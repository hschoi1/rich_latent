import pickle
import os
from absl import flags
from tools.get_data import *
from tools.statistics import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
params = FLAGS.flag_values_dict()

# restore ckpt from i th model in model_dir and calculate the values of keys
def fetch(input_fn, model_fn, model_dir, keys,  i, checkpoint_step):

    model_dir = model_dir + str(i)
    print('Evaluating eval samples for %s' % model_dir)
    if checkpoint_step is None:
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        assert checkpoint_path is not None
    else:
        checkpoint_path = os.path.join(model_dir, 'model.ckpt-%d' % checkpoint_step)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params)

    batch_results_ = list(estimator.predict(
        input_fn,
        predict_keys=keys,
        checkpoint_path=checkpoint_path,
        yield_single_examples=False))

    tuples = []

    for key in keys:
        if batch_results_[0][key].shape[-1] in [1, 3, 30]: #when fetching adversarial examples
            each_key = np.concatenate([b[key] for b in batch_results_], axis=0)
        else:
            if key in ['approx_posterior_mean', 'approx_posterior_stddev']:
                each_key = np.concatenate([b[key] for b in batch_results_], axis=0)
            else:
                each_key = np.concatenate([b[key].T for b in batch_results_], axis=0)
            each_key = np.mean(each_key, axis=1)

        tuples.append(each_key)
    return tuples


# Plot Rate-Distortion Curve for true data (good model minimizes them both)
def plot_rd(eval_input_fn, model_fn, model_dir):
  keys = ['rate', 'distortion']
  results = fetch(eval_input_fn, model_fn, model_dir, keys, 0)
  plt.scatter(results[0], results[1])
  plt.set_xlabel('Rate')
  plt.set_ylabel('Distortion')

  plt.show()


# plot Ensemble ELBO Mean vs Ensemble ELBO Variance
def plot_ensemble_stats(eval_input_fn, model_fn, model_dir):
  M = 5
  keys = ['elbo']
  f, axes = plt.subplots(1, len(keys), figsize=(15, 5))
  for i,key in enumerate(keys):
      ensemble = []

      for j in range(M):
        results = fetch(eval_input_fn, model_fn, model_dir, keys, j)
        ensemble.append(results)

      mean = np.mean(ensemble, axis=0)
      var = np.var(ensemble, axis=0)
      axes[i].scatter(mean, var)
      axes[i].set_xlabel('Ensemble %s mean' % key)
      axes[i].set_ylabel('Ensemble %s variance' % key)


# adversarially perturb a normal/uniform noise and get the adversarial example from base_model
# and calculate the values of keys for the adversarial example using apply_model
def adversarial_fetch(eval_input_fn, batch_size, model_fn, model_dir, keys, base_model, apply_model, checkpoint_step=None):
    adv_keys = ['adversarial_normal_noise', 'adversarial_uniform_noise']

    fetched = fetch(eval_input_fn, model_fn, model_dir, adv_keys, base_model, checkpoint_step)
    adversarial_normal_noise_results = fetched[0]
    adversarial_uniform_noise_results = fetched[1]
    adv_normal_eval_dataset = tf.data.Dataset.from_tensor_slices((adversarial_normal_noise_results))
    adv_normal_eval_dataset = adv_normal_eval_dataset.batch(batch_size)

    adv_uniform_eval_dataset = tf.data.Dataset.from_tensor_slices((adversarial_uniform_noise_results))
    adv_uniform_eval_dataset = adv_uniform_eval_dataset.batch(batch_size)

    adv_normal_eval_input_fn = lambda: adv_normal_eval_dataset.make_one_shot_iterator().get_next()
    adv_uniform_eval_input_fn = lambda: adv_uniform_eval_dataset.make_one_shot_iterator().get_next()

    adversarial_normal_noise_results = fetch(adv_normal_eval_input_fn, model_fn, model_dir, keys, apply_model, checkpoint_step)
    adversarial_uniform_noise_results = fetch(adv_uniform_eval_input_fn, model_fn, model_dir, keys, apply_model, checkpoint_step)

    return adversarial_normal_noise_results, adversarial_uniform_noise_results


# adversarially perturb a normal/uniform noise and get the adversarial example from base_model
# and use that adversarial example to calculate the values of keys for all 5 models.
def adversarial_ensemble_fetch(base, batch_size, model_fn, model_dir, keys, base_model, each_size=1000):
    eval_input_fn = get_eval_dataset(base, batch_size, each_size=each_size)

    # collect ensemble elbo for adversarial noise input
    adversarial_normal_noise_ensemble = []
    adversarial_uniform_noise_ensemble = []
    M = 5
    for i in range(M):
        adversarial_normal_noise_results, adversarial_uniform_noise_results = adversarial_fetch(eval_input_fn,
                                                                                batch_size, model_fn, model_dir, keys, base_model, i)

        adversarial_normal_noise_ensemble.append(adversarial_normal_noise_results)
        adversarial_uniform_noise_ensemble.append(adversarial_uniform_noise_results)

    return adversarial_normal_noise_ensemble, adversarial_uniform_noise_ensemble

# plot elbo for each dataset and also plot single elbo vs. ensemble variance
def ensemble_OoD(datasets, expand_last_dim,  noised_list, noise_type_list, batch_size,
                 model_fn, model_dir, show_adv, adv_base, feature_shape=(28,28), each_size=1000):
    from tools.statistics import analysis_helper, name_helper, plot_analysis
    M = 5
    f, axes = plt.subplots(1, 2, figsize=(12, 5))

    keys = ['elbo']  # or rate
    ensemble_elbos = []

    # construct input for all models
    converted_datasets, datasets_names = name_helper(datasets, noised_list, noise_type_list)
    input_fn = build_eval_multiple_datasets(converted_datasets, 100, expand_last_dim, noised_list,noise_type_list, feature_shape, each_size)  # corrupted indistribution attached automatically
    datasets_names += ['gaussian_noise_c', 'shot_noise_c', 'impulse_noise_c', 'defocus_blur_c',
                       'glass_blur_c', 'motion_blur_c', 'zoom_blur_c', 'snow_c', 'frost_c', 'fog_c',
                       'brightness_c', 'contrast_c', 'elastic_transform_c', 'pixelate_c', 'jpeg_compression_c',
                       'speckle_noise_c', 'gaussian_blur_c', 'spatter_c', 'saturate_c', 'gaussian_noise_p', 'shot_noise_p',
                       'motion_blur_p', 'zoom_blur_p', 'snow_p', 'brightness_p', 'translate_p', 'rotate_p', 'tilt_p',
                       'scale_p', 'speckle_noise_p', 'gaussian_blur_p', 'spatter_p', 'shear_p'] # manually attach dataset names

    for i in range(M):
        single_results = analysis_helper(input_fn, converted_datasets, None, model_fn,model_dir, i, i, keys)
        single_elbo = single_results[0]
        ensemble_elbos.append(single_elbo)

    ensemble_elbos = np.array(ensemble_elbos)
    ensemble_var = np.var(ensemble_elbos, axis=0)

    # histogram of elbo of the last model on different datasets
    if each_size==1000:
        bin_range = (-2000, 1000)
        bin_dict = {'single_elbo':(-2000, 1000,300), 'ensemble_elbo_mean':(-2000, 1000,300)}
    else:
        bin_range = (-100, 0)
        bin_dict = {'single_elbo': (-100, 0, 300), 'ensemble_elbo_mean': (-100, 0, 300)}
    bins = 300
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[0].hist(single_elbo[each_size*i:each_size*(i+1)], label=label, alpha=0.5, bins=bins, range=bin_range)


    # scatter plot of single elbo vs enesmble variance on each dataset
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[1].scatter(single_elbo[each_size*i:each_size*(i+1)], ensemble_var[each_size*i:each_size*(i+1)], label=label, alpha=0.3)

    if show_adv is not None:
        adversarial_normal_noise_ensemble, adversarial_uniform_noise_ensemble = adversarial_ensemble_fetch(datasets[0],
                                                                        batch_size, model_fn, model_dir, keys, adv_base, each_size=each_size)
        # get ensemble statistics on adversarial noise
        adversarial_normal_noise_ensemble = np.array(adversarial_normal_noise_ensemble)[:,0,:]
        adversarial_uniform_noise_ensemble = np.array(adversarial_uniform_noise_ensemble)[:,0,:]

        # elbo of the last model
        single_adv_normal_elbo = adversarial_normal_noise_ensemble[-1]
        single_adv_uniform_elbo = adversarial_uniform_noise_ensemble[-1]

        # histogram of elbo of the last model on adversarial noise
        axes[0].hist(single_adv_normal_elbo, label='adversarial normal noise', alpha=0.5, bins=100,
                     range=bin_range)
        axes[0].hist(single_adv_uniform_elbo, label='adversarial uniform noise', alpha=0.5, bins=100,
                     range=bin_range)

        # get ensemble var of elbos for adversarial examples
        adv_normal_ensemble_var = np.var(adversarial_normal_noise_ensemble, axis=0)
        adv_uniform_ensemble_var = np.var(adversarial_uniform_noise_ensemble, axis=0)

        # scatter plot of single elbo vs enesmble variance on adversarial noise
        axes[1].scatter(single_adv_normal_elbo, adv_normal_ensemble_var, label='adversarial normal noise', alpha=0.1)
        axes[1].scatter(single_adv_uniform_elbo, adv_uniform_ensemble_var, label='adversarial uniform noise', alpha=0.1)

        # add single elbo/ensemble statistics of adversarial examples for score analysis
        single_elbo = np.concatenate([single_elbo, single_adv_normal_elbo, single_adv_uniform_elbo], axis=0)
        ensemble_elbos = np.concatenate([ensemble_elbos, adversarial_normal_noise_ensemble,
                                         adversarial_uniform_noise_ensemble], axis=1)

        datasets_names += ['adv_normal_noise', 'adv_uniform_noise']

    ensemble_mean = np.mean(ensemble_elbos, axis=0)
    ensemble_var = np.var(ensemble_elbos, axis=0)
    WAIC = ensemble_mean - ensemble_var
    ensemble_keys = ['single_elbo', 'ensemble_elbo_mean', 'ensemble_elbo_var', 'WAIC']
    ensemble_results = [single_elbo, ensemble_mean, ensemble_var, WAIC]
    if keys[0] == 'rate':
        ensemble_keys = ['single_rate']
        ensemble_results = [single_elbo]


    plot_analysis(ensemble_results, datasets_names, ensemble_keys, bins=bin_dict, each_size=each_size)

    # adjust range
    axes[0].set_xlabel('single ELBO of each dataset')
    axes[0].set_ylabel('frequency')
    axes[1].set_xlabel('ELBO of single model')
    axes[1].set_ylabel('ensemble variance')
    if each_size==1000:
        top = 40000
    else:
        top = 500
    axes[1].set_xlim(bin_range)
    axes[1].set_ylim(bottom=0, top=top)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    f.savefig(os.path.join(FLAGS.model_dir,"elbo.eps"), bbox_inches="tight", format='eps', dpi=1000)


def ensemble_perturbations(base_dataset, expand_last_dim,  noised_list, noise_type_list,
                 model_fn, model_dir, feature_shape=(28,28), each_size=1000):
    from tools.statistics import analysis_helper, name_helper, plot_analysis

    converted_datasets, datasets_names = name_helper([base_dataset], noised_list, noise_type_list)
    input_fn = build_perturbation_datasets(converted_datasets[0], 100, expand_last_dim, feature_shape, each_size)
    perturbation_names = ['gaussian_noise_p', 'shot_noise_p', 'motion_blur_p', 'zoom_blur_p', 'snow_p', 'brightness_p', 'translate_p',
                       'rotate_p', 'tilt_p', 'scale_p', 'speckle_noise_p', 'gaussian_blur_p', 'spatter_p', 'shear_p']
    for perturbation_name in perturbation_names:
        for severity in range(30):
            datasets_names.append(perturbation_name+str(severity)) # attach severity info

    M = 5
    keys = ['elbo']  # or rate
    ensemble_elbos = []
    for i in range(M):
        single_results = analysis_helper(input_fn, converted_datasets, None, model_fn, model_dir, i, i, keys)
        single_elbo = single_results[0]
        ensemble_elbos.append(single_elbo)

    ensemble_elbos = np.array(ensemble_elbos)
    ensemble_mean = np.mean(ensemble_elbos, axis=0)
    ensemble_var = np.var(ensemble_elbos, axis=0)
    WAIC = ensemble_mean - ensemble_var
    ensemble_keys = ['single_elbo', 'ensemble_elbo_mean', 'ensemble_elbo_var', 'WAIC']
    ensemble_results = [single_elbo, ensemble_mean, ensemble_var, WAIC]
    if keys[0] == 'rate':
        ensemble_keys = ['single_rate']
        ensemble_results = [single_elbo]

    plot_analysis(ensemble_results, datasets_names, ensemble_keys, each_size=each_size)


def ensemble_corruptions(base_dataset, expand_last_dim,  noised_list, noise_type_list,
                 model_fn, model_dir, feature_shape=(28,28), each_size=1000):
    from tools.statistics import analysis_helper, name_helper, plot_analysis

    converted_datasets, datasets_names = name_helper([base_dataset], noised_list, noise_type_list)
    input_fn = build_corruption_datasets(converted_datasets[0], 100, expand_last_dim, feature_shape, each_size, severity=1)
    datasets_names += ['gaussian_noise_c', 'shot_noise_c', 'impulse_noise_c', 'defocus_blur_c',
                       'glass_blur_c', 'motion_blur_c', 'zoom_blur_c', 'snow_c', 'frost_c', 'fog_c',
                       'brightness_c', 'contrast_c', 'elastic_transform_c', 'pixelate_c', 'jpeg_compression_c',
                       'speckle_noise_c', 'gaussian_blur_c', 'spatter_c', 'saturate_c']
    M = 5
    keys = ['elbo']  # or rate
    ensemble_elbos = []
    for i in range(M):
        single_results = analysis_helper(input_fn, converted_datasets, None, model_fn, model_dir, i, i, keys)
        single_elbo = single_results[0]
        ensemble_elbos.append(single_elbo)

    ensemble_elbos = np.array(ensemble_elbos)
    ensemble_mean = np.mean(ensemble_elbos, axis=0)
    ensemble_var = np.var(ensemble_elbos, axis=0)
    WAIC = ensemble_mean - ensemble_var
    ensemble_keys = ['single_elbo', 'ensemble_elbo_mean', 'ensemble_elbo_var', 'WAIC']
    ensemble_results = [single_elbo, ensemble_mean, ensemble_var, WAIC]
    if keys[0] == 'rate':
        ensemble_keys = ['single_rate']
        ensemble_results = [single_elbo]

    plot_analysis(ensemble_results, datasets_names, ensemble_keys,  each_size=each_size)


def history_compare_elbo(datasets, expand_last_dim,  noised_list, noise_type_list, batch_size,
                 model_fn, model_dir, show_adv, adv_base, feature_shape=(28,28), each_size=1000):
    """Compute plt.scatter(elbo, auroc) for each checkpoint """
    from tools.statistics import analysis_helper
    from tools.statistics import get_scores
    M = 5
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    step_arr = np.arange(0, FLAGS.max_steps, FLAGS.viz_steps)

    keys = ['elbo', 'approx_posterior_mean', 'approx_posterior_stddev']
    full_results = {}    
    # tmp
    # step_arr = [5000]
    for step in step_arr:
        print('step: %d' % step)

        ensemble_elbos = []
        ensemble_posterior_means = []
        ensemble_posterior_vars = []

        for i in range(M):
            single_results, datasets_names = analysis_helper(datasets, expand_last_dim,  noised_list, noise_type_list, None, model_fn,model_dir, i, i, keys, feature_shape, each_size, step)
            single_elbo = single_results[0]
            single_posterior_mean = single_results[1]
            single_posterior_var = single_results[2]
            ensemble_elbos.append(single_elbo)
            ensemble_posterior_means.append(single_posterior_mean)
            ensemble_posterior_vars.append(single_posterior_var)
        ensemble_elbos = np.array(ensemble_elbos)

        # analyze statistics
        ensemble_var = np.var(ensemble_elbos, axis=0)
        ensemble_mean = np.mean(ensemble_elbos, axis=0)
        # Perform classication based on ensemble var, as a function of ensemble mean scores.
        results  = get_scores(ensemble_mean, ensemble_var, datasets_names, each_size, False)
        full_results[step] = results
    
    with open(os.path.join(FLAGS.model_dir, 'train_history_scores.pkl'), 'wb') as f:
      pickle.dump(full_results, f)
