from absl import flags
from tools.get_data import *
from tools.statistics import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
params = FLAGS.flag_values_dict()

# restore ckpt from i th model in model_dir and calculate the values of keys
def fetch(input_fn, model_fn, model_dir, keys,  i):

    model_dir = model_dir + str(i)
    print('Evaluating eval samples for %s' % model_dir)
    assert tf.train.latest_checkpoint(model_dir) is not None

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=model_dir, ))

    batch_results_ = list(estimator.predict(
        input_fn,
        predict_keys=keys,
        yield_single_examples=False))

    tuples = []

    for key in keys:
        if batch_results_[0][key].shape[-1] in [1, 3]:
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
def adversarial_fetch(eval_input_fn, batch_size, model_fn, model_dir, keys, base_model, apply_model):
    adv_keys = ['adversarial_normal_noise', 'adversarial_uniform_noise']

    fetched = fetch(eval_input_fn, model_fn, model_dir, adv_keys, base_model)
    adversarial_normal_noise_results = fetched[0]
    adversarial_uniform_noise_results = fetched[1]
    adv_normal_eval_dataset = tf.data.Dataset.from_tensor_slices((adversarial_normal_noise_results))
    adv_normal_eval_dataset = adv_normal_eval_dataset.batch(batch_size)

    adv_uniform_eval_dataset = tf.data.Dataset.from_tensor_slices((adversarial_uniform_noise_results))
    adv_uniform_eval_dataset = adv_uniform_eval_dataset.batch(batch_size)

    adv_normal_eval_input_fn = lambda: adv_normal_eval_dataset.make_one_shot_iterator().get_next()
    adv_uniform_eval_input_fn = lambda: adv_uniform_eval_dataset.make_one_shot_iterator().get_next()

    adversarial_normal_noise_results = fetch(adv_normal_eval_input_fn, model_fn, model_dir, keys, apply_model)
    adversarial_uniform_noise_results = fetch(adv_uniform_eval_input_fn, model_fn, model_dir, keys, apply_model)

    return adversarial_normal_noise_results, adversarial_uniform_noise_results


# adversarially perturb a normal/uniform noise and get the adversarial example from base_model
# and use that adversarial example to calculate the values of keys for all 5 models.
def adversarial_ensemble_fetch(base, batch_size, model_fn, model_dir, keys, base_model):
    eval_input_fn = get_eval_dataset(base, batch_size)

    # collect ensemble elbo for adversarial noise input
    adversarial_normal_noise_ensemble = []
    adversarial_uniform_noise_ensemble = []
    M = 5
    for i in range(M,10):
        adversarial_normal_noise_results, adversarial_uniform_noise_results = adversarial_fetch(eval_input_fn,
                                                                                batch_size, model_fn, model_dir, keys, base_model, i)

        adversarial_normal_noise_ensemble.append(adversarial_normal_noise_results)
        adversarial_uniform_noise_ensemble.append(adversarial_uniform_noise_results)

    return adversarial_normal_noise_ensemble, adversarial_uniform_noise_ensemble

#plot single critic predictions vs Ensemble Variance
def compare_critic_preds(datasets, expand_last_dim, noised_list, noise_type_list, batch_size,
                 model_fn, model_dir, show_adv, adv_base):
    M = 5
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    keys = ['true_logit']
    ensemble_logits = []
    from tools.statistics import analysis_helper
    for i in range(M, 10):
        single_logit, datasets_names = analysis_helper(datasets, expand_last_dim, noised_list, noise_type_list, False,
                                                      model_fn, model_dir, i, i, keys)
        single_logit = single_logit[0][:, 0]  # remove meaningless dim
        ensemble_logits.append(single_logit)

    ensemble_keys = ['single_logit']
    ensemble_results = [single_logit]
    ensemble_logit_mean = np.mean(ensemble_logits, axis=0)
    ensemble_var = np.var(ensemble_logits, axis=0)
    ensemble_results += [ensemble_logit_mean, ensemble_var]
    ensemble_keys += ['ensemble_mean_logit', 'ensemble_logit_var']

    # histogram of critic logit of the last model on different datasets
    bin_range = (-5, 5)
    bins = 100
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[0].hist(single_logit[1000 * i:1000 * (i + 1)], label=label, alpha=0.5, bins=bins, range=bin_range)


    # scatter plot of single logit vs enesmble variance on each dataset
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[1].scatter(single_logit[1000 * i:1000 * (i + 1)], ensemble_var[1000 * i:1000 * (i + 1)], label=label,
                        alpha=0.3)

    if show_adv is not None:
        adversarial_normal_noise_ensemble, adversarial_uniform_noise_ensemble = adversarial_ensemble_fetch(datasets[0],
                                                                        batch_size, model_fn, model_dir, keys, adv_base)

        adversarial_normal_noise_ensemble = np.array(adversarial_normal_noise_ensemble)[:, 0, :, 0]  # remove meaningless dims
        adversarial_uniform_noise_ensemble = np.array(adversarial_uniform_noise_ensemble)[:, 0, :, 0]
        # critic logit of the last model
        single_adv_normal_logit = adversarial_normal_noise_ensemble[-1]
        single_adv_uniform_logit = adversarial_uniform_noise_ensemble[-1]

        # histogram of critic of the last model on adversarial noise
        axes[0].hist(single_adv_normal_logit, label='adversarial normal noise', alpha=0.5, bins=100,
                     range=bin_range)
        axes[0].hist(single_adv_uniform_logit, label='adversarial uniform noise', alpha=0.5, bins=100,
                     range=bin_range)

        # get ensemble mean/var on adversarial noise
        adv_normal_ensemble_mean = np.mean(adversarial_normal_noise_ensemble, axis=0)
        adv_normal_ensemble_var = np.var(adversarial_normal_noise_ensemble, axis=0)
        adv_uniform_ensemble_mean = np.mean(adversarial_uniform_noise_ensemble, axis=0)
        adv_uniform_ensemble_var = np.var(adversarial_uniform_noise_ensemble, axis=0)

        # scatter plot of single logit vs enesmble variance on adversarial noise
        axes[1].scatter(single_adv_normal_logit, adv_normal_ensemble_var, label='adversarial normal noise', alpha=0.1)
        axes[1].scatter(single_adv_uniform_logit, adv_uniform_ensemble_var, label='adversarial uniform noise', alpha=0.1)

        # add single logit/ensemble logit mean/var of adversarial examples for score analysis
        ensemble_results[0] = np.concatenate([ensemble_results[0], single_adv_normal_logit, single_adv_uniform_logit], axis=0)
        ensemble_results[1] = np.concatenate([ensemble_results[1], adv_normal_ensemble_mean, adv_uniform_ensemble_mean], axis=0)
        ensemble_results[2] = np.concatenate([ensemble_results[2], adv_normal_ensemble_var, adv_uniform_ensemble_var], axis=0)
        datasets_names += ['adv_normal_noise', 'adv_uniform_noise']

    from tools.statistics import plot_analysis
    plot_analysis(ensemble_results, datasets_names, ensemble_keys)

    # adjust range
    axes[0].set_xlabel('single true logit of each dataset')
    axes[0].set_ylabel('frequency')
    axes[0].legend()
    axes[1].set_xlabel('true logit of single model')
    axes[1].set_ylabel('ensemble variance')
    axes[1].legend()
    plt.legend()
    plt.show()
    f.savefig("logits")

# plot elbo for each dataset and also plot single elbo vs. ensemble variance
def compare_elbo(datasets, expand_last_dim,  noised_list, noise_type_list, batch_size,
                 model_fn, model_dir, show_adv, adv_base):
    from tools.statistics import analysis_helper
    M = 5
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    keys = ['elbo', 'approx_posterior_mean', 'approx_posterior_stddev']
    ensemble_elbos = []
    ensemble_posterior_means = []
    ensemble_posterior_vars = []

    for i in range(5,10):
        single_results, datasets_names = analysis_helper(datasets, expand_last_dim,  noised_list, noise_type_list, False,
                                                 model_fn,model_dir, i, i, keys)
        single_elbo = single_results[0]
        single_posterior_mean = single_results[1]
        single_posterior_var = single_results[2]
        ensemble_elbos.append(single_elbo)
        ensemble_posterior_means.append(single_posterior_mean)
        ensemble_posterior_vars.append(single_posterior_var)

    ensemble_elbos = np.array(ensemble_elbos)
    ensemble_posterior_means = np.array(ensemble_posterior_means)
    ensemble_posterior_vars = np.array(ensemble_posterior_vars)

    # histogram of elbo of the last model on different datasets
    bin_range = (-2000, 1000)
    bins = 300
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[0].hist(single_elbo[1000*i:1000*(i+1)], label=label, alpha=0.5, bins=bins, range=bin_range)

    # get ensemble var for scatter plot below
    ensemble_var = np.var(ensemble_elbos, axis=0)

    # scatter plot of single elbo vs enesmble variance on each dataset
    for i in range(len(datasets)):
        label = datasets_names[i]
        axes[1].scatter(single_elbo[1000*i:1000*(i+1)], ensemble_var[1000*i:1000*(i+1)], label=label, alpha=0.3)

    if show_adv:
        adversarial_normal_noise_ensemble, adversarial_uniform_noise_ensemble = adversarial_ensemble_fetch(datasets[0],
                                                                        batch_size, model_fn, model_dir, keys, adv_base)
        # get ensemble statistics on adversarial noise
        adversarial_normal_noise_ensemble = np.array(adversarial_normal_noise_ensemble)
        adversarial_normal_noise_ensemble = np.transpose(adversarial_normal_noise_ensemble, (1, 0, 2))
        adversarial_uniform_noise_ensemble = np.array(adversarial_uniform_noise_ensemble)
        adversarial_uniform_noise_ensemble = np.transpose(adversarial_uniform_noise_ensemble, (1, 0, 2))
        # for axis0, index 0:elbo 1:posterior mean 2:posterior variance

        # elbo of the last model
        single_adv_normal_elbo = adversarial_normal_noise_ensemble[0][-1]
        single_adv_uniform_elbo = adversarial_uniform_noise_ensemble[0][-1]

        # histogram of elbo of the last model on adversarial noise
        axes[0].hist(single_adv_normal_elbo, label='adversarial normal noise', alpha=0.5, bins=100,
                     range=bin_range)
        axes[0].hist(single_adv_uniform_elbo, label='adversarial uniform noise', alpha=0.5, bins=100,
                     range=bin_range)

        # get ensemble var of elbos for adversarial examples
        adv_normal_ensemble_var = np.var(adversarial_normal_noise_ensemble[0], axis=0)
        adv_uniform_ensemble_var = np.var(adversarial_uniform_noise_ensemble[0], axis=0)

        # scatter plot of single elbo vs enesmble variance on adversarial noise
        axes[1].scatter(single_adv_normal_elbo, adv_normal_ensemble_var, label='adversarial normal noise', alpha=0.1)
        axes[1].scatter(single_adv_uniform_elbo, adv_uniform_ensemble_var, label='adversarial uniform noise', alpha=0.1)

        # add single elbo/ensemble statistics of adversarial examples for score analysis
        single_elbo = np.concatenate([single_elbo, single_adv_normal_elbo, single_adv_uniform_elbo], axis=0)
        ensemble_elbos = np.concatenate([ensemble_elbos, adversarial_normal_noise_ensemble[0],
                                         adversarial_uniform_noise_ensemble[0]], axis=1)
        ensemble_posterior_means = np.concatenate([ensemble_posterior_means, adversarial_normal_noise_ensemble[1],
                                                   adversarial_uniform_noise_ensemble[1]], axis=1)
        ensemble_posterior_vars = np.concatenate([ensemble_posterior_vars, adversarial_normal_noise_ensemble[2],
                                                   adversarial_uniform_noise_ensemble[2]], axis=1)
        datasets_names += ['adv_normal_noise', 'adv_uniform_noise']

    ensemble_keys = ['single_elbo', 'ensemble_elbo_mean', 'ensemble_elbo_var', 'ensemble_posterior_means_mean',
                     'ensemble_posterior_means_var', 'ensemble_posterior_vars_mean', 'ensemble_posterior_vars_var']
    ensemble_results = [single_elbo]
    ensemble_results += vae_ensemble_statistics(ensemble_elbos, ensemble_posterior_means, ensemble_posterior_vars)

    from tools.statistics import plot_analysis
    plot_analysis(ensemble_results, datasets_names, ensemble_keys)

    # adjust range
    axes[0].set_xlabel('single ELBO of each dataset')
    axes[0].set_ylabel('frequency')
    axes[0].legend()
    axes[1].set_xlabel('ELBO of single model')
    axes[1].set_ylabel('ensemble variance')

    top = 40000
    axes[1].set_xlim(bin_range)
    axes[1].set_ylim(bottom=0, top=top)
    axes[1].legend()
    plt.legend()
    plt.show()
    f.savefig("elbo")


def vae_ensemble_statistics(ensemble_elbos, ensemble_posterior_means, ensemble_posterior_vars):

    ensemble_elbo_mean = np.mean(ensemble_elbos, axis=0)
    ensemble_elbo_var = np.var(ensemble_elbos, axis=0)

    ensemble_posterior_means_mean = np.mean(ensemble_posterior_means, axis=0)
    ensemble_posterior_means_var = np.var(ensemble_posterior_means, axis=0)

    ensemble_posterior_vars_mean = np.mean(ensemble_posterior_vars, axis=0)
    ensemble_posterior_vars_var = np.var(ensemble_posterior_vars, axis=0)

    dics = {'ensemble_elbo_mean': ensemble_elbo_mean,
            'ensemble_elbo_var': ensemble_elbo_var,
            'ensemble_posterior_means_mean': ensemble_posterior_means_mean,
            'ensemble_posterior_means_var': ensemble_posterior_means_var,
            'ensemble_posterior_vars_mean': ensemble_posterior_vars_mean,
            'ensemble_posterior_vars_var': ensemble_posterior_vars_var}

    results = [ensemble_elbo_mean, ensemble_elbo_var, ensemble_posterior_means_mean,
               ensemble_posterior_means_var, ensemble_posterior_vars_mean, ensemble_posterior_vars_var]

    return results



