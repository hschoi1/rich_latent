from absl import flags
from get_data import *
from model import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
       FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

flags.DEFINE_bool(
    'noise_to_input',default=False, help="whether to add noise to training data")

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=2,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_bool(
    "use_NF",
    default = True,
    help = "If False, normalizing flows are not applied")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")
flags.DEFINE_integer("n_flows", default=6, help="Number of Normalizing Flows")
flags.DEFINE_float("elbo_threshold", default=5.0, help="anomaly threshold for whole elbo")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
         "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
         "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
         "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
         "then you must also specify `mixture_components=1`.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=1000, help="Frequency at which to save visualizations.")

flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")


FLAGS = flags.FLAGS
params = FLAGS.flag_values_dict()
params["activation"] = getattr(tf.nn, params["activation"])

compare_datasets = ['fashion_mnist','mnist','noise']


def fetch(input_fn, model_fn, model_dir, keys,  i):

    FLAGS.model_dir = model_dir + str(i)
    print('Evaluating eval samples for %s' % FLAGS.model_dir)
    assert tf.train.latest_checkpoint(FLAGS.model_dir) is not None

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir, ))

    batch_results_ = list(estimator.predict(
        input_fn,
        predict_keys=keys,
        yield_single_examples=False))

    tuples = []

    for key in keys:
        each_key = np.concatenate([b[key].T for b in batch_results_], axis=0)
        each_key = np.sum(each_key, axis=1)
        tuples.append(each_key)

    return tuples


# Plot Rate-Distortion Curve for true data (good model minimizes them both)
def plot_rd(eval_input_fn, model_fn, model_dir):
  keys = ['rate', 'distortion']
  results = fetch(eval_input_fn, model_fn, model_dir, keys, j)
  plt.scatter(results[0], results[1])
  plt.set_xlabel('Rate')
  plt.set_ylabel('Distortion')

  plt.show()


# plot Ensemble Mean vs Ensemble Variance
def plot_ensemble_stats(eval_input_fn, model_fn, model_dir, keys):
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


def compare_elbo(std_dataset, datasets, model_fn, model_dir):
    M = 5
    f, axes = plt.subplots(1, 2, figsize=(10, 5))

    adversarial_noise_ensemble = []
    adv_keys = ['adv_elbo']
    keys = ['elbo']
    # collect ensemble elbo for adversarial noise input
    for i in range(M):
        _, eval_input_fn = get_dataset(std_dataset, FLAGS.batch_size)
        adversarial_noise_results = fetch(eval_input_fn, model_fn, model_dir, adv_keys, i)
        adversarial_noise_ensemble.append(adversarial_noise_results)

    #histogram of elbo of the last model on adversairal noise
    axes[0].hist(adversarial_noise_results, label='adversarial noise')
    # elbo of the last model
    single_adv_elbo = adversarial_noise_results[0]
    # get ensemble var
    adv_ensemble = adversarial_noise_ensemble[0]
    adv_ensemble_var = np.var(adv_ensemble, axis=0)

    # scatter plot of single elbo vs enesmble variance on adversarial noise
    axes[1].scatter(single_adv_elbo, adv_ensemble_var, label='adversarial noise')

    for dataset in datasets:
        ensemble = []
        each_dataset = []
        _, eval_input_fn = get_dataset(dataset, FLAGS.batch_size)
        for i in range(M):
            results = fetch(eval_input_fn, model_fn, model_dir, keys, i)
            each_dataset.append(results)
        each_dataset = np.stack(each_dataset)
        ensemble.append(each_dataset)

        #histogram of elbo of the last model on each dataset
        axes[0].hist(results, label=dataset)

        #elbo of the last model
        single_elbo = results[0]
        # get ensemble var
        ensemble = ensemble[0]
        ensemble_var = np.var(ensemble, axis=0)

        # scatter plot of single elbo vs enesmble variance on each dataset
        axes[1].scatter(single_elbo, ensemble_var, label=dataset)


    # histogram of elbo of different datasets under cifar10/mnist
    axes[0].set_xlabel('single ELBO of each dataset')
    axes[0].set_ylabel('frequency')
    axes[0].legend()
    axes[1].set_xlabel('ELBO of single model')
    axes[1].set_ylabel('ensemble variance')
    axes[1].legend()
    plt.legend()
    plt.show()
    f.savefig("elbo")



def main(argv):
   #compare with other datasets under mnist likelihood
   FLAGS.model_dir = "gs://hyunsun/image_vae/mnist/model"
   compare_elbo('mnist', compare_datasets, model_fn, FLAGS.model_dir)


if __name__ == "__main__":
    tf.app.run()
