# Dependency imports
from absl import flags
import tensorflow_probability as tfp
from tools.get_data import *
from tools.statistics import *
from tools.analysis import *
tfd = tfp.distributions
tfb = tfp.bijectors


# Flags
# IMAGE_SHAPE = (28, 28, 1)
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
       FLAGS.__delattr__(keys)

#del_all_flags(tf.flags.FLAGS)

flags = tf.app.flags
FLAGS = tf.flags.FLAGS


flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_float(
    "beta", default=1.0, help="Beta in a Beta-VAE.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "latent_size",
    default=16,
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
    "viz_steps", default=500, help="Frequency at which to save visualizations.")

flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

flags.DEFINE_bool('skip_train', default=False, help='Whether to skip training the model ensembles and go straight to analysis.')

flags.DEFINE_string(
    "train_dataset",
    default="mnist",
    help="mnist/fashion_mnist for VAE, mnist/fashion_mnist/cifar10 for wgan")


def main(argv):
    del argv
    M = 5  # number of models in ensemble
    for i in range(M):
        print('Running VAE experiment %d' % i)
        params = FLAGS.flag_values_dict()
        params["activation"] = getattr(tf.nn, params["activation"])
       
        from vae.model import model_fn
        #FLAGS.model_dir = os.path.join(FLAGS.model_dir, str(i))
        model_dir = FLAGS.model_dir + str(i)

        if FLAGS.delete_existing and tf.gfile.Exists(model_dir):
            tf.logging.warn("Deleting old log directory at {}".format(model_dir))
            tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.MakeDirs(model_dir)

        if FLAGS.train_dataset == "mnist":
            train_input_fn, eval_input_fn = get_dataset('mnist', FLAGS.batch_size)
        elif FLAGS.train_dataset == "fashion_mnist":
            train_input_fn, eval_input_fn = get_dataset('fashion_mnist', FLAGS.batch_size)

        estimator = tf.estimator.Estimator(
            model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=model_dir,
                save_checkpoints_steps=FLAGS.viz_steps,
            ),
        )
        if not FLAGS.skip_train:
            for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
                estimator.train(train_input_fn, steps=FLAGS.viz_steps)
            # Evaluate once after training.
            eval_results = estimator.evaluate(eval_input_fn)
            print("Evaluation_results:\n\t%s\n" % eval_results)

    



    # the first dataset in compare_datasets is the base distribution we use
    # to compare Out of distribution samples against. Should be the same as the training dataset
    if FLAGS.train_dataset == "mnist":
        compare_datasets = ['mnist', 'omniglot', 'notMNIST', 'fashion_mnist', 'mnist', 'mnist',
                            'uniform_noise', 'normal_noise']
    elif FLAGS.train_dataset == "fashion_mnist":
        compare_datasets = ['fashion_mnist', 'omniglot', 'notMNIST', 'mnist', 'fashion_mnist', 'fashion_mnist', 'uniform_noise',
                            'normal_noise']


    # whether to noise each dataset or not
    noised_list = [False, False, False, False, True, True, False, False]
    # if the element in noised_list is true for a dataset then what kind of noise/transformations to apply?
    # if the above element is set False, any noise/transformation will not be processed.
    noise_type_list = ['normal', 'normal', 'normal', 'normal', 'hor_flip', 'ver_flip', 'normal', 'normal']

    # add adversarially perturbed noise as OoD
    show_adv_examples = 'normal'
    expand_last_dim = True


    #which model to use to create adversarially perturbed noise for ensemble analysis
    adv_base = 0

    # ensembles on OoD datasets
    ensemble_OoD(compare_datasets, expand_last_dim, noised_list, noise_type_list, FLAGS.batch_size, model_fn, FLAGS.model_dir, show_adv_examples, adv_base, each_size=10000)
    # ensembles on corrupted indistribution
    #ensemble_corruptions(compare_datasets[0], expand_last_dim, noised_list, noise_type_list, model_fn, FLAGS.model_dir, each_size=1000)
    # ensembles on perturbed indistribution
    #ensemble_perturbations(compare_datasets[0], expand_last_dim, noised_list, noise_type_list, model_fn, FLAGS.model_dir, each_size=1000)


if __name__ == "__main__":
    tf.app.run()

