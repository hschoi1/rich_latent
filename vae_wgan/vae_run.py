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




def main(argv):
    del argv
    M = 5  # number of models in ensemble
    for i in range(M):
        print('Running VAE experiment %d' % i)
        params = FLAGS.flag_values_dict()
        params["activation"] = getattr(tf.nn, params["activation"])
       
        from vae.model import model_fn
        #FLAGS.model_dir = "gs://hyunsun/image_vae/mnist/model%d" % i
        model_dir = os.path.join(FLAGS.model_dir, str(i))
        if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
            tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)

        train_input_fn, eval_input_fn = get_dataset('mnist', FLAGS.batch_size)

        estimator = tf.estimator.Estimator(
            model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=FLAGS.model_dir,
                save_checkpoints_steps=FLAGS.viz_steps,
            ),
        )
        if not FLAGS.skip_train:
            estimator.train(train_input_fn, max_steps=FLAGS.max_steps)
            # Evaluate once after training.
            eval_results = estimator.evaluate(eval_input_fn)
            print("Evaluation_results:\n\t%s\n" % eval_results)

    
    # plot values of variables defined in keys and plot them for each dataset

    # keys are threshold variables
    keys = ['elbo']

    # the first dataset in compare_datasets is the base distribution we use
    # to compare Out of distribution samples against. Should be the same as the training dataset
    compare_datasets = ['mnist', 'mnist', 'mnist', 'mnist', 'mnist', 'mnist', 'notMNIST', 'fashion_mnist',
                        'fashion_mnist', 'normal_noise', 'uniform_noise', 'omniglot']

    # whether to noise each dataset or not
    noised_list = [False, True, True, True, True, True, False, False, True, False, False, False]
    # if the element in noised_list is true for a dataset then what kind of noise/transformations to apply?
    # if the above element is set False, any noise/transformation will not be processed.

    noise_type_list = ['normal', 'normal', 'uniform', 'brighten', 'hor_flip', 'ver_flip', 'normal', 'normal', 'normal',
                       'normal', 'normal', 'normal']

    # whether to add adversarially perturbed noise
    # if perturbed normal noise: normal, if perturbed uniform noise: uniform , if nothing: None
    show_adv_examples = 'normal'

    #if there is a specific range to look at, add a tuple of (low, high, #of bins) for the value
    bins = {'elbo':(-2000,1000,300)}
   
    #out of the 5 models, which model to use for single analysis
    which_model = 0
    expand_last_dim = True  #for MNIST, True, for CIFAR10: False
    #FLAGS.model_dir = "gs://hyunsun/image_vae/mnist/model"

    #for single models
    #single_analysis(compare_datasets, expand_last_dim, noised_list, noise_type_list, show_adv_examples, model_fn, FLAGS.model_dir, which_model, which_model, keys, bins)  # for cifar10, attach feature_shape=(32,32,3)

    #which model to use to create adversarially perturbed noise for ensemble analysis
    adv_base = 0

    #for ensembles
    #ensemble_analysis(compare_datasets, expand_last_dim, noised_list, noise_type_list, FLAGS.batch_size, model_fn, FLAGS.model_dir, show_adv_examples, adv_base)
                                                   # for cifar10, attach feature_shape=(32,32,3)
    # history analysis
    history_compare_elbo(compare_datasets, expand_last_dim, noised_list, noise_type_list, FLAGS.batch_size, model_fn, FLAGS.model_dir, show_adv_examples, adv_base)

if __name__ == "__main__":
    tf.app.run()

