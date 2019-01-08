from absl import flags
import numpy as np
import tensorflow as tf
from tools.get_data import *
import os
from tools.statistics import *
from anomaly.model import anomaly_model_fn

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
       FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

flags.DEFINE_float(
    "learning_rate", default=0.0001, help="Initial learning rate.")
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
    "viz_steps", default=1000, help="Frequency at which to save visualizations.")

flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

flags.DEFINE_bool('skip_train', default=False, help='Whether to skip training the model ensembles and go straight to analysis.')

FLAGS = flags.FLAGS


def main(argv):

    M = 5  # number of models in ensemble
    for i in range(M):
        params = FLAGS.flag_values_dict()
        params["activation"] = getattr(tf.nn, params["activation"])
   
        model_dir = FLAGS.model_dir + str(i)

        if FLAGS.delete_existing and tf.gfile.Exists(model_dir):
            tf.logging.warn("Deleting old log directory at {}".format(model_dir))
            tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.MakeDirs(model_dir)

        train_input_fn, eval_input_fn = get_dataset('credit_card',FLAGS.batch_size)
       
        estimator = tf.estimator.Estimator(
            anomaly_model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=model_dir,
                save_checkpoints_steps=FLAGS.viz_steps,
            ),
        )
        if not FLAGS.skip_train:
            for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
                estimator.train(train_input_fn, steps=FLAGS.viz_steps)
                #eval_results = estimator.evaluate(eval_input_fn)
                #print("Evaluation_results:\n\t%s\n" % eval_results)


    # plot values of variables defined in keys and plot them for each dataset

    # keys are threshold variables
    keys = ['elbo']
    compare_datasets = ['credit_card_normal', 'credit_card_anomalies', 'normal_noise', 'uniform_noise']
    # whether to noise each dataset or not
    noised_list = [False, False, False, False]
    # if the element in noised_list is true for a dataset then what kind of noise/transformations to apply?
    # if the above element is set False, any noise/transformation will not be processed.

    noise_type_list = ['normal', 'normal', 'normal', 'normal']

    # whether to add adversarially perturbed noise
    # if perturbed normal noise: normal, if perturbed uniform noise: uniform , if nothing: None
    show_adv_examples = None

    # if there is a specific range to look at, add a tuple of (low, high, #of bins) for the value
    bins = {'elbo': (-200, 0, 100)}

    # out of the 5 models, which model to use for analysis
    which_model = 0
    expand_last_dim = False

    #single_analysis(compare_datasets, expand_last_dim, noised_list, noise_type_list, show_adv_examples, anomaly_model_fn,
    #         FLAGS.model_dir, which_model,
    #         which_model, keys, bins, feature_shape=(30,), each_size=492)

    #which model to use to create adversarially perturbed noise for ensemble analysis
    adv_base = 0
    ensemble_OoD(compare_datasets, expand_last_dim, noised_list, noise_type_list, FLAGS.batch_size,
                 anomaly_model_fn, FLAGS.model_dir, show_adv_examples, adv_base, feature_shape=(30,), each_size=492)

if __name__ == "__main__":
    tf.app.run()

