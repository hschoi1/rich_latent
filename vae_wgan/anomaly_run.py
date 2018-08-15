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
flags.DEFINE_bool(
    'noise_to_input',default=False, help="whether to add noise to training data")
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
    "viz_steps", default=1000, help="Frequency at which to save visualizations.")

flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")


FLAGS = flags.FLAGS


def main(argv):

    M = 5  # number of models in ensemble
    for i in range(M):
        params = FLAGS.flag_values_dict()
        params["activation"] = getattr(tf.nn, params["activation"])
   
        FLAGS.model_dir = "gs://hyunsun/anomaly/model%d" % i

        if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
            tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)

        train_input_fn, eval_input_fn = get_dataset('credit_card',FLAGS.batch_size)
       
        estimator = tf.estimator.Estimator(
            anomaly_model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=FLAGS.model_dir,
                save_checkpoints_steps=FLAGS.viz_steps,
            ),
        )

        for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
            estimator.train(train_input_fn, steps=FLAGS.viz_steps)
            #eval_results = estimator.evaluate(eval_input_fn)
            #print("Evaluation_results:\n\t%s\n" % eval_results)


    print("regular test set of normal data and anomalies")
    ensemble_analysis(eval_input_fn, 'anomalies')

    #normal_eval_input_fn = get_eval_dataset('normal_noise_credit_card', FLAGS.batch_size)
    #print("normal noise in place of anomalies")
    #ensemble_analysis(normal_eval_input_fn, 'normal_noise')

    #uniform_eval_input_fn = get_eval_dataset('uniform_noise_credit_card', FLAGS.batch_size)
    #print("uniform noise in place of anomalies")
    #ensemble_analysis(uniform_eval_input_fn, 'uniform_noise')




def ensemble_analysis(eval_input_fn, name):
    ensemble_elbos = []
    ensemble_posterior_means = []
    ensemble_posterior_vars = []
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])

    M = 5  # number of models in ensemble
    for i in range(M):
        FLAGS.model_dir = "gs://hyunsun/anomaly/model%d" % i

        estimator = tf.estimator.Estimator(
            anomaly_model_fn,
            params=params,
            config=tf.estimator.RunConfig(
                model_dir=FLAGS.model_dir,
                save_checkpoints_steps=FLAGS.viz_steps,
            ),
        )

        batch_results_ = list(estimator.predict(eval_input_fn,
                                                predict_keys=['elbo_local_mean', 'sigmoid', 'approx_posterior_mean',
                                                              'approx_posterior_stddev']))
        elbo_local_mean = np.array([b['elbo_local_mean'] for b in batch_results_])
        mean = np.array([b['approx_posterior_mean'] for b in batch_results_])
        stddev = np.array([b['approx_posterior_stddev'] for b in batch_results_])
        ensemble_elbos.append(elbo_local_mean)
        ensemble_posterior_means.append(mean)
        ensemble_posterior_vars.append(stddev ** 2)


    ensemble_elbos = np.array(ensemble_elbos)
    ensemble_posterior_means = np.array(ensemble_posterior_means)
    ensemble_posterior_vars = np.array(ensemble_posterior_vars)
    ensemble_elbo_mean = np.mean(ensemble_elbos, axis=0)
    ensemble_elbo_var = np.var(ensemble_elbos, axis=0)
    ensemble_posterior_means_mean = np.mean(np.mean(ensemble_posterior_means, axis=2), axis=0)
    ensemble_posterior_means_var = np.var(np.mean(ensemble_posterior_means, axis=2), axis=0)
    ensemble_posterior_vars_mean = np.mean(np.mean(ensemble_posterior_vars, axis=2), axis=0)
    ensemble_posterior_vars_var = np.var(np.mean(ensemble_posterior_vars, axis=2), axis=0)
    truth = np.concatenate([np.zeros(492), np.ones(492)])

    dics = {'elbo_local_mean':elbo_local_mean, 'ensemble_elbo_mean':ensemble_elbo_mean,'ensemble_elbo_var':ensemble_elbo_var,
            'ensemble_posterior_means_mean':ensemble_posterior_means_mean, 'ensemble_posterior_means_var':ensemble_posterior_means_var,
            'ensemble_posterior_vars_mean':ensemble_posterior_vars_mean, 'ensemble_posterior_vars_var':ensemble_posterior_vars_var}
    bins = {'elbo_local_mean': (-50.0, 50.0, 100), 'ensemble_elbo_mean': (-50.0, 50.0, 100),
            'ensemble_elbo_var': (0.0, 200, 200),
            'ensemble_posterior_means_mean': (-0.6, 0.4, 100), 'ensemble_posterior_means_var': (0, 0.5, 100),
            'ensemble_posterior_vars_mean': (0, 3, 100), 'ensemble_posterior_vars_var': (0, 0.4, 100)}

    keys = ['elbo_local_mean','ensemble_elbo_mean','ensemble_elbo_var','ensemble_posterior_means_mean',
            'ensemble_posterior_means_var','ensemble_posterior_vars_mean', 'ensemble_posterior_vars_var']
    results = [elbo_local_mean, ensemble_elbo_mean, ensemble_elbo_var, ensemble_posterior_means_mean,
               ensemble_posterior_means_var, ensemble_posterior_vars_mean, ensemble_posterior_vars_var]


    plot_analysis(results, ['valid', name], keys, bins=bins, each_size=492)

if __name__ == "__main__":
    tf.app.run()

