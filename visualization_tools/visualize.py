import tensorflow as tf
import numpy as np
from absl import flags
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from model import *

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

def main(argv):
    FLAGS.model_dir = "gs://hyunsun/image_vae/mnist/model0"
    params["activation"] = getattr(tf.nn, params["activation"])

    eval_input_fn_list = visualize_mnist_dataset()
    posterior = []
    prior = []

    adv_posterior = []
    adv_prior = []

    for eval_input_fn in eval_input_fn_list:

        estimator = tf.estimator.Estimator(
                    model_fn,
                    params=params,
                    config=tf.estimator.RunConfig(
                        model_dir=FLAGS.model_dir,
                    )
                )

        batch_results_ = list(
            estimator.predict(eval_input_fn,
                              predict_keys=['visualized_posterior_samples', 'inversed_posterior_samples',
                                            'adv_visualized_posterior_samples','adv_inversed_posterior_samples']))
        visualized_posterior_samples = np.array([b['visualized_posterior_samples'] for b in batch_results_])
        inversed_posterior_samples = np.array([b['inversed_posterior_samples'] for b in batch_results_])
        adv_visualized_posterior_samples = np.array([b['adv_visualized_posterior_samples'] for b in batch_results_])
        adv_inversed_posterior_samples = np.array([b['adv_inversed_posterior_samples'] for b in batch_results_])

        # collect samples from each digit from normal input
        posterior.append(visualized_posterior_samples)
        prior.append(inversed_posterior_samples)

        # collect samples from each digit from adversarially perturbed noise
        adv_posterior.append(adv_visualized_posterior_samples)
        adv_prior.append(adv_inversed_posterior_samples)



    # visualize latent of normal digits with color

    f, arr = plt.subplots(1, 2, figsize=(10, 5))
    colors = cm.rainbow(np.linspace(0, 1, 10))

    for num, c in zip(range(10),colors):
        X1 = posterior[num]
        arr[0].scatter(X1[:, :, 0], X1[:, :, 1], s=1, color=c, label=num, alpha=0.5)

        X2 = prior[num]
        arr[1].scatter(X2[:, :, 0], X2[:, :, 1], s=1, color=c, label=num, alpha=0.5)


    plt.legend()

    f.savefig("gs://hyunsun/image_vae/mnist/normal_latent.png")

    # visualize latent of adversarial noise examples with color
    f, arr = plt.subplots(1, 2, figsize=(10, 5))
    colors = cm.rainbow(np.linspace(0, 1, 10))

    for num, c in zip(range(10), colors):
        X1 = adv_posterior[num]
        arr[0].scatter(X1[:, :, 0], X1[:, :, 1], s=1, color=c, label=num, alpha=0.5)


        X2 = adv_prior[num]
        arr[1].scatter(X2[:, :, 0], X2[:, :, 1], s=1, color=c, label=num, alpha=0.5)


    plt.legend()

    f.savefig("gs://hyunsun/image_vae/mnist/adversarial_noise_latent.png")


# split eval dataset into each digit. 100 input for each digit.
# however we sample 100 points from posterior dist per input, so 10000 points will be plotted per digit.
def visualize_mnist_dataset():
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = np.expand_dims(x_test, axis=-1)
    x_test = x_test.astype(np.float32) / 255.

    numbers = []
    for i in range(10):
        each_digit = [y_test== i]
        each_digit_y = y_test[each_digit][:100]
        each_digit_x = x_test[each_digit][:100]
        #eval_dataset = tf.data.Dataset.from_tensor_slices((each_digit_x, each_digit_y)).batch(FLAGS.batch_size)
        eval_dataset = tf.data.Dataset.from_tensor_slices((each_digit_x, each_digit_y)).batch(100)
        eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()
        numbers.append(eval_input_fn)
    return numbers

if __name__ == "__main__":
    tf.app.run()

