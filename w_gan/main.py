from __future__ import print_function
from absl import flags
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from model import *
from get_data import *


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
       FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)


flags.DEFINE_integer(
    "max_steps", default=20001, help="note the one step indicates that "
                                     "a Citers updates of critic and one update of generator")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=64,
    help="Batch size.")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_integer(
    "z_dim", default=128, help="dimension of latent z")
flags.DEFINE_float(
    "learning_rate_ger",
    default=5e-5,
    help="learning rate of the generator")
flags.DEFINE_float(
    "learning_rate_dis",
    default=5e-5,
    help="learning rate of the discriminator")
flags.DEFINE_integer(
    "img_size",
    default=32,
    help="size of the input image")
flags.DEFINE_integer(
    "channel",
    default=1,
    help="channel = 3 if is_svhn is True else 1")
flags.DEFINE_integer(
    "ngf",
    default=32,
    help="hidden layer size if mlp is chosen, ignore if otherwise")
flags.DEFINE_integer(
    "ndf",
    default=32,
    help="hidden layer size if mlp is chosen, ignore if otherwise")
flags.DEFINE_integer(
    "Citers",
    default=5,
    help="update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)")
flags.DEFINE_float(
    "clamp_lower",
    default=-0.01,
    help="the lower bound of parameters in critic")
flags.DEFINE_float(
    "clamp_upper",
    default=0.01,
    help="the upper bound of parameters in critic")
flags.DEFINE_bool(
    "is_mlp",
    default=False,
    help="where to use mlp or dcgan structure")
flags.DEFINE_bool(
    "is_adam",
    default=False,
    help="whether to use adam for parameter update, if the flag is set False"
         " use tf.train.RMSPropOptimizer as recommended in paper")
flags.DEFINE_bool(
    "is_svhn",
    default=False,
    help="whether to use SVHN or MNIST, set false and MNIST is used")
flags.DEFINE_string(
    "mode",
    default='gp',
    help="'gp' for gp WGAN and 'regular' for vanilla")
flags.DEFINE_float(
    "lam",
    default=10.,
    help="only when the mode is gp")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join('jeju', "/w_gan"),
    help="Directory to put the model's fit.")

flags.DEFINE_integer(
    "viz_steps", default=1000, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS



def main(argv):
    ensemble = []
    M = 5  # number of models in ensemble
    for i in range(M):
        params = FLAGS.flag_values_dict()
        params["activation"] = getattr(tf.nn, params["activation"])

        FLAGS.model_dir = "gs://hyunsun/w_gan/mnist/model%d" % i

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

        for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
            estimator.train(train_input_fn, steps=FLAGS.viz_steps)
            #eval_results = estimator.evaluate(eval_input_fn)
            #print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
    tf.app.run()

