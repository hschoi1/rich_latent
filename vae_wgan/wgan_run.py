# referred to  https://github.com/Zardinality/WGAN-tensorflow/blob/master/WGAN.ipynb
# both on MNIST/FashionMNIST and CIFAR10

from __future__ import print_function
import tensorflow as tf
import pdb
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import tensorflow.contrib.layers as ly
import functools
from functools import partial
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

flags = tf.app.flags
FLAGS = tf.flags.FLAGS


flags.DEFINE_integer(
    "max_steps", default=20001,
    help="Number of generator updates. one generator update followed by 'Citers' times critic updates.")
flags.DEFINE_integer(
    "extra_steps", default=4001, help="Number of extra training steps for a discriminative model.")
flags.DEFINE_integer(
    "z_dim", default=128, help="dimension of latent z")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=64,
    help="Batch size.")

flags.DEFINE_float(
    "learning_rate_ger",
    default=5e-5,
    help="learning rate of the generator")
flags.DEFINE_float(
    "learning_rate_dis",
    default=5e-5,
    help="learning rate of the discriminator")

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
    "is_adam",
    default=False,
    help="whether to use adam for parameter update, if the flag is set False"
         " use tf.train.RMSPropOptimizer as recommended in paper")
flags.DEFINE_string(
    "mode",
    default='gp',
    help="'gp' for gp WGAN and 'regular' for vanilla")
flags.DEFINE_float(
    "lam",
    default=10.,
    help="only when the mode is gp")

flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"),
    help="Directory to put the model's fit.")

flags.DEFINE_bool('skip_train', default=False, help='Whether to skip training the model ensembles and go straight to analysis.')

flags.DEFINE_string(
    "train_dataset",
    default="mnist",
    help="mnist/fashion_mnist for VAE, mnist/fashion_mnist/cifar10 for wgan")


def make_critic(activation, latent_size, base_depth):

    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    critic_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_size, activation=None),
    ])

    def critic(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = critic_net(images)
        logit = ly.fully_connected(tf.reshape(
            net, [-1, latent_size]), 1, activation_fn=None)
        return logit

    return critic

def make_generator(activation, latent_size, output_shape, base_depth):

    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)

    # first filter has size 7 for mnist, 8 for cifar (32x32)
    filter_width = 7 if output_shape[0] == 28 else 8

    generator_net = tf.keras.Sequential([
        deconv(2 * base_depth, filter_width, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(output_shape[-1], 5, activation=None),
    ])

    def generator(codes):
        original_shape = tf.shape(codes)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        codes = tf.reshape(codes, (-1, 1, 1, latent_size))
        logits = generator_net(codes)
        logits = tf.reshape(
            logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        return logits

    return generator



tf.reset_default_graph()
### start graph
if FLAGS.train_dataset == "cifar10":
    features = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x')
else:
    features = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x')
params = FLAGS.flag_values_dict()
params["activation"] = getattr(tf.nn, params["activation"])

batch_shape = tf.shape(features)[0]
noise_dist = tf.contrib.distributions.Normal(0., 1.)
z = noise_dist.sample((batch_shape, params["z_dim"]))
critic = make_critic(params["activation"],
                       params["z_dim"],
                       params["base_depth"])

image_shape = features.get_shape().as_list()[1:]

generator = make_generator(params["activation"],
                       params["z_dim"],
                       image_shape,
                       params["base_depth"])


with tf.variable_scope('generator'):
    train = generator(z)   #generated images

with tf.variable_scope('critic'):
    true_logit = critic(features)


# adversarially perturbed noise wrt true logit
grad, = tf.gradients(true_logit, features)
normal_noise = tfd.Normal(loc=tf.zeros([]), scale=tf.ones([]))
uniform_noise = tfd.Uniform(0., 1.)
normal_noise_samples = normal_noise.sample(sample_shape=tf.shape(features))
uniform_noise_samples = uniform_noise.sample(sample_shape=tf.shape(features))
adversarial_normal_noise = normal_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.
adversarial_uniform_noise = uniform_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.


with tf.variable_scope('critic', reuse=True):
    fake_logit = critic(train)

c_loss = tf.reduce_mean(fake_logit - true_logit) #the sign should not matter as the norm is symmetric

if params['mode'] is 'gp':
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((batch_shape, 1, 1, 1))
    interpolated = features + alpha * (train - features)
    with tf.variable_scope('critic', reuse=True):
      inte_logit = critic(interpolated)
    gradients = tf.gradients(inte_logit, [interpolated, ])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
    #gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
    #grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
    c_loss += params['lam'] * gradient_penalty

g_loss = tf.reduce_mean(-fake_logit)

tf.summary.scalar("g_loss", g_loss)
tf.summary.scalar("c_loss", c_loss)
tf.summary.image("img", train, max_outputs=8)
theta_g = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
theta_c = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

# to keep track of # of updates
counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
extra_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
global_step = tf.train.get_or_create_global_step()


opt_c = ly.optimize_loss(loss=c_loss, learning_rate=params['learning_rate_dis'],
                         optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                           beta2=0.9) if params[
                                                             'is_adam'] is True else tf.train.RMSPropOptimizer,
                         variables=theta_c, global_step=counter_c,

                         summaries=['gradient_norm'])


opt_g = ly.optimize_loss(loss=g_loss, learning_rate=params['learning_rate_ger'],
                         optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                           beta2=0.9) if params[
                                                             'is_adam'] is True else tf.train.RMSPropOptimizer,
                         variables=theta_g, global_step=counter_g,
                         summaries=['gradient_norm'])


if params['mode'] is 'regular':   #not used in our experiments
    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, params['clamp_lower'], params['clamp_upper'])) for var in theta_c]
    # merge the clip operations on critic variables
    with tf.control_dependencies([opt_c]):
        opt_c = tf.tuple(clipped_var_c)

if not params['mode'] in ['gp', 'regular']:
    raise (NotImplementedError('Only two modes'))



# after done training wgan, train critic a few more steps to make it a discriminative classifier
discrim_predictions = tf.nn.sigmoid(true_logit)
discrim_logits = tf.concat([true_logit, fake_logit], 0)
discrim_labels = tf.concat([tf.ones_like(true_logit), tf.zeros_like(fake_logit)], 0)
discrim_c_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=discrim_labels, logits=discrim_logits)
discrim_loss = tf.reduce_mean(discrim_c_loss)
tf.summary.scalar("discrim_loss", discrim_loss)


extra_train_op = ly.optimize_loss(loss=discrim_loss, learning_rate=params['learning_rate_dis'],
                                  optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                                    beta2=0.9) if params[
                                                                      'is_adam'] is True else tf.train.RMSPropOptimizer,
                                  variables=theta_c, global_step=extra_counter,
                                  summaries=['gradient_norm'])

tf.summary.scalar("critic counter", counter_c)
tf.summary.scalar("generator counter", counter_g)

summ_op = tf.summary.merge_all()

### end of graph



# data preparation
if FLAGS.train_dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
elif FLAGS.train_dataset == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
elif FLAGS.train_dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # if train on cifar10, change to mnist to cifar10

x_train = x_train.astype(np.float32) / 255.
def next_feed_dict():
    choice = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    train_img = x_train[choice]
    feed_dict = {features: train_img}
    return feed_dict



#FLAGS.model_dir = 'gs://hyunsun/w_gan/mnist/model'

### TRAINING
if not FLAGS.skip_train:
    for model_num in range(5):
        with tf.Session() as sess:

            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.model_dir+str(model_num), sess.graph)

            sess.run(tf.global_variables_initializer())

            for g_update in range(FLAGS.max_steps):
                if g_update % 5000 == 0:
                    saver.save(sess, FLAGS.model_dir+str(model_num)+'/model.ckpt', global_step=g_update)

                if (g_update < 25) or (g_update % 500 == 0):
                    citers = 100
                else:
                    citers = 5

                # update critic
                for i in range(citers):
                    feed_dict = next_feed_dict()
                    fetch_dict={'train_op': opt_c, 'loss': c_loss}
                    results = sess.run(fetch_dict, feed_dict)

                # update generator
                feed_dict = next_feed_dict()
                fetch_dict = {'train_op': opt_g, 'loss': g_loss,
                              'counter_g': counter_g, 'counter_c': counter_c}
                results = sess.run(fetch_dict, feed_dict)

                if g_update % 500 == 0:
                    print('\n')
                    print('counter_c', results['counter_c'], 'counter_g', results['counter_g'])
                    print('\n')
                    summ = sess.run(summ_op, feed_dict)
                    summary_writer.add_summary(summ, g_update)
    ### END OF TRAINING


    ### EXTRA TRAINING
    for model_num in range(5):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir+str(model_num)+'/'))
            summary_writer = tf.summary.FileWriter(FLAGS.model_dir + str(model_num) + '/extra')

            for extra_step in range(FLAGS.extra_steps):
                if extra_step % 1000 == 0:
                    saver.save(sess, FLAGS.model_dir+str(model_num)+'/extra/model.ckpt', global_step=extra_step)

                feed_dict = next_feed_dict()
                fetch_dict = {'train_op': extra_train_op, 'loss': discrim_loss, 'extra_counter': extra_counter}
                results = sess.run(fetch_dict, feed_dict)

                if extra_step % 100 == 0:
                    print(results['loss'], results['extra_counter'])
                    summ = sess.run({'sum': summ_op}, feed_dict)
                    summary_writer.add_summary(summ['sum'], extra_step)

    ### END OF EXTRA TRAINING


### ANALYSIS

from tools.get_data import build_eval_multiple_datasets2
from tools.statistics import plot_analysis

# the first dataset in dataset_list is the base distribution we use
# to compare Out of distribution samples against. Should be the same as the training dataset
if FLAGS.train_dataset == "mnist":
    dataset_list = [tf.keras.datasets.mnist, 'omniglot', 'notMNIST',
                    tf.keras.datasets.fashion_mnist, tf.keras.datasets.mnist, tf.keras.datasets.mnist,
                    'uniform_noise', 'normal_noise']
    datasets_names = ['mnist',  'omniglot', 'notMNIST', 'fashion_mnist',
                      'mnist nsd by hor_flip', 'mnist nsd by ver_flip', 'uniform_noise','normal_noise']


elif FLAGS.train_dataset == "fashion_mnist":
    dataset_list = [tf.keras.datasets.fashion_mnist, 'omniglot', 'notMNIST',
                    tf.keras.datasets.mnist, tf.keras.datasets.fashion_mnist, tf.keras.datasets.fashion_mnist,
                    'uniform_noise', 'normal_noise']
    datasets_names = ['fashion_mnist', 'omniglot', 'notMNIST', 'mnist', 'fashion_mnist nsd by hor_flip',
                      'fashion_mnist nsd by ver_flip', 'uniform_noise', 'normal_noise']

elif FLAGS.train_dataset == "cifar10":
    dataset_list = [tf.keras.datasets.cifar10, 'celebA', 'SVHN', 'ImageNet','uniform_noise','normal_noise',
                    tf.keras.datasets.cifar10,tf.keras.datasets.cifar10]
    datasets_names = ['cifar10', 'celebA', 'SVHN', 'ImageNet', 'uniform_noise', 'normal noise',
                      'cifar10 nsd by hor_flip', 'cifar10 Vflip nsd by ver_flip']


# construct a np array of all the above datasets, each of which has 1000 samples.
if (FLAGS.train_dataset == "mnist") or (FLAGS.train_dataset == "fashion_mnist"):
    noised_list = [False, False, False, False, True, True, False, False]
    noise_type_list = ['normal', 'normal', 'normal', 'normal', 'hor_flip', 'ver_flip', 'normal', 'normal']
    eval_data = build_eval_multiple_datasets2(dataset_list, expand_last_dim=True, noised_list=noised_list,
                                   noise_type_list=noise_type_list, feature_shape=(28,28), each_size=1000)

elif FLAGS.train_dataset == "cifar10":
    noised_list = [False, False, False, False, False, False, True, True]
    noise_type_list = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'hor_flip', 'ver_flip']
    eval_data = build_eval_multiple_datasets2(dataset_list, expand_last_dim=False, noised_list=noised_list,
                                          noise_type_list=noise_type_list, feature_shape=(32, 32, 3), each_size=1000)





# for plotting labels
keys = ['single_logits','logits_ens_mean','logits_ens_var', 'WAIC']
show_adv = True

### get adversarial examples from model0
if show_adv:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir+'0/extra/'))
        feed_dict = {features: eval_data[:1000]}   # eval_data[:1000] is mnist
        fetch_dict = {'adv_normal':adversarial_normal_noise, 'adv_uniform': adversarial_uniform_noise}
        results = sess.run(fetch_dict, feed_dict)
        ## attach adversarial examples to eval_data for evaluation
        eval_data = np.concatenate([eval_data, results['adv_normal'], results['adv_uniform']],axis=0)
        datasets_names.append('adv_normal')
        datasets_names.append('adv_uniform')


### Iterate through all the extra trained models and collect logits
logits_list = []

for model_num in range(5):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir+str(model_num)+'/extra/'))
        '''
        # check if the critic is classifying well
        feed_dict = {features: eval_data[:1000]}
        fetch_dict = {'discrim_logits':discrim_logits}
        fetched_results = sess.run(fetch_dict, feed_dict)
        discrim_results = fetched_results['discrim_logits']
        discrim_probs = np.exp(discrim_results)/(1+np.exp(discrim_results))
        discrim_preds = (discrim_probs > 0.5)
        truth = np.concatenate([np.ones(1000), np.zeros(1000)])
        truth = np.expand_dims(truth, axis=-1)
        correct = sum(discrim_preds == truth)
        print(correct)   # how many correct classifications out of 2000
        '''

        feed_dict = {features: eval_data}
        fetch_dict = {'true_logit': true_logit}
        fetched_results = sess.run(fetch_dict, feed_dict)
        logits = fetched_results['true_logit']
    logits_list.append(logits)

logits_mean = np.mean(logits_list, axis=0)
logits_var = np.var(logits_list, axis=0)
waic = logits_mean - logits_var
results = [logits_list[0], logits_mean, logits_var, waic]
extra_trained = plot_analysis(results, datasets_names, keys, bins=None, each_size=1000)
# extra_trained is an array of AUROC of size (3, num of datasets (maybe including adversarial examples))
# extra_trained[0] is the scores of datasets for single critic logit, [1] for ensmeble mean of the logits
# [2] is for the ensemble variance of the logits


### Iterate through all the not extra trained models and collect logits
'''
not_extra_trained_logits_list = []

for model_num in range(5):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_dir + str(model_num) + '/extra/model.ckpt-0')
        feed_dict = {features: eval_data}
        fetch_dict = {'true_logit': true_logit}
        fetched_results = sess.run(fetch_dict, feed_dict)
        logits = fetched_results['true_logit']
    not_extra_trained_logits_list.append(logits)

not_extra_trained_logits_mean = np.mean(not_extra_trained_logits_list, axis=0)
not_extra_trained_logits_var = np.var(not_extra_trained_logits_list, axis=0)
not_extra_trained_results = [not_extra_trained_logits_list[0], not_extra_trained_logits_mean, not_extra_trained_logits_var]
not_extra_trained = plot_analysis(not_extra_trained_results, datasets_names, keys, bins=None, each_size=1000)
# not_extra_trained is an array of AUROC of size (3, num of OOD datasets (maybe including adversarial examples))
# not_extra_trained[0] is the scores of datasets for single critic logit, [1] for ensmeble mean of the logits
# [2] is for the ensemble variance of the logits

ind = np.arange(len(not_extra_trained[0]))
fig, ax = plt.subplots(1,3,figsize=(15,5))
rects1 = ax[0].bar(ind, extra_trained[0], width=0.3) # auroc of single logits
rects2 = ax[0].bar(ind+0.3, not_extra_trained[0], width=0.3)
ax[0].set_ylabel('AUROC')
ax[0].set_xlabel('single_logit')
#ax[0].set_xticks(ind + 0.3 / 2)
#ax[0].set_xticklabels(datasets_names)
ax[0].legend((rects1[0], rects2[0]), ['extra_trained', 'not_extra_trained'])
rects1 = ax[1].bar(ind, extra_trained[1], width=0.3)  # auroc of ensemble mean
rects2 = ax[1].bar(ind+0.3, not_extra_trained[1], width=0.3)
ax[1].set_ylabel('AUROC')
ax[1].set_xlabel('ensemble_logit_mean')
#ax[1].set_xticks(ind + 0.3 / 2)
#ax[1].set_xticklabels(datasets_names)
ax[1].legend((rects1[0], rects2[0]), ['extra_trained', 'not_extra_trained'])
rects1 = ax[2].bar(ind, extra_trained[2], width=0.3)  # auroc of ensemble var
rects2 = ax[2].bar(ind+0.3, not_extra_trained[2], width=0.3)
ax[2].set_ylabel('AUROC')
ax[2].set_xlabel('ensemble_logit_var')
#ax[2].set_xticks(ind + 0.3 / 2)
#ax[2].set_xticklabels(datasets_names)
ax[2].legend((rects1[0], rects2[0]), ['extra_trained', 'not_extra_trained'])

fig.savefig(os.path.join(FLAGS.model_dir, "auroc_wgans.eps"), format='eps', dpi=1000)


#if want a different version of the same plot where each subplot represents each dataset

ind = np.arange(2)
fig, ax = plt.subplots(1,len(not_extra_trained[0]),figsize=(5*len(not_extra_trained[0]),5))
for i in range(len(not_extra_trained[0])):
    rects1 = ax[i].bar(ind[0], extra_trained[1][i], width=0.3, color='b')  # auroc of ensemble mean
    rects2 = ax[i].bar(ind[0] + 0.3, not_extra_trained[1][i], width=0.3, color='y')
    rects1 = ax[i].bar(ind[1], extra_trained[2][i], width=0.3, color='b')  # auroc of ensemble var
    rects2 = ax[i].bar(ind[1] + 0.3, not_extra_trained[2][i], width=0.3, color='y')
    ax[i].set_ylabel('AUROC')
    ax[i].set_xticks(ind+0.5)
    ax[i].set_xticklabels(['ensemble_logit_mean     ensmeble_logit_var'])
    ax[i].legend((rects1[0], rects2[0]), ['extra_trained', 'not_extra_trained'])
    ax[i].set_title(datasets_names[i+1])  # exclude the base distribuiton (just plain mnist)

fig.savefig(os.path.join(FLAGS.model_dir, "auroc_wgans_version2.eps"), format='eps', dpi=1000)
'''