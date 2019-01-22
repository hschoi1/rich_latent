from __future__ import print_function
import tensorflow as tf
import pdb
import os
import tensorflow.contrib.layers as ly
import functools
import matplotlib.pyplot as plt
import pickle
from functools import partial
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

flags = tf.app.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_integer(
    "max_steps", default=5001,
    help="Number of generator updates. one generator update followed by 'Citers' times critic updates.")
flags.DEFINE_integer(
    "extra_steps", default=10001, help="Number of extra training steps for a discriminative model.")
flags.DEFINE_integer(
    "z_dim", default=4, help="dimension of latent z")

flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=128,
    help="Batch size.")

flags.DEFINE_float(
    "learning_rate_ger",
    default=5e-4,
    help="learning rate of the generator")
flags.DEFINE_float(
    "learning_rate_dis",
    default=5e-4,
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


def make_critic(activation):
    critic_net = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=activation),
        tf.keras.layers.Dense(5, activation=activation),
    ])
    def critic(images):
        net = critic_net(images)
        logit = ly.fully_connected(net, 1, activation_fn=None)
        return logit
    return critic

def make_generator(activation):
    generator_net = tf.keras.Sequential([
        #tf.keras.layers.Dense(10, activation=activation),
        tf.keras.layers.Dense(10, activation=None),
        tf.keras.layers.Dense(2, activation=None)
    ])
    def generator(codes):
        logits = generator_net(codes)
        return logits
    return generator


tf.reset_default_graph()
### start graph
features = tf.placeholder(dtype=tf.float32, shape=[None,2], name='x')
params = FLAGS.flag_values_dict()
params["activation"] = getattr(tf.nn, params["activation"])

batch_shape = tf.shape(features)[0]
noise_dist = tf.contrib.distributions.Normal(0., 1.)
z = noise_dist.sample((batch_shape, params["z_dim"]))
critic = make_critic(params["activation"])

generator = make_generator(params["activation"])

with tf.variable_scope('generator'):
    train = generator(z)  # generated images

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

c_loss = tf.reduce_mean(fake_logit - true_logit)  # the sign should not matter as the norm is symmetric

if params['mode'] is 'gp':
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((batch_shape, 1, 1, 1))
    interpolated = features + alpha * (train - features)
    with tf.variable_scope('critic', reuse=True):
        inte_logit = critic(interpolated)
    gradients = tf.gradients(inte_logit, [interpolated, ])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
    # gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
    # grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
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

if params['mode'] is 'regular':  # not used in our experiments
    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, params['clamp_lower'], params['clamp_upper'])) for var in
                     theta_c]
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

extra_train_op = ly.optimize_loss(loss=discrim_loss, learning_rate=1e-4,
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

x_train = np.random.normal(loc=[5.,5.], scale=[.5,.5], size=(50000,2))

def next_feed_dict():
    choice = np.random.choice(x_train.shape[0], FLAGS.batch_size)
    train_batch = x_train[choice]
    feed_dict = {features: train_batch}
    return feed_dict


### TRAINING

for model_num in range(4):
    with tf.Session() as sess:

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for g_update in range(FLAGS.max_steps):
            citers = 5

            # update critic
            for i in range(citers):
                feed_dict = next_feed_dict()
                fetch_dict = {'train_op': opt_c, 'loss': c_loss}
                results = sess.run(fetch_dict, feed_dict)

            # update generator
            feed_dict = next_feed_dict()
            fetch_dict = {'train_op': opt_g, 'loss': g_loss, 'generated': train,
                          'counter_g': counter_g, 'counter_c': counter_c, 'features':features}
            results = sess.run(fetch_dict, feed_dict)

            if g_update % 2500 == 0:
                saver.save(sess, FLAGS.model_dir + str(model_num) + '/model.ckpt', global_step=g_update)
                print('\n')
                print('counter_c', results['counter_c'], 'counter_g', results['counter_g'], results['loss'])
                print('\n')
                fake_scatter = results['generated']
                true_scatter = results['features']
                f, axes = plt.subplots(1, 2, figsize=(10,5),sharex='row',sharey='row')
                axes[0].scatter(fake_scatter[:,0], fake_scatter[:,1])
                axes[1].scatter(true_scatter[:,0], true_scatter[:,1])
                plt.show()

### END OF TRAINING



### EXTRA TRAINING
for model_num in range(4):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir + str(model_num) + '/'))

        for extra_step in range(FLAGS.extra_steps):
            if extra_step % 10000 == 0:
                saver.save(sess, FLAGS.model_dir + str(model_num) + '/extra/model.ckpt', global_step=extra_step)

            feed_dict = next_feed_dict()
            fetch_dict = {'train_op': extra_train_op, 'loss': discrim_loss, 'extra_counter': extra_counter}
            results = sess.run(fetch_dict, feed_dict)

            if extra_step % 500 == 0:
                print(results['loss'], results['extra_counter'])

                # check if the critic is classifying well
                feed_dict = {features: x_train[np.random.choice(50000,1000)]}
                fetch_dict = {'discrim_logits':discrim_logits}
                fetched_results = sess.run(fetch_dict, feed_dict)

                discrim_results = fetched_results['discrim_logits']
                discrim_probs = np.exp(discrim_results)/(1+np.exp(discrim_results))
                discrim_preds = (discrim_probs > 0.5)
                truth = np.concatenate([np.ones(1000), np.zeros(1000)])
                truth = np.expand_dims(truth, axis=-1)
                correct = sum(discrim_preds == truth)
                print(correct)  # how many correct classifications out of 2000
### END OF EXTRA TRAINING



#PLOT
# axes[0] is a plot of the critic output after extra training
# axes[1] is a plot of the |fake_logit - true_logit| (e.g Wasserstein distance) without extra training
# axes[2] is a plot of generated images
# first 4 columns correspond to each single model. The last column is the ensemble mean of them.

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
whole = np.zeros((400,2))
z = np.zeros((20,20))
for i in range(20):
    for j in range(20):
        whole[20 * i + j][0] = x[i]
        whole[20 * i + j][1] = y[j]

ensemble = []
generated_collect = []
# plot extra trained
for model_num in range(4):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir + str(model_num)+ '/extra/'))
        feed_dict = {features: whole}
        fetch_dict = {'discrim_predictions': discrim_predictions, 'train':train}
        results = sess.run(fetch_dict, feed_dict)
        values = results['discrim_predictions']

        z = np.reshape(values, (20,20), order='C')
        ensemble.append(z)

        generated = results['train']     #collected generated data for plot
        generated_collect.append(generated)


fig, axes = plt.subplots(3,5,figsize=(25,15), sharex='all',sharey='all')
X, Y = np.meshgrid(x, y)
for i in range(4):  # each single model
    axes[0,i].contourf(X, Y, ensemble[i], 20, cmap=plt.cm.bone)
    axes[2,i].scatter(generated_collect[i][:, 0], generated_collect[i][:, 1], alpha=0.1, label=str(i))
z=np.mean(ensemble,axis=0)  # if want to plot variance, change this to np.var
generated_mean = np.mean(generated_collect, axis=0)
axes[0,4].contourf(X, Y, z, 20, cmap=plt.cm.bone)
axes[2,4].scatter(generated_mean[:, 0], generated_mean[:, 1], alpha=0.1)

#plot not extra trained wgan
ensemble = []
for model_num in range(4):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_dir + str(model_num) + '/extra/model.ckpt-0')
        feed_dict = {features: whole}
        fetch_dict = {'fake': fake_logit, 'true':true_logit}
        results = sess.run(fetch_dict, feed_dict)
        values = abs(results['fake'] - results['true'])
        z = np.reshape(values, (20,20), order='C')
        ensemble.append(z)

# plot not extra trained wgan distance
X, Y = np.meshgrid(x, y)
for i in range(4):
    axes[1,i].contourf(X, Y, ensemble[i], 20, cmap=plt.cm.bone)
z=np.mean(ensemble,axis=0)
axes[1,4].contourf(X, Y, z, 20, cmap=plt.cm.bone)

fig.savefig('wgan_heatmap.eps', format='eps', dpi=1000)
plt.show()