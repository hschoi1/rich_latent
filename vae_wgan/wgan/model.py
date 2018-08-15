# referred to  https://github.com/Zardinality/WGAN-tensorflow/blob/master/WGAN.ipynb

from __future__ import print_function
from absl import flags
import tensorflow as tf
import pdb
import tensorflow.contrib.layers as ly
import functools
from functools import partial
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
FLAGS = flags.FLAGS


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def generator_conv(z):
    train = ly.fully_connected(
        z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4, 512))
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, FLAGS.channel, 3, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    print(train.name)
    return train


def critic_conv(img, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        logit = ly.fully_connected(tf.reshape(
            img, [FLAGS.batch_size, -1]), 1, activation_fn=None)
    return logit

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
    """Create the decoder function.
    Args:
      activation: Activation function to use.
      latent_size: Dimensionality of the encoding.
      output_shape: The output image shape.
      base_depth: Smallest depth for a layer.
    Returns:
      decoder: A `callable` mapping a `Tensor` of encodings to a
        `tf.distributions.Distribution` instance over images.
    """
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



#change to Estimator grammar


def model_fn(features, labels, mode, params, config):
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    del labels, config
    predictions = {}

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
        train = generator(z)
    with tf.variable_scope('critic'):
        true_logit = critic(features)
    predictions['true_logit'] = true_logit


    if mode == tf.estimator.ModeKeys.PREDICT:
        # adversarial perturbation to random noise
        grad, = tf.gradients(true_logit, features)
        normal_noise = tfd.Normal(loc=tf.zeros([]), scale=tf.ones([]))
        uniform_noise = tfd.Uniform(0., 1.)
        normal_noise_samples = normal_noise.sample(sample_shape=tf.shape(features))
        uniform_noise_samples = uniform_noise.sample(sample_shape=tf.shape(features))
        adversarial_normal_noise = normal_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.
        adversarial_uniform_noise = uniform_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.
        predictions['adversarial_normal_noise'] = adversarial_normal_noise
        predictions['adversarial_uniform_noise'] = adversarial_uniform_noise


    with tf.variable_scope('critic', reuse=True):
        fake_logit = critic(train)
    predictions['fake_logit'] = fake_logit
    c_loss = tf.reduce_mean(fake_logit - true_logit)
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
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    img_sum = tf.summary.image("img", train, max_outputs=4)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    i = tf.train.get_global_step()
    j = tf.train.get_global_step()
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=params['learning_rate_ger'],
                             optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                               beta2=0.9) if params['is_adam'] is True else tf.train.RMSPropOptimizer,
                             variables=theta_g, global_step=j,
                             summaries=['gradient_norm'])
    #counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c = ly.optimize_loss(loss=c_loss, learning_rate=params['learning_rate_dis'],
                             optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                               beta2=0.9) if params['is_adam'] is True else tf.train.RMSPropOptimizer,
                             variables=theta_c, global_step=i,
                             summaries=['gradient_norm'])
    if params['mode'] is 'regular':
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, params['clamp_lower'], params['clamp_upper'])) for var in theta_c]
        # merge the clip operations on critic variables
        with tf.control_dependencies([opt_c]):
            opt_c = tf.tuple(clipped_var_c)
    if not params['mode'] in ['gp', 'regular']:
        raise (NotImplementedError('Only two modes'))



    citers = tf.cond(tf.logical_or(tf.less(j, tf.constant(25, dtype=tf.int64)),
                                   tf.equal(tf.divide(tf.train.get_global_step(), tf.constant(500, dtype=tf.int64)), tf.zeros([],dtype=tf.float64))),
                     lambda: tf.constant(100, dtype=tf.int64), lambda: tf.constant(params['Citers'], dtype=tf.int64))


    train_op = tf.cond(tf.equal(tf.divide(i, citers), tf.zeros([],dtype=tf.float64)),
                       lambda: opt_c, lambda: opt_g)
    loss = tf.cond(tf.equal(tf.divide(i, citers), tf.zeros([],dtype=tf.float64)),
                   lambda: c_loss, lambda: g_loss)


    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "c_loss": tf.metrics.mean(c_loss),
            "g_loss": tf.metrics.mean(g_loss)},
        predictions=predictions,
    )

