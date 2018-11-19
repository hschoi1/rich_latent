from absl import flags
import tensorflow as tf
import functools
import itertools
import tensorflow_probability as tfp
import numpy as np
import pdb
tfd = tfp.distributions
tfb = tfp.bijectors
flags=tf.app.flags
FLAGS = tf.app.flags.FLAGS

def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.log(tf.expm1(x))


def make_encoder(activation, latent_size, base_depth):
    """Create the encoder function.
    Args:
      activation: Activation function to use.
      latent_size: The dimensionality of the encoding.
      base_depth: The lowest depth for a layer.
    Returns:
      encoder: A `callable` mapping a `Tensor` of images to a
        `tf.distributions.Distribution` instance over encodings.
    """
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=activation)
 
    encoder_net = tf.keras.Sequential([
        conv(base_depth, 5, 1),
        conv(base_depth, 5, 2),
        conv(2 * base_depth, 5, 1),
        conv(2 * base_depth, 5, 2),
        conv(4 * latent_size, 7, padding="VALID"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])

    def encoder(images):
        images = 2 * tf.cast(images, dtype=tf.float32) - 1
        net = encoder_net(images)
        return tfd.MultivariateNormalDiag(
            loc=net[..., :latent_size],
            scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                      _softplus_inverse(1.0)),
            name="code")

    return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
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

    decoder_net = tf.keras.Sequential([
        deconv(2 * base_depth, filter_width, padding="VALID"),
        deconv(2 * base_depth, 5),
        deconv(2 * base_depth, 5, 2),
        deconv(base_depth, 5),
        deconv(base_depth, 5, 2),
        deconv(base_depth, 5),
        conv(output_shape[-1], 5, activation=None),
    ])

    def decoder(codes):
        original_shape = tf.shape(codes)
        # Collapse the sample and batch dimension and convert to rank-4 tensor for
        # use with a convolutional decoder network.
        codes = tf.reshape(codes, (-1, 1, 1, latent_size))
        logits = decoder_net(codes)
        logits = tf.reshape(
            logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
        return tfd.Independent(
            tfd.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(output_shape),
            name="image")

    return decoder


def init_once(x, name):
    return tf.get_variable(name, initializer=x, trainable=False)


def make_arflow(z_dist, latent_size, n_flows, hidden_size=(512, 512), invert=False):
    chain = list(itertools.chain.from_iterable([
                                                   tfb.MaskedAutoregressiveFlow(
                                                       shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                                                           hidden_size)),
                                                   tfb.Permute(
                                                       init_once(np.random.permutation(latent_size), 'permute_%d' % i)),
                                               ] for i in range(n_flows)))
    return tfd.TransformedDistribution(distribution=z_dist, bijector=tfb.Chain(chain[:-1]))


def make_NF_prior(latent_size, n_flows):
    return make_arflow(tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size], dtype=tf.float32),
        scale_diag=tf.ones([latent_size], dtype=tf.float32)),
        latent_size, n_flows)


def make_mixture_prior(latent_size, mixture_components):
    """Create the mixture of Gaussians prior distribution. Prior is learned.
    Args:
      latent_size: The dimensionality of the latent representation.
      mixture_components: Number of elements of the mixture.
    Returns:
      random_prior: A `tf.distributions.Distribution` instance
        representing the distribution over encodings in the absence of any
        evidence.
    """
    if mixture_components == 1:
        # See the module docstring for why we don't learn the parameters here.
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0)
    else:
        loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
        raw_scale_diag = tf.get_variable(
            name="raw_scale_diag", shape=[mixture_components, latent_size])
        mixture_logits = tf.get_variable(
            name="mixture_logits", shape=[mixture_components])

        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.nn.softplus(raw_scale_diag)),
            mixture_distribution=tfd.Categorical(logits=mixture_logits),
            name="prior")


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def model_fn(features, labels, mode, params, config):
    """Build the model function for use in an estimator.
    Arguments:
      features: The input features for the estimator.
      labels: The labels, unused here.
      mode: Signifies whether it is train or test or predict.
      params: Some hyperparameters as a dictionary.
      config: The RunConfig, unused here.
    Returns:
      EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """

   
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])
    del labels, config
    predictions = {}

    if params["analytic_kl"] and params["latent_size"] != 1:
        raise NotImplementedError(
            "Using `analytic_kl` is only supported when `mixture_components = 1` "
            "since there's no closed form otherwise.")

    encoder = make_encoder(params["activation"],
                           params["latent_size"],
                           params["base_depth"])

    image_shape = features.get_shape().as_list()[1:]
    decoder = make_decoder(params["activation"],
                           params["latent_size"],
                           image_shape,
                           params["base_depth"])
    if params['use_NF']:
        latent_prior = make_NF_prior(params["latent_size"], params["n_flows"])
    else:
        latent_prior = make_mixture_prior(params["latent_size"],
                                          params["mixture_components"])

    image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
    image_tile_summary(
        "recon/sample",
        tf.to_float(decoder_likelihood.sample()[:3, :16]),
        rows=3,
        cols=16)
    image_tile_summary(
        "recon/mean",
        decoder_likelihood.mean()[:3, :16],
        rows=3,
        cols=16)

    # `distortion` is just the negative log likelihood.
    distortion = -decoder_likelihood.log_prob(features)
    avg_distortion = tf.reduce_mean(distortion)
    tf.summary.scalar("distortion", avg_distortion)

    if params["analytic_kl"]:
        raise ValueError('Not Completely Implemented!')
        rate = tfd.kl_divergence(approx_posterior, latent_prior)
    else:
        rate = (approx_posterior.log_prob(approx_posterior_sample)
                - latent_prior.log_prob(approx_posterior_sample))
    avg_rate = tf.reduce_mean(rate)
    tf.summary.scalar("rate", avg_rate)

    elbo_local = -(rate + distortion)

    elbo = tf.reduce_mean(elbo_local)
    tf.summary.scalar("elbo", elbo)

    loss = tf.reduce_mean(FLAGS.beta * rate + distortion)

    # negative log-likelihood of encoded inputs under likelihood model p(x|z)
    # lower is better
    predictions['distortion'] = distortion
    predictions['rate'] = rate
    predictions['elbo'] = elbo_local

    importance_weighted_elbo = tf.reduce_mean(
        tf.reduce_logsumexp(elbo_local, axis=0) -
        tf.log(tf.to_float(params["n_samples"])))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

    # Decode samples from the prior for visualization.
    random_image = decoder(latent_prior.sample(16))
    image_tile_summary(
        "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
    image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # estimator predictions for inference + visualization
    predictions['approx_posterior_mean'] = approx_posterior.mean()
    predictions['approx_posterior_stddev'] = approx_posterior.scale.diag

    if mode == tf.estimator.ModeKeys.PREDICT:
        # adversarial perturbation to random noise
        grad, = tf.gradients(loss, features)
        normal_noise = tfd.Normal(loc=tf.zeros([]), scale=tf.ones([]))
        uniform_noise = tfd.Uniform(0.,1.)
        normal_noise_samples = normal_noise.sample(sample_shape=tf.shape(features))
        uniform_noise_samples = uniform_noise.sample(sample_shape=tf.shape(features))
        adversarial_normal_noise = normal_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.
        adversarial_uniform_noise = uniform_noise_samples - .3 * tf.sign(grad)  # optimize the gibberish to minimize loss.
        predictions['adversarial_normal_noise'] = adversarial_normal_noise
        predictions['adversarial_uniform_noise'] = adversarial_uniform_noise

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.metrics.mean(elbo),
            "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
            "rate": tf.metrics.mean(avg_rate),
            "distortion": tf.metrics.mean(avg_distortion), },
        predictions=predictions,
    )



