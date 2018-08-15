# Credit Card Model
from absl import flags
import tensorflow as tf
import functools
import itertools
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
FLAGS = flags.FLAGS

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

    encoder_net = tf.keras.Sequential([
        tf.keras.layers.Dense(2 * latent_size, activation=None),
    ])

    def encoder(images):
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

    decoder_net = tf.keras.Sequential([
        tf.keras.layers.Dense(30),
    ])

    def decoder(codes):
        # pdb.set_trace()
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


def anomaly_model_fn(features, labels, mode, params, config):
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
    # del labels, config
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

    # pdb.set_trace()

    approx_posterior = encoder(features)
    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)

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
    loss = -elbo
    tf.summary.scalar("elbo", elbo)

    # negative log-likelihood of encoded inputs under likelihood model p(x|z)
    # lower is better
    predictions['distortion'] = distortion
    predictions['rate'] = rate
    predictions['elbo'] = elbo_local

    importance_weighted_elbo = tf.reduce_mean(
        tf.reduce_logsumexp(elbo_local, axis=0) -
        tf.log(tf.to_float(params["n_samples"])))
    tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

    # Perform variational inference by minimizing the -ELBO.
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(0.0001, global_step,
                                          params["max_steps"])
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # estimator predictions for inference + visualization
    predictions['approx_posterior_mean'] = approx_posterior.mean()
    predictions['approx_posterior_stddev'] = approx_posterior.scale.diag

    # adversarial perturbation
    grad, = tf.gradients(loss, features)
    adversarial_example = features - .1 * tf.sign(grad)  # optimize the gibberish to minimize loss.
    predictions['adversarial_example'] = adversarial_example
    # pdb.set_trace()
    elbo_local.set_shape((FLAGS.n_samples, None))
    elbo_local_mean = tf.reduce_mean(elbo_local, axis=0)
    predictions['elbo_local_mean'] = elbo_local_mean
    elbo_local_mean = tf.sigmoid(elbo_local_mean)
    predictions['sigmoid'] = elbo_local_mean
    mask = tf.greater(elbo_local_mean, params['elbo_threshold'])
    predictions['class'] = tf.cast(mask, tf.int32)
    # elbo_local_mean = tf.clip_by_value(elbo_local_mean, 0.0,1.0)
    # thresholds = [0.0, 0.5, 1.0]
    thresholds = np.arange(0.0, 0.2, 0.01).tolist()

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops={
                "elbo": tf.metrics.mean(elbo),
                "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
                "rate": tf.metrics.mean(avg_rate),
                "distortion": tf.metrics.mean(avg_distortion),
            },
            predictions=predictions
        )

    labels = tf.cast(labels, tf.int32)

    (metric_tensor1, update_op1) = tf.metrics.true_positives_at_thresholds(labels=labels, predictions=elbo_local_mean,
                                                                           thresholds=thresholds)
    (metric_tensor2, update_op2) = tf.metrics.true_negatives_at_thresholds(labels=labels, predictions=elbo_local_mean,
                                                                           thresholds=thresholds)
    (metric_tensor3, update_op3) = tf.metrics.false_positives_at_thresholds(labels=labels, predictions=elbo_local_mean,
                                                                            thresholds=thresholds)
    (metric_tensor4, update_op4) = tf.metrics.false_negatives_at_thresholds(labels=labels, predictions=elbo_local_mean,
                                                                            thresholds=thresholds)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.model_dir,
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            "elbo": tf.metrics.mean(elbo),
            "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
            "rate": tf.metrics.mean(avg_rate),
            "distortion": tf.metrics.mean(avg_distortion),
            "TP": (tf.cast(metric_tensor1, tf.float32), tf.cast(update_op1, tf.float32)),
            "TN": (tf.cast(metric_tensor2, tf.float32), tf.cast(update_op2, tf.float32)),
            "FP": (tf.cast(metric_tensor3, tf.float32), tf.cast(update_op3, tf.float32)),
            "FN": (tf.cast(metric_tensor4, tf.float32), tf.cast(update_op4, tf.float32)),
            "auc": tf.metrics.auc(labels=labels, predictions=elbo_local_mean),
        },
        predictions=predictions, training_hooks=[summary_hook]
    )



