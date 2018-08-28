import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np
import pdb

tfd = tfp.distributions
tfb = tfp.bijectors

tf.reset_default_graph()

batch_size = 512
X = np.random.normal(loc=[5., 5.], scale=[1., 1.], size=(10000, 2))
plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
plt.show()
dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=X.shape[0])
dataset = dataset.prefetch(3 * batch_size)
dataset = dataset.batch(batch_size)
data_iterator = dataset.make_one_shot_iterator()
x_samples = data_iterator.get_next()

base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], tf.float32))


class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        # event_dims = self._event_dims_tensor(y)
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=1)


d, r = 2, 2
bijectors = []
num_layers = 4
for i in range(num_layers):
    with tf.variable_scope('bijector_%d' % i):
        V = tf.get_variable('V', [d, r], dtype=np.float32)  # factor loading
        shift = tf.get_variable('shift', [d], dtype=np.float32)  # affine shift
        L = tf.get_variable('L', [d * (d + 1) / 2],
                            dtype=np.float32)  # lower triangular
        bijectors.append(tfb.Affine(
            scale_tril=tfd.fill_triangular(L),
            scale_perturb_factor=V,
            shift=shift,
        ))
        alpha = tf.abs(tf.get_variable('alpha', [], dtype=np.float32)) + .01
        bijectors.append(LeakyReLU(alpha=alpha))
# Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
# that they are applied..
mlp_bijector = tfb.Chain(
    list(reversed(bijectors[:-1])), name='2d_mlp_bijector')

dist = tfd.TransformedDistribution(
    distribution=base_dist,
    bijector=mlp_bijector
)

x = base_dist.sample(batch_size)
samples = [x]
names = [base_dist.name]
for bijector in reversed(dist.bijector.bijectors):
    x = bijector.forward(x)
    samples.append(x)
    names.append(bijector.name)

features = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x')
log_prob = dist.log_prob(features)
loss = -tf.reduce_mean(dist.log_prob(x_samples))
train_op = tf.train.RMSPropOptimizer(1e-5).minimize(loss)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    NUM_STEPS = int(1e5)
    global_step = []
    np_losses = []
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss], {
            features: np.zeros((100, 2))})  # 'features' is a meaningless input here. it is for later testing
        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(np_loss)

        if i % int(5e3) == 0:
            print(i, np_loss)
    start = 10
    plt.plot(np_losses[start:])
    plt.show()

    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)

    whole = np.zeros((400, 2))
    for i in range(20):
        for j in range(20):
            whole[20 * i + j][0] = x[i]
            whole[20 * i + j][1] = y[j]

    feed_dict = {features: whole.astype(np.float32)}
    fetch_dict = {'log_prob': log_prob}
    results = sess.run(fetch_dict, feed_dict)

    z = np.reshape(results['log_prob'], (20, 20), order='F')  # feed in all the points
    X, Y = np.meshgrid(x, y)

    plt.contourf(X, Y, z, 20, cmap=plt.cm.bone)
    plt.savefig('NF_heatmap.eps', format='eps', dpi=1000)

    plt.show()



