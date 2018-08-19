import tensorflow as tf
import os
import pdb
from scipy.io import loadmat
from six.moves import urllib
import numpy as np
import pandas as pd


# random noise from normal distribution
def build_normal_noise_fns(batch_size, eval_repeat=1, image_shape=(28, 28, 1)):
    """Build fake MNIST-style data for unit testing."""
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.normal(0,1,size=(1000, *image_shape)).astype("float32")).map(
        lambda row: (row, 0)).batch(batch_size)

    eval_input_fn = lambda: dataset.repeat(eval_repeat).make_one_shot_iterator().get_next()
    return eval_input_fn

# random noise from uniform distribution
def build_uniform_noise_fns(batch_size, eval_repeat=1, image_shape=(28, 28, 1)):
    """Build fake MNIST-style data for unit testing."""
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.uniform(low=0., high=1., size=(1000, *image_shape)).astype("float32")).map(
        lambda row: (row, 0)).batch(batch_size)

    eval_input_fn = lambda: dataset.repeat(eval_repeat).make_one_shot_iterator().get_next()
    return eval_input_fn

def credit_dataset_helper():
    df = pd.read_csv('data/creditcard.csv.zip', header=0, sep=',', quotechar='"')
    features_index = df.columns[:-2]  # exclude amount and class
    whole = df[features_index]
    # normalize amount
    amount = df['Amount']
    amount = (amount - amount.mean()) / amount.std()
    whole = pd.concat([whole, amount], axis=1)  # attach back normalized amount
    normal = whole[df.Class == 0].as_matrix().astype(np.float32)
    anormal = whole[df.Class == 1].as_matrix().astype(np.float32)  # size of 492
    whole = np.concatenate([normal, anormal], axis=0)

    feature_mins = np.min(whole, axis=0)
    feature_maxs = np.max(whole, axis=0)
    feature_ranges = feature_maxs - feature_mins

    # squeeze all features into [0,1]
    # first adjust min of each feature to 0.
    for feature_index, feature_min in enumerate(feature_mins):
        whole[:, feature_index] -= feature_min
    # divide each feature by its range so that new max is 1
    whole = whole / feature_ranges

    n_anormal = anormal.shape[0]  # 492
    normal_X = whole[:-n_anormal]
    anormal_X = whole[-n_anormal:]
    return normal_X, anormal_X

def build_credit_dataset(batch_size, noise=False, noise_type='normal'):

    normal_X, anormal_X = credit_dataset_helper()
    # split into train/test splits
    np.random.shuffle(normal_X)
    n_anormal = anormal_X.shape[0]  # 492
    test_normal_X = normal_X[:n_anormal]
    train_normal_X = normal_X[n_anormal:]

    dataset = tf.data.Dataset.from_tensor_slices((train_normal_X, np.zeros(train_normal_X.shape[0])))
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    eval_labels = np.zeros(n_anormal)
    eval_dataset = tf.data.Dataset.from_tensor_slices((test_normal_X, eval_labels))
    eval_dataset = eval_dataset.batch(batch_size)

    train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()
    return train_input_fn, eval_input_fn  # both consisting of normal data


def build_eval_multiple_datasets(dataset_list, batch_size, expand_last_dim=False, noised_list=None,
                               noise_type_list=None, feature_shape=(28,28), each_size=1000):

    x_test_list = []
    for i, dataset in enumerate(dataset_list):
        if noised_list != None:
            x_test = build_eval_helper(dataset, expand_last_dim, noised_list[i], noise_type_list[i],feature_shape, each_size)
        else:
            x_test = build_eval_helper(dataset, expand_last_dim, feature_shape=feature_shape, each_size=each_size)

        x_test_list.append(x_test)

    eval_dataset = tf.data.Dataset.from_tensor_slices((np.concatenate(x_test_list, axis=0)))
    eval_dataset = eval_dataset.batch(batch_size)

    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

    return eval_input_fn


def build_eval_helper(dataset, expand_last_dim=False, noised=False, noise_type='normal', feature_shape=(28,28), each_size=1000):

    if dataset in ['uniform_noise', 'normal_noise']:
        if dataset == 'uniform_noise':
            x_test = np.random.uniform(low=0., high=1., size=(each_size, *feature_shape)).astype("float32")
            x_test = 255. * x_test # will be divided by 255. later
        elif dataset == 'normal_noise':
            x_test = np.random.normal(loc=0., scale=1., size=(each_size, *feature_shape)).astype("float32")
            x_test = 255. * x_test

    elif dataset == 'notMNIST':
        loaded = np.fromfile(file='data/t10k-images-idx3-ubyte', dtype=np.uint8)
        x_test = np.reshape(loaded[16:], (-1, 28, 28))
    elif dataset == 'SVHN':
        loaded = loadmat('test_32x32.mat')['X']
        reshaped = loaded.transpose((3, 0, 1, 2))
        x_test = np.reshape(reshaped, (-1, 32, 32, 3))

    elif dataset == 'ImageNet':
        x_test = np.load('imagenet.npy')

    elif dataset == 'celebA':
        x_test = np.load('celeba.npy')

    elif dataset == 'credit_card_normal':
        x_test = credit_dataset_helper()[0] * 255.  #will be divided by 255 later

    elif dataset == 'credit_card_anomalies':
        x_test = credit_dataset_helper()[1] * 255.
    else:
        _, (x_test, _) = dataset.load_data()


    choice = np.random.choice(x_test.shape[0],each_size)
    x_test = x_test[choice]
    x_test = x_test.astype(np.float32) / 255.

    if expand_last_dim:
        x_test = np.expand_dims(x_test, axis=-1)

    image_shape = x_test.shape[1:]
    if noised:
        if noise_type == 'normal':
            noise = np.random.normal(loc=0., scale=1., size=(each_size, *image_shape)).astype("float32")
        elif noise_type == 'uniform':
            noise = np.random.uniform(low=0., high=1., size=(each_size, *image_shape)).astype("float32")
        elif noise_type == 'brighten':
            noise = np.ones_like(x_test).astype("float32")
        elif noise_type == 'hor_flip':
            x_test = np.flip(x_test, 2).astype("float32")
        elif noise_type == 'ver_flip':
            x_test = np.flip(x_test, 1).astype("float32")

        if noise_type in ['normal', 'uniform', 'brighten']:
            x_test += 0.1*noise

    x_test = np.clip(x_test,0., 1.)

    return x_test


def build_keras_dataset(keras_dataset, batch_size, expand_last_dim=False):
    (x_train, y_train), (x_test, y_test) = keras_dataset.load_data()

    if expand_last_dim:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    train_input_fn = lambda: train_dataset.repeat().make_one_shot_iterator().get_next()
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()
    return train_input_fn, eval_input_fn



def build_eval_dataset(dataset, batch_size, expand_last_dim=False, noised=False, noise_type='normal', feature_shape=(28,28),
                       each_size=1000):
    x_test = build_eval_helper(dataset, expand_last_dim, noised, noise_type, feature_shape, each_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices((x_test)).batch(batch_size)
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()
    # for glow
    #return x_test
    return eval_input_fn

def get_eval_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        return build_eval_dataset(tf.keras.datasets.mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'fashion_mnist':
        return build_eval_dataset(tf.keras.datasets.fashion_mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'notMNIST':
        return build_eval_dataset('notMNIST', batch_size, expand_last_dim=True)
    elif dataset_name == 'cifar10':
        return build_eval_dataset(tf.keras.datasets.cifar10, batch_size)
    elif dataset_name == 'cifar100':
        return build_eval_dataset(tf.keras.datasets.cifar100, batch_size)
    elif dataset_name == 'SVHN':
        return build_eval_dataset('SVHN', batch_size)
    elif dataset_name == 'credit_card_normal':
        return build_eval_dataset('credit_card_normal', batch_size, feature_shape=(30,), each_size=492)
    elif dataset_name == 'credit_card_anomalies':
        return build_eval_dataset('credit_card_anomalies', batch_size, feature_shape=(30,), each_size=492)



def get_dataset(dataset_name, batch_size):  #for training and eval
    # unified dataset loading fn
    if dataset_name == 'mnist':
        return build_keras_dataset(tf.keras.datasets.mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'fashion_mnist':
        return build_keras_dataset(tf.keras.datasets.fashion_mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'cifar10':
        return build_keras_dataset(tf.keras.datasets.cifar10, batch_size)
    elif dataset_name == 'credit_card':
        return build_credit_dataset(batch_size)




