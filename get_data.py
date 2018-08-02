import tensorflow as tf
import os
from six.moves import urllib
import numpy as np
import pandas as pd

ROOT_PATH = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
FILE_TEMPLATE = "binarized_mnist_{split}.amat"


def download(directory, filename):
    """Download a file."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = os.path.join(ROOT_PATH, filename)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


def static_mnist_dataset(directory, split_name):
    """Return binary static MNIST tf.data.Dataset."""
    amat_file = download(directory, FILE_TEMPLATE.format(split=split_name))
    dataset = tf.data.TextLineDataset(amat_file)
    str_to_arr = lambda string: np.array([char == "1" for char in string.split()])

    def _parser(s):
        booltensor = tf.py_func(str_to_arr, [s], tf.bool)
        reshaped = tf.reshape(booltensor, [28, 28, 1])
        return tf.to_float(reshaped), tf.constant(0, tf.int32)

    return dataset.map(_parser)


def build_fake_input_fns(batch_size, eval_repeat=1, image_shape=(28, 28, 1)):
    """Build fake MNIST-style data for unit testing."""
    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.rand(batch_size, *image_shape).astype("float32")).map(
        lambda row: (row, 0)).batch(batch_size)

    train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
    eval_input_fn = lambda: dataset.repeat(eval_repeat).make_one_shot_iterator().get_next()
    return train_input_fn, eval_input_fn


def build_mnist_input_fns(data_dir, batch_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = static_mnist_dataset(data_dir, "train")
    training_dataset = training_dataset.shuffle(50000).repeat().batch(batch_size)
    train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.
    eval_dataset = static_mnist_dataset(data_dir, "valid")
    eval_dataset = eval_dataset.batch(batch_size)
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

    return train_input_fn, eval_input_fn


def build_credit_dataset(batch_size):
    df = pd.read_csv('drive/Colab Notebooks/creditcard.csv.zip', header=0, sep=',', quotechar='"')
    normal = df[df.Class == 0]
    anormal = df[df.Class == 1]

    X_index = df.columns[1:-2]
    normal_features = normal[X_index]
    normal_time_amount = normal[['Time', 'Amount']]
    normal_time_amount = (normal_time_amount - normal_time_amount.mean()) / normal_time_amount.std()
    normal_X = pd.concat([normal_features, normal_time_amount], axis=1)
    normal_X = normal_X.as_matrix()
    normal_X = normal_X.astype(np.float32)

    # split into train/test splits
    test_normal_X = normal_X[:492]
    train_normal_X = normal_X[492:]

    anormal_features = anormal[X_index]
    anormal_time_amount = anormal[['Time', 'Amount']]
    anormal_time_amount = (anormal_time_amount - anormal_time_amount.mean()) / anormal_time_amount.std()
    anormal_X = pd.concat([anormal_features, anormal_time_amount], axis=1)
    anormal_X = anormal_X.as_matrix()
    anormal_X = anormal_X.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((train_normal_X, np.zeros(train_normal_X.shape[0])))
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    eval_labels = np.concatenate([np.zeros(test_normal_X.shape[0]), np.ones(anormal_X.shape[0])])
    eval_dataset = tf.data.Dataset.from_tensor_slices((np.concatenate([test_normal_X, anormal_X], axis=0), eval_labels))
    eval_dataset = eval_dataset.batch(batch_size)

    train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
    eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()
    return train_input_fn, eval_input_fn


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


def get_dataset(dataset_name, batch_size):
    # unified dataset loading fn
    if dataset_name == 'mnist':
        return build_keras_dataset(tf.keras.datasets.mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'fashion_mnist':
        return build_keras_dataset(tf.keras.datasets.fashion_mnist, batch_size, expand_last_dim=True)
    elif dataset_name == 'noise':
        return build_fake_input_fns(1000, eval_repeat=10)
    elif dataset_name == 'cifar10':
        return build_keras_dataset(tf.keras.datasets.cifar10, batch_size)
    elif dataset_name == 'cifar100':
        return build_keras_dataset(tf.keras.datasets.cifar100, batch_size)
    elif dataset_name == 'credit_card':
        return build_credit_dataset(batch_size)
