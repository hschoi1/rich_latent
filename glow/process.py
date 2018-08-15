import tensorflow as tf
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import glob 

def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[1] % resolution == 0
    assert x.shape[2] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [s[0], resolution, s[1] // resolution,
                       resolution, s[2] // resolution, s[3]])
    x = np.mean(x, (2, 4))
    return x


def x_to_uint8(x):
    x = np.clip(np.floor(x), 0, 255)
    return x.astype(np.uint8)

 
with tf.Session() as sess:
    data_path = glob.glob('celeba-tfr/validation/*.tfrecords')

    filename_queue = tf.train.string_input_producer(data_path)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([1], tf.int64)})
    # label is always 0 if uncondtional
    # to get CelebA attr, add 'attr': tf.FixedLenFeature([40], tf.int64)
    data, label, shape = features['data'], features['label'], features['shape']
    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
    img = tf.decode_raw(data, tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    imgs = []
    for i in range(1000):
       image, lbl = sess.run([img, label])
       imgs.append(image)
    imgs = np.array(imgs)
    imgs = downsample(imgs.astype(np.float32), 32)
    imgs = x_to_uint8(imgs)
    pdb.set_trace()
    np.save('celeba',imgs)
    coord.request_stop()
    coord.join(threads)
    sess.close()



