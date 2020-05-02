import os

import tensorflow as tf

from multiprocessing import Process, Queue


def process_write_out(write_fn, fn_args, max_queue_size=5000):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    write_queue = Queue(maxsize=max_queue_size)
    process = Process(target=write_fn, args=fn_args + (write_queue,))
    process.start()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    return process, write_queue


def gaussFilter(fx, fy, sigma):
    x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
    y = x
    Y, X = tf.meshgrid(x, y)

    sigma = -2 * (sigma ** 2)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    k = 2 * tf.exp(tf.divide(z, sigma))
    k = tf.divide(k, tf.reduce_sum(k))
    return k

def gaussian_blur(image, filtersize, sigma):
    n_channels = image.shape[-1]

    fx, fy = filtersize[0], filtersize[1]
    filt = gaussFilter(fx, fy, sigma)
    filt = tf.stack([filt] * n_channels, axis=2)
    filt = tf.expand_dims(filt, 3)

    padded_image = tf.pad(image, [[0, 0], [fx, fx], [fy, fy], [0, 0]], constant_values=0.0)

    res = tf.nn.depthwise_conv2d(padded_image, filt, strides=[1, 1, 1, 1], padding="SAME")
    return res[:, fx:-fx, fy:-fy, :]