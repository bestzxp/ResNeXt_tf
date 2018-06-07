import tensorflow as tf
import tflearn

def global_average_pooling(x):
    return tflearn.layers.conv.global_avg_pool(x, name='global_average_pooling')


def conv_layer(input, filter, kernel, stride=1, padding='SAME', name='conv'):
    return tf.layers.conv2d(inputs=input, use_bias=False,
                            filters=filter, kernel_size=kernel,
                            strides=stride, padding=padding, name=name)


def batch_normalization(x, training, name=''):
    return tf.layers.batch_normalization(inputs=x, training=training, name=name)


def relu(x):
    return tf.nn.relu(x)

def concatenation(layers, axis=3):
    return tf.concat(layers, axis=axis)

def dense(input, out, name='linear'):
    return tf.layers.dense(inputs=input, use_bias=False, units=out, name=name)

def max_pooling(input, pool_size, stride):
    return tf.layers.max_pooling2d(input, pool_size=pool_size, strides=stride, padding='SAME')


