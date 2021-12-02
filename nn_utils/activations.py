import tensorflow as tf


def to_hdr_activation(x):
    return tf.exp(tf.nn.relu(x)) - 1


def from_hdr_activation(x):
    return tf.math.log(1 + tf.nn.relu(x))


def softplus_1m(x):
    return tf.math.softplus(x - 1)


def padded_sigmoid(x, padding: float, upper_padding=True, lower_padding=True):
    # If padding is positive it can have values from 0-padding < x < 1+padding,
    # if negative 0+padding < x 1-padding
    x = tf.nn.sigmoid(x)

    mult = upper_padding + lower_padding  # Evil cast to int
    x = x * (1 + mult * padding)
    if lower_padding:
        x = x - padding
    return x
