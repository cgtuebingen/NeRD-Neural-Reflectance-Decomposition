import tensorflow as tf


def to_8b(x):
    return tf.cast(255 * tf.clip_by_value(x, 0, 1), tf.uint8)


def horizontal_image_log(name, *xs):
    [x.shape.assert_has_rank(4) for x in xs]
    stacked = tf.concat(xs, 2)
    tf.summary.image(name, stacked)


def vertical_image_log(name, *xs):
    [x.shape.assert_has_rank(4) for x in xs]
    stacked = tf.concat(xs, 1)
    tf.summary.image(name, stacked)


def hdr_to_tb(name, data):
    tf.summary.image(
        name,
        tf.clip_by_value(  # Just for safety
            tf.math.pow(
                data / (tf.ones_like(data) + data),
                1.0 / 2.2,
            ),
            0,
            1,
        ),  # Reinhard tone mapping
    )
