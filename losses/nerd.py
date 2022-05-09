import tensorflow as tf


def l2_norm(axis=-1, global_batch_size=None):
    @tf.function
    def run(x):
        norm = tf.reduce_sum(tf.square(x), axis=axis)

        if global_batch_size is not None:
            return tf.reduce_sum(norm / global_batch_size)
        else:
            return tf.reduce_mean(norm)

    return run


def mask_mse(global_batch_size=None):
    @tf.function
    def run(true, pred, mask):
        loss = tf.math.square(true - pred) * mask

        if global_batch_size is not None:
            loss_sum = tf.reduce_sum(loss)
            term_mean = loss_sum / global_batch_size / loss.shape[-1]
            return term_mean

        return tf.reduce_mean(loss)

    return run


def cosine_weighted_mse(global_batch_size=None):
    @tf.function
    def run(true, pred, cos_theta_i):
        # We want to optimize true and pred and not the cos theta
        # value
        # This would make all normals face away from the camera
        # So stop backprop to the normals for the cos theta
        cos_theta_stop = tf.stop_gradient(tf.nn.relu(cos_theta_i))

        term = tf.math.square((true - pred) * cos_theta_stop)

        if global_batch_size is not None:
            term_sum = tf.reduce_sum(term)
            term_mean = term_sum / global_batch_size / term.shape[-1]
            return term_mean

        return tf.reduce_mean(term)

    return run


def segmentation_mask_loss(global_batch_size=None):
    @tf.function
    def call(individual_alpha, acc_alpha, mask, fg_scaler=1):
        # The background loss punishes all values directly
        loss_background = tf.reduce_sum(
            tf.abs(individual_alpha - mask), -1, keepdims=True
        )

        # In the foreground we do not know where information should be placed
        loss_foreground = (
            tf.abs(acc_alpha - mask) if fg_scaler > 0 else tf.zeros_like(mask)
        )

        mask_binary = tf.where(mask > 0.5, tf.ones_like(mask), tf.zeros_like(mask))

        return (
            tf.reduce_sum(
                tf.where(
                    mask > 0.5,
                    loss_foreground * mask_binary * fg_scaler,
                    loss_background * (1 - mask_binary),
                )
            )
            / global_batch_size
        )

    return call
