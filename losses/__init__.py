import tensorflow as tf
import inspect


def multi_gpu_wrapper(loss_fn, global_batch_size):
    if inspect.isclass(loss_fn) and issubclass(loss_fn, tf.keras.losses.Loss):
        loss_obj = loss_fn(reduction=tf.keras.losses.Reduction.NONE)
    else:
        loss_obj = loss_fn

    def calculate_loss(*loss_args):
        per_example_loss = loss_obj(*loss_args)
        return tf.reduce_sum(per_example_loss) / global_batch_size

    return calculate_loss


def l2_regularization(x):
    return tf.reduce_mean(tf.square(x), -1)
