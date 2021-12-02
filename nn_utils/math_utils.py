from typing import Optional
import tensorflow as tf
import numpy as np

EPS = 1e-7


def softplus_1m(x):
    return tf.math.softplus(x - 1)


def ev100_to_exp(ev100):
    maxL = 1.2 * tf.pow(2.0, ev100)
    return tf.maximum(1.0 / tf.maximum(maxL, EPS), tf.ones_like(maxL) * EPS)


def create_exp_decay(val, decay_rate, decay_steps):
    def exp_decay(global_step):
        return val * decay_rate ** (global_step / decay_steps)

    return exp_decay


def flatten(x: tf.Tensor, start_dim: int = 0, end_dim: Optional[int] = None):
    shape = tf.shape(x)
    if end_dim is None:
        end_dim = len(shape)

    new_shape = tf.concat([shape[:start_dim], [-1], shape[end_dim:]], 0)

    return tf.reshape(x, new_shape)


def repeat(x: tf.Tensor, n: int, axis: int) -> tf.Tensor:
    with tf.name_scope("repeat"):
        repeat = [1 for _ in range(len(x.shape))]
        repeat[axis] = n

        return tf.tile(x, repeat)


def saturate(x, low=0.0, high=1.0):
    with tf.name_scope("saturate"):
        return tf.clip_by_value(x, low, high)


def mix(x, y, a):
    with tf.name_scope("mix"):
        a = tf.clip_by_value(a, 0, 1)
        return x * (1 - a) + y * a


def to_vec3(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("to_vec3"):
        return repeat(x, 3, -1)


def vector_fill(shape, vector):
    with tf.name_scope("vector_fill"):
        return tf.stack(
            [
                tf.fill(shape, vector[0]),
                tf.fill(shape, vector[1]),
                tf.fill(shape, vector[2]),
            ]
        )


def fill_like(x, val):
    return tf.ones_like(x) * val


@tf.function(experimental_relax_shapes=True)
def background_compose(x: tf.Tensor, y: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    maskClip = saturate(mask)  # Ensure mask is 0 to 1
    return x * maskClip + (1.0 - maskClip) * y


@tf.function(experimental_relax_shapes=True)
def white_background_compose(x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    return background_compose(x, tf.ones_like(x), mask)


def srgb_to_linear(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("srgb_to_linear"):
        x = saturate(x)

        switch_val = 0.04045
        return tf.where(
            tf.math.greater_equal(x, switch_val),
            tf.pow((tf.maximum(x, switch_val) + 0.055) / 1.055, 2.4),
            x / 12.92,
        )


def linear_to_srgb(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("linear_to_srgb"):
        x = saturate(x)

        switch_val = 0.0031308
        return tf.where(
            tf.math.greater_equal(x, switch_val),
            1.055 * tf.pow(tf.maximum(x, switch_val), 1.0 / 2.4) - 0.055,
            x * 12.92,
        )


def soft_hdr(x, up_threshold=0.9, low_threshold=0.1):
    return tf.where(
        tf.less_equal(x, up_threshold),
        tf.where(
            tf.greater_equal(x, low_threshold),
            x,
            (
                low_threshold * safe_exp((x - low_threshold) / low_threshold)
                if low_threshold > 0
                else x
            ),
        ),
        (
            (
                1
                - (1 - up_threshold)
                * safe_exp(-((x - up_threshold) / (1 - up_threshold)))
            )
            if up_threshold < 1
            else x
        ),
    )


def uncharted2_tonemap_partial(x: tf.Tensor):
    A = 0.15
    B = 0.50
    C = 0.10
    D = 0.20
    E = 0.02
    F = 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def uncharted2_filmic(v: tf.Tensor):
    exposure_bias = 2.0
    curr = uncharted2_tonemap_partial(v * exposure_bias)

    W = 11.2
    white_scale = 1.0 / uncharted2_tonemap_partial(W)
    return curr * white_scale


def aces_approx(v: tf.Tensor):
    v = v * 0.6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return saturate((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0)


def isclose(x: tf.Tensor, val: float, threshold: float = EPS) -> tf.Tensor:
    with tf.name_scope("is_close"):
        return tf.less_equal(tf.abs(x - val), threshold)


def safe_sqrt(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("safe_sqrt"):
        sqrt_in = tf.maximum(x, EPS)
        return tf.sqrt(sqrt_in)


def safe_log(x):
    """The same as tf.math.log(x), but clamps the input to prevent NaNs."""
    return tf.math.log(tf.minimum(x, fill_like(x, 33e37)))


def safe_log1p(x):
    """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
    return tf.math.log1p(tf.minimum(x, fill_like(x, 33e37)))


def safe_exp(x):
    """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
    return tf.math.exp(tf.minimum(x, fill_like(x, 87.5)))


def safe_expm1(x):
    """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
    return tf.math.expm1(tf.minimum(x, fill_like(x, 87.5)))


def dot(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("dot"):
        return tf.reduce_sum(x * y, axis=-1, keepdims=True)


def magnitude(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("magnitude"):
        return safe_sqrt(dot(x, x))


def l2Norm(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("l2_norm"):
        return dot(x, x)


def normalize(x: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("normalize"):
        magn = magnitude(x)
        # If the magnitude is too low just return a zero vector
        return tf.where(magn <= safe_sqrt(float(0)), tf.zeros_like(x), x / magn)


def cross(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("cross"):
        # To keep everything short
        return tf.linalg.cross(x, y)


def reflect(d: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
    with tf.name_scope("reflect"):
        return d - 2 * dot(d, n) * n


def spherical_to_cartesian(theta: tf.Tensor, phi: tf.Tensor, r=1) -> tf.Tensor:
    """
    The referrence plane is the Cartesian xy plane
    phi is inclination from the z direction.
    theta is measured from the Cartesian x axis (so that the axis has theta = +90°).

    Args:
        theta: is azimuth [0, 2pi)
        phi: is inclination [0, Pi],
        r: is length  [0, inf)

    Returns:
        The cartesian vector (x,y,z)
    """
    x = r * tf.sin(phi) * tf.sin(theta)
    y = r * tf.cos(phi)
    z = r * tf.sin(phi) * tf.cos(theta)

    return tf.concat([x, y, z], -1)


def cartesian_to_spherical(vec: tf.Tensor) -> tf.Tensor:
    """
    Args:
        vec: cartesian vector

    Returns:
        Theta is azimuth [0, 2pi)
        Phi is inclination [0, Pi],
        r is length [0, inf)
    """
    x, y, z = vec[..., 0:1], vec[..., 1:2], vec[..., 2:3]

    r = magnitude(vec)
    theta = tf.math.atan2(x, z)
    # atan2 outputs value from -pi to pi.
    # We expect 0 to 2pi.
    # The negative values need to map to second quadrant
    theta = tf.where(theta > 0, theta, 2 * np.pi + theta)
    # Lastly theta should never reach 2pi so wrap around
    theta = tf.math.floormod(theta, 2 * np.pi - EPS)

    # Phi is just acos and safety to avoid div by 0
    phi = tf.math.acos(tf.clip_by_value(tf.math.divide_no_nan(y, r), -1, 1))

    return theta, phi, r


def spherical_to_uv(spherical: tf.Tensor) -> tf.Tensor:
    # Turst no one
    theta = tf.clip_by_value(
        spherical[..., 0],
        0,
        2 * np.pi - EPS,
    )  # [0, 2pi)
    phi = tf.clip_by_value(spherical[..., 1], 0, np.pi)  # [0, pi]

    u = theta / (2 * np.pi)  # TODO this might be a np.fliplr candidate
    v = phi / np.pi

    return tf.math.abs(tf.stack([u, v], -1))


def direction_to_uv(d: tf.Tensor) -> tf.Tensor:
    theta, phi, r = cartesian_to_spherical(d)
    return spherical_to_uv(tf.concat([theta, phi], -1))


def uv_to_spherical(uvs: tf.Tensor) -> tf.Tensor:
    # Turst no one
    u = tf.clip_by_value(uvs[..., 0], 0, 1)  # TODO this might be a np.fliplr candidate
    v = tf.clip_by_value(uvs[..., 1], 0, 1)

    theta = tf.clip_by_value(
        (2 * u * np.pi),
        0,
        2 * np.pi - EPS,
    )  # [-pi, pi)
    phi = tf.clip_by_value(np.pi * v, 0, np.pi)  # [0, pi]

    return tf.stack([theta, phi], -1)


def uv_to_direction(uvs: tf.Tensor) -> tf.Tensor:
    spherical = uv_to_spherical(uvs)
    theta = spherical[..., 0:1]
    phi = spherical[..., 1:2]
    return spherical_to_cartesian(theta, phi)


def shape_to_uv(height: int, width: int) -> tf.Tensor:
    # UV
    # 0,0              1,0
    # 0,1              1,1
    us, vs = tf.meshgrid(
        tf.linspace(
            0.0 + 0.5 / tf.cast(width, tf.float32),
            1.0 - 0.5 / tf.cast(width, tf.float32),
            width,
        ),
        tf.linspace(
            0.0 + 0.5 / tf.cast(height, tf.float32),
            1.0 - 0.5 / tf.cast(height, tf.float32),
            height,
        ),
    )  # Use pixel centers
    return tf.cast(tf.stack([us, vs], -1), tf.float32)


def logx(val, base):
    return tf.math.log(val) / tf.math.log(base)
