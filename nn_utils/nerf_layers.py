from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from nn_utils.activations import softplus_1m


class FourierEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        num_frequencies: int,
        include_input: bool = True,
        input_dim: int = 3,
        **kwargs
    ):
        super(FourierEmbedding, self).__init__(trainable=False, **kwargs)

        self.input_dims = input_dim
        self.out_dims = 0
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if include_input:
            self.out_dims += input_dim

        if num_frequencies >= 1:
            frequency_dims = (
                2 * num_frequencies * input_dim
            )  # 2 (sin, cos) * num_frequencies * num input_dim
            self.out_dims += frequency_dims

        self.scales = tf.convert_to_tensor(
            2.0 ** np.linspace(0.0, self.num_frequencies - 1, self.num_frequencies),
            tf.float32,
        )

    # @tf.function
    def call(self, x):
        assert (
            x.shape[-1] == self.input_dims
        ), "Channel dimension is %d but should be %d" % (x.shape[-1], self.input_dims)

        x_shape = tf.shape(x)

        xb = tf.reshape(
            x[..., None, :] * self.scales[:, None],
            tf.concat([x_shape[:-1], self.num_frequencies * x_shape[-1:]], 0),
        )

        four_feat = tf.math.sin(tf.concat([xb, xb + 0.5 * np.pi], axis=-1))

        ret = []
        if self.include_input:
            ret += [x]

        return tf.concat(ret + [four_feat], axis=-1)

    def get_output_dimensionality(self):
        return self.out_dims


class AnnealedFourierEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, num_frequencies, include_input: bool = True, input_dim: int = 3, **kwargs
    ):
        super(AnnealedFourierEmbedding, self).__init__(trainable=False, **kwargs)

        self.input_dims = input_dim
        self.base_embedder = FourierEmbedding(
            num_frequencies, include_input=False, input_dim=input_dim
        )
        self.num_frequencies = num_frequencies
        self.include_input = include_input

    # @tf.function
    def call(self, x, alpha):
        embed_initial = self.base_embedder(x)

        embed = tf.reshape(
            embed_initial, (-1, 2, self.num_frequencies, self.input_dims)
        )
        window = tf.reshape(self.cosine_easing_window(alpha), (1, 1, -1, 1))
        embed = window * embed

        embed = tf.reshape(embed, tf.shape(embed_initial))
        if self.include_input:
            embed = tf.concat([x, embed], -1)

        return embed

    def cosine_easing_window(self, alpha):
        x = tf.clip_by_value(
            alpha - tf.range(0, self.num_frequencies, dtype=tf.float32), 0.0, 1.0
        )
        return 0.5 * (1 + tf.cos(np.pi * x + np.pi))

    @classmethod
    def calculate_alpha(cls, num_frequencies, step, end_step):
        return (num_frequencies * step) / end_step

    def get_output_dimensionality(self):
        return self.base_embedder.get_output_dimensionality() + (
            self.input_dims if self.include_input else 0
        )


def add_base_args(parser):
    """Add the base nerf arguments to the parser

    Args:
        parser (ArgumentParser): the current ArgumentParser.

    Returns:
        ArgumentParser: the modified ArgumentParser for call chaining
    """
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    return parser


def split_sigma_and_payload(raw) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calls the network and separates the sigma from the payload.

    Args:
        raw (tf.Tensor(float32) [...., 1 + payload_channels]): the
            raw output data from the network.

    Returns:
        sigma (tf.Tensor(float32), [..., 1]): the density
            along the ray.
        payload (tf.Tensor(float32), [..., payload_channels]): the
            payload along the ray.
    """
    # Separete the sigma value from the payload
    sigma = raw[..., :1]
    payload = raw[..., 1:]

    return sigma, payload


def volumetric_rendering(
    sigma: tf.Tensor,
    payload: tf.Tensor,
    z_samples: tf.Tensor,
    rays_direction: tf.Tensor,
    payload_to_parmeters: Callable[[tf.Tensor], Dict[str, tf.Tensor]],
    white_background_parameters: List[str] = [],
    sigma_activation=softplus_1m,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Volumetric Rendering Function.

    Args:
        sigma (tf.Tensor(float32), [batch, num_samples, 1]): the density per sample.
        payload (tf.Tensor(float32), [batch, num_samples, num_payloads_channels]): the
            payload per samples. Can contain anything from RGB to a BRDF.
        z_samples (tf.Tensor(float32), [batch, num_samples]): the z samples along the
            ray.
        ray_directions (tf.Tensor(float32), [batch, 3]): the ray directions.
        payload_to_parmeters (Callable[[tf.Tensor], Dict[str, tf.Tensor]]): function
            which splits the payload into a dict with distinct parameters. Can also
            apply activations functions to the linear payload.
        white_background_parameters (List[str], optional): list of parameters that are
            composed on a white background. Defaults to [] (No white background
            composing).

    Returns:
        payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the raymarched
            payload dictionary.
        weights (tf.Tensor(float32) [batch, num_samples]): the weights along the ray.
            That is the accumulated product of the individual alphas.
    """
    eps = 1e-10
    sigma = sigma[..., 0]  # Remove channel dimensions

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_samples[..., 1:] - z_samples[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = tf.concat(
        [dists, tf.broadcast_to([eps], tf.shape(dists[..., :1]))], axis=-1
    )  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * tf.linalg.norm(rays_direction[..., None, :], axis=-1)

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha = 1.0 - tf.exp(-sigma_activation(sigma) * dists)

    # Compute weight for payload of each sample along each ray. A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # [N_rays, N_samples]
    accum_prod = tf.concat(
        [
            tf.ones_like(alpha[..., :1], alpha.dtype),
            tf.math.cumprod(1.0 - alpha[..., :-1] + eps, axis=-1, exclusive=True),
        ],
        -1,
    )
    weights = alpha * accum_prod

    # Is the accumulated density along the ray
    accumulated_weights = tf.reduce_sum(weights, -1)

    # Estimated depth is expected distance.
    depth = tf.reduce_sum(weights * z_samples, axis=-1)

    # Disparity is inverse depth.
    inv_eps = 1 / eps
    disp = tf.math.divide_no_nan(accumulated_weights, tf.maximum(depth, eps))
    disparity = tf.where(
        (disp > 0) & (disp < inv_eps) & (accumulated_weights > eps), disp, inv_eps
    )

    # Apply weights to payload of each ray sample
    payload_dict = payload_to_parmeters(payload)
    payload_raymarched = {
        k: tf.reduce_sum(weights[..., None] * v, axis=-2)
        for k, v in payload_dict.items()
    }

    # White background compose
    for parameter in white_background_parameters:
        payload_raymarched[parameter] = math_utils.white_background_compose(
            payload_raymarched[parameter], accumulated_weights[..., None]
        )

    payload_raymarched["depth"] = depth
    payload_raymarched["disparity"] = disparity
    payload_raymarched["acc_alpha"] = accumulated_weights
    payload_raymarched["individual_alphas"] = alpha

    return (
        payload_raymarched,
        weights,
    )


def cast_rays(rays_origin: tf.Tensor, rays_direction: tf.Tensor, z_samples: tf.Tensor):
    """Build the ray sampling points.

    Args:
        ray_origins (tf.Tensor(float32), [batch, ..., 3]): the ray origin.
        ray_directions (tf.Tensor(float32), [batch, ...., 3]): the ray direction.
        z_samples (tf.Tensor(float32), [batch, ..., samples]): the z distances from
            a sampling function.

    Returns:
        tf.Tensor(float32), [batch, ..., samples, 3]: the points along the ray.
    """
    return (
        rays_origin[..., None, :]
        + rays_direction[..., None, :] * z_samples[..., :, None]
    )


@tf.function(experimental_relax_shapes=True)
def setup_fixed_grid_sampling(
    rays_origin: tf.Tensor,
    rays_direction: tf.Tensor,
    near_bound: float,
    far_bound: float,
    num_samples: int,
    randomized: bool = False,
    linear_disparity: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Creates a fixed sampling pattern.

    Args:
        ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
        ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
        near_bound (float): the near clipping point.
        far_bound (float): the far clipping point.
        num_samples (int): the number of samples to generate.
        randomized (bool, optional): Activates noise and pertub ray features.
            Defaults to False.
        linear_disparity (bool, optional): Sample in linearly in disparity
            instead of depth. Defaults to False.

    Returns:
        points (tf.Tensor(float32), [batch, samples, 3]): the points to sample
            along the ray.
        z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
            along the ray.
    """
    # Decide where to sample along each ray.
    # Under the logic, all rays will be sampled at the same times.
    # Ensure the sample values match with the ray origin's shape
    t_vals = (
        tf.reshape(
            tf.linspace(0.0, 1.0, num_samples),
            (*[1 for _ in rays_origin.shape[:-1]], -1),
        )
        * tf.ones_like(rays_origin[..., :1])
    )

    # Add sample dimension
    near_bound = tf.expand_dims(near_bound * tf.ones_like(t_vals[..., 0]), -1)
    far_bound = tf.expand_dims(far_bound * tf.ones_like(t_vals[..., 0]), -1)
    # Ensure near and far bound have the same shape as t_vals and also add
    # an additional dimension for the samples

    if linear_disparity:
        # Sample linearly in inverse depth (disparity).
        z_samples = 1.0 / (
            tf.math.divide_no_nan(1.0, near_bound) * (1.0 - t_vals)
            + tf.math.divide_no_nan(1.0, far_bound) * t_vals
        )
    else:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_samples = near_bound * (1.0 - t_vals) + far_bound * t_vals

    # Perturb sampling time along each ray.
    if randomized:
        # get intervals between samples
        mids = 0.5 * (z_samples[..., 1:] + z_samples[..., :-1])
        upper = tf.concat([mids, z_samples[..., -1:]], -1)
        lower = tf.concat([z_samples[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(tf.shape(z_samples))
        z_samples = lower + (upper - lower) * t_rand

    pts = cast_rays(rays_origin, rays_direction, z_samples)

    return pts, z_samples


@tf.function(experimental_relax_shapes=True)
def setup_hierachical_sampling(
    rays_origin: tf.Tensor,
    rays_direction: tf.Tensor,
    previous_z_samples: tf.Tensor,
    previous_weights: tf.Tensor,
    num_samples: int,
    randomized: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Creates a hierachical sampling pattern.

    Args:
        ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
        ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
        previous_z_samples (tf.Tensor(float32), [batch, samples]): the previous z
            distances from a previous sampling step.
        previous_weights (tf.Tensor(float32), [batch, samples]): the previous weights
            from a previous sampling step.
        num_samples (int): the number of samples to generate.
        randomized (bool, optional): Activates noise and pertub ray features.
            Defaults to False.

    Returns:
        points (tf.Tensor(float32), [batch, samples, 3]): the points to sample
            along the ray.
        z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
            along the ray.
    """
    z_samples_mid = 0.5 * (previous_z_samples[..., 1:] + previous_z_samples[..., :-1])

    return sample_pdf(
        z_samples_mid,
        previous_weights[..., 1:-1],
        rays_origin,
        rays_direction,
        previous_z_samples,
        num_samples,
        randomized,
    )


@tf.function(experimental_relax_shapes=True)
def piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling.

    Args:
        bins (tf.Tensor(float32), [batch, num_bins + 1]): bins for sampling.
        weights (tf.Tensor(float32), [batch, num_bins]): weights per bin.
        num_samples (int): The number of samples to generate
        randomized (bool): Use randomized samples.

    Returns:
        z_samples (tf.Tensor(float32), [batch, num_samples]): the z samples along the
            ray.
    """
    # Get pdf
    eps = 1e-5
    weight_sum = tf.reduce_sum(weights, -1, keepdims=True)
    padding = tf.maximum(0.0, eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = tf.math.divide_no_nan(weights, weight_sum)
    cdf = tf.minimum(1.0, tf.cumsum(pdf[..., :-1], -1))
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf, tf.ones_like(cdf[..., :1])], -1)

    # Take uniform samples
    if randomized:
        s = 1.0 / num_samples
        u = tf.cast(tf.range(num_samples), tf.float32) * s
        cdf_shape = tf.shape(cdf)
        u = u + tf.random.uniform(
            tf.concat([cdf_shape[:-1], [num_samples]], 0),
            maxval=s - np.finfo("float32").eps,
        )
        # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = tf.minimum(u, 1.0 - np.finfo("float32").eps)
    else:
        # Match the behavior of random.uniform() by spanning [0, 1-eps].
        u = tf.linspace(0.0, 1.0 - np.finfo("float32").eps, num_samples)
        broadcast_shape = tf.concat([tf.shape(cdf)[:-1], [num_samples]], 0)
        u = u * tf.ones(broadcast_shape, dtype=tf.float32)

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = tf.reduce_max(tf.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = tf.reduce_min(
            tf.where(tf.logical_not(mask), x[..., None], x[..., -1:, None]), -2
        )
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = tf.clip_by_value(tf.math.divide_no_nan(u - cdf_g0, cdf_g1 - cdf_g0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through `samples`.
    return tf.stop_gradient(samples)


def sample_pdf(
    bins,
    weights,
    origins,
    directions,
    previous_z_samples,
    num_samples,
    randomized: bool,
):
    """Hierarchical sampling.

    Args:
        bins (tf.Tensor(float32), [batch, num_bins + 1]): bins for sampling.
        weights (tf.Tensor(float32), [batch, num_bins]): weights per bin.
        origins (tf.Tensor(float32), [batch, 3]): the ray origins.
        directions (tf.Tensor(float32), [batch, 3]): the ray directions
        previous_z_samples (tf.Tensor(float32), [batch, num_previous_samples]): the
            previous z samples.
        num_samples (int): The number of samples to generate
        randomized (bool): use randomized samples.

    Returns:
        points (tf.Tensor(float32), [batch, num_samples + num_previous_samples, 3]): the
            points to sample along the ray.
        z_samples (tf.Tensor(float32), [batch, num_samples + num_previous_samples]): the
            z samples along the ray.
    """
    z_samples = piecewise_constant_pdf(bins, weights, num_samples, randomized)

    # Compute united z_samples and sample points
    z_samples = tf.stop_gradient(  # Again make sure we do not backprop
        tf.sort(tf.concat([previous_z_samples, z_samples], axis=-1), axis=-1)
    )

    pts = cast_rays(origins, directions, z_samples)

    return (pts, z_samples)


def add_gaussian_noise(raw, noise_std, randomized):
    if (noise_std is not None) and noise_std > 0.0 and randomized:
        return raw + tf.random.normal(raw.shape, dtype=raw.dtype) * noise_std
    else:
        return raw


def get_full_image_eval_grid(
    H: int,
    W: int,
    focal: float,
    c2w: tf.Tensor,
    jitter: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get ray origins, directions from a pinhole camera.

    Args:
        H (int): the height of the image.
        W (int): the width of the image.
        focal (float): the focal length of the camera.
        c2w (tf.Tensor [3, 3]): the camera matrix.
        jitter (Optional[tf.Tensor], optional [H, W, 2]): the jitter offsets.
            Defaults to None which disables jittering.

    Returns:
        rays_origin (tf.Tensor, [H, W, 3]): the rays origin.
        rays_direction (tf.Tensor, [H, W, 3]): the rays direction.
    """
    i, j = tf.meshgrid(
        tf.range(W, dtype=tf.float32) + 0.5,
        tf.range(H, dtype=tf.float32) + 0.5,
        indexing="xy",
    )

    if jitter is not None:
        i = i + jitter[:, :, 1]
        j = j + jitter[:, :, 0]

    dirs = tf.stack(
        [
            (i - float(W) * float(0.5)) / float(focal),
            -(j - float(H) * float(0.5)) / float(focal),
            -tf.ones_like(i),
        ],
        -1,
    )
    rays_d = tf.reduce_sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d
