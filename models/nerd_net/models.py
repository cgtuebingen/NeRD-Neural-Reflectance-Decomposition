from typing import Dict, Tuple, Optional

import tensorflow as tf

import nn_utils.math_utils as math_utils
from losses import multi_gpu_wrapper
from losses.nerd import segmentation_mask_loss
from nn_utils.nerf_layers import (
    FourierEmbedding,
    add_gaussian_noise,
    setup_fixed_grid_sampling,
    setup_hierachical_sampling,
    split_sigma_and_payload,
    volumetric_rendering,
)
from nn_utils.sg_rendering import SgRenderer
from utils.training_setup_utils import get_num_gpus


class SgsCondenseModel(tf.keras.Model):
    def __init__(self, condense_features: int = 48, normalize=True, **kwargs):
        super(SgsCondenseModel, self).__init__(**kwargs)

        self.normalize = normalize

        self.condense_features = condense_features

        self.condenser = None
        if condense_features > 0:
            self.condenser = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer((24 * 7,)),
                    tf.keras.layers.Dense(condense_features, activation="linear"),
                ]
            )  # Just a single dense layer
            print("SGs condense\n", self.condenser.summary())

    @tf.function
    def call(self, sgs):
        # The amplitude can have a huge range. Normalize the amplitude
        if self.normalize:
            sgs = tf.reshape(sgs, [-1, 24, 7])
            # This is not a per channel norm
            normAmpl = sgs[..., :3] / tf.maximum(
                tf.reduce_max(sgs[..., :3], -1, keepdims=True),
                1e-3,
            )
            sgs = tf.concat([normAmpl, sgs[..., 3:]], -1)

        sgs = tf.reshape(sgs, [-1, 24 * 7])

        if self.condense_features > 0:
            return self.condenser(sgs)
        else:
            return sgs


class NerdCoarseModel(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(NerdCoarseModel, self).__init__(**kwargs)

        self.num_samples = args.coarse_samples
        self.linear_disparity_sampling = args.linear_disparity_sampling
        self.raw_noise_std = args.raw_noise_std

        sgs_condense_features = args.sgs_condense_features

        # Start with fourier embedding
        self.pos_embedder = FourierEmbedding(args.fourier_frequency)
        main_net = [
            tf.keras.layers.InputLayer(
                (self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        # Then add the main layers
        for _ in range(args.net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.net_width,
                    activation="relu",
                )
            )
        # Build network stack
        self.main_net_first = tf.keras.Sequential(main_net)

        main_net = [
            tf.keras.layers.InputLayer(
                (args.net_width + self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        for _ in range(args.net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.net_width,
                    activation="relu",
                )
            )
        self.main_net_second = tf.keras.Sequential(main_net)

        # Sigma is a own output not conditioned on the illumination
        self.sigma_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((args.net_width,)),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )
        print("Coarse sigma\n", self.sigma_net.summary())

        self.sgs_condense_net = SgsCondenseModel(sgs_condense_features, **kwargs)

        # Build a small conditional net which gets the embedding from the main net
        # plus the illumination
        self.conditional_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((args.net_width + sgs_condense_features,)),
                tf.keras.layers.Dense(args.net_width, activation="relu"),
                tf.keras.layers.Dense(3, activation="linear"),
            ]
        )
        print("Coarse conditional\n", self.conditional_net.summary())

        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu
        self.mse = multi_gpu_wrapper(
            tf.keras.losses.MeanSquaredError,
            self.global_batch_size,
        )
        self.alpha_loss = segmentation_mask_loss(
            self.global_batch_size,
        )

    def payload_to_parmeters(
        self, raymarched_payload: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        ret = {"rgb": tf.sigmoid(raymarched_payload)}  # Just RGB here

        for k in ret:
            tf.debugging.check_numerics(
                ret[k], "output {}: {}".format(k, tf.math.is_nan(ret[k]))
            )
        return ret

    @tf.function
    def call(
        self, pts: tf.Tensor, sgs_illumination: tf.Tensor, randomized: bool = False
    ) -> tf.Tensor:
        """Evaluates the network for all points and condition it on the illumination

        Args:
            pts (tf.Tensor(float32), [..., 3]): the points where to evaluate the
                network.
            sgs_illumination (tf.Tensor(float32), [1, 24, 7]): the sgs for a single
                image.
            randomized (bool): use randomized sigma noise. Defaults to False.

        Returns:
            sigma_payload (tf.Tensor(float32), [..., 1 + 3]): the sigma and the
                rgb.
        """
        # Condense the illumination first
        # Do not optimize the context from the coarse network
        sgs_illumination = tf.stop_gradient(sgs_illumination)
        illumination_context = self.sgs_condense_net(sgs_illumination)

        pts_embed = self.pos_embedder(tf.reshape(pts, (-1, pts.shape[-1])))
        # Run points through main net. The embedding is flat B*S, C
        main_embd = self.main_net_first(pts_embed)
        main_embd = self.main_net_second(tf.concat([main_embd, pts_embed], -1))

        # Extract sigma
        sigma = self.sigma_net(main_embd)
        sigma = add_gaussian_noise(sigma, self.raw_noise_std, randomized)

        # Prepare the illumination context to fit the shape
        # Tile in batch direction (which is batch * samples)
        illumination_context = illumination_context * tf.ones_like(main_embd[:, :1])

        # Concat main embedding and the context
        conditional_input = tf.concat([main_embd, illumination_context], -1)
        # Predict the conditional RGB
        rgb = self.conditional_net(conditional_input)

        # Build the final output
        sigma_payload_flat = tf.concat([sigma, rgb], -1)
        new_shape = tf.concat([tf.shape(pts)[:-1], sigma_payload_flat.shape[-1:]], 0)
        return tf.reshape(sigma_payload_flat, new_shape)

    @tf.function
    def render_rays(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        near_bound: float,
        far_bound: float,
        sgs_illumination: tf.Tensor,
        randomized: bool = False,
        overwrite_num_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Render the rays

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            near_bound (float): the near clipping point.
            far_bound (float): the far clipping point.
            sgs_illumination (tf.Tensor(float32), [1, 24, 7]): the sgs for a single
                image.
            randomized (bool, optional): Activates noise and pertub ray features.
                Defaults to False.

        Returns:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
                along the ray.
            weights (tf.Tensor(float32) [batch, num_samples]): the weights along the
                ray. That is the accumulated product of the individual alphas.
        """
        points, z_samples = setup_fixed_grid_sampling(
            ray_origins,
            ray_directions,
            near_bound,
            far_bound,
            self.num_samples
            if overwrite_num_samples is None
            else overwrite_num_samples,
            randomized=randomized,
            linear_disparity=self.linear_disparity_sampling,
        )

        raw = self.call(points, sgs_illumination, randomized=randomized)

        sigma, payload_raw = split_sigma_and_payload(raw)

        payload, weights = volumetric_rendering(
            sigma,
            payload_raw,
            z_samples,
            ray_directions,
            self.payload_to_parmeters,
            ["rgb"],
            sigma_activation=tf.nn.relu,
        )

        return payload, z_samples, weights

    @tf.function
    def calculate_losses(
        self,
        payload: Dict[str, tf.Tensor],
        target: tf.Tensor,
        target_mask: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Calculates the losses

        Args:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            target (tf.Tensor(float32), [batch, 3]): the RGB target of the
                respective ray
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target for the respective ray.

        Returns:
            Dict[str, tf.Tensor]: a dict of loss names with the evaluated losses.
                "loss" stores the final loss of the layer.
        """
        inverse_advanced = 1 - lambda_advanced_loss

        target_masked = tf.stop_gradient(
            math_utils.white_background_compose(target, target_mask)
        )

        alpha_loss = self.alpha_loss(
            payload["individual_alphas"],
            payload["acc_alpha"][..., None],
            target_mask,
            0,
        )

        image_loss = self.mse(target_masked, payload["rgb"])

        final_loss = image_loss + alpha_loss * 0.4 * inverse_advanced

        losses = {
            "loss": final_loss,
            "image_loss": image_loss,
            "alpha_loss": alpha_loss,
        }

        for k in losses:
            tf.debugging.check_numerics(
                losses[k], "output {}: {}".format(k, tf.math.is_nan(losses[k]))
            )

        return losses


class NerdBrdfAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim=5, **kwargs):
        super(NerdBrdfAutoEncoder, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((input_dim,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(
                    latent_dim,
                    activation="linear",
                    kernel_regularizer=tf.keras.regularizers.L2(0.1),
                ),
                tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -40, 40)),
            ]
        )  # Condense to latent dim features
        print("BRDF Encoder\n", self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((latent_dim,)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(output_dim, activation="linear"),
            ]
        )  # And expand it again
        print("BRDF Decoder\n", self.decoder.summary())

    @tf.function
    def call(self, x):
        z = self.encoder(x)
        dec = self.decoder(z)

        return dec, z

    def get_kernel_regularization(self):
        return tf.math.reduce_sum(self.encoder.losses)


class NerdFineModel(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(NerdFineModel, self).__init__(**kwargs)

        self.num_samples = args.fine_samples
        self.coarse_samples = args.coarse_samples
        self.raw_noise_std = args.raw_noise_std

        self.direct_rgb = args.direct_rgb
        self.mlp_normal = args.mlp_normal

        # Start with fourier embedding
        self.pos_embedder = FourierEmbedding(args.fourier_frequency)
        main_net = [
            tf.keras.layers.InputLayer(
                (self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        # Then add the main layers
        for _ in range(args.net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.net_width,
                    activation="relu",
                )
            )
        # Build network stack
        self.main_net_first = tf.keras.Sequential(main_net)

        main_net = [
            tf.keras.layers.InputLayer(
                (args.net_width + self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        for _ in range(args.net_depth // 2):
            main_net.append(
                tf.keras.layers.Dense(
                    args.net_width,
                    activation="relu",
                )
            )
        self.main_net_second = tf.keras.Sequential(main_net)

        # Add a final layer for the main net which predicts sigma plus eventual
        # direct_rgb or mlp_normals
        self.main_final_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(args.net_width),
                tf.keras.layers.Dense(
                    # Decide how many features we need (at least 1 - sigma)
                    1
                    + (3 if args.direct_rgb else 0)
                    + (3 if self.mlp_normal else 0)
                    + (5 if args.brdf_interpolation_z <= 0 else 0),
                    activation="linear",
                ),
            ]
        )
        print("Fine final\n", self.main_final_net.summary())

        # Build the BRDF auto encoder
        self.brdf_encoder = (
            NerdBrdfAutoEncoder(args.net_width, args.brdf_interpolation_z, **kwargs)
            if args.brdf_interpolation_z > 0
            else None
        )

        # Add the renderer
        self.renderer = SgRenderer(eval_background=True)

        # Add losses
        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu

        self.mse = multi_gpu_wrapper(
            tf.keras.losses.MeanSquaredError, self.global_batch_size
        )
        self.alpha_loss = segmentation_mask_loss(self.global_batch_size)

    def payload_to_parmeters(
        self, raymarched_payload: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        ret = {}
        start = 0  # Index where the extraction starts
        if self.direct_rgb:
            ret["direct_rgb"] = tf.sigmoid(raymarched_payload[..., start : start + 3])
            start += 3

        # Ensure the value range is -1 to 1 if mlp normals are used
        normal_fn = tf.nn.tanh if self.mlp_normal else tf.identity
        ret["normal"] = math_utils.normalize(
            normal_fn(raymarched_payload[..., start : start + 3])
        )
        start += 3

        # BRDF parameters
        ret["basecolor"] = tf.sigmoid(raymarched_payload[..., start : start + 3])
        start += 3
        ret["metallic"] = tf.sigmoid(raymarched_payload[..., start : start + 1])
        start += 1
        ret["roughness"] = tf.sigmoid(raymarched_payload[..., start : start + 1])
        start += 1

        if raymarched_payload.shape[-1] > start:
            ret["brdf_embedding"] = raymarched_payload[..., start:]

        for k in ret:
            tf.debugging.check_numerics(
                ret[k], "output {}: {}".format(k, tf.math.is_nan(ret[k]))
            )

        return ret

    @tf.function
    def call(self, pts, randomized=False) -> tf.Tensor:
        """Evaluates the network for all points

        Args:
            pts (tf.Tensor(float32), [..., 3]): the points where to evaluate the
                network.
            randomized (bool): use randomized sigma noise. Defaults to False.

        Returns:
            sigma_payload (tf.Tensor(float32), [..., 1 + payload_channels]): the
                sigma and the payload.
        """
        # Tape to calculate the normal gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            if not self.mlp_normal:  # Normals are not directly predicted
                tape.watch(pts)  # Watch pts as it is not a variable

            pts_flat = tf.reshape(pts, (-1, pts.shape[-1]))
            pts_embed = self.pos_embedder(pts_flat)

            # Call the main network
            main_embd = self.main_net_first(pts_embed)
            main_embd = self.main_net_second(tf.concat([main_embd, pts_embed], -1))

            # Split sigma and payload
            sigma_payload = self.main_final_net(main_embd)
            sigma = sigma_payload[..., :1]

            payload = sigma_payload[..., 1:]

        # Build payload list
        main_payload_end = (3 if self.direct_rgb else 0) + (3 if self.mlp_normal else 0)
        full_payload_list = [payload[..., :main_payload_end]]

        if not self.mlp_normal:  # Normals are not directly predicted
            # Normals are derived from the gradient of sigma wrt. to the input points
            normal = -1 * tape.gradient(sigma, pts_flat)
            full_payload_list.append(normal)

        # Evaluate the BRDF
        if self.brdf_encoder is not None:
            brdf, brdf_embedding = self.brdf_encoder(main_embd)
            full_payload_list.append(brdf)
            full_payload_list.append(brdf_embedding)
        else:
            brdf = payload[..., main_payload_end:]
            full_payload_list.append(brdf)

        # Add noise
        sigma = add_gaussian_noise(sigma, self.raw_noise_std, randomized)

        # Build the final output
        sigma_payload_flat = tf.concat([sigma] + full_payload_list, -1)
        new_shape = tf.concat([tf.shape(pts)[:-1], sigma_payload_flat.shape[-1:]], 0)
        return tf.reshape(sigma_payload_flat, new_shape)

    @tf.function
    def render_rays(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        previous_z_samples: tf.Tensor,
        previous_weights: tf.Tensor,
        sgs_illumination: tf.Tensor,
        ev100: tf.Tensor,
        randomized: bool = False,
        skip_rendering: bool = False,
        overwrite_num_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Render the rays

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            previous_z_samples (tf.Tensor(float32), [batch, samples]): the previous
                distances to sample along the ray.
            previous_weights (tf.Tensor(float32) [batch, num_samples]): the previous
                weights along the ray. That is the accumulated product of the
                individual alphas.
            sgs_illumination (tf.Tensor(float32), [1, 24, 7]): the sgs for a single
                image.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            randomized (bool, optional): Activates noise and pertub ray features.
                Defaults to False.

        Returns:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
                along the ray.
            weights (tf.Tensor(float32) [batch, num_samples]): the weights along the
                ray. That is the accumulated product of the individual alphas.
        """
        points, z_samples = setup_hierachical_sampling(
            ray_origins,
            ray_directions,
            previous_z_samples,
            previous_weights,
            self.num_samples
            if overwrite_num_samples is None
            else overwrite_num_samples,
            randomized=randomized,
        )

        raw = self.call(points, randomized=randomized)

        sigma, payload_raw = split_sigma_and_payload(raw)

        payload, weights = volumetric_rendering(
            sigma,
            payload_raw,
            z_samples,
            ray_directions,
            self.payload_to_parmeters,
            ["basecolor", "metallic", "roughness"]
            + (
                ["direct_rgb"] if self.direct_rgb else []
            ),  # Check if direct rgb is requested
            sigma_activation=tf.nn.relu,
        )

        # Ensure the raymarched normal is still normalized
        payload["normal"] = math_utils.white_background_compose(
            math_utils.normalize(payload["normal"]),
            payload["acc_alpha"][:, None],
        )

        if not skip_rendering:
            # View direction is inverse ray direction.
            # Which points from surface to camera
            view_direction = math_utils.normalize(-1 * ray_directions)

            rgb_render = self.renderer(
                sg_illuminations=sgs_illumination,
                basecolor=payload["basecolor"],
                metallic=payload["metallic"],
                roughness=payload["roughness"],
                normal=payload["normal"],
                alpha=payload["acc_alpha"],
                view_dir=view_direction,
            )
            tf.debugging.check_numerics(
                rgb_render,
                "output Re-render: {}".format(tf.math.is_nan(rgb_render)),
            )
            payload["hdr_rgb"] = rgb_render
            payload["rgb"] = math_utils.white_background_compose(
                self.camera_post_processing(rgb_render, ev100),
                payload["acc_alpha"][..., None],
            )

        return (
            payload,
            z_samples,
            weights,
        )

    @tf.function
    def camera_post_processing(self, hdr_rgb: tf.Tensor, ev100: tf.Tensor) -> tf.Tensor:
        """Applies the camera auto-exposure post-processing

        Args:
            hdr_rgb (tf.Tensor(float32), [..., 3]): the HDR input fromt the
                rendering step.
            ev100 ([type]): [description]

        Returns:
            tf.Tensor: [description]
        """
        exp_val = tf.stop_gradient(math_utils.ev100_to_exp(ev100))
        ldr_rgb = math_utils.linear_to_srgb(math_utils.saturate(hdr_rgb * exp_val))

        return ldr_rgb

    @tf.function
    def calculate_losses(
        self,
        payload: Dict[str, tf.Tensor],
        target: tf.Tensor,
        target_mask: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
        lambda_color_loss: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Calculates the losses

        Args:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            target (tf.Tensor(float32), [batch, 3]): the RGB target of the
                respective ray
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target for the respective ray.
            lambda_advanced_loss (tf.Tensor(float32), [1]): current advanced loss
                interpolation value.
            lambda_color_loss (tf.Tensor(float32), [1]): current color loss
                interpolation value.

        Returns:
            Dict[str, tf.Tensor]: a dict of loss names with the evaluated losses.
                "loss" stores the final loss of the layer.
        """
        inverse_advanced = 1 - lambda_advanced_loss
        inverse_color = 1 - lambda_color_loss

        target_masked = tf.stop_gradient(
            math_utils.white_background_compose(target, target_mask)
        )

        alpha_loss = self.alpha_loss(
            payload["individual_alphas"],
            payload["acc_alpha"][..., None],
            target_mask,
            0,
        )

        # Calculate losses
        direct_img_loss = 0
        if self.direct_rgb:
            direct_img_loss = self.mse(target_masked, payload["direct_rgb"])

        image_loss = self.mse(target_masked, payload["rgb"])

        brdf_embedding_loss = 0
        if "brdf_embedding" in payload:
            brdf_embedding_loss = self.brdf_encoder.get_kernel_regularization()

        final_loss = (
            image_loss * tf.maximum(inverse_color, 0.1)
            + alpha_loss * inverse_advanced
            + direct_img_loss * lambda_color_loss
            + brdf_embedding_loss
        )

        losses = {
            "loss": final_loss,
            "image_loss": image_loss,
            "alpha_loss": alpha_loss,
            "brdf_embedding_loss": brdf_embedding_loss,
        }

        if self.direct_rgb:
            losses["direct_rgb_loss"] = direct_img_loss

        for k in losses:
            tf.debugging.check_numerics(
                losses[k], "output {}: {}".format(k, tf.math.is_nan(losses[k]))
            )

        return losses
