from typing import Dict, List, Optional, Tuple

import tensorflow as tf

import nn_utils.math_utils as math_utils
from losses import multi_gpu_wrapper
from models.nerd_net.models import NerdCoarseModel, NerdFineModel
from models.nerd_net.sgs_store import SgsStore
from nn_utils.nerf_layers import add_base_args
from nn_utils.sg_rendering import SgRenderer
from utils.training_setup_utils import (
    StateRestoration,
    StateRestorationItem,
    get_num_gpus,
)


class NerdModel(tf.keras.Model):
    def __init__(self, num_images, args, **kwargs):
        super(NerdModel, self).__init__(**kwargs)

        # Setup the models
        self.coarse_model = NerdCoarseModel(args, **kwargs)
        self.fine_model = NerdFineModel(args, **kwargs)

        # Randomize if training
        self.randomized = args.perturb == 1.0
        print("Running with pertubation:", self.randomized)

        # Setup the place where the SGs are stored
        num_illuminations = 1 if args.single_env else num_images
        self.sgs_store = SgsStore(num_illuminations, args.num_sgs, args.mean_sgs_path)

        self.rotating_object = args.rotating_object

        # Add the renderer
        self.renderer = SgRenderer()
        self.num_gpu = max(1, get_num_gpus())

        # Setup the state restoration
        states = [
            StateRestorationItem("coarse", self.coarse_model),
            StateRestorationItem("fine", self.fine_model),
            StateRestorationItem("sgs", self.sgs_store),
        ]
        self.state_restoration = StateRestoration(args, states)

    def save(self, step):
        # Save weights for step
        self.state_restoration.save(step)

    def restore(self, step: Optional[int] = None) -> int:
        # Restore weights from step or if None the latest one
        return self.state_restoration.restore(step)

    @tf.function
    def call(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        sg_illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        training=False,
        skip_rendering: bool = False,
        high_quality=False,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Evaluate the network for given ray origins and directions and camera pose

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [1, 3, 3]): the camera matrix.
            near_bound (float): the near clipping point.
            far_bound (float): the far clipping point.
            sg_illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            training (bool, optional): Whether this is a training step or not.
                Activates noise and pertub ray features if requested. Defaults to True.

        Returns:
            coarse_payload (Dict[str, tf.Tensor]): dict with the payload for the coarse
                network.
            fine_payload (Dict[str, tf.Tensor]): dict with the payload for the fine
                network.
        """

        # Get the current illumination
        sg_illumination = self.sgs_store(
            sg_illumination_idx, camera_pose[0] if self.rotating_object else None
        )

        sg_illumination = self.sgs_store.validate_sgs(sg_illumination)

        # Coarse step
        (
            coarse_payload,
            coarse_z_samples,
            coarse_weights,
        ) = self.coarse_model.render_rays(
            ray_origins,
            ray_directions,
            near_bound,
            far_bound,
            tf.stop_gradient(sg_illumination),
            randomized=training and self.randomized,
            overwrite_num_samples=(self.coarse_model.num_samples * 2)
            if high_quality
            else None,
        )

        fine_payload, _, _ = self.fine_model.render_rays(
            ray_origins,
            ray_directions,
            coarse_z_samples,
            coarse_weights,
            sg_illumination,
            ev100,
            randomized=training and self.randomized,
            skip_rendering=skip_rendering,
            overwrite_num_samples=(self.fine_model.num_samples * 2)
            if high_quality
            else None,
        )

        return coarse_payload, fine_payload

    def distributed_call(
        self,
        strategy,
        chunk_size: int,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        sg_illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        training=False,
        illumination_context_override=None,
        high_quality=False,
    ):
        if illumination_context_override is not None:
            sg_illumination_idx = tf.cast(
                tf.ones_like(sg_illumination_idx) * illumination_context_override,
                tf.int32,
            )

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        dp_df = (
            tf.data.Dataset.from_tensor_slices(
                (ray_origins.numpy(), ray_directions.numpy())
            )
            .batch(chunk_size // (2 if high_quality else 1) * get_num_gpus())
            .with_options(options)
        )

        dp_dist_df = strategy.experimental_distribute_dataset(dp_df)
        coarse_payloads: Dict[str, List[tf.Tensor]] = {}
        fine_payloads: Dict[str, List[tf.Tensor]] = {}

        def add_to_dict(to_add, main_dict):
            for k, v in to_add.items():
                arr = main_dict.get(
                    k,
                    [],
                )
                arr.extend(v)
                main_dict[k] = arr

            return main_dict

        for dp in dp_dist_df:
            rays_o, rays_d = dp
            # Render image.
            coarse_result_per_replica, fine_result_per_replica = strategy.run(
                self.call,
                (
                    rays_o,
                    rays_d,
                    camera_pose,
                    near_bound,
                    far_bound,
                    sg_illumination_idx,
                    ev100,
                    training,
                    illumination_context_override,
                    high_quality,
                ),
            )

            coarse_result = {
                k: strategy.experimental_local_results(v)
                for k, v in coarse_result_per_replica.items()
            }
            fine_result = {
                k: strategy.experimental_local_results(v)
                for k, v in fine_result_per_replica.items()
            }
            coarse_payloads = add_to_dict(coarse_result, coarse_payloads)
            fine_payloads = add_to_dict(fine_result, fine_payloads)

        coarse_payloads = {k: tf.concat(v, 0) for k, v in coarse_payloads.items()}
        fine_payloads = {k: tf.concat(v, 0) for k, v in fine_payloads.items()}

        return coarse_payloads, fine_payloads

    @tf.function
    def train_step(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        sg_illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        is_wb_ref_image: tf.Tensor,
        wb_input_value: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        target: tf.Tensor,
        target_mask: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
        lambda_color_loss: tf.Tensor,
        stop_sgs: bool,
    ) -> Tuple[
        Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor
    ]:
        """Perform a single training step.

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [1, 3, 3]): the camera matrix.
            near_bound (tf.Tensor(float32), [1]): the near clipping point.
            far_bound (tf.Tensor(float32), [1]): the far clipping point.
            sg_illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            is_wb_ref_image (tf.Tensor(bool) [1]): whether the current image is
                a reference whitebalance image.
            wb_input_value (tf.Tensor(float32) [1, 3]): if `is_wb_ref_image` then
                this is defines the whitebalance value.
            optimizer (tf.keras.optimizers.Optimizer): the optimizer to use in the
                train step.
            target (tf.Tensor(float32), [batch, 3]): the rgb target from the image.
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target from the image.
            lambda_advanced_loss (tf.Tensor(float32), [1]): current advanced loss
                interpolation value.
            lambda_color_loss (tf.Tensor(float32), [1]): current color loss
                interpolation value.

        Returns:
            coarse_payload (Dict[str, tf.Tensor]): dict with the payload for the
                coarse network.
            fine_payload (Dict[str, tf.Tensor]): dict with the payload for the fine
                network.
            loss (tf.Tensor(float32), [1]): the joint loss.
            coarse_losses (Dict[str, tf.Tensor]): a dict of loss names with the
                evaluated losses. "loss" stores the final loss of the layer.
            fine_losses (Dict[str, tf.Tensor]): a dict of loss names with the evaluated
                losses. "loss" stores the final loss of the layer.
        """
        # fine_result, coarse_result, loss, coarse_losses, fine_losses
        if is_wb_ref_image[0]:
            self.sgs_store.apply_whitebalance_to_idx(
                sg_illumination_idx,
                wb_input_value,
                ray_origins,
                ev100,
                clip_range=None if stop_sgs else (0.99, 1.01),
            )
        else:
            if stop_sgs:
                self.sgs_store.apply_whitebalance_to_idx(
                    sg_illumination_idx,
                    tf.constant([[0.8, 0.8, 0.8]], dtype=tf.float32),
                    ray_origins,
                    ev100,
                    clip_range=None if stop_sgs else (0.99, 1.01),
                    grayscale=True,
                )

        with tf.GradientTape() as tape:
            coarse_result, fine_result = self.call(
                ray_origins,
                ray_directions,
                camera_pose,
                near_bound,
                far_bound,
                sg_illumination_idx,
                ev100,
                training=True,
            )

            coarse_losses = self.coarse_model.calculate_losses(
                coarse_result, target, target_mask, lambda_advanced_loss
            )

            fine_losses = self.fine_model.calculate_losses(
                fine_result,
                target,
                target_mask,
                lambda_advanced_loss,
                lambda_color_loss,
            )

            loss = coarse_losses["loss"] + fine_losses["loss"]

        grad_vars = (
            self.coarse_model.trainable_variables
            + self.fine_model.trainable_variables
            + self.sgs_store.trainable_variables
        )
        gradients = tape.gradient(loss, grad_vars)

        gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        optimizer.apply_gradients(zip(gradients, grad_vars))

        return fine_result, coarse_result, loss, coarse_losses, fine_losses

    @tf.function
    def illumination_single_step(
        self,
        ray_directions,
        basecolor,
        metallic,
        roughness,
        normal,
        alpha,
        camera_pose,
        sg_illumination_idx,
        target,
        ev100,
        mse,
        optimizer,
    ):
        view_direction = math_utils.normalize(-1 * ray_directions)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.sgs_store.trainable_variables)

            sg = self.sgs_store(
                sg_illumination_idx,
                camera_pose if self.rotating_object else None,
            )  # Receive latest sgs

            render = self.renderer(
                sg_illuminations=sg,
                basecolor=basecolor,
                metallic=metallic,
                roughness=roughness,
                normal=normal,
                alpha=alpha,
                view_dir=view_direction,
            )

            # Auto exposure + srgb to model camera setup
            render = math_utils.white_background_compose(
                self.fine_model.camera_post_processing(render, ev100), alpha
            )

            sgs_loss = mse(
                math_utils.white_background_compose(tf.reshape(target, (-1, 3)), alpha),
                render,
            )
            tf.debugging.check_numerics(
                sgs_loss, "loss sgs: {}".format(tf.math.is_nan(sgs_loss))
            )

        grad_vars = self.sgs_store.trainable_variables
        gradients = tape.gradient(sgs_loss, grad_vars)

        optimizer.apply_gradients(zip(gradients, grad_vars))

        # self.sgs_store.ensure_sgs_correct(sg_illumination_idx)

        return sgs_loss

    def illumination_steps(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound,
        far_bound,
        sg_illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        target: tf.Tensor,
        steps: int,
        chunk_size: int = 1024,
        strategy=tf.distribute.get_strategy(),
    ) -> tf.Tensor:
        """Perform a illumination optimization step. This only performs the illumination
        with a fixed network.

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [batch, 3, 4]): the camera matrix.
            near_bound (tf.Tensor(float32), [1]): the near clipping point.
            far_bound (tf.Tensor(float32), [1]): the far clipping point.
            sg_illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            optimizer (tf.keras.optimizers.Optimizer): the optimizer to use in the
                train step.
            target (tf.Tensor(float32), [batch, 3]): the rgb target from the image.
            steps (int): the number of optimization steps to perform.
            chunk_size (int): If specified runs the sampling in
                batches. Runs everything jointly if 0.

        Returns:
            tf.Tensor(float32), [1]: the loss after the optimization
        """
        mse = multi_gpu_wrapper(tf.keras.losses.MeanSquaredError, target.shape[0])

        _, fine_result = self.distributed_call(
            strategy,
            chunk_size,
            ray_origins,
            ray_directions,
            camera_pose,
            near_bound,
            far_bound,
            sg_illumination_idx,
            ev100,
            False,
        )

        dp_df = tf.data.Dataset.from_tensor_slices(
            (
                ray_directions,
                target,
                fine_result["basecolor"],
                fine_result["metallic"],
                fine_result["roughness"],
                fine_result["normal"],
                fine_result["acc_alpha"][..., None],
            )
        ).batch(chunk_size * get_num_gpus())
        dp_dist_df = strategy.experimental_distribute_dataset(dp_df)

        for i in tf.range(steps):
            total_loss = 0
            for dp in dp_dist_df:
                ray_d, trgt, bcol, mtl, rgh, nrm, alp = dp
                illum_loss_per_replica = strategy.run(
                    self.illumination_single_step,
                    (
                        ray_d,
                        bcol,
                        mtl,
                        rgh,
                        nrm,
                        alp,
                        camera_pose[:1],
                        sg_illumination_idx[:1],
                        trgt,
                        ev100,
                        mse,
                        optimizer,
                    ),
                )
                illum_loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, illum_loss_per_replica, axis=None
                )
                total_loss = total_loss + illum_loss

        return total_loss

    @classmethod
    def add_args(cls, parser):
        """Add the base nerf arguments to the parser with addition
        to the specific NeRD ones.

        Args:
            parser (ArgumentParser): the current ArgumentParser.

        Returns:
            ArgumentParser: the modified ArgumentParser for call chaining
        """
        add_base_args(parser)
        parser.add_argument(
            "--coarse_samples",
            type=int,
            default=64,
            help="number of coarse samples per ray in a fixed grid",
        )
        parser.add_argument(
            "--fine_samples",
            type=int,
            default=128,
            help="number of additional samples per ray based on the coarse samples",
        )
        parser.add_argument(
            "--fourier_frequency",
            type=int,
            default=10,
            help="log2 of max freq for positional encoding",
        )
        parser.add_argument(
            "--net_width", type=int, default=256, help="channels per layer"
        )
        parser.add_argument(
            "--net_depth", type=int, default=8, help="layers in network"
        )

        parser.add_argument(
            "--num_sgs",
            type=int,
            default=24,
            help=(
                "Number of spherical Gaussians per illumination. "
                "Default is 24 and mean"
            ),
        )
        parser.add_argument(
            "--mean_sgs_path",
            type=str,
            default="data/nerd/mean_sgs.npy",
            help="Path to the initial mean spherical Gaussians initialization file.",
        )

        parser.add_argument(
            "--rotating_object",
            action="store_true",
            help=(
                "The object is rotating instead of the camera. The illumination then "
                "needs to stay static"
            ),
        )
        parser.add_argument(
            "--single_env",
            action="store_true",
            help="All input images are captured under a single environment",
        )

        # Coarse configs
        parser.add_argument(
            "-lindisp",
            "--linear_disparity_sampling",
            action="store_true",
            help="Coarse sampling linearly in disparity rather than depth",
        )

        parser.add_argument(
            "--sgs_condense_features",
            type=int,
            default=16,
            help=(
                "The number features the sgs condense network reduces the spherical "
                "gaussians to."
            ),
        )

        # Fine configs

        parser.add_argument(
            "--brdf_interpolation_z",
            type=int,
            default=2,
            choices=[0, 1, 2, 3],
            help=(
                "The BRDF is not predicted directly but a BRDF embedding space is "
                "created which is then used decoded to a BRDF. This defines the size "
                "of the latent vector. 0 disables the interpolation",
            ),
        )

        parser.add_argument(
            "--direct_rgb",
            action="store_true",
            help=(
                "Also performs a direct RGB color prediction. This is useful in the "
                "beginning of the training."
            ),
        )

        parser.add_argument(
            "--mlp_normal",
            action="store_true",
            help="The MLP generates the normal instead of deriving it from gradients.",
        )

        return parser
