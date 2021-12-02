from typing import Optional

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from nn_utils.sg_rendering import SgRenderer


class SgsStore(tf.keras.layers.Layer):
    def __init__(
        self,
        num_samples: int,
        num_sgs: int = 24,
        base_sgs_path: Optional[str] = None,
        compress_sharpness: bool = False,
        compress_amplitude: bool = False,
        **kwargs,
    ):
        super(SgsStore, self).__init__(**kwargs)

        self.compress_amplitude = compress_amplitude
        self.compress_sharpness = compress_sharpness
        if base_sgs_path is not None and base_sgs_path != "None":
            base_sgs = np.load(base_sgs_path)
        else:
            base_axis_sharpness = SgsStore.setup_uniform_axis_sharpness(num_sgs)
            base_amplitude = np.ones_like(base_axis_sharpness[..., :3])
            base_sgs = np.concatenate([base_amplitude, base_axis_sharpness])

        if compress_sharpness or compress_amplitude:
            base_sgs = tf.concat(
                [
                    math_utils.safe_log(base_sgs[..., 0:3])
                    if compress_amplitude
                    else base_sgs[..., 0:3],
                    base_sgs[..., 3:6],
                    math_utils.safe_log(base_sgs[..., 6:7])
                    if compress_sharpness
                    else base_sgs[..., 6:7],
                ],
                -1,
            )

        self.num_samples = num_samples
        self.num_sgs = num_sgs
        sgs_np = [base_sgs for _ in range(num_samples)]

        self.init_sgs = tf.convert_to_tensor(sgs_np, dtype=tf.float32)
        self.sgs = tf.Variable(initial_value=self.init_sgs, trainable=True)

        self.renderer = SgRenderer()

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],  # Batch size
            self.num_sgs,
            7,
        )

    @tf.function
    def apply_whitebalance_to_idx(
        self,
        idx,
        wb_value,
        rays_o,
        ev100,
        clip_range=(0.99, 1.01),
        grayscale=False,
    ):
        if self.num_samples == 1:
            idx = tf.convert_to_tensor(0, tf.int32)
        else:
            idx = idx[0]

        exp_val = math_utils.ev100_to_exp(ev100)

        normal_dir = math_utils.normalize(rays_o[:1])
        wb_scene = (
            self.renderer(
                sg_illuminations=self.call(idx),
                basecolor=tf.constant([[0.8, 0.8, 0.8]], dtype=tf.float32),
                metallic=tf.constant([[0.0]], dtype=tf.float32),
                roughness=tf.constant([[1.0]], dtype=tf.float32),
                normal=normal_dir,
                alpha=tf.constant([1.0], dtype=tf.float32),
                view_dir=normal_dir,
            )
            * exp_val
        )  # [B, C]

        factor = wb_value / tf.maximum(wb_scene, 1e-1)

        if grayscale:
            factor = tf.reduce_mean(factor, -1, keepdims=True)

        if clip_range is not None:
            factor = tf.clip_by_value(factor, clip_range[0], clip_range[1])

        if self.compress_amplitude:
            environment_ampl = math_utils.safe_log(
                math_utils.safe_exp(self.sgs[idx, :, :3]) * factor
            )
        else:
            environment_ampl = self.sgs[idx, :, :3] * factor
        environment_remain = self.sgs[idx, :, 3:]
        new_env = tf.concat([environment_ampl, environment_remain], -1)

        new_sgs = tf.tensor_scatter_nd_update(
            self.sgs, [[idx]], new_env[None, ...]
        )  # Ensure we do not backprop to factor. This is a fixed step

        self.sgs.assign(new_sgs)

    def validate_sgs(self, sgs):
        if self.compress_amplitude:
            ampl = math_utils.safe_log(math_utils.safe_exp(sgs[..., :3]))
        else:
            ampl = tf.abs(sgs[..., :3])
        axis = math_utils.normalize(sgs[..., 3:6])
        if self.compress_sharpness:
            sharpness = math_utils.safe_log(
                math_utils.saturate(math_utils.safe_exp(sgs[..., 6:]), 0.5, 30)
            )
        else:
            sharpness = tf.math.maximum(sgs[..., 6:], 1e-6)

        # Ensure we do not backprop. This is a fixed step
        return tf.concat([ampl, axis, sharpness], -1)

    def ensure_sgs_correct(self, idx):
        sgs = self.call(idx)

        new_env = self.validate_sgs(sgs)

        new_sgs = tf.tensor_scatter_nd_update(self.sgs, [[idx]], new_env[None, ...])
        self.sgs.assign(new_sgs)

    def call(self, idxs, rotating_object_pose: Optional[tf.Tensor] = None):
        if self.num_samples == 1:
            # Only a single sgs exist. No need to gather
            # But check if a counterrotation is required
            if rotating_object_pose is not None:
                rotation_matrix = tf.linalg.inv(rotating_object_pose[:3, :3])
                environment_ampl = self.sgs[..., :3]
                environment_axis = self.sgs[..., 3:6]
                environment_sharpness = self.sgs[..., 6:]

                environment_axis = environment_axis[..., None, :] * rotation_matrix
                environment_axis = tf.reduce_sum(environment_axis, -1)

                return tf.concat(
                    [environment_ampl, environment_axis, environment_sharpness], -1
                )
            else:
                return self.sgs
        else:
            dtype = idxs.dtype
            if dtype != "int32" and dtype != "int64":
                idxs = tf.cast(idxs, "int32")

            sgs = tf.gather(self.sgs, idxs)

            return sgs

    @classmethod
    def setup_uniform_axis_sharpness(cls, num_sgs) -> np.ndarray:
        def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.sum(x * y, axis=-1, keepdims=True)

        def magnitude(x: np.ndarray) -> np.ndarray:
            return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))

        def normalize(x: np.ndarray) -> np.ndarray:
            return x / magnitude(x)

        axis = []
        inc = np.pi * (3.0 - np.sqrt(5.0))
        off = 2.0 / num_sgs
        for k in range(num_sgs):
            y = k * off - 1.0 + (off / 2.0)
            r = np.sqrt(1.0 - y * y)
            phi = k * inc
            axis.append(normalize(np.array([np.cos(phi) * r, np.sin(phi) * r, y])))

        minDp = 1.0
        for a in axis:
            h = normalize(a + axis[0])
            minDp = min(minDp, dot(h, axis[0]))

        sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)

        axis = np.stack(axis, 0)  # Shape: num_sgs, 3
        sharpnessNp = np.ones((num_sgs, 1)) * sharpness
        return np.concatenate([axis, sharpnessNp], -1)
