from typing import Tuple

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils


class SgRenderer(tf.keras.layers.Layer):
    def __init__(
        self,
        eval_background: bool = False,
        compress_sharpness: bool = False,
        compress_amplitude: bool = False,
        **kwargs
    ):
        super(SgRenderer, self).__init__(**kwargs)
        self.eval_background = eval_background
        self.compress_sharpness = compress_sharpness
        self.compress_amplitude = compress_amplitude

    @tf.function
    def call(
        self,
        sg_illuminations: tf.Tensor,
        basecolor: tf.Tensor,
        metallic: tf.Tensor,
        roughness: tf.Tensor,
        normal: tf.Tensor,
        alpha: tf.Tensor,
        view_dir: tf.Tensor,
    ):
        with tf.name_scope("Renderer"):
            lin_basecolor = math_utils.srgb_to_linear(basecolor)
            diffuse = lin_basecolor * (1 - metallic)  # Only diffuse is metallic is 0
            # Interpolate between 0.04 base reflectivity where non-metallic
            # and specular color (from basecolor)
            specular = math_utils.mix(
                tf.ones_like(lin_basecolor) * 0.04, lin_basecolor, metallic
            )

            normal = tf.where(normal == tf.zeros_like(normal), view_dir, normal)

            diffuse = tf.expand_dims(diffuse, 1)
            specular = tf.expand_dims(specular, 1)
            roughness = tf.expand_dims(roughness, 1)
            normal = tf.expand_dims(math_utils.normalize(normal), 1)
            view_dir = tf.expand_dims(math_utils.normalize(view_dir), 1)

            env = None
            if self.eval_background:
                # Evaluate everything for the environment
                env = self._sg_evaluate(sg_illuminations, view_dir)
                # And sum all contributions
                env = tf.reduce_sum(env, 1)

            # Evaluate BRDF
            brdf = self._brdf_eval(
                sg_illuminations,
                diffuse,
                specular,
                roughness,
                normal,
                view_dir,
            )
            # And sum the contributions
            brdf = tf.reduce_sum(brdf, 1)

            if self.eval_background:
                if len(alpha.shape) == 1:
                    alpha = tf.expand_dims(alpha, 1)
                alpha = tf.clip_by_value(alpha, 0, 1)

                return tf.nn.relu(brdf * alpha + env * (1 - alpha))
            else:
                return tf.nn.relu(brdf)

    @tf.function
    def _brdf_eval(
        self,
        sg_illuminations: tf.Tensor,
        diffuse: tf.Tensor,
        specular: tf.Tensor,
        roughness: tf.Tensor,
        normal: tf.Tensor,
        view_dir: tf.Tensor,
    ):
        with tf.name_scope("BRDF_evaluation"):
            v = view_dir
            diff = diffuse
            spec = specular
            norm = normal
            rogh = roughness

            ndf = self._distribution_term(norm, rogh)

            warped_ndf = self._sg_warp_distribution(ndf, v)
            _, warpDir, _ = self._extract_sg_components(warped_ndf)

            ndl = math_utils.saturate(math_utils.dot(norm, warpDir))
            ndv = math_utils.saturate(math_utils.dot(norm, v))
            h = math_utils.normalize(warpDir + v)
            ldh = math_utils.saturate(math_utils.dot(warpDir, h))

            diffuse_eval = self._evaluate_diffuse(sg_illuminations, diff, norm)  # * ndl
            specular_eval = self._evaluate_specular(
                sg_illuminations, spec, rogh, warped_ndf, ndl, ndv, ldh
            )

            tf.debugging.check_numerics(
                diffuse_eval,
                "output diffuse_eval: {}".format(tf.math.is_nan(diffuse_eval)),
            )
            tf.debugging.check_numerics(
                specular_eval,
                "output specular_eval: {}".format(tf.math.is_nan(specular_eval)),
            )

            return diffuse_eval + specular_eval

    @tf.function
    def _evaluate_diffuse(
        self, sg_illuminations: tf.Tensor, diffuse: tf.Tensor, normal: tf.Tensor
    ) -> tf.Tensor:
        with tf.name_scope("Diffuse"):
            diff = diffuse / np.pi

            _, s_axis, s_sharpness = self._extract_sg_components(sg_illuminations)
            mudn = math_utils.saturate(math_utils.dot(s_axis, normal))

            c0 = 0.36
            c1 = 1.0 / (4.0 * c0)

            eml = math_utils.safe_exp(-s_sharpness)
            em2l = eml * eml
            rl = tf.math.reciprocal_no_nan(s_sharpness)

            scale = 1.0 + 2.0 * em2l - rl
            bias = (eml - em2l) * rl - em2l

            x = math_utils.safe_sqrt(1.0 - scale)
            x0 = c0 * mudn
            x1 = c1 * x

            n = x0 + x1

            y_cond = tf.less_equal(tf.abs(x0), x1)
            y_true = n * (n / tf.maximum(x, 1e-6))
            y_false = mudn
            y = tf.where(y_cond, y_true, y_false)

            res = scale * y + bias

            res = res * self._sg_integral(sg_illuminations) * diff

            return res

    @tf.function
    def _evaluate_specular(
        self,
        sg_illuminations: tf.Tensor,
        specular: tf.Tensor,
        roughness: tf.Tensor,
        warped_ndf: tf.Tensor,
        ndl: tf.Tensor,
        ndv: tf.Tensor,
        ldh: tf.Tensor,
    ) -> tf.Tensor:
        with tf.name_scope("Specular"):
            a2 = math_utils.saturate(roughness * roughness, 1e-3)

            with tf.name_scope("Distribution"):
                D = self._sg_inner_product(warped_ndf, sg_illuminations)

            with tf.name_scope("Geometry"):
                G = self._ggx(a2, ndl) * self._ggx(a2, ndv)

            with tf.name_scope("Fresnel"):
                powTerm = tf.pow(1.0 - ldh, 5)
                F = specular + (1.0 - specular) * powTerm

            output = D * G * F * ndl

            return tf.nn.relu(output)

    @tf.function
    def _ggx(self, a2: tf.Tensor, ndx: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("Geometric"):
            return tf.math.reciprocal_no_nan(
                ndx + math_utils.safe_sqrt(a2 + (1 - a2) * ndx * ndx)
            )

    @tf.function
    def _distribution_term(self, d: tf.Tensor, roughness: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("Distribution"):
            a2 = math_utils.saturate(roughness * roughness, 1e-3)

            ret = self._stack_sg_components(
                math_utils.to_vec3(tf.math.reciprocal_no_nan(np.pi * a2)),
                d,
                2.0 / tf.maximum(a2, 1e-6),
            )

            return ret

    @tf.function
    def _sg_warp_distribution(self, ndfs: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("WarpDistribution"):
            ndf_amplitude, ndf_axis, ndf_sharpness = self._extract_sg_components(ndfs)

            ret = tf.concat(
                [
                    ndf_amplitude,
                    math_utils.reflect(-v, ndf_axis),
                    tf.math.divide_no_nan(
                        ndf_sharpness,
                        (4.0 * math_utils.saturate(math_utils.dot(ndf_axis, v), 1e-4)),
                    ),
                ],
                -1,
            )

            return ret

    @tf.function
    def _stack_sg_components(self, amplitude, axis, sharpness):
        return tf.concat(
            [
                math_utils.safe_log(amplitude)
                if self.compress_amplitude
                else amplitude,
                axis,
                math_utils.safe_log(math_utils.saturate(sharpness, 0.5, 30))
                if self.compress_sharpness
                else sharpness,
            ],
            -1,
        )

    @tf.function
    def _extract_sg_components(
        self, sg: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.name_scope("SG_Extract"):
            s_amplitude = (
                math_utils.safe_exp(sg[..., 0:3])
                if self.compress_amplitude
                else sg[..., 0:3]
            )
            s_axis = sg[..., 3:6]
            s_sharpness = (
                math_utils.safe_exp(sg[..., 6:7])
                if self.compress_amplitude
                else sg[..., 6:7]
            )

            return (
                tf.abs(s_amplitude),
                math_utils.normalize(s_axis),
                math_utils.saturate(s_sharpness, 0.5, 30),
            )

    @tf.function
    def _sg_integral(self, sg: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("SG_Integral"):
            s_amplitude, _, s_sharpness = self._extract_sg_components(sg)

            expTerm = 1.0 - math_utils.safe_exp(-2.0 * s_sharpness)
            return 2 * np.pi * tf.math.divide_no_nan(s_amplitude, s_sharpness) * expTerm

    @tf.function
    def _sg_inner_product(self, sg1: tf.Tensor, sg2: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("SG_InnerProd"):
            s1_amplitude, s1_axis, s1_sharpness = self._extract_sg_components(sg1)
            s2_amplitude, s2_axis, s2_sharpness = self._extract_sg_components(sg2)

            umLength = math_utils.magnitude(
                s1_sharpness * s1_axis + s2_sharpness * s2_axis
            )
            expo = (
                math_utils.safe_exp(umLength - s1_sharpness - s2_sharpness)
                * s1_amplitude
                * s2_amplitude
            )

            other = 1.0 - math_utils.safe_exp(-2.0 * umLength)

            return tf.math.divide_no_nan(2.0 * np.pi * expo * other, umLength)

    @tf.function
    def _sg_evaluate(self, sg: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("SG_Evaluate"):
            s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

            cosAngle = math_utils.dot(d, s_axis)
            return s_amplitude * math_utils.safe_exp(s_sharpness * (cosAngle - 1.0))

    @tf.function
    def visualize_fit(self, shape, sgs: tf.Tensor):
        if len(sgs.shape) == 4:
            if tf.shape(sgs)[0] == 1:
                sgs = sgs[0]
        print(sgs.shape)

        shape = (
            shape[0],
            shape[1],
            3,
        )

        output = tf.zeros([shape[0] * shape[1], shape[2]], dtype=tf.float32)

        uvs = math_utils.shape_to_uv(shape[0], shape[1])
        d = math_utils.uv_to_direction(uvs)
        d = tf.reshape(d, [-1, 1, 3])

        for i in range(sgs.shape[1]):
            sg = sgs[:, i : i + 1]
            output += self._sg_evaluate(sg, d)[:, 0]

        output = tf.reshape(output, shape)

        return output
