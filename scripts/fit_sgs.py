import argparse
import os
import time
from typing import List, Tuple

import imageio
import numpy as np
import pyexr
import tensorflow as tf
from tqdm import tqdm

import nn_utils.math_utils as math_utils
from models.nerd_net.sgs_store import SgsStore
from nn_utils.sg_rendering import SgRenderer
from skimage.transform import resize


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("envmap", type=str, help="Path to environment map.")
    parser.add_argument(
        "--num_sgs",
        type=int,
        help="Fits x sgs_levels lobes to the environment map.",
        default=24,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of optimization-steps for each sgs level",
    )
    parser.add_argument(
        "--env_height",
        type=int,
        default=128,
        help="Height of environment map used for optimization",
    )

    parser.add_argument(
        "--out_path", type=str, required=True, help="Path where to store the results."
    )
    parser.add_argument("--gpu", type=int, required=True, help="Which GPU to use.")

    args = parser.parse_args()
    return args


def optimize(
    sgs: tf.Tensor,
    directions: tf.Tensor,
    target: tf.Tensor,
    optimizer,
    sg_render: SgRenderer,
) -> Tuple[tf.Tensor]:
    """A single optimization step.

    Renders the color for each direction and compares it to the target.

    Returns:
        Loss and rendered envmap.
    """
    with tf.GradientTape() as tape:
        tape.watch(sgs)
        dir_flat = tf.reshape(directions, (-1, 1, 3))
        evaled_flat = sg_render._sg_evaluate(sgs, dir_flat)
        evaled_flat = tf.reduce_sum(evaled_flat, 1)
        evaled = tf.reshape(evaled_flat, target.shape)
        loss = tf.reduce_mean(
            tf.math.abs(tf.math.log(1 + target) - tf.math.log(1 + evaled))
        )
    grad_vars = (sgs,)
    gradients = tape.gradient(loss, grad_vars)
    optimizer.apply_gradients(zip(gradients, grad_vars))

    # optimization constraints
    ampl = tf.math.maximum(sgs[..., :3], 0.01)
    axis = math_utils.normalize(sgs[..., 3:6])
    sharpness = tf.math.maximum(sgs[..., 6:], 0.5)
    sgs.assign(tf.concat([ampl, axis, sharpness], -1))
    return loss, evaled


def fit_sgs(
    env_map: np.ndarray,
    directions: tf.Tensor,
    num_sgs: int,
    steps: int,
    sg_render: SgRenderer,
    path: str,
) -> Tuple[List[np.ndarray]]:
    # env map to tensor
    env_map = tf.convert_to_tensor(env_map, dtype=tf.float32)

    start_time = time.time()
    sgs_axis_sharpness = SgsStore.setup_uniform_axis_sharpness(num_sgs)
    sgs_amplitude = np.ones_like(sgs_axis_sharpness[..., :3])
    sgs = np.concatenate([sgs_amplitude, sgs_axis_sharpness], -1)

    sgs_tf = tf.Variable(
        tf.convert_to_tensor(sgs[np.newaxis, ...], tf.float32),
        trainable=True,
        name="sgs_{}".format(num_sgs),
    )
    optimizer = tf.keras.optimizers.Adam(1e-2)
    bar = tqdm(range(steps), desc="lobes: {}".format(num_sgs))
    for _ in bar:
        loss, sg_envmap = optimize(sgs_tf, directions, env_map, optimizer, sg_render)
        bar.set_postfix({"loss": loss.numpy()})

    total_time = time.time() - start_time
    print("Fitting took", total_time)

    # write to disk
    sgs_path = os.path.join(path, "{0:03d}_sg.npy".format(num_sgs))
    env_path = os.path.join(path, "{0:03d}_sg_env.exr".format(num_sgs))
    np.save(sgs_path, sgs_tf.numpy())
    pyexr.write(env_path, sg_envmap.numpy())


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    os.makedirs(args.out_path, exist_ok=True)

    if args.envmap[-3:] == "exr":
        env_map = pyexr.open(args.envmap).get()[..., :3]
    else:
        env_map = imageio.imread(args.envmap)

    env_map = resize(env_map, (args.env_height, args.env_height * 2))

    directions = math_utils.uv_to_direction(
        math_utils.shape_to_uv(env_map.shape[0], env_map.shape[1])
    )
    sg_render = SgRenderer()

    fit_sgs(env_map, directions, args.num_sgs, args.steps, sg_render, args.out_path)


if __name__ == "__main__":
    main()
