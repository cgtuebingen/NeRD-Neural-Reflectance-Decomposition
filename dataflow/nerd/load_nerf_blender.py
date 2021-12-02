import json
import os

import imageio
import numpy as np
import tensorflow as tf


def trans_t(t):
    return tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def rot_phi(phi):
    return tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, tf.cos(phi), -tf.sin(phi), 0],
            [0, tf.sin(phi), tf.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def rot_theta(th):
    return tf.convert_to_tensor(
        [
            [tf.cos(th), 0, -tf.sin(th), 0],
            [0, 1, 0, 0],
            [tf.sin(th), 0, tf.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, trainskip=1, testskip=1, valskip=1):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_masks = []
    all_poses = []
    all_ev100 = []
    counts = [0]
    meta = None
    for s in splits:
        meta = metas[s]
        imgs = []
        masks = []
        poses = []
        if s == "train":
            skip = max(trainskip, 1)
        elif s == "val":
            skip = max(valskip, 1)
        else:
            skip = max(testskip, 1)

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            img_file = (imageio.imread(fname) / 255).astype(np.float32)
            imgs.append(img_file[..., 0:3])
            masks.append(img_file[..., 3:])

            # Read the poses
            poses.append(np.array(frame["transform_matrix"]))

            all_ev100.append(8)

        imgs = np.array(imgs).astype(np.float32)
        # Continue with the masks.
        # They only require values to be between 0 and 1
        # Clip to be sure
        masks = np.clip(np.array(masks).astype(np.float32), 0, 1)

        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0).astype(np.float32)
    masks = np.concatenate(all_masks, 0).astype(np.float32)
    poses = np.concatenate(all_poses, 0)
    ev100s = np.stack(all_ev100, 0).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = tf.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = H // 2
        W = W // 2
        imgs = tf.image.resize(imgs, [H, W], method="area").numpy()
        masks = tf.image.resize(masks, [H, W], method="area").numpy()
        focal = focal / 2.0

    return imgs, masks, poses, ev100s, render_poses, [H, W, focal], i_split
