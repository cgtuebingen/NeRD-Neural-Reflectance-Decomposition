import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import nn_utils.math_utils as math_utils
from dataflow.nerd.load_blender import load_blender_data
from dataflow.nerd.load_nerf_blender import load_blender_data as load_nerf_blender_data
from dataflow.nerd.load_real_world import load_llff_data
from nn_utils.nerf_layers import get_full_image_eval_grid
from utils.training_setup_utils import get_num_gpus


def add_args(parser):
    parser.add_argument(
        "--datadir",
        required=True,
        type=str,
        help="Path to dataset location.",
    )

    parser.add_argument(
        "--jitter_coords",
        action="store_true",
        help=(
            "Jitters the sampling coordinates. The RGB training color is "
            "then handled with a lookup. May reduce aliasing artifacts and improve "
            "opacity gradient based normal."
        ),
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="blender",
        help="Currently only blender or real_world is available",
        choices=["blender", "real_world", "nerf_blender"],
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=1,
        help="will load 1/N images from test/val sets, useful for large datasets",
    )
    parser.add_argument(
        "--valskip",
        type=int,
        default=16,
        help="will load 1/N images from test/val sets, useful for large datasets",
    )
    parser.add_argument(
        "--trainskip",
        type=int,
        default=1,
        help="will load 1/N images from train sets, useful for large datasets",
    )

    parser.add_argument(
        "--spherify", action="store_true", help="Real-world dataset captured in 360."
    )
    parser.add_argument(
        "--near",
        type=float,
        help=(
            "Multiplicator of near clip distance for real-world datasets, absolute "
            "near clip distance for blender."
        ),
    )
    parser.add_argument(
        "--far",
        type=float,
        help=(
            "Multiplocator of far clip distance for real-world datasets, absolute far "
            "clip distance for blender."
        ),
    )

    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )
    parser.add_argument(
        "--rwfactor",
        type=int,
        default=8,
        help="Factor for real-world sample downscaling",
    )
    parser.add_argument(
        "--rwholdout", type=int, default=16, help="Real world holdout stride"
    )

    return parser


def pick_correct_dataset(args):
    wbs = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)
    if args.dataset_type == "blender":
        (
            images,
            masks,
            poses,
            ev100s,
            render_poses,
            hwf,
            i_split,
            wbs,
        ) = load_blender_data(
            basedir=args.datadir,
            half_res=args.half_res,
            testskip=args.testskip,
            valskip=args.valskip,
            trainskip=args.trainskip,
        )
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        wb_ref_image = np.array([False for _ in range(images.shape[0])])
        wb_ref_image[i_train] = True

        near = args.near if args.near is not None else 2
        far = args.far if args.far is not None else 6
    elif args.dataset_type == "nerf_blender":
        (
            images,
            masks,
            poses,
            ev100s,
            render_poses,
            hwf,
            i_split,
        ) = load_nerf_blender_data(
            basedir=args.datadir,
            half_res=args.half_res,
            testskip=args.testskip,
            valskip=args.valskip,
            trainskip=args.trainskip,
        )
        print(
            "Loaded nerf blender", images.shape, render_poses.shape, hwf, args.datadir
        )
        i_train, i_val, i_test = i_split

        wb_ref_image = np.array([False for _ in range(images.shape[0])])
        # TODO make this configurable. Currently the reference image
        # is a completely random one
        ref_idx = np.random.choice(i_train, 1)
        wb_ref_image[ref_idx] = True

        near = args.near if args.near is not None else 2
        far = args.far if args.far is not None else 6
    elif args.dataset_type == "real_world":
        (images, masks, ev100s, poses, bds, render_poses, i_test,) = load_llff_data(
            basedir=args.datadir,
            factor=args.rwfactor,
            spherify=args.spherify,
        )

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded real world", images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.rwholdout > 0:
            print("Auto Real World holdout,", args.rwholdout)
            i_test = np.arange(images.shape[0])[:: args.rwholdout]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        wb_ref_image = np.array([False for _ in range(images.shape[0])])
        # TODO make this configurable. Currently the reference image
        # is a completely random one
        ref_idx = np.random.choice(i_train, 1)
        wb_ref_image[ref_idx] = True

        near = args.near if args.near is not None else 0.9
        far = args.far if args.far is not None else 1.0

        near = tf.reduce_min(bds) * near
        far = tf.reduce_max(bds) * far

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    mean_ev100 = np.mean(ev100s)

    return (
        i_train,
        i_val,
        i_test,
        images,
        masks,
        poses,
        ev100s,
        mean_ev100,
        wbs,
        wb_ref_image,
        render_poses,
        hwf,
        near,
        far,
    )


@tf.function
def build_rays(
    hwf,
    image,
    mask,
    pose,
    ev100,
    wb,
    wb_ref_image,
    num_rand_per_gpu,
    num_gpu,
    should_jitter_coords: bool,
):
    H, W, focal = hwf
    # Setup ray jittering
    jitter_coords = None
    if should_jitter_coords:
        jitter_coords = tf.random.uniform([H, W, 2], minval=-0.5, maxval=0.5)

    input_ev100 = tf.reshape(ev100, (1,))  # [1]
    pose = pose[:3, :4]

    rays_o, rays_d = get_full_image_eval_grid(H, W, focal, pose, jitter=jitter_coords)
    if num_rand_per_gpu > 0:
        coordsFull = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing="ij"), -1)
        coords = tf.reshape(coordsFull, [-1, 2])

        select_inds = tf.random.uniform_candidate_sampler(
            tf.range(coords.shape[0], dtype=tf.int64)[None, :],
            coords.shape[0],
            num_rand_per_gpu * num_gpu,
            True,
            coords.shape[0],
        )[0]

        select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
        rays_o = tf.gather_nd(rays_o, select_inds)
        rays_d = tf.gather_nd(rays_d, select_inds)

        if should_jitter_coords:
            # Jitter coords
            jitteredCoords = tf.cast(coordsFull, tf.float32) + jitter_coords
            jitteredCoords = tf.gather_nd(jitteredCoords, select_inds)
            # Interpolate. Add in fake batch dimensions and remove them
            target = tfa.image.interpolate_bilinear(image[None], jitteredCoords[None])[
                0
            ]
            input_mask = tfa.image.interpolate_bilinear(
                mask[None], jitteredCoords[None]
            )[0]
        else:
            target = tf.gather_nd(image, select_inds)
            input_mask = tf.gather_nd(mask, select_inds)
    else:
        input_mask = tf.reshape(mask, (-1, 1))
        target = tf.reshape(image, (-1, 3))
        rays_o = tf.reshape(rays_o, (-1, 3))
        rays_d = tf.reshape(rays_d, (-1, 3))

    return (
        rays_o,
        rays_d,
        tf.cast(math_utils.repeat(pose[None, ...], num_gpu, 0), tf.float32),
        input_mask,
        math_utils.repeat(input_ev100, num_gpu, 0),
        math_utils.repeat(tf.cast(wb, tf.float32)[None, ...], num_gpu, 0),
        math_utils.repeat(tf.convert_to_tensor([wb_ref_image]), num_gpu, 0),
        target,
    )


def dataflow(
    args,
    hwf,
    images,
    masks,
    poses,
    ev100s,
    wbs,
    wb_ref_image,
    select_idxs,
    is_train=False,
):
    train_samples = select_idxs.shape[0]
    repeats = max(args.steps_per_epoch // train_samples, 1)

    main_flow = tf.data.Dataset.from_tensor_slices(
        {
            "image": images[select_idxs],
            "mask": masks[select_idxs],
            "pose": poses[select_idxs],
            "ev100": ev100s[select_idxs],
            "wb": wbs[select_idxs],
            "wb_ref_image": wb_ref_image[select_idxs],
        }
    )
    idx_flow = tf.data.Dataset.from_tensor_slices(select_idxs)

    base_dataflow = tf.data.Dataset.zip((idx_flow, main_flow))
    if is_train:
        base_dataflow = base_dataflow.shuffle(
            100, reshuffle_each_iteration=True
        ).repeat(repeats)

    num_gpus = max(get_num_gpus(), 1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    dataflow = (
        base_dataflow.map(
            lambda idx, dp: (
                math_utils.repeat(
                    idx[
                        None,
                    ],
                    num_gpus if is_train else 1,
                    0,
                ),
                *build_rays(
                    hwf,
                    dp["image"],
                    dp["mask"],
                    dp["pose"],
                    dp["ev100"],
                    dp["wb"],
                    dp["wb_ref_image"],
                    args.batch_size if is_train else 0,
                    num_gpus if is_train else 1,
                    args.jitter_coords if is_train else None,
                ),  # idx, rays_o, rays_d, pose, input_mask, input_ev100, wb, target
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
        .with_options(options)
    )

    return dataflow


def create_dataflow(args):
    # Def return all sgs and the dataflow
    # The dataflow should return the ray directions, and corresponding values
    # As well as the corresponding index to the sgs
    (
        idx_train,
        idx_val,
        idx_test,
        images,
        masks,
        poses,
        ev100s,
        mean_ev100s,
        wbs,
        wb_ref_image,
        render_poses,
        hwf,
        near,
        far,
    ) = pick_correct_dataset(args)

    # Make sure all images are selected
    if wbs.shape[0] == 1:
        wbs = wbs.repeat(images.shape[0], 0)

    return (
        hwf,
        near,
        far,
        render_poses,
        images.shape[0],
        mean_ev100s,
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_train,
            is_train=True,
        ),
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_val,
            is_train=False,
        ),
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_test,
            is_train=False,
        ),
    )
