import os
import shutil
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from typing import List, Optional, Union

import configargparse
import numpy as np

from utils.import_finder import find_files_to_copy


def setup_parser():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--basedir",
        required=True,
        type=str,
        help="Basedir to store the runs. Runs are organized in folder EXPNAME.",
    )
    parser.add_argument(
        "--expname", required=True, type=str, help="Train folder to create."
    )

    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size per GPU"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=100000)

    parser.add_argument(
        "--gpu",
        type=str,
        help="Comma separated list of GPUs to use (PCI Bus Order), do not pass if it should use all available GPUs",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Activates NaN and Inf debugging."
    )

    return parser


def parse_args_file_without_nones(parser, overwrite=None):
    fake_parser = configargparse.ArgParser()
    fake_parser.add_argument("--config", help="config file path")

    args, remaining_args = fake_parser.parse_known_args(overwrite)

    config_string = []
    with open(args.config, "r") as pyFile:
        for line in pyFile:
            # ignore comments
            line = line.strip()
            if "config =" in line:
                continue

            if "= None" not in line:
                if "= True" in line:
                    line = line.replace("= True", "= true")
                elif "= False" in line:
                    line = line.replace("= False", "= false")
                config_string.append(line)

    config_string = "\n".join(config_string)

    return parser.parse_known_args(
        args=remaining_args, config_file_contents=config_string
    )[
        0
    ]  # Only return the parsed args and omit not found ones


def get_num_gpus():
    import tensorflow as tf

    return len(tf.config.list_physical_devices("GPU"))
    # return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def adjust_learning_rate_to_replica(args) -> float:
    return args.learning_rate * get_num_gpus()


@dataclass
class StateRestorationItem:
    name: str
    obj: object
    is_variable: bool = False


class StateRestoration:
    def __init__(self, args, to_watch: List[StateRestorationItem]):
        self.args = args
        self.to_watch = to_watch

    def save(self, i):
        for sri in self.to_watch:
            if sri.is_variable:
                StateRestoration.save_variable(self.args, sri.obj, sri.name, i)
            else:
                StateRestoration.save_weights(self.args, sri.obj, sri.name, i)

    def restore(self, step: Optional[int] = None) -> int:
        if step is None:
            name_to_step = lambda name: int(
                name.replace(self.to_watch[0].name + "_", "").replace(".npy", ""),
            )
            ckpts = sorted(
                [
                    f
                    for f in os.listdir(
                        os.path.join(self.args.basedir, self.args.expname)
                    )
                    if self.to_watch[0].name in f
                ],
                key=name_to_step,
            )
            print("Found ckpts", ckpts)
            if len(ckpts) > 0:
                step = name_to_step(ckpts[-1])
            else:
                return 0

        for sri in self.to_watch:
            print("Restoring", sri.obj, sri.name, step)
            if sri.is_variable:
                StateRestoration.restore_variable(self.args, sri.obj, sri.name, step)
            else:
                StateRestoration.restore_weights(self.args, sri.obj, sri.name, step)

        return step

    @classmethod
    def build_save_path(cls, args, prefix, i):
        return os.path.join(
            args.basedir, args.expname, "{}_{:06d}.npy".format(prefix, i)
        )

    @classmethod
    def save_weights(cls, args, net, prefix, i):
        path = StateRestoration.build_save_path(args, prefix, i)
        np.save(path, net.get_weights())
        print("saved weights", prefix, "at", path)

    @classmethod
    def save_variable(cls, args, variable, prefix, i):
        path = StateRestoration.build_save_path(args, prefix, i)
        np.save(path, variable)
        print("saved variable", prefix, "at", path)

    @classmethod
    def restore_weights(cls, args, net, prefix, i):
        path = StateRestoration.build_save_path(args, prefix, i)
        net.set_weights(np.load(path, allow_pickle=True))
        print("reloaded weights", prefix, "from", path)

    @classmethod
    def restore_variable(cls, args, variable, prefix, i):
        path = StateRestoration.build_save_path(args, prefix, i)
        data = np.load(path, allow_pickle=True)
        variable.assign(data)
        print("reloaded variable", prefix, "from", path)


class SetupDirectory:
    def __init__(
        self,
        args,
        copy_files: bool = False,
        main_script: Optional[str] = None,
        copy_data: Optional[Union[List[str]]] = None,
    ) -> None:
        import tensorflow as tf

        if args.gpu is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        else:
            print("No gpu override requested!")
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        print("Utilizing %d GPUs for training." % get_num_gpus())

        write_dir = os.path.join(args.basedir, args.expname)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        f = os.path.join(write_dir, "args.txt")

        logdir = os.path.join(args.basedir, "summaries", args.expname)

        if copy_files:
            with open(f, "w") as file:
                for arg in sorted(vars(args)):
                    attr = getattr(args, arg)
                    file.write("{} = {}\n".format(arg, attr))
            if args.config is not None:
                f = os.path.join(write_dir, "config.txt")
                with open(f, "w") as file:
                    file.write(open(args.config, "r").read())

            assert main_script is not None
            all_files = find_files_to_copy(main_script)
            copy_paths = [os.path.join(write_dir, "src", f) for f in all_files]

            for src, dst in zip(all_files, copy_paths):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

            if copy_data is not None:
                if isinstance(copy_data, str):
                    copy_data = [copy_data]
                for cdata in copy_data:
                    if os.path.exists(cdata):
                        data_dst = os.path.join(write_dir, "src", cdata)
                        copy_tree(cdata, data_dst)

        self.writer = tf.summary.create_file_writer(logdir)

        if args.debug:
            tf.debugging.enable_check_numerics()

    def __enter__(self):
        self._w = self.writer.as_default()
        return self._w.__enter__()

    def __exit__(self, type, value, traceback):
        self._w.__exit__(type, value, traceback)
