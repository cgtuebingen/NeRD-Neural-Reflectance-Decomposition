import argparse
import os

datasets_synthetic = [
    ("Globe", "https://github.com/vork/globe_syn_photogrammetry.git"),
    ("Car", "https://github.com/vork/car_syn_photogrammetry.git"),
    ("Chair", "https://github.com/vork/chair_syn_photogrammetry.git"),
]

datasets_real_world = [
    ("Gnome", "https://github.com/vork/gnomes-photogrammetry.git"),
    ("GoldCape", "https://github.com/vork/moldGoldCape.git"),
    ("EthiopianHead", "https://github.com/vork/ethiopianHead.git"),
    ("StatueOfLiberty", "https://github.com/vork/StatueOfLiberty-Photogrammetry.git"),
    ("MotherChild", "https://github.com/vork/mother_child-photogrammetry.git"),
]


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="The root location of the datasets.")

    return parser.parse_args()


def main(args):
    dataset_root = args.dataset_root

    variants = [("synthetic", datasets_synthetic), ("real_world", datasets_real_world)]

    for name, data in variants:
        dataset_variant_root = os.path.join(dataset_root, name)
        os.makedirs(dataset_variant_root, exist_ok=True)

        for dname, durl in data:
            dataset_path = os.path.join(dataset_variant_root, dname)

            if not os.path.exists(dataset_path):
                os.system("git clone %s %s" % (durl, dataset_path))


if __name__ == "__main__":
    main(args())
