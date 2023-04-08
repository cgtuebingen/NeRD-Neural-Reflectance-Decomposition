# NeRD: Neural Reflectance Decomposition from Image Collections

### [Project Page](https://markboss.me/publication/2021-nerd/) | [Video](https://youtu.be/JL-qMTXw9VU) | [Paper](https://arxiv.org/abs/2012.03918) | [Dataset](download_datasets.py)

Implementation for NeRD. A novel method which decomposes multiple images into shape, BRDF and illumination.
<br><br>
[NeRD: Neural Reflectance Decomposition from Image Collections](https://markboss.me/publication/2021-nerd/)<br>
[Mark Boss](https://markboss.me)<sup>1</sup>, [Raphael Braun](https://uni-tuebingen.de/en/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/computergrafik/lehrstuhl/mitarbeiter/raphael-braun/)<sup>1</sup>, [Varun Jampani](https://varunjampani.github.io)<sup>2</sup>, [Jonathan T. Barron](https://jonbarron.info)<sup>2</sup>, [Ce Liu](http://people.csail.mit.edu/celiu/)<sup>2</sup>, [Hendrik P. A. Lensch](https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/computer-science/lehrstuehle/computergrafik/computer-graphics/staff/prof-dr-ing-hendrik-lensch/)<sup>1</sup><br>
<sup>1</sup>University of Tübingen, <sup>2</sup>Google Research 
<br><br>
![](images/teaser.jpg)

**Also check our follow-up work: [Neural-PIL](https://github.com/cgtuebingen/Neural-PIL)**

## Setup

A conda environment is used for dependency management

```
conda env create -f environment.yml
conda activate nerd
```
## Running

Replace the specific `[]` placeholders:

```
python train_nerd.py --datadir [DIR_TO_DATASET_FOLDER] --basedir [TRAIN_DIR] --expname [EXPERIMENT_NAME] --gpu [COMMA_SEPARATED_GPU_LIST]
```

### Specific Arguments per Dataset

Most setup is handled by configurations files located in [configs/nerd/](configs/nerd/).
#### Our Synthethic Scenes

```
--config configs/nerd/blender.txt
```

#### NeRF Synthethic Scenes

```
--config configs/nerd/nerf_blender.txt
```

#### Real-World

```
--config configs/nerd/real_world.txt 
```

Often objects are captured in a spherical manner and the flag `--spherify` should be applied for those scenes.

## Datasets

All datasets are uploaded in individual git repositories. We have created a [download script](download_datasets.py) which automatically fetches all datasets and downloads them to a specified folder. Usage: 

```shell
python download_datasets.py /path/to/dataset/root
```
## Run Your Own Data

Mainly camera poses and segmentation masks are required. For the poses the scripts from [NeRF](https://github.com/bmild/nerf#generating-poses-for-your-own-scenes) are used to prepare the scenes. The dataset then needs to be put in the following file structure:

```
images/
    [IMG_NAME_1].jpg
    [IMG_NAME_2].jpg
    ...
masks/
    [IMG_NAME_1].jpg
    [IMG_NAME_2].jpg
    ...
poses_bounds.npy
```

The `poses_bounds.npy` is generated from the [LLFF script](https://github.com/bmild/nerf#dont-have-poses).

## Evaluation

The [train_nerd.py](train_nerd.py) can be called with a `--render_only` flag and the `--config` flag pointing to the `args.txt` of the experiments folder. The SGs can be converted to regular environment maps with the `visualize_fit` function in [nn_utils/sg_rendering.py](nn_utils/sg_rendering.py) (Example in [train_nerd.py](train_nerd.py#L631)).
## Citation

```
@inproceedings{boss2021nerd,
  title         = {NeRD: Neural Reflectance Decomposition from Image Collections},
  author        = {Boss, Mark and Braun, Raphael and Jampani, Varun and Barron, Jonathan T. and Liu, Ce and Lensch, Hendrik P.A.},
  booktitle     = {IEEE International Conference on Computer Vision (ICCV)},
  year          = {2021},
}
```
