# Simple EasyHec: Accurate and Automatic Camera Calibration

This repo provides (mostly) pip installable code to easily calibrate a camera and get its extrinsics with respect to some object (like a robot). It is a cleaned up and simplified version of the original [EasyHeC project](https://github.com/ootts/EasyHeC). It works by taking a dataset of segmentation masks of the object and the object poses, and based on the camera intrinsics and an initial guess of the extrinsics, optimizes that guess into an accurate estimate of the true extrinsics. The optimization process leverages [Nvdiffrast](https://github.com/NVlabs/nvdiffrast) for differentiable rendering to run this optimization process.

<!-- TODO (stao): add some images -->

## Installation

Install from pip
```bash
pip install easyhec
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
```

Install from git (We highly recommend you create a conda/mamba environment to manage packages)

```bash
git clone https://github.com/StoneT2000/simple-easyhec
conda create -n easyhec "python==3.11"
pip install -e .
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
```

The code relies on Nvdiffrast which can sometimes be tricky to setup as it can have some dependencies that need to be installed outside of pip. If you have issues installing Nvdiffrast (which provides dockerized options) see their [documentation](https://nvlabs.github.io/nvdiffrast/) or use our google colab script.

For those who don't want to manually segment their robot images you can use [SAM2](https://github.com/facebookresearch/sam2). Instructions for using that with the differentiable rendering optimizer for camera calibration are detailed in the [real usage section](#real)

## Usage

We provide two examples, one in simulation and one for a real robot arm

### Simulation

To test in simulation we provide an example through maniskill. The example generates 10 synthetic images of the Franka/Panda robot with segmentation masks of the robot in random joint positions sampled around the initial joint position, and an initial extrinsic camera guess noisly sampled around the ground truth.

```bash
pip install easyhec[sim-maniskill]
# Franka arm, camera used here is a offhand 512x512 camera
python -m easyhec.examples.sim.maniskill -e StackCube-v1 --samples 10
```

You can also try out other robots and cameras that ManiSkill provides in other environments like the base camera for SO100


```bash
# SO100 arm, base_camera here is a 128x128 camera 
python -m easyhec.examples.sim.maniskill -e SO100GraspCube-v1 --samples 10 \
    --camera-name base_camera \
    --initial-extrinsic-guess-rot-error 15 --early-stopping-threshold 25 --initial-extrinsic-guess-pos-error 0.1
```

Wrist cameras are also possible but are harder to get working. The amount of initial extrinsic error must also be lowered since the robot already takes up a large part of the image. If the robot is far away more error is possible to solve from.

```bash
python -m easyhec.examples.sim.maniskill -e StackCube-v1 --samples 10 \
    --camera-name hand_camera \
    --initial-extrinsic-guess-rot-error 5 --early-stopping-threshold 25 --initial-extrinsic-guess-pos-error 0.01
```

### Real



## Tuning Tips

- It is recommended to get a diverse range of sample images that show the robot in different orientations. This is particularly more important for wrist cameras, which often only see the robot gripper.
- The amount of error that constitutes a good camera calibration depends on how far away the robot is and the resolution of that rebot in pixel space. You can lower or increase it depending on this factor. Alternatively just optimize for a few thousand iterations and excluding entering local minima the optimization process should covnerge to a good estimate
- The initial guess of the camera extrinsics does not have be good, but if the robot is up very close it may need to be more accurate. This can be the case for wrist cameras.
- If loss does not seem to be going down or is seems to have stalled, the optimization process has likely failed, usually due to too poor of a initial camera extrinsic guess.
- To ensure best results make sure you have fairly accurate visual meshes for the robot the camera is attached on / is pointing at. It is okay if the colors do not match, we just shapes to match
- While it is best to have accurate visual meshes, this optimization can still work even if you don't include some parts from the real world. It may be useful to edit out poor segmentations

## Citation

If you find this code useful for your research, please use the following BibTeX entries

```
@article{chen2023easyhec,
  title={EasyHec: Accurate and Automatic Hand-eye Calibration via Differentiable Rendering and Space Exploration},
  author={Chen, Linghao and Qin, Yuzhe and Zhou, Xiaowei and Su, Hao},
  journal={IEEE Robotics and Automation Letters (RA-L)}, 
  year={2023}
}
@article{Laine2020diffrast,
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
  journal = {ACM Transactions on Graphics},
  year    = {2020},
  volume  = {39},
  number  = {6}
}
```