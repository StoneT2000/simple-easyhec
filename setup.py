from setuptools import find_packages, setup

setup(
    name="easyhec",
    version="0.1.4",
    packages=find_packages(),
    author="Stone Tao",
    homepage="https://github.com/stonet2000/easyhec",
    summary="EasyHec is a library for fast and automatic camera extrinsic calibration",
    license="BSD-3-Clause",
    url="https://github.com/stonet2000/easyhec",
    python_requires=">=3.9",
    install_requires=[
        "tyro",
        "torch",
        "tqdm",
        "opencv-python",
        "trimesh",
        "transforms3d",
        "matplotlib",
        # ninja is used by nvdiffrast
        "ninja>=1.11",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pre-commit",
        ],
        "sim-maniskill": [
            "mani_skill-nightly",
        ],
    },
)
