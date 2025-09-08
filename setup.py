from setuptools import find_packages, setup

setup(
    name="easyhec",
    version="0.1.1",
    packages=find_packages(),
    author="Stone Tao",
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
