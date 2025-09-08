from setuptools import find_packages, setup

setup(
    name="easyhec",
    version="0.1.6",
    packages=find_packages(),
    package_data={"easyhec": ["examples/real/robot_definitions/**"]},
    author="Stone Tao",
    homepage="https://github.com/stonet2000/easyhec",
    description="EasyHec is a library for fast and automatic camera extrinsic calibration",
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
        "urchin",
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
