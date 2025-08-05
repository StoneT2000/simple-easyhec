from setuptools import find_packages, setup

setup(
    name="easyhec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tyro",
        "torch",
        "tqdm",
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
