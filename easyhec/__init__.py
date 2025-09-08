from pathlib import Path
try:
    import sam2
except ModuleNotFoundError:
    print("sam2 is not installed. Features using SAM2 for segmentation will not work. Follow the instructions at https://github.com/facebookresearch/sam2 to install it.")

try:
    import nvdiffrast
except ModuleNotFoundError:
    print("nvdiffrast is not installed. Please install it with pip install 'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8'")
    exit(1)

ROBOT_DEFINITIONS_DIR = Path(__file__).parent / "examples" / "real" / "robot_definitions"