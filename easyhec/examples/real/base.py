from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:

    urdf_path: Optional[str] = None
    """path to the URDF file of the robot. Some robot specific scripts will provide a URDF file or generate meshes for you and this argument can be left as None."""
    batch_size: Optional[int] = None
    """batch size for the optimization. If none will use whole batch optimization"""
    train_steps: int = 5000
    """number of optimization steps. The default is 5000 which is usually more than enough to converge"""
    early_stopping_steps: int = 200
    """if after this many steps of optimization the loss has not improved, then optimization will stop. If this value is 0 then early stopping is disabled."""
    seed: int = 0

    ### sam2 segmentation related configs ###
    model_cfg: str = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    """the model config for sam2"""
    checkpoint: str = "sam2/checkpoints/sam2.1_hiera_large.pt"
    """the checkpoint for sam2"""

    output_dir: str = "results"
    """Where to save the calibration results and any visualizations"""
