from dataclasses import dataclass
from typing import Annotated, Optional

import tyro


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "StackCube-v1"
    """the simulated environment ID if you want to test easy hec on a sim environment. With the sim environment it will use ground truth segmentation masks for optimization"""
    samples: int = 10
    """number of synthetic samples to generate to try and calibrate the camera with"""
    test_samples: int = 10
    """number of synthetic samples to generate to test the predicted extrinsics with"""
    initial_extrinsic_guess_pos_error: float = 0.1
    """error in the initial extrinsic translation guess in meters."""
    initial_extrinsic_guess_rot_error: float = 15.0
    """error in the initial extrinsic rotation guess in degrees."""
    batch_size: Optional[int] = None
    """batch size for the optimization. If none will use whole batch optimization"""
    train_steps: int = 5000
    """number of optimization steps. The default is 5000 which is usually more than enough to converge"""
    early_stopping_steps: int = 200
    """if after this many steps of optimization the loss has not improved, then optimization will stop. If this value is 0 then early stopping is disabled."""
    seed: int = 0

    ### sam2 segmentation related configs ###
    use_ground_truth_segmentation: bool = True
    """if true will optimize against the ground truth segmentation masks. If false will open a GUI with an interactive segmentation tool that a user can click points on that are then sent to the SAM2 segmentation model to predict segmentation masks"""
    model_cfg: str = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    """the model config for sam2"""
    checkpoint: str = "sam2/checkpoints/sam2.1_hiera_large.pt"
    """the checkpoint for sam2"""

    output_dir: str = "results/sim"
    """Where to save the calibration results and any visualizations"""
