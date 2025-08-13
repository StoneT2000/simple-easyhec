import os.path as osp
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrealsense2 as rs
import torch
import trimesh
import tyro
from transforms3d.euler import euler2mat

from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros


@dataclass
class RealPaperArgs(Args):
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

    paper_type: str = "letter"
    """The type of paper to use to calibrate against. Options are 'letter' or 'a4' (or any other sized A paper) """


paper_sizes = {
    "letter": {
        "width": 0.2159,  # 8.5 inches in mm
        "height": 0.2794,  # 11 inches in mm
    }
}


def main(args: RealPaperArgs):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams
    camera_width = 640
    camera_height = 360
    config.enable_stream(
        rs.stream.color, camera_width, camera_height, rs.format.bgr8, 30
    )
    # Get the color stream profile and its intrinsics
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    intrinsic = np.array(
        [
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    print("Camera Intrinsics:\n", intrinsic)

    while True:
        frames = pipeline.wait_for_frames()
        cframe = frames.get_color_frame()
        if not cframe:
            print("No frame")
            continue
        image = np.asanyarray(cframe.get_data())
        break
    # TODO (stao): allow more image captures
    images = [image]

    # use what we put in sim as the initial guess
    initial_extrinsic_guess = np.eye(4)
    initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, np.pi / 2)
    initial_extrinsic_guess[:3, 3] = np.array([0.0, 0.0, 0.4])
    print("Initial extrinsic guess", initial_extrinsic_guess)
    meshes = [osp.join(osp.dirname(__file__), "letter_paper.stl")]

    # Create a box mesh representing the letter paper (in meters)
    # if args.paper_type is of the format "a0", "a1", ..., "a5" do something
    import re

    if re.match(r"^a[0-5]$", args.paper_type.lower()):
        a_paper_number = int(args.paper_type[1])
        # Standard A-series paper sizes: A0 is 841mm x 1189mm, each size down halves the area (width and height divided by sqrt(2))
        base_width = 0.841  # meters (A0 width)
        base_height = 1.189  # meters (A0 height)
        factor = 2 ** (a_paper_number / 2)
        paper_width = base_width / factor
        paper_height = base_height / factor
    else:
        paper_width = paper_sizes[args.paper_type]["width"]
        paper_height = paper_sizes[args.paper_type]["height"]
    print(
        f"Calibrating against {args.paper_type} paper which has dimensions {paper_width}m x {paper_height}m"
    )
    paper_box = trimesh.creation.box(extents=(paper_width, paper_height, 1e-3))
    meshes = [paper_box]
    link_poses_dataset = np.stack(np.eye(4)).reshape(1, 1, 4, 4)
    camera_mount_poses = None

    interactive_segmentation = InteractiveSegmentation(
        segmentation_model="sam2",
        segmentation_model_cfg=dict(
            checkpoint=args.checkpoint, model_cfg=args.model_cfg
        ),
    )
    robot_masks = interactive_segmentation.get_segmentation(images)

    ### run the optimization given the data ###
    predicted_camera_extrinsic_opencv = (
        optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            robot_masks=torch.from_numpy(robot_masks).float().to(device),
            link_poses_dataset=torch.from_numpy(link_poses_dataset).float().to(device),
            initial_extrinsic_guess=torch.tensor(initial_extrinsic_guess)
            .float()
            .to(device),
            meshes=meshes,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_mount_poses=(
                torch.from_numpy(camera_mount_poses).float().to(device)
                if camera_mount_poses is not None
                else None
            ),
            gt_camera_pose=None,
            iterations=args.train_steps,
            early_stopping_steps=args.early_stopping_steps,
        )
        .cpu()
        .numpy()
    )
    predicted_camera_extrinsic_ros = opencv2ros(predicted_camera_extrinsic_opencv)

    ### Print predicted results ###

    print(f"Predicted camera extrinsic")
    print(f"OpenCV:\n{predicted_camera_extrinsic_opencv}")
    print(f"ROS/SAPIEN/ManiSkill:\n{predicted_camera_extrinsic_ros}")

    visualization.visualize_extrinsic_results(
        images=images,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        intrinsic=intrinsic,
        extrinsics=np.stack(
            [initial_extrinsic_guess, predicted_camera_extrinsic_opencv]
        ),
        labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
        output_dir="results/paper",
    )
    print("Visualizations saved to results/paper")


if __name__ == "__main__":
    main(tyro.cli(RealPaperArgs))
