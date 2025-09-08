from dataclasses import dataclass
from pathlib import Path

import cv2
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
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv


@dataclass
class RealPaperArgs(Args):
    """Calibrate a (realsense) camera with just a piece of standard sized paper. Note that this script might not work with your particular realsense camera, modify as needed.Other cameras can work if you modify the code to get the camera intrinsics and a single color image from the camera."""
    output_dir: str = "results/paper"
    paper_type: str = "letter"
    """The type of paper to use to calibrate against. Options are 'letter' or 'a4'"""
    realsense_camera_serial_id: str = "none"
    """The serial id of the realsense camera to use for calibration"""
    # TODO (stao): A1, A2, A3, follow a nice structure, we can just generate the meshes for those.


paper_sizes = {
    "letter": {
        "width": 0.2159,  # 8.5 inches in mm
        "height": 0.2794,  # 11 inches in mm
    },
    "a4": {
        "width": 0.210,  # 8.27 inches in mm
        "height": 0.297,  # 11.69 inches in mm
    },
}


def main(args: RealPaperArgs):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found.")

    # Configure streams
    camera_width = 1280
    camera_height = 720
    if args.realsense_camera_serial_id == "none":
        print("No realsense camera serial id provided, using the first device found")
        realsense_camera_serial_id = devices[0].get_info(rs.camera_info.serial_number)
    else:
        realsense_camera_serial_id = args.realsense_camera_serial_id
    print(f"RealSense device id: {realsense_camera_serial_id}")
    config.enable_device(realsense_camera_serial_id)
    config.enable_stream(
        rs.stream.color, camera_width, camera_height, rs.format.bgr8, 30
    )
    # Get the color stream profile and its intrinsics
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)

    ### Fetch Intrinsics ###
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    intrinsic = np.array(
        [
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    ### Fetch one color image ###
    skip_frames = 60
    print("Starting camera and warming it up...")
    for _ in range(skip_frames):
        frames = pipeline.wait_for_frames()
        cframe = frames.get_color_frame()
        if not cframe:
            print("No frame")
            continue
        image = np.asanyarray(cframe.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Camera Intrinsics:\n {repr(intrinsic)}")
    images = [image]

    ### Make an initial guess for the extrinsic ###
    # use what we put in sim as the initial guess
    initial_extrinsic_guess = np.eye(4)

    # the guess says we are at position xyz=[-0.4, 0.0, 0.4] and angle the camerea downwards by np.pi / 4 radians  or 45 degrees
    # note that this convention is more natural for robotics (follows the typical convention for ROS and various simulators), where +Z is moving up towards the sky, +Y is to the left, +X is forward
    initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, 0)
    initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.4])
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

    print("Initial extrinsic guess", initial_extrinsic_guess)

    # Create a box mesh representing the letter paper (in meters)
    paper_width = paper_sizes[args.paper_type]["width"]
    paper_height = paper_sizes[args.paper_type]["height"]
    paper_box = trimesh.creation.box(extents=(paper_width, paper_height, 1e-3))
    meshes = [paper_box]
    # We assume the world frame is centered at the paper and oriented to be perpendicular to the paper
    link_poses_dataset = np.stack(np.eye(4)).reshape(1, 1, 4, 4)

    camera_mount_poses = None

    interactive_segmentation = InteractiveSegmentation(
        segmentation_model="sam2",
        segmentation_model_cfg=dict(
            checkpoint=args.checkpoint, model_cfg=args.model_cfg
        ),
    )
    masks = interactive_segmentation.get_segmentation(images)

    ### run the optimization given the data ###
    predicted_camera_extrinsic_opencv = (
        optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            masks=torch.from_numpy(masks).float().to(device),
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
    print(f"OpenCV:\n{repr(predicted_camera_extrinsic_opencv)}")
    print(f"ROS/SAPIEN/ManiSkill/Mujoco/Isaac:\n{repr(predicted_camera_extrinsic_ros)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(
        Path(args.output_dir) / "camera_extrinsic_opencv.npy",
        predicted_camera_extrinsic_opencv,
    )
    np.save(
        Path(args.output_dir) / "camera_extrinsic_ros.npy",
        predicted_camera_extrinsic_ros,
    )
    np.save(Path(args.output_dir) / "camera_intrinsic.npy", intrinsic)

    visualization.visualize_extrinsic_results(
        images=images,
        link_poses_dataset=link_poses_dataset,
        meshes=meshes,
        intrinsic=intrinsic,
        extrinsics=np.stack(
            [initial_extrinsic_guess, predicted_camera_extrinsic_opencv]
        ),
        masks=masks,
        labels=["Initial Extrinsic Guess", "Predicted Extrinsic"],
        output_dir=args.output_dir,
    )
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(RealPaperArgs))
