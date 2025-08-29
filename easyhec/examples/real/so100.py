from dataclasses import dataclass
from pathlib import Path
import logging
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import trimesh
import tyro
from transforms3d.euler import euler2mat
from collections import defaultdict
from easyhec.examples.real.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv

from pathlib import Path
import gymnasium as gym
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.robots.utils import make_robot_from_config
import numpy as np
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense import RealSenseCamera
from urchin import URDF
from lerobot.motors.motors_bus import MotorNormMode

def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": RealSenseCameraConfig(serial_number_or_name="105422061670", fps=30, width=1280, height=720)
            },
            id="stone_home",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot

def merge_meshes(meshes: list[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None

@dataclass
class SO100Args(Args):
    """Calibrate a (realsense) camera with LeRobot SO100. Note that this script might not work with your particular realsense camera, modify as needed. Other cameras can work if you modify the code to get the camera intrinsics and a single color image from the camera."""
    output_dir: str = "results/so100"
    # TODO (stao): A1, A2, A3, follow a nice structure, we can just generate the meshes for those.


# This is extrememly important to tune!
CALIBRATION_OFFSET = {
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_flex": 0,
    "wrist_roll": 0,
    "gripper": 0,
}

# For the author's SO100 they used this calibration offset. Yours might be different
CALIBRATION_OFFSET = {
    "shoulder_pan": -3,
    "shoulder_lift": -3,
    "elbow_flex": 0,
    "wrist_flex": 5,
    "wrist_roll": 0,
    "gripper": 0,
}


def main(args: SO100Args):
    user_tuned_calibration_offset = False
    for k in CALIBRATION_OFFSET.keys():
        if CALIBRATION_OFFSET[k] != 0:
            user_tuned_calibration_offset = True
            break
    if not user_tuned_calibration_offset:
        logging.warning("The calibration offset for sim2real/real2sim is not tuned!! Unless you are absolutely sure you will most likely get poor results.")


    robot: SO100Follower = create_real_robot("so100")
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    robot.connect()

    cameras_ft = robot._cameras_ft
    print(f"Found {len(cameras_ft)} cameras to calibrate")
    joint_position_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

    # get camera intrinsics for realsense cameras.
    intrinsics = dict()
    for cam_name, cam in robot.cameras.items():
        if isinstance(cam, RealSenseCamera):
            streams = cam.rs_profile.get_streams()
            assert len(streams) == 1, "Only one stream per camera is supported at the moment and it must be the color steam. Make sure to not enable any other streams."
            color_stream = streams[0]
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            intrinsic = np.array(
                [
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1],
                ]
            )
            intrinsics[cam_name] = intrinsic

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

    def get_qpos(robot: SO100Follower, flat: bool = True):
        obs = robot.bus.sync_read("Present_Position")
        for k in CALIBRATION_OFFSET.keys():
            obs[k] = obs[k] - CALIBRATION_OFFSET[k]
        for k in obs.keys():
            obs[k] = np.deg2rad(obs[k])
        if not flat:
            return obs
        joint_positions = []
        for k, v in obs.items():
            joint_positions.append(v)
        joint_positions = np.array(joint_positions)
        return joint_positions
    
    def set_target_qpos(robot: SO100Follower, qpos: np.ndarray):
        action = {}
        for name, qpos_val in zip(joint_position_names, qpos):
            action[name] = np.rad2deg(qpos_val) + CALIBRATION_OFFSET[name.removesuffix(".pos")]
        robot.send_action(action)
    
    robot_def_path = Path(__file__).parent / "robot_definitions" / "so100"
    robot_urdf = URDF.load(str(robot_def_path / "so100.urdf"))

    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        meshes.append(merge_meshes(link_meshes))

    # reference qpos positions to calibrate with    
    qpos_samples = [
        np.array([
            0, 0, 0, np.pi / 2, np.pi / 2, 0.2
        ]),
        np.array([
            np.pi / 3, -np.pi / 6, 0, np.pi / 2, np.pi / 2, 0
        ])
    ]
    control_freq = 15
    max_radians_per_step = 0.05

    # generate our link pose dataset and image pairs.
    link_poses_dataset = np.zeros((len(qpos_samples), len(meshes), 4, 4))
    image_dataset = defaultdict(list)

    for i in range(len(qpos_samples)):
        goal_qpos = qpos_samples[i]
        target_qpos = get_qpos(robot)
        for _ in range(int(20*control_freq)):
            start_loop_t = time.perf_counter()
            delta_qpos = (goal_qpos - target_qpos)
            delta_step = delta_qpos.clip(
                min=-max_radians_per_step, max=max_radians_per_step
            )
            if np.linalg.norm(delta_qpos) < 1e-4:
                break
            target_qpos += delta_step
            dt_s = time.perf_counter() - start_loop_t
            set_target_qpos(robot, target_qpos)
            time.sleep(1 / control_freq - dt_s)
        time.sleep(1) # give some time for the robot to settle, cheap arms don't hold up as well
        qpos_dict = get_qpos(robot, flat=False)
        for cam_name, cam in robot.cameras.items():
            image_dataset[cam_name].append(cam.async_read())
            
        # get link poses
        cfg = dict()
        for k in robot_urdf.joint_map.keys():
            cfg[k] = qpos_dict[k]
        link_poses = robot_urdf.link_fk(cfg=cfg, use_names=True)
        for link_idx, v in enumerate(link_poses.values()):
            link_poses_dataset[i, link_idx] = v
    for k in image_dataset.keys():
        image_dataset[k] = np.stack(image_dataset[k])

    ### Make an initial guess for the extrinsic for each camera ###
    initial_extrinsic_guesses = dict()
    for k in cameras_ft.keys():
        initial_extrinsic_guess = np.eye(4)

        # the guess says we are at position xyz=[-0.4, 0.0, 0.4] and angle the camerea downwards by np.pi / 4 radians  or 45 degrees
        # note that this convention is more natural for robotics (follows the typical convention for ROS and various simulators), where +Z is moving up towards the sky, +Y is to the left, +X is forward
        initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
        initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.5])
        initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

        initial_extrinsic_guesses[k] = initial_extrinsic_guess

    print("Initial extrinsic guesses")
    for k in initial_extrinsic_guesses.keys():
        print(f"Camera {k}:\n{repr(initial_extrinsic_guesses[k])}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in initial_extrinsic_guesses.keys():
        print(f"Calibrating camera {k}")
        initial_extrinsic_guess = initial_extrinsic_guesses[k]
        intrinsic = intrinsics[k]
        images = image_dataset[k]
        camera_mount_poses = None # TODO (stao): support this
        camera_width = images.shape[2]
        camera_height = images.shape[1]
        
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
    main(tyro.cli(SO100Args))