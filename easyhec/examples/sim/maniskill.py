from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
import sapien
import torch
import transforms3d
import tyro
from mani_skill.envs.sapien_env import BaseEnv

from easyhec.examples.sim.base import Args
from easyhec.optim.optimize import optimize
from easyhec.segmentation.interactive import InteractiveSegmentation
from easyhec.utils import visualization
from easyhec.utils.camera_conversions import opencv2ros, ros2opencv


def generate_synthetic_data(env: BaseEnv, samples: int, camera_name: str):
    if camera_name == "human_render_camera":
        render_camera = next(iter(env._human_render_cameras.values()))
    elif camera_name in env._sensors:
        render_camera = env._sensors[camera_name]
    else:
        raise ValueError(f"Camera {camera_name} not found in environment")
    mount_link = render_camera.camera.mount
    intrinsic = render_camera.camera.get_intrinsic_matrix()[0].cpu().numpy()

    # gets ground truth camera pose. If the camera is mounted then this extrinsic is in the reference frame of the mount link
    # in sapien the poses are stored in the ROS convention
    camera_pose = ros2opencv(
        render_camera.camera.get_local_pose().sp.to_transformation_matrix()
    )
    camera_height, camera_width = (
        render_camera.config.height,
        render_camera.config.width,
    )

    # find all visible links
    visible_links = [
        x
        for x in env.agent.robot.links
        if x._objs[0].entity.find_component_by_type(sapien.render.RenderBodyComponent)
        is not None
    ]

    # generate synthetic rgb images, segmentation images, and link poses of the robot in the scene.
    images = []
    segmentation_masks = []
    link_poses_dataset = []
    camera_mount_poses = []
    init_qpos = env.agent.robot.get_qpos().cpu().numpy()[0]
    for i in range(samples):
        noise = np.random.randn(init_qpos.shape[0]) * 0.2
        qpos = init_qpos.copy() + noise
        for active_joint in env.agent.robot.active_joints:
            qpos[active_joint.active_index] = np.clip(
                qpos[active_joint.active_index],
                active_joint.limits[0, 0],
                active_joint.limits[0, 1],
            )
        env.agent.robot.set_qpos(qpos)
        env.scene.update_render()
        render_camera.capture()
        data = render_camera.get_obs(rgb=True, segmentation=True)
        images.append(data["rgb"].clone())
        segmentation_masks.append(data["segmentation"][..., 0].clone())

        if mount_link is not None:
            camera_mount_poses.append(
                mount_link.pose.sp.inv().to_transformation_matrix()
            )

        link_poses = []
        for link in visible_links:
            for render_shape in (
                link._objs[0]
                .entity.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes
            ):
                link_visual_mesh_pose = link.pose.sp * render_shape.local_pose
                link_poses.append(link_visual_mesh_pose.to_transformation_matrix())
        link_poses_dataset.append(np.stack(link_poses))
    link_poses_dataset = np.stack(link_poses_dataset)
    if len(camera_mount_poses) > 0:
        camera_mount_poses = np.stack(camera_mount_poses)

    segmentation_images = []
    robot_masks = []
    for i in range(len(images)):
        images[i] = images[i][0].cpu().numpy()
        segmentation_masks[i] = segmentation_masks[i][0].cpu().numpy()
    for i in range(len(segmentation_masks)):
        segmentation_image = images[i].copy()
        segment_ids = []
        for link in env.agent.robot.links:
            segment_ids.append(link.per_scene_id[0])
        robot_mask = np.isin(segmentation_masks[i], segment_ids)
        segmentation_image[robot_mask] //= 4
        robot_masks.append(robot_mask)
        segmentation_images.append(segmentation_image)
    robot_masks = np.stack(robot_masks)

    mesh_paths = []
    for link in visible_links:
        # assumes there is only one render body component
        rb = link._objs[0].entity.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if rb is None:
            continue
        for render_shape in rb.render_shapes:
            mesh_filename = render_shape.filename
            mesh_paths.append(mesh_filename)

    return dict(
        images=images,
        segmentation_images=segmentation_images,
        mesh_paths=mesh_paths,
        segmentation_masks=segmentation_masks,
        robot_masks=robot_masks,
        link_poses_dataset=link_poses_dataset,
        intrinsic=intrinsic,
        camera_mount_poses=camera_mount_poses if len(camera_mount_poses) > 0 else None,
        camera_pose=camera_pose,
        camera_width=camera_width,
        camera_height=camera_height,
    )


@dataclass
class ManiSkillArgs(Args):
    output_dir: str = "results/maniskill"
    shader: str = "default"
    """Choice of shader to modify the rendering of the environment. default is a fast and cheap option, you can also try "rt" for ray-tracing option"""

    camera_name: str = "human_render_camera"
    """the name of the camera to try and calibrate the extrinsics for. ManiSkill envs have several cameras typically, one for human viewing labeled the human_render_camera and others for actual the actual cameras used for inputs to a robotics model. This script prints out all available cameras in the env and you can change the camera_name argument here to try and calibrate a different camera."""


def main(args: ManiSkillArgs):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### create the sim environment ###
    env = gym.make(
        args.env_id,
        human_render_camera_configs=dict(shader_pack=args.shader),
        sensor_configs=dict(shader_pack=args.shader),
    )
    env.reset()
    base_env: BaseEnv = env.unwrapped

    # print out all available cameras
    print("Available sensors/cameras used for models in the environment:")
    print(list(base_env._sensors.keys()))
    print("Available human render cameras in the environment:")
    print(list(base_env._human_render_cameras.keys()))
    if args.camera_name == "human_render_camera":
        print("Using human render camera for visualization")
    else:
        print("Using camera: ", args.camera_name)

    ### generate the synthetic data ###
    synthetic_data = generate_synthetic_data(
        base_env, args.samples + args.test_samples, args.camera_name
    )

    if not args.use_ground_truth_segmentation:
        interactive_segmentation = InteractiveSegmentation(
            segmentation_model="sam2",
            segmentation_model_cfg=dict(
                checkpoint=args.checkpoint, model_cfg=args.model_cfg
            ),
        )
        synthetic_data["robot_masks"] = interactive_segmentation.get_segmentation(
            synthetic_data["images"][: args.samples]
        )

    # data that you would need to collect in the real world
    link_poses_dataset = synthetic_data["link_poses_dataset"]
    intrinsic = synthetic_data["intrinsic"]
    camera_width = synthetic_data["camera_width"]
    camera_height = synthetic_data["camera_height"]
    mesh_paths = synthetic_data["mesh_paths"]
    robot_masks = synthetic_data["robot_masks"]
    # specific to cameras that are not fixed
    camera_mount_poses = synthetic_data["camera_mount_poses"]

    # data used just for visualization
    images = synthetic_data["images"]

    ### generate an initial guess around the ground truth pose ###
    ground_truth_camera_pose = synthetic_data["camera_pose"]
    initial_extrinsic_guess = ground_truth_camera_pose.copy()

    # pick a random direction to perturb the camera position.
    rand_vec = np.random.randn(3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    rand_vec = rand_vec * args.initial_extrinsic_guess_pos_error
    initial_extrinsic_guess[:3, 3] -= rand_vec

    rand_vec = np.random.randn(3)
    rand_vec = rand_vec / np.linalg.norm(rand_vec)
    rand_vec = rand_vec * args.initial_extrinsic_guess_rot_error
    initial_extrinsic_guess[:3, :3] = (
        transforms3d.axangles.axangle2mat(
            rand_vec, np.deg2rad(args.initial_extrinsic_guess_rot_error)
        )
        @ initial_extrinsic_guess[:3, :3]
    )

    ### run the optimization given the data ###

    predicted_camera_extrinsic_opencv = (
        optimize(
            camera_intrinsic=torch.from_numpy(intrinsic).float().to(device),
            masks=torch.from_numpy(robot_masks[: args.samples])
            .float()
            .to(device),
            link_poses_dataset=torch.from_numpy(link_poses_dataset[: args.samples])
            .float()
            .to(device),
            initial_extrinsic_guess=torch.from_numpy(initial_extrinsic_guess)
            .float()
            .to(device),
            meshes=mesh_paths,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_mount_poses=(
                torch.from_numpy(camera_mount_poses[: args.samples]).float().to(device)
                if camera_mount_poses is not None
                else None
            ),
            gt_camera_pose=torch.from_numpy(ground_truth_camera_pose)
            .float()
            .to(device),
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
        images=images[args.samples :],
        link_poses_dataset=link_poses_dataset[args.samples :],
        meshes=mesh_paths,
        intrinsic=intrinsic,
        extrinsics=np.stack(
            [
                initial_extrinsic_guess,
                predicted_camera_extrinsic_opencv,
                ground_truth_camera_pose,
            ]
        ),
        camera_mount_poses=camera_mount_poses[args.samples :] if camera_mount_poses is not None else None,
        labels=["Initial Extrinsic Guess", "Predicted Extrinsic", "Ground Truth"],
        output_dir=args.output_dir,
    )
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(ManiSkillArgs))
