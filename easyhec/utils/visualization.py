import os.path as osp
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer


def visualize_extrinsic_results(
    images,
    link_poses_dataset,
    meshes,
    intrinsic: np.ndarray,
    extrinsics: np.ndarray,
    camera_mount_poses: Optional[np.ndarray] = None,
    labels: List[str] = [],
    output_dir="results/",
):
    """
    Visualizes a given list of extrinsic matrices and draws the mask cameras at those extrinsics would project on the original RGB images.

    Args:
        images (np.ndarray, shape (N, H, W, 3)): List of RGB images to visualize.
        link_poses_dataset (np.ndarray, shape (N, L, 4, 4)): Link poses relative to any frame (e.g. the robot base frame), where N is the number of samples, L is the number of links
        meshes (List[str | trimesh.Trimesh]): List of mesh paths or trimesh.Trimesh objects for each of the links
        intrinsic (np.ndarray, shape (3, 3)): Camera intrinsic matrix
        extrinsics (np.ndarray, shape (M, 4, 4)): Extrinsic matrices to visualize
        camera_mount_poses (np.ndarray, shape (N, 4, 4)): Camera mount poses relative to the robot base frame, where N is the number of samples. If none then camera is assumed to be fixed.
        labels (List[str]): List of labels for each of the extrinsics
        output_dir (str): Directory to save the visualizations
    """
    ### visualization code for the predicted extrinsic ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_height, camera_width = images[0].shape[:2]
    renderer = NVDiffrastRenderer(camera_height, camera_width)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    if camera_mount_poses is not None:
        camera_mount_poses = torch.from_numpy(camera_mount_poses).float().to(device)
    link_poses_dataset = torch.from_numpy(link_poses_dataset).float().to(device)

    for i in range(len(meshes)):
        if isinstance(meshes[i], str):
            meshes[i] = trimesh.load(osp.expanduser(meshes[i]), force="mesh")
    link_vertices = [mesh.vertices.copy() for mesh in meshes]
    link_faces = [mesh.faces.copy() for mesh in meshes]
    link_vertices = [
        torch.from_numpy(mesh.vertices).float().to(device) for mesh in meshes
    ]
    link_faces = [torch.from_numpy(mesh.faces).int().to(device) for mesh in meshes]

    def get_mask_from_camera_pose(camera_pose):
        mask = torch.zeros((camera_height, camera_width), device=device)
        for j, link_pose in enumerate(link_poses_dataset[i]):
            link_mask = renderer.render_mask(
                link_vertices[j],
                link_faces[j],
                intrinsic,
                camera_pose @ link_pose,
            )
            link_mask = link_mask.detach()
            mask[link_mask > 0] = 1
        return mask

    for i in tqdm(range(len(images))):
        overlaid_images = []
        for j in range(len(extrinsics)):
            if camera_mount_poses is not None:
                mask = get_mask_from_camera_pose(extrinsics[j] @ camera_mount_poses[i])
            else:
                mask = get_mask_from_camera_pose(extrinsics[j])
            mask = mask.cpu().numpy()
            overlaid_images.append(images[i].copy())
            overlaid_images[-1][mask > 0] = overlaid_images[-1][mask > 0] // 4

        plt.figure(figsize=(7 * len(extrinsics), 7))
        for j in range(len(extrinsics)):
            plt.subplot(1, len(extrinsics), j + 1)
            plt.imshow(overlaid_images[j])
            plt.axis("off")
            plt.title(labels[j])

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/{i}.png")
