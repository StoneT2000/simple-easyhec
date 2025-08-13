import nvdiffrast.torch as dr
import torch

from easyhec.utils import utils_3d


class NVDiffrastRenderer:
    def __init__(self, height: int, width: int):
        self.H, self.W = height, width
        self.resolution = (height, width)
        blender2opencv = (
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            .float()
            .cuda()
        )
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        intrinsic: torch.Tensor,
        object_pose: torch.Tensor,
        anti_aliasing: bool = True,
    ) -> torch.Tensor:
        """
        Differentiable rendering of given vertices and faces given the pose of the object and intrinsics of the camera

        Parameters:
            verts (torch.Tensor, shape (N, 3)): vertices of the object
            faces (torch.Tensor, shape (M, 3)): faces of the object
            intrinsic (torch.Tensor, shape (3, 3)): intrinsic matrix of the camera
            object_pose (torch.Tensor, shape (4, 4)): pose of the object
            anti_aliasing (bool): Default is True. If True, will use antialiasing.

        """
        proj = utils_3d.K_to_projection(intrinsic, self.H, self.W)
        pose = self.opencv2blender @ object_pose
        pos_clip = utils_3d.transform_pos(proj @ pose, verts)
        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, faces, resolution=self.resolution
        )
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask
