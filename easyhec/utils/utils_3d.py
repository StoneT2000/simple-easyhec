import cv2
import numpy as np
import torch
import trimesh

from easyhec.utils.pytorch3d_se3 import (_get_se3_V_input, _se3_V_matrix,
                                         se3_exp_map)
from easyhec.utils.pytorch3d_se3 import se3_log_map as se3_log_map_pytorch3d


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


def K_to_projection(K, H, W, n=0.001, f=10.0):
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = (
        torch.tensor(
            [
                [2 * fu / W, 0, -2 * cu / W + 1, 0],
                [0, 2 * fv / H, 2 * cv / H - 1, 0],
                [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                [0, 0, -1, 0],
            ]
        )
        .cuda()
        .float()
    )
    return proj


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def se3_log_map(
    transform: torch.Tensor,
    eps: float = 1e-4,
    cos_bound: float = 1e-4,
    backend=None,
    test_acc=True,
):
    if backend is None:
        backend = "pytorch3d"
    if backend == "pytorch3d":
        dof6 = se3_log_map_pytorch3d(transform, eps, cos_bound)
    elif backend == "opencv":
        # from pytorch3d.common.compat import solve
        log_rotation = []
        for tsfm in transform:
            cv2_rot = -cv2.Rodrigues(tsfm[:3, :3].detach().cpu().numpy().astype(float))[
                0
            ]
            log_rotation.append(
                torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float()
            )
        log_rotation = torch.stack(log_rotation, dim=0)
        T = transform[:, 3, :3]
        V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
        log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
        dof6 = torch.cat((log_translation, log_rotation), dim=1)
    else:
        raise NotImplementedError()
    if test_acc:
        err = (se3_exp_map(dof6) - transform).abs().max()
        if err > 0.1:
            raise RuntimeError()
    return dof6
