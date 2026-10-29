"""Bridge TIGRE cone-beam projection and PyTorch autograd."""

import itertools
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import tigre
import tigre.utilities.gpu as gpu


class _TigreAxAtbLegacy(torch.autograd.Function):
    """Use TIGRE Ax for forward projection and matched Atb for backward gradients."""

    @staticmethod
    def forward(
        ctx, volume_input: torch.Tensor, geo, angles: np.ndarray, gpuids: List[int]
    ) -> torch.Tensor:
        """Project the trailing [Z, X, Y] volume dimensions and preserve leading batch dimensions."""
        device = volume_input.device
        # TIGRE operates on NumPy arrays; autograd is supplied by backward().
        vol_np = volume_input.detach().cpu().numpy()

        ctx.geo = geo
        ctx.angles = angles
        ctx.gpuids = gpuids
        ctx.vol_shape = tuple(volume_input.shape[-3:])

        extra_dims = list(volume_input.shape[:-3])

        if len(extra_dims) == 0:
            sample = tigre.Ax(vol_np, geo, angles, projection_type="interpolated", gpuids=gpuids)
            sample_t = torch.from_numpy(sample).to(device)
            out = volume_input.new_empty(tuple(sample_t.shape), dtype=torch.float32)
            out.copy_(sample_t)
            return out

        sample_idx = tuple(0 for _ in extra_dims)
        sample = tigre.Ax(
            vol_np[sample_idx], geo, angles, projection_type="interpolated", gpuids=gpuids
        )
        sample_t = torch.from_numpy(sample).to(device)

        out = volume_input.new_empty(tuple(extra_dims) + tuple(sample_t.shape), dtype=torch.float32)
        out[sample_idx] = sample_t

        for subspace in itertools.product(*[range(d) for d in extra_dims]):
            if subspace == sample_idx:
                continue
            out[subspace] = torch.from_numpy(
                tigre.Ax(
                    vol_np[subspace], geo, angles, projection_type="interpolated", gpuids=gpuids
                )
            ).to(device)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Back-project upstream projection gradients using the existing matched TIGRE operator."""
        device = grad_output.device
        grad_np = grad_output.detach().cpu().numpy()

        extra_dims = list(grad_output.shape[:-3])

        if len(extra_dims) == 0:
            vol = tigre.Atb(
                grad_np, ctx.geo, ctx.angles, backprojection_type="matched", gpuids=ctx.gpuids
            )
            vol_t = torch.from_numpy(vol).to(device)
            return vol_t, None, None, None

        vol_out = grad_output.new_empty(
            tuple(extra_dims) + tuple(ctx.vol_shape), dtype=torch.float32
        )

        for subspace in itertools.product(*[range(d) for d in extra_dims]):
            vol_out[subspace] = torch.from_numpy(
                tigre.Atb(
                    grad_np[subspace],
                    ctx.geo,
                    ctx.angles,
                    backprojection_type="matched",
                    gpuids=ctx.gpuids,
                )
            ).to(device)

        return vol_out, None, None, None


class ConeBeam3DProjector:
    """Provide the existing cone-beam projection interface."""

    def __init__(
        self,
        image_size: Sequence[int],
        image_reso: Sequence[float],
        euler_deg: Sequence[float],
        proj_size: Sequence[int],
        proj_reso: Sequence[float],
        dde: float,
        dso: float,
        dsd: Optional[float] = None,
        tigre_gpu_id: Optional[int] = None,
        torch_device: Optional[Union[str, torch.device]] = None,
    ):
        self.image_size = np.array(image_size, dtype=np.int32)
        self.image_reso = np.array(image_reso, dtype=np.float32)
        self.proj_size = np.array(proj_size, dtype=np.int32)
        self.proj_reso = np.array(proj_reso, dtype=np.float32)

        self.dso = float(dso)
        self.dsd = float(dsd) if dsd is not None else float(dso) + float(dde)

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(torch_device)

        if tigre_gpu_id is None:
            ids = gpu.getGpuIds()
            tigre_gpu_id = int(ids[0]) if isinstance(ids, (list, tuple)) and len(ids) > 0 else 0

        # Preserve the existing selection of the first visible GPU.
        self.gpuids = gpu.getGpuIds()

        try:
            self.gpuids.devices = [0]
        except Exception as e:
            raise RuntimeError(f"Failed to set tigre gpuids.devices. gpuids={self.gpuids}, err={e}")

        geo = tigre.geometry()
        geo.offDetector = np.array([0, 0], dtype=np.float32)
        geo.accuracy = 1
        geo.COR = 0
        geo.rotDetector = np.array([0, 0, 0], dtype=np.float32)
        geo.mode = "cone"

        geo.nDetector = np.array([int(self.proj_size[0]), int(self.proj_size[1])], dtype=np.int32)
        geo.dDetector = np.array(
            [float(self.proj_reso[0]), float(self.proj_reso[1])], dtype=np.float32
        )
        geo.sDetector = geo.nDetector * geo.dDetector

        geo.nVoxel = np.array(
            [int(self.image_size[0]), int(self.image_size[1]), int(self.image_size[2])],
            dtype=np.int32,
        )
        geo.dVoxel = np.array(
            [float(self.image_reso[0]), float(self.image_reso[1]), float(self.image_reso[2])],
            dtype=np.float32,
        )
        geo.sVoxel = geo.nVoxel * geo.dVoxel

        geo.DSD = float(self.dsd)
        geo.DSO = float(self.dso)

        geo.offOrigin = np.array([0, 0, 0], dtype=np.float32)

        angles = np.array([[float(euler_deg[0]), float(euler_deg[1]), 0.0]], dtype=np.float32)
        angles = angles / 180.0 * np.pi
        geo.angles = angles

        self.geo = geo
        self.angles = angles

    def get_main_ray_direction(self):
        """Return the existing main-ray direction used by the directional TV penalty."""
        src = np.array([0.0, -self.dso, 0.0], dtype=np.float32)
        det = np.array([0.0, self.dsd - self.dso, 0.0], dtype=np.float32)

        ray = det - src
        ray = ray / np.linalg.norm(ray)

        pri, sec, _ = self.angles[0]

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pri), -np.sin(pri)],
                [0, np.sin(pri), np.cos(pri)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(sec), 0, np.sin(sec)],
                [0, 1, 0],
                [-np.sin(sec), 0, np.cos(sec)],
            ]
        )

        # Retain the existing rotation convention used by the regularizer.
        R = Ry @ Rx
        ray_world = R @ ray
        ray_world = ray_world / np.linalg.norm(ray_world)

        return torch.tensor(ray_world, dtype=torch.float32)

    def _ensure_5d(self, vol: torch.Tensor) -> torch.Tensor:
        """Normalize volume layout to [batch, channel, Z, X, Y]."""
        if vol.ndim == 3:
            return vol[None, None, ...]
        if vol.ndim == 4:
            return vol[:, None, ...]
        if vol.ndim == 5:
            return vol
        raise ValueError(f"Unexpected volume ndim={vol.ndim}, shape={tuple(vol.shape)}")

    def _normalize_projection_shape(self, projs: torch.Tensor) -> torch.Tensor:
        """Normalize supported TIGRE layouts to [batch, view, height, width]."""
        if projs.ndim == 3:
            return projs.unsqueeze(1)

        if projs.ndim != 4:
            raise ValueError(f"Unexpected projection ndim={projs.ndim}, shape={tuple(projs.shape)}")

        if projs.shape[1] == 1:
            return projs

        if projs.shape[-1] == 1:
            return projs.permute(0, 3, 1, 2).contiguous()

        return projs

    def forward_project(self, volume: torch.Tensor) -> torch.Tensor:
        """Project a volume using the existing geometry and differentiable operator."""
        volume = self._ensure_5d(volume).contiguous().to(self.torch_device).float()

        if volume.shape[1] != 1:
            raise ValueError(f"Expected channel C=1, got volume shape {tuple(volume.shape)}")

        vol3d = volume[:, 0]

        projs = _TigreAxAtbLegacy.apply(vol3d, self.geo, self.angles, self.gpuids)

        projs = self._normalize_projection_shape(projs)

        return projs
