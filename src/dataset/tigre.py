"""Build multi-view projection tensors and centered voxel coordinates."""

import torch
import numpy as np

from torch.utils.data import Dataset


class ConeGeometry(object):
    """Store the geometry values used by this dataset implementation."""

    def __init__(self, data, index=0):

        def _at(x, i):
            """Select a per-view geometry value while accepting shared scalar values."""
            if isinstance(x, (list, tuple, np.ndarray)):
                return x[i]
            if torch.is_tensor(x):
                return x[i].item() if x.ndim > 0 else x.item()
            return x

        self.DSD = _at(data["DSD"], index)
        self.DSO = _at(data["DSO"], index)

        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"])
        self.sDetector = self.nDetector * self.dDetector

        self.nVoxel = np.array(data["nVoxel"])
        self.dVoxel = np.array(data["dVoxel"])
        self.sVoxel = self.nVoxel * self.dVoxel


class TIGREDataset(Dataset):
    """Provide projection observations and voxel-center coordinates."""

    def __init__(self, data, device="cuda"):
        super().__init__()

        projs = torch.as_tensor(data["projections"], dtype=torch.float32, device=device)

        if projs.ndim == 3:
            projs = projs.unsqueeze(1)
        if projs.ndim != 4:
            raise ValueError(
                f"Expected projections with shape [N, V, H, W] (or [N, H, W]); got {tuple(projs.shape)}"
            )

        self.projs = projs
        self.n_samples = int(data["numTrain"])
        self.n_views = int(self.projs.shape[1])

        self.geos = [ConeGeometry(data, index=i) for i in range(self.n_views)]
        geo0 = self.geos[0]

        xs = torch.linspace(0, geo0.nDetector[1] - 1, int(geo0.nDetector[1]), device=device)
        ys = torch.linspace(0, geo0.nDetector[0] - 1, int(geo0.nDetector[0]), device=device)

        try:
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        except TypeError:
            grid_x, grid_y = torch.meshgrid(xs, ys)

        coords = torch.stack((grid_x, grid_y), dim=-1)

        if coords.shape[0] == int(geo0.nDetector[1]) and coords.shape[1] == int(geo0.nDetector[0]):
            coords = coords.permute(1, 0, 2)

        self.coords = torch.reshape(coords, [-1, 2])
        self.voxels = torch.as_tensor(self.get_voxels(geo0), dtype=torch.float32, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        projs = self.projs[index]
        return {"projs": projs}

    def get_voxels(self, geo: ConeGeometry):
        """Generate voxel centers on the existing centered reconstruction grid."""
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3),
            indexing="ij",
        )
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
