"""Retain the original two-view dataset with geometry in meters."""

import torch
import numpy as np

from torch.utils.data import Dataset


class ConeGeometry(object):
    """Store the geometry values used by this dataset implementation."""

    def __init__(self, data, index=0):

        self.DSD = data["DSD"][index] / 1000
        self.DSO = data["DSO"][index] / 1000

        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000
        self.sDetector = self.nDetector * self.dDetector

        self.nVoxel = np.array(data["nVoxel"])
        self.dVoxel = np.array(data["dVoxel"]) / 1000
        self.sVoxel = self.nVoxel * self.dVoxel


class TIGREDataset(Dataset):
    """Provide projection observations and voxel-center coordinates."""

    def __init__(self, data, device="cuda"):
        super().__init__()

        self.projs = torch.tensor(data["projections"], dtype=torch.float32, device=device)

        self.geo_one = ConeGeometry(data, index=0)
        self.geo_two = ConeGeometry(data, index=1)
        self.n_samples = data["numTrain"]
        geo = self.geo_one
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, geo.nDetector[1] - 1, geo.nDetector[1], device=device),
                torch.linspace(0, geo.nDetector[0] - 1, geo.nDetector[0], device=device),
            ),
            -1,
        )
        self.coords = torch.reshape(coords, [-1, 2])
        self.voxels = torch.tensor(self.get_voxels(geo), dtype=torch.float32, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        projs = self.projs[index]
        out = {
            "projs": projs,
        }

        return out

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
