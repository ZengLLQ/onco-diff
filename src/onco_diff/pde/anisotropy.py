"""Diffusion Tensor Imaging (DTI) Anisotropy and Brain Tissue Microstructure."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class TissueProperties:
    """Diffusion rates and carrying capacity per brain tissue class."""
    diff_white_matter_mm2_day: float = 0.15  # Fast axonal migration
    diff_gray_matter_mm2_day: float = 0.03   # Isotropic slower diffusion
    proliferation_rate_per_day: float = 0.025 # Tumor growth rate rho
    carrying_capacity: float = 1.0           # Max normalized tumor cell density


class DTITensorField:
    """Manages spatially varying anisotropic diffusion tensors derived from DTI."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        voxel_size_mm: float = 1.0,
        default_properties: Optional[TissueProperties] = None,
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.voxel_size_mm = voxel_size_mm
        self.props = default_properties or TissueProperties()

        # Tissue masks: 0: CSF/background, 1: Gray matter, 2: White matter
        self.tissue_map = np.ones(shape, dtype=int)
        
        # Tensor matrix D(x): shape (ndim, ndim, *shape)
        self.d_tensor = np.zeros((self.ndim, self.ndim, *shape), dtype=np.float64)
        self._initialize_isotropic_tensors()

    def _initialize_isotropic_tensors(self) -> None:
        """Initialize diagonal isotropic diffusion according to gray matter."""
        for d in range(self.ndim):
            self.d_tensor[d, d] = self.props.diff_gray_matter_mm2_day

    def set_white_matter_tract(
        self,
        mask: np.ndarray,
        principal_direction: Sequence[float],
        anisotropy_ratio: float = 5.0,
    ) -> None:
        """
        Assign an anisotropic white matter fiber tract.
        
        Args:
            mask: Boolean array indicating tract voxels.
            principal_direction: Unit vector along fiber tract (e.g. [1, 0] or [1, 0, 0]).
            anisotropy_ratio: Ratio of longitudinal diffusion to transverse diffusion (Dw / Dg).
        """
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match field shape {self.shape}")

        v = np.asarray(principal_direction, dtype=np.float64)
        norm_v = np.linalg.norm(v)
        if norm_v > 0:
            v = v / norm_v
        else:
            v[0] = 1.0

        d_long = self.props.diff_white_matter_mm2_day
        d_trans = d_long / max(1.0, anisotropy_ratio)

        # Projection matrix P = v * v^T
        p_proj = np.outer(v, v)
        i_mat = np.eye(self.ndim)
        
        # Tensor D = d_trans * I + (d_long - d_trans) * (v * v^T)
        tract_d = d_trans * i_mat + (d_long - d_trans) * p_proj

        self.tissue_map[mask] = 2  # White matter
        for i in range(self.ndim):
            for j in range(self.ndim):
                self.d_tensor[i, j, mask] = tract_d[i, j]

    def set_csf_ventricles(self, mask: np.ndarray) -> None:
        """Set cerebrospinal fluid ventricles where tumor cell diffusion is blocked."""
        self.tissue_map[mask] = 0
        for i in range(self.ndim):
            for j in range(self.ndim):
                self.d_tensor[i, j, mask] = 0.0

    def compute_fractional_anisotropy(self) -> np.ndarray:
        """
        Compute point-wise Fractional Anisotropy (FA) map.
        FA ranges from 0 (perfect spherical isotropy) to 1 (linear tract).
        """
        fa = np.zeros(self.shape, dtype=np.float64)
        
        # Reshape for batch eigenvalue computation
        n_vox = int(np.prod(self.shape))
        flat_tensors = np.moveaxis(self.d_tensor, [0, 1], [-2, -1]).reshape(n_vox, self.ndim, self.ndim)
        
        evals = np.linalg.eigvalsh(flat_tensors)  # Eigenvalues in ascending order
        mean_eval = np.mean(evals, axis=-1, keepdims=True)

        var_eval = np.sum((evals - mean_eval) ** 2, axis=-1)
        norm_eval = np.sum(evals ** 2, axis=-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.sqrt(3.0 / 2.0) * np.sqrt(var_eval) / np.sqrt(norm_eval + 1e-12)
            val = np.where(norm_eval > 1e-12, val, 0.0)

        fa = np.clip(val.reshape(self.shape), 0.0, 1.0)
        return fa
