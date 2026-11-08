"""Virtual Neuroimaging: Multi-Modal MRI Lesion Segmentation and Synthesis."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VirtualMRISegmentation:
    """Clinical MRI-derived tumor sub-regions based on cell density thresholds."""
    t1gd_enhancing_core: np.ndarray    # Dense macroscopic tumor (c >= 0.80)
    t2_flair_hyperintense: np.ndarray  # Infiltration & edema (c >= 0.16)
    invisible_infiltrative: np.ndarray # Sub-radiological infiltration (0.01 <= c < 0.16)
    t1gd_volume_cm3: float
    t2_flair_volume_cm3: float
    infiltrative_volume_cm3: float


class VirtualMRIGenerator:
    """
    Translates continuous in silico cell density distributions into
    standard clinical diagnostic neuroimaging representations (T1Gd and T2/FLAIR).
    """

    def __init__(
        self,
        t1gd_threshold: float = 0.80,
        t2_flair_threshold: float = 0.16,
        subradiological_threshold: float = 0.01,
        voxel_volume_mm3: float = 1.0,
    ):
        self.t1gd_threshold = t1gd_threshold
        self.t2_flair_threshold = t2_flair_threshold
        self.subrad_threshold = subradiological_threshold
        self.vox_vol_mm3 = voxel_volume_mm3

    def segment_tumor(self, cell_density: np.ndarray) -> VirtualMRISegmentation:
        """Segment continuous density into clinical MRI signal categories."""
        t1gd_mask = (cell_density >= self.t1gd_threshold)
        t2_mask = (cell_density >= self.t2_flair_threshold)
        invis_mask = (cell_density >= self.subrad_threshold) & (cell_density < self.t2_flair_threshold)

        # 1 cm^3 = 1000 mm^3
        t1gd_vol = (float(np.sum(t1gd_mask)) * self.vox_vol_mm3) / 1000.0
        t2_vol = (float(np.sum(t2_mask)) * self.vox_vol_mm3) / 1000.0
        invis_vol = (float(np.sum(invis_mask)) * self.vox_vol_mm3) / 1000.0

        return VirtualMRISegmentation(
            t1gd_enhancing_core=t1gd_mask,
            t2_flair_hyperintense=t2_mask,
            invisible_infiltrative=invis_mask,
            t1gd_volume_cm3=t1gd_vol,
            t2_flair_volume_cm3=t2_vol,
            infiltrative_volume_cm3=invis_vol,
        )

    def synthesize_t1_contrast_image(
        self,
        cell_density: np.ndarray,
        parenchyma_mask: np.ndarray,
        ventricle_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize T1 post-contrast weighted MRI image.
        Normal brain ~ 0.5, CSF ~ 0.1, Enhancing rim ~ 0.95, Necrotic core ~ 0.2.
        """
        img = np.zeros_like(cell_density, dtype=np.float64)
        img[parenchyma_mask] = 0.50
        img[ventricle_mask] = 0.10

        # Necrotic core (very high density c > 0.9)
        img[cell_density >= 0.90] = 0.20
        # Enhancing rim (0.8 <= c < 0.9)
        enhancing = (cell_density >= self.t1gd_threshold) & (cell_density < 0.90)
        img[enhancing] = 0.95

        return img
