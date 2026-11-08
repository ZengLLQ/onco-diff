"""Linear-Quadratic (LQ) Radiobiology and Surgical Resection Modeling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np


@dataclass
class RadiosensitivityParameters:
    """Linear-Quadratic model parameters for tumor tissue."""
    alpha_per_gy: float = 0.05    # Intrinsic single-hit radiosensitivity (1/Gy)
    beta_per_gy2: float = 0.005   # Double-hit radiosensitivity (1/Gy^2)
    hypoxia_reduction_factor: float = 0.5  # Radioresistance in necrotic/hypoxic core

    @property
    def alpha_beta_ratio_gy(self) -> float:
        return self.alpha_per_gy / self.beta_per_gy2 if self.beta_per_gy2 > 0 else 10.0


@dataclass
class RadiationFraction:
    """A scheduled radiotherapy session."""
    day: float
    dose_gy: float = 2.0
    target_mask: Optional[np.ndarray] = None


class RadiotherapyProtocol:
    """Applies fractionated radiotherapy and surgical resection interventions."""

    def __init__(self, params: Optional[RadiosensitivityParameters] = None):
        self.params = params or RadiosensitivityParameters()

    def compute_surviving_fraction(
        self,
        dose_gy: float,
        is_hypoxic_mask: Optional[np.ndarray] = None,
    ) -> float | np.ndarray:
        """
        Evaluate LQ surviving fraction: S(d) = exp(-alpha*d - beta*d^2).
        """
        p = self.params
        if is_hypoxic_mask is not None:
            # Reduced alpha in hypoxic regions
            eff_alpha = np.where(is_hypoxic_mask, p.alpha_per_gy * p.hypoxia_reduction_factor, p.alpha_per_gy)
            eff_beta = np.where(is_hypoxic_mask, p.beta_per_gy2 * p.hypoxia_reduction_factor, p.beta_per_gy2)
            return np.exp(-eff_alpha * dose_gy - eff_beta * (dose_gy ** 2))
        else:
            return float(np.exp(-p.alpha_per_gy * dose_gy - p.beta_per_gy2 * (dose_gy ** 2)))

    def apply_fraction(
        self,
        tumor_density: np.ndarray,
        dose_gy: float,
        radiation_field_mask: Optional[np.ndarray] = None,
        hypoxic_threshold: float = 0.85,
    ) -> np.ndarray:
        """
        Apply a discrete radiotherapy fraction to tumor cell density array.
        """
        c = np.copy(tumor_density)
        is_hypoxic = (c >= hypoxic_threshold)
        surv_fraction = self.compute_surviving_fraction(dose_gy, is_hypoxic)

        if radiation_field_mask is not None:
            c[radiation_field_mask] *= surv_fraction[radiation_field_mask] if isinstance(surv_fraction, np.ndarray) else surv_fraction
        else:
            c *= surv_fraction

        return c

    @staticmethod
    def apply_surgical_resection(
        tumor_density: np.ndarray,
        resection_threshold: float = 0.80,
        margin_voxels: int = 1,
    ) -> np.ndarray:
        """
        Simulate surgical resection removing all macroscopically visible tumor
        above resection_threshold, plus a microscopic safety margin.
        """
        from scipy.ndimage import binary_dilation
        c = np.copy(tumor_density)
        resection_core = (c >= resection_threshold)
        
        if margin_voxels > 0:
            resection_zone = binary_dilation(resection_core, iterations=margin_voxels)
        else:
            resection_zone = resection_core

        c[resection_zone] = 0.0
        return c
