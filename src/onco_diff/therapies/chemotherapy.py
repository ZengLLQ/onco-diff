"""Alkylating Chemotherapy (Temozolomide TMZ) and MGMT Biomarker Modeling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TemozolomideParameters:
    """Pharmacodynamic parameters for TMZ cytotoxicity."""
    max_kill_rate_per_day: float = 0.045  # Daily fractional cell kill at saturated drug exposure
    ec50_mg_l: float = 2.0  # TMZ concentration for half-maximal cytotoxicity
    mgmt_methylated: bool = True  # Methylated (silenced repair) vs Unmethylated (resistant)
    mgmt_resistance_factor: float = 0.35  # Efficacy fraction retained when MGMT is unmethylated


class ChemotherapyRegimen:
    """Models systemic TMZ cycles and daily continuous cytotoxicity."""

    def __init__(self, params: Optional[TemozolomideParameters] = None):
        self.params = params or TemozolomideParameters()

    def compute_daily_kill_rate(self, plasma_conc_mg_l: float = 5.0) -> float:
        """
        Compute daily fractional cell death rate:
        kill = k_max * (C / (EC50 + C)) * (1.0 if methylated else resistance_factor)
        """
        p = self.params
        c = max(0.0, plasma_conc_mg_l)
        saturation = c / (p.ec50_mg_l + c) if (p.ec50_mg_l + c) > 0 else 0.0

        methylation_efficiency = 1.0 if p.mgmt_methylated else p.mgmt_resistance_factor
        return float(p.max_kill_rate_per_day * saturation * methylation_efficiency)

    def apply_chemotherapy_kill(
        self,
        tumor_density: np.ndarray,
        dt_days: float,
        plasma_conc_mg_l: float = 5.0,
    ) -> np.ndarray:
        """Apply continuous TMZ cytotoxicity across time step dt."""
        rate = self.compute_daily_kill_rate(plasma_conc_mg_l)
        # Exponential survival over dt: c_new = c * exp(-rate * dt)
        return tumor_density * np.exp(-rate * dt_days)
