"""Clinical Glioblastoma Invasiveness and Survival Prognostics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class GlioblastomaPrognosticReport:
    """Quantitative invasiveness score and clinical prognosis."""
    diffusion_coefficient_mm2_day: float
    proliferation_rate_per_day: float
    invasiveness_ratio_mm2: float  # D / rho
    tumor_phenotype: Literal["DIFFUSE_INFILTRATIVE", "INTERMEDIATE", "NODULAR_CIRCUMSCRIBED"]
    recurrence_risk_level: Literal["EXTREME", "HIGH", "MODERATE"]
    predicted_median_survival_months: float
    clinical_guidance: str


class TumorBiomarkerEvaluator:
    """Computes Swanson invasiveness ratio D/rho and prognostic survival indices."""

    @staticmethod
    def evaluate_phenotype(
        diffusion_rate: float,
        proliferation_rate: float,
        mgmt_methylated: bool = True,
    ) -> GlioblastomaPrognosticReport:
        """
        Evaluate D/rho invasiveness index:
        D/rho > 5 mm^2: Diffuse, widely infiltrative, early distant recurrence.
        D/rho < 1 mm^2: Nodular, focal, high local cell density.
        """
        ratio = diffusion_rate / max(1e-6, proliferation_rate)

        if ratio >= 4.0:
            phenotype = "DIFFUSE_INFILTRATIVE"
            risk = "EXTREME"
            base_survival = 12.0
            guidance = (
                f"Highly diffuse infiltration (D/rho = {ratio:.2f} mm^2). Sub-radiological margins extend > 2-3 cm "
                f"beyond visible T2/FLAIR hyperintensity. Extended margin radiotherapy and adjuvant TMZ mandatory."
            )
        elif ratio >= 1.5:
            phenotype = "INTERMEDIATE"
            risk = "HIGH"
            base_survival = 15.0
            guidance = (
                f"Intermediate invasiveness (D/rho = {ratio:.2f} mm^2). Standard RTOG 2 cm margin conformal field recommended."
            )
        else:
            phenotype = "NODULAR_CIRCUMSCRIBED"
            risk = "MODERATE"
            base_survival = 20.0
            guidance = (
                f"Predominantly nodular morphology (D/rho = {ratio:.2f} mm^2). High potential for gross total resection (GTR)."
            )

        # MGMT promoter methylation confers approximately +6-8 months overall survival benefit
        if mgmt_methylated:
            median_survival = base_survival + 6.5
            guidance += " MGMT promoter is METHYLATED: Expect robust response to alkylating Temozolomide."
        else:
            median_survival = base_survival
            guidance += " MGMT promoter is UNMETHYLATED: Inherent resistance to alkylating agents; consider clinical trials."

        return GlioblastomaPrognosticReport(
            diffusion_coefficient_mm2_day=diffusion_rate,
            proliferation_rate_per_day=proliferation_rate,
            invasiveness_ratio_mm2=ratio,
            tumor_phenotype=phenotype,
            recurrence_risk_level=risk,
            predicted_median_survival_months=median_survival,
            clinical_guidance=guidance,
        )
