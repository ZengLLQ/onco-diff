"""ASCII Visualizer and Clinical Oncology Report Renderer."""

from __future__ import annotations
import numpy as np
from onco_diff.imaging.virtual_mri import VirtualMRISegmentation
from onco_diff.imaging.tumor_metrics import GlioblastomaPrognosticReport


def render_ascii_tumor_slice(density_slice: np.ndarray, height: int = 12, width: int = 36) -> str:
    """
    Render 2D tumor cell density slice as ASCII density map:
    ' ' : Normal tissue (< 0.01)
    '.' : Infiltration (0.01 - 0.16)
    '*' : T2/FLAIR edema (0.16 - 0.80)
    '#' : Macroscopic enhancing core (>= 0.80)
    """
    from scipy.ndimage import zoom
    zoom_y = height / density_slice.shape[0]
    zoom_x = width / density_slice.shape[1]
    resampled = zoom(density_slice, (zoom_y, zoom_x), order=1)

    lines = ["+" + "-" * width + "+"]
    for r in range(height):
        row_chars = []
        for c in range(width):
            val = resampled[r, c]
            if val < 0.01:
                ch = " "
            elif val < 0.16:
                ch = "."
            elif val < 0.80:
                ch = "*"
            else:
                ch = "#"
            row_chars.append(ch)
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def render_oncology_report(
    report: GlioblastomaPrognosticReport,
    seg: VirtualMRISegmentation,
    total_cell_burden: float,
) -> str:
    """Render clinical computational oncology report."""
    border = "=" * 70
    sub_border = "-" * 70

    out = [
        border,
        "    OncoDiff: Spatiotemporal Glioblastoma Neuro-Oncology Report      ",
        border,
        "  VIRTUAL MRI TUMOR SUB-REGION VOLUMES:",
        f"    T1Gd Enhancing Core (GTV)      : {seg.t1gd_volume_cm3:6.2f} cm^3 (Dense tumor >= 0.80)",
        f"    T2/FLAIR Hyperintensity (CTV)  : {seg.t2_flair_volume_cm3:6.2f} cm^3 (Edema & invasion >= 0.16)",
        f"    Sub-radiological Infiltration  : {seg.infiltrative_volume_cm3:6.2f} cm^3 (Microscopic margin)",
        f"    Estimated Total Cell Burden    : {total_cell_burden:6.1f} normalized units",
        sub_border,
        "  BIOPHYSICAL INVASIVENESS & PROGNOSTIC STRATIFICATION:",
        f"    Diffusion Coefficient D        : {report.diffusion_coefficient_mm2_day:6.3f} mm^2/day",
        f"    Proliferation Rate rho         : {report.proliferation_rate_per_day:6.3f} /day",
        f"    Swanson Invasiveness Ratio     : {report.invasiveness_ratio_mm2:6.2f} mm^2 (D/rho)",
        f"    Tumor Invasiveness Phenotype   : {report.tumor_phenotype}",
        f"    Recurrence Risk Level          : {report.recurrence_risk_level}",
        f"    Predicted Median Survival      : {report.predicted_median_survival_months:4.1f} months",
        sub_border,
        "  CLINICAL ONCOLOGY GUIDANCE:",
        f"    Recommendation                 : {report.clinical_guidance}",
        border,
    ]
    return "\n".join(out)
