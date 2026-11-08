"""Command-Line Interface (CLI) for OncoDiff."""

from __future__ import annotations
import argparse
import sys
import numpy as np
from onco_diff.pde import (
    TissueProperties,
    DTITensorField,
    GliomaReactionDiffusionSolver,
)
from onco_diff.therapies import (
    RadiotherapyProtocol,
    ChemotherapyRegimen,
    TemozolomideParameters,
)
from onco_diff.imaging import (
    VirtualMRIGenerator,
    TumorBiomarkerEvaluator,
)
from onco_diff.visualizer import render_ascii_tumor_slice, render_oncology_report


def setup_synthetic_brain_domain(dim: int = 30) -> tuple[DTITensorField, np.ndarray]:
    """Create 2D synthetic brain domain with corpus callosum white matter tract and tumor seed."""
    field = DTITensorField((dim, dim), voxel_size_mm=1.0)
    
    y, x = np.ogrid[:dim, :dim]
    center = dim / 2.0

    # Non-brain exterior
    outside = ((x - center) ** 2 + (y - center) ** 2) > (dim * 0.44) ** 2
    field.set_csf_ventricles(outside)

    # Ventricles (center)
    ventricles = ((x - center) ** 2 + ((y - center) * 1.5) ** 2) <= (dim * 0.08) ** 2
    field.set_csf_ventricles(ventricles)

    # Horizontal White Matter Tract (Corpus Callosum along x-axis / axis 1)
    wm_mask = (y >= int(center - 2)) & (y <= int(center + 2)) & ~outside & ~ventricles
    field.set_white_matter_tract(wm_mask, principal_direction=[0.0, 1.0], anisotropy_ratio=6.0)

    # Initial seed in left hemisphere white matter
    c0 = np.zeros((dim, dim), dtype=np.float64)
    seed_mask = ((x - (center - 5)) ** 2 + (y - center) ** 2) <= 2.0 ** 2
    c0[seed_mask] = 0.90

    return field, c0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="onco-diff",
        description="OncoDiff: Glioblastoma Spatiotemporal Reaction-Diffusion & Radiochemotherapy Simulator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Simulate
    sim_p = subparsers.add_parser("simulate", help="Simulate untreated glioma invasion along white matter tracts")
    sim_p.add_argument("--days", type=float, default=60.0, help="Simulation duration in days (default: 60.0)")
    sim_p.add_argument("--diff-wm", type=float, default=0.15, help="White matter diffusion mm^2/day (default: 0.15)")
    sim_p.add_argument("--rho", type=float, default=0.025, help="Proliferation rate /day (default: 0.025)")

    # 2. Stupp Protocol
    stupp_p = subparsers.add_parser("stupp", help="Simulate Stupp protocol (Resection + Radiotherapy 60 Gy + TMZ)")
    stupp_p.add_argument("--days", type=float, default=90.0, help="Simulation duration in days (default: 90.0)")
    stupp_p.add_argument("--mgmt-methylated", action="store_true", default=True, help="Set MGMT promoter methylated")
    stupp_p.add_argument("--mgmt-unmethylated", dest="mgmt_methylated", action="store_false")

    # 3. Prognosis
    prog_p = subparsers.add_parser("prognosis", help="Calculate Swanson D/rho ratio and clinical prognostic score")
    prog_p.add_argument("--diff", type=float, default=0.15, help="Diffusion rate D in mm^2/day (default: 0.15)")
    prog_p.add_argument("--rho", type=float, default=0.025, help="Proliferation rate rho in /day (default: 0.025)")
    prog_p.add_argument("--mgmt-methylated", action="store_true", default=True)
    prog_p.add_argument("--mgmt-unmethylated", dest="mgmt_methylated", action="store_false")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "simulate":
        field, c0 = setup_synthetic_brain_domain(dim=28)
        field.props.diff_white_matter_mm2_day = args.diff_wm
        field.props.proliferation_rate_per_day = args.rho

        solver = GliomaReactionDiffusionSolver(field)
        history = solver.simulate(c0, total_days=args.days, dt_days=0.2, snapshot_interval_days=15.0)

        generator = VirtualMRIGenerator()
        seg = generator.segment_tumor(history.final_density)

        evaluator = TumorBiomarkerEvaluator()
        report = evaluator.evaluate_phenotype(args.diff_wm, args.rho, mgmt_methylated=True)

        print(render_oncology_report(report, seg, history.total_cell_burdens[-1]))
        print("\nFinal Tumor Cell Infiltration Density Map:")
        print(render_ascii_tumor_slice(history.final_density))
        return 0

    elif args.command == "stupp":
        field, c0 = setup_synthetic_brain_domain(dim=28)
        solver = GliomaReactionDiffusionSolver(field)
        
        # Grow for 30 days prior to clinical detection
        history_pre = solver.simulate(c0, total_days=30.0, dt_days=0.2)
        c_at_diagnosis = history_pre.final_density

        # Day 30: Gross Total Resection
        c_post_surgery = RadiotherapyProtocol.apply_surgical_resection(c_at_diagnosis, resection_threshold=0.80)

        # Days 30 to 72: Stupp Radiochemotherapy (30 fractions of 2 Gy + daily TMZ)
        rt = RadiotherapyProtocol()
        tmz_params = TemozolomideParameters(mgmt_methylated=args.mgmt_methylated)
        tmz = ChemotherapyRegimen(tmz_params)

        def stupp_intervention(t: float, c_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            # Apply 2 Gy radiation fraction on weekdays (5 days a week)
            # t in [30, 72]
            day_of_treatment = t - 30.0
            if 0.0 <= day_of_treatment <= 42.0:
                fraction_day = int(day_of_treatment)
                # If weekday and start of day
                if (fraction_day % 7 < 5) and (abs(day_of_treatment - fraction_day) < 0.1):
                    c_curr = rt.apply_fraction(c_curr, dose_gy=2.0)
                # Continuous TMZ daily kill rate
                kill_rate = np.ones_like(c_curr) * tmz.compute_daily_kill_rate(plasma_conc_mg_l=5.0)
                return c_curr, kill_rate
            return c_curr, None

        history_treatment = solver.simulate(
            c_post_surgery,
            total_days=args.days,
            dt_days=0.2,
            intervention_callback=stupp_intervention,
        )

        generator = VirtualMRIGenerator()
        seg = generator.segment_tumor(history_treatment.final_density)

        evaluator = TumorBiomarkerEvaluator()
        report = evaluator.evaluate_phenotype(0.15, 0.025, mgmt_methylated=args.mgmt_methylated)

        print(render_oncology_report(report, seg, history_treatment.total_cell_burdens[-1]))
        print("\nPost-Stupp Protocol Residual Tumor Density:")
        print(render_ascii_tumor_slice(history_treatment.final_density))
        return 0

    elif args.command == "prognosis":
        evaluator = TumorBiomarkerEvaluator()
        report = evaluator.evaluate_phenotype(args.diff, args.rho, mgmt_methylated=args.mgmt_methylated)
        print("=" * 65)
        print("    OncoDiff: Clinical Invasiveness and Survival Prognosis      ")
        print("=" * 65)
        print(f"  Diffusion Coefficient D    : {report.diffusion_coefficient_mm2_day:.4f} mm^2/day")
        print(f"  Proliferation Rate rho     : {report.proliferation_rate_per_day:.4f} /day")
        print(f"  Swanson Invasiveness Ratio : {report.invasiveness_ratio_mm2:.2f} mm^2 (D/rho)")
        print(f"  Invasiveness Phenotype     : {report.tumor_phenotype}")
        print(f"  Recurrence Risk Level      : {report.recurrence_risk_level}")
        print(f"  Predicted Median Survival  : {report.predicted_median_survival_months:.1f} months")
        print("-" * 65)
        print(f"  Guidance: {report.clinical_guidance}")
        print("=" * 65)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
