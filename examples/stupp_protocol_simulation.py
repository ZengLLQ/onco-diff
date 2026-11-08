#!/usr/bin/env python3
"""Clinical Example: Standard-of-Care Stupp Protocol for Glioblastoma."""

import numpy as np
from onco_diff.pde import DTITensorField, GliomaReactionDiffusionSolver
from onco_diff.therapies import RadiotherapyProtocol, ChemotherapyRegimen, TemozolomideParameters
from onco_diff.imaging import VirtualMRIGenerator, TumorBiomarkerEvaluator
from onco_diff.visualizer import render_oncology_report


def run_stupp_protocol():
    print("Executing Clinical Glioblastoma Stupp Protocol Simulation...")

    dim = 24
    field = DTITensorField((dim, dim), voxel_size_mm=1.0)
    
    # Brain boundary
    y, x = np.ogrid[:dim, :dim]
    outside = ((x - dim/2)**2 + (y - dim/2)**2) > (dim * 0.44)**2
    field.set_csf_ventricles(outside)

    # Initial tumor nidus
    c0 = np.zeros((dim, dim), dtype=np.float64)
    c0[int(dim/2)-2:int(dim/2)+2, int(dim/2)-2:int(dim/2)+2] = 0.85

    solver = GliomaReactionDiffusionSolver(field)

    # 1. Untreated growth (20 days)
    hist_pre = solver.simulate(c0, total_days=20.0, dt_days=0.2)
    c_pre = hist_pre.final_density

    # 2. Surgical Resection
    c_post_surg = RadiotherapyProtocol.apply_surgical_resection(c_pre, resection_threshold=0.75, margin_voxels=1)
    
    # 3. Chemoradiotherapy (MGMT Methylated)
    rt = RadiotherapyProtocol()
    tmz = ChemotherapyRegimen(TemozolomideParameters(mgmt_methylated=True))

    def treatment_callback(t: float, c_curr: np.ndarray):
        # 2 Gy fraction delivered every weekday
        day_int = int(t)
        if (day_int % 7 < 5) and (abs(t - day_int) < 0.15):
            c_curr = rt.apply_fraction(c_curr, dose_gy=2.0)
        kill = np.ones_like(c_curr) * tmz.compute_daily_kill_rate(plasma_conc_mg_l=5.0)
        return c_curr, kill

    hist_tx = solver.simulate(c_post_surg, total_days=40.0, dt_days=0.2, intervention_callback=treatment_callback)

    gen = VirtualMRIGenerator()
    seg = gen.segment_tumor(hist_tx.final_density)
    evaluator = TumorBiomarkerEvaluator()
    report = evaluator.evaluate_phenotype(0.15, 0.025, mgmt_methylated=True)

    print(render_oncology_report(report, seg, hist_tx.total_cell_burdens[-1]))

    assert hist_tx.total_cell_burdens[-1] < hist_pre.total_cell_burdens[-1]
    print("SUCCESS: Stupp Protocol simulation completed without errors.")


if __name__ == "__main__":
    run_stupp_protocol()
