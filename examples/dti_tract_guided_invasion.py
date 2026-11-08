#!/usr/bin/env python3
"""Clinical Example: DTI Anisotropic White Matter Guided Tumor Invasion."""

import numpy as np
from onco_diff.pde import DTITensorField, GliomaReactionDiffusionSolver
from onco_diff.visualizer import render_ascii_tumor_slice


def run_dti_tract_guided_invasion():
    print("Executing DTI White Matter Fiber Tract Invasion Simulation...")

    dim = 26
    field = DTITensorField((dim, dim), voxel_size_mm=1.0)
    
    # Brain boundary
    y, x = np.ogrid[:dim, :dim]
    outside = ((x - dim/2)**2 + (y - dim/2)**2) > (dim * 0.45)**2
    field.set_csf_ventricles(outside)

    # Horizontal White Matter Tract (Corpus Callosum along x-axis / axis 1)
    center = dim / 2.0
    tract_mask = (y >= int(center - 1)) & (y <= int(center + 1)) & ~outside
    field.set_white_matter_tract(tract_mask, principal_direction=[0.0, 1.0], anisotropy_ratio=8.0)

    fa_map = field.compute_fractional_anisotropy()
    print(f"  Peak White Matter Fractional Anisotropy (FA): {np.max(fa_map):.3f}")

    # Seed tumor at center of tract
    c0 = np.zeros((dim, dim), dtype=np.float64)
    c0[int(center), int(center)] = 0.95

    solver = GliomaReactionDiffusionSolver(field)
    history = solver.simulate(c0, total_days=35.0, dt_days=0.15)

    final_c = history.final_density
    
    # Assess horizontal spread vs vertical spread along tract
    horiz_span = np.sum(final_c[int(center), :] > 0.05)
    vert_span = np.sum(final_c[:, int(center)] > 0.05)
    
    print("=" * 65)
    print("       Anisotropic Glioma Spread Along Axonal Tracts            ")
    print("=" * 65)
    print(f"  Horizontal (Axonal) Spread Span : {horiz_span} voxels")
    print(f"  Vertical (Transverse) Spread    : {vert_span} voxels")
    print(f"  Anisotropic Elongation Ratio    : {horiz_span / max(1, vert_span):.2f}x")
    print("-" * 65)
    print("Tumor Cell Distribution on Corpus Callosum:")
    print(render_ascii_tumor_slice(final_c, height=10, width=30))
    print("=" * 65)

    assert horiz_span > vert_span
    print("SUCCESS: DTI tract guided invasion verified.")


if __name__ == "__main__":
    run_dti_tract_guided_invasion()
