"""Unit tests for Fisher-KPP reaction-diffusion solver."""

import numpy as np
import pytest
from onco_diff.pde import DTITensorField, GliomaReactionDiffusionSolver


def test_fisher_kpp_growth_and_bounds():
    field = DTITensorField((16, 16), voxel_size_mm=1.0)
    solver = GliomaReactionDiffusionSolver(field)

    c0 = np.zeros((16, 16))
    c0[7:9, 7:9] = 0.5  # Small seed

    history = solver.simulate(c0, total_days=15.0, dt_days=0.2)
    final_c = history.final_density

    # Tumor must proliferate
    assert history.total_cell_burdens[-1] > history.total_cell_burdens[0]
    # Cells must respect carrying capacity bounds [0, 1]
    assert np.all(final_c >= 0.0)
    assert np.all(final_c <= 1.0)


def test_csf_containment():
    field = DTITensorField((16, 16), voxel_size_mm=1.0)
    # Block right half with CSF
    csf_mask = np.zeros((16, 16), dtype=bool)
    csf_mask[:, 10:] = True
    field.set_csf_ventricles(csf_mask)

    solver = GliomaReactionDiffusionSolver(field)
    c0 = np.zeros((16, 16))
    c0[7:9, 4:6] = 0.8  # Left side seed

    history = solver.simulate(c0, total_days=20.0, dt_days=0.2)
    final_c = history.final_density

    # No tumor should cross into CSF
    assert np.all(final_c[csf_mask] == 0.0)
