"""Unit tests for Linear-Quadratic radiotherapy and surgical resection."""

import numpy as np
import pytest
from onco_diff.therapies import (
    RadiosensitivityParameters,
    RadiotherapyProtocol,
)


def test_linear_quadratic_surviving_fraction():
    params = RadiosensitivityParameters(alpha_per_gy=0.06, beta_per_gy2=0.006)
    rt = RadiotherapyProtocol(params)

    # 2 Gy fraction: S = exp(-0.06*2 - 0.006*4) = exp(-0.12 - 0.024) = exp(-0.144) ~ 0.8659
    sf = rt.compute_surviving_fraction(dose_gy=2.0)
    assert sf == pytest.approx(np.exp(-0.144), rel=1e-3)

    # Hypoxic survival is higher (radioresistant)
    sf_hypoxic = rt.compute_surviving_fraction(dose_gy=2.0, is_hypoxic_mask=np.array([True]))
    assert sf_hypoxic[0] > sf


def test_surgical_resection():
    c = np.zeros((12, 12))
    c[4:8, 4:8] = 0.90  # Dense macroscopic tumor
    c[2:10, 2:10] += 0.20 # Infiltrative margin

    c_post = RadiotherapyProtocol.apply_surgical_resection(c, resection_threshold=0.80, margin_voxels=0)
    assert np.all(c_post[4:8, 4:8] == 0.0)
    # Surrounding lower density cells remain
    assert np.sum(c_post) > 0.0
