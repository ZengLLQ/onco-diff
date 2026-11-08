"""Unit tests for Temozolomide alkylating chemotherapy."""

import pytest
import numpy as np
from onco_diff.therapies import TemozolomideParameters, ChemotherapyRegimen


def test_chemotherapy_mgmt_methylation_sensitivity():
    p_methyl = TemozolomideParameters(mgmt_methylated=True)
    p_unmethyl = TemozolomideParameters(mgmt_methylated=False)

    tmz_methyl = ChemotherapyRegimen(p_methyl)
    tmz_unmethyl = ChemotherapyRegimen(p_unmethyl)

    kill_methyl = tmz_methyl.compute_daily_kill_rate(plasma_conc_mg_l=4.0)
    kill_unmethyl = tmz_unmethyl.compute_daily_kill_rate(plasma_conc_mg_l=4.0)

    # Methylated tumors suffer significantly higher cytotoxicity
    assert kill_methyl > kill_unmethyl
    assert kill_unmethyl == pytest.approx(kill_methyl * p_unmethyl.mgmt_resistance_factor, rel=1e-3)


def test_chemotherapy_apply_kill():
    tmz = ChemotherapyRegimen()
    c = np.ones((8, 8)) * 0.5
    c_after = tmz.apply_chemotherapy_kill(c, dt_days=1.0)
    assert np.all(c_after < c)
