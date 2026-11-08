"""Unit tests for virtual MRI segmentation and Swanson invasiveness metrics."""

import numpy as np
import pytest
from onco_diff.imaging import (
    VirtualMRIGenerator,
    TumorBiomarkerEvaluator,
)


def test_virtual_mri_segmentation():
    gen = VirtualMRIGenerator(t1gd_threshold=0.80, t2_flair_threshold=0.16)
    
    # Grid with 10 voxels at 0.90, 20 voxels at 0.40, and 50 voxels at 0.05
    c = np.zeros((10, 10))
    c.ravel()[:10] = 0.90
    c.ravel()[10:30] = 0.40
    c.ravel()[30:80] = 0.05

    seg = gen.segment_tumor(c)
    assert np.sum(seg.t1gd_enhancing_core) == 10
    # T2 flair encompasses all c >= 0.16 (both core and edema = 10 + 20 = 30)
    assert np.sum(seg.t2_flair_hyperintense) == 30
    assert np.sum(seg.invisible_infiltrative) == 50


def test_swanson_invasiveness_prognosis():
    evaluator = TumorBiomarkerEvaluator()
    # High D / low rho: Diffuse infiltrative
    rep_diffuse = evaluator.evaluate_phenotype(diffusion_rate=0.20, proliferation_rate=0.02, mgmt_methylated=True)
    assert rep_diffuse.tumor_phenotype == "DIFFUSE_INFILTRATIVE"
    assert rep_diffuse.recurrence_risk_level == "EXTREME"
    assert rep_diffuse.invasiveness_ratio_mm2 == pytest.approx(10.0)

    # Low D / high rho: Nodular
    rep_nodular = evaluator.evaluate_phenotype(diffusion_rate=0.02, proliferation_rate=0.03, mgmt_methylated=False)
    assert rep_nodular.tumor_phenotype == "NODULAR_CIRCUMSCRIBED"
    assert rep_nodular.recurrence_risk_level == "MODERATE"
