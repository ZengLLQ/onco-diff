"""Unit tests for DTI tensor field and Fractional Anisotropy."""

import numpy as np
import pytest
from onco_diff.pde import DTITensorField, TissueProperties


def test_dti_isotropic_initialization():
    field = DTITensorField((16, 16), voxel_size_mm=1.0)
    assert field.d_tensor.shape == (2, 2, 16, 16)
    # Off-diagonals must be zero for isotropic gray matter
    np.testing.assert_allclose(field.d_tensor[0, 1], 0.0)
    np.testing.assert_allclose(field.d_tensor[1, 0], 0.0)
    # Diagonals equal gray matter diffusion rate
    np.testing.assert_allclose(field.d_tensor[0, 0], field.props.diff_gray_matter_mm2_day)


def test_dti_white_matter_tract_anisotropy():
    field = DTITensorField((20, 20), voxel_size_mm=1.0)
    tract_mask = np.zeros((20, 20), dtype=bool)
    tract_mask[8:12, :] = True

    field.set_white_matter_tract(tract_mask, principal_direction=[1.0, 0.0], anisotropy_ratio=5.0)
    
    # Diffusion along tract (x-axis, [0, 0]) should be 5x transverse diffusion ([1, 1])
    d_long = field.d_tensor[0, 0, tract_mask]
    d_trans = field.d_tensor[1, 1, tract_mask]
    assert np.all(d_long > d_trans)

    # Compute FA
    fa = field.compute_fractional_anisotropy()
    assert np.all(fa >= 0.0)
    assert np.all(fa <= 1.0)
    assert np.mean(fa[tract_mask]) > np.mean(fa[~tract_mask])


def test_csf_ventricles_zero_diffusion():
    field = DTITensorField((10, 10))
    vent_mask = np.zeros((10, 10), dtype=bool)
    vent_mask[4:6, 4:6] = True
    field.set_csf_ventricles(vent_mask)

    assert np.all(field.d_tensor[:, :, vent_mask] == 0.0)
