import numpy as np
import os
import tempfile

import pytest

from backend.pipeline import (
    load_raw,
    apply_flog2_curve,
    xyz_to_rec2020,
    apply_lut,
)


def test_load_raw_py_loader(tmp_path):
    # Prepare a tiny XYZ array and save as .npy
    arr = np.array([[[0.0, 0.25, 0.5], [0.75, 1.0, 0.0]]], dtype=np.float32)  # shape (1,2,3)
    path = tmp_path / "data.npy"
    np.save(path, arr)

    loaded = load_raw(str(path))
    assert loaded.shape == arr.shape
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, arr)


def test_apply_flog2_curve_shape_and_range():
    xyz = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)  # shape (1,1,3)
    out = apply_flog2_curve(xyz)
    assert out.shape == xyz.shape
    assert np.all((out >= 0.0) & (out <= 1.0))
    # Check endpoints map correctly for the chosen curve
    assert pytest.approx(out[0, 0, 0], rel=1e-5, abs=1e-6) == 0.0
    assert pytest.approx(out[0, 0, 2], rel=1e-5, abs=1e-6) == 1.0


def test_xyz_to_rec2020_matrix_application():
    # A simple known input, size 1x1
    xyz = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    out = xyz_to_rec2020(xyz)
    # Verify shape and range
    assert out.shape == xyz.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Compute expected via the same matrix to ensure determinism
    M = np.array([
        [1.71665119, -0.35567078, -0.25336658],
        [-0.66668439, 1.616507, 0.0157685],
        [0.0176397, -0.0427706, 0.94210345],
    ], dtype=np.float32)
    expected = (np.array([0.5, 0.5, 0.5], dtype=np.float32) @ M.T)
    expected = np.clip(expected, 0.0, 1.0)
    assert np.allclose(out.reshape(3), expected)


def test_apply_flog2_curve_out_of_range_clamps():
    xyz_neg = np.array([[[-0.5, -1.0, 2.0]]], dtype=np.float32)
    out = apply_flog2_curve(xyz_neg)
    assert out.shape == xyz_neg.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_end_to_end_synthetic():
    from backend.pipeline import load_raw, apply_flog2_curve, apply_lut
    xyz = np.array([[[0.1, 0.5, 0.9], [0.3, 0.7, 1.0]]], dtype=np.float32)
    flog2 = apply_flog2_curve(xyz)
    assert flog2.shape == xyz.shape
    assert (flog2 >= 0.0).all() and (flog2 <= 1.0).all()
    N = 4
    lut = np.zeros((N, N, N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                lut[i, j, k] = np.array([i / (N - 1), j / (N - 1), k / (N - 1)], dtype=np.float32)
    graded = apply_lut(flog2, lut)
    assert graded.shape == xyz.shape
    assert graded.dtype == np.float32
    assert (graded >= 0.0).all() and (graded <= 1.0).all()


def test_apply_lut_basic_identity_mapping():
    # Create a tiny LUT where the output is exactly the coordinates of the input
    N = 4
    lut = np.zeros((N, N, N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                lut[i, j, k] = np.array([i / (N - 1), j / (N - 1), k / (N - 1)], dtype=np.float32)

    flog2 = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    # Two pixels: one at 0,0,0 and one at 1,1,1
    out = __import__('backend.pipeline', fromlist=['apply_lut']).apply_lut(flog2, lut)
    assert out.shape == flog2.shape
    assert np.allclose(out[0, 0, :], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(out[0, 1, :], np.array([1.0, 1.0, 1.0], dtype=np.float32))
