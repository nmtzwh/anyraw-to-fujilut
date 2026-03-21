"""
CPU-based, pure-Python backend pipeline helpers for converting raw XYZ data.

This module provides a small, deterministic API suitable for unit testing
and potential later integration via a FastAPI service. It deliberately avoids
external dependencies beyond NumPy and a light 3D LUT implementation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def load_raw(path: str) -> np.ndarray:
    """Load an XYZ image from disk as a float32 array in [0, 1].

    The function accepts common NumPy-based formats (.npy or .npz). The loaded
    data is coerced to float32 and clipped to [0, 1]. Cropping or flipping is
    left to the caller or the data producer; this function only loads the array.

    Expected shape: (..., 3) where the last dimension represents the X, Y, Z
    channels. The result is a numpy array with dtype float32.
    """

    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".npz"):
        npz = np.load(path)
        # Load the first array stored in the archive
        first_key = next(iter(npz.files))
        data = npz[first_key]
    else:
        raise ValueError(f"Unsupported raw format for path: {path}")

    data = np.asarray(data, dtype=np.float32)
    # Normalize into [0, 1] if needed; tests will provide in-range data
    data = np.clip(data, 0.0, 1.0)
    # Ensure last dimension is 3 (XYZ)
    if data.shape[-1] != 3:
        raise ValueError("Loaded data must have 3 channels in the last dimension")
    return data


def apply_flog2_curve(xyz: np.ndarray) -> np.ndarray:
    """Apply the Fujifilm F-Log2 OETF (Opto-Electronic Transfer Function).

    Based on F-Log2 Data Sheet Ver.1.1.
    Uses a piecewise function: logarithmic shoulder for values above cut1,
    linear toe for values below.
    """
    a = 5.555556
    b = 0.064829
    c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    cut1 = 0.000889

    v = np.asarray(xyz, dtype=np.float32)
    v = np.clip(v, 0.0, 1.0)
    out = np.zeros_like(v)

    mask_log = v >= cut1
    mask_linear = v < cut1

    out[mask_log] = c * np.log10(a * v[mask_log] + b) + d
    out[mask_linear] = e * v[mask_linear] + f

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def xyz_to_rec2020(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to Rec.2020-like linear space using a fixed 3x3 matrix.

    This is a pure, deterministic affine transform. The exact matrix is chosen
    to be a reasonable approximation for unit tests and educational purposes.
    The function supports an array of shape (..., 3).
    """

    M = np.array([
        [1.71665119, -0.35567078, -0.25336658],
        [-0.66668439, 1.616507, 0.0157685],
        [0.0176397, -0.0427706, 0.94210345],
    ], dtype=np.float32)

    a = np.asarray(xyz, dtype=np.float32)
    if a.shape[-1] != 3:
        raise ValueError("xyz_to_rec2020 expects input with 3 channels in the last dimension")

    flat = a.reshape(-1, 3)
    transformed = flat @ M.T  # shape (N, 3)
    out = transformed.reshape(a.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_exposure_gain(xyz_image: np.ndarray, target_grey: float = 0.18, ev_offset: float = 0.0) -> float:
    """Calculate gain to normalize image exposure based on geometric mean of Y channel.

    Extracts the Y (luminance) channel from XYZ data, computes the geometric mean,
    and returns a gain factor to normalize exposure to a target grey level.
    """
    Y_channel = xyz_image[::10, ::10, 1]
    epsilon = 1e-6
    log_Y = np.log(np.maximum(Y_channel, epsilon))
    geom_mean = np.exp(np.mean(log_Y))
    safe_mean = max(geom_mean, epsilon)
    gain = target_grey / safe_mean
    return gain * (2.0 ** ev_offset)


def apply_lut(flog2: np.ndarray, lut_table: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to a per-pixel 3-channel input.

    Parameters:
      flog2: numpy array of shape (H, W, 3) with values in [0, 1]. These are
             treated as normalized coordinates into the LUT.
      lut_table: numpy array with shape (N, N, N, 3). The LUT values are in [0, 1].

    Returns:
      numpy array of shape (H, W, 3) with values in [0, 1].
    """

    fl = np.asarray(flog2, dtype=np.float32)
    if fl.shape[-1] != 3:
        raise ValueError("flog2 input must have 3 channels in the last dimension")
    lut = np.asarray(lut_table, dtype=np.float32)
    if lut.ndim != 4 or lut.shape[-1] != 3:
        raise ValueError("lut_table must have shape (N,N,N,3)")
    N = lut.shape[0]
    if not (0.0 <= fl).all() or not (fl <= 1.0).all():
        fl = np.clip(fl, 0.0, 1.0)

    # Scale coordinates to LUT grid indexes [0, N-1]
    coord = fl * (N - 1)
    i0 = np.floor(coord[..., 0]).astype(int)
    j0 = np.floor(coord[..., 1]).astype(int)
    k0 = np.floor(coord[..., 2]).astype(int)
    i1 = np.minimum(i0 + 1, N - 1)
    j1 = np.minimum(j0 + 1, N - 1)
    k1 = np.minimum(k0 + 1, N - 1)
    fx = coord[..., 0] - i0
    fy = coord[..., 1] - j0
    fz = coord[..., 2] - k0

    # Gather corners
    c000 = lut[i0, j0, k0]
    c100 = lut[i1, j0, k0]
    c010 = lut[i0, j1, k0]
    c110 = lut[i1, j1, k0]
    c001 = lut[i0, j0, k1]
    c101 = lut[i1, j0, k1]
    c011 = lut[i0, j1, k1]
    c111 = lut[i1, j1, k1]

    def lerp(a, b, t):
        return a + (b - a) * t[..., None]

    c00 = lerp(c000, c100, fx)
    c01 = lerp(c001, c101, fx)
    c10 = lerp(c010, c110, fx)
    c11 = lerp(c011, c111, fx)
    c0 = lerp(c00, c10, fy)
    c1 = lerp(c01, c11, fy)
    c = lerp(c0, c1, fz)

    return np.clip(c, 0.0, 1.0).astype(np.float32)


def prewarm() -> None:
    """Optional warm-up to improve first-call performance.

    This function does a tiny no-op computation to trigger any one-time
    initialization that might be expensive on first use.
    """
    # Tiny no-op call to avoid import-time surprises
    _ = apply_flog2_curve(np.zeros((1, 1, 3), dtype=np.float32))


__all__ = [
    "load_raw",
    "apply_flog2_curve",
    "xyz_to_rec2020",
    "apply_lut",
    "get_exposure_gain",
    "prewarm",
]
