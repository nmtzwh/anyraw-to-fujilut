"""Load images and convert to linear XYZ float32 in [0, 1]."""
from __future__ import annotations

import os
import numpy as np
import imageio.v3 as iio
import tifffile

SUPPORTED_RAW = {".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF", ".RW2"}
SUPPORTED_TIFF = {".TIFF", ".TIF"}
SUPPORTED_IMG = {".JPG", ".JPEG", ".PNG"}


def load_image_to_xyz(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].upper()
    if ext in SUPPORTED_RAW:
        return _load_raw_to_xyz(path)
    elif ext in SUPPORTED_TIFF:
        return _load_tiff_to_xyz(path)
    elif ext in SUPPORTED_IMG:
        return _load_image_to_xyz_fallback(path)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


def _load_raw_to_xyz(path: str) -> np.ndarray:
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            user_black=None,
            user_sat=None,
            dcb_iteration=False,
            dcb_enhance=False,
            faa_reduction=False,
            median_filter_passes=0,
            erfc_reduce_noise=False,
        )
        rgb_f = rgb.astype(np.float32) / 65535.0
        rgb_f = np.clip(rgb_f, 0.0, 1.0)

    M_rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1374375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )

    H, W, _ = rgb_f.shape
    xyz = rgb_f.reshape(-1, 3) @ M_rgb_to_xyz.T
    max_val = xyz.max()
    if max_val > 0:
        xyz = xyz / max_val
    xyz = np.clip(xyz, 0.0, 1.0)
    return xyz.reshape(H, W, 3).astype(np.float32)


def _load_tiff_to_xyz(path: str) -> np.ndarray:
    data = tifffile.imread(path).astype(np.float32)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3 and data.shape[2] == 4:
        data = data[:, :, :3]
    data = np.clip(data, 0.0, 1.0)
    return data.astype(np.float32)


def _load_image_to_xyz_fallback(path: str) -> np.ndarray:
    img = iio.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.where(
        img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4
    )
    img = np.clip(img, 0.0, 1.0)
    M_srgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1374375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    H, W, _ = img.shape
    xyz = img.reshape(-1, 3) @ M_srgb_to_xyz.T
    xyz = np.clip(xyz, 0.0, 1.0)
    return xyz.reshape(H, W, 3).astype(np.float32)


__all__ = ["load_image_to_xyz"]
