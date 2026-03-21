"""Load images and convert to linear XYZ float32 in [0, 1]."""
from __future__ import annotations

import os
import numpy as np
import imageio.v3 as iio
import tifffile

SUPPORTED_RAW = {".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF", ".RW2"}
SUPPORTED_TIFF = {".TIFF", ".TIF"}
SUPPORTED_IMG = {".JPG", ".JPEG", ".PNG"}


def crop_raw_with_flips(xyz_img: np.ndarray, imagesize) -> np.ndarray:
    """Crop the image to the effective area, accounting for sensor flip orientation."""
    flip = imagesize.flip

    match flip:
        case 0:
            left = imagesize.crop_left_margin
            top = imagesize.crop_top_margin
            right = left + imagesize.crop_width
            bottom = top + imagesize.crop_height
            return xyz_img[top:bottom, left:right]
        case 3:
            left = imagesize.raw_width - imagesize.crop_left_margin - imagesize.crop_width
            top = imagesize.raw_height - imagesize.crop_top_margin - imagesize.crop_height
            right = left + imagesize.crop_width
            bottom = top + imagesize.crop_height
            return xyz_img[top:bottom, left:right]
        case 5:
            left = imagesize.crop_top_margin
            top = imagesize.raw_width - imagesize.crop_left_margin - imagesize.crop_width
            right = left + imagesize.crop_height
            bottom = top + imagesize.crop_width
            return xyz_img[top:bottom, left:right]
        case 6:
            left = imagesize.raw_height - imagesize.crop_top_margin - imagesize.crop_height
            top = imagesize.crop_left_margin
            right = left + imagesize.crop_height
            bottom = top + imagesize.crop_width
            return xyz_img[top:bottom, left:right]
        case _:
            raise ValueError(f"Unknown flip: {flip}")


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
        params = rawpy.Params(
            use_camera_wb=True,
            no_auto_bright=True,
            bright=1.0,
            user_sat=None,
            output_color=rawpy.ColorSpace.XYZ,
            output_bps=16,
            gamma=[1, 1],
            dcb_iterations=0,
            dcb_enhance=False,
            median_filter_passes=0,
        )
        rgb = raw.postprocess(params)
        rgb_f = rgb.astype(np.float32) / 65535.0
        rgb_f = crop_raw_with_flips(rgb_f, raw.sizes)

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
