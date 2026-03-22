"""Load images and convert to linear XYZ float32 in [0, 1]."""
from __future__ import annotations

import os
import numpy as np
import imageio.v3 as iio
import tifffile

SUPPORTED_RAW = {".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF", ".RW2"}
SUPPORTED_TIFF = {".TIFF", ".TIF"}
SUPPORTED_IMG = {".JPG", ".JPEG", ".PNG"}


def crop_raw_with_flips(xyz_img: np.ndarray, imagesize, shrink: int = 1) -> np.ndarray:
    """Crop the image to the effective area, accounting for sensor flip orientation and shrink factor."""
    flip = imagesize.flip
    
    # Scale margins and dimensions by the shrink factor
    s_left = imagesize.crop_left_margin // shrink
    s_top = imagesize.crop_top_margin // shrink
    s_width = imagesize.crop_width // shrink
    s_height = imagesize.crop_height // shrink
    
    # Scale total dimensions if needed (for flips that use raw_width/height)
    s_raw_w = imagesize.raw_width // shrink
    s_raw_h = imagesize.raw_height // shrink

    match flip:
        case 0:
            left = s_left
            top = s_top
            right = left + s_width
            bottom = top + s_height
            return xyz_img[top:bottom, left:right]
        case 3:
            left = s_raw_w - s_left - s_width
            top = s_raw_h - s_top - s_height
            right = left + s_width
            bottom = top + s_height
            return xyz_img[top:bottom, left:right]
        case 5:
            left = s_top
            top = s_raw_w - s_left - s_width
            right = left + s_height
            bottom = top + s_width
            return xyz_img[top:bottom, left:right]
        case 6:
            left = s_raw_h - s_top - s_height
            top = s_left
            right = left + s_height
            bottom = top + s_width
            return xyz_img[top:bottom, left:right]
        case _:
            raise ValueError(f"Unknown flip: {flip}")


def load_image_to_xyz(path: str, shrink: int = 1) -> np.ndarray:
    ext = os.path.splitext(path)[1].upper()
    if ext in SUPPORTED_RAW:
        return _load_raw_to_xyz(path, shrink=shrink)
    elif ext in SUPPORTED_TIFF:
        return _load_tiff_to_xyz(path)
    elif ext in SUPPORTED_IMG:
        return _load_image_to_xyz_fallback(path)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


def _load_raw_to_xyz(path: str, shrink: int = 1) -> np.ndarray:
    import rawpy
    
    # rawpy doesn't have a 'shrink' parameter in Params. 
    # 'half_size=True' provides a 2x downsampling which is fast.
    use_half = shrink >= 2
    
    with rawpy.imread(path) as raw:
        params = rawpy.Params(
            use_camera_wb=True,
            no_auto_bright=True,
            bright=1.0,
            output_color=rawpy.ColorSpace.XYZ,
            output_bps=16,
            gamma=[1, 1],
            half_size=use_half,
        )
        rgb = raw.postprocess(params)
        rgb_f = rgb.astype(np.float32) / 65535.0
        
        # The effective shrink factor for coordinate scaling is 2 if half_size was used
        eff_shrink = 2 if use_half else 1
        xyz_cropped = crop_raw_with_flips(rgb_f, raw.sizes, shrink=eff_shrink)

    xyz = np.clip(xyz_cropped, 0.0, 1.0)
    return xyz.astype(np.float32)


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
