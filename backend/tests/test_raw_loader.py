"""Tests for raw_loader — image loading and RGB→XYZ conversion."""
import io
import os
import numpy as np
import pytest
import tempfile

import tifffile
import imageio.v3 as iio

from backend.raw_loader import load_image_to_xyz, _load_tiff_to_xyz, _load_image_to_xyz_fallback


class TestLoadTIFFToXYZ:
    def test_loads_float32_tiff(self, tmp_path):
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        path = tmp_path / "test.tiff"
        tifffile.imwrite(str(path), arr)
        result = _load_tiff_to_xyz(str(path))
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.float32
        assert (result >= 0).all() and (result <= 1).all()

    def test_tiff_grayscale_expands_to_3ch(self, tmp_path):
        arr = np.random.rand(8, 8).astype(np.float32)
        path = tmp_path / "gray.tiff"
        tifffile.imwrite(str(path), arr)
        result = _load_tiff_to_xyz(str(path))
        assert result.shape == (8, 8, 3)

    def test_tiff_clips_to_01(self, tmp_path):
        arr = np.array([[[2.0, -1.0, 0.5]]], dtype=np.float32)
        path = tmp_path / "clip.tiff"
        tifffile.imwrite(str(path), arr)
        result = _load_tiff_to_xyz(str(path))
        assert (result >= 0).all() and (result <= 1).all()
        assert result[0, 0, 0] == 1.0
        assert result[0, 0, 1] == 0.0


class TestLoadImageToXYZFallback:
    def test_loads_jpeg(self, tmp_path):
        arr = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        path = tmp_path / "test.jpg"
        iio.imwrite(str(path), arr, format="jpeg")
        result = _load_image_to_xyz_fallback(str(path))
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.float32
        assert (result >= 0).all() and (result <= 1).all()

    def test_loads_png(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "test.png"
        iio.imwrite(str(path), arr, format="png")
        result = _load_image_to_xyz_fallback(str(path))
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.float32

    def test_jpeg_grayscale_expands_to_3ch(self, tmp_path):
        arr = (np.random.rand(8, 8) * 255).astype(np.uint8)
        path = tmp_path / "gray.jpg"
        iio.imwrite(str(path), arr, format="jpeg")
        result = _load_image_to_xyz_fallback(str(path))
        assert result.shape == (8, 8, 3)


class TestLoadImageToXYZ:
    def test_dispatches_tiff(self, tmp_path):
        arr = np.random.rand(8, 8, 3).astype(np.float32)
        path = tmp_path / "dispatch.tiff"
        tifffile.imwrite(str(path), arr)
        result = load_image_to_xyz(str(path))
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.float32

    def test_dispatches_jpg(self, tmp_path):
        arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        path = tmp_path / "dispatch.jpg"
        iio.imwrite(str(path), arr, format="jpeg")
        result = load_image_to_xyz(str(path))
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.float32

    def test_rejects_unsupported_extension(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_bytes(b"FAKE")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image_to_xyz(str(path))
