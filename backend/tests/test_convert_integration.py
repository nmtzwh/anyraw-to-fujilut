"""Integration tests: end-to-end /convert pipeline with real TIFF + real .cube LUT."""
import base64
import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from backend.main import app


def _synthetic_tiff_bytes() -> bytes:
    arr = np.random.rand(8, 8, 3).astype(np.float32)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def _identity_cube_bytes(n: int = 2) -> bytes:
    vals = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                vals.append(f"{i/(n-1):.6f} {j/(n-1):.6f} {k/(n-1):.6f}")
    return f"LUT_3D_SIZE {n}\n" + "\n".join(vals) + "\n"


class TestConvertIntegration:
    def test_convert_returns_valid_jpegs_for_each_lut(self):
        client = TestClient(app, raise_server_exceptions=False)
        files = [
            ("image", ("test.tiff", _synthetic_tiff_bytes(), "image/tiff")),
            ("luts", ("identity.cube", _identity_cube_bytes(2), "text/plain")),
            ("luts", ("fujifilm_std.cube", _identity_cube_bytes(2), "text/plain")),
        ]
        resp = client.post("/convert", files=files)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["results"]) == 2
        for r in data["results"]:
            jpeg_bytes = base64.b64decode(r["image_base64_jpeg"])
            assert jpeg_bytes[:2] == b"\xff\xd8", "Not a JPEG"

    def test_convert_lut_name_matches_filename(self):
        client = TestClient(app, raise_server_exceptions=False)
        files = [
            ("image", ("test.tiff", _synthetic_tiff_bytes(), "image/tiff")),
            ("luts", ("my_custom_lut.cube", _identity_cube_bytes(2), "text/plain")),
        ]
        resp = client.post("/convert", files=files)
        assert resp.status_code == 200
        assert resp.json()["results"][0]["lut_name"] == "my_custom_lut"

    def test_convert_preserves_original_extension_in_temp_path(self):
        client = TestClient(app, raise_server_exceptions=False)
        files = [
            ("image", ("photo.ARW", _synthetic_tiff_bytes(), "image/tiff")),
            ("luts", ("lut.cube", _identity_cube_bytes(2), "text/plain")),
        ]
        resp = client.post("/convert", files=files)
        assert resp.status_code == 200

    def test_convert_jpeg_is_reasonable_size(self):
        client = TestClient(app, raise_server_exceptions=False)
        arr = np.random.rand(128, 128, 3).astype(np.float32)
        buf = io.BytesIO()
        tifffile.imwrite(buf, arr)
        files = [
            ("image", ("test.tiff", buf.getvalue(), "image/tiff")),
            ("luts", ("lut.cube", _identity_cube_bytes(4), "text/plain")),
        ]
        resp = client.post("/convert", files=files)
        assert resp.status_code == 200
        jpeg_bytes = base64.b64decode(resp.json()["results"][0]["image_base64_jpeg"])
        assert 100 < len(jpeg_bytes) < 1_000_000, "JPEG size unreasonable"
