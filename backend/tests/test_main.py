"""Integration tests for the FastAPI backend endpoints."""
import base64
import io

import httpx
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok_and_version(self):
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert isinstance(data["version"], str)


class TestConvertEndpoint:
    def _synthetic_tiff_bytes(self) -> bytes:
        import numpy as np
        import tifffile

        arr = np.zeros((1, 1, 3), dtype=np.float32) + 0.5
        buf = io.BytesIO()
        tifffile.imwrite(buf, arr)
        return buf.getvalue()

    def _dummy_cube_content(self) -> bytes:
        return b"""TITLE "Test LUT"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""

    def test_convert_missing_image_returns_422(self):
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        files = {"luts": ("lut.cube", self._dummy_cube_content(), "text/plain")}
        response = client.post("/convert", files=files)

        assert response.status_code == 422

    def test_convert_missing_luts_returns_422(self):
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        files = {"image": ("test.tiff", b"FAKE TIFF DATA", "image/tiff")}
        response = client.post("/convert", files=files)

        assert response.status_code == 422

    def test_convert_missing_both_returns_422(self):
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/convert")

        assert response.status_code == 422

    def test_convert_with_image_and_luts_returns_results(self):
        from backend.main import app

        client = TestClient(app, raise_server_exceptions=False)

        img_bytes = self._synthetic_tiff_bytes()
        lut_bytes = self._dummy_cube_content()

        files = [
            ("image", ("input.tiff", img_bytes, "image/tiff")),
            ("luts", ("fujifilm_std.cube", lut_bytes, "text/plain")),
            ("luts", ("fujifilm_pro.cube", lut_bytes, "text/plain")),
        ]
        response = client.post("/convert", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2

        for result in data["results"]:
            assert "lut_name" in result
            assert "image_base64_jpeg" in result
            decoded = base64.b64decode(result["image_base64_jpeg"])
            assert decoded.startswith(b"\xff\xd8"), "Expected JPEG magic bytes"


class TestTorchFree:
    def test_no_torch_imports_in_backend(self):
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "-c", "import backend; import backend.main; import backend.pipeline"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
