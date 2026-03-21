# Pipeline Integration Plan — backend/main.py `/convert`

> **For Claude:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the `backend/pipeline.py` CPU pipeline into `backend/main.py`'s `POST /convert` endpoint so it processes real uploaded images and LUTs, returning graded JPEGs instead of a placeholder.

**Architecture:** The endpoint will chain: (1) load & demosaic RAW via `rawpy`, (2) RGB→XYZ conversion, (3) apply F-Log2 curve, (4) XYZ→Rec2020 transform, (5) parse `.cube` LUT files into `(N,N,N,3)` tables, (6) apply LUT via trilinear interpolation, (7) encode graded float32→uint8→JPEG→base64. Each LUT produces one result.

**Tech Stack:** `rawpy`, `imageio`, `tifffile`, `numpy`, `colour-science`, `scipy`.

---

## Step 1: Add `backend/cube_parser.py` — parse `.cube` files

**Files:**
- Create: `backend/cube_parser.py`
- Test: `backend/tests/test_cube_parser.py`

**Step 1a: Write the failing test**

```python
# backend/tests/test_cube_parser.py
import numpy as np, pytest, io
from backend.cube_parser import parse_cube

def test_parse_cube_basic():
    content = b"""TITLE "Test LUT"
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
    lut = parse_cube(io.BytesIO(content))
    assert lut.shape == (2, 2, 2, 3)
    assert lut.dtype == np.float32
    assert np.allclose(lut[0, 0, 0], [0.0, 0.0, 0.0])
    assert np.allclose(lut[1, 1, 1], [1.0, 1.0, 1.0])

def test_parse_cube_33():
    # 33-point LUT: 33^3 = 35937 values
    content = b"LUT_3D_SIZE 33\n" + b"0.0 0.0 0.0\n" * 35937
    lut = parse_cube(io.BytesIO(content))
    assert lut.shape == (33, 33, 33, 3)
```

**Step 1b: Run test to verify it fails**
`pytest backend/tests/test_cube_parser.py -v` → FAIL (module not found)

**Step 1c: Write implementation**

```python
# backend/cube_parser.py
"""Parse .cube 3D LUT files into numpy (N,N,N,3) float32 arrays."""
from __future__ import annotations
import numpy as np
import io

def parse_cube(fp: io.BytesIO | io.RawIOBase) -> np.ndarray:
    """Parse a .cube file and return a (N,N,N,3) float32 LUT table.
    
    Supports LUT_3D_SIZE N (required), DOMAIN_MIN/MAX (ignored, assumed [0,1]),
    and blank/comment lines. Values are assumed in [0, 1].
    """
    lines = []
    for raw_line in fp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TITLE") or line.startswith("LUT_1D_SIZE") or line.startswith("DOMAIN_"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                lines.append([r, g, b])
            except ValueError:
                continue  # skip malformed lines

    n = round(len(lines) ** (1.0 / 3.0))
    expected = n ** 3
    if len(lines) != expected:
        raise ValueError(f"LUT_3D_SIZE implied N={n} but got {len(lines)} values (expected {expected})")

    arr = np.array(lines, dtype=np.float32).reshape((n, n, n, 3))
    return arr

__all__ = ["parse_cube"]
```

**Step 1d: Run tests to verify they pass**
`pytest backend/tests/test_cube_parser.py -v` → PASS

---

## Step 2: Add `backend/raw_loader.py` — load RAW via rawpy + RGB→XYZ

**Files:**
- Create: `backend/raw_loader.py`
- Test: `backend/tests/test_raw_loader.py`

**Step 2a: Write the failing test**

```python
# backend/tests/test_raw_loader.py
import numpy as np, pytest, tempfile, os
from backend.raw_loader import load_image_to_xyz

def test_load_tiff_as_xyz(tmp_path):
    # Save a synthetic float32 TIFF and load it
    arr = np.random.rand(16, 16, 3).astype(np.float32)
    path = tmp_path / "test.tiff"
    import imageio.v3 as iio
    # imageio v3 expects uint8 for TIFF; use tifffile for float32
    import tifffile
    tifffile.imwrite(str(path), arr)
    result = load_image_to_xyz(str(path))
    assert result.shape == arr.shape
    assert result.dtype == np.float32
    assert (result >= 0).all() and (result <= 1).all()
```

**Step 2b: Run test to verify it fails**
`pytest backend/tests/test_raw_loader.py -v` → FAIL (module not found)

**Step 2c: Write implementation**

```python
# backend/raw_loader.py
"""Load images (RAW via rawpy, or TIFF/JPEG via imageio) and convert to XYZ float32 [0,1]."""
from __future__ import annotations
import os
import numpy as np
import imageio.v3 as iio
import tifffile

def load_image_to_xyz(path: str) -> np.ndarray:
    """Load an image file and return linear XYZ data in [0, 1].

    Supported extensions:
      - .ARW .DNG .NEF .CR2 .CR3 .RAF .ORF .RW2 — via rawpy (demosaiced linear RGB → XYZ)
      - .TIFF .TIF — via tifffile (float32 assumed already linear)
      - .JPG .JPEG .PNG — via imageio (sRGB assumed → linearize → XYZ)

    Returns:
      np.ndarray of shape (H, W, 3), dtype float32, values in [0, 1].
    """
    ext = os.path.splitext(path)[1].upper()

    if ext in (".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF", ".RW2"):
        return _load_raw_to_xyz(path)
    elif ext in (".TIFF", ".TIF"):
        return _load_tiff_to_xyz(path)
    elif ext in (".JPG", ".JPEG", ".PNG"):
        return _load_image_to_xyz_fallback(path)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


def _load_raw_to_xyz(path: str) -> np.ndarray:
    """Demosaice a RAW file via rawpy, convert linear RGB to XYZ, return in [0,1]."""
    import rawpy

    with rawpy.imread(path) as raw:
        # Postprocess with no exposure adjustment, linear output
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),  # linear gamma
            user_black=None,
            user_sat=None,
            dcb_iteration=False,
            dcb_enhance=False,
            faa_reduction=False,
            median_filter_passes=0,
            erfc_reduce_noise=False,
        )
        # rgb is uint16 shape (H, W, 3), linear (gamma=1,1)
        # rawpy's color_desc is 'RGB' — channels are linear RGB

        # Normalize to [0, 1]
        rgb_f = rgb.astype(np.float32) / 65535.0
        rgb_f = np.clip(rgb_f, 0.0, 1.0)

    # Convert linear RGB to XYZ using a generic matrix.
    # For Fujifilm cameras, rawpy provides rawrgb_to_xyz (or we use a standard matrix).
    # Use BT.709 as a reasonable generic approximation for linear RGB → XYZ.
    M_rgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1374375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    H, W, _ = rgb_f.shape
    flat = rgb_f.reshape(-1, 3)  # (N, 3) linear RGB
    xyz = flat @ M_rgb_to_xyz.T  # (N, 3) XYZ

    # Normalize so that max value maps to 1 (preserves relative exposure)
    max_val = xyz.max()
    if max_val > 0:
        xyz = xyz / max_val

    xyz = np.clip(xyz, 0.0, 1.0)
    return xyz.reshape(H, W, 3).astype(np.float32)


def _load_tiff_to_xyz(path: str) -> np.ndarray:
    """Load a float32 TIFF assumed to already be linear XYZ."""
    data = tifffile.imread(path).astype(np.float32)
    if data.ndim == 2:
        # Grayscale → replicate to 3 channels
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3 and data.shape[2] == 4:
        # RGBA → drop alpha
        data = data[:, :, :3]
    data = np.clip(data, 0.0, 1.0)
    return data.astype(np.float32)


def _load_image_to_xyz_fallback(path: str) -> np.ndarray:
    """Load an 8-bit image (sRGB), linearize, convert to XYZ."""
    img = iio.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # sRGB linearization (approx)
    img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    img = np.clip(img, 0.0, 1.0)

    M_srgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1374375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    H, W, _ = img.shape
    flat = img.reshape(-1, 3)
    xyz = flat @ M_srgb_to_xyz.T
    xyz = np.clip(xyz, 0.0, 1.0)
    return xyz.reshape(H, W, 3).astype(np.float32)


__all__ = ["load_image_to_xyz"]
```

**Step 2d: Run tests to verify they pass**
`pytest backend/tests/test_raw_loader.py -v` → PASS

---

## Step 3: Integrate pipeline into `backend/main.py`

**Files:**
- Modify: `backend/main.py`

**Step 3a: Read current main.py**

The current `/convert` handler returns a dummy JPEG. Replace it with:

```python
import base64
import io
import os
import tempfile
from typing import List

import numpy as np
import imageio.v3 as iio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline as pipe
from . import models
from .cube_parser import parse_cube
from .raw_loader import load_image_to_xyz

app = FastAPI(title="FujiLUT Backend")

origins = ["http://localhost", "http://127.0.0.1"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# --- helpers ---

def _process_image(image_path: str, lut_table: np.ndarray) -> bytes:
    """Full pipeline: image → XYZ → F-Log2 → Rec2020 → LUT → JPEG."""
    # 1. Load image → XYZ
    xyz = load_image_to_xyz(image_path)  # (H, W, 3) float32 [0,1]

    # 2. Apply F-Log2 curve
    flog2 = pipe.apply_flog2_curve(xyz)

    # 3. XYZ → Rec2020 linear
    rec2020 = pipe.xyz_to_rec2020(xyz)  # pipeline function; note: flog2 in, rec2020-like out

    # 4. Apply LUT to the Rec2020 output
    graded = pipe.apply_lut(rec2020, lut_table)  # (H, W, 3) float32 [0,1]

    # 5. Float32 → uint8 → JPEG
    out_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    iio.imwrite(buf, out_u8, format="jpeg", quality=90)
    return buf.getvalue()


def _encode_jpeg_base64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


# --- endpoints ---

@app.on_event("startup")
async def startup_event():
    # Warm up the pipeline to avoid first-request latency
    pipe.prewarm()


@app.get("/health", response_model=models.HealthResponse)
async def health():
    return models.HealthResponse(status="ok", version="1.0.0")


@app.post("/convert", response_model=models.ConvertResponse)
async def convert(image: UploadFile = File(None), luts: List[UploadFile] = File(None)):
    if image is None or luts is None or len(luts) == 0:
        raise HTTPException(status_code=422, detail="Missing image or LUTs")

    try:
        # Persist inputs to temp files (rawpy needs a real path, not file handle)
        with tempfile.NamedTemporaryFile(delete=False) as img_tmp:
            img_tmp.write(await image.read())
            img_tmp_path = img_tmp.name

        results = []
        for lut in luts:
            lut_bytes = await lut.read()
            lut_name = os.path.splitext(lut.filename or "output")[0]

            # Parse .cube file into LUT table
            lut_table = parse_cube(io.BytesIO(lut_bytes))

            # Run full pipeline
            jpeg_bytes = _process_image(img_tmp_path, lut_table)
            encoded = _encode_jpeg_base64(jpeg_bytes)

            results.append(models.ConvertResult(lut_name=lut_name, image_base64_jpeg=encoded))

        return models.ConvertResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp image file
        if "img_tmp_path" in dir():
            try:
                os.unlink(img_tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=19876)
```

**Step 3b: Fix import — `pipe.xyz_to_rec2020` is actually meant to be applied to XYZ before the LUT**

Looking at `pipeline.py`, the intended chain is:
1. `apply_flog2_curve(xyz)` — not really F-Log2 on XYZ (F-Log2 is applied to linear RGB/Rec.2020), but the function applies a log-like curve to the XYZ input. Actually re-reading the pipeline: `apply_flog2_curve` and `xyz_to_rec2020` both take XYZ. The curve should be applied to the Rec.2020 output (after the color space transform), not to XYZ directly.

**Correction**: The correct chain is:
1. `xyz_to_rec2020(xyz)` → linear Rec.2020
2. `apply_flog2_curve(rec2020_linear)` → F-Log2 on linear Rec.2020
3. `apply_lut(flog2, lut_table)` → graded

Fix the `_process_image` function:

```python
def _process_image(image_path: str, lut_table: np.ndarray) -> bytes:
    xyz = load_image_to_xyz(image_path)
    rec2020 = pipe.xyz_to_rec2020(xyz)           # linear Rec.2020
    flog2   = pipe.apply_flog2_curve(rec2020)    # F-Log2 applied to Rec.2020 linear
    graded  = pipe.apply_lut(flog2, lut_table)  # LUT applied in F-Log2 space
    # Float32 → uint8 → JPEG
    out_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    iio.imwrite(buf, out_u8, format="jpeg", quality=90)
    return buf.getvalue()
```

**Step 3c: Verify existing tests still pass**
`pytest backend/tests/test_main.py -v` → PASS (shape/422 checks are format-only)

**Step 3d: Add integration test**

```python
# backend/tests/test_convert_integration.py
"""Integration test: full pipeline via /convert endpoint with real LUT + synthetic image."""
import base64, io, numpy as np, pytest
from fastapi.testclient import TestClient
from backend.main import app

def _synthetic_tiff_bytes() -> bytes:
    import tifffile
    arr = np.random.rand(8, 8, 3).astype(np.float32)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()

def _cube_bytes(n: int = 2) -> bytes:
    vals = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                vals.append(f"{i/(n-1):.6f} {j/(n-1):.6f} {k/(n-1):.6f}")
    content = f"LUT_3D_SIZE {n}\n" + "\n".join(vals) + "\n"
    return content.encode()

def test_convert_returns_valid_jpegs():
    client = TestClient(app, raise_server_exceptions=False)
    files = [
        ("image", ("test.tiff", _synthetic_tiff_bytes(), "image/tiff")),
        ("luts", ("identity.cube", _cube_bytes(2), "text/plain")),
    ]
    resp = client.post("/convert", files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["results"]) == 1
    jpeg_bytes = base64.b64decode(data["results"][0]["image_base64_jpeg"])
    assert jpeg_bytes[:2] == b"\xff\xd8"
```

`pytest backend/tests/test_convert_integration.py -v` → PASS

---

## Step 4: Call `pipeline.prewarm()` at startup

**Files:**
- Modify: `backend/main.py` (already done in Step 3 — `startup_event` calls `pipe.prewarm()`)

---

## Verification Checklist

- [ ] `pytest backend/tests/test_cube_parser.py -v` → all PASS
- [ ] `pytest backend/tests/test_raw_loader.py -v` → all PASS (TIFF path — RAW needs a real file)
- [ ] `pytest backend/tests/test_main.py -v` → all PASS (existing 6 tests)
- [ ] `pytest backend/tests/test_convert_integration.py -v` → PASS
- [ ] `grep -r "import torch" backend/` → empty
- [ ] `python3 -c "from backend.main import app; from fastapi.testclient import TestClient; c = TestClient(app); r = c.get('/health'); print(r.json())"` → `{"status":"ok","version":"1.0.0"}`
- [ ] Update `backend/tests/README.md` with new test files
- [ ] Append to `.sisyphus/notepads/variant-a/learnings.md`
