import base64
import io
import os
import sys
import tempfile
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

# When run as `python backend/main.py`, the parent dir isn't on sys.path,
# so `backend` can't be imported as a package.  Fix: add the repo root.
if __package__ is None:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import numpy as np
import imageio.v3 as iio
import traceback
from threadpoolctl import threadpool_limits
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from backend import pipeline as pipe
from backend import models
from backend.cube_parser import parse_cube
from backend.raw_loader import load_image_to_xyz
from backend.cache import PipelineCache


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # pipe.prewarm() # Temporarily disabled to debug startup hang
    yield


app = FastAPI(title="FujiLUT Backend", lifespan=lifespan)

origins = ["http://localhost", "http://127.0.0.1"]

# Global pipeline cache — shared across all requests
cache = PipelineCache()


def _process_image(image_path: str, lut_table: np.ndarray, lut_hash: str, shrink: int = 1, ev_offset: float = 0.0) -> bytes:
    """Process image through the full pipeline, leveraging the cache."""
    return cache.get_or_render_jpeg(image_path, shrink, ev_offset, lut_hash, lut_table, quality=90)


def _batch_export_image(image_path: str, lut_table: np.ndarray, lut_hash: str, save_path: str, ev_offset: float = 0.0):
    """Export at full resolution. Uses cache for XYZ decode but not for JPEG (different quality)."""
    # Throttle OpenMP/BLAS threading to 2 cores maximum when exporting in background
    with threadpool_limits(limits=2):
        flog2 = cache.get_or_compute_flog2(image_path, shrink=1, ev_offset=ev_offset)
        graded = pipe.apply_lut(flog2, lut_table)
        out_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
        iio.imwrite(save_path, out_u8, format="jpeg", quality=95)


@app.get("/health")
async def health():
    return models.HealthResponse(status="ok", version="1.0.0")


@app.post("/convert")
async def convert(
    image: UploadFile = File(default=None),
    luts: List[UploadFile] = File(default=None),
    preview: bool = Form(default=True),
    ev_offset: float = Form(default=0.0),
    include_original: bool = Form(default=False),
):
    if image is None or (luts is None and not include_original) or (luts is not None and len(luts) == 0 and not include_original):
        raise HTTPException(status_code=422, detail="Missing image or LUTs")

    img_tmp_path = None
    try:
        orig_ext = os.path.splitext(image.filename or "")[1]
        img_tmp_path = tempfile.mktemp(suffix=orig_ext)
        with open(img_tmp_path, "wb") as img_tmp:
            img_tmp.write(await image.read())

        results = []
        # Use shrink=4 for previews, 1 for full resolution
        shrink = 4 if preview else 1
        
        if include_original:
            orig_jpeg = cache.render_original_jpeg(img_tmp_path, shrink, ev_offset)
            results.append(
                models.ConvertResult(
                    lut_name="__Original__",
                    image_base64_jpeg=base64.b64encode(orig_jpeg).decode("ascii"),
                )
            )

        if luts:
            for lut in luts:
                lut_bytes = await lut.read()
                lut_table, lut_hash = cache.get_or_parse_lut(lut_bytes)
                jpeg_bytes = _process_image(img_tmp_path, lut_table, lut_hash, shrink=shrink, ev_offset=ev_offset)
                lut_name = os.path.splitext(lut.filename or "output")[0]
                results.append(
                    models.ConvertResult(
                        lut_name=lut_name,
                        image_base64_jpeg=base64.b64encode(jpeg_bytes).decode("ascii"),
                    )
                )

        return models.ConvertResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if img_tmp_path:
            try:
                os.unlink(img_tmp_path)
            except OSError:
                pass


@app.post("/export")
async def export_batch(req: models.ExportRequest):
    try:
        if not os.path.exists(req.output_dir):
            os.makedirs(req.output_dir, exist_ok=True)

        count = 0
        base_name = os.path.splitext(os.path.basename(req.image_path))[0]

        for lut_path in req.lut_paths:
            if not os.path.exists(lut_path):
                continue

            lut_table, lut_hash = cache.get_or_parse_lut_from_path(lut_path)

            lut_name = os.path.splitext(os.path.basename(lut_path))[0]
            save_name = f"{base_name}_{lut_name}.jpeg"
            save_path = os.path.join(req.output_dir, save_name)

            # Offload heavy CPU work to a background thread to prevent blocking the Event Loop
            await asyncio.to_thread(
                _batch_export_image, req.image_path, lut_table, lut_hash, save_path, req.ev_offset
            )
            count += 1

        return models.ExportResponse(
            count=count, message=f"Exported {count} images to {req.output_dir}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Path-based conversion (no file upload) ─────────────────────────────────

class ConvertByPathRequest(BaseModel):
    image_path: str
    lut_paths: List[str] = []
    preview: bool = True
    ev_offset: float = 0.0
    include_original: bool = False


@app.post("/convert-by-path")
async def convert_by_path(req: ConvertByPathRequest):
    """Convert a RAW image using file paths instead of uploads.
    
    This endpoint avoids the overhead of re-uploading large RAW files
    over HTTP. The backend reads files directly from the filesystem,
    leveraging the multi-level pipeline cache for near-instant responses
    when the same image or LUT has been seen before.
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    try:
        shrink = 4 if req.preview else 1
        results = []

        if req.include_original:
            orig_jpeg = cache.render_original_jpeg(req.image_path, shrink, req.ev_offset)
            results.append(
                models.ConvertResult(
                    lut_name="__Original__",
                    image_base64_jpeg=base64.b64encode(orig_jpeg).decode("ascii"),
                )
            )

        for lut_path in req.lut_paths:
            if not os.path.exists(lut_path):
                continue
            lut_table, lut_hash = cache.get_or_parse_lut_from_path(lut_path)
            jpeg_bytes = cache.get_or_render_jpeg(
                req.image_path, shrink, req.ev_offset, lut_hash, lut_table, quality=90
            )
            lut_name = os.path.splitext(os.path.basename(lut_path))[0]
            results.append(
                models.ConvertResult(
                    lut_name=lut_name,
                    image_base64_jpeg=base64.b64encode(jpeg_bytes).decode("ascii"),
                )
            )

        return models.ConvertResponse(results=results)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Cache management ─────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """Return cache hit/miss statistics and current entry counts."""
    return cache.stats()


@app.post("/cache/clear")
async def cache_clear():
    """Flush all in-memory caches."""
    cache.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19876)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
