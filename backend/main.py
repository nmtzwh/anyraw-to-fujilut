import base64
import io
import os
import tempfile
from typing import List

import numpy as np
import imageio.v3 as iio
from fastapi import FastAPI, File, UploadFile, HTTPException

from . import pipeline as pipe
from . import models
from .cube_parser import parse_cube
from .raw_loader import load_image_to_xyz

app = FastAPI(title="FujiLUT Backend")

origins = ["http://localhost", "http://127.0.0.1"]


def _process_image(image_path: str, lut_table: np.ndarray) -> bytes:
    xyz = load_image_to_xyz(image_path)
    rec2020 = pipe.xyz_to_rec2020(xyz)
    flog2 = pipe.apply_flog2_curve(rec2020)
    graded = pipe.apply_lut(flog2, lut_table)
    out_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    iio.imwrite(buf, out_u8, format="jpeg", quality=90)
    return buf.getvalue()


@app.on_event("startup")
async def startup_event():
    pipe.prewarm()


@app.get("/health")
async def health():
    return models.HealthResponse(status="ok", version="1.0.0")


@app.post("/convert")
async def convert(
    image: UploadFile = File(default=None), luts: List[UploadFile] = File(default=None)
):
    if image is None or luts is None or len(luts) == 0:
        raise HTTPException(status_code=422, detail="Missing image or LUTs")

    img_tmp_path = None
    try:
        orig_ext = os.path.splitext(image.filename or "")[1]
        img_tmp_path = tempfile.mktemp(suffix=orig_ext)
        with open(img_tmp_path, "wb") as img_tmp:
            img_tmp.write(await image.read())

        results = []
        for lut in luts:
            lut_table = parse_cube(io.BytesIO(await lut.read()))
            jpeg_bytes = _process_image(img_tmp_path, lut_table)
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
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if img_tmp_path:
            try:
                os.unlink(img_tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=19876)
