from pydantic import BaseModel
from typing import List


class ConvertRequest(BaseModel):
    pass  # multipart/form-data — no JSON body needed


class ConvertResult(BaseModel):
    lut_name: str
    image_base64_jpeg: str  # base64-encoded JPEG


class ConvertResponse(BaseModel):
    results: List[ConvertResult]


class HealthResponse(BaseModel):
    status: str  # "ok"
    version: str


class ExportRequest(BaseModel):
    image_path: str
    lut_paths: List[str]
    output_dir: str
    ev_offset: float = 0.0


class ExportResponse(BaseModel):
    count: int
    message: str
