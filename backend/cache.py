"""
Multi-level pipeline cache for the RAW → LUT processing pipeline.

Cache Levels:
  L0  XYZ decode cache      — keyed by (path, mtime, shrink)
  L1  LUT table cache       — keyed by content hash, disk-persisted
  L2  F-Log2 intermediate   — keyed by (image_key, ev_offset)
  L3  Final JPEG bytes      — keyed by (image_key, lut_hash, shrink, ev)

All levels use an LRU eviction policy with configurable max sizes.
Thread-safe via a single reentrant lock.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import imageio.v3 as iio

from backend import pipeline as pipe
from backend.raw_loader import load_image_to_xyz
from backend.cube_parser import parse_cube


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_XYZ_ENTRIES = 20        # L0 – decoded XYZ arrays (preview res)
MAX_LUT_ENTRIES = 10        # L1 – parsed LUT tables
MAX_FLOG2_ENTRIES = 20      # L2 – F-Log2 intermediates
MAX_JPEG_ENTRIES = 100      # L3 – final encoded JPEGs

# Disk cache directory for LUT tables
_LUT_DISK_DIR: Path | None = None


def _get_lut_disk_dir() -> Path:
    """Return (and lazily create) the disk cache directory for LUT tables."""
    global _LUT_DISK_DIR
    if _LUT_DISK_DIR is None:
        base = Path(os.environ.get("FUJILUT_CACHE_DIR", ""))
        if not base.is_dir():
            import tempfile
            base = Path(tempfile.gettempdir()) / "fujilut_cache"
        _LUT_DISK_DIR = base / "luts"
        _LUT_DISK_DIR.mkdir(parents=True, exist_ok=True)
    return _LUT_DISK_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_key(path: str, shrink: int) -> Tuple[str, float, int]:
    """Generate a cache key from file path + mtime + shrink."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (os.path.normpath(path), mtime, shrink)


def _content_hash(data: bytes) -> str:
    """Fast content hash for LUT bytes."""
    return hashlib.md5(data).hexdigest()


def _lru_put(cache: OrderedDict, key, value, max_size: int) -> None:
    """Insert into an LRU OrderedDict, evicting oldest if over capacity."""
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
    else:
        if len(cache) >= max_size:
            cache.popitem(last=False)  # evict oldest
        cache[key] = value


def _lru_get(cache: OrderedDict, key):
    """Look up a key, returning None on miss. Moves hit to end (most recent)."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


# ---------------------------------------------------------------------------
# PipelineCache
# ---------------------------------------------------------------------------

class PipelineCache:
    """Four-level LRU cache for the RAW → LUT preview pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # L0: XYZ arrays — key: (norm_path, mtime, shrink)
        self._xyz: OrderedDict[Tuple, np.ndarray] = OrderedDict()

        # L1: LUT tables — key: content_hash (str)
        self._luts: OrderedDict[str, np.ndarray] = OrderedDict()

        # L2: F-Log2 intermediates — key: (norm_path, mtime, shrink, ev_offset)
        self._flog2: OrderedDict[Tuple, np.ndarray] = OrderedDict()

        # L3: Final JPEG bytes — key: (norm_path, mtime, shrink, ev_offset, lut_hash)
        self._jpeg: OrderedDict[Tuple, bytes] = OrderedDict()

        # Stats
        self.hits = {"xyz": 0, "lut": 0, "flog2": 0, "jpeg": 0}
        self.misses = {"xyz": 0, "lut": 0, "flog2": 0, "jpeg": 0}

    # -- L0: XYZ decode -------------------------------------------------------

    def get_or_decode_xyz(self, path: str, shrink: int) -> np.ndarray:
        """Return cached XYZ or decode from RAW file."""
        key = _file_key(path, shrink)
        with self._lock:
            cached = _lru_get(self._xyz, key)
            if cached is not None:
                self.hits["xyz"] += 1
                return cached

        # Decode outside the lock (CPU-bound, may be slow)
        xyz = load_image_to_xyz(path, shrink=shrink)

        with self._lock:
            self.misses["xyz"] += 1
            _lru_put(self._xyz, key, xyz, MAX_XYZ_ENTRIES)
        return xyz

    # -- L1: LUT tables -------------------------------------------------------

    def get_or_parse_lut(self, lut_bytes: bytes) -> Tuple[np.ndarray, str]:
        """Return (lut_table, content_hash). Checks memory then disk cache."""
        h = _content_hash(lut_bytes)

        with self._lock:
            cached = _lru_get(self._luts, h)
            if cached is not None:
                self.hits["lut"] += 1
                return cached, h

        # Check disk cache
        disk_path = _get_lut_disk_dir() / f"{h}.npy"
        if disk_path.exists():
            try:
                lut_table = np.load(str(disk_path))
                with self._lock:
                    self.hits["lut"] += 1
                    _lru_put(self._luts, h, lut_table, MAX_LUT_ENTRIES)
                return lut_table, h
            except Exception:
                pass  # corrupt file, re-parse

        # Parse from bytes
        lut_table = parse_cube(io.BytesIO(lut_bytes))

        # Save to disk
        try:
            np.save(str(disk_path), lut_table)
        except OSError:
            pass  # non-fatal

        with self._lock:
            self.misses["lut"] += 1
            _lru_put(self._luts, h, lut_table, MAX_LUT_ENTRIES)
        return lut_table, h

    def get_or_parse_lut_from_path(self, lut_path: str) -> Tuple[np.ndarray, str]:
        """Convenience: read a .cube file from disk and cache it."""
        with open(lut_path, "rb") as f:
            lut_bytes = f.read()
        return self.get_or_parse_lut(lut_bytes)

    # -- L2: F-Log2 intermediate -----------------------------------------------

    def get_or_compute_flog2(
        self, path: str, shrink: int, ev_offset: float
    ) -> np.ndarray:
        """Return cached F-Log2 or compute stages 1→5."""
        fk = _file_key(path, shrink)
        flog2_key = (*fk, ev_offset)

        with self._lock:
            cached = _lru_get(self._flog2, flog2_key)
            if cached is not None:
                self.hits["flog2"] += 1
                return cached

        # L0 hit possible even on L2 miss (e.g. EV changed)
        xyz = self.get_or_decode_xyz(path, shrink)

        # Stages 2-5
        gain = pipe.get_exposure_gain(xyz, ev_offset=ev_offset)
        xyz_exposed = xyz * gain
        rec2020 = pipe.xyz_to_rec2020(xyz_exposed)
        flog2 = pipe.apply_flog2_curve(rec2020)

        with self._lock:
            self.misses["flog2"] += 1
            _lru_put(self._flog2, flog2_key, flog2, MAX_FLOG2_ENTRIES)
        return flog2

    # -- L3: Final JPEG --------------------------------------------------------

    def get_or_render_jpeg(
        self,
        path: str,
        shrink: int,
        ev_offset: float,
        lut_hash: str,
        lut_table: np.ndarray,
        quality: int = 90,
    ) -> bytes:
        """Return cached JPEG or apply LUT + encode."""
        fk = _file_key(path, shrink)
        jpeg_key = (*fk, ev_offset, lut_hash)

        with self._lock:
            cached = _lru_get(self._jpeg, jpeg_key)
            if cached is not None:
                self.hits["jpeg"] += 1
                return cached

        # L2 hit possible
        flog2 = self.get_or_compute_flog2(path, shrink, ev_offset)

        # Stage 6: apply LUT
        graded = pipe.apply_lut(flog2, lut_table)

        # Stage 7: encode JPEG
        out_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
        buf = io.BytesIO()
        iio.imwrite(buf, out_u8, format="jpeg", quality=quality)
        jpeg_bytes = buf.getvalue()

        with self._lock:
            self.misses["jpeg"] += 1
            _lru_put(self._jpeg, jpeg_key, jpeg_bytes, MAX_JPEG_ENTRIES)
        return jpeg_bytes

    # -- Original (no-LUT) preview --------------------------------------------

    def render_original_jpeg(
        self, path: str, shrink: int, ev_offset: float, quality: int = 90
    ) -> bytes:
        """Render the F-Log2 image (no LUT applied) as JPEG."""
        flog2 = self.get_or_compute_flog2(path, shrink, ev_offset)
        out_u8 = (np.clip(flog2, 0.0, 1.0) * 255.0).astype(np.uint8)
        buf = io.BytesIO()
        iio.imwrite(buf, out_u8, format="jpeg", quality=quality)
        return buf.getvalue()

    # -- Management ------------------------------------------------------------

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": {
                    "xyz": len(self._xyz),
                    "lut": len(self._luts),
                    "flog2": len(self._flog2),
                    "jpeg": len(self._jpeg),
                },
                "limits": {
                    "xyz": MAX_XYZ_ENTRIES,
                    "lut": MAX_LUT_ENTRIES,
                    "flog2": MAX_FLOG2_ENTRIES,
                    "jpeg": MAX_JPEG_ENTRIES,
                },
                "hits": dict(self.hits),
                "misses": dict(self.misses),
            }

    def clear(self) -> None:
        """Flush all in-memory caches. Disk LUT cache is preserved."""
        with self._lock:
            self._xyz.clear()
            self._luts.clear()
            self._flog2.clear()
            self._jpeg.clear()
            for k in self.hits:
                self.hits[k] = 0
            for k in self.misses:
                self.misses[k] = 0


__all__ = ["PipelineCache"]
