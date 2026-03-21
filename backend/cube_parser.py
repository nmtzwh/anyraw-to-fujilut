"""Parse .cube 3D LUT files into numpy (N,N,N,3) float32 arrays."""
from __future__ import annotations

import io
import numpy as np


def parse_cube(fp: io.BytesIO | io.RawIOBase) -> np.ndarray:
    lines = []
    for raw_line in fp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        if (
            line.startswith("TITLE")
            or line.startswith("LUT_1D_SIZE")
            or line.startswith("DOMAIN_")
        ):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                lines.append([r, g, b])
            except ValueError:
                continue

    n = round(len(lines) ** (1.0 / 3.0))
    expected = n ** 3
    if len(lines) != expected:
        raise ValueError(
            f"LUT_3D_SIZE implied N={n} but got {len(lines)} values (expected {expected})"
        )

    lut = np.array(lines, dtype=np.float32).reshape((n, n, n, 3))
    return np.transpose(lut, (2, 1, 0, 3))


__all__ = ["parse_cube"]
