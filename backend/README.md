Backend Wave1 Pipeline: API contract and usage

Overview
- Pure-Python, CPU-based image processing pipeline for XYZ data -> F-Log2 -> 3D LUT -> 8-bit output.
- No PyTorch or GUI dependencies. Meant to be importable by backend/main.py and testable in unit tests.

Public API (backend.pipeline)
- load_raw(path: str) -> np.ndarray
  Load a 3-channel XYZ image (H, W, 3) as float32 in [0, 1]. Supports .npy, .npz.
- apply_flog2_curve(xyz: np.ndarray) -> np.ndarray
  Apply a deterministic F-Log2-like curve to XYZ values. Returns same shape, in [0, 1].
- xyz_to_rec2020(xyz: np.ndarray) -> np.ndarray
  Affine transform from XYZ to Rec.2020-like space. Returns in [0, 1].
- apply_lut(flog2: np.ndarray, lut_table: np.ndarray) -> np.ndarray
  Apply a 3D LUT to the 3-channel input. Expects lut_table of shape (N, N, N, 3).
- prewarm() -> None
  Optional warm-up to improve first-call performance.

Usage example (synthetic):
- from backend.pipeline import load_raw, apply_flog2_curve, xyz_to_rec2020, apply_lut
- xyz = load_raw('data.npy')
- flog2 = apply_flog2_curve(xyz)
- rec2020 = xyz_to_rec2020(xyz)
- lut_out = apply_lut(flog2, lut_table)

Notes
- The plan references a Torch-free approach and a test-driven development style,
  focused on deterministic, pure-Python implementations suitable for unit tests.
- This repository uses a backend-wave1 worktree; do not commit to main branch.

Requirements
- Python 3.11.x required (see backend/requirements.txt for pinned dependencies).
- Install: `pip install -r backend/requirements.txt`
- Verify no torch installed: `python -c "import torch"` should raise ImportError
