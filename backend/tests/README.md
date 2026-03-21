# backend/tests — Test Suite

## Overview

Hermetic, CPU-only pytest suite for the Torch-free backend pipeline. No external resources required; all tests use synthetic data.

## Run

```bash
# From repo root
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ -v --tb=short

# Collect-only (dry-run)
pytest backend/tests/ --collect-only
```

## Structure

| File | What it tests |
|---|---|
| `test_pipeline.py` | Pure pipeline functions: load_raw, apply_flog2_curve, xyz_to_rec2020, apply_lut |
| `test_cube_parser.py` | .cube LUT file parsing: basic, 33-point, identity, error cases |
| `test_raw_loader.py` | Image loading (TIFF, JPEG, PNG) and RGB→XYZ conversion |
| `test_main.py` | FastAPI endpoints: GET /health, POST /convert; Torch-free import guard |
| `test_convert_integration.py` | End-to-end /convert: TIFF + .cube → graded JPEG per LUT |

## Test inventory

**test_pipeline.py** (6 tests)
- `test_load_raw_py_loader` — roundtrip .npy save+load preserves shape/dtype/values
- `test_apply_flog2_curve_shape_and_range` — monotonic curve, endpoints map to 0/1
- `test_apply_flog2_curve_out_of_range_clamps` — values <0 and >1 clamp to [0,1]
- `test_xyz_to_rec2020_matrix_application` — shape/dtype/range + deterministic matrix
- `test_end_to_end_synthetic` — full sequence: xyz → flog2 → lut; checks shape/dtype/range
- `test_apply_lut_basic_identity_mapping` — identity LUT preserves pixel values

**test_cube_parser.py** (6 tests)
- `test_parse_cube_basic_shape_and_values` — 2-point LUT: shape (2,2,2,3), corner values correct
- `test_parse_cube_strips_comments` — comments and TITLE lines are ignored
- `test_parse_cube_33_point` — 33-point LUT (35937 values) parses to shape (33,33,33,3)
- `test_parse_cube_3_point_identity` — 3-point identity LUT preserves coordinates
- `test_parse_cube_rejects_wrong_value_count` — wrong number of values raises ValueError
- `test_parse_cube_empty_lines_ignored` — blank lines are skipped without error

**test_raw_loader.py** (9 tests)
- `test_loads_float32_tiff` — loads float32 TIFF → (H,W,3) float32 [0,1]
- `test_tiff_grayscale_expands_to_3ch` — 2D TIFF replicates to 3 channels
- `test_tiff_clips_to_01` — values outside [0,1] are clipped
- `test_loads_jpeg` — loads JPEG → (H,W,3) float32 [0,1]
- `test_loads_png` — loads PNG → (H,W,3) float32 [0,1]
- `test_jpeg_grayscale_expands_to_3ch` — 2D JPEG replicates to 3 channels
- `test_dispatches_tiff` — load_image_to_xyz dispatches TIFF files correctly
- `test_dispatches_jpg` — load_image_to_xyz dispatches JPEG files correctly
- `test_rejects_unsupported_extension` — unknown extensions raise ValueError

**test_main.py** (6 tests)
- `test_health_returns_ok_and_version` — GET /health returns 200 with status+version
- `test_convert_missing_image_returns_422` — POST /convert without image → 422
- `test_convert_missing_luts_returns_422` — POST /convert without LUTs → 422
- `test_convert_missing_both_returns_422` — POST /convert with neither → 422
- `test_convert_with_image_and_luts_returns_results` — 200, N results, base64 JPEGs valid
- `test_no_torch_imports_in_backend` — torch cannot be imported through backend package

**test_convert_integration.py** (4 tests)
- `test_convert_returns_valid_jpegs_for_each_lut` — multiple LUTs each return valid JPEG
- `test_convert_lut_name_matches_filename` — lut_name field matches the .cube filename
- `test_convert_preserves_original_extension_in_temp_path` — ARW/DNG/etc. extensions preserved
- `test_convert_jpeg_is_reasonable_size` — output JPEG is 100B–1MB (not empty or huge)

## Torch-free constraint

No test in this directory may import `torch`, `torch.*`, `PyQt5`, or any PyQt5 submodule. The `TestTorchFree` class enforces this programmatically.

## Dependencies

All required packages are listed in `backend/requirements.txt`. Install with:

```bash
pip install -r backend/requirements.txt
```

Minimum versions: numpy>=1.26, pytest>=7.0, httpx>=0.24, fastapi>=0.100.
