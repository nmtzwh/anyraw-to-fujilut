# Issues

## Final Verification (Wave 3 completion)

All 4 gates passed on 2026-03-21:

| Gate | Check | Status |
|---|---|---|
| F1 | Backend: torch-free import, 6 pipeline tests, 3 cube_parser tests, 6 endpoint tests, 1 integration test | ✅ PASS |
| F2 | Frontend: IPC handlers, preload API, types.d.ts, error modal HTML/JS, 9 Playwright tests | ✅ PASS |
| F3 | Packaging: electron-builder.yml (appId, nsis, asar), package.json scripts, CI workflow (5 jobs), playwright.config.ts | ✅ PASS |
| F4 | Docs: all READMEs, notepads (learnings/decisions/issues), 3 plan files | ✅ PASS |
- PyTest not installed in environment; tests could not be executed here.
- Ensure that the 3D LUT handling accounts for edge cases where input is exactly at LUT grid boundaries.

## Wave 1.5 Additions (RESOLVED)
- backend/main.py now has full pipeline integration: load_image_to_xyz → xyz_to_rec2020 → apply_flog2_curve → apply_lut → JPEG encode → base64.
- pipeline.prewarm() is now called in FastAPI startup_event.

## Wave 1.3 Additions
- Root requirements.txt still contains torch and PyQt5; only the backend/ subdir is torch-free. Downstream packaging must be careful not to include root requirements.
- scipy added as explicit dep (colour-science transitive); worth confirming if colour-science LUT utilities truly need scipy at runtime.

## Wave 2A.4 Additions
- src/raw-decoder.ts does not exist in electron-wave2a worktree (confirmed absent). Only stubs for color.ts/lut.ts remain. No libraw-wasm imports anywhere in this worktree's src/. package.json devDeps do not include libraw-wasm for this worktree.
