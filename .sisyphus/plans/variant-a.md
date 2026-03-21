# Plan: anyraw-to-fujilut — Cross-Platform Architecture Refactor

## TL;DR

Refactor the half-built app into a proper TS frontend (Electron) + Python backend (FastAPI, torch-free) architecture, packaged as single-file installers per platform.

**Recommended path:** Electron + embedded Python subprocess (FastAPI service over `localhost:19876`).

**Variant comparison:** Electron vs. Tauri — Electron chosen for MVP (existing TS, mature packaging); Tauri documented as v2 migration target.

---

## 1. Current Architecture (as-is, confirmed by direct exploration)

```
┌─────────────────────────────────────────────────────────┐
│  src/ (TS client-side pipeline — works in browser)      │
│  raw-decoder.ts → libraw-wasm (WASM demosaicing)        │
│  color.ts        → F-Log2 + XYZ→Rec2020 (pure TS)       │
│  lut.ts          → WebGL2 3D-LUT applier                │
│  app.ts          → UI orchestration                     │
│                                                         │
│  server.js → Express static server (localhost:3000)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Python layer                                           │
│  convert_raw.py        → NumPy, CPU-only, verified ✅   │
│  convert_raw_torch.py  → PyTorch GPU, CLI ✅           │
│  raw_converter_gui.py → PyQt5 GUI ⚠️ imports torch!   │
└─────────────────────────────────────────────────────────┘

requirements.txt  →  torch, PyQt5, rawpy, numpy, colour-science, imageio, tifffile
package.json      →  express, libraw-wasm, multer, parse-cube-lut, esbuild, typescript
```

**Critical coupling found:**
- `raw_converter_gui.py` **imports torch** via `from convert_raw_torch import ...` — GUI cannot start without PyTorch.
- `requirements.txt` includes `torch` and `PyQt5` as hard dependencies for the GUI.
- No IPC layer exists between TS frontend and Python backend.
- TS frontend is WASM/browser-only — no desktop integration (no native file dialogs, menus, or system tray).
- No packaging config for desktop distribution.
- The standalone CLI (`convert_raw.py`) and the GUI are completely separate codebases with duplicated logic.

---

## 2. Architecture Decisions

### Decision 1 — Frontend: Electron (primary), Tauri (fallback)

**Electron (primary — for MVP velocity)**

| Pro | Con |
|---|---|
| Existing TS pipeline in `src/` adapts directly (color.ts, lut.ts reusable) | ~150–200 MB app size (Chromium + Node runtime) |
| Mature electron-builder → NSIS one-file .exe, AppImage, DMG | Higher memory footprint than Tauri |
| Straightforward IPC via `ipcMain`/`ipcRenderer` → Python subprocess over HTTP | Larger attack surface (mitigated by contextIsolation) |
| Node.js ecosystem for file system, native dialogs, menus | |
| `contextIsolation: true`, `nodeIntegration: false` for security | |

**Tauri (v2 migration target — for size/security)**

| Pro | Con |
|---|---|
| ~10 MB binary, minimal attack surface | Requires Rust toolchain setup |
| System WebView (no bundled Chromium) | Current TS pipeline needs adaptation (no Node.js in renderer) |
| electron-builder-equivalent via `@tauri-apps/cli` | Steeper learning curve |

**Decision:** Proceed with Electron as primary. Document Tauri as a v2 migration path.

### Decision 2 — Backend: Python FastAPI, torch-free

**Remove torch entirely.** `convert_raw.py` (NumPy + colour-science + rawpy) is verified and fast enough on CPU for typical RAW sizes. The PyTorch GPU path adds ~2 GB of dependencies for marginal speed gains on small batches. Removing torch eliminates packaging complexity.

**Architecture:**
- Backend runs as a **subprocess** spawned by Electron on startup.
- Python process starts **FastAPI on fixed port 19876** (`http://127.0.0.1:19876`).
  - Fixed port chosen for predictable wiring, zero discovery overhead, and no race conditions during startup.
  - If port 19876 is in use, FastAPI fails fast with a user-visible error — no silent fallback.
- Electron main process proxies HTTP requests; renderer uses `ipcRenderer.invoke` → main → HTTP → Python.
- Backend is **stateless per request** (no GPU state).
- Backend pre-warms on startup (runs a dummy conversion on first boot) to eliminate first-request latency.

**Python version:** Python 3.11.x. Pinned in `backend/requirements.txt`.

**API surface (minimal, localhost-only):**

```
POST /convert
  Body: multipart/form-data { image: File, luts: File[] }
  Response: JSON { results: [{ lut_name: string, image_base64_jpeg: string }] }

GET /health
  Response: JSON { status: "ok", version: string }
```

**Preview vs. Export pipeline:**
- **Preview** (renderer, fast): Keep `src/color.ts` + `src/lut.ts` for live WebGL2 LUT preview in the renderer. This gives instant feedback when switching LUTs.
- **Export** (backend, accurate): `POST /convert` returns the final graded JPEG for each LUT, used when the user triggers export. Ensures bit-exact colour-science LUT application.
- `src/raw-decoder.ts` is **removed** — RAW decoding always goes through the Python backend (rawpy) for broad format support.

### Decision 3 — IPC: HTTP over localhost:19876

- Fixed port 19876 for predictable, debuggable wiring — no discovery dance.
- FastAPI is HTTP-native → minimal glue.
- Electron main process acts as HTTP proxy → renderer never makes raw HTTP calls (avoids CORS complexity).
- Local-only process → no TLS needed on localhost; no network exposure.
- Error handling: if backend fails to bind 19876, show modal error with "Retry" button that re-spawns the subprocess.

### Decision 4 — Packaging: electron-builder

| Platform | Format | Single-file? |
|---|---|---|
| Windows | NSIS installer (`.exe`) | ✅ `nsis.oneFile: true` |
| macOS | DMG (`.dmg`) | ✅ bundled in single DMG |
| Linux | AppImage (`.AppImage`) | ✅ `linux.target: AppImage` |

- Python venv bundled inside Electron at `resources/`.
- Auto-starts/stops Python subprocess with Electron lifecycle.
- No separate Python installation required on end-user machine.

---

## 3. Scope

### IN
- Electron app scaffold (main + renderer process structure).
- FastAPI backend service wrapping `convert_raw.py` pipeline.
- IPC layer: Electron main process ↔ Python FastAPI over localhost:19876.
- Strip PyQt5 and torch from all packaged outputs.
- Single-file installers per platform via electron-builder.
- `convert_raw.py` CLI preserved as standalone entry point.
- `src/color.ts` and `src/lut.ts` reused as renderer preview pipeline.
- `src/raw-decoder.ts` **removed** — RAW decoding always via Python backend.
- `src/app.ts` rewritten for Electron (native dialogs, file export, backend IPC).
- `public/` and `server.js` kept for browser-only development/demo, excluded from Electron packaging.
- Error UX: backend crash → modal dialog with "Restart Backend" button.
- RAW format support: rawpy (CPU) handles decoding in backend; libraw-wasm excluded from packaged app.

### OUT
- PyQt5 GUI — completely removed from packaged app; do not port.
- PyTorch pipeline — CPU NumPy path only; torch never imported in packaged outputs.
- `convert_raw_torch.py` — kept as reference, not packaged.
- Code signing — v2 follow-up.
- libraw-wasm — excluded from packaged Electron app; kept only for browser demo path.

---

## 4. Target Module Map (after refactor)

```
/
├── src/                          # Electron renderer (TS)
│   ├── app.ts                    # UI logic, state, IPC calls
│   ├── color.ts                  # Keep (F-Log2, XYZ→Rec2020, exposure)
│   ├── lut.ts                    # Keep (WebGL2 LUT applier for preview)
│   ├── types.d.ts                # Keep
│   └── electron/                  # New
│       ├── main.ts               # App lifecycle, subprocess spawn/kill
│       ├── preload.ts            # Secure IPC bridge (contextBridge)
│       └── ipc.ts                # HTTP proxy to Python backend
│
├── backend/                      # New: Python backend service
│   ├── __init__.py
│   ├── main.py                   # FastAPI app + lifespan
│   ├── pipeline.py               # Refactored from convert_raw.py
│   │                             #   load_raw(path) → np.ndarray (XYZ)
│   │                             #   apply_flog2(xyz)  → np.ndarray (F-Log2)
│   │                             #   apply_lut(flog2, lut_table) → np.ndarray
│   ├── models.py                 # Pydantic request/response models
│   └── requirements.txt          # numpy, rawpy, colour-science, uvicorn,
│                                  # fastapi, python-multipart
│
├── convert_raw.py                 # Keep (standalone CLI, unchanged)
├── convert_raw_torch.py           # Keep (reference only)
├── raw_converter_gui.py           # Keep (reference only)
│
├── requirements.txt              # Simplify: remove torch, PyQt5
├── electron-builder.yml           # New: packaging config
├── package.json                  # Update: add electron, electron-builder
└── tsconfig.json                # Update: multi-target (renderer + main)
```

---

## 5. Execution Waves

### Wave 1 — Backend Foundation
**1.1** Extract pipeline from `convert_raw.py` into `backend/pipeline.py`:
- `load_raw(path: str) → np.ndarray` — XYZ float32
- `apply_flog2(xyz: np.ndarray, ev_offset: float) → np.ndarray` — F-Log2 float32
- `apply_lut(flog2: np.ndarray, lut_table: np.ndarray) → np.ndarray` — graded uint8

**1.2** Build `backend/main.py` (FastAPI):
- `POST /convert` — accepts RAW + N LUT files → returns base64 JPEG per LUT
- `GET /health` — health check; returns `{ status: "ok", version: string }`
- CORS disabled (localhost-only)
- Fixed port **19876** — fail fast with a clear OS-level error if port is unavailable.
- Backend pre-warms on startup: runs a null conversion to JIT-compile the NumPy path before the first user request.

**1.3** Update `backend/requirements.txt`: remove `torch`, `PyQt5`; add `fastapi`, `uvicorn`, `python-multipart`; pin `numpy>=1.26,<2`, `rawpy`, `colour-science`, `imageio`, `tifffile`. Python version: **3.11.x**.

**1.4** Add backend tests (pytest) — use `convert_raw.py` output as regression oracle

### Wave 2A — Electron Frontend (parallel with Wave 1)
**2A.1** Scaffold Electron:
- `npm install electron @electron/remote electron-builder`
- `src/electron/main.ts` — window creation, Python subprocess lifecycle
- `src/electron/preload.ts` — `contextBridge` IPC bridge
- `src/electron/ipc.ts` — HTTP proxy (renderer → preload → main → HTTP → Python)

**2A.2** Adapt renderer:
- **Remove** `src/raw-decoder.ts` entirely — no WASM decoding in packaged app.
- **Keep** `src/color.ts`, `src/lut.ts` for renderer-side preview pipeline (WebGL2 LUT preview for instant feedback).
- **Rewrite** `src/app.ts` for Electron: use `dialog.showOpenDialog` for file selection, `fs.writeFile` for export, and IPC calls for conversion.
- Update build: add esbuild for main/preload TS compilation.

**2A.3** `tsconfig.json` multi-target: renderer (web) + main (Node)

### Wave 2B — Electron Packaging (parallel with Wave 2A)
**2B.1** `electron-builder.yml`:
```yaml
appId: com.fujilut.app
productName: FujiLUT
directories: { output: dist }
files: ['build/**/*', 'backend/**/*', 'venv/**/*']
extraResources:
  - from: backend/
    to: backend/
    filter: ['**/*']
asar: true
nsis:
  oneFile: true      # Windows single-file .exe
mac:
  target: dmg
linux:
  target: AppImage
```

**2B.2** Bundle Python venv in Electron:
```ts
// src/electron/main.ts
const venvPython = path.join(process.resourcesPath, 'backend', 'venv', 'bin', 'python');
const backendScript = path.join(process.resourcesPath, 'backend', 'main.py');
const backendProcess = spawn(venvPython, [backendScript], { stdio: 'pipe' });
// Write port to temp file when backend starts
// Read port, wait for /health, then show window
```

**2B.3** Verify all three platform installers build

### Wave 3 — Stabilization
- End-to-end tests (Playwright) covering full pipeline (open RAW → select LUTs → convert → export JPEG).
- GitHub Actions CI for cross-platform build verification.
- Error UX: Python crash → Electron modal dialog with "Restart Backend" button (no silent retry).
- Performance: benchmark NumPy pipeline on 24 MP images; add LUT result caching if p95 latency > 2s per LUT.

---

## 6. Verification (agent-executable)

| # | Check | Method |
|---|---|---|
| 1 | Backend has zero torch imports | `grep -r "import torch" backend/` → empty |
| 2 | Backend starts and /health responds | `curl http://localhost:{port}/health` → `{"status":"ok"}` |
| 3 | POST /convert returns valid results | POST RAW + LUTs → parse JSON → N base64 strings |
| 4 | requirements.txt clean | no `torch`, no `PyQt5` |
| 5 | Electron app starts subprocess | launch app → backend process visible in OS process list |
| 6 | Full pipeline end-to-end | RAW + LUTs → convert → export JPEG → verify file |
| 7 | Windows one-file .exe | build → run installer → app launches offline |
| 8 | macOS .dmg | build → mount → drag to Applications → app launches |
| 9 | Linux .AppImage | build → chmod +x → run → app launches |
| 10 | Standalone CLI still works | `python convert_raw.py -i RAW -l LUT/` → output JPEG |

---

## 7. Resolved Decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Python version | **Python 3.11.x** | Stable ABI, broad wheel support for numpy/rawpy/colour-science |
| 2 | Backend port | **Fixed 19876** | Predictable wiring, zero discovery overhead, no race conditions |
| 3 | TS module reuse | **Keep** `color.ts` + `lut.ts`; **rewrite** `app.ts`; **remove** `raw-decoder.ts` | Color/LUT logic is reusable; backend handles all RAW decoding |
| 4 | `public/` + `server.js` | **Keep** for browser demo; **exclude** from Electron packaging | Useful for dev/demo without bundling Python |
| 5 | Error UX | **Modal dialog + "Restart Backend" button** | Desktop UX should be explicit; no silent retry to avoid confusion |
| 6 | LUT preview pipeline | **Renderer WebGL2** for live preview; **Python backend** for export | Instant feedback + bit-exact final output |
| 7 | RAW format support | **rawpy (CPU)** in backend; **libraw-wasm excluded** from packaged app | Broadest format coverage via rawpy; simpler packaging without WASM |
