/**
 * app.ts — Electron renderer entry point.
 *
 * Architecture: the renderer is sandboxed. All file access and network calls go
 * through window.electronAPI (contextBridge). RAW decoding is delegated entirely
 * to the Python backend — this file only handles UI state and IPC calls.
 */

const api = window.electronAPI;

function qe<T extends HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

interface ConversionResult {
  lutName: string;
  dataUrl: string;
  selected: boolean;
}

let rawFilePath: string | null = null;
let rawFileName: string | null = null;
let lutFilePaths: string[] = [];
let lutNames: string[] = [];
let results: ConversionResult[] = [];

function showErrorModal(msg: string): void {
  const modal = qe<HTMLDivElement>("error-modal");
  const msgEl = qe<HTMLParagraphElement>("error-modal-message");
  if (modal) modal.style.display = "flex";
  if (msgEl) msgEl.textContent = msg;
}

function hideErrorModal(): void {
  const modal = qe<HTMLDivElement>("error-modal");
  if (modal) modal.style.display = "none";
}

function setStatus(msg: string, isError = false): void {
  const el = qe<HTMLDivElement>("status-label");
  if (!el) return;
  el.textContent = msg;
  el.className = isError ? "status-label error" : "status-label";
}

function setProgress(pct: number): void {
  const el = qe<HTMLDivElement>("progress-fill");
  const bar = qe<HTMLDivElement>("progress-bar");
  if (el) el.style.width = `${pct}%`;
  if (bar) bar.style.display = pct < 100 ? "block" : "none";
}

function enableControls(enabled: boolean): void {
  const ids = ["btn-convert", "btn-export", "btn-apply-exposure"] as const;
  ids.forEach((id) => {
    const btn = qe<HTMLButtonElement>(id);
    if (btn) btn.disabled = !enabled;
  });
}

function checkConvertReady(): void {
  const btn = qe<HTMLButtonElement>("btn-convert");
  if (btn) btn.disabled = !(rawFilePath && lutFilePaths.length > 0);
}

async function openRawFile(): Promise<void> {
  const filePath = await api.dialog.openFile([
    { name: "RAW Images", extensions: ["arw", "dng", "nef", "cr2", "cr3", "raf", "orf", "rw2"] },
    { name: "All Files", extensions: ["*"] },
  ]);
  if (!filePath) return;
  rawFilePath = filePath;
  rawFileName = filePath.split(/[\\/]/).pop() ?? filePath;
  setStatus(`Selected: ${rawFileName}`);
  checkConvertReady();
}

async function selectLutFiles(): Promise<void> {
  const MAX_LUTS = 20;
  const selected: string[] = [];
  for (let i = 0; i < MAX_LUTS; i++) {
    const path = await api.dialog.openFile([
      { name: "Cube LUT", extensions: ["cube"] },
      { name: "All Files", extensions: ["*"] },
    ]);
    if (!path) break;
    selected.push(path);
    setStatus(`Selected ${selected.length} LUT file(s)\u2026`);
  }
  if (selected.length === 0) return;
  lutFilePaths = selected;
  lutNames = selected.map((p, idx) => p.split(/[\\/]/).pop()?.replace(".cube", "") ?? `lut_${idx}`);
  setStatus(`${lutNames.length} LUT(s) selected`);
  checkConvertReady();
}

async function startConversion(): Promise<void> {
  if (!rawFilePath || lutFilePaths.length === 0) return;
  enableControls(false);
  setProgress(5);
  setStatus("Reading RAW file\u2026");

  let imageBuffer: ArrayBuffer;
  try {
    imageBuffer = await api.fs.readFile(rawFilePath);
  } catch (err) {
    setStatus(`Failed to read file: ${(err as Error).message}`, true);
    enableControls(true);
    return;
  }

  setProgress(15);
  setStatus("Reading LUT files\u2026");

  let lutBuffers: ArrayBuffer[];
  try {
    lutBuffers = await Promise.all(lutFilePaths.map((p) => api.fs.readFile(p)));
  } catch (err) {
    setStatus(`Failed to read LUT: ${(err as Error).message}`, true);
    enableControls(true);
    return;
  }

  setProgress(25);
  setStatus("Sending to backend\u2026");

  let response: { results: Array<{ lut_name: string; image_base64_jpeg: string }> };
  try {
    response = await api.backend.convert({
      imageBuffer,
      imageName: rawFileName ?? "image",
      lutBuffers,
      lutNames,
    });
  } catch (err) {
    showErrorModal(`Backend error: ${(err as Error).message}`);
    enableControls(true);
    return;
  }

  setProgress(90);
  results = response.results.map((r) => ({
    lutName: r.lut_name,
    dataUrl: `data:image/jpeg;base64,${r.image_base64_jpeg}`,
    selected: false,
  }));

  displayResults();
  setProgress(100);
  setStatus(`Converted ${results.length} image(s)`);
  enableControls(true);
}

function displayResults(): void {
  const container = qe<HTMLDivElement>("thumbnail-container");
  if (!container) return;
  container.innerHTML = "";

  results.forEach((result, idx) => {
    const thumb = document.createElement("div");
    thumb.style.cssText = "flex:0 0 100px; cursor:pointer; text-align:center;";
    thumb.innerHTML = `
      <img src="${result.dataUrl}" alt="${result.lutName}"
           style="width:96px;height:64px;object-fit:cover;border-radius:4px;border:2px solid transparent;">
      <div style="font-size:0.7rem;margin-top:2px;color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${result.lutName}</div>
    `;
    thumb.addEventListener("click", () => showPreview(idx));
    container.appendChild(thumb);
  });

  if (results.length > 0) showPreview(0);
}

function showPreview(idx: number): void {
  const img = qe<HTMLImageElement>("preview-image");
  const noImg = qe<HTMLDivElement>("no-image-selected");
  if (!img || !noImg) return;

  const result = results[idx];
  img.src = result.dataUrl;
  img.style.display = "block";
  noImg.style.display = "none";

  const container = qe<HTMLDivElement>("thumbnail-container");
  container?.querySelectorAll("img").forEach((el, i) => {
    (el as HTMLElement).style.borderColor = i === idx ? "#4caf50" : "transparent";
  });
}

async function exportSelected(): Promise<void> {
  const selected = results.filter((r) => r.selected);
  if (selected.length === 0) {
    setStatus("No images selected for export.", true);
    return;
  }

  for (const result of selected) {
    const defaultName = rawFileName
      ? rawFileName.replace(/\.[^.]+$/, "") + "_" + result.lutName + ".jpg"
      : result.lutName + ".jpg";

    const savePath = await api.dialog.saveFile(defaultName);
    if (!savePath) continue;

    const base64 = result.dataUrl.split(",")[1];
    const binary = atob(base64);
    const buf = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buf[i] = binary.charCodeAt(i);

    try {
      await api.fs.writeFile(savePath, buf.buffer);
    } catch (err) {
      setStatus(`Export failed: ${(err as Error).message}`, true);
    }
  }
  setStatus(`Exported ${selected.length} file(s)`);
}

function updateExposureLabel(): void {
  const slider = qe<HTMLInputElement>("exposure-slider");
  const label = qe<HTMLSpanElement>("exposure-value");
  if (slider && label) {
    const ev = parseInt(slider.value) / 10;
    label.textContent = `${ev.toFixed(1)} EV`;
  }
}

function init(): void {
  const dot = qe<HTMLDivElement>("backend-dot");
  const lbl = qe<HTMLSpanElement>("backend-label");
  const ver = qe<HTMLSpanElement>("backend-version");

  (async () => {
    try {
      const health = await api.backend.health();
      if (dot) dot.className = "status-dot ok";
      if (lbl) lbl.textContent = `Backend ready — v${health.version}`;
      if (ver) ver.textContent = health.version;
    } catch {
      if (dot) dot.className = "status-dot error";
      if (lbl) lbl.textContent = "Backend unreachable";
    }
  })();

  const slider = qe<HTMLInputElement>("exposure-slider");
  slider?.addEventListener("input", updateExposureLabel);
  updateExposureLabel();

  const selectAll = qe<HTMLInputElement>("select-all-cb");
  selectAll?.addEventListener("change", () => {
    results.forEach((r) => { r.selected = selectAll.checked; });
  });

  qe<HTMLButtonElement>("btn-open-raw")?.addEventListener("click", openRawFile);
  qe<HTMLButtonElement>("btn-select-luts")?.addEventListener("click", selectLutFiles);
  qe<HTMLButtonElement>("btn-convert")?.addEventListener("click", startConversion);
  qe<HTMLButtonElement>("btn-export")?.addEventListener("click", exportSelected);
  qe<HTMLButtonElement>("btn-apply-exposure")?.addEventListener("click", startConversion);

  qe<HTMLButtonElement>("btn-retry-conversion")?.addEventListener("click", () => {
    hideErrorModal();
    startConversion();
  });

  qe<HTMLButtonElement>("btn-dismiss-error")?.addEventListener("click", hideErrorModal);

  // Test hook: allows Playwright tests to inject error states
  (window as typeof window & Record<"__showTestErrorModal", typeof showErrorModal>).__showTestErrorModal = showErrorModal;

  qe<HTMLInputElement>("raw-file-input")?.addEventListener("change", () => {
    const input = qe<HTMLInputElement>("raw-file-input");
    if (input?.files?.[0]) setStatus(`Browser file: ${input.files[0].name}`);
  });
  qe<HTMLInputElement>("lut-files-input")?.addEventListener("change", () => {
    const input = qe<HTMLInputElement>("lut-files-input");
    if (input?.files?.length) setStatus(`${input.files.length} LUT file(s) selected`);
  });
}

document.addEventListener("DOMContentLoaded", init);
