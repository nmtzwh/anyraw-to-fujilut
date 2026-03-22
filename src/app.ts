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
  loading?: boolean;
}

let rawFilePath: string | null = null;
let rawFileName: string | null = null;
let lutFilePaths: string[] = [];
let lutNames: string[] = [];
let results: ConversionResult[] = [];
let currentEvOffset = 0;
let currentViewMode: 'single' | 'grid' = 'single';
let selectedResultIdx = 0;

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
  const ids = ["btn-convert", "btn-export", "btn-apply-exposure", "btn-open-raw", "btn-select-luts"] as const;
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
  const selected = await api.dialog.openFile(
    [{ name: "RAW Images", extensions: ["arw", "dng", "nef", "cr2", "cr3", "raf", "orf", "rw2"] }, { name: "All Files", extensions: ["*"] }],
    false,
  );
  if (!selected) return;
  const filePath = Array.isArray(selected) ? selected[0] : selected;
  rawFilePath = filePath;
  rawFileName = filePath.split(/[\\/]/).pop() ?? filePath;
  setStatus(`Selected: ${rawFileName}`);
  checkConvertReady();
}

async function selectLutFiles(paths?: string[]): Promise<void> {
  const selected = paths ?? await api.dialog.openFile(
    [{ name: "Cube LUT", extensions: ["cube"] }, { name: "All Files", extensions: ["*"] }],
    true,
  );
  if (!selected || (Array.isArray(selected) && selected.length === 0)) return;

  const finalPaths = Array.isArray(selected) ? selected : [selected];
  lutFilePaths = finalPaths;
  lutNames = finalPaths.map((p, idx) => p.split(/[\\/]/).pop()?.replace(/\.cube$/i, "") ?? `lut_${idx}`);
  
  // Persist paths
  localStorage.setItem('persistent-lut-paths', JSON.stringify(lutFilePaths));
  
  setStatus(`${lutNames.length} LUT(s) selected`);
  checkConvertReady();
}

async function startConversion(evOffset?: number): Promise<void> {
  if (!rawFilePath || lutFilePaths.length === 0) return;
  if (evOffset !== undefined) currentEvOffset = evOffset;
  
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

  setProgress(10);
  setStatus("Preparing LUTs\u2026");

  // Initialize results with placeholders
  const selectAllCb = qe<HTMLInputElement>("select-all-cb");
  const initiallySelected = selectAllCb ? selectAllCb.checked : true;
  
  results = lutNames.map(name => ({
    lutName: name,
    dataUrl: "",
    selected: initiallySelected,
    loading: true
  }));
  
  displayResults(); // Show placeholders immediately

  // Process one by one
  for (let i = 0; i < lutFilePaths.length; i++) {
    const path = lutFilePaths[i];
    const name = lutNames[i];
    
    setStatus(`Applying LUT ${i + 1}/${lutNames.length}: ${name}\u2026`);
    const progress = 10 + Math.floor((i / lutFilePaths.length) * 85);
    setProgress(progress);

    try {
      const lutBuffer = await api.fs.readFile(path);
      const response = await api.backend.convert({
        imageBuffer,
        imageName: rawFileName ?? "image",
        lutBuffers: [lutBuffer],
        lutNames: [name],
        preview: true,
        evOffset: currentEvOffset,
      } as any);

      if (response && Array.isArray(response.results) && response.results.length > 0) {
        const r = response.results[0];
        const res = results[i];
        res.lutName = r.lut_name;
        res.dataUrl = `data:image/jpeg;base64,${r.image_base64_jpeg}`;
        res.loading = false;
        updateResultUI(i);
      }
    } catch (err) {
      console.error(`Failed to apply LUT ${name}:`, err);
      results[i].loading = false;
      // We could show an error icon here, but for now just stop loading
      updateResultUI(i);
    }
  }

  setProgress(100);
  setStatus(`Generated ${results.length} preview(s)`);
  enableControls(true);
}

function updateResultUI(idx: number): void {
  const result = results[idx];
  
  // Update sidebar list item
  const lutList = qe<HTMLUListElement>("lut-list");
  const li = lutList?.children[idx] as HTMLElement | undefined;
  if (li) {
    li.className = result.loading ? "lut-item loading" : "lut-item";
    const img = li.querySelector("img");
    if (img) img.src = result.dataUrl;
  }

  // Update grid item
  const grid = qe<HTMLDivElement>("grid-container");
  const gridItem = grid?.children[idx] as HTMLElement | undefined;
  if (gridItem) {
    gridItem.className = result.loading ? "grid-item loading" : "grid-item";
    if (result.selected) gridItem.classList.add("export-selected");
    
    const img = gridItem.querySelector("img");
    if (img) {
      img.src = result.dataUrl;
      img.style.visibility = result.loading ? "hidden" : "visible";
    }
  }
  
  // If the currently selected item is done, refresh it
  if (idx === selectedResultIdx) {
      showPreview(idx);
  }
}

function updateViewModeUI() {
  const singleBtn = qe<HTMLButtonElement>("btn-view-single");
  const gridBtn = qe<HTMLButtonElement>("btn-view-grid");
  const previewImg = qe<HTMLImageElement>("preview-image");
  const gridCont = qe<HTMLDivElement>("grid-container");
  const noImg = qe<HTMLDivElement>("no-image-selected");

  if (currentViewMode === 'single') {
      singleBtn?.classList.add("active");
      gridBtn?.classList.remove("active");
      if (gridCont) gridCont.style.display = "none";
      
      if (results.length > 0) {
          if (previewImg) previewImg.style.display = "block";
          if (noImg) noImg.style.display = "none";
      } else {
          if (previewImg) previewImg.style.display = "none";
          if (noImg) noImg.style.display = "flex";
      }
  } else {
      singleBtn?.classList.remove("active");
      gridBtn?.classList.add("active");
      if (previewImg) previewImg.style.display = "none";
      if (noImg) noImg.style.display = "none";
      
      if (results.length > 0) {
          if (gridCont) gridCont.style.display = "grid";
      }
  }
}

function updateExportSelections() {
    const grid = qe<HTMLDivElement>("grid-container");
    grid?.querySelectorAll(".grid-item").forEach((el, i) => {
        if (results[i]?.selected) el.classList.add("export-selected");
        else el.classList.remove("export-selected");
    });

    const selectAllCb = qe<HTMLInputElement>("select-all-cb");
    if (selectAllCb) {
        selectAllCb.checked = results.length > 0 && results.every(r => r.selected);
        selectAllCb.indeterminate = !selectAllCb.checked && results.some(r => r.selected);
    }
}

function displayResults(): void {
  const grid = qe<HTMLDivElement>("grid-container");
  const lutList = qe<HTMLUListElement>("lut-list");
  const noLutsMsg = qe<HTMLDivElement>("no-luts-msg");
  const viewToggle = qe<HTMLDivElement>("view-toggle");

  if (viewToggle) viewToggle.style.display = results.length > 0 ? "flex" : "none";
  if (noLutsMsg) noLutsMsg.style.display = results.length > 0 ? "none" : "block";

  if (grid) grid.innerHTML = "";
  if (lutList) lutList.innerHTML = "";

  results.forEach((result, idx) => {
    const li = document.createElement("li");
    li.className = result.loading ? "lut-item loading" : "lut-item";
    li.innerHTML = `
        <img src="${result.dataUrl || ''}" alt="${result.lutName}">
        <span class="lut-item-name">${result.lutName}</span>
    `;
    li.addEventListener("click", () => {
        if (results[idx].loading) return;
        currentViewMode = 'single';
        updateViewModeUI();
        showPreview(idx);
    });
    lutList?.appendChild(li);

    const gridItem = document.createElement("div");
    gridItem.className = result.loading ? "grid-item loading" : "grid-item";
    if (result.selected) gridItem.classList.add("export-selected");
    
    gridItem.innerHTML = `
        <div class="grid-item-img-wrapper">
            <img src="${result.dataUrl || ''}" alt="${result.lutName}" style="${result.loading ? 'visibility:hidden' : ''}">
            <div class="grid-item-badge">${result.lutName}</div>
            <div class="grid-item-export-check"></div>
        </div>
    `;

    const imgWrapper = gridItem.querySelector("img");
    imgWrapper?.addEventListener("click", () => {
        if (results[idx].loading) return;
        currentViewMode = 'single';
        updateViewModeUI();
        showPreview(idx);
    });

    const exportCheck = gridItem.querySelector(".grid-item-export-check");
    exportCheck?.addEventListener("click", (e) => {
        e.stopPropagation();
        results[idx].selected = !results[idx].selected;
        updateExportSelections();
    });

    grid?.appendChild(gridItem);
  });

  if (results.length > 0) {
      showPreview(0);
      updateViewModeUI();
      updateExportSelections();
  }
}

function showPreview(idx: number): void {
  selectedResultIdx = idx;
  const img = qe<HTMLImageElement>("preview-image");
  const container = document.querySelector(".preview-content");
  if (!img || !container) return;

  const result = results[idx];
  if (!result) return;

  if (result.loading) {
    container.classList.add("loading");
    img.src = "";
    img.style.visibility = "hidden";
  } else {
    container.classList.remove("loading");
    img.src = result.dataUrl;
    img.style.visibility = "visible";
  }

  const lutList = qe<HTMLUListElement>("lut-list");
  lutList?.querySelectorAll(".lut-item").forEach((el, i) => {
    if (i === idx) el.classList.add("selected");
    else el.classList.remove("selected");
  });

  const grid = qe<HTMLDivElement>("grid-container");
  grid?.querySelectorAll(".grid-item").forEach((el, i) => {
    if (i === idx) el.classList.add("selected");
    else el.classList.remove("selected");
  });
}

async function exportSelected(): Promise<void> {
  const selectedIndices = results
    .map((r, i) => (r.selected && !r.loading ? i : -1))
    .filter((i) => i !== -1);

  if (selectedIndices.length === 0) {
    if (results.some(r => r.loading)) {
        setStatus("Please wait for conversions to finish.", true);
    } else {
        setStatus("No images selected for export.", true);
    }
    return;
  }

  if (!rawFilePath) return;

  const outputDir = await api.dialog.selectDirectory();
  if (!outputDir) return;

  enableControls(false);
  setProgress(5);
  setStatus(`Preparing to export ${selectedIndices.length} image(s)\u2026`);

  let exportedCount = 0;
  try {
    for (let i = 0; i < selectedIndices.length; i++) {
        const idx = selectedIndices[i];
        const lutPath = lutFilePaths[idx];
        const lutName = lutNames[idx];
        
        setStatus(`Exporting ${i + 1}/${selectedIndices.length}: ${lutName}\u2026`);
        
        await api.backend.export({
          imagePath: rawFilePath,
          lutPaths: [lutPath],
          outputDir,
          evOffset: currentEvOffset,
        });
        
        exportedCount++;
        setProgress(5 + Math.floor((exportedCount / selectedIndices.length) * 95));
    }
    setStatus(`Successfully exported ${exportedCount} image(s) to ${outputDir}`);
  } catch (err) {
    setStatus(`Export failed after ${exportedCount} image(s): ${(err as Error).message}`, true);
  } finally {
    setProgress(100);
    enableControls(true);
  }
}

function updateExposureLabel(): void {
  const slider = qe<HTMLInputElement>("exposure-slider");
  const label = qe<HTMLSpanElement>("exposure-value");
  if (slider && label) {
    const ev = parseInt(slider.value) / 10;
    label.textContent = `${ev > 0 ? '+' : ''}${ev.toFixed(1)} EV`;
  }
}

function applyExposure(): void {
  const slider = qe<HTMLInputElement>("exposure-slider");
  const evOffset = slider ? parseInt(slider.value) / 10 : 0;
  startConversion(evOffset);
}

function init(): void {
  const dot = qe<HTMLDivElement>("backend-dot");
  const lbl = qe<HTMLSpanElement>("backend-label");
  const ver = qe<HTMLSpanElement>("backend-version");

  (async () => {
    try {
      const health = await api.backend.health();
      if (dot) dot.className = "status-dot ok";
      if (lbl) lbl.textContent = `Backend ready`;
      if (ver) {
          ver.textContent = `v${health.version}`;
          ver.style.display = "inline";
      }
    } catch {
      if (dot) dot.className = "status-dot error";
      if (lbl) lbl.textContent = "Backend unreachable";
    }
    
    // Load persistent LUTs after healthcare check
    const savedLuts = localStorage.getItem('persistent-lut-paths');
    if (savedLuts) {
        try {
            const paths = JSON.parse(savedLuts);
            if (Array.isArray(paths) && paths.length > 0) {
                await api.fs.approveReadPaths(paths);
                selectLutFiles(paths);
            }
        } catch (e) {
            console.warn("Failed to load persistent LUTs:", e);
        }
    }
  })();

  const slider = qe<HTMLInputElement>("exposure-slider");
  slider?.addEventListener("input", updateExposureLabel);
  updateExposureLabel();

  const selectAll = qe<HTMLInputElement>("select-all-cb");
  selectAll?.addEventListener("change", () => {
    results.forEach((r) => { r.selected = selectAll.checked; });
    updateExportSelections();
  });

  qe<HTMLButtonElement>("btn-open-raw")?.addEventListener("click", openRawFile);
  qe<HTMLButtonElement>("btn-select-luts")?.addEventListener("click", () => selectLutFiles());
  qe<HTMLButtonElement>("btn-convert")?.addEventListener("click", () => startConversion());
  qe<HTMLButtonElement>("btn-export")?.addEventListener("click", exportSelected);
  qe<HTMLButtonElement>("btn-apply-exposure")?.addEventListener("click", applyExposure);
  qe<HTMLButtonElement>("btn-retry-conversion")?.addEventListener("click", () => {
    hideErrorModal();
    startConversion();
  });
  qe<HTMLButtonElement>("btn-dismiss-error")?.addEventListener("click", hideErrorModal);

  qe<HTMLButtonElement>("btn-view-single")?.addEventListener("click", () => {
      currentViewMode = 'single';
      updateViewModeUI();
  });
  qe<HTMLButtonElement>("btn-view-grid")?.addEventListener("click", () => {
      currentViewMode = 'grid';
      updateViewModeUI();
  });
}

document.addEventListener("DOMContentLoaded", init);

