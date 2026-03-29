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

let currentZoom = 1;
let isPanning = false;
let startX = 0, startY = 0;
let panX = 0, panY = 0;
let originalDataUrl: string | null = null;
let currentFolderImages: string[] = [];
let fetchThumbnailsGeneration = 0;

function resetZoom() {
  currentZoom = 1;
  panX = 0; panY = 0;
  updateZoomTransform();
}

function updateZoomTransform() {
  const img = qe<HTMLImageElement>("preview-image");
  if (img) img.style.transform = `translate(${panX}px, ${panY}px) scale(${currentZoom})`;
}

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
  selectRawFile(filePath);
}

function selectRawFile(filePath: string) {
  rawFilePath = filePath;
  rawFileName = filePath.split(/[\\/]/).pop() ?? filePath;
  setStatus(`Selected: ${rawFileName}`);
  checkConvertReady();
  if (lutFilePaths.length > 0) startConversion();
}

async function openFolder(): Promise<void> {
  const dirPath = await api.dialog.selectDirectory();
  if (!dirPath) return;

  const files = await api.fs.readDir(dirPath);
  currentFolderImages = files.sort();
  
  const filmstrip = qe<HTMLDivElement>("filmstrip-container");
  if (filmstrip) {
    if (files.length > 0) {
      filmstrip.style.display = "flex";
      filmstrip.innerHTML = "";
      files.forEach((f: string, i: number) => {
        const item = document.createElement("div");
        item.className = "filmstrip-item";
        item.innerHTML = `<img src="" data-path="${f}" style="display:none;"><div class="filmstrip-label">${f.split(/[\\/]/).pop()}</div>`;
        item.addEventListener("click", () => {
          document.querySelectorAll('.filmstrip-item').forEach(el => el.classList.remove('active'));
          item.classList.add('active');
          selectRawFile(f);
        });
        filmstrip.appendChild(item);
      });
      selectRawFile(files[0]);
      filmstrip.children[0]?.classList.add('active');
      
      // Kick off background thumbnails
      fetchThumbnailsGeneration++;
      fetchThumbnails(files, fetchThumbnailsGeneration);

    } else {
      filmstrip.style.display = "none";
      setStatus("No RAW files found in directory.", true);
    }
  }
}

async function fetchThumbnails(files: string[], generation: number) {
  const filmstrip = qe<HTMLDivElement>("filmstrip-container");
  for (const file of files) {
      if (fetchThumbnailsGeneration !== generation) return; // Abort if a new folder is opened
      
      try {
          const imgBuffer = await api.fs.readFile(file);
          const response = await api.backend.convert({
              imageBuffer: imgBuffer,
              imageName: file.split(/[\\/]/).pop()!,
              lutBuffers: [],
              lutNames: [],
              preview: true,
              evOffset: 0,
              include_original: true
          });
          
          if (fetchThumbnailsGeneration !== generation) return;
          if (response && response.results.length > 0) {
              const dataUrl = `data:image/jpeg;base64,${response.results[0].image_base64_jpeg}`;
              
              if (filmstrip) {
                  const imgEl = filmstrip.querySelector(`img[data-path="${file.replace(/\\/g, '\\\\')}"]`) as HTMLImageElement;
                  if (imgEl) {
                      imgEl.src = dataUrl;
                      imgEl.style.display = "block";
                  }
              }
          }
      } catch (e) {
          console.warn(`Failed to fetch thumbnail for ${file}`);
      }
  }
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

  // Initialize results
  const selectAllCb = qe<HTMLInputElement>("select-all-cb");
  const initiallySelected = selectAllCb ? selectAllCb.checked : true;
  
  results = lutNames.map(name => ({
    lutName: name,
    dataUrl: "",
    selected: initiallySelected,
    loading: true
  }));
  
  displayResults(); 
  originalDataUrl = null;
  resetZoom();

  // Fetch Original Base Image first
  try {
      const response = await api.backend.convert({
        imageBuffer,
        imageName: rawFileName ?? "image",
        lutBuffers: [],
        lutNames: [],
        preview: true,
        evOffset: currentEvOffset,
        include_original: true
      });
      if (response && response.results.length > 0) {
          originalDataUrl = `data:image/jpeg;base64,${response.results[0].image_base64_jpeg}`;
      }
  } catch(e) { console.error("Could not fetch original image", e); }

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

function drawHistogram(dataUrl: string) {
  const canvas = qe<HTMLCanvasElement>("histogram-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  
  const img = new Image();
  img.onload = () => {
    const off = document.createElement("canvas");
    off.width = 128; off.height = 128;
    const octx = off.getContext("2d");
    if (!octx) return;
    octx.drawImage(img, 0, 0, 128, 128);
    const data = octx.getImageData(0,0,128,128).data;
    
    const histR = new Array(256).fill(0);
    const histG = new Array(256).fill(0);
    const histB = new Array(256).fill(0);
    
    let maxCount = 0;
    for (let i = 0; i < data.length; i += 4) {
      histR[data[i]]++;
      histG[data[i+1]]++;
      histB[data[i+2]]++;
      maxCount = Math.max(maxCount, histR[data[i]], histG[data[i+1]], histB[data[i+2]]);
    }
    
    ctx.clearRect(0,0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'screen';
    
    const w = canvas.width / 256;
    
    const drawChannel = (hist: number[], color: string) => {
        ctx.fillStyle = color;
        for (let i = 0; i < 256; i++) {
            const h = (hist[i] / maxCount) * canvas.height;
            ctx.fillRect(i * w, canvas.height - h, Math.ceil(w), h);
        }
    };

    drawChannel(histR, "rgba(255, 0, 0, 0.6)");
    drawChannel(histG, "rgba(0, 255, 0, 0.6)");
    drawChannel(histB, "rgba(0, 0, 255, 0.6)");
    
    ctx.globalCompositeOperation = 'source-over';
  };
  img.src = dataUrl;
}

function updateViewModeUI() {
  const singleBtn = qe<HTMLButtonElement>("btn-view-single");
  const gridBtn = qe<HTMLButtonElement>("btn-view-grid");
  const previewImgWrapper = qe<HTMLDivElement>("preview-image")?.parentElement;
  const gridCont = qe<HTMLDivElement>("grid-container");
  const noImg = qe<HTMLDivElement>("no-image-selected");

  if (currentViewMode === 'single') {
      singleBtn?.classList.add("active");
      gridBtn?.classList.remove("active");
      if (gridCont) gridCont.style.display = "none";
      
      if (results.length > 0) {
          if (previewImgWrapper) previewImgWrapper.style.display = "flex";
          if (noImg) noImg.style.display = "none";
      } else {
          if (previewImgWrapper) previewImgWrapper.style.display = "none";
          if (noImg) noImg.style.display = "flex";
      }
  } else {
      singleBtn?.classList.remove("active");
      gridBtn?.classList.add("active");
      if (previewImgWrapper) previewImgWrapper.style.display = "none";
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
    drawHistogram(result.dataUrl);
    
    const toggleBtn = qe<HTMLDivElement>("btn-toggle-original");
    if (toggleBtn && originalDataUrl) {
        toggleBtn.style.display = "block";
    }
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
  qe<HTMLButtonElement>("btn-open-folder")?.addEventListener("click", openFolder);
  qe<HTMLButtonElement>("btn-select-luts")?.addEventListener("click", () => selectLutFiles());
  qe<HTMLButtonElement>("btn-convert")?.addEventListener("click", () => {
      resetZoom();
      startConversion();
  });
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

  const toggleBtn = qe<HTMLDivElement>("btn-toggle-original");
  const previewImg = qe<HTMLImageElement>("preview-image");
  if (previewImg) previewImg.ondragstart = () => false;

  if (toggleBtn && previewImg) {
      const showOrig = () => { if (originalDataUrl && !results[selectedResultIdx]?.loading) { previewImg.src = originalDataUrl; drawHistogram(originalDataUrl); } };
      const showLut = () => { if (!results[selectedResultIdx]?.loading) { previewImg.src = results[selectedResultIdx].dataUrl; drawHistogram(results[selectedResultIdx].dataUrl); } };
      toggleBtn.addEventListener("mousedown", showOrig);
      toggleBtn.addEventListener("mouseup", showLut);
      toggleBtn.addEventListener("mouseleave", showLut);
      toggleBtn.addEventListener("touchstart", showOrig);
      toggleBtn.addEventListener("touchend", showLut);
  }

  const imgWrapper = qe<HTMLDivElement>("preview-image")?.parentElement;
  if (imgWrapper) {
      imgWrapper.style.touchAction = "none";
      imgWrapper.addEventListener("wheel", (e) => {
          if (currentViewMode !== 'single') return;
          e.preventDefault();
          const zoomSensitivity = 0.002;
          currentZoom = Math.max(0.5, Math.min(5, currentZoom - e.deltaY * zoomSensitivity));
          updateZoomTransform();
      });

      imgWrapper.addEventListener("pointerdown", (e) => {
          if (currentViewMode !== 'single') return;
          isPanning = true;
          startX = e.clientX - panX;
          startY = e.clientY - panY;
          imgWrapper.setPointerCapture(e.pointerId);
      });

      imgWrapper.addEventListener("pointermove", (e) => {
          if (!isPanning) return;
          panX = e.clientX - startX;
          panY = e.clientY - startY;
          updateZoomTransform();
      });

      imgWrapper.addEventListener("pointerup", (e) => {
          isPanning = false;
          imgWrapper.releasePointerCapture(e.pointerId);
      });
      imgWrapper.addEventListener("pointercancel", (e) => {
          isPanning = false;
      });
  }
}

document.addEventListener("DOMContentLoaded", init);

