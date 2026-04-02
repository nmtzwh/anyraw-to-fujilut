/**
 * ipc.ts — HTTP proxy layer between the Electron main process and the Python FastAPI backend.
 *
 * Architecture: the renderer NEVER makes raw HTTP calls. Instead it calls into the
 * preload bridge (contextBridge). The preload calls ipcRenderer.invoke(), which routes to
 * main. Main makes the HTTP request to Python at http://127.0.0.1:19876 and returns
 * the parsed response to the renderer.
 *
 * This keeps the renderer sandboxed (no direct network) and lets the main process
 * own all external communication.
 */

import { ipcMain, shell, dialog, IpcMainInvokeEvent, OpenDialogOptions } from "electron";

export const BACKEND_ORIGIN = "http://127.0.0.1:19876";
export const HEALTH_PATH = "/health";
export const CONVERT_PATH = "/convert";

/** Approved file paths — only files selected via dialogs can be read/written. */
const approvedReadPaths = new Set<string>();
const approvedWritePaths = new Set<string>();

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ConvertResult {
  lut_name: string;
  image_base64_jpeg: string;
}

export interface ConvertResponse {
  results: ConvertResult[];
}

/** Fetch helpers — all network calls stay in the main process. */

async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = `${BACKEND_ORIGIN}${path}`;
  const resp = await fetch(url, init);
  if (!resp.ok) {
    throw new Error(`Backend request failed: ${resp.status} ${resp.statusText} — ${path}`);
  }
  return resp;
}

async function backendGet<T>(path: string): Promise<T> {
  const resp = await backendFetch(path);
  return resp.json() as Promise<T>;
}

async function backendPost<T>(path: string, body: unknown): Promise<T> {
  const resp = await backendFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return resp.json() as Promise<T>;
}

/** Wire up all IPC handlers. Call once from main.ts during app startup. */
export function registerIpcHandlers(): void {
  // ── Test-mode error injection (Playwright) ─────────────────────────────────
  let simulateConvertError = false;
  ipcMain.handle("backend:simulateError", (_ev: IpcMainInvokeEvent, enabled: boolean) => {
    simulateConvertError = enabled;
  });

  // ── Health check ────────────────────────────────────────────────────────────
  ipcMain.handle("backend:health", async (): Promise<HealthResponse> => {
    try {
      return await backendGet<HealthResponse>(HEALTH_PATH);
    } catch (err) {
      throw new Error(`Backend unreachable: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  // ── Convert a RAW image through N LUTs ──────────────────────────────────────
  // Renderer sends ArrayBuffer chunks keyed by filename. Main assembles a
  // multipart/form-data POST and forwards to the Python backend.
  ipcMain.handle(
    "backend:convert",
    async (
      _ev: IpcMainInvokeEvent,
      payload: { 
        imageBuffer: ArrayBuffer; 
        imageName: string; 
        lutBuffers: ArrayBuffer[]; 
        lutNames: string[];
        preview?: boolean;
        evOffset?: number;
        include_original?: boolean;
      }
    ): Promise<ConvertResponse> => {
      if (simulateConvertError) {
        throw new Error("Simulated backend error — inject backend:simulateError(false) to disable");
      }
      const { imageBuffer, imageName, lutBuffers, lutNames, preview = true, evOffset = 0, include_original = false } = payload;

      const form = new FormData();
      const imageBlob = new Blob([imageBuffer], { type: "application/octet-stream" });
      form.append("image", imageBlob, imageName);
      form.append("preview", preview ? "true" : "false");
      form.append("ev_offset", evOffset.toString());
      form.append("include_original", include_original ? "true" : "false");

      for (let i = 0; i < lutBuffers.length; i++) {
        const lutBlob = new Blob([lutBuffers[i]], { type: "text/plain" });
        form.append("luts", lutBlob, lutNames[i] ?? `lut_${i}.cube`);
      }

      const resp = await fetch(`${BACKEND_ORIGIN}${CONVERT_PATH}`, { method: "POST", body: form });
      if (!resp.ok) {
        throw new Error(`Convert failed: ${resp.status} ${resp.statusText}`);
      }
      return resp.json() as Promise<ConvertResponse>;
    }
  );

  // ── Convert by path (no file upload — uses backend cache) ────────────────
  ipcMain.handle(
    "backend:convertByPath",
    async (
      _ev: IpcMainInvokeEvent,
      payload: {
        imagePath: string;
        lutPaths: string[];
        preview?: boolean;
        evOffset?: number;
        includeOriginal?: boolean;
      }
    ): Promise<ConvertResponse> => {
      if (simulateConvertError) {
        throw new Error("Simulated backend error — inject backend:simulateError(false) to disable");
      }
      const { imagePath, lutPaths, preview = true, evOffset = 0, includeOriginal = false } = payload;

      const resp = await fetch(`${BACKEND_ORIGIN}/convert-by-path`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: imagePath,
          lut_paths: lutPaths,
          preview,
          ev_offset: evOffset,
          include_original: includeOriginal,
        }),
      });
      if (!resp.ok) {
        throw new Error(`ConvertByPath failed: ${resp.status} ${resp.statusText}`);
      }
      return resp.json() as Promise<ConvertResponse>;
    }
  );

  ipcMain.handle(
    "backend:export",
    async (
      _ev: IpcMainInvokeEvent,
      payload: { imagePath: string; lutPaths: string[]; outputDir: string; evOffset: number }
    ): Promise<{ count: number; message: string }> => {
      const { imagePath, lutPaths, outputDir, evOffset } = payload;
      const resp = await fetch(`${BACKEND_ORIGIN}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: imagePath,
          lut_paths: lutPaths,
          output_dir: outputDir,
          ev_offset: evOffset,
        }),
      });
      if (!resp.ok) {
        throw new Error(`Export failed: ${resp.status} ${resp.statusText}`);
      }
      return resp.json() as Promise<{ count: number; message: string }>;
    }
  );

  // ── File dialogs (proxied so renderer can't bypass sandbox) ───────────────────
  ipcMain.handle("dialog:openFile", async (_ev: IpcMainInvokeEvent, filters?: OpenDialogOptions["filters"], allowMultiple?: boolean): Promise<string | string[] | null> => {
    const props: OpenDialogOptions["properties"] = ["openFile"];
    if (allowMultiple) props.push("multiSelections");
    const result = await dialog.showOpenDialog({
      properties: props,
      filters: filters ?? [
        { name: "RAW Images", extensions: ["arw", "dng", "nef", "cr2", "cr3", "raf", "orf", "rw2"] },
        { name: "All Files", extensions: ["*"] },
      ],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    for (const p of result.filePaths) approvedReadPaths.add(p);
    return allowMultiple ? result.filePaths : result.filePaths[0];
  });

  ipcMain.handle(
    "dialog:saveFile",
    async (_ev: IpcMainInvokeEvent, defaultPath?: string): Promise<string | null> => {
      const result = await dialog.showSaveDialog({
        defaultPath,
        filters: [
          { name: "JPEG Image", extensions: ["jpg", "jpeg"] },
          { name: "TIFF Image", extensions: ["tif", "tiff"] },
        ],
      });
      if (result.canceled || !result.filePath) return null;
      approvedWritePaths.add(result.filePath);
      return result.filePath;
    }
  );

  ipcMain.handle("dialog:selectDirectory", async (): Promise<string | null> => {
    const result = await dialog.showOpenDialog({
      properties: ["openDirectory"],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  // ── Read Directory for RAW files ──────────────────────────────────────────
  ipcMain.handle("fs:readDir", async (_ev: IpcMainInvokeEvent, dirPath: string): Promise<string[]> => {
    const fs = await import("fs/promises");
    const path = await import("path");
    const files = await fs.readdir(dirPath);
    const validExts = new Set([".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2"]);
    const rawFiles = [];
    for (const f of files) {
      if (validExts.has(path.extname(f).toLowerCase())) {
        const full = path.join(dirPath, f);
        approvedReadPaths.add(full);
        rawFiles.push(full);
      }
    }
    return rawFiles;
  });

  // ── Read file as ArrayBuffer (renderer can't access fs directly) ───────────
  ipcMain.handle("fs:readFile", async (_ev: IpcMainInvokeEvent, filePath: string): Promise<ArrayBuffer> => {
    if (!approvedReadPaths.has(filePath)) {
      throw new Error("Forbidden: file path not approved by user dialog");
    }
    const fs = await import("fs/promises");
    const buf = await fs.readFile(filePath);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  });

  // ── Re-approve paths from previous session ────────────────────────────────
  ipcMain.handle("fs:approveReadPaths", (_ev: IpcMainInvokeEvent, paths: string[]): void => {
    for (const p of paths) {
      if (typeof p === "string" && p.length > 0) {
        approvedReadPaths.add(p);
      }
    }
  });

  // ── Write bytes to a path (for exporting results) ───────────────────────────
  ipcMain.handle("fs:writeFile", async (_ev: IpcMainInvokeEvent, path: string, buffer: ArrayBuffer): Promise<void> => {
    if (!approvedWritePaths.has(path)) {
      throw new Error("Forbidden: file path not approved by save dialog");
    }
    const fs = await import("fs/promises");
    await fs.writeFile(path, Buffer.from(buffer));
  });

  // ── Open external URL in system browser ──────────────────────────────────────
  ipcMain.handle("shell:openExternal", async (_ev: IpcMainInvokeEvent, url: string): Promise<void> => {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      throw new Error(`Forbidden URL scheme: ${parsed.protocol}`);
    }
    await shell.openExternal(url);
  });
}
