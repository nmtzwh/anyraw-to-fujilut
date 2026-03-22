/**
 * preload.ts — Secure IPC bridge exposed to the renderer via contextBridge.
 *
 * Security model:
 * - Renderer is sandboxed (nodeIntegration: false, contextIsolation: true).
 * - Only the API surface declared below is reachable from the renderer.
 * - No raw HTTP, no filesystem, no subprocess access from the renderer.
 */

import { contextBridge, ipcRenderer } from "electron";

export interface ConvertPayload {
  imageBuffer: ArrayBuffer;
  imageName: string;
  lutBuffers: ArrayBuffer[];
  lutNames: string[];
  preview?: boolean;
  evOffset?: number;
}

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

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface ElectronAPI {
  backend: {
    health: () => Promise<HealthResponse>;
    convert: (payload: ConvertPayload) => Promise<ConvertResponse>;
    export: (payload: { imagePath: string; lutPaths: string[]; outputDir: string; evOffset: number }) => Promise<{ count: number; message: string }>;
    simulateError: (enabled: boolean) => Promise<void>;
  };
  dialog: {
    openFile: (filters?: FileFilter[], allowMultiple?: boolean) => Promise<string | string[] | null>;
    saveFile: (defaultPath?: string) => Promise<string | null>;
    selectDirectory: () => Promise<string | null>;
  };
  fs: {
    readFile: (path: string) => Promise<ArrayBuffer>;
    writeFile: (path: string, buffer: ArrayBuffer) => Promise<void>;
  };
  shell: {
    openExternal: (url: string) => Promise<void>;
  };
}

const api: ElectronAPI = {
  backend: {
    health: () => ipcRenderer.invoke("backend:health"),
    convert: (payload: ConvertPayload) => ipcRenderer.invoke("backend:convert", payload),
    export: (payload: { imagePath: string; lutPaths: string[]; outputDir: string }) => ipcRenderer.invoke("backend:export", payload),
    simulateError: (enabled: boolean) => ipcRenderer.invoke("backend:simulateError", enabled),
  },
  dialog: {
    openFile: (filters?: FileFilter[], allowMultiple?: boolean) => ipcRenderer.invoke("dialog:openFile", filters, allowMultiple),
    saveFile: (defaultPath?: string) => ipcRenderer.invoke("dialog:saveFile", defaultPath),
    selectDirectory: () => ipcRenderer.invoke("dialog:selectDirectory"),
  },
  fs: {
    readFile: (path: string) => ipcRenderer.invoke("fs:readFile", path) as Promise<ArrayBuffer>,
    writeFile: (path: string, buffer: ArrayBuffer) =>
      ipcRenderer.invoke("fs:writeFile", path, buffer),
  },
  shell: {
    openExternal: (url: string) => ipcRenderer.invoke("shell:openExternal", url),
  },
};

contextBridge.exposeInMainWorld("electronAPI", api);
