/**
 * types.d.ts — Shared TypeScript type declarations for the Electron renderer.
 * Exposes the electronAPI injected by preload.ts via contextBridge.
 */

export interface ConvertPayload {
  imageBuffer: ArrayBuffer;
  imageName: string;
  lutBuffers: ArrayBuffer[];
  lutNames: string[];
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
    simulateError: (enabled: boolean) => Promise<void>;
  };
  dialog: {
    openFile: (filters?: FileFilter[]) => Promise<string | null>;
    saveFile: (defaultPath?: string) => Promise<string | null>;
  };
  fs: {
    readFile: (path: string) => Promise<ArrayBuffer>;
    writeFile: (path: string, buffer: ArrayBuffer) => Promise<void>;
  };
  shell: {
    openExternal: (url: string) => Promise<void>;
  };
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
