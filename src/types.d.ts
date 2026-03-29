/**
 * types.d.ts — Shared TypeScript type declarations for the Electron renderer.
 * Exposes the electronAPI injected by preload.ts via contextBridge.
 */

export interface ConvertPayload {
  imageBuffer: ArrayBuffer;
  imageName: string;
  lutBuffers: ArrayBuffer[];
  lutNames: string[];
  preview?: boolean;
  evOffset?: number;
  include_original?: boolean;
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
    readDir: (path: string) => Promise<string[]>;
    readFile: (path: string) => Promise<ArrayBuffer>;
    writeFile: (path: string, buffer: ArrayBuffer) => Promise<void>;
    approveReadPaths: (paths: string[]) => Promise<void>;
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
