/**
 * lut.ts — LUT parsing and application stubs for the renderer.
 * Full implementation: see gui-dev worktree src/lut.ts (WebGL2 3D LUT).
 * These stubs allow build:renderer to succeed without the parse-cube-lut browser dep.
 */

export interface LutData {
  title: string;
  size: number;
  domain: [number, number, number, number, number, number];
  table: number[][];
}

export function parseLut(_file: File): Promise<LutData> {
  return Promise.resolve({ title: "", size: 0, domain: [0, 1, 0, 1, 0, 1], table: [] });
}

export function applyLut(_image: ImageData, _lutData: LutData): Promise<ImageData> {
  return Promise.resolve(new ImageData(1, 1));
}
