/**
 * color.ts — Color space conversion stubs.
 * RAW decoding is handled by the Python backend (rawpy). These stubs are here
 * so build:renderer succeeds without depending on the full color.ts from gui-dev.
 * Full implementation: see gui-dev worktree src/color.ts.
 */

export function applyFlog2Curve(_linear_data: Float32Array): Float32Array {
  return new Float32Array(0);
}

export function xyzToRec2020(_xyz_image: Float32Array): Float32Array {
  return new Float32Array(0);
}

export function getExposureGain(
  _xyz_image: Float32Array,
  _target_grey?: number,
  _ev_offset?: number
): number {
  return 1.0;
}
