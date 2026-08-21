/**
 * Tangential smoothing of a spherical chart: a few steps of uniform Laplacian
 * smoothing on the sphere (each vertex moves toward the normalized mean of its
 * neighbours). It removes the sub-triangle-scale kinks that SDEM's local fold
 * repairs leave behind while keeping the large-scale area distribution; callers
 * repair folds afterwards.
 */
import type { Mesh } from '../mesh/types.ts';
import { vertexAdjacency } from './meshops.ts';

export function smoothChart(m: Mesh, S: Float64Array, iterations = 5, alpha = 0.5): void {
  const adj = vertexAdjacency(m);
  const next = new Float64Array(S.length);
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < m.nv; i++) {
      let x = 0, y = 0, z = 0, n = 0;
      for (let p = adj.ptr[i]; p < adj.ptr[i + 1]; p++) { const j = adj.idx[p]; x += S[3 * j]; y += S[3 * j + 1]; z += S[3 * j + 2]; n++; }
      x = (1 - alpha) * S[3 * i] + (alpha * x) / n; y = (1 - alpha) * S[3 * i + 1] + (alpha * y) / n; z = (1 - alpha) * S[3 * i + 2] + (alpha * z) / n;
      const r = Math.hypot(x, y, z) || 1;
      next[3 * i] = x / r; next[3 * i + 1] = y / r; next[3 * i + 2] = z / r;
    }
    S.set(next);
  }
}
