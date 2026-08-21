/**
 * Fold repair (repair_spherical_folds.m): vertices of flipped spherical
 * triangles, grown by one ring, are moved to the normalized average of their
 * neighbours until no triangle is flipped. Only the fold neighbourhoods move.
 */
import type { Mesh } from '../mesh/types.ts';
import { vertexAdjacency } from './meshops.ts';

export function repairSphericalFolds(m: Mesh, S: Float64Array, maxIt = 200): { folds: number; passes: number } {
  const { faces: f, nf, nv } = m;
  const adj = vertexAdjacency(m);
  const flipped = (): Uint8Array => {
    const vol = new Float64Array(nf);
    let pos = 0;
    for (let t = 0; t < nf; t++) {
      const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
      const bx = S[3 * b], by = S[3 * b + 1], bz = S[3 * b + 2], cx = S[3 * c], cy = S[3 * c + 1], cz = S[3 * c + 2];
      vol[t] = S[3 * a] * (by * cz - bz * cy) + S[3 * a + 1] * (bz * cx - bx * cz) + S[3 * a + 2] * (bx * cy - by * cx);
      if (vol[t] > 0) pos++;
    }
    const maj = pos >= nf / 2 ? 1 : -1;
    const bad = new Uint8Array(nf);
    for (let t = 0; t < nf; t++) if (Math.sign(vol[t]) !== maj) bad[t] = 1;
    return bad;
  };
  let passes = 0;
  for (;;) {
    const bad = flipped();
    let nbad = 0; for (let t = 0; t < nf; t++) nbad += bad[t];
    if (nbad === 0 || passes === maxIt) return { folds: nbad, passes };
    const touch = new Uint8Array(nv);
    for (let t = 0; t < nf; t++) if (bad[t]) for (let k = 0; k < 3; k++) touch[f[3 * t + k]] = 1;
    const grown = touch.slice();
    for (let i = 0; i < nv; i++) if (touch[i]) for (let p = adj.ptr[i]; p < adj.ptr[i + 1]; p++) grown[adj.idx[p]] = 1;
    const next = S.slice();
    for (let i = 0; i < nv; i++) {
      if (!grown[i]) continue;
      let x = 0, y = 0, z = 0;
      for (let p = adj.ptr[i]; p < adj.ptr[i + 1]; p++) { const j = adj.idx[p]; x += S[3 * j]; y += S[3 * j + 1]; z += S[3 * j + 2]; }
      const r = Math.hypot(x, y, z) || 1;
      next[3 * i] = x / r; next[3 * i + 1] = y / r; next[3 * i + 2] = z / r;
    }
    S.set(next);
    passes++;
  }
}
