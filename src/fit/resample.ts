/**
 * Resample a per-vertex field of a charted mesh at query directions on the
 * sphere (port of interp_spherical_mesh.m): the continuous piecewise-linear
 * spherical signal of Zhou et al. Each query direction is located in the
 * spherical triangle containing it (gnomonic test: [a b c] w = p with w >= 0)
 * and the field is interpolated barycentrically.
 *
 * Triangle location: the chart is star-shaped about the origin, so the
 * containing triangle is (almost always) incident to the query's nearest chart
 * vertex. Nearest vertices come from a uniform grid hash over the unit cube;
 * misses widen to the 1-ring's triangles, then to a full sweep, then to the
 * nearest vertex's value.
 */
import type { Mesh } from '../mesh/types.ts';
import { vertexFaces, vertexAdjacency } from '../chart/meshops.ts';

export class SphereLocator {
  private readonly S: Float64Array;
  private readonly f: Uint32Array;
  private readonly vf: { ptr: Int32Array; idx: Int32Array };
  private readonly adj: { ptr: Int32Array; idx: Int32Array };
  private readonly G: number;
  private readonly cellHead: Int32Array;
  private readonly nextInCell: Int32Array;

  constructor(mesh: Mesh, chart: ArrayLike<number>) {
    const nv = mesh.nv;
    this.S = new Float64Array(3 * nv);
    for (let i = 0; i < nv; i++) {
      const r = Math.hypot(chart[3 * i], chart[3 * i + 1], chart[3 * i + 2]) || 1;
      for (let c = 0; c < 3; c++) this.S[3 * i + c] = chart[3 * i + c] / r;
    }
    this.f = mesh.faces;
    this.vf = vertexFaces(mesh);
    this.adj = vertexAdjacency(mesh);
    this.G = Math.max(4, Math.ceil(Math.sqrt(nv / 3)));
    this.cellHead = new Int32Array(this.G ** 3).fill(-1);
    this.nextInCell = new Int32Array(nv);
    for (let i = 0; i < nv; i++) {
      const c = this.cell(this.S[3 * i], this.S[3 * i + 1], this.S[3 * i + 2]);
      this.nextInCell[i] = this.cellHead[c]; this.cellHead[c] = i;
    }
  }

  private cell(x: number, y: number, z: number): number {
    const G = this.G, q = (t: number) => Math.min(G - 1, Math.max(0, Math.floor(((t + 1) / 2) * G)));
    return q(x) + G * (q(y) + G * q(z));
  }

  private nearest(x: number, y: number, z: number): number {
    const G = this.G;
    const q = (t: number) => Math.min(G - 1, Math.max(0, Math.floor(((t + 1) / 2) * G)));
    const cx = q(x), cy = q(y), cz = q(z);
    let best = -1, bestDot = -2;
    for (let rad = 1; rad <= G; rad++) {
      for (let i = Math.max(0, cx - rad); i <= Math.min(G - 1, cx + rad); i++)
        for (let j = Math.max(0, cy - rad); j <= Math.min(G - 1, cy + rad); j++)
          for (let k = Math.max(0, cz - rad); k <= Math.min(G - 1, cz + rad); k++) {
            if (rad > 1 && Math.abs(i - cx) < rad && Math.abs(j - cy) < rad && Math.abs(k - cz) < rad) continue; // shell only
            for (let v = this.cellHead[i + G * (j + G * k)]; v !== -1; v = this.nextInCell[v]) {
              const d = x * this.S[3 * v] + y * this.S[3 * v + 1] + z * this.S[3 * v + 2];
              if (d > bestDot) { bestDot = d; best = v; }
            }
          }
      if (best !== -1 && rad >= 1) {
        // a vertex at angular distance acos(bestDot) beats anything outside the searched shells once the shells cover that radius
        const covered = (rad * 2) / G; // chord covered by the searched cube (conservative)
        if (Math.sqrt(2 - 2 * bestDot) < covered) break;
      }
    }
    return best;
  }

  /** Barycentric weights of direction p in triangle t (gnomonic), or null if outside. */
  private inTriangle(t: number, px: number, py: number, pz: number, tol = -1e-12): [number, number, number] | null {
    const S = this.S, f = this.f;
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    // solve [Sa Sb Sc] w = p by Cramer's rule
    const m00 = S[3 * a], m10 = S[3 * a + 1], m20 = S[3 * a + 2];
    const m01 = S[3 * b], m11 = S[3 * b + 1], m21 = S[3 * b + 2];
    const m02 = S[3 * c], m12 = S[3 * c + 1], m22 = S[3 * c + 2];
    const det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12) + m02 * (m10 * m21 - m20 * m11);
    if (Math.abs(det) < 1e-14) return null;
    const w0 = (px * (m11 * m22 - m21 * m12) - m01 * (py * m22 - pz * m12) + m02 * (py * m21 - pz * m11)) / det;
    const w1 = (m00 * (py * m22 - pz * m12) - px * (m10 * m22 - m20 * m12) + m02 * (m10 * pz - m20 * py)) / det;
    const w2 = (m00 * (m11 * pz - m21 * py) - m01 * (m10 * pz - m20 * py) + px * (m10 * m21 - m20 * m11)) / det;
    if (w0 < tol || w1 < tol || w2 < tol) return null;
    const s = w0 + w1 + w2;
    return [w0 / s, w1 / s, w2 / s];
  }

  /**
   * Locate each query direction: containing triangle and barycentric weights
   * (face -1 with weights on the nearest vertex when nothing contains it).
   */
  locate(queries: ArrayLike<number>): { face: Int32Array; bary: Float64Array; fallbacks: number } {
    const m = queries.length / 3, face = new Int32Array(m).fill(-1), bary = new Float64Array(3 * m), f = this.f;
    let fallbacks = 0;
    const tried = new Int32Array(this.f.length / 3).fill(-1);
    for (let q = 0; q < m; q++) {
      let px = queries[3 * q], py = queries[3 * q + 1], pz = queries[3 * q + 2];
      const r = Math.hypot(px, py, pz) || 1; px /= r; py /= r; pz /= r;
      const v0 = this.nearest(px, py, pz);
      let hit: { t: number; w: [number, number, number] } | null = null;
      const tryVertex = (v: number): boolean => {
        for (let p = this.vf.ptr[v]; p < this.vf.ptr[v + 1]; p++) {
          const t = this.vf.idx[p];
          if (tried[t] === q) continue;
          tried[t] = q;
          const w = this.inTriangle(t, px, py, pz);
          if (w) { hit = { t, w }; return true; }
        }
        return false;
      };
      if (!tryVertex(v0)) {
        for (let p = this.adj.ptr[v0]; p < this.adj.ptr[v0 + 1] && !hit; p++) tryVertex(this.adj.idx[p]);
        if (!hit) {
          fallbacks++;
          for (let t = 0; t < f.length / 3 && !hit; t++) { const w = this.inTriangle(t, px, py, pz); if (w) hit = { t, w }; }
        }
      }
      if (hit) {
        const { t, w } = hit as { t: number; w: [number, number, number] };
        face[q] = t; bary[3 * q] = w[0]; bary[3 * q + 1] = w[1]; bary[3 * q + 2] = w[2];
      } else {
        // nearest-vertex fallback: weight 1 on that vertex in one of its faces
        const t = this.vf.idx[this.vf.ptr[v0]];
        face[q] = t;
        for (let k = 0; k < 3; k++) bary[3 * q + k] = f[3 * t + k] === v0 ? 1 : 0;
      }
    }
    return { face, bary, fallbacks };
  }

  /**
   * Interpolate `field` (nv x d, interleaved) at the `queries` (m x 3, interleaved directions).
   * Returns m x d interleaved values plus the count of queries that needed the fallbacks.
   */
  interpolate(field: ArrayLike<number>, d: number, queries: ArrayLike<number>): { values: Float64Array; fallbacks: number } {
    const m = queries.length / 3, out = new Float64Array(m * d), f = this.f;
    let fallbacks = 0;
    const tried = new Int32Array(this.f.length / 3).fill(-1);
    for (let q = 0; q < m; q++) {
      let px = queries[3 * q], py = queries[3 * q + 1], pz = queries[3 * q + 2];
      const r = Math.hypot(px, py, pz) || 1; px /= r; py /= r; pz /= r;
      const v0 = this.nearest(px, py, pz);
      let hit: { t: number; w: [number, number, number] } | null = null;
      const tryVertex = (v: number): boolean => {
        for (let p = this.vf.ptr[v]; p < this.vf.ptr[v + 1]; p++) {
          const t = this.vf.idx[p];
          if (tried[t] === q) continue;
          tried[t] = q;
          const w = this.inTriangle(t, px, py, pz);
          if (w) { hit = { t, w }; return true; }
        }
        return false;
      };
      if (!tryVertex(v0)) {
        for (let p = this.adj.ptr[v0]; p < this.adj.ptr[v0 + 1] && !hit; p++) tryVertex(this.adj.idx[p]);
        if (!hit) {
          fallbacks++;
          for (let t = 0; t < f.length / 3 && !hit; t++) { const w = this.inTriangle(t, px, py, pz); if (w) hit = { t, w }; }
        }
      }
      if (hit) {
        const { t, w } = hit as { t: number; w: [number, number, number] };
        for (let c = 0; c < d; c++)
          out[q * d + c] = w[0] * field[f[3 * t] * d + c] + w[1] * field[f[3 * t + 1] * d + c] + w[2] * field[f[3 * t + 2] * d + c];
      } else {
        for (let c = 0; c < d; c++) out[q * d + c] = field[v0 * d + c];
      }
    }
    return { values: out, fallbacks };
  }
}
