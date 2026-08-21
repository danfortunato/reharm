/** Mesh operators shared by the charts (ports of the MATLAB helpers). */
import type { Mesh } from '../mesh/types.ts';
import { fromTriplets, type CSC } from './sparse.ts';

export const faceAreas = (f: Uint32Array, p: ArrayLike<number>, nf = f.length / 3): Float64Array => {
  const A = new Float64Array(nf);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const ux = p[3 * b] - p[3 * a], uy = p[3 * b + 1] - p[3 * a + 1], uz = p[3 * b + 2] - p[3 * a + 2];
    const vx = p[3 * c] - p[3 * a], vy = p[3 * c + 1] - p[3 * a + 1], vz = p[3 * c + 2] - p[3 * a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    A[t] = 0.5 * Math.hypot(nx, ny, nz);
  }
  return A;
};

/** Triplets of the cotangent Laplacian L (off-diagonal +cot, diagonal -sum): negative semidefinite. */
export function cotangentLaplacian(m: Mesh): CSC {
  const { faces: f, positions: p, nf, nv } = m;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  const len = (a: number, b: number) => Math.hypot(p[3 * a] - p[3 * b], p[3 * a + 1] - p[3 * b + 1], p[3 * a + 2] - p[3 * b + 2]);
  for (let t = 0; t < nf; t++) {
    const f1 = f[3 * t], f2 = f[3 * t + 1], f3 = f[3 * t + 2];
    const l1 = len(f2, f3), l2 = len(f3, f1), l3 = len(f1, f2);
    const s = (l1 + l2 + l3) / 2;
    const area = Math.sqrt(Math.max(s * (s - l1) * (s - l2) * (s - l3), 1e-300));
    const cot12 = (l1 * l1 + l2 * l2 - l3 * l3) / area / 2;
    const cot23 = (l2 * l2 + l3 * l3 - l1 * l1) / area / 2;
    const cot31 = (l1 * l1 + l3 * l3 - l2 * l2) / area / 2;
    I.push(f1, f2, f2, f3, f3, f1, f1, f2, f3);
    J.push(f2, f1, f3, f2, f1, f3, f1, f2, f3);
    V.push(cot12, cot12, cot23, cot23, cot31, cot31, -cot12 - cot31, -cot12 - cot23, -cot31 - cot23);
  }
  return fromTriplets(nv, I, J, V);
}

/** Tutte (uniform-weight) Laplacian: off-diagonal = number of faces sharing the edge / 2 ... summed to 1 per edge; diagonal -degree. */
export function tutteLaplacian(m: Mesh): CSC {
  const { faces: f, nf, nv } = m;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  for (let t = 0; t < nf; t++)
    for (let k = 0; k < 3; k++) {
      const a = f[3 * t + k], b = f[3 * t + ((k + 1) % 3)];
      I.push(a, b, a, b); J.push(b, a, a, b); V.push(0.5, 0.5, -0.5, -0.5);
    }
  return fromTriplets(nv, I, J, V);
}

/** Vertex -> incident faces (CSR). */
export function vertexFaces(m: Mesh): { ptr: Int32Array; idx: Int32Array } {
  const ptr = new Int32Array(m.nv + 1);
  for (let k = 0; k < m.faces.length; k++) ptr[m.faces[k] + 1]++;
  for (let i = 0; i < m.nv; i++) ptr[i + 1] += ptr[i];
  const idx = new Int32Array(ptr[m.nv]), next = ptr.slice(0, m.nv);
  for (let t = 0; t < m.nf; t++) for (let k = 0; k < 3; k++) idx[next[m.faces[3 * t + k]]++] = t;
  return { ptr, idx };
}

/** Vertex adjacency (CSR, symmetric, no self loops). */
export function vertexAdjacency(m: Mesh): { ptr: Int32Array; idx: Int32Array } {
  const L = tutteLaplacian(m);
  const ptr = new Int32Array(m.nv + 1), idx: number[] = [];
  for (let j = 0; j < m.nv; j++) {
    for (let p = L.colptr[j]; p < L.colptr[j + 1]; p++) if (L.rowidx[p] !== j) idx.push(L.rowidx[p]);
    ptr[j + 1] = idx.length;
  }
  return { ptr, idx: Int32Array.from(idx) };
}

/** The most regular (equilateral-like) triangle, used to puncture the sphere. */
export function mostRegularTriangle(m: Mesh): number {
  const { faces: f, positions: p } = m;
  let best = 0, bestReg = Infinity;
  const len = (a: number, b: number) => Math.hypot(p[3 * a] - p[3 * b], p[3 * a + 1] - p[3 * b + 1], p[3 * a + 2] - p[3 * b + 2]);
  for (let t = 0; t < m.nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const e1 = len(b, c), e2 = len(a, c), e3 = len(a, b), s = e1 + e2 + e3;
    const reg = Math.abs(e1 / s - 1 / 3) + Math.abs(e2 / s - 1 / 3) + Math.abs(e3 / s - 1 / 3);
    if (reg < bestReg) { bestReg = reg; best = t; }
  }
  return best;
}

/** Number of spherical triangles whose orientation disagrees with the majority. */
export function countFolds(f: Uint32Array, S: ArrayLike<number>): number {
  const nf = f.length / 3, vol = new Float64Array(nf);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const bx = S[3 * b], by = S[3 * b + 1], bz = S[3 * b + 2], cx = S[3 * c], cy = S[3 * c + 1], cz = S[3 * c + 2];
    vol[t] = S[3 * a] * (by * cz - bz * cy) + S[3 * a + 1] * (bz * cx - bx * cz) + S[3 * a + 2] * (bx * cy - by * cx);
  }
  let pos = 0; for (let t = 0; t < nf; t++) if (vol[t] > 0) pos++;
  const majority = pos >= nf / 2 ? 1 : -1;
  let bad = 0; for (let t = 0; t < nf; t++) if (Math.sign(vol[t]) !== majority) bad++;
  return bad;
}

/** Conformal-factor statistics: per-face sphere area / surface area. */
export function lambdaStats(m: Mesh, S: ArrayLike<number>): { spread: number; ratio: number; lam: Float64Array } {
  const A0 = faceAreas(m.faces, m.positions), A1 = faceAreas(m.faces, S);
  const lam = new Float64Array(m.nf);
  let s = 0, s2 = 0, lo = Infinity, hi = 0;
  for (let t = 0; t < m.nf; t++) {
    lam[t] = A1[t] / Math.max(A0[t], 1e-300);
    s += lam[t]; s2 += lam[t] * lam[t];
    if (lam[t] < lo) lo = lam[t]; if (lam[t] > hi) hi = lam[t];
  }
  const mean = s / m.nf, std = Math.sqrt(Math.max(s2 / m.nf - mean * mean, 0));
  return { spread: std / mean, ratio: hi / Math.max(lo, 1e-300), lam };
}

/** Solve K x = 0 on the free vertices with x prescribed on `fixed`: K_FF x_F = -K_FB x_B. K must be SPD on the free set. */
export function dirichletSolve(
  K: CSC, coords: ArrayLike<number>, fixed: ArrayLike<number>, fixedVals: ArrayLike<number>, nrhs: number,
  solver: (A: CSC, coords: ArrayLike<number>, b: ArrayLike<number>, nrhs: number) => Float64Array,
): Float64Array {
  const n = K.n;
  const isFixed = new Uint8Array(n), fval = new Float64Array(n * nrhs);
  for (let k = 0; k < fixed.length; k++) { isFixed[fixed[k]] = 1; for (let r = 0; r < nrhs; r++) fval[r * n + fixed[k]] = fixedVals[r * fixed.length + k]; }
  const newIdx = new Int32Array(n).fill(-1);
  let nfree = 0;
  for (let i = 0; i < n; i++) if (!isFixed[i]) newIdx[i] = nfree++;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  const b = new Float64Array(nfree * nrhs), fc = new Float64Array(nfree * 3);
  for (let j = 0; j < n; j++)
    for (let p = K.colptr[j]; p < K.colptr[j + 1]; p++) {
      const i = K.rowidx[p];
      if (isFixed[i]) continue;
      if (isFixed[j]) { for (let r = 0; r < nrhs; r++) b[r * nfree + newIdx[i]] -= K.val[p] * fval[r * n + j]; }
      else { I.push(newIdx[i]); J.push(newIdx[j]); V.push(K.val[p]); }
    }
  for (let i = 0; i < n; i++) if (!isFixed[i]) for (let c = 0; c < 3; c++) fc[3 * newIdx[i] + c] = coords[3 * i + c];
  const xf = solver(fromTriplets(nfree, I, J, V), fc, b, nrhs);
  const x = new Float64Array(n * nrhs);
  for (let i = 0; i < n; i++)
    for (let r = 0; r < nrhs; r++) x[r * n + i] = isFixed[i] ? fval[r * n + i] : xf[r * nfree + newIdx[i]];
  return x;
}

/**
 * Chart roughness: the jump of log(conformal factor) across adjacent faces --
 * mean and 99th percentile. A chart whose stretch changes abruptly between
 * neighbouring triangles is a kinked function of (theta, phi), and kinks are
 * what a band-limited fit turns into ringing. The conformal map is the
 * smooth reference (Spot 0.13, David 0.03); SDEM charts land at 0.16-0.4.
 */
export function chartRoughness(m: Mesh, S: ArrayLike<number>): { mean: number; p99: number } {
  const { lam } = lambdaStats(m, S);
  const f = m.faces, key = (a: number, b: number) => (a < b ? a * m.nv + b : b * m.nv + a);
  const edge = new Map<number, number>(), jumps: number[] = [];
  for (let t = 0; t < m.nf; t++)
    for (let k = 0; k < 3; k++) {
      const kk = key(f[3 * t + k], f[3 * t + ((k + 1) % 3)]);
      const o = edge.get(kk);
      if (o === undefined) edge.set(kk, t); else jumps.push(Math.abs(Math.log(lam[t]) - Math.log(lam[o])));
    }
  jumps.sort((a, b) => a - b);
  return { mean: jumps.reduce((a, b) => a + b, 0) / (jumps.length || 1), p99: jumps[Math.floor(0.99 * (jumps.length - 1))] ?? 0 };
}
