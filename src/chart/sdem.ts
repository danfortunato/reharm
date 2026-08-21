/**
 * Spherical density-equalizing map — a port of SDEM.m (Lyu, Lui, Choi, SIAM
 * J. Imaging Sci. 2024) with its overlap correction
 * (update_and_correct_overlap.m). A density rho = population / face area is
 * diffused on the current sphere mesh, vertices move along -grad(rho)/rho
 * projected to the tangent plane, and after each step folds are detected as
 * |mu| >= 1 in a stereographic chart (punctured at a regular triangle, then at
 * its antipode) and removed by a linear Beltrami solve with |mu| clamped to
 * 1 - delta; if that fails the step is halved.
 *
 * population = face areas (the default here) targets uniform geometric area --
 * sphere area proportional to surface area, i.e. a uniform conformal factor,
 * which is what a solver wants; ones targets uniform mesh density.
 *
 * Fold handling: 'repair' (default) takes the step and removes any folds by
 * local Tutte relaxation, halving the step if that fails -- ~7x faster than the
 * paper's stereographic Beltrami correction ('lbs', kept as an option) and, on
 * the meshes tested, it converges where 'lbs' stalled.
 */
import type { Mesh } from '../mesh/types.ts';
import { faceAreas, countFolds } from './meshops.ts';
import { repairSphericalFolds } from './repair.ts';
import { fromTriplets, pcgJacobi, type CSC } from './sparse.ts';
import { linearBeltramiSolverFaces } from './conformal.ts';

// lightweight profiling hooks (no-ops unless globalThis.__sdemProf exists)
const prof = (name: string, t0: number): void => {
  const P = (globalThis as any).__sdemProf as Record<string, { ms: number; n: number }> | undefined;
  if (!P) return;
  const e = (P[name] ??= { ms: 0, n: 0 }); e.ms += performance.now() - t0; e.n++;
};

const normalizeRows = (r: Float64Array): void => {
  for (let i = 0; i < r.length; i += 3) {
    const n = Math.hypot(r[i], r[i + 1], r[i + 2]) || 1;
    r[i] /= n; r[i + 1] /= n; r[i + 2] /= n;
  }
};

/** face values -> vertex values, area-weighted (f2v_area). */
function faceToVertex(f: Uint32Array, nv: number, area: Float64Array, fval: Float64Array, d: number): Float64Array {
  const out = new Float64Array(nv * d), w = new Float64Array(nv);
  for (let t = 0; t < f.length / 3; t++)
    for (let k = 0; k < 3; k++) {
      const v = f[3 * t + k];
      w[v] += area[t];
      for (let c = 0; c < d; c++) out[v * d + c] += area[t] * fval[t * d + c];
    }
  for (let v = 0; v < nv; v++) for (let c = 0; c < d; c++) out[v * d + c] /= w[v] || 1;
  return out;
}

/** (A + dt L): lumped mass plus dt times the half-cotangent Laplace-Beltrami (PSD). */
function massPlusLaplacian(f: Uint32Array, nv: number, r: Float64Array, dt: number): { M: CSC; A: Float64Array } {
  const nf = f.length / 3;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  const A = new Float64Array(nv);
  const len = (a: number, b: number) => Math.hypot(r[3 * a] - r[3 * b], r[3 * a + 1] - r[3 * b + 1], r[3 * a + 2] - r[3 * b + 2]);
  for (let t = 0; t < nf; t++) {
    const f1 = f[3 * t], f2 = f[3 * t + 1], f3 = f[3 * t + 2];
    const l1 = len(f2, f3), l2 = len(f3, f1), l3 = len(f1, f2), s = (l1 + l2 + l3) / 2;
    const area = Math.sqrt(Math.max(s * (s - l1) * (s - l2) * (s - l3), 1e-300));
    const cot12 = (l1 * l1 + l2 * l2 - l3 * l3) / area / 4, cot23 = (l2 * l2 + l3 * l3 - l1 * l1) / area / 4, cot31 = (l1 * l1 + l3 * l3 - l2 * l2) / area / 4;
    I.push(f1, f2, f2, f3, f3, f1, f1, f2, f3); J.push(f2, f1, f3, f2, f1, f3, f1, f2, f3);
    V.push(-cot12, -cot12, -cot23, -cot23, -cot31, -cot31, cot12 + cot31, cot12 + cot23, cot31 + cot23);
    A[f1] += area / 3; A[f2] += area / 3; A[f3] += area / 3;
  }
  for (let k = 0; k < V.length; k++) V[k] *= dt / 2;
  for (let v = 0; v < nv; v++) { I.push(v); J.push(v); V.push(A[v]); }
  return { M: fromTriplets(nv, I, J, V), A };
}

/** Per-face gradient of a vertex scalar on the embedded mesh (compute_gradient_3D). */
function faceGradient(f: Uint32Array, r: Float64Array, g: Float64Array): Float64Array {
  const nf = f.length / 3, out = new Float64Array(3 * nf);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const e1 = [r[3 * c] - r[3 * b], r[3 * c + 1] - r[3 * b + 1], r[3 * c + 2] - r[3 * b + 2]];
    const e2 = [r[3 * a] - r[3 * c], r[3 * a + 1] - r[3 * c + 1], r[3 * a + 2] - r[3 * c + 2]];
    const e3 = [r[3 * b] - r[3 * a], r[3 * b + 1] - r[3 * a + 1], r[3 * b + 2] - r[3 * a + 2]];
    const cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
    const area2 = Math.hypot(cx, cy, cz); // = 2 area
    const N = [cx / area2, cy / area2, cz / area2];
    const tx = g[a] * e1[0] + g[b] * e2[0] + g[c] * e3[0], ty = g[a] * e1[1] + g[b] * e2[1] + g[c] * e3[1], tz = g[a] * e1[2] + g[b] * e2[2] + g[c] * e3[2];
    out[3 * t] = (N[1] * tz - N[2] * ty) / area2; out[3 * t + 1] = (N[2] * tx - N[0] * tz) / area2; out[3 * t + 2] = (N[0] * ty - N[1] * tx) / area2;
  }
  return out;
}

/** regular_triangle.m: the triangle whose vertices have the smallest averaged irregularity. */
function regularTriangle(f: Uint32Array, nv: number, r: Float64Array): number {
  const nf = f.length / 3, Rf = new Float64Array(nf), Rv = new Float64Array(nv), cnt = new Float64Array(nv);
  const len = (a: number, b: number) => Math.hypot(r[3 * a] - r[3 * b], r[3 * a + 1] - r[3 * b + 1], r[3 * a + 2] - r[3 * b + 2]);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2], e1 = len(b, c), e2 = len(a, c), e3 = len(a, b), s = e1 + e2 + e3;
    Rf[t] = Math.abs(e1 / s - 1 / 3) + Math.abs(e2 / s - 1 / 3) + Math.abs(e3 / s - 1 / 3);
    for (let k = 0; k < 3; k++) { Rv[f[3 * t + k]] += Rf[t] / 3; cnt[f[3 * t + k]]++; }
  }
  let best = 0, bestVal = Infinity;
  for (let t = 0; t < nf; t++) {
    const v = (Rv[f[3 * t]] + Rv[f[3 * t + 1]] + Rv[f[3 * t + 2]]) / 3;
    if (v < bestVal) { bestVal = v; best = t; }
  }
  return best;
}

/** rotate_sphere.m: rotation taking the centroid direction of face `t` to the north pole; returns M (row-major 3x3) and its inverse. */
function rotationToNorth(f: Uint32Array, r: Float64Array, t: number): { M: number[]; Minv: number[] } {
  const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
  let cx = (r[3 * a] + r[3 * b] + r[3 * c]) / 3, cy = (r[3 * a + 1] + r[3 * b + 1] + r[3 * c + 1]) / 3, cz = (r[3 * a + 2] + r[3 * b + 2] + r[3 * c + 2]) / 3;
  const n = Math.hypot(cx, cy, cz); cx /= n; cy /= n; cz /= n;
  const rxy = Math.hypot(cx, cy) || 1e-300;
  const sz = -cy / rxy, czz = cx / rxy;
  const b1 = czz * cx - sz * cy, b3 = cz; // rot_z applied: (b1, 0, b3)
  const rb = Math.hypot(b1, b3);
  const sy = -b1 / rb, cyy = b3 / rb;
  // M = rot_y * rot_z
  const rz = [czz, -sz, 0, sz, czz, 0, 0, 0, 1], ry = [cyy, 0, sy, 0, 1, 0, -sy, 0, cyy];
  const mul = (P: number[], Q: number[]) => { const R = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) R[3 * i + j] += P[3 * i + k] * Q[3 * k + j]; return R; };
  const M = mul(ry, rz);
  const Minv = mul([czz, sz, 0, -sz, czz, 0, 0, 0, 1], [cyy, 0, -sy, 0, 1, 0, sy, 0, cyy]);
  return { M, Minv };
}

const rotate = (R: number[], r: Float64Array): Float64Array => {
  const out = new Float64Array(r.length);
  for (let i = 0; i < r.length; i += 3) {
    const x = r[i], y = r[i + 1], z = r[i + 2];
    out[i] = R[0] * x + R[1] * y + R[2] * z; out[i + 1] = R[3] * x + R[4] * y + R[5] * z; out[i + 2] = R[6] * x + R[7] * y + R[8] * z;
  }
  return out;
};

/** sphere -> plane (x/(1-z), y/(1-z)); NaN -> Infinity as in the MATLAB. */
function stereo(r: Float64Array): Float64Array {
  const n = r.length / 3, p = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { const d = 1 - r[3 * i + 2]; p[2 * i] = d === 0 ? Infinity : r[3 * i] / d; p[2 * i + 1] = d === 0 ? Infinity : r[3 * i + 1] / d; }
  return p;
}
/** plane -> sphere; non-finite points go to the north pole. */
function unstereo(p: Float64Array): Float64Array {
  const n = p.length / 2, r = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const x = p[2 * i], y = p[2 * i + 1], z = 1 + x * x + y * y;
    if (!Number.isFinite(z)) { r[3 * i + 2] = 1; continue; }
    r[3 * i] = (2 * x) / z; r[3 * i + 1] = (2 * y) / z; r[3 * i + 2] = (-1 + x * x + y * y) / z;
  }
  return r;
}

/** Beltrami coefficient of a planar map P -> Q on a face list (the 2-column branch of beltrami_coefficient.m). */
function beltrami2D(f: Uint32Array, P: Float64Array, Q: Float64Array): { re: Float64Array; im: Float64Array } {
  const nf = f.length / 3, re = new Float64Array(nf), im = new Float64Array(nf);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const e1x = P[2 * c] - P[2 * b], e1y = P[2 * c + 1] - P[2 * b + 1];
    const e2x = P[2 * a] - P[2 * c], e2y = P[2 * a + 1] - P[2 * c + 1];
    const e3x = P[2 * b] - P[2 * a], e3y = P[2 * b + 1] - P[2 * a + 1];
    const area = (-e2x * e1y + e1x * e2y) / 2;
    const Mx = [e1y, e2y, e3y].map((e) => e / area / 2), My = [e1x, e2x, e3x].map((e) => -e / area / 2);
    const idx = [a, b, c];
    let fxr = 0, fxi = 0, fyr = 0, fyi = 0;
    for (let k = 0; k < 3; k++) { fxr += Mx[k] * Q[2 * idx[k]]; fxi += Mx[k] * Q[2 * idx[k] + 1]; fyr += My[k] * Q[2 * idx[k]]; fyi += My[k] * Q[2 * idx[k] + 1]; }
    // Dz = (Dx - i Dy)/2, Dc = (Dx + i Dy)/2 applied to z = Q_x + i Q_y
    const dzr = (fxr + fyi) / 2, dzi = (fxi - fyr) / 2, dcr = (fxr - fyi) / 2, dci = (fxi + fyr) / 2;
    const den = dzr * dzr + dzi * dzi;
    let mr = (dcr * dzr + dci * dzi) / den, mi = (dci * dzr - dcr * dzi) / den;
    if (!Number.isFinite(mr) || !Number.isFinite(mi)) { mr = 1; mi = 0; }
    re[t] = mr; im[t] = mi;
  }
  return { re, im };
}

/** The stereographic charts of S and r about `pole`, the punctured face list, and the ignored cap. */
function hemisphere(f: Uint32Array, nv: number, S: Float64Array, r: Float64Array, pole: number) {
  const th = performance.now();
  try { return hemisphereImpl(f, nv, S, r, pole); } finally { prof('  hemisphere setup', th); }
}
function hemisphereImpl(f: Uint32Array, nv: number, S: Float64Array, r: Float64Array, pole: number) {
  const { M } = rotationToNorth(f, S, pole);
  const Mr = rotationToNorth(f, r, pole); // S and r are each rotated by the rotation of its OWN centroid of `pole`
  const Srot = rotate(M, S), rrot = rotate(Mr.M, r);
  const pS = stereo(Srot), pr = stereo(rrot);
  const nf = f.length / 3;
  const fp = new Uint32Array(3 * (nf - 1));
  for (let t = 0, q = 0; t < nf; t++) if (t !== pole) { fp[q++] = f[3 * t]; fp[q++] = f[3 * t + 1]; fp[q++] = f[3 * t + 2]; }
  const order = Int32Array.from({ length: nv }, (_, i) => i).sort((i, j) => Srot[3 * j + 2] - Srot[3 * i + 2]);
  const nig = Math.max(Math.round(nv / 10), 3), ig = order.slice(0, nig);
  const isIg = new Uint8Array(nv); for (const v of ig) isIg[v] = 1;
  const countOverlaps = (mu: { re: Float64Array; im: Float64Array }) => {
    let n = 0;
    for (let t = 0; t < fp.length / 3; t++) {
      if (isIg[fp[3 * t]] || isIg[fp[3 * t + 1]] || isIg[fp[3 * t + 2]]) continue;
      if (Math.hypot(mu.re[t], mu.im[t]) >= 1) n++;
    }
    return n;
  };
  return { Mr, rrot, pS, pr, fp, ig, nig, countOverlaps };
}

/** Overlaps of the map S -> r in the chart about `pole` (no correction). */
function overlapCount(f: Uint32Array, nv: number, S: Float64Array, r: Float64Array, pole: number): number {
  const h = hemisphere(f, nv, S, r, pole);
  return h.countOverlaps(beltrami2D(h.fp, h.pS, h.pr));
}

/** One hemisphere pass of the overlap correction. Accepts when the corrected map has no more overlaps than `baseline`. */
function correctHemisphere(f: Uint32Array, nv: number, S: Float64Array, r: Float64Array, pole: number, delta: number, baseline: number): { r: Float64Array; ok: boolean } {
  const h = hemisphere(f, nv, S, r, pole);
  const { Mr, rrot, pS, pr, fp, ig, nig, countOverlaps } = h;
  const mu = beltrami2D(fp, pS, pr);
  const overlaps = countOverlaps(mu);
  if (overlaps === 0) return { r: rotate(Mr.Minv, rrot), ok: true };
  // Clamp |mu| < 1 on EVERY face, not only the counted overlaps: the Beltrami
  // operator is elliptic only where |mu| < 1, and faces touching the ignored
  // polar cap can carry |mu| >= 1 too (MATLAB's general LU tolerates the
  // resulting indefinite system; a Cholesky solve needs it positive definite).
  for (let t = 0; t < fp.length / 3; t++) {
    const a = Math.hypot(mu.re[t], mu.im[t]);
    if (a >= 1) { mu.re[t] *= (1 - delta) / a; mu.im[t] *= (1 - delta) / a; }
  }
  const target = new Float64Array(2 * nig);
  for (let k = 0; k < nig; k++) { target[2 * k] = pr[2 * ig[k]]; target[2 * k + 1] = pr[2 * ig[k] + 1]; }
  let pl: Float64Array;
  const tl = performance.now();
  try { pl = linearBeltramiSolverFaces(fp, nv, pS, mu, ig, target); } catch { prof('  LBS solve', tl); return { r, ok: false }; }
  prof('  LBS solve', tl);
  const left = countOverlaps(beltrami2D(fp, pS, pl));
  return { r: rotate(Mr.Minv, unstereo(pl)), ok: left <= baseline };
}

function southPole(f: Uint32Array, r: Float64Array, bigtri: number): number {
  const nf = f.length / 3, cen = (t: number) => {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const x = (r[3 * a] + r[3 * b] + r[3 * c]) / 3, y = (r[3 * a + 1] + r[3 * b + 1] + r[3 * c + 1]) / 3, z = (r[3 * a + 2] + r[3 * b + 2] + r[3 * c + 2]) / 3;
    const n = Math.hypot(x, y, z); return [x / n, y / n, z / n];
  };
  const c0 = cen(bigtri);
  let best = 0, bestD = -1;
  for (let t = 0; t < nf; t++) { const c = cen(t); const d = (c[0] - c0[0]) ** 2 + (c[1] - c0[1]) ** 2 + (c[2] - c0[2]) ** 2; if (d > bestD) { bestD = d; best = t; } }
  return best;
}

/** update_and_correct_overlap.m, with acceptance relative to the current map and the halving capped. */
function updateAndCorrectOverlap(f: Uint32Array, nv: number, S: Float64Array, r0: Float64Array, bigtri: number, dr: Float64Array, dt: number): Float64Array {
  const delta = 0.1;
  const baseN = overlapCount(f, nv, S, r0, bigtri);
  const southF = southPole(f, r0, bigtri);
  const baseS = overlapCount(f, nv, S, r0, southF);
  for (let halvings = 0; halvings <= 12; halvings++, dt /= 2) {
    const r = new Float64Array(r0.length);
    for (let i = 0; i < r.length; i++) r[i] = r0[i] + dt * dr[i];
    normalizeRows(r);
    const north = correctHemisphere(f, nv, S, r, bigtri, delta, baseN);
    if (!north.ok) continue;
    const south = correctHemisphere(f, nv, S, north.r, southPole(f, r, bigtri), delta, baseS);
    if (south.ok) return south.r;
  }
  return r0;
}

export interface SdemOptions {
  population?: Float64Array; dt?: number; epsilon?: number; maxIter?: number;
  /** 'lbs': the paper's overlap correction (stereographic Beltrami solves with step halving);
   *  'repair': take the step and remove any folds by local Tutte relaxation (much faster) */
  correction?: 'lbs' | 'repair';
  /** stop after this many consecutive rejected steps (map no longer moving) */
  maxRejected?: number;
  onStep?: (step: number, err: number) => void;
  /** live cap on iterations, read before every step (a UI can change it
   *  mid-run); overrides maxIter when present */
  maxIterLive?: () => number;
  /** awaited once per step: lets a worker pump its message queue mid-run so
   *  maxIterLive can actually change */
  yieldStep?: () => Promise<void>;
}

/** The density-equalizing map from an initial spherical map S (e.g. the Tutte map). */
export async function sdem(m: Mesh, S: Float64Array, opt: SdemOptions = {}): Promise<{ S: Float64Array; steps: number; spread: number; repairs: number; stopped: 'converged' | 'stalled' | 'max steps' }> {
  const { faces: f, nv, nf } = m;
  const dt = opt.dt ?? 0.1, eps = opt.epsilon ?? 1e-3, maxIter = opt.maxIter ?? 300;
  // The density spread is not a usable convergence signal: SDEM progresses in
  // bursts (a long plateau, then a fold repair unlocks another drop). What is
  // reliable is whether the map still moves: once every step is rejected (all
  // halvings exhausted, map returned unchanged) nothing can change any more.
  const maxRejected = opt.maxRejected ?? 20;
  let rejected = 0;
  let stopped: 'converged' | 'stalled' | 'max steps' = 'max steps';
  const population = opt.population ?? faceAreas(f, m.positions);
  const r = S.slice(); normalizeRows(r);
  const bigtri = regularTriangle(f, nv, r);
  const density = (): Float64Array => {
    const area = faceAreas(f, r), rf = new Float64Array(nf);
    for (let t = 0; t < nf; t++) rf[t] = population[t] / Math.max(area[t], 1e-300);
    return faceToVertex(f, nv, area, rf, 1);
  };
  const spread = (x: Float64Array) => { let s = 0, s2 = 0; for (const v of x) { s += v; s2 += v * v; } const mean = s / x.length; return Math.sqrt(Math.max(s2 / x.length - mean * mean, 0)) / mean; };
  let rho = density(), err = spread(rho), step = 0;
  opt.onStep?.(0, err);
  const rhoTemp = rho.slice();
  let repairs = 0;
  const capOf = opt.maxIterLive ?? (() => maxIter);
  while (err >= eps && step < capOf()) {
    if (opt.yieldStep) await opt.yieldStep();
    if (step >= capOf()) break;   // the cap may have dropped during the yield
    // keep the map bijective throughout: a fold left behind by one hemisphere's
    // solve (in the cap the other hemisphere ignores) would otherwise veto every
    // later step, freezing the map
    const tr = performance.now();
    if (countFolds(f, r) > 0) { repairSphericalFolds(m, r); repairs++; rho = density(); }
    prof('fold check/repair', tr);
    let tt = performance.now();
    const { M, A } = massPlusLaplacian(f, nv, r, dt);
    prof('assembly', tt); tt = performance.now();
    const rhs = new Float64Array(nv); for (let v = 0; v < nv; v++) rhs[v] = A[v] * rho[v];
    rhoTemp.set(rho);
    const it = pcgJacobi(M, rhs, rhoTemp, 1e-10, 5000);
    prof(`pcg (avg ${0}it)`, tt); (globalThis as any).__pcgIts = ((globalThis as any).__pcgIts ?? 0) + it.iterations; tt = performance.now();
    const gf = faceGradient(f, r, rhoTemp);
    const gv = faceToVertex(f, nv, faceAreas(f, r), gf, 3);
    prof('gradient', tt);
    const dr = new Float64Array(3 * nv);
    for (let v = 0; v < nv; v++) {
      const dx = -gv[3 * v] / rhoTemp[v], dy = -gv[3 * v + 1] / rhoTemp[v], dz = -gv[3 * v + 2] / rhoTemp[v];
      const dot = dx * r[3 * v] + dy * r[3 * v + 1] + dz * r[3 * v + 2];
      dr[3 * v] = dx - dot * r[3 * v]; dr[3 * v + 1] = dy - dot * r[3 * v + 1]; dr[3 * v + 2] = dz - dot * r[3 * v + 2];
    }
    tt = performance.now();
    let rNew: Float64Array;
    if ((opt.correction ?? 'repair') === 'repair') {
      // take the step; if it folds, repair locally; if the repair cannot untangle it, halve the step
      rNew = r;
      for (let h = 0, step_ = dt; h <= 8; h++, step_ /= 2) {
        const cand = new Float64Array(r.length);
        for (let i = 0; i < r.length; i++) cand[i] = r[i] + step_ * dr[i];
        normalizeRows(cand);
        if (countFolds(f, cand) > 0) { repairs++; if (repairSphericalFolds(m, cand, 60).folds > 0) continue; }
        rNew = cand; break;
      }
    } else {
      rNew = updateAndCorrectOverlap(f, nv, S, r, bigtri, dr, dt);
    }
    prof('overlap correction', tt);
    rejected = rNew === r ? rejected + 1 : 0;   // the routine returns r0 itself when every halving failed
    r.set(rNew);
    step++;
    err = spread(rhoTemp);
    opt.onStep?.(step, err);
    rho = density();
    if (rejected >= maxRejected) { stopped = 'stalled'; break; }
  }
  if (err < eps) stopped = 'converged';
  return { S: r, steps: step, spread: err, repairs, stopped };
}
