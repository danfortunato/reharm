/**
 * Spherical conformal map of a genus-0 closed mesh — a port of Choi et al.'s
 * spherical_conformal_map.m (FLASH, SIAM J. Imaging Sci. 2015):
 *
 *   1. harmonic map to the plane with the most regular triangle pinned
 *      (cotangent Laplacian, Dirichlet on its three vertices), inverse
 *      stereographic projection, and a Möbius rescaling that balances the
 *      pinned triangle against the innermost one;
 *   2. the south cap is held and the Beltrami coefficient of the planar map is
 *      corrected with one linear Beltrami solve (a quasi-conformal
 *      composition), then projected back to the sphere.
 *
 * Every linear system is the free block of a symmetric (semi)definite
 * operator, solved by sparse Cholesky. NaN results fall back exactly as the
 * MATLAB does (Tutte map; larger landmark set; the uncorrected map).
 */
import type { Mesh } from '../mesh/types.ts';
import { fromTriplets, spdSolve, type CSC } from './sparse.ts';
import { cotangentLaplacian, tutteLaplacian, dirichletSolve, mostRegularTriangle } from './meshops.ts';

const negate = (A: CSC): CSC => ({ ...A, val: A.val.map((v) => -v) });

/** inverse stereographic projection, plane -> sphere (z = (-1 + |z|^2)/(1 + |z|^2)) */
function toSphere(re: Float64Array, im: Float64Array, flipZ = false): Float64Array {
  const n = re.length, S = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r2 = re[i] * re[i] + im[i] * im[i], d = 1 + r2;
    S[3 * i] = (2 * re[i]) / d; S[3 * i + 1] = (2 * im[i]) / d; S[3 * i + 2] = (flipZ ? 1 - r2 : r2 - 1) / d;
  }
  return S;
}

/** The common tail of the conformal and Tutte maps: centre, project, balance the puncture against the innermost triangle. */
function finishPlanarMap(zr: Float64Array, zi: Float64Array, f: Uint32Array, bigtri: number): Float64Array {
  const n = zr.length, nf = f.length / 3;
  let mr = 0, mi = 0;
  for (let i = 0; i < n; i++) { mr += zr[i]; mi += zi[i]; }
  mr /= n; mi /= n;
  for (let i = 0; i < n; i++) { zr[i] -= mr; zi[i] -= mi; }
  const S = toSphere(zr, zi);
  // w: projection from the other pole
  const wr = new Float64Array(n), wi = new Float64Array(n);
  for (let i = 0; i < n; i++) { wr[i] = S[3 * i] / (1 + S[3 * i + 2]); wi[i] = S[3 * i + 1] / (1 + S[3 * i + 2]); }
  let inner = -1, innerVal = Infinity, second = -1, secondVal = Infinity;
  for (let t = 0; t < nf; t++) {
    let s = 0;
    for (let k = 0; k < 3; k++) { const v = f[3 * t + k]; s += Math.hypot(zr[v], zi[v]); }
    if (s < innerVal) { second = inner; secondVal = innerVal; inner = t; innerVal = s; }
    else if (s < secondVal) { second = t; secondVal = s; }
  }
  if (inner === bigtri) inner = second;
  const side = (re: Float64Array, im: Float64Array, t: number) => {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    return (Math.hypot(re[a] - re[b], im[a] - im[b]) + Math.hypot(re[b] - re[c], im[b] - im[c]) + Math.hypot(re[c] - re[a], im[c] - im[a])) / 3;
  };
  const north = side(zr, zi, bigtri), south = side(wr, wi, inner);
  const scale = Math.sqrt(north * south) / north;
  for (let i = 0; i < n; i++) { zr[i] *= scale; zi[i] *= scale; }
  return toSphere(zr, zi);
}

/** Harmonic map to the plane with the puncture triangle's vertices pinned (conformal: cotangent weights; Tutte: uniform). */
function punctureAndSolve(m: Mesh, L: CSC, bigtri: number, pinned: [number, number][]): { zr: Float64Array; zi: Float64Array } {
  const fixed = [m.faces[3 * bigtri], m.faces[3 * bigtri + 1], m.faces[3 * bigtri + 2]];
  const vals = new Float64Array(6);
  for (let k = 0; k < 3; k++) { vals[k] = pinned[k][0]; vals[3 + k] = pinned[k][1]; }
  const x = dirichletSolve(negate(L), m.positions, fixed, vals, 2, spdSolve);
  return { zr: x.subarray(0, m.nv).slice(), zi: x.subarray(m.nv).slice() };
}

export function sphericalTutteMap(m: Mesh, bigtri = mostRegularTriangle(m)): Float64Array {
  const pinned: [number, number][] = [0, 1, 2].map((k) => [Math.cos((2 * Math.PI * k) / 3), Math.sin((2 * Math.PI * k) / 3)]);
  const { zr, zi } = punctureAndSolve(m, tutteLaplacian(m), bigtri, pinned);
  return finishPlanarMap(zr, zi, m.faces, bigtri);
}

/** Beltrami coefficient of the map plane (P, 2D) -> surface (m.positions), per face. */
export function beltramiCoefficient(m: Mesh, P: Float64Array): { re: Float64Array; im: Float64Array } {
  const { faces: f, positions: v, nf } = m;
  const re = new Float64Array(nf), im = new Float64Array(nf);
  for (let t = 0; t < nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const e1x = P[2 * c] - P[2 * b], e1y = P[2 * c + 1] - P[2 * b + 1];
    const e2x = P[2 * a] - P[2 * c], e2y = P[2 * a + 1] - P[2 * c + 1];
    const e3x = P[2 * b] - P[2 * a], e3y = P[2 * b + 1] - P[2 * a + 1];
    const area = (-e2x * e1y + e1x * e2y) / 2;
    const Dx = [e1y, e2y, e3y].map((e) => e / area / 2), Dy = [e1x, e2x, e3x].map((e) => -e / area / 2);
    let E = 0, G = 0, F = 0;
    for (let c3 = 0; c3 < 3; c3++) {
      const du = Dx[0] * v[3 * a + c3] + Dx[1] * v[3 * b + c3] + Dx[2] * v[3 * c + c3];
      const dv = Dy[0] * v[3 * a + c3] + Dy[1] * v[3 * b + c3] + Dy[2] * v[3 * c + c3];
      E += du * du; G += dv * dv; F += du * dv;
    }
    const den = E + G + 2 * Math.sqrt(Math.max(E * G - F * F, 0));
    re[t] = (E - G) / den; im[t] = (2 * F) / den;
  }
  return { re, im };
}

/** Linear Beltrami solver (Lui et al.): the planar map with Beltrami coefficient mu and the landmarks pinned. */
export function linearBeltramiSolver(
  m: Mesh, P: Float64Array, mu: { re: Float64Array; im: Float64Array }, landmarks: Int32Array | number[], target: Float64Array,
): Float64Array {
  return linearBeltramiSolverFaces(m.faces, m.nv, P, mu, landmarks, target);
}

/** Same, on an explicit face list (the SDEM overlap correction punctures a triangle). */
export function linearBeltramiSolverFaces(
  f: Uint32Array, nv: number, P: Float64Array, mu: { re: Float64Array; im: Float64Array }, landmarks: Int32Array | number[], target: Float64Array,
): Float64Array {
  const nf = f.length / 3;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  for (let t = 0; t < nf; t++) {
    const mr = mu.re[t], mi = mu.im[t], m2 = mr * mr + mi * mi, den = 1 - m2;
    const af = (1 - 2 * mr + m2) / den, bf = (-2 * mi) / den, gf = (1 + 2 * mr + m2) / den;
    const f0 = f[3 * t], f1 = f[3 * t + 1], f2 = f[3 * t + 2];
    const ux = [P[2 * f1 + 1] - P[2 * f2 + 1], P[2 * f2 + 1] - P[2 * f0 + 1], P[2 * f0 + 1] - P[2 * f1 + 1]];
    const uy = [P[2 * f2] - P[2 * f1], P[2 * f0] - P[2 * f2], P[2 * f1] - P[2 * f0]];
    const l = ux.map((x, k) => Math.hypot(x, uy[k])), s = (l[0] + l[1] + l[2]) / 2;
    const area = Math.sqrt(Math.max(s * (s - l[0]) * (s - l[1]) * (s - l[2]), 1e-300));
    const q = (i: number, j: number) => (af * ux[i] * ux[j] + bf * ux[i] * uy[j] + bf * ux[j] * uy[i] + gf * uy[i] * uy[j]) / area;
    const idx = [f0, f1, f2];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { I.push(idx[i]); J.push(idx[j]); V.push(-q(i, j) / 2); }
  }
  // A = sparse(I,J,-V) is negative semidefinite; the solve uses K = -A (PSD, free block PD)
  const K = negate(fromTriplets(nv, I, J, V));
  const coords2 = new Float64Array(3 * nv);
  for (let i = 0; i < nv; i++) { coords2[3 * i] = P[2 * i]; coords2[3 * i + 1] = P[2 * i + 1]; }
  const fv = new Float64Array(2 * landmarks.length);
  for (let k = 0; k < landmarks.length; k++) { fv[k] = target[2 * k]; fv[landmarks.length + k] = target[2 * k + 1]; }
  const x = dirichletSolve(K, coords2, landmarks, fv, 2, spdSolve);
  const out = new Float64Array(2 * nv);
  for (let i = 0; i < nv; i++) { out[2 * i] = x[i]; out[2 * i + 1] = x[nv + i]; }
  return out;
}

const hasNaN = (a: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return true; return false; };

/** The spherical conformal map: nv*3 unit-sphere coordinates. */
export function sphericalConformalMap(m: Mesh): Float64Array {
  const { faces: f, positions: v, nv } = m;
  const bigtri = mostRegularTriangle(m);
  const p1 = f[3 * bigtri], p2 = f[3 * bigtri + 1], p3 = f[3 * bigtri + 2];
  // planar positions of the pinned triangle: (0,0), (1,0), and the third by similarity
  const a = [v[3 * p2] - v[3 * p1], v[3 * p2 + 1] - v[3 * p1 + 1], v[3 * p2 + 2] - v[3 * p1 + 2]];
  const b = [v[3 * p3] - v[3 * p1], v[3 * p3 + 1] - v[3 * p1 + 1], v[3 * p3 + 2] - v[3 * p1 + 2]];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const cx = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const sin1 = Math.hypot(...cx) / (na * nb);
  const ratio = 1 / na, y3 = nb * sin1 * ratio, x3 = Math.sqrt(Math.max(nb * nb * ratio * ratio - y3 * y3, 0));
  let { zr, zi } = punctureAndSolve(m, cotangentLaplacian(m), bigtri, [[0, 0], [1, 0], [x3, y3]]);
  let S = finishPlanarMap(zr, zi, f, bigtri);
  if (hasNaN(S)) S = sphericalTutteMap(m, bigtri);

  // quasi-conformal correction with the south cap held
  const order = Int32Array.from({ length: nv }, (_, i) => i).sort((i, j) => S[3 * i + 2] - S[3 * j + 2]);
  const P = new Float64Array(2 * nv);
  for (let i = 0; i < nv; i++) { P[2 * i] = S[3 * i] / (1 + S[3 * i + 2]); P[2 * i + 1] = S[3 * i + 1] / (1 + S[3 * i + 2]); }
  const mu = beltramiCoefficient(m, P);
  const tryFix = (fixnum: number): Float64Array | null => {
    const fixed = order.slice(0, Math.min(nv, fixnum));
    const target = new Float64Array(2 * fixed.length);
    for (let k = 0; k < fixed.length; k++) { target[2 * k] = P[2 * fixed[k]]; target[2 * k + 1] = P[2 * fixed[k] + 1]; }
    try { const r = linearBeltramiSolver(m, P, mu, fixed, target); return hasNaN(r) ? null : r; } catch { return null; }
  };
  const fixnum = Math.max(Math.round(nv / 10), 3);
  const map = tryFix(fixnum) ?? tryFix(fixnum * 5) ?? P;
  const re = new Float64Array(nv), im = new Float64Array(nv);
  for (let i = 0; i < nv; i++) { re[i] = map[2 * i]; im[i] = map[2 * i + 1]; }
  return toSphere(re, im, true);
}
