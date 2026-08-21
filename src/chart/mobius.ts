/**
 * Möbius area correction (Choi & Rycroft 2018; mobius_area_correction_spherical.m):
 * among the Möbius transformations of the sphere -- which keep the map
 * conformal -- pick the one that minimizes the mean |log(area ratio)| over the
 * faces. fmincon over the 8 real parameters of (az+b)/(cz+d) is replaced by
 * Nelder-Mead (derivative-free, same objective, same start).
 */
import type { Mesh } from '../mesh/types.ts';
import { faceAreas } from './meshops.ts';

function stereo(S: ArrayLike<number>, n: number): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { const d = 1 - S[3 * i + 2]; re[i] = S[3 * i] / d; im[i] = S[3 * i + 1] / d; }
  return { re, im };
}

function unstereo(re: Float64Array, im: Float64Array, out: Float64Array): Float64Array {
  for (let i = 0; i < re.length; i++) {
    const r2 = re[i] * re[i] + im[i] * im[i], d = 1 + r2;
    out[3 * i] = (2 * re[i]) / d; out[3 * i + 1] = (2 * im[i]) / d; out[3 * i + 2] = (r2 - 1) / d;
  }
  return out;
}

function applyMobius(x: ArrayLike<number>, zr: Float64Array, zi: Float64Array, wr: Float64Array, wi: Float64Array): void {
  const [ar, ai, br, bi, cr, ci, dr, di] = Array.from(x);
  for (let k = 0; k < zr.length; k++) {
    const nr = ar * zr[k] - ai * zi[k] + br, ni = ar * zi[k] + ai * zr[k] + bi;
    const qr = cr * zr[k] - ci * zi[k] + dr, qi = cr * zi[k] + ci * zr[k] + di;
    const q2 = qr * qr + qi * qi;
    wr[k] = (nr * qr + ni * qi) / q2; wi[k] = (ni * qr - nr * qi) / q2;
  }
}

export function mobiusAreaCorrection(m: Mesh, S: Float64Array, maxEval = 1500): { S: Float64Array; x: Float64Array; iterations: number } {
  const { faces: f, nv, nf } = m;
  const areaV = faceAreas(f, m.positions);
  let tot = 0; for (let t = 0; t < nf; t++) tot += areaV[t];
  for (let t = 0; t < nf; t++) areaV[t] /= tot;
  const { re: zr, im: zi } = stereo(S, nv);
  const wr = new Float64Array(nv), wi = new Float64Array(nv), tmp = new Float64Array(3 * nv);
  const objective = (x: ArrayLike<number>): number => {
    applyMobius(x, zr, zi, wr, wi);
    const A = faceAreas(f, unstereo(wr, wi, tmp));
    let s = 0; for (let t = 0; t < nf; t++) s += A[t];
    let acc = 0, cnt = 0;
    for (let t = 0; t < nf; t++) { const v = Math.abs(Math.log(A[t] / s / areaV[t])); if (Number.isFinite(v)) { acc += v; cnt++; } }
    return cnt ? acc / cnt : Infinity;
  };
  const { x, evals } = nelderMead(objective, [1, 0, 0, 0, 0, 0, 1, 0], 0.15, maxEval, 1e-7);
  applyMobius(x, zr, zi, wr, wi);
  return { S: unstereo(wr, wi, new Float64Array(3 * nv)), x: Float64Array.from(x), iterations: evals };
}

/** Standard Nelder-Mead simplex minimization. */
export function nelderMead(fn: (x: number[]) => number, x0: number[], step: number, maxEval: number, ftol: number): { x: number[]; f: number; evals: number } {
  const n = x0.length;
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step; simplex.push(p); }
  let fs = simplex.map(fn), evals = n + 1;
  const order = () => { const idx = fs.map((_, i) => i).sort((a, b) => fs[a] - fs[b]); simplex.splice(0, n + 1, ...idx.map((i) => simplex[i])); fs = idx.map((i) => fs[i]); };
  order();
  while (evals < maxEval) {
    if (Math.abs(fs[n] - fs[0]) <= ftol * (Math.abs(fs[0]) + 1e-12)) break;
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    const worst = simplex[n];
    const reflect = centroid.map((c, j) => c + (c - worst[j]));
    const fr = fn(reflect); evals++;
    if (fr < fs[0]) {
      const expand = centroid.map((c, j) => c + 2 * (c - worst[j]));
      const fe = fn(expand); evals++;
      if (fe < fr) { simplex[n] = expand; fs[n] = fe; } else { simplex[n] = reflect; fs[n] = fr; }
    } else if (fr < fs[n - 1]) {
      simplex[n] = reflect; fs[n] = fr;
    } else {
      const outside = fr < fs[n];
      const contract = centroid.map((c, j) => c + 0.5 * ((outside ? reflect[j] : worst[j]) - c));
      const fc = fn(contract); evals++;
      if (fc < (outside ? fr : fs[n])) { simplex[n] = contract; fs[n] = fc; }
      else { // shrink toward the best
        for (let i = 1; i <= n; i++) { simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j])); fs[i] = fn(simplex[i]); evals++; }
      }
    }
    order();
  }
  return { x: simplex[0], f: fs[0], evals };
}
