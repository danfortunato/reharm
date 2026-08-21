/**
 * Coefficient-space filters: a gain per degree l, applied to the x, y, z
 * coefficient sets. Degree-only gains are rotation-invariant. Layouts follow
 * src/sht/layout.ts (SHTns m-major, interleaved re/im, m >= 0).
 *
 * These are the filters of the MATLAB module (filter_spherical_harmonics,
 * filter_spherical_harmonics_zhou 'gaussian'): hard truncation is an ideal
 * low-pass and rings near sharp features; the trapeziform ramp and the
 * Gaussian (heat-kernel) taper do not. Gaussian gains are strictly positive
 * and monotone in l.
 */
import { lmIndex } from '../sht/layout.ts';
import { gaussNodesWeights } from '../sht/gauss.ts';

export type FilterKind = 'none' | 'hard' | 'trapeziform' | 'gaussian' | 'fraction';

export interface FilterSpec {
  kind: FilterKind;
  /** cutoff degree (hard / trapeziform pass-band edge) */
  N: number;
  /** trapeziform: degrees >= N + ramp are removed; linear ramp in between */
  ramp: number;
  /** Gaussian kernel width exp(-theta^2 / (2 sigma^2)), radians */
  sigma: number;
  /** 'fraction': keep the lowest 1/r of the coefficients (Gu et al. Fig. 8) */
  r: number;
}

/** Degree at which keeping 1/r of (lmax+1)^2 coefficients truncates. */
export const fractionCutoff = (lmax: number, r: number): number =>
  Math.max(0, Math.round((lmax + 1) / Math.sqrt(r)) - 1);

/** Per-degree gains g[0..lmax] for a filter spec. */
export function degreeGains(lmax: number, spec: FilterSpec): Float64Array {
  const g = new Float64Array(lmax + 1).fill(1);
  switch (spec.kind) {
    case 'none':
      break;
    case 'hard':
      for (let l = 0; l <= lmax; l++) g[l] = l <= spec.N ? 1 : 0;
      break;
    case 'fraction': {
      const N = fractionCutoff(lmax, spec.r);
      for (let l = 0; l <= lmax; l++) g[l] = l <= N ? 1 : 0;
      break;
    }
    case 'trapeziform': {
      const nstop = spec.N + Math.max(1, spec.ramp);
      for (let l = 0; l <= lmax; l++) g[l] = Math.min(1, Math.max(0, (nstop - l) / (nstop - spec.N)));
      break;
    }
    case 'gaussian':
      return gaussianGains(lmax, spec.sigma);
  }
  return g;
}

/**
 * Convolution-theorem gains of the kernel G(theta) = exp(-theta^2 / 2 sigma^2),
 * g(l) = int G P_l(cos theta) sin theta dtheta, DC-normalized (port of
 * gaussian_gains in filter_spherical_harmonics_zhou.m). Gauss-Legendre
 * quadrature in x = cos theta.
 */
export function gaussianGains(lmax: number, sigma: number): Float64Array {
  const nq = Math.max(4 * lmax, 200);
  const { x, w } = gaussNodesWeights(nq);
  const G = new Float64Array(nq);
  for (let k = 0; k < nq; k++) {
    const th = Math.acos(Math.max(-1, Math.min(1, x[k])));
    G[k] = Math.exp(-(th * th) / (2 * sigma * sigma));
  }
  const g = new Float64Array(lmax + 1);
  const p0 = new Float64Array(nq).fill(1), p1 = Float64Array.from(x);
  let pm1 = p0, pl = p1;
  for (let l = 0; l <= lmax; l++) {
    let s = 0;
    const P = l === 0 ? p0 : pl;
    for (let k = 0; k < nq; k++) s += w[k] * G[k] * P[k];
    g[l] = s;
    if (l >= 1) {
      // P_{l+1} = ((2l+1) x P_l - l P_{l-1}) / (l+1)
      const next = new Float64Array(nq);
      for (let k = 0; k < nq; k++) next[k] = ((2 * l + 1) * x[k] * pl[k] - l * pm1[k]) / (l + 1);
      pm1 = pl; pl = next;
    }
  }
  const g0 = g[0];
  for (let l = 0; l <= lmax; l++) g[l] /= g0;
  return g;
}

/** Multiply each (l, m) coefficient by g[l]. Returns a new array. */
export function applyGains(qlm: Float32Array, lmax: number, mmax: number, g: ArrayLike<number>): Float32Array {
  const out = new Float32Array(qlm.length);
  for (let m = 0; m <= mmax; m++)
    for (let l = m; l <= lmax; l++) {
      const i = 2 * lmIndex(lmax, l, m);
      out[i] = qlm[i] * g[l];
      out[i + 1] = qlm[i + 1] * g[l];
    }
  return out;
}

/** Energy per degree, sum over m (m > 0 counted twice, real field) and over the given fields. */
export function powerSpectrum(fields: Float32Array[], lmax: number, mmax: number): Float64Array {
  const P = new Float64Array(lmax + 1);
  for (const q of fields)
    for (let m = 0; m <= mmax; m++)
      for (let l = m; l <= lmax; l++) {
        const i = 2 * lmIndex(lmax, l, m);
        const e = q[i] * q[i] + q[i + 1] * q[i + 1];
        P[l] += m === 0 ? e : 2 * e;
      }
  return P;
}
