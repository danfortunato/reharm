/**
 * Recurrence coefficients for the first theta-derivative of orthonormal
 * associated Legendre functions (Condon-Shortley phase included), matching
 * the alpha^+/alpha^- recurrence in evolving_surface/notes/algos.tex Sec 2.1:
 *
 *   sin(theta) d/dtheta Y_l^m = alpha^+(l,m) Y_{l+1}^m + alpha^-(l,m) Y_{l-1}^m
 *
 * so the coefficients of sin(theta)*dtheta(u), by degree, are
 *
 *   v_l^m = alpha^+(l-1,m) u_{l-1}^m + alpha^-(l+1,m) u_{l+1}^m
 *
 * dropping any term referring to a degree outside 0 <= l <= lmax. Baked to
 * zero at each m-block's first/last element (rather than left undefined), so
 * a consuming WGSL kernel needs only an in-bounds check, not a validity check.
 */
import { lmIndex, nlmCalc } from './layout.ts';

export interface DerivCoeffs {
  /** aPlus[lm] = alpha^+(l-1,m) when l>m, else 0 -- multiplies u_{l-1}^m. */
  aPlus: Float64Array;
  /** aMinus[lm] = alpha^-(l+1,m) when l<lmax, else 0 -- multiplies u_{l+1}^m. */
  aMinus: Float64Array;
  /** m of the coefficient at flat index lm (the phi-derivative needs only this). */
  mOf: Uint32Array;
}

export function alphaPlus(l: number, m: number): number {
  return l * Math.sqrt(((l - m + 1) * (l + m + 1)) / ((2 * l + 1) * (2 * l + 3)));
}

export function alphaMinus(l: number, m: number): number {
  return -(l + 1) * Math.sqrt(((l - m) * (l + m)) / ((2 * l - 1) * (2 * l + 1)));
}

export function derivCoeffs(lmax: number, mmax: number): DerivCoeffs {
  const nlm = nlmCalc(lmax, mmax);
  const aPlus = new Float64Array(nlm);
  const aMinus = new Float64Array(nlm);
  const mOf = new Uint32Array(nlm);
  for (let m = 0; m <= mmax; m++) {
    for (let l = m; l <= lmax; l++) {
      const lm = lmIndex(lmax, l, m);
      mOf[lm] = m;
      if (l - 1 >= m) aPlus[lm] = alphaPlus(l - 1, m);
      if (l + 1 <= lmax) aMinus[lm] = alphaMinus(l + 1, m);
    }
  }
  return { aPlus, aMinus, mOf };
}

/**
 * CPU application of the alpha shuffle to interleaved [re, im] coefficients:
 * the coefficients of sin(theta) * dtheta(u). Same index arithmetic as the
 * WGSL kernel (l +- 1 are the adjacent flat indices within an m-block; the
 * baked zeros in aPlus/aMinus kill the cross-block terms).
 */
export function applyDthetac(q: Float32Array, dc: DerivCoeffs): Float32Array {
  const nlm = dc.aPlus.length;
  const out = new Float32Array(2 * nlm);
  for (let lm = 0; lm < nlm; lm++) {
    let re = 0, im = 0;
    if (lm > 0 && dc.aPlus[lm] !== 0) { re += dc.aPlus[lm] * q[2 * lm - 2]; im += dc.aPlus[lm] * q[2 * lm - 1]; }
    if (lm + 1 < nlm && dc.aMinus[lm] !== 0) { re += dc.aMinus[lm] * q[2 * lm + 2]; im += dc.aMinus[lm] * q[2 * lm + 3]; }
    out[2 * lm] = re; out[2 * lm + 1] = im;
  }
  return out;
}

/** CPU application of dphi in coefficient space: (dphi u)_l^m = i m u_l^m. */
export function applyDphic(q: Float32Array, dc: DerivCoeffs): Float32Array {
  const nlm = dc.mOf.length;
  const out = new Float32Array(2 * nlm);
  for (let lm = 0; lm < nlm; lm++) {
    const m = dc.mOf[lm];
    out[2 * lm] = -m * q[2 * lm + 1];
    out[2 * lm + 1] = m * q[2 * lm];
  }
  return out;
}
