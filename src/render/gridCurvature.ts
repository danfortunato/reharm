/**
 * Pointwise curvature of a surface x(theta, phi) on the display grid, from the
 * SPECTRAL sin-weighted derivative fields (SurfaceRenderer.derivatives): exact
 * for the band-limited surface, no finite differences — and no division by
 * sin(theta) anywhere. With the sin-weighted fundamental forms
 *
 *   Ê = sin² E = At·At      F̂ = sin F = At·x_φ     G = x_φ·x_φ
 *   L̂ = sin² L = (Att − cosθ·At)·n̂    M̂ = sin M = Atp·n̂    N = x_φφ·n̂
 *   n̂ = (At × x_φ)/|At × x_φ|         (the unit normal — scale drops out)
 *
 * the sin factors cancel exactly in the ratios:
 *
 *   2H = -(Ê N − 2 F̂ M̂ + G L̂)/(Ê G − F̂²)     K = (L̂ N − M̂²)/(Ê G − F̂²)
 *
 * so the parameterization's 1/sin² never amplifies the fields' fp32 noise in
 * any intermediate (the same undivided discipline as turing-surface's
 * flux-form Laplacian). What remains near the poles is the loss of
 * significance in the fields themselves (their signal vanishes like sin
 * there while their absolute fp32 noise does not), which the polar cap fill
 * below still covers.
 *
 * Sign convention: the normal is x_theta x x_phi (outward for the standard
 * sphere parameterization), and mean curvature is negated so the unit sphere
 * gets H = +1 (and K = +1).
 */
import type { DerivFields } from '../fit/renderGrid.ts';

/**
 * Per-grid-point curvature field (nlat * nphi), NaN where the parameterization
 * is degenerate. `theta` holds the latitudes (poles excluded).
 */
export function gridCurvature(
  d: DerivFields,
  theta: Float64Array,
  nphi: number,
  kind: 'mean' | 'gauss',
): Float32Array {
  const nlat = theta.length;
  const out = new Float32Array(nlat * nphi);
  const { At, Att, Atp, xp, xpp } = d;

  for (let i = 0; i < nlat; i++) {
    const c = Math.cos(theta[i]);
    for (let j = 0; j < nphi; j++) {
      const p = i * nphi + j;
      const ax = At[0][p], ay = At[1][p], az = At[2][p];
      const bx = xp[0][p], by = xp[1][p], bz = xp[2][p];
      const Eh = ax * ax + ay * ay + az * az;
      const Fh = ax * bx + ay * by + az * bz;
      const G = bx * bx + by * by + bz * bz;
      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;
      const nn = Math.hypot(nx, ny, nz);
      const W = Eh * G - Fh * Fh;
      if (!(W > 1e-12 * Eh * G) || !(nn > 0)) { out[p] = NaN; continue; }
      const Lh = ((Att[0][p] - c * ax) * nx + (Att[1][p] - c * ay) * ny + (Att[2][p] - c * az) * nz) / nn;
      const Mh = (Atp[0][p] * nx + Atp[1][p] * ny + Atp[2][p] * nz) / nn;
      const N = (xpp[0][p] * nx + xpp[1][p] * ny + xpp[2][p] * nz) / nn;
      out[p] = kind === 'gauss'
        ? (Lh * N - Mh * Mh) / W
        : -(Eh * N - 2 * Fh * Mh + G * Lh) / (2 * W);
    }
  }

  // Near the poles the fields' signal vanishes like sin(theta) while their
  // absolute fp32 analysis noise does not (m=0 content piling up in phase —
  // see colorbar.ts): those rows copy the nearest reliable ring.
  const RELIABLE = 0.2;
  let lo = 0; while (lo < nlat - 1 && Math.sin(theta[lo]) < RELIABLE) lo++;
  let hi = nlat - 1; while (hi > lo && Math.sin(theta[hi]) < RELIABLE) hi--;
  for (let i = 0; i < lo; i++) out.copyWithin(i * nphi, lo * nphi, (lo + 1) * nphi);
  for (let i = hi + 1; i < nlat; i++) out.copyWithin(i * nphi, hi * nphi, (hi + 1) * nphi);
  return out;
}
