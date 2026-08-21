/**
 * Synthesis on a display grid chosen independently of the analysis band-limit:
 * a plan at degree min(lmax, Nplot) on the Gauss grid of degree Nplot. The
 * coefficients are re-laid out (truncated or zero-padded) into that plan's
 * SHTns layout. Coarser than lmax means the display cannot show degrees
 * above Nplot -- they are dropped, which is what sampling them on that grid
 * would alias anyway; finer than lmax is exact interpolation.
 */
import { ShtPlan } from '../sht/sht.ts';
import { applyDphic, applyDthetac, derivCoeffs, type DerivCoeffs } from '../sht/derivCoeffs.ts';
import { gridForLmax, lmIndex, type ShtConfig } from '../sht/layout.ts';
import { applyGains } from './filters.ts';
import type { Coefficients } from './gridFit.ts';

/** Copy coefficients between SHTns layouts of different band-limits. */
export function relayout(q: Float32Array, lmaxFrom: number, lmaxTo: number): Float32Array {
  if (lmaxFrom === lmaxTo) return q;
  const out = new Float32Array(((lmaxTo + 1) * (lmaxTo + 2)) / 2 * 2);
  const L = Math.min(lmaxFrom, lmaxTo);
  for (let m = 0; m <= L; m++)
    for (let l = m; l <= L; l++) {
      const a = 2 * lmIndex(lmaxFrom, l, m), b = 2 * lmIndex(lmaxTo, l, m);
      out[b] = q[a]; out[b + 1] = q[a + 1];
    }
  return out;
}

/** Sin-weighted (UNDIVIDED) derivative fields of each coordinate on the grid:
 *  At = sinθ·x_θ, Att = synth(dthetac² q) = sinθcosθ·x_θ + sin²θ·x_θθ,
 *  Atp = sinθ·x_θφ, xp = x_φ, xpp = x_φφ — all direct syntheses, no 1/sinθ
 *  anywhere. Curvature combines them so the sin factors cancel exactly on
 *  paper (gridCurvature.ts), instead of numerically after a 1/sin² blowup —
 *  the same undivided-flux discipline as turing-surface's Laplacian. */
export interface DerivFields {
  At: Float32Array[]; Att: Float32Array[]; Atp: Float32Array[];
  xp: Float32Array[]; xpp: Float32Array[];
}

export class SurfaceRenderer {
  readonly cfg: ShtConfig;
  readonly plan: ShtPlan;
  readonly phi: Float64Array;
  private dc: DerivCoeffs | null = null;
  private constructor(plan: ShtPlan, cfg: ShtConfig) {
    this.plan = plan; this.cfg = cfg;
    this.phi = new Float64Array(cfg.nphi);
    for (let j = 0; j < cfg.nphi; j++) this.phi[j] = (2 * Math.PI * j) / cfg.nphi;
  }
  /** Display grid of degree `nplot` (nlat ~ nplot+1, nphi = next power of two >= 2 nplot+1), carrying degrees up to min(lmax, nplot). */
  static async create(device: GPUDevice, lmax: number, nplot: number): Promise<SurfaceRenderer> {
    const { nlat, nphi } = gridForLmax(nplot, 1);
    const l = Math.min(lmax, nplot);
    const cfg: ShtConfig = { lmax: l, mmax: l, nlat, nphi };
    return new SurfaceRenderer(await ShtPlan.create(device, cfg), cfg);
  }
  async synthesize(coef: Coefficients, gains?: ArrayLike<number>): Promise<Float32Array> {
    const fields: Float32Array[] = [];
    for (const q0 of coef.q) {
      const q = gains ? applyGains(q0, coef.lmax, coef.mmax, gains) : q0;
      fields.push(await this.plan.synth(relayout(q, coef.lmax, this.cfg.lmax)));   // sequential: shared staging buffer
    }
    const n = this.cfg.nlat * this.cfg.nphi, out = new Float32Array(n * 3);
    for (let p = 0; p < n; p++) { out[3 * p] = fields[0][p]; out[3 * p + 1] = fields[1][p]; out[3 * p + 2] = fields[2][p]; }
    return out;
  }
  /**
   * Sin-weighted derivatives of each coordinate of the (filtered) surface on
   * the display grid, fully spectrally: the derivative operators are applied
   * in coefficient space on the CPU (dtheta via the alpha shuffle, dphi as
   * i*m, composed for the second derivatives) and each result synthesized by
   * the existing plan. No finite differences and NO division by sin(theta)
   * anywhere — the fields are returned undivided (see DerivFields) and the
   * curvature formulas cancel the sin factors analytically:
   *
   *   At  := synth(dthetac q)            = sin * x_t
   *   Att := synth(dthetac dthetac q)    = sin * d/dtheta(At) = sin cos x_t + sin^2 x_tt
   *   Atp := synth(dphic dthetac q)      = sin * x_tp
   *
   * Sequential for the same reason as synthesize(); must run on the same
   * serialized GPU queue.
   */
  async derivatives(coef: Coefficients, gains?: ArrayLike<number>): Promise<DerivFields> {
    this.dc ??= derivCoeffs(this.cfg.lmax, this.cfg.mmax);
    const out: DerivFields = { At: [], Att: [], Atp: [], xp: [], xpp: [] };
    for (const q0 of coef.q) {
      const q = gains ? applyGains(q0, coef.lmax, coef.mmax, gains) : q0;
      const ql = relayout(q, coef.lmax, this.cfg.lmax);
      const qA = applyDthetac(ql, this.dc), qC = applyDphic(ql, this.dc);
      out.At.push(await this.plan.synth(qA));                          // sequential:
      out.Att.push(await this.plan.synth(applyDthetac(qA, this.dc)));  // shared staging buffer
      out.xp.push(await this.plan.synth(qC));
      out.Atp.push(await this.plan.synth(applyDphic(qA, this.dc)));
      out.xpp.push(await this.plan.synth(applyDphic(qC, this.dc)));
    }
    return out;
  }
  destroy(): void { this.plan.destroy(); }
}
