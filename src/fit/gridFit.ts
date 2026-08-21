/**
 * The transform engine: the three coordinate fields sampled on the
 * Gauss-Legendre x equispaced grid go through the GPU analysis once; every
 * filter change is three GPU syntheses of the re-weighted coefficients. This
 * is the Gu et al. (2004) route of the MATLAB module
 * (fit_spherical_harmonics_grid), with shtns-webgpu as the transform.
 */
import { ShtPlan } from '../sht/sht.ts';
import { gridForLmax, nlmCalc, type ShtConfig } from '../sht/layout.ts';
import { applyGains } from './filters.ts';

export interface Coefficients {
  lmax: number;
  mmax: number;
  /** SHTns-layout complex coefficients of x, y, z (interleaved re/im). */
  q: [Float32Array, Float32Array, Float32Array];
}

export class GridFitter {
  readonly cfg: ShtConfig;
  readonly plan: ShtPlan;
  /** colatitudes of the grid rows, in the order of the spatial layout */
  readonly theta: Float64Array;
  readonly phi: Float64Array;

  private constructor(plan: ShtPlan, cfg: ShtConfig) {
    this.plan = plan;
    this.cfg = cfg;
    this.theta = Float64Array.from(plan.cosTheta, (c) => Math.acos(Math.max(-1, Math.min(1, c))));
    this.phi = new Float64Array(cfg.nphi);
    for (let j = 0; j < cfg.nphi; j++) this.phi[j] = (2 * Math.PI * j) / cfg.nphi;
  }

  /**
   * oversample 1: the minimal exact quadrature grid (nlat ~ lmax+1, nphi = next power of two
   * >= 2 lmax+1); k: roughly k times as many latitudes and longitudes. The resampled surface
   * is not band-limited, and a finer grid aliases less of its content above lmax into the
   * retained coefficients; cost grows as k^2.
   */
  static async create(device: GPUDevice, lmax: number, oversample: number): Promise<GridFitter> {
    const { nlat, nphi } = gridForLmax(lmax, 2 * oversample - 1); // gridForLmax's rule: nlat ~ (pdeg+1)(lmax+1)/2
    const cfg: ShtConfig = { lmax, mmax: lmax, nlat, nphi };
    const plan = await ShtPlan.create(device, cfg);
    return new GridFitter(plan, cfg);
  }

  get nlm(): number {
    return nlmCalc(this.cfg.lmax, this.cfg.mmax);
  }

  /** Unit-sphere directions of the grid points, interleaved xyz, row-major (lat, phi). */
  gridDirections(): Float64Array {
    const { nlat, nphi } = this.cfg;
    const d = new Float64Array(nlat * nphi * 3);
    for (let i = 0; i < nlat; i++) {
      const st = Math.sin(this.theta[i]), ct = Math.cos(this.theta[i]);
      for (let j = 0; j < nphi; j++) {
        const p = 3 * (i * nphi + j);
        d[p] = st * Math.cos(this.phi[j]);
        d[p + 1] = st * Math.sin(this.phi[j]);
        d[p + 2] = ct;
      }
    }
    return d;
  }

  /**
   * Forward transform of the three coordinate fields sampled on the grid (f64
   * samples), with ONE mixed-precision refinement pass. The fp32 analysis
   * leaves every coefficient an ABSOLUTE error ~eps32*||f||, which drowns the
   * tiny high-degree coefficients of a smooth surface — after the ~l^2
   * amplification of second derivatives this was the polar curvature rings.
   * Analyzing the f64 residual f - synth(q1) once more and adding contracts
   * that error by ~eps32 (classic iterative refinement; the transform is
   * well-conditioned): the coefficients become accurate RELATIVE to their own
   * size, which differentiation tolerates, and the fp32 storage they return
   * to only costs relative rounding. On the exact sphere the high degrees
   * come out exactly zero. Costs one extra synthesis + analysis per field,
   * once per fit.
   */
  async analyze(spat: [Float64Array, Float64Array, Float64Array], refine = true): Promise<Coefficients> {
    // the plan's convenience transforms share a staging buffer: run them one at a time
    const n = this.cfg.nlat * this.cfg.nphi;
    const f32 = new Float32Array(n);
    const q: Float32Array[] = [];
    for (const s of spat) {
      for (let p = 0; p < n; p++) f32[p] = s[p];
      const q1 = await this.plan.analys(f32);
      if (!refine) { q.push(q1); continue; }
      const s1 = await this.plan.synth(q1);
      for (let p = 0; p < n; p++) f32[p] = s[p] - s1[p];   // f64 residual, small enough for f32 storage
      const dq = await this.plan.analys(f32);
      const out = new Float32Array(q1.length);
      for (let k = 0; k < q1.length; k++) out[k] = q1[k] + dq[k];
      q.push(out);
    }
    return { lmax: this.cfg.lmax, mmax: this.cfg.mmax, q: q as [Float32Array, Float32Array, Float32Array] };
  }

  /** Synthesize (optionally re-weighted) coefficients on the grid: interleaved xyz per grid point. */
  async synthesize(coef: Coefficients, gains?: ArrayLike<number>): Promise<Float32Array> {
    const qs = gains ? coef.q.map((q) => applyGains(q, coef.lmax, coef.mmax, gains)) : coef.q;
    const fields: Float32Array[] = [];
    for (const q of qs) fields.push(await this.plan.synth(q));   // sequential: shared staging buffer
    const n = this.cfg.nlat * this.cfg.nphi;
    const out = new Float32Array(n * 3);
    for (let p = 0; p < n; p++) {
      out[3 * p] = fields[0][p];
      out[3 * p + 1] = fields[1][p];
      out[3 * p + 2] = fields[2][p];
    }
    return out;
  }

  destroy(): void {
    this.plan.destroy();
  }
}
