/**
 * Sample the Loop limit surface of a charted mesh at directions on the sphere.
 * Positions and chart are carried as one 6-dimensional field, so both are
 * smooth limit functions over the control mesh. A query direction is located
 * piecewise-linearly on the level-1 control chart, then refined on the LIMIT
 * chart by Newton steps (Jacobian from the control triangle), and the surface
 * is evaluated at the converged (face, barycentric). The parameterization
 * sphere -> surface is therefore smooth everywhere the limit surface is.
 */
import type { Mesh } from '../mesh/types.ts';
import { LoopLimit } from '../mesh/loopLimit.ts';
import { SphereLocator } from './resample.ts';

export class LoopLimitSampler {
  readonly limit: LoopLimit;
  private readonly locator: SphereLocator;
  private readonly chart1: Float64Array;   // level-1 control chart, normalized
  private readonly across: Int32Array;     // face across edge k of face t: across[3t+k], edges (0,1),(1,2),(2,0)

  constructor(mesh: Mesh, chart: ArrayLike<number>) {
    const nv = mesh.nv, field = new Float64Array(6 * nv);
    for (let i = 0; i < nv; i++) {
      field[6 * i] = mesh.positions[3 * i]; field[6 * i + 1] = mesh.positions[3 * i + 1]; field[6 * i + 2] = mesh.positions[3 * i + 2];
      field[6 * i + 3] = chart[3 * i]; field[6 * i + 4] = chart[3 * i + 1]; field[6 * i + 5] = chart[3 * i + 2];
    }
    this.limit = new LoopLimit(mesh, field, 6);
    const m1 = this.limit.mesh;
    this.chart1 = new Float64Array(3 * m1.nv);
    for (let i = 0; i < m1.nv; i++) {
      const x = this.limit.field[6 * i + 3], y = this.limit.field[6 * i + 4], z = this.limit.field[6 * i + 5], r = Math.hypot(x, y, z) || 1;
      this.chart1[3 * i] = x / r; this.chart1[3 * i + 1] = y / r; this.chart1[3 * i + 2] = z / r;
    }
    this.locator = new SphereLocator(m1, this.chart1);
    // face adjacency across edges
    const f = m1.faces, nf = m1.nf, edge = new Map<number, number>();
    this.across = new Int32Array(3 * nf).fill(-1);
    for (let t = 0; t < nf; t++)
      for (let k = 0; k < 3; k++) {
        const a = f[3 * t + k], b = f[3 * t + ((k + 1) % 3)];
        const kk = a < b ? a * m1.nv + b : b * m1.nv + a;
        const other = edge.get(kk);
        if (other === undefined) edge.set(kk, 3 * t + k);
        else { this.across[3 * t + k] = other >> 0 === other ? Math.floor(other / 3) : -1; this.across[other] = t; }
      }
  }

  /** barycentrics of direction X in control-chart triangle t (may be outside: negative weights). */
  private baryIn(t: number, x: number, y: number, z: number): [number, number, number] {
    const f = this.limit.mesh.faces, C = this.chart1, a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const m00 = C[3 * a], m10 = C[3 * a + 1], m20 = C[3 * a + 2], m01 = C[3 * b], m11 = C[3 * b + 1], m21 = C[3 * b + 2], m02 = C[3 * c], m12 = C[3 * c + 1], m22 = C[3 * c + 2];
    const det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12) + m02 * (m10 * m21 - m20 * m11);
    const w0 = (x * (m11 * m22 - m21 * m12) - m01 * (y * m22 - z * m12) + m02 * (y * m21 - z * m11)) / det;
    const w1 = (m00 * (y * m22 - z * m12) - x * (m10 * m22 - m20 * m12) + m02 * (m10 * z - m20 * y)) / det;
    const w2 = (m00 * (m11 * z - m21 * y) - m01 * (m10 * z - m20 * y) + x * (m10 * m21 - m20 * m11)) / det;
    const s = w0 + w1 + w2;
    return [w0 / s, w1 / s, w2 / s];
  }

  /** the level-1 control mesh (for reporting) */
  get controlMesh(): Mesh { return this.limit.mesh; }

  /** Surface positions (m x 3) at the query directions; also reports the Newton residual statistics. */
  sample(queries: ArrayLike<number>, newtonSteps = 8): { values: Float64Array; fallbacks: number; maxResidual: number } {
    const m = queries.length / 3, out = new Float64Array(3 * m);
    const { face, bary, fallbacks } = this.locator.locate(queries);
    const f = this.limit.mesh.faces, C = this.chart1;
    const val = new Float64Array(6), gv = new Float64Array(6), gw = new Float64Array(6);
    let maxRes = 0;
    for (let q = 0; q < m; q++) {
      let px = queries[3 * q], py = queries[3 * q + 1], pz = queries[3 * q + 2];
      const rr = Math.hypot(px, py, pz) || 1; px /= rr; py /= rr; pz /= rr;
      let t = face[q];
      let u = bary[3 * q], v = bary[3 * q + 1], w = bary[3 * q + 2];
      let res = 0, hops = 0, nudged = false;
      for (let it = 0; it < newtonSteps; it++) {
        this.limit.evaluate(t, u, v, w, val, gv, gw);
        const cx = val[3], cy = val[4], cz = val[5], cr = Math.hypot(cx, cy, cz) || 1;
        const rx = px - cx / cr, ry = py - cy / cr, rz = pz - cz / cr;
        res = Math.hypot(rx, ry, rz);
        if (res < 1e-12) break;
        // Newton with the limit chart's own Jacobian (projected to the sphere to first order)
        const J0 = gv[3] / cr, J1 = gv[4] / cr, J2 = gv[5] / cr, J3 = gw[3] / cr, J4 = gw[4] / cr, J5 = gw[5] / cr;
        const g11 = J0 * J0 + J1 * J1 + J2 * J2, g12 = J0 * J3 + J1 * J4 + J2 * J5, g22 = J3 * J3 + J4 * J4 + J5 * J5;
        const det = g11 * g22 - g12 * g12;
        if (!(det > 1e-300)) {
          // singular Jacobian: the iterate is within the evaluator's unresolvable radius (~2^-16) of an
          // extraordinary vertex, where it returns no tangent. Nudge toward the centroid once and
          // continue; if Newton brings it back there, the solution IS that vertex to ~1e-5 of a face: accept.
          if (nudged) break;
          nudged = true;
          u = (1 - 1e-4) * u + 1e-4 / 3; v = (1 - 1e-4) * v + 1e-4 / 3; w = 1 - u - v;
          continue;
        }
        const b1 = J0 * rx + J1 * ry + J2 * rz, b2 = J3 * rx + J4 * ry + J5 * rz;
        v += (g22 * b1 - g12 * b2) / det; w += (g11 * b2 - g12 * b1) / det; u = 1 - v - w;
        // left the triangle: hop across the crossed edge and re-express the point there
        if ((u < 0 || v < 0 || w < 0) && hops < 6) {
          const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
          const k = u < 0 ? 1 : v < 0 ? 2 : 0;
          const nt = this.across[3 * t + k];
          if (nt >= 0) {
            const X = [u * C[3 * a] + v * C[3 * b] + w * C[3 * c], u * C[3 * a + 1] + v * C[3 * b + 1] + w * C[3 * c + 1], u * C[3 * a + 2] + v * C[3 * b + 2] + w * C[3 * c + 2]];
            t = nt; [u, v, w] = this.baryIn(t, X[0], X[1], X[2]); hops++;
          }
        }
        if (u < 0 || v < 0 || w < 0) { v = Math.max(v, 0); w = Math.max(w, 0); const s = v + w; if (s > 1) { v /= s; w /= s; } u = 1 - v - w; }
      }
      this.limit.evaluate(t, u, v, w, val);
      out[3 * q] = val[0]; out[3 * q + 1] = val[1]; out[3 * q + 2] = val[2];
      if (res > maxRes) maxRes = res;
    }
    return { values: out, fallbacks, maxResidual: maxRes };
  }
}
