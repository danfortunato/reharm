/**
 * Balanced chart: relax a fold-free spherical map (in practice SDEM's
 * area-equalized one) to a trade-off between conformal and area distortion,
 * minimizing over vertex positions on S^2
 *
 *   E = sum_f w_f [ alpha (s1^2 + s2^2) / (2 s1 s2) + (1 - alpha) (J + 1/J) / 2 ]
 *
 * w_f = face area share on the surface, s1, s2 = singular values of the
 * face's map (surface triangle -> chord triangle), J = s1 s2 relative to the
 * total-area ratio. Both terms are >= 1 with equality for a similarity / an
 * exactly area-preserving face, and both blow up as a face degenerates, so a
 * line search that refuses non-positive orientation keeps the map fold-free.
 *
 * Why: SDEM forces exact area equality, which through a narrow neck (a bulb
 * on a stalk) costs strong anisotropy -- 6:1 on such bulbs, 21:1 at the 99th
 * percentile on a blob with ~20 necked arms -- and the isotropic SH filter
 * then flattens those bulbs into blades. alpha = 0.5 takes that to ~1.7:1
 * while the most-crowded faces still get > 0.2x their area share; the lmax-127
 * truncation error of that blob drops 4x.
 *
 * Terms per face (a, b, c the mapped unit vectors; As, cot_* the surface
 * triangle's area and corner cotangents):
 *   s1^2 + s2^2 = (cot_a |c-b|^2 + cot_b |a-c|^2 + cot_c |b-a|^2) / (2 As)
 *   s1 s2       = a . (b x c) / (2 As)   (signed: continuous through a face
 *                                          degenerating onto a great circle)
 * Minimized by L-BFGS on the sphere with a per-vertex diagonal preconditioner
 * (incident chart area over incident surface weight): the gradient is
 * largest on the tiniest chart faces, and an unscaled step folds them.
 */
import type { Mesh } from '../mesh/types.ts';

export interface BalancedOptions {
  alpha?: number;
  maxIter?: number;
  /** stop once the energy has dropped by less than this relative amount over the last 50 iterations */
  tol?: number;
  onIter?: (iter: number, energy: number) => void;
  yieldStep?: () => Promise<void>;
}

export async function balancedRelax(m: Mesh, S0: Float64Array, opt: BalancedOptions = {}): Promise<{ S: Float64Array; iterations: number; energy: number }> {
  const { nf, nv } = m;
  const P = m.positions;
  const alpha = opt.alpha ?? 0.5, maxIter = opt.maxIter ?? 1000, tol = opt.tol ?? 1e-5;
  const n3 = 3 * nv;
  // work with the face orientation under which the map faces outward (uploads come either way)
  let f = m.faces;
  {
    let pos = 0;
    for (let t = 0; t < nf; t++) {
      const ia = 3 * f[3 * t], ib = 3 * f[3 * t + 1], ic = 3 * f[3 * t + 2];
      const vol = S0[ia] * (S0[ib + 1] * S0[ic + 2] - S0[ib + 2] * S0[ic + 1]) + S0[ia + 1] * (S0[ib + 2] * S0[ic] - S0[ib] * S0[ic + 2])
                + S0[ia + 2] * (S0[ib] * S0[ic + 1] - S0[ib + 1] * S0[ic]);
      if (vol > 0) pos++;
    }
    if (pos < nf / 2) {
      f = f.slice();
      for (let t = 0; t < nf; t++) { const b = f[3 * t + 1]; f[3 * t + 1] = f[3 * t + 2]; f[3 * t + 2] = b; }
    }
  }

  // surface triangles: area weights and corner cotangents
  const As = new Float64Array(nf), cot = new Float64Array(3 * nf);
  let Atot = 0;
  for (let t = 0; t < nf; t++) {
    const i = [f[3 * t], f[3 * t + 1], f[3 * t + 2]];
    const p = i.map((v) => [P[3 * v], P[3 * v + 1], P[3 * v + 2]]);
    const ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
    const vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
    const A = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    As[t] = A; Atot += A;
    for (let k = 0; k < 3; k++) {
      const o = p[k], q = p[(k + 1) % 3], r = p[(k + 2) % 3];
      const ax = q[0] - o[0], ay = q[1] - o[1], az = q[2] - o[2];
      const bx = r[0] - o[0], by = r[1] - o[1], bz = r[2] - o[2];
      cot[3 * t + k] = (ax * bx + ay * by + az * bz) / (2 * A);   // cot = dot / |cross|, |cross| = 2A
    }
  }
  const w = Float64Array.from(As, (a) => a / Atot);
  const scale = (4 * Math.PI) / Atot;

  // energy (Infinity if any face is folded or degenerate) and, optionally, its gradient w.r.t. X on the sphere
  const evaluate = (X: Float64Array, grad: Float64Array | null): number => {
    if (grad) grad.fill(0);
    let E = 0;
    for (let t = 0; t < nf; t++) {
      const ia = 3 * f[3 * t], ib = 3 * f[3 * t + 1], ic = 3 * f[3 * t + 2];
      const ax = X[ia], ay = X[ia + 1], az = X[ia + 2];
      const bx = X[ib], by = X[ib + 1], bz = X[ib + 2];
      const cx = X[ic], cy = X[ic + 1], cz = X[ic + 2];
      const bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;   // b x c
      const det = (ax * bcx + ay * bcy + az * bcz) / (2 * As[t]);
      if (!(det > 0)) return Infinity;
      const ca = cot[3 * t], cb = cot[3 * t + 1], cc = cot[3 * t + 2];
      const ebcx = cx - bx, ebcy = cy - by, ebcz = cz - bz;   // c - b (opposite a)
      const ecax = ax - cx, ecay = ay - cy, ecaz = az - cz;   // a - c (opposite b)
      const eabx = bx - ax, eaby = by - ay, eabz = bz - az;   // b - a (opposite c)
      const fro2 = (ca * (ebcx * ebcx + ebcy * ebcy + ebcz * ebcz) + cb * (ecax * ecax + ecay * ecay + ecaz * ecaz)
                  + cc * (eabx * eabx + eaby * eaby + eabz * eabz)) / (2 * As[t]);
      const Jn = det / scale;
      E += w[t] * (alpha * fro2 / (2 * det) + (1 - alpha) * 0.5 * (Jn + 1 / Jn));
      if (!grad) continue;
      const gF = w[t] * alpha / (2 * det);                                                    // dE/dfro2
      const gD = w[t] * (-alpha * fro2 / (2 * det * det) + (1 - alpha) * 0.5 * (1 / scale - scale / (det * det)));   // dE/ddet
      const kF = gF / As[t], kD = gD / (2 * As[t]);
      // d fro2: a: (cot_b (a-c) - cot_c (b-a)) / As, b: (cot_c (b-a) - cot_a (c-b)) / As, c: (cot_a (c-b) - cot_b (a-c)) / As
      // d det : a: (b x c), b: (c x a), c: (a x b), all / (2 As)
      const caxx = cy * az - cz * ay, caxy = cz * ax - cx * az, caxz = cx * ay - cy * ax;   // c x a
      const abxx = ay * bz - az * by, abxy = az * bx - ax * bz, abxz = ax * by - ay * bx;   // a x b
      grad[ia] += kF * (cb * ecax - cc * eabx) + kD * bcx;
      grad[ia + 1] += kF * (cb * ecay - cc * eaby) + kD * bcy;
      grad[ia + 2] += kF * (cb * ecaz - cc * eabz) + kD * bcz;
      grad[ib] += kF * (cc * eabx - ca * ebcx) + kD * caxx;
      grad[ib + 1] += kF * (cc * eaby - ca * ebcy) + kD * caxy;
      grad[ib + 2] += kF * (cc * eabz - ca * ebcz) + kD * caxz;
      grad[ic] += kF * (ca * ebcx - cb * ecax) + kD * abxx;
      grad[ic + 1] += kF * (ca * ebcy - cb * ecay) + kD * abxy;
      grad[ic + 2] += kF * (ca * ebcz - cb * ecaz) + kD * abxz;
    }
    if (grad)   // tangential projection (X stays on the unit sphere)
      for (let v = 0; v < nv; v++) {
        const x = X[3 * v], y = X[3 * v + 1], z = X[3 * v + 2];
        const d = grad[3 * v] * x + grad[3 * v + 1] * y + grad[3 * v + 2] * z;
        grad[3 * v] -= d * x; grad[3 * v + 1] -= d * y; grad[3 * v + 2] -= d * z;
      }
    return E;
  };

  const normalize = (X: Float64Array) => {
    for (let i = 0; i < n3; i += 3) { const r = Math.hypot(X[i], X[i + 1], X[i + 2]) || 1; X[i] /= r; X[i + 1] /= r; X[i + 2] /= r; }
  };
  /** per-vertex inverse-Hessian-diagonal estimate: incident chart area / incident surface weight */
  const precond = (X: Float64Array): Float64Array => {
    const a = new Float64Array(nv), ww = new Float64Array(nv);
    for (let t = 0; t < nf; t++) {
      const ia = 3 * f[3 * t], ib = 3 * f[3 * t + 1], ic = 3 * f[3 * t + 2];
      const ux = X[ib] - X[ia], uy = X[ib + 1] - X[ia + 1], uz = X[ib + 2] - X[ia + 2];
      const vx = X[ic] - X[ia], vy = X[ic + 1] - X[ia + 1], vz = X[ic + 2] - X[ia + 2];
      const At = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      for (let k = 0; k < 3; k++) { a[f[3 * t + k]] += At; ww[f[3 * t + k]] += w[t]; }
    }
    const H = new Float64Array(n3);
    for (let v = 0; v < nv; v++) H[3 * v] = H[3 * v + 1] = H[3 * v + 2] = a[v] / ww[v];
    return H;
  };
  const dot = (x: Float64Array, y: Float64Array) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; };

  let X = S0.slice(); normalize(X);
  let g = new Float64Array(n3);
  let E = evaluate(X, g);
  if (!Number.isFinite(E)) return { S: X, iterations: 0, energy: E };   // folded start: nothing to do
  const sHist: Float64Array[] = [], yHist: Float64Array[] = [];
  const M = 10;
  const Ehist: number[] = [E];
  const cand = new Float64Array(n3), gNew = new Float64Array(n3);
  let iter = 0;
  for (; iter < maxIter; iter++) {
    if (opt.yieldStep && iter % 10 === 0) await opt.yieldStep();
    const H0 = precond(X);
    // L-BFGS two-loop recursion with the diagonal H0
    const q = g.slice();
    const al: number[] = [];
    for (let k = sHist.length - 1; k >= 0; k--) {
      const rho = 1 / dot(yHist[k], sHist[k]);
      const a = rho * dot(sHist[k], q); al[k] = a;
      for (let i = 0; i < n3; i++) q[i] -= a * yHist[k][i];
    }
    let gamma = 1;
    if (sHist.length) {
      const s = sHist[sHist.length - 1], y = yHist[yHist.length - 1];
      let yHy = 0; for (let i = 0; i < n3; i++) yHy += y[i] * H0[i] * y[i];
      gamma = dot(s, y) / yHy;
    }
    for (let i = 0; i < n3; i++) q[i] *= gamma * H0[i];
    for (let k = 0; k < sHist.length; k++) {
      const rho = 1 / dot(yHist[k], sHist[k]);
      const b = rho * dot(yHist[k], q);
      for (let i = 0; i < n3; i++) q[i] += sHist[k][i] * (al[k] - b);
    }
    let p = q; for (let i = 0; i < n3; i++) p[i] = -p[i];
    let slope = dot(g, p);
    const steepest = () => {
      sHist.length = 0; yHist.length = 0;
      p = new Float64Array(n3); for (let i = 0; i < n3; i++) p[i] = -H0[i] * g[i];
      slope = dot(g, p);
      let mx = 0; for (const v of p) mx = Math.max(mx, Math.abs(v));
      return 1e-2 / mx;
    };
    let step = 1;
    if (!(slope < 0)) step = steepest();
    else if (!sHist.length) { let mx = 0; for (const v of p) mx = Math.max(mx, Math.abs(v)); step = 1e-2 / mx; }
    // backtracking Armijo line search; Infinity (a fold) is always rejected
    let En = Infinity, accepted = false, restarted = !sHist.length;
    for (let h = 0; h < 60; h++) {
      for (let i = 0; i < n3; i++) cand[i] = X[i] + step * p[i];
      normalize(cand);
      En = evaluate(cand, null);
      if (En <= E + 1e-4 * step * slope) { accepted = true; break; }
      step *= 0.5;
      if (step * Math.abs(slope) < 1e-16 * Math.abs(E)) {
        if (restarted) break;
        step = steepest(); restarted = true;
      }
    }
    if (!accepted) break;
    evaluate(cand, gNew);
    const s = new Float64Array(n3), y = new Float64Array(n3);
    for (let i = 0; i < n3; i++) { s[i] = cand[i] - X[i]; y[i] = gNew[i] - g[i]; }
    const sy = dot(s, y);
    if (sy > 1e-12 * Math.sqrt(dot(s, s) * dot(y, y))) {
      sHist.push(s); yHist.push(y);
      if (sHist.length > M) { sHist.shift(); yHist.shift(); }
    }
    X.set(cand); g.set(gNew); E = En;
    Ehist.push(E);
    opt.onIter?.(iter + 1, E);
    if (Ehist.length > 50 && Ehist[Ehist.length - 51] - E < tol * Math.abs(E)) { iter++; break; }
  }
  return { S: X, iterations: iter, energy: E };
}
