/**
 * Spherical conformal parameterization of a genus-0 POINT CLOUD — a port of
 * the MATLAB module's pc_spherical_conformal_map/calc_pc_laplacian:
 *
 *   G. P.-T. Choi, K. T. Ho, L. M. Lui, "Spherical Conformal Parameterization
 *   of Genus-0 Point Clouds for Meshing", SIAM J. Imaging Sci. 9(4), 2016.
 *
 * Pipeline: MLS approximation of the Laplace-Beltrami operator on the points
 * (quadratic fit over the k nearest neighbors in a PCA tangent frame,
 * Gaussian-type weights) → punctured Laplace solve at the most regular local
 * triple → inverse north-pole stereographic projection → south-pole solve
 * with the lowest 20 % fixed (replaces the mesh algorithm's quasi-conformal
 * composition; paper Thm 6.1) → one-ring balancing at the poles → north-south
 * reiterations until the map stabilizes → final balancing. All solves use the
 * no-pivot sparse LU (lu.ts) — the MLS Laplacian is nonsymmetric.
 *
 * Everything follows the MATLAB verbatim, including the 20 % fixed-set ratio
 * (the paper says 10 %), the magic 2.6562^2 scaling of the boundary triple,
 * and the convergence rule (stop below 1e-3 mean squared displacement or on
 * the first worsening, keeping the possibly-worse last map).
 */
import { fromTriplets, type CSC } from './sparse.ts';
import { lu, luOrdering, luSolve, type LUFactor } from './lu.ts';

// ---------------------------------------------------------------- k nearest neighbors
/** Indices of the k nearest points (including the point itself, first),
 *  sorted by distance — MATLAB knnsearch(vertex, vertex, 'K', k). Uniform
 *  grid binning with an expanding shell search. */
export function knn(pos: ArrayLike<number>, nv: number, k: number): Int32Array {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < nv; v++)
    for (let c = 0; c < 3; c++) { const x = pos[3 * v + c]; if (x < lo[c]) lo[c] = x; if (x > hi[c]) hi[c] = x; }
  const vol = Math.max((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]), 1e-300);
  const h = Math.max(Math.cbrt((vol * k) / nv), 1e-12);   // ~k points per cell
  const nx = Math.max(1, Math.floor((hi[0] - lo[0]) / h) + 1);
  const ny = Math.max(1, Math.floor((hi[1] - lo[1]) / h) + 1);
  const nz = Math.max(1, Math.floor((hi[2] - lo[2]) / h) + 1);
  const cellOf = (v: number): number => {
    const ix = Math.min(nx - 1, Math.floor((pos[3 * v] - lo[0]) / h));
    const iy = Math.min(ny - 1, Math.floor((pos[3 * v + 1] - lo[1]) / h));
    const iz = Math.min(nz - 1, Math.floor((pos[3 * v + 2] - lo[2]) / h));
    return ix + nx * (iy + ny * iz);
  };
  const count = new Int32Array(nx * ny * nz + 1);
  for (let v = 0; v < nv; v++) count[cellOf(v) + 1]++;
  for (let c = 0; c < nx * ny * nz; c++) count[c + 1] += count[c];
  const bucket = new Int32Array(nv), cursor = count.slice(0, nx * ny * nz);
  for (let v = 0; v < nv; v++) bucket[cursor[cellOf(v)]++] = v;

  const out = new Int32Array(nv * k);
  const cand: number[] = [], dist: number[] = [];
  for (let v = 0; v < nv; v++) {
    const px = pos[3 * v], py = pos[3 * v + 1], pz = pos[3 * v + 2];
    const cx = Math.min(nx - 1, Math.floor((px - lo[0]) / h));
    const cy = Math.min(ny - 1, Math.floor((py - lo[1]) / h));
    const cz = Math.min(nz - 1, Math.floor((pz - lo[2]) / h));
    cand.length = 0; dist.length = 0;
    // expand Chebyshev shells (each cell visited exactly once, clamping or not)
    // until the k-th distance is safely inside the searched region
    for (let ring = 0; ; ring++) {
      for (let iz = Math.max(0, cz - ring); iz <= Math.min(nz - 1, cz + ring); iz++)
        for (let iy = Math.max(0, cy - ring); iy <= Math.min(ny - 1, cy + ring); iy++)
          for (let ix = Math.max(0, cx - ring); ix <= Math.min(nx - 1, cx + ring); ix++) {
            if (Math.max(Math.abs(ix - cx), Math.abs(iy - cy), Math.abs(iz - cz)) !== ring) continue;
            const c = ix + nx * (iy + ny * iz);
            for (let p = count[c]; p < count[c + 1]; p++) {
              const w = bucket[p];
              const d = (pos[3 * w] - px) ** 2 + (pos[3 * w + 1] - py) ** 2 + (pos[3 * w + 2] - pz) ** 2;
              cand.push(w); dist.push(d);
            }
          }
      const covered = cx - ring <= 0 && cx + ring >= nx - 1 && cy - ring <= 0 && cy + ring >= ny - 1
        && cz - ring <= 0 && cz + ring >= nz - 1;
      if (cand.length >= k) {
        const idx = cand.map((_, i) => i).sort((a, b) => dist[a] - dist[b] || cand[a] - cand[b]);
        if (Math.sqrt(dist[idx[k - 1]]) <= ring * h || covered) {
          for (let i = 0; i < k; i++) out[v * k + i] = cand[idx[i]];
          break;
        }
      } else if (covered) {
        throw new Error(`knn: fewer than k=${k} points`);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- small dense helpers
/** Eigenvectors of a symmetric 3x3 (columns, eigenvalues descending) by Jacobi sweeps. */
function eig3(S: Float64Array): Float64Array {
  const a = Float64Array.from(S);
  const V = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]);
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[3 * p + q];
      if (apq === 0) continue;
      const theta = (a[3 * q + q] - a[3 * p + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let i = 0; i < 3; i++) {
        const aip = a[3 * i + p], aiq = a[3 * i + q];
        a[3 * i + p] = c * aip - s * aiq;
        a[3 * i + q] = s * aip + c * aiq;
      }
      for (let i = 0; i < 3; i++) {
        const api = a[3 * p + i], aqi = a[3 * q + i];
        a[3 * p + i] = c * api - s * aqi;
        a[3 * q + i] = s * api + c * aqi;
      }
      for (let i = 0; i < 3; i++) {
        const vip = V[3 * i + p], viq = V[3 * i + q];
        V[3 * i + p] = c * vip - s * viq;
        V[3 * i + q] = s * vip + c * viq;
      }
    }
  }
  const ev = [a[0], a[4], a[8]];
  const ord = [0, 1, 2].sort((i, j) => ev[j] - ev[i]);
  const out = new Float64Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) out[3 * r + c] = V[3 * r + ord[c]];
  return out;
}

/** Solve the 6x6 system A X = B (B is 6 x m, row-major) in place by partial-pivot elimination. */
function solve6(A: Float64Array, B: Float64Array, m: number): void {
  const n = 6;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[n * r + col]) > Math.abs(A[n * piv + col])) piv = r;
    if (piv !== col) {
      for (let c = col; c < n; c++) { const t = A[n * col + c]; A[n * col + c] = A[n * piv + c]; A[n * piv + c] = t; }
      for (let c = 0; c < m; c++) { const t = B[m * col + c]; B[m * col + c] = B[m * piv + c]; B[m * piv + c] = t; }
    }
    const d = A[n * col + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[n * r + col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[n * r + c] -= f * A[n * col + c];
      for (let c = 0; c < m; c++) B[m * r + c] -= f * B[m * col + c];
    }
  }
  for (let col = n - 1; col >= 0; col--) {
    const d = A[n * col + col];
    for (let c = 0; c < m; c++) {
      let t = B[m * col + c];
      for (let r = col + 1; r < n; r++) t -= A[n * col + r] * B[m * r + c];
      B[m * col + c] = t / d;
    }
  }
}

// ---------------------------------------------------------------- local Delaunay (rings)
/** Bowyer-Watson Delaunay of a small 2D point set; returns triangle index triples. */
export function delaunay2(xs: Float64Array, ys: Float64Array): number[][] {
  const n = xs.length;
  // super-triangle enclosing everything
  let lo0 = Infinity, hi0 = -Infinity, lo1 = Infinity, hi1 = -Infinity;
  for (let i = 0; i < n; i++) { lo0 = Math.min(lo0, xs[i]); hi0 = Math.max(hi0, xs[i]); lo1 = Math.min(lo1, ys[i]); hi1 = Math.max(hi1, ys[i]); }
  const dm = Math.max(hi0 - lo0, hi1 - lo1) || 1, mx = (lo0 + hi0) / 2, my = (lo1 + hi1) / 2;
  const px = [...xs, mx - 20 * dm, mx, mx + 20 * dm];
  const py = [...ys, my - dm, my + 20 * dm, my - dm];
  let tris: number[][] = [[n, n + 1, n + 2]];
  const circum = (t: number[]): [number, number, number] => {
    const ax = px[t[0]], ay = py[t[0]], bx = px[t[1]], by = py[t[1]], cx = px[t[2]], cy = py[t[2]];
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    return [ux, uy, (ux - ax) ** 2 + (uy - ay) ** 2];
  };
  let cc = tris.map(circum);
  for (let i = 0; i < n; i++) {
    const bad: number[] = [];
    for (let t = 0; t < tris.length; t++) {
      const [ux, uy, r2] = cc[t];
      if ((px[i] - ux) ** 2 + (py[i] - uy) ** 2 <= r2) bad.push(t);
    }
    // boundary of the bad region = edges appearing once
    const edges = new Map<string, [number, number]>();
    for (const t of bad)
      for (const [a, b] of [[tris[t][0], tris[t][1]], [tris[t][1], tris[t][2]], [tris[t][2], tris[t][0]]] as const) {
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        if (edges.has(key)) edges.delete(key);
        else edges.set(key, [a, b]);
      }
    const keep: number[][] = [], keepCC: [number, number, number][] = [];
    for (let t = 0; t < tris.length; t++) if (!bad.includes(t)) { keep.push(tris[t]); keepCC.push(cc[t]); }
    for (const [a, b] of edges.values()) { keep.push([a, b, i]); keepCC.push(circum([a, b, i])); }
    tris = keep; cc = keepCC;
  }
  return tris.filter((t) => t[0] < n && t[1] < n && t[2] < n);
}

// ---------------------------------------------------------------- MLS Laplacian
export interface PCLaplacian {
  L: CSC;
  /** k-nearest indices, self first, per point (row-major nv x k) */
  knnInd: Int32Array;
  /** local one-ring triangles per point (global indices), concatenated,
   *  with ringPtr delimiting each point's block (triples) */
  ringTris: Int32Array;
  ringPtr: Int32Array;
}

/** MLS Laplace-Beltrami approximation and local one-rings (calc_pc_laplacian). */
export function pcLaplacian(pos: ArrayLike<number>, nv: number, k = 25): PCLaplacian {
  const knnInd = knn(pos, nv, k);
  const I = new Int32Array(nv * k), J = new Int32Array(nv * k), V = new Float64Array(nv * k);
  const kx = new Float64Array(k), ky = new Float64Array(k), kz = new Float64Array(k);
  const lx = new Float64Array(k), ly = new Float64Array(k), lz = new Float64Array(k);
  const w = new Float64Array(k);
  const A = new Float64Array(36), B = new Float64Array(6 * k), b = new Float64Array(6);
  const ringTris: number[] = [], ringPtr = new Int32Array(nv + 1);
  for (let j = 0; j < nv; j++) {
    // neighbors centered on the point; the first is the point itself (zero)
    for (let i = 0; i < k; i++) {
      const q = knnInd[j * k + i];
      kx[i] = pos[3 * q] - pos[3 * j];
      ky[i] = pos[3 * q + 1] - pos[3 * j + 1];
      kz[i] = pos[3 * q + 2] - pos[3 * j + 2];
    }
    // PCA tangent frame: covariance about the NEIGHBOR MEAN (as MATLAB's pca),
    // eigenvalues descending; the smallest-variance direction is the normal
    let mx = 0, my = 0, mz = 0;
    for (let i = 0; i < k; i++) { mx += kx[i]; my += ky[i]; mz += kz[i]; }
    mx /= k; my /= k; mz /= k;
    const C = new Float64Array(9);
    for (let i = 0; i < k; i++) {
      const dx = kx[i] - mx, dy = ky[i] - my, dz = kz[i] - mz;
      C[0] += dx * dx; C[1] += dx * dy; C[2] += dx * dz;
      C[4] += dy * dy; C[5] += dy * dz; C[8] += dz * dz;
    }
    C[3] = C[1]; C[6] = C[2]; C[7] = C[5];
    const E = eig3(C);
    for (let i = 0; i < k; i++) {
      lx[i] = kx[i] * E[0] + ky[i] * E[3] + kz[i] * E[6];
      ly[i] = kx[i] * E[1] + ky[i] * E[4] + kz[i] * E[7];
      lz[i] = kx[i] * E[2] + ky[i] * E[5] + kz[i] * E[8];
    }
    // Gaussian-type weights, the center pinned to 1
    let maxnorm = 0;
    for (let i = 0; i < k; i++) maxnorm = Math.max(maxnorm, lx[i] * lx[i] + ly[i] * ly[i]);
    for (let i = 0; i < k; i++) w[i] = Math.exp((-Math.sqrt(k) * (lx[i] * lx[i] + ly[i] * ly[i])) / maxnorm) / k;
    w[0] = 1;
    // weighted quadratic fit: basis [1, x, y, x^2, xy, y^2]
    A.fill(0); B.fill(0);
    for (let i = 0; i < k; i++) {
      b[0] = 1; b[1] = lx[i]; b[2] = ly[i]; b[3] = lx[i] * lx[i]; b[4] = lx[i] * ly[i]; b[5] = ly[i] * ly[i];
      for (let r = 0; r < 6; r++) {
        B[k * r + i] = w[i] * b[r];
        for (let c = 0; c < 6; c++) A[6 * r + c] += w[i] * b[r] * b[c];
      }
    }
    solve6(A, B, k);   // B becomes c (6 x k): data values -> fit coefficients
    // derivatives of the height function over the tangent plane at the center
    let D1 = 0, D2 = 0, D3 = 0, D4 = 0, D5 = 0;
    for (let i = 0; i < k; i++) {
      D1 += B[k * 1 + i] * lz[i];
      D2 += B[k * 2 + i] * lz[i];
      D3 += 2 * B[k * 3 + i] * lz[i];
      D4 += B[k * 4 + i] * lz[i];
      D5 += 2 * B[k * 5 + i] * lz[i];
    }
    // the LB row: c' g with g the exact combination from the MATLAB (Eq 6.10)
    const g1 = 2 * D1 * D1 * D2 * D4 - D1 * D3 - D1 * D2 * D2 * D3 - D1 * D5 - D1 * D1 * D1 * D5;
    const g2 = 2 * D1 * D2 * D2 * D4 - D2 * D3 - D2 * D2 * D2 * D3 - D2 * D5 - D1 * D1 * D2 * D5;
    const g3 = (1 + D1 * D1 + D2 * D2) * (1 + D2 * D2);
    const g4 = (1 + D1 * D1 + D2 * D2) * (-2 * D1 * D2);
    const g5 = (1 + D1 * D1 + D2 * D2) * (1 + D1 * D1);
    for (let i = 0; i < k; i++) {
      I[j * k + i] = j;
      J[j * k + i] = knnInd[j * k + i];
      V[j * k + i] = B[k * 1 + i] * g1 + B[k * 2 + i] * g2 + 2 * B[k * 3 + i] * g3 + B[k * 4 + i] * g4 + 2 * B[k * 5 + i] * g5;
    }
    // local one-ring: Delaunay triangles of the projected neighbors that contain the center
    ringPtr[j] = ringTris.length / 3;
    for (const t of delaunay2(lx, ly))
      if (t.includes(0)) ringTris.push(knnInd[j * k + t[0]], knnInd[j * k + t[1]], knnInd[j * k + t[2]]);
  }
  ringPtr[nv] = ringTris.length / 3;
  return { L: fromTriplets(nv, I, J, V), knnInd, ringTris: Int32Array.from(ringTris), ringPtr };
}

// ---------------------------------------------------------------- the map
/** Replace the given rows of L with identity rows (MATLAB's L2 construction). */
function fixRows(L: CSC, rows: ArrayLike<number>): CSC {
  const isFixed = new Uint8Array(L.n);
  for (let i = 0; i < rows.length; i++) isFixed[rows[i]] = 1;
  const I: number[] = [], J: number[] = [], V: number[] = [];
  for (let j = 0; j < L.n; j++)
    for (let p = L.colptr[j]; p < L.colptr[j + 1]; p++) {
      const i = L.rowidx[p];
      if (!isFixed[i]) { I.push(i); J.push(j); V.push(L.val[p]); }
    }
  for (let r = 0; r < rows.length; r++) { I.push(rows[r]); J.push(rows[r]); V.push(1); }
  return fromTriplets(L.n, I, J, V);
}

export interface PCMapInfo {
  iterations: number;
  worstResidual: number;
  timeMs: number;
}

/** The full Algorithm 1: point cloud -> spherical conformal map (unit directions, 3 per point). */
export function pcSphericalMap(
  pos: ArrayLike<number>, nv: number,
  pc: PCLaplacian,
  onProgress?: (msg: string) => void,
  /** override of the boundary triple (validation only: MATLAB's delaunay
   *  orders the same most-regular triangle differently, and the ORDER sets
   *  the flattening target, i.e. the map's Mobius normalization) */
  boundaryTriple?: [number, number, number],
): { S: Float64Array; info: PCMapInfo } {
  const t0 = performance.now();
  const { L, ringTris, ringPtr } = pc;
  const ordering = luOrdering(L, pos);
  let worstResidual = 0;
  const solveFixed = (rows: ArrayLike<number>, dRe: Float64Array, dIm: Float64Array): [Float64Array, Float64Array] => {
    const L2 = fixRows(L, rows);
    const F: LUFactor = lu(L2, ordering);
    const b = new Float64Array(2 * nv);
    b.set(dRe); b.set(dIm, nv);
    const { x, residual } = luSolve(F, L2, b, 2);
    if (residual > worstResidual) worstResidual = residual;
    return [x.slice(0, nv), x.slice(nv)];
  };

  // most regular local triangle (over every point's one-ring triangles)
  let best = Infinity, bd0 = 0, bd1 = 0, bd2 = 0;
  if (boundaryTriple) [bd0, bd1, bd2] = boundaryTriple;
  else for (let t = 0; t < ringTris.length; t += 3) {
    const a = ringTris[t], b = ringTris[t + 1], c = ringTris[t + 2];
    const e1 = Math.hypot(pos[3 * b] - pos[3 * c], pos[3 * b + 1] - pos[3 * c + 1], pos[3 * b + 2] - pos[3 * c + 2]);
    const e2 = Math.hypot(pos[3 * a] - pos[3 * c], pos[3 * a + 1] - pos[3 * c + 1], pos[3 * a + 2] - pos[3 * c + 2]);
    const e3 = Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
    const s = e1 + e2 + e3;
    const reg = Math.abs(e1 / s - 1 / 3) + Math.abs(e2 / s - 1 / 3) + Math.abs(e3 / s - 1 / 3);
    if (reg < best) { best = reg; bd0 = a; bd1 = b; bd2 = c; }
  }
  onProgress?.('point cloud: north-pole solve…');

  // flatten the triple to the plane (the 2.6562^2 scale is the MATLAB's)
  const l12 = Math.hypot(pos[3 * bd0] - pos[3 * bd1], pos[3 * bd0 + 1] - pos[3 * bd1 + 1], pos[3 * bd0 + 2] - pos[3 * bd1 + 2]);
  const l23 = Math.hypot(pos[3 * bd1] - pos[3 * bd2], pos[3 * bd1 + 1] - pos[3 * bd2 + 1], pos[3 * bd1 + 2] - pos[3 * bd2 + 2]);
  const l31 = Math.hypot(pos[3 * bd0] - pos[3 * bd2], pos[3 * bd0 + 1] - pos[3 * bd2 + 1], pos[3 * bd0 + 2] - pos[3 * bd2 + 2]);
  const ang = Math.acos((l12 * l12 + l23 * l23 - l31 * l31) / (2 * l12 * l23));
  const scale = (2.6562 * 2.6562) / l12;
  const tri = [
    [0, 0],
    [l12 * scale, 0],
    [(-l23 * Math.cos(ang) + l12) * scale, -l23 * Math.sin(ang) * scale],
  ];
  const cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3, cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3;
  const dRe = new Float64Array(nv), dIm = new Float64Array(nv);
  const bd = [bd0, bd1, bd2];
  for (let i = 0; i < 3; i++) { dRe[bd[i]] = tri[i][0] - cx; dIm[bd[i]] = tri[i][1] - cy; }
  let [zr, zi] = solveFixed(bd, dRe, dIm);
  let mr = 0, mi = 0;
  for (let i = 0; i < nv; i++) { mr += zr[i]; mi += zi[i]; }
  mr /= nv; mi /= nv;
  const S = new Float64Array(3 * nv);
  const invNorth = (): void => {
    for (let i = 0; i < nv; i++) {
      const a = zr[i], b2 = zi[i], n2 = 1 + a * a + b2 * b2;
      S[3 * i] = (2 * a) / n2; S[3 * i + 1] = (2 * b2) / n2; S[3 * i + 2] = (-1 + a * a + b2 * b2) / n2;
    }
  };
  const invSouth = (): void => {
    for (let i = 0; i < nv; i++) {
      const a = zr[i], b2 = zi[i], n2 = 1 + a * a + b2 * b2;
      S[3 * i] = (2 * a) / n2; S[3 * i + 1] = (2 * b2) / n2; S[3 * i + 2] = -(a * a + b2 * b2 - 1) / n2;
    }
  };
  for (let i = 0; i < nv; i++) { zr[i] -= mr; zi[i] -= mi; }
  invNorth();

  // south-pole solve: the lowest 20 % anchor the plane
  const ratio = 0.2;
  const nfix = Math.floor(nv * ratio);
  const byZ = Int32Array.from({ length: nv }, (_, i) => i);
  const southFixed = (): Int32Array => Int32Array.from(byZ.sort((a, b) => S[3 * a + 2] - S[3 * b + 2]).slice(0, nfix));
  const northFixed = (): Int32Array => Int32Array.from(byZ.sort((a, b) => S[3 * b + 2] - S[3 * a + 2]).slice(0, nfix));
  const projSouth = (): void => { for (let i = 0; i < nv; i++) { const d = 1 + S[3 * i + 2]; dRe[i] = S[3 * i] / d; dIm[i] = S[3 * i + 1] / d; } };
  const projNorth = (): void => { for (let i = 0; i < nv; i++) { const d = 1 - S[3 * i + 2]; dRe[i] = S[3 * i] / d; dIm[i] = S[3 * i + 1] / d; } };
  onProgress?.('point cloud: south-pole solve…');
  {
    const fixed = southFixed();
    projSouth();
    const rhsRe = new Float64Array(nv), rhsIm = new Float64Array(nv);
    for (const f of fixed) { rhsRe[f] = dRe[f]; rhsIm[f] = dIm[f]; }
    [zr, zi] = solveFixed(fixed, rhsRe, rhsIm);
    invSouth();
  }

  // one-ring balancing at the poles (before and after the reiterations)
  const balance = (): void => {
    let north = 0, south = 0;
    for (let i = 1; i < nv; i++) { if (S[3 * i + 2] > S[3 * north + 2]) north = i; if (S[3 * i + 2] < S[3 * south + 2]) south = i; }
    const ringOf = (v: number): Set<number> => {
      const set = new Set<number>();
      for (let t = ringPtr[v]; t < ringPtr[v + 1]; t++)
        for (let c = 0; c < 3; c++) set.add(ringTris[3 * t + c]);
      set.delete(v);
      return set;
    };
    const side = (v: number, ring: Set<number>, southPole: boolean): number => {
      const d0 = southPole ? 1 + S[3 * v + 2] : 1 - S[3 * v + 2];
      const a0 = S[3 * v] / d0, b0 = S[3 * v + 1] / d0;
      let s = 0;
      for (const u of ring) {
        const d = southPole ? 1 + S[3 * u + 2] : 1 - S[3 * u + 2];
        s += Math.hypot(S[3 * u] / d - a0, S[3 * u + 1] / d - b0);
      }
      return s / ring.size;
    };
    const NT = side(north, ringOf(north), false);
    const ST = side(south, ringOf(south), true);
    const f = Math.sqrt(NT * ST) / ST;
    projSouth();
    for (let i = 0; i < nv; i++) { zr[i] = dRe[i] * f; zi[i] = dIm[i] * f; }
    invSouth();
  };
  balance();

  // north-south reiterations
  let prev = Float64Array.from(S);
  let diffPrev = Infinity;
  let iterations = 0;
  for (;;) {
    iterations++;
    onProgress?.(`point cloud: N-S reiteration ${iterations}…`);
    {
      const fixed = northFixed();
      projNorth();
      const rhsRe = new Float64Array(nv), rhsIm = new Float64Array(nv);
      for (const f of fixed) { rhsRe[f] = dRe[f]; rhsIm[f] = dIm[f]; }
      [zr, zi] = solveFixed(fixed, rhsRe, rhsIm);
      invNorth();
    }
    {
      const fixed = southFixed();
      projSouth();
      const rhsRe = new Float64Array(nv), rhsIm = new Float64Array(nv);
      for (const f of fixed) { rhsRe[f] = dRe[f]; rhsIm[f] = dIm[f]; }
      [zr, zi] = solveFixed(fixed, rhsRe, rhsIm);
      invSouth();
    }
    let diff = 0;
    for (let i = 0; i < 3 * nv; i++) diff += (S[i] - prev[i]) ** 2;
    diff /= nv;
    prev = Float64Array.from(S);
    if (diff < 0.001 || diff > diffPrev) break;
    diffPrev = diff;
  }
  balance();
  return { S, info: { iterations, worstResidual, timeMs: performance.now() - t0 } };
}
