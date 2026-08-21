/**
 * Sparse symmetric positive-definite solves for the chart computations.
 *
 *  - triplets -> compressed-sparse-column (duplicates summed)
 *  - fill-reducing ordering: geometric nested dissection. Every matrix here is
 *    the graph of a surface mesh whose vertices have coordinates; a plane cut
 *    through such a mesh is a curve of O(sqrt n) vertices, so recursively
 *    cutting the longest bounding-box axis and ordering the cut vertices last
 *    gives the O(n log n) fill of a planar-graph ordering without a quotient
 *    graph.
 *  - Cholesky: up-looking, after Davis' CSparse (cs_etree / cs_ereach /
 *    cs_chol): the elimination tree gives each row's pattern as a tree reach,
 *    so the symbolic and numeric passes are the same loop.
 */

export interface CSC {
  n: number;
  colptr: Int32Array;
  rowidx: Int32Array;
  val: Float64Array;
}

/** Assemble an n x n CSC matrix from triplets, summing duplicates. */
export function fromTriplets(n: number, I: ArrayLike<number>, J: ArrayLike<number>, V: ArrayLike<number>): CSC {
  const nnz = I.length;
  const count = new Int32Array(n + 1);
  for (let k = 0; k < nnz; k++) count[J[k] + 1]++;
  for (let j = 0; j < n; j++) count[j + 1] += count[j];
  const rows = new Int32Array(nnz), vals = new Float64Array(nnz), next = count.slice(0, n);
  for (let k = 0; k < nnz; k++) { const p = next[J[k]]++; rows[p] = I[k]; vals[p] = V[k]; }
  // sort rows within each column and merge duplicates
  const colptr = new Int32Array(n + 1);
  const rowidx = new Int32Array(nnz), val = new Float64Array(nnz);
  let q = 0;
  const idx: number[] = [];
  for (let j = 0; j < n; j++) {
    colptr[j] = q;
    idx.length = 0;
    for (let p = count[j]; p < count[j + 1]; p++) idx.push(p);
    idx.sort((a, b) => rows[a] - rows[b]);
    let last = -1;
    for (const p of idx) {
      if (rows[p] === last) val[q - 1] += vals[p];
      else { rowidx[q] = rows[p]; val[q] = vals[p]; last = rows[p]; q++; }
    }
  }
  colptr[n] = q;
  return { n, colptr: colptr, rowidx: rowidx.slice(0, q), val: val.slice(0, q) };
}

/** y = A x (general CSC). */
export function multiply(A: CSC, x: ArrayLike<number>, y = new Float64Array(A.n)): Float64Array {
  y.fill(0);
  for (let j = 0; j < A.n; j++) {
    const xj = x[j];
    if (xj === 0) continue;
    for (let p = A.colptr[j]; p < A.colptr[j + 1]; p++) y[A.rowidx[p]] += A.val[p] * xj;
  }
  return y;
}

/** Symmetric permutation B = P A P^T with pinv[old] = new (full pattern kept). */
function permuteSymmetric(A: CSC, pinv: Int32Array): CSC {
  const nnz = A.colptr[A.n];
  const I = new Int32Array(nnz), J = new Int32Array(nnz);
  let k = 0;
  for (let j = 0; j < A.n; j++)
    for (let p = A.colptr[j]; p < A.colptr[j + 1]; p++) { I[k] = pinv[A.rowidx[p]]; J[k] = pinv[j]; k++; }
  return fromTriplets(A.n, I, J, A.val);
}

/**
 * Geometric nested dissection: returns perm (new -> old). `coords` holds
 * 3 numbers per vertex. Leaves of <= `leaf` vertices are ordered as they come.
 */
export function nestedDissection(A: CSC, coords: ArrayLike<number>, leaf = 48): Int32Array {
  const n = A.n;
  const perm = new Int32Array(n);
  const side = new Int8Array(n); // scratch: which half a vertex is in (0 = not in the current subset)
  // post-order emission (separator after both halves) with an explicit task stack
  type Task = { kind: 'split' | 'emit'; set: Int32Array };
  const tasks: Task[] = [{ kind: 'split', set: Int32Array.from({ length: n }, (_, i) => i) }];
  const order: number[] = []; // final order built front-to-back
  while (tasks.length) {
    const t = tasks.pop()!;
    if (t.kind === 'emit') { for (const v of t.set) order.push(v); continue; }
    const set = t.set;
    if (set.length <= leaf) { for (const v of set) order.push(v); continue; }
    // longest bounding-box axis, median split
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const v of set) for (let c = 0; c < 3; c++) { const x = coords[3 * v + c]; if (x < lo[c]) lo[c] = x; if (x > hi[c]) hi[c] = x; }
    let ax = 0; for (let c = 1; c < 3; c++) if (hi[c] - lo[c] > hi[ax] - lo[ax]) ax = c;
    const sorted = set.slice().sort((a, b) => coords[3 * a + ax] - coords[3 * b + ax]);
    const half = sorted.length >> 1;
    for (let i = 0; i < sorted.length; i++) side[sorted[i]] = i < half ? 1 : 2;
    // separator: vertices of side 1 adjacent to side 2
    const sepA: number[] = [], restA: number[] = [], restB: number[] = [];
    for (let i = 0; i < half; i++) {
      const v = sorted[i];
      let cut = false;
      for (let p = A.colptr[v]; p < A.colptr[v + 1]; p++) if (side[A.rowidx[p]] === 2) { cut = true; break; }
      (cut ? sepA : restA).push(v);
    }
    for (let i = half; i < sorted.length; i++) restB.push(sorted[i]);
    for (const v of set) side[v] = 0;
    if (sepA.length === 0 || restA.length === 0 || restB.length === 0) {
      // disconnected or degenerate split: just order the set
      for (const v of set) order.push(v);
      continue;
    }
    // post-order: halves first (pushed last = popped first), separator after
    tasks.push({ kind: 'emit', set: Int32Array.from(sepA) });
    tasks.push({ kind: 'split', set: Int32Array.from(restB) });
    tasks.push({ kind: 'split', set: Int32Array.from(restA) });
  }
  for (let i = 0; i < n; i++) perm[i] = order[i];
  return perm;
}

export interface CholeskyFactor {
  n: number;
  perm: Int32Array;  // new -> old
  pinv: Int32Array;  // old -> new
  Lp: Int32Array;    // CSC of L (lower triangular, diagonal first in each column)
  Li: Int32Array;
  Lx: Float64Array;
}

/** Elimination tree of the (permuted, symmetric) matrix from its upper triangle. */
function etree(C: CSC): Int32Array {
  const n = C.n, parent = new Int32Array(n).fill(-1), ancestor = new Int32Array(n).fill(-1);
  for (let k = 0; k < n; k++)
    for (let p = C.colptr[k]; p < C.colptr[k + 1]; p++) {
      let i = C.rowidx[p];
      while (i !== -1 && i < k) {
        const inext = ancestor[i];
        ancestor[i] = k;
        if (inext === -1) parent[i] = k;
        i = inext;
      }
    }
  return parent;
}

/** Nonzero pattern of row k of L: reach of column k's upper entries in the etree. Returns top; pattern is s[top..n). */
function ereach(C: CSC, k: number, parent: Int32Array, s: Int32Array, w: Int32Array): number {
  const n = C.n;
  let top = n;
  w[k] = k; // mark k
  for (let p = C.colptr[k]; p < C.colptr[k + 1]; p++) {
    let i = C.rowidx[p];
    if (i > k) continue;
    let len = 0;
    while (w[i] !== k) { s[len++] = i; w[i] = k; i = parent[i]; }
    while (len > 0) s[--top] = s[--len];
  }
  return top;
}

/**
 * Cholesky factorization P A P^T = L L^T of a symmetric positive-definite CSC
 * matrix (full pattern supplied). Throws if a pivot is not positive.
 */
export function cholesky(A: CSC, perm: Int32Array): CholeskyFactor {
  const n = A.n;
  const pinv = new Int32Array(n);
  for (let i = 0; i < n; i++) pinv[perm[i]] = i;
  const C = permuteSymmetric(A, pinv);
  const parent = etree(C);
  const s = new Int32Array(n), w = new Int32Array(n).fill(-1);
  // symbolic: column counts of L
  const cnt = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const top = ereach(C, k, parent, s, w);
    for (let t = top; t < n; t++) cnt[s[t]]++;
    cnt[k]++; // diagonal
  }
  const Lp = new Int32Array(n + 1);
  for (let j = 0; j < n; j++) Lp[j + 1] = Lp[j] + cnt[j];
  const Li = new Int32Array(Lp[n]), Lx = new Float64Array(Lp[n]);
  const c = Lp.slice(0, n); // next free slot per column
  const x = new Float64Array(n);
  w.fill(-1);
  for (let k = 0; k < n; k++) {
    const top = ereach(C, k, parent, s, w);
    x[k] = 0;
    for (let p = C.colptr[k]; p < C.colptr[k + 1]; p++) if (C.rowidx[p] <= k) x[C.rowidx[p]] = C.val[p];
    let d = x[k];
    x[k] = 0;
    for (let t = top; t < n; t++) {
      const i = s[t];
      const lki = x[i] / Lx[Lp[i]];
      x[i] = 0;
      for (let p = Lp[i] + 1; p < c[i]; p++) x[Li[p]] -= Lx[p] * lki;
      d -= lki * lki;
      const p = c[i]++;
      Li[p] = k; Lx[p] = lki;
    }
    if (!(d > 0)) throw new Error(`cholesky: matrix not positive definite (pivot ${d} at ${k})`);
    const p = c[k]++;
    Li[p] = k; Lx[p] = Math.sqrt(d);
  }
  return { n, perm, pinv, Lp, Li, Lx };
}

/** Solve A x = b with a factor of A. `b` may have several right-hand sides (column-major, n each). */
export function solve(F: CholeskyFactor, b: ArrayLike<number>, nrhs = 1): Float64Array {
  const { n, perm, Lp, Li, Lx } = F;
  const x = new Float64Array(n * nrhs);
  const y = new Float64Array(n);
  for (let r = 0; r < nrhs; r++) {
    for (let i = 0; i < n; i++) y[i] = b[r * n + perm[i]];
    // forward: L y = y
    for (let j = 0; j < n; j++) {
      const yj = (y[j] /= Lx[Lp[j]]);
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) y[Li[p]] -= Lx[p] * yj;
    }
    // backward: L^T z = y
    for (let j = n - 1; j >= 0; j--) {
      let t = y[j];
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) t -= Lx[p] * y[Li[p]];
      y[j] = t / Lx[Lp[j]];
    }
    for (let i = 0; i < n; i++) x[r * n + perm[i]] = y[i];
  }
  return x;
}

/** Convenience: factor with nested dissection and solve. */
export function spdSolve(A: CSC, coords: ArrayLike<number>, b: ArrayLike<number>, nrhs = 1): Float64Array {
  const perm = nestedDissection(A, coords);
  return solve(cholesky(A, perm), b, nrhs);
}

/** Jacobi-preconditioned conjugate gradients for SPD A: x0 is both the warm start and the output. */
export function pcgJacobi(A: CSC, b: ArrayLike<number>, x: Float64Array, tol = 1e-10, maxit = 2000): { iterations: number; residual: number } {
  const n = A.n;
  const dinv = new Float64Array(n);
  for (let j = 0; j < n; j++) for (let p = A.colptr[j]; p < A.colptr[j + 1]; p++) if (A.rowidx[p] === j) dinv[j] = 1 / A.val[p];
  const r = new Float64Array(n), z = new Float64Array(n), pv = new Float64Array(n), Ap = new Float64Array(n);
  multiply(A, x, Ap);
  let bnorm = 0;
  for (let i = 0; i < n; i++) { r[i] = b[i] - Ap[i]; bnorm += b[i] * b[i]; }
  bnorm = Math.sqrt(bnorm) || 1;
  for (let i = 0; i < n; i++) { z[i] = dinv[i] * r[i]; pv[i] = z[i]; }
  let rz = 0; for (let i = 0; i < n; i++) rz += r[i] * z[i];
  let it = 0, rnorm = Infinity;
  for (; it < maxit; it++) {
    rnorm = 0; for (let i = 0; i < n; i++) rnorm += r[i] * r[i];
    rnorm = Math.sqrt(rnorm);
    if (rnorm <= tol * bnorm) break;
    multiply(A, pv, Ap);
    let pAp = 0; for (let i = 0; i < n; i++) pAp += pv[i] * Ap[i];
    const alpha = rz / pAp;
    for (let i = 0; i < n; i++) { x[i] += alpha * pv[i]; r[i] -= alpha * Ap[i]; z[i] = dinv[i] * r[i]; }
    let rzNew = 0; for (let i = 0; i < n; i++) rzNew += r[i] * z[i];
    const beta = rzNew / rz; rz = rzNew;
    for (let i = 0; i < n; i++) pv[i] = z[i] + beta * pv[i];
  }
  return { iterations: it, residual: rnorm / bnorm };
}
