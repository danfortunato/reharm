/**
 * Sparse LU for the point-cloud Laplacian solves (chosen over normal
 * equations: the MLS Laplacian is nonsymmetric and squaring it would square
 * its conditioning).
 *
 * Gilbert-Peierls left-looking factorization WITHOUT pivoting: the matrix is
 * ordered once by the same geometric nested dissection the Cholesky path
 * uses (on the symmetrized pattern), and each column's pattern is the graph
 * reach of A(:,j) through the columns of L computed so far. No pivoting is
 * justified empirically, not structurally — the MLS rows put their dominant
 * weight on the diagonal but are not strictly dominant — so solve() applies
 * one step of iterative refinement and reports the relative residual; the
 * caller decides what residual is acceptable.
 */
import { fromTriplets, multiply, nestedDissection, type CSC } from './sparse.ts';

export interface LUFactor {
  n: number;
  perm: Int32Array;   // new -> old
  pinv: Int32Array;   // old -> new
  Lp: Int32Array; Li: Int32Array; Lx: Float64Array;   // unit lower (diagonal implicit)
  Up: Int32Array; Ui: Int32Array; Ux: Float64Array;   // upper, diagonal LAST in each column
}

class Cols {
  p: number[] = [0];
  i: number[] = [];
  x: number[] = [];
  push(i: number, x: number): void { this.i.push(i); this.x.push(x); }
  close(): void { this.p.push(this.i.length); }
}

/** Factor P A P^T = L U with the given ordering (no pivoting). Throws on a
 *  zero pivot; near-zero pivots surface as a large solve() residual. */
export function lu(A: CSC, perm: Int32Array): LUFactor {
  const n = A.n;
  const pinv = new Int32Array(n);
  for (let i = 0; i < n; i++) pinv[perm[i]] = i;
  const L = new Cols(), U = new Cols();
  const x = new Float64Array(n);
  const mark = new Int32Array(n).fill(-1);
  const stack = new Int32Array(n), estate = new Int32Array(n);
  const order = new Int32Array(n);   // topological order of the reach, reversed
  for (let j = 0; j < n; j++) {
    // scatter column perm[j] of A into x (permuted rows)
    const cj = perm[j];
    let top = n;
    let nord = 0;
    for (let p = A.colptr[cj]; p < A.colptr[cj + 1]; p++) {
      const i0 = pinv[A.rowidx[p]];
      x[i0] = A.val[p];
      // DFS from i0 through columns of L (< j) to find the reach
      if (mark[i0] === j) continue;
      let head = 0;
      stack[0] = i0; estate[0] = L.p[Math.min(i0, j)];   // edge cursor of the node (only i < j has a column)
      mark[i0] = j;
      while (head >= 0) {
        const v = stack[head];
        if (v >= j) { order[nord++] = v; head--; continue; }   // no column yet: leaf
        const pend = L.p[v + 1];
        let e = estate[head], advanced = false;
        while (e < pend) {
          const w = L.i[e];
          e++;
          if (mark[w] !== j) {
            estate[head] = e;
            mark[w] = j;
            head++;
            stack[head] = w; estate[head] = w < j ? L.p[w] : 0;
            advanced = true;
            break;
          }
        }
        if (!advanced) {
          if (e >= pend) { order[nord++] = v; head--; }
          else estate[head] = e;
        }
      }
      top = n; // (top unused; kept for clarity)
    }
    // topological order: nodes were emitted children-first; process in reverse
    for (let t = nord - 1; t >= 0; t--) {
      const i = order[t];
      if (i >= j) continue;
      const xi = x[i];
      if (xi === 0) continue;
      for (let p = L.p[i]; p < L.p[i + 1]; p++) x[L.i[p]] -= L.x[p] * xi;
    }
    // gather: U gets i <= j (diagonal last), L gets i > j scaled by the pivot
    let piv = 0;
    const uis: number[] = [], lis: number[] = [];
    for (let t = 0; t < nord; t++) {
      const i = order[t];
      if (i < j) uis.push(i);
      else if (i === j) piv = x[i];
      else lis.push(i);
    }
    if (piv === 0) throw new Error(`lu: zero pivot at column ${j}`);
    uis.sort((a, b) => a - b);
    lis.sort((a, b) => a - b);
    for (const i of uis) { U.push(i, x[i]); x[i] = 0; }
    U.push(j, piv); x[j] = 0;
    U.close();
    for (const i of lis) { L.push(i, x[i] / piv); x[i] = 0; }
    L.close();
  }
  return {
    n, perm, pinv,
    Lp: Int32Array.from(L.p), Li: Int32Array.from(L.i), Lx: Float64Array.from(L.x),
    Up: Int32Array.from(U.p), Ui: Int32Array.from(U.i), Ux: Float64Array.from(U.x),
  };
}

/** x = U \ (L \ P b), permuted back. `b` column-major for several right-hand sides. */
function luApply(F: LUFactor, b: ArrayLike<number>, nrhs: number): Float64Array {
  const { n, perm, Lp, Li, Lx, Up, Ui, Ux } = F;
  const out = new Float64Array(n * nrhs);
  const y = new Float64Array(n);
  for (let r = 0; r < nrhs; r++) {
    for (let i = 0; i < n; i++) y[i] = b[r * n + perm[i]];
    for (let j = 0; j < n; j++) {              // L y = y (unit diagonal)
      const yj = y[j];
      if (yj === 0) continue;
      for (let p = Lp[j]; p < Lp[j + 1]; p++) y[Li[p]] -= Lx[p] * yj;
    }
    for (let j = n - 1; j >= 0; j--) {         // U z = y (diagonal last per column)
      const pd = Up[j + 1] - 1;
      const zj = (y[j] /= Ux[pd]);
      if (zj === 0) continue;
      for (let p = Up[j]; p < pd; p++) y[Ui[p]] -= Ux[p] * zj;
    }
    for (let i = 0; i < n; i++) out[r * n + perm[i]] = y[i];
  }
  return out;
}

/** Solve A x = b with one step of iterative refinement (no-pivot LU needs the
 *  safety net); returns the worst relative residual across right-hand sides. */
export function luSolve(F: LUFactor, A: CSC, b: ArrayLike<number>, nrhs = 1): { x: Float64Array; residual: number } {
  const n = F.n;
  const x = luApply(F, b, nrhs);
  const r = new Float64Array(n * nrhs);
  const Ax = new Float64Array(n);
  let worst = 0;
  for (let k = 0; k < nrhs; k++) {
    multiply(A, x.subarray(k * n, (k + 1) * n), Ax);
    for (let i = 0; i < n; i++) r[k * n + i] = b[k * n + i] - Ax[i];
  }
  const dx = luApply(F, r, nrhs);
  for (let i = 0; i < n * nrhs; i++) x[i] += dx[i];
  for (let k = 0; k < nrhs; k++) {
    multiply(A, x.subarray(k * n, (k + 1) * n), Ax);
    let rn = 0, bn = 0;
    for (let i = 0; i < n; i++) { rn += (b[k * n + i] - Ax[i]) ** 2; bn += b[k * n + i] ** 2; }
    const rel = Math.sqrt(rn) / (Math.sqrt(bn) || 1);
    if (rel > worst) worst = rel;
  }
  return { x, residual: worst };
}

/** Ordering for lu(): nested dissection on the symmetrized pattern (the
 *  splitter only reads adjacency, which must see both A(i,j) and A(j,i)). */
export function luOrdering(A: CSC, coords: ArrayLike<number>): Int32Array {
  const nnz = A.colptr[A.n];
  const I = new Int32Array(2 * nnz), J = new Int32Array(2 * nnz), V = new Float64Array(2 * nnz);
  let k = 0;
  for (let j = 0; j < A.n; j++)
    for (let p = A.colptr[j]; p < A.colptr[j + 1]; p++) {
      I[k] = A.rowidx[p]; J[k] = j; k++;
      I[k] = j; J[k] = A.rowidx[p]; k++;
    }
  V.fill(1);
  return nestedDissection(fromTriplets(A.n, I, J, V), coords);
}
