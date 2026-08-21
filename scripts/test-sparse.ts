// node scripts/test-sparse.ts  (Node >= 22.6 strips types natively)
import { fromTriplets, cholesky, nestedDissection, solve, multiply } from '../src/chart/sparse.ts';

function rng(seed: number) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }

// 1) random SPD grid-graph Laplacian + identity, compare A x = b residual and against a dense solve
const W = 12, H = 10, n = W * H, r = rng(7);
const I: number[] = [], J: number[] = [], V: number[] = [], coords = new Float64Array(3 * n);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const i = y * W + x; coords[3 * i] = x; coords[3 * i + 1] = y; coords[3 * i + 2] = 0.1 * r();
  const nb = [[x + 1, y], [x, y + 1]];
  for (const [nx, ny] of nb) if (nx < W && ny < H) {
    const j = ny * W + nx, w = 0.5 + r();
    I.push(i, j, i, j); J.push(j, i, i, j); V.push(-w, -w, w, w);
  }
  I.push(i); J.push(i); V.push(1 + r());
}
const A = fromTriplets(n, I, J, V);
const b = Float64Array.from({ length: n }, () => r() - 0.5);
const perm = nestedDissection(A, coords);
const seen = new Set(perm); if (seen.size !== n) throw new Error('perm is not a permutation');
const F = cholesky(A, perm);
const x = solve(F, b);
const res = multiply(A, x); let err = 0; for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(res[i] - b[i]));
console.log(`grid ${W}x${H}: nnz(A)=${A.colptr[n]} nnz(L)=${F.Lp[n]} residual ${err.toExponential(2)}`);
if (err > 1e-10) throw new Error('residual too large');

// 2) dense reference on a small random SPD matrix
const m = 30, M: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) { const v = r() - 0.5; M[i][j] += v; M[j][i] += v; }
for (let i = 0; i < m; i++) M[i][i] += m;
const It: number[] = [], Jt: number[] = [], Vt: number[] = [];
for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) { It.push(i); Jt.push(j); Vt.push(M[i][j]); }
const Ad = fromTriplets(m, It, Jt, Vt), bd = Float64Array.from({ length: m }, () => r());
const xs = solve(cholesky(Ad, nestedDissection(Ad, new Float64Array(3 * m).map(() => r()))), bd);
// Gaussian elimination reference
const G = M.map((row, i) => [...row, bd[i]]);
for (let c = 0; c < m; c++) { for (let rr = c + 1; rr < m; rr++) { const f = G[rr][c] / G[c][c]; for (let k = c; k <= m; k++) G[rr][k] -= f * G[c][k]; } }
const xr = new Float64Array(m);
for (let i = m - 1; i >= 0; i--) { let t = G[i][m]; for (let k = i + 1; k < m; k++) t -= G[i][k] * xr[k]; xr[i] = t / G[i][i]; }
let d = 0; for (let i = 0; i < m; i++) d = Math.max(d, Math.abs(xs[i] - xr[i]));
console.log(`dense ref (m=${m}): max |x - x_ref| = ${d.toExponential(2)}`);
if (d > 1e-9) throw new Error('mismatch vs dense solve');

// 3) not positive definite must throw
let threw = false;
try { cholesky(fromTriplets(2, [0, 1, 0, 1], [0, 1, 1, 0], [1, 1, 2, 2]), Int32Array.from([0, 1])); } catch { threw = true; }
console.log(`indefinite throws: ${threw}`);
if (!threw) throw new Error('expected throw');
console.log('sparse: ok');
