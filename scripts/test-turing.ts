// node scripts/test-turing.ts -- turing-surface export normalization
//
// The payload sent by "Export to turing-surface" is centered (q00 zeroed) and
// scaled to rms radius 1 by Parseval (src/fit/exportTuring.ts). Pinned here on
// the one surface whose answer is exact: the unit sphere, whose coordinates
// are pure degree-1 harmonics with known coefficients under the shared
// convention (orthonormal, Condon-Shortley, m >= 0 storage) — so its rms
// radius must come out exactly 1, and offsets/scalings must be recovered.
import { normalizeGeometry } from '../src/fit/exportTuring.ts';
import { lmIndex, nlmCalc } from '../src/sht/layout.ts';

const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

const lmax = 4, mmax = 4, nlm = nlmCalc(lmax, mmax);
const Y00 = 1 / (2 * Math.sqrt(Math.PI));

// x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta):
// q_11(x) = -sqrt(2*pi/3), q_11(y) = i*sqrt(2*pi/3), q_10(z) = sqrt(4*pi/3).
const sphere = (): [Float32Array, Float32Array, Float32Array] => {
  const X = new Float32Array(2 * nlm), Y = new Float32Array(2 * nlm), Z = new Float32Array(2 * nlm);
  X[2 * lmIndex(lmax, 1, 1)] = -Math.sqrt((2 * Math.PI) / 3);
  Y[2 * lmIndex(lmax, 1, 1) + 1] = Math.sqrt((2 * Math.PI) / 3);
  Z[2 * lmIndex(lmax, 1, 0)] = Math.sqrt((4 * Math.PI) / 3);
  return [X, Y, Z];
};

// ---- the unit sphere is already normalized --------------------------------
{
  const [X, Y, Z] = sphere();
  const n = normalizeGeometry(X, Y, Z, lmax, mmax);
  check(close(n.scale, 1, 1e-6), `unit sphere rms radius: got ${n.scale}, want 1`);
  check(n.center.every((c) => close(c, 0, 1e-7)), `unit sphere center: got ${n.center.join(', ')}`);
  for (let i = 0; i < 2 * nlm; i++) {
    check(close(n.X[i], X[i], 1e-6) && close(n.Y[i], Y[i], 1e-6) && close(n.Z[i], Z[i], 1e-6),
      `unit sphere coefficients changed at ${i}`);
  }
}

// ---- offset and scale are recovered, and the result is the unit sphere ----
{
  const [X, Y, Z] = sphere();
  const scale = 3, center = [1, -2, 0.5];
  for (const [q, c] of [[X, center[0]], [Y, center[1]], [Z, center[2]]] as const) {
    for (let i = 0; i < q.length; i++) q[i] *= scale;
    q[0] = c / Y00; // the (0,0) coefficient holding the mean position
  }
  const snapshot = X.slice();
  const n = normalizeGeometry(X, Y, Z, lmax, mmax);
  check(close(n.scale, scale, 1e-5), `scaled sphere: got scale ${n.scale}, want ${scale}`);
  check(
    n.center.every((c, k) => close(c, center[k], 1e-6)),
    `offset sphere: got center ${n.center.join(', ')}, want ${center.join(', ')}`,
  );
  const [uX, uY, uZ] = sphere();
  for (let i = 0; i < 2 * nlm; i++) {
    check(close(n.X[i], uX[i], 1e-6) && close(n.Y[i], uY[i], 1e-6) && close(n.Z[i], uZ[i], 1e-6),
      `normalized result is not the unit sphere at ${i}`);
  }
  check(X.every((v, i) => v === snapshot[i]), 'normalizeGeometry mutated its input');
}

// ---- a degenerate (all-zero) surface is refused ----------------------------
{
  let threw = false;
  try {
    normalizeGeometry(new Float32Array(2 * nlm), new Float32Array(2 * nlm), new Float32Array(2 * nlm), lmax, mmax);
  } catch {
    threw = true;
  }
  check(threw, 'zero surface must throw, not divide by zero');
}

console.log('test-turing: all checks passed');
