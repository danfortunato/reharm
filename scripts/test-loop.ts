// node scripts/test-loop.ts -- checks of the exact Loop limit evaluator
import { LoopLimit, loopSubdivideField } from '../src/mesh/loopLimit.ts';
import { syntheticMesh, syntheticShapes } from '../src/mesh/presets.ts';
import { makeMesh } from '../src/mesh/types.ts';

// control mesh: coarse lat-long bumpy sphere (poles are extraordinary: valence nphi; the rest valence 6)
const { mesh: ctrl } = syntheticMesh(syntheticShapes.bumpy.fn, 10, 16);
const field = Float64Array.from(ctrl.positions);
const L = new LoopLimit(ctrl, field, 3);
const f = L.mesh.faces;

// 1) at a corner (u = 1) the evaluation must equal that vertex's limit position, to DEPTH precision
let worst = 0;
for (let t = 0; t < L.mesh.nf; t += 7) {
  for (let k = 0; k < 3; k++) {
    const bc = [0, 0, 0]; bc[k] = 1;
    const e = L.evaluate(t, bc[0], bc[1], bc[2]);
    const l = L.limitOfVertex(f[3 * t + k]);
    worst = Math.max(worst, Math.hypot(e[0] - l[0], e[1] - l[1], e[2] - l[2]));
  }
}
console.log(`corner evaluation vs limit mask: max |diff| = ${worst.toExponential(2)} (expect < 1e-8)`);
if (worst > 1e-8) throw new Error('corner test failed');

// 2) continuity across a shared edge: same point from both faces
const key = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
const edgeFaces = new Map<string, number[]>();
for (let t = 0; t < L.mesh.nf; t++) for (let k = 0; k < 3; k++) { const kk = key(f[3 * t + k], f[3 * t + ((k + 1) % 3)]); (edgeFaces.get(kk) ?? edgeFaces.set(kk, []).get(kk)!).push(t); }
let worstEdge = 0, tested = 0;
for (const [kk, ts] of edgeFaces) {
  if (ts.length !== 2 || tested > 60) continue;
  const [a, b] = kk.split(',').map(Number);
  const bary = (t: number, s: number) => { const bc = [0, 0, 0]; for (let k = 0; k < 3; k++) { if (f[3 * t + k] === a) bc[k] = s; if (f[3 * t + k] === b) bc[k] = 1 - s; } return bc; };
  for (const s of [0.3, 0.71]) {
    const p = L.evaluate(ts[0], ...(bary(ts[0], s) as [number, number, number]));
    const q = L.evaluate(ts[1], ...(bary(ts[1], s) as [number, number, number]));
    worstEdge = Math.max(worstEdge, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
  }
  tested++;
}
console.log(`continuity across edges (${tested} edges): max |diff| = ${worstEdge.toExponential(2)} (expect < 1e-8)`);
if (worstEdge > 1e-8) throw new Error('continuity test failed');

// 3) against deep global subdivision: control point after 6 levels ~ limit to O(4^-6); compare at a random interior point
//    (the level-6 mesh's vertex nearest to the evaluated point should be within the level-6 edge length)
let m = L.mesh, fld = L.field;
for (let k = 0; k < 5; k++) ({ mesh: m, field: fld } = loopSubdivideField(m, fld, 3));
let worst3 = 0;
for (let t = 0; t < L.mesh.nf; t += 11) {
  const e = L.evaluate(t, 0.2, 0.3, 0.5);
  let best = Infinity;
  for (let i = 0; i < m.nv; i++) best = Math.min(best, Math.hypot(fld[3 * i] - e[0], fld[3 * i + 1] - e[1], fld[3 * i + 2] - e[2]));
  worst3 = Math.max(worst3, best);
}
console.log(`distance from evaluated points to the level-6 control net: max ${worst3.toExponential(2)} (edge length there ~ ${(0.3 / 32).toExponential(1)})`);
if (worst3 > 0.3 / 32) throw new Error('deep-subdivision test failed');

// 4) icosahedron control mesh -> limit surface should be round-ish and timing
const t0 = performance.now();
let n = 0;
for (let t = 0; t < L.mesh.nf; t++) for (let s = 0; s < 20; s++) { L.evaluate(t, 0.1 + 0.02 * s, 0.3, 0.6 - 0.02 * s); n++; }
console.log(`${n} evaluations in ${(performance.now() - t0).toFixed(0)} ms (${((performance.now() - t0) / n * 1000).toFixed(1)} µs each)`);
console.log('loop limit: ok');
