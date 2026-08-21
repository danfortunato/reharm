// node scripts/test-pointcloud.ts <refdir> -- validate the point-cloud parameterization
// port against MATLAB references produced by make_refs.m (see the pcrefs scratch dir):
// kNN, MLS Laplacian, local rings, the full map, and the spherical-Delaunay hull.
// Generate the references with MATLAB R2026a:
//   matlab -batch "make_pc_refs('<refdir>')" ; matlab -batch "make_pc_stages('<refdir>')"
// (scripts/make_pc_refs.m, make_pc_stages.m; needs sphere-surf/pointcloudsphericalconformalmap).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { knn, pcLaplacian, pcSphericalMap } from '../src/chart/pointcloud.ts';
import { lu, luOrdering, luSolve } from '../src/chart/lu.ts';
import { fromTriplets } from '../src/chart/sparse.ts';
import { convexHull } from '../src/mesh/hull.ts';

const refdir = process.argv[2];
if (!refdir) throw new Error('usage: node scripts/test-pointcloud.ts <refdir>');
const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };

function readBin(name: string): { rows: number; cols: number; data: Float64Array } {
  const buf = readFileSync(join(refdir, name));
  const rows = buf.readUInt32LE(0), cols = buf.readUInt32LE(4);
  const data = new Float64Array(buf.buffer.slice(buf.byteOffset + 8, buf.byteOffset + 8 + rows * cols * 8));
  return { rows, cols, data };
}

// ---- LU sanity on a small random nonsymmetric system vs a dense solve
{
  const n = 60;
  let s = 42 >>> 0;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296 - 0.5; };
  const I: number[] = [], J: number[] = [], V: number[] = [];
  const dense = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i !== j && rnd() > -0.35) continue;
      const v = i === j ? 8 + rnd() : rnd();
      I.push(i); J.push(j); V.push(v); dense[n * i + j] += v;
    }
  const A = fromTriplets(n, I, J, V);
  const coords = Float64Array.from({ length: 3 * n }, () => rnd());
  const b = Float64Array.from({ length: n }, () => rnd());
  const { x, residual } = luSolve(lu(A, luOrdering(A, coords)), A, b);
  // dense elimination for reference
  const M = Float64Array.from(dense), y = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[n * r + c]) > Math.abs(M[n * piv + c])) piv = r;
    for (let k = 0; k < n; k++) { const t = M[n * c + k]; M[n * c + k] = M[n * piv + k]; M[n * piv + k] = t; }
    const t = y[c]; y[c] = y[piv]; y[piv] = t;
    for (let r = c + 1; r < n; r++) {
      const f = M[n * r + c] / M[n * c + c];
      for (let k = c; k < n; k++) M[n * r + k] -= f * M[n * c + k];
      y[r] -= f * y[c];
    }
  }
  for (let c = n - 1; c >= 0; c--) { for (let k = c + 1; k < n; k++) y[c] -= M[n * c + k] * y[k]; y[c] /= M[n * c + c]; }
  let err = 0;
  for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(x[i] - y[i]));
  check(err < 1e-10 && residual < 1e-12, `lu vs dense: err ${err.toExponential(2)}, residual ${residual.toExponential(2)}`);
  console.log(`lu ok (dense diff ${err.toExponential(1)}, residual ${residual.toExponential(1)})`);
}

const canonicalTris = (faces: ArrayLike<number>): Set<string> => {
  const set = new Set<string>();
  for (let t = 0; t + 2 < faces.length; t += 3) set.add([faces[t], faces[t + 1], faces[t + 2]].sort((a, b) => a - b).join(','));
  return set;
};

for (const name of ['bumpy', 'david']) {
  const cloud = readBin(`${name}_cloud.bin`);
  const nv = cloud.rows, pos = cloud.data;
  const k = 25;
  console.log(`\n${name}: ${nv} points`);

  // kNN: same neighbor sets, self first (ordering ties may differ)
  const refKnn = readBin(`${name}_knn.bin`).data;
  let t0 = performance.now();
  const mine = knn(pos, nv, k);
  let knnRowsDiffer = 0;
  for (let v = 0; v < nv; v++) {
    check(mine[v * k] === v, `knn: row ${v} does not start with itself`);
    const a = Array.from(mine.slice(v * k, v * k + k)).sort((x, y) => x - y).join(',');
    const b = Array.from(refKnn.slice(v * k, v * k + k)).map(Number).sort((x, y) => x - y).join(',');
    if (a !== b) knnRowsDiffer++;
  }
  console.log(`  knn: ${(performance.now() - t0).toFixed(0)} ms, ${knnRowsDiffer} rows differ (ties)`);
  check(knnRowsDiffer <= nv * 0.001, `knn: ${knnRowsDiffer} rows differ from MATLAB`);

  // MLS Laplacian entries
  t0 = performance.now();
  const pc = pcLaplacian(pos, nv, k);
  const tL = performance.now() - t0;
  const refL = readBin(`${name}_L.bin`);
  const refMap = new Map<number, number>();
  for (let r = 0; r < refL.rows; r++) refMap.set(refL.data[3 * r] * nv + refL.data[3 * r + 1], refL.data[3 * r + 2]);
  let maxRel = 0, checked = 0;
  const { L } = pc;
  for (let j = 0; j < nv; j++)
    for (let p = L.colptr[j]; p < L.colptr[j + 1]; p++) {
      const ref = refMap.get(L.rowidx[p] * nv + j);
      if (ref === undefined) continue;   // ties in knn change the pattern for a few rows
      checked++;
      const rel = Math.abs(L.val[p] - ref) / Math.max(Math.abs(ref), 1e-6);
      if (rel > maxRel) maxRel = rel;
    }
  console.log(`  L: ${tL.toFixed(0)} ms, ${checked}/${refL.rows} entries compared, max rel err ${maxRel.toExponential(2)}`);
  check(maxRel < 1e-6, `L mismatch: max rel err ${maxRel.toExponential(2)}`);
  check(checked >= refL.rows * 0.995, `L pattern mismatch: only ${checked}/${refL.rows} compared`);

  // local rings (canonical triangle sets)
  const refRings = readBin(`${name}_rings.bin`);
  const refSet = canonicalTris(refRings.data);
  const mineSet = canonicalTris(pc.ringTris);
  let common = 0;
  for (const t of mineSet) if (refSet.has(t)) common++;
  const frac = common / Math.max(refSet.size, mineSet.size);
  console.log(`  rings: ${mineSet.size} vs ${refSet.size} unique triangles, ${(100 * frac).toFixed(2)} % common`);
  check(frac > 0.98, `rings differ too much (${(100 * frac).toFixed(1)} % common)`);

  // the full map, from MATLAB's boundary triple: the most regular triangle is
  // the same, but MATLAB's delaunay orders its vertices differently and the
  // order is the map's Mobius normalization — inject theirs to compare maps
  const refBd = Array.from(readBin(`${name}_bd.bin`).data).map((v) => v - 1) as [number, number, number];
  const mineBdSet = (() => {
    let best = Infinity, bd: number[] = [];
    const rt = pc.ringTris;
    for (let t = 0; t + 2 < rt.length; t += 3) {
      const a = rt[t], b = rt[t + 1], c = rt[t + 2];
      const e1 = Math.hypot(pos[3 * b] - pos[3 * c], pos[3 * b + 1] - pos[3 * c + 1], pos[3 * b + 2] - pos[3 * c + 2]);
      const e2 = Math.hypot(pos[3 * a] - pos[3 * c], pos[3 * a + 1] - pos[3 * c + 1], pos[3 * a + 2] - pos[3 * c + 2]);
      const e3 = Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
      const su = e1 + e2 + e3;
      const reg = Math.abs(e1 / su - 1 / 3) + Math.abs(e2 / su - 1 / 3) + Math.abs(e3 / su - 1 / 3);
      if (reg < best) { best = reg; bd = [a, b, c]; }
    }
    return bd.sort((x, y) => x - y).join(',');
  })();
  check(mineBdSet === [...refBd].sort((x, y) => x - y).join(','), `regular triple differs: {${mineBdSet}} vs {${refBd.join(',')}}`);
  t0 = performance.now();
  const { S, info } = pcSphericalMap(pos, nv, pc, undefined, refBd);
  const refS = readBin(`${name}_map.bin`).data;
  let maxD = 0, meanD = 0;
  for (let v = 0; v < nv; v++) {
    const d = Math.hypot(S[3 * v] - refS[3 * v], S[3 * v + 1] - refS[3 * v + 1], S[3 * v + 2] - refS[3 * v + 2]);
    if (d > maxD) maxD = d;
    meanD += d;
  }
  meanD /= nv;
  console.log(`  map: ${(performance.now() - t0).toFixed(0)} ms, ${info.iterations} N-S iterations, worst solve residual ${info.worstResidual.toExponential(1)}, vs MATLAB mean ${meanD.toExponential(2)} max ${maxD.toExponential(2)}`);
  check(info.worstResidual < 1e-8, `solve residual too large: ${info.worstResidual.toExponential(2)}`);
  check(maxD < 1e-4, `map differs from MATLAB: max ${maxD.toExponential(2)}`);

  // hull of the REFERENCE map vs the reference spherical Delaunay
  const refFaces = readBin(`${name}_faces.bin`);
  t0 = performance.now();
  const hull = convexHull(refS, nv);
  const refFaceSet = canonicalTris(refFaces.data);
  const hullSet = canonicalTris(hull);
  let commonF = 0;
  for (const t of hullSet) if (refFaceSet.has(t)) commonF++;
  const fracF = commonF / Math.max(refFaceSet.size, hullSet.size);
  console.log(`  hull: ${(performance.now() - t0).toFixed(0)} ms, ${hullSet.size} faces vs ${refFaceSet.size} reference, ${(100 * fracF).toFixed(2)} % common`);
  check(hull.length / 3 === 2 * nv - 4, `hull is not a triangulated sphere over all points (${hull.length / 3} faces)`);
  check(fracF > 0.99, `hull differs from sphere_delaunay (${(100 * fracF).toFixed(1)} % common)`);
}
console.log('\npoint-cloud port ok');
