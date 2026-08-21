// node scripts/test-intersect.ts -- self-intersection checker (port of the fixed
// check_self_intersections.m) and collapsed-face count
import { faceAreas, collapsedCount, selfIntersections, triTriIntersects } from '../src/mesh/intersect.ts';
import { buildTopology } from '../src/render/sphereMesh.ts';
import { gaussNodesWeights } from '../src/sht/gauss.ts';

const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };
const soup = (v: number[], f: number[]) => ({ pos: Float64Array.from(v), faces: Uint32Array.from(f) });

// 1) narrow-phase cases: a piercing pair counts; separated, vertex-sharing
// (adjacent triangles always touch) and zero-area configurations do not
const A = [0, 0, 0, 2, 0, 0, 0, 2, 0];
let m = soup([...A, 0.3, 0.3, -1, 0.9, 0.3, 1, 0.3, 0.9, 1], [0, 1, 2, 3, 4, 5]);
check(selfIntersections(m.pos, m.faces) === 1, 'piercing pair not counted');
m = soup([...A, 0.3, 0.3, 9, 0.9, 0.3, 11, 0.3, 0.9, 11], [0, 1, 2, 3, 4, 5]);
check(selfIntersections(m.pos, m.faces) === 0, 'separated pair counted');
m = soup([...A, 0.9, 0.3, -1, 0.3, 0.9, 1], [0, 1, 2, 0, 3, 4]);   // crosses the plane but shares vertex 0
check(selfIntersections(m.pos, m.faces) === 0, 'vertex-sharing pair counted');
m = soup([...A, 0.5, 0.5, -1, 0.5, 0.5, 0, 0.5, 0.5, 1], [0, 1, 2, 3, 4, 5]);   // colinear: zero area
check(selfIntersections(m.pos, m.faces) === 0, 'zero-area sliver counted');
console.log('narrow-phase cases ok');

// 2) the unit-sphere tensor-grid mesh (poles merged) is embedded -- this exact
// mesh reported thousands of "intersections" before the unit-normal fix made
// the polar slivers grazing contacts instead of garbage
const { x } = gaussNodesWeights(64);
const phi = new Float64Array(128);
for (let j = 0; j < 128; j++) phi[j] = (2 * Math.PI * j) / 128;
const topo = buildTopology(x, phi);
let t = performance.now();
const nSphere = selfIntersections(topo.sphereRef, topo.indices);
console.log(`sphere grid mesh 64x128 (${topo.indices.length / 3} faces): ${nSphere} self-intersections, ${(performance.now() - t).toFixed(0)} ms`);
check(nSphere === 0, `unit-sphere grid mesh must be embedded (got ${nSphere})`);

// 3) two overlapping unit spheres as one mesh: many intersecting pairs
const nv = topo.numVertices;
const two = new Float64Array(2 * 3 * nv);
two.set(topo.sphereRef);
for (let i = 0; i < nv; i++) {
  two[3 * (nv + i)] = topo.sphereRef[3 * i] + 0.8;
  two[3 * (nv + i) + 1] = topo.sphereRef[3 * i + 1];
  two[3 * (nv + i) + 2] = topo.sphereRef[3 * i + 2];
}
const twoFaces = new Uint32Array(2 * topo.indices.length);
twoFaces.set(topo.indices);
for (let k = 0; k < topo.indices.length; k++) twoFaces[topo.indices.length + k] = topo.indices[k] + nv;
const nTwo = selfIntersections(two, twoFaces);
console.log(`two overlapping spheres: ${nTwo} intersecting pairs`);
check(nTwo > 100, 'overlapping spheres must report many intersections');

// 4) broad-phase completeness: on a random triangle soup the binned count
// must equal brute force over all pairs with the same narrow phase
let s = 987654321 >>> 0;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
const NT = 300;
const sp = new Float64Array(9 * NT);
for (let i = 0; i < NT; i++) {
  const cx = rnd(), cy = rnd(), cz = rnd();
  for (let v = 0; v < 3; v++) {
    sp[9 * i + 3 * v] = cx + 0.2 * (rnd() - 0.5);
    sp[9 * i + 3 * v + 1] = cy + 0.2 * (rnd() - 0.5);
    sp[9 * i + 3 * v + 2] = cz + 0.2 * (rnd() - 0.5);
  }
}
const sf = Uint32Array.from({ length: 3 * NT }, (_, k) => k);
const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let k = 0; k < 9 * NT; k++) {
  const ax = k % 3;
  if (sp[k] < lo[ax]) lo[ax] = sp[k];
  if (sp[k] > hi[ax]) hi[ax] = sp[k];
}
const tol = 1e-10 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
let brute = 0;
for (let i = 0; i < NT; i++)
  for (let j = i + 1; j < NT; j++) if (triTriIntersects(sp, sf, i, j, tol)) brute++;
const binned = selfIntersections(sp, sf);
console.log(`random soup ${NT} triangles: brute ${brute}, binned ${binned}`);
check(brute === binned, `broad phase dropped pairs: brute ${brute} != binned ${binned}`);
check(brute > 0, 'soup produced no intersections: test is vacuous');

// 5) collapsed faces: ratio below 10% of the reference area, zero-area
// reference skipped; a uniformly shrunk mesh collapses every face
check(collapsedCount(Float64Array.from([1, 0.05, 1, 0.2]), Float64Array.from([1, 1, 0, 1])) === 1, 'collapsedCount semantics');
const shrunk = Float64Array.from(topo.sphereRef, (v) => 0.1 * v);
const nCollapsed = collapsedCount(faceAreas(shrunk, topo.indices), faceAreas(topo.sphereRef, topo.indices));
check(nCollapsed === topo.indices.length / 3, `shrunk mesh: all faces must collapse (got ${nCollapsed})`);
console.log('collapsed-face count ok');
