// node scripts/test-loaders.ts -- PLY (ascii/binary) and STL (ascii/binary) readers
// on a synthetic cube: 8 welded vertices, 6 quads -> 12 triangles, Euler chi = 2.
import { parsePLY, parseSTL } from '../src/mesh/loaders.ts';
import { eulerCharacteristic, type Mesh } from '../src/mesh/types.ts';

const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };

// cube: vertices 0..7, quad faces (outward order irrelevant here)
const V = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const QUADS = [
  [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
  [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7],
];
const closedCube = (m: Mesh, what: string): void => {
  check(m.nv === 8, `${what}: nv ${m.nv} != 8`);
  check(m.nf === 12, `${what}: nf ${m.nf} != 12 (fan-triangulated quads)`);
  check(eulerCharacteristic(m) === 2, `${what}: not a closed surface`);
};

// 1) ascii PLY, with extra vertex properties (normals) around x,y,z and a
// leading comment, so the property indexing is exercised
const asciiPly = [
  'ply', 'format ascii 1.0', 'comment synthetic cube',
  'element vertex 8',
  'property float nx', 'property float x', 'property float y', 'property float z', 'property float ny',
  'element face 6',
  'property list uchar int vertex_indices',
  'end_header',
  ...V.map((v) => `9 ${v[0]} ${v[1]} ${v[2]} 9`),
  ...QUADS.map((q) => `4 ${q.join(' ')}`),
].join('\n');
closedCube(parsePLY(new TextEncoder().encode(asciiPly).buffer as ArrayBuffer), 'ascii PLY');

// 2) binary_little_endian PLY: vertex = double x,y,z + uchar quality; face = uchar-count int list
const head = new TextEncoder().encode([
  'ply', 'format binary_little_endian 1.0',
  'element vertex 8',
  'property double x', 'property double y', 'property double z', 'property uchar quality',
  'element face 6',
  'property list uchar int vertex_indices',
  'end_header', '',
].join('\n'));
const body = new ArrayBuffer(8 * 25 + 6 * (1 + 4 * 4));
const dv = new DataView(body);
let off = 0;
for (const v of V) {
  dv.setFloat64(off, v[0], true); dv.setFloat64(off + 8, v[1], true); dv.setFloat64(off + 16, v[2], true);
  dv.setUint8(off + 24, 7); off += 25;
}
for (const q of QUADS) {
  dv.setUint8(off, 4); off += 1;
  for (const i of q) { dv.setInt32(off, i, true); off += 4; }
}
const binPly = new Uint8Array(head.length + body.byteLength);
binPly.set(head); binPly.set(new Uint8Array(body), head.length);
closedCube(parsePLY(binPly.buffer as ArrayBuffer), 'binary PLY');

// triangles of the same cube, as vertex coordinate triples (for STL)
const TRIS: number[][][] = QUADS.flatMap((q) => [[V[q[0]], V[q[1]], V[q[2]]], [V[q[0]], V[q[2]], V[q[3]]]]);

// 3) ascii STL: repeated per-facet vertices must weld back to 8
const asciiStl = ['solid cube',
  ...TRIS.flatMap((t) => ['facet normal 0 0 0', 'outer loop',
    ...t.map((v) => `vertex ${v[0]} ${v[1]} ${v[2]}`), 'endloop', 'endfacet']),
  'endsolid cube'].join('\n');
closedCube(parseSTL(new TextEncoder().encode(asciiStl).buffer as ArrayBuffer), 'ascii STL');

// 4) binary STL (80-byte header + count + 50 bytes per facet)
const stl = new ArrayBuffer(84 + 50 * TRIS.length);
const sdv = new DataView(stl);
sdv.setUint32(80, TRIS.length, true);
TRIS.forEach((t, k) => {
  const base = 84 + 50 * k + 12;
  t.forEach((v, j) => { sdv.setFloat32(base + 12 * j, v[0], true); sdv.setFloat32(base + 12 * j + 4, v[1], true); sdv.setFloat32(base + 12 * j + 8, v[2], true); });
});
closedCube(parseSTL(stl), 'binary STL');

// 5) a binary STL whose header happens to start with "solid" must still parse as binary
const solidStl = stl.slice(0);
new Uint8Array(solidStl).set(new TextEncoder().encode('solid maybe-not'), 0);
closedCube(parseSTL(solidStl), 'binary STL with solid header');

console.log('PLY ascii/binary and STL ascii/binary readers ok (cube: 8 v, 12 f, chi 2)');
