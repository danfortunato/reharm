// node scripts/test-h5.ts -- HDF5 geometry export round-trip through h5wasm/node
import * as h5wasmNode from 'h5wasm/node';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { writeFileSync, unlinkSync } from 'node:fs';
import { encodeGeometryH5, filterLabel, FORMAT_VERSION, type H5Module } from '../src/fit/exportH5.ts';
import { lmIndex, nlmCalc } from '../src/sht/layout.ts';

const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };
const h5 = h5wasmNode as unknown as H5Module & {
  File: new (path: string, mode: string) => {
    attrs: Record<string, { value: unknown }>;
    get(name: string): { attrs: Record<string, { value: unknown }>; get(name: string): { value: unknown; dtype?: unknown } };
    close(): void;
  };
};

// coefficients with values planted at known (l, m) slots, so the file's byte
// layout is pinned to the shared SHTns m-major convention, not just to
// "whatever encode wrote"
const lmax = 4, mmax = 4, nlm = nlmCalc(lmax, mmax);
const X = new Float32Array(2 * nlm), Y = new Float32Array(2 * nlm), Z = new Float32Array(2 * nlm);
X[2 * lmIndex(lmax, 2, 1)] = 0.5;
X[2 * lmIndex(lmax, 2, 1) + 1] = -0.25;
Y[2 * lmIndex(lmax, 0, 0)] = 1.25;
Z[2 * lmIndex(lmax, 4, 4)] = 3;
Z[2 * lmIndex(lmax, 4, 4) + 1] = 0.125;

const bytes = await encodeGeometryH5(h5, {
  lmax, mmax, nlat: 6, nphi: 16, X, Y, Z,
  model: 'spot', map: 'conformal + Möbius', sampling: 'uniform', filter: filterLabel({ kind: 'gaussian', N: 4, ramp: 1, sigma: 0.05 }),
}, join(tmpdir(), `reharm-test-h5-${process.pid}-encode.h5`));
check(bytes.length > 0 && bytes[0] === 0x89 && bytes[1] === 0x48 && bytes[2] === 0x44 && bytes[3] === 0x46,
  'not an HDF5 file (bad magic)');

const path = join(tmpdir(), `reharm-test-h5-${process.pid}-decode.h5`);
writeFileSync(path, bytes);
await (h5 as unknown as { ready: Promise<unknown> }).ready;
const f = new h5.File(path, 'r');
try {
  const attr = (name: string) => f.attrs[name]?.value;
  check(attr('app') === 'reharm', `app attr: ${String(attr('app'))}`);
  check(Number(attr('format_version')) === FORMAT_VERSION, 'format_version');
  check(attr('model') === 'spot' && attr('map') === 'conformal + Möbius', 'provenance attrs');
  check(attr('filter') === 'gaussian sigma=0.05', `filter attr: ${String(attr('filter'))}`);
  check(attr('sampling') === 'uniform', 'sampling attr');
  const grid = f.get('grid');
  const gattr = (name: string) => Number(grid.attrs[name]?.value);
  check(gattr('lmax') === lmax && gattr('mmax') === mmax && gattr('nlat') === 6 && gattr('nphi') === 16 && gattr('nlm') === nlm,
    'grid attrs');
  const geom = f.get('geometry');
  for (const [name, q] of [['Gx', X], ['Gy', Y], ['Gz', Z]] as const) {
    const v = geom.get(name).value;
    check(v instanceof Float32Array && v.length === 2 * nlm, `${name} is not float32[2*nlm]`);
    for (let k = 0; k < 2 * nlm; k++) check((v as Float32Array)[k] === q[k], `${name}[${k}] round-trip`);
  }
} finally {
  f.close();
  unlinkSync(path);
}
console.log(`h5 geometry export round-trip ok (${bytes.length} bytes, nlm ${nlm})`);
