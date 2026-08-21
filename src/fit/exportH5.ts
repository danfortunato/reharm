/**
 * Geometry export: the fitted surface's spherical-harmonic coefficients as
 * one HDF5 file, in turing-surface's geometry layout (the /grid + /geometry
 * subset of its reference/cache files — docs/ellipsoid-reference-spec.md
 * there), so the PDE side, Python (h5py) and MATLAB (h5read) all read it
 * without conversion:
 *
 *   /            attrs: app='reharm', format_version, created_utc,
 *                       model, map, sampling, filter
 *   /grid        attrs: lmax, mmax, nlat, nphi, nlm      (analysis grid)
 *   /geometry    Gx, Gy, Gz    float32[2*nlm]
 *
 * The datasets are the coefficients of x, y, z WITH the current filter gains
 * applied — the surface on screen, the one the output row certifies —
 * in the shared convention (src/sht/layout.ts): orthonormal harmonics with
 * Condon-Shortley phase, m >= 0 only, SHTns m-major ordering, [re, im]
 * interleaved.
 *
 * The h5wasm module is passed in by the caller: the page imports 'h5wasm'
 * lazily (the wasm is ~4 MB and should not load before the first export) and
 * the node test imports 'h5wasm/node'; this module stays ignorant of which.
 */
import { nlmCalc } from '../sht/layout.ts';

/** The slice of h5wasm this writer touches (see turing-surface-cache's h5file.ts). */
export interface H5Module {
  ready: Promise<unknown>;
  File: new (path: string, mode: string) => H5Obj & { close(): void };
}
interface H5Obj {
  get(name: string): unknown;
  create_group(name: string): unknown;
  create_attribute(name: string, data: unknown): void;
  create_dataset(args: { name: string; data: unknown; dtype?: string }): void;
}
interface EmFS {
  readFile(path: string): Uint8Array;
  unlink(path: string): void;
}

export interface GeometryFileData {
  /** analysis band-limit and grid */
  lmax: number;
  mmax: number;
  nlat: number;
  nphi: number;
  /** filtered coefficients of x, y, z, 2*nlm each */
  X: Float32Array;
  Y: Float32Array;
  Z: Float32Array;
  /** provenance: the mesh, the map, the sampling, and the filter that produced them */
  model: string;
  map: string;
  sampling: string;
  filter: string;
}

export const FORMAT_VERSION = 1;

/** Serialize to HDF5 bytes. `scratch` must be a path h5wasm may create and
 *  delete (any absolute path in the browser's in-memory FS; a real temporary
 *  file under node's NODERAWFS build). */
export async function encodeGeometryH5(h5: H5Module, data: GeometryFileData, scratch: string): Promise<Uint8Array> {
  const { FS } = (await h5.ready) as { FS: EmFS };
  const nlm = nlmCalc(data.lmax, data.mmax);
  for (const [name, q] of [['Gx', data.X], ['Gy', data.Y], ['Gz', data.Z]] as const) {
    if (q.length !== 2 * nlm) throw new Error(`${name}: ${q.length} values, expected 2*nlm = ${2 * nlm}`);
  }
  const file = new h5.File(scratch, 'w');
  try {
    file.create_attribute('app', 'reharm');
    file.create_attribute('format_version', FORMAT_VERSION);
    file.create_attribute('created_utc', new Date().toISOString());
    file.create_attribute('model', data.model);
    file.create_attribute('map', data.map);
    file.create_attribute('sampling', data.sampling);
    file.create_attribute('filter', data.filter);

    const grid = file.create_group('grid') as H5Obj;
    grid.create_attribute('lmax', data.lmax);
    grid.create_attribute('mmax', data.mmax);
    grid.create_attribute('nlat', data.nlat);
    grid.create_attribute('nphi', data.nphi);
    grid.create_attribute('nlm', nlm);

    const geom = file.create_group('geometry') as H5Obj;
    geom.create_dataset({ name: 'Gx', data: data.X, dtype: '<f4' });
    geom.create_dataset({ name: 'Gy', data: data.Y, dtype: '<f4' });
    geom.create_dataset({ name: 'Gz', data: data.Z, dtype: '<f4' });
  } finally {
    file.close();
  }
  try {
    return FS.readFile(scratch);
  } finally {
    FS.unlink(scratch);
  }
}

/** The filter as one human-readable provenance attribute. */
export function filterLabel(spec: { kind: string; N: number; ramp: number; sigma: number }): string {
  switch (spec.kind) {
    case 'hard': return `hard N=${spec.N}`;
    case 'trapeziform': return `trapeziform N=${spec.N} ramp=${spec.ramp}`;
    case 'gaussian': return `gaussian sigma=${spec.sigma}`;
    default: return 'off';
  }
}
