/**
 * Preset geometries: scanned/modelled meshes shipped in public/presets (see
 * presets.json for provenance), and synthetic shapes built on a latitude-
 * longitude grid. A synthetic shape comes with its chart for free -- each
 * vertex IS a sphere point -- so it exercises the whole transform pipeline
 * without any parameterization step.
 */
import { makeMesh, type Mesh } from './types.ts';
import { parseSHM } from './loaders.ts';

export interface PresetEntry {
  key: string;
  label: string;
  file: string;
  nv: number;
  nf: number;
  euler: number;
  source: string;
}

export async function loadManifest(base = './presets/'): Promise<PresetEntry[]> {
  const res = await fetch(`${base}presets.json`);
  if (!res.ok) throw new Error(`presets.json: ${res.status}`);
  return (await res.json()) as PresetEntry[];
}

export async function loadPreset(entry: PresetEntry, base = './presets/'): Promise<Mesh> {
  const res = await fetch(`${base}${entry.file}`);
  if (!res.ok) throw new Error(`${entry.file}: ${res.status}`);
  return parseSHM(await res.arrayBuffer());
}

/** A mesh together with a spherical chart: unit-sphere point per vertex. */
export interface ChartedMesh {
  mesh: Mesh;
  chart: Float32Array; // nv*3, on the unit sphere
}

/** Radial profile r(theta, phi) of the synthetic shapes, or a full map. */
export type ShapeFn = (theta: number, phi: number) => [number, number, number];

export const syntheticShapes: Record<string, { label: string; fn: ShapeFn }> = {
  bumpy: {
    // a polynomial on the sphere (sin^4 cos4φ ∝ Re Y_4^4, cos3θ ∝ P_3(cosθ)), so x, y, z
    // are exactly band-limited at degree 5: the analysis round-trips to fp32 precision
    label: 'bumpy sphere',
    fn: (th, ph) => radial(th, ph, 1 + 0.2 * Math.sin(th) ** 4 * Math.cos(4 * ph) + 0.1 * Math.cos(3 * th)),
  },
  ellipsoid: {
    label: 'ellipsoid 1:1:3',
    fn: (th, ph) => [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), 3 * Math.cos(th)],
  },
  peanut: {
    label: 'peanut',
    fn: (th, ph) => {
      const st = Math.sin(th), r = 1 - 0.6 * st * st;
      return [r * st * Math.cos(ph), r * st * Math.sin(ph), 1.4 * r * Math.cos(th)];
    },
  },
  sphere: { label: 'sphere', fn: (th, ph) => radial(th, ph, 1) },
};

function radial(th: number, ph: number, r: number): [number, number, number] {
  return [r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph), r * Math.cos(th)];
}

/**
 * Triangulate a shape on an (nlat x nphi) latitude-longitude grid with two
 * pole vertices. The chart is exact: vertex (theta_i, phi_j) sits at that
 * point of the unit sphere.
 */
export function syntheticMesh(fn: ShapeFn, nlat = 96, nphi = 192): ChartedMesh {
  const nv = nlat * nphi + 2;
  const positions = new Float32Array(nv * 3);
  const chart = new Float32Array(nv * 3);
  const put = (k: number, th: number, ph: number) => {
    const [x, y, z] = fn(th, ph);
    positions.set([x, y, z], 3 * k);
    chart.set([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)], 3 * k);
  };
  for (let i = 0; i < nlat; i++) {
    const th = (Math.PI * (i + 1)) / (nlat + 1);
    for (let j = 0; j < nphi; j++) put(i * nphi + j, th, (2 * Math.PI * j) / nphi);
  }
  const north = nlat * nphi, south = north + 1;
  put(north, 0, 0); put(south, Math.PI, 0);

  const faces: number[] = [];
  for (let i = 0; i + 1 < nlat; i++)
    for (let j = 0; j < nphi; j++) {
      const j2 = (j + 1) % nphi;
      const a = i * nphi + j, b = i * nphi + j2, c = (i + 1) * nphi + j, d = (i + 1) * nphi + j2;
      faces.push(a, c, b, b, c, d);
    }
  for (let j = 0; j < nphi; j++) {
    const j2 = (j + 1) % nphi;
    faces.push(north, j, j2);
    faces.push((nlat - 1) * nphi + j, south, (nlat - 1) * nphi + j2);
  }
  return { mesh: makeMesh(positions, Uint32Array.from(faces)), chart };
}
