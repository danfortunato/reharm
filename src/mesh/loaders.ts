/**
 * Mesh readers: the preset binary ('SHM1'), Wavefront OBJ, OFF, PLY (ascii
 * and both binary flavors), and STL (ascii and binary).
 * Faces with more than three vertices are fan-triangulated. STL carries no
 * connectivity, so its repeated per-facet vertices are welded by exact
 * coordinate match (the downstream Euler-characteristic check needs a real
 * closed mesh, not a triangle soup).
 */
import { makeMesh, type Mesh } from './types.ts';

/** Preset binary: 'SHM1', uint32 nv, uint32 nf, float32[nv*3], uint32[nf*3]; little-endian. */
export function parseSHM(buf: ArrayBuffer): Mesh {
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== 'SHM1') throw new Error(`not an SHM1 mesh (magic ${JSON.stringify(magic)})`);
  const hdr = new DataView(buf, 4, 8);
  const nv = hdr.getUint32(0, true), nf = hdr.getUint32(4, true);
  const positions = new Float32Array(buf, 12, nv * 3);
  const faces = new Uint32Array(buf, 12 + nv * 12, nf * 3);
  return makeMesh(positions.slice(), faces.slice());
}

export function parseOBJ(text: string): Mesh {
  const pos: number[] = [], fac: number[] = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'v') pos.push(+t[1], +t[2], +t[3]);
    else if (t[0] === 'f') {
      const idx = t.slice(1).map((s) => parseInt(s.split('/')[0], 10) - 1);
      for (let k = 1; k + 1 < idx.length; k++) fac.push(idx[0], idx[k], idx[k + 1]);
    }
  }
  return makeMesh(Float32Array.from(pos), Uint32Array.from(fac));
}

export function parseOFF(text: string): Mesh {
  const lines = text.split(/\r?\n/).map((l) => l.replace(/#.*/, '').trim()).filter((l) => l);
  if (!/^OFF$/i.test(lines[0])) throw new Error('not an OFF file');
  const [nv, nf] = lines[1].split(/\s+/).map(Number);
  const pos = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    const t = lines[2 + i].split(/\s+/).map(Number);
    pos[3 * i] = t[0]; pos[3 * i + 1] = t[1]; pos[3 * i + 2] = t[2];
  }
  const fac: number[] = [];
  for (let i = 0; i < nf; i++) {
    const t = lines[2 + nv + i].split(/\s+/).map(Number);
    const idx = t.slice(1, 1 + t[0]);
    for (let k = 1; k + 1 < idx.length; k++) fac.push(idx[0], idx[k], idx[k + 1]);
  }
  return makeMesh(pos, Uint32Array.from(fac));
}

// ---------------------------------------------------------------- PLY
const PLY_SIZES: Record<string, number> = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};
const plyRead = (dv: DataView, off: number, type: string, le: boolean): number => {
  switch (PLY_SIZES[type]) {
    case 1: return type[0] === 'u' ? dv.getUint8(off) : dv.getInt8(off);
    case 2: return type[0] === 'u' ? dv.getUint16(off, le) : dv.getInt16(off, le);
    case 8: return dv.getFloat64(off, le);
    default: return type[0] === 'f' ? dv.getFloat32(off, le)
      : type[0] === 'u' ? dv.getUint32(off, le) : dv.getInt32(off, le);
  }
};

interface PlyProp { name: string; type: string; listCountType?: string }
interface PlyElement { name: string; count: number; props: PlyProp[] }

export function parsePLY(buf: ArrayBuffer): Mesh {
  // the header is ASCII up to an 'end_header' line; the body follows its newline
  const probe = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 65536)));
  const endMatch = /end_header\r?\n/.exec(probe);
  if (!probe.startsWith('ply') || !endMatch) throw new Error('not a PLY file');
  const bodyStart = endMatch.index + endMatch[0].length;
  let format = '';
  const elements: PlyElement[] = [];
  for (const line of probe.slice(0, endMatch.index).split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'format') format = t[1];
    else if (t[0] === 'element') elements.push({ name: t[1], count: Number(t[2]), props: [] });
    else if (t[0] === 'property') {
      const el = elements[elements.length - 1];
      if (!el) throw new Error('PLY property before any element');
      if (t[1] === 'list') el.props.push({ name: t[4], type: t[3], listCountType: t[2] });
      else el.props.push({ name: t[2], type: t[1] });
    }
  }
  const pos: number[] = [], fac: number[] = [];
  const vertexEl = elements.find((e) => e.name === 'vertex');
  if (vertexEl?.props.some((p) => p.listCountType !== undefined))
    throw new Error('PLY vertex elements with list properties are not supported');
  // per-vertex value arrays align with the scalar property order in both bodies
  const coordAt = (name: string): number => {
    const i = vertexEl?.props.findIndex((p) => p.name === name) ?? -1;
    if (i < 0) throw new Error(`PLY vertex has no '${name}' property`);
    return i;
  };
  const [ix, iy, iz] = vertexEl ? [coordAt('x'), coordAt('y'), coordAt('z')] : [0, 1, 2];
  const takeVertex = (values: number[]): void => { pos.push(values[ix], values[iy], values[iz]); };
  const isIndexList = (p: PlyProp): boolean =>
    p.listCountType !== undefined && (p.name === 'vertex_indices' || p.name === 'vertex_index');

  if (format === 'ascii') {
    const lines = new TextDecoder().decode(new Uint8Array(buf, bodyStart)).split(/\r?\n/).filter((l) => l.trim());
    let ln = 0;
    for (const el of elements)
      for (let i = 0; i < el.count; i++) {
        const tok = lines[ln++].trim().split(/\s+/).map(Number);
        if (el.name === 'vertex') takeVertex(tok);
        else if (el.name === 'face') {
          let o = 0;
          for (const p of el.props) {
            const isList = p.listCountType !== undefined;
            const n = isList ? tok[o++] : 1;
            if (isIndexList(p)) for (let k = 1; k + 1 < n; k++) fac.push(tok[o], tok[o + k], tok[o + k + 1]);
            o += n;
          }
        }
      }
  } else if (format === 'binary_little_endian' || format === 'binary_big_endian') {
    const le = format === 'binary_little_endian';
    const dv = new DataView(buf);
    let off = bodyStart;
    const values: number[] = [];
    for (const el of elements)
      for (let i = 0; i < el.count; i++) {
        values.length = 0;
        for (const p of el.props) {
          if (p.listCountType !== undefined) {
            const n = plyRead(dv, off, p.listCountType, le);
            off += PLY_SIZES[p.listCountType];
            if (isIndexList(p) && el.name === 'face') {
              const base = off, sz = PLY_SIZES[p.type];
              for (let k = 1; k + 1 < n; k++)
                fac.push(plyRead(dv, base, p.type, le), plyRead(dv, base + k * sz, p.type, le), plyRead(dv, base + (k + 1) * sz, p.type, le));
            }
            off += n * PLY_SIZES[p.type];
          } else {
            if (el.name === 'vertex') values.push(plyRead(dv, off, p.type, le));
            off += PLY_SIZES[p.type];
          }
        }
        if (el.name === 'vertex') takeVertex(values);
      }
  } else throw new Error(`unsupported PLY format '${format}'`);
  return makeMesh(Float32Array.from(pos), Uint32Array.from(fac));
}

// ---------------------------------------------------------------- STL
/** Weld exactly-repeated coordinates into shared vertices as triangles stream in. */
class VertexWelder {
  readonly pos: number[] = [];
  readonly fac: number[] = [];
  private readonly seen = new Map<string, number>();
  add(x: number, y: number, z: number): void {
    const key = `${x},${y},${z}`;
    let i = this.seen.get(key);
    if (i === undefined) { i = this.pos.length / 3; this.seen.set(key, i); this.pos.push(x, y, z); }
    this.fac.push(i);
  }
  mesh(): Mesh { return makeMesh(Float32Array.from(this.pos), Uint32Array.from(this.fac)); }
}

export function parseSTL(buf: ArrayBuffer): Mesh {
  const w = new VertexWelder();
  // binary iff the 80-byte header + count + 50 bytes per facet adds up exactly
  // ("solid" at byte 0 is no proof of ascii: binary exporters write it too)
  const dv = new DataView(buf);
  if (buf.byteLength >= 84 && 84 + 50 * dv.getUint32(80, true) === buf.byteLength) {
    const n = dv.getUint32(80, true);
    for (let t = 0; t < n; t++) {
      const base = 84 + 50 * t + 12;   // skip the facet normal
      for (let v = 0; v < 3; v++)
        w.add(dv.getFloat32(base + 12 * v, true), dv.getFloat32(base + 12 * v + 4, true), dv.getFloat32(base + 12 * v + 8, true));
    }
  } else {
    const text = new TextDecoder().decode(buf);
    if (!/^\s*solid/.test(text)) throw new Error('not an STL file');
    const re = /vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;
    for (let m = re.exec(text); m; m = re.exec(text)) w.add(Number(m[1]), Number(m[2]), Number(m[3]));
    if (w.fac.length % 3 !== 0) throw new Error(`STL facet with ${w.fac.length % 3} vertices left over`);
  }
  return w.mesh();
}

/** Dispatch on the file name. */
export async function readMeshFile(file: File): Promise<Mesh> {
  const name = file.name.toLowerCase();
  if (name.endsWith('.mesh')) return parseSHM(await file.arrayBuffer());
  if (name.endsWith('.obj')) return parseOBJ(await file.text());
  if (name.endsWith('.off')) return parseOFF(await file.text());
  if (name.endsWith('.ply')) return parsePLY(await file.arrayBuffer());
  if (name.endsWith('.stl')) return parseSTL(await file.arrayBuffer());
  throw new Error(`unsupported mesh format: ${file.name} (use .obj, .off, .ply, .stl or .mesh)`);
}
