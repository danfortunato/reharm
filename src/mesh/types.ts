/** A closed triangle mesh: positions interleaved xyz (nv*3), 0-based faces (nf*3). */
export interface Mesh {
  readonly positions: Float32Array;
  readonly faces: Uint32Array;
  readonly nv: number;
  readonly nf: number;
}

export const makeMesh = (positions: Float32Array, faces: Uint32Array): Mesh => ({
  positions,
  faces,
  nv: positions.length / 3,
  nf: faces.length / 3,
});

/** Euler characteristic V - E + F (2 for a closed genus-0 surface). */
export function eulerCharacteristic(m: Mesh): number {
  const edges = new Set<number>();
  const key = (a: number, b: number) => (a < b ? a * m.nv + b : b * m.nv + a);
  const f = m.faces;
  for (let t = 0; t < m.nf; t++) {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    edges.add(key(a, b)); edges.add(key(b, c)); edges.add(key(c, a));
  }
  return m.nv - edges.size + m.nf;
}

/** Bounding box diagonal, the length scale used for relative errors. */
export function bboxDiagonal(m: Mesh): number {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const p = m.positions;
  for (let i = 0; i < m.nv; i++)
    for (let c = 0; c < 3; c++) {
      const v = p[3 * i + c];
      if (v < lo[c]) lo[c] = v;
      if (v > hi[c]) hi[c] = v;
    }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}

/** Centre at the origin and scale to unit bounding-box diagonal (in place). */
export function normalize(m: Mesh): Mesh {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const p = m.positions;
  for (let i = 0; i < m.nv; i++)
    for (let c = 0; c < 3; c++) {
      const v = p[3 * i + c];
      if (v < lo[c]) lo[c] = v;
      if (v > hi[c]) hi[c] = v;
    }
  const ctr = [0, 1, 2].map((c) => (lo[c] + hi[c]) / 2);
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  for (let i = 0; i < m.nv; i++)
    for (let c = 0; c < 3; c++) p[3 * i + c] = (p[3 * i + c] - ctr[c]) / diag;
  return m;
}
