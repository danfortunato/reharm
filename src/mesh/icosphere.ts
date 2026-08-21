/**
 * Icosphere: the regular icosahedron with `subdiv` rounds of midpoint
 * subdivision, every vertex normalized to the unit sphere. Near-uniform
 * vertex distribution and no distinguished poles — the honest sphere mesh
 * for testing (the lat-long tessellation clusters vertices at its poles,
 * which confounds pole-noise tests and makes a lousy point cloud).
 */
import { makeMesh, type Mesh } from './types.ts';

export function icosphere(subdiv: number): Mesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let pos: number[] = [
    -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
    0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
    t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1,
  ];
  let faces: number[] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
    1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
    3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
    4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ];
  for (let i = 0; i < pos.length; i += 3) {
    const n = Math.hypot(pos[i], pos[i + 1], pos[i + 2]);
    pos[i] /= n; pos[i + 1] /= n; pos[i + 2] /= n;
  }
  for (let s = 0; s < subdiv; s++) {
    const mid = new Map<number, number>();
    const nv0 = pos.length / 3;
    const midpoint = (a: number, b: number): number => {
      const key = a < b ? a * nv0 + b : b * nv0 + a;
      let m = mid.get(key);
      if (m === undefined) {
        const x = pos[3 * a] + pos[3 * b], y = pos[3 * a + 1] + pos[3 * b + 1], z = pos[3 * a + 2] + pos[3 * b + 2];
        const n = Math.hypot(x, y, z);
        m = pos.length / 3;
        pos.push(x / n, y / n, z / n);
        mid.set(key, m);
      }
      return m;
    };
    const next: number[] = [];
    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f], b = faces[f + 1], c = faces[f + 2];
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    faces = next;
  }
  return makeMesh(Float32Array.from(pos), Uint32Array.from(faces));
}
