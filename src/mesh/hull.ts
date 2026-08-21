/**
 * 3D convex hull (quickhull with conflict lists). For points lying on a
 * sphere the hull is exactly the spherical Delaunay triangulation (the
 * lifting equivalence) — the meshing step of the point-cloud pipeline, the
 * same construction as MATLAB's convhulln/sphere_delaunay. Returns
 * outward-oriented triangles over the ORIGINAL point indices.
 */

interface Face {
  a: number; b: number; c: number;
  nx: number; ny: number; nz: number; off: number;   // plane: n·x = off
  nbr: [Face | null, Face | null, Face | null];      // across edges (a,b), (b,c), (c,a)
  outside: number[];
  dead: boolean;
}

export function convexHull(pos: ArrayLike<number>, nv: number): Uint32Array {
  if (nv < 4) throw new Error('convexHull: need at least 4 points');
  const px = (i: number) => pos[3 * i], py = (i: number) => pos[3 * i + 1], pz = (i: number) => pos[3 * i + 2];
  let scale = 0;
  for (let i = 0; i < nv; i++) scale = Math.max(scale, Math.abs(px(i)), Math.abs(py(i)), Math.abs(pz(i)));
  // tight: on a unit sphere the sagitta of a d-wide facet is ~d^2/2, so a
  // larger eps swallows every point whose local spacing is below sqrt(2 eps) —
  // crowded conformal maps (spot: lambda ratio ~1e7) really have such regions
  const eps = 1e-14 * (scale || 1);

  const makeFace = (a: number, b: number, c: number): Face => {
    const ux = px(b) - px(a), uy = py(b) - py(a), uz = pz(b) - pz(a);
    const vx = px(c) - px(a), vy = py(c) - py(a), vz = pz(c) - pz(a);
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    return { a, b, c, nx, ny, nz, off: nx * px(a) + ny * py(a) + nz * pz(a), nbr: [null, null, null], outside: [], dead: false };
  };
  const above = (f: Face, i: number): number => f.nx * px(i) + f.ny * py(i) + f.nz * pz(i) - f.off;

  // initial simplex: extremes along x, then farthest from the line, then from the plane
  let i0 = 0, i1 = 0;
  for (let i = 1; i < nv; i++) { if (px(i) < px(i0)) i0 = i; if (px(i) > px(i1)) i1 = i; }
  if (i0 === i1) { i1 = i0 === 0 ? 1 : 0; }
  let i2 = -1, bestA = -1;
  for (let i = 0; i < nv; i++) {
    if (i === i0 || i === i1) continue;
    const f = makeFace(i0, i1, i);
    const a2 = f.nx * f.nx + f.ny * f.ny + f.nz * f.nz;
    if (a2 > bestA) { bestA = a2; i2 = i; }
  }
  const base = makeFace(i0, i1, i2);
  let i3 = -1, bestD = 0;
  for (let i = 0; i < nv; i++) {
    const d = Math.abs(above(base, i));
    if (d > bestD) { bestD = d; i3 = i; }
  }
  if (i3 < 0 || bestD <= eps) throw new Error('convexHull: points are coplanar');
  if (above(base, i3) > 0) { const t = base.b; base.b = base.c; base.c = t; }   // orient the base away from i3

  const faces: Face[] = [];
  const f0 = makeFace(base.a, base.b, base.c);
  const f1 = makeFace(base.a, i3, base.b);
  const f2 = makeFace(base.b, i3, base.c);
  const f3 = makeFace(base.c, i3, base.a);
  // adjacency of the tetrahedron (edge order (a,b),(b,c),(c,a))
  f0.nbr = [f1, f2, f3]; f1.nbr = [f3, f2, f0]; f2.nbr = [f1, f3, f0]; f3.nbr = [f2, f1, f0];
  faces.push(f0, f1, f2, f3);

  // conflict lists
  for (let i = 0; i < nv; i++) {
    if (i === base.a || i === base.b || i === base.c || i === i3) continue;
    for (const f of faces) if (above(f, i) > eps) { f.outside.push(i); break; }
  }

  const queue: Face[] = faces.filter((f) => f.outside.length > 0);
  while (queue.length) {
    const f = queue.pop()!;
    if (f.dead || f.outside.length === 0) continue;
    // farthest conflict point
    let p = f.outside[0], best = -Infinity;
    for (const i of f.outside) { const d = above(f, i); if (d > best) { best = d; p = i; } }
    // visible region (BFS) and its horizon, walked in order
    const visible: Face[] = [f];
    f.dead = true;
    const horizon: { u: number; v: number; hidden: Face }[] = [];
    for (let h = 0; h < visible.length; h++) {
      const g = visible[h];
      const vs = [g.a, g.b, g.c];
      for (let e = 0; e < 3; e++) {
        const n = g.nbr[e]!;
        if (n.dead) continue;
        if (above(n, p) > eps) { n.dead = true; visible.push(n); }
        else horizon.push({ u: vs[e], v: vs[(e + 1) % 3], hidden: n });
      }
    }
    // cone of new faces over the horizon (edge orientation keeps them outward)
    const cone = new Map<number, Face>();   // keyed by the edge start u: face (u, v, p)
    for (const { u, v, hidden } of horizon) {
      const nf = makeFace(u, v, p);
      nf.nbr[0] = hidden;
      // fix the hidden face's neighbor pointer across (v, u)
      const hv = [hidden.a, hidden.b, hidden.c];
      for (let e = 0; e < 3; e++) if (hv[e] === v && hv[(e + 1) % 3] === u) hidden.nbr[e] = nf;
      cone.set(u, nf);
      faces.push(nf);
    }
    for (const nf of cone.values()) {
      nf.nbr[1] = cone.get(nf.b)!;          // edge (v, p) borders the next cone face (v, w, p)
      cone.get(nf.b)!.nbr[2] = nf;          // whose edge (p, v) borders this one
    }
    // redistribute the conflict points of the dead region
    for (const g of visible)
      for (const i of g.outside) {
        if (i === p) continue;
        for (const nf of cone.values()) if (above(nf, i) > eps) { nf.outside.push(i); break; }
      }
    for (const nf of cone.values()) if (nf.outside.length) queue.push(nf);
  }

  const out: number[] = [];
  for (const f of faces) if (!f.dead) out.push(f.a, f.b, f.c);
  return Uint32Array.from(out);
}
