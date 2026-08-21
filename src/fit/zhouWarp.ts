/**
 * Adaptive sampling warp of Zhou, Bao & Shi, "3D surface filtering using
 * spherical harmonics", CAD 36 (2004), Sec 3.4 — the scheme behind the
 * MATLAB module's fit_spherical_harmonics_zhou: iteratively redistribute the
 * grid mesh's vertices with a density-ratio weighted Laplacian so their
 * density on the sphere approaches the parameterization's own vertex
 * density. Sampling the surface at the warped positions and treating the
 * values as the regular grid signal makes the transformed signal f∘w — much
 * smoother where the chart crowds, so far fewer degrees are needed for the
 * same fidelity. The price: the coefficients encode f∘w, tied to this mesh's
 * vertex distribution (not comparable across meshings — keep uniform
 * sampling for coefficients-as-data).
 *
 * The scheme is grid-agnostic (the paper used Driscoll–Healy; here it feeds
 * the Gauss–Legendre grid): the grid mesh is the tensor-grid triangulation
 * with pole caps, and the warp moves all its vertices; the caps just are not
 * sampled afterwards.
 */
import type { Mesh } from '../mesh/types.ts';
import type { SphereLocator } from './resample.ts';

/**
 * Vertex density of Zhou et al. Sec 3.4: D(v) = (|N(v)|+1) / (incident area
 * / total area). Valence is counted as directed loop edges out of v, which
 * equals the unique-neighbor count on a closed consistently-oriented mesh
 * (each undirected edge appears once per direction); an inconsistently
 * oriented upload only perturbs the density's numerator slightly.
 */
export function vertexDensity(P: ArrayLike<number>, F: ArrayLike<number>, nv: number): Float64Array {
  const nf = (F.length / 3) | 0;
  const Av = new Float64Array(nv), val = new Float64Array(nv);
  let total = 0;
  for (let t = 0; t < nf; t++) {
    const a = 3 * F[3 * t], b = 3 * F[3 * t + 1], c = 3 * F[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const A = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    total += A;
    for (let k = 0; k < 3; k++) {
      Av[F[3 * t + k]] += A;
      val[F[3 * t + k]]++;
    }
  }
  const D = new Float64Array(nv);
  let minPos = Infinity;
  for (let v = 0; v < nv; v++)
    if (Av[v] > 0) {
      D[v] = (val[v] + 1) / (Av[v] / total);
      if (D[v] < minPos) minPos = D[v];
    }
  for (let v = 0; v < nv; v++) if (Av[v] === 0) D[v] = minPos;   // guard: isolated vertex
  return D;
}

/**
 * Warp the grid mesh (`dirs`, `gridFaces`; unit directions) toward the
 * density of the charted mesh (`mesh` placed on the sphere at `chart`,
 * PL-interpolated through `locator`). Returns the warped unit directions;
 * `dirs` is not modified.
 */
export function adaptiveWarp(
  mesh: Mesh, chart: ArrayLike<number>, locator: SphereLocator,
  dirs: Float64Array, gridFaces: ArrayLike<number>, iters = 30,
): Float64Array {
  const m = dirs.length / 3;
  const nf = (gridFaces.length / 3) | 0;
  const D0 = vertexDensity(chart, mesh.faces, mesh.nv);
  let P2 = Float64Array.from(dirs);
  const R = new Float64Array(m);
  const num = new Float64Array(3 * m), den = new Float64Array(m);
  for (let it = 0; it < iters; it++) {
    const D2 = vertexDensity(P2, gridFaces, m);
    const D0at = locator.interpolate(D0, 1, P2).values;
    for (let v = 0; v < m; v++) R[v] = D0at[v] / D2[v];
    // v_new = Normalize( sum R(p) p / sum R(p) ), p over the star's neighbors:
    // directed loop edges (a -> b) visit each neighbor of a exactly once
    num.fill(0); den.fill(0);
    for (let t = 0; t < nf; t++)
      for (let k = 0; k < 3; k++) {
        const a = gridFaces[3 * t + k], b = gridFaces[3 * t + ((k + 1) % 3)];
        const r = R[b];
        num[3 * a] += r * P2[3 * b]; num[3 * a + 1] += r * P2[3 * b + 1]; num[3 * a + 2] += r * P2[3 * b + 2];
        den[a] += r;
      }
    const next = new Float64Array(3 * m);
    for (let v = 0; v < m; v++) {
      const x = num[3 * v] / den[v], y = num[3 * v + 1] / den[v], z = num[3 * v + 2] / den[v];
      const n = Math.hypot(x, y, z) || 1;
      next[3 * v] = x / n; next[3 * v + 1] = y / n; next[3 * v + 2] = z / n;
    }
    P2 = next;
  }
  return P2;
}
