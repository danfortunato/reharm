# reharm

Spherical-harmonic fitting of genus-0 closed surfaces, live in the browser:
map a mesh — or a raw point cloud — onto the sphere, resample its exact Loop
limit surface onto a Gauss–Legendre grid, one GPU analysis, then filter by
degree in real time. Sibling of
[turing-surface](https://github.com/concept-collection/turing-surface), whose
transform (shtns-webgpu, vendored under `src/sht/`) and renderer
(`src/render/`) it reuses; **Download .h5** exports the fitted coefficients in
exactly the geometry layout that solver consumes (`/geometry` Gx/Gy/Gz +
`/grid`, readable with h5py and MATLAB's `h5read`).

This is the browser counterpart of the MATLAB module
`sphere-surf/spherical-harmonic-fitting`, and carries the conclusions reached
there: compute coefficients once by quadrature projection (bounded at any
band-limit), never hard-truncate for display — taper (trapeziform) or smooth
(Gaussian / heat-kernel) instead — and choose the map by geometry: conformal
+ Möbius unless the conformal factor is too spread, then area-equalized.

## Pipeline

```
geometry ──► map (unit-sphere point per vertex) ──► sample x,y,z on the grid ──► GPU analysis ──► coefficients
   │                                                (+ one mixed-precision                            │
   └ point cloud ──► MLS conformal map ──►             refinement pass)                               │
     spherical Delaunay ──► induced mesh                                                              │
                                                                                                      │
                    filter gains per degree (CPU) ──► GPU synthesis on the display grid ──► surface ◄─┘
```

Analysis runs once per (geometry, lmax, oversample). Moving the filter
controls re-weights the coefficients and re-synthesizes: three small GPU
transforms, so the N slider is live.

## Features

| piece | notes |
|---|---|
| models | Spot, bunny, David, Max Planck, brain ×2, lion (`SHM1` presets), an exact icosphere, and uploads: OBJ, OFF, PLY (ascii + binary), STL (ascii + binary, vertices welded) |
| point clouds | a face-less upload — or any model via **use as: point cloud** — is mapped with the Choi–Ho–Lui MLS conformal map (nonsymmetric MLS Laplacian solved by a no-pivot sparse LU) and meshed by spherical Delaunay (= convex hull on the sphere); the induced mesh then flows through everything else. Validated against the MATLAB reference to 1e-12 |
| maps | conformal + Möbius (Nelder–Mead area correction), area-equalized (Tutte → SDEM → repair, ~5 s on Spot), Tutte, point-cloud; `auto` switches to area-equalized when the conformal factor's max/min exceeds 1e4. All in a Web Worker with progress; λ statistics match MATLAB to 3 digits |
| smooth input | exact Loop limit surface of mesh *and* map (Stam-style evaluation, numerically built per-valence operators, Newton point location on the limit chart; 0.6 µs/eval, residual ≤ 1e-12), or the raw PL mesh |
| sampling | uniform grid, or **adaptive** (Zhou et al.: 30 density-ratio Laplacian warp iterations toward the map's vertex density — recovers features a crowded map starves, at the price that coefficients encode f∘w) |
| analysis | fp32 GPU quadrature with one mixed-precision refinement pass (f64 residual re-analyzed): coefficients accurate relative to their own size, so second derivatives — and the exported spectra — are genuine |
| filters | truncate, trapeziform, Gaussian (σ defaults to 3/lmax; gain at lmax reported); live N slider; normalized log spectrum with removed-power ghosts |
| output metrics | grid residual (rms/max), and the embeddedness signal: **self-intersection count** (fixed Möller checker: unit normals, sliver skip, poles merged) and **collapsed faces** (< 10 % of the same face's area on the input surface), computed in a worker per filter change |
| coloring | mean/Gaussian curvature or radius: fully spectral sin-weighted derivatives (no 1/sin anywhere), dealiased per Veerapaneni et al. (differentiate on a ≥ 2× grid, filter to the surface band, restrict) |
| view | display oversampling (auto targets ~512 render latitudes), wireframe (input mesh edges / output collocation grid), sphere morph, ingested-points toggle, collocation-grid overlay, pole editing (ctrl+drag the marker, then apply: an exact symmetry of the fit that moves where the grid clusters) |
| export | Download .h5: filtered coefficients + grid attrs + provenance (model, map, sampling, filter) in turing-surface's geometry layout |

## Development

```
npm install
npm run dev        # local dev server
npm run build      # type-check + production build to dist/
```

Headless checks (Chrome + SwiftShader WebGPU; build first):

```
node scripts/smoke.mjs out.png                       # fit, slide N, map Spot, screenshot
node scripts/shot.mjs out.png bunny area 127 127     # any configuration: geometry map lmax N [subdiv] [filter] [color]
node scripts/test-upload.mjs                         # upload flow: OBJ + point-cloud PLY end to end
node scripts/test-export.mjs <dir>                   # .h5 download round trip
node scripts/measure-curv.mjs                        # curvature accuracy with/without refinement
```

`shot.mjs` env switches: `WIRE=in|out|both`, `MORPH=1`, `POLE=1`, `GRID=1`,
`CLOUD=1` (ingest as point cloud), `POINTS=1`, `SAMPLING=adaptive`,
`PLOTOS=1|2|4|8`, `CHARTSMOOTH=n`.

Unit tests (plain node):

```
node scripts/test-sparse.ts        # Cholesky vs dense reference
node scripts/test-chart.ts spot david brain   # map statistics vs the MATLAB module
node scripts/test-resample.ts      # resampling accuracy and timing
node scripts/test-loop.ts          # Loop subdivision operators
node scripts/test-limit-sample.ts  # exact limit evaluation (1e-16 corners, gradients vs FD)
node scripts/test-intersect.ts     # self-intersection checker (incl. brute-force parity)
node scripts/test-loaders.ts       # PLY/STL readers (a cube in all four encodings)
node scripts/test-h5.ts            # HDF5 export round trip (h5wasm/node)
node scripts/test-zhou.ts          # adaptive-sampling warp equalizes density
node scripts/test-pointcloud.ts <refdir>   # point-cloud pipeline vs MATLAB references
                                   # (generate refdir with scripts/make_pc_refs.m + make_pc_stages.m)
```

Why an exact limit surface: a band-limited fit of a flat mesh reproduces its facets once lmax resolves them,
and a smooth surface seen through a piecewise-linear *map* is still a kinked function of (θ, φ) that
rings. So both the surface and the map are evaluated as Loop limit maps, which needs the map inverted
at the grid nodes (Newton). The evaluator builds the subdivision operators numerically from canonical patches
(no hand-coded tables) and interpolates the regular-patch box-spline basis on the principal lattice; the
underlying descent is accurate to ~1e-10 (4^-depth interpolation vs 2^depth roundoff), which is well below
the fp32 transforms. OpenSubdiv via WASM was evaluated as the alternative: neither available port exposes
Loop limit evaluation, so it would have meant writing Emscripten bindings to `Bfr::Surface`.

SDEM's default here ('fold repair') is a simplification of SDEM.m that is both ~7× faster and better
converged: each density-equalizing step is taken outright and any folds it creates are removed by local
Tutte relaxation (halving the step if that fails), instead of the paper's stereographic Beltrami-solve
overlap correction, which profiling showed was 76 % of every step. The density target is the geometric
one (population = face areas, i.e. a uniform conformal factor) rather than the paper's mesh-density
target. Measured: Spot 4.9 s, λ std/mean 0.24, max/min 24 (paper route: 14 s, 1.53, 252; MATLAB stalled
at density spread 0.45); David converges in 29 steps to λ std/mean 0.02; bunny 7.3 s, λ max/min 730
(ears). The paper-route port itself differs from SDEM.m in four deliberate ways, each needed for a
Cholesky-based solver and for robustness: |μ| is clamped below 1 on *every* face before the linear
Beltrami solve, folds are repaired inside the loop whenever they appear, a step is accepted when it does
not *increase* the overlap count (halving capped at 12), and the loop stops once 20 consecutive steps are
rejected — the density spread is not a usable stopping signal because progress comes in bursts.

Presets live in `public/presets/` (`presets.json` records provenance; meshes are
`SHM1` binaries: magic, uint32 nv, uint32 nf, float32 xyz, uint32 faces,
little-endian, centred and scaled to unit bounding-box diagonal). Sample point
clouds (`data/*.ply`) are the Spot and David vertices.

Numerics: transforms are fp32 (SHTns layout: orthonormal, Condon–Shortley, m ≥ 0)
with one mixed-precision refinement pass on analysis; the grid is Gauss–Legendre ×
equispaced, analysis oversampling `auto` = 2× (4× for a PL input, 3× for a rough
map). On the exact sphere at lmax 127 the mean-curvature field is uniform to
0.03 % rms (unrefined: 1.0 %).

## References

The page footer lists the papers each piece implements: Choi–Lam–Lui (FLASH
conformal map, SIAM JIS 2015), Choi–Leung-Liu–Gu–Lui (Möbius area correction
via partial welding, SIAM JIS 2020), Lyu–Lui–Choi (spherical density-equalizing
map, SIAM JIS 2024), Choi–Ho–Lui (point-cloud parameterization, SIAM JIS 2016),
Gu–Wang–Chan–Thompson–Yau (grid analysis & filtering, IEEE TMI 2004),
Zhou–Bao–Shi (adaptive sampling, CAD 2004), Stam (Loop evaluation, SIGGRAPH 98),
Veerapaneni–Rahimian–Biros–Zorin (curvature dealiasing, JCP 2011).

## License

CECILL-2.1 (inherited from SHTns via the vendored shtns-webgpu transforms).
