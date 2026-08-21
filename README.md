# reharm

Spherical-harmonic fitting of genus-0 closed surfaces, live in the browser:
chart the mesh onto the sphere, resample onto a Gauss–Legendre grid, one GPU
analysis, then filter by degree in real time. Sibling of
[turing-surface](https://github.com/concept-collection/turing-surface), whose
transform (shtns-webgpu, vendored under `src/sht/`) and renderer
(`src/render/`) it reuses; a fitted surface is exactly the geometry format that
solver consumes.

This is the browser counterpart of the MATLAB module
`sphere-surf/spherical-harmonic-fitting`, and carries the conclusions reached
there: compute coefficients once by quadrature projection (bounded at any
band-limit), never hard-truncate for display — taper (trapeziform) or smooth
(Gaussian / heat-kernel) instead — and choose the chart by geometry: conformal
+ Möbius unless the conformal factor is too spread, then area-equalized.

## Pipeline

```
geometry ──► chart (unit-sphere point per vertex) ──► sample x,y,z on the grid ──► GPU analysis ──► coefficients
                                                                                                       │
                    filter gains per degree (CPU) ──► GPU synthesis on the display grid ──► surface ◄──┘
```

Analysis runs once per (geometry, lmax, oversample). Moving the filter
controls re-weights the coefficients and re-synthesizes: three small GPU
transforms, so the N slider is live.

## Status

| piece | state |
|---|---|
| synthetic presets (bumpy sphere, ellipsoid, peanut, sphere) — exact chart, fit end-to-end | done |
| mesh presets (Spot, bunny, David, Max Planck, brain ×2, lion) + OBJ/OFF upload, viewing | done |
| filters: hard cutoff, trapeziform, Gaussian, keep-1/r; live N; morph; power spectrum | done |
| charts in the browser: conformal + Möbius (Nelder–Mead), Tutte, fold repair, chart policy; sparse Cholesky with geometric nested dissection (`src/chart/`) | done — matches MATLAB's λ statistics to 3 digits |
| mesh → grid resampling: piecewise-linear over the spherical triangulation, grid-hash point location (`src/fit/resample.ts`) | done |
| area-equalized chart (Tutte → SDEM → repair) for crowded shapes; `auto` switches to it when the conformal factor's max/min exceeds 1e4 | done — restores the bunny's ears and Spot's legs |
| charts run in a Web Worker with progress (SDEM takes ~25 s on a 3k-vertex mesh) | done |
| smooth input: exact Loop limit surface (default) or the raw PL mesh (a band-limited fit of a flat mesh shows its facets once lmax resolves them; a smooth surface must also be seen through a smooth chart, so the limit chart is evaluated too) | done |
| display oversampling: synthesize on a grid finer than lmax, turing-surface-style (own synthesis plan; coefficients re-laid out); color by mean/Gaussian curvature or radius, with colorbar | done |
| **exact Loop limit surface** (`src/mesh/loopLimit.ts`, `src/fit/limitSample.ts`): Stam-style evaluation with numerically built per-valence operators and an analytically evaluated quartic box spline on regular patches; the chart is a limit map too, and grid directions are located on it by Newton (neighbour hopping, vertex-singularity handling). Default `smooth input`. 0.6 µs per evaluation; Newton to 1e-12 on all 131k queries of a 256×512 grid in 0.6 s | done |
| Gaussian σ defaults to 3/lmax (gain ≈ 1 % at lmax) and the gain at lmax is reported — σ ≪ π/lmax is a no-op that looks like ringing | done |
| metrics: vertex error (scattered synthesis), collapsed faces, self-intersections at grid resolution | later |
| Zhou adaptive sampling; numbl `.m` presets; coefficient export | later |

## Development

```
npm install
npm run dev        # local dev server
npm run build      # type-check + production build to dist/
node scripts/smoke.mjs out.png   # headless Chrome (SwiftShader WebGPU): fit, slide N, chart Spot, screenshot
node scripts/test-sparse.ts      # Cholesky vs dense reference
node scripts/test-chart.ts spot david brain   # chart statistics (compare with the MATLAB module's numbers)
node scripts/test-resample.ts    # resampling accuracy and timing
```

Chart timings in Node: conformal + Möbius — Spot 0.08 s, David 0.23 s, brain (48k vertices) 1.7 s;
area-equalized (200 SDEM steps) — Spot and bunny ~25 s. Resampling onto a 256×512 grid adds 0.04–0.12 s.
`node scripts/shot.mjs out.png bunny area 127 127` screenshots any configuration headlessly;
`node scripts/test-loop.ts` / `test-limit-sample.ts` check the limit evaluator (corner values vs the limit
mask and edge continuity to 1e-16; analytic gradients vs finite differences).

Why an exact limit surface: a band-limited fit of a flat mesh reproduces its facets once lmax resolves them,
and a smooth surface seen through a piecewise-linear *chart* is still a kinked function of (θ, φ) that
rings. So both the surface and the chart are evaluated as Loop limit maps, which needs the chart inverted
at the grid nodes (Newton). The evaluator builds the subdivision operators numerically from canonical patches
(no hand-coded tables) and interpolates the regular-patch box-spline basis on the principal lattice; the
underlying descent is accurate to ~1e-10 (4^-depth interpolation vs 2^depth roundoff), which is well below
the fp32 transforms. OpenSubdiv via WASM was evaluated as the alternative: neither available port exposes
Loop limit evaluation, so it would have meant writing Emscripten bindings to `Bfr::Surface`.

SDEM's default here ('fold repair') is a simplification of SDEM.m that is both ~7× faster and better
converged: each density-equalizing step is taken outright and any folds it creates are removed by local
Tutte relaxation (halving the step if that fails), instead of the paper's stereographic Beltrami-solve
overlap correction, which is still available ('Beltrami solves (paper)'). Profiling showed that
correction was 76 % of every step (about five Beltrami solves per step from rejected attempts). The
density target is the geometric one (population = face areas, i.e. a uniform conformal factor) rather
than the paper's mesh-density target. Measured: Spot 4.9 s, λ std/mean 0.24, max/min 24 (paper route:
14 s, 1.53, 252; MATLAB stalled at density spread 0.45); David converges in 29 steps to λ std/mean 0.02;
bunny 7.3 s, λ max/min 730 (ears). The paper-route port itself differs from SDEM.m in four deliberate
ways, each needed for a Cholesky-based solver and for robustness: |μ| is clamped below 1 on *every* face before the linear Beltrami solve
(the operator is elliptic only there; MATLAB's LU tolerated the indefinite system), folds are repaired
inside the loop whenever they appear (a fold left in the cap one hemisphere ignores would otherwise veto
every later step), a step is accepted when it does not *increase* the overlap count with the step
halving capped at 12, and the loop stops once 20 consecutive steps are rejected (the map no longer
moves) — MATLAB's ε = 10⁻³ on the density spread is unreachable, and the spread itself is not a usable
stopping signal because progress comes in bursts. The density spread reaches 0.24 on Spot (149 steps)
and 0.25 on the bunny (cap), versus 0.45 where the MATLAB run stalled. Five tangential smoothing
steps then take the repair kinks out of the chart before anything is resampled through it.

Presets live in `public/presets/` (`presets.json` records provenance; meshes are
`SHM1` binaries: magic, uint32 nv, uint32 nf, float32 xyz, uint32 faces,
little-endian, centred and scaled to unit bounding-box diagonal).

Numerics: transforms are fp32 (SHTns layout: orthonormal, Condon–Shortley, m ≥ 0);
a band-limited shape round-trips to ~1e-6 relative at lmax 63. The grid is
Gauss–Legendre × equispaced, `oversample 2` doubling both directions.

## License

CECILL-2.1 (inherited from SHTns via the vendored shtns-webgpu transforms).
