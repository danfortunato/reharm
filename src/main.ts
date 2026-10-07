/**
 * reharm: page wiring.
 *
 *   geometry ──► chart ──► sample on the Gauss grid ──► GPU analysis ──► coefficients
 *                                                                            │
 *            filter gains (CPU, per degree) ──► GPU synthesis ──► surface ◄──┘
 *
 * The analysis runs once per (geometry, lmax, oversample); moving the filter
 * controls re-weights the coefficients and re-synthesizes -- three small GPU
 * transforms and one buffer upload, so the slider is live.
 */
import { SphereScene } from './render/SphereScene.ts';
import { buildTopology, fillPositions, fillFieldValues, fillColors, bestFitRadius, type SphereMeshTopology } from './render/sphereMesh.ts';
import { colormaps, colormapNames } from './render/colormaps.ts';
import { Colorbar, floorRange } from './render/colorbar.ts';
import { gridCurvature } from './render/gridCurvature.ts';
import { GridFitter, type Coefficients } from './fit/gridFit.ts';
import { applyGains, degreeGains, powerSpectrum, type FilterSpec } from './fit/filters.ts';
import { encodeGeometryH5, filterLabel, type H5Module } from './fit/exportH5.ts';
import {
  normalizeGeometry,
  sendGeometry,
  TURING_SURFACE_URL,
  type TuringGeometryPayload,
} from './fit/exportTuring.ts';
import { bboxDiagonal, eulerCharacteristic, makeMesh, normalize, type Mesh } from './mesh/types.ts';
import { icosphere } from './mesh/icosphere.ts';
import { readMeshFile } from './mesh/loaders.ts';
import type { ChartInfo, ChartKind } from './chart/policy.ts';
import type { ChartRequest } from './chart/chart.worker.ts';
import { SphereLocator } from './fit/resample.ts';
import { SurfaceRenderer, type DerivFields } from './fit/renderGrid.ts';
import { gridForLmax } from './sht/layout.ts';
import { gaussNodesWeights } from './sht/gauss.ts';
import type { SampleInit, SampleRequest, SampleResult } from './fit/sample.worker.ts';
import type { GridMetricsRequest, GridMetricsResult } from './fit/metrics.worker.ts';
import {
  loadManifest, loadPreset, syntheticMesh, syntheticShapes,
  type ChartedMesh, type PresetEntry, type ShapeFn,
} from './mesh/presets.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const elGeometry = $<HTMLSelectElement>('geometry');
const elUploadFallback = $<HTMLInputElement>('upload-fallback');
const elInputForm = $<HTMLSelectElement>('inputform');
const elChart = $<HTMLSelectElement>('chart');
const elSdemSteps = $<HTMLInputElement>('sdemsteps');
const elChartSmooth = $<HTMLInputElement>('chartsmooth');
const elSubdiv = $<HTMLSelectElement>('subdiv');
const elLmax = $<HTMLSelectElement>('lmax');
const elOversample = $<HTMLSelectElement>('oversample');
const elSampling = $<HTMLSelectElement>('sampling');
const elFilter = $<HTMLSelectElement>('filter');
const elN = $<HTMLInputElement>('N');
const elNnum = $<HTMLInputElement>('Nnum');
const elRamp = $<HTMLInputElement>('ramp');
const elSigma = $<HTMLInputElement>('sigma');
const elSigmaSlider = $<HTMLInputElement>('sigmaslider');
const elPlotOs = $<HTMLSelectElement>('plotos');
const elColor = $<HTMLSelectElement>('color');
const elColormap = $<HTMLSelectElement>('colormap');
const elColorbar = $<HTMLDivElement>('colorbar-fit');
const elWire = $<HTMLSelectElement>('wire');
const elPole = $<HTMLButtonElement>('polemode');
let poleArmed = false;
function disarmPoleButton(): void { poleArmed = false; elPole.classList.remove('armed'); }
const wireIn = () => elWire.value === 'in' || elWire.value === 'both';
const wireOut = () => elWire.value === 'out' || elWire.value === 'both';
const elMorph = $<HTMLInputElement>('morph');
const elShowPoints = $<HTMLInputElement>('showpoints');
const elLabPoints = $<HTMLElement>('lab-points');
const elExport = $<HTMLButtonElement>('exporth5');
const elExportTs = $<HTMLButtonElement>('exportts');
const elStatus = $<HTMLParagraphElement>('status');
const elMetrics = $<HTMLPreElement>('metrics');
const elSpectrum = $<HTMLCanvasElement>('spectrum');
const viewOrig = $<HTMLDivElement>('view-orig');
const viewFit = $<HTMLDivElement>('view-fit');
const noteOrig = $<HTMLDivElement>('note-orig');
const noteFit = $<HTMLDivElement>('note-fit');
const busyFit = $<HTMLDivElement>('busy-fit');
const busyText = $<HTMLDivElement>('busy-text');
const busyFill = $<HTMLDivElement>('busy-fill');

const FLAT = [0.80, 0.82, 0.93] as const;

// ---------------------------------------------------------------- state
let device: GPUDevice | null = null;
let fitter: GridFitter | null = null;
let fitterKey = '';
let renderer: SurfaceRenderer | null = null;
let rendererKey = '';
/** synthesis on the ANALYSIS grid, for the residual against the resampled/exact surface */
let residCoords: Float32Array | null = null;
let presets: PresetEntry[] = [];

let mesh: Mesh | null = null;
let meshLabel = '';
/** the last uploaded mesh; kept so the model dropdown's named entry can be
 *  returned to after visiting presets (survives resetGeometryState) */
let uploadedMesh: Mesh | null = null;
let uploadedName = '';
/** the dropdown value to snap back to when the custom… file dialog is cancelled */
let lastGeometryKey = '';
/** exact sampler for synthetic shapes (no resampling step needed) */
let sampler: ShapeFn | null = null;
let charted: ChartedMesh | null = null;
let chartInfo: ChartInfo | null = null;
/** the mesh actually resampled: the input, or its Loop-subdivided smooth version */
let fitMesh: Mesh | null = null;
/** the raw ingested cloud (pre-compaction) when the input was a point cloud —
 *  the 'points' checkbox swaps the input pane back to it */
let cloudPositions: Float32Array | null = null;

/** The input pane is currently showing the ingested points, not the mesh. */
const pointsActive = (): boolean =>
  elShowPoints.checked && !!cloudPositions && !inputSphereView && !!mesh && mesh.nf > 0;

/** Swap the input pane between the induced mesh and the ingested points.
 *  Only in the surface view — the map (sphere) view ignores the toggle. The
 *  mesh wireframe follows: no edges over the points view, and the dropdown's
 *  choice comes back when the points go away. */
function applyPointsView(): void {
  if (!sceneOrig || !mesh || mesh.nf === 0) return;
  const on = pointsActive();
  sceneOrig.setMeshVisible(!on);
  sceneOrig.setPoints(on ? cloudPositions : null);
  if (!inputSphereView) sceneOrig.setWireframe(wireIn() && !on);
}
/** All mesh sampling (exact Loop limit or PL, uniform or Zhou-warped) runs in
 *  a worker — limit construction and Newton evaluation block for seconds at
 *  high lmax, and the adaptive warp's 30 density iterations are not free
 *  either. One request at a time (fits are serialized through the GPU
 *  queue); killing the worker resolves the pending request with null so a
 *  queued fit can notice it is stale instead of awaiting a reply that will
 *  never come. */
let sampleWorker: Worker | null = null;
let samplePending: ((r: SampleResult | null) => void) | null = null;
let sampleRequestId = 0;
let lastNewtonResidual = 0;

function killSampleWorker(): void {
  sampleWorker?.terminate(); sampleWorker = null;
  const p = samplePending; samplePending = null; p?.(null);
}

function startSampleWorker(m: Mesh, chart: Float64Array, mode: SampleInit['mode']): void {
  killSampleWorker();
  sampleWorker = new Worker(new URL('./fit/sample.worker.ts', import.meta.url), { type: 'module' });
  sampleWorker.onmessage = (e: MessageEvent<SampleResult>) => {
    if (e.data.id !== sampleRequestId) return;
    const p = samplePending; samplePending = null; p?.(e.data);
  };
  const req: SampleInit = { init: true, positions: m.positions, faces: m.faces, chart, mode };
  sampleWorker.postMessage(req);
}

function sampleSurface(queries: Float64Array, warp?: SampleRequest['warp']): Promise<SampleResult | null> {
  if (!sampleWorker) return Promise.resolve(null);
  return new Promise((resolve) => {
    samplePending = resolve;
    const transfers = [queries.buffer, ...(warp ? [warp.capDirs.buffer, warp.gridFaces.buffer] : [])];
    sampleWorker!.postMessage({ id: ++sampleRequestId, queries, warp } satisfies SampleRequest, transfers);
  });
}
let chartWorker: Worker | null = null;
let chartRequestId = 0;
/** bumped whenever the geometry changes; queued fits for an older geometry are dropped */
let geometryGen = 0;

let coef: Coefficients | null = null;
let spectrum: Float64Array | null = null;
let topo: SphereMeshTopology | null = null;
let gridTheta: Float64Array | null = null; // display-grid latitudes, for the curvature coloring
let gridExact: Float32Array | null = null; // exact surface on the grid (synthetic): for the error
let posBuf: Float32Array | null = null;
let coords: Float32Array | null = null;
/** radius of the sphere the 'sphere' toggle jumps to (the surface's rms radius) */
const morphT = () => (elMorph.checked ? 0 : 1);
let morphRadius = 1;

let sceneOrig: SphereScene | null = null;
let sceneFit: SphereScene | null = null;
let colorbar: Colorbar | null = null;
let resizeObs: ResizeObserver | null = null;

/** The PDE collocation-grid overlay (rings and meridians of the exact
 *  degree-lmax Gauss grid, gridForLmax(lmax, 1)), drawn as curves sampled at
 *  the display grid's resolution — see updateGridOverlay. */
let gridLinesKey = '';
let gridLinesIndices: Uint32Array | null = null;
let gridLinesTheta: Float64Array | null = null;   // the PDE grid's latitudes, ascending
let gridLinesNphi = 0;                            // the PDE grid's longitude count

let lastFitMs = 0;
/** wall time of the whole last fit: sampling + analysis + display and filter syntheses */
let lastTotalMs = 0;

/** Embeddedness of the fitted surface, measured on the analysis grid's tensor
 *  mesh (poles merged): self-intersection count and collapsed-face count vs
 *  the input surface at the same grid points. Computed in a worker; null
 *  while a fit or filter change is in flight ("checking…" in the panel). */
let gridMetrics: GridMetricsResult | null = null;
let metricsWorker: Worker | null = null;
let metricsId = 0;
let metricsBusy = false;
/** the surface changed while the worker was busy: rerun on the latest state */
let metricsDirty = false;

/** Progress overlay on the fit pane. `fraction` in [0,1], or undefined for indeterminate. */
function setBusy(text: string | null, fraction?: number): void {
  busyFit.hidden = text === null;
  if (text === null) return;
  busyText.textContent = text;
  busyFill.classList.toggle('indeterminate', fraction === undefined);
  busyFill.style.width = fraction === undefined ? '' : `${Math.round(100 * fraction)}%`;
}

/** The status line shows errors only — no live progress chatter (the busy
 *  overlay carries progress); an informational call clears any stale error. */
const status = (msg: string, err = false) => {
  elStatus.textContent = err ? msg : '';
  elStatus.classList.toggle('err', err);
  if (err) console.error(msg);
};

// ---------------------------------------------------------------- filter controls
function filterSpec(): FilterSpec {
  return {
    kind: elFilter.value as FilterSpec['kind'],
    N: Number(elN.value),
    ramp: Number(elRamp.value),
    sigma: Number(elSigma.value),
    r: 1,   // the keep-1/r filter is not offered in the UI
  };
}

function syncFilterControls(): void {
  const k = elFilter.value;
  $('lab-N').hidden = !(k === 'hard' || k === 'trapeziform');
  $('lab-ramp').hidden = k !== 'trapeziform';
  $('lab-sigma').hidden = k !== 'gaussian';
}

function setNRange(lmax: number): void {
  elN.max = String(lmax); elNnum.max = String(lmax);
  if (Number(elN.value) > lmax) { elN.value = String(lmax); elNnum.value = String(lmax); }
}

/** Default Gaussian width for a band-limit: gain(lmax) ~ 1 % (sigma = 3/lmax). */
const defaultSigma = (lmax: number) => Number((3 / lmax).toPrecision(2));

/** Set both σ controls: the number box holds the value, the slider its log10. */
function setSigmaControls(v: number): void {
  elSigma.value = String(v);
  elSigmaSlider.value = String(Math.log10(Math.min(Math.max(v, 1e-3), 1)));
}

// ---------------------------------------------------------------- scenes
function flatColors(n: number): Float32Array {
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) c.set(FLAT, 3 * i);
  return c;
}

function newScene(container: HTMLElement, nv: number, indices: Uint32Array, positions: Float32Array): SphereScene {
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--sphere-bg').trim();
  const s = new SphereScene(container, nv, indices, positions, bg);
  s.updateColors(flatColors(nv));
  s.resize(container.clientWidth, container.clientHeight);
  s.fitCamera();
  return s;
}

let inputSphereView = false;

function showOriginal(m: Mesh): void {
  sceneOrig?.dispose();
  inputSphereView = false;
  // A face-less input is a point cloud: render the points themselves (the
  // induced Delaunay mesh replaces this view once the map is computed).
  if (m.nf === 0) {
    const pts = Float32Array.from(m.positions);
    sceneOrig = newScene(viewOrig, m.nv, new Uint32Array(0), pts);
    sceneOrig.setPoints(pts);
    if (sceneFit) sceneOrig.syncCamerasWith(sceneFit);
    noteOrig.textContent = '';
    updateMetrics();
    return;
  }
  // The input is a PL mesh and is rendered faithfully as one: a vertex per
  // face corner, so computed normals are per-face — honest flat shading
  // (smooth shading would hide the facets the fit has to contend with).
  const n = m.nf * 3, pos = new Float32Array(n * 3), idx = new Uint32Array(n);
  for (let k = 0; k < n; k++) { pos.set(m.positions.subarray(3 * m.faces[k], 3 * m.faces[k] + 3), 3 * k); idx[k] = k; }
  sceneOrig = newScene(viewOrig, n, idx, pos);
  sceneOrig.setWireframe(wireIn());
  if (sceneFit) sceneOrig.syncCamerasWith(sceneFit);
  noteOrig.textContent = '';
  restorePoleMode();
  applyInputMorph();
  updateMetrics();   // the summary line shows the new mesh's counts right away
}

/** A rebuilt input scene comes up without the marker: restore it if armed. */
function restorePoleMode(): void {
  if (!poleArmed || !sceneOrig || !poleDir) return;
  placePoleMarker(poleDir);
}

/**
 * The input's map view: the triangulation projected onto the sphere of the
 * mesh's rms radius, with each triangle subdivided (midpoints reprojected)
 * until every edge spans at most ARC_TARGET — big map triangles drawn as flat
 * chords cut visibly inside the sphere. The ORIGINAL edges are kept separately
 * as curved polylines for the wireframe overlay (the fill's subdivision edges
 * are clutter). Cached per (mesh, map).
 */
let sphereView: { pos: Float32Array; idx: Uint32Array; linePos: Float32Array; lineIdx: Uint32Array } | null = null;
const ARC_TARGET = 0.06;   // max chord arc in radians (~3.4 degrees)
const MAX_DEPTH = 5;

function buildSphereView(): typeof sphereView {
  if (sphereView || !mesh || !charted || charted.mesh !== mesh) return sphereView;
  const S = charted.chart, f = mesh.faces, nf = mesh.nf, nv = mesh.nv;
  let s2 = 0;
  for (let v = 0; v < nv; v++) s2 += mesh.positions[3 * v] ** 2 + mesh.positions[3 * v + 1] ** 2 + mesh.positions[3 * v + 2] ** 2;
  const r = Math.sqrt(s2 / nv);
  const arc = (a: number, b: number) =>
    2 * Math.asin(Math.min(1, Math.hypot(S[3 * a] - S[3 * b], S[3 * a + 1] - S[3 * b + 1], S[3 * a + 2] - S[3 * b + 2]) / 2));
  const depthOf = (t: number): number => {
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const m = Math.max(arc(a, b), arc(b, c), arc(c, a));
    return Math.min(MAX_DEPTH, Math.max(0, Math.ceil(Math.log2(m / ARC_TARGET))));
  };
  // fill: per-face principal-lattice subdivision with SHARED vertices so the
  // shading is smooth — corner-duplicated flat shading would render every
  // subdivision facet and read as a dense wireframe
  const unit = (v: number) => [S[3 * v], S[3 * v + 1], S[3 * v + 2]];
  const dOf = new Uint8Array(nf);
  let nV = 0, nT = 0;
  for (let t = 0; t < nf; t++) {
    const d = 1 << depthOf(t);
    dOf[t] = d; nV += ((d + 1) * (d + 2)) / 2; nT += d * d;
  }
  const pos = new Float32Array(nV * 3);
  const idx = new Uint32Array(nT * 3);
  let vw = 0, iw = 0;
  for (let t = 0; t < nf; t++) {
    const d = dOf[t];
    const A = unit(f[3 * t]), B = unit(f[3 * t + 1]), C = unit(f[3 * t + 2]);
    const base = vw / 3;
    const rowStart: number[] = [];
    for (let i = 0, count = 0; i <= d; i++) { rowStart.push(count); count += i + 1; }
    for (let i = 0; i <= d; i++)
      for (let j = 0; j <= i; j++) {
        const wa = (d - i) / d, wb = (i - j) / d, wc = j / d;
        const x = wa * A[0] + wb * B[0] + wc * C[0];
        const y = wa * A[1] + wb * B[1] + wc * C[1];
        const z = wa * A[2] + wb * B[2] + wc * C[2];
        const n2 = Math.hypot(x, y, z) || 1;
        pos[vw++] = (r * x) / n2; pos[vw++] = (r * y) / n2; pos[vw++] = (r * z) / n2;
      }
    for (let i = 0; i < d; i++)
      for (let j = 0; j <= i; j++) {
        const a = base + rowStart[i] + j, b = base + rowStart[i + 1] + j, c = base + rowStart[i + 1] + j + 1;
        idx[iw++] = a; idx[iw++] = b; idx[iw++] = c;
        if (j < i) { idx[iw++] = a; idx[iw++] = c; idx[iw++] = base + rowStart[i] + j + 1; }
      }
  }
  // wireframe: the original edges as curved polylines
  const edges: number[] = [];
  const seen = new Set<number>();
  for (let t = 0; t < nf; t++)
    for (let k = 0; k < 3; k++) {
      const a = f[3 * t + k], b = f[3 * t + ((k + 1) % 3)];
      const key = Math.min(a, b) * nv + Math.max(a, b);
      if (!seen.has(key)) { seen.add(key); edges.push(a, b); }
    }
  // Lines are sampled 4x finer than the fill (their chords then sag 16x less
  // than the fill's facets, so they never dip below it) and get a hair of
  // radial clearance on top.
  const lineArc = ARC_TARGET / 4;
  const rLine = r * 1.002;
  let nPts = 0, nSegs = 0;
  const segsOf = (a: number, b: number) => Math.min(256, Math.max(1, Math.ceil(arc(a, b) / lineArc)));
  for (let e = 0; e < edges.length; e += 2) { const s = segsOf(edges[e], edges[e + 1]); nPts += s + 1; nSegs += s; }
  const linePos = new Float32Array(nPts * 3);
  const lineIdx = new Uint32Array(nSegs * 2);
  let lp = 0, li = 0, base = 0;
  for (let e = 0; e < edges.length; e += 2) {
    const a = unit(edges[e]), b = unit(edges[e + 1]);
    const s = segsOf(edges[e], edges[e + 1]);
    for (let k = 0; k <= s; k++) {
      const t = k / s;
      const x = (1 - t) * a[0] + t * b[0], y = (1 - t) * a[1] + t * b[1], z = (1 - t) * a[2] + t * b[2];
      const n2 = Math.hypot(x, y, z) || 1;
      linePos[lp++] = (rLine * x) / n2; linePos[lp++] = (rLine * y) / n2; linePos[lp++] = (rLine * z) / n2;
      if (k < s) { lineIdx[li++] = base + k; lineIdx[li++] = base + k + 1; }
    }
    base += s + 1;
  }
  sphereView = { pos, idx, linePos, lineIdx };
  return sphereView;
}

/** Rebuild the input pane as its map on the sphere. */
function showInputSphere(): void {
  const sv = buildSphereView();
  if (!sv) return;
  sceneOrig?.dispose();
  sceneOrig = newScene(viewOrig, sv.pos.length / 3, sv.idx, sv.pos);
  if (wireIn()) sceneOrig.setGridLines(sv.linePos, sv.lineIdx);
  if (sceneFit) sceneOrig.syncCamerasWith(sceneFit);
  noteOrig.textContent = '';
  restorePoleMode();
}

/** Swap the input pane between the mesh and its map on the sphere. */
function applyInputMorph(): void {
  if (!mesh) return;
  const wantSphere = elMorph.checked && !!charted && charted.mesh === mesh;
  if (wantSphere) { showInputSphere(); inputSphereView = true; }
  else if (inputSphereView) { inputSphereView = false; showOriginal(mesh); }
}

function installResize(): void {
  resizeObs?.disconnect();
  resizeObs = new ResizeObserver(() => {
    sceneOrig?.resize(viewOrig.clientWidth, viewOrig.clientHeight);
    sceneFit?.resize(viewFit.clientWidth, viewFit.clientHeight);
  });
  resizeObs.observe(viewOrig);
  resizeObs.observe(viewFit);
  // the spectrum canvas stretches with the log box's height: redraw on resize
  new ResizeObserver(() => drawSpectrum(lastGains ?? undefined)).observe(elSpectrum);
}

// ---------------------------------------------------------------- geometry
function resetGeometryState(): void {
  geometryGen++;
  coef = null; spectrum = null; gridExact = null; sampler = null; charted = null; chartInfo = null; fitMesh = null; baseChart = null; derivCache = null; lastGains = null; sphereView = null; inputSphereView = false; lastTotalMs = 0;
  cloudPositions = null; elLabPoints.hidden = true;   // the checkbox itself stays: the preference survives reloads of the same cloud
  killSampleWorker();
  poleRot = null; poleDir = null; poleDragging = false; disarmPoleButton();
  chartWorker?.terminate(); chartWorker = null;
  // a checker run on the old geometry could take seconds: kill it outright
  metricsWorker?.terminate(); metricsWorker = null;
  gridMetrics = null; metricsBusy = false; metricsDirty = false; metricsId++;
  sceneFit?.dispose(); sceneFit = null;       // do not leave the previous surface on screen
  viewFit.querySelectorAll('canvas').forEach((c) => c.remove());
  noteFit.textContent = ''; setBusy(null);
  updateMetrics(); drawSpectrum();
}

async function selectGeometry(key: string): Promise<void> {
  resetGeometryState();
  if (key === 'uploaded') {
    if (!uploadedMesh) throw new Error('no uploaded mesh');
    mesh = uploadedMesh; meshLabel = uploadedName;
  } else if (key.startsWith('synthetic:')) {
    const name = key.slice('synthetic:'.length);
    const shape = syntheticShapes[name];
    if (name === 'sphere') {
      // an icosphere, not the lat-long tessellation: near-uniform vertices and
      // no distinguished poles (honest input display and a proper test cloud);
      // its unit vertices ARE its exact chart. The fit still samples the
      // analytic sphere exactly through the sampler.
      const m = icosphere(4);
      mesh = m; charted = { mesh: m, chart: Float32Array.from(m.positions) };
    } else {
      const cm = syntheticMesh(shape.fn);
      mesh = cm.mesh; charted = cm;
    }
    sampler = shape.fn; meshLabel = shape.label;
    // 'use as: point cloud' discards the exact chart too — the vertices go
    // through the full cloud pipeline like any upload
    if (elInputForm.value === 'cloud') { sampler = null; charted = null; }
  } else {
    const entry = presets.find((p) => p.key === key);
    if (!entry) throw new Error(`unknown preset ${key}`);
    status(`loading ${entry.label}…`);
    mesh = await loadPreset(entry);
    meshLabel = entry.label;
  }
  stripToCloud();
  showOriginal(mesh);
  await computeChart();
  await runFit();
}

/** 'use as: point cloud' throws the model's connectivity away — the map
 *  pipeline rediscovers it by spherical Delaunay (synthetic shapes keep
 *  their exact charts and are exempt). */
function stripToCloud(): void {
  if (!mesh || sampler) return;
  if (elInputForm.value === 'cloud' && mesh.nf > 0) mesh = makeMesh(mesh.positions, new Uint32Array(0));
  if (mesh.nf === 0) elShowPoints.checked = true;   // the points view defaults on for cloud inputs
}

async function uploadGeometry(file: File): Promise<void> {
  // parse before touching any state, so a bad file leaves the current fit alone
  status(`reading ${file.name}…`);
  const m = normalize(await readMeshFile(file));
  uploadedMesh = m; uploadedName = file.name;
  upsertUploadedOption(file.name);
  elGeometry.value = 'uploaded'; lastGeometryKey = 'uploaded';
  resetGeometryState();
  mesh = m; meshLabel = file.name;
  stripToCloud();
  showOriginal(mesh);
  await computeChart();
  await runFit();
}

/** The uploaded file's own entry in the model dropdown, just above custom…. */
function upsertUploadedOption(label: string): void {
  let o = elGeometry.querySelector<HTMLOptionElement>('option[value="uploaded"]');
  if (!o) {
    o = document.createElement('option'); o.value = 'uploaded';
    elGeometry.insertBefore(o, elGeometry.querySelector('option[value="custom"]'));
  }
  o.textContent = label;
}

/** Chart a mesh onto the sphere in a worker (synthetic shapes carry their own exact chart). */
async function computeChart(): Promise<void> {
  if (!mesh) return;
  if (sampler) { status('synthetic shape: its map is exact, the map selector does not apply'); return; }
  // a face-less upload is a point cloud: the worker maps it (Choi-Ho-Lui MLS
  // conformal) and returns the spherical-Delaunay mesh it induces, which then
  // IS the mesh — the Euler check applies only once there are faces
  const isCloud = mesh.nf === 0;
  if (!isCloud && eulerCharacteristic(mesh) !== 2) return;
  chartWorker?.terminate();
  const worker = new Worker(new URL('./chart/chart.worker.ts', import.meta.url), { type: 'module' });
  chartWorker = worker;
  const id = ++chartRequestId;
  const kind = elChart.value as ChartKind;
  setBusy(`mapping: ${isCloud ? 'point cloud…' : kind === 'area' || kind === 'balanced' ? 'Tutte map…' : 'conformal map…'}`);
  const m = mesh;
  try {
    const { S, faces, positions } = await new Promise<{ S: Float64Array; faces?: Uint32Array; positions?: Float32Array }>((resolve, reject) => {
      worker.onmessage = (e: MessageEvent) => {
        if (e.data.id !== id) return;
        if (e.data.progress) {
          const msg = e.data.progress as string;
          const step = /SDEM step (\d+)\/(\d+)/.exec(msg);
          setBusy(`mapping: ${msg}`, step ? Number(step[1]) / Number(step[2]) : undefined);
          return;
        }
        if (e.data.error) { reject(new Error(e.data.error)); return; }
        chartInfo = e.data.info as ChartInfo;
        resolve({ S: e.data.S as Float64Array, faces: e.data.faces as Uint32Array | undefined, positions: e.data.positions as Float32Array | undefined });
      };
      worker.onerror = (ev) => reject(new Error(ev.message));
      const req: ChartRequest = { id, positions: m.positions, faces: m.faces, kind, maxLambdaRatio: 1e4, sdemMaxSteps: Number(elSdemSteps.value), chartSmoothing: Number(elChartSmooth.value) };
      worker.postMessage(req);
    });
    if (id !== chartRequestId) return; // superseded
    setBusy('resampling and transforming…');
    let installed = m;
    if (faces && faces.length) {
      installed = makeMesh(positions ?? m.positions, faces);   // the induced triangulation
      mesh = installed;
      cloudPositions = Float32Array.from(m.positions);
      elLabPoints.hidden = false;
      // the induced mesh replaces the points view of the SAME object: keep
      // whatever zoom/pan the user did while the map was computing
      const pose = sceneOrig?.cameraState();
      showOriginal(installed);
      if (pose) sceneOrig?.setCameraState(pose);
      applyPointsView();
    }
    applyChart(installed, S);
  } catch (e) {
    if (id === chartRequestId) { status(`map failed: ${(e as Error).message}`, true); setBusy(null); }
  } finally {
    worker.terminate();
    if (chartWorker === worker) chartWorker = null;
  }
}

/** Install a chart: hand the mesh and map to the sampling worker (exact Loop
 *  limit or PL, per the smoothing select). The raw map is kept in baseChart;
 *  the user's pole rotation (an exact symmetry of the fit) is applied on top
 *  here. */
let baseChart: Float64Array | null = null;
const chartOf = (): Float64Array => baseChart!;
function applyChart(m: Mesh, S: Float64Array): void {
  baseChart = S;
  const Sr = poleRot ? rotateChart(S, poleRot) : S;
  fitMesh = m;
  charted = { mesh: m, chart: Float32Array.from(Sr) };
  startSampleWorker(m, Sr, elSubdiv.value === 'limit' ? 'limit' : 'pl');
  sphereView = null;      // the map changed; refresh the input's sphere view
  applyInputMorph();
}

// ---------------------------------------------------------------- pole editing
/** Cumulative rotation applied to the raw map (null = identity). */
let poleRot: Float64Array | null = null;
/** While armed: the marked chart direction (in the current rotated map). */
let poleDir: [number, number, number] | null = null;
let poleMoved = false;
let poleDragging = false;

function rotateChart(S: Float64Array, R: Float64Array): Float64Array {
  const out = new Float64Array(S.length);
  for (let i = 0; i < S.length; i += 3) {
    const x = S[i], y = S[i + 1], z = S[i + 2];
    out[i] = R[0] * x + R[1] * y + R[2] * z;
    out[i + 1] = R[3] * x + R[4] * y + R[5] * z;
    out[i + 2] = R[6] * x + R[7] * y + R[8] * z;
  }
  return out;
}

/** The minimal rotation taking unit vector d to +z (Rodrigues). */
function alignToZ(d: [number, number, number]): Float64Array {
  const [x, y, z] = d;
  const s = Math.hypot(x, y);
  if (s < 1e-12) return z > 0 ? Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1) : Float64Array.of(1, 0, 0, 0, -1, 0, 0, 0, -1);
  const kx = y / s, ky = -x / s;   // unit axis d x z (kz = 0)
  const c = z, v = 1 - c;
  return Float64Array.of(
    c + kx * kx * v, kx * ky * v, ky * s,
    kx * ky * v, c + ky * ky * v, -kx * s,
    -ky * s, kx * s, c,
  );
}

const matmul3 = (A: Float64Array, B: Float64Array): Float64Array => {
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      R[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return R;
};

/** Marker at the surface point the current map sends to chart direction `dir`. */
function placePoleMarker(dir: [number, number, number]): void {
  if (!sceneOrig || !mesh || !charted) return;
  if (inputSphereView) {
    let s2 = 0;
    for (let v = 0; v < mesh.nv; v++) s2 += mesh.positions[3 * v] ** 2 + mesh.positions[3 * v + 1] ** 2 + mesh.positions[3 * v + 2] ** 2;
    const r = Math.sqrt(s2 / mesh.nv);
    sceneOrig.setMarker([r * dir[0], r * dir[1], r * dir[2]]);
    return;
  }
  const loc = new SphereLocator(mesh, Float64Array.from(charted.chart));   // throwaway: one query
  const p = loc.interpolate(mesh.positions, 3, Float64Array.from(dir)).values;
  sceneOrig.setMarker([p[0], p[1], p[2]]);
}

/** Pointer drag on the input while armed: marker follows the surface under the
 *  cursor; the chart direction of the hit is remembered for the refit. */
function movePole(e: PointerEvent): void {
  if (!sceneOrig || !mesh || !charted) return;
  const hit = sceneOrig.raycastSurface(e.clientX, e.clientY);
  if (!hit) return;
  if (inputSphereView) {
    const n = Math.hypot(...hit.point) || 1;
    poleDir = [hit.point[0] / n, hit.point[1] / n, hit.point[2] / n];
  } else {
    // corner-duplicated input geometry: rendered triangle k IS mesh face k
    const S = charted.chart, f = mesh.faces, t = hit.faceIndex, [u, v, w] = hit.bary;
    const a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const x = u * S[3 * a] + v * S[3 * b] + w * S[3 * c];
    const y = u * S[3 * a + 1] + v * S[3 * b + 1] + w * S[3 * c + 1];
    const z = u * S[3 * a + 2] + v * S[3 * b + 2] + w * S[3 * c + 2];
    const n = Math.hypot(x, y, z) || 1;
    poleDir = [x / n, y / n, z / n];
  }
  poleMoved = true;
  sceneOrig.setMarker(hit.point);
}

/** Arm/disarm pole editing; disarming applies the rotation and refits once.
 *  While armed, normal drags still orbit — moving the marker takes ctrl+drag
 *  (intercepted in the capture phase so OrbitControls never sees it). */
function setPoleMode(on: boolean): void {
  if (on) {
    if (sampler) {
      status('synthetic shape: its map is exact and never enters the fit — pole editing does not apply (ingest as a point cloud to get a real map)');
      disarmPoleButton(); return;
    }
    if (!mesh || !charted) { disarmPoleButton(); return; }
    poleDir = [0, 0, 1];
    poleMoved = false;
    placePoleMarker(poleDir);
  } else {
    sceneOrig?.setMarker(null);
    if (poleMoved && poleDir && mesh && baseChart) {
      const R = alignToZ(poleDir);
      poleRot = poleRot ? matmul3(R, poleRot) : R;
      applyChart(mesh, baseChart);
      void runFit();
    }
    poleDir = null; poleDragging = false;
  }
}

/** 'auto' display oversampling targets this many render latitudes: the factor
 *  is the smallest power of two (up to 8) that reaches it. An lmax grid already
 *  this fine gains nothing visually and is not oversampled. */
const AUTO_RENDER_NLAT = 512;

/** The display oversampling factor the UI currently asks for. */
function resolveOversample(lmax: number): number {
  if (elPlotOs.value !== 'auto') return Number(elPlotOs.value);
  let os = 1;
  while (os < 8 && os * (lmax + 1) < AUTO_RENDER_NLAT) os *= 2;
  return os;
}

/** The curvature-dealiasing plan: the same spectral band as the display, on a
 *  2x grid — used only when the display grid itself is below 2x (see
 *  updateFitColors; Veerapaneni et al. 2011 Sec 4). */
let curvRenderer: SurfaceRenderer | null = null;
let curvRendererKey = '';
async function ensureCurvRenderer(lmax: number): Promise<SurfaceRenderer> {
  if (!device) throw new Error('WebGPU is not available in this browser');
  const nplot = 2 * (lmax + 1) - 1;
  const key = `${lmax}/${nplot}`;
  if (!curvRenderer || curvRendererKey !== key) {
    curvRenderer?.destroy();
    curvRenderer = await SurfaceRenderer.create(device, lmax, nplot);
    curvRendererKey = key;
  }
  return curvRenderer;
}

/** The display grid: the degree-lmax Gauss grid scaled by the oversampling factor. */
async function ensureRenderer(lmax: number): Promise<SurfaceRenderer> {
  if (!device) throw new Error('WebGPU is not available in this browser');
  const nplot = resolveOversample(lmax) * (lmax + 1) - 1;
  const key = `${lmax}/${nplot}`;
  if (!renderer || rendererKey !== key) {
    renderer?.destroy();
    renderer = await SurfaceRenderer.create(device, lmax, nplot);
    rendererKey = key;
  }
  return renderer;
}

// ---------------------------------------------------------------- fit
/** 'auto' analysis oversampling: 2× normally; 4× for a PL input (facet kinks
 *  decay slowly, so there is heavy content above lmax to alias); 3× for a
 *  rough chart (metric jitter has the same effect). */
function resolveAnalysisOversample(): number {
  if (elOversample.value !== 'auto') return Number(elOversample.value);
  if (!sampler && elSubdiv.value !== 'limit') return 4;
  if (chartInfo && chartInfo.roughnessMean > 0.25) return 3;
  return 2;
}

async function ensureFitter(): Promise<GridFitter> {
  if (!device) throw new Error('WebGPU is not available in this browser');
  const lmax = Number(elLmax.value);
  const os = resolveAnalysisOversample();
  const key = `${lmax}/${os}`;
  if (!fitter || fitterKey !== key) {
    fitter?.destroy();            // safe: all GPU work is serialized through the queue
    fitter = await GridFitter.create(device, lmax, os);
    fitterKey = key;
  }
  return fitter;
}

// All GPU work (fits and filter syntheses) runs through one queue: a plan's
// staging buffers are shared, and a plan may be replaced between tasks.
let gpuQueue: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void>): Promise<void> {
  const run = gpuQueue.then(task, task);
  gpuQueue = run.catch(() => {});
  return run;
}
const runFit = (): Promise<void> => { const gen = geometryGen; return enqueue(() => (gen === geometryGen ? doRunFit() : Promise.resolve())); };
let filterQueued = false;
/** Re-weight and re-synthesize; rapid slider events coalesce into the next queued run. */
function applyFilter(): Promise<void> {
  if (filterQueued) return Promise.resolve();
  filterQueued = true;
  return enqueue(async () => { filterQueued = false; await doApplyFilter(); });
}

async function doRunFit(): Promise<void> {
  if (!mesh) return;
  const gen = geometryGen;
  const stale = () => gen !== geometryGen;   // geometry changed while we were on the GPU
  const chi = eulerCharacteristic(mesh);
  if (chi !== 2) {
    noteFit.textContent = `Euler characteristic ${chi}: not a closed genus-0 surface, cannot be charted onto the sphere.`;
    sceneFit?.dispose(); sceneFit = null;
    updateMetrics(); return;
  }
  if (!sampler && !sampleWorker) {
    setBusy(null);
    noteFit.textContent = 'No map: the mesh could not be parameterized onto the sphere.';
    sceneFit?.dispose(); sceneFit = null;
    coef = null; updateMetrics(); drawSpectrum(); return;
  }
  try {
    const tAll = performance.now();
    if (!sampler) { setBusy('resampling and transforming…'); await new Promise((r) => setTimeout(r, 0)); }
    if (stale()) return;
    const f = await ensureFitter();
    if (stale()) return;
    setNRange(f.cfg.lmax);
    const t0 = performance.now();
    const { nlat, nphi } = f.cfg;
    const n = nlat * nphi;
    // f64 sample fields: the analysis refines against them (GridFitter.analyze)
    const spat: [Float64Array, Float64Array, Float64Array] = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    // sample the surface on the grid: exactly for synthetic shapes, by piecewise-linear
    // interpolation over the charted triangulation for meshes
    gridExact = new Float32Array(n * 3);
    if (sampler) {
      for (let i = 0; i < nlat; i++)
        for (let j = 0; j < nphi; j++) {
          const p = i * nphi + j;
          const [x, y, z] = sampler(f.theta[i], f.phi[j]);
          spat[0][p] = x; spat[1][p] = y; spat[2][p] = z;
          gridExact[3 * p] = x; gridExact[3 * p + 1] = y; gridExact[3 * p + 2] = z;
        }
    } else {
      // Zhou adaptive sampling warps the closed grid mesh (grid nodes plus
      // the pole caps) toward the map's vertex density before sampling; the
      // sampled values then stand as the regular grid signal (f∘w).
      let warp: SampleRequest['warp'];
      if (elSampling.value === 'adaptive') {
        const gtopo = buildTopology(f.plan.cosTheta, f.phi);
        warp = {
          capDirs: Float64Array.from(gtopo.sphereRef.subarray(3 * n)),
          gridFaces: gtopo.indices,
          iters: 30,
        };
      }
      const r = await sampleSurface(f.gridDirections(), warp);
      if (stale() || !r) return;                 // geometry changed mid-sample (worker killed)
      if (r.error !== undefined) throw new Error(r.error);
      lastNewtonResidual = r.maxResidual!;
      const values = r.values!;
      for (let p = 0; p < n; p++) {
        spat[0][p] = values[3 * p]; spat[1][p] = values[3 * p + 1]; spat[2][p] = values[3 * p + 2];
      }
      gridExact.set(values);
    }
    const analyzed = await f.analyze(spat, !(globalThis as Record<string, unknown>).__noRefine);   // debug switch, as __sdemProf
    if (stale()) return;
    coef = analyzed;
    lastFitMs = performance.now() - t0;
    spectrum = powerSpectrum(coef.q, coef.lmax, coef.mmax);

    const r = await ensureRenderer(f.cfg.lmax);
    if (stale()) return;
    topo = buildTopology(r.plan.cosTheta, r.phi);
    gridTheta = Float64Array.from(r.plan.cosTheta, Math.acos);
    posBuf = new Float32Array(topo.numVertices * 3);
    const synthesized = await r.synthesize(coef);
    if (stale()) return;
    coords = synthesized;
    morphRadius = bestFitRadius(coords, topo);
    fillPositions(posBuf, coords, topo, morphT(), morphRadius);
    sceneFit?.dispose();
    sceneFit = newScene(viewFit, topo.numVertices, topo.indices, posBuf);
    if (sceneOrig) sceneFit.syncCamerasWith(sceneOrig);
    noteFit.textContent = '';
    setBusy(null);
    await doApplyFilter();
    // full fit wall time: sampling + analysis + display and filter syntheses
    lastTotalMs = performance.now() - tAll;
    updateMetrics(lastGains ?? undefined);
  } catch (e) {
    setBusy(null);
    status(`fit failed: ${(e as Error).message}`, true);
  }
}

async function doApplyFilter(): Promise<void> {
  if (!coef || !fitter || !renderer || !topo || !posBuf) return;
  const gen = geometryGen;
  const spec = filterSpec();
  const gains = degreeGains(coef.lmax, spec);
  const shown = await renderer.synthesize(coef, gains);
  if (gen !== geometryGen) return;
  const resid = await fitter.synthesize(coef, gains);     // analysis grid: comparable with gridExact
  if (gen !== geometryGen) return;
  coords = shown; residCoords = resid;
  lastGains = gains; derivCache = null;
  gridMetrics = null; kickGridMetrics();
  morphRadius = bestFitRadius(coords, topo);
  fillPositions(posBuf, coords, topo, morphT(), morphRadius);
  sceneFit?.updatePositions(posBuf);
  await updateFitColors();
  updateGridOverlay();
  updateMetrics(gains);
  drawSpectrum(gains);
}

/** Recompute the embeddedness metrics for the current filtered surface in the
 *  worker. Latest-wins: a change arriving while the worker is busy sets the
 *  dirty flag and reruns on completion, so rapid slider events cost one
 *  in-flight computation plus one final rerun. */
function kickGridMetrics(): void {
  if (!fitter || !residCoords || !gridExact) return;
  if (metricsBusy) { metricsDirty = true; return; }
  if (!metricsWorker) {
    metricsWorker = new Worker(new URL('./fit/metrics.worker.ts', import.meta.url), { type: 'module' });
    metricsWorker.onmessage = (e: MessageEvent<GridMetricsResult>) => {
      metricsBusy = false;
      if (metricsDirty) { metricsDirty = false; kickGridMetrics(); return; }
      if (e.data.id !== metricsId) return;   // superseded by a geometry change
      gridMetrics = e.data;
      updateMetrics(lastGains ?? undefined);
    };
  }
  metricsBusy = true;
  const req: GridMetricsRequest = {
    id: ++metricsId,
    cosTheta: Float64Array.from(fitter.plan.cosTheta),
    phi: fitter.phi.slice(),
    coords: residCoords.slice(),
    exact: gridExact.slice(),
  };
  metricsWorker.postMessage(req, [req.cosTheta.buffer, req.phi.buffer, req.coords.buffer, req.exact.buffer]);
}

// ---------------------------------------------------------------- export
let exportCounter = 0;
/** Download the displayed surface's coefficients (current gains applied) as
 *  an HDF5 file in turing-surface's geometry layout (src/fit/exportH5.ts).
 *  h5wasm's wasm is ~4 MB, so it loads on the first click, not with the page. */
async function exportGeometry(): Promise<void> {
  if (!coef || !fitter) return;
  const c = coef;
  elExport.disabled = true;
  try {
    const h5 = (await import('h5wasm')) as unknown as H5Module;
    const gains = lastGains;
    const [X, Y, Z] = c.q.map((q) => (gains ? applyGains(q, c.lmax, c.mmax, gains) : q));
    const bytes = await encodeGeometryH5(h5, {
      lmax: c.lmax, mmax: c.mmax, nlat: fitter.cfg.nlat, nphi: fitter.cfg.nphi,
      X, Y, Z,
      model: meshLabel,
      map: sampler ? 'exact (synthetic)' : chartInfo?.type ?? '',
      sampling: elSampling.value === 'adaptive' ? 'adaptive (Zhou)' : 'uniform',
      filter: filterLabel(filterSpec()),
    }, `/export-${exportCounter++}.h5`);
    const url = URL.createObjectURL(new Blob([bytes.slice()], { type: 'application/x-hdf5' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${meshLabel.replace(/\W+/g, '-')}-lmax${c.lmax}.h5`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    status(`export failed: ${(e as Error).message}`, true);
  } finally {
    elExport.disabled = false;
  }
}

/** Hand the displayed surface to a fresh turing-surface tab as its geometry
 *  (src/fit/exportTuring.ts): same coefficients as the .h5 download, centered
 *  and scaled to rms radius 1. The tab must open synchronously in the click —
 *  popup blockers only allow that — so this is not async itself. */
function exportToTuringSurface(): void {
  if (!coef || !fitter) {
    status('nothing to export yet — wait for a fit to finish', true);
    return;
  }
  const c = coef;
  const base = localStorage.getItem('reharm-turing-surface-url') ?? TURING_SURFACE_URL;
  let url: URL;
  try {
    url = new URL(base);
  } catch {
    status(`export failed: bad turing-surface URL '${base}'`, true);
    return;
  }
  url.searchParams.set('import', 'reharm');
  const win = window.open(url.href, '_blank');
  if (!win) {
    status('export failed: the browser blocked the turing-surface tab', true);
    return;
  }
  elExportTs.disabled = true;
  void (async () => {
    try {
      const gains = lastGains;
      const [X, Y, Z] = c.q.map((q) => (gains ? applyGains(q, c.lmax, c.mmax, gains) : q));
      const norm = normalizeGeometry(X, Y, Z, c.lmax, c.mmax);
      const payload: TuringGeometryPayload = {
        type: 'reharm-geometry',
        version: 1,
        lmax: c.lmax,
        mmax: c.mmax,
        Gx: norm.X,
        Gy: norm.Y,
        Gz: norm.Z,
        name: meshLabel,
        provenance: {
          app: 'reharm',
          model: meshLabel,
          map: sampler ? 'exact (synthetic)' : chartInfo?.type ?? '',
          sampling: elSampling.value === 'adaptive' ? 'adaptive (Zhou)' : 'uniform',
          filter: filterLabel(filterSpec()),
          center: norm.center,
          scale: norm.scale,
        },
      };
      await sendGeometry(win, url.origin, payload);
    } catch (e) {
      status(`export to turing-surface failed: ${(e as Error).message}`, true);
    } finally {
      elExportTs.disabled = false;
    }
  })();
}

// ---------------------------------------------------------------- coloring
/** Robust colormap range: 2–98 percentiles (curvature has pole/sliver outliers),
 *  floored so a numerically-constant field is drawn as uniform. */
function robustRange(values: Float32Array): { lo: number; hi: number } {
  const stride = Math.max(1, Math.floor(values.length / 65536));
  const sample: number[] = [];
  for (let i = 0; i < values.length; i += stride) if (Number.isFinite(values[i])) sample.push(values[i]);
  if (!sample.length) return { lo: 0, hi: 1 };
  sample.sort((a, b) => a - b);
  const q = (t: number) => sample[Math.round(t * (sample.length - 1))];
  return floorRange(q(0.02), q(0.98));
}

/** Spectral derivative fields of the displayed (filtered) surface, for the
 *  curvature coloring; dropped whenever the displayed field changes. */
let derivCache: DerivFields | null = null;
let lastGains: Float64Array | null = null;

/** Color the fitted surface by the selected field of the FULL surface (morphing
 *  carries the colors along, showing where features map on the sphere).
 *  Must run on the serialized GPU queue: curvature synthesizes derivatives. */
async function updateFitColors(): Promise<void> {
  if (!sceneFit || !topo) return;
  const kind = elColor.value;
  elColorbar.hidden = kind === 'solid';
  if (kind === 'solid' || !coords || !gridTheta) {
    sceneFit.updateColors(flatColors(topo.numVertices));
    return;
  }
  let field: Float32Array;
  if (kind === 'radius') {
    const n = coords.length / 3;
    field = new Float32Array(n);
    for (let p = 0; p < n; p++) field[p] = Math.hypot(coords[3 * p], coords[3 * p + 1], coords[3 * p + 2]);
  } else {
    if (!coef || !renderer || !gridTheta) return;
    // Dealiasing (Veerapaneni et al. 2011, Sec 4 / Fig. 2): the curvature of
    // a band-limited surface has roughly TWICE the surface's bandwidth, so
    // its nonlinear pointwise evaluation must happen on a grid of >= 2x the
    // band or the tail folds in. The display grid is that grid under auto
    // oversampling; when the user forces 1x, the derivatives go on a
    // dedicated 2x plan instead and the filtered curvature is restricted to
    // the display grid (both plans carry the same spectral band, so the
    // coefficients transfer directly — the paper's upsample -> differentiate
    // -> filter -> restrict, with the display grid as "original").
    const cr = renderer.cfg.nlat >= 2 * (renderer.cfg.lmax + 1) ? renderer : await ensureCurvRenderer(renderer.cfg.lmax);
    if (!derivCache) {
      const gen = geometryGen;
      const d = await cr.derivatives(coef, lastGains ?? undefined);
      if (gen !== geometryGen || !sceneFit || !topo) return;
      derivCache = d;
    }
    const crTheta = cr === renderer ? gridTheta : Float64Array.from(cr.plan.cosTheta, Math.acos);
    const raw = gridCurvature(derivCache, crTheta, cr.cfg.nphi, kind as 'mean' | 'gauss');
    // filter to the surface's band: the above-band part of H holds both its
    // aliased tail and the derivative-amplified analysis noise
    const gen = geometryGen;
    const clean = Float32Array.from(raw, (v) => (Number.isNaN(v) ? 0 : v));
    const qH = await cr.plan.analys(clean);
    const proj = await renderer.plan.synth(qH);
    if (gen !== geometryGen || !sceneFit || !topo) return;
    if (cr === renderer) {
      field = raw;
      for (let p = 0; p < field.length; p++) if (!Number.isNaN(field[p])) field[p] = proj[p];
    } else {
      field = proj;   // restriction to the display grid
    }
    (globalThis as Record<string, unknown>).__curv = field;   // debug hook for headless measurements
  }
  const values = new Float32Array(topo.numVertices);
  fillFieldValues(values, field, topo);
  let { lo, hi } = robustRange(values);
  if (kind !== 'radius') {
    // fp32 analysis noise, amplified ~lmax^2 by the two derivatives, leaves
    // the curvature of a constant-curvature surface ~0.3% noisy in rings
    // (m=0 pole pileup, see colorbar.ts) — floor the color span so that
    // noise reads as uniform. Real curvature variation is far larger.
    const minSpan = 0.03 * Math.max(Math.abs(lo), Math.abs(hi));
    if (hi - lo < minSpan) { const mid = (lo + hi) / 2; lo = mid - minSpan / 2; hi = mid + minSpan / 2; }
  }
  const cmap = colormaps[elColormap.value];
  const colors = new Float32Array(topo.numVertices * 3);
  fillColors(colors, values, lo, hi, cmap);
  sceneFit.updateColors(colors);
  colorbar?.update(cmap, lo, hi);
}

/** Re-render on a new display grid (no analysis needed). */
const rebuildDisplay = (): Promise<void> => { const gen = geometryGen; return enqueue(async () => {
  if (gen !== geometryGen || !coef || !fitter) return;
  const r = await ensureRenderer(fitter.cfg.lmax);
  if (gen !== geometryGen) return;
  topo = buildTopology(r.plan.cosTheta, r.phi);
  gridTheta = Float64Array.from(r.plan.cosTheta, Math.acos);
  posBuf = new Float32Array(topo.numVertices * 3);
  coords = await r.synthesize(coef);
  if (gen !== geometryGen) return;
  morphRadius = bestFitRadius(coords, topo);
  fillPositions(posBuf, coords, topo, morphT(), morphRadius);
  sceneFit?.dispose();
  sceneFit = newScene(viewFit, topo.numVertices, topo.indices, posBuf);
  if (sceneOrig) sceneFit.syncCamerasWith(sceneOrig);
  await doApplyFilter();
}); };

function applyMorph(): void {
  if (!coords || !topo || !posBuf) return;
  fillPositions(posBuf, coords, topo, morphT(), morphRadius);
  sceneFit?.updatePositions(posBuf);
  applyGridPositions();
}

// ------------------------------------------------- PDE collocation grid overlay
/**
 * Rings and meridians of the EXACT degree-lmax Gauss grid (gridForLmax(lmax,1)
 * — the collocation grid a PDE solve at this lmax uses), drawn as curves ON
 * the displayed surface: each line is sampled at the display grid's
 * resolution by linear interpolation of the blended display positions, so its
 * segments hug the rendered surface and the fill's polygonOffset resolves
 * visibility. (A single chord per collocation cell sags kappa h^2/8 below the
 * surface and gets clipped — no uniform lift can track that.) Gauss latitudes
 * do not nest, hence the theta interpolation; the longitudes are uniform in
 * both grids, so meridians land on (or between) display columns. CPU-only:
 * positions follow the filter and the morph through posBuf.
 */
function updateGridOverlay(): void {
  if (!sceneFit) return;
  if (!wireOut() || !coef || !topo || !gridTheta || !posBuf) { sceneFit.setGridLines(null, null); return; }
  const { nlat: nlatP, nphi: nphiP } = gridForLmax(coef.lmax, 1);
  const key = `${coef.lmax}/${topo.nlat}x${topo.nphi}`;
  if (gridLinesKey !== key || !gridLinesIndices) {
    gridLinesKey = key;
    gridLinesTheta = Float64Array.from(gaussNodesWeights(nlatP).x, Math.acos).sort();
    gridLinesNphi = nphiP;
    // vertex layout: nlatP rings of nphiD points, then nphiP meridians of nlatD points
    const nlatD = topo.nlat, nphiD = topo.nphi;
    const segs = new Uint32Array(2 * (nlatP * nphiD + nphiP * (nlatD - 1)));
    let k = 0;
    for (let r = 0; r < nlatP; r++)
      for (let j = 0; j < nphiD; j++) { const b = r * nphiD; segs[k++] = b + j; segs[k++] = b + (j + 1) % nphiD; }
    const mb = nlatP * nphiD;
    for (let m = 0; m < nphiP; m++)
      for (let i = 0; i + 1 < nlatD; i++) { const b = mb + m * nlatD; segs[k++] = b + i; segs[k++] = b + i + 1; }
    gridLinesIndices = segs;
  }
  applyGridPositions();
}

/** Re-sample the overlay's positions from the blended display grid (cheap, CPU). */
function applyGridPositions(): void {
  if (!sceneFit || !wireOut() || !gridLinesIndices || !gridLinesTheta || !topo || !posBuf || !gridTheta) return;
  const nlatD = topo.nlat, nphiD = topo.nphi, nlatP = gridLinesTheta.length, nphiP = gridLinesNphi;
  const pos = new Float32Array(3 * (nlatP * nphiD + nphiP * nlatD));
  // rings: interpolate between the two display rows bracketing each PDE latitude
  let a = 0;
  for (let r = 0; r < nlatP; r++) {
    const t = gridLinesTheta[r];
    while (a < nlatD - 2 && gridTheta[a + 1] <= t) a++;
    const b = a + 1;
    const w = Math.min(1, Math.max(0, (t - gridTheta[a]) / (gridTheta[b] - gridTheta[a])));
    for (let j = 0; j < nphiD; j++) {
      const pa = 3 * (a * nphiD + j), pb = 3 * (b * nphiD + j), q = 3 * (r * nphiD + j);
      pos[q] = (1 - w) * posBuf[pa] + w * posBuf[pb];
      pos[q + 1] = (1 - w) * posBuf[pa + 1] + w * posBuf[pb + 1];
      pos[q + 2] = (1 - w) * posBuf[pa + 2] + w * posBuf[pb + 2];
    }
  }
  // meridians: interpolate between the two display columns bracketing each PDE longitude
  const mb = nlatP * nphiD;
  for (let m = 0; m < nphiP; m++) {
    const c = (m * nphiD) / nphiP;
    const c0 = Math.floor(c) % nphiD, c1 = (c0 + 1) % nphiD, cw = c - Math.floor(c);
    for (let i = 0; i < nlatD; i++) {
      const p0 = 3 * (i * nphiD + c0), p1 = 3 * (i * nphiD + c1), q = 3 * (mb + m * nlatD + i);
      pos[q] = (1 - cw) * posBuf[p0] + cw * posBuf[p1];
      pos[q + 1] = (1 - cw) * posBuf[p0 + 1] + cw * posBuf[p1 + 1];
      pos[q + 2] = (1 - cw) * posBuf[p0 + 2] + cw * posBuf[p1 + 2];
    }
  }
  sceneFit.setGridLines(pos, gridLinesIndices);
}

// ---------------------------------------------------------------- readouts
function updateMetrics(gains?: Float64Array): void {
  // One fact per line behind an aligned label column. Labels keep their colons:
  // the headless scripts key on 'map:', 'filter:', and 'transform: lmax'.
  const lines: string[] = [];
  const row = (label: string, value: string) => lines.push(label.padEnd(12) + value);
  const num = (n: number) => n.toLocaleString('en-US');
  if (chartInfo) {
    const type = chartInfo.type.replace(/\s*\(.*\)$/, '');   // the pipeline parenthetical is detail
    const folds = chartInfo.foldsRepaired || chartInfo.foldsLeft
      ? ` · ${chartInfo.foldsRepaired} fold${chartInfo.foldsRepaired === 1 ? '' : 's'} repaired${chartInfo.foldsLeft ? ` (${chartInfo.foldsLeft} LEFT)` : ''}`
      : '';
    row('map:', `${type}${folds}`);
    if (chartInfo.sdemSteps !== undefined) row('  SDEM:', `${chartInfo.sdemSteps} steps (${chartInfo.sdemStopped}) · spread ${chartInfo.sdemSpread?.toFixed(3)}`);
    if (chartInfo.balancedIters !== undefined) row('  balance:', `${chartInfo.balancedIters} iterations (alpha 0.5)`);
    const ratio = chartInfo.lambdaRatio;
    row('  λ:', `std/mean ${chartInfo.lambdaSpread.toFixed(3)} · max/min ${ratio >= 1e4 ? ratio.toExponential(1) : String(Number(ratio.toPrecision(2)))}${chartInfo.crowded && !chartInfo.type.startsWith('area') && !chartInfo.type.startsWith('balanced') ? '  — crowded: try the area-equalized or balanced map' : ''}`);
    row('  rough:', `mean ${chartInfo.roughnessMean.toFixed(3)} · p99 ${chartInfo.roughnessP99.toFixed(2)}   (conformal ≈ 0.03–0.13)`);
  }
  if (coef && fitter) {
    const { lmax, nlat, nphi } = fitter.cfg;
    const aos = Math.round(nlat / (lmax + 1));
    row('transform:', `analysis grid ${nlat}×${nphi}${aos > 1 ? ` (${aos}×)` : ''} · ${num((lmax + 1) ** 2)} coeffs per coordinate`);
    if (gains) {
      let kept = 0, energy = 0, total = 0;
      for (let l = 0; l <= lmax; l++) {
        kept += (2 * l + 1) * gains[l] * gains[l];
        if (spectrum) { energy += gains[l] * gains[l] * spectrum[l]; total += spectrum[l]; }
      }
      row('filter:', `gain at lmax ${(100 * gains[lmax]).toPrecision(2)} % · coeffs kept ${(100 * kept / (lmax + 1) ** 2).toFixed(1)} % · energy ${(100 * energy / (total || 1)).toFixed(2)} %`);
    }
    if (gridExact && residCoords && mesh) {
      const rc = residCoords;
      const diag = bboxDiagonal(mesh);
      let s2 = 0, mx = 0;
      const n = gridExact.length / 3;
      for (let p = 0; p < n; p++) {
        const d = Math.hypot(rc[3 * p] - gridExact[3 * p], rc[3 * p + 1] - gridExact[3 * p + 1], rc[3 * p + 2] - gridExact[3 * p + 2]);
        s2 += d * d; if (d > mx) mx = d;
      }
      const pct = (v: number) => `${(100 * v / diag).toPrecision(3)} %`;
      row('residual:', `rms ${pct(Math.sqrt(s2 / n))} · max ${pct(mx)}`);
    }
    if (gridExact && residCoords) {
      const g = gridMetrics;
      row('output:', g
        ? `${num(g.selfIntersections)} self-intersection${g.selfIntersections === 1 ? '' : 's'} · ${num(g.collapsed)} collapsed face${g.collapsed === 1 ? '' : 's'}`
        : 'checking…');
    }
  }
  elMetrics.textContent = lines.join('\n');
  // the live summary line above the panes; errors overwrite it until the next
  // successful update
  const parts: string[] = [];
  if (mesh) parts.push(mesh.nf === 0 ? `${num(mesh.nv)} points` : `${num(mesh.nv)} vertices · ${num(mesh.nf)} faces`);
  if (coef && fitter && lastTotalMs) {
    const fmt = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);
    parts.push(`total time ${fmt((chartInfo?.timeMs ?? 0) + lastTotalMs)}`);
  }
  if (parts.length) { elStatus.textContent = parts.join(' · '); elStatus.classList.remove('err'); }
}

function drawSpectrum(gains?: Float64Array): void {
  const ctx = elSpectrum.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const W = elSpectrum.clientWidth, H = elSpectrum.clientHeight;
  if (elSpectrum.width !== W * dpr || elSpectrum.height !== H * dpr) { elSpectrum.width = W * dpr; elSpectrum.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink-2').trim() || '#888';
  ctx.font = '11px system-ui'; ctx.fillStyle = ink;
  if (!spectrum) return;   // the box's 'spectrum' caption is enough of a placeholder
  // fixed axis, 1 down to 1e-8 with a tick every two decades; the top pad
  // leaves room for the box's 'spectrum' caption
  const L = spectrum.length, pad = { l: 44, r: 10, t: 44, b: 22 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const hi = 0, lo = -8;
  const x = (l: number) => pad.l + (pw * (l + 0.5)) / L;
  const y = (v: number) => pad.t + ph * (1 - (v - lo) / (hi - lo));
  const clamp = (v: number) => Math.min(Math.max(v, lo), hi);
  ctx.strokeStyle = ink; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph); ctx.stroke();
  const SUP: Record<string, string> = { '-': '\u207b', '0': '\u2070', '2': '\u00b2', '4': '\u2074', '6': '\u2076', '8': '\u2078' };
  const pow10 = (v: number): string => '10' + String(v).split('').map((c) => SUP[c]).join('');
  ctx.textAlign = 'right';
  for (let v = hi; v >= lo; v -= 2) { ctx.fillText(pow10(v), pad.l - 8, y(v) + 4); }
  ctx.textAlign = 'left';
  ctx.fillText('degree l', pad.l + pw / 2 - 20, H - 6);
  ctx.fillText(String(L - 1), pad.l + pw - 14, H - 6); ctx.fillText('0', pad.l, H - 6);
  // Bars are the FILTERED power gain^2 * S_l; a faint ghost above each bar is
  // the part of the raw spectrum the filter removed (gain <= 1, so the ghost
  // always extends the bar upward). A zero gain leaves a zero-height bar.
  ctx.fillStyle = ink;
  const bw = Math.max(1, pw / L - 1);
  // relative power: normalized by the largest raw bar, so the axis top (1) is
  // the true maximum and the scale does not rescale as the filter moves
  let smax = 0;
  for (let l = 0; l < L; l++) if (spectrum[l] > smax) smax = spectrum[l];
  smax = smax || 1;
  for (let l = 0; l < L; l++) {
    const raw = Math.log10(Math.max(spectrum[l] / smax, 1e-30));
    const yRaw = y(clamp(raw));
    if (gains) {
      ctx.globalAlpha = 0.3;
      ctx.fillRect(x(l) - bw / 2, yRaw, bw, pad.t + ph - yRaw);
      ctx.globalAlpha = 1;
      const v = Math.log10(Math.max(gains[l] * gains[l] * spectrum[l] / smax, 1e-30));
      const yy = y(clamp(v));
      ctx.fillRect(x(l) - bw / 2, yy, bw, pad.t + ph - yy);
    } else {
      ctx.fillRect(x(l) - bw / 2, yRaw, bw, pad.t + ph - yRaw);
    }
  }
}

// ---------------------------------------------------------------- wiring
async function initGPU(): Promise<void> {
  if (!('gpu' in navigator)) { status('WebGPU is not available in this browser; meshes can be viewed but not fitted.', true); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { status('No WebGPU adapter found.', true); return; }
  device = await adapter.requestDevice();
  device.lost.then((info) => status(`WebGPU device lost: ${info.message}`, true));
}

function populateGeometries(): void {
  // synthetic shapes (syntheticShapes) are kept in code for tests but not
  // offered in the UI
  for (const p of presets) {
    const o = document.createElement('option'); o.value = p.key; o.textContent = p.label; elGeometry.appendChild(o);
  }
  // the exact unit sphere, back in the UI as the curvature-coloring test case
  // (H = K = 1: any structure in the colors is pipeline noise, not geometry)
  const sph = document.createElement('option');
  sph.value = 'synthetic:sphere'; sph.textContent = 'sphere';
  elGeometry.appendChild(sph);
  const custom = document.createElement('option');
  custom.value = 'custom'; custom.textContent = 'custom…';
  custom.title = 'Load your own mesh: OBJ, OFF, PLY, STL, or the preset SHM1 binary (.mesh). Must be a closed genus-0 triangle mesh.';
  elGeometry.appendChild(custom);
}

async function main(): Promise<void> {
  installResize();
  syncFilterControls();
  for (const name of colormapNames) {
    const o = document.createElement('option'); o.value = name; o.textContent = name; elColormap.appendChild(o);
  }
  elColormap.value = 'jet';
  colorbar = new Colorbar(elColorbar);
  try { presets = await loadManifest(); } catch (e) { status(`presets unavailable: ${(e as Error).message}`, true); }
  populateGeometries();
  await initGPU();

  // The model dropdown's custom… entry opens the file dialog. Cancelling (or a
  // bad file) snaps the selection back to what it was, so the dropdown never
  // shows custom… as a state — only real geometries. showOpenFilePicker names
  // the formats in the OS dialog's file-type dropdown (behind 'Show Options'
  // on macOS; the accept attribute alone only greys files out); browsers
  // without it get the plain file input instead.
  interface FilePickerWindow {
    showOpenFilePicker?: (opts: {
      types: { description: string; accept: Record<string, string[]> }[];
    }) => Promise<{ getFile(): Promise<File> }[]>;
  }
  const picker = (window as FilePickerWindow).showOpenFilePicker;
  const revert = () => { elGeometry.value = lastGeometryKey; };
  const pickCustomMesh = async (): Promise<void> => {
    if (!picker) { elUploadFallback.click(); return; }   // fallback reverts via its own events
    try {
      const [h] = await picker({
        types: [{
          description: 'Surface meshes (.obj, .off, .ply, .stl, .mesh)',
          accept: { 'application/octet-stream': ['.obj', '.off', '.ply', '.stl', '.mesh'] },
        }],
      });
      await uploadGeometry(await h.getFile());
    } catch (e) {
      revert();
      if ((e as Error).name !== 'AbortError') status(String(e), true);   // cancel is not an error
    }
  };
  elGeometry.addEventListener('change', () => {
    const key = elGeometry.value;
    if (key === 'custom') { void pickCustomMesh(); return; }
    lastGeometryKey = key;
    void selectGeometry(key).catch((e) => status(String(e), true));
  });
  elUploadFallback.addEventListener('change', () => {
    const f = elUploadFallback.files?.[0];
    if (f) void uploadGeometry(f).catch((e) => { revert(); status(String(e), true); });
    else revert();
  });
  elUploadFallback.addEventListener('cancel', revert);
  elChart.addEventListener('change', () => void (async () => { await computeChart(); await runFit(); })());
  elSdemSteps.addEventListener('change', () => {
    // a chart in flight takes the new cap live (SDEM re-reads it every step);
    // otherwise recompute as for any other chart parameter
    if (chartWorker) { chartWorker.postMessage({ id: chartRequestId, sdemMaxSteps: Number(elSdemSteps.value) }); return; }
    if (chartInfo?.sdemSteps !== undefined || elChart.value === 'area' || elChart.value === 'balanced') void (async () => { await computeChart(); await runFit(); })();
  });
  elChartSmooth.addEventListener('change', () => { if (chartInfo?.sdemSteps !== undefined || elChart.value === 'area' || elChart.value === 'balanced') void (async () => { await computeChart(); await runFit(); })(); });
  elSubdiv.addEventListener('change', () => { if (mesh && charted && !sampler) { applyChart(mesh, Float64Array.from(chartOf())); void runFit(); } });
  elLmax.addEventListener('change', () => { setSigmaControls(defaultSigma(Number(elLmax.value))); void runFit(); });
  elPlotOs.addEventListener('change', () => void rebuildDisplay());
  elColor.addEventListener('change', () => void enqueue(updateFitColors));
  elColormap.addEventListener('change', () => void enqueue(updateFitColors));
  elWire.addEventListener('change', () => {
    // input pane: in the sphere view the wireframe is the curved original-edge
    // overlay, not the fill geometry's (subdivided) edges
    if (inputSphereView && sphereView) sceneOrig?.setGridLines(wireIn() ? sphereView.linePos : null, wireIn() ? sphereView.lineIdx : null);
    else sceneOrig?.setWireframe(wireIn() && !pointsActive());   // no mesh edges over the points view
    updateGridOverlay();   // output pane
  });
  elPole.addEventListener('click', () => {
    poleArmed = !poleArmed;
    elPole.classList.toggle('armed', poleArmed);
    setPoleMode(poleArmed);
  });
  // capture phase: a ctrl+drag must not also reach OrbitControls on the canvas
  viewOrig.addEventListener('pointerdown', (e) => {
    if (!poleArmed || !e.ctrlKey) return;
    e.stopPropagation(); e.preventDefault();
    poleDragging = true; movePole(e);
  }, { capture: true });
  window.addEventListener('pointermove', (e) => { if (poleDragging) movePole(e); });
  window.addEventListener('pointerup', () => { poleDragging = false; });
  setSigmaControls(defaultSigma(Number(elLmax.value)));
  elOversample.addEventListener('change', () => void runFit());
  elSampling.addEventListener('change', () => void runFit());
  elInputForm.addEventListener('change', () => {
    const key = elGeometry.value;
    if (key && key !== 'custom') { lastGeometryKey = key; void selectGeometry(key).catch((e) => status(String(e), true)); }
  });
  elFilter.addEventListener('change', () => { syncFilterControls(); void applyFilter(); });
  elN.addEventListener('input', () => { elNnum.value = elN.value; void applyFilter(); });
  elNnum.addEventListener('change', () => { elN.value = elNnum.value; void applyFilter(); });
  elRamp.addEventListener('input', () => void applyFilter());
  elSigmaSlider.addEventListener('input', () => {
    elSigma.value = String(Number(Math.pow(10, Number(elSigmaSlider.value)).toPrecision(2)));
    void applyFilter();
  });
  elSigma.addEventListener('input', () => {
    // move the slider only — rewriting the box would fight the user's typing
    const v = Number(elSigma.value);
    if (Number.isFinite(v) && v > 0) elSigmaSlider.value = String(Math.log10(Math.min(Math.max(v, 1e-3), 1)));
    void applyFilter();
  });
  elMorph.addEventListener('change', () => { applyMorph(); applyInputMorph(); applyPointsView(); });
  elShowPoints.addEventListener('change', applyPointsView);
  // Reset one pane and copy its pose to the other, so the panes stay in lockstep
  // even when their geometries frame slightly differently.
  elExport.addEventListener('click', () => void exportGeometry());
  elExportTs.addEventListener('click', () => exportToTuringSurface());
  $('resetview').addEventListener('click', () => {
    const lead = sceneOrig ?? sceneFit;
    lead?.resetCamera();
    if (lead && sceneFit && lead !== sceneFit) sceneFit.setCameraState(lead.cameraState());
  });
  // Cancel must not dead-end (there is no fit button): fall back to the fast
  // conformal chart, visibly — the dropdown reflects it.
  $('cancel-chart').addEventListener('click', () => {
    if (!chartWorker) return;
    chartWorker.terminate(); chartWorker = null; chartRequestId++;
    setBusy(null);
    status('map cancelled — falling back to conformal + Möbius');
    elChart.value = 'conformal';
    void (async () => { await computeChart(); await runFit(); })();
  });
  window.addEventListener('resize', () => drawSpectrum());

  if (presets.length) { lastGeometryKey = presets[0].key; await selectGeometry(presets[0].key); }
}

void main();
