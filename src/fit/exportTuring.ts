/**
 * "Export to turing-surface": hand the fitted surface to a live turing-surface
 * tab, no file in between.
 *
 * The payload is the same three coefficient arrays the .h5 export writes
 * (filter gains applied — the surface on screen), normalized so the surface
 * lands at turing-surface's scale: its shapes are all O(1) around the origin
 * and its model parameters are tuned there, and the Laplace-Beltrami operator
 * scales as 1/length^2, so an un-normalized surface would silently change
 * every pattern's wavelength. Centering is zeroing the (0,0) coefficient
 * (mean position = q_00 * Y_00 with orthonormal harmonics); the rms radius is
 * Parseval on what remains, with weight 2 on m > 0 for the implied
 * negative-m partners of the m >= 0 storage convention.
 *
 * Transport: the two apps are on different origins, so the channel is
 * window.open + postMessage — structured clone carries the Float32Arrays
 * without encoding. The protocol (mirrored in turing-surface's src/main.ts
 * import block): ping {type:'reharm-import-ping'} until the opened tab
 * answers {type:'reharm-import-ready'} — it does so only once its boot has a
 * live session to swap the surface into — then post the geometry once and
 * wait for {type:'reharm-import-ack'} or an error.
 */

/** Where the button points. Overridable for local testing via
 *  localStorage['reharm-turing-surface-url']. */
export const TURING_SURFACE_URL = 'https://concept-collection.github.io/turing-surface/';

export interface TuringGeometryPayload {
  type: 'reharm-geometry';
  version: 1;
  lmax: number;
  mmax: number;
  /** Normalized coefficients of x, y, z, 2*nlm each, shared SHTns layout. */
  Gx: Float32Array;
  Gy: Float32Array;
  Gz: Float32Array;
  /** Short label for the dropdown entry over there. */
  name: string;
  provenance: {
    app: 'reharm';
    model: string;
    map: string;
    sampling: string;
    filter: string;
    /** Mean position subtracted and rms radius divided out, so the original
     *  surface is recoverable: x_orig = scale * x + center. */
    center: [number, number, number];
    scale: number;
  };
}

export interface NormalizedGeometry {
  X: Float32Array;
  Y: Float32Array;
  Z: Float32Array;
  center: [number, number, number];
  scale: number;
}

const Y00 = 1 / (2 * Math.sqrt(Math.PI));

/** Center the surface and scale it to rms radius 1. The inputs are copied,
 *  never written — they are the fit's own coefficients. */
export function normalizeGeometry(
  X: Float32Array,
  Y: Float32Array,
  Z: Float32Array,
  lmax: number,
  mmax: number,
): NormalizedGeometry {
  const qs = [X.slice(), Y.slice(), Z.slice()];
  const center = qs.map((q) => {
    const mean = q[0] * Y00;
    q[0] = 0;
    q[1] = 0;
    return mean;
  }) as [number, number, number];
  // Parseval over the centered coefficients: mean |x|^2 over the sphere is
  // sum_lm w_m |q_lm|^2 / 4pi, w_0 = 1, w_m>0 = 2. The m-major loop walks the
  // storage order, so idx just increments.
  let sum = 0;
  for (const q of qs) {
    let idx = 0;
    for (let m = 0; m <= mmax; m++) {
      const w = m === 0 ? 1 : 2;
      for (let l = m; l <= lmax; l++, idx++) {
        const re = q[2 * idx];
        const im = q[2 * idx + 1];
        sum += w * (re * re + im * im);
      }
    }
  }
  const scale = Math.sqrt(sum / (4 * Math.PI));
  if (!(scale > 0)) throw new Error('degenerate surface: zero rms radius');
  const inv = 1 / scale;
  for (const q of qs) for (let i = 0; i < q.length; i++) q[i] *= inv;
  return { X: qs[0], Y: qs[1], Z: qs[2], center, scale };
}

/**
 * Run the handshake against an already-opened tab and deliver the payload.
 * The caller opens the window synchronously in the click handler — popup
 * blockers only allow that — and this settles when the tab acks, refuses, or
 * the timeout passes (boot over there includes a WebGPU model compile, so the
 * default is generous).
 */
export function sendGeometry(
  win: Window,
  targetOrigin: string,
  payload: TuringGeometryPayload,
  timeoutMs = 120_000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let sent = false;
    let done = false;
    const finish = (err?: Error): void => {
      if (done) return;
      done = true;
      clearInterval(ping);
      clearTimeout(timer);
      window.removeEventListener('message', onMessage);
      if (err) reject(err);
      else resolve();
    };
    const onMessage = (ev: MessageEvent): void => {
      if (ev.source !== win) return;
      const d = ev.data as Record<string, unknown> | null;
      if (!d || typeof d !== 'object') return;
      // 'ready' can arrive twice — once unsolicited at the tab's boot, once
      // as a ping reply — so the payload is posted exactly once.
      if (d.type === 'reharm-import-ready' && !sent) {
        sent = true;
        win.postMessage(payload, targetOrigin);
      } else if (d.type === 'reharm-import-ack') {
        finish();
      } else if (d.type === 'reharm-import-error') {
        finish(new Error(String(d.message ?? 'import refused')));
      }
    };
    window.addEventListener('message', onMessage);
    const ping = setInterval(() => {
      if (win.closed) {
        finish(new Error('the turing-surface tab was closed'));
        return;
      }
      win.postMessage({ type: 'reharm-import-ping' }, targetOrigin);
    }, 300);
    const timer = setTimeout(
      () =>
        finish(
          new Error(
            'turing-surface did not respond — the site there may not support imports yet',
          ),
        ),
      timeoutMs,
    );
  });
}
