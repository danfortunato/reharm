import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Three.js scene wrapper: a single indexed triangle mesh with dynamic
 * per-vertex positions and colors, orbit controls, and optional camera
 * synchronization with sibling scenes. The topology is fixed by the grid; the
 * positions are the surface, so they change when the geometry or the morph
 * does, and the colors every frame.
 *
 * Rendering is on demand: the animation loop ticks every frame (it has to,
 * to drive OrbitControls damping), but only re-renders when the colors,
 * camera, or canvas size actually changed.
 *
 * Adapted from figpack's SphereEmbedding view (figpack_experimental).
 */
export class SphereScene {
  #scene: THREE.Scene;
  #camera: THREE.PerspectiveCamera;
  #renderer: THREE.WebGLRenderer;
  #controls: OrbitControls;
  #geometry: THREE.BufferGeometry;
  #mesh: THREE.Mesh;
  #wire: THREE.LineSegments | null = null;
  #points: THREE.Points | null = null;
  #grid: THREE.LineSegments | null = null;
  #marker: THREE.Mesh | null = null;
  #animationId: number | null = null;
  #defaultCameraState: {
    position: THREE.Vector3;
    target: THREE.Vector3;
  } | null = null;
  #syncing = false;
  #needsRender = true;
  #lastW = -1;
  #lastH = -1;

  constructor(
    container: HTMLElement,
    numVertices: number,
    indices: Uint32Array,
    positions: Float32Array,
    background = '#14161c',
  ) {
    this.#scene = new THREE.Scene();
    this.#scene.background = new THREE.Color(background);

    this.#camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);

    this.#renderer = new THREE.WebGLRenderer({ antialias: true });
    this.#renderer.setPixelRatio(window.devicePixelRatio || 1);
    // The canvas always fills its container via CSS; resize() then only
    // updates the drawing buffer
    this.#renderer.domElement.style.width = '100%';
    this.#renderer.domElement.style.height = '100%';
    this.#renderer.domElement.style.display = 'block';
    container.appendChild(this.#renderer.domElement);

    // Lighting: a hemisphere for soft ambient with a vertical gradient, plus
    // a key/fill pair attached to the camera (key well off the viewing axis,
    // MATLAB camlight-style) so the form reads as 3D from any orbit angle. A
    // flat ambient + headlight washes the shading out: the gradient the eye
    // needs comes from light arriving obliquely.
    this.#scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(1.7, 1.4, 1.0);
    this.#camera.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-1.5, -0.4, 0.6);
    this.#camera.add(fill);
    this.#scene.add(this.#camera);

    this.#geometry = new THREE.BufferGeometry();
    // Positions move with the morph slider, so they are dynamic too.
    const positionAttr = new THREE.BufferAttribute(positions, 3);
    positionAttr.setUsage(THREE.DynamicDrawUsage);
    const colorAttr = new THREE.BufferAttribute(
      new Float32Array(numVertices * 3),
      3,
    );
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    this.#geometry.setAttribute('position', positionAttr);
    this.#geometry.setAttribute('color', colorAttr);
    this.#geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.#geometry.computeVertexNormals();
    this.#geometry.computeBoundingSphere();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 55,
      specular: new THREE.Color(0x606060),
      // pushed back slightly so a wireframe overlay does not z-fight
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    this.#mesh = new THREE.Mesh(this.#geometry, material);
    this.#scene.add(this.#mesh);

    this.#controls = new OrbitControls(this.#camera, this.#renderer.domElement);
    this.#controls.enableDamping = true;
    this.#controls.dampingFactor = 0.1;
    // Fires on user input and on every damping-tail update, so the flag stays
    // set until the camera has fully settled.
    this.#controls.addEventListener('change', () => {
      this.#needsRender = true;
    });

    this.#animate();
  }

  #animate = () => {
    this.#animationId = requestAnimationFrame(this.#animate);
    this.#controls.update();
    if (!this.#needsRender) return;
    this.#needsRender = false;
    this.#renderer.render(this.#scene, this.#camera);
  };

  updateColors(colors: Float32Array): void {
    const attr = this.#geometry.getAttribute('color') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(colors);
    attr.needsUpdate = true;
    this.#needsRender = true;
  }

  /**
   * Move the vertices — for the sphere/surface morph. Normals have to be
   * recomputed with them or the shading stays that of the old shape, which is
   * the whole thing the eye reads a curved surface by.
   */
  updatePositions(positions: Float32Array): void {
    const attr = this.#geometry.getAttribute('position') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(positions);
    attr.needsUpdate = true;
    this.#geometry.computeVertexNormals();
    this.#geometry.computeBoundingSphere();
    this.#needsRender = true;
  }

  /**
   * Show/hide a wireframe overlay of the mesh's triangle edges. The line
   * geometry shares the mesh's position attribute, so it follows
   * updatePositions() for free; it is built lazily on first use.
   */
  setWireframe(on: boolean): void {
    if (on && !this.#wire) {
      const idx = this.#geometry.getIndex()!;
      const edges = new Uint32Array(idx.count * 2);
      for (let t = 0; t < idx.count; t += 3) {
        const a = idx.getX(t), b = idx.getX(t + 1), c = idx.getX(t + 2);
        edges.set([a, b, b, c, c, a], 2 * t);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', this.#geometry.getAttribute('position'));
      g.setIndex(new THREE.BufferAttribute(edges, 1));
      this.#wire = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x30363d }));
      this.#scene.add(this.#wire);
    }
    if (this.#wire) this.#wire.visible = on;
    this.#needsRender = true;
  }

  /** Hide/show the surface fill (the points view swaps it for the cloud). */
  setMeshVisible(on: boolean): void {
    this.#mesh.visible = on;
    this.#needsRender = true;
  }

  /**
   * Show the vertices as a point cloud (the honest view of a face-less
   * input), or hide it with null. Unlit: points have no normals to shade.
   */
  setPoints(positions: Float32Array | null): void {
    if (this.#points) {
      this.#scene.remove(this.#points);
      this.#points.geometry.dispose();
      (this.#points.material as THREE.Material).dispose();
      this.#points = null;
    }
    if (positions) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      g.computeBoundingSphere();
      const size = (g.boundingSphere?.radius ?? 1) * 0.012;
      this.#points = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8892a8, size, sizeAttenuation: true }));
      this.#scene.add(this.#points);
    }
    this.#needsRender = true;
  }

  /**
   * Show a line overlay with caller-supplied positions and segment indices
   * (e.g. a collocation grid drawn on the surface), or hide it with null.
   * Positions may be updated per-call (morph); the geometry is rebuilt only
   * when the sizes change.
   */
  setGridLines(positions: Float32Array | null, indices: Uint32Array | null): void {
    if (!positions || !indices) {
      if (this.#grid) this.#grid.visible = false;
      this.#needsRender = true;
      return;
    }
    const same = this.#grid
      && (this.#grid.geometry.getAttribute('position') as THREE.BufferAttribute).count === positions.length / 3
      && this.#grid.geometry.getIndex()!.count === indices.length;
    if (!same) {
      if (this.#grid) {
        this.#scene.remove(this.#grid);
        this.#grid.geometry.dispose();
        (this.#grid.material as THREE.Material).dispose();
      }
      const g = new THREE.BufferGeometry();
      const attr = new THREE.BufferAttribute(positions, 3);
      attr.setUsage(THREE.DynamicDrawUsage);
      g.setAttribute('position', attr);
      g.setIndex(new THREE.BufferAttribute(indices, 1));
      this.#grid = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x30363d }));
      this.#scene.add(this.#grid);
    } else {
      const attr = this.#grid!.geometry.getAttribute('position') as THREE.BufferAttribute;
      (attr.array as Float32Array).set(positions);
      attr.needsUpdate = true;
    }
    this.#grid!.visible = true;
    this.#needsRender = true;
  }

  /** Enable/disable the orbit controls (e.g. while a gizmo drag owns the pointer). */
  set controlsEnabled(v: boolean) {
    this.#controls.enabled = v;
  }

  /** Show a small marker sphere at a position, or hide it with null. */
  setMarker(p: [number, number, number] | null): void {
    if (!p) {
      if (this.#marker) this.#marker.visible = false;
      this.#needsRender = true;
      return;
    }
    if (!this.#marker) {
      const r = (this.#geometry.boundingSphere?.radius ?? 1) * 0.025;
      this.#marker = new THREE.Mesh(
        new THREE.SphereGeometry(r, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xe0483e }),
      );
      this.#scene.add(this.#marker);
    }
    this.#marker.position.set(p[0], p[1], p[2]);
    this.#marker.visible = true;
    this.#needsRender = true;
  }

  /** Raycast a pointer event's position against the surface mesh: the hit
   *  point, the triangle index, and its barycentric coordinates. */
  raycastSurface(clientX: number, clientY: number): { point: [number, number, number]; faceIndex: number; bary: [number, number, number] } | null {
    const rect = this.#renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -(((clientY - rect.top) / rect.height) * 2 - 1),
    );
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.#camera);
    const hit = rc.intersectObject(this.#mesh, false)[0];
    const faceIndex = hit?.faceIndex;
    if (!hit || faceIndex === undefined || faceIndex === null) return null;
    const idx = this.#geometry.getIndex()!;
    const pos = this.#geometry.getAttribute('position') as THREE.BufferAttribute;
    const i = faceIndex * 3;
    const a = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(i));
    const b = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(i + 1));
    const c = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(i + 2));
    const bc = new THREE.Vector3();
    THREE.Triangle.getBarycoord(hit.point, a, b, c, bc);
    return { point: [hit.point.x, hit.point.y, hit.point.z], faceIndex, bary: [bc.x, bc.y, bc.z] };
  }

  /** The renderer's canvas, for capturing frames. */
  get canvas(): HTMLCanvasElement {
    return this.#renderer.domElement;
  }

  /**
   * Render immediately, outside the animation loop. A WebGL canvas without
   * preserveDrawingBuffer keeps its drawing buffer only until the browser next
   * composites, so a capturer must render and copy within one task.
   */
  renderNow(): void {
    this.#needsRender = false;
    this.#renderer.render(this.#scene, this.#camera);
  }

  /** Mirror this scene's camera whenever the other scene's controls move.
   *  This scene (the newly built one) first adopts the other's current pose:
   *  without that, a scene rebuilt while the user was orbiting its sibling
   *  comes up at the default framing and the panes are out of sync until the
   *  next drag. */
  syncCamerasWith(other: SphereScene): void {
    this.setCameraState(other.cameraState());
    const follow = (src: SphereScene, dst: SphereScene) => {
      src.#controls.addEventListener('change', () => {
        if (dst.#syncing) return;
        src.#syncing = true;
        dst.#camera.position.copy(src.#camera.position);
        dst.#camera.zoom = src.#camera.zoom;
        dst.#camera.updateProjectionMatrix();
        dst.#controls.target.copy(src.#controls.target);
        dst.#controls.update();
        dst.#needsRender = true;
        src.#syncing = false;
      });
    };
    follow(this, other);
    follow(other, this);
  }

  /** Orbit the camera about the up axis by `angle` radians, keeping the
   *  target. Synced sibling scenes follow via their controls, as with a drag. */
  orbitBy(angle: number): void {
    const offset = this.#camera.position.clone().sub(this.#controls.target);
    offset.applyAxisAngle(this.#camera.up, angle);
    this.#camera.position.copy(this.#controls.target).add(offset);
    this.#controls.update();
    this.#needsRender = true;
  }

  /** Camera pose, for carrying the view across a scene rebuild. */
  cameraState(): { position: THREE.Vector3; target: THREE.Vector3; zoom: number } {
    return {
      position: this.#camera.position.clone(),
      target: this.#controls.target.clone(),
      zoom: this.#camera.zoom,
    };
  }

  setCameraState(s: {
    position: THREE.Vector3;
    target: THREE.Vector3;
    zoom: number;
  }): void {
    this.#camera.position.copy(s.position);
    this.#camera.zoom = s.zoom;
    this.#camera.updateProjectionMatrix();
    this.#controls.target.copy(s.target);
    this.#controls.update();
    this.#needsRender = true;
  }

  /**
   * Position the camera to comfortably frame the geometry.
   *
   * The distance is generous on purpose. The bounding sphere is of the surface
   * currently loaded, but the camera is *kept* across a geometry change and
   * across the morph, so a frame that only just fits the shape at hand would
   * clip the next one. Leaving room means switching shapes never needs a
   * camera reset to see what happened.
   */
  fitCamera(): void {
    this.#geometry.computeBoundingSphere();
    const bs = this.#geometry.boundingSphere;
    if (!bs) return;
    const radius = Math.max(bs.radius, 1e-6);
    const distance = radius * 3.4;
    this.#controls.target.copy(bs.center);
    this.#camera.position.set(
      bs.center.x + distance * 0.55,
      bs.center.y + distance * 0.35,
      bs.center.z + distance * 0.75,
    );
    this.#camera.near = radius * 0.01;
    this.#camera.far = radius * 100;
    this.#camera.updateProjectionMatrix();
    this.#controls.update();
    this.#needsRender = true;
    this.#defaultCameraState = {
      position: this.#camera.position.clone(),
      target: this.#controls.target.clone(),
    };
  }

  resetCamera(): void {
    if (this.#defaultCameraState) {
      this.#camera.position.copy(this.#defaultCameraState.position);
      this.#controls.target.copy(this.#defaultCameraState.target);
      this.#controls.update();
      this.#needsRender = true;
    } else {
      this.fitCamera();
    }
  }

  resize(width: number, height: number): void {
    // Setting canvas.width clears the canvas even at the same value, which
    // shows as a blank flash until the next render — skip no-op resizes.
    if (width === this.#lastW && height === this.#lastH) return;
    this.#lastW = width;
    this.#lastH = height;
    this.#camera.aspect = width / Math.max(1, height);
    this.#camera.updateProjectionMatrix();
    // updateStyle=false: the canvas keeps its 100%/100% CSS sizing
    this.#renderer.setSize(width, height, false);
    // setSize clears the drawing buffer, so a re-render is required even
    // though nothing in the scene moved
    this.#needsRender = true;
  }

  /**
   * Set the drawing buffer to an exact square pixel size, independent of the
   * container and devicePixelRatio — for capturing at a chosen resolution.
   * The canvas keeps its CSS sizing, so on screen it just rescales. Undo with
   * restoreSize().
   */
  captureSize(px: number): void {
    this.#renderer.setPixelRatio(1);
    this.#renderer.setSize(px, px, false);
    this.#camera.aspect = 1;
    this.#camera.updateProjectionMatrix();
    this.#needsRender = true;
  }

  /** Return from captureSize() to the container-driven buffer size. */
  restoreSize(): void {
    this.#renderer.setPixelRatio(window.devicePixelRatio || 1);
    if (this.#lastW > 0 && this.#lastH > 0) {
      this.#renderer.setSize(this.#lastW, this.#lastH, false);
      this.#camera.aspect = this.#lastW / Math.max(1, this.#lastH);
      this.#camera.updateProjectionMatrix();
    }
    this.#needsRender = true;
  }

  dispose(): void {
    if (this.#animationId !== null) {
      cancelAnimationFrame(this.#animationId);
      this.#animationId = null;
    }
    this.#controls.dispose();
    if (this.#wire) {
      this.#wire.geometry.dispose();
      (this.#wire.material as THREE.Material).dispose();
      this.#wire = null;
    }
    if (this.#grid) {
      this.#grid.geometry.dispose();
      (this.#grid.material as THREE.Material).dispose();
      this.#grid = null;
    }
    if (this.#marker) {
      this.#marker.geometry.dispose();
      (this.#marker.material as THREE.Material).dispose();
      this.#marker = null;
    }
    this.#geometry.dispose();
    (this.#mesh.material as THREE.Material).dispose();
    if (this.#renderer.domElement.parentNode) {
      this.#renderer.domElement.parentNode.removeChild(
        this.#renderer.domElement,
      );
    }
    this.#renderer.dispose();
  }
}
