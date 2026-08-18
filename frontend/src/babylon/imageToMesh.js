import {
  Color3,
  Mesh,
  MeshBuilder,
  PBRMaterial,
  StandardMaterial,
  Texture,
  TransformNode,
  Vector3,
} from "@babylonjs/core";
import earcut from "earcut";

const DEFAULT_HEIGHT = 1;
const DEFAULT_ALPHA = 32;
const MAX_TRACE = 180;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function uniqueName(prefix) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

export function loadRasterImage(src) {
  return new Promise((resolve, reject) => {
    if (!src) {
      reject(new Error("No image source"));
      return;
    }

    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () =>
      reject(new Error(`Unable to load image: ${src}`));
    image.src = src;
  });
}

export function rasterToCanvas(image, maxSize = 512) {
  const scale = Math.min(
    1,
    maxSize / Math.max(image.width, image.height, 1)
  );
  const width = Math.max(2, Math.round(image.width * scale));
  const height = Math.max(2, Math.round(image.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);
  return canvas;
}

function createTexture(scene, url, invertY = false) {
  const texture = new Texture(
    url,
    scene,
    true,
    invertY,
    Texture.TRILINEAR_SAMPLINGMODE
  );
  texture.hasAlpha = true;
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  return texture;
}

function makePbrMaterial(scene, name, options = {}) {
  const material = new PBRMaterial(name, scene);
  material.metallic = options.metallic ?? 0.04;
  material.roughness = options.roughness ?? 0.42;
  material.backFaceCulling = options.backFaceCulling ?? false;
  material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHATESTANDBLEND;
  material.albedoColor = options.albedoColor || Color3.White();
  if (options.albedoTexture) {
    material.albedoTexture = options.albedoTexture;
    material.useAlphaFromAlbedoTexture = true;
  }
  if (options.emissiveColor) {
    material.emissiveColor = options.emissiveColor;
  }
  material.emissiveIntensity = options.emissiveIntensity ?? 0.08;
  material.environmentIntensity = options.environmentIntensity ?? 1;
  return material;
}

function makeStandardMaterial(scene, name, options = {}) {
  const material = new StandardMaterial(name, scene);
  material.diffuseColor = options.albedoColor || Color3.White();
  material.specularColor = new Color3(0.12, 0.12, 0.12);
  material.backFaceCulling = options.backFaceCulling ?? false;
  material.useAlphaFromDiffuseTexture = true;
  if (options.albedoTexture) {
    material.diffuseTexture = options.albedoTexture;
    material.diffuseTexture.hasAlpha = true;
  }
  if (options.emissiveColor) {
    material.emissiveColor = options.emissiveColor;
  }
  return material;
}

function createSurfaceMaterial(scene, name, options = {}) {
  if (options.useStandard) {
    return makeStandardMaterial(scene, name, options);
  }
  return makePbrMaterial(scene, name, options);
}

export function computeNodeBounds(node) {
  if (!node) {
    return {
      min: Vector3.Zero(),
      max: Vector3.Zero(),
      center: Vector3.Zero(),
      size: Vector3.Zero(),
    };
  }

  node.computeWorldMatrix(true);
  const { min, max } = node.getHierarchyBoundingVectors(true);
  return {
    min,
    max,
    center: min.add(max).scale(0.5),
    size: max.subtract(min),
  };
}

export function centerNodePivot(root) {
  const bounds = computeNodeBounds(root);
  const offset = bounds.center.subtract(root.getAbsolutePosition());
  (root.getChildMeshes ? root.getChildMeshes(false) : []).forEach((mesh) => {
    mesh.position.subtractInPlace(offset);
  });
  root.position.addInPlace(offset);
  return computeNodeBounds(root);
}

export function normalizeNodeHeight(root, height = DEFAULT_HEIGHT) {
  const bounds = computeNodeBounds(root);
  const current = Math.max(bounds.size.y, bounds.size.x, bounds.size.z, 0.0001);
  const scale = height / current;
  root.scaling.scaleInPlace(scale);
  return computeNodeBounds(root);
}

export function enableMeshShadows(root, shadowGenerator) {
  const meshes = root.getChildMeshes ? root.getChildMeshes(false) : [root];
  meshes.forEach((mesh) => {
    mesh.receiveShadows = true;
    if (shadowGenerator && mesh.getTotalVertices?.() > 0) {
      shadowGenerator.addShadowCaster(mesh, false);
    }
  });
}

export function applyNodeTransform(root, {
  position,
  rotation,
  scaling,
} = {}) {
  if (position) {
    root.position.copyFrom(position);
  }
  if (rotation) {
    root.rotation.copyFrom(rotation);
  }
  if (scaling) {
    root.scaling.copyFrom(scaling);
  }
}

export function frameCameraOnNode(camera, node, padding = 1.55) {
  if (!camera || !node) {
    return false;
  }

  const bounds = computeNodeBounds(node);
  const radius = Math.max(bounds.size.x, bounds.size.y, bounds.size.z) * padding;
  camera.setTarget(bounds.center);

  if (typeof camera.radius === "number") {
    camera.radius = Math.max(radius, 0.8);
  } else {
    const dir = camera.getForwardRay(1).direction.scale(-1);
    camera.position.copyFrom(bounds.center.add(dir.scale(radius)));
  }

  return true;
}

function disposeHandle(handle) {
  if (!handle || handle._disposed) {
    return;
  }

  handle._disposed = true;
  (handle.textures || []).forEach((texture) => {
    try {
      texture.dispose();
    } catch (_error) {
      /* ignore */
    }
  });
  (handle.materials || []).forEach((material) => {
    try {
      material.dispose();
    } catch (_error) {
      /* ignore */
    }
  });
  (handle.meshes || []).forEach((mesh) => {
    try {
      mesh.dispose(false, true);
    } catch (_error) {
      /* ignore */
    }
  });
  try {
    handle.root?.dispose();
  } catch (_error) {
    /* ignore */
  }
}

function finishHandle({
  scene,
  root,
  meshes,
  materials,
  textures,
  options = {},
  kind,
}) {
  const height = options.height ?? DEFAULT_HEIGHT;
  centerNodePivot(root);
  normalizeNodeHeight(root, height);
  applyNodeTransform(root, options);
  if (options.shadowGenerator || options.receiveShadows !== false) {
    enableMeshShadows(root, options.shadowGenerator);
  }

  const bounds = computeNodeBounds(root);
  const handle = {
    kind,
    root,
    meshes,
    materials,
    textures,
    bounds,
    scene,
    dispose() {
      disposeHandle(handle);
    },
  };

  if (options.frameCamera) {
    frameCameraOnNode(options.frameCamera, root, options.framePadding);
  }

  return handle;
}

function rdp(points, epsilon) {
  if (points.length < 3) {
    return points;
  }

  const first = points[0];
  const last = points[points.length - 1];
  let maxDist = 0;
  let index = 0;

  for (let i = 1; i < points.length - 1; i += 1) {
    const dist = perpendicularDistance(points[i], first, last);
    if (dist > maxDist) {
      maxDist = dist;
      index = i;
    }
  }

  if (maxDist > epsilon) {
    const left = rdp(points.slice(0, index + 1), epsilon);
    const right = rdp(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }

  return [first, last];
}

function perpendicularDistance(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy) || 1;
  return Math.abs(dy * point.x - dx * point.y + b.x * a.y - b.y * a.x) / length;
}

function shoelace(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const j = (i + 1) % points.length;
    area += points[i].x * points[j].y - points[j].x * points[i].y;
  }
  return area / 2;
}

function ensureWinding(points, clockwise) {
  const area = shoelace(points);
  const isClockwise = area < 0;
  if (isClockwise !== clockwise) {
    return points.slice().reverse();
  }
  return points;
}

function sampleAlpha(data, width, height, x, y, threshold) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return 0;
  }
  return data[(y * width + x) * 4 + 3] >= threshold ? 1 : 0;
}

function marchingSquares(canvas, threshold) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const { data } = ctx.getImageData(0, 0, width, height);
  const segments = [];

  const push = (ax, ay, bx, by) => {
    segments.push({
      a: { x: ax, y: ay },
      b: { x: bx, y: by },
    });
  };

  for (let y = 0; y < height - 1; y += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      const tl = sampleAlpha(data, width, height, x, y, threshold);
      const tr = sampleAlpha(data, width, height, x + 1, y, threshold);
      const br = sampleAlpha(data, width, height, x + 1, y + 1, threshold);
      const bl = sampleAlpha(data, width, height, x, y + 1, threshold);
      const code = (tl << 3) | (tr << 2) | (br << 1) | bl;

      const mx = x + 0.5;
      const my = y + 0.5;
      const x1 = x + 1;
      const y1 = y + 1;

      switch (code) {
        case 1:
        case 14:
          push(x, my, mx, y1);
          break;
        case 2:
        case 13:
          push(mx, y1, x1, my);
          break;
        case 3:
        case 12:
          push(x, my, x1, my);
          break;
        case 4:
        case 11:
          push(mx, y, x1, my);
          break;
        case 6:
        case 9:
          push(mx, y, mx, y1);
          break;
        case 7:
        case 8:
          push(x, my, mx, y);
          break;
        case 5:
          push(x, my, mx, y);
          push(mx, y1, x1, my);
          break;
        case 10:
          push(mx, y, x1, my);
          push(x, my, mx, y1);
          break;
        default:
          break;
      }
    }
  }

  return stitchSegments(segments);
}

function keyOf(point) {
  return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
}

function stitchSegments(segments) {
  const unused = segments.slice();
  const loops = [];

  while (unused.length) {
    const start = unused.pop();
    const loop = [start.a, start.b];
    let guard = unused.length + 2;

    while (guard > 0) {
      guard -= 1;
      const tail = loop[loop.length - 1];
      const tailKey = keyOf(tail);
      let found = -1;

      for (let i = 0; i < unused.length; i += 1) {
        const seg = unused[i];
        if (keyOf(seg.a) === tailKey) {
          loop.push(seg.b);
          found = i;
          break;
        }
        if (keyOf(seg.b) === tailKey) {
          loop.push(seg.a);
          found = i;
          break;
        }
      }

      if (found < 0) {
        break;
      }

      unused.splice(found, 1);

      if (keyOf(loop[loop.length - 1]) === keyOf(loop[0])) {
        break;
      }
    }

    if (loop.length >= 4) {
      if (keyOf(loop[0]) === keyOf(loop[loop.length - 1])) {
        loop.pop();
      }
      loops.push(loop);
    }
  }

  return loops.sort((a, b) => Math.abs(shoelace(b)) - Math.abs(shoelace(a)));
}

function pointInPoly(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const intersect =
      a.y > point.y !== b.y > point.y &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / ((b.y - a.y) || 1e-6) + a.x;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

function toShapeVectors(loop, canvas) {
  const aspect = canvas.width / canvas.height;
  return loop.map((point) => {
    const nx = point.x / canvas.width - 0.5;
    const ny = 0.5 - point.y / canvas.height;
    return new Vector3(nx * aspect, 0, ny);
  });
}

export function extractImageSilhouette(canvas, {
  alphaThreshold = DEFAULT_ALPHA,
  simplify = 1.35,
} = {}) {
  const loops = marchingSquares(canvas, alphaThreshold);
  if (!loops.length) {
    return null;
  }

  const outerRaw = rdp(loops[0], simplify);
  if (outerRaw.length < 4) {
    return null;
  }

  const outer = ensureWinding(outerRaw, false);
  const holes = loops
    .slice(1)
    .filter((loop) => Math.abs(shoelace(loop)) > 8)
    .filter((loop) => pointInPoly(loop[0], outer))
    .map((loop) => ensureWinding(rdp(loop, simplify), true))
    .filter((loop) => loop.length >= 4);

  return {
    outer: toShapeVectors(outer, canvas),
    holes: holes.map((loop) => toShapeVectors(loop, canvas)),
    aspect: canvas.width / canvas.height,
  };
}

function attachCommonMeshFlags(mesh) {
  mesh.isPickable = true;
  mesh.receiveShadows = true;
  mesh.alwaysSelectAsActiveMesh = false;
}

export async function createImagePlane(options) {
  const {
    scene,
    image,
    billboard = false,
    name = uniqueName("image-plane"),
  } = options;

  const raster = await loadRasterImage(image);
  const aspect = raster.width / Math.max(raster.height, 1);
  const height = options.height ?? DEFAULT_HEIGHT;
  const width = options.width ?? height * aspect;

  const root = new TransformNode(`${name}-root`, scene);
  const texture = createTexture(scene, image, false);
  const material = createSurfaceMaterial(scene, `${name}-mat`, {
    albedoTexture: texture,
    useStandard: options.useStandard,
    roughness: 0.55,
    metallic: 0.02,
  });

  const plane = MeshBuilder.CreatePlane(
    name,
    {
      width,
      height,
      sideOrientation: Mesh.DOUBLESIDE,
    },
    scene
  );
  plane.parent = root;
  plane.material = material;
  attachCommonMeshFlags(plane);

  if (billboard) {
    plane.billboardMode = Mesh.BILLBOARDMODE_ALL;
  }

  return finishHandle({
    scene,
    root,
    meshes: [plane],
    materials: [material],
    textures: [texture],
    options: { ...options, height },
    kind: billboard ? "billboard" : "plane",
  });
}

export async function createThickImage(options) {
  const {
    scene,
    image,
    depth = 0.06,
    name = uniqueName("thick-image"),
    sideColor = new Color3(0.16, 0.14, 0.12),
  } = options;

  const raster = await loadRasterImage(image);
  const aspect = raster.width / Math.max(raster.height, 1);
  const height = options.height ?? DEFAULT_HEIGHT;
  const width = options.width ?? height * aspect;

  const root = new TransformNode(`${name}-root`, scene);
  const frontTex = createTexture(scene, image, false);
  const frontMat = createSurfaceMaterial(scene, `${name}-front`, {
    albedoTexture: frontTex,
    useStandard: options.useStandard,
    roughness: 0.38,
    metallic: 0.05,
  });
  const sideMat = createSurfaceMaterial(scene, `${name}-side`, {
    albedoColor: sideColor,
    useStandard: options.useStandard,
    roughness: 0.5,
    metallic: options.sideMetallic ?? 0.18,
    backFaceCulling: true,
  });

  const box = MeshBuilder.CreateBox(
    name,
    {
      width,
      height,
      depth,
      wrap: true,
    },
    scene
  );
  box.parent = root;
  box.subMeshes = [];
  box.material = frontMat;

  const sides = MeshBuilder.CreateBox(
    `${name}-shell`,
    { width, height, depth },
    scene
  );
  sides.parent = root;
  sides.scaling.set(1.001, 1.001, 1);
  sides.material = sideMat;

  const face = MeshBuilder.CreatePlane(
    `${name}-face`,
    { width: width * 0.995, height: height * 0.995 },
    scene
  );
  face.parent = root;
  face.position.z = depth / 2 + 0.001;
  face.material = frontMat;
  attachCommonMeshFlags(box);
  attachCommonMeshFlags(sides);
  attachCommonMeshFlags(face);

  return finishHandle({
    scene,
    root,
    meshes: [box, sides, face],
    materials: [frontMat, sideMat],
    textures: [frontTex],
    options: { ...options, height },
    kind: "thick",
  });
}

export async function createExtrudedImage(options) {
  const {
    scene,
    image,
    depth = 0.08,
    bevel = 0.02,
    name = uniqueName("extruded-image"),
    sideColor = new Color3(0.78, 0.62, 0.28),
    alphaThreshold = DEFAULT_ALPHA,
  } = options;

  const raster = await loadRasterImage(image);
  const canvas = rasterToCanvas(raster, MAX_TRACE);
  const silhouette = extractImageSilhouette(canvas, {
    alphaThreshold,
    simplify: Math.max(0.8, 1.1 + bevel * 12),
  });

  if (!silhouette) {
    return createThickImage({
      ...options,
      depth: Math.max(depth, 0.04),
      sideColor,
    });
  }

  const root = new TransformNode(`${name}-root`, scene);
  const frontTex = createTexture(scene, image, false);
  const frontMat = createSurfaceMaterial(scene, `${name}-front`, {
    albedoTexture: frontTex,
    useStandard: options.useStandard,
    roughness: 0.36,
    metallic: 0.08,
  });
  const sideMat = createSurfaceMaterial(scene, `${name}-side`, {
    albedoColor: sideColor,
    useStandard: options.useStandard,
    roughness: 0.32,
    metallic: options.sideMetallic ?? 0.72,
    emissiveColor: sideColor.scale(0.15),
    backFaceCulling: true,
  });

  let body = null;
  try {
    body = MeshBuilder.ExtrudePolygon(
      name,
      {
        shape: silhouette.outer,
        holes: silhouette.holes,
        depth,
        sideOrientation: Mesh.DOUBLESIDE,
      },
      scene,
      earcut
    );
  } catch (error) {
    console.warn("ExtrudePolygon failed, using thick image", error);
    root.dispose();
    return createThickImage({
      ...options,
      depth,
      sideColor,
    });
  }

  body.parent = root;
  body.material = sideMat;
  body.rotation.x = -Math.PI / 2;
  attachCommonMeshFlags(body);

  const rimMeshes = [body];

  if (bevel > 0) {
    const rim = body.clone(`${name}-rim`);
    rim.parent = root;
    rim.scaling.set(1 + bevel * 1.8, 1 + bevel * 1.8, 0.22);
    rim.position.z = -depth * 0.12;
    rim.material = sideMat;
    attachCommonMeshFlags(rim);
    rimMeshes.push(rim);
  }

  const faceHeight = 1;
  const faceWidth = silhouette.aspect;
  const face = MeshBuilder.CreatePlane(
    `${name}-face`,
    {
      width: faceWidth,
      height: faceHeight,
      sideOrientation: Mesh.DOUBLESIDE,
    },
    scene
  );
  face.parent = root;
  face.position.z = depth / 2 + 0.002;
  face.material = frontMat;
  attachCommonMeshFlags(face);

  const meshes = root.getChildMeshes(false);

  return finishHandle({
    scene,
    root,
    meshes,
    materials: [frontMat, sideMat],
    textures: [frontTex],
    options: { ...options, height: options.height ?? DEFAULT_HEIGHT },
    kind: "extruded",
  });
}

export async function createDepthImage(options) {
  const {
    scene,
    image,
    depthMap,
    strength = 0.28,
    subdivisions = 80,
    name = uniqueName("depth-image"),
  } = options;

  if (!depthMap) {
    return createThickImage({
      ...options,
      depth: Math.max(0.04, strength * 0.4),
    });
  }

  const colorImage = await loadRasterImage(image);
  const depthImage = await loadRasterImage(depthMap);
  const depthCanvas = rasterToCanvas(depthImage, subdivisions + 1);
  const depthCtx = depthCanvas.getContext("2d");
  const depthData = depthCtx.getImageData(
    0,
    0,
    depthCanvas.width,
    depthCanvas.height
  ).data;

  const aspect = colorImage.width / Math.max(colorImage.height, 1);
  const height = options.height ?? DEFAULT_HEIGHT;
  const width = options.width ?? height * aspect;

  const root = new TransformNode(`${name}-root`, scene);
  const ground = MeshBuilder.CreateGround(
    name,
    {
      width,
      height: width / aspect,
      subdivisions: clamp(subdivisions, 16, 180),
      updatable: true,
    },
    scene
  );
  ground.parent = root;

  const positions = ground.getVerticesData("position");
  const normalsNeeded = true;
  const subdiv = clamp(subdivisions, 16, 180);
  const cols = subdiv + 1;

  for (let i = 0; i < positions.length / 3; i += 1) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const u = col / subdiv;
    const v = 1 - row / subdiv;
    const px = clamp(Math.floor(u * (depthCanvas.width - 1)), 0, depthCanvas.width - 1);
    const py = clamp(
      Math.floor(v * (depthCanvas.height - 1)),
      0,
      depthCanvas.height - 1
    );
    const offset = (py * depthCanvas.width + px) * 4;
    const luminance =
      (depthData[offset] * 0.299 +
        depthData[offset + 1] * 0.587 +
        depthData[offset + 2] * 0.114) /
      255;
    positions[i * 3 + 1] = (luminance - 0.5) * strength;
  }

  ground.updateVerticesData("position", positions);
  if (normalsNeeded) {
    ground.createNormals(true);
  }

  const colorTex = createTexture(scene, image, false);
  const material = createSurfaceMaterial(scene, `${name}-mat`, {
    albedoTexture: colorTex,
    useStandard: options.useStandard,
    roughness: 0.48,
    metallic: 0.04,
  });
  ground.material = material;
  attachCommonMeshFlags(ground);

  return finishHandle({
    scene,
    root,
    meshes: [ground],
    materials: [material],
    textures: [colorTex],
    options: { ...options, height },
    kind: "depth",
  });
}

export async function create3DLogo(options) {
  const {
    image,
    height = 1.2,
    depth = 0.08,
    bevel = 0.025,
    scene,
  } = options;

  return createExtrudedImage({
    ...options,
    scene,
    image,
    height,
    depth,
    bevel,
    sideColor: options.sideColor || new Color3(0.83, 0.67, 0.28),
    sideMetallic: 0.78,
    name: options.name || uniqueName("nhl-logo"),
  });
}

export const ImageMesh = {
  createImagePlane,
  createThickImage,
  createExtrudedImage,
  createDepthImage,
  create3DLogo,
  frameCameraOnNode,
  computeNodeBounds,
  dispose: disposeHandle,
};
