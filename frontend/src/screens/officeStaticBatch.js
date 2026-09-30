import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

/*
 * Render-only draw-call batching for static office geometry.
 *
 * Children render exactly as before (React still owns them). After mount, opaque meshes
 * whose materials share every setting except base colour are merged into one mesh per
 * material "family", with each mesh's colour baked into a vertex-colour attribute; the
 * originals are hidden, not removed. Nothing here touches pointer handling, game state
 * or anything interactive — only wrap subtrees with no hover/click/animation.
 * `rebuildKey` must change whenever the wrapped subtree's contents change. `includeTransparent`
 * is only safe for non-overlapping decals: merged pieces lose per-object depth sorting.
 */

const BATCHABLE_TYPES = new Set([
  "MeshStandardMaterial",
  "MeshPhysicalMaterial",
  "MeshBasicMaterial",
  "MeshLambertMaterial",
  "MeshPhongMaterial",
]);

const SCALAR_KEYS = [
  "emissiveIntensity", "roughness", "metalness", "opacity", "transparent", "side",
  "flatShading", "toneMapped", "envMapIntensity", "clearcoat", "clearcoatRoughness",
  "sheen", "sheenRoughness", "transmission", "thickness", "ior", "reflectivity",
  "specularIntensity", "iridescence", "anisotropy", "dispersion", "shininess", "wireframe",
  "fog", "depthWrite", "depthTest", "alphaTest", "polygonOffset", "polygonOffsetFactor",
  "polygonOffsetUnits", "bumpScale", "blending", "colorWrite", "dithering",
  "premultipliedAlpha", "alphaToCoverage", "aoMapIntensity", "lightMapIntensity",
];

const COLOR_KEYS = ["emissive", "specular", "sheenColor", "specularColor", "attenuationColor"];

const MAP_KEYS = [
  "map", "bumpMap", "normalMap", "roughnessMap", "metalnessMap", "emissiveMap", "aoMap",
  "alphaMap", "envMap", "lightMap", "displacementMap", "clearcoatMap", "clearcoatNormalMap",
  "clearcoatRoughnessMap", "specularMap", "transmissionMap", "thicknessMap", "sheenColorMap",
  "sheenRoughnessMap", "iridescenceMap", "anisotropyMap", "specularIntensityMap",
  "specularColorMap", "gradientMap",
];

const KEEP_ATTRIBUTES = new Set(["position", "normal", "uv"]);

// three sets these itself on MeshStandard/MeshPhysical; anything else means a custom shader.
const BUILTIN_DEFINES = new Set(["STANDARD", "PHYSICAL"]);

function materialFamilyKey(material) {
  const parts = [material.type];
  for (const key of SCALAR_KEYS) {
    if (key in material) parts.push(`${key}=${material[key]}`);
  }
  for (const key of COLOR_KEYS) {
    if (material[key]?.isColor) parts.push(`${key}=${material[key].getHexString()}`);
  }
  for (const key of MAP_KEYS) {
    if (material[key]) parts.push(`${key}=${material[key].uuid}`);
  }
  if (material.normalScale) parts.push(`ns=${material.normalScale.x},${material.normalScale.y}`);
  return parts.join("|");
}

function isBatchableMesh(mesh, includeTransparent) {
  const material = mesh.material;
  if (!mesh.isMesh || mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.isTroikaText) return false;
  if (!material || Array.isArray(material) || !BATCHABLE_TYPES.has(material.type)) return false;
  if (material.transparent && !includeTransparent) return false;
  if (material.vertexColors || material.isShaderMaterial) return false;
  // Custom shader hooks (troika text, drei effects) can't be shared safely.
  if (Object.prototype.hasOwnProperty.call(material, "onBeforeCompile")) return false;
  const customDefines = Object.keys(material.defines || {}).filter((d) => !BUILTIN_DEFINES.has(d));
  if (customDefines.length) return false;
  if (mesh.morphTargetInfluences?.length || mesh.renderOrder !== 0) return false;
  if (Object.prototype.hasOwnProperty.call(mesh, "onBeforeRender")) return false;
  const geometry = mesh.geometry;
  if (!geometry?.isBufferGeometry || !geometry.attributes.position) return false;
  return true;
}

function collectMeshes(root, includeTransparent) {
  const out = [];
  const visit = (object, visible) => {
    if (object.userData?.noBatch) return;
    const shown = visible && object.visible;
    if (object !== root && shown && isBatchableMesh(object, includeTransparent)) out.push(object);
    for (const child of object.children) visit(child, shown);
  };
  visit(root, true);
  return out;
}

function prepareGeometry(mesh, toRootSpace) {
  const geometry = mesh.geometry.clone();
  for (const name of Object.keys(geometry.attributes)) {
    if (!KEEP_ATTRIBUTES.has(name)) geometry.deleteAttribute(name);
  }
  geometry.morphAttributes = {};
  geometry.clearGroups();
  if (!geometry.attributes.normal) geometry.computeVertexNormals();
  geometry.applyMatrix4(toRootSpace);

  const color = mesh.material.color?.isColor ? mesh.material.color : new THREE.Color(1, 1, 1);
  const count = geometry.attributes.position.count;
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function buildBatch(root, includeTransparent) {
  root.updateWorldMatrix(true, true);
  const rootInverse = root.matrixWorld.clone().invert();
  const families = new Map();

  for (const mesh of collectMeshes(root, includeTransparent)) {
    const toRootSpace = rootInverse.clone().multiply(mesh.matrixWorld);
    const geometry = mesh.geometry;
    const key = [
      materialFamilyKey(mesh.material),
      mesh.castShadow ? "cs" : "",
      mesh.receiveShadow ? "rs" : "",
      geometry.index ? "idx" : "flat",
      geometry.attributes.uv ? "uv" : "nouv",
      toRootSpace.determinant() < 0 ? "mirror" : "",
    ].join("#");
    if (!families.has(key)) families.set(key, []);
    families.get(key).push({ mesh, toRootSpace });
  }

  const created = [];
  const hidden = [];
  for (const entries of families.values()) {
    if (entries.length < 2) continue;
    const geometries = entries.map(({ mesh, toRootSpace }) => prepareGeometry(mesh, toRootSpace));
    let merged = null;
    try {
      merged = mergeGeometries(geometries, false);
    } catch (err) {
      merged = null;
    }
    geometries.forEach((g) => g.dispose());
    if (!merged) continue;

    const source = entries[0].mesh;
    const material = source.material.clone();
    if (material.color?.isColor) material.color.setRGB(1, 1, 1);
    material.vertexColors = true;

    const batched = new THREE.Mesh(merged, material);
    batched.name = "office-static-batch";
    batched.castShadow = source.castShadow;
    batched.receiveShadow = source.receiveShadow;
    batched.raycast = () => null;
    batched.matrixAutoUpdate = false;
    batched.userData.noBatch = true;
    root.add(batched);
    created.push(batched);

    for (const { mesh } of entries) {
      mesh.visible = false;
      hidden.push(mesh);
    }
  }

  return () => {
    for (const mesh of created) {
      root.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    for (const mesh of hidden) mesh.visible = true;
  };
}

export function StaticBatch({ children, rebuildKey = "", enabled = true, includeTransparent = false }) {
  const ref = useRef(null);
  const invalidate = useThree((state) => state.invalidate);

  useEffect(() => {
    const root = ref.current;
    if (!root || !enabled) return undefined;
    let undo = null;
    // One frame later so child matrices and lazily-attached materials are settled.
    const frame = requestAnimationFrame(() => {
      undo = buildBatch(root, includeTransparent);
      invalidate();
    });
    return () => {
      cancelAnimationFrame(frame);
      if (undo) undo();
      invalidate();
    };
  }, [rebuildKey, enabled, includeTransparent, invalidate]);

  return <group ref={ref}>{children}</group>;
}
