/**
 * Live 3D uniform preview on the same React Three Fiber + drei stack as the office hub.
 *
 * Cost model: one low-poly jersey (shared geometries), one torso + one sleeve canvas
 * texture redrawn only when the uniform config changes (old textures disposed), and
 * frameloop="demand" so the GPU renders only on a change or while the user drags.
 * Unmounting the Canvas releases the WebGL context; nothing keeps running off-screen.
 */
import React, { useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { FONT_FAMILY, crestToSvgString } from "./crest";

const TEX_W = 1024;
const TEX_H = 768;
// Cylinder seam sits on the left side: u=0.25 is the chest, u=0.75 the back.
const FRONT_X = TEX_W * 0.25;
const BACK_X = TEX_W * 0.75;

function stripeBands(kind) {
  switch (kind) {
    case "single": return [[0, 0.07]];
    case "double": return [[0, 0.045], [0.07, 0.045]];
    case "triple": return [[0, 0.035], [0.055, 0.035], [0.11, 0.035]];
    case "heritage": return [[0, 0.03], [0.045, 0.07], [0.13, 0.03]];
    case "split": return [[0, 0.09]];
    default: return [];
  }
}

function drawStripes(ctx, u, y0, h, w) {
  const bands = stripeBands(u.stripe);
  bands.forEach(([off, size], i) => {
    ctx.fillStyle = u.stripe === "heritage" && i !== 1 ? u.stripe_accent : u.stripe_color;
    ctx.fillRect(0, y0 + off * h, w, size * h);
    if (u.stripe === "split") {
      ctx.fillStyle = u.stripe_accent;
      ctx.fillRect(0, y0 + off * h + size * h * 0.42, w, size * h * 0.16);
    }
  });
}

function drawArchedText(ctx, text, cx, cy, radius, size, color, font) {
  ctx.save();
  ctx.font = `900 ${size}px ${font}`;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const chars = String(text || "").toUpperCase().split("");
  const widths = chars.map((c) => ctx.measureText(c).width + size * 0.08);
  const total = widths.reduce((a, b) => a + b, 0);
  let angle = -total / radius / 2;
  chars.forEach((c, i) => {
    const a = angle + widths[i] / radius / 2;
    ctx.save();
    ctx.translate(cx + Math.sin(a) * radius, cy + radius - Math.cos(a) * radius);
    ctx.rotate(a);
    ctx.fillText(c, 0, 0);
    ctx.restore();
    angle += widths[i] / radius;
  });
  ctx.restore();
}

function drawTorso(canvas, u, preview, crestImg) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = u.body;
  ctx.fillRect(0, 0, TEX_W, TEX_H);
  // shoulder yoke
  ctx.fillStyle = u.shoulder;
  ctx.fillRect(0, 0, TEX_W, TEX_H * 0.17);
  // waist stripes
  if (u.waist_stripe) drawStripes(ctx, u, TEX_H * 0.74, TEX_H, TEX_W);
  // collar: V notch on the chest
  ctx.fillStyle = u.collar_color;
  if (u.collar === "crew") {
    ctx.fillRect(FRONT_X - 70, 0, 140, TEX_H * 0.03);
  } else {
    const depth = u.collar === "lace" || u.collar === "laced-v" ? 0.12 : 0.09;
    ctx.beginPath();
    ctx.moveTo(FRONT_X - 70, 0);
    ctx.lineTo(FRONT_X + 70, 0);
    ctx.lineTo(FRONT_X, TEX_H * depth);
    ctx.closePath();
    ctx.fill();
    if (u.collar !== "v-neck") {
      ctx.strokeStyle = u.body;
      ctx.lineWidth = 3;
      for (let i = 1; i <= 3; i += 1) {
        const y = (TEX_H * depth * i) / 4;
        ctx.beginPath();
        ctx.moveTo(FRONT_X - 14, y);
        ctx.lineTo(FRONT_X + 14, y);
        ctx.stroke();
      }
    }
  }
  // crest
  if (crestImg) {
    const size = 250 * Number(u.crest_scale || 1);
    const cy = TEX_H * (0.4 + Number(u.crest_y || 0) * 0.5);
    ctx.drawImage(crestImg, FRONT_X - size / 2, cy - size / 2, size, size);
  }
  const font = FONT_FAMILY[u.number_font] || FONT_FAMILY.block;
  // captain letter on the chest
  if (u.captain_letter && preview.letter) {
    ctx.save();
    ctx.font = `900 64px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = u.number_outline;
    ctx.strokeText(preview.letter, FRONT_X + 140, TEX_H * 0.22);
    ctx.fillStyle = u.number_color;
    ctx.fillText(preview.letter, FRONT_X + 140, TEX_H * 0.22);
    ctx.restore();
  }
  // nameplate + back number
  const name = String(preview.name || "").toUpperCase();
  if (name) {
    if (u.nameplate === "straight") {
      ctx.save();
      ctx.font = `900 46px ${font}`;
      ctx.fillStyle = u.name_color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(name, BACK_X, TEX_H * 0.2);
      ctx.restore();
    } else {
      drawArchedText(ctx, name, BACK_X, TEX_H * (u.nameplate === "vertical-arch" ? 0.15 : 0.18), 420, 44, u.name_color, font);
    }
  }
  const num = String(preview.number ?? "");
  if (num) {
    ctx.save();
    ctx.font = `900 250px ${font}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 18;
    ctx.strokeStyle = u.number_outline;
    ctx.strokeText(num, BACK_X, TEX_H * 0.47);
    ctx.fillStyle = u.number_color;
    ctx.fillText(num, BACK_X, TEX_H * 0.47);
    ctx.restore();
  }
}

function drawSleeve(canvas, u, preview) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = u.sleeve;
  ctx.fillRect(0, 0, w, h);
  // sleeve stripes near the cuff (canvas bottom = cuff end of the cylinder)
  drawStripes(ctx, u, h * 0.66, h, w);
  const num = String(preview.number ?? "");
  if (num) {
    ctx.save();
    ctx.font = `900 ${h * 0.22}px ${FONT_FAMILY[u.number_font] || FONT_FAMILY.block}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = u.number_outline;
    ctx.strokeText(num, w * 0.25, h * 0.32);
    ctx.fillStyle = u.number_color;
    ctx.fillText(num, w * 0.25, h * 0.32);
    ctx.restore();
  }
}

function useCrestImage(crest, fallbackSrc, onReady) {
  const [img, setImg] = React.useState(null);
  const svg = useMemo(() => (crest ? crestToSvgString(crest, 512) : null), [crest]);
  useEffect(() => {
    let url = null;
    let alive = true;
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      if (alive) {
        setImg(im);
        onReady?.();
      }
    };
    im.onerror = () => alive && setImg(null);
    if (svg) {
      url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
      im.src = url;
    } else if (fallbackSrc) {
      im.src = fallbackSrc;
    } else {
      setImg(null);
    }
    return () => {
      alive = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [svg, fallbackSrc, onReady]);
  return img;
}

function Jersey({ uniform, crest, crestFallbackSrc, preview }) {
  const invalidate = useThree((s) => s.invalidate);
  const crestImg = useCrestImage(crest, crestFallbackSrc, invalidate);
  const torsoCanvas = useMemo(() => Object.assign(document.createElement("canvas"), { width: TEX_W, height: TEX_H }), []);
  const sleeveCanvas = useMemo(() => Object.assign(document.createElement("canvas"), { width: 512, height: 256 }), []);
  const torsoTex = useMemo(() => {
    const t = new THREE.CanvasTexture(torsoCanvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return t;
  }, [torsoCanvas]);
  const sleeveTex = useMemo(() => {
    const t = new THREE.CanvasTexture(sleeveCanvas);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [sleeveCanvas]);
  const geo = useMemo(() => {
    // Lathe profile bottom -> neck: flared hem, straight body, chest, sloped shoulders.
    // Same UV layout as a cylinder (u around, v up), seam on the left side.
    const profile = [
      [0.7, -0.78], [0.67, -0.55], [0.63, -0.2], [0.62, 0.2], [0.63, 0.42], [0.6, 0.56],
      [0.5, 0.67], [0.36, 0.74], [0.24, 0.78], [0.2, 0.8],
    ].map(([r, y]) => new THREE.Vector2(r, y));
    const torso = new THREE.LatheGeometry(profile, 64, -Math.PI / 2);
    // Lathe V follows profile index; remap to height so the texture isn't squeezed into the shoulders.
    const pos = torso.attributes.position;
    const uv = torso.attributes.uv;
    for (let i = 0; i < uv.count; i += 1) uv.setY(i, (pos.getY(i) + 0.78) / 1.58);
    uv.needsUpdate = true;
    return {
      torso,
      sleeve: new THREE.CylinderGeometry(0.22, 0.26, 0.8, 28, 1, true),
      cuff: new THREE.TorusGeometry(0.26, 0.03, 8, 28),
      hem: new THREE.TorusGeometry(0.7, 0.025, 8, 64),
      collar: new THREE.TorusGeometry(0.21, 0.035, 8, 32),
    };
  }, []);
  const mats = useMemo(() => ({
    torso: new THREE.MeshStandardMaterial({ map: torsoTex, side: THREE.DoubleSide }),
    sleeve: new THREE.MeshStandardMaterial({ map: sleeveTex, side: THREE.DoubleSide }),
    trim: new THREE.MeshStandardMaterial({}),
  }), [torsoTex, sleeveTex]);

  // Redraw textures + recolour materials only when the config changes; then ask for one frame.
  useEffect(() => {
    drawTorso(torsoCanvas, uniform, preview, crestImg);
    drawSleeve(sleeveCanvas, uniform, preview);
    torsoTex.needsUpdate = true;
    sleeveTex.needsUpdate = true;
    const rough = Number(uniform.fabric_roughness ?? 0.78);
    Object.values(mats).forEach((m) => {
      m.roughness = rough;
      m.metalness = 0.02;
    });
    mats.trim.color.set(uniform.collar_color);
    invalidate();
  }, [uniform, preview, crestImg, torsoCanvas, sleeveCanvas, torsoTex, sleeveTex, mats, invalidate]);

  // On-demand loop: the first frame can land before layout gives the canvas its size.
  useEffect(() => {
    let n = 0;
    let id = 0;
    const tick = () => {
      invalidate();
      n += 1;
      if (n < 3) id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [invalidate]);

  useEffect(() => () => {
    Object.values(geo).forEach((g) => g.dispose());
    Object.values(mats).forEach((m) => m.dispose());
    torsoTex.dispose();
    sleeveTex.dispose();
  }, [geo, mats, torsoTex, sleeveTex]);

  return (
    <group position={[0, -0.05, 0]} scale={[1, 1, 0.62]}>
      <mesh geometry={geo.torso} material={mats.torso} />
      <mesh geometry={geo.hem} material={mats.trim} position={[0, -0.78, 0]} rotation={[Math.PI / 2, 0, 0]} />
      <mesh geometry={geo.collar} material={mats.trim} position={[0, 0.8, 0]} rotation={[Math.PI / 2, 0, 0]} />
      {[-1, 1].map((side) => (
        <group key={side} position={[side * 0.56, 0.58, 0]} rotation={[0, 0, side * 0.82]}>
          <mesh geometry={geo.sleeve} material={mats.sleeve} position={[0, -0.36, 0]} scale={[1, 1, 1.25]} />
          <mesh geometry={geo.cuff} material={mats.trim} position={[0, -0.76, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[1, 1.25, 1]} />
        </group>
      ))}
    </group>
  );
}

const VIEW_ANGLE = { front: 0, side: Math.PI / 2, back: Math.PI };

function ViewRig({ view }) {
  const controls = useRef(null);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    const c = controls.current;
    if (!c || view == null) return;
    c.setAzimuthalAngle(VIEW_ANGLE[view] ?? 0);
    c.update();
    invalidate();
  }, [view, invalidate]);
  return (
    <OrbitControls
      ref={controls}
      enablePan={false}
      enableDamping={false}
      minDistance={2.2}
      maxDistance={5}
      minPolarAngle={Math.PI * 0.3}
      maxPolarAngle={Math.PI * 0.62}
    />
  );
}

export default function UniformStudio3D({ uniform, crest, crestFallbackSrc, preview, view = "front", height = 420 }) {
  if (!uniform) return null;
  return (
    <div className="lh-studio-canvas" style={{ height }}>
      <Canvas
        frameloop="demand"
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true, powerPreference: "default", preserveDrawingBuffer: false }}
        camera={{ position: [0, 0.1, 3.4], fov: 38, near: 0.1, far: 20 }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[2.5, 3, 4]} intensity={1.15} />
        <directionalLight position={[-3, 1.5, -3]} intensity={0.45} color="#8fb6ff" />
        <Jersey uniform={uniform} crest={crest} crestFallbackSrc={crestFallbackSrc} preview={preview} />
        <ViewRig view={view} />
      </Canvas>
    </div>
  );
}
