/* Concept: "Dynasty Plaza" — a from-scratch Double Dash–style 3D hub.
   Mockup only. Injected over the live hub so the REAL station silhouettes
   (cloned from .office-menu-icon svg) and the real team logo are reused untouched.
   Everything around them (world, platforms, frames, HUD) is new. */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const CFG = window.__ddHub || {};
const TEAM = CFG.team || { name: "Toronto Maple Leafs", abbr: "TOR", record: "0-0-0", date: "Sep 15 · Preseason", cap: "$13.31M", logo: "" };

/* ---------- harvest real station art ---------- */
const PALETTE = ["#ff3b3b", "#ffcc00", "#2fd35a", "#2b8cff", "#b45cff", "#ff8a1f", "#ff4fa3", "#16d9c9", "#ff6b4a", "#7a6bff"];
const stations = [...document.querySelectorAll(".office-menu-icon")]
  .filter((el) => el.querySelector("svg"))
  .map((el, i) => ({
    name: el.querySelector(".office-menu-icon__name")?.textContent.trim() || "",
    kicker: el.querySelector(".office-menu-icon__kicker")?.textContent.trim() || "",
    svg: el.querySelector("svg").cloneNode(true),
    color: PALETTE[i % PALETTE.length],
  }));

/* ---------- DOM shell ---------- */
document.getElementById("dd-hub")?.remove();
const root = document.createElement("div");
root.id = "dd-hub";
document.body.appendChild(root);
const style = document.createElement("style");
style.textContent = `
#dd-hub{position:fixed;inset:0;z-index:99999;overflow:hidden;font-family:"Luckiest Guy",cursive;color:#fff;--ink:#1a0d05;--y:#ffe23a;--gh:#f8d58a;--g:#c98a3c;--gl:#6e4314;--rim:#e2231a}
#dd-hub canvas{position:absolute;inset:0;display:block}
.ddh-card{position:absolute;left:0;top:0;width:150px;transform-origin:50% 100%;pointer-events:auto;cursor:pointer;will-change:transform}
.ddh-med{position:relative;width:150px;height:150px;border-radius:28px;display:grid;place-items:center;
  background:radial-gradient(circle at 50% 30%,#2a4c9e,#0b1838 75%);
  box-shadow:0 0 0 3px var(--gl),0 0 0 6px var(--gh),0 0 0 9px var(--gl),0 0 0 13px var(--c),0 14px 24px rgba(0,0,0,.45);transition:transform .18s cubic-bezier(.3,1.6,.5,1),box-shadow .18s}
.ddh-med svg{width:118px!important;height:118px!important}
.ddh-med::after{content:"";position:absolute;inset:0;border-radius:28px;background:linear-gradient(115deg,transparent 30%,rgba(255,255,255,.28) 45%,transparent 60%);background-size:250% 100%;background-position:120% 0;transition:background-position .6s}
.ddh-card:hover .ddh-med{transform:translateY(-10px) scale(1.08);box-shadow:0 0 0 3px var(--gl),0 0 0 6px var(--gh),0 0 0 9px var(--gl),0 0 0 13px var(--c),0 0 34px 10px var(--c),0 22px 30px rgba(0,0,0,.45)}
.ddh-card:hover .ddh-med::after{background-position:-40% 0}
.ddh-name{margin-top:18px;text-align:center;font-size:19px;line-height:1;color:var(--y);-webkit-text-stroke:3px var(--ink);paint-order:stroke fill;text-shadow:0 3px 0 var(--ink);white-space:nowrap}
.ddh-kick{position:absolute;top:-12px;left:50%;transform:translateX(-50%) rotate(-4deg);font-size:12px;letter-spacing:.06em;background:var(--c);border:2px solid var(--ink);border-radius:7px;padding:3px 9px 0;box-shadow:0 3px 0 var(--ink);white-space:nowrap;text-shadow:0 2px 0 rgba(0,0,0,.35)}
.ddh-top{position:absolute;left:24px;top:20px;display:flex;align-items:center;gap:14px;padding:10px 22px 8px 10px;border-radius:999px;background:rgba(11,24,56,.78);
  box-shadow:0 0 0 3px var(--gl),0 0 0 6px var(--gh),0 0 0 9px var(--gl),0 10px 20px rgba(0,0,0,.4)}
.ddh-top img{width:58px;height:58px;object-fit:contain;filter:drop-shadow(0 3px 4px rgba(0,0,0,.6))}
.ddh-team{font-size:26px;color:var(--y);-webkit-text-stroke:3px var(--ink);paint-order:stroke fill;text-shadow:0 3px 0 var(--ink);line-height:1}
.ddh-meta{font-family:Nunito,Inter,sans-serif;font-weight:900;font-size:13px;letter-spacing:.06em;color:#bfe9ff;margin-top:4px}
.ddh-cap{position:absolute;right:24px;top:24px;text-align:right;padding:10px 20px 8px;border-radius:16px;background:linear-gradient(180deg,#2fd35a,#138a35);border:3px solid var(--ink);box-shadow:0 5px 0 var(--ink),0 0 24px rgba(47,211,90,.5)}
.ddh-cap small{display:block;font-size:12px;letter-spacing:.1em;color:#eaffef}
.ddh-cap b{font-weight:400;font-size:30px;-webkit-text-stroke:2px var(--ink);paint-order:stroke fill;text-shadow:0 3px 0 var(--ink)}
.ddh-go{position:absolute;right:28px;bottom:58px;display:flex;flex-direction:column;align-items:flex-end;gap:12px}
.ddh-btn{font-family:inherit;font-size:30px;letter-spacing:.04em;color:#ffe7c4;padding:18px 34px 13px;cursor:pointer;border:3px solid;border-color:var(--gh) var(--g) var(--gl) var(--g);border-radius:8px;
  text-shadow:0 3px 0 #5e1606,0 4px 8px rgba(0,0,0,.6);background:linear-gradient(180deg,#ff6a2a,#c2330f 55%,#5e1606);
  box-shadow:0 0 0 2px var(--gl),0 0 0 4px var(--gh),0 0 0 6px var(--gl),inset 0 3px 0 rgba(255,200,150,.4),0 10px 18px rgba(0,0,0,.45),0 0 34px rgba(255,120,40,.55);transition:transform .12s,filter .12s;animation:ddhPulse 2.2s ease-in-out infinite}
.ddh-btn:hover{transform:translateY(-3px);filter:brightness(1.12)}.ddh-btn:active{transform:translateY(2px)}
.ddh-sims{display:flex;gap:8px}
.ddh-chip{font-family:inherit;font-size:15px;color:#fff;padding:9px 14px 6px;border-radius:999px;border:2px solid var(--ink);background:linear-gradient(180deg,#2b8cff,#1554b8);box-shadow:0 3px 0 var(--ink);cursor:pointer;text-shadow:0 2px 0 rgba(0,0,0,.4)}
.ddh-chip:hover{filter:brightness(1.15);transform:translateY(-2px)}
.ddh-next{position:absolute;left:24px;bottom:58px;padding:14px 18px 12px;border-radius:16px;border:4px solid var(--rim);background:rgba(11,24,56,.8);box-shadow:inset 0 0 0 2px rgba(255,140,120,.35),0 10px 18px rgba(0,0,0,.4)}
.ddh-next small{position:absolute;top:-14px;left:12px;font-size:13px;background:var(--rim);border:2px solid var(--ink);border-radius:6px;padding:2px 8px 0}
.ddh-next b{font-weight:400;font-size:22px}.ddh-next span{display:block;font-family:Nunito,sans-serif;font-weight:900;font-size:13px;color:#ffb48a;margin-top:2px}
.ddh-arrow{position:absolute;top:50%;width:64px;height:64px;margin-top:-32px;border-radius:50%;font-size:26px;color:#fff;cursor:pointer;border:3px solid var(--ink);
  background:radial-gradient(circle at 35% 30%,#ffe36b,#ffb000 55%,#c97a00);box-shadow:0 5px 0 var(--ink),0 0 20px rgba(255,200,0,.55);text-shadow:0 2px 0 var(--ink);font-family:inherit}
.ddh-arrow:hover{transform:scale(1.1)}.ddh-arrow:active{transform:translateY(3px)}
.ddh-arrow--l{left:22px}.ddh-arrow--r{right:22px}
.ddh-foot{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);display:flex;gap:20px;font-size:14px;padding:6px 18px 4px;border-radius:999px;background:rgba(0,0,0,.35);text-shadow:0 2px 0 var(--ink)}
.ddh-foot i{font-style:normal;display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;border:2px solid var(--ink);margin-right:6px;background:#2fd35a}
@keyframes ddhPulse{50%{box-shadow:0 0 0 2px var(--gl),0 0 0 4px var(--gh),0 0 0 6px var(--gl),inset 0 3px 0 rgba(255,200,150,.4),0 10px 18px rgba(0,0,0,.45),0 0 54px 6px rgba(255,150,60,.75)}}
@media (prefers-reduced-motion:reduce){.ddh-btn{animation:none}}
`;
root.appendChild(style);

/* ---------- renderer / scene ---------- */
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
root.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog("#7fb8ff", 34, 80);
const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.1, 300);
const camBase = new THREE.Vector3(0, 13.5, 28);
const lookBase = new THREE.Vector3(0, 2.4, -2);
camera.position.copy(camBase);

/* sky: saturated arcade gradient */
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(150, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { t: { value: 0 } },
    vertexShader: `varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `varying vec3 vP; uniform float t;
      void main(){ float h = vP.y;
        vec3 top=vec3(.05,.13,.55), mid=vec3(.13,.62,1.), hor=vec3(1.,.62,.48), pink=vec3(1.,.36,.68);
        vec3 c = mix(hor, mid, smoothstep(.0,.28,h)); c = mix(c, top, smoothstep(.28,.9,h));
        c = mix(c, pink, smoothstep(.12,0.,abs(h-.02))*.55*(.6+.4*sin(vP.x*3.+t*.2)));
        gl_FragColor = vec4(c,1.); }`,
  })
));

/* lights: hemisphere + warm key + coloured rims */
scene.add(new THREE.HemisphereLight("#bfe6ff", "#3a1a6a", 1.1));
const key = new THREE.DirectionalLight("#fff1d6", 2.2);
key.position.set(8, 18, 10);
scene.add(key);
const rimA = new THREE.PointLight("#16d9c9", 60, 40); rimA.position.set(-14, 6, -6); scene.add(rimA);
const rimB = new THREE.PointLight("#ff4fa3", 60, 40); rimB.position.set(14, 6, -6); scene.add(rimB);

/* ---------- helpers ---------- */
const radialTex = (inner, outer, size = 128) => {
  const c = document.createElement("canvas"); c.width = c.height = size;
  const g = c.getContext("2d"), gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gr.addColorStop(0, inner); gr.addColorStop(1, outer); g.fillStyle = gr; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
};
const shadowTex = radialTex("rgba(10,10,40,.55)", "rgba(10,10,40,0)");
function contactShadow(r) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(r * 2, r * 2), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; return m;
}
/* rounded, bevelled platform from a lathe profile (procedural, no hard edges) */
function platformGeo(r, h, bevel) {
  const pts = [new THREE.Vector2(0, 0)];
  for (let i = 0; i <= 8; i++) { const a = (i / 8) * Math.PI / 2; pts.push(new THREE.Vector2(r - bevel + Math.sin(a) * bevel, bevel - Math.cos(a) * bevel)); }
  for (let i = 0; i <= 8; i++) { const a = (i / 8) * Math.PI / 2; pts.push(new THREE.Vector2(r - bevel + Math.cos(a) * bevel, h - bevel + Math.sin(a) * bevel)); }
  pts.push(new THREE.Vector2(0, h));
  return new THREE.LatheGeometry(pts, 48);
}

/* ---------- island: gently domed ice with gradient vertex colours ---------- */
{
  const g = new THREE.CircleGeometry(19, 96, 0, Math.PI * 2);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position, col = [];
  const cIn = new THREE.Color("#e9fbff"), cMid = new THREE.Color("#7fe0ff"), cOut = new THREE.Color("#2b6bff");
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i), d = Math.hypot(x, z) / 19;
    pos.setY(i, (1 - d * d) * 0.6 + Math.sin(x * 0.6) * Math.cos(z * 0.5) * 0.05);
    const c = d < 0.5 ? cIn.clone().lerp(cMid, d / 0.5) : cMid.clone().lerp(cOut, (d - 0.5) / 0.5);
    col.push(c.r, c.g, c.b);
  }
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.25, metalness: 0.05 })));
  /* island rim: thick candy bevel */
  const rim = new THREE.Mesh(new THREE.TorusGeometry(19, 0.55, 16, 128), new THREE.MeshStandardMaterial({ color: "#ffcc00", roughness: 0.3, metalness: 0.3, emissive: "#ff8a1f", emissiveIntensity: 0.25 }));
  rim.rotation.x = Math.PI / 2; rim.position.y = 0.05; scene.add(rim);
  /* faint rink rings on the ice */
  for (const [r, c] of [[5.2, "#ff3b3b"], [9.5, "#2b8cff"]]) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.12, 96), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.45 }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.62 - (r / 17) ** 2 * 0.6 + 0.02; scene.add(ring);
  }
}

/* ---------- race track ribbon around the island (spline + red/white curbs) ---------- */
{
  const ctrl = [];
  for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2; const r = 23 + Math.sin(a * 3) * 1.6; ctrl.push(new THREE.Vector3(Math.cos(a) * r, -0.35 + Math.sin(a * 2) * 0.25, Math.sin(a) * r)); }
  const curve = new THREE.CatmullRomCurve3(ctrl, true, "catmullrom", 0.5);
  const N = 400, W = 2.4, pos = [], col = [], idx = [];
  const road = new THREE.Color("#5a2bd6"), red = new THREE.Color("#ff2d2d"), white = new THREE.Color("#ffffff");
  for (let i = 0; i <= N; i++) {
    const t = i / N, p = curve.getPointAt(t), tan = curve.getTangentAt(t), side = new THREE.Vector3(-tan.z, 0, tan.x).normalize();
    const curb = Math.floor(i / 6) % 2 ? red : white;
    [[-W, curb], [-W + 0.35, road], [W - 0.35, road], [W, curb]].forEach(([o, c]) => { const v = p.clone().addScaledVector(side, o); pos.push(v.x, v.y, v.z); col.push(c.r, c.g, c.b); });
    if (i < N) for (let k = 0; k < 3; k++) { const a = i * 4 + k, b = a + 4; idx.push(a, b, a + 1, a + 1, b, b + 1); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx); g.computeVertexNormals();
  scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide })));
}

/* ---------- centre: gold pedestal + team crest ---------- */
const crest = new THREE.Group();
{
  const ped = new THREE.Mesh(platformGeo(2.6, 1.2, 0.35), new THREE.MeshStandardMaterial({ color: "#ffcc33", metalness: 0.75, roughness: 0.25, emissive: "#ff9a1f", emissiveIntensity: 0.2 }));
  ped.position.y = 0.4; crest.add(ped);
  const ped2 = new THREE.Mesh(platformGeo(1.9, 0.6, 0.25), new THREE.MeshStandardMaterial({ color: "#ff3b3b", metalness: 0.3, roughness: 0.35 }));
  ped2.position.y = 1.6; crest.add(ped2);
  const sh = contactShadow(4.2); sh.position.y = 0.62; crest.add(sh);
  if (TEAM.logo) {
    const tex = new THREE.TextureLoader().load(TEAM.logo); tex.colorSpace = THREE.SRGBColorSpace;
    const logo = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
    logo.scale.set(4.4, 4.4, 1); logo.position.y = 4.9; logo.name = "logo"; crest.add(logo);
  }
  /* glow halo behind crest */
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTex("rgba(255,240,150,.9)", "rgba(255,180,60,0)"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  halo.scale.set(8.5, 8.5, 1); halo.position.y = 4.9; crest.add(halo);
  /* animated selection rings */
  for (let i = 0; i < 2; i++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(3 + i * 0.5, 0.07, 8, 96, Math.PI * 1.4), new THREE.MeshBasicMaterial({ color: i ? "#16d9c9" : "#ffe23a" }));
    ring.position.y = 4.9; ring.name = "ring" + i; crest.add(ring);
  }
  /* spotlight beams */
  for (const x of [-1, 1]) {
    const beam = new THREE.Mesh(new THREE.ConeGeometry(2.6, 16, 32, 1, true), new THREE.MeshBasicMaterial({ color: "#fff3b0", transparent: true, opacity: 0.08, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    beam.position.set(x * 2.4, 11, -1); beam.rotation.z = x * 0.22; crest.add(beam);
  }
}
crest.position.set(0, 0, -5);
crest.scale.setScalar(1.25);
scene.add(crest);

/* ---------- station platforms on an arc ---------- */
const N = stations.length;
/* Turntable ring around the crest (Hick's law: ~5 stations face the player, the rest rotate in) */
const RING_C = new THREE.Vector3(0, 0, -5), RING_R = 13;
let spin = 0, spinGoal = 0;
const arc = stations.map((s, i) => {
  const g = new THREE.Group();
  g.position.set(0, 0.25, 0);
  const mat = new THREE.MeshStandardMaterial({ color: s.color, roughness: 0.32, metalness: 0.15, emissive: s.color, emissiveIntensity: 0.12 });
  const plat = new THREE.Mesh(platformGeo(1.7, 0.7, 0.3), mat); g.add(plat);
  const cap = new THREE.Mesh(platformGeo(1.25, 0.18, 0.08), new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.2 }));
  cap.position.y = 0.7; g.add(cap);
  const sh = contactShadow(2.6); sh.position.y = 0.42; scene.add(sh); sh.position.x = g.position.x; sh.position.z = g.position.z;
  const ring = new THREE.Mesh(new THREE.RingGeometry(1.9, 2.15, 64), new THREE.MeshBasicMaterial({ color: "#ffe23a", transparent: true, opacity: 0, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.95; g.add(ring);
  scene.add(g);

  const card = document.createElement("div");
  card.className = "ddh-card";
  card.style.setProperty("--c", s.color);
  card.innerHTML = `<div class="ddh-med">${s.kicker ? `<span class="ddh-kick">${s.kicker}</span>` : ""}</div><div class="ddh-name">${s.name}</div>`;
  card.querySelector(".ddh-med").appendChild(s.svg);
  root.appendChild(card);
  const st = { g, plat, ring, sh, card, mat, i, hover: 0, vel: 0, phase: i * 1.7 };
  card.addEventListener("pointerenter", () => (hovered = st));
  card.addEventListener("pointerleave", () => hovered === st && (hovered = null));
  return st;
});
let hovered = null;

/* ---------- Double Dash item boxes ---------- */
const qTex = (() => {
  const c = document.createElement("canvas"); c.width = c.height = 128; const g = c.getContext("2d");
  g.fillStyle = "rgba(255,255,255,.18)"; g.fillRect(0, 0, 128, 128);
  g.strokeStyle = "#fff"; g.lineWidth = 8; g.strokeRect(6, 6, 116, 116);
  g.font = "900 92px Luckiest Guy, Arial Black"; g.textAlign = "center"; g.textBaseline = "middle";
  g.lineWidth = 10; g.strokeStyle = "#1a0d05"; g.strokeText("?", 64, 70); g.fillStyle = "#fff"; g.fillText("?", 64, 70);
  return new THREE.CanvasTexture(c);
})();
const boxMat = new THREE.ShaderMaterial({
  transparent: true, uniforms: { t: { value: 0 }, map: { value: qTex } },
  vertexShader: `varying vec2 vUv; varying vec3 vN; void main(){ vUv=uv; vN=normalize(normalMatrix*normal); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader: `varying vec2 vUv; varying vec3 vN; uniform float t; uniform sampler2D map;
    vec3 hue(float h){ return clamp(abs(mod(h*6.+vec3(0,4,2),6.)-3.)-1.,0.,1.); }
    void main(){ vec3 c = hue(fract(vUv.x*.5+vUv.y*.5+t*.25)); vec4 q = texture2D(map,vUv);
      float fres = pow(1.-abs(vN.z),2.); c = mix(c*.85, q.rgb, q.a*.9) + fres*.5;
      gl_FragColor = vec4(c, .82); }`,
});
const boxes = [[-6, 6.2, -9], [7, 7, -10], [-15, 4.5, 4], [15.5, 5, 3], [0, 9.5, -14]].map(([x, y, z], i) => {
  const b = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.5, 1.5), boxMat);
  b.position.set(x, y, z); b.userData.y = y; b.userData.p = i * 1.3; scene.add(b);
  return b;
});

/* ---------- confetti sparkles ---------- */
const spark = (() => {
  const n = 260, p = new Float32Array(n * 3), c = new Float32Array(n * 3), col = new THREE.Color();
  for (let i = 0; i < n; i++) {
    p.set([(Math.random() - 0.5) * 50, Math.random() * 18 + 1, (Math.random() - 0.5) * 40 - 4], i * 3);
    col.set(PALETTE[i % PALETTE.length]); c.set([col.r, col.g, col.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.22, vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(pts); return pts;
})();

/* ---------- HUD (DOM) ---------- */
root.insertAdjacentHTML("beforeend", `
  <div class="ddh-top">${TEAM.logo ? `<img src="${TEAM.logo}" alt="">` : ""}<div><div class="ddh-team">${TEAM.name}</div><div class="ddh-meta">${TEAM.date} · ${TEAM.record}</div></div></div>
  <div class="ddh-cap"><small>CAP SPACE</small><b>${TEAM.cap}</b></div>
  <div class="ddh-next"><small>NEXT UP</small><b>PRESEASON</b><span>No game listed yet</span></div>
  <div class="ddh-go"><div class="ddh-sims"><button class="ddh-chip">7 DAYS</button><button class="ddh-chip">15 DAYS</button><button class="ddh-chip">30 DAYS</button><button class="ddh-chip">SEASON</button></div>
    <button class="ddh-btn">▶ ADVANCE DAY</button></div>
  <button class="ddh-arrow ddh-arrow--l" aria-label="Previous stations">◀</button><button class="ddh-arrow ddh-arrow--r" aria-label="Next stations">▶</button>
  <div class="ddh-foot"><span><i>A</i>ENTER</span><span><i style="background:#ffcc00">M</i>DIRECTORY</span><span><i style="background:#2b8cff">◀▶</i>SPIN</span></div>`);

/* ---------- interaction / animation ---------- */
const step = (d) => (spinGoal += d * (Math.PI * 2) / N);
root.querySelector(".ddh-arrow--l").onclick = () => step(1);
root.querySelector(".ddh-arrow--r").onclick = () => step(-1);
addEventListener("keydown", (e) => { if (!document.getElementById("dd-hub")) return; if (e.key === "ArrowLeft") step(1); if (e.key === "ArrowRight") step(-1); });
const mouse = new THREE.Vector2();
addEventListener("pointermove", (e) => mouse.set(e.clientX / innerWidth - 0.5, e.clientY / innerHeight - 0.5));
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clock = new THREE.Clock(), look = lookBase.clone(), tmp = new THREE.Vector3();
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

function frame() {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime * (reduce ? 0 : 1);
  scene.children[0].material.uniforms.t.value = t;
  boxMat.uniforms.t.value = t;

  /* camera: drift + parallax + ease toward hovered station */
  const target = hovered ? hovered.g.position : lookBase;
  look.x = damp(look.x, target.x * (hovered ? 0.35 : 1), 3, dt);
  look.y = damp(look.y, hovered ? 2.6 : lookBase.y, 3, dt);
  look.z = damp(look.z, hovered ? target.z * 0.3 : lookBase.z, 3, dt);
  camera.position.x = damp(camera.position.x, camBase.x + mouse.x * 2.2 + Math.sin(t * 0.13) * 0.6, 2, dt);
  camera.position.y = damp(camera.position.y, camBase.y - mouse.y * 1.2 + Math.sin(t * 0.17) * 0.25, 2, dt);
  camera.position.z = camBase.z;
  camera.lookAt(look);

  /* crest */
  crest.getObjectByName("ring0").rotation.z = t * 0.8;
  crest.getObjectByName("ring1").rotation.z = -t * 0.55;
  crest.getObjectByName("ring1").rotation.x = 0.3 * Math.sin(t * 0.5);
  const logo = crest.getObjectByName("logo"); if (logo) logo.position.y = 4.9 + Math.sin(t * 1.4) * 0.15;

  /* stations: layered idle sine + damped spring on hover */
  spin = damp(spin, spinGoal, 5, dt);
  for (const s of arc) {
    const ang = Math.PI / 2 + (s.i / N) * Math.PI * 2 + spin;
    s.g.position.x = RING_C.x + Math.cos(ang) * RING_R;
    s.g.position.z = RING_C.z + Math.sin(ang) * RING_R * 0.82;
    s.sh.position.set(s.g.position.x, 0.42, s.g.position.z);
    const front = THREE.MathUtils.smoothstep(Math.sin(ang), -0.35, 0.25);
    s.card.style.opacity = front.toFixed(2);
    s.card.style.pointerEvents = front > 0.4 ? "auto" : "none";
    const goal = hovered === s ? 1 : 0;
    s.vel += (goal - s.hover) * 120 * dt; s.vel *= Math.exp(-14 * dt); s.hover += s.vel * dt;
    const bob = Math.sin(t * 1.1 + s.phase) * 0.08 + Math.sin(t * 2.3 + s.phase * 0.7) * 0.03;
    s.g.position.y = 0.25 + bob + s.hover * 0.7;
    s.g.scale.setScalar(1 + s.hover * 0.08);
    s.mat.emissiveIntensity = 0.12 + s.hover * 0.6;
    s.ring.material.opacity = s.hover; s.ring.rotation.z = t * 2;
    s.sh.material.opacity = 1 - s.hover * 0.45; s.sh.scale.setScalar(1 + s.hover * 0.3);

    tmp.copy(s.g.position); tmp.y += 1.1; tmp.project(camera);
    const x = (tmp.x * 0.5 + 0.5) * innerWidth, y = (-tmp.y * 0.5 + 0.5) * innerHeight;
    const dist = camera.position.distanceTo(s.g.position), sc = THREE.MathUtils.clamp(22 / dist, 0.6, 1.12);
    s.card.style.transform = `translate(${x - 75}px,${y - 210}px) scale(${sc})`;
    s.card.style.zIndex = String(1000 - Math.round(dist * 10));
  }
  for (const b of boxes) { b.rotation.x = t * 0.7 + b.userData.p; b.rotation.y = t * 0.9; b.position.y = b.userData.y + Math.sin(t * 1.3 + b.userData.p) * 0.4; }
  spark.rotation.y = t * 0.02;

  renderer.render(scene, camera);
  if (document.getElementById("dd-hub")) requestAnimationFrame(frame);
}
frame();
