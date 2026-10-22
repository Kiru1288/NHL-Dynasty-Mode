/* Concept v2: "Front Office Gallery" — toned-down Double Dash hub.
   Large 2D station plaques hung on a curved 3D wall. Real silhouettes are cloned
   from the live hub (.office-menu-icon svg); Franchise Command, Calendar and
   League Ops get matching hand-drawn silhouettes [NEW]. Every old HUD element
   (top bar, war-room tab, desk alerts, wire ticker, directory, sims, chemistry,
   key hints) is kept, restyled. Mockup only. */
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const T = Object.assign({ name: "Toronto Maple Leafs", date: "2026-09-15 (Tue)", phase: "Preseason", record: "0-0-0", cap: "$13.31M", next: "No game listed", wire: "League wire quiet — advance a day to fill the crawl.", logo: "" }, (window.__ddHub || {}).team);

/* ---------- station art ---------- */
const live = {};
document.querySelectorAll(".office-menu-icon").forEach((el) => {
  const svg = el.querySelector("svg"); if (!svg) return;
  live[el.querySelector(".office-menu-icon__name")?.textContent.trim()] = {
    svg: svg.outerHTML, kicker: el.querySelector(".office-menu-icon__kicker")?.textContent.trim() || "",
    accent: el.style.getPropertyValue("--menu-accent") || "#c4a46a" };
});
const DRAWN = {
  "Franchise Command": { kicker: "Command", accent: "#9fd6ea", svg: `<svg viewBox="0 0 440 440"><defs><linearGradient id="fcBody" x1="50%" y1="0%" x2="50%" y2="100%"><stop offset="0%" stop-color="#1f3442"/><stop offset="100%" stop-color="#04080c"/></linearGradient><radialGradient id="fcGlow" cx="50%" cy="45%" r="55%"><stop offset="0%" stop-color="#9fd6ea" stop-opacity=".35"/><stop offset="100%" stop-color="#9fd6ea" stop-opacity="0"/></radialGradient></defs><ellipse cx="220" cy="230" rx="180" ry="150" fill="url(#fcGlow)"/><path d="M110 120 L330 120 L330 280 L110 280 Z" fill="url(#fcBody)" stroke="#9fd6ea" stroke-width="2.4" stroke-opacity=".45" rx="8"/><path d="M128 138 L312 138 L312 262 L128 262 Z" fill="#0b1a24" stroke="#9fd6ea" stroke-opacity=".25"/><path d="M146 230 L186 196 L222 214 L262 170 L296 184" fill="none" stroke="#dff6ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity=".8"/><circle cx="262" cy="170" r="7" fill="#9fd6ea"/><path d="M80 290 L360 290 L384 330 L56 330 Z" fill="url(#fcBody)" stroke="#9fd6ea" stroke-width="2" stroke-opacity=".35"/><path d="M190 306 L250 306" stroke="#9fd6ea" stroke-width="5" stroke-opacity=".35" stroke-linecap="round"/></svg>` },
  "Calendar": { kicker: "Season Timeline", accent: "#dbe7ef", svg: `<svg viewBox="0 0 440 440"><defs><linearGradient id="caBody" x1="50%" y1="0%" x2="50%" y2="100%"><stop offset="0%" stop-color="#26343e"/><stop offset="100%" stop-color="#05090d"/></linearGradient></defs><path d="M96 118 L344 118 L344 362 L96 362 Z" fill="url(#caBody)" stroke="#dbe7ef" stroke-width="2.4" stroke-opacity=".4"/><path d="M96 118 L344 118 L344 168 L96 168 Z" fill="#c0554d" opacity=".55"/><path d="M150 96 L150 140 M290 96 L290 140" stroke="#dbe7ef" stroke-width="9" stroke-linecap="round" opacity=".55"/>${[0, 1, 2, 3].map((r) => [0, 1, 2, 3, 4].map((c) => `<rect x="${116 + c * 44}" y="${186 + r * 42}" width="30" height="28" rx="4" fill="#dbe7ef" opacity="${r === 1 && c === 2 ? 0.95 : 0.12}"/>`).join("")).join("")}<circle cx="220" cy="242" r="24" fill="none" stroke="#e0a06a" stroke-width="3.5"/></svg>` },
  "League Operations": { kicker: "League Office", accent: "#7fb8f0", svg: `<svg viewBox="0 0 440 440"><defs><linearGradient id="loSuit" x1="50%" y1="0%" x2="50%" y2="100%"><stop offset="0%" stop-color="#1a2c3c"/><stop offset="100%" stop-color="#02050a"/></linearGradient><radialGradient id="loHalo" cx="50%" cy="30%" r="50%"><stop offset="0%" stop-color="#7fb8f0" stop-opacity=".45"/><stop offset="100%" stop-color="#7fb8f0" stop-opacity="0"/></radialGradient></defs><ellipse cx="220" cy="150" rx="150" ry="120" fill="url(#loHalo)"/><path d="M220 96 c20 0 36 16 36 38 c0 16 -8 28 -19 34 c38 10 62 40 66 86 l10 146 l-186 0 l10 -146 c4 -46 28 -76 66 -86 c-11 -6 -19 -18 -19 -34 c0 -22 16 -38 36 -38 z" fill="url(#loSuit)" stroke="#7fb8f0" stroke-width="2.2" stroke-opacity=".4"/><path d="M220 176 l-14 26 l14 20 l14 -20 z" fill="#dbe7ef" opacity=".25"/><path d="M196 186 L220 260 L244 186" fill="none" stroke="#7fb8f0" stroke-width="2" stroke-opacity=".3"/></svg>` },
};
const pick = (n) => live[n] || DRAWN[n] || { svg: "", kicker: "", accent: "#c4a46a" };
const TOP = ["Draft Class", "Trade Hub", "Roster Board", null, "Strategy Board", "Stats", "League Operations"];
const BOT = ["Franchise Command", "Calendar", "Contracts", "Franchise Identity", "Standings", "Storylines", "Legacy Wall"];

/* ---------- DOM shell ---------- */
document.getElementById("dd-hub")?.remove();
const root = document.createElement("div");
root.id = "dd-hub";
document.body.appendChild(root);
root.insertAdjacentHTML("beforeend", `<style>
#dd-hub{position:fixed;inset:0;z-index:99999;overflow:hidden;font-family:"Luckiest Guy",cursive;color:#fff;
  --ink:#1a0d05;--y:#ffe23a;--gh:#f8d58a;--g:#c98a3c;--gl:#6e4314;--rim:#e2231a;--teal:#2ee6d6;--peach:#ffb48a;--glass:rgba(14,30,72,.84);--ui:Nunito,Inter,sans-serif}
#dd-hub canvas{position:absolute;inset:0}
.gx-plaque{position:absolute;left:0;top:0;width:var(--w);transform-origin:50% 50%;cursor:pointer;will-change:transform}
.gx-face{position:relative;height:calc(var(--w) * .92);border-radius:16px;display:grid;place-items:center;background:linear-gradient(180deg,rgba(255,180,138,.07),rgba(8,18,46,.92) 70%),var(--glass);
  box-shadow:0 0 0 2px var(--gl),0 0 0 4px var(--gh),0 0 0 6px var(--gl),inset 0 0 0 2px color-mix(in srgb,var(--a) 45%,transparent),inset 0 -30px 40px rgba(0,0,0,.35),0 16px 26px rgba(0,0,0,.45);
  transition:transform .2s cubic-bezier(.3,1.5,.5,1),box-shadow .2s;overflow:hidden}
.gx-face svg{width:84%!important;height:84%!important}
.gx-face::after{content:"";position:absolute;inset:0;background:linear-gradient(115deg,transparent 35%,rgba(255,255,255,.16) 48%,transparent 60%);background-size:260% 100%;background-position:130% 0;transition:background-position .7s ease}
.gx-kick{position:absolute;top:-11px;left:12px;z-index:2;font-size:11px;letter-spacing:.07em;color:var(--ink);background:var(--a);border:2px solid var(--ink);border-radius:6px;padding:3px 8px 0;box-shadow:0 2px 0 var(--ink);white-space:nowrap}
.gx-name{margin-top:12px;text-align:center;font-size:17px;line-height:1;color:var(--y);-webkit-text-stroke:2.5px var(--ink);paint-order:stroke fill;text-shadow:0 3px 0 var(--ink);white-space:nowrap}
.gx-plaque:hover .gx-face{transform:translateY(-6px);box-shadow:0 0 0 2px var(--gl),0 0 0 4px var(--gh),0 0 0 6px var(--gl),inset 0 0 0 2px var(--a),0 0 30px 4px color-mix(in srgb,var(--a) 55%,transparent),0 22px 30px rgba(0,0,0,.45)}
.gx-plaque:hover .gx-face::after{background-position:-40% 0}
.gx-crest .gx-face{border-radius:50%;background:radial-gradient(circle at 50% 38%,rgba(255,226,140,.28),rgba(10,22,56,.95) 70%)}
.gx-crest .gx-face img{width:78%;height:78%;object-fit:contain;filter:drop-shadow(0 6px 10px rgba(0,0,0,.55))}
.gx-crest .gx-name{font-size:20px}
.gx-hud{position:absolute;left:0;right:0;top:0;height:54px;display:flex;align-items:center;gap:14px;padding:0 18px;background:linear-gradient(180deg,rgba(10,22,54,.92),rgba(10,22,54,.75));box-shadow:0 2px 0 var(--g),0 4px 0 var(--gl),0 8px 18px rgba(0,0,0,.35)}
.gx-hud img{width:38px;height:38px;object-fit:contain}
.gx-team{font-size:22px;color:var(--y);-webkit-text-stroke:2px var(--ink);paint-order:stroke fill;text-shadow:0 2px 0 var(--ink)}
.gx-office{font-size:12px;letter-spacing:.12em;color:var(--teal)}
.gx-pill{font-family:var(--ui);font-weight:900;font-size:12px;letter-spacing:.08em;text-transform:uppercase;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.06);border:1.5px solid rgba(255,180,138,.35);color:#e6eef6}
.gx-pill b{color:var(--peach)}.gx-pill.cap{border-color:rgba(82,223,148,.5)}.gx-pill.cap b{color:#7dfcb0}
.gx-grow{flex:1}
.gx-edge{position:absolute;left:0;top:50%;transform:translateY(-50%);writing-mode:vertical-rl;rotate:180deg;font-family:inherit;font-size:15px;letter-spacing:.06em;color:#ffe7c4;padding:18px 9px;cursor:pointer;
  border:3px solid;border-left:0;border-color:var(--gh) var(--g) var(--gl) var(--g);border-radius:0 10px 10px 0;background:linear-gradient(90deg,#183a94,#2f5fd0);box-shadow:0 0 0 2px var(--gl),0 8px 16px rgba(0,0,0,.4);text-shadow:0 2px 0 #071440}
.gx-alert{position:absolute;left:18px;top:66px;font-family:var(--ui);font-weight:900;font-size:12px;letter-spacing:.1em;text-transform:uppercase;padding:7px 12px;border-radius:999px;background:rgba(10,22,54,.8);border:2px solid #52df94;color:#a8ffc9}
.gx-alert::before{content:"✓ ";color:#52df94}
.gx-bar{position:absolute;left:18px;right:18px;bottom:44px;display:flex;align-items:center;gap:12px}
.gx-btn{font-family:inherit;font-size:24px;letter-spacing:.04em;color:#ffe7c4;padding:14px 26px 10px;cursor:pointer;border:3px solid;border-color:var(--gh) var(--g) var(--gl) var(--g);border-radius:7px;
  text-shadow:0 2px 0 #5e1606,0 3px 6px rgba(0,0,0,.6);background:linear-gradient(180deg,#e0521f,#a8300f 55%,#5e1606);box-shadow:0 0 0 2px var(--gl),0 0 0 4px var(--gh),0 0 0 6px var(--gl),inset 0 2px 0 rgba(255,200,150,.35),0 8px 14px rgba(0,0,0,.5);transition:transform .12s,filter .12s}
.gx-btn.sec{font-size:18px;padding:11px 18px 8px;background:linear-gradient(180deg,#2f5fd0,#183a94 55%,#0a1d55);text-shadow:0 2px 0 #071440}
.gx-btn:hover{transform:translateY(-2px);filter:brightness(1.1)}.gx-btn:active{transform:translateY(2px)}
.gx-chip{font-family:inherit;font-size:14px;color:#e6eef6;padding:9px 13px 6px;border-radius:999px;cursor:pointer;background:rgba(10,22,54,.85);border:2px solid rgba(46,230,214,.4)}
.gx-chip:hover{border-color:var(--teal);color:#fff}
.gx-quiet{font-family:inherit;font-size:16px;color:var(--peach);background:none;border:0;cursor:pointer;text-shadow:0 2px 0 var(--ink);padding:8px}
.gx-quiet:hover{color:#fff}
.gx-wire{position:absolute;left:0;right:0;bottom:0;height:34px;display:flex;align-items:center;background:rgba(6,14,36,.92);border-top:2px solid var(--g)}
.gx-wire span.flag{height:100%;display:grid;place-items:center;padding:0 16px;font-size:15px;background:var(--rim);box-shadow:inset 0 -3px 0 rgba(0,0,0,.25)}
.gx-wire .crawl{flex:1;overflow:hidden;white-space:nowrap;font-family:var(--ui);font-weight:900;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#dbe7ef;padding-left:16px;
  -webkit-mask:linear-gradient(90deg,#000 85%,transparent)}
.gx-wire .keys{padding:0 16px;font-family:var(--ui);font-weight:800;font-size:11px;letter-spacing:.08em;color:rgba(219,231,239,.6);white-space:nowrap}
</style>`);

/* ---------- 3D room ---------- */
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
root.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0c1a3e");
scene.fog = new THREE.Fog("#0c1a3e", 18, 42);
const camera = new THREE.PerspectiveCamera(46, innerWidth / innerHeight, 0.1, 100);
const camBase = new THREE.Vector3(0, 3.9, 13), look = new THREE.Vector3(0, 3.85, -8);

/* curved back wall: cylinder segment seen from inside, panel grooves + vertical gradient in shader */
const WALL_R = 22, WALL_CZ = 13;            // wall centre is behind the camera, so it curves toward us at the edges
const wallMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, uniforms: { t: { value: 0 } },
  vertexShader: `varying vec2 vUv; varying vec3 vW; void main(){ vUv=uv; vW=(modelMatrix*vec4(position,1.)).xyz; gl_Position=projectionMatrix*viewMatrix*vec4(vW,1.); }`,
  fragmentShader: `varying vec2 vUv; varying vec3 vW; uniform float t;
    void main(){
      float y = vW.y;
      vec3 base = mix(vec3(.07,.14,.33), vec3(.13,.24,.5), smoothstep(-.5,7.,y));
      float groove = smoothstep(.0,.02,abs(fract(vUv.x*46.)-.5)-.47);          // vertical panel seams
      base *= 1. - groove*.25;
      float rail = smoothstep(.07,.0,abs(y-7.75)) + smoothstep(.05,.0,abs(y-.2)); // brass rails
      base = mix(base, vec3(.85,.62,.3), rail*.8);
      float teal = exp(-pow((vW.x+11.)/7.,2.)) * smoothstep(8.,1.,y) * .22;      // turquoise wash left
      float peach= exp(-pow((vW.x-11.)/7.,2.)) * smoothstep(8.,1.,y) * .22;      // peach wash right
      base += vec3(.18,.9,.84)*teal + vec3(1.,.7,.54)*peach;
      gl_FragColor = vec4(base,1.);
    }`,
});
const wall = new THREE.Mesh(new THREE.CylinderGeometry(WALL_R, WALL_R, 9, 96, 1, true, Math.PI * 0.72, Math.PI * 0.56), wallMat);
wall.position.set(0, 4.5, WALL_CZ);
scene.add(wall);

/* floor: dark polished boards with a soft peach pool */
{
  const g = new THREE.PlaneGeometry(60, 40, 1, 1); g.rotateX(-Math.PI / 2);
  const m = new THREE.ShaderMaterial({
    vertexShader: `varying vec3 vW; void main(){ vW=(modelMatrix*vec4(position,1.)).xyz; gl_Position=projectionMatrix*viewMatrix*vec4(vW,1.); }`,
    fragmentShader: `varying vec3 vW; void main(){ vec3 c = vec3(.05,.09,.2);
      c *= .85 + .15*smoothstep(.0,.04,abs(fract(vW.x*.8)-.5)-.46);
      c += vec3(1.,.62,.45) * exp(-dot(vW.xz-vec2(0.,-2.), vW.xz-vec2(0.,-2.))/60.) * .16;
      c += vec3(.18,.9,.84) * exp(-dot(vW.xz-vec2(-9.,-5.), vW.xz-vec2(-9.,-5.))/30.) * .07;
      gl_FragColor = vec4(c,1.); }`,
  });
  scene.add(new THREE.Mesh(g, m));
}

/* light pools behind each plaque (canvas radial sprites, additive) */
const glowTex = (() => { const c = document.createElement("canvas"); c.width = c.height = 128; const g = c.getContext("2d"); const r = g.createRadialGradient(64, 64, 0, 64, 64, 64); r.addColorStop(0, "rgba(255,255,255,.55)"); r.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = r; g.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c); })();

/* wall anchor for column c (0..6) on row r: point on the cylinder */
const anchor = (col, y) => {
  const ang = THREE.MathUtils.lerp(-0.42, 0.44, col / 6);          // radians across the visible arc
  return new THREE.Vector3(Math.sin(ang) * WALL_R, y, WALL_CZ - Math.cos(ang) * WALL_R + 0.25);
};

const plaques = [];
const make = (name, col, y, crest = false) => {
  const s = crest ? { svg: "", kicker: "", accent: "#ffe23a" } : pick(name);
  const el = document.createElement("div");
  el.className = "gx-plaque" + (crest ? " gx-crest" : "");
  el.style.setProperty("--a", s.accent);
  el.innerHTML = `<div class="gx-face">${s.kicker ? `<span class="gx-kick">${s.kicker}</span>` : ""}${crest ? (T.logo ? `<img src="${T.logo}" alt="">` : "") : s.svg}</div>${crest ? "" : `<div class="gx-name">${name}</div>`}`;
  root.appendChild(el);
  const pos = anchor(col, y);
  const pool = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: new THREE.Color(s.accent), transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }));
  pool.position.copy(pos).add(new THREE.Vector3(0, 0, 0.1)); pool.scale.set(crest ? 6 : 4, crest ? 6 : 4, 1); scene.add(pool);
  const p = { el, pos, pool, col, crest, h: 0 };
  el.addEventListener("pointerenter", () => (p.hot = true));
  el.addEventListener("pointerleave", () => (p.hot = false));
  plaques.push(p);
};
TOP.forEach((n, c) => make(n || "crest", c, 6.05, n === null));
BOT.forEach((n, c) => make(n, c, 1.75));

/* ---------- HUD ---------- */
root.insertAdjacentHTML("beforeend", `
<div class="gx-hud">${T.logo ? `<img src="${T.logo}" alt="">` : ""}<div><div class="gx-office">GM OFFICE</div><div class="gx-team">${T.name}</div></div>
  <div class="gx-grow"></div><span class="gx-pill">${T.date} · <b>${T.phase}</b></span><span class="gx-pill">Record <b>${T.record}</b></span><span class="gx-pill cap">Cap <b>${T.cap}</b></span><span class="gx-pill">Next <b>${T.next}</b></span></div>
<span class="gx-alert">Desk is clear</span>
<button class="gx-edge">DRAFT WAR ROOM</button>
<div class="gx-bar"><button class="gx-btn sec">☰ DIRECTORY</button><button class="gx-quiet">TEAM CHEMISTRY</button><div class="gx-grow"></div>
  <button class="gx-chip">7 DAYS</button><button class="gx-chip">15 DAYS</button><button class="gx-chip">30 DAYS</button><button class="gx-chip">REST OF SEASON</button><button class="gx-btn">▶ ADVANCE ONE DAY</button></div>
<div class="gx-wire"><span class="flag">WIRE</span><div class="crawl">${T.wire}</div><span class="keys">↑↓ SELECT · ENTER CONFIRM · ←→ ADJUST · ESC BACK · M DIRECTORY</span></div>`);

/* ---------- loop: restrained drift, parallax, plaque projection ---------- */
const mouse = new THREE.Vector2();
addEventListener("pointermove", (e) => mouse.set(e.clientX / innerWidth - 0.5, e.clientY / innerHeight - 0.5));
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clock = new THREE.Clock(), v = new THREE.Vector3(), damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));
const plaqueW = () => Math.min(158, innerWidth / 8.6);

function frame() {
  const dt = Math.min(clock.getDelta(), 0.05), t = reduce ? 0 : clock.elapsedTime;
  camera.position.x = damp(camera.position.x, camBase.x + mouse.x * 0.9 + Math.sin(t * 0.11) * 0.18, 2.5, dt);
  camera.position.y = damp(camera.position.y, camBase.y - mouse.y * 0.45 + Math.sin(t * 0.15) * 0.06, 2.5, dt);
  camera.position.z = camBase.z;
  camera.lookAt(look);
  const W = plaqueW();
  for (const p of plaques) {
    p.h = damp(p.h, p.hot ? 1 : 0, 10, dt);
    p.pool.material.opacity = 0.3 + p.h * 0.45 + Math.sin(t * 0.8 + p.col) * 0.03;
    v.copy(p.pos).project(camera);
    const x = (v.x * 0.5 + 0.5) * innerWidth, y = (-v.y * 0.5 + 0.5) * innerHeight;
    const yaw = (p.col - 3) * -5;                       // slight turn so plaques sit flush on the curved wall
    const w = p.crest ? W * 1.12 : W;
    p.el.style.setProperty("--w", w + "px");
    p.el.style.transform = `translate(${x - w / 2}px,${y - w * 0.62}px) perspective(900px) rotateY(${yaw}deg)`;
  }
  renderer.render(scene, camera);
  if (document.getElementById("dd-hub") === root) requestAnimationFrame(frame);
}
frame();
