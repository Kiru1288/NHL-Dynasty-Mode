/**
 * Low-poly PS2-era face generator for generated (non-NHL) players.
 *
 * Builds a faceted mesh (flat-shaded triangles lit from the upper left) in a
 * 100 x 122 space, deterministic per player: skin tone, head shape, hair
 * style/colour, facial hair, expression, helmet / goalie cage, and a jersey
 * collar in team colours. The mesh is rasterised onto a tiny canvas
 * (44 x 54), posterized and outlined, then shown with pixelated scaling, so it
 * sits next to the restyled real NHL photos as one art style.
 */

import { makeCanvas, ps2Process } from "./ps2Pixel";

const SKIN = {
  fair: "#f2c9a8",
  light: "#e2ad85",
  medium: "#c98d62",
  olive: "#b5865a",
  tan: "#a46d42",
  brown: "#7b4a2b",
  deep: "#55311c",
};
const HAIR = {
  black: "#1c1814",
  dark_brown: "#3b2617",
  brown: "#62401f",
  auburn: "#7d3820",
  red: "#ab4a20",
  blond: "#d9b45c",
  dirty_blond: "#a5864a",
  grey: "#9b9a98",
  white: "#dedbd5",
};
const SKIN_KEYS = Object.keys(SKIN);
const HAIR_KEYS = Object.keys(HAIR);
const STYLES = ["swept", "crop", "buzz", "curly", "red_part", "bald", "flow", "spiky", "flat_top", "afro", "fade", "long_blond", "messy_dark", "grey", "helmet"];
const BEARDS = ["none", "none", "none", "stubble", "beard", "mustache", "goatee"];

const LIGHT = (() => {
  const v = [-0.5, -0.55, 0.67];
  const m = Math.hypot(v[0], v[1], v[2]);
  return v.map((x) => x / m);
})();

function hashStr(s) {
  let h = 2166136261;
  const str = String(s);
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function rng(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexToRgb(hex) {
  const h = String(hex || "").replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6), 16);
  if (!Number.isFinite(n)) return [128, 128, 128];
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]) {
  const c = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

function shadeHex(hex, k) {
  const [r, g, b] = hexToRgb(hex);
  if (k <= 1) return rgbToHex([r * k, g * k, b * k]);
  const t = k - 1;
  return rgbToHex([r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t]);
}

function mixHex(a, b, t) {
  const x = hexToRgb(a);
  const y = hexToRgb(b);
  return rgbToHex([x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t]);
}

/** Flat lambert shade of a 3D triangle, quantized to 5 steps. */
function facetShade(a, b, c) {
  const ux = b[0] - a[0];
  const uy = b[1] - a[1];
  const uz = (b[2] - a[2]) * 34;
  const vx = c[0] - a[0];
  const vy = c[1] - a[1];
  const vz = (c[2] - a[2]) * 34;
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  const m = Math.hypot(nx, ny, nz) || 1;
  nx /= m;
  ny /= m;
  nz /= m;
  if (nz < 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
  const d = Math.max(0, nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]);
  const raw = 0.58 + d * 0.62;
  return Math.round(raw * 5) / 5;
}

function tri(out, a, b, c, base, extra = {}) {
  out.push({ p: [a[0], a[1], b[0], b[1], c[0], c[1]], c: shadeHex(base, facetShade(a, b, c)), ...extra });
}

/** Triangle fan from a raised centre to an outline: instant faceted dome. */
function fan(out, outline, center, base, zEdge = 0.1, extra = {}) {
  for (let i = 0; i < outline.length; i += 1) {
    const a = outline[i];
    const b = outline[(i + 1) % outline.length];
    tri(out, center, [a[0], a[1], a[2] ?? zEdge], [b[0], b[1], b[2] ?? zEdge], base, extra);
  }
}

function flat(out, pts, color, extra = {}) {
  out.push({ p: pts.flat(), c: color, ...extra });
}

function mirror(p) {
  return [100 - p[0], p[1], p[2]];
}

/** Read the face traits (falls back to seeded picks when a payload omits them). */
export function faceTraits(player = {}) {
  const seed =
    Number(player.avatar_seed) > 0
      ? Number(player.avatar_seed)
      : hashStr(player.id || player.player_id || player.name || "x");
  const r = rng(seed ^ 0x9e3779b9);
  const pick = (arr) => arr[Math.floor(r() * arr.length) % arr.length];
  const pos = String(player.position || player.pos || "").toUpperCase();
  const skin = SKIN[player.skin_tone] ? player.skin_tone : pick(SKIN_KEYS);
  const hairColor = HAIR[player.hair_color] ? player.hair_color : pick(HAIR_KEYS);
  let style = STYLES.includes(player.hair_style) ? player.hair_style : pick(STYLES);
  const beard = player.facial_hair || pick(BEARDS);
  const expression = player.expression || "neutral";
  const helmetRoll = r();
  let helmet = style === "helmet" || helmetRoll < 0.18;
  const cage = pos === "G" && helmetRoll < 0.55;
  if (cage) helmet = true;
  if (style === "helmet") style = "crop";
  return {
    seed,
    skin,
    hairColor: style === "grey" ? "grey" : style === "long_blond" ? "blond" : hairColor,
    style,
    beard,
    expression,
    helmet,
    cage,
    visor: helmet && !cage && r() < 0.5,
    hw: 0.92 + r() * 0.16,
    jaw: 0.86 + r() * 0.24,
    chinY: 78 + r() * 5,
    top: 16 + r() * 4,
    neck: r(),
    collar: r() < 0.5 ? "v" : "crew",
    yoke: r() < 0.6,
    backdrop: Math.floor(r() * 3),
  };
}

/** Build the polygon list (back to front) for a face. */
export function buildFaceModel(t, colors = {}) {
  const jersey = colors.primary || "#63263b";
  const jersey2 = colors.secondary || "#ffffff";
  const skin = SKIN[t.skin];
  const hair = HAIR[t.hairColor];
  const out = [];
  const L = (x) => 50 - x;
  const { hw, jaw, chinY, top } = t;

  // --- hair volume behind head (long styles)
  if (!t.helmet && ["flow", "long_blond", "afro", "curly", "messy_dark"].includes(t.style)) {
    const long = t.style === "flow" || t.style === "long_blond";
    const big = t.style === "afro" ? 9 : t.style === "curly" ? 5 : 3;
    const low = long ? 70 : t.style === "afro" ? 58 : 52;
    fan(
      out,
      [
        [L(27 * hw + big), low],
        [L(30 * hw + big), 30],
        [L(18 * hw + big * 0.6), top - big],
        [50, top - 4 - big],
        [50 + 18 * hw + big * 0.6, top - big],
        [50 + 30 * hw + big, 30],
        [50 + 27 * hw + big, low],
        [50 + 14, low - 6],
        [L(14), low - 6],
      ],
      [50, 34, 0.7],
      shadeHex(hair, 0.82)
    );
  }

  // --- jersey + shoulders
  const shoulders = [
    [0, 122, 0.05],
    [1, 106, 0.1],
    [13, 96, 0.25],
    [33, 89, 0.4],
    [40, 87, 0.5],
    [60, 87, 0.5],
    [67, 89, 0.4],
    [87, 96, 0.25],
    [99, 106, 0.1],
    [100, 122, 0.05],
  ];
  fan(out, shoulders, [50, 116, 0.8], jersey);
  if (t.yoke) {
    flat(out, [[1, 106], [13, 96], [19, 101], [7, 112]], shadeHex(jersey2, 0.95));
    flat(out, [[99, 106], [87, 96], [81, 101], [93, 112]], shadeHex(jersey2, 0.78));
  }

  // --- neck
  const nw = 9 + t.neck * 3;
  tri(out, [L(nw), 72, 0.3], [50, 74, 0.6], [L(nw + 1), 90, 0.3], skin);
  tri(out, [50, 74, 0.6], [50, 95, 0.6], [L(nw + 1), 90, 0.3], skin);
  tri(out, [50 + nw, 72, 0.3], [50, 74, 0.6], [50 + nw + 1, 90, 0.3], skin);
  tri(out, [50, 74, 0.6], [50, 95, 0.6], [50 + nw + 1, 90, 0.3], skin);
  flat(out, [[L(nw), 72], [50 + nw, 72], [50 + 6, 79], [L(6), 79]], shadeHex(skin, 0.62)); // jaw shadow

  // --- collar
  if (t.collar === "v") {
    flat(out, [[39, 87], [50, 102], [61, 87], [57, 87], [50, 97], [43, 87]], jersey2);
    flat(out, [[43, 87], [50, 97], [57, 87]], shadeHex(skin, 0.7));
  } else {
    flat(out, [[37, 87], [50, 93], [63, 87], [64, 91], [50, 98], [36, 91]], jersey2);
    flat(out, [[40, 87], [50, 92], [60, 87]], shadeHex(skin, 0.68));
  }

  // --- ears
  const earL = [[L(22 * hw), 42, 0.2], [L(27.5 * hw), 41, 0.2], [L(27 * hw), 58, 0.2], [L(23 * hw), 54, 0.2]];
  fan(out, earL, [L(25 * hw), 49, 0.35], skin);
  fan(out, earL.map(mirror), [50 + 25 * hw, 49, 0.35], skin);

  // --- head mesh
  const V = {
    T: [50, top, 0.5],
    F: [50, top + 14, 0.82],
    C0: [50, 44, 0.92],
    N: [50, 55, 1.0],
    M: [50, 65, 0.85],
    K: [50, chinY, 0.6],
    FL: [L(16 * hw), top + 5, 0.35],
    TL: [L(23.5 * hw), 37, 0.15],
    BL: [L(11), 41, 0.75],
    EL: [L(11), 48, 0.6],
    CL: [L(22 * hw), 54, 0.2],
    NL: [L(4.5), 56, 0.82],
    QL: [L(13), 62, 0.62],
    JL: [L(16.5 * hw * jaw), 68, 0.28],
    KL: [L(7 * jaw), chinY - 2, 0.55],
  };
  const R = (k) => (k.endsWith("L") ? mirror(V[k]) : V[k]);
  const faces = [
    ["T", "FL", "F"],
    ["FL", "BL", "F"],
    ["FL", "TL", "BL"],
    ["TL", "EL", "BL"],
    ["TL", "CL", "EL"],
    ["F", "BL", "C0"],
    ["BL", "EL", "C0"],
    ["EL", "NL", "C0"],
    ["C0", "NL", "N"],
    ["EL", "CL", "QL"],
    ["EL", "QL", "NL"],
    ["CL", "JL", "QL"],
    ["QL", "JL", "KL"],
    ["QL", "KL", "M"],
    ["QL", "M", "NL"],
    ["NL", "M", "N"],
    ["KL", "K", "M"],
  ];
  faces.forEach(([a, b, c]) => {
    tri(out, V[a], V[b], V[c], skin);
    tri(out, R(a), R(b), R(c), skin);
  });

  // --- facial hair
  const beardBase = t.beard === "grey_beard" ? HAIR.grey : hair;
  const beardOutline = (drop) => [
    [L(22 * hw), 54, 0.2],
    [V.JL[0], V.JL[1] + drop * 0.4, 0.3],
    [V.KL[0], V.KL[1] + drop, 0.5],
    [50, chinY + drop + 1, 0.6],
    [100 - V.KL[0], V.KL[1] + drop, 0.5],
    [100 - V.JL[0], V.JL[1] + drop * 0.4, 0.3],
    [50 + 22 * hw, 54, 0.2],
    [50 + 15, 61, 0.6],
    [50 + 6, 63, 0.8],
    [50, 62, 0.85],
    [L(6), 63, 0.8],
    [L(15), 61, 0.6],
  ];
  if (t.beard === "beard" || t.beard === "grey_beard" || t.beard === "playoff_beard") {
    fan(out, beardOutline(t.beard === "playoff_beard" ? 7 : 2), [50, 72, 0.8], beardBase);
  } else if (t.beard === "stubble") {
    fan(out, beardOutline(0), [50, 72, 0.8], mixHex(skin, hair, 0.45), 0.1, { a: 0.75 });
  } else if (t.beard === "goatee") {
    fan(out, [[44, 66], [56, 66], [56.5, 74], [50, chinY + 2], [43.5, 74]], [50, 71, 0.8], beardBase);
  }
  if (t.beard === "mustache" || t.beard === "beard" || t.beard === "playoff_beard" || t.beard === "grey_beard" || t.beard === "goatee") {
    flat(out, [[43, 61.5], [50, 59.5], [57, 61.5], [58, 64.5], [50, 62.5], [42, 64.5]], shadeHex(beardBase, 0.85));
  }

  // --- nose
  flat(out, [[50, 45], [54, 55.5], [50, 57]], shadeHex(skin, 0.7));
  flat(out, [[47, 56.5], [53, 56.5], [52, 58], [48, 58]], shadeHex(skin, 0.55));

  // --- eyes + brows (expression)
  const ex = t.expression;
  const eye = "#15131c";
  const ey = ex === "tired" ? 47.5 : 46.5;
  const eh = ex === "tired" ? 2 : ex === "focused" || ex === "angry" ? 2.4 : 3;
  flat(out, [[L(14.5), ey], [L(7.5), ey - 0.4], [L(8), ey + eh], [L(14), ey + eh]], eye);
  flat(out, [[50 + 14.5, ey], [50 + 7.5, ey - 0.4], [50 + 8, ey + eh], [50 + 14, ey + eh]], eye);
  flat(out, [[L(12.5), ey + 0.4], [L(11), ey + 0.4], [L(11), ey + 1.6], [L(12.5), ey + 1.6]], "#f4f1e6");
  flat(out, [[50 + 9.5, ey + 0.4], [50 + 11, ey + 0.4], [50 + 11, ey + 1.6], [50 + 9.5, ey + 1.6]], "#f4f1e6");
  const brow = shadeHex(t.style === "bald" ? mixHex(hair, skin, 0.4) : hair, 0.8);
  const inner = ex === "angry" || ex === "focused" ? 2.2 : ex === "tired" ? -1.2 : 0;
  const lift = ex === "confident" ? -1.6 : 0;
  flat(out, [[L(16), 41.5], [L(7), 41.5 + inner], [L(7), 43.5 + inner], [L(16), 43.8]], brow);
  flat(out, [[50 + 16, 41.5 + lift], [50 + 7, 41.5 + inner], [50 + 7, 43.5 + inner], [50 + 16, 43.8 + lift]], brow);

  // --- mouth
  const lip = mixHex(shadeHex(skin, 0.55), "#7a2b2b", 0.35);
  if (ex === "smile" || ex === "confident") {
    flat(out, [[43.5, 65.5], [56.5, 65.5], [53.5, 69.5], [46.5, 69.5]], lip);
    flat(out, [[45.5, 66], [54.5, 66], [53.5, 67.4], [46.5, 67.4]], "#f1ede2");
  } else if (ex === "angry") {
    flat(out, [[44.5, 67.5], [55.5, 67.5], [54, 66], [46, 66]], lip);
  } else {
    flat(out, [[45, 66.2], [55, 66.2], [54.5, 67.8], [45.5, 67.8]], lip);
  }

  // --- hair cap / helmet
  if (t.helmet) {
    const helm = [
      [L(26 * hw), 56, 0.15],
      [L(28.5 * hw), 38, 0.15],
      [L(25 * hw), 24, 0.25],
      [L(14 * hw), top - 4, 0.4],
      [50, top - 7, 0.5],
      [50 + 14 * hw, top - 4, 0.4],
      [50 + 25 * hw, 24, 0.25],
      [50 + 28.5 * hw, 38, 0.15],
      [50 + 26 * hw, 56, 0.15],
      [50 + 20 * hw, 54, 0.3],
      [50 + 21 * hw, 40, 0.35],
      [50 + 13, 35, 0.6],
      [50, 34, 0.7],
      [L(13), 35, 0.6],
      [L(21 * hw), 40, 0.35],
      [L(20 * hw), 54, 0.3],
    ];
    fan(out, helm, [50, top + 6, 0.95], jersey);
    flat(out, [[L(3), top - 6.5], [50 + 3, top - 6.5], [50 + 3, 34.5], [L(3), 34.5]], shadeHex(jersey2, 0.9)); // centre stripe
    flat(out, [[L(26 * hw), 56], [L(20 * hw), 54], [L(21 * hw), 49], [L(27.5 * hw), 49]], shadeHex(jersey, 0.6)); // ear guard
    flat(out, [[50 + 26 * hw, 56], [50 + 20 * hw, 54], [50 + 21 * hw, 49], [50 + 27.5 * hw, 49]], shadeHex(jersey, 0.5));
    if (t.visor) {
      flat(out, [[L(20), 35], [50 + 20, 35], [50 + 19, 44], [L(19), 44]], "#9fd8ff", { a: 0.42 });
      flat(out, [[L(20), 35], [50 + 20, 35], [50 + 20, 36.4], [L(20), 36.4]], "#e9f7ff", { a: 0.8 });
    }
    if (t.cage) {
      const bar = "#b4bcc8";
      for (let x = 36; x <= 64; x += 7) flat(out, [[x - 0.7, 36], [x + 0.7, 36], [x + 0.7, chinY - 1], [x - 0.7, chinY - 1]], bar, { a: 0.8 });
      for (let y = 44; y <= chinY - 2; y += 8) flat(out, [[33, y - 0.7], [67, y - 0.7], [67, y + 0.7], [33, y + 0.7]], bar, { a: 0.8 });
    }
  } else if (t.style !== "bald") {
    const short = t.style === "buzz" || t.style === "fade";
    const vol = { afro: 9, curly: 5, flat_top: 4, spiky: 3, messy_dark: 3, swept: 3, flow: 2, long_blond: 2, red_part: 2, crop: 1.5, grey: 1.5 }[t.style] ?? 1.5;
    const hl = short ? 28 : t.style === "flow" || t.style === "messy_dark" || t.style === "long_blond" ? 34 : 31; // hairline y
    const capColor = short ? mixHex(hair, skin, 0.35) : hair;
    const topY = t.style === "flat_top" ? top - 7 : top - (short ? 1 : vol);
    const cap = [
      [L(24 * hw), 44, 0.15],
      [L((25 + vol * 0.4) * hw), 30, 0.2],
      [L((17 + vol * 0.5) * hw), topY + 4, 0.35],
      [t.style === "flat_top" ? L(12) : 50, topY, 0.5],
      ...(t.style === "flat_top" ? [[50 + 12, topY, 0.5]] : []),
      [50 + (17 + vol * 0.5) * hw, topY + 4, 0.35],
      [50 + (25 + vol * 0.4) * hw, 30, 0.2],
      [50 + 24 * hw, 44, 0.15],
      [50 + 22 * hw, hl + 4, 0.35],
      [50 + 10, hl, 0.7],
      [50, hl - (t.style === "red_part" ? 2 : 0), 0.8],
      [L(10), hl + (t.style === "swept" ? 3 : 0), 0.7],
      [L(22 * hw), hl + 4, 0.35],
    ];
    fan(out, cap, [50, top + 4, 0.9], capColor);
    if (t.style === "spiky") {
      for (let i = -2; i <= 2; i += 1) {
        const x = 50 + i * 8;
        flat(out, [[x - 4, topY + 6], [x + 1, topY - 5], [x + 4, topY + 6]], shadeHex(hair, i < 0 ? 1.1 : 0.85));
      }
    }
    if (t.style === "swept") {
      flat(out, [[L(20), hl - 2], [50 + 12, hl - 4], [L(4), hl + 6]], shadeHex(hair, 1.12));
    }
    if (t.style === "red_part") {
      flat(out, [[L(8), topY + 3], [L(6.5), topY + 3], [L(6), hl - 1], [L(7.5), hl - 1]], shadeHex(capColor, 0.55));
    }
    if (t.style === "flow" || t.style === "long_blond") {
      flat(out, [[L(25 * hw), 40], [L(21 * hw), 40], [L(22 * hw), 68], [L(27 * hw), 70]], shadeHex(hair, 1.05));
      flat(out, [[50 + 25 * hw, 40], [50 + 21 * hw, 40], [50 + 22 * hw, 68], [50 + 27 * hw, 70]], shadeHex(hair, 0.78));
    }
  }
  return out;
}

const faceCache = new Map();
const FACE_W = 44;
const FACE_H = 54;

function backdropFor(t, colors) {
  const base = colors.primary || "#1b3260";
  const navy = "#0d1a36";
  return [mixHex(base, navy, 0.72), mixHex(base, navy, 0.55), mixHex(base, navy, 0.82)];
}

/**
 * Rasterise a face into a small posterized PNG (cached by traits + colours).
 * Returns "" when canvas is unavailable (tests / SSR).
 */
export function renderLowPolyFace(player, colors = {}) {
  const t = faceTraits(player);
  const key = `${t.seed}|${t.skin}|${t.hairColor}|${t.style}|${t.beard}|${t.expression}|${t.helmet}|${t.cage}|${colors.primary || ""}|${colors.secondary || ""}`;
  if (faceCache.has(key)) return faceCache.get(key);
  const fg = makeCanvas(FACE_W, FACE_H);
  const bg = makeCanvas(FACE_W, FACE_H);
  if (!fg || !bg) return "";
  const sx = FACE_W / 100;
  const sy = FACE_H / 122;
  const drawPolys = (ctx, polys) => {
    ctx.save();
    ctx.scale(sx, sy);
    polys.forEach((poly) => {
      const p = poly.p;
      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
      ctx.closePath();
      ctx.globalAlpha = poly.a ?? 1;
      ctx.fillStyle = poly.c;
      ctx.fill();
      // Seal hairline gaps between adjacent facets.
      if ((poly.a ?? 1) === 1) {
        ctx.strokeStyle = poly.c;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    });
    ctx.restore();
  };
  try {
    drawPolys(fg.ctx, buildFaceModel(t, colors));
    const img = fg.ctx.getImageData(0, 0, FACE_W, FACE_H);
    ps2Process(img, { levels: 9, dither: 0.08, contrast: 1.05, saturation: 1.1, ink: 0.2, alphaCut: 120 });
    fg.ctx.putImageData(img, 0, 0);
    // Faceted backdrop shards in a muted team tint.
    const [b1, b2, b3] = backdropFor(t, colors);
    const shards =
      t.backdrop === 0
        ? [{ p: [0, 0, 100, 0, 100, 122, 0, 122], c: b1 }, { p: [0, 0, 70, 0, 0, 80], c: b2 }, { p: [100, 40, 100, 122, 40, 122], c: b3 }]
        : t.backdrop === 1
        ? [{ p: [0, 0, 100, 0, 100, 122, 0, 122], c: b1 }, { p: [30, 0, 100, 0, 100, 70], c: b2 }, { p: [0, 50, 60, 122, 0, 122], c: b3 }]
        : [{ p: [0, 0, 100, 0, 100, 122, 0, 122], c: b3 }, { p: [0, 20, 100, 0, 100, 60, 0, 90], c: b1 }, { p: [0, 0, 40, 0, 0, 30], c: b2 }];
    drawPolys(bg.ctx, shards);
    bg.ctx.drawImage(fg.canvas, 0, 0);
    const url = bg.canvas.toDataURL("image/png");
    if (faceCache.size > 600) faceCache.clear();
    faceCache.set(key, url);
    return url;
  } catch {
    return "";
  }
}

/** Normalise any CSS colour string to #rrggbb (null when unparseable). */
let probe = null;
export function cssColorToHex(value) {
  const v = String(value || "").trim();
  if (!v || v.includes("var(")) return null;
  if (/^#[0-9a-f]{6}$/i.test(v)) return v.toLowerCase();
  if (!probe) probe = makeCanvas(1, 1);
  if (!probe) return null;
  try {
    probe.ctx.fillStyle = "#010203";
    probe.ctx.fillStyle = v;
    const out = String(probe.ctx.fillStyle);
    if (out === "#010203" && v.toLowerCase() !== "#010203") return null;
    if (out.startsWith("#")) return out;
    const m = out.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const [r, g, b] = m[1].split(",").map((x) => Number(x));
    return rgbToHex([r, g, b]);
  } catch {
    return null;
  }
}
