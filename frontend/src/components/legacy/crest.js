/**
 * Crest primitives: the editable crest is a list of layers (backend: services/legacy_hall.py
 * validate_identity). Shapes and symbols here are original geometric marks drawn for this
 * game; nothing is traced from a real club logo.
 *
 * Coordinates: viewBox -100..100. Layer x/y shift in those units, scale multiplies.
 */
import React from "react";

export const SHAPE_PATHS = {
  shield: "M-78,-82 L78,-82 L78,-8 C78,44 34,74 0,92 C-34,74 -78,44 -78,-8 Z",
  circle: "M0,-88 A88,88 0 1,1 0,88 A88,88 0 1,1 0,-88 Z",
  oval: "M0,-70 C62,-70 96,-34 96,0 C96,34 62,70 0,70 C-62,70 -96,34 -96,0 C-96,-34 -62,-70 0,-70 Z",
  hexagon: "M0,-90 L78,-45 L78,45 L0,90 L-78,45 L-78,-45 Z",
  diamond: "M0,-94 L82,0 L0,94 L-82,0 Z",
  banner: "M-80,-80 L80,-80 L80,70 L0,40 L-80,70 Z",
  roundel: "M0,-90 A90,90 0 1,1 0,90 A90,90 0 1,1 0,-90 Z M0,-62 A62,62 0 1,0 0,62 A62,62 0 1,0 0,-62 Z",
};

export const SYMBOL_PATHS = {
  none: "",
  star: "M0,-60 L14,-19 L57,-19 L22,7 L35,49 L0,23 L-35,49 L-22,7 L-57,-19 L-14,-19 Z",
  crown: "M-56,30 L-56,-30 L-28,0 L0,-46 L28,0 L56,-30 L56,30 Z M-56,38 L56,38 L56,50 L-56,50 Z",
  puck: "M-50,-14 C-50,-30 50,-30 50,-14 L50,14 C50,30 -50,30 -50,14 Z M-50,-14 C-50,2 50,2 50,-14",
  sticks: "M-48,-56 L-38,-62 L22,40 L46,40 L46,52 L14,52 Z M48,-56 L38,-62 L-22,40 L-46,40 L-46,52 L-14,52 Z",
  maple: "M0,-62 L10,-34 L26,-42 L20,-12 L46,-22 L38,0 L56,8 L26,22 L30,36 L4,30 L4,58 L-4,58 L-4,30 L-30,36 L-26,22 L-56,8 L-38,0 L-46,-22 L-20,-12 L-26,-42 L-10,-34 Z",
  bolt: "M14,-64 L-34,6 L-4,6 L-16,64 L34,-8 L4,-8 Z",
  peak: "M-64,44 L-22,-30 L0,4 L22,-46 L64,44 Z M22,-46 L32,-28 L12,-28 Z",
  wave: "M-66,0 C-44,-24 -22,-24 0,0 C22,24 44,24 66,0 L66,22 C44,46 22,46 0,22 C-22,-2 -44,-2 -66,22 Z",
  wolf: "M-44,-54 L-20,-24 L20,-24 L44,-54 L46,-8 L26,30 L0,52 L-26,30 L-46,-8 Z M-20,-2 L-8,6 L-20,10 Z M20,-2 L8,6 L20,10 Z",
  bear: "M-46,-40 A16,16 0 1,1 -22,-52 L22,-52 A16,16 0 1,1 46,-40 L48,8 C48,40 24,56 0,56 C-24,56 -48,40 -48,8 Z M-6,22 L6,22 L0,32 Z",
  bird: "M-62,-6 C-30,-40 6,-40 26,-18 L60,-30 L38,-6 L58,8 L28,6 C14,34 -18,42 -44,30 L-20,14 Z",
  anchor: "M-6,-60 A12,12 0 1,1 6,-60 L6,30 C22,28 36,18 44,4 L56,12 C44,40 22,52 0,52 C-22,52 -44,40 -56,12 L-44,4 C-36,18 -22,28 -6,30 Z M-28,-30 L28,-30 L28,-20 L-28,-20 Z",
};

export const FONT_FAMILY = {
  block: '"Archivo Black", "Arial Black", sans-serif',
  athletic: '"Barlow Condensed", "Arial Narrow", sans-serif',
  serif: 'Georgia, "Times New Roman", serif',
  rounded: '"Nunito", "Arial Rounded MT Bold", sans-serif',
  condensed: '"Rajdhani", "Arial Narrow", sans-serif',
};

function layerTransform(L) {
  const x = Number(L.x || 0);
  const y = Number(L.y || 0);
  const s = Number(L.scale == null ? 1 : L.scale);
  const r = Number(L.rotation || 0);
  return `translate(${x} ${y}) rotate(${r}) scale(${s})`;
}

function paint(v) {
  return !v || v === "none" ? "none" : v;
}

export function CrestLayer({ layer: L }) {
  if (!L || L.visible === false) return null;
  const common = {
    fill: paint(L.fill),
    stroke: paint(L.stroke),
    strokeWidth: Number(L.stroke_width || 0),
    strokeLinejoin: "round",
    vectorEffect: "non-scaling-stroke",
  };
  if (L.kind === "shape") {
    return <path d={SHAPE_PATHS[L.shape] || SHAPE_PATHS.shield} transform={layerTransform(L)} fillRule="evenodd" {...common} />;
  }
  if (L.kind === "symbol") {
    const d = SYMBOL_PATHS[L.symbol] || "";
    return d ? <path d={d} transform={layerTransform(L)} fillRule="evenodd" {...common} /> : null;
  }
  if (L.kind === "stripe") {
    return <rect x={-100} y={-9} width={200} height={18} transform={layerTransform(L)} {...common} />;
  }
  if (L.kind === "text") {
    const text = String(L.text || "").slice(0, 6);
    const size = text.length <= 1 ? 104 : text.length === 2 ? 82 : text.length === 3 ? 62 : 44;
    return (
      <text
        transform={layerTransform(L)}
        textAnchor="middle"
        dominantBaseline="central"
        fontFamily={FONT_FAMILY[L.font] || FONT_FAMILY.block}
        fontWeight="900"
        fontSize={size}
        paintOrder="stroke fill"
        {...common}
      >
        {text}
      </text>
    );
  }
  return null;
}

/** Renders a saved crest config. Clip keeps layers inside the crest silhouette. */
export function CrestSVG({ crest, size = 160, title, className = "" }) {
  const layers = Array.isArray(crest?.layers) ? crest.layers : [];
  const clipShape = SHAPE_PATHS[crest?.shape] || SHAPE_PATHS.shield;
  const clipId = React.useId().replace(/:/g, "");
  return (
    <svg
      className={className}
      viewBox="-100 -100 200 200"
      width={size}
      height={size}
      role="img"
      aria-label={title || "Team crest"}
    >
      <defs>
        <clipPath id={`crest-clip-${clipId}`}>
          <path d={clipShape} transform="scale(1.06)" />
        </clipPath>
      </defs>
      <g clipPath={`url(#crest-clip-${clipId})`}>
        {layers.map((L) => (
          <CrestLayer key={L.id} layer={L} />
        ))}
      </g>
    </svg>
  );
}

/** Same crest as a standalone SVG string (for canvas/texture use in the 3D studio). */
export function crestToSvgString(crest, px = 512) {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const layers = Array.isArray(crest?.layers) ? crest.layers : [];
  const body = layers
    .filter((L) => L && L.visible !== false)
    .map((L) => {
      const t = layerTransform(L);
      const f = paint(L.fill);
      const st = paint(L.stroke);
      const sw = Number(L.stroke_width || 0);
      if (L.kind === "shape" || L.kind === "symbol") {
        const d = L.kind === "shape" ? SHAPE_PATHS[L.shape] || SHAPE_PATHS.shield : SYMBOL_PATHS[L.symbol] || "";
        return d ? `<path d="${d}" transform="${t}" fill="${f}" stroke="${st}" stroke-width="${sw}" fill-rule="evenodd" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>` : "";
      }
      if (L.kind === "stripe") {
        return `<rect x="-100" y="-9" width="200" height="18" transform="${t}" fill="${f}" stroke="${st}" stroke-width="${sw}"/>`;
      }
      if (L.kind === "text") {
        const text = esc(String(L.text || "").slice(0, 6));
        const size = text.length <= 1 ? 104 : text.length === 2 ? 82 : text.length === 3 ? 62 : 44;
        return `<text transform="${t}" text-anchor="middle" dominant-baseline="central" font-family='${FONT_FAMILY[L.font] || FONT_FAMILY.block}' font-weight="900" font-size="${size}" fill="${f}" stroke="${st}" stroke-width="${sw}" paint-order="stroke fill">${text}</text>`;
      }
      return "";
    })
    .join("");
  const clip = SHAPE_PATHS[crest?.shape] || SHAPE_PATHS.shield;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${px}" height="${px}" viewBox="-100 -100 200 200"><defs><clipPath id="c"><path d="${clip}" transform="scale(1.06)"/></clipPath></defs><g clip-path="url(#c)">${body}</g></svg>`;
}

/** Presets: broad starting directions, all editable afterwards. */
export function crestPreset(kind, colors, initials = "FC") {
  const c = colors || {};
  const base = (shape, fill, stroke, sw = 6) => ({ id: "base", kind: "shape", shape, fill, stroke, stroke_width: sw, x: 0, y: 0, scale: 1, rotation: 0, visible: true });
  const txt = (fill, stroke, font = "block", y = 0, scale = 1) => ({ id: "mark", kind: "text", text: initials, fill, stroke, stroke_width: 2, font, x: 0, y, scale, rotation: 0, visible: true });
  switch (kind) {
    case "heritage":
      return { shape: "shield", layers: [base("shield", c.primary, c.secondary), { ...base("shield", "none", c.accent, 3), id: "inner", scale: 0.82 }, txt(c.neutral, c.secondary, "serif", 6)] };
    case "minimal":
      return { shape: "circle", layers: [base("circle", c.primary, "none", 0), txt(c.neutral, "none", "rounded", 2, 0.9)] };
    case "arena":
      return { shape: "hexagon", layers: [base("hexagon", c.secondary, c.primary, 8), { id: "sym", kind: "symbol", symbol: "bolt", fill: c.primary, stroke: c.accent, stroke_width: 3, x: 0, y: -2, scale: 1.1, rotation: 0, visible: true }] };
    case "vintage":
      return { shape: "roundel", layers: [base("roundel", c.primary, c.neutral, 4), { id: "sym", kind: "symbol", symbol: "star", fill: c.secondary, stroke: "none", stroke_width: 0, x: 0, y: 0, scale: 0.6, rotation: 0, visible: true }] };
    case "geometric":
      return { shape: "diamond", layers: [base("diamond", c.primary, c.accent, 5), { id: "band", kind: "stripe", fill: c.secondary, stroke: "none", stroke_width: 0, x: 0, y: 0, scale: 1, rotation: -30, visible: true }, txt(c.neutral, c.primary, "condensed", 0, 0.8)] };
    case "alternate":
      return { shape: "banner", layers: [base("banner", c.secondary, c.primary, 6), { id: "sym", kind: "symbol", symbol: "sticks", fill: c.primary, stroke: "none", stroke_width: 0, x: 0, y: -6, scale: 1, rotation: 0, visible: true }] };
    default:
      return null;
  }
}

export const CREST_PRESETS = [
  { id: "heritage", label: "Traditional heritage" },
  { id: "minimal", label: "Modern minimalist" },
  { id: "arena", label: "Aggressive arena" },
  { id: "vintage", label: "Vintage revival" },
  { id: "geometric", label: "Geometric contemporary" },
  { id: "alternate", label: "Classic alternate" },
];

/** WCAG contrast ratio, mirrors the backend check so the UI warns before saving. */
export function contrastRatio(a, b) {
  const lum = (hex) => {
    const m = /^#([0-9a-f]{6})$/i.exec(String(hex || ""));
    if (!m) return 0;
    const c = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  };
  const la = lum(a);
  const lb = lum(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}
