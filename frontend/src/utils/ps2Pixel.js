/**
 * PS2-era pixel pipeline for player portraits.
 *
 * Real NHL photos keep the actual face but get restyled on a tiny canvas:
 * downscale -> contrast/saturation punch -> 4x4 ordered dither -> posterize
 * -> luma edge ink -> hard alpha + chunky silhouette outline. The result is a
 * small PNG data URL rendered with `image-rendering: pixelated`.
 *
 * Everything is cached (memory + sessionStorage) and processed through a
 * small idle-time queue so a 50-row roster never stalls the main thread.
 */

import { getApiBaseUrl } from "../services/api";

const CACHE_VERSION = "ps2v2";
const STORE_PREFIX = `${CACHE_VERSION}:`;
const memCache = new Map(); // src -> dataURL | "" (failed)
const inflight = new Map(); // src -> Promise
const queue = [];
let active = 0;
const MAX_ACTIVE = 3;

// Remember which loading strategy works so we stop paying for failed requests.
let directCorsOk = null; // null = unknown, true/false after first attempt
let proxyOk = null;
let bothFailedStreak = 0;

const BAYER4 = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

function storeGet(key) {
  try {
    return window.sessionStorage.getItem(STORE_PREFIX + key);
  } catch {
    return null;
  }
}

function storeSet(key, value) {
  try {
    window.sessionStorage.setItem(STORE_PREFIX + key, value);
  } catch {
    /* quota or private mode: memory cache still works */
  }
}

export function makeCanvas(w, h) {
  if (typeof document === "undefined") return null;
  try {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    return ctx ? { canvas: c, ctx } : null;
  } catch {
    return null;
  }
}

/**
 * In-place PS2 restyle of an ImageData.
 * opts: levels (posterize steps), dither (0..1 strength), contrast, saturation,
 *       ink (edge darkening strength 0..1), outline (hex or null), alphaCut.
 */
export function ps2Process(img, opts = {}) {
  const {
    levels = 7,
    dither = 0.55,
    contrast = 1.16,
    saturation = 1.2,
    ink = 0.45,
    outline = "#0b0f1a",
    alphaCut = 110,
  } = opts;
  const { width: w, height: h, data } = img;
  const n = w * h;
  const step = 255 / (levels - 1);
  const luma = new Float32Array(n);

  // 1) colour punch + hard alpha
  for (let i = 0; i < n; i += 1) {
    const o = i * 4;
    let r = data[o];
    let g = data[o + 1];
    let b = data[o + 2];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    r = l + (r - l) * saturation;
    g = l + (g - l) * saturation;
    b = l + (b - l) * saturation;
    r = (r - 128) * contrast + 128;
    g = (g - 128) * contrast + 128;
    b = (b - 128) * contrast + 128;
    data[o] = r < 0 ? 0 : r > 255 ? 255 : r;
    data[o + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    data[o + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
    data[o + 3] = data[o + 3] >= alphaCut ? 255 : 0;
    luma[i] = 0.299 * data[o] + 0.587 * data[o + 1] + 0.114 * data[o + 2];
  }

  // 2) ordered dither + posterize, 3) edge ink
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      const o = i * 4;
      if (!data[o + 3]) continue;
      const t = (BAYER4[(y & 3) * 4 + (x & 3)] / 16 - 0.5) * step * dither;
      let k = 1;
      if (ink > 0 && x > 0 && y > 0 && x < w - 1 && y < h - 1) {
        const gx = luma[i + 1] - luma[i - 1];
        const gy = luma[i + w] - luma[i - w];
        const mag = Math.sqrt(gx * gx + gy * gy);
        if (mag > 70) k = 1 - ink;
        else if (mag > 45) k = 1 - ink * 0.45;
      }
      for (let c = 0; c < 3; c += 1) {
        let v = data[o + c] * k + t;
        v = Math.round(v / step) * step;
        data[o + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }

  // 4) chunky silhouette outline on transparent neighbours
  if (outline) {
    const or = parseInt(outline.slice(1, 3), 16);
    const og = parseInt(outline.slice(3, 5), 16);
    const ob = parseInt(outline.slice(5, 7), 16);
    const solid = new Uint8Array(n);
    for (let i = 0; i < n; i += 1) solid[i] = data[i * 4 + 3] ? 1 : 0;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const i = y * w + x;
        if (solid[i]) continue;
        const near =
          (x > 0 && solid[i - 1]) ||
          (x < w - 1 && solid[i + 1]) ||
          (y > 0 && solid[i - w]) ||
          (y < h - 1 && solid[i + w]);
        if (near) {
          const o = i * 4;
          data[o] = or;
          data[o + 1] = og;
          data[o + 2] = ob;
          data[o + 3] = 255;
        }
      }
    }
  }
  return img;
}

function loadImage(url, cors) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    if (cors) im.crossOrigin = "anonymous";
    im.decoding = "async";
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("load"));
    im.src = url;
  });
}

function proxyUrl(src) {
  let base = "";
  try {
    base = getApiBaseUrl() || "";
  } catch {
    base = "";
  }
  return `${base}/api/headshot-proxy?url=${encodeURIComponent(src)}`;
}

function rasterizePhoto(im, targetW = 48) {
  const iw = im.naturalWidth || im.width;
  const ih = im.naturalHeight || im.height;
  if (!iw || !ih) return "";
  const w = targetW;
  const h = Math.max(1, Math.round((targetW * ih) / iw));
  // Two-step downscale keeps faces readable instead of aliased mush.
  const mid = makeCanvas(w * 3, h * 3);
  const out = makeCanvas(w, h);
  if (!mid || !out) return "";
  mid.ctx.imageSmoothingEnabled = true;
  mid.ctx.imageSmoothingQuality = "high";
  mid.ctx.drawImage(im, 0, 0, w * 3, h * 3);
  out.ctx.imageSmoothingEnabled = true;
  out.ctx.imageSmoothingQuality = "high";
  out.ctx.drawImage(mid.canvas, 0, 0, w, h);
  const data = out.ctx.getImageData(0, 0, w, h); // throws if tainted
  ps2Process(data);
  out.ctx.putImageData(data, 0, 0);
  return out.canvas.toDataURL("image/png");
}

async function processPhoto(src) {
  const attempts = [];
  if (directCorsOk !== false) attempts.push(["direct", src]);
  if (proxyOk !== false) attempts.push(["proxy", proxyUrl(src)]);
  const failed = [];
  for (const [kind, url] of attempts) {
    try {
      const im = await loadImage(url, true);
      const out = rasterizePhoto(im);
      if (kind === "direct") directCorsOk = true;
      else proxyOk = true;
      // The other strategy failed on an image this one could read: stop trying it.
      if (failed.includes("direct") && directCorsOk == null) directCorsOk = false;
      if (out) return out;
    } catch {
      failed.push(kind);
    }
  }
  // Both routes failing repeatedly means CORS is blocked and no proxy is up:
  // settle on the CSS fallback instead of re-requesting every headshot.
  if (failed.length === attempts.length && failed.length) {
    bothFailedStreak += 1;
    if (bothFailedStreak >= 3) {
      if (directCorsOk == null) directCorsOk = false;
      if (proxyOk == null) proxyOk = false;
    }
  } else {
    bothFailedStreak = 0;
  }
  return "";
}

function idle(fn) {
  if (typeof window !== "undefined" && typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(fn, { timeout: 400 });
  } else {
    setTimeout(fn, 16);
  }
}

function pump() {
  while (active < MAX_ACTIVE && queue.length) {
    const job = queue.shift();
    active += 1;
    idle(() => {
      processPhoto(job.src)
        .then((url) => {
          memCache.set(job.src, url);
          if (url) storeSet(job.src, url);
          job.resolve(url);
        })
        .catch(() => {
          memCache.set(job.src, "");
          job.resolve("");
        })
        .finally(() => {
          active -= 1;
          inflight.delete(job.src);
          pump();
        });
    });
  }
}

/** Synchronous cache peek: dataURL, "" (known failure) or undefined. */
export function peekPs2Photo(src) {
  if (!src) return "";
  if (memCache.has(src)) return memCache.get(src);
  const stored = typeof window !== "undefined" ? storeGet(src) : null;
  if (stored) {
    memCache.set(src, stored);
    return stored;
  }
  return undefined;
}

/** Resolve a PS2-restyled data URL for a photo ("" when pixels are unreadable). */
export function requestPs2Photo(src) {
  const hit = peekPs2Photo(src);
  if (hit !== undefined) return Promise.resolve(hit);
  if (inflight.has(src)) return inflight.get(src);
  const p = new Promise((resolve) => {
    queue.push({ src, resolve });
  });
  inflight.set(src, p);
  pump();
  return p;
}

let filterMounted = false;
/**
 * CSS-only fallback when canvas pixels are blocked: an SVG posterize filter
 * (discrete transfer) referenced as `filter: url(#ph-ps2-posterize)`.
 */
export function ensurePs2CssFilter() {
  if (filterMounted || typeof document === "undefined" || !document.body) return;
  filterMounted = true;
  try {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "0");
    svg.setAttribute("height", "0");
    svg.setAttribute("aria-hidden", "true");
    svg.style.position = "absolute";
    const table = "0 0.2 0.4 0.6 0.8 1";
    svg.innerHTML =
      `<filter id="ph-ps2-posterize" color-interpolation-filters="sRGB">` +
      `<feComponentTransfer><feFuncR type="discrete" tableValues="${table}"/>` +
      `<feFuncG type="discrete" tableValues="${table}"/><feFuncB type="discrete" tableValues="${table}"/>` +
      `</feComponentTransfer></filter>`;
    document.body.appendChild(svg);
  } catch {
    /* ignore */
  }
}
