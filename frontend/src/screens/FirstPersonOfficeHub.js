import React, {
    Suspense,
    useCallback,
    useMemo,
    useRef,
    useState,
    useEffect,
  } from "react";
  
  import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
  
  import {
    Html,
    OrbitControls,
    Text,
    RoundedBox,
    ContactShadows,
    Environment,
    SoftShadows,
    AccumulativeShadows,
    RandomizedLight,
    useGLTF,
  } from "@react-three/drei";
  
  import {
    EffectComposer,
    Bloom,
    Vignette,
  } from "@react-three/postprocessing";
  
  import { motion, AnimatePresence } from "framer-motion";
  import * as THREE from "three";
  import TeamLogoBadge from "../components/ui/TeamLogoBadge";
  import PlayerHeadshot from "../components/PlayerHeadshot";
  import { resolveFranchiseTeamLogo, toLogoUrl } from "../utils/teamLogos";
  import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
  import { SCREENS } from "../game/constants";
  import "./FirstPersonOfficeHub.css";
  import officeFontBold from "../styles/ArchivoBlack-Regular.ttf";
  import retroOfficePackGlb from "../styles/Retro Office Pack/Itch Upload/90s Retro Office Pack.glb";
  import officeWallTextureSrc from "../pictures/gray-abstract-texture-background.jpg";

  /**
   * One standardized landmark footprint for every menu destination.
   *
   * `artPx` is the square box each scene is fitted into (drei's Html transform
   * maps 1px to `distanceFactor / 400` world units), so a portrait scene and a
   * wide scene end up with the same perceived weight instead of one dwarfing
   * the other. Hitboxes, labels and vignettes all derive from the same numbers.
   */
  const MENU_LANDMARK = {
    artPx: 208,
    padPx: 17,
    distanceFactor: 1.92,
    hitBox: [1.16, 1.3, 0.52],
    hitBoxOffset: [0, -0.04, 0.28],
    vignette: [1.26, 1.36],
    /** Two hanging lines plus the elevated crest medallion above them. */
    crestY: 2.66,
    upperBandY: 2.46,
    lowerBandY: 1.18,
  };

  /**
   * Back-wall grid. The columns are deliberately clustered rather than evenly
   * spaced — team building to the left, the club in the middle, league
   * intelligence to the right — so the wall reads as three installations
   * instead of one row of identical posters.
   */
  const MENU_COLUMNS = {
    farLeft: -3.72,
    left: -2.44,
    innerLeft: -1.16,
    center: 0,
    innerRight: 1.16,
    right: 2.44,
    farRight: 3.72,
    lowLeftOuter: -3.38,
    lowLeftInner: -2.08,
    lowRightInner: 2.08,
    lowRightOuter: 3.38,
  };

  /**
   * Interaction volumes. Wall landmarks all share `MENU_LANDMARK.hitBox`; the
   * entries below cover the physical props that are not standardized scenes.
   */
  const OFFICE_HITBOXES = {
    dashboard: [1.15, 0.95, 0.85],
    calendar: [0.86, 0.7, 0.9],
    scouting: [1.55, 1.15, 0.42],
    gameDayPuck: [0.24, 0.14, 0.24],
    tasks: [0.42, 0.16, 0.52],
    draft: [2.05, 2.35, 0.55],
    arenaWindow: [1.85, 1.08, 0.24],
  };

  /**
   * First-person executive seated eye line. The rest position sits dead centre
   * on the room axis so the look-around allowance is symmetric — the previous
   * off-axis rest pose spent most of its right-hand travel before the player
   * touched the mouse, which left the right wall unreachable.
   */
  const OFFICE_CAMERA = {
    position: [0, 1.96, 4.05],
    target: [0, 1.62, -1.15],
    fov: 46,
    minDistance: 2.4,
    maxDistance: 6.8,
  };

  /** League Ops diorama focal point — used for hover/click camera nudge */
  const LEAGUE_OPS_FOCUS = [3.72, 2.46, -3.34];

  /**
   * Rear-facing executive — authored SVG paths (code only).
   * ViewBox 500×760. Center X = 250.
   */
  const LO_HEAD =
    "M 250 42 C 268 42 282 49 288 63 C 292 73 293 84 291 96" +
    "C 290 108 286 120 279 130 C 272 139 263 145 254 148" +
    "C 251 149 248 149 245 148 C 235 145 226 139 220 130" +
    "C 213 120 209 108 208 96 C 207 84 208 73 212 63" +
    "C 218 49 232 42 250 42 Z";

  const LO_HAIR_CROWN =
    "M 216 72 C 220 55 234 45 251 45 C 267 45 280 53 285 68" +
    "C 281 64 277 61 272 59 C 269 54 263 51 257 50" +
    "C 250 47 244 51 238 51 C 230 53 223 60 216 72 Z";

  const LO_HAIR_SIDES =
    "M 211 78 C 208 94 212 113 220 128 L 225 133 C 219 116 218 96 221 76 Z" +
    "M 289 78 C 292 94 288 113 280 128 L 275 133 C 281 116 282 96 279 76 Z";

  const LO_NAPE =
    "M 226 125 C 233 139 240 146 250 148 C 260 146 268 139 274 125" +
    "C 270 143 262 153 250 154 C 238 153 230 143 226 125 Z";

  const LO_NECK =
    "M 232 135 C 233 149 232 163 229 176 C 240 184 260 184 271 176" +
    "C 268 163 267 149 268 135 C 259 145 241 145 232 135 Z";

  const LO_COLLAR_SHIRT =
    "M 224 169 C 236 178 264 178 276 169 L 278 178" +
    "C 264 188 236 188 222 178 Z";

  const LO_COLLAR_SUIT =
    "M 212 177 C 225 186 235 192 250 194 C 265 192 275 186 288 177" +
    "L 302 199 C 284 204 269 210 250 220 C 231 210 216 204 198 199 Z";

  const LO_JACKET =
    "M 230 180 C 194 184 158 193 126 207 C 134 220 147 231 163 238" +
    "C 168 292 173 354 178 414 C 181 464 178 514 171 558" +
    "C 194 572 220 578 250 576 C 280 578 306 572 329 558" +
    "C 322 514 319 464 322 414 C 327 354 332 292 337 238" +
    "C 353 231 366 220 374 207 C 342 193 306 184 270 180" +
    "C 260 190 240 190 230 180 Z";

  const LO_SLEEVE_L =
    "M 128 206 C 106 216 91 235 84 259 C 78 287 82 321 91 354" +
    "C 100 389 110 421 122 450 C 132 474 145 496 160 515" +
    "C 170 527 183 531 194 523 C 196 514 191 504 184 495" +
    "C 174 475 166 450 158 421 C 150 387 145 351 144 318" +
    "C 143 283 149 254 163 238 C 153 225 140 214 128 206 Z";

  const LO_SLEEVE_R =
    "M 372 206 C 394 216 409 235 416 259 C 422 287 418 321 409 354" +
    "C 400 389 390 421 378 450 C 368 474 355 496 340 515" +
    "C 330 527 317 531 306 523 C 304 514 309 504 316 495" +
    "C 326 475 334 450 342 421 C 350 387 355 351 356 318" +
    "C 357 283 351 254 337 238 C 347 225 360 214 372 206 Z";

  const LO_HAND_L =
    "M 160 515 C 168 528 180 538 196 541 C 205 537 209 529 207 519" +
    "C 199 515 191 507 184 495 C 181 512 173 521 160 515 Z";

  const LO_HAND_R =
    "M 340 515 C 332 528 320 538 304 541 C 295 537 291 529 293 519" +
    "C 301 515 309 507 316 495 C 319 512 327 521 340 515 Z";

  const LO_SHOULDER_PLANE =
    "M 128 205 C 174 188 215 183 250 184 C 285 183 326 188 372 205" +
    "C 364 217 350 228 335 234 C 306 222 278 216 250 216" +
    "C 222 216 194 222 165 234 C 150 228 136 217 128 205 Z";

  const LO_TORSO_PANEL =
    "M 168 235 C 177 300 183 365 187 424 C 190 476 194 525 202 563" +
    "C 218 570 235 573 250 572 C 265 573 282 570 298 563" +
    "C 306 525 310 476 313 424 C 317 365 323 300 332 235" +
    "C 302 224 198 224 168 235 Z";

  const LO_RIM_HEAD =
    "M 212 111 C 206 94 207 75 212 63 C 218 49 232 42 250 42" +
    "C 268 42 282 49 288 63 C 293 75 294 94 288 111";
  const LO_RIM_SHOULDERS =
    "M 126 207 C 166 190 205 182 230 180 C 240 190 260 190 270 180" +
    "C 295 182 334 190 374 207";
  const LO_RIM_ARM_L =
    "M 128 207 C 104 220 89 239 84 263 C 78 298 85 337 95 372";
  const LO_RIM_ARM_R =
    "M 372 207 C 396 220 411 239 416 263 C 422 298 415 337 405 372";

  const LO_SEAM_CENTER = "M 250 218 L 250 568";
  const LO_SEAM_SHOULDER_L = "M 230 190 C 194 196 174 212 164 236";
  const LO_SEAM_SHOULDER_R = "M 270 190 C 306 196 326 212 336 236";
  const LO_SEAM_SIDE_L = "M 169 250 C 176 330 181 416 176 520";
  const LO_SEAM_SIDE_R = "M 331 250 C 324 330 319 416 324 520";
  const LO_SEAM_SLEEVE_L = "M 141 230 C 125 302 132 396 161 484";
  const LO_SEAM_SLEEVE_R = "M 359 230 C 375 302 368 396 339 484";
  const LO_HEM = "M 173 555 C 197 570 225 578 250 574 C 275 578 303 570 327 555";

  const LO_SHIELD =
    "M 86 268 L 118 262 L 150 268 L 148 312 C 148 332 128 348 118 354" +
    "C 108 348 88 332 88 312 Z";
  const LO_TROPHY =
    "M 370 268 L 382 268 L 386 292 L 402 292 C 406 304 398 312 386 314" +
    "L 382 334 L 402 348 L 350 348 L 370 334 L 366 314" +
    "C 354 312 346 304 350 292 L 366 292 Z";

  let leagueOpsVignetteTexture = null;

  function getLeagueOpsVignetteTexture() {
    if (leagueOpsVignetteTexture) return leagueOpsVignetteTexture;
    if (typeof document === "undefined") return null;

    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    const grad = ctx.createRadialGradient(128, 118, 18, 128, 128, 148);
    grad.addColorStop(0, "rgba(5,8,12,0.62)");
    grad.addColorStop(0.52, "rgba(4,7,11,0.38)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    leagueOpsVignetteTexture = new THREE.CanvasTexture(canvas);
    leagueOpsVignetteTexture.needsUpdate = true;
    return leagueOpsVignetteTexture;
  }

  /**
   * Deep navy-charcoal room. The walls used to be bright teal, which made every
   * wall-mounted destination read as a coloured poster; the landmarks carry the
   * colour now, the architecture stays dark.
   */
  const OFFICE_PALETTE = {
    void: "#06161b",
    wall: "#122e37",
    wallLight: "#1b4653",
    wallDeep: "#0a2027",
    wallWainscot: "#081d23",
    wallPanel: "#0d262d",
    panel: "#0f2830",
    walnut: "#3d2a1c",
    gunmetal: "#2a3038",
    leather: "#1c1816",
    gold: "#c4a46a",
    goldDim: "#8a7048",
    monitor: "#0c141c",
    monitorGlow: "#1a3850",
    alert: "#8a3028",
  };

  /** Gray abstract photo + executive teal tint, cached after first bake. */
  let tintedOfficeWallCache = null;

  function buildTintedOfficeWallMaps(sourceImage) {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");

    const pattern = ctx.createPattern(sourceImage, "repeat");
    ctx.fillStyle = pattern;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.globalCompositeOperation = "multiply";
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, OFFICE_PALETTE.wallLight);
    grad.addColorStop(0.32, OFFICE_PALETTE.wall);
    grad.addColorStop(0.62, OFFICE_PALETTE.wallDeep);
    grad.addColorStop(1, OFFICE_PALETTE.wallWainscot);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    /* One low wainscot break only. The wall used to be scored into a grid of
       panels, which drew a rectangle around every landmark and made the whole
       room read as a poster board. */
    const railY = canvas.height * 0.74;
    ctx.fillStyle = "rgba(5, 26, 33, 0.5)";
    ctx.fillRect(0, railY + 4, canvas.width, canvas.height - railY - 4);

    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
    ctx.fillRect(0, railY, canvas.width, 3);
    ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
    ctx.fillRect(0, railY, canvas.width, 1.2);

    const edge = ctx.createRadialGradient(
      canvas.width / 2,
      canvas.height / 2,
      canvas.width * 0.18,
      canvas.width / 2,
      canvas.height / 2,
      canvas.width * 0.78
    );
    edge.addColorStop(0, "rgba(0,0,0,0)");
    edge.addColorStop(1, "rgba(0,0,0,0.24)");
    ctx.fillStyle = edge;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const colorMap = new THREE.CanvasTexture(canvas);
    colorMap.wrapS = THREE.RepeatWrapping;
    colorMap.wrapT = THREE.RepeatWrapping;
    colorMap.colorSpace = THREE.SRGBColorSpace;

    const bumpCanvas = document.createElement("canvas");
    bumpCanvas.width = canvas.width;
    bumpCanvas.height = canvas.height;
    const bctx = bumpCanvas.getContext("2d");
    bctx.drawImage(canvas, 0, 0);
    const pixels = bctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const lum =
        pixels.data[i] * 0.299 +
        pixels.data[i + 1] * 0.587 +
        pixels.data[i + 2] * 0.114;
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = lum;
    }
    bctx.putImageData(pixels, 0, 0);

    const bumpMap = new THREE.CanvasTexture(bumpCanvas);
    bumpMap.wrapS = THREE.RepeatWrapping;
    bumpMap.wrapT = THREE.RepeatWrapping;

    return { colorMap, bumpMap };
  }

  function getTintedOfficeWallMaps(sourceImage) {
    if (!tintedOfficeWallCache) {
      tintedOfficeWallCache = buildTintedOfficeWallMaps(sourceImage);
    }
    return tintedOfficeWallCache;
  }

  function createExecutiveWallMaterial(sourceImage, repeatX = 2, repeatY = 1.4) {
    const { colorMap, bumpMap } = getTintedOfficeWallMaps(sourceImage);
    const color = colorMap.clone();
    const bump = bumpMap.clone();
    color.repeat.set(repeatX, repeatY);
    bump.repeat.set(repeatX, repeatY);
    return new THREE.MeshStandardMaterial({
      map: color,
      bumpMap: bump,
      bumpScale: 0.024,
      roughness: 0.84,
      metalness: 0.02,
      envMapIntensity: 0.28,
    });
  }

  function ExecutiveWallSurface({ size, position, rotation = [0, 0, 0], repeat = [2.2, 1.5] }) {
    const [repeatX, repeatY] = repeat;
    // TextureLoader lives on fiber's useLoader, not drei.
    const sourceTexture = useLoader(THREE.TextureLoader, officeWallTextureSrc);

    const material = useMemo(() => {
      const img = sourceTexture?.image;
      if (!img?.width) return null;
      return createExecutiveWallMaterial(img, repeatX, repeatY);
    }, [sourceTexture, repeatX, repeatY]);

    useEffect(() => () => material?.dispose(), [material]);

    if (!material) return null;

    return (
      <mesh position={position} rotation={rotation} receiveShadow castShadow raycast={() => null}>
        <boxGeometry args={size} />
        <primitive object={material} attach="material" />
      </mesh>
    );
  }

  useLoader.preload(THREE.TextureLoader, officeWallTextureSrc);

  /** Google Maps Weather API (optional). Set REACT_APP_GOOGLE_WEATHER_API_KEY in frontend/.env */
  const GOOGLE_WEATHER_API_KEY =
    typeof process !== "undefined"
      ? process.env.REACT_APP_GOOGLE_WEATHER_API_KEY || process.env.REACT_APP_GOOGLE_MAPS_API_KEY || ""
      : "";

  function parseFranchiseDateParts(currentDate) {
    const raw = String(currentDate || "").trim();
    const match = raw.match(/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/);
    if (match) {
      return {
        year: Number(match[1]),
        month: Number(match[2]),
        day: Number(match[3]),
      };
    }
    const d = new Date(raw);
    if (!Number.isNaN(d.getTime())) {
      return { year: d.getFullYear(), month: d.getMonth() + 1, day: d.getDate() };
    }
    return { year: 0, month: 9, day: 15 };
  }

  /** Seasonal weather from the franchise sim date. Google Weather only covers live/near-term, not arbitrary sim years. */
  function deriveSeasonalWeather(currentDate) {
    const { month } = parseFranchiseDateParts(currentDate);
    if (month >= 11 || month <= 2) {
      return {
        condition: "snow",
        label: "Snow / overcast",
        sky: "#6a7a90",
        haze: "#d8e4f0",
        light: "#c8d8e8",
        precip: "snow",
      };
    }
    if (month === 3 || month === 4 || month === 10) {
      return {
        condition: "rain",
        label: "Cool rain",
        sky: "#4a5a6e",
        haze: "#8a9aac",
        light: "#a8b8c8",
        precip: "rain",
      };
    }
    if (month >= 5 && month <= 8) {
      return {
        condition: "clear",
        label: "Clear summer",
        sky: "#4a7ab8",
        haze: "#c8dff8",
        light: "#f0e8d0",
        precip: "none",
      };
    }
    return {
      condition: "clear",
      label: "Clear autumn",
      sky: "#3a5a88",
      haze: "#b8c8e0",
      light: "#e8d8b8",
      precip: "none",
    };
  }

  function mapGoogleWeatherCondition(payload) {
    const type = String(
      payload?.weatherCondition?.type ||
        payload?.weatherCondition?.description?.text ||
        payload?.condition ||
        ""
    ).toUpperCase();
    if (/SNOW|ICE|BLIZZARD|FLURRY/.test(type)) {
      return { condition: "snow", label: "Snow", precip: "snow", sky: "#6a7a90", haze: "#d8e4f0", light: "#c8d8e8" };
    }
    if (/RAIN|SHOWER|STORM|THUNDER|DRIZZLE/.test(type)) {
      return { condition: "rain", label: "Rain", precip: "rain", sky: "#4a5a6e", haze: "#8a9aac", light: "#a8b8c8" };
    }
    if (/CLOUD|OVERCAST|FOG/.test(type)) {
      return { condition: "cloudy", label: "Cloudy", precip: "none", sky: "#5a6a80", haze: "#a8b8c8", light: "#d0d8e0" };
    }
    return { condition: "clear", label: "Clear", precip: "none", sky: "#4a7ab8", haze: "#c8dff8", light: "#f0e8d0" };
  }

  const USE_RETRO_OFFICE_PACK = false;
  const USE_PROCEDURAL_ROOM_SHELL = true;

  /** Bundled from src — source: styles/Retro Office Pack/Itch Upload/ */
  const RETRO_OFFICE_MODEL_PATH = retroOfficePackGlb;

  const RETRO_OFFICE_TRANSFORM = {
    position: [0, 0, -0.95],
    rotation: [0, Math.PI, 0],
    scale: 0.82,
  };

  const pictureContext = (() => {
    try {
      return require.context("../pictures", false, /\.(png|jpe?g|webp|svg)$/i);
    } catch (err) {
      return null;
    }
  })();
  
  function getOfficePictures() {
    if (!pictureContext) return [];
  
    return pictureContext.keys().map((key) => {
      const asset = pictureContext(key);
  
      return {
        key,
        src: asset?.default || asset,
        name: key.replace("./", "").replace(/\.[^/.]+$/, ""),
      };
    });
  }

  /* Backend labels can arrive with UTF-8 punctuation that was decoded through a
     legacy code page (season labels show up as "2025<mojibake>2026"). Escapes
     are written as code units so this repair cannot itself be mangled. */
  const MOJIBAKE_REPAIRS = [
    [/\u0393\u00c7\u00d6/g, "\u2019"],
    [/\u0393\u00c7\u00a3/g, "\u201c"],
    [/\u0393\u00c7\u00a5/g, "\u201d"],
    [/\u0393\u00c7\u00f4/g, "\u2013"],
    [/\u0393\u00c7\u00f6/g, "\u2014"],
    [/\u0393\u00c7\u00a2/g, "\u00b7"],
    [/\u00e2\u20ac\u2122/g, "\u2019"],
    [/\u00e2\u20ac\u009c/g, "\u201c"],
    [/\u00e2\u20ac\u009d/g, "\u201d"],
    [/\u00e2\u20ac\u201c/g, "\u2013"],
    [/\u00e2\u20ac\u201d/g, "\u2014"],
    [/\u0393\u00c7./g, "-"],
  ];

  function repairEncoding(text) {
    let out = text;
    for (const [pattern, replacement] of MOJIBAKE_REPAIRS) {
      out = out.replace(pattern, replacement);
    }
    return out;
  }

  function safeText(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    return repairEncoding(String(value));
  }
  
  function initialsFromTeam(teamName = "NHL") {
    return String(teamName)
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((word) => word[0])
      .join("")
      .toUpperCase();
  }
  
  function formatRecord(record) {
    if (!record) return "0-0-0";
    if (typeof record === "string") return record;
  
    return `${record.w ?? record.wins ?? 0}-${record.l ?? record.losses ?? 0}-${
      record.otl ?? record.ot ?? record.overtime_losses ?? 0
    }`;
  }
  
  function formatMoney(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "string" && value.startsWith("$")) return value;

    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);

    // Backend contract/cap fields are millions; legacy dollar amounts are large.
    const millions = Math.abs(n) >= 500 ? n / 1000000 : n;
    const abs = Math.abs(millions);
    if (abs >= 10) return `$${millions.toFixed(1)}M`;
    return `$${millions.toFixed(2)}M`;
  }

  function officeCapMillions(value) {
    const n = officeSafeNumber(value, NaN);
    if (!Number.isFinite(n)) return NaN;
    return Math.abs(n) >= 500 ? n / 1000000 : n;
  }

  function titleCaseWords(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (m) => m.toUpperCase());
  }

  function formatOfficeMode(mode) {
    return titleCaseWords(mode || "regular");
  }

  function formatStandingsLine(line) {
    const text = String(line || "Standings");
    return text.replace(/^(\d+)\s/, (_, n) => {
      const num = Number(n);
      const mod10 = num % 10;
      const mod100 = num % 100;
      const suffix =
        mod100 >= 11 && mod100 <= 13
          ? "th"
          : mod10 === 1
            ? "st"
            : mod10 === 2
              ? "nd"
              : mod10 === 3
                ? "rd"
                : "th";
      return `${num}${suffix} `;
    });
  }

  function formatNextGameLabel(nextGame, phase = "") {
    const text = String(nextGame || "");
    if (text && text !== "No game listed" && text !== "Upcoming Game") return text;
    const ph = String(phase || "").toLowerCase();
    if (ph.includes("offseason")) return "Offseason — no game scheduled";
    if (ph.includes("complete")) return "Season complete";
    return "No game on schedule";
  }

  function officeSafeNumber(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function officeSafeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function officePhaseText(franchiseState) {
    const ph = franchiseState?.season_phase || franchiseState?.phase || "";
    const stage = franchiseState?.offseason_stage;
    const ui = franchiseState?.nhl_today?.ui_phase;
    if (ph === "offseason" && stage) return `${ph} ${stage}`;
    if (ui) return String(ui);
    return String(ph || "regular");
  }

  function countOfficeInjuries(franchiseState, team) {
    const pools = [
      franchiseState?.injuries,
      franchiseState?.medical?.injuries,
      franchiseState?.team?.injuries,
      franchiseState?.user_team?.injuries,
      team?.injuries,
    ];
    for (const pool of pools) {
      if (Array.isArray(pool)) return pool.length;
    }
    return officeSafeNumber(franchiseState?.injury_count, 0);
  }

  function parseStreakDirection(streak) {
    if (!streak) return null;
    const text = String(streak).toLowerCase();
    if (text.includes("w") || text.includes("win")) return "win";
    if (text.includes("l") || text.includes("loss")) return "loss";
    const n = officeSafeNumber(streak, NaN);
    if (Number.isFinite(n)) return n > 0 ? "win" : n < 0 ? "loss" : null;
    return null;
  }

  function parseStreakLength(streak) {
    if (!streak) return 0;
    const match = String(streak).match(/(\d+)/);
    return match ? officeSafeNumber(match[1], 0) : 0;
  }

  function deriveOfficeMood(franchiseState, team, officeSummary = {}) {
    const fs = franchiseState || {};
    const calSeg = String(fs.nhl_today?.segment || fs.nhl_today?.season_segment || "").toLowerCase();
    const ph = String(
      calSeg === "preseason"
        ? "preseason"
        : fs.season_phase || fs.phase || (calSeg || "regular")
    ).toLowerCase();
    const stage = String(fs.offseason_stage || "").toLowerCase();
    const uiPhase = String(fs.nhl_today?.ui_phase || "").toLowerCase();
    const combined = `${ph} ${stage} ${uiPhase} ${calSeg}`;

    const isOffseason = ph === "offseason" || combined.includes("offseason");
    const isPlayoffs =
      ph === "playoffs" ||
      ph === "playoff_ready" ||
      ph === "post_cup" ||
      combined.includes("playoff");
    const isCupFinal =
      combined.includes("cup final") ||
      combined.includes("stanley cup final") ||
      stage.includes("cup_final") ||
      stage.includes("final");
    const isTradeDeadline =
      combined.includes("trade deadline") ||
      combined.includes("deadline") ||
      uiPhase.includes("deadline") ||
      stage.includes("trade_deadline");
    const isDraftWeek =
      combined.includes("draft") &&
      (stage.includes("draft") || uiPhase.includes("draft") || isOffseason);
    const isFreeAgency =
      stage.includes("free_agency") ||
      stage.includes("free agency") ||
      combined.includes("free agency") ||
      combined.includes("free_agency");

    const streak =
      team?.streak ||
      team?.current_streak ||
      fs.streak ||
      fs.current_streak ||
      officeSummary?.streak;
    const streakDir = parseStreakDirection(streak);
    const streakLen = parseStreakLength(streak);
    const isLosingStreak = streakDir === "loss" && streakLen >= 3;
    const isHotStreak = streakDir === "win" && streakLen >= 3;

    const injuryCount = countOfficeInjuries(fs, team);
    const hasInjuryCrisis = injuryCount >= 3;

    const ownerConf = officeSafeNumber(
      fs.owner_confidence ??
        fs.owner?.confidence ??
        fs.owner?.approval ??
        fs.management?.owner_confidence,
      NaN
    );
    const hasOwnerPressure =
      (Number.isFinite(ownerConf) && ownerConf < 45) ||
      officeSafeNumber(officeSummary?.pendingTasks, 0) >= 4;

    const unread = officeSafeNumber(
      officeSummary?.unreadMessages ??
        fs.unread_messages ??
        fs.unreadMessages,
      0
    );
    const pending = officeSafeNumber(
      officeSummary?.pendingTasks ??
        fs.pending_tasks ??
        fs.pendingTasks,
      0
    );
    const hasUrgentDecisions =
      pending > 0 ||
      unread > 0 ||
      hasOwnerPressure ||
      hasInjuryCrisis ||
      officeSafeArray(fs.urgent_decisions).length > 0;

    let teamForm = "steady";
    if (isHotStreak) teamForm = "hot";
    else if (isLosingStreak) teamForm = "cold";
    else if (isPlayoffs) teamForm = "stakes";

    let pressureLevel = "low";
    if (hasOwnerPressure || isCupFinal || (isPlayoffs && isLosingStreak)) {
      pressureLevel = "critical";
    } else if (
      isTradeDeadline ||
      hasInjuryCrisis ||
      isLosingStreak ||
      hasUrgentDecisions
    ) {
      pressureLevel = "high";
    } else if (isDraftWeek || isFreeAgency || isPlayoffs || pending > 0) {
      pressureLevel = "medium";
    }

    let officeMode = "regular_season";
    if (isCupFinal) officeMode = "cup_final";
    else if (isPlayoffs) officeMode = "playoffs";
    else if (isTradeDeadline) officeMode = "trade_deadline";
    else if (isDraftWeek) officeMode = "draft_week";
    else if (isFreeAgency) officeMode = "free_agency";
    else if (isOffseason) officeMode = "offseason";
    else if (ph === "preseason") officeMode = "preseason";

    return {
      seasonPhase: ph || "regular",
      officeMode,
      pressureLevel,
      teamForm,
      isTradeDeadline,
      isDraftWeek,
      isFreeAgency,
      isPlayoffs,
      isOffseason,
      isCupFinal,
      isLosingStreak,
      isHotStreak,
      hasInjuryCrisis,
      hasOwnerPressure,
      hasUrgentDecisions,
      injuryCount,
      unreadMessages: unread,
      pendingTasks: pending,
    };
  }

  function buildOfficeUrgentItems(franchiseState, team, officeSummary = {}) {
    const items = [];
    const fs = franchiseState || {};
    const mood = deriveOfficeMood(fs, team, officeSummary);
    const push = (item) => {
      if (!item?.id || !item?.title) return;
      items.push({
        severity: "low",
        detail: "",
        target: OFFICE_NAV_TARGETS.DASHBOARD,
        ...item,
      });
    };

    const unread = mood.unreadMessages;
    if (unread > 0) {
      push({
        id: "unread-messages",
        type: "messages",
        severity: unread >= 5 ? "high" : "medium",
        title: `${unread} unread message${unread === 1 ? "" : "s"}`,
        detail: "Trade calls and league noise may need a response.",
        target: OFFICE_NAV_TARGETS.INBOX,
      });
    }

    const pending = mood.pendingTasks;
    if (pending > 0) {
      push({
        id: "pending-tasks",
        type: "tasks",
        severity: pending >= 3 ? "high" : "medium",
        title: `${pending} decision${pending === 1 ? "" : "s"} on the desk`,
        detail: "Front office priorities are waiting for your call.",
        target: OFFICE_NAV_TARGETS.TASKS,
      });
    }

    if (mood.hasOwnerPressure) {
      push({
        id: "owner-pressure",
        type: "owner",
        severity: "high",
        title: "Owner pressure rising",
        detail: "Leadership confidence may be narrowing your runway.",
        target: OFFICE_NAV_TARGETS.OWNER,
      });
    }

    if (mood.hasInjuryCrisis) {
      push({
        id: "injury-crisis",
        type: "injuries",
        severity: "high",
        title: "Injury report requires attention",
        detail: `${mood.injuryCount} active injuries are affecting roster stability.`,
        target: OFFICE_NAV_TARGETS.INJURIES,
      });
    }

    const capRaw =
      officeSummary?.capSpaceMillions ??
      team?.cap_space ??
      team?.capSpace ??
      fs.cap_space ??
      officeSummary?.capSpaceRaw;
    const capMillions = officeCapMillions(capRaw);
    if (Number.isFinite(capMillions) && capMillions < 1.5) {
      push({
        id: "cap-tight",
        type: "contracts",
        severity: capMillions < 0 ? "critical" : "medium",
        title: capMillions < 0 ? "Cap space is underwater" : "Cap space is tight",
        detail: "Contract moves may require creativity before the next major decision.",
        target: OFFICE_NAV_TARGETS.SALARY_CAP,
      });
    }

    if (mood.isTradeDeadline) {
      push({
        id: "trade-deadline",
        type: "trade",
        severity: "high",
        title: "Trade market activity detected",
        detail: "Deadline pressure is live. Calls and offers may not wait.",
        target: OFFICE_NAV_TARGETS.TRADE_CALLS,
      });
    }

    if (mood.isDraftWeek) {
      push({
        id: "draft-board",
        type: "draft",
        severity: "medium",
        title: "Draft board update available",
        detail: "Final tier review is recommended before selections lock in.",
        target: OFFICE_NAV_TARGETS.DRAFT_BOARD,
      });
    }

    if (mood.isFreeAgency || mood.isOffseason) {
      push({
        id: "contract-decisions",
        type: "contracts",
        severity: "medium",
        title: "Contract decisions pending",
        detail: "RFAs, UFAs, and extension timing are on the clock.",
        target: OFFICE_NAV_TARGETS.CONTRACTS,
      });
    }

    const storyCount = officeSafeNumber(officeSummary?.activeStorylines, 0);
    if (storyCount > 0) {
      push({
        id: "storylines",
        type: "news",
        severity: "low",
        title: `${storyCount} active storyline${storyCount === 1 ? "" : "s"}`,
        detail: "Locker room and league narratives may need management.",
        target: OFFICE_NAV_TARGETS.STORYLINES,
      });
    }

    const nextGame = officeSummary?.nextGame;
    if (nextGame && nextGame !== "No game listed") {
      push({
        id: "next-game",
        type: "game",
        severity: mood.isPlayoffs ? "high" : "low",
        title: "Next game preparation available",
        detail: `Upcoming: ${nextGame}`,
        target: OFFICE_NAV_TARGETS.GAME_PREVIEW,
      });
    }

    if (mood.isLosingStreak) {
      push({
        id: "losing-streak",
        type: "performance",
        severity: "medium",
        title: "Losing streak flagged by staff",
        detail: "Analytics and coaching are tracking a slide in form.",
        target: OFFICE_NAV_TARGETS.TEAM_STATS,
      });
    }

    const tradeOffers = officeSafeArray(fs.trade_offers || fs.incoming_trades);
    if (tradeOffers.length > 0) {
      push({
        id: "trade-offers",
        type: "trade",
        severity: "medium",
        title: `${tradeOffers.length} trade offer${tradeOffers.length === 1 ? "" : "s"} on file`,
        detail: "Some proposals expire after the next game or phase advance.",
        target: OFFICE_NAV_TARGETS.TRADE_CALLS,
      });
    }

    const severityRank = { critical: 0, high: 1, medium: 2, low: 3 };
    return items.sort(
      (a, b) =>
        (severityRank[a.severity] ?? 9) - (severityRank[b.severity] ?? 9)
    );
  }

  const LOW_POWER_STORAGE_KEY = "nhlOfficeLowPowerMode";

  function detectWebGLSupport() {
    try {
      const canvas = document.createElement("canvas");
      return !!(
        window.WebGLRenderingContext &&
        (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
      );
    } catch (err) {
      return false;
    }
  }
    
  const OFFICE_PANEL_IDS = {
    DASHBOARD: "dashboard",
    MESSAGES: "messages",
    CALENDAR: "calendar",
    SCOUTING: "scouting",
    CONTRACTS: "contracts",
    STATS: "stats",
    LINES: "lines",
    NEWS: "news",
    AWARDS: "awards",
    DRAFT: "draft",
    DRAFT_CLASS: "draftClass",
    ROSTER: "roster",
    STANDINGS: "standings",
    GAME_DAY: "gameDay",
    TEAM_IDENTITY: "teamIdentity",
    TASKS: "tasks",
    LEAGUE_CENTRAL: "leagueCentral",
  };

  const OFFICE_NAV_TARGETS = {
    DASHBOARD: "dashboard",
    SIM_NEXT_GAME: "sim-next-game",
    TEAM_REPORT: "team-report",
    OWNER_GOALS: "owner-goals",
    ROSTER: "roster",
    INJURIES: "injuries",

    INBOX: "inbox",
    TRADE_CALLS: "trade-calls",
    STAFF: "staff",
    OWNER: "owner",

    CALENDAR: "calendar",
    SIM_TO_DATE: "sim-to-date",
    EVENTS: "events",
    NEXT_GAME: "next-game",

    DRAFT_CLASS: "draft-class",
    SCOUTING: "scouting",
    WATCHLIST: "watchlist",
    ASSIGN_SCOUTS: "assign-scouts",

    CONTRACTS: "contracts",
    EXTENSIONS: "extensions",
    FREE_AGENCY: "free-agency",
    SALARY_CAP: "salary-cap",

    SKATER_STATS: "skater-stats",
    GOALIE_STATS: "goalie-stats",
    TEAM_STATS: "team-stats",
    ADVANCED_STATS: "advanced-stats",

    LINES: "lines",
    POWERPLAY: "powerplay",
    PENALTYKILL: "penaltykill",
    DEPTH_CHART: "depth-chart",

    STORYLINES: "storylines",
    LEAGUE_NEWS: "league-news",
    GAME_RECAPS: "recaps",
    RUMORS: "rumors",

    AWARDS: "awards",
    RECORDS: "records",
    HISTORY: "history",
    RETIRED_NUMBERS: "retired-numbers",

    DRAFT_BOARD: "draft-board",
    PROSPECT_RANKINGS: "prospect-rankings",
    TEAM_NEEDS: "team-needs",
    DRAFT_LOTTERY: "draft-lottery",

    STANDINGS: "standings",
    PLAYOFF_RACE: "playoff-race",
    POWER_RANKINGS: "power-rankings",
    DIVISION: "division",

    GAME_PREVIEW: "game-preview",
    SIM_GAME: "sim-game",
    BROADCAST: "broadcast",
    MATCHUP: "matchup",

    TEAM_PROFILE: "team-profile",
    FANBASE: "fanbase",
    MORALE: "morale",
    OWNERSHIP: "ownership",

    TASKS: "tasks",
    OBJECTIVES: "objectives",
    URGENT_DECISIONS: "urgent-decisions",
    STAFF_NOTES: "staff-notes",

    LEAGUE_CENTRAL: "league-central",
  };

  /*
   * ==========================================================================
   * LEAGUE OPERATIONS — LIVE INTELLIGENCE DISPLAY
   * ==========================================================================
   * Wired end-to-end:
   *   - Screen: frontend/src/screens/LeagueOperations.js (SCREENS.LEAGUE_OPERATIONS / GM_WORLD)
   *   - API: GET /api/franchise/league-operations → services/league_operations.py
   *   - Also embedded (slim) on franchise /state as league_operations + franchise_pulse
   *   - Office wall "League Central" navigates here for CBA desk, cap forecast, markets, risk
   *
   * Read-only: CBA negotiations are display-only pressure estimates (no vote/apply yet).
   * ==========================================================================
   */

  export const FRANCHISE_COMMAND_GROUPS = {
    primary: { id: "primary", label: "Primary Commands" },
    operations: { id: "operations", label: "Hockey Operations" },
    frontOffice: { id: "frontOffice", label: "League & Front Office" },
    future: { id: "future", label: "Reserved / Coming Soon" },
  };

  const PLACEHOLDER_COPY = {
    gmPhone: {
      title: "GM Phone",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for inbox, trade calls, owner messages, and staff updates.",
    },
    legacyWall: {
      title: "Legacy Wall",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for awards, records, team history, and retired numbers.",
    },
    arenaWindow: {
      title: "Arena Window",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for game preview, broadcast, matchup reports, and game-day prep.",
    },
    decisionDesk: {
      title: "Decision Desk",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for tasks, objectives, urgent decisions, and staff notes.",
    },
    cultureWall: {
      title: "Franchise Culture Wall",
      subtitle: "This feature does not have a dedicated screen yet.",
      description:
        "Reserved for team profile, fanbase, morale dashboards, and ownership direction.",
    },
    leagueCentral: {
      title: "League Operations",
      subtitle: "CBA desk, cap forecast, and team revenue.",
      description:
        "League-wide economics — salary cap growth, escrow, relocation risk, and team money.",
    },
    inbox: {
      title: "Inbox",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for GM inbox and front office messaging.",
    },
    ownerDesk: {
      title: "Owner Desk",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for owner goals, approval, and executive pressure.",
    },
    leagueNews: {
      title: "League News",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for league-wide news wire and transaction feed.",
    },
    gameRecaps: {
      title: "Game Recaps",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for nightly scoresheets and game recaps.",
    },
    assignScouts: {
      title: "Assign Scouts",
      subtitle: "This feature does not have a dedicated screen yet.",
      description: "Reserved for scout assignments and coverage maps.",
    },
  };

  export const FRANCHISE_COMMAND_REGISTRY = [
    {
      id: "command-center",
      label: "Franchise Office",
      eyebrow: "Command Center",
      description: "Executive hub, franchise overview, and office systems.",
      group: "primary",
      target: "command-center",
      type: "hub",
      highlight: true,
      enabled: true,
    },
    {
      id: "roster",
      label: "Roster",
      eyebrow: "Personnel",
      description: "NHL roster, depth chart, roles, and injury list.",
      group: "primary",
      target: "roster",
      type: "navigate",
      screen: SCREENS.ROSTER,
      highlight: true,
      enabled: true,
    },
    {
      id: "calendar",
      label: "Calendar",
      eyebrow: "Schedule",
      description: "Season schedule, upcoming games, and league dates.",
      group: "primary",
      target: "calendar",
      type: "navigate",
      screen: SCREENS.CALENDAR,
      highlight: true,
      enabled: true,
    },
    {
      id: "strategy-board",
      label: "Strategy Board",
      eyebrow: "Tactics",
      description: "Forward lines, defensive pairs, and matchup planning.",
      group: "primary",
      target: "strategy-board",
      type: "navigate",
      screen: SCREENS.EDIT_LINES,
      highlight: true,
      enabled: true,
    },
    {
      id: "standings",
      label: "Standings",
      eyebrow: "League Table",
      description: "Division standings, playoff race, and points picture.",
      group: "primary",
      target: "standings",
      type: "navigate",
      screen: SCREENS.STATS,
      tab: "team",
      highlight: true,
      enabled: true,
    },
    {
      id: "draft-war-room",
      label: "Draft War Room",
      eyebrow: "Draft Mode",
      description: "Draft board prep, tiers, and selection strategy.",
      group: "primary",
      target: "draft-war-room",
      type: "navigate",
      screen: SCREENS.DRAFT_CLASS,
      highlight: true,
      enabled: true,
    },
    {
      id: "lines",
      label: "Lines",
      eyebrow: "Lineup",
      description: "Edit even-strength lines and defensive pairs.",
      group: "operations",
      target: "lines",
      type: "navigate",
      screen: SCREENS.EDIT_LINES,
      enabled: true,
    },
    {
      id: "power-play",
      label: "Power Play",
      eyebrow: "Special Teams",
      description: "Power play units and deployment.",
      group: "operations",
      target: "powerplay",
      type: "navigate",
      screen: SCREENS.POWER_PLAY,
      enabled: true,
    },
    {
      id: "penalty-kill",
      label: "Penalty Kill",
      eyebrow: "Special Teams",
      description: "Penalty kill units and structure.",
      group: "operations",
      target: "penaltykill",
      type: "navigate",
      screen: SCREENS.PENALTY_KILL,
      enabled: true,
    },
    {
      id: "scouting",
      label: "Scouting",
      eyebrow: "Amateur Ops",
      description: "Prospect reports, watchlists, and draft intel.",
      group: "operations",
      target: "scouting",
      type: "navigate",
      screen: SCREENS.SCOUTING,
      enabled: true,
    },
    {
      id: "team-needs",
      label: "Team Needs",
      eyebrow: "Roster Planning",
      description: "Positional needs and draft priorities.",
      group: "operations",
      target: "team-needs",
      type: "navigate",
      screen: SCREENS.TEAM_NEEDS,
      enabled: true,
    },
    {
      id: "contracts",
      label: "Contracts",
      eyebrow: "Cap Ledger",
      description: "Active contracts, extensions, and cap hits.",
      group: "operations",
      target: "contracts",
      type: "navigate",
      screen: SCREENS.CAP_LEDGER,
      tab: "contracts",
      enabled: true,
    },
    {
      id: "stats-analytics",
      label: "Stats / Analytics",
      eyebrow: "Performance Intel",
      description: "Skater, goalie, team, and advanced analytics.",
      group: "frontOffice",
      target: "stats-analytics",
      type: "navigate",
      screen: SCREENS.STATS,
      tab: "overview",
      enabled: true,
    },
    {
      id: "trade-hub",
      label: "Trade Hub",
      eyebrow: "Trade Floor",
      description: "Trade calls, offers, and roster moves.",
      group: "frontOffice",
      target: "trade-hub",
      type: "navigate",
      screen: SCREENS.TRADE,
      enabled: true,
    },
    {
      id: "free-agency",
      label: "Free Agency",
      eyebrow: "Market Wire",
      description: "UFA/RFA market, bids, and signings.",
      group: "frontOffice",
      target: "free-agency",
      type: "navigate",
      screen: SCREENS.FREE_AGENCY,
      enabled: true,
    },
    {
      id: "storylines",
      label: "Storylines / News",
      eyebrow: "Narrative",
      description: "League storylines, drama, and narrative beats.",
      group: "frontOffice",
      target: "storylines",
      type: "navigate",
      screen: SCREENS.STORYLINES,
      enabled: true,
    },
    {
      id: "chemistry",
      label: "Chemistry / Morale",
      eyebrow: "Locker Room",
      description: "Line chemistry, morale, and culture metrics.",
      group: "frontOffice",
      target: "chemistry",
      type: "navigate",
      screen: SCREENS.CHEMISTRY,
      enabled: true,
    },
    {
      id: "draft-class",
      label: "Draft Class",
      eyebrow: "Prospects",
      description: "Full draft class rankings and scouting dossiers.",
      group: "frontOffice",
      target: "draft-class",
      type: "navigate",
      screen: SCREENS.DRAFT_CLASS,
      enabled: true,
    },
    {
      id: "league-central",
      label: "League Operations",
      eyebrow: "League Economics",
      description: "CBA desk, cap forecast, team revenue, and relocation watch.",
      group: "frontOffice",
      target: "league-central",
      type: "navigate",
      screen: SCREENS.LEAGUE_OPERATIONS,
      enabled: true,
    },
    {
      id: "gm-phone",
      label: "GM Phone",
      eyebrow: "Communications",
      description: "Inbox, trade calls, owner messages, and staff updates.",
      group: "future",
      target: "gm-phone",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.gmPhone,
      enabled: true,
    },
    {
      id: "legacy-wall",
      label: "Legacy Wall",
      eyebrow: "History",
      description: "Awards, records, banners, and retired numbers.",
      group: "future",
      target: "legacy-wall",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.legacyWall,
      enabled: true,
    },
    {
      id: "arena-window",
      label: "Arena Window",
      eyebrow: "Game Day",
      description: "Game preview, broadcast, and matchup prep.",
      group: "future",
      target: "arena-window",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.arenaWindow,
      enabled: true,
    },
    {
      id: "decision-desk",
      label: "Decision Desk",
      eyebrow: "Tasks",
      description: "Pending decisions, objectives, and urgent items.",
      group: "future",
      target: "decision-desk",
      type: "placeholder",
      placeholder: PLACEHOLDER_COPY.decisionDesk,
      enabled: true,
    },
  ];

  function commandToRoute(cmd) {
    if (!cmd) return null;
    if (cmd.type === "hub") return { type: "hub" };
    if (cmd.type === "placeholder") {
      return {
        type: "placeholder",
        placeholder: { ...cmd.placeholder, targetId: cmd.target },
      };
    }
    if (cmd.type === "navigate" && cmd.screen) {
      const route = { type: "screen", screen: cmd.screen };
      if (cmd.screen === SCREENS.CAP_LEDGER && cmd.tab) route.capTab = cmd.tab;
      if (cmd.screen === SCREENS.STATS && cmd.tab) route.statsTab = cmd.tab;
      return route;
    }
    return null;
  }

  const COMMAND_TARGET_ROUTES = (() => {
    const routes = {};

    FRANCHISE_COMMAND_REGISTRY.forEach((cmd) => {
      const route = commandToRoute(cmd);
      if (route) routes[cmd.target] = route;
    });

    const screenRoute = (screen, extras = {}) => ({ type: "screen", screen, ...extras });
    const ph = (copy, targetId) => ({
      type: "placeholder",
      placeholder: { ...copy, targetId },
    });

    routes[OFFICE_NAV_TARGETS.DASHBOARD] = { type: "hub" };
    routes[OFFICE_NAV_TARGETS.ROSTER] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.INJURIES] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.DEPTH_CHART] = screenRoute(SCREENS.ROSTER);
    routes[OFFICE_NAV_TARGETS.CALENDAR] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.EVENTS] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.NEXT_GAME] = screenRoute(SCREENS.CALENDAR);
    routes[OFFICE_NAV_TARGETS.DRAFT_CLASS] = screenRoute(SCREENS.DRAFT_CLASS);
    routes[OFFICE_NAV_TARGETS.DRAFT_BOARD] = screenRoute(SCREENS.DRAFT_CLASS);
    routes[OFFICE_NAV_TARGETS.PROSPECT_RANKINGS] = screenRoute(SCREENS.DRAFT_CLASS);
    routes["draft-war-room"] = screenRoute(SCREENS.DRAFT_CLASS);
    routes["strategy-board"] = screenRoute(SCREENS.EDIT_LINES);
    routes["stats-analytics"] = screenRoute(SCREENS.STATS, { statsTab: "overview" });
    routes["trade-hub"] = screenRoute(SCREENS.TRADE);
    routes["command-center"] = { type: "hub" };
    routes[OFFICE_NAV_TARGETS.DRAFT_LOTTERY] = screenRoute(SCREENS.DRAFT_LOTTERY);
    routes[OFFICE_NAV_TARGETS.TEAM_NEEDS] = screenRoute(SCREENS.TEAM_NEEDS);
    routes[OFFICE_NAV_TARGETS.SCOUTING] = screenRoute(SCREENS.SCOUTING);
    routes[OFFICE_NAV_TARGETS.WATCHLIST] = screenRoute(SCREENS.SCOUTING);
    routes[OFFICE_NAV_TARGETS.CONTRACTS] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "contracts" });
    routes[OFFICE_NAV_TARGETS.EXTENSIONS] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "contracts" });
    // Hub Free Agency opens the Wire screen (same UI as offseason) — not Cap Ledger,
    // and not an offseason stage reopen (works in regular season / playoffs).
    routes[OFFICE_NAV_TARGETS.FREE_AGENCY] = screenRoute(SCREENS.FREE_AGENCY);
    routes[OFFICE_NAV_TARGETS.SALARY_CAP] = screenRoute(SCREENS.CAP_LEDGER, { capTab: "salaryCap" });
    routes[OFFICE_NAV_TARGETS.SKATER_STATS] = screenRoute(SCREENS.STATS, { statsTab: "players" });
    routes[OFFICE_NAV_TARGETS.GOALIE_STATS] = screenRoute(SCREENS.STATS, { statsTab: "goalies" });
    routes[OFFICE_NAV_TARGETS.TEAM_STATS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.ADVANCED_STATS] = screenRoute(SCREENS.STATS, { statsTab: "advanced" });
    routes[OFFICE_NAV_TARGETS.STANDINGS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.PLAYOFF_RACE] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.POWER_RANKINGS] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.DIVISION] = screenRoute(SCREENS.STATS, { statsTab: "team" });
    routes[OFFICE_NAV_TARGETS.LINES] = screenRoute(SCREENS.EDIT_LINES);
    routes[OFFICE_NAV_TARGETS.POWERPLAY] = screenRoute(SCREENS.POWER_PLAY);
    routes[OFFICE_NAV_TARGETS.PENALTYKILL] = screenRoute(SCREENS.PENALTY_KILL);
    routes[OFFICE_NAV_TARGETS.STORYLINES] = screenRoute(SCREENS.STORYLINES);
    routes[OFFICE_NAV_TARGETS.RUMORS] = screenRoute(SCREENS.STORYLINES);
    routes[OFFICE_NAV_TARGETS.MORALE] = screenRoute(SCREENS.CHEMISTRY);
    routes[OFFICE_NAV_TARGETS.TRADE_CALLS] = screenRoute(SCREENS.TRADE);

    routes[OFFICE_NAV_TARGETS.INBOX] = ph(PLACEHOLDER_COPY.inbox, OFFICE_NAV_TARGETS.INBOX);
    routes[OFFICE_NAV_TARGETS.STAFF] = ph(PLACEHOLDER_COPY.gmPhone, OFFICE_NAV_TARGETS.STAFF);
    routes[OFFICE_NAV_TARGETS.OWNER] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.OWNER);
    routes[OFFICE_NAV_TARGETS.TEAM_REPORT] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.TEAM_REPORT);
    routes[OFFICE_NAV_TARGETS.OWNER_GOALS] = ph(PLACEHOLDER_COPY.ownerDesk, OFFICE_NAV_TARGETS.OWNER_GOALS);
    routes[OFFICE_NAV_TARGETS.TASKS] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.TASKS);
    routes[OFFICE_NAV_TARGETS.OBJECTIVES] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.OBJECTIVES);
    routes[OFFICE_NAV_TARGETS.URGENT_DECISIONS] = ph(
      PLACEHOLDER_COPY.decisionDesk,
      OFFICE_NAV_TARGETS.URGENT_DECISIONS
    );
    routes[OFFICE_NAV_TARGETS.STAFF_NOTES] = ph(PLACEHOLDER_COPY.decisionDesk, OFFICE_NAV_TARGETS.STAFF_NOTES);
    routes[OFFICE_NAV_TARGETS.AWARDS] = screenRoute(SCREENS.STATS, { statsTab: "overview" });
    routes[OFFICE_NAV_TARGETS.RECORDS] = ph(PLACEHOLDER_COPY.legacyWall, OFFICE_NAV_TARGETS.RECORDS);
    routes[OFFICE_NAV_TARGETS.HISTORY] = ph(PLACEHOLDER_COPY.legacyWall, OFFICE_NAV_TARGETS.HISTORY);
    routes[OFFICE_NAV_TARGETS.RETIRED_NUMBERS] = ph(
      PLACEHOLDER_COPY.legacyWall,
      OFFICE_NAV_TARGETS.RETIRED_NUMBERS
    );
    routes[OFFICE_NAV_TARGETS.GAME_PREVIEW] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.GAME_PREVIEW);
    routes[OFFICE_NAV_TARGETS.BROADCAST] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.BROADCAST);
    routes[OFFICE_NAV_TARGETS.MATCHUP] = ph(PLACEHOLDER_COPY.arenaWindow, OFFICE_NAV_TARGETS.MATCHUP);
    routes[OFFICE_NAV_TARGETS.SIM_GAME] = null;
    routes[OFFICE_NAV_TARGETS.TEAM_PROFILE] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.TEAM_PROFILE);
    routes[OFFICE_NAV_TARGETS.FANBASE] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.FANBASE);
    routes[OFFICE_NAV_TARGETS.OWNERSHIP] = ph(PLACEHOLDER_COPY.cultureWall, OFFICE_NAV_TARGETS.OWNERSHIP);
    routes[OFFICE_NAV_TARGETS.LEAGUE_CENTRAL] = screenRoute(SCREENS.LEAGUE_OPERATIONS);
    routes[OFFICE_NAV_TARGETS.LEAGUE_NEWS] = ph(PLACEHOLDER_COPY.leagueNews, OFFICE_NAV_TARGETS.LEAGUE_NEWS);
    routes[OFFICE_NAV_TARGETS.GAME_RECAPS] = ph(PLACEHOLDER_COPY.gameRecaps, OFFICE_NAV_TARGETS.GAME_RECAPS);
    routes[OFFICE_NAV_TARGETS.ASSIGN_SCOUTS] = ph(
      PLACEHOLDER_COPY.assignScouts,
      OFFICE_NAV_TARGETS.ASSIGN_SCOUTS
    );
    routes["gm-phone"] = ph(PLACEHOLDER_COPY.gmPhone, "gm-phone");
    routes["legacy-wall"] = ph(PLACEHOLDER_COPY.legacyWall, "legacy-wall");
    routes["arena-window"] = ph(PLACEHOLDER_COPY.arenaWindow, "arena-window");
    routes["decision-desk"] = ph(PLACEHOLDER_COPY.decisionDesk, "decision-desk");
    routes["culture-wall"] = ph(PLACEHOLDER_COPY.cultureWall, "culture-wall");

    return routes;
  })();

  export function resolveCommandTarget(target) {
    if (!target) return null;
    return COMMAND_TARGET_ROUTES[target] || null;
  }

  const QUICK_MENU_BADGE_TITLES = {
    Deadline: "Trade deadline window",
    Draft: "Draft week priority",
    FA: "Free agency period",
    Offseason: "Offseason operations",
    Pressure: "Owner pressure elevated",
    Injuries: "Injury crisis active",
    Playoffs: "Playoff push",
    Slide: "Losing streak flagged",
    Urgent: "Urgent desk item",
  };

  export {
    OFFICE_PANEL_IDS,
    OFFICE_NAV_TARGETS,
    deriveOfficeMood,
    buildOfficeUrgentItems,
    getDynamicPanelCopy,
    getContextualCommandRegistry,
    LOW_POWER_STORAGE_KEY,
  };

  const PANEL_TO_COMMAND_TARGET = {
    [OFFICE_PANEL_IDS.DASHBOARD]: "command-center",
    [OFFICE_PANEL_IDS.MESSAGES]: "trade-hub",
    [OFFICE_PANEL_IDS.CALENDAR]: "calendar",
    [OFFICE_PANEL_IDS.SCOUTING]: "scouting",
    [OFFICE_PANEL_IDS.CONTRACTS]: "contracts",
    [OFFICE_PANEL_IDS.STATS]: "stats-analytics",
    [OFFICE_PANEL_IDS.LINES]: "lines",
    [OFFICE_PANEL_IDS.NEWS]: "storylines",
    [OFFICE_PANEL_IDS.AWARDS]: "stats-analytics",
    [OFFICE_PANEL_IDS.DRAFT]: "draft-war-room",
    [OFFICE_PANEL_IDS.DRAFT_CLASS]: "draft-class",
    [OFFICE_PANEL_IDS.ROSTER]: "roster",
    [OFFICE_PANEL_IDS.STANDINGS]: "standings",
    [OFFICE_PANEL_IDS.GAME_DAY]: "arena-window",
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: "culture-wall",
    [OFFICE_PANEL_IDS.TASKS]: "decision-desk",
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: "league-central",
  };

  const OFFICE_INTERACTIVE_PANEL_IDS = [
    OFFICE_PANEL_IDS.DASHBOARD,
    OFFICE_PANEL_IDS.MESSAGES,
    OFFICE_PANEL_IDS.CALENDAR,
    OFFICE_PANEL_IDS.SCOUTING,
    OFFICE_PANEL_IDS.CONTRACTS,
    OFFICE_PANEL_IDS.STATS,
    OFFICE_PANEL_IDS.NEWS,
    OFFICE_PANEL_IDS.TASKS,
    OFFICE_PANEL_IDS.TEAM_IDENTITY,
    OFFICE_PANEL_IDS.LINES,
    OFFICE_PANEL_IDS.STANDINGS,
    OFFICE_PANEL_IDS.LEAGUE_CENTRAL,
    OFFICE_PANEL_IDS.DRAFT,
    OFFICE_PANEL_IDS.DRAFT_CLASS,
    OFFICE_PANEL_IDS.ROSTER,
    OFFICE_PANEL_IDS.AWARDS,
    OFFICE_PANEL_IDS.GAME_DAY,
  ];
  
  const PANEL_CONTENT = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      title: "Command Interface",
      eyebrow: "Executive Command Screen",
      description:
        "Review your franchise overview, roster status, owner goals, cap pressure, injuries, staff notes, and next decisions.",
      actions: [
        ["Sim Next Game", OFFICE_NAV_TARGETS.SIM_NEXT_GAME],
        ["Team Report", OFFICE_NAV_TARGETS.TEAM_REPORT],
        ["Owner Goals", OFFICE_NAV_TARGETS.OWNER_GOALS],
        ["Roster Status", OFFICE_NAV_TARGETS.ROSTER],
        ["Injury Watch", OFFICE_NAV_TARGETS.INJURIES],
      ],
    },
  
    [OFFICE_PANEL_IDS.MESSAGES]: {
      title: "Trade Desk",
      eyebrow: "Negotiation Table",
      description:
        "Trade calls, offers, counter-proposals, and front-office negotiation paperwork.",
      actions: [
        ["Trade Calls", OFFICE_NAV_TARGETS.TRADE_CALLS],
        ["Inbox", OFFICE_NAV_TARGETS.INBOX],
        ["Staff Updates", OFFICE_NAV_TARGETS.STAFF],
        ["Owner Messages", OFFICE_NAV_TARGETS.OWNER],
      ],
    },
  
    [OFFICE_PANEL_IDS.CALENDAR]: {
      title: "Season Calendar",
      eyebrow: "Desk Calendar",
      description:
        "Review the schedule, upcoming games, league events, simulation dates, and important deadlines.",
      actions: [
        ["Schedule", OFFICE_NAV_TARGETS.CALENDAR],
        ["Sim to Date", OFFICE_NAV_TARGETS.SIM_TO_DATE],
        ["Important Dates", OFFICE_NAV_TARGETS.EVENTS],
        ["Next Game", OFFICE_NAV_TARGETS.NEXT_GAME],
      ],
    },
  
    [OFFICE_PANEL_IDS.SCOUTING]: {
      title: "Scouting Room",
      eyebrow: "Scouting Kit",
      description:
        "Review prospects, assignments, scouting reports, draft rankings, and watchlists.",
      actions: [
        ["Draft Class", OFFICE_NAV_TARGETS.DRAFT_CLASS],
        ["Scouting Reports", OFFICE_NAV_TARGETS.SCOUTING],
        ["Watchlist", OFFICE_NAV_TARGETS.WATCHLIST],
        ["Assign Scouts", OFFICE_NAV_TARGETS.ASSIGN_SCOUTS],
      ],
    },
  
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      title: "Contract Office",
      eyebrow: "Cap Ledger",
      description:
        "Manage contracts, free agency, salary cap, and roster money.",
      actions: [
        ["Contracts", OFFICE_NAV_TARGETS.CONTRACTS],
        ["Extensions", OFFICE_NAV_TARGETS.EXTENSIONS],
        ["Free Agency", OFFICE_NAV_TARGETS.FREE_AGENCY],
        ["Salary Cap", OFFICE_NAV_TARGETS.SALARY_CAP],
      ],
    },
  
    [OFFICE_PANEL_IDS.STATS]: {
      title: "Analytics Room",
      eyebrow: "Analytics Tablet",
      description:
        "Study player performance, team analytics, xGF%, CF%, PDO, power play, penalty kill, and trends.",
      actions: [
        ["Skater Stats", OFFICE_NAV_TARGETS.SKATER_STATS],
        ["Goalie Stats", OFFICE_NAV_TARGETS.GOALIE_STATS],
        ["Team Analytics", OFFICE_NAV_TARGETS.TEAM_STATS],
        ["Advanced Metrics", OFFICE_NAV_TARGETS.ADVANCED_STATS],
      ],
    },
  
    [OFFICE_PANEL_IDS.LINES]: {
      title: "Line Strategy Board",
      eyebrow: "Rink Whiteboard",
      description:
        "Edit forward lines, defensive pairs, special teams, matchup plans, and tactical setup.",
      actions: [
        ["Edit Lines", OFFICE_NAV_TARGETS.LINES],
        ["Power Play", OFFICE_NAV_TARGETS.POWERPLAY],
        ["Penalty Kill", OFFICE_NAV_TARGETS.PENALTYKILL],
        ["Depth Chart", OFFICE_NAV_TARGETS.DEPTH_CHART],
      ],
    },
  
    [OFFICE_PANEL_IDS.NEWS]: {
      title: "League Storylines",
      eyebrow: "Newspaper Stack",
      description:
        "Read headlines, rumors, game recaps, player drama, league movement, and front office noise.",
      actions: [
        ["Storylines", OFFICE_NAV_TARGETS.STORYLINES],
        ["League News", OFFICE_NAV_TARGETS.LEAGUE_NEWS],
        ["Game Recaps", OFFICE_NAV_TARGETS.GAME_RECAPS],
        ["Rumors", OFFICE_NAV_TARGETS.RUMORS],
      ],
    },
  
    [OFFICE_PANEL_IDS.AWARDS]: {
      title: "Legacy Wall",
      eyebrow: "Trophy Shelf",
      description:
        "View awards, records, team history, retired numbers, banners, and legacy moments.",
      actions: [
        ["Awards", OFFICE_NAV_TARGETS.AWARDS],
        ["Records", OFFICE_NAV_TARGETS.RECORDS],
        ["Team History", OFFICE_NAV_TARGETS.HISTORY],
        ["Retired Numbers", OFFICE_NAV_TARGETS.RETIRED_NUMBERS],
      ],
    },
  
    [OFFICE_PANEL_IDS.DRAFT]: {
      title: "Draft War Room",
      eyebrow: "Entry Draft Floor",
      description:
        "Conduct the NHL Entry Draft — board, pick order, and selection strategy.",
      actions: [
        ["Draft Board", OFFICE_NAV_TARGETS.DRAFT_BOARD],
        ["Prospect Rankings", OFFICE_NAV_TARGETS.PROSPECT_RANKINGS],
        ["Draft Lottery", OFFICE_NAV_TARGETS.DRAFT_LOTTERY],
        ["Team Needs", OFFICE_NAV_TARGETS.TEAM_NEEDS],
      ],
    },

    [OFFICE_PANEL_IDS.DRAFT_CLASS]: {
      title: "Draft Class",
      eyebrow: "Prospect Research",
      description:
        "Research the draft pool — rankings, dossiers, tiers, and scouting notes before draft day.",
      actions: [
        ["Draft Class", OFFICE_NAV_TARGETS.DRAFT_CLASS],
        ["Prospect Rankings", OFFICE_NAV_TARGETS.PROSPECT_RANKINGS],
        ["Scouting", OFFICE_NAV_TARGETS.SCOUTING],
        ["Watchlist", OFFICE_NAV_TARGETS.WATCHLIST],
      ],
    },

    [OFFICE_PANEL_IDS.ROSTER]: {
      title: "Roster Board",
      eyebrow: "Dressing Room",
      description:
        "NHL roster, depth chart, scratches, and injury list.",
      actions: [
        ["Roster", OFFICE_NAV_TARGETS.ROSTER],
        ["Depth Chart", OFFICE_NAV_TARGETS.DEPTH_CHART],
        ["Injury Watch", OFFICE_NAV_TARGETS.INJURIES],
        ["Lines", OFFICE_NAV_TARGETS.LINES],
      ],
    },

    [OFFICE_PANEL_IDS.STANDINGS]: {
      title: "League Standings",
      eyebrow: "Standings Wall",
      description:
        "Track division races, playoff odds, conference battles, league rankings, and power movement.",
      actions: [
        ["Standings", OFFICE_NAV_TARGETS.STANDINGS],
        ["Playoff Race", OFFICE_NAV_TARGETS.PLAYOFF_RACE],
        ["Power Rankings", OFFICE_NAV_TARGETS.POWER_RANKINGS],
        ["Division View", OFFICE_NAV_TARGETS.DIVISION],
      ],
    },
  
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      title: "Arena Window",
      eyebrow: "Arena View",
      description:
        "Prepare for the next matchup, review lines, watch broadcast, check injuries, and simulate.",
      actions: [
        ["Game Preview", OFFICE_NAV_TARGETS.GAME_PREVIEW],
        ["Sim Game", OFFICE_NAV_TARGETS.SIM_GAME],
        ["Broadcast", OFFICE_NAV_TARGETS.BROADCAST],
        ["Matchup Report", OFFICE_NAV_TARGETS.MATCHUP],
      ],
    },
  
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      title: "Franchise Culture Wall",
      eyebrow: "Logo Wall",
      description:
        "Review team branding, culture, fanbase, morale, ownership direction, and long-term identity.",
      actions: [
        ["Team Profile", OFFICE_NAV_TARGETS.TEAM_PROFILE],
        ["Fanbase", OFFICE_NAV_TARGETS.FANBASE],
        ["Morale", OFFICE_NAV_TARGETS.MORALE],
        ["Ownership", OFFICE_NAV_TARGETS.OWNERSHIP],
      ],
    },
  
    [OFFICE_PANEL_IDS.TASKS]: {
      title: "Decision Desk",
      eyebrow: "Clipboard",
      description:
        "Review pending decisions, reminders, urgent items, owner pressure, and front office priorities.",
      actions: [
        ["Tasks", OFFICE_NAV_TARGETS.TASKS],
        ["Objectives", OFFICE_NAV_TARGETS.OBJECTIVES],
        ["Urgent Decisions", OFFICE_NAV_TARGETS.URGENT_DECISIONS],
        ["Staff Notes", OFFICE_NAV_TARGETS.STAFF_NOTES],
      ],
    },

    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      title: "League Operations",
      eyebrow: "League Economics",
      description: "CBA rules, cap growth, team revenue, and relocation risk.",
      actions: [
        ["Cap Forecast", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["Team Money", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["CBA Desk", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
        ["Relocation Watch", OFFICE_NAV_TARGETS.LEAGUE_CENTRAL],
      ],
    },
  };

  const PANEL_STAFF_SPEAKERS = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      role: "Assistant GM",
      fallback: "Start with what matters most today, then work outward.",
    },
    [OFFICE_PANEL_IDS.MESSAGES]: {
      role: "Assistant GM",
      fallback: "If the phone keeps ringing, something in the market is moving.",
    },
    [OFFICE_PANEL_IDS.CALENDAR]: {
      role: "Assistant GM",
      fallback: "The calendar is the truth. Miss a date and the league will not wait.",
    },
    [OFFICE_PANEL_IDS.SCOUTING]: {
      role: "Head Scout",
      fallback: "The public board is not your board. Review the tiers before draft day.",
    },
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      role: "Cap Specialist",
      fallback: "Do not approve anything long-term until we know the projected cap hit.",
    },
    [OFFICE_PANEL_IDS.STATS]: {
      role: "Analytics Director",
      fallback: "The standings are one story. The underlying numbers may be another.",
    },
    [OFFICE_PANEL_IDS.LINES]: {
      role: "Head Coach",
      fallback: "Matchups and minutes matter as much as names on the card.",
    },
    [OFFICE_PANEL_IDS.NEWS]: {
      role: "Assistant GM",
      fallback: "Narrative pressure is real even when the box score looks fine.",
    },
    [OFFICE_PANEL_IDS.AWARDS]: {
      role: "Assistant GM",
      fallback: "Legacy wins recruiting battles you never see on the scoresheet.",
    },
    [OFFICE_PANEL_IDS.DRAFT]: {
      role: "Head Scout",
      fallback: "Your board should be opinionated, tiered, and ready for chaos.",
    },
    [OFFICE_PANEL_IDS.DRAFT_CLASS]: {
      role: "Head Scout",
      fallback: "Know the pool before you walk into the war room.",
    },
    [OFFICE_PANEL_IDS.ROSTER]: {
      role: "Head Coach",
      fallback: "The lineup board is where roles become minutes.",
    },
    [OFFICE_PANEL_IDS.STANDINGS]: {
      role: "Analytics Director",
      fallback: "Playoff probability and division math are tightening every week.",
    },
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      role: "Head Coach",
      fallback: "Game-day prep is where culture meets execution.",
    },
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      role: "Owner",
      fallback: "The building feels what the franchise believes about itself.",
    },
    [OFFICE_PANEL_IDS.TASKS]: {
      role: "Assistant GM",
      fallback: "Urgent does not always mean important, but ignored urgent becomes expensive.",
    },
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      role: "Assistant GM",
      fallback: "League money drives the next cap number.",
    },
  };

  const PANEL_PRESSURE_COPY = {
    [OFFICE_PANEL_IDS.CONTRACTS]:
      "Ignored contracts can turn into arbitration pressure or expensive July panic.",
    [OFFICE_PANEL_IDS.SCOUTING]:
      "If the board is stale by draft week, your staff may miss risers.",
    [OFFICE_PANEL_IDS.LINES]:
      "If injuries are ignored, fatigue and role mismatch can snowball.",
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]:
      "Low confidence can narrow your rebuild runway.",
    [OFFICE_PANEL_IDS.MESSAGES]:
      "Some offers expire after the next game or phase advance.",
    [OFFICE_PANEL_IDS.TASKS]:
      "Deferred decisions tend to arrive louder and more expensive.",
  };

  const PANEL_CAMERA_TARGETS = {
    [OFFICE_PANEL_IDS.DASHBOARD]: {
      position: [1.05, 1.48, 2.28],
      target: [0.22, 1.22, 0.18],
      fov: 38,
    },
    [OFFICE_PANEL_IDS.MESSAGES]: {
      position: [-1.5, 2.4, -0.75],
      target: [-2.44, 2.44, -3.3],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.CALENDAR]: {
      position: [1.72, 1.42, 2.05],
      target: [1.48, 1.02, 0.62],
      fov: 32,
    },
    [OFFICE_PANEL_IDS.SCOUTING]: {
      position: [-2.0, 1.7, -1.35],
      target: [-4.1, 1.62, -2.35],
      fov: 38,
    },
    [OFFICE_PANEL_IDS.CONTRACTS]: {
      position: [-2.05, 1.3, -0.85],
      target: [-3.38, 1.12, -3.3],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.STATS]: {
      position: [1.5, 2.4, -0.75],
      target: [2.44, 2.44, -3.3],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.NEWS]: {
      position: [1.3, 1.3, -0.85],
      target: [2.08, 1.12, -3.28],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.TASKS]: {
      position: [1.35, 1.52, 2.22],
      target: [1.22, 1.1, 0.95],
      fov: 36,
    },
    [OFFICE_PANEL_IDS.TEAM_IDENTITY]: {
      position: [0, 2.6, -0.72],
      target: [0, 2.68, -3.34],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.LINES]: {
      position: [0.72, 2.4, -0.78],
      target: [1.16, 2.44, -3.32],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.STANDINGS]: {
      position: [2.05, 1.3, -0.85],
      target: [3.38, 1.12, -3.32],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: {
      position: [2.28, 2.4, -0.7],
      target: [3.72, 2.46, -3.34],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.DRAFT]: {
      position: [-1.55, 1.72, 1.85],
      target: [-4.15, 1.65, 0.45],
      fov: 40,
    },
    [OFFICE_PANEL_IDS.DRAFT_CLASS]: {
      position: [-2.28, 2.4, -0.7],
      target: [-3.72, 2.46, -3.34],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.ROSTER]: {
      position: [-0.72, 2.4, -0.78],
      target: [-1.16, 2.44, -3.32],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.AWARDS]: {
      position: [-1.3, 1.3, -0.85],
      target: [-2.08, 1.12, -3.26],
      fov: 30,
    },
    [OFFICE_PANEL_IDS.GAME_DAY]: {
      position: [2.05, 1.62, 1.85],
      target: [3.85, 1.65, 1.0],
      fov: 36,
    },
  };

  function getDynamicPanelCopy(
    panelId,
    basePanel,
    franchiseState,
    team,
    officeMood,
    urgentItems,
    officeSummary = null
  ) {
    const panel = basePanel || PANEL_CONTENT[panelId] || {};
    const mood = officeMood || deriveOfficeMood(franchiseState, team, officeSummary || {});
    const urgent = officeSafeArray(urgentItems);
    const urgentCount = urgent.length;
    const phase = officePhaseText(franchiseState);
    const topUrgent = urgent[0]?.title || "routine franchise maintenance";
    const speaker = PANEL_STAFF_SPEAKERS[panelId] || {
      role: "Assistant GM",
      fallback: "Keep the room calm and the decisions sharp.",
    };

    let description = panel.description || "";
    let staffNote = speaker.fallback;
    let pressureLine = PANEL_PRESSURE_COPY[panelId] || "";

    if (panelId === OFFICE_PANEL_IDS.DASHBOARD) {
      description = `Your front office has ${urgentCount} urgent item${urgentCount === 1 ? "" : "s"}, current phase is ${phase}, and the next major decision is ${topUrgent}.`;
      staffNote = `We have ${urgentCount} fires and ${mood.pendingTasks || 0} queued decisions. Triage before you get pulled into noise.`;
    } else if (panelId === OFFICE_PANEL_IDS.MESSAGES) {
      const tradeTone =
        mood.isTradeDeadline || mood.unreadMessages > 2 ? "active" : "quiet";
      description = `${mood.unreadMessages || 0} unread messages. Trade calls are ${tradeTone} depending on league phase.`;
      staffNote =
        mood.isTradeDeadline
          ? "Phones are hot. Filter noise and protect your leverage."
          : speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.CONTRACTS) {
      if (mood.isFreeAgency || mood.isOffseason) {
        description =
          "Contract season is live. Market pressure, comparables, and cap timing are all moving.";
      }
      const capRaw =
        officeSummary?.capSpaceMillions ??
        team?.cap_space ??
        team?.capSpace ??
        franchiseState?.cap_space;
      const capMillions = officeCapMillions(capRaw);
      if (Number.isFinite(capMillions) && capMillions < 2.0) {
        description = `Cap space is tight at ${formatMoney(capMillions)}. Every move needs a second look.`;
        pressureLine = PANEL_PRESSURE_COPY[panelId];
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.SCOUTING) {
      if (mood.isDraftWeek || mood.isOffseason) {
        description =
          "Final board review is recommended before selections and tier calls lock in.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.LINES) {
      if (mood.hasInjuryCrisis) {
        description =
          "Lineup decisions are unstable with multiple injuries affecting roles and minutes.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.STATS) {
      if (mood.isLosingStreak) {
        description =
          "Analytics staff is flagging performance issues beneath the recent results.";
      }
      staffNote = speaker.fallback;
    } else if (panelId === OFFICE_PANEL_IDS.TEAM_IDENTITY) {
      if (mood.hasOwnerPressure) {
        description =
          "Ownership expectations are elevated. Culture and results are being measured together.";
        staffNote =
          "They are watching the room as closely as the standings. Keep the message consistent.";
        pressureLine = PANEL_PRESSURE_COPY[panelId];
      }
    } else if (panelId === OFFICE_PANEL_IDS.TASKS) {
      description = `${mood.pendingTasks || 0} pending decisions and ${urgentCount} urgent desk items need executive attention.`;
      staffNote = speaker.fallback;
    }

    if (panelId === OFFICE_PANEL_IDS.MESSAGES && mood.isTradeDeadline) {
      pressureLine = PANEL_PRESSURE_COPY[panelId];
    }
    if (panelId === OFFICE_PANEL_IDS.LINES && mood.hasInjuryCrisis) {
      pressureLine = PANEL_PRESSURE_COPY[panelId];
    }

    return {
      ...panel,
      description,
      staffNote,
      staffRole: speaker.role,
      pressureLine: pressureLine || PANEL_PRESSURE_COPY[panelId] || null,
    };
  }

  const QUICK_MENU_BADGE_RULES = {
    [OFFICE_PANEL_IDS.MESSAGES]: ["trade_deadline", "owner_pressure"],
    [OFFICE_PANEL_IDS.CONTRACTS]: ["free_agency", "offseason", "trade_deadline"],
    [OFFICE_PANEL_IDS.DRAFT]: ["draft_week"],
    [OFFICE_PANEL_IDS.SCOUTING]: ["draft_week"],
    [OFFICE_PANEL_IDS.TASKS]: ["owner_pressure"],
    [OFFICE_PANEL_IDS.LINES]: ["injury_crisis"],
    [OFFICE_PANEL_IDS.GAME_DAY]: ["playoffs", "injury_crisis"],
    [OFFICE_PANEL_IDS.STATS]: ["losing_streak"],
  };

  function getQuickMenuBadge(panelId, officeMood, urgentItems) {
    const mood = officeMood || {};
    const rules = QUICK_MENU_BADGE_RULES[panelId] || [];
    const urgent = officeSafeArray(urgentItems);

    if (rules.includes("trade_deadline") && mood.isTradeDeadline) return "Deadline";
    if (rules.includes("draft_week") && mood.isDraftWeek) return "Draft";
    if (rules.includes("free_agency") && mood.isFreeAgency) return "FA";
    if (rules.includes("offseason") && mood.isOffseason) return "Offseason";
    if (rules.includes("owner_pressure") && mood.hasOwnerPressure) return "Pressure";
    if (rules.includes("injury_crisis") && mood.hasInjuryCrisis) return "Injuries";
    if (rules.includes("playoffs") && mood.isPlayoffs) return "Playoffs";
    if (rules.includes("losing_streak") && mood.isLosingStreak) return "Slide";

    const panelUrgent = urgent.find((item) => {
      if (panelId === OFFICE_PANEL_IDS.MESSAGES) return item.type === "messages" || item.type === "trade";
      if (panelId === OFFICE_PANEL_IDS.TASKS) return item.type === "tasks";
      if (panelId === OFFICE_PANEL_IDS.CONTRACTS) return item.type === "contracts";
      if (panelId === OFFICE_PANEL_IDS.SCOUTING || panelId === OFFICE_PANEL_IDS.DRAFT) {
        return item.type === "draft";
      }
      if (panelId === OFFICE_PANEL_IDS.LINES) return item.type === "injuries";
      if (panelId === OFFICE_PANEL_IDS.GAME_DAY) return item.type === "game";
      return false;
    });

    if (panelUrgent?.severity === "critical" || panelUrgent?.severity === "high") {
      return "Urgent";
    }

    return "";
  }

  function getContextualCommandRegistry(baseRegistry, officeMood, urgentItems) {
    const mood = officeMood || {};
    const priority = [];

    const pushIds = (ids) => {
      ids.forEach((id) => {
        if (!priority.includes(id)) priority.push(id);
      });
    };

    if (mood.isTradeDeadline) {
      pushIds(["trade-hub", "league-central", "contracts", "stats-analytics", "lines"]);
    } else if (mood.isDraftWeek) {
      pushIds(["draft-war-room", "draft-class", "scouting", "team-needs", "contracts"]);
    } else if (mood.isFreeAgency || mood.isOffseason) {
      pushIds(["free-agency", "contracts", "scouting", "team-needs", "draft-class"]);
    } else if (mood.isPlayoffs) {
      pushIds(["strategy-board", "lines", "standings", "stats-analytics", "storylines"]);
    } else {
      pushIds(["calendar", "strategy-board", "lines", "standings", "roster"]);
    }

    if (mood.hasOwnerPressure) {
      pushIds(["decision-desk", "command-center", "storylines"]);
    }
    if (mood.hasInjuryCrisis) {
      pushIds(["lines", "roster", "strategy-board"]);
    }

    const rank = new Map(priority.map((id, index) => [id, index]));

    return officeSafeArray(baseRegistry)
      .map((item) => ({
        ...item,
        badge: getQuickMenuBadgeForCommand(item, mood, urgentItems),
      }))
      .sort((a, b) => {
        const aRank = rank.has(a.id) ? rank.get(a.id) : 99;
        const bRank = rank.has(b.id) ? rank.get(b.id) : 99;
        return aRank - bRank;
      });
  }

  function getQuickMenuBadgeForCommand(cmd, officeMood, urgentItems) {
    const panelId = cmd.panelId || cmd.id;
    return getQuickMenuBadge(panelId, officeMood, urgentItems);
  }

  function validateOfficeNavigation() {
    if (process.env.NODE_ENV === "production") return;

    const panelIds = new Set(Object.keys(PANEL_CONTENT));

    OFFICE_INTERACTIVE_PANEL_IDS.forEach((panelId) => {
      if (!panelIds.has(panelId)) {
        console.warn(
          "[OfficeNav] Interactive object opens missing panel:",
          panelId
        );
      }
    });

    const suspiciousPairs = [
      { labelIncludes: "game preview", targetMustInclude: "game-preview" },
      { labelIncludes: "calendar", targetMustInclude: "calendar" },
      { labelIncludes: "schedule", targetMustInclude: "calendar" },
      { labelIncludes: "broadcast", targetMustInclude: "broadcast" },
      { labelIncludes: "standings", targetMustInclude: "standings" },
      { labelIncludes: "draft board", targetMustInclude: "draft-board" },
      { labelIncludes: "salary cap", targetMustInclude: "salary-cap" },
      { labelIncludes: "contracts", targetMustInclude: "contracts" },
      { labelIncludes: "lines", targetMustInclude: "lines" },
      { labelIncludes: "power play", targetMustInclude: "powerplay" },
      { labelIncludes: "penalty kill", targetMustInclude: "penaltykill" },
    ];

    Object.entries(PANEL_CONTENT).forEach(([panelId, panel]) => {
      if (!panel?.title || !Array.isArray(panel.actions)) {
        console.warn("[OfficeNav] Bad panel config:", panelId, panel);
        return;
      }

      const seenTargets = new Set();

      panel.actions.forEach(([label, target]) => {
        if (!label || !target) {
          console.warn("[OfficeNav] Empty action label/target:", panelId, label, target);
        }

        if (seenTargets.has(target)) {
          console.warn("[OfficeNav] Duplicate target in panel:", panelId, target);
        }

        seenTargets.add(target);

        const normalizedLabel = String(label).toLowerCase();
        const normalizedTarget = String(target).toLowerCase();

        suspiciousPairs.forEach((rule) => {
          if (
            normalizedLabel.includes(rule.labelIncludes) &&
            !normalizedTarget.includes(rule.targetMustInclude)
          ) {
            console.warn(
              `[OfficeNav] Suspicious action mapping in ${panelId}: "${label}" -> "${target}". Expected target to include "${rule.targetMustInclude}".`
            );
          }
        });
      });
    });
  }
  
  const QUICK_MENU = FRANCHISE_COMMAND_REGISTRY;
  
  function CameraRig({
    resetToken,
    activePanel,
    lowPowerMode = false,
    prefersReducedMotion = false,
    hoveredId = null,
    leagueOpsClickToken = 0,
  }) {
    const controlsRef = useRef(null);
    const { camera } = useThree();
    const focusRef = useRef({
      position: new THREE.Vector3(...OFFICE_CAMERA.position),
      target: new THREE.Vector3(...OFFICE_CAMERA.target),
      fov: OFFICE_CAMERA.fov,
    });
    const hoverBlendRef = useRef(0);
    const clickBlendRef = useRef(0);
    const clickStartRef = useRef(0);
    const leagueFocusRef = useRef(new THREE.Vector3(...LEAGUE_OPS_FOCUS));
    const [camX, camY, camZ] = OFFICE_CAMERA.position;
    const [tgtX, tgtY, tgtZ] = OFFICE_CAMERA.target;

    useEffect(() => {
      if (leagueOpsClickToken > 0) {
        clickStartRef.current = performance.now();
        clickBlendRef.current = 0;
      }
    }, [leagueOpsClickToken]);

    useEffect(() => {
      const snap = prefersReducedMotion || lowPowerMode;
      const panelTarget = activePanel ? PANEL_CAMERA_TARGETS[activePanel] : null;
      const nextPos = panelTarget?.position || OFFICE_CAMERA.position;
      const nextTarget = panelTarget?.target || OFFICE_CAMERA.target;

      focusRef.current.position.set(...nextPos);
      focusRef.current.target.set(...nextTarget);
      focusRef.current.fov = panelTarget?.fov || OFFICE_CAMERA.fov;

      if (snap) {
        camera.position.set(...nextPos);
        camera.fov = focusRef.current.fov;
        camera.updateProjectionMatrix();
        if (controlsRef.current) {
          controlsRef.current.target.set(...nextTarget);
          controlsRef.current.update();
        } else {
          camera.lookAt(...nextTarget);
        }
      }
    }, [
      resetToken,
      activePanel,
      camera,
      lowPowerMode,
      prefersReducedMotion,
      camX,
      camY,
      camZ,
      tgtX,
      tgtY,
      tgtZ,
    ]);

    useFrame(() => {
      if (!controlsRef.current) return;
      const snap = prefersReducedMotion || lowPowerMode;
      const lerpFactor = snap ? 1 : 0.085;

      const panelTarget = activePanel ? PANEL_CAMERA_TARGETS[activePanel] : null;
      const basePos = new THREE.Vector3(...(panelTarget?.position || OFFICE_CAMERA.position));
      const baseTarget = new THREE.Vector3(...(panelTarget?.target || OFFICE_CAMERA.target));
      const baseFov = panelTarget?.fov || OFFICE_CAMERA.fov;

      let destPos = basePos;
      let destTarget = baseTarget;

      if (!snap && !activePanel) {
        const hoverTarget = hoveredId === "leagueCentral" ? 1 : 0;
        hoverBlendRef.current +=
          (hoverTarget - hoverBlendRef.current) * (hoverTarget ? 0.07 : 0.11);

        if (leagueOpsClickToken > 0 && clickStartRef.current) {
          const elapsed = (performance.now() - clickStartRef.current) / 1000;
          const clickT = Math.min(elapsed / 0.82, 1);
          clickBlendRef.current = Math.sin(clickT * Math.PI * 0.5);
          if (elapsed > 0.92) clickBlendRef.current *= 0.86;
        } else {
          clickBlendRef.current *= 0.88;
        }

        const blend = hoverBlendRef.current * 0.032 + clickBlendRef.current * 0.06;
        if (blend > 0.0005) {
          destPos = basePos.clone().lerp(leagueFocusRef.current, blend);
          destTarget = baseTarget.clone().lerp(leagueFocusRef.current, blend * 0.92);
        }
      } else if (activePanel) {
        hoverBlendRef.current *= 0.85;
        clickBlendRef.current *= 0.85;
      }

      focusRef.current.position.lerp(destPos, lerpFactor);
      focusRef.current.target.lerp(destTarget, lerpFactor);
      focusRef.current.fov += (baseFov - focusRef.current.fov) * lerpFactor;

      camera.position.lerp(focusRef.current.position, lerpFactor);
      camera.fov += (focusRef.current.fov - camera.fov) * lerpFactor;
      camera.updateProjectionMatrix();
      controlsRef.current.target.lerp(focusRef.current.target, lerpFactor);
      controlsRef.current.update();
    });

    return (
        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom
          minDistance={OFFICE_CAMERA.minDistance}
          maxDistance={OFFICE_CAMERA.maxDistance}
          zoomSpeed={0.38}
          enableDamping
          dampingFactor={0.11}
          rotateSpeed={0.28}
          minPolarAngle={Math.PI / 2.78}
          maxPolarAngle={Math.PI / 2.06}
          minAzimuthAngle={-0.64}
          maxAzimuthAngle={0.64}
          target={OFFICE_CAMERA.target}
        />
    );
  }
  function StationPlaque({ text, hovered = false, width = 0.52, position = [0, 0.02, 0.38] }) {
    return (
      <group position={[0, 0.02, 0.38]} rotation={[-0.18, 0, 0]} raycast={() => null}>
        <RoundedBox args={[width, 0.07, 0.018]} radius={0.008} smoothness={4}>
          <meshStandardMaterial
            color={hovered ? "#1a1610" : "#12141a"}
            roughness={0.55}
            metalness={0.12}
            emissive={hovered ? OFFICE_PALETTE.goldDim : "#000000"}
            emissiveIntensity={hovered ? 0.18 : 0}
          />
        </RoundedBox>
        <mesh position={[0, 0.028, 0.012]} raycast={() => null}>
          <boxGeometry args={[width * 0.92, 0.006, 0.008]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.gold}
            roughness={0.42}
            metalness={0.62}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={hovered ? 0.22 : 0.08}
          />
        </mesh>
        <WallText position={[0, 0, 0.014]} size={0.028} color={hovered ? "#f0e4c8" : "#d8d0c4"}>
          {text}
        </WallText>
      </group>
    );
  }

  function InteractCorners({ args = [0.75, 0.5, 0.2], position = [0, 0, 0], hovered = false }) {
    if (!hovered) return null;
    const [w, h] = args;
    const hw = w * 0.48;
    const hh = h * 0.48;
    const marks = [
      [-hw, hh],
      [hw, hh],
      [-hw, -hh],
      [hw, -hh],
    ];
    return (
      <group position={position} raycast={() => null}>
        {marks.map(([x, y], i) => (
          <mesh key={`corner-${i}`} position={[x, y, 0.04]}>
            <boxGeometry args={[0.04, 0.04, 0.006]} />
            <meshStandardMaterial
              color={OFFICE_PALETTE.gold}
              emissive={OFFICE_PALETTE.goldDim}
              emissiveIntensity={0.45}
              roughness={0.4}
              metalness={0.55}
            />
          </mesh>
        ))}
      </group>
    );
  }

  function HoverLabel({
    visible,
    label,
    description,
    badge,
    compact = false,
    position = [0, 0.34, 0.08],
  }) {
    if (!visible) return null;

    return (
      <Html center distanceFactor={compact ? 9.2 : 7.2} position={position} zIndexRange={[40, 0]}>
        <div className={compact ? "office-proximity office-proximity--compact" : "office-proximity"}>
          <strong>{label}</strong>
          {!compact && description ? <span>{description}</span> : null}
          <em>{badge ? badge : "Enter"}</em>
        </div>
      </Html>
    );
  }
  function BlinkingNotificationLight({ active, position = [0, 0, 0], color = "#d94a41" }) {
    const ref = useRef();

    useFrame((state) => {
      if (!ref.current || !active) return;
      const pulse = 0.35 + Math.sin(state.clock.elapsedTime * 5.5) * 0.25;
      ref.current.material.emissiveIntensity = pulse;
    });

    if (!active) return null;

    return (
      <mesh ref={ref} position={position} raycast={() => null}>
        <sphereGeometry args={[0.028, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.45}
          roughness={0.35}
        />
      </mesh>
    );
  }

  function InteractiveGroup({
    id,
    label,
    description,
    badge,
    children,
    position = [0, 0, 0],
    rotation = [0, 0, 0],
    scale = 1,
    hoveredId,
    setHoveredId,
    onOpen,
    hoverScale = 1.008,
    hoverLift = 0,
    hitBoxArgs,
    hitBoxPosition = [0, 0.18, 0],
    openId = id,
    lowPowerMode = false,
    showPlaque = false,
    plaqueText = "",
    plaqueWidth = 0.52,
    plaquePosition = [0, 0.02, 0.38],
    labelCompact = false,
    labelPosition = [0, 0.34, 0.08],
    showHoverCorners = true,
    hideHoverLabel = false,
    activateOnPointerDown = false,
  }) {
    const groupRef = useRef();
    const scaleVec = useRef(new THREE.Vector3(1, 1, 1));
    const isHovered = hoveredId === id;
    const effectiveHoverScale = lowPowerMode ? 1.004 : hoverScale;
    const effectiveHoverLift = lowPowerMode ? 0.0006 : hoverLift;

    useFrame((state) => {
      if (!groupRef.current) return;

      const targetScale = isHovered ? effectiveHoverScale : 1;

      groupRef.current.scale.lerp(
        scaleVec.current.set(
          targetScale * scale,
          targetScale * scale,
          targetScale * scale
        ),
        lowPowerMode ? 0.22 : 0.14
      );
  
      if (isHovered) {
        groupRef.current.position.y =
          position[1] +
          Math.sin(state.clock.elapsedTime * 2.2) * effectiveHoverLift;
      } else {
        groupRef.current.position.y +=
          (position[1] - groupRef.current.position.y) * 0.12;
      }
    });
  
    return (
      <group
        ref={groupRef}
        position={position}
        rotation={rotation}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHoveredId(id);
          document.body.classList.add("office-cursor-active");
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHoveredId(null);
          document.body.classList.remove("office-cursor-active");
        }}
        onPointerDown={
          activateOnPointerDown
            ? (e) => {
                e.stopPropagation();
                onOpen(openId);
              }
            : undefined
        }
        onClick={(e) => {
          e.stopPropagation();
          if (!activateOnPointerDown) onOpen(openId);
        }}
      >
        <mesh
          position={hitBoxPosition}
          renderOrder={999}
          onPointerDown={
            activateOnPointerDown
              ? (e) => {
                  e.stopPropagation();
                  onOpen(openId);
                }
              : undefined
          }
          onClick={(e) => {
            e.stopPropagation();
            if (!activateOnPointerDown) onOpen(openId);
          }}
        >
          <boxGeometry
            args={hitBoxArgs || OFFICE_HITBOXES[id] || [0.75, 0.5, 0.75]}
          />
          <meshBasicMaterial
            transparent
            opacity={0}
            depthWrite={false}
            depthTest={false}
            color="#ffffff"
          />
        </mesh>
  
        {children(isHovered)}

        <InteractCorners
          args={hitBoxArgs || OFFICE_HITBOXES[id] || [0.75, 0.5, 0.75]}
          position={hitBoxPosition}
          hovered={showHoverCorners && isHovered}
        />

        {!hideHoverLabel ? (
          <HoverLabel
            visible={isHovered}
            label={label}
            description={description}
            badge={badge}
            compact={labelCompact}
            position={labelPosition}
          />
        ) : null}
      </group>
    );
  }
  
  function GlowMaterial({
    color = "#1b2536",
    emissive = "#000000",
    intensity = 0.2,
    roughness = 0.55,
    metalness = 0.1,
  }) {
    return (
      <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={intensity}
          roughness={roughness}
          metalness={metalness}
          envMapIntensity={0.35}
        />
    );
  }
  
  function WallText({
    children,
    position,
    rotation = [0, 0, 0],
    size = 0.12,
    color = "#f8ead5",
    anchorX = "center",
    anchorY = "middle",
    maxWidth = 2,
  }) {
    return (
      <Text
        font={officeFontBold}
        position={position}
        rotation={rotation}
        fontSize={size}
        color={color}
        anchorX={anchorX}
        anchorY={anchorY}
        maxWidth={maxWidth}
        textAlign="center"
      >
        {children}
      </Text>
    );
  }
  
  function ScreenGlassMaterial({ hovered }) {
    return (
      <meshPhysicalMaterial
        color={hovered ? "#081420" : OFFICE_PALETTE.monitor}
        emissive={hovered ? "#1e5a82" : OFFICE_PALETTE.monitorGlow}
        emissiveIntensity={hovered ? 0.48 : 0.26}
        roughness={0.08}
        metalness={0.12}
        envMapIntensity={0.55}
        clearcoat={0.82}
        clearcoatRoughness={0.14}
        transparent
        opacity={0.96}
      />
    );
  }

  function SmokedGlassMaterial({ opacity = 0.14, hovered = false }) {
    return (
      <meshPhysicalMaterial
        color={hovered ? "#1a2230" : "#0e1218"}
        emissive={hovered ? "#2a3a4a" : "#101820"}
        emissiveIntensity={hovered ? 0.12 : 0.05}
        roughness={0.08}
        metalness={0.22}
        transparent
        opacity={opacity}
        clearcoat={0.9}
        clearcoatRoughness={0.08}
      />
    );
  }

  function WoodMaterial({
    color = "#4a3424",
    roughness = 0.78,
    metalness = 0.02,
  }) {
    return (
      <meshPhysicalMaterial
        color={color}
        roughness={roughness}
        metalness={metalness}
        envMapIntensity={0.28}
        clearcoat={0.08}
        clearcoatRoughness={0.62}
      />
    );
  }

  function PaperMaterial({ color = "#e8dfcc", roughness = 0.96 }) {
    return (
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={0}
        envMapIntensity={0.08}
      />
    );
  }

  function LeatherMaterial({ color = "#231f1d", roughness = 0.88 }) {
    return (
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={0.02}
        envMapIntensity={0.12}
      />
    );
  }

  function MetalMaterial({
    color = "#8a7350",
    roughness = 0.42,
    metalness = 0.78,
  }) {
    return (
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={metalness}
        envMapIntensity={0.7}
      />
    );
  }

  function PlasticMaterial({ color = "#1a1f28", roughness = 0.62 }) {
    return (
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={0.04}
        envMapIntensity={0.22}
      />
    );
  }

  function GlassMaterial({ opacity = 0.22 }) {
    return (
      <meshPhysicalMaterial
        color="#c8dce8"
        roughness={0.08}
        metalness={0.04}
        transparent
        opacity={opacity}
        clearcoat={0.85}
        clearcoatRoughness={0.12}
      />
    );
  }

  function DustMotes({ count = 48, enabled = true }) {
    const pointsRef = useRef();
    const positions = useMemo(() => {
      const arr = new Float32Array(count * 3);
      for (let i = 0; i < count; i += 1) {
        arr[i * 3] = (Math.random() - 0.5) * 6.4;
        arr[i * 3 + 1] = 0.35 + Math.random() * 2.6;
        arr[i * 3 + 2] = (Math.random() - 0.5) * 5.2;
      }
      return arr;
    }, [count]);

    useFrame((state) => {
      if (!pointsRef.current || !enabled) return;
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.008;
      pointsRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.12) * 0.04;
    });

    if (!enabled) return null;

    return (
      <points ref={pointsRef} raycast={() => null}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.016}
          color="#c4bba8"
          transparent
          opacity={0.11}
          depthWrite={false}
          sizeAttenuation
        />
      </points>
    );
  }

  function PracticalLights({ lowPowerMode = false, prefersReducedMotion = false }) {
    const deskRef = useRef();
    const monitorRef = useRef();

    useFrame((state) => {
      if (prefersReducedMotion || lowPowerMode) return;
      const t = state.clock.elapsedTime;
      if (deskRef.current) {
        deskRef.current.intensity = 0.55 + Math.sin(t * 0.35) * 0.03;
      }
      if (monitorRef.current) {
        monitorRef.current.intensity = 0.22 + Math.sin(t * 0.9) * 0.02;
      }
    });

    return (
      <>
        <directionalLight
          position={[-5.2, 3.6, 0.4]}
          intensity={1.35}
          color="#c8dce8"
          castShadow={!lowPowerMode}
          shadow-mapSize-width={lowPowerMode ? 1024 : 1536}
          shadow-mapSize-height={lowPowerMode ? 1024 : 1536}
          shadow-bias={-0.00018}
          shadow-normalBias={0.04}
          shadow-camera-near={0.4}
          shadow-camera-far={16}
          shadow-camera-left={-6}
          shadow-camera-right={6}
          shadow-camera-top={6}
          shadow-camera-bottom={-6}
        />
        <spotLight
          position={[0, 3.5, -2.2]}
          angle={0.95}
          penumbra={0.92}
          intensity={0.42}
          color="#a8d0dc"
          distance={8}
        />
        <spotLight
          position={[-3.8, 2.8, -1.5]}
          angle={0.75}
          penumbra={0.88}
          intensity={0.28}
          color="#88b8c4"
          distance={6}
        />
        <spotLight
          position={[3.8, 2.8, -1.5]}
          angle={0.75}
          penumbra={0.88}
          intensity={0.28}
          color="#88b8c4"
          distance={6}
        />
        <spotLight
          ref={deskRef}
          position={[-1.2, 2.35, 0.85]}
          angle={0.62}
          penumbra={0.88}
          intensity={0.72}
          color="#e8c898"
          distance={6}
          castShadow={false}
        />
        <directionalLight
          position={[3.8, 2.6, 4.2]}
          intensity={0.42}
          color="#e4ddd0"
          castShadow={false}
        />
      </>
    );
  }

  function WoodGrainLines({ width = 4.35, depth = 1.38, y = 1.031, z = 1.03, count = 16 }) {
    return (
      <group>
        {Array.from({ length: count }).map((_, i) => (
          <mesh
            key={`grain-${i}`}
            position={[-width / 2 + (i / (count - 1)) * width, y, z]}
            raycast={() => null}
          >
            <boxGeometry args={[0.005, 0.002, depth]} />
            <meshStandardMaterial
              color="#352018"
              roughness={0.62}
              transparent
              opacity={0.28}
            />
          </mesh>
        ))}
      </group>
    );
  }

  function FloorPlanks({ width = 9, depth = 8, count = 22 }) {
    return (
      <group position={[0, 0.004, -1.1]} rotation={[-Math.PI / 2, 0, 0]}>
        {Array.from({ length: count }).map((_, i) => (
          <mesh
            key={`plank-${i}`}
            position={[(-width / 2 + (i / (count - 1)) * width), 0, 0]}
            raycast={() => null}
          >
            <planeGeometry args={[0.012, depth]} />
            <meshBasicMaterial color="#1a100a" transparent opacity={0.22} />
          </mesh>
        ))}
      </group>
    );
  }

  function OfficeRug() {
    return (
      <group position={[0, 0.012, 0.35]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <mesh receiveShadow>
          <planeGeometry args={[3.1, 2.0]} />
          <meshStandardMaterial color="#12100e" roughness={0.94} metalness={0.01} />
        </mesh>
        <mesh position={[0, 0.001, 0]}>
          <planeGeometry args={[2.85, 1.75]} />
          <meshStandardMaterial color="#181410" roughness={0.9} transparent opacity={0.42} />
        </mesh>
      </group>
    );
  }

  function ExecutiveChairSilhouette() {
    return (
      <group position={[0, 0.78, 2.95]} raycast={() => null}>
        <mesh position={[0, 0.62, 0]} castShadow>
          <boxGeometry args={[0.88, 1.18, 0.07]} />
          <LeatherMaterial color={OFFICE_PALETTE.leather} roughness={0.9} />
        </mesh>
        {[-0.48, 0.48].map((x) => (
          <mesh key={`arm-${x}`} position={[x, 0.14, 0.18]} rotation={[0.42, 0, 0]}>
            <boxGeometry args={[0.12, 0.05, 0.38]} />
            <LeatherMaterial color="#080807" roughness={0.88} />
          </mesh>
        ))}
        <mesh position={[0, 0.08, 0.32]}>
          <boxGeometry args={[0.72, 0.16, 0.42]} />
          <LeatherMaterial color="#0d0c0b" roughness={0.86} />
        </mesh>
      </group>
    );
  }

  function CeilingLightStrip() {
    return (
      <group position={[0, 3.98, -1.2]} raycast={() => null}>
        <mesh>
          <boxGeometry args={[5.2, 0.04, 0.18]} />
          <MetalMaterial color="#1a1c22" roughness={0.35} metalness={0.72} />
        </mesh>
        <mesh position={[0, -0.018, 0]}>
          <boxGeometry args={[4.8, 0.012, 0.1]} />
          <meshStandardMaterial
            color="#f0ddb0"
            emissive="#d4b060"
            emissiveIntensity={0.62}
            roughness={0.55}
          />
        </mesh>
      </group>
    );
  }

  function WallDisplayFrame({ width = 1.85, height = 1.08, children, accent = "#c9a86a" }) {
    return (
      <group>
        <RoundedBox
          position={[0, 0, -0.055]}
          args={[width + 0.28, height + 0.28, 0.1]}
          radius={0.04}
          smoothness={6}
          castShadow
          raycast={() => null}
        >
          <MetalMaterial color="#12151c" roughness={0.55} metalness={0.55} />
        </RoundedBox>
        <mesh position={[0, 0, -0.018]} raycast={() => null}>
          <boxGeometry args={[width + 0.08, height + 0.08, 0.04]} />
          <meshStandardMaterial color="#080a10" roughness={0.82} metalness={0.06} />
        </mesh>
        <mesh position={[0, 0, -0.002]} raycast={() => null}>
          <boxGeometry args={[width, height, 0.012]} />
          <meshStandardMaterial color="#0a1218" roughness={0.35} metalness={0.08} emissive="#102030" emissiveIntensity={0.18} />
        </mesh>
        <mesh position={[0, height / 2 + 0.08, 0.0]} raycast={() => null}>
          <boxGeometry args={[width + 0.12, 0.016, 0.02]} />
          <meshStandardMaterial
            color={accent}
            emissive={accent}
            emissiveIntensity={0.12}
            roughness={0.48}
            metalness={0.62}
          />
        </mesh>
        <group position={[0, 0, -0.01]}>
          {children}
        </group>
      </group>
    );
  }

  function Baseboards() {
    const trimMat = (
      <meshStandardMaterial
        color="#1a120e"
        roughness={0.55}
        metalness={0.12}
        envMapIntensity={0.2}
      />
    );
    const capMat = (
      <meshStandardMaterial
        color={OFFICE_PALETTE.goldDim}
        roughness={0.38}
        metalness={0.55}
        emissive={OFFICE_PALETTE.goldDim}
        emissiveIntensity={0.08}
      />
    );

    return (
      <group raycast={() => null}>
        {[
          [0, -3.48, 0, [8.6, 0.14, 0.06]],
          [-4.38, -0.15, Math.PI / 2, [6.7, 0.14, 0.06]],
          [4.38, -0.15, Math.PI / 2, [6.7, 0.14, 0.06]],
        ].map(([x, z, rot, args], i) => (
          <group key={`base-${i}`}>
            <mesh position={[x, 0.08, z]} rotation={[0, rot, 0]} receiveShadow>
              <boxGeometry args={args} />
              {trimMat}
            </mesh>
            <mesh position={[x, 0.16, z]} rotation={[0, rot, 0]}>
              <boxGeometry args={[args[0], 0.018, args[2] + 0.02]} />
              {capMat}
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  function WallPanelStrips() {
    const railMat = (
      <meshStandardMaterial
        color="#2a2018"
        roughness={0.48}
        metalness={0.18}
        envMapIntensity={0.35}
      />
    );
    const brassMat = (
      <meshStandardMaterial
        color={OFFICE_PALETTE.goldDim}
        roughness={0.35}
        metalness={0.62}
        emissive={OFFICE_PALETTE.goldDim}
        emissiveIntensity={0.12}
      />
    );

    /* Chair-rail height only — a rail at landmark height cut every scene in
       half and finished the "framed poster" illusion. */
    const rails = [
      { pos: [0, 0.68, -3.46], rot: 0, len: 8.2 },
      { pos: [-4.36, 0.68, -0.15], rot: Math.PI / 2, len: 6.6 },
      { pos: [4.36, 0.68, -0.15], rot: Math.PI / 2, len: 6.6 },
    ];

    const crowns = [
      { pos: [0, 3.72, -3.46], rot: 0, len: 8.2 },
      { pos: [-4.36, 3.72, -0.15], rot: Math.PI / 2, len: 6.6 },
      { pos: [4.36, 3.72, -0.15], rot: Math.PI / 2, len: 6.6 },
    ];

    return (
      <group raycast={() => null}>
        {rails.map(({ pos, rot, len }, i) => (
          <group key={`rail-${i}`} position={pos} rotation={[0, rot, 0]}>
            <mesh position={[0, 0, 0.04]}>
              <boxGeometry args={[len, 0.055, 0.045]} />
              {railMat}
            </mesh>
            <mesh position={[0, 0.028, 0.05]}>
              <boxGeometry args={[len, 0.012, 0.02]} />
              {brassMat}
            </mesh>
          </group>
        ))}
        {crowns.map(({ pos, rot, len }, i) => (
          <group key={`crown-${i}`} position={pos} rotation={[0, rot, 0]}>
            <mesh position={[0, 0, 0.04]}>
              <boxGeometry args={[len, 0.08, 0.05]} />
              {railMat}
            </mesh>
            <mesh position={[0, 0.035, 0.055]}>
              <boxGeometry args={[len, 0.014, 0.018]} />
              {brassMat}
            </mesh>
          </group>
        ))}
        {/* Continuous picture rail the upper landmarks hang from. One line of
            architecture for the whole wall, never a box around a scene. */}
        <mesh position={[0, 3.22, -3.42]}>
          <boxGeometry args={[8.2, 0.024, 0.024]} />
          <meshStandardMaterial
            color="#5d4c2f"
            roughness={0.42}
            metalness={0.5}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={0.04}
          />
        </mesh>
        {/* Wall sconces — pushed into the corners so no lamp sits inside a menu
            composition */}
        {[
          [-4.2, 2.9, -3.42],
          [4.2, 2.9, -3.42],
        ].map(([x, y, z], i) => (
          <group key={`sconce-${i}`} position={[x, y, z]}>
            <mesh position={[0, 0, 0.03]}>
              <boxGeometry args={[0.08, 0.22, 0.06]} />
              <MetalMaterial color="#2a2620" roughness={0.45} metalness={0.55} />
            </mesh>
            <mesh position={[0, -0.02, 0.06]}>
              <sphereGeometry args={[0.045, 12, 12]} />
              <meshStandardMaterial
                color="#f0ddb0"
                emissive="#d4b060"
                emissiveIntensity={0.42}
                roughness={0.4}
              />
            </mesh>
            <pointLight
              position={[0, -0.05, 0.15]}
              intensity={0.24}
              color="#e8c898"
              distance={2.2}
            />
          </group>
        ))}
      </group>
    );
  }

  function SmallRivets({ radius = 0.75, count = 4 }) {
    const positions = [
      [-radius, radius],
      [radius, radius],
      [-radius, -radius],
      [radius, -radius],
    ].slice(0, count);

    return positions.map(([x, y], i) => (
      <mesh key={`rivet-${i}`} position={[x, y, 0.04]} raycast={() => null}>
        <cylinderGeometry args={[0.012, 0.012, 0.018, 8]} />
        <MetalMaterial color="#6a5a42" roughness={0.35} metalness={0.78} />
      </mesh>
    ));
  }

  function WallFrame({ width = 2.28, height = 1.24, depth = 0.05, children }) {
    return (
      <group>
        <mesh position={[0, 0, -0.01]} castShadow raycast={() => null}>
          <boxGeometry args={[width, height, depth]} />
          <WoodMaterial color="#151210" roughness={0.55} />
        </mesh>
        <mesh position={[0, 0, 0.018]} castShadow raycast={() => null}>
          <boxGeometry args={[width - 0.14, height - 0.14, 0.022]} />
          <MetalMaterial color="#3a342c" roughness={0.42} metalness={0.55} />
        </mesh>
        {children}
      </group>
    );
  }

  function DeskLamp({ position = [-2.05, 0.89, 0.12] }) {
    return (
      <group position={position} raycast={() => null}>
        <RoundedBox args={[0.14, 0.04, 0.14]} radius={0.02} smoothness={4} position={[0, 0.02, 0]} castShadow>
          <MetalMaterial color="#2a241c" roughness={0.48} metalness={0.55} />
        </RoundedBox>
        <mesh position={[0, 0.11, 0]} castShadow>
          <cylinderGeometry args={[0.045, 0.055, 0.14, 16]} />
          <meshStandardMaterial color="#1c1814" roughness={0.72} metalness={0.08} />
        </mesh>
        <mesh position={[0, 0.18, 0]}>
          <cylinderGeometry args={[0.07, 0.05, 0.05, 20]} />
          <meshStandardMaterial
            color="#3a3228"
            emissive="#e8c898"
            emissiveIntensity={0.22}
            roughness={0.55}
          />
        </mesh>
        <pointLight position={[0, 0.16, 0.08]} intensity={0.42} color="#e8c898" distance={1.6} />
      </group>
    );
  }

  function DeskPen({ position = [1.05, 1.034, 0.92] }) {
    return (
      <group position={position} rotation={[0, -0.4, 0]} raycast={() => null}>
        <mesh rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.006, 0.006, 0.14, 8]} />
          <PlasticMaterial color="#1c2533" />
        </mesh>
        <mesh position={[0.07, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.009, 0.009, 0.025, 8]} />
          <MetalMaterial color="#c4a86a" roughness={0.25} metalness={0.8} />
        </mesh>
      </group>
    );
  }

  function DeskClutter() {
    return (
      <group raycast={() => null}>
        <group position={[0.55, 1.034, 0.92]} rotation={[0, 0.12, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.22, 0.004, 0.28]} />
            <PaperMaterial color="#d8d0c0" />
          </mesh>
          <mesh position={[0.03, 0.004, 0.04]} rotation={[0, -0.08, 0.02]} castShadow>
            <boxGeometry args={[0.18, 0.003, 0.22]} />
            <PaperMaterial color="#ece6d8" />
          </mesh>
        </group>
      </group>
    );
  }

  function DeskDrawerFaces() {
    const drawers = [
      [-1.42, 0.78, 1.62],
      [-1.42, 0.64, 1.62],
      [1.42, 0.78, 1.62],
      [1.42, 0.64, 1.62],
    ];

    return (
      <group raycast={() => null}>
        {drawers.map(([x, y, z], i) => (
          <group key={`drawer-${i}`} position={[x, y, z]}>
            <mesh castShadow>
              <boxGeometry args={[0.72, 0.22, 0.04]} />
              <WoodMaterial color="#3a2114" roughness={0.48} />
            </mesh>
            <mesh position={[0, 0, 0.028]}>
              <boxGeometry args={[0.58, 0.008, 0.012]} />
              <MetalMaterial color="#7a6848" roughness={0.3} metalness={0.75} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  function MarkerTray({ position = [0, -0.48, 0.07] }) {
    const markers = ["#d83e37", "#2c65b9", "#111111", "#2e8b45"];
    return (
      <group position={position} raycast={() => null}>
        <mesh>
          <boxGeometry args={[1.35, 0.035, 0.06]} />
          <MetalMaterial color="#4a4a4a" roughness={0.35} metalness={0.6} />
        </mesh>
        {markers.map((color, i) => (
          <mesh key={color} position={[-0.42 + i * 0.28, 0.03, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.012, 0.012, 0.09, 8]} />
            <PlasticMaterial color={color} roughness={0.45} />
          </mesh>
        ))}
        <mesh position={[0.52, 0.028, 0]}>
          <boxGeometry args={[0.08, 0.04, 0.04]} />
          <meshStandardMaterial color="#d8d0c4" roughness={0.75} />
        </mesh>
      </group>
    );
  }
  
  function useTeamLogoTexture(teamLogo) {
    const [texture, setTexture] = useState(null);
    const [loadFailed, setLoadFailed] = useState(false);
    const logoUrl = toLogoUrl(teamLogo);

    useEffect(() => {
      if (!logoUrl) {
        setTexture(null);
        setLoadFailed(false);
        return undefined;
      }

      let active = true;
      setLoadFailed(false);

      const loader = new THREE.TextureLoader();
      loader.load(
        logoUrl,
        (tex) => {
          if (!active) {
            tex.dispose();
            return;
          }
          tex.colorSpace = THREE.SRGBColorSpace;
          setTexture(tex);
        },
        undefined,
        () => {
          if (active) {
            setTexture(null);
            setLoadFailed(true);
          }
        }
      );

      return () => {
        active = false;
      };
    }, [logoUrl]);

    useEffect(() => {
      return () => {
        texture?.dispose();
      };
    }, [texture]);

    return { texture, loadFailed };
  }

  function TeamLogoPlane({
    teamLogo,
    teamName,
    hovered = false,
    width = 1.15,
    height = 1.15,
    opacity = 1,
    circularFallback = true,
  }) {
    const { texture, loadFailed } = useTeamLogoTexture(teamLogo);
    const initials = initialsFromTeam(teamName);
    const showTexture = texture && !loadFailed;

    if (showTexture) {
      return (
        <mesh raycast={() => null}>
          <planeGeometry args={[width, height]} />
          <meshBasicMaterial
            map={texture}
            transparent
            opacity={opacity}
            toneMapped={false}
            depthWrite={opacity >= 0.95}
          />
        </mesh>
      );
    }

    const fallbackRadius = Math.min(width, height) * 0.5;

    return (
      <group>
        {circularFallback ? (
          <mesh raycast={() => null}>
            <circleGeometry args={[fallbackRadius, 64]} />
            <meshStandardMaterial
              color={hovered ? "#f3d78a" : "#1c2533"}
              roughness={0.4}
              metalness={0.16}
              transparent={opacity < 1}
              opacity={opacity}
            />
          </mesh>
        ) : (
          <mesh raycast={() => null}>
            <planeGeometry args={[width, height]} />
            <meshStandardMaterial
              color={hovered ? "#2a3548" : "#121820"}
              roughness={0.45}
              metalness={0.12}
              transparent={opacity < 1}
              opacity={opacity}
            />
          </mesh>
        )}

        <WallText
          position={[0, 0, 0.02]}
          size={Math.min(width, height) * 0.19}
          color="#fff4d8"
          maxWidth={width * 0.9}
        >
          {initials}
        </WallText>
      </group>
    );
  }

  /** Flat logo decal for desk mat, laptop, binders */
  function TeamLogoDecal({
    teamLogo,
    teamName,
    position = [0, 0, 0],
    rotation = [0, 0, 0],
    width = 0.35,
    height = 0.35,
    opacity = 0.22,
    hovered = false,
  }) {
    return (
      <group position={position} rotation={rotation}>
        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={width}
          height={height}
          opacity={opacity}
          circularFallback={false}
        />
      </group>
    );
  }
  
  function LaptopObject({
    hovered,
    focused = false,
    teamName,
    teamLogo,
    currentDate,
    nextGame,
    record = "0-0-0",
    priorityCount = 0,
    seasonPhase = "",
  }) {
    const screenRef = useRef(null);
    const pulseRef = useRef(null);
    const stripRefs = useRef([]);
    const phaseLabel = safeText(String(seasonPhase || "").replace(/_/g, " "), "");
    const showDetail = focused || hovered;

    useFrame((state) => {
      const t = state.clock.elapsedTime;

      if (screenRef.current?.material) {
        screenRef.current.material.emissiveIntensity = hovered
          ? 0.44 + Math.sin(t * 1.8) * 0.04
          : 0.28 + Math.sin(t * 1.1) * 0.02;
      }

      if (pulseRef.current?.material) {
        pulseRef.current.material.opacity = hovered
          ? 0.14 + Math.sin(t * 2.2) * 0.04
          : 0.08 + Math.sin(t * 1.4) * 0.02;
      }

      // Telemetry strips idle almost flat and come alive under the cursor.
      stripRefs.current.forEach((strip, i) => {
        if (!strip) return;
        const wave = 0.5 + Math.sin(t * (1.6 + i * 0.45) + i) * 0.5;
        const target = hovered ? 0.35 + wave * 0.65 : 0.16 + wave * 0.1;
        strip.scale.x += (target - strip.scale.x) * 0.14;
        strip.position.x = -0.52 + (strip.scale.x * 0.42) / 2;
      });
    });

    return (
      <group scale={[0.62, 0.62, 0.62]} rotation={[0, 0.22, 0]} position={[0.16, 0.02, 0.06]}>
        <RoundedBox
          position={[0, 0.02, 0.02]}
          args={[1.92, 0.08, 0.88]}
          radius={0.04}
          smoothness={8}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <MetalMaterial color="#0a0c10" roughness={0.38} metalness={0.62} />
        </RoundedBox>

        <mesh position={[0, 0.058, 0.02]} raycast={() => null}>
          <boxGeometry args={[1.78, 0.006, 0.78]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.goldDim}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={hovered ? 0.14 : 0.06}
            roughness={0.35}
            metalness={0.55}
          />
        </mesh>

        <group position={[0, 0.48, -0.18]} rotation={[-0.22, 0, 0]}>
          <RoundedBox
            args={[1.68, 1.02, 0.07]}
            radius={0.04}
            smoothness={10}
            castShadow
            receiveShadow
            raycast={() => null}
          >
            <meshPhysicalMaterial
              color="#050810"
              emissive={hovered ? "#0e2840" : "#061420"}
              emissiveIntensity={hovered ? 0.22 : 0.1}
              roughness={0.32}
              metalness={0.45}
              clearcoat={0.45}
              clearcoatRoughness={0.28}
              envMapIntensity={0.4}
            />
          </RoundedBox>

          <RoundedBox
            ref={screenRef}
            position={[0, 0, 0.042]}
            args={[1.54, 0.88, 0.014]}
            radius={0.028}
            smoothness={8}
            raycast={() => null}
          >
            <ScreenGlassMaterial hovered={hovered} />
          </RoundedBox>

          <mesh ref={pulseRef} position={[0, 0, 0.036]} visible={false} raycast={() => null}>
            <planeGeometry args={[1.62, 0.94]} />
            <meshBasicMaterial color="#4a9ac8" transparent opacity={0} depthWrite={false} />
          </mesh>

          <WallText position={[0, 0.38, 0.056]} size={0.028} color="#c4a46a">
            FRANCHISE COMMAND
          </WallText>

          <TeamLogoDecal
            teamLogo={teamLogo}
            teamName={teamName}
            position={[-0.52, 0.22, 0.055]}
            width={0.18}
            height={0.18}
            opacity={0.95}
            hovered={hovered}
          />

          <WallText position={[-0.28, 0.26, 0.058]} size={0.048} color="#f0f4f8" anchorX="left">
            {teamName}
          </WallText>

          <WallText position={[-0.28, 0.16, 0.058]} size={0.022} color="#8aa0b0" anchorX="left">
            {safeText(currentDate, "Today")}
          </WallText>

          <WallText position={[0.42, 0.22, 0.058]} size={0.04} color="#e8f0f4">
            {safeText(record, "0-0-0")}
          </WallText>
          <WallText position={[0.42, 0.14, 0.058]} size={0.016} color="#6a8898">
            RECORD
          </WallText>

          <WallText position={[-0.52, 0.0, 0.058]} size={0.018} color="#6a8898" anchorX="left">
            NEXT GAME
          </WallText>
          <WallText position={[-0.52, -0.08, 0.058]} size={0.026} color="#d8e0e8" anchorX="left" maxWidth={1.2}>
            {safeText(nextGame, "No game listed")}
          </WallText>

          {phaseLabel ? (
            <WallText position={[0.42, 0.0, 0.058]} size={0.018} color="#c4a46a">
              {phaseLabel}
            </WallText>
          ) : null}

          {[0, 1, 2].map((i) => (
            <mesh
              key={`command-strip-${i}`}
              ref={(node) => {
                stripRefs.current[i] = node;
              }}
              position={[-0.52, -0.4 + i * 0.055, 0.056]}
              scale={[0.2, 1, 1]}
              raycast={() => null}
            >
              <planeGeometry args={[0.42, 0.012]} />
              <meshBasicMaterial
                color={i === 0 ? "#c4a46a" : "#4a9ac8"}
                transparent
                opacity={hovered ? 0.72 : 0.34}
                depthWrite={false}
              />
            </mesh>
          ))}

          {showDetail ? (
            <>
              <WallText position={[-0.52, -0.22, 0.058]} size={0.022} color="#c4a46a" anchorX="left">
                {Number(priorityCount || 0) > 0
                  ? `${priorityCount} URGENT GM ITEM${Number(priorityCount) === 1 ? "" : "S"}`
                  : "DESK CLEAR"}
              </WallText>
              <WallText position={[-0.52, -0.32, 0.058]} size={0.016} color="#6a8898" anchorX="left">
                OPEN COMMAND CENTER
              </WallText>
            </>
          ) : (
            <WallText position={[-0.52, -0.26, 0.058]} size={0.018} color="#5a7080" anchorX="left">
              YOUR TEAM · YOUR DAY
            </WallText>
          )}
        </group>

        {[
          ["ROSTER", -0.46],
          ["CAP", 0],
          ["OPS", 0.46],
        ].map(([label, x]) => (
          <group key={label} position={[x, 0.068, 0.28]} raycast={() => null}>
            <RoundedBox args={[0.4, 0.022, 0.24]} radius={0.018} smoothness={4}>
              <meshStandardMaterial color="#12151c" roughness={0.52} metalness={0.2} />
            </RoundedBox>
            <WallText
              position={[0, 0.016, 0]}
              rotation={[-Math.PI / 2, 0, 0]}
              size={0.022}
              color="#8a8070"
            >
              {label}
            </WallText>
          </group>
        ))}

        <pointLight
          position={[0, 0.35, 0.1]}
          intensity={hovered ? 0.32 : 0.16}
          color="#6a7a88"
          distance={1.4}
        />

        {/* Desk pool light — the command station warms up when addressed */}
        <pointLight
          position={[0, 0.22, 0.42]}
          intensity={hovered ? 0.42 : 0.1}
          color="#e8c898"
          distance={1.8}
        />
      </group>
    );
  }
  
  function PhoneObject({
    hovered,
    unreadMessages,
    hasTradeActivity = false,
    callerLabel = "LEAGUE GM",
  }) {
    const notify = hasTradeActivity || Number(unreadMessages || 0) > 0;

    return (
      <group rotation={[0, 0.12, 0]} scale={[0.82, 0.82, 0.82]}>
        {/* Negotiation blotter */}
        <RoundedBox args={[0.72, 0.06, 0.88]} radius={0.035} smoothness={5} castShadow>
          <WoodMaterial color={hovered ? "#3a2a1c" : "#2e2218"} roughness={0.62} />
        </RoundedBox>
        {/* Two opposing team sides */}
        <RoundedBox args={[0.28, 0.02, 0.34]} radius={0.015} smoothness={3} position={[-0.18, 0.045, -0.12]}>
          <PaperMaterial color="#d8cfc0" />
        </RoundedBox>
        <RoundedBox args={[0.28, 0.02, 0.34]} radius={0.015} smoothness={3} position={[0.18, 0.045, -0.12]}>
          <PaperMaterial color="#c8d0d8" />
        </RoundedBox>
        <WallText position={[-0.18, 0.06, -0.22]} rotation={[-Math.PI / 2, 0, 0]} size={0.022} color="#5a4030">
          US
        </WallText>
        <WallText position={[0.18, 0.06, -0.22]} rotation={[-Math.PI / 2, 0, 0]} size={0.022} color="#304050">
          THEM
        </WallText>
        {/* Trade arrow */}
        <mesh position={[0, 0.055, -0.12]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
          <boxGeometry args={[0.12, 0.02, 0.008]} />
          <meshStandardMaterial color="#c4a46a" emissive="#c4a46a" emissiveIntensity={hovered ? 0.35 : 0.12} />
        </mesh>
        {/* Desk phone */}
        <RoundedBox args={[0.18, 0.08, 0.22]} radius={0.02} smoothness={4} position={[0, 0.07, 0.28]}>
          <meshStandardMaterial
            color="#14181e"
            roughness={0.5}
            metalness={0.2}
            emissive={hovered ? OFFICE_PALETTE.goldDim : "#000000"}
            emissiveIntensity={hovered ? 0.12 : 0}
          />
        </RoundedBox>
        <mesh position={[0.06, 0.12, 0.32]} rotation={[0.4, 0, 0.2]} raycast={() => null}>
          <cylinderGeometry args={[0.025, 0.03, 0.16, 12]} />
          <meshStandardMaterial color="#1a1e24" roughness={0.45} metalness={0.25} />
        </mesh>
        <WallText
          position={[0, 0.06, 0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.024}
          color="#c8c0b4"
        >
          {safeText(callerLabel, "TRADE DESK")}
        </WallText>
        {Number(unreadMessages || 0) > 0 ? (
          <WallText position={[0, 0.06, 0.2]} rotation={[-Math.PI / 2, 0, 0]} size={0.018} color="#c4a46a">
            {`${unreadMessages} OFFERS`}
          </WallText>
        ) : null}
        <BlinkingNotificationLight active={notify} position={[0.28, 0.08, 0.35]} color={OFFICE_PALETTE.gold} />
        <pointLight position={[0, 0.25, 0.1]} intensity={hovered ? 0.22 : 0.08} color="#e8c898" distance={0.9} />
      </group>
    );
  }
  
  function ScoutingKitObject({ hovered, teamLogo, teamName, draftWeek = false }) {
    return (
      <group rotation={[0, 0.16, 0]}>
        <RoundedBox args={[0.92, 0.08, 0.66]} radius={0.035} smoothness={5}>
          <GlowMaterial
            color={hovered ? "#d9b15a" : "#8e6b37"}
            emissive={hovered ? "#b58222" : "#000000"}
            intensity={hovered ? 0.18 : 0}
            roughness={0.72}
          />
        </RoundedBox>
  
        <mesh position={[-0.22, 0.038, -0.28]}>
          <boxGeometry args={[0.45, 0.035, 0.16]} />
          <meshStandardMaterial color="#e5bd69" roughness={0.65} />
        </mesh>
  
        <mesh position={[0.08, 0.062, -0.02]}>
          <boxGeometry args={[0.72, 0.018, 0.42]} />
          <meshStandardMaterial color="#efe5ce" roughness={0.8} />
        </mesh>
  
        <WallText
          position={[0.02, 0.08, -0.02]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.06}
          color="#1c1710"
        >
          SCOUTING
        </WallText>

        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[0.28, 0.082, -0.22]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.12}
          height={0.12}
          opacity={0.72}
          hovered={hovered}
        />
  
        <group position={[-0.23, 0.11, 0.18]} rotation={[-Math.PI / 2, 0, 0]}>
          <mesh>
            <cylinderGeometry args={[0.07, 0.07, 0.12, 24]} />
            <meshStandardMaterial color="#111820" roughness={0.36} />
          </mesh>
  
          <mesh position={[0.14, 0, 0]}>
            <cylinderGeometry args={[0.07, 0.07, 0.12, 24]} />
            <meshStandardMaterial color="#111820" roughness={0.36} />
          </mesh>
  
          <mesh position={[0.07, 0, 0]}>
            <boxGeometry args={[0.08, 0.035, 0.035]} />
            <meshStandardMaterial color="#303945" roughness={0.45} />
          </mesh>
        </group>
  
        <WallText
          position={[0.17, 0.087, 0.18]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.028}
          color="#49351c"
        >
          INTL TRIP NOTES
        </WallText>

        {[
          ["TIER 1", -0.18, -0.02, "#c9a86a"],
          ["RISERS", 0.02, 0.02, "#7eb896"],
          ["WATCH", 0.2, -0.04, "#7eb8d4"],
        ].map(([label, x, z, color]) => (
          <group key={label} position={[x, 0.084, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <mesh raycast={() => null}>
              <boxGeometry args={[0.14, 0.09, 0.004]} />
              <meshStandardMaterial color="#efe5ce" roughness={0.82} />
            </mesh>
            <WallText position={[0, 0.004, 0.004]} size={0.018} color={color}>
              {label}
            </WallText>
          </group>
        ))}

        {draftWeek ? (
          <mesh position={[0, 0.095, 0]} raycast={() => null}>
            <planeGeometry args={[0.72, 0.42]} />
            <meshBasicMaterial color="#c9a86a" transparent opacity={0.08} depthWrite={false} />
          </mesh>
        ) : null}
      </group>
    );
  }
  
  function ContractLedgerObject({ hovered, capSpace, capPressure = false }) {
    return (
      <group rotation={[0, -0.08, 0]}>
        <RoundedBox args={[0.98, 0.075, 0.7]} radius={0.03} smoothness={6}>
          <GlowMaterial
            color={hovered ? "#8ed0ad" : "#355c50"}
            emissive={hovered ? "#5db88b" : "#000000"}
            intensity={hovered ? 0.18 : 0}
            roughness={0.68}
          />
        </RoundedBox>
  
        <mesh position={[0.04, 0.06, 0]}>
          <boxGeometry args={[0.72, 0.02, 0.5]} />
          <meshStandardMaterial color="#f2ead6" roughness={0.82} />
        </mesh>
  
        <WallText
          position={[0.04, 0.083, -0.18]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.048}
          color="#172820"
        >
          CAP LEDGER
        </WallText>
  
        <WallText
          position={[0.04, 0.086, -0.04]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.035}
          color="#375244"
        >
          ROOM: {formatMoney(capSpace)}
        </WallText>
  
        {[0, 1, 2].map((i) => (
          <mesh key={i} position={[0.04, 0.087, 0.08 + i * 0.09]}>
            <boxGeometry args={[0.48, 0.005, 0.012]} />
            <meshStandardMaterial color="#77a98c" roughness={0.7} />
          </mesh>
        ))}

        {[
          ["RFA", -0.18, 0.1],
          ["UFA", 0.02, 0.14],
          ["NMC", 0.2, 0.08],
          ["CAP", -0.05, 0.2],
        ].map(([label, x, z]) => (
          <WallText
            key={label}
            position={[0.04 + x, 0.092, z]}
            rotation={[-Math.PI / 2, 0, 0]}
            size={0.022}
            color="#2d4a3d"
          >
            {label}
          </WallText>
        ))}

        <mesh position={[0.22, 0.09, 0.24]} rotation={[-Math.PI / 2, 0, 0.12]} raycast={() => null}>
          <boxGeometry args={[0.18, 0.12, 0.004]} />
          <PaperMaterial color="#f7f1df" />
        </mesh>

        <mesh position={[0.22, 0.091, 0.3]} rotation={[-Math.PI / 2, 0, 0.12]} raycast={() => null}>
          <boxGeometry args={[0.1, 0.004, 0.004]} />
          <meshStandardMaterial color="#4a4034" roughness={0.7} />
        </mesh>

        {capPressure ? (
          <WallText
            position={[0.22, 0.095, 0.24]}
            rotation={[-Math.PI / 2, 0, 0.12]}
            size={0.028}
            color="#b72a20"
          >
            CAP WARN
          </WallText>
        ) : null}
  
        <group position={[-0.33, 0.105, 0.2]} rotation={[0, 0, 0.65]}>
          <mesh>
            <cylinderGeometry args={[0.018, 0.018, 0.42, 16]} />
            <meshStandardMaterial color="#111111" roughness={0.35} />
          </mesh>
  
          <mesh position={[0, 0.23, 0]}>
            <cylinderGeometry args={[0.015, 0.015, 0.06, 16]} />
            <meshStandardMaterial color="#d0a24a" metalness={0.4} roughness={0.25} />
          </mesh>
        </group>
      </group>
    );
  }
  
  function TabletObject({ hovered }) {
    return (
      <group rotation={[0, 0.24, 0]}>
        <RoundedBox args={[0.72, 0.055, 0.92]} radius={0.055} smoothness={7}>
          <GlowMaterial
            color="#090d12"
            emissive={hovered ? "#67c9ff" : "#1e6487"}
            intensity={hovered ? 0.55 : 0.25}
            roughness={0.42}
          />
        </RoundedBox>
  
        <mesh position={[0, 0.035, 0]}>
          <boxGeometry args={[0.58, 0.012, 0.72]} />
          <meshStandardMaterial
            color="#071725"
            emissive="#184d6a"
            emissiveIntensity={0.35}
          />
        </mesh>
  
        <WallText
          position={[0, 0.05, -0.25]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.055}
          color="#d9f8ff"
        >
          STATS
        </WallText>
  
        <WallText
          position={[0, 0.052, -0.03]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.035}
          color="#95e8ff"
        >
          CF% xGF% PDO
        </WallText>
  
        {[0, 1, 2, 3].map((i) => (
          <mesh
            key={i}
            position={[-0.22 + i * 0.15, 0.055, 0.24]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <planeGeometry args={[0.08, 0.08 + i * 0.03]} />
            <meshBasicMaterial
              color="#64d6ff"
              transparent
              opacity={hovered ? 0.32 : 0.18}
            />
          </mesh>
        ))}
      </group>
    );
  }
  
  /** Sep → Jun reading order, matching how a hockey season actually runs. */
  const SEASON_TIMELINE_MONTHS = [
    "SEP",
    "OCT",
    "NOV",
    "DEC",
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "JUN",
  ];

  function CalendarObject({ hovered, currentDate, nextGame, teamLogo, teamName }) {
    const { month } = parseFranchiseDateParts(currentDate);
    const timelineIndex = SEASON_TIMELINE_MONTHS.indexOf(
      new Date(2000, Math.max(month - 1, 0), 1)
        .toLocaleString("en-US", { month: "short" })
        .toUpperCase()
    );
    const nowIndex = timelineIndex >= 0 ? timelineIndex : 0;

    return (
      <group rotation={[0, -0.18, 0]} scale={[0.78, 0.78, 0.78]}>
        <RoundedBox args={[0.78, 0.05, 0.88]} radius={0.03} smoothness={6} castShadow>
          <PaperMaterial color={hovered ? "#d2c6ae" : "#c4b79c"} />
        </RoundedBox>
        <mesh position={[0, 0.032, -0.34]} raycast={() => null}>
          <boxGeometry args={[0.78, 0.022, 0.16]} />
          <meshStandardMaterial color="#6a2a28" roughness={0.62} metalness={0.06} />
        </mesh>
        <WallText
          position={[0, 0.048, -0.34]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.034}
          color="#e8dcc8"
        >
          SEASON
        </WallText>
        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[0.28, 0.045, -0.2]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.08}
          height={0.08}
          opacity={0.7}
          hovered={hovered}
        />
        <WallText
          position={[-0.08, 0.045, -0.18]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.028}
          color="#3a3228"
          anchorX="left"
        >
          {safeText(currentDate, "Today")}
        </WallText>
        {/* Season timeline — the desk plans time, not dates in a month grid */}
        <mesh position={[0, 0.042, 0.06]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
          <planeGeometry args={[0.62, 0.008]} />
          <meshBasicMaterial color="#6a5f4c" transparent opacity={0.75} />
        </mesh>

        {SEASON_TIMELINE_MONTHS.map((month, i) => {
          const x = -0.28 + i * 0.0933;
          const isNow = i === nowIndex;
          return (
            <group key={`season-tick-${month}`} position={[x, 0, 0]}>
              <mesh
                position={[0, 0.043, 0.06]}
                rotation={[-Math.PI / 2, 0, 0]}
                raycast={() => null}
              >
                <planeGeometry args={[0.009, 0.038]} />
                <meshBasicMaterial
                  color={isNow ? "#e8c07a" : "#7a6e58"}
                  transparent
                  opacity={isNow ? 0.95 : 0.55}
                />
              </mesh>
              <WallText
                position={[0, 0.044, 0.13]}
                rotation={[-Math.PI / 2, 0, 0]}
                size={0.019}
                color={isNow ? "#2a2118" : "#6a5f4c"}
              >
                {month}
              </WallText>
              {i % 2 === 0 ? (
                <mesh
                  position={[0.024, 0.043, 0.005]}
                  rotation={[-Math.PI / 2, 0, 0]}
                  raycast={() => null}
                >
                  <planeGeometry args={[0.03, 0.026]} />
                  <meshBasicMaterial
                    color="#3a5040"
                    transparent
                    opacity={hovered ? 0.8 : 0.45}
                  />
                </mesh>
              ) : null}
            </group>
          );
        })}

        {/* Today marker — a puck sitting on the rail */}
        <mesh
          position={[-0.28 + nowIndex * 0.0933, 0.052, 0.06]}
          raycast={() => null}
        >
          <cylinderGeometry args={[0.026, 0.026, 0.016, 20]} />
          <meshStandardMaterial
            color="#14100c"
            emissive="#e8c07a"
            emissiveIntensity={hovered ? 0.35 : 0.12}
            roughness={0.5}
          />
        </mesh>

        <mesh
          position={[-0.055, 0.041, 0.06]}
          rotation={[-Math.PI / 2, 0, 0]}
          raycast={() => null}
        >
          <planeGeometry args={[0.45, 0.02]} />
          <meshBasicMaterial
            color="#e8c07a"
            transparent
            opacity={hovered ? 0.22 : 0.08}
            depthWrite={false}
          />
        </mesh>
        <WallText
          position={[0, 0.045, 0.36]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.02}
          color="#2a241c"
          maxWidth={0.7}
        >
          {`NEXT · ${safeText(nextGame, "Open")}`}
        </WallText>
      </group>
    );
  }

  function NewspaperObject({ hovered, activeStorylines }) {
    return (
      <group rotation={[0, -0.28, 0]}>
        <mesh position={[0, 0.01, 0]} raycast={() => null}>
          <boxGeometry args={[0.62, 0.012, 0.42]} />
          <PaperMaterial color={hovered ? "#d4cbb8" : "#c4baa6"} />
        </mesh>
  
        <WallText
          position={[0, 0.028, -0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.038}
          color="#2a241c"
        >
          DOSSIER
        </WallText>
  
        <WallText
          position={[0, 0.03, 0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.028}
          color="#4a4034"
        >
          {Number(activeStorylines || 0)} storylines
        </WallText>
      </group>
    );
  }
  
  function ClipboardObject({ hovered, pendingTasks }) {
    return (
      <group rotation={[0, 0.5, 0]}>
        <RoundedBox args={[0.58, 0.055, 0.76]} radius={0.025} smoothness={5} raycast={() => null}>
          <GlowMaterial color={hovered ? "#fff6d7" : "#e7dcbd"} roughness={0.8} />
        </RoundedBox>
  
        <mesh position={[0, 0.055, -0.31]} raycast={() => null}>
          <boxGeometry args={[0.32, 0.04, 0.08]} />
          <meshStandardMaterial color="#20252d" roughness={0.5} metalness={0.16} />
        </mesh>
  
        <WallText
          position={[0, 0.076, -0.08]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.043}
          color="#1b1b1b"
        >
          DECISIONS
        </WallText>
  
        {[0, 1, 2].map((i) => (
          <group key={i} position={[-0.16, 0.077, 0.07 + i * 0.12]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
              <circleGeometry args={[0.018, 18]} />
              <meshStandardMaterial color={i === 0 ? "#d9473e" : "#7a8a77"} />
            </mesh>
  
            <mesh position={[0.16, 0, 0]} raycast={() => null}>
              <boxGeometry args={[0.24, 0.004, 0.014]} />
              <meshStandardMaterial color="#6a5b45" roughness={0.8} />
            </mesh>
          </group>
        ))}
  
        <WallText
          position={[0, 0.079, 0.3]}
          rotation={[-Math.PI / 2, 0, 0]}
          size={0.033}
          color="#3a3327"
        >
          {Number(pendingTasks || 0)} pending
        </WallText>
      </group>
    );
  }
  
  function CoffeeAndPuck() {
    return (
      <group position={[1.62, 0.91, 0.12]} raycast={() => null}>
        <mesh castShadow receiveShadow>
          <cylinderGeometry args={[0.13, 0.13, 0.045, 32]} />
          <meshStandardMaterial color="#060606" roughness={0.38} metalness={0.12} />
        </mesh>
        <mesh position={[0, 0.028, 0]}>
          <torusGeometry args={[0.118, 0.005, 8, 32]} />
          <meshStandardMaterial color="#1a1a1a" roughness={0.32} metalness={0.18} />
        </mesh>
      </group>
    );
  }

  function Desk({ children, teamName, teamLogo }) {
    return (
      <group>
        {[[-1.62, 0.28, 1.12], [1.62, 0.28, 1.12], [-1.62, 0.28, 0.38], [1.62, 0.28, 0.38]].map(
          ([x, y, z], i) => (
            <RoundedBox
              key={`leg-${i}`}
              position={[x, y, z]}
              args={[0.16, 0.56, 0.16]}
              radius={0.03}
              smoothness={5}
              castShadow
              receiveShadow
              raycast={() => null}
            >
              <MetalMaterial color="#12151c" roughness={0.48} metalness={0.72} />
            </RoundedBox>
          )
        )}

        <RoundedBox
          position={[0, 0.52, 0.78]}
          args={[3.85, 0.58, 1.38]}
          radius={0.07}
          smoothness={7}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <meshStandardMaterial color="#161920" roughness={0.78} metalness={0.08} />
        </RoundedBox>

        <RoundedBox
          position={[-1.92, 0.48, 0.72]}
          args={[0.62, 0.5, 1.18]}
          radius={0.06}
          smoothness={6}
          castShadow
          raycast={() => null}
        >
          <meshStandardMaterial color="#14181f" roughness={0.8} metalness={0.06} />
        </RoundedBox>

        <RoundedBox
          position={[0, 0.84, 0.74]}
          args={[4.12, 0.08, 1.58]}
          radius={0.055}
          smoothness={8}
          castShadow
          receiveShadow
          raycast={() => null}
        >
          <WoodMaterial color="#4a3222" roughness={0.58} metalness={0.04} />
        </RoundedBox>

        <mesh position={[0, 0.76, 1.46]} castShadow raycast={() => null}>
          <boxGeometry args={[3.35, 0.16, 0.22]} />
          <meshStandardMaterial color="#101318" roughness={0.72} metalness={0.1} />
        </mesh>
        <mesh position={[0, 0.685, 1.52]} raycast={() => null}>
          <boxGeometry args={[3.2, 0.012, 0.04]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.goldDim}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={0.22}
            roughness={0.4}
            metalness={0.7}
          />
        </mesh>

        <RoundedBox
          position={[0.28, 0.89, 0.62]}
          args={[1.55, 0.02, 0.72]}
          radius={0.02}
          smoothness={4}
          receiveShadow
          raycast={() => null}
        >
          <meshStandardMaterial color="#0e1218" roughness={0.7} metalness={0.12} />
        </RoundedBox>

        <TeamLogoDecal
          teamLogo={teamLogo}
          teamName={teamName}
          position={[-1.35, 0.892, 0.35]}
          rotation={[-Math.PI / 2, 0, 0]}
          width={0.18}
          height={0.18}
          opacity={0.22}
        />

        <DeskLamp position={[-2.08, 0.89, 0.08]} />
        {children}
      </group>
    );
  }
  
  function getBestPlayer(players = []) {
    const rating = (player) =>
      Number(
        player?.overall ||
          player?.ovr ||
          player?.rating ||
          player?.trueOverall ||
          player?.true_ovr ||
          player?.calculatedOverall ||
          0
      );

    return [...players].sort((a, b) => rating(b) - rating(a))[0];
  }

  function getPlayerName(player) {
    return (
      player?.name ||
      player?.full_name ||
      `${player?.first_name || player?.firstName || ""} ${player?.last_name || player?.lastName || ""}`.trim() ||
      "Franchise Player"
    );
  }

  function WallPlayerPortrait({ player }) {
    const resolvedPlayer = useMemo(() => ensurePlayerHeadshotFields(player || {}), [player]);

    if (!player) {
      return (
        <mesh position={[0, 0, 0.045]} raycast={() => null}>
          <boxGeometry args={[2.02, 1.02, 0.02]} />
          <meshStandardMaterial
            color="#142638"
            emissive="#102943"
            emissiveIntensity={0.1}
            roughness={0.55}
          />
        </mesh>
      );
    }

    return (
      <>
        <mesh position={[0, 0, 0.045]} raycast={() => null}>
          <boxGeometry args={[2.02, 1.02, 0.02]} />
          <meshStandardMaterial color="#142638" roughness={0.55} />
        </mesh>
        <Html transform position={[0, 0.02, 0.06]} scale={0.42} center style={{ pointerEvents: "none" }}>
          <div className="office-wall-portrait">
            <PlayerHeadshot player={resolvedPlayer} size="xl" variant="card" />
          </div>
        </Html>
      </>
    );
  }

  function OfficeFurniture({ teamLogo, teamName, mood = {}, championshipCount = 0 }) {
    return (
      <group raycast={() => null}>
        {/* Filing cabinet directly beneath the Contracts station, so the deal
            sheet reads as paperwork belonging to a real piece of furniture */}
        <RoundedBox position={[-3.38, 0.21, -3.02]} args={[0.74, 0.42, 0.42]} radius={0.03} smoothness={4} castShadow>
          <meshStandardMaterial color="#1b2028" roughness={0.62} metalness={0.16} />
        </RoundedBox>
        {[0.12, 0.3].map((y) => (
          <mesh key={`cab-pull-${y}`} position={[-3.38, y, -2.79]} raycast={() => null}>
            <boxGeometry args={[0.2, 0.016, 0.02]} />
            <MetalMaterial color="#6a5a42" roughness={0.4} metalness={0.7} />
          </mesh>
        ))}
        {/* Credenza carrying the trophy plinth */}
        <RoundedBox position={[-2.08, 0.21, -3.04]} args={[1.05, 0.42, 0.44]} radius={0.04} smoothness={4} castShadow>
          <WoodMaterial color="#33231a" roughness={0.6} />
        </RoundedBox>
        <mesh position={[-2.08, 0.43, -3.04]} raycast={() => null}>
          <boxGeometry args={[1.1, 0.02, 0.48]} />
          <WoodMaterial color="#4a3222" roughness={0.5} />
        </mesh>
        {/* Warm shrine light for the trophy */}
        <pointLight position={[-2.08, 1.0, -2.75]} intensity={0.4} color="#e8c07a" distance={2.4} />
        {/* Draft table tucked near the war-room entrance */}
        <RoundedBox position={[-3.55, 0.52, 0.45]} args={[0.95, 0.07, 0.55]} radius={0.03} smoothness={3} castShadow>
          <WoodMaterial color="#2a1c14" />
        </RoundedBox>
        <pointLight position={[-4.1, 1.9, 0.45]} intensity={mood.isDraftWeek ? 0.45 : 0.16} color="#c4a46a" distance={2.8} />
        <pointLight position={[-4.05, 2.0, -1.55]} intensity={0.14} color="#e8d8b8" distance={2.2} />
        {mood.isTradeDeadline ? (
          <pointLight position={[-1.5, 1.2, 0.7]} intensity={0.32} color="#c47848" distance={2.0} />
        ) : null}
        {championshipCount > 0 ? (
          <group position={[-2.62, 0.55, -3.04]}>
            <mesh>
              <cylinderGeometry args={[0.05, 0.07, 0.18, 12]} />
              <meshStandardMaterial color="#c4a46a" metalness={0.55} roughness={0.35} />
            </mesh>
          </group>
        ) : null}
      </group>
    );
  }

  function CityWeatherWindow({ currentDate, weather }) {
    const planeRef = useRef();
    const precipRefs = useRef([]);
    const wx = weather || deriveSeasonalWeather(currentDate);

    useFrame((state) => {
      const t = state.clock.elapsedTime;
      if (planeRef.current) {
        planeRef.current.position.x = -0.55 + ((t * 0.12) % 1.4);
        planeRef.current.position.y = 0.42 + Math.sin(t * 0.7) * 0.04;
      }
      precipRefs.current.forEach((mesh, i) => {
        if (!mesh) return;
        mesh.position.y -= wx.precip === "snow" ? 0.004 : 0.012;
        if (mesh.position.y < -0.55) {
          mesh.position.y = 0.55;
          mesh.position.x = -0.7 + (i % 8) * 0.18 + (Math.random() * 0.05);
        }
      });
    });

    const buildings = [
      [-0.62, 0.05, 0.22, 0.55],
      [-0.38, -0.05, 0.18, 0.42],
      [-0.12, 0.12, 0.28, 0.68],
      [0.18, 0.0, 0.2, 0.5],
      [0.42, 0.08, 0.24, 0.6],
      [0.65, -0.02, 0.16, 0.38],
    ];

    return (
      <group position={[4.28, 2.05, 0.35]} rotation={[0, -Math.PI / 2, 0]}>
        <RoundedBox args={[2.05, 1.55, 0.08]} radius={0.03} smoothness={3} position={[0, 0, -0.06]}>
          <meshStandardMaterial color="#0c1018" roughness={0.7} metalness={0.15} />
        </RoundedBox>
        {/* Sky */}
        <mesh position={[0, 0.1, -0.02]} raycast={() => null}>
          <planeGeometry args={[1.85, 1.28]} />
          <meshBasicMaterial color={wx.sky} />
        </mesh>
        {/* Haze / sun wash */}
        <mesh position={[0.35, 0.35, -0.01]} raycast={() => null}>
          <circleGeometry args={[0.22, 24]} />
          <meshBasicMaterial color={wx.light} transparent opacity={wx.condition === "clear" ? 0.55 : 0.18} />
        </mesh>
        {/* City silhouette */}
        {buildings.map(([x, y, w, h], i) => (
          <mesh key={`bldg-${i}`} position={[x, -0.45 + h / 2, 0]} raycast={() => null}>
            <boxGeometry args={[w, h, 0.04]} />
            <meshStandardMaterial color="#0a1220" roughness={0.9} />
          </mesh>
        ))}
        {/* Distant windows */}
        {buildings.map(([x, y, w, h], i) =>
          [0, 1, 2].map((row) => (
            <mesh
              key={`win-${i}-${row}`}
              position={[x, -0.35 + row * 0.14, 0.03]}
              raycast={() => null}
            >
              <boxGeometry args={[w * 0.35, 0.035, 0.01]} />
              <meshBasicMaterial color="#c4a46a" transparent opacity={0.35} />
            </mesh>
          ))
        )}
        {/* Aircraft */}
        <group ref={planeRef} position={[-0.4, 0.42, 0.02]}>
          <mesh raycast={() => null}>
            <boxGeometry args={[0.12, 0.018, 0.02]} />
            <meshBasicMaterial color="#e8eef4" />
          </mesh>
          <mesh position={[0, 0, 0]} raycast={() => null}>
            <boxGeometry args={[0.04, 0.004, 0.08]} />
            <meshBasicMaterial color="#d0d8e0" />
          </mesh>
        </group>
        {/* Precipitation */}
        {wx.precip !== "none"
          ? Array.from({ length: 14 }).map((_, i) => (
              <mesh
                key={`precip-${i}`}
                ref={(el) => {
                  precipRefs.current[i] = el;
                }}
                position={[-0.65 + (i % 7) * 0.2, 0.4 - (i % 5) * 0.12, 0.04]}
                raycast={() => null}
              >
                <boxGeometry
                  args={
                    wx.precip === "snow"
                      ? [0.03, 0.03, 0.01]
                      : [0.012, 0.08, 0.01]
                  }
                />
                <meshBasicMaterial
                  color={wx.precip === "snow" ? "#f0f4f8" : "#9ab0c4"}
                  transparent
                  opacity={0.55}
                />
              </mesh>
            ))
          : null}
        {/* Glass */}
        <mesh position={[0, 0.05, 0.06]} raycast={() => null}>
          <planeGeometry args={[1.9, 1.35]} />
          <meshPhysicalMaterial
            color="#8aa0b8"
            transparent
            opacity={0.12}
            roughness={0.15}
            metalness={0.05}
            transmission={0.4}
          />
        </mesh>
        <WallText position={[0, -0.72, 0.08]} size={0.028} color="#c4a46a">
          {safeText(wx.label, "Outside")}
        </WallText>
        <pointLight position={[-0.2, 0.2, 0.35]} intensity={0.35} color={wx.light} distance={2.2} />
      </group>
    );
  }

  function RoomShell() {
    return (
      <group>
        <mesh position={[0, 0, -1.1]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[9, 8]} />
          <meshStandardMaterial color="#3a342c" roughness={0.88} metalness={0.02} envMapIntensity={0.16} />
        </mesh>

        <mesh position={[0, 0.008, -1.1]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[8.6, 7.6]} />
          <meshStandardMaterial color="#4a4238" roughness={0.72} />
        </mesh>

        <FloorPlanks />
        <OfficeRug />
        <Baseboards />

        <ExecutiveWallSurface
          position={[0, 2.1, -3.55]}
          size={[8.8, 4.2, 0.12]}
          repeat={[3.6, 1.65]}
        />

        <WallPanelStrips />

        <ExecutiveWallSurface
          position={[-4.43, 2.1, -0.15]}
          rotation={[0, Math.PI / 2, 0]}
          size={[6.9, 4.2, 0.12]}
          repeat={[2.8, 1.65]}
        />

        <ExecutiveWallSurface
          position={[4.43, 2.1, -0.15]}
          rotation={[0, Math.PI / 2, 0]}
          size={[6.9, 4.2, 0.12]}
          repeat={[2.8, 1.65]}
        />

        <mesh position={[0, 4.15, -0.2]} rotation={[Math.PI / 2, 0, 0]} receiveShadow raycast={() => null}>
          <planeGeometry args={[8.8, 7]} />
          <meshStandardMaterial color="#0e1820" roughness={0.92} metalness={0.02} />
        </mesh>

        <mesh position={[0, 4.08, -0.15]} raycast={() => null}>
          <boxGeometry args={[8.4, 0.05, 0.08]} />
          <MetalMaterial color="#1a1c22" roughness={0.38} metalness={0.72} />
        </mesh>

        <CeilingLightStrip />

        {/* Crest recess — a turned walnut disc behind the franchise crest, so the
            centre of the wall reads as an installation rather than a poster */}
        <mesh
          position={[0, 2.938, -3.46]}
          rotation={[Math.PI / 2, 0, 0]}
          raycast={() => null}
        >
          <cylinderGeometry args={[0.36, 0.36, 0.07, 56]} />
          <WoodMaterial color="#241a12" roughness={0.6} />
        </mesh>
        <mesh position={[0, 2.938, -3.43]} raycast={() => null}>
          <circleGeometry args={[0.35, 56]} />
          <meshStandardMaterial color="#0a0f14" roughness={0.9} metalness={0.04} />
        </mesh>

        {/* Left wall architecture for Draft Class + War Room */}
        <mesh position={[-4.28, 1.35, 1.55]} rotation={[0, Math.PI / 2, 0]} raycast={() => null}>
          <boxGeometry args={[1.15, 2.35, 0.06]} />
          <meshStandardMaterial color={OFFICE_PALETTE.wallDeep} roughness={0.88} metalness={0.03} />
        </mesh>
        <RoundedBox position={[-4.34, 1.85, -1.65]} rotation={[0, Math.PI / 2, 0]} args={[2.15, 1.85, 0.1]} radius={0.04} smoothness={4} raycast={() => null}>
          <WoodMaterial color="#1c1410" roughness={0.7} />
        </RoundedBox>
        <mesh position={[-4.2, 2.85, 0.45]} rotation={[0, Math.PI / 2, 0]} raycast={() => null}>
          <boxGeometry args={[2.2, 0.08, 0.12]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.goldDim}
            emissive={OFFICE_PALETTE.goldDim}
            emissiveIntensity={0.2}
            metalness={0.5}
            roughness={0.4}
          />
        </mesh>

        {/* Window opening cut on right wall */}
        <mesh position={[4.28, 2.05, 0.35]} rotation={[0, -Math.PI / 2, 0]} raycast={() => null}>
          <boxGeometry args={[2.15, 1.65, 0.05]} />
          <meshStandardMaterial color={OFFICE_PALETTE.wallDeep} roughness={0.85} />
        </mesh>

        {[-4.38, 4.38].map((x) => (
          <group key={`side-trim-${x}`} position={[x, 2.05, -0.15]}>
            <mesh position={[0, 0, 0.08]} raycast={() => null}>
              <boxGeometry args={[0.04, 3.6, 0.02]} />
              <meshStandardMaterial
                color={OFFICE_PALETTE.goldDim}
                emissive={OFFICE_PALETTE.goldDim}
                emissiveIntensity={0.14}
                roughness={0.45}
                metalness={0.42}
              />
            </mesh>
          </group>
        ))}

        <mesh position={[0, 2.85, -3.42]} raycast={() => null}>
          <planeGeometry args={[1.4, 0.22]} />
          <meshBasicMaterial color="#c9a86a" transparent opacity={0.03} depthWrite={false} />
        </mesh>
      </group>
    );
  }
  
  function WallLogo({ hovered, teamLogo, teamName, scale = 1.18 }) {
    return (
      <group scale={[scale, scale, scale]}>
        <mesh position={[0, 0, -0.04]} castShadow raycast={() => null}>
          <boxGeometry args={[1.95, 1.95, 0.06]} />
          <WoodMaterial color="#1a1612" roughness={0.58} />
        </mesh>

        <mesh position={[0, 0, -0.018]} raycast={() => null}>
          <boxGeometry args={[1.72, 1.72, 0.04]} />
          <MetalMaterial color="#2a2620" roughness={0.45} metalness={0.42} />
        </mesh>

        <mesh position={[0, 0, -0.015]} raycast={() => null}>
          <circleGeometry args={[0.86, 64]} />
          <meshStandardMaterial
            color={hovered ? "#3a3228" : "#1c2028"}
            metalness={0.22}
            roughness={0.48}
          />
        </mesh>

        <mesh position={[0, 0, -0.02]} raycast={() => null}>
          <planeGeometry args={[2.05, 2.05]} />
          <meshBasicMaterial
            color={hovered ? "#ffd8a0" : "#c9a86a"}
            transparent
            opacity={hovered ? 0.14 : 0.08}
            depthWrite={false}
          />
        </mesh>
  
        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={1.28}
          height={1.28}
        />

        <SmallRivets radius={0.78} />
  
        <WallText position={[0, -0.87, 0.045]} size={0.068} color="#c9a86a">
          FRONT OFFICE
        </WallText>

        <mesh position={[0, 0.95, 0.02]} raycast={() => null}>
          <planeGeometry args={[1.4, 0.2]} />
          <meshBasicMaterial color="#ffd8a0" transparent opacity={0.06} depthWrite={false} />
        </mesh>
      </group>
    );
  }

  /**
   * Restrained franchise crest — smoked-glass wall emblem. It is the physical
   * centre of the Franchise Identity landmark, so it takes its placement from
   * that landmark's standardized footprint.
   */
  function WallHeroLogo({
    teamLogo,
    teamName,
    hovered = false,
    position = [0, 2.52, -3.32],
    scale = 0.82,
  }) {
    return (
      <group position={position} scale={[scale, scale, scale]} raycast={() => null}>
        {/* Turned medallion, not a framed picture — the crest is set into the
            wall inside a brass bezel so nothing rectangular sits behind it. */}
        <mesh position={[0, 0, -0.05]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.72, 0.72, 0.05, 56]} />
          <MetalMaterial color="#11141a" roughness={0.5} metalness={0.58} />
        </mesh>

        <mesh position={[0, 0, -0.026]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.63, 0.63, 0.03, 56]} />
          <SmokedGlassMaterial opacity={0.2} hovered={hovered} />
        </mesh>

        <mesh position={[0, 0, -0.014]}>
          <ringGeometry args={[0.6, 0.66, 56]} />
          <meshStandardMaterial
            color={OFFICE_PALETTE.gold}
            emissive={OFFICE_PALETTE.gold}
            emissiveIntensity={hovered ? 0.34 : 0.16}
            roughness={0.38}
            metalness={0.72}
          />
        </mesh>

        <TeamLogoPlane
          teamLogo={teamLogo}
          teamName={teamName}
          hovered={hovered}
          width={0.78}
          height={0.78}
          opacity={hovered ? 0.92 : 0.74}
        />

        <pointLight
          position={[0, 0, 0.28]}
          intensity={hovered ? 0.38 : 0.18}
          color="#c9a86a"
          distance={1.8}
        />
      </group>
    );
  }
  
  function HockeySticks() {
    return (
      <group position={[-3.75, 0.75, -2.88]} rotation={[0, 0, -0.15]}>
        {[0, 1, 2].map((i) => (
          <group
            key={i}
            position={[i * 0.08, 0, i * 0.035]}
            rotation={[0, 0, i * 0.13]}
          >
            <mesh position={[0, 0.63, 0]} rotation={[0, 0, 0.08]}>
              <boxGeometry args={[0.035, 1.45, 0.035]} />
              <meshStandardMaterial color="#5f3a21" roughness={0.58} />
            </mesh>

            <mesh position={[0.1, -0.1, 0]} rotation={[0, 0, 0.55]}>
              <boxGeometry args={[0.34, 0.045, 0.045]} />
              <meshStandardMaterial color="#1b1b1b" roughness={0.5} />
            </mesh>
          </group>
        ))}
      </group>
    );
  }

  function ScoutingStation({ hovered }) {
    return (
      <group>
        <RoundedBox args={[1.72, 1.22, 0.1]} radius={0.04} smoothness={4} position={[0, 0, -0.05]} castShadow>
          <meshStandardMaterial color="#1a1814" roughness={0.72} />
        </RoundedBox>
        {/* Regional map */}
        <RoundedBox args={[0.85, 0.72, 0.03]} radius={0.02} smoothness={3} position={[-0.35, 0.12, 0.02]}>
          <meshStandardMaterial color={hovered ? "#2a4a3a" : "#1e3830"} roughness={0.65} emissive="#143028" emissiveIntensity={0.12} />
        </RoundedBox>
        {[[-0.55, 0.28], [-0.28, 0.05], [-0.42, -0.12], [-0.18, 0.22]].map(([x, y], i) => (
          <mesh key={`pin-${i}`} position={[x, y, 0.05]} raycast={() => null}>
            <sphereGeometry args={[0.028, 10, 10]} />
            <meshStandardMaterial color="#c4a46a" emissive="#c4a46a" emissiveIntensity={0.3} />
          </mesh>
        ))}
        {/* Prospect cards */}
        {[0, 1, 2].map((i) => (
          <group key={`card-${i}`} position={[0.42, 0.28 - i * 0.28, 0.04]} rotation={[0, -0.08, 0.04 * (i - 1)]}>
            <RoundedBox args={[0.42, 0.24, 0.02]} radius={0.015} smoothness={3}>
              <PaperMaterial color={hovered ? "#d8d0c0" : "#c8bfae"} />
            </RoundedBox>
            <mesh position={[-0.12, 0.02, 0.015]} raycast={() => null}>
              <boxGeometry args={[0.12, 0.14, 0.008]} />
              <meshStandardMaterial color="#2a3038" roughness={0.7} />
            </mesh>
            <WallText position={[0.08, 0.04, 0.016]} size={0.028} color="#2a241c">
              {`#${i + 1}`}
            </WallText>
          </group>
        ))}
        <RoundedBox args={[0.55, 0.12, 0.08]} radius={0.02} smoothness={3} position={[0.35, -0.42, 0.06]}>
          <meshStandardMaterial color="#3a2a18" roughness={0.6} />
        </RoundedBox>
        <WallText position={[0, 0.52, 0.04]} size={0.048} color="#c4a46a">
          SCOUTING
        </WallText>
        <pointLight position={[0.2, 0.3, 0.5]} intensity={hovered ? 0.28 : 0.14} color="#e8d8b8" distance={1.4} />
      </group>
    );
  }

  function DraftWarRoomEntrance({ hovered, draftWeek = false }) {
    return (
      <group>
        {/* Doorway frame */}
        <RoundedBox args={[2.35, 2.55, 0.18]} radius={0.03} smoothness={3} position={[0, 0.1, -0.35]} castShadow>
          <meshStandardMaterial color="#12151a" roughness={0.75} />
        </RoundedBox>
        <mesh position={[0, 0.15, -0.22]} raycast={() => null}>
          <boxGeometry args={[1.55, 2.05, 0.08]} />
          <meshStandardMaterial color="#06080c" roughness={0.9} />
        </mesh>
        {/* Light spill from room */}
        <mesh position={[0, 0.2, -0.18]} raycast={() => null}>
          <planeGeometry args={[1.4, 1.85]} />
          <meshBasicMaterial color="#c4a46a" transparent opacity={hovered || draftWeek ? 0.14 : 0.07} depthWrite={false} />
        </mesh>
        {/* Draft board silhouette inside */}
        <group position={[0, 0.35, -0.28]}>
          <mesh raycast={() => null}>
            <boxGeometry args={[1.05, 0.85, 0.04]} />
            <meshStandardMaterial color="#1a1612" roughness={0.6} />
          </mesh>
          {[0, 1, 2, 3, 4].map((r) =>
            [0, 1, 2].map((c) => (
              <mesh key={`cell-${r}-${c}`} position={[-0.32 + c * 0.32, 0.28 - r * 0.14, 0.03]} raycast={() => null}>
                <boxGeometry args={[0.26, 0.1, 0.01]} />
                <meshBasicMaterial color={r === 0 ? "#c4a46a" : "#2a3038"} />
              </mesh>
            ))
          )}
        </group>
        {/* Pick clock */}
        <group position={[0.72, 0.85, -0.12]}>
          <mesh raycast={() => null}>
            <cylinderGeometry args={[0.14, 0.14, 0.05, 24]} />
            <meshStandardMaterial color="#1a1814" roughness={0.45} metalness={0.3} />
          </mesh>
          <WallText position={[0, 0, 0.04]} size={0.045} color="#c4a46a" rotation={[Math.PI / 2, 0, 0]}>
            :45
          </WallText>
        </group>
        {/* Overhead sign */}
        <RoundedBox args={[1.85, 0.28, 0.1]} radius={0.02} smoothness={3} position={[0, 1.28, 0.02]}>
          <meshStandardMaterial
            color="#0c0e12"
            emissive={hovered || draftWeek ? "#3a3020" : "#1a1810"}
            emissiveIntensity={hovered || draftWeek ? 0.45 : 0.2}
            roughness={0.4}
            metalness={0.25}
          />
        </RoundedBox>
        <WallText position={[0, 1.28, 0.08]} size={0.065} color="#f0e4c8">
          DRAFT WAR ROOM
        </WallText>
        <pointLight position={[0, 0.6, 0.2]} intensity={hovered || draftWeek ? 0.55 : 0.28} color="#e8c898" distance={2.4} />
      </group>
    );
  }

  function LeagueOpsSilhouette({ hovered = false, prefersReducedMotion = false }) {
    const hoverT = useRef(0);
    const iconRef = useRef(null);
    const vignetteTex = useMemo(() => getLeagueOpsVignetteTexture(), []);

    useFrame((_, delta) => {
      const step = prefersReducedMotion ? 1 : Math.min(delta / 0.32, 1);
      const target = hovered ? 1 : 0;
      hoverT.current += (target - hoverT.current) * step * 4.2;
      const ease = hoverT.current * hoverT.current * (3 - 2 * hoverT.current);

      if (iconRef.current) {
        iconRef.current.style.setProperty("--league-ops-hover", String(ease));
      }
    });

    const seam = {
      fill: "none",
      stroke: "#9eb4c4",
      strokeWidth: 1.15,
      strokeLinecap: "round",
      opacity: 0.14,
    };

    return (
      <group>
        {vignetteTex ? (
          <mesh position={[0, 0.04, -0.04]} raycast={() => null}>
            <planeGeometry args={MENU_LANDMARK.vignette} />
            <meshBasicMaterial
              map={vignetteTex}
              transparent
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        ) : null}

        <Html
          center
          transform
          distanceFactor={1.92}
          position={[0, 0.02, 0.018]}
          zIndexRange={[35, 0]}
          wrapperClass="league-ops-html-layer"
          style={{ pointerEvents: "none" }}
        >
          <div
            ref={iconRef}
            className={`league-ops-icon${hovered ? " league-ops-icon--hover" : ""}`}
            style={{ "--league-ops-hover": 0 }}
          >
            <span className="league-ops-icon__corner league-ops-icon__corner--tl" />
            <span className="league-ops-icon__corner league-ops-icon__corner--tr" />
            <span className="league-ops-icon__corner league-ops-icon__corner--bl" />
            <span className="league-ops-icon__corner league-ops-icon__corner--br" />

            <svg
              viewBox="0 0 500 640"
              className="league-ops-icon__scene"
              aria-hidden="true"
              focusable="false"
            >
              <defs>
                <linearGradient id="loPanelL" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#05080c" stopOpacity="0" />
                  <stop offset="35%" stopColor="#0a1218" stopOpacity="0.82" />
                  <stop offset="100%" stopColor="#060a10" stopOpacity="0.55" />
                </linearGradient>
                <linearGradient id="loPanelR" x1="100%" y1="0%" x2="0%" y2="0%">
                  <stop offset="0%" stopColor="#05080c" stopOpacity="0" />
                  <stop offset="35%" stopColor="#0a1218" stopOpacity="0.82" />
                  <stop offset="100%" stopColor="#060a10" stopOpacity="0.55" />
                </linearGradient>
                <linearGradient id="loDoor" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#d8edf8" stopOpacity="0.16" />
                  <stop offset="18%" stopColor="#e8f4fc" stopOpacity="0.68" />
                  <stop offset="44%" stopColor="#9fc4d8" stopOpacity="0.34" />
                  <stop offset="78%" stopColor="#6a8aa0" stopOpacity="0.08" />
                  <stop offset="100%" stopColor="#6a8aa0" stopOpacity="0" />
                </linearGradient>
                <linearGradient id="loDoorWide" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#7ca2b8" stopOpacity="0" />
                  <stop offset="28%" stopColor="#9fcce2" stopOpacity="0.12" />
                  <stop offset="50%" stopColor="#e7f5fc" stopOpacity="0.58" />
                  <stop offset="72%" stopColor="#9fcce2" stopOpacity="0.12" />
                  <stop offset="100%" stopColor="#7ca2b8" stopOpacity="0" />
                </linearGradient>
                <radialGradient id="loDoorBloom" cx="50%" cy="28%" r="62%">
                  <stop offset="0%" stopColor="#f4fbff" stopOpacity="0.95" />
                  <stop offset="38%" stopColor="#9ec8de" stopOpacity="0.38" />
                  <stop offset="100%" stopColor="#9ec8de" stopOpacity="0" />
                </radialGradient>
                {/* userSpaceOnUse: these strokes run down a zero-width path, and
                    an objectBoundingBox gradient on a degenerate box does not
                    paint at all. */}
                <linearGradient
                  id="loTrim"
                  gradientUnits="userSpaceOnUse"
                  x1="0"
                  y1="50"
                  x2="0"
                  y2="530"
                >
                  <stop offset="0%" stopColor="#6a5a3a" stopOpacity="0.15" />
                  <stop offset="45%" stopColor="#c4a46a" stopOpacity="0.42" />
                  <stop offset="100%" stopColor="#6a5a3a" stopOpacity="0.08" />
                </linearGradient>
                <linearGradient id="loJacket" x1="50%" y1="0%" x2="50%" y2="100%">
                  <stop offset="0%" stopColor="#2a3944" />
                  <stop offset="42%" stopColor="#17232d" />
                  <stop offset="100%" stopColor="#0a1118" />
                </linearGradient>
                <linearGradient id="loSleeveL" x1="100%" y1="10%" x2="0%" y2="80%">
                  <stop offset="0%" stopColor="#22303a" />
                  <stop offset="55%" stopColor="#111b24" />
                  <stop offset="100%" stopColor="#05090d" />
                </linearGradient>
                <linearGradient id="loSleeveR" x1="0%" y1="10%" x2="100%" y2="80%">
                  <stop offset="0%" stopColor="#1e2b35" />
                  <stop offset="55%" stopColor="#101922" />
                  <stop offset="100%" stopColor="#05090d" />
                </linearGradient>
                <linearGradient id="loShoulder" x1="50%" y1="0%" x2="50%" y2="100%">
                  <stop offset="0%" stopColor="#354652" />
                  <stop offset="100%" stopColor="#16222c" />
                </linearGradient>
                <linearGradient id="loHair" x1="40%" y1="0%" x2="70%" y2="100%">
                  <stop offset="0%" stopColor="#161a1e" />
                  <stop offset="100%" stopColor="#08090b" />
                </linearGradient>
                <pattern id="loSuitWeave" width="5" height="5" patternUnits="userSpaceOnUse">
                  <path d="M 0 1 L 5 1" stroke="#8fa5b4" strokeWidth="0.35" opacity="0.16" />
                  <path d="M 1 0 L 1 5" stroke="#020407" strokeWidth="0.35" opacity="0.3" />
                </pattern>
                <filter id="loRimBlur" x="-18%" y="-8%" width="136%" height="116%">
                  <feGaussianBlur stdDeviation="2.7" />
                </filter>
                <filter id="loGlowBlur" x="-40%" y="-20%" width="180%" height="140%">
                  <feGaussianBlur stdDeviation="15" />
                </filter>
              </defs>

              <rect x="28" y="36" width="132" height="510" fill="url(#loPanelL)" />
              <rect x="340" y="36" width="132" height="510" fill="url(#loPanelR)" />
              <path d="M 154 50 L 154 530" stroke="url(#loTrim)" strokeWidth="2" opacity="0.48" />
              <path d="M 346 50 L 346 530" stroke="url(#loTrim)" strokeWidth="2" opacity="0.48" />
              <path d="M 170 76 L 170 520" stroke="#7b93a2" strokeWidth="1" opacity="0.12" />
              <path d="M 330 76 L 330 520" stroke="#7b93a2" strokeWidth="1" opacity="0.12" />

              <path
                d="M 146 34 C 180 104 194 196 184 320 C 178 414 194 500 220 570
                   L 280 570 C 306 500 322 414 316 320 C 306 196 320 104 354 34 Z"
                fill="url(#loDoorWide)"
                filter="url(#loGlowBlur)"
                opacity="0.72"
              />
              <ellipse
                cx="250"
                cy="198"
                rx="112"
                ry="236"
                fill="url(#loDoorBloom)"
                filter="url(#loGlowBlur)"
                opacity="0.48"
              />
              <path
                d="M 206 38 C 216 144 214 280 204 536 L 296 536
                   C 286 280 284 144 294 38 Z"
                fill="url(#loDoor)"
                opacity="0.72"
              />

              <path d={LO_SHIELD} fill="none" stroke="#8a9aaa" strokeWidth="1.4" opacity="0.16" />
              <path d={LO_TROPHY} fill="#8a9aaa" opacity="0.12" />

              <g className="league-ops-icon__figure-layer league-ops-icon__rim">
                <path
                  d={LO_RIM_HEAD}
                  fill="none"
                  stroke="#d9f1fc"
                  strokeWidth="6"
                  filter="url(#loRimBlur)"
                  opacity="0.78"
                />
                <path
                  d={LO_RIM_SHOULDERS}
                  fill="none"
                  stroke="#c7e6f5"
                  strokeWidth="6"
                  filter="url(#loRimBlur)"
                  opacity="0.7"
                />
                <path
                  d={LO_RIM_ARM_L}
                  fill="none"
                  stroke="#98c5da"
                  strokeWidth="4"
                  filter="url(#loRimBlur)"
                  opacity="0.34"
                />
                <path
                  d={LO_RIM_ARM_R}
                  fill="none"
                  stroke="#b6d7e7"
                  strokeWidth="3"
                  filter="url(#loRimBlur)"
                  opacity="0.22"
                />
              </g>

              <g className="league-ops-icon__figure-layer league-ops-icon__man">
                <path d={LO_JACKET} fill="url(#loJacket)" />
                <path d={LO_TORSO_PANEL} fill="#15212a" opacity="0.38" />
                <path d={LO_SHOULDER_PLANE} fill="url(#loShoulder)" opacity="0.7" />
                <path d={LO_JACKET} fill="url(#loSuitWeave)" opacity="0.1" />
                <path d={LO_SLEEVE_L} fill="url(#loSleeveL)" />
                <path d={LO_SLEEVE_R} fill="url(#loSleeveR)" />
                <path d={LO_HAND_L} fill="#030507" />
                <path d={LO_HAND_R} fill="#030507" />
                <path d={LO_NECK} fill="#1a1412" />
                <path d={LO_COLLAR_SHIRT} fill="#cbd8df" opacity="0.78" />
                <path d={LO_COLLAR_SUIT} fill="#0a1016" />
                <path d={LO_HEAD} fill="url(#loHair)" />
                <path d={LO_HAIR_CROWN} fill="#050608" opacity="0.55" />
                <path d={LO_HAIR_SIDES} fill="#0a0c0e" opacity="0.4" />
                <path d={LO_NAPE} fill="#121418" opacity="0.45" />
                <path d={LO_SEAM_CENTER} {...seam} />
                <path d={LO_SEAM_SHOULDER_L} {...seam} />
                <path d={LO_SEAM_SHOULDER_R} {...seam} />
                <path d={LO_SEAM_SIDE_L} {...seam} />
                <path d={LO_SEAM_SIDE_R} {...seam} />
                <path d={LO_SEAM_SLEEVE_L} {...seam} opacity="0.1" />
                <path d={LO_SEAM_SLEEVE_R} {...seam} opacity="0.1" />
                <path d={LO_HEM} fill="none" stroke="#121820" strokeWidth="1.6" opacity="0.35" />
              </g>
            </svg>

            <div className="league-ops-icon__title">
              <span className="league-ops-icon__kicker">Executive</span>
              <span className="league-ops-icon__name">League Operations</span>
            </div>
          </div>
        </Html>
      </group>
    );
  }

  /* ---------------------------------------------------------------
     Shared menu-diorama engine.

     Every wall destination is authored entirely in code (inline SVG +
     CSS) and framed by the same vignette / label / hover grammar that
     made League Operations read as a landmark instead of a UI tile.
     The hover value is eased on the render loop and published as the
     `--menu-hover` custom property so each scene can drive its own
     animation from one number.
     --------------------------------------------------------------- */
  /** Fit any scene aspect inside the standardized square art box. */
  function fitLandmarkArt(viewBox) {
    const parts = String(viewBox || "")
      .trim()
      .split(/\s+/)
      .map(Number);
    const vw = parts[2] > 0 ? parts[2] : 1;
    const vh = parts[3] > 0 ? parts[3] : 1;
    const box = MENU_LANDMARK.artPx;
    return vw >= vh
      ? { width: box, height: Math.round((box * vh) / vw) }
      : { width: Math.round((box * vw) / vh), height: box };
  }

  /**
   * Physical mounts. These are the only backing geometry a landmark gets — a plinth,
   * a rail or a shallow recess — so the artwork itself defines the silhouette
   * instead of sitting on a coloured backing plate.
   */
  function LandmarkMount({ kind, accent = "#c4a46a" }) {
    if (kind === "plinth") {
      return (
        <group position={[0, -0.7, 0.14]} raycast={() => null}>
          <mesh>
            <boxGeometry args={[0.66, 0.07, 0.24]} />
            <meshStandardMaterial color="#15181c" roughness={0.6} metalness={0.24} />
          </mesh>
          <mesh position={[0, 0.042, 0.006]}>
            <boxGeometry args={[0.62, 0.008, 0.21]} />
            <meshStandardMaterial
              color={accent}
              emissive={accent}
              emissiveIntensity={0.18}
              roughness={0.4}
              metalness={0.55}
            />
          </mesh>
        </group>
      );
    }

    /* Suspension wires up to the wall's picture rail. Reads as a hung panel of
       glass rather than a sticker on the plaster. */
    if (kind === "wire") {
      return (
        <group raycast={() => null}>
          {[-0.26, 0.26].map((x) => (
            <mesh key={`wire-${x}`} position={[x, 0.69, 0.03]}>
              <boxGeometry args={[0.007, 0.18, 0.007]} />
              <meshStandardMaterial
                color={accent}
                emissive={accent}
                emissiveIntensity={0.14}
                roughness={0.38}
                metalness={0.68}
              />
            </mesh>
          ))}
        </group>
      );
    }

    /* Two slim brackets at the sides, the way a glass sign is stood off a
       wall. Deliberately not a frame — nothing crosses behind the artwork. */
    if (kind === "standoff") {
      return (
        <group raycast={() => null}>
          {[-0.42, 0.42].map((x) => (
            <group key={`standoff-${x}`} position={[x, 0, 0.035]}>
              <mesh>
                <cylinderGeometry args={[0.014, 0.014, 0.5, 10]} />
                <meshStandardMaterial
                  color="#2a2620"
                  emissive={accent}
                  emissiveIntensity={0.1}
                  roughness={0.4}
                  metalness={0.6}
                />
              </mesh>
            </group>
          ))}
        </group>
      );
    }

    if (kind === "ledge") {
      return (
        <group position={[0, -0.72, 0.18]} raycast={() => null}>
          <mesh>
            <boxGeometry args={[0.94, 0.06, 0.32]} />
            <WoodMaterial color="#2a1d14" roughness={0.6} />
          </mesh>
          <mesh position={[0, 0.036, 0.0]}>
            <boxGeometry args={[0.9, 0.008, 0.3]} />
            <meshStandardMaterial
              color={accent}
              emissive={accent}
              emissiveIntensity={0.12}
              roughness={0.42}
              metalness={0.5}
            />
          </mesh>
          {[-0.34, 0.34].map((x) => (
            <mesh key={`corbel-${x}`} position={[x, -0.06, -0.04]}>
              <boxGeometry args={[0.05, 0.09, 0.16]} />
              <WoodMaterial color="#1c130d" roughness={0.66} />
            </mesh>
          ))}
        </group>
      );
    }

    return null;
  }

  function MenuDiorama({
    hovered = false,
    prefersReducedMotion = false,
    kicker = "",
    name,
    viewBox = "0 0 440 440",
    accent = "#9fd6ea",
    mount = null,
    mountAccent,
    children,
  }) {
    const rootRef = useRef(null);
    const hoverT = useRef(0);
    const vignetteTex = useMemo(() => getLeagueOpsVignetteTexture(), []);
    const art = useMemo(() => fitLandmarkArt(viewBox), [viewBox]);

    useFrame((_, delta) => {
      const step = prefersReducedMotion ? 1 : Math.min(delta / 0.32, 1);
      hoverT.current += ((hovered ? 1 : 0) - hoverT.current) * step * 4.2;
      const ease =
        hoverT.current * hoverT.current * (3 - 2 * hoverT.current);
      if (rootRef.current) {
        rootRef.current.style.setProperty("--menu-hover", ease.toFixed(3));
      }
    });

    return (
      <group>
        {vignetteTex ? (
          <mesh position={[0, 0.08, -0.035]} raycast={() => null}>
            <planeGeometry args={MENU_LANDMARK.vignette} />
            <meshBasicMaterial
              map={vignetteTex}
              transparent
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        ) : null}

        <LandmarkMount kind={mount} accent={mountAccent || accent} />

        <Html
          center
          transform
          distanceFactor={MENU_LANDMARK.distanceFactor}
          position={[0, 0, 0.02]}
          zIndexRange={[30, 0]}
          wrapperClass="office-menu-html"
          style={{ pointerEvents: "none" }}
        >
          <div
            ref={rootRef}
            className={`office-menu-icon${
              hovered ? " office-menu-icon--hover" : ""
            }`}
            style={{
              "--menu-accent": accent,
              width: `${MENU_LANDMARK.artPx + MENU_LANDMARK.padPx * 2}px`,
            }}
          >
            <span className="office-menu-icon__corner office-menu-icon__corner--tl" />
            <span className="office-menu-icon__corner office-menu-icon__corner--tr" />
            <span className="office-menu-icon__corner office-menu-icon__corner--bl" />
            <span className="office-menu-icon__corner office-menu-icon__corner--br" />

            <div
              className="office-menu-icon__frame"
              style={{ height: `${MENU_LANDMARK.artPx}px` }}
            >
              <svg
                viewBox={viewBox}
                className="office-menu-icon__scene"
                width={art.width}
                height={art.height}
                aria-hidden="true"
                focusable="false"
              >
                {children}
              </svg>
            </div>

            <div className="office-menu-icon__title">
              {kicker ? (
                <span className="office-menu-icon__kicker">{kicker}</span>
              ) : null}
              <span className="office-menu-icon__name">{name}</span>
            </div>
          </div>
        </Html>
      </group>
    );
  }

  /**
   * Title-only landmark plate for the desk destinations, so the physical props
   * carry exactly the same label system as the wall scenes.
   */
  function LandmarkLabel({
    hovered = false,
    prefersReducedMotion = false,
    kicker = "",
    name,
    accent = "#c4a46a",
    position = [0, 0, 0],
    /* Desk props sit roughly half as far from the camera as the wall, so their
       titles need half the world scale to read at the same size on screen. */
    distanceFactor = MENU_LANDMARK.distanceFactor * 0.5,
  }) {
    const rootRef = useRef(null);
    const hoverT = useRef(0);

    useFrame((_, delta) => {
      const step = prefersReducedMotion ? 1 : Math.min(delta / 0.32, 1);
      hoverT.current += ((hovered ? 1 : 0) - hoverT.current) * step * 4.2;
      const ease =
        hoverT.current * hoverT.current * (3 - 2 * hoverT.current);
      if (rootRef.current) {
        rootRef.current.style.setProperty("--menu-hover", ease.toFixed(3));
      }
    });

    return (
      <Html
        center
        transform
        distanceFactor={distanceFactor}
        position={position}
        zIndexRange={[30, 0]}
        wrapperClass="office-menu-html"
        style={{ pointerEvents: "none" }}
      >
        <div
          ref={rootRef}
          className={`office-menu-icon office-menu-icon--label-only${
            hovered ? " office-menu-icon--hover" : ""
          }`}
          style={{ "--menu-accent": accent }}
        >
          <div className="office-menu-icon__title">
            {kicker ? (
              <span className="office-menu-icon__kicker">{kicker}</span>
            ) : null}
            <span className="office-menu-icon__name">{name}</span>
          </div>
        </div>
      </Html>
    );
  }

  /** Strategy Board — wide tactical rink with a coach leaning in. */
  function StrategyDiorama({ hovered, prefersReducedMotion }) {
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Tactics"
        name="Strategy Board"
        accent="#8fd8f0"
        mount="standoff"
        mountAccent="#5f8ea6"
      >
        <defs>
          <radialGradient id="stIce" cx="50%" cy="44%" r="62%">
            <stop offset="0%" stopColor="#71c4e2" stopOpacity="0.34" />
            <stop offset="55%" stopColor="#2b6c88" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#0a1a24" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Rink: light held by the ice, edge described by a thin board line */}
        <g className="menu-glow">
          <rect x="24" y="96" width="412" height="212" rx="106" fill="url(#stIce)" />
        </g>
        <rect
          x="24"
          y="96"
          width="412"
          height="212"
          rx="106"
          fill="none"
          stroke="#8fd8f0"
          strokeWidth="2.4"
          strokeOpacity="0.38"
        />

        <path d="M 230 100 L 230 304" stroke="#c0554d" strokeWidth="3" opacity="0.42" />
        <path d="M 148 106 L 148 298" stroke="#5f95c9" strokeWidth="2.4" opacity="0.3" />
        <path d="M 312 106 L 312 298" stroke="#5f95c9" strokeWidth="2.4" opacity="0.3" />
        <circle cx="230" cy="202" r="40" fill="none" stroke="#8fc4dc" strokeWidth="2" opacity="0.3" />
        <circle cx="86" cy="202" r="20" fill="none" stroke="#c0554d" strokeWidth="1.8" opacity="0.22" />
        <circle cx="374" cy="202" r="20" fill="none" stroke="#c0554d" strokeWidth="1.8" opacity="0.22" />

        {/* One attacking route, drawn on hover */}
        <path
          className="menu-route"
          d="M 96 258 C 168 240 190 168 264 154 C 330 142 372 178 398 226"
          fill="none"
          stroke="#eaf9ff"
          strokeWidth="3.6"
          strokeLinecap="round"
          strokeDasharray="360"
        />
        <circle className="menu-dot menu-dot--a" cx="96" cy="258" r="9" fill="#f4fcff" />
        <circle className="menu-dot menu-dot--b" cx="264" cy="154" r="7.5" fill="#9fd6ea" opacity="0.72" />
        <circle className="menu-dot menu-dot--c" cx="398" cy="226" r="7.5" fill="#9fd6ea" opacity="0.46" />
        <circle cx="170" cy="228" r="6" fill="#c0554d" opacity="0.5" />
        <circle cx="300" cy="252" r="6" fill="#c0554d" opacity="0.5" />

        {/* Coach leaning over the board, foreground */}
        <g className="menu-hero">
          <path
            d="M 152 440 c -6 -74 14 -122 56 -142 c -21 -16 -28 -46 -14 -70 c 13 -23 44 -31 66 -18 c 23 13 31 45 17 68 c -5 9 -13 16 -21 20 c 44 19 63 68 57 142 z"
            fill="#04080c"
          />
          <path
            d="M 208 298 c 26 -12 52 -12 78 0"
            fill="none"
            stroke="#8fd8f0"
            strokeWidth="2.6"
            strokeOpacity="0.28"
            strokeLinecap="round"
          />
          <path
            d="M 158 400 c 8 -50 26 -82 50 -96"
            fill="none"
            stroke="#8fd8f0"
            strokeWidth="2.2"
            strokeOpacity="0.14"
            strokeLinecap="round"
          />
        </g>
      </MenuDiorama>
    );
  }

  /** Roster Board — three skaters on a bench rail, centre one stepping forward. */
  function RosterDiorama({ hovered, prefersReducedMotion }) {
    const skater =
      "M 0 -128 c 15 0 27 12 27 27 c 0 11 -6 20 -14 25 c 22 6 36 20 42 42 l 16 60 l -26 8 l -16 -50 l -6 132 l -20 0 l -6 -96 l -6 96 l -20 0 l -6 -132 l -16 50 l -26 -8 l 16 -60 c 6 -22 20 -36 42 -42 c -8 -5 -14 -14 -14 -25 c 0 -15 12 -27 27 -27 z";
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Lineup"
        name="Roster Board"
        accent="#dbe7ef"
        mount="wire"
        mountAccent="#6f7d86"
      >
        <defs>
          <linearGradient id="roHero" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#2d3f4a" />
            <stop offset="100%" stopColor="#05090d" />
          </linearGradient>
          <linearGradient id="roBack" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#141f27" />
            <stop offset="100%" stopColor="#04070a" />
          </linearGradient>
          <linearGradient id="roFloor" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#c9dbe6" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#c9dbe6" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Arena floor wash so the group stands on something */}
        <ellipse cx="220" cy="386" rx="182" ry="28" fill="url(#roFloor)" />

        {/* Two support skaters set back and outboard: further away, dimmer, and
            edge-lit only, so the trio reads as depth rather than a pattern. */}
        <g className="menu-wing menu-wing--l">
          <g transform="translate(92 312) scale(0.54)">
            <path
              d={skater}
              fill="url(#roBack)"
              stroke="#9dbccd"
              strokeWidth="3.4"
              strokeOpacity="0.4"
            />
          </g>
        </g>
        <g className="menu-wing menu-wing--r">
          <g transform="translate(348 312) scale(0.54)">
            <path
              d={skater}
              fill="url(#roBack)"
              stroke="#9dbccd"
              strokeWidth="3.4"
              strokeOpacity="0.4"
            />
          </g>
        </g>

        {/* Front skater, closer and taller, catching the arena light */}
        <g className="menu-hero">
          <g transform="translate(220 272) scale(0.98)">
            <path
              d={skater}
              fill="url(#roHero)"
              stroke="#e8f2f8"
              strokeWidth="2"
              strokeOpacity="0.34"
            />
            <path
              d="M -22 -74 c 14 -7 30 -7 44 0"
              fill="none"
              stroke="#e8f2f8"
              strokeWidth="3"
              strokeOpacity="0.4"
              strokeLinecap="round"
            />
          </g>
        </g>

        {/* Depth chart rungs rather than a nameplate card */}
        {[0, 1, 2].map((i) => (
          <path
            key={`ro-rung-${i}`}
            className="menu-bar"
            style={{ "--i": i }}
            d={`M ${152 + i * 8} ${400 + i * 15} L ${288 - i * 8} ${400 + i * 15}`}
            stroke="#c9dbe6"
            strokeWidth="4"
            strokeLinecap="round"
            opacity={0.32 - i * 0.09}
          />
        ))}
      </MenuDiorama>
    );
  }

  /** Trade Hub — two asset dossiers with a live transaction beam. */
  function TradeHubDiorama({ hovered, prefersReducedMotion }) {
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Deadline"
        name="Trade Hub"
        accent="#e0a06a"
      >
        <defs>
          <linearGradient id="trBodyL" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#263946" />
            <stop offset="100%" stopColor="#05090d" />
          </linearGradient>
          <linearGradient id="trBodyR" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#3a2e22" />
            <stop offset="100%" stopColor="#07080a" />
          </linearGradient>
          <linearGradient id="trBeam" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#7fd4ea" stopOpacity="0" />
            <stop offset="50%" stopColor="#f6fbff" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#e0a06a" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Two opposing asset groups: a player bust plus his paperwork on each
            side, edge-lit in opposite colours so the exchange reads instantly */}
        <g className="menu-mass menu-mass--l">
          <path
            d="M 44 262 c 0 -50 22 -82 62 -94 c -19 -13 -28 -38 -21 -60 c 7 -23 32 -36 55 -30 c 24 6 38 30 32 54 c -4 16 -14 27 -27 34 c 41 12 62 46 62 96 z"
            fill="url(#trBodyL)"
            stroke="#9fd6ea"
            strokeWidth="2.6"
            strokeOpacity="0.4"
          />
          <path
            d="M 32 288 L 168 262 L 184 356 L 48 382 Z"
            fill="#0b1319"
            stroke="#7fa8bd"
            strokeWidth="2.2"
            strokeOpacity="0.44"
          />
          <path d="M 54 306 L 152 288" stroke="#9fd6ea" strokeWidth="7" strokeOpacity="0.5" strokeLinecap="round" />
          <path d="M 58 332 L 128 319" stroke="#9fd6ea" strokeWidth="6" strokeOpacity="0.28" strokeLinecap="round" />
        </g>

        <g className="menu-mass menu-mass--r">
          <path
            d="M 396 262 c 0 -50 -22 -82 -62 -94 c 19 -13 28 -38 21 -60 c -7 -23 -32 -36 -55 -30 c -24 6 -38 30 -32 54 c 4 16 14 27 27 34 c -41 12 -62 46 -62 96 z"
            fill="url(#trBodyR)"
            stroke="#e8c090"
            strokeWidth="2.6"
            strokeOpacity="0.4"
          />
          <path
            d="M 408 288 L 272 262 L 256 356 L 392 382 Z"
            fill="#100d0a"
            stroke="#c49a6a"
            strokeWidth="2.2"
            strokeOpacity="0.44"
          />
          <path d="M 386 306 L 288 288" stroke="#e8c090" strokeWidth="7" strokeOpacity="0.46" strokeLinecap="round" />
          <path d="M 382 332 L 312 319" stroke="#e8c090" strokeWidth="6" strokeOpacity="0.26" strokeLinecap="round" />
        </g>

        {/* Transaction path with the pick travelling along it */}
        <path
          className="menu-path"
          d="M 148 148 C 186 76 254 76 292 148"
          fill="none"
          stroke="#8fa8b8"
          strokeWidth="2.2"
          strokeOpacity="0.34"
          strokeDasharray="7 9"
        />
        <rect className="menu-beam" x="130" y="238" width="180" height="3.5" fill="url(#trBeam)" />

        <g className="menu-token">
          <path
            d="M 220 62 l 32 18 l 0 36 l -32 18 l -32 -18 l 0 -36 z"
            fill="#100e0b"
            stroke="#e0a06a"
            strokeWidth="2.6"
            strokeOpacity="0.85"
          />
          <circle cx="220" cy="98" r="6" fill="#f0d0a8" opacity="0.9" />
        </g>
        <circle className="menu-alert" cx="296" cy="74" r="7" fill="#e0705f" opacity="0.8" />
      </MenuDiorama>
    );
  }

  /** Contracts — lit deal sheet, signature line, cap figure. */
  function ContractsDiorama({
    hovered,
    prefersReducedMotion,
    capSpace,
    capPressure = false,
  }) {
    const capText = formatMoney(capSpace);
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Cap Space"
        name="Contracts"
        accent={capPressure ? "#e0705f" : "#c4a46a"}
        mount="ledge"
        mountAccent="#8a7048"
      >
        <defs>
          <linearGradient id="ctPage" x1="16%" y1="0%" x2="84%" y2="100%">
            <stop offset="0%" stopColor="#8e8878" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#26262a" stopOpacity="0.46" />
          </linearGradient>
          <linearGradient id="ctPen" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#31261a" />
            <stop offset="100%" stopColor="#0a0906" />
          </linearGradient>
        </defs>

        {/* Deal sheet resting on the ledge, lit only along the brass edge */}
        <path d="M 104 152 L 330 118 L 352 356 L 126 390 Z" fill="#04070a" opacity="0.6" />
        <path
          d="M 96 144 L 322 110 L 344 348 L 118 382 Z"
          fill="url(#ctPage)"
          stroke="#c4a46a"
          strokeWidth="2"
          strokeOpacity="0.45"
        />
        <path d="M 96 144 L 322 110 L 324 128 L 98 162 Z" fill="#e8dcc0" opacity="0.14" />

        {[0, 1, 2].map((i) => (
          <path
            key={`ct-rule-${i}`}
            d={`M 120 ${196 + i * 26} L ${286 - i * 34} ${171 + i * 26}`}
            stroke="#1c211f"
            strokeWidth="7"
            strokeOpacity={0.24 - i * 0.05}
            strokeLinecap="round"
          />
        ))}

        <g transform="rotate(-8.5 130 288)">
          <text
            x="130"
            y="288"
            fontSize="36"
            fontWeight="800"
            fill={capPressure ? "#7c231c" : "#171c1a"}
            opacity="0.82"
          >
            {capText}
          </text>
          <text
            x="132"
            y="312"
            fontSize="16"
            fontWeight="700"
            fill="#2e332f"
            opacity="0.55"
            letterSpacing="2.4"
          >
            CAP SPACE
          </text>
        </g>

        <path
          className="menu-sign"
          d="M 126 350 C 160 330 178 360 204 342 C 230 324 250 354 296 330"
          fill="none"
          stroke="#0d1116"
          strokeWidth="4.5"
          strokeLinecap="round"
          strokeDasharray="240"
        />

        <path d="M 290 190 L 372 102 L 396 124 L 314 212 L 282 224 Z" fill="url(#ctPen)" />
        <path d="M 282 224 L 314 212 L 292 234 Z" fill="#c4a46a" />
        <path d="M 360 114 L 384 136" stroke="#c4a46a" strokeWidth="4" opacity="0.4" />
      </MenuDiorama>
    );
  }

  /** Standings — vertical division ladder, user rung lit. */
  function StandingsDiorama({
    hovered,
    prefersReducedMotion,
    standingsRank,
    record,
  }) {
    const parsedRank = parseInt(String(standingsRank || "").trim(), 10);
    const activeRow = Number.isFinite(parsedRank)
      ? Math.min(Math.max(parsedRank - 1, 0), 4)
      : 1;
    const footer = safeText(record, "") || safeText(standingsRank, "");
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Division Race"
        name="Standings"
        accent="#8fd8f0"
      >
        <defs>
          {/* The rungs are flat horizontal strokes, so their bounding box has
              no height and an objectBoundingBox gradient paints nothing. */}
          <linearGradient
            id="sdRung"
            gradientUnits="userSpaceOnUse"
            x1="96"
            y1="0"
            x2="338"
            y2="0"
          >
            <stop offset="0%" stopColor="#8fb6cb" />
            <stop offset="100%" stopColor="#1b2f3c" />
          </linearGradient>
          <linearGradient
            id="sdRungLive"
            gradientUnits="userSpaceOnUse"
            x1="96"
            y1="0"
            x2="338"
            y2="0"
          >
            <stop offset="0%" stopColor="#b6ecff" />
            <stop offset="100%" stopColor="#2e708c" />
          </linearGradient>
        </defs>

        {/* Ladder: rungs only, no plate behind them */}
        <path d="M 84 56 L 84 372" stroke="#3c4c57" strokeWidth="2.5" opacity="0.5" />

        {[0, 1, 2, 3, 4].map((i) => {
          const live = i === activeRow;
          return (
            <g key={`sd-${i}`} className="menu-bar" style={{ "--i": i }}>
              <path
                d={`M 96 ${72 + i * 66} L ${338 - i * 44} ${72 + i * 66}`}
                stroke={live ? "url(#sdRungLive)" : "url(#sdRung)"}
                strokeWidth={live ? 22 : 16}
                strokeLinecap="round"
                opacity={live ? 1 : 0.82}
              />
              <text
                x="52"
                y={82 + i * 66}
                fontSize="28"
                fontWeight="800"
                fill={live ? "#dff4ff" : "#6e879a"}
              >
                {i + 1}
              </text>
              {live ? (
                <circle
                  className="menu-alert"
                  cx={356 - i * 44}
                  cy={72 + i * 66}
                  r="8"
                  fill="#e9f8ff"
                />
              ) : null}
            </g>
          );
        })}

        {footer ? (
          <text
            x="96"
            y="412"
            fontSize="26"
            fontWeight="800"
            fill="#9fc0d1"
            opacity="0.8"
            letterSpacing="2"
          >
            {footer}
          </text>
        ) : null}
      </MenuDiorama>
    );
  }

  /** Storylines — newswire strips surfacing out of the dark. */
  function StorylinesDiorama({
    hovered,
    prefersReducedMotion,
    activeStorylines,
  }) {
    const count = Number(activeStorylines || 0);
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Newswire"
        name="Storylines"
        accent="#dfe8f0"
      >
        {/* Three headline slips hanging at different depths and angles — the
            nearest is bright and square-on, the ones behind fall away. Drawn
            back-to-front so the top slip overlaps the others. */}
        {[2, 1, 0].map((i) => {
          const y = 108 + i * 112;
          const x = 54 + i * 30;
          const fade = [1, 0.58, 0.36][i];
          const tilt = [-1.6, 2.2, -3][i];
          const size = [1, 0.9, 0.8][i];
          return (
            <g
              key={`sl-${i}`}
              transform={`rotate(${tilt} ${x} ${y}) scale(${size}) translate(${
                (x * (1 - size)) / size
              } ${(y * (1 - size)) / size})`}
            >
              {/* Inner group carries the hover class: a CSS transform would
                  otherwise replace the static tilt above. */}
              <g className="menu-strip" style={{ "--i": i }}>
                <path
                  d={`M ${x} ${y - 24} L ${x} ${y + 44}`}
                  stroke="#e07a5f"
                  strokeWidth="4"
                  strokeOpacity={0.62 * fade}
                  strokeLinecap="round"
                />
                <path
                  d={`M ${x + 18} ${y - 26} L ${x + 18 + 66} ${y - 26}`}
                  stroke="#8fb4c8"
                  strokeWidth="5"
                  strokeOpacity={0.4 * fade}
                  strokeLinecap="round"
                />
                <path
                  d={`M ${x + 18} ${y} L ${x + 18 + (258 - i * 54)} ${y}`}
                  stroke="#f6fafd"
                  strokeWidth="16"
                  strokeOpacity={0.82 * fade}
                  strokeLinecap="round"
                />
                <path
                  d={`M ${x + 18} ${y + 26} L ${x + 18 + (196 - i * 46)} ${y + 26}`}
                  stroke="#c3d4e0"
                  strokeWidth="7"
                  strokeOpacity={0.46 * fade}
                  strokeLinecap="round"
                />
              </g>
            </g>
          );
        })}

        <circle className="menu-alert" cx="356" cy="76" r="9" fill="#e0705f" />

        {count > 0 ? (
          <text
            x="392"
            y="404"
            textAnchor="end"
            fontSize="26"
            fontWeight="800"
            fill="#c3d4e0"
            opacity="0.6"
            letterSpacing="1.5"
          >
            {`${count} ACTIVE`}
          </text>
        ) : null}
      </MenuDiorama>
    );
  }

  /** Draft Class — prospect walking into the stage spotlight. */
  function DraftClassDiorama({
    hovered,
    prefersReducedMotion,
    seasonYear,
    currentDate,
  }) {
    const yearNum = Number(seasonYear);
    const fallbackYear = parseFranchiseDateParts(currentDate).year;
    const yearLabel =
      Number.isFinite(yearNum) && yearNum > 1900
        ? String(yearNum)
        : fallbackYear > 1900
        ? String(fallbackYear)
        : "";
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker={yearLabel ? `${yearLabel} Class` : "Prospects"}
        name="Draft Class"
        viewBox="0 0 440 460"
        accent="#e8c890"
      >
        <defs>
          <linearGradient id="dcSpot" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#fff4de" stopOpacity="0.34" />
            <stop offset="55%" stopColor="#e8c890" stopOpacity="0.1" />
            <stop offset="100%" stopColor="#e8c890" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="dcPool" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#f4e0b8" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#f4e0b8" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="dcArch" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#c9a468" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#c9a468" stopOpacity="0.06" />
          </linearGradient>
          <linearGradient id="dcSuit" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#0f1922" />
            <stop offset="100%" stopColor="#010305" />
          </linearGradient>
        </defs>

        <path
          d="M 66 400 L 66 176 A 154 154 0 0 1 374 176 L 374 400"
          fill="none"
          stroke="url(#dcArch)"
          strokeWidth="9"
        />
        <path
          d="M 96 400 L 96 182 A 124 124 0 0 1 344 182 L 344 400"
          fill="none"
          stroke="#c9a468"
          strokeWidth="2"
          strokeOpacity="0.16"
        />

        <path className="menu-spot" d="M 220 60 L 142 398 L 298 398 Z" fill="url(#dcSpot)" />
        <ellipse className="menu-spot" cx="220" cy="398" rx="86" ry="15" fill="url(#dcPool)" />

        {/* The prospect is nearly black against the spotlight and carries a
            warm rim on the stage side, so the silhouette does the reading. */}
        <g className="menu-hero">
          <path
            d="M 220 152 c 18 0 32 15 32 34 c 0 14 -7 25 -17 30 c 28 10 43 36 46 76 l 12 106 l -38 0 l -8 -80 l -6 80 l -38 0 l -6 -80 l -8 80 l -38 0 l 12 -106 c 3 -40 18 -66 46 -76 c -10 -5 -17 -16 -17 -30 c 0 -19 14 -34 32 -34 z"
            fill="url(#dcSuit)"
            stroke="#f0dcb0"
            strokeWidth="2"
            strokeOpacity="0.26"
          />
          {/* Suit lapel notch keeps it reading as tailored rather than a cone */}
          <path
            d="M 220 224 l -13 22 l 13 16 l 13 -16 z"
            fill="#f0e4c4"
            opacity="0.16"
          />
          <path
            d="M 252 236 l 40 14 l -9 52 l -38 -13 z"
            fill="#e8dcc0"
            opacity="0.7"
          />
          <path d="M 252 236 l 40 14" stroke="#8a7a58" strokeWidth="2" opacity="0.45" />
        </g>

        {[0, 1, 2].map((i) => (
          <g key={`dc-${i}`} className="menu-card" style={{ "--i": i }}>
            <rect
              x={62 + i * 116}
              y="410"
              width="60"
              height="40"
              rx="4"
              fill="#0d1218"
              stroke="#c9a468"
              strokeWidth="1.5"
              strokeOpacity={i === 0 ? 0.7 : 0.28}
            />
            <text
              x={92 + i * 116}
              y="438"
              textAnchor="middle"
              fontSize="21"
              fontWeight="800"
              fill={i === 0 ? "#f0dcb0" : "#6f7d88"}
            >
              {i + 1}
            </text>
          </g>
        ))}
      </MenuDiorama>
    );
  }

  /**
   * Franchise Identity — brass crest ring around the real club crest (that plate
   * is 3D geometry showing through the transparent middle of this scene) with a
   * hanging home sweater below it.
   */
  function IdentityDiorama({ hovered, prefersReducedMotion }) {
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="The Club"
        name="Franchise Identity"
        accent="#c4a46a"
      >
        <defs>
          <linearGradient id="idSweater" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#24384a" />
            <stop offset="100%" stopColor="#060b11" />
          </linearGradient>
          <linearGradient id="idSweep" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0" />
            <stop offset="50%" stopColor="#ffffff" stopOpacity="0.42" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          <clipPath id="idSweaterClip">
            <path d="M 176 272 C 188 262 204 256 214 256 C 219 265 221 265 226 256 C 236 256 252 262 264 272 L 300 300 L 284 340 L 266 332 L 261 424 L 179 424 L 174 332 L 156 340 L 140 300 Z" />
          </clipPath>
        </defs>

        {/* The crest medallion itself is 3D geometry showing through here, so
            this scene only adds the hanging sweater and its rod. */}
        <path d="M 116 250 L 324 250" stroke="#c4a46a" strokeWidth="2.4" strokeOpacity="0.32" strokeLinecap="round" />
        <path d="M 214 236 C 214 226 226 226 226 236 L 220 252" fill="none" stroke="#c4a46a" strokeWidth="2.6" strokeOpacity="0.42" strokeLinecap="round" />

        <g className="menu-hero">
          <path
            d="M 176 272 C 188 262 204 256 214 256 C 219 265 221 265 226 256 C 236 256 252 262 264 272 L 300 300 L 284 340 L 266 332 L 261 424 L 179 424 L 174 332 L 156 340 L 140 300 Z"
            fill="url(#idSweater)"
            stroke="#c4a46a"
            strokeWidth="1.8"
            strokeOpacity="0.34"
          />
          <g clipPath="url(#idSweaterClip)">
            <path d="M 140 392 L 300 392" stroke="#c4a46a" strokeWidth="7" strokeOpacity="0.28" />
            <path d="M 140 406 L 300 406" stroke="#dbe7ef" strokeWidth="4" strokeOpacity="0.14" />
            <path d="M 144 306 L 296 306" stroke="#c4a46a" strokeWidth="5" strokeOpacity="0.2" />
            {/* Sleeve cuffs and shoulder yoke — the cues that separate a
                sweater on a hanger from a plain trapezoid */}
            <path d="M 140 296 L 178 316" stroke="#c4a46a" strokeWidth="6" strokeOpacity="0.24" />
            <path d="M 262 316 L 300 296" stroke="#c4a46a" strokeWidth="6" strokeOpacity="0.24" />
            <path
              d="M 178 278 C 200 292 240 292 262 278"
              fill="none"
              stroke="#dbe7ef"
              strokeWidth="2.4"
              strokeOpacity="0.22"
            />
            <rect className="menu-sweep" x="-160" y="250" width="70" height="184" fill="url(#idSweep)" />
          </g>
          {/* Collar opening */}
          <path
            d="M 206 258 C 212 270 228 270 234 258"
            fill="none"
            stroke="#c4a46a"
            strokeWidth="2.6"
            strokeOpacity="0.4"
            strokeLinecap="round"
          />
        </g>
      </MenuDiorama>
    );
  }

  /**
   * Stats — performance hologram: skater inside an analytics ring, live team
   * record as the headline figure, shot map dots and a rising trend line.
   */
  function StatsDiorama({ hovered, prefersReducedMotion, record }) {
    const recordText = safeText(record, "");
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Performance"
        name="Stats"
        accent="#7fe4f0"
      >
        <defs>
          <linearGradient id="saSkater" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#123240" />
            <stop offset="100%" stopColor="#02070a" />
          </linearGradient>
          <radialGradient id="saHalo" cx="50%" cy="52%" r="52%">
            <stop offset="0%" stopColor="#7fe4f0" stopOpacity="0.26" />
            <stop offset="100%" stopColor="#7fe4f0" stopOpacity="0" />
          </radialGradient>
        </defs>

        <circle className="menu-glow" cx="220" cy="238" r="150" fill="url(#saHalo)" />

        {/* Analytics ring — completes itself on hover */}
        <circle
          className="menu-ring"
          cx="220"
          cy="238"
          r="132"
          fill="none"
          stroke="#7fe4f0"
          strokeWidth="3"
          strokeOpacity="0.5"
          strokeDasharray="830"
          strokeLinecap="round"
          transform="rotate(-90 220 238)"
        />
        <circle cx="220" cy="238" r="146" fill="none" stroke="#7fe4f0" strokeWidth="1" strokeOpacity="0.14" />
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <path
            key={`sa-tick-${i}`}
            d="M 220 96 L 220 108"
            stroke="#7fe4f0"
            strokeWidth="2"
            strokeOpacity="0.22"
            transform={`rotate(${i * 60} 220 238)`}
          />
        ))}

        {/* Skater mid-stride, built from separate masses so the pose stays
            legible at hub distance: head ahead of the hips, trailing leg
            extended, stick reaching down into the shooting lane. */}
        <g className="menu-hero">
          {[
            "M 196 272 L 216 290 L 138 332 L 122 310 Z",
            "M 210 206 L 172 232 L 162 216 L 202 190 Z",
            "M 234 182 C 262 192 270 214 258 240 L 222 290 L 184 268 L 208 204 C 214 188 222 178 234 182 Z",
            "M 222 284 L 252 302 L 264 356 L 234 362 Z",
            "M 250 214 L 288 244 L 276 260 L 240 232 Z",
          ].map((d, i) => (
            <path
              key={`sa-mass-${i}`}
              d={d}
              fill="url(#saSkater)"
              stroke="#a8ecf8"
              strokeWidth="2"
              strokeOpacity="0.34"
            />
          ))}
          <circle
            cx="252"
            cy="166"
            r="25"
            fill="url(#saSkater)"
            stroke="#a8ecf8"
            strokeWidth="2"
            strokeOpacity="0.34"
          />
          {/* Blades and stick — the only bright hardware in the scene */}
          <path d="M 228 366 L 272 360" stroke="#dff6ff" strokeWidth="3" strokeOpacity="0.5" strokeLinecap="round" />
          <path d="M 114 316 L 142 336" stroke="#dff6ff" strokeWidth="3" strokeOpacity="0.4" strokeLinecap="round" />
          <path d="M 286 248 L 364 302" stroke="#c9dbe6" strokeWidth="6" strokeOpacity="0.58" strokeLinecap="round" />
          <path d="M 364 302 L 392 308" stroke="#c9dbe6" strokeWidth="8" strokeOpacity="0.58" strokeLinecap="round" />
        </g>

        {/* Shot-location dots */}
        {[
          [110, 196, 0.5],
          [92, 258, 0.32],
          [136, 320, 0.42],
          [340, 196, 0.28],
          [356, 244, 0.36],
          [318, 340, 0.24],
        ].map(([cx, cy, o], i) => (
          <circle
            key={`sa-dot-${i}`}
            className="menu-dot"
            style={{ "--i": i }}
            cx={cx}
            cy={cy}
            r="6"
            fill="#7fe4f0"
            opacity={o}
          />
        ))}

        {/* Rising trend line */}
        <path
          className="menu-graph"
          d="M 96 386 L 148 366 L 192 376 L 244 340 L 296 348 L 348 306"
          fill="none"
          stroke="#e8fbff"
          strokeWidth="3.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {recordText ? (
          <g className="menu-figure">
            <text
              x="220"
              y="72"
              textAnchor="middle"
              fontSize="38"
              fontWeight="800"
              fill="#d8f4fa"
              opacity="0.9"
              letterSpacing="1"
            >
              {recordText}
            </text>
            <text
              x="220"
              y="92"
              textAnchor="middle"
              fontSize="14"
              fontWeight="700"
              fill="#7fe4f0"
              opacity="0.5"
              letterSpacing="4"
            >
              RECORD
            </text>
          </g>
        ) : null}
      </MenuDiorama>
    );
  }

  /** Legacy Wall — trophy shrine under warm brass light. */
  function LegacyDiorama({ hovered, prefersReducedMotion }) {
    return (
      <MenuDiorama
        hovered={hovered}
        prefersReducedMotion={prefersReducedMotion}
        kicker="Honours"
        name="Legacy Wall"
        accent="#e6c878"
        mount="plinth"
        mountAccent="#c9a468"
      >
        <defs>
          <linearGradient id="lgCup" x1="20%" y1="0%" x2="80%" y2="100%">
            <stop offset="0%" stopColor="#f0d79a" />
            <stop offset="42%" stopColor="#b08c4c" />
            <stop offset="100%" stopColor="#3a2c15" />
          </linearGradient>
          <linearGradient id="lgBanner" x1="50%" y1="0%" x2="50%" y2="100%">
            <stop offset="0%" stopColor="#1a1610" />
            <stop offset="100%" stopColor="#070604" />
          </linearGradient>
          <linearGradient id="lgSweep" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#fff4d8" stopOpacity="0" />
            <stop offset="50%" stopColor="#fff4d8" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#fff4d8" stopOpacity="0" />
          </linearGradient>
          <radialGradient id="lgWarm" cx="50%" cy="34%" r="56%">
            <stop offset="0%" stopColor="#e6c878" stopOpacity="0.24" />
            <stop offset="100%" stopColor="#e6c878" stopOpacity="0" />
          </radialGradient>
          <clipPath id="lgCupClip">
            <path d="M 176 96 L 264 96 L 258 158 C 254 190 240 208 220 214 C 200 208 186 190 182 158 Z" />
          </clipPath>
        </defs>

        <ellipse className="menu-glow" cx="220" cy="180" rx="170" ry="160" fill="url(#lgWarm)" />

        {/* Retired-number banners hanging behind the cup */}
        {[
          [104, 0],
          [336, 1],
        ].map(([x, i]) => (
          <g key={`lg-banner-${i}`} className="menu-wing" style={{ "--i": i }}>
            <path
              d={`M ${x - 30} 54 L ${x + 30} 54 L ${x + 30} 232 L ${x} 262 L ${x - 30} 232 Z`}
              fill="url(#lgBanner)"
              stroke="#c9a468"
              strokeWidth="1.6"
              strokeOpacity="0.3"
            />
            <path d={`M ${x - 14} 108 L ${x + 14} 108`} stroke="#e6c878" strokeWidth="6" strokeOpacity="0.34" strokeLinecap="round" />
            <path d={`M ${x - 14} 136 L ${x + 14} 136`} stroke="#e6c878" strokeWidth="6" strokeOpacity="0.22" strokeLinecap="round" />
            <path d={`M ${x - 14} 164 L ${x + 14} 164`} stroke="#e6c878" strokeWidth="6" strokeOpacity="0.14" strokeLinecap="round" />
          </g>
        ))}

        {/* The cup */}
        <g className="menu-hero">
          <path
            d="M 176 96 L 264 96 L 258 158 C 254 190 240 208 220 214 C 200 208 186 190 182 158 Z"
            fill="url(#lgCup)"
          />
          <g clipPath="url(#lgCupClip)">
            <path className="menu-sweep" d="M -60 90 L 10 90 L -20 220 L -90 220 Z" fill="url(#lgSweep)" />
          </g>
          <path
            d="M 176 106 C 152 108 146 132 156 148 C 162 158 172 162 180 162"
            fill="none"
            stroke="#b08c4c"
            strokeWidth="9"
            strokeLinecap="round"
          />
          <path
            d="M 264 106 C 288 108 294 132 284 148 C 278 158 268 162 260 162"
            fill="none"
            stroke="#b08c4c"
            strokeWidth="9"
            strokeLinecap="round"
          />
          <path d="M 172 84 L 268 84 L 264 98 L 176 98 Z" fill="#f2dda6" opacity="0.9" />
          <path d="M 212 214 L 228 214 L 232 254 L 208 254 Z" fill="#8f7338" />
          <path d="M 186 254 L 254 254 L 262 282 L 178 282 Z" fill="#6d5729" />
          <path d="M 170 282 L 270 282 L 280 312 L 160 312 Z" fill="#4a3a1c" />
          <path d="M 160 312 L 280 312 L 280 322 L 160 322 Z" fill="#e6c878" opacity="0.32" />
        </g>
      </MenuDiorama>
    );
  }

  function PhysicalDraftBoard({ hovered, draftWeek = false }) {
    return <DraftWarRoomEntrance hovered={hovered} draftWeek={draftWeek} />;
  }

  function BroadcastScoreboard({ hovered, record, nextGame }) {
    return (
      <WallDisplayFrame width={1.78} height={1.02} accent={OFFICE_PALETTE.gold}>
        <RoundedBox
          position={[0, 0, 0.02]}
          args={[1.62, 0.88, 0.04]}
          radius={0.02}
          smoothness={6}
          raycast={() => null}
        >
          <meshStandardMaterial
            color="#060810"
            emissive={hovered ? "#142838" : "#0a1828"}
            emissiveIntensity={hovered ? 0.32 : 0.16}
            roughness={0.44}
            metalness={0.1}
          />
        </RoundedBox>

        <mesh position={[0, 0, 0.048]} raycast={() => null}>
          <planeGeometry args={[1.48, 0.74]} />
          <GlassMaterial opacity={0.07} />
        </mesh>

        <WallText position={[0, 0.36, 0.055]} size={0.042} color="#c9a86a">
          LEAGUE OPERATIONS
        </WallText>

        <WallText position={[0, 0.28, 0.055]} size={0.022} color="#6a8a9a">
          STANDINGS • SCORES • HEADLINES
        </WallText>

        <WallText position={[0, 0.06, 0.055]} size={0.034} color="#8aaaba">
          Record {safeText(record)}
        </WallText>

        <WallText position={[0, -0.06, 0.055]} size={0.028} color="#d8e0e8">
          Next {safeText(nextGame, "No game listed")}
        </WallText>

        <mesh position={[0, -0.3, 0.054]} raycast={() => null}>
          <boxGeometry args={[1.32, 0.06, 0.01]} />
          <meshStandardMaterial
            color="#101820"
            emissive="#1a3040"
            emissiveIntensity={hovered ? 0.22 : 0.1}
            roughness={0.5}
          />
        </mesh>

        <WallText position={[0, -0.3, 0.062]} size={0.022} color="#7a9aaa">
          BROADCAST • NEWS • LEAGUE FEED
        </WallText>
      </WallDisplayFrame>
    );
  }
  
  function ArenaWindowObject({ hovered, nextGame, seasonYear }) {
    return (
      <group>
        <mesh position={[0, 0, -0.04]} castShadow raycast={() => null}>
          <boxGeometry args={[1.35, 0.72, 0.06]} />
          <WoodMaterial color="#2a2418" roughness={0.58} />
        </mesh>

        <mesh raycast={() => null}>
          <boxGeometry args={[1.22, 0.58, 0.03]} />
          <meshStandardMaterial
            color={hovered ? "#1a4a6a" : "#122838"}
            emissive="#143850"
            emissiveIntensity={hovered ? 0.28 : 0.14}
            transparent
            opacity={0.88}
            roughness={0.35}
          />
        </mesh>

        <WallText position={[0, 0.18, 0.04]} size={0.04} color="#c4a46a">
          GAME DAY
        </WallText>
        <WallText position={[0, 0.02, 0.04]} size={0.028} color="#d8e0e8" maxWidth={1.1}>
          {safeText(nextGame, "No game listed")}
        </WallText>
        <WallText position={[0, -0.18, 0.04]} size={0.022} color="#6a8898">
          {Number(seasonYear) > 0 ? String(seasonYear) : "PREVIEW"}
        </WallText>
        <pointLight position={[0, 0.1, 0.35]} intensity={hovered ? 0.25 : 0.1} color="#6a9aba" distance={1.2} />
      </group>
    );
  }
  
  // Phase 2 idea:
  // Split the full GLB in Blender into individual runtime props:
  // desk.glb, chair.glb, crt-monitor.glb, phone.glb, keyboard-mouse.glb,
  // binders.glb, filing-cabinet.glb, lamp.glb, mugs.glb, printer.glb.
  // Then replace procedural desk-zone props one by one while keeping InteractiveGroup hitboxes.
  function RetroOfficeModel({
    enabled = USE_RETRO_OFFICE_PACK,
    lowPowerMode = false,
    transform = RETRO_OFFICE_TRANSFORM,
  }) {
    const { scene } = useGLTF(RETRO_OFFICE_MODEL_PATH);

    const clonedScene = useMemo(() => {
      if (!scene) return null;
      return scene.clone(true);
    }, [scene]);

    useEffect(() => {
      if (!clonedScene) return;

      clonedScene.traverse((child) => {
        if (!child.isMesh) return;

        child.castShadow = !lowPowerMode;
        child.receiveShadow = true;

        // Critical: imported art should never block current InteractiveGroup hitboxes.
        child.raycast = () => null;

        const materials = Array.isArray(child.material)
          ? child.material
          : child.material
            ? [child.material]
            : [];

        materials.forEach((mat) => {
          if (!mat) return;
          mat.needsUpdate = true;

          if ("roughness" in mat && mat.roughness == null) {
            mat.roughness = 0.72;
          }

          if ("metalness" in mat && mat.metalness == null) {
            mat.metalness = 0.05;
          }
        });
      });

      if (process.env.NODE_ENV !== "production") {
        const meshNames = [];
        clonedScene.traverse((child) => {
          if (child.isMesh && child.name) meshNames.push(child.name);
        });
        console.log("[RetroOfficePack] Loaded mesh count:", meshNames.length);
        console.log("[RetroOfficePack] Sample meshes:", meshNames.slice(0, 40));
      }
    }, [clonedScene, lowPowerMode]);

    if (!enabled || !clonedScene) return null;

    return (
      <group
        position={transform.position}
        rotation={transform.rotation}
        scale={transform.scale}
        raycast={() => null}
      >
        <primitive object={clonedScene} />
      </group>
    );
  }

  useGLTF.preload(RETRO_OFFICE_MODEL_PATH);

  function OfficeScene({
    teamName,
    teamLogo,
    seasonYear,
    currentDate,
    record,
    capSpace,
    nextGame,
    standingsRank,
    unreadMessages,
    pendingTasks,
    activeStorylines,
    hoveredId,
    setHoveredId,
    handleOpenPanel,
    resetToken,
    officePictures,
    bestPlayer,
    officeMood,
    activePanel,
    lowPowerMode = false,
    prefersReducedMotion = false,
    championshipCount = 0,
    capPressure = false,
    officeWeather = null,
  }) {
    const mood = officeMood || {};
    const [leagueOpsClickToken, setLeagueOpsClickToken] = useState(0);

    const handleLeagueOpsOpen = useCallback(() => {
      setLeagueOpsClickToken((token) => token + 1);
      handleOpenPanel(OFFICE_PANEL_IDS.LEAGUE_CENTRAL);
    }, [handleOpenPanel]);

    const tradeActivity =
      mood.isTradeDeadline ||
      Number(unreadMessages || 0) > 0 ||
      mood.hasUrgentDecisions;
    const weather = officeWeather || deriveSeasonalWeather(currentDate);

    return (
      <>
        <color attach="background" args={[OFFICE_PALETTE.void]} />
        <fog attach="fog" args={[OFFICE_PALETTE.void, 16, 34]} />
  
        <CameraRig
          resetToken={resetToken}
          activePanel={activePanel}
          lowPowerMode={lowPowerMode}
          prefersReducedMotion={prefersReducedMotion}
          hoveredId={hoveredId}
          leagueOpsClickToken={leagueOpsClickToken}
        />
        {!lowPowerMode ? <SoftShadows size={18} samples={10} focus={0.55} /> : null}
        <Environment preset="city" environmentIntensity={0.34} />
  
        <hemisphereLight intensity={0.55} color="#b8dce8" groundColor="#1a3840" />
        <ambientLight intensity={0.4} color="#88a8b0" />
        <PracticalLights
          lowPowerMode={lowPowerMode}
          prefersReducedMotion={prefersReducedMotion}
        />
        {/* Window daylight spill */}
        <pointLight position={[3.6, 2.1, 0.4]} intensity={0.55} color={weather.light} distance={5.5} />
  
        {!lowPowerMode ? (
          <AccumulativeShadows
            temporal
            frames={48}
            color="#1a1814"
            colorBlend={0.85}
            opacity={0.18}
            scale={8}
            position={[0, 0.018, 0]}
          >
            <RandomizedLight
              amount={4}
              radius={2.8}
              ambient={0.28}
              intensity={0.62}
              position={[1.6, 4.2, 1.8]}
              color="#c8a868"
            />
          </AccumulativeShadows>
        ) : null}

        {USE_RETRO_OFFICE_PACK ? <RetroOfficeModel lowPowerMode={lowPowerMode} /> : null}

        <RoomShell />
        <OfficeFurniture
          teamLogo={teamLogo}
          teamName={teamName}
          mood={mood}
          championshipCount={championshipCount}
        />

        <CityWeatherWindow currentDate={currentDate} weather={weather} />

        <Desk teamName={teamName} teamLogo={teamLogo}>
          <InteractiveGroup
            id="dashboard"
            label=""
            position={[0.08, 0.885, 0.18]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hoverScale={1}
            hoverLift={0}
            hitBoxArgs={OFFICE_HITBOXES.dashboard}
            hitBoxPosition={[0, 0.3, 0.02]}
            activateOnPointerDown
            hideHoverLabel
            showHoverCorners={false}
          >
            {(hovered) => (
              <>
                <LaptopObject
                  hovered={hovered}
                  focused={activePanel === OFFICE_PANEL_IDS.DASHBOARD}
                  teamName={teamName}
                  teamLogo={teamLogo}
                  currentDate={currentDate}
                  nextGame={nextGame}
                  record={record}
                  priorityCount={Number(activeStorylines || 0)}
                  seasonPhase={mood.seasonPhase || mood.officeMode || ""}
                />
                <LandmarkLabel
                  hovered={hovered}
                  prefersReducedMotion={prefersReducedMotion}
                  kicker="Command"
                  name="Franchise Command"
                  accent="#9fd6ea"
                  position={[0, 0.66, 0.12]}
                />
              </>
            )}
          </InteractiveGroup>

          {/* Desk dressing — the interactive Trade Hub lives on the wall */}
          <group position={[-1.52, 0.885, 0.62]}>
            <PhoneObject
              hovered={hoveredId === "messages"}
              unreadMessages={unreadMessages}
              hasTradeActivity={tradeActivity}
              callerLabel="TRADE DESK"
            />
          </group>

          <InteractiveGroup
            id="calendar"
            label=""
            position={[1.48, 0.885, 0.62]}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
            onOpen={handleOpenPanel}
            hoverScale={1}
            hoverLift={0}
            hitBoxArgs={OFFICE_HITBOXES.calendar}
            hitBoxPosition={[0, 0.24, 0]}
            activateOnPointerDown
            hideHoverLabel
            showHoverCorners={false}
          >
            {(hovered) => (
              <>
                <CalendarObject
                  hovered={hovered}
                  currentDate={currentDate}
                  nextGame={nextGame}
                  teamLogo={teamLogo}
                  teamName={teamName}
                />
                <LandmarkLabel
                  hovered={hovered}
                  prefersReducedMotion={prefersReducedMotion}
                  kicker="Season Timeline"
                  name="Calendar"
                  accent="#dbe7ef"
                  position={[0, 0.26, 0.06]}
                />
              </>
            )}
          </InteractiveGroup>

          {/* Desk dressing — the interactive Storylines wall carries the panel */}
          <group position={[-0.72, 0.885, 0.78]}>
            <NewspaperObject
              hovered={hoveredId === "news"}
              activeStorylines={activeStorylines}
            />
          </group>

          {mood.isTradeDeadline || mood.isDraftWeek ? <DeskClutter /> : null}
          <CoffeeAndPuck />
        </Desk>

        {/* ---- Back wall. Three clustered installations rather than one even
                row: team building on the left, the club in the middle, league
                intelligence on the right. Every landmark uses the same
                footprint, hitbox and label system; only the mount, depth and
                accent change. ---- */}

        <InteractiveGroup
          id="draftClass"
          label=""
          position={[MENU_COLUMNS.farLeft, MENU_LANDMARK.upperBandY, -3.36]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <DraftClassDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
              seasonYear={seasonYear}
              currentDate={currentDate}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="lines"
          label=""
          position={[MENU_COLUMNS.innerRight, MENU_LANDMARK.upperBandY, -3.34]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <StrategyDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="contracts"
          label=""
          position={[MENU_COLUMNS.lowLeftOuter, MENU_LANDMARK.lowerBandY, -3.3]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <ContractsDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
              capSpace={capSpace}
              capPressure={capPressure}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="messages"
          label=""
          position={[MENU_COLUMNS.left, MENU_LANDMARK.upperBandY, -3.3]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
          lowPowerMode={lowPowerMode}
        >
          {(hovered) => (
            <TradeHubDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="teamIdentity"
          label=""
          position={[MENU_COLUMNS.center, MENU_LANDMARK.crestY, -3.4]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <>
              <WallHeroLogo
                teamLogo={teamLogo}
                teamName={teamName}
                hovered={hovered}
                position={[0, 0.278, 0.07]}
                scale={0.42}
              />
              <IdentityDiorama
                hovered={hovered}
                prefersReducedMotion={prefersReducedMotion}
              />
            </>
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="roster"
          label=""
          position={[MENU_COLUMNS.innerLeft, MENU_LANDMARK.upperBandY, -3.34]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <RosterDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="leagueCentral"
          label=""
          position={[MENU_COLUMNS.farRight, MENU_LANDMARK.upperBandY, -3.36]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleLeagueOpsOpen}
          openId={OFFICE_PANEL_IDS.LEAGUE_CENTRAL}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <LeagueOpsSilhouette
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="news"
          label=""
          position={[MENU_COLUMNS.lowRightInner, MENU_LANDMARK.lowerBandY, -3.28]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <StorylinesDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
              activeStorylines={activeStorylines}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="standings"
          label=""
          position={[MENU_COLUMNS.lowRightOuter, MENU_LANDMARK.lowerBandY, -3.32]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <StandingsDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
              standingsRank={standingsRank}
              record={record}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="stats"
          label=""
          position={[MENU_COLUMNS.right, MENU_LANDMARK.upperBandY, -3.3]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <StatsDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
              record={record}
            />
          )}
        </InteractiveGroup>

        {/* Trophy shrine at credenza height — a lit plinth the player can walk
            their eye down to, rather than a plaque lost on the side wall. */}
        <InteractiveGroup
          id="awards"
          label=""
          position={[MENU_COLUMNS.lowLeftInner, MENU_LANDMARK.lowerBandY, -3.26]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hoverScale={1}
          hoverLift={0}
          hitBoxArgs={MENU_LANDMARK.hitBox}
          hitBoxPosition={MENU_LANDMARK.hitBoxOffset}
          activateOnPointerDown
          hideHoverLabel
          showHoverCorners={false}
        >
          {(hovered) => (
            <LegacyDiorama
              hovered={hovered}
              prefersReducedMotion={prefersReducedMotion}
            />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="draft"
          label="Draft War Room"
          description="Enter the draft floor"
          position={[-4.28, 1.55, 0.45]}
          rotation={[0, Math.PI / 2, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.draft}
        >
          {(hovered) => (
            <PhysicalDraftBoard hovered={hovered} draftWeek={mood.isDraftWeek} />
          )}
        </InteractiveGroup>

        <InteractiveGroup
          id="scouting"
          label="Scouting Station"
          description="Prospects · reports · watchlist"
          position={[-4.24, 1.62, -2.35]}
          rotation={[0, Math.PI / 2 - 0.42, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={OFFICE_HITBOXES.scouting}
        >
          {(hovered) => <ScoutingStation hovered={hovered} />}
        </InteractiveGroup>

        <InteractiveGroup
          id="gameDay"
          label="Game Day"
          description="Preview · simulate · matchup"
          position={[4.02, 0.95, 1.45]}
          rotation={[0, -Math.PI / 2 + 0.26, 0]}
          hoveredId={hoveredId}
          setHoveredId={setHoveredId}
          onOpen={handleOpenPanel}
          hitBoxArgs={[1.35, 0.72, 0.32]}
          hitBoxPosition={[0, 0, 0.1]}
        >
          {(hovered) => (
            <ArenaWindowObject
              hovered={hovered}
              nextGame={nextGame}
              seasonYear={seasonYear}
            />
          )}
        </InteractiveGroup>
  
        <ContactShadows
          position={[0, 0.012, 0]}
          opacity={0.48}
          scale={8}
          blur={3.2}
          far={4.2}
          color="#1a1612"
        />
      </>
    );
  }
  
  function OfficeHud({
    teamName,
    teamLogo,
    currentDate,
    record,
    capSpace,
    nextGame,
    officeMood,
    urgentItems = [],
    onUrgentSelect,
    onReset,
    onQuickMenu,
    onExitOffice,
    onOpenStation,
    lowPowerMode = false,
    onToggleLowPower,
  }) {
    const mood = officeMood || {};
    const urgent = officeSafeArray(urgentItems);
    const [briefingOpen, setBriefingOpen] = useState(false);
    const [tourOpen, setTourOpen] = useState(() => {
      try {
        return localStorage.getItem("nhl-office-tour-v1") !== "1";
      } catch (err) {
        return true;
      }
    });
    const phaseLabel = safeText(
      mood.seasonPhase || mood.officeMode || "",
      ""
    ).replace(/_/g, " ");
    const topStory = urgent[0];

    const dismissTour = () => {
      setTourOpen(false);
      try {
        localStorage.setItem("nhl-office-tour-v1", "1");
      } catch (err) {
        /* ignore */
      }
    };

    return (
      <div className="office-hud office-hud--cinematic">
        <div className="office-reticle" aria-hidden="true" />

        <div className="office-broadcast">
          <span className="office-broadcast__where">GM office</span>
          <TeamLogoBadge teamLogo={teamLogo} teamName={teamName} size={28} variant="badge" />
          <strong>{teamName}</strong>
          <span>{safeText(currentDate, "Today")}</span>
          {phaseLabel ? <em>{phaseLabel}</em> : null}
          <span>Record {safeText(record, "0-0-0")}</span>
          <span>{formatMoney(capSpace)} available</span>
          {nextGame ? <span>Next {safeText(nextGame, "")}</span> : null}
        </div>

        {tourOpen ? (
          <div className="office-tour">
            <p>
              Hover a brass corner to see what a station does. Click to enter.
              Drag to look around. R returns home. M opens the directory.
            </p>
            <button type="button" onClick={dismissTour}>
              Got it
            </button>
          </div>
        ) : null}

        {/* Only the war room keeps a screen-edge shortcut: it is a doorway on
            the side wall. Draft Class and Legacy Wall are now readable
            landmarks in the room, so duplicating them here just put floating
            menu chips back over the composition. */}
        <button
          type="button"
          className="office-edge office-edge--left"
          onClick={() => onOpenStation?.(OFFICE_PANEL_IDS.DRAFT)}
        >
          Draft war room
        </button>

        <button
          type="button"
          className={`office-urgent-desk office-urgent-desk--compact${briefingOpen ? " is-open" : ""}`}
          onClick={() => setBriefingOpen((open) => !open)}
        >
          <span>
            {topStory
              ? topStory.title
              : urgent.length
              ? `${urgent.length} open items`
              : "Desk is clear"}
          </span>
          {briefingOpen ? (
            urgent.length ? (
              <ul className="office-urgent-desk__list">
                {urgent.slice(0, 3).map((item) => (
                  <li
                    key={item.id}
                    className={`office-urgent-desk__item office-urgent-desk__item--${item.severity || "low"}`}
                  >
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onUrgentSelect?.(item.target, item);
                      }}
                    >
                      <strong>{item.title}</strong>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="office-urgent-desk__empty">No open storylines.</p>
            )
          ) : null}
        </button>

        <div className="office-control-bar">
          <p className="office-control-bar__hint">Drag to look · Esc back · R home · M directory</p>
          <button type="button" className="office-control-bar__primary" onClick={onQuickMenu}>
            Directory
          </button>
          {onExitOffice ? (
            <button type="button" className="office-control-bar__utility" onClick={onExitOffice}>
              Leave office
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  function OfficePanel({
    activePanel,
    teamName,
    teamLogo,
    record,
    capSpace,
    nextGame,
    standingsRank,
    onClose,
    onNavigate,
    panelCopy,
    briefingNote,
    urgentItems = [],
  }) {
    const panel = panelCopy || (activePanel ? PANEL_CONTENT[activePanel] : null);

    if (!panel) return null;

    const panelUrgent = officeSafeArray(urgentItems).filter((item) => {
      if (activePanel === OFFICE_PANEL_IDS.MESSAGES) {
        return item.type === "messages" || item.type === "trade";
      }
      if (activePanel === OFFICE_PANEL_IDS.CONTRACTS) return item.type === "contracts";
      if (activePanel === OFFICE_PANEL_IDS.SCOUTING || activePanel === OFFICE_PANEL_IDS.DRAFT) {
        return item.type === "draft";
      }
      if (activePanel === OFFICE_PANEL_IDS.LINES) return item.type === "injuries";
      if (activePanel === OFFICE_PANEL_IDS.TASKS) return item.type === "tasks";
      return false;
    });

    return (
      <AnimatePresence>
        <motion.aside
          className="office-panel"
          initial={{ opacity: 0, x: 80, scale: 0.96 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: 80, scale: 0.96 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
        >
          <TeamLogoBadge
            className="office-panel-watermark"
            teamLogo={teamLogo}
            teamName={teamName}
            size={200}
            variant="watermark"
            opacity={0.1}
          />

          <div className="office-panel-header">
            <TeamLogoBadge
              teamLogo={teamLogo}
              teamName={teamName}
              size={72}
              variant="framed"
            />

            <div>
              <span>{panel.eyebrow}</span>
              <h2>{panel.title}</h2>
            </div>

            <button type="button" className="office-panel-close" onClick={onClose}>
              ×
            </button>
          </div>

          {briefingNote ? (
            <p className="office-panel-briefing">{briefingNote}</p>
          ) : null}

          <p>{panel.description}</p>

          {panel.staffNote ? (
            <div className="office-panel-staff-note">
              <span>{panel.staffRole || "Staff Note"}</span>
              <p>{panel.staffNote}</p>
            </div>
          ) : null}

          {panel.pressureLine ? (
            <p className="office-panel-pressure">
              <strong>If ignored:</strong> {panel.pressureLine}
            </p>
          ) : null}

          {panelUrgent.length ? (
            <div className="office-panel-urgent">
              <span>Desk Priority</span>
              <ul>
                {panelUrgent.slice(0, 3).map((item) => (
                  <li key={item.id}>
                    <button type="button" onClick={() => onNavigate(item.target)}>
                      {item.title}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="office-panel-stats">
            <div>
              <span>Record</span>
              <strong>{safeText(record, "0-0-0")}</strong>
            </div>

            <div>
              <span>Cap</span>
              <strong>{formatMoney(capSpace)}</strong>
            </div>

            <div>
              <span>Standing</span>
              <strong>{safeText(standingsRank, "—")}</strong>
            </div>
          </div>

          <div className="office-panel-next">
            <span>Next Game</span>
            <strong>{safeText(nextGame, "No game listed")}</strong>
          </div>

          <div className="office-panel-actions">
            {panel.actions.map(([label, target]) => (
              <button key={target} type="button" onClick={() => onNavigate(target)}>
                {label}
              </button>
            ))}
          </div>
        </motion.aside>
      </AnimatePresence>
    );
  }

  function QuickMenu({
    open,
    onClose,
    onNavigate,
    onSimNextGame,
    simDisabled = false,
    menuItems = QUICK_MENU,
  }) {
    const grouped = useMemo(() => {
      const buckets = {};
      officeSafeArray(menuItems).forEach((item) => {
        const groupId = item.group || "primary";
        if (!buckets[groupId]) buckets[groupId] = [];
        buckets[groupId].push(item);
      });
      return buckets;
    }, [menuItems]);

    const groupOrder = ["primary", "operations", "frontOffice", "future"];

    return (
      <AnimatePresence>
        {open ? (
          <motion.div
            className="office-quick-menu"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="office-quick-menu-card"
              initial={{ y: 32, scale: 0.98 }}
              animate={{ y: 0, scale: 1 }}
              exit={{ y: 32, scale: 0.98 }}
            >
              <div className="office-quick-menu-head">
                <div>
                  <span className="office-quick-menu-kicker">Desk Terminal // CMD</span>
                  <h3>Operations Directory</h3>
                </div>

                <button type="button" className="office-quick-menu-close" onClick={onClose} aria-label="Close">
                  ×
                </button>
              </div>

              {onSimNextGame ? (
                <div className="office-quick-menu-sim">
                  <button
                    type="button"
                    className="office-quick-menu-sim-btn"
                    disabled={simDisabled}
                    onClick={() => {
                      onSimNextGame();
                      onClose();
                    }}
                  >
                    <strong>Sim Next Game</strong>
                    <small>Advance franchise to the next scheduled game</small>
                  </button>
                </div>
              ) : null}

              {groupOrder.map((groupId) => {
                const items = grouped[groupId];
                if (!items?.length) return null;
                const meta = FRANCHISE_COMMAND_GROUPS[groupId] || { label: groupId };

                return (
                  <section key={groupId} className={`office-quick-menu-section office-quick-menu-section--${groupId}`}>
                    <header className="office-quick-menu-section-head">
                      <span>{meta.label}</span>
                    </header>

                    <div className="office-quick-menu-grid">
                      {items.map((item) => {
                        const isPlaceholder = item.type === "placeholder";
                        const isHub = item.type === "hub";

                        return (
                          <button
                            type="button"
                            key={item.id}
                            className={[
                              "office-quick-menu-item",
                              item.highlight ? "is-highlight" : "",
                              isPlaceholder ? "is-placeholder" : "",
                              item.enabled === false ? "is-disabled" : "",
                            ]
                              .filter(Boolean)
                              .join(" ")}
                            onClick={() => {
                              if (item.enabled === false) return;
                              onNavigate?.(item.target);
                              onClose();
                            }}
                          >
                            <span className="office-quick-menu-item-eyebrow">
                              {item.eyebrow}
                              {item.badge ? ` • ${item.badge}` : ""}
                              {isPlaceholder ? " • Soon" : ""}
                            </span>
                            <strong>{item.label}</strong>
                            <small>{item.description}</small>
                            {isHub ? <em className="office-quick-menu-item-note">Stay in office</em> : null}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                );
              })}
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    );
  }
  
  function WebGLFallback({
    onNavigate,
    teamName,
    teamLogo,
    phase,
    record,
    capSpace,
    urgentItems = [],
    menuItems = QUICK_MENU,
    onSimNextGame,
    simDisabled = false,
  }) {
    const urgent = officeSafeArray(urgentItems);
    const grouped = useMemo(() => {
      const buckets = {};
      officeSafeArray(menuItems).forEach((item) => {
        const groupId = item.group || "primary";
        if (!buckets[groupId]) buckets[groupId] = [];
        buckets[groupId].push(item);
      });
      return buckets;
    }, [menuItems]);
    const groupOrder = ["primary", "operations", "frontOffice", "future"];

    return (
      <div className="office-fallback">
        <div className="office-fallback-hero">
          <TeamLogoBadge
            teamLogo={teamLogo}
            teamName={teamName}
            size={72}
            variant="framed"
          />
          <div>
            <span>Executive Office Fallback</span>
            <h2>{safeText(teamName, "Franchise Club")}</h2>
            <p>
              WebGL could not load, but your command center is still online. Phase{" "}
              {safeText(phase, "—")} • Record {safeText(record, "0-0-0")} • Cap{" "}
              {formatMoney(capSpace)}
            </p>
          </div>
        </div>

        <div className="office-fallback-briefing">
          <span>Executive Briefing</span>
          {urgent.length ? (
            <ul className="office-urgent-desk__list">
              {urgent.slice(0, 6).map((item) => (
                <li
                  key={item.id}
                  className={`office-urgent-desk__item office-urgent-desk__item--${item.severity || "low"}`}
                >
                  <button
                    type="button"
                    onClick={() => onNavigate?.(item.target)}
                  >
                    <strong>{item.title}</strong>
                    <small>{item.detail}</small>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="office-urgent-desk__empty">
              No urgent fires on the desk. That either means you are doing well, or
              the league is waiting to ruin your week.
            </p>
          )}
        </div>

        {onSimNextGame ? (
          <div className="office-quick-menu-sim office-fallback-sim">
            <button
              type="button"
              className="office-quick-menu-sim-btn"
              disabled={simDisabled}
              onClick={onSimNextGame}
            >
              <strong>Sim Next Game</strong>
              <small>Advance franchise to the next scheduled game</small>
            </button>
          </div>
        ) : null}

        {groupOrder.map((groupId) => {
          const items = grouped[groupId];
          if (!items?.length) return null;
          const meta = FRANCHISE_COMMAND_GROUPS[groupId] || { label: groupId };
          return (
            <section key={groupId} className="office-fallback-section">
              <header className="office-quick-menu-section-head">
                <span>{meta.label}</span>
              </header>
              <div className="office-fallback-grid">
                {items.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={[
                      item.highlight ? "is-highlight" : "",
                      item.type === "placeholder" ? "is-placeholder" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    onClick={() => onNavigate?.(item.target)}
                  >
                    <span>
                      {item.eyebrow}
                      {item.badge ? ` • ${item.badge}` : ""}
                    </span>
                    <strong>{item.label}</strong>
                    <small>{item.description}</small>
                  </button>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    );
  }
  
  class OfficeErrorBoundary extends React.Component {
    constructor(props) {
      super(props);
      this.state = { hasError: false };
    }
  
    static getDerivedStateFromError() {
      return { hasError: true };
    }
  
    componentDidCatch(error) {
      console.error("Office scene crashed:", error);
    }
  
    render() {
      if (this.state.hasError) {
        return (
          <WebGLFallback
            onNavigate={this.props.onNavigate}
            teamName={this.props.teamName}
            teamLogo={this.props.teamLogo}
            phase={this.props.phase}
            record={this.props.record}
            capSpace={this.props.capSpace}
            urgentItems={this.props.urgentItems}
            menuItems={this.props.menuItems}
            onSimNextGame={this.props.onSimNextGame}
            simDisabled={this.props.simDisabled}
          />
        );
      }
  
      return this.props.children;
    }
  }
  
  export default function FirstPersonOfficeHub({
    teamName = "Franchise Club",
    teamLogo = "",
    seasonYear = "Season",
    currentDate = "Today",
    record = "0-0-0",
    capSpace = "—",
    capSpaceMillions = null,
    nextGame = "No game listed",
    standingsRank = "Standings",
    unreadMessages = 0,
    pendingTasks = 0,
    activeStorylines = 0,
    franchiseState = null,
    team = null,
    officeMood: officeMoodProp = null,
    urgentItems: urgentItemsProp = null,
    officeSummary = null,
    panelRequest = null,
    onNavigate,
    onOpenPanel,
    onExitOffice,
    onPanelRequestHandled,
    onSimNextGame,
    simDisabled = false,
    players = [],
  }) {
    const [hoveredId, setHoveredId] = useState(null);
    const [activePanel, setActivePanel] = useState(null);
    const [showQuickMenu, setShowQuickMenu] = useState(false);
    const [resetToken, setResetToken] = useState(0);
    const [briefingNote, setBriefingNote] = useState("");
    const canvasHostRef = useRef(null);
    const [canvasReady, setCanvasReady] = useState(false);
    const [lowPowerMode, setLowPowerMode] = useState(() => {
      try {
        return localStorage.getItem(LOW_POWER_STORAGE_KEY) === "1";
      } catch (err) {
        return false;
      }
    });
    const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
    const [officeWeather, setOfficeWeather] = useState(() =>
      deriveSeasonalWeather(currentDate)
    );

    const bestPlayer = useMemo(() => getBestPlayer(players), [players]);
  
    const officePictures = useMemo(() => getOfficePictures(), []);
    const normalizedRecord = useMemo(() => formatRecord(record), [record]);

    useEffect(() => {
      const seasonal = deriveSeasonalWeather(currentDate);
      setOfficeWeather(seasonal);

      if (!GOOGLE_WEATHER_API_KEY) return undefined;

      const { year, month, day } = parseFranchiseDateParts(currentDate);
      const simStamp = year > 0 ? Date.UTC(year, month - 1, day) : NaN;
      const now = Date.now();
      const nearLive =
        Number.isFinite(simStamp) && Math.abs(now - simStamp) < 1000 * 60 * 60 * 24 * 12;

      if (!nearLive) return undefined;

      let cancelled = false;
      const lat = Number(team?.latitude || team?.lat || 45.4215);
      const lon = Number(team?.longitude || team?.lng || -75.6972);
      const url =
        `https://weather.googleapis.com/v1/currentConditions:lookup` +
        `?key=${encodeURIComponent(GOOGLE_WEATHER_API_KEY)}` +
        `&location.latitude=${lat}&location.longitude=${lon}`;

      fetch(url)
        .then((res) => (res.ok ? res.json() : null))
        .then((payload) => {
          if (cancelled || !payload) return;
          setOfficeWeather({
            ...seasonal,
            ...mapGoogleWeatherCondition(payload),
            source: "google",
          });
        })
        .catch(() => {
          /* keep seasonal */
        });

      return () => {
        cancelled = true;
      };
    }, [currentDate, team]);
    const effectiveTeamLogo = useMemo(() => {
      const fromName = resolveFranchiseTeamLogo(
        { name: teamName, team_name: teamName },
        teamName
      );
      if (fromName) return fromName;
      return toLogoUrl(teamLogo);
    }, [teamLogo, teamName]);

    const summary = useMemo(
      () => ({
        ...(officeSummary || {}),
        unreadMessages,
        pendingTasks,
        activeStorylines,
        nextGame,
        record: normalizedRecord,
        capSpace,
        capSpaceMillions:
          Number.isFinite(Number(capSpaceMillions))
            ? Number(capSpaceMillions)
            : Number.isFinite(Number(officeSummary?.capSpaceMillions))
              ? Number(officeSummary.capSpaceMillions)
              : null,
        capSpaceRaw:
          Number.isFinite(Number(capSpaceMillions))
            ? Number(capSpaceMillions)
            : team?.cap_space ??
              team?.capSpace ??
              franchiseState?.cap_space,
      }),
      [
        unreadMessages,
        pendingTasks,
        activeStorylines,
        nextGame,
        normalizedRecord,
        capSpace,
        capSpaceMillions,
        team,
        franchiseState,
        officeSummary,
      ]
    );

    const officeMood = useMemo(
      () =>
        officeMoodProp ||
        deriveOfficeMood(franchiseState, team, summary),
      [officeMoodProp, franchiseState, team, summary]
    );

    const urgentItems = useMemo(
      () =>
        urgentItemsProp ||
        buildOfficeUrgentItems(franchiseState, team, summary),
      [urgentItemsProp, franchiseState, team, summary]
    );

    const contextualCommandRegistry = useMemo(
      () => getContextualCommandRegistry(FRANCHISE_COMMAND_REGISTRY, officeMood, urgentItems),
      [officeMood, urgentItems]
    );

    const activePanelCopy = useMemo(() => {
      if (!activePanel) return null;
      return getDynamicPanelCopy(
        activePanel,
        PANEL_CONTENT[activePanel],
        franchiseState,
        team,
        officeMood,
        urgentItems,
        summary
      );
    }, [activePanel, franchiseState, team, officeMood, urgentItems, summary]);

    const championshipCount = useMemo(() => {
      const cups =
        franchiseState?.championships ??
        franchiseState?.stanley_cups ??
        team?.championships ??
        team?.stanley_cups;
      if (Array.isArray(cups)) return cups.length;
      return officeSafeNumber(cups, 0);
    }, [franchiseState, team]);

    const franchisePulse = useMemo(
      () => franchiseState?.franchise_pulse || null,
      [franchiseState]
    );

    const capPressure = useMemo(() => {
      const raw =
        summary.capSpaceMillions ??
        team?.cap_space ??
        team?.capSpace ??
        franchiseState?.cap_space ??
        summary.capSpaceRaw;
      const millions = officeCapMillions(raw);
      return Number.isFinite(millions) && millions < 2.0;
    }, [team, franchiseState, summary]);

    const webglSupported = useMemo(() => detectWebGLSupport(), []);
  
    const handleOpenPanel = useCallback(
      (panelId) => {
        const commandTarget = PANEL_TO_COMMAND_TARGET[panelId];
        if (commandTarget && onNavigate) {
          onNavigate(commandTarget);
          return;
        }

        if (!PANEL_CONTENT[panelId]) {
          console.warn("[OfficeNav] Missing panel:", panelId);
          if (onNavigate) onNavigate(panelId);
          return;
        }

        setActivePanel(panelId);

        if (onOpenPanel) {
          onOpenPanel(panelId);
        }
      },
      [onNavigate, onOpenPanel]
    );
  
    const handleNavigate = useCallback(
      (target) => {
        if (onNavigate) {
          onNavigate(target);
        } else {
          console.log("Navigate:", target);
        }
        setActivePanel(null);
        setBriefingNote("");
      },
      [onNavigate]
    );

    const handleToggleLowPower = useCallback(() => {
      setLowPowerMode((prev) => {
        const next = !prev;
        try {
          localStorage.setItem(LOW_POWER_STORAGE_KEY, next ? "1" : "0");
        } catch (err) {
          /* ignore storage failures */
        }
        return next;
      });
    }, []);

    useEffect(() => {
      if (!panelRequest?.panelId) return;
      handleOpenPanel(panelRequest.panelId);
      if (panelRequest.note) {
        setBriefingNote(panelRequest.note);
      }
      onPanelRequestHandled?.();
    }, [panelRequest, handleOpenPanel, onPanelRequestHandled]);

    useEffect(() => {
      if (typeof window === "undefined" || !window.matchMedia) return undefined;
      const media = window.matchMedia("(prefers-reduced-motion: reduce)");
      const apply = () => setPrefersReducedMotion(Boolean(media.matches));
      apply();
      media.addEventListener?.("change", apply);
      return () => media.removeEventListener?.("change", apply);
    }, []);

    useEffect(() => {
      let active = true;

      const enableCanvas = () => {
        if (active && canvasHostRef.current && webglSupported) {
          setCanvasReady(true);
        }
      };

      // Defer mount until the host div exists — avoids R3F connect() hitting null under StrictMode.
      const frameId = window.requestAnimationFrame(enableCanvas);

      return () => {
        active = false;
        window.cancelAnimationFrame(frameId);
        setCanvasReady(false);
      };
    }, [webglSupported]);
  
    useEffect(() => {
      validateOfficeNavigation();
    }, []);
  
    useEffect(() => {
      return () => document.body.classList.remove("office-cursor-active");
    }, []);
  
    useEffect(() => {
      const onKeyDown = (e) => {
        const key = e.key.toLowerCase();
  
        if (key === "escape") {
          if (showQuickMenu) {
            setShowQuickMenu(false);
            return;
          }
          setActivePanel(null);
          setBriefingNote("");
        }
  
        if (key === "m") {
          setShowQuickMenu((open) => !open);
        }
  
        if (key === "r") {
          setResetToken((v) => v + 1);
        }
      };
  
      window.addEventListener("keydown", onKeyDown);
      return () => window.removeEventListener("keydown", onKeyDown);
    }, [showQuickMenu]);
  
    const showFallback = !webglSupported;
    const effectiveLowPower = lowPowerMode || prefersReducedMotion;
    const phaseLabelText = officePhaseText(franchiseState) || seasonYear;

    if (showFallback) {
      return (
        <section
          className="office-hub office-hub--fallback register-office"
          data-register="office"
          data-office-mode={officeMood.officeMode}
          data-season-phase={officeMood.seasonPhase}
          data-pressure={officeMood.pressureLevel}
        >
          <WebGLFallback
            onNavigate={handleNavigate}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            phase={phaseLabelText}
            record={normalizedRecord}
            capSpace={capSpace}
            urgentItems={urgentItems}
            menuItems={contextualCommandRegistry}
            onSimNextGame={onSimNextGame}
            simDisabled={simDisabled}
          />
          <OfficePanel
            activePanel={activePanel}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            record={normalizedRecord}
            capSpace={capSpace}
            nextGame={nextGame}
            standingsRank={standingsRank}
            panelCopy={activePanelCopy}
            briefingNote={briefingNote}
            urgentItems={urgentItems}
            onClose={() => {
              setActivePanel(null);
              setBriefingNote("");
            }}
            onNavigate={handleNavigate}
          />
        </section>
      );
    }

    return (
      <section
        className={`office-hub register-office ${effectiveLowPower ? "office-hub--low-power" : ""}`}
        data-register="office"
        data-hovered={hoveredId || "none"}
        data-office-mode={officeMood.officeMode}
        data-season-phase={officeMood.seasonPhase}
        data-pressure={officeMood.pressureLevel}
        data-team-form={officeMood.teamForm}
        data-deadline={officeMood.isTradeDeadline ? "true" : "false"}
        data-draft-week={officeMood.isDraftWeek ? "true" : "false"}
        data-free-agency={officeMood.isFreeAgency ? "true" : "false"}
        data-playoffs={officeMood.isPlayoffs ? "true" : "false"}
        data-offseason={officeMood.isOffseason ? "true" : "false"}
        data-injury-crisis={officeMood.hasInjuryCrisis ? "true" : "false"}
        data-owner-pressure={officeMood.hasOwnerPressure ? "true" : "false"}
      >
        <div className="office-canvas" ref={canvasHostRef}>
          <OfficeErrorBoundary
            onOpenPanel={handleOpenPanel}
            onNavigate={handleNavigate}
            teamName={teamName}
            teamLogo={effectiveTeamLogo}
            phase={phaseLabelText}
            record={normalizedRecord}
            capSpace={capSpace}
            urgentItems={urgentItems}
            menuItems={contextualCommandRegistry}
            onSimNextGame={onSimNextGame}
            simDisabled={simDisabled}
          >
            {canvasReady ? (
            <Canvas
              shadows={!effectiveLowPower}
              dpr={effectiveLowPower ? [1, 1] : [1, 2]}
              eventSource={canvasHostRef}
              camera={{
                position: OFFICE_CAMERA.position,
                fov: OFFICE_CAMERA.fov,
              }}
              gl={{
                antialias: !effectiveLowPower,
                powerPreference: effectiveLowPower ? "default" : "high-performance",
              }}
              onPointerMissed={() => {
                setHoveredId(null);
                document.body.classList.remove("office-cursor-active");
              }}
              onCreated={({ gl }) => {
                gl.toneMapping = THREE.ACESFilmicToneMapping;
                gl.toneMappingExposure = 1.14;
                gl.outputColorSpace = THREE.SRGBColorSpace;
              }}
            >
              <Suspense fallback={null}>
                <OfficeScene
                  teamName={teamName}
                  teamLogo={effectiveTeamLogo}
                  seasonYear={seasonYear}
                  currentDate={currentDate}
                  record={normalizedRecord}
                  capSpace={capSpace}
                  nextGame={nextGame}
                  standingsRank={standingsRank}
                  unreadMessages={unreadMessages}
                  pendingTasks={pendingTasks}
                  activeStorylines={activeStorylines}
                  hoveredId={hoveredId}
                  setHoveredId={setHoveredId}
                  handleOpenPanel={handleOpenPanel}
                  resetToken={resetToken}
                  officePictures={officePictures}
                  bestPlayer={bestPlayer}
                  officeMood={officeMood}
                  activePanel={activePanel}
                  lowPowerMode={effectiveLowPower}
                  prefersReducedMotion={prefersReducedMotion}
                  championshipCount={championshipCount}
                  capPressure={capPressure}
                  officeWeather={officeWeather}
                />
  
                {!effectiveLowPower ? (
                  <EffectComposer enableNormalPass={false} multisampling={2}>
                    <Bloom
                      intensity={0.06}
                      luminanceThreshold={0.78}
                      luminanceSmoothing={0.88}
                    />
                    <Vignette eskil={false} offset={0.42} darkness={0.18} />
                  </EffectComposer>
                ) : (
                  <EffectComposer multisampling={0}>
                    <Vignette eskil={false} offset={0.42} darkness={0.14} />
                  </EffectComposer>
                )}
              </Suspense>
            </Canvas>
            ) : null}
          </OfficeErrorBoundary>
        </div>
  
        <div className="office-vignette" aria-hidden="true" />
  
        <OfficeHud
          teamName={teamName}
          teamLogo={effectiveTeamLogo}
          seasonYear={seasonYear}
          currentDate={currentDate}
          record={normalizedRecord}
          capSpace={capSpace}
          nextGame={nextGame}
          standingsRank={standingsRank}
          officeMood={officeMood}
          franchisePulse={franchisePulse}
          urgentItems={urgentItems}
          onUrgentSelect={handleNavigate}
          lowPowerMode={effectiveLowPower}
          prefersReducedMotion={prefersReducedMotion}
          onToggleLowPower={handleToggleLowPower}
          onReset={() => setResetToken((v) => v + 1)}
          onQuickMenu={() => setShowQuickMenu(true)}
          onOpenStation={handleOpenPanel}
          onExitOffice={onExitOffice}
        />

        <OfficePanel
          activePanel={activePanel}
          teamName={teamName}
          teamLogo={effectiveTeamLogo}
          record={normalizedRecord}
          capSpace={capSpace}
          nextGame={nextGame}
          standingsRank={standingsRank}
          panelCopy={activePanelCopy}
          briefingNote={briefingNote}
          urgentItems={urgentItems}
          onClose={() => {
            setActivePanel(null);
            setBriefingNote("");
          }}
          onNavigate={handleNavigate}
        />

        <QuickMenu
          open={showQuickMenu}
          onClose={() => setShowQuickMenu(false)}
          onNavigate={handleNavigate}
          onSimNextGame={onSimNextGame}
          simDisabled={simDisabled}
          menuItems={contextualCommandRegistry}
        />
      </section>
    );
  }