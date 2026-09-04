/**
 * Deterministic CSS headshot metadata (mirrors SimEngine player_headshots.py).
 * Used when API rows omit avatar_seed / headshot_id (older saves, partial payloads).
 */

const HEADSHOT_MIN = 1;
const HEADSHOT_MAX = 60;

const SKIN_TONES = ["fair", "light", "medium", "olive", "tan", "brown", "deep"];
const HAIR_STYLES = [
  "swept",
  "crop",
  "buzz",
  "curly",
  "red_part",
  "bald",
  "flow",
  "spiky",
  "flat_top",
  "afro",
  "fade",
  "long_blond",
  "messy_dark",
  "grey",
  "helmet",
];
const HAIR_COLORS = ["black", "dark_brown", "brown", "auburn", "red", "blond", "dirty_blond", "grey", "white"];
const FACIAL_HAIR = ["none", "stubble", "beard", "playoff_beard", "mustache", "goatee", "grey_beard"];
const EXPRESSIONS = ["neutral", "smile", "serious", "focused", "confident", "tired", "angry"];

const GOALIE_HEADSHOTS = [10, 21, 22, 36, 37, 38, 52, 55];
const VETERAN_HEADSHOTS = [8, 9, 15, 17, 18, 29, 33, 41, 48];
const ROOKIE_HEADSHOTS = [2, 3, 16, 19, 23, 27, 31, 44, 49];
const PROSPECT_HEADSHOTS = [16, 27, 31, 44, 49, 50];

const K = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

function rotr(n, x) {
  return (x >>> n) | (x << (32 - n));
}

function sha256Hex(input) {
  const msg = new TextEncoder().encode(String(input));
  const bitLen = msg.length * 8;
  const padLen = ((msg.length + 9 + 63) & ~63) - msg.length;
  const padded = new Uint8Array(msg.length + padLen);
  padded.set(msg);
  padded[msg.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 4, bitLen, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let i = 0; i < padded.length; i += 64) {
    for (let t = 0; t < 16; t += 1) {
      w[t] = view.getUint32(i + t * 4, false);
    }
    for (let t = 16; t < 64; t += 1) {
      const s0 = rotr(7, w[t - 15]) ^ rotr(18, w[t - 15]) ^ (w[t - 15] >>> 3);
      const s1 = rotr(17, w[t - 2]) ^ rotr(19, w[t - 2]) ^ (w[t - 2] >>> 10);
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let t = 0; t < 64; t += 1) {
      const S1 = rotr(6, e) ^ rotr(11, e) ^ rotr(25, e);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + K[t] + w[t]) >>> 0;
      const S0 = rotr(2, a) ^ rotr(13, a) ^ rotr(22, a);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((v) => v.toString(16).padStart(8, "0"))
    .join("");
}

function pick(options, seed, salt = "") {
  if (!options?.length) return "";
  const digest = sha256Hex(`${seed}|${salt}`);
  const idx = parseInt(digest.slice(0, 8), 16) % options.length;
  return options[idx];
}

export function deriveAvatarSeed({
  player_id = "",
  full_name = "",
  birth_year = 0,
  age = 0,
  nationality = "",
  position = "",
  shoots = "",
  team_id = "",
} = {}) {
  const material = [
    String(player_id || "").trim(),
    String(full_name || "").trim().toLowerCase(),
    String(Number(birth_year || 0)),
    String(Number(age || 0)),
    String(nationality || "").trim().toLowerCase(),
    String(position || "").trim().toUpperCase(),
    String(shoots || "").trim().toUpperCase(),
    String(team_id || "").trim(),
  ].join("|");

  const digest = sha256Hex(material);
  let seed = parseInt(digest.slice(0, 12), 16);
  if (!Number.isFinite(seed) || seed <= 0) seed = 1;
  return seed;
}

export function ageBucket(age) {
  const a = Number(age || 0);
  if (a <= 19) return "prospect";
  if (a <= 22) return "rookie";
  if (a <= 30) return "prime";
  if (a <= 36) return "veteran";
  return "legend";
}

export function nationalityCode(nationality = "") {
  const raw = String(nationality || "").trim().toUpperCase();
  const mapping = {
    CANADA: "CAN",
    CAN: "CAN",
    "UNITED STATES": "USA",
    USA: "USA",
    US: "USA",
    SWEDEN: "SWE",
    SWE: "SWE",
    FINLAND: "FIN",
    FIN: "FIN",
    "CZECH REPUBLIC": "CZE",
    CZECHIA: "CZE",
    CZE: "CZE",
    SLOVAKIA: "SVK",
    SVK: "SVK",
    RUSSIA: "RUS",
    RUS: "RUS",
    GERMANY: "GER",
    GER: "GER",
    SWITZERLAND: "SUI",
    SUI: "SUI",
    AUSTRIA: "AUT",
    AUT: "AUT",
    NORWAY: "NOR",
    NOR: "NOR",
    DENMARK: "DEN",
    DEN: "DEN",
    LATVIA: "LAT",
    LAT: "LAT",
    BELARUS: "BLR",
    BLR: "BLR",
    UKRAINE: "UKR",
    UKR: "UKR",
    FRANCE: "FRA",
    FRA: "FRA",
  };

  if (mapping[raw]) return mapping[raw];
  for (const [key, code] of Object.entries(mapping)) {
    if (raw.includes(key) || key.includes(raw)) return code;
  }
  if (raw.length >= 3) return raw.slice(0, 3);
  return raw || "NHL";
}

function headshotIdForProfile(seed, { age, position, facial_hair }) {
  const pos = String(position || "C").trim().toUpperCase();
  const bucket = ageBucket(age);

  let pool;
  if (pos === "G") {
    pool = GOALIE_HEADSHOTS;
  } else if (bucket === "prospect") {
    pool = PROSPECT_HEADSHOTS;
  } else if (bucket === "rookie") {
    pool = ROOKIE_HEADSHOTS;
  } else if (
    bucket === "veteran" ||
    bucket === "legend" ||
    ["beard", "playoff_beard", "grey_beard", "mustache"].includes(facial_hair)
  ) {
    pool = VETERAN_HEADSHOTS;
  } else {
    pool = Array.from({ length: HEADSHOT_MAX - HEADSHOT_MIN + 1 }, (_, i) => i + HEADSHOT_MIN);
  }

  return pool[seed % pool.length];
}

export function generatePlayerHeadshotMetadata(player = {}) {
  const playerId = String(player.id || player.player_id || player.key || "");
  const fullName = String(
    player.name ||
      player.full_name ||
      [player.firstName, player.lastName].filter(Boolean).join(" ") ||
      ""
  );
  const age = Number(player.age || 20);
  const birthYear = Number(player.birth_year || player.birthYear || 0);
  const nationality = String(player.nationality || player.country || player.nat || player.birth_country || "");
  const position = String(player.position || player.pos || "C");
  const shoots = String(player.shoots || player.hand || player.handedness || "L");
  const teamId = String(player.team_id || player.current_team_id || "");

  const seed =
    player.avatar_seed != null && Number(player.avatar_seed) > 0
      ? Number(player.avatar_seed)
      : deriveAvatarSeed({
          player_id: playerId,
          full_name: fullName,
          birth_year: birthYear,
          age,
          nationality,
          position,
          shoots,
          team_id: teamId,
        });

  const bucket = ageBucket(age);
  const skin_tone = pick(SKIN_TONES, seed, "skin");
  const hair_style = pick(HAIR_STYLES, seed, "hair_style");
  const hair_color = pick(HAIR_COLORS, seed, "hair_color");
  let facial_hair = pick(FACIAL_HAIR, seed, "facial_hair");

  if (bucket === "veteran" || bucket === "legend") {
    if (facial_hair === "none" && seed % 5 < 3) {
      facial_hair = bucket === "legend" ? "grey_beard" : "beard";
    }
  }
  if (bucket === "prospect" && facial_hair !== "none" && facial_hair !== "stubble") {
    facial_hair = "none";
  }

  const expression = pick(EXPRESSIONS, seed, "expression");
  const headshot_id = headshotIdForProfile(seed, { age, position, facial_hair });

  return {
    avatar_seed: seed,
    headshot_id,
    face_variant: headshot_id,
    skin_tone,
    hair_style,
    hair_color,
    facial_hair,
    expression,
    age_bucket: bucket,
    nationality_code: nationalityCode(nationality),
  };
}

export function ensurePlayerHeadshotFields(player = {}) {
  if (!player || typeof player !== "object") return player;

  const existingId = Number(player.headshot_id || player.face_variant || 0);
  const existingSeed = Number(player.avatar_seed || 0);
  if (existingId >= HEADSHOT_MIN && existingId <= HEADSHOT_MAX && existingSeed > 0) {
    return {
      ...player,
      headshot_id: existingId,
      face_variant: Number(player.face_variant || existingId),
      avatar_seed: existingSeed,
      age_bucket: player.age_bucket || ageBucket(player.age),
      nationality_code:
        player.nationality_code ||
        nationalityCode(player.nationality || player.country || player.nat || ""),
      expression: player.expression || "neutral",
    };
  }

  const meta = generatePlayerHeadshotMetadata(player);
  return { ...player, ...meta };
}

const NHL_HEADSHOT_HOSTS = new Set([
  "assets.nhle.com",
  "cms.nhl.bamgrid.com",
]);

function safeNhlHeadshotUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    const parsed = new URL(raw);
    return parsed.protocol === "https:" && NHL_HEADSHOT_HOSTS.has(parsed.hostname.toLowerCase())
      ? parsed.href
      : "";
  } catch {
    return "";
  }
}

export function pickHeadshotIdentityFields(source = {}) {
  if (!source || typeof source !== "object") return {};
  const out = {};
  const assign = (key, value) => {
    if (value !== undefined && value !== null && value !== "") {
      out[key] = value;
    }
  };

  assign("nhl_player_id", source.nhl_player_id ?? source.nhl_id);
  assign("nhl_id", source.nhl_id ?? source.nhl_player_id);
  assign(
    "nhl_headshot_url",
    source.nhl_headshot_url ?? source.nhlHeadshotUrl
  );
  assign("headshot_url", source.headshot_url ?? source.headshotUrl);
  assign("headshot", source.headshot);
  assign("portrait_url", source.portrait_url ?? source.portrait);
  assign("portrait_source", source.portrait_source);
  assign("real_nhl_import", source.real_nhl_import);
  assign("headshot_id", source.headshot_id ?? source.face_variant);
  assign("face_variant", source.face_variant ?? source.headshot_id);
  assign("avatar_seed", source.avatar_seed);
  assign("skin_tone", source.skin_tone);
  assign("hair_style", source.hair_style);
  assign("hair_color", source.hair_color);
  assign("facial_hair", source.facial_hair);
  assign("expression", source.expression);
  assign("age_bucket", source.age_bucket);
  assign("nationality_code", source.nationality_code);

  return out;
}

export function mergePlayerHeadshotIdentity(row = {}, rosterRow = null) {
  return ensurePlayerHeadshotFields({
    ...row,
    ...pickHeadshotIdentityFields(rosterRow),
    ...pickHeadshotIdentityFields(row),
  });
}

/**
 * Resolve a real NHL photograph without weakening the deterministic portrait
 * fallback. Metadata lookup happens during backend import, never in React.
 */
export function getNhlHeadshotUrl(player = {}) {
  if (!player || typeof player !== "object") return "";

  const direct = safeNhlHeadshotUrl(
    player.nhl_headshot_url ||
      player.nhlHeadshotUrl ||
      player.headshot_url ||
      player.headshotUrl ||
      player.headshot ||
      player.portrait_url ||
      player.portrait
  );
  if (direct) return direct;

  const hasNhlIdentity = Boolean(
    player.nhl_player_id ||
      player.nhl_id ||
      player.real_nhl_import ||
      player.portrait_source === "nhl"
  );
  if (!hasNhlIdentity) return "";

  return "";
}

export function resolvePlayerHeadshot(player = {}) {
  const generatedPlayer = ensurePlayerHeadshotFields(player);
  const nhlUrl = getNhlHeadshotUrl(generatedPlayer);
  return {
    player: generatedPlayer,
    src: nhlUrl,
    source: nhlUrl ? "nhl" : "generated",
  };
}
