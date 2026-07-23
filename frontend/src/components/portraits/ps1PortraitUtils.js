/**
 * PS1 portrait helpers — deterministic face paths and team jersey colors.
 * Reuses existing player id / seed fields; no backend changes required.
 */

const FACE_PLACEHOLDER_COUNT = 24;

const FRANCHISE_DEFAULT_COLORS = {
  primary: "#1a2744",
  secondary: "#c9a86a",
  accent: "#d4af37",
  jersey: "#1a2744",
  jersey2: "#0d1526",
  skin: "#c58b5f",
  hair: "#111927",
};

function hashString(input) {
  const text = String(input ?? "");
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function padFaceIndex(index) {
  return String(index).padStart(3, "0");
}

/**
 * Deterministic local placeholder index (1..24) so the same player keeps the same face slot.
 */
export function getFallbackFaceIndex(player = {}) {
  const seed =
    player.id ??
    player.player_id ??
    player.person_id ??
    player.avatar_seed ??
    player.name ??
    "unknown";
  return (hashString(seed) % FACE_PLACEHOLDER_COUNT) + 1;
}

/**
 * Resolve face image URL priority:
 * ps1FaceSrc → faceSrc → headshotUrl → /portraits/ps1/face_NNN.png
 */
export function getPS1FaceSrc(player = {}, faceSrcOverride) {
  if (faceSrcOverride) return faceSrcOverride;
  if (player.ps1FaceSrc) return player.ps1FaceSrc;
  if (player.faceSrc) return player.faceSrc;
  const remote = player.headshotUrl || player.headshot_url;
  if (remote) return remote;
  return `/portraits/ps1/face_${padFaceIndex(getFallbackFaceIndex(player))}.png`;
}

/**
 * Pull team jersey colors from player/team objects when present.
 */
export function getTeamPortraitColors(player = {}, team) {
  const source = team && typeof team === "object" ? team : player;
  const primary =
    source.primary_color ||
    source.primaryColor ||
    source.team_primary ||
    source.color_primary ||
    player.team_primary ||
    FRANCHISE_DEFAULT_COLORS.primary;
  const secondary =
    source.secondary_color ||
    source.secondaryColor ||
    source.team_secondary ||
    source.color_secondary ||
    player.team_secondary ||
    FRANCHISE_DEFAULT_COLORS.secondary;

  return {
    primary,
    secondary,
    accent: FRANCHISE_DEFAULT_COLORS.accent,
    jersey: primary,
    jersey2: secondary,
    skin: FRANCHISE_DEFAULT_COLORS.skin,
    hair: FRANCHISE_DEFAULT_COLORS.hair,
  };
}

export function mapPortraitSizeToHeadshot(size = "md") {
  if (size === "xs" || size === "tiny") return "xs";
  if (size === "sm") return "sm";
  if (size === "lg" || size === "profile") return "lg";
  if (size === "xl" || size === "xxl") return "xl";
  return "md";
}

/** Per-headshot skin/hair/style — mirrors CSS headshot palette variety. */
const APPEARANCE_BY_ID = [
  { skin: "#b9855e", skinShadow: "#815237", hair: "#122032", hairStyle: "swept" },
  { skin: "#e0b084", skinShadow: "#a66f48", hair: "#d9b45a", hairStyle: "crop" },
  { skin: "#8f5f41", skinShadow: "#5d3828", hair: "#1b1614", hairStyle: "buzz" },
  { skin: "#5e3a29", skinShadow: "#392117", hair: "#080808", hairStyle: "curly" },
  { skin: "#f0bc91", skinShadow: "#b9784f", hair: "#a8421e", hairStyle: "part" },
  { skin: "#c89267", skinShadow: "#875538", hair: "transparent", hairStyle: "bald" },
  { skin: "#d2a078", skinShadow: "#91603e", hair: "#2a1b12", hairStyle: "flow" },
  { skin: "#b87b52", skinShadow: "#77462e", hair: "#19110d", hairStyle: "beard" },
  { skin: "#db9d6f", skinShadow: "#9c623c", hair: "#312014", hairStyle: "crop" },
  { skin: "#d7a47c", skinShadow: "#8e5c3c", hair: "#151515", hairStyle: "mask" },
  { skin: "#c58b5f", skinShadow: "#8d5b3d", hair: "#2b1810", hairStyle: "spiky" },
  { skin: "#7b4d36", skinShadow: "#44291d", hair: "#0b0b0b", hairStyle: "afro" },
  { skin: "#e0b084", skinShadow: "#a66f48", hair: "#d9b45a", hairStyle: "long" },
  { skin: "#c99672", skinShadow: "#875b42", hair: "#4a4034", hairStyle: "messy" },
  { skin: "#a96e4b", skinShadow: "#70422c", hair: "#14100e", hairStyle: "grey" },
];

export function derivePortraitAppearance(player = {}) {
  const idRaw = Number(player.headshot_id || player.face_variant || 0);
  const seed = Number(player.avatar_seed || hashString(player.id || player.name || 1));
  const headshotId =
    idRaw >= 1 && idRaw <= 60 ? idRaw : (Math.abs(seed) % APPEARANCE_BY_ID.length) + 1;
  const palette = APPEARANCE_BY_ID[(headshotId - 1) % APPEARANCE_BY_ID.length];
  const tilt = ((seed % 360) / 360) * 0.14 - 0.07;
  const eyeOffset = 0.11 + (seed % 7) * 0.004;
  return {
    headshotId,
    ...palette,
    headTilt: tilt,
    eyeSpacing: eyeOffset,
    mouthWidth: 0.1 + (seed % 5) * 0.012,
  };
}

export function getPortraitCamera(size = "profile") {
  const presets = {
    sm: { position: [0, 0.04, 1.05], fov: 42 },
    md: { position: [0, 0.04, 1.12], fov: 40 },
    lg: { position: [0, 0.04, 1.18], fov: 38 },
    profile: { position: [0, 0.04, 1.22], fov: 36 },
    xl: { position: [0, 0.04, 1.28], fov: 34 },
  };
  return presets[size] || presets.profile;
}

export function probeImageSrc(src) {
  return new Promise((resolve) => {
    if (!src || typeof src !== "string") {
      resolve(false);
      return;
    }
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = src;
  });
}
