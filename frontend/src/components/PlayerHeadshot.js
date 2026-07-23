import React, { useMemo } from "react";
import "./PlayerHeadshot.css";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { resolveCountryCode, flagApiUrl } from "../utils/countryFlags";

function clampHeadshotId(value, seed = 0) {
  const n = Number(value);
  if (Number.isFinite(n) && n >= 1 && n <= 60) return Math.floor(n);
  const s = Number(seed);
  if (Number.isFinite(s)) return (Math.abs(Math.floor(s)) % 60) + 1;
  return 1;
}

function nationalityCode(player = {}) {
  return (
    player.nationality_code ||
    player.nationalityCode ||
    (typeof player.nationality === "string" && player.nationality.length <= 3
      ? player.nationality.toUpperCase()
      : "") ||
    ""
  );
}

function roleClass(player = {}) {
  const pos = String(player.position || "").toUpperCase();
  if (pos === "G") return "role-goalie";
  if (pos === "D") return "role-defense";
  if (player.captain || player.is_captain) return "role-captain";
  const arch = String(player.archetype || player.role || "").toLowerCase();
  if (arch.includes("enforcer")) return "role-enforcer";
  if (arch.includes("playmaker")) return "role-playmaker";
  if (arch.includes("sniper")) return "role-sniper";
  if (arch.includes("two")) return "role-two-way";
  if (player.age_bucket === "prospect" || Number(player.age) <= 20) return "role-prospect";
  return "role-forward";
}

/**
 * CSS-only NHL franchise headshot renderer.
 * Expects backend fields: headshot_id, avatar_seed, expression, age_bucket, nationality_code.
 */
export default function PlayerHeadshot({
  player = {},
  size = "md",
  variant = "",
  mood = "",
  badge = null,
  badgeVariant = "",
  number = null,
  flag = null,
  teamColors = null,
  className = "",
  selected = false,
  dimmed = false,
  scouted = false,
  hidden = false,
  locked = false,
  scoutLevel = "",
  injuryState = "",
  draftState = "",
  animate = "",
  title,
  style = {},
  ...rest
}) {
  const resolved = useMemo(() => ensurePlayerHeadshotFields(player), [player]);

  const headshotId = useMemo(
    () => clampHeadshotId(resolved.headshot_id || resolved.face_variant, resolved.avatar_seed),
    [resolved.headshot_id, resolved.face_variant, resolved.avatar_seed]
  );

  const resolvedMood = mood || resolved.expression || "neutral";
  const resolvedFlag = flag ?? nationalityCode(resolved);
  // Prefer a real nation flag image. Resolve an ISO code from the passed flag prop or any
  // nationality/country field on the player, then hit the shared flag CDN.
  const flagIso = useMemo(
    () =>
      resolveCountryCode(resolvedFlag) ||
      resolveCountryCode(resolved.nationality_code) ||
      resolveCountryCode(resolved.countryCode) ||
      resolveCountryCode(resolved.country_code) ||
      resolveCountryCode(resolved.nationality) ||
      resolveCountryCode(resolved.country) ||
      resolveCountryCode(resolved.birth_country) ||
      resolveCountryCode(resolved.birthCountry),
    [
      resolvedFlag,
      resolved.nationality_code,
      resolved.countryCode,
      resolved.country_code,
      resolved.nationality,
      resolved.country,
      resolved.birth_country,
      resolved.birthCountry,
    ]
  );
  const flagUrl = flagIso ? flagApiUrl(flagIso, 64, "flat") : null;
  const resolvedNumber =
    number ?? resolved.jersey_number ?? resolved.number ?? resolved.jersey ?? resolved.num ?? null;

  const classes = [
    "player-headshot",
    `headshot-${headshotId}`,
    size ? `size-${size}` : "",
    variant ? `variant-${variant}` : "",
    resolvedMood ? `mood-${resolvedMood}` : "",
    resolved.age_bucket ? `age-${resolved.age_bucket}` : "",
    roleClass(resolved),
    badgeVariant ? `ph-badge-context-${badgeVariant}` : "",
    selected ? "is-selected" : "",
    dimmed ? "is-dimmed" : "",
    scouted ? "is-scouted" : "",
    hidden ? "is-hidden" : "",
    locked ? "is-locked" : "",
    scoutLevel ? `scout-${scoutLevel}` : "",
    injuryState ? `injury-${injuryState}` : "",
    draftState ? `draft-${draftState}` : "",
    animate ? `animate-${animate}` : "",
    teamColors ? "team-branded" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const mergedStyle = teamColors
    ? {
        ...style,
        "--team-primary": teamColors.primary,
        "--team-secondary": teamColors.secondary,
        "--team-accent": teamColors.accent,
      }
    : style;

  const badgeClass = badgeVariant ? `ph-badge ${badgeVariant}` : "ph-badge";

  return (
    <div
      className={classes}
      style={mergedStyle}
      title={title || resolved.name || [resolved.firstName, resolved.lastName].filter(Boolean).join(" ")}
      aria-label={resolved.name ? `${resolved.name} headshot` : "Player headshot"}
      {...rest}
    >
      <span className="ph-hair" aria-hidden="true" />
      <span className="ph-face" aria-hidden="true" />
      <span className="ph-eyes" aria-hidden="true" />
      <span className="ph-mouth" aria-hidden="true" />
      <span className="ph-extra" aria-hidden="true" />
      {badge ? <span className={badgeClass}>{badge}</span> : null}
      {resolvedNumber != null && resolvedNumber !== "" ? (
        <span className="ph-number">{resolvedNumber}</span>
      ) : null}
      {flagUrl ? (
        <span className="ph-flag ph-flag--img" title={String(resolvedFlag || flagIso || "")}>
          <img
            className="ph-flag__img"
            src={flagUrl}
            alt={String(resolvedFlag || flagIso || "flag")}
            loading="lazy"
            draggable={false}
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
        </span>
      ) : resolvedFlag ? (
        <span className="ph-flag">
          <span className="ph-flag__code">{resolvedFlag}</span>
        </span>
      ) : null}
    </div>
  );
}

export { clampHeadshotId, nationalityCode, roleClass };
