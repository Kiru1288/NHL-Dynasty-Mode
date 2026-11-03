import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ensurePs2CssFilter, peekPs2Photo, requestPs2Photo } from "../utils/ps2Pixel";
import { cssColorToHex, renderLowPolyFace } from "../utils/lowPolyFace";
import { resolvePlayerHeadshot } from "../utils/playerHeadshots";
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
 * Shared player portrait renderer. NHL photography is layered above the
 * deterministic CSS portrait, which remains mounted as the failure fallback.
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
  preferPhoto = true,
  showFlag = true,
  ...rest
}) {
  const portrait = useMemo(() => resolvePlayerHeadshot(player), [player]);
  const resolved = portrait.player;
  const [failedNhlSrc, setFailedNhlSrc] = useState("");
  const showNhlPhoto = Boolean(preferPhoto && portrait.src && failedNhlSrc !== portrait.src);
  const rootRef = useRef(null);

  // PS2 restyle of real photos: cached canvas pass, CSS posterize until ready / if blocked.
  const photoSrc = showNhlPhoto ? portrait.src : "";
  const [ps2Photo, setPs2Photo] = useState(() => (photoSrc ? peekPs2Photo(photoSrc) || "" : ""));
  useEffect(() => {
    if (!photoSrc) return undefined;
    ensurePs2CssFilter();
    const hit = peekPs2Photo(photoSrc);
    if (hit !== undefined) {
      setPs2Photo(hit || "");
      return undefined;
    }
    setPs2Photo("");
    let live = true;
    let io = null;
    const go = () => {
      requestPs2Photo(photoSrc).then((url) => {
        if (live && url) setPs2Photo(url);
      });
    };
    const el = rootRef.current;
    if (el && typeof IntersectionObserver !== "undefined") {
      io = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            io.disconnect();
            io = null;
            go();
          }
        },
        { rootMargin: "300px" }
      );
      io.observe(el);
    } else {
      go();
    }
    return () => {
      live = false;
      if (io) io.disconnect();
    };
  }, [photoSrc]);

  const headshotIdKey = resolved.headshot_id || resolved.face_variant || "";
  // Generated players: low-poly PS2 face in the headshot's jersey colours.
  const [cssJersey, setCssJersey] = useState(null);
  const useIsoLayout = typeof window !== "undefined" ? useLayoutEffect : useEffect;
  useIsoLayout(() => {
    if (showNhlPhoto || (teamColors && teamColors.primary)) return;
    const el = rootRef.current;
    if (!el || typeof getComputedStyle !== "function") return;
    try {
      const cs = getComputedStyle(el);
      const primary = cssColorToHex(cs.getPropertyValue("--jersey"));
      const secondary = cssColorToHex(cs.getPropertyValue("--jersey2"));
      if (!cssJersey || cssJersey.primary !== (primary || undefined) || cssJersey.secondary !== (secondary || undefined)) {
        setCssJersey({ primary: primary || undefined, secondary: secondary || undefined });
      }
    } catch {
      if (!cssJersey) setCssJersey({});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showNhlPhoto, teamColors && teamColors.primary, className, variant, size, roleClass(resolved), headshotIdKey, style && style["--team-primary"], style && style["--jersey"]]);
  const faceColors = useMemo(() => {
    if (teamColors && teamColors.primary) {
      return {
        primary: cssColorToHex(teamColors.primary) || undefined,
        secondary: cssColorToHex(teamColors.secondary) || cssColorToHex(teamColors.accent) || undefined,
      };
    }
    return cssJersey || {};
  }, [teamColors, cssJersey]);
  const lowPolySrc = useMemo(
    // Wait for the jersey colours (read in a layout effect, before paint) so each face renders once.
    () => (showNhlPhoto || !((teamColors && teamColors.primary) || cssJersey) ? "" : renderLowPolyFace(resolved, faceColors)),
    [showNhlPhoto, resolved, faceColors, teamColors, cssJersey]
  );

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
    showNhlPhoto ? "has-nhl-headshot photo-only" : "",
    showNhlPhoto ? (ps2Photo ? "ph-ps2-ready" : "ph-ps2-pending") : "",
    !showNhlPhoto && lowPolySrc ? "photo-only ph-lowpoly" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const mergedStyle = {
    ...(teamColors
      ? {
          "--team-primary": teamColors.primary,
          "--team-secondary": teamColors.secondary,
          "--team-accent": teamColors.accent,
        }
      : {}),
    ...style,
  };

  const badgeClass = badgeVariant ? `ph-badge ${badgeVariant}` : "ph-badge";

  return (
    <div
      ref={rootRef}
      className={classes}
      style={mergedStyle}
      title={title || resolved.name || [resolved.firstName, resolved.lastName].filter(Boolean).join(" ")}
      aria-label={resolved.name ? `${resolved.name} headshot` : "Player headshot"}
      {...rest}
    >
      {showNhlPhoto ? (
        <img
          className="ph-nhl-image ph-nhl-image--solo"
          src={ps2Photo || portrait.src}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
          onError={(event) => {
            event.currentTarget.style.display = "none";
            setFailedNhlSrc(portrait.src);
          }}
        />
      ) : lowPolySrc ? (
        <img
          className="ph-nhl-image ph-nhl-image--solo ph-lowpoly-img"
          src={lowPolySrc}
          alt=""
          decoding="async"
          draggable={false}
        />
      ) : (
        <>
          <span className="ph-hair" aria-hidden="true" />
          <span className="ph-face" aria-hidden="true" />
          <span className="ph-eyes" aria-hidden="true" />
          <span className="ph-mouth" aria-hidden="true" />
          <span className="ph-extra" aria-hidden="true" />
        </>
      )}
      {badge ? <span className={badgeClass}>{badge}</span> : null}
      {resolvedNumber != null && resolvedNumber !== "" ? (
        <span className="ph-number">{resolvedNumber}</span>
      ) : null}
      {showFlag && flagUrl ? (
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
      ) : showFlag && resolvedFlag ? (
        <span className="ph-flag">
          <span className="ph-flag__code">{resolvedFlag}</span>
        </span>
      ) : null}
    </div>
  );
}

export { clampHeadshotId, nationalityCode, roleClass };
