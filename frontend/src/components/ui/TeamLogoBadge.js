import React, { useState, useCallback, useEffect } from "react";
import { toLogoUrl } from "../../utils/teamLogos";

export function initialsFromTeam(teamName = "NHL") {
  const words = String(teamName).split(/\s+/).filter(Boolean);
  // League / club codes ("KHL", "AHL", "SHL") read as themselves, not one letter.
  if (words.length === 1 && words[0].length <= 5) return words[0].toUpperCase();
  return words
    .slice(0, 2)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
}

/**
 * Reusable team logo for HUD panels and overlays.
 * variant: badge | circle | watermark | framed
 */
export default function TeamLogoBadge({
  teamLogo = "",
  teamName = "Team",
  size = 48,
  variant = "badge",
  opacity = 1,
  className = "",
  title,
}) {
  const [imgError, setImgError] = useState(false);
  const initials = initialsFromTeam(teamName);
  const logoUrl = toLogoUrl(teamLogo);
  const showImage = Boolean(logoUrl) && !imgError;

  useEffect(() => {
    setImgError(false);
  }, [logoUrl]);

  const handleError = useCallback(() => {
    setImgError(true);
  }, []);

  const style = {
    "--team-logo-size": `${size}px`,
    opacity,
  };

  const classes = [
    "team-logo-badge",
    `team-logo-badge--${variant}`,
    showImage ? "team-logo-badge--image" : "team-logo-badge--initials",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={classes}
      style={style}
      title={title || teamName}
      aria-hidden={variant === "watermark" ? true : undefined}
    >
      {showImage ? (
        <img
          src={logoUrl}
          alt=""
          onError={handleError}
          draggable={false}
        />
      ) : (
        <span>{initials}</span>
      )}
    </div>
  );
}
