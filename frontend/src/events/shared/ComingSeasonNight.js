import React from "react";
import CinematicEventShell from "./CinematicEventShell";

/**
 * First-class placeholder for season nights that are not built yet.
 * Never return null from a routed event menu.
 */
export default function ComingSeasonNight({
  phaseLabel = "SEASON NIGHT",
  title = "Coming this season",
  eyebrow = "On the calendar",
  body = "This night is on the league calendar. The full broadcast will land here.",
  ctaLabel = "Return to Hub",
  onContinue,
  onBack,
  franchiseState = {},
}) {
  const leave = onBack || onContinue;
  // Theme slug from the night's label ("TRADE DEADLINE" -> "trade-deadline"); game.css dresses each one.
  const theme = String(phaseLabel || "season-night").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const season =
    franchiseState?.season_label ||
    (franchiseState?.season_year
      ? `${franchiseState.season_year}–${Number(franchiseState.season_year) + 1}`
      : "");

  return (
    <>
      {null}
      <CinematicEventShell
        prefix="comingnight"
        phaseLabel={phaseLabel}
        seasonLabel={season}
        title={title}
        hideTitle
        hideEyebrow
        ctaLabel={ctaLabel}
        revealDelayMs={0}
        showSkip={false}
        hideTicker
        onContinue={leave}
        onBack={leave}
        heroContent={
          <div className={`comingnight-hero comingnight-hero--${theme}`}>
            <span className="comingnight-prop" aria-hidden="true" />
            <p className="comingnight-kicker">{eyebrow}</p>
            <h1 className="comingnight-title">{title}</h1>
            <p className="comingnight-body">{body}</p>
          </div>
        }
        railTitle="Tonight"
        railContent={
          <p className="comingnight-rail">
            Leave to the hub and keep running the club. This night will open as a full
            broadcast when it ships.
          </p>
        }
      />
    </>
  );
}


