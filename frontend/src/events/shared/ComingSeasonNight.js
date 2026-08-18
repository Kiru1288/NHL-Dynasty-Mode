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
  const season =
    franchiseState?.season_label ||
    (franchiseState?.season_year
      ? `${franchiseState.season_year}–${Number(franchiseState.season_year) + 1}`
      : "");

  return (
    <>
      <style>{COMING_NIGHT_CSS}</style>
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
          <div className="comingnight-hero">
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

const COMING_NIGHT_CSS = `
.comingnight-hero {
  display: grid;
  gap: 12px;
  max-width: 42rem;
}
.comingnight-kicker {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ops-cyan, #13d8e7);
}
.comingnight-title {
  margin: 0;
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ops-text, #e9f7fb);
  line-height: 1.05;
}
.comingnight-body,
.comingnight-rail {
  margin: 0;
  color: var(--ops-text, #e9f7fb);
  font-size: 0.95rem;
  line-height: 1.5;
}
.comingnight-rail {
  color: var(--ops-muted, #8096a8);
  font-size: 0.85rem;
}
`;
