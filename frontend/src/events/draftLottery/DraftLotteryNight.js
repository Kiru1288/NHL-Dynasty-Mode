import React, { useCallback, useEffect, useMemo, useState } from "react";
import { pickFranchiseData } from "../shared/eventHelpers";
import {
  buildRevealSequence,
  formatMovement,
  normalizeLotteryPicks,
  pickOrdinal,
  revealPaceMs,
} from "./draftLotteryHelpers";
import "./DraftLottery.css";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function TeamLogo({ src, label, size = "md" }) {
  if (src) {
    return (
      <span className={`dlot-logo ${size}`}>
        <img src={src} alt="" loading="lazy" />
      </span>
    );
  }
  return (
    <span className={`dlot-logo ${size} dlot-logo-fallback`}>
      {(label || "?").slice(0, 3).toUpperCase()}
    </span>
  );
}

function MovementChip({ movement }) {
  const { label, tone } = formatMovement(movement);
  return <span className={`dlot-chip tone-${tone}`}>{label}</span>;
}

function PickCard({ pick, active, pending, isUser }) {
  const movement = formatMovement(pick.movement);
  return (
    <article
      className={[
        "dlot-card",
        active ? "is-active" : "",
        pending ? "is-pending" : "",
        isUser ? "is-user" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="dlot-card-rank">#{pick.pick}</div>
      <div className="dlot-card-body">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <TeamLogo src={pick.logoSrc} label={pick.team_name} size="md" />
          <div>
            <strong>{pick.team_name}</strong>
            {pick.is_traded && pick.via_abbr ? (
              <div className="dlot-via">
                {pick.viaLogoSrc ? <TeamLogo src={pick.viaLogoSrc} label={pick.via_abbr} size="xs" /> : null}
                <span>via {pick.via_abbr}</span>
              </div>
            ) : null}
            <div className="dlot-card-details">
              <span>Was #{pick.original_rank}</span>
              <span className={`tone-${movement.tone}`}>{movement.label}</span>
              {pick.odds != null ? <span>{pick.odds}% odds</span> : null}
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}

function RevealOverlay({ pick, revealIndex, revealTotal, onSkip, isUser }) {
  const isTopPick = pick.pick <= 3;

  return (
    <>
      <div className={`dlot-lower-third${isUser ? " is-user" : ""}`} aria-live="polite">
        <div className="dlot-lower-third__panel">
          <div className="dlot-lower-third__pick">#{pick.pick}</div>
          <div className="dlot-lower-third__body">
            <strong>{pick.team_name}</strong>
            <span>
              {isUser ? "Your franchise · " : ""}
              {pickOrdinal(pick.pick).toUpperCase()} overall
            </span>
          </div>
        </div>
      </div>
      <div className={`dlot-reveal-overlay${isUser ? " is-user-team" : ""}`}>
      <p className="dlot-reveal-kicker">
        {isTopPick ? "THE" : "WITH THE"} {pickOrdinal(pick.pick).toUpperCase()} OVERALL PICK
      </p>
      <p className="dlot-reveal-pick">#{pick.pick}</p>
      <TeamLogo src={pick.logoSrc} label={pick.team_name} size="xl" />
      <h2 className="dlot-reveal-team">{pick.team_name}</h2>
      {pick.is_traded && (pick.via_team_name || pick.via_abbr) ? (
        <p className="dlot-reveal-via">
          {pick.viaLogoSrc ? <TeamLogo src={pick.viaLogoSrc} label={pick.via_abbr} size="sm" /> : null}
          <span>via {pick.via_team_name || pick.via_abbr}</span>
        </p>
      ) : null}
      <div className="dlot-reveal-meta">
        <MovementChip movement={pick.movement} />
        <span className="dlot-chip">Projected #{pick.original_rank}</span>
        {pick.odds != null ? <span className="dlot-chip">{pick.odds}% odds</span> : null}
      </div>
      <p className="dlot-reveal-progress">
        Revealing {revealTotal - revealIndex} of {revealTotal} remaining
      </p>
      <button type="button" className="dlot-skip-btn" onClick={onSkip} style={{ marginTop: 20 }}>
        Skip to All Picks
      </button>
      </div>
    </>
  );
}

function ResultsBoard({ picks, userTeamId }) {
  return (
    <section className="dlot-results">
      <div className="dlot-results-head">
        <h2>Draft Order Set</h2>
        <p>Final lottery results — picks 1 through {picks.length || 16}</p>
      </div>
      <div className="dlot-board">
        {picks.map((pick) => {
          const isUser =
            userTeamId &&
            (String(pick.team_id).toLowerCase() === String(userTeamId).toLowerCase() ||
              String(pick.lottery_team_id || "").toLowerCase() === String(userTeamId).toLowerCase());
          const jumped = Number(pick.movement) > 0;
          return (
            <article
              key={`${pick.pick}-${pick.team_id}-${pick.lottery_team_id || ""}`}
              className={[
                "dlot-board-row",
                isUser ? "is-user" : "",
                jumped ? "is-jump" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              <div className="dlot-board-pick">#{pick.pick}</div>
              <TeamLogo src={pick.logoSrc} label={pick.team_name} size="md" />
              <div className="dlot-board-team">
                <div>
                  <strong>{pick.team_name}</strong>
                  {pick.is_traded && pick.via_abbr ? (
                    <span className="dlot-via-inline">
                      {pick.viaLogoSrc ? <TeamLogo src={pick.viaLogoSrc} label={pick.via_abbr} size="xs" /> : null}
                      via {pick.via_abbr}
                    </span>
                  ) : isUser ? (
                    <span>Your team</span>
                  ) : null}
                </div>
              </div>
              <div className="dlot-board-col">Was #{pick.original_rank}</div>
              <div className="dlot-board-col">
                <MovementChip movement={pick.movement} />
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export default function DraftLotteryNight({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
}) {
  const raw = pickFranchiseData(franchiseState, eventData, [
    "draft_lottery",
    "offseason.draft_lottery",
  ]);

  const picks = useMemo(() => normalizeLotteryPicks(raw), [raw]);
  const revealSequence = useMemo(() => buildRevealSequence(picks), [picks]);
  const userTeamId = franchiseState?.user_team_id || franchiseState?.team?.id || "";

  const [stage, setStage] = useState("intro");
  const [revealIndex, setRevealIndex] = useState(0);

  const currentReveal = revealSequence[revealIndex] || null;
  const revealedIds = useMemo(
    () => new Set(revealSequence.slice(0, revealIndex + 1).map((p) => p.pick)),
    [revealSequence, revealIndex]
  );

  const skipToResults = useCallback(() => {
    setStage("results");
    setRevealIndex(revealSequence.length);
  }, [revealSequence.length]);

  const startReveal = useCallback(() => {
    setRevealIndex(0);
    setStage("reveal");
  }, []);

  useEffect(() => {
    if (stage !== "reveal" || !currentReveal) return undefined;

    const pace = revealPaceMs(currentReveal.pick);
    const timer = window.setTimeout(() => {
      if (revealIndex >= revealSequence.length - 1) {
        window.setTimeout(() => setStage("results"), 900);
        return;
      }
      setRevealIndex((i) => i + 1);
    }, pace);

    return () => window.clearTimeout(timer);
  }, [stage, revealIndex, currentReveal, revealSequence.length]);

  const tickerItems = [
    "LOTTERY NIGHT",
    "BALLS DROP",
    "ORDER SET",
    "FUTURE STARS",
    picks.length ? `${picks.length} TEAMS` : "DRAFT ORDER",
  ];

  return (
    <section className="dlot-root">
      <div className="dlot-bg" aria-hidden="true">
        <div className="dlot-bg-scrim" />
        <div className="dlot-bg-noise" />
        <div className="dlot-spotlight" />
      </div>

      <header className="dlot-topbar">
        <button type="button" className="dlot-ghost-btn" onClick={onBack}>
          ← Hub World
        </button>
        <div className="dlot-status-pill">LOTTERY NIGHT</div>
        <div className="dlot-season">{seasonLabel(franchiseState)}</div>
      </header>

      {stage === "intro" && (
        <main className="dlot-intro">
          <p className="dlot-eyebrow">NHL Draft Lottery</p>
          <h1 className="dlot-title">Lottery Night</h1>
          <p className="dlot-subtitle">
            Watch the reveal from the {revealSequence[0]?.pick || 16}
            {revealSequence[0]?.pick === 1 ? "st" : "th"} pick up to #1, or skip straight to the
            full draft order.
          </p>
          {picks.length ? (
            <div className="dlot-intro-actions">
              <button type="button" className="dlot-cta-btn" onClick={startReveal}>
                Watch Reveal
              </button>
              <button type="button" className="dlot-secondary-btn" onClick={skipToResults}>
                View All Picks
              </button>
            </div>
          ) : (
            <p className="dlot-empty">Lottery data unavailable</p>
          )}
        </main>
      )}

      {stage === "reveal" && picks.length ? (
        <>
          <main className="dlot-stage">
            <section>
              <p className="dlot-eyebrow">Live reveal</p>
              <h1 className="dlot-title">Pick Order</h1>
              <p className="dlot-subtitle" style={{ textAlign: "left", marginBottom: 0 }}>
                Revealing from #{revealSequence[0]?.pick} down to #1
              </p>
            </section>
            <aside className="dlot-panel">
              <h2>Revealed Picks</h2>
              <div className="dlot-rail">
                {revealSequence.map((pick) => {
                  const revealed = revealedIds.has(pick.pick);
                  const active = currentReveal?.pick === pick.pick;
                  const isUser =
                    userTeamId &&
                    (String(pick.team_id).toLowerCase() === String(userTeamId).toLowerCase() ||
                      String(pick.lottery_team_id || "").toLowerCase() === String(userTeamId).toLowerCase());
                  return (
                    <PickCard
                      key={pick.pick}
                      pick={pick}
                      active={active}
                      pending={!revealed}
                      isUser={isUser}
                    />
                  );
                })}
              </div>
            </aside>
          </main>
          {currentReveal ? (
            <RevealOverlay
              pick={currentReveal}
              revealIndex={revealIndex}
              revealTotal={revealSequence.length}
              onSkip={skipToResults}
              isUser={
                Boolean(userTeamId) &&
                (String(currentReveal.team_id).toLowerCase() === String(userTeamId).toLowerCase() ||
                  String(currentReveal.lottery_team_id || "").toLowerCase() === String(userTeamId).toLowerCase())
              }
            />
          ) : null}
        </>
      ) : null}

      {stage === "results" && picks.length ? (
        <ResultsBoard picks={picks} userTeamId={userTeamId} />
      ) : null}

      <footer className="dlot-footer">
        <div className="dlot-ticker">
          <div className="dlot-ticker-track">
            <div className="dlot-ticker-group">
              {tickerItems.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="dlot-footer-actions">
          {stage === "reveal" ? (
            <button type="button" className="dlot-skip-btn" onClick={skipToResults}>
              Skip Cinematic
            </button>
          ) : null}
          <button
            type="button"
            className="dlot-cta-btn"
            onClick={() => {
              if (stage !== "results") {
                skipToResults();
                return;
              }
              onContinue?.();
            }}
            disabled={typeof onContinue !== "function"}
          >
            {stage === "results" ? "Enter Combine" : "Skip to Enter Combine"}
          </button>
        </div>
      </footer>
    </section>
  );
}
