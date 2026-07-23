import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import {
  buildAwardsCeremonySlides,
  buildAwardsFanTweets,
  buildAwardsNightSummary,
  buildAwardTickerItems,
  buildCeremonyRailGroups,
  buildFallbackAwardFans,
  getCeremonyRevealStatus,
  normalizeAwardsPayload,
  SEASON_MILESTONES,
} from "./awardHelpers";
import "./AwardsNight.css";

const PHASES = {
  INTRO: "intro",
  NOMINEES: "nominees",
  WALK: "walk",
  REVEAL: "reveal",
  WINNER: "winner",
  SUMMARY: "summary",
};

const PHASE_ORDER = [
  PHASES.INTRO,
  PHASES.NOMINEES,
  PHASES.WALK,
  PHASES.REVEAL,
  PHASES.WINNER,
];

const PHASE_TIME = {
  [PHASES.NOMINEES]: 3400,
  [PHASES.WALK]: 1700,
  [PHASES.REVEAL]: 2200,
  [PHASES.WINNER]: 5200,
};

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "Season Complete";
}

function initials(name = "") {
  return String(name)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

function ShowTrophy({ slide, compact = false }) {
  return (
    <div className={`an-show-trophy${compact ? " is-compact" : ""}`} aria-hidden="true">
      <div className="an-show-trophy__glow" />
      <div className="an-show-trophy__cup">
        <span>{slide?.awardShort || "AWD"}</span>
      </div>
      <div className="an-show-trophy__base" />
    </div>
  );
}

function PortraitVisual({ card, slide, fallback = "?" }) {
  const isTeam = slide?.awardKind === "team" || (card?.teamLogoSrc && !card?.player);

  if (isTeam) {
    if (card?.teamLogoSrc || slide?.winnerLogoSrc) {
      return <img className="an-team-logo" src={card?.teamLogoSrc || slide?.winnerLogoSrc} alt="" />;
    }
    return (
      <div className="an-team-logo an-team-logo--fallback">
        {initials(card?.label || slide?.winnerLabel || fallback)}
      </div>
    );
  }

  const player = card?.player || slide?.winnerPlayer;
  if (player) {
    return (
      <div className="an-player-portrait">
        <PlayerHeadshot player={player} size="lg" variant="hero" mood="neutral" animate="in" />
      </div>
    );
  }

  return (
    <div className="an-player-portrait an-player-portrait--fallback">
      {initials(card?.label || slide?.winnerLabel || fallback)}
    </div>
  );
}

function StageCopy({ slide, phase }) {
  if (!slide) return null;

  const isIntro = phase === PHASES.INTRO;
  const isReveal = phase === PHASES.REVEAL || phase === PHASES.WINNER;
  const whyLine = Array.isArray(slide.whyTheyWon) ? slide.whyTheyWon[0] : slide.rationale;

  return (
    <div className={`an-stage-copy${isIntro ? " is-intro" : ""}${isReveal ? " is-reveal" : ""}`}>
      <span>{slide.title || slide.awardLabel}</span>
      {isIntro ? (
        <>
          <h1>{slide.awardLabel}</h1>
          <p>{slide.stageLine || slide.rationale || "League honors handed out tonight."}</p>
        </>
      ) : isReveal ? (
        <>
          <h1>The winner is</h1>
          <p className="an-reveal-name">{slide.winnerLabel}</p>
          {slide.winnerTeamName && slide.awardKind !== "team" ? (
            <p>{slide.winnerTeamName}</p>
          ) : null}
        </>
      ) : (
        <>
          <h1>{slide.awardLabel}</h1>
          <p>{slide.stageLine || "Finalists on stage."}</p>
        </>
      )}
      {phase === PHASES.WINNER && whyLine ? <p>{whyLine}</p> : null}
    </div>
  );
}

function NomineeLineup({ slide, phase }) {
  const cards = (slide?.finalistCards?.length ? slide.finalistCards : slide?.candidateCards || []).slice(0, 3);
  const revealed = phase === PHASES.REVEAL || phase === PHASES.WINNER;

  if (!cards.length) {
    return (
      <div className="an-nominee-empty">
        <ShowTrophy slide={slide} compact />
        <p>Finalists on stage — winner still sealed.</p>
      </div>
    );
  }

  return (
    <div className="an-nominee-lineup">
      {cards.map((card) => {
        const isWinner = Boolean(card.isWinner);
        return (
          <article
            key={`${card.rank}-${card.label}`}
            className={`an-nominee-card${revealed ? " is-revealed" : ""}${revealed && isWinner ? " is-winner" : ""}`}
          >
            <div className="an-nominee-card__portrait">
              <PortraitVisual card={card} slide={slide} />
            </div>
            <div className="an-nominee-card__copy">
              <span>{revealed && isWinner ? "Winner" : `Finalist #${card.rank}`}</span>
              <strong>{card.label}</strong>
              <em>{card.teamName || card.stat || card.subline || ""}</em>
            </div>
            <div className="an-nominee-card__shine" aria-hidden="true" />
          </article>
        );
      })}
    </div>
  );
}

function Presenter({ phase, slide }) {
  const visible = phase === PHASES.WALK || phase === PHASES.REVEAL || phase === PHASES.WINNER;
  if (!visible) return null;

  return (
    <div className="an-presenter" aria-hidden="true">
      <div className="an-presenter__shadow" />
      <div className="an-presenter__body">
        <div className="an-presenter__head" />
        <div className="an-presenter__torso">
          <div className="an-presenter__lapel" />
          <div className="an-presenter__tie" />
          <div className="an-presenter__mic" />
        </div>
        <div className="an-presenter__card">{slide?.awardShort || "AWD"}</div>
      </div>
    </div>
  );
}

function WinnerMoment({ slide, phase }) {
  const visible = phase === PHASES.WINNER;
  const stats = slide?.statCards || [];
  const badges = slide?.heroBadges || [];
  const why = Array.isArray(slide?.whyTheyWon) ? slide.whyTheyWon.slice(0, 2) : [];

  return (
    <div className={`an-winner-moment${visible ? " is-visible" : ""}${stats.length ? " has-stats" : ""}`}>
      <div className="an-winner-moment__visual">
        <PortraitVisual slide={slide} card={{ player: slide?.winnerPlayer, teamLogoSrc: slide?.winnerLogoSrc || slide?.winnerTeamLogoSrc, label: slide?.winnerLabel }} />
        {slide?.winnerTeamLogoSrc && slide?.awardKind !== "team" ? (
          <div className="an-winner-team-badge">
            <img className="an-team-logo" src={slide.winnerTeamLogoSrc} alt="" />
          </div>
        ) : null}
      </div>
      <div className="an-winner-moment__copy">
        <span className="an-winner-moment__label">{slide?.awardLabel}</span>
        <h2>{slide?.winnerLabel}</h2>
        {slide?.winnerTeamName && slide?.awardKind !== "team" ? <p>{slide.winnerTeamName}</p> : null}
        {why.map((line) => (
          <p key={line}>{line}</p>
        ))}
        {badges.length ? (
          <div className="an-winner-badges">
            {badges.map((badge) => (
              <span
                key={badge.label}
                className={`an-winner-badge${badge.tone === "gold" || badge.tone === "primary" ? " is-gold" : badge.tone === "accent" ? " is-accent" : ""}`}
              >
                {badge.label}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {stats.length ? (
        <div className="an-winner-stats">
          {stats.slice(0, 4).map((stat) => (
            <div key={stat.label} className={`an-winner-stat${stat.tone === "primary" ? " is-primary" : ""}`}>
              <span>{stat.label}</span>
              <strong>
                {stat.value}
                {stat.suffix || ""}
              </strong>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function AwardsRail({ groups, activeIndex, totalSlides, phase, revealedIds, onSelect }) {
  const isAwardRevealed = (index, slide) =>
    index < activeIndex ||
    revealedIds.has(slide.id) ||
    (index === activeIndex && (phase === PHASES.REVEAL || phase === PHASES.WINNER));

  const railSubline = (index, slide, status) => {
    if (isAwardRevealed(index, slide)) return slide.winnerLabel;
    if (index === activeIndex) {
      if (phase === PHASES.INTRO) return "Up next";
      return "On stage";
    }
    if (status === "up-next") return "Up next";
    return "Locked";
  };

  return (
    <aside className="an-awards-rail">
      <h2 className="an-panel-title">Ceremony Order</h2>
      <div className="an-awards-rail__list">
        {groups.map((group) => (
          <div key={group.id} className="an-awards-rail__group">
            <h3>{group.label}</h3>
            {group.items.map(({ slide, index }) => {
              const status = getCeremonyRevealStatus(index, activeIndex, totalSlides);
              return (
                <button
                  key={slide.id}
                  type="button"
                  className={`an-awards-rail__item is-${status}${index === activeIndex ? " is-active" : ""}`}
                  style={{ "--award-accent": slide.accent }}
                  onClick={() => onSelect(index)}
                >
                  <span className="an-awards-rail__badge">{slide.awardShort || index + 1}</span>
                  <span className="an-awards-rail__copy">
                    <strong>{slide.awardLabel}</strong>
                    <em>{railSubline(index, slide, status)}</em>
                  </span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}

function LiveFanFeed({ tweets = [], spoilerMode = true }) {
  const items = tweets.slice(0, 12);

  const tweetClass = (tweet) => {
    const tone = String(tweet?.tone || "live");
    if (tone === "disgust" || tone === "debate" || tone === "shock") return " is-disgust";
    if (tone === "hot-take") return " is-hot-take";
    if (tone === "anticipation" || tone === "hype") return " is-support";
    if (tone === "support" || tone === "celebration") return " is-support";
    return " is-support";
  };

  return (
    <aside className="an-live-feed">
      <div className="an-live-feed__header">
        <span className="an-live-feed__dot" />
        <strong>Fan Pulse</strong>
        <em>{spoilerMode ? "Live · No spoilers" : "Live · Reactions"}</em>
      </div>
      <div className="an-live-feed__stream">
        {!items.length ? (
          <div className="an-live-tweet">
            <div className="an-live-tweet__avatar"><span>RW</span></div>
            <div className="an-live-tweet__body">
              <p>Waiting on fan reactions…</p>
            </div>
          </div>
        ) : (
          items.map((tweet) => (
            <article
              key={tweet.id}
              className={`an-live-tweet${tweetClass(tweet)}${tweet?.tone === "hot-take" ? " is-compact" : ""}`}
            >
              <div className="an-live-tweet__avatar">
                {tweet.avatarSrc || tweet.fan?.avatarSrc ? (
                  <img src={tweet.avatarSrc || tweet.fan?.avatarSrc} alt="" />
                ) : (
                  <span>{initials(tweet.displayName || tweet.fan?.displayName || "Fan")}</span>
                )}
              </div>
              <div className="an-live-tweet__body">
                <div className="an-live-tweet__top">
                  <strong>{tweet.displayName || tweet.fan?.displayName || "Fan"}</strong>
                  <span>{tweet.handle || tweet.fan?.handle || "@fan"}</span>
                </div>
                <p className={tweet?.tone === "hot-take" ? "an-live-tweet__hot" : ""}>{tweet.text}</p>
                <div className="an-live-tweet__meta">
                  <span>{tweet.awardLabel || "Awards Night"}</span>
                  <span>{tweet.createdAtLabel || "now"}</span>
                </div>
              </div>
            </article>
          ))
        )}
      </div>
    </aside>
  );
}

function SummaryScreen({ summary, slides, ticker, tweets, onContinue, continuing }) {
  return (
    <div className="an-summary">
      <section className="an-summary__card">
        <span>Awards Night Complete</span>
        <h1>{summary.headline}</h1>
        <p>{summary.subline}</p>
        <div className="an-summary__winners">
          {(summary.heroAwards || []).map((award) => (
            <div key={award.awardKey} className="an-summary__winner">
              <strong>{award.awardLabel}</strong>
              <span>{award.winnerLabel}</span>
            </div>
          ))}
        </div>
        {ticker.length ? (
          <div className="an-summary__ticker">
            {ticker.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        ) : null}
        <button type="button" className="an-gold-btn" onClick={onContinue} disabled={continuing}>
          {continuing ? "Continuing…" : "Continue to Retirements"}
        </button>
      </section>
      <LiveFanFeed tweets={tweets} />
    </div>
  );
}

/**
 * Permanent Awards Night franchise ceremony.
 */
export default function AwardsNight({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
}) {
  const awards = useMemo(
    () => normalizeAwardsPayload(franchiseState, eventData),
    [franchiseState, eventData]
  );

  const slides = useMemo(() => buildAwardsCeremonySlides(awards), [awards]);
  const summary = useMemo(() => buildAwardsNightSummary(awards), [awards]);
  const railGroups = useMemo(() => buildCeremonyRailGroups(slides), [slides]);
  const ticker = useMemo(() => buildAwardTickerItems(awards), [awards]);
  const fanPool = useMemo(() => buildFallbackAwardFans(24, "awards-night-live"), []);
  const preShowTweets = useMemo(
    () =>
      buildAwardsFanTweets(awards, {
        fans: fanPool,
        maxTweets: 48,
        tweetsPerAward: 4,
        seed: "awards-night-pre",
        spoilerFree: true,
        includeSummaryTweets: false,
        includeHotTakes: true,
        includeExtremeTakes: false,
      }),
    [awards, fanPool]
  );
  const reactionTweets = useMemo(
    () =>
      buildAwardsFanTweets(awards, {
        fans: fanPool,
        maxTweets: 48,
        tweetsPerAward: 4,
        seed: "awards-night-post",
        spoilerFree: false,
        includeSummaryTweets: false,
        includeHotTakes: true,
        includeExtremeTakes: true,
      }),
    [awards, fanPool]
  );

  const [activeIndex, setActiveIndex] = useState(0);
  const [phase, setPhase] = useState(PHASES.INTRO);
  const [autoRun, setAutoRun] = useState(true);
  const [voiceOn, setVoiceOn] = useState(true);
  const [revealedIds, setRevealedIds] = useState(() => new Set());
  const [continuing, setContinuing] = useState(false);
  const [continueError, setContinueError] = useState("");
  const timerRef = useRef(null);
  const spokenRef = useRef("");
  const { openFranchiseEvent } = useGameUI() || {};

  const activeSlide = slides[activeIndex] || null;
  const awardRevealed =
    Boolean(activeSlide?.id && revealedIds.has(activeSlide.id)) ||
    phase === PHASES.REVEAL ||
    phase === PHASES.WINNER;

  const liveTweets = useMemo(() => {
    if (!activeSlide?.awardKey) return preShowTweets.slice(0, 10);
    const pool = awardRevealed ? reactionTweets : preShowTweets;
    const scoped = pool.filter((tweet) => tweet.awardKey === activeSlide.awardKey);
    return (scoped.length ? scoped : pool).slice(0, 12);
  }, [activeSlide?.awardKey, awardRevealed, preShowTweets, reactionTweets]);

  const speak = useCallback(
    (text) => {
      if (!voiceOn || !text || typeof window === "undefined" || !window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 0.94;
      utter.pitch = 1;
      window.speechSynthesis.speak(utter);
    },
    [voiceOn]
  );

  const cancelSpeech = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const jumpToAward = useCallback(
    (indexValue) => {
      const next = Math.max(0, Math.min(slides.length - 1, Number(indexValue) || 0));
      clearTimer();
      cancelSpeech();
      spokenRef.current = "";
      setActiveIndex(next);
      setPhase(PHASES.INTRO);
    },
    [cancelSpeech, clearTimer, slides.length]
  );

  const advancePhase = useCallback(() => {
    clearTimer();
    setPhase((current) => {
      const idx = PHASE_ORDER.indexOf(current);
      if (idx < 0 || idx >= PHASE_ORDER.length - 1) return current;
      return PHASE_ORDER[idx + 1];
    });
  }, [clearTimer]);

  const nextAward = useCallback(() => {
    clearTimer();
    if (activeIndex >= slides.length - 1) {
      setPhase(PHASES.SUMMARY);
      return;
    }
    setActiveIndex((i) => i + 1);
    setPhase(PHASES.INTRO);
  }, [activeIndex, clearTimer, slides.length]);

  const revealNow = useCallback(() => {
    clearTimer();
    setPhase(PHASES.REVEAL);
  }, [clearTimer]);

  const beginShow = useCallback(() => {
    clearTimer();
    setPhase(PHASES.NOMINEES);
  }, [clearTimer]);

  const skipWholeShow = useCallback(() => {
    clearTimer();
    cancelSpeech();
    setPhase(PHASES.SUMMARY);
  }, [cancelSpeech, clearTimer]);

  useEffect(() => {
    setActiveIndex(0);
    setPhase(PHASES.INTRO);
    setRevealedIds(new Set());
    spokenRef.current = "";
  }, [slides.length]);

  useEffect(() => {
    if ((phase === PHASES.REVEAL || phase === PHASES.WINNER) && activeSlide?.id) {
      setRevealedIds((prev) => {
        if (prev.has(activeSlide.id)) return prev;
        const next = new Set(prev);
        next.add(activeSlide.id);
        return next;
      });
    }
  }, [activeSlide?.id, phase]);

  useEffect(() => {
    if (!voiceOn || !activeSlide) return undefined;
    if (phase !== PHASES.REVEAL && phase !== PHASES.WINNER) return undefined;

    const line = `The ${activeSlide.awardLabel} goes to ${activeSlide.winnerLabel}.`;
    const key = `${activeSlide.id}:${line}`;
    if (spokenRef.current === key) return undefined;

    spokenRef.current = key;
    speak(line);
    return cancelSpeech;
  }, [activeSlide, cancelSpeech, phase, speak, voiceOn]);

  useEffect(() => {
    if (!autoRun || phase === PHASES.INTRO || phase === PHASES.SUMMARY) return undefined;
    const delay = PHASE_TIME[phase];
    if (!delay) return undefined;

    timerRef.current = window.setTimeout(() => {
      if (phase === PHASES.WINNER) {
        nextAward();
      } else {
        advancePhase();
      }
    }, delay);

    return clearTimer;
  }, [advancePhase, autoRun, clearTimer, nextAward, phase]);

  const handleContinue = useCallback(async () => {
    if (continuing || typeof onContinue !== "function") return;
    setContinuing(true);
    setContinueError("");
    try {
      await onContinue();
      if (typeof openFranchiseEvent === "function") openFranchiseEvent();
    } catch (error) {
      const message =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        (error?.message && /network/i.test(String(error.message))
          ? "Network error advancing the offseason. Return to Hub and use Resume Offseason Timeline."
          : null) ||
        error?.message ||
        "Could not continue the offseason.";
      setContinueError(String(message));
    } finally {
      setContinuing(false);
    }
  }, [continuing, onContinue, openFranchiseEvent]);

  if (!slides.length) {
    return (
      <section className="an-root an-awards-show">
        <div className="an-stage-bg" />
        <div className="an-stage-noise" />
        <div className="an-empty-show">
          <h1>Awards Night</h1>
          <p>No award winners were found. Complete the playoffs to reveal season hardware.</p>
          {typeof onBack === "function" ? (
            <button type="button" className="an-gold-btn" onClick={onBack}>
              Hub World
            </button>
          ) : null}
          <button type="button" className="an-gold-btn" onClick={handleContinue} disabled={continuing}>
            Continue
          </button>
        </div>
      </section>
    );
  }

  const rootStyle = {
    "--an-accent": activeSlide?.accent || "#f6c453",
    "--an-glow": activeSlide?.glow || "rgba(246, 196, 83, 0.38)",
  };

  if (phase === PHASES.SUMMARY) {
    return (
      <section className="an-root an-awards-show is-summary" style={rootStyle}>
        <div className="an-stage-bg" />
        <div className="an-stage-noise" />
        <header className="an-show-topbar">
          <div className="an-show-brand">
            <span className="an-live-dot" />
            <div>
              <strong>Awards Night</strong>
              <em>{seasonLabel(franchiseState)}</em>
            </div>
          </div>
        </header>
        <SummaryScreen
          summary={summary}
          slides={slides}
          ticker={ticker}
          tweets={summary.fanTweets || reactionTweets}
          onContinue={handleContinue}
          continuing={continuing}
        />
        {continueError ? <p className="an-empty-show">{continueError}</p> : null}
      </section>
    );
  }

  return (
    <section className={`an-root an-awards-show phase-${phase}`} style={rootStyle}>
      <div className="an-stage-bg" />
      <div className="an-stage-noise" />

      <header className="an-show-topbar">
        <div className="an-show-brand">
          <span className="an-live-dot" />
          <div>
            <strong>Awards Night</strong>
            <em>{seasonLabel(franchiseState)}</em>
          </div>
        </div>

        <div className="an-show-actions">
          {typeof onBack === "function" ? (
            <button type="button" className="an-dark-btn" onClick={onBack}>
              Leave to Hub
            </button>
          ) : null}
          <select
            className="an-jump-select"
            value={activeIndex}
            onChange={(event) => jumpToAward(event.target.value)}
            aria-label="Skip to award"
          >
            {slides.map((slide, index) => (
              <option key={slide.id} value={index}>
                {slide.awardLabel}
              </option>
            ))}
          </select>
          <button type="button" className="an-dark-btn" onClick={() => setVoiceOn((v) => !v)}>
            TTS {voiceOn ? "On" : "Off"}
          </button>
          <button type="button" className="an-dark-btn" onClick={() => setAutoRun((v) => !v)}>
            {autoRun ? "Pause" : "Play"}
          </button>
          <button type="button" className="an-dark-btn" onClick={skipWholeShow}>
            Skip Show
          </button>
        </div>
      </header>

      <main className="an-show-layout">
        <AwardsRail
          groups={railGroups}
          activeIndex={activeIndex}
          totalSlides={slides.length}
          phase={phase}
          revealedIds={revealedIds}
          onSelect={jumpToAward}
        />

        <section className="an-main-stage">
          <div className="an-curtain an-curtain--left" aria-hidden="true" />
          <div className="an-curtain an-curtain--right" aria-hidden="true" />
          <div className="an-curtain-valance" aria-hidden="true" />
          <div className="an-stage-light an-stage-light--left" aria-hidden="true" />
          <div className="an-stage-light an-stage-light--right" aria-hidden="true" />
          <div className="an-stage-floor" aria-hidden="true" />

          <div className="an-stage-screen">
            <div className="an-stage-screen__top">
              <span>Now Presenting</span>
              <strong>{activeSlide?.awardLabel}</strong>
            </div>

            <StageCopy slide={activeSlide} phase={phase} />

            {phase === PHASES.INTRO ? (
              <div className="an-intro-controls">
                <button type="button" className="an-gold-btn" onClick={beginShow}>
                  Begin Show
                </button>
                <button type="button" className="an-glass-btn" onClick={skipWholeShow}>
                  Skip Entire Show
                </button>
              </div>
            ) : (
              <>
                <NomineeLineup slide={activeSlide} phase={phase} />
                <Presenter phase={phase} slide={activeSlide} />
                <WinnerMoment slide={activeSlide} phase={phase} />
              </>
            )}
          </div>
        </section>

        <LiveFanFeed tweets={liveTweets} spoilerMode={!awardRevealed} />
      </main>

      <footer className="an-show-footer">
        <div className="an-milestones">
          {SEASON_MILESTONES.map((milestone) => (
            <span key={milestone.id} className={milestone.id === "awards" ? "is-current" : ""}>
              {milestone.label}
            </span>
          ))}
        </div>

        <div className="an-footer-controls">
          <button type="button" className="an-dark-btn" disabled={phase === PHASES.INTRO} onClick={revealNow}>
            Reveal Now
          </button>
          <button type="button" className="an-dark-btn" onClick={nextAward}>
            Skip Award
          </button>
          <button
            type="button"
            className="an-gold-btn"
            onClick={phase === PHASES.INTRO ? beginShow : phase === PHASES.WINNER ? nextAward : advancePhase}
          >
            {phase === PHASES.INTRO ? "Start Ceremony" : phase === PHASES.WINNER ? "Next Award" : "Continue"}
          </button>
        </div>
      </footer>
    </section>
  );
}
