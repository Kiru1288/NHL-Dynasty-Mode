import React, { useCallback, useEffect, useState } from "react";
import { buildCinematicCss } from "./eventHelpers";

/**
 * Shared cinematic layout matching PlayoffStartMenu design language.
 */
export default function CinematicEventShell({
  prefix = "evt",
  phaseLabel = "EVENT",
  seasonLabel = "",
  title = "EVENT",
  eyebrow = "",
  heroContent = null,
  railTitle = "Details",
  railContent = null,
  tickerItems = [],
  ctaLabel = "Continue",
  onContinue,
  onBack,
  revealDelayMs = 1200,
  revealKey,
  showSkip = true,
  hubLabel = "Hub World",
}) {
  const [introDone, setIntroDone] = useState(false);
  const css = buildCinematicCss(prefix);
  const p = prefix;
  const revealId = revealKey ?? title;

  const skipReveal = useCallback(() => setIntroDone(true), []);

  useEffect(() => {
    setIntroDone(false);
    const t = window.setTimeout(() => setIntroDone(true), revealDelayMs);
    return () => window.clearTimeout(t);
  }, [revealId, revealDelayMs]);

  const items = tickerItems.length ? tickerItems : ["FRANCHISE MODE"];

  return (
    <section className={`${p}-root`}>
      <style>{css}</style>
      <div className={`${p}-bg`}>
        <div className={`${p}-bg-scrim`} />
        <div className={`${p}-bg-noise`} />
        <div className={`${p}-spotlight`} />
      </div>

      <header className={`${p}-topbar`}>
        <div className={`${p}-topbar-left`}>
          <button type="button" onClick={onBack} className={`${p}-ghost-btn`}>
            ← {hubLabel}
          </button>
        </div>
        <div className={`${p}-status-pill`}>{phaseLabel}</div>
        <div className={`${p}-season`}>{seasonLabel || ""}</div>
      </header>

      <main className={`${p}-stage`}>
        <section className={`${p}-reveal`}>
          {eyebrow ? <p className={`${p}-eyebrow`}>{eyebrow}</p> : null}
          <h1 className={`${p}-title`}>{title}</h1>
          {heroContent}
        </section>
        <aside className={`${p}-panel`}>
          <h2>{railTitle}</h2>
          <div className={`${p}-rail`}>{railContent}</div>
        </aside>
      </main>

      <footer className={`${p}-footer`}>
        <div className={`${p}-ticker`}>
          <div className={`${p}-ticker-track`}>
            <div className={`${p}-ticker-group`}>
              {items.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
        </div>
        <div className={`${p}-footer-actions`}>
          {typeof onBack === "function" ? (
            <button type="button" className={`${p}-ghost-btn`} onClick={onBack}>
              Leave to Hub
            </button>
          ) : null}
          {showSkip && !introDone ? (
            <button type="button" className={`${p}-skip-btn`} onClick={skipReveal}>
              Skip Reveal
            </button>
          ) : null}
          <button
            type="button"
            className={`${p}-cta-btn`}
            onClick={onContinue}
            disabled={!introDone || typeof onContinue !== "function"}
          >
            {introDone ? ctaLabel : "Revealing…"}
          </button>
        </div>
      </footer>
    </section>
  );
}
