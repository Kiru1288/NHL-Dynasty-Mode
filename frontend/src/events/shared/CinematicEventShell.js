import React, { useCallback, useEffect, useState } from "react";
import { buildCinematicCss } from "./eventHelpers";

/**
 * Shared cinematic layout matching PlayoffStartMenu design language.
 */
export default function CinematicEventShell({
  prefix = "evt",
  phaseLabel = "EVENT",
  phaseStyle = "pill",
  seasonLabel = "",
  title = "EVENT",
  titleVariant = "",
  eyebrow = "",
  hideTitle = false,
  hideEyebrow = false,
  heroContent = null,
  railTitle = "Details",
  railContent = null,
  railHint = null,
  tickerItems = [],
  tickerInteractive = false,
  activeTicker = null,
  onTickerSelect = null,
  hideTicker = false,
  ctaLabel = "Continue",
  onContinue,
  onBack,
  revealDelayMs = 0,
  revealKey,
  persistRevealKey = null,
  showSkip = true,
  hubLabel = "Hub",
  footerAlign = "center",
  rootClassName = "",
  register = "ops",
}) {
  const readPersisted = () => {
    if (!persistRevealKey || typeof window === "undefined") return false;
    try {
      return window.sessionStorage.getItem(persistRevealKey) === "1";
    } catch {
      return false;
    }
  };

  const skipCeremony = Number(revealDelayMs) <= 0;
  const [introDone, setIntroDone] = useState(() => skipCeremony || readPersisted());
  const css = buildCinematicCss(prefix);
  const p = prefix;
  const revealId = revealKey ?? title;

  const markRevealed = useCallback(() => {
    setIntroDone(true);
    if (!persistRevealKey || typeof window === "undefined") return;
    try {
      window.sessionStorage.setItem(persistRevealKey, "1");
    } catch {
      /* ignore */
    }
  }, [persistRevealKey]);

  const skipReveal = useCallback(() => markRevealed(), [markRevealed]);

  useEffect(() => {
    if (skipCeremony || (persistRevealKey && readPersisted())) {
      setIntroDone(true);
      return undefined;
    }
    setIntroDone(false);
    const t = window.setTimeout(() => markRevealed(), revealDelayMs);
    return () => window.clearTimeout(t);
  }, [revealId, revealDelayMs, persistRevealKey, markRevealed, skipCeremony]);

  const items = tickerItems.length ? tickerItems : ["FRANCHISE MODE"];
  const registerClass =
    register === "office" ? "register-office" : register === "shell" ? "register-shell" : "register-ops";
  const rootClasses = [
    `${p}-root`,
    registerClass,
    titleVariant ? `${p}-root--${titleVariant}` : "",
    rootClassName,
  ]
    .filter(Boolean)
    .join(" ");
  const titleClasses = [`${p}-title`, titleVariant ? `${p}-title--${titleVariant}` : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={rootClasses} data-register={register}>
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
        {phaseStyle === "pill" ? (
          <div className={`${p}-status-pill`}>{phaseLabel}</div>
        ) : phaseStyle === "text" ? (
          <div className={`${p}-phase-text`}>{phaseLabel}</div>
        ) : (
          <div className={`${p}-phase-spacer`} aria-hidden />
        )}
        <div className={`${p}-season`}>{seasonLabel || ""}</div>
      </header>

      <main className={`${p}-stage`}>
        <section className={`${p}-reveal`}>
          {!hideEyebrow && eyebrow ? <p className={`${p}-eyebrow`}>{eyebrow}</p> : null}
          {!hideTitle ? <h1 className={titleClasses}>{title}</h1> : null}
          {heroContent}
        </section>
        <aside className={`${p}-panel`}>
          <h2>{railTitle}</h2>
          <div className={`${p}-rail`}>{railContent}</div>
          {railHint ? <p className={`${p}-rail-hint`}>{railHint}</p> : null}
        </aside>
      </main>

      <footer className={`${p}-footer${hideTicker ? " is-compact" : ""}`}>
        {!hideTicker ? (
          <div className={`${p}-ticker`}>
            <div className={`${p}-ticker-track`}>
              <div className={`${p}-ticker-group`}>
                {items.map((item) =>
                  tickerInteractive ? (
                    <button
                      key={item}
                      type="button"
                      className={`${p}-ticker-tab${activeTicker === item ? " is-active" : ""}`}
                      onClick={() => onTickerSelect?.(item)}
                    >
                      {item}
                    </button>
                  ) : (
                    <span key={item}>{item}</span>
                  )
                )}
              </div>
            </div>
          </div>
        ) : null}
        <div className={`${p}-footer-actions ${footerAlign === "split" ? "is-split" : ""}`}>
          {showSkip && !introDone && !skipCeremony ? (
            <button type="button" className={`${p}-skip-btn`} onClick={skipReveal}>
              Skip cinematic
            </button>
          ) : null}
          <button
            type="button"
            className={`${p}-cta-btn`}
            onClick={() => {
              if (!introDone && !skipCeremony) markRevealed();
              onContinue?.();
            }}
            disabled={typeof onContinue !== "function"}
            title={typeof onContinue !== "function" ? "Resolve required decisions to continue" : undefined}
          >
            {ctaLabel}
          </button>
        </div>
      </footer>
    </section>
  );
}
