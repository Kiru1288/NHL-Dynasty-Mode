import React, { useMemo } from "react";
import {
  buildTickerItems,
  deriveTickerStateLabel,
  normalizeTickerGames,
} from "./tickerGames";
import "./BroadcastTicker.css";

function useReducedMotion() {
  return useMemo(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);
}

/**
 * ESPN-style broadcast crawl. Reads schedule/results from franchise payload.
 * @param {object} props
 * @param {object} [props.payload] - franchise / sim state
 * @param {object[]} [props.games] - pre-normalized games (optional)
 * @param {string} [props.label] - left bug label, e.g. NHL SIM
 * @param {string} [props.stateLabel] - optional second bug override
 * @param {boolean} [props.compact]
 */
export default function BroadcastTicker({
  payload,
  games: gamesProp,
  label = "NHL SIM",
  stateLabel,
  compact = false,
}) {
  const reducedMotion = useReducedMotion();

  const normalizedGames = useMemo(() => {
    if (Array.isArray(gamesProp) && gamesProp.length) return gamesProp;
    return normalizeTickerGames(payload);
  }, [gamesProp, payload]);

  const items = useMemo(() => buildTickerItems(normalizedGames), [normalizedGames]);

  const stateBug = stateLabel || deriveTickerStateLabel(normalizedGames);

  const loopItems = useMemo(() => {
    if (reducedMotion || items.length <= 1) return items;
    return [...items, ...items];
  }, [items, reducedMotion]);

  const isStatic = reducedMotion || items.length <= 2;

  return (
    <footer
      className={`broadcast-ticker${compact ? " is-compact" : ""}${isStatic ? " is-static" : ""}`}
      aria-label="Broadcast ticker"
      role="marquee"
    >
      <div className="broadcast-ticker__bug">{label}</div>
      {stateBug ? <div className="broadcast-ticker__state">{stateBug}</div> : null}
      <div className="broadcast-ticker__viewport">
        <div className="broadcast-ticker__track">
          {loopItems.map((item, index) => (
            <React.Fragment key={`${item.id}-${index}`}>
              {index > 0 ? <span className="broadcast-ticker__separator" aria-hidden="true" /> : null}
              <span className="broadcast-ticker__item" title={item.text}>
                {item.text}
              </span>
            </React.Fragment>
          ))}
        </div>
      </div>
    </footer>
  );
}
