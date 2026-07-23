import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getFranchiseSessionId } from "../../../services/api";
import {
  buildAwardsFanTweets,
  normalizeAwardsPayload,
} from "../../../events/awardsNight/awardHelpers";

const DEV = process.env.NODE_ENV === "development";

const TONE_ORDER = ["homer", "skeptic", "stat nerd", "rival fan", "chaos fan", "analyst", "main-feed"];

function pick(obj, snake, camel, fallback = "") {
  if (!obj || typeof obj !== "object") return fallback;
  const snakeVal = obj[snake];
  if (snakeVal !== undefined && snakeVal !== null && snakeVal !== "") return snakeVal;
  const camelVal = obj[camel];
  if (camelVal !== undefined && camelVal !== null && camelVal !== "") return camelVal;
  return fallback;
}

function normalizeFan(raw = {}) {
  return {
    id: pick(raw, "id", "id", ""),
    displayName: pick(raw, "display_name", "displayName", "Rink Watcher"),
    handle: pick(raw, "handle", "handle", "@puckwatch117"),
    avatarSrc: pick(raw, "avatar_src", "avatarSrc", ""),
    persona: pick(raw, "persona", "persona", "fan"),
    market: pick(raw, "market", "market", "League Feed"),
    nat: pick(raw, "nat", "nat", ""),
  };
}

function normalizeTweet(raw = {}) {
  const fan = normalizeFan(raw.fan || {});
  const context = raw.context || {};

  return {
    id: pick(raw, "id", "id", "tweet-fallback"),
    type: pick(raw, "type", "type", "award_reaction"),
    awardKey: pick(raw, "award_key", "awardKey", ""),
    awardLabel: pick(raw, "award_label", "awardLabel", "Awards Night"),
    winnerLabel: pick(raw, "winner_label", "winnerLabel", ""),
    winnerTeamName: pick(raw, "winner_team_name", "winnerTeamName", ""),
    text: String(raw.text || "").trim(),
    tone: pick(raw, "tone", "tone", fan.persona || "reaction"),
    createdAtLabel: pick(raw, "created_at_label", "createdAtLabel", "now"),
    fan,
    metrics: {
      replies: Number(raw.metrics?.replies || 0),
      reposts: Number(raw.metrics?.reposts || 0),
      quotes: Number(raw.metrics?.quotes || 0),
      likes: Number(raw.metrics?.likes || 0),
    },
    context: {
      topStat: pick(context, "top_stat", "topStat", ""),
      runnerUp: pick(context, "runner_up", "runnerUp", ""),
      legacy: pick(context, "legacy", "legacy", ""),
    },
  };
}

function fanInitials(name = "") {
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "RW";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0] || ""}${parts[1][0] || ""}`.toUpperCase();
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function buildLocalTweets({ awards, franchiseState, eventData, maxTweets, seed, awardKey }) {
  const normalized =
    Array.isArray(awards) && awards.length
      ? awards
      : normalizeAwardsPayload(franchiseState || {}, eventData || {});

  const scopedAwards = awardKey
    ? normalized.filter((a) => a.awardKey === awardKey)
    : normalized;

  return buildAwardsFanTweets(scopedAwards.length ? scopedAwards : normalized, {
    maxTweets,
    tweetsPerAward: awardKey ? 4 : 2,
    seed: awardKey ? `${seed}:${awardKey}` : seed || "awards-night-local",
    includeSummaryTweets: !awardKey,
  }).map((tweet) =>
    normalizeTweet({
      id: tweet.id,
      type: tweet.type,
      award_key: tweet.awardKey,
      award_label: tweet.awardLabel,
      winner_label: tweet.context?.winnerLabel || tweet.winnerLabel || "",
      winner_team_name: tweet.context?.winnerTeamName || tweet.winnerTeamName || "",
      text: tweet.text,
      tone: tweet.tone,
      created_at_label: tweet.createdAtLabel,
      fan: {
        display_name: tweet.fan?.displayName || tweet.displayName,
        handle: tweet.fan?.handle || tweet.handle,
        avatar_src: tweet.fan?.avatarSrc || tweet.avatarSrc,
        persona: tweet.fan?.persona || tweet.persona,
        market: tweet.fan?.market || tweet.market,
      },
      metrics: tweet.metrics,
      context: tweet.context,
    })
  );
}

function buildEndpoint(sessionId, { endpoint, maxTweets, seed, eventType, awardKey }) {
  if (endpoint) return endpoint;
  const params = new URLSearchParams();
  params.set("count", String(maxTweets));
  if (seed) params.set("seed", seed);
  if (awardKey) params.set("award_key", awardKey);
  if (eventType && eventType !== "awards") params.set("event_type", eventType);
  return `/api/franchise/${encodeURIComponent(sessionId)}/fan-reactions/awards?${params.toString()}`;
}

function pickVisibleTweets(tweets, { awardKey, visibleCount }) {
  let pool = [...safeArray(tweets)];
  if (awardKey) {
    const scoped = pool.filter((t) => t.awardKey === awardKey);
    if (scoped.length) pool = scoped;
  }

  const byTone = new Map();
  pool.forEach((tweet) => {
    const key = tweet.tone || tweet.fan.persona || "fan";
    if (!byTone.has(key)) byTone.set(key, tweet);
  });

  const diverse = [];
  TONE_ORDER.forEach((tone) => {
    if (byTone.has(tone)) diverse.push(byTone.get(tone));
  });
  pool.forEach((tweet) => {
    if (!diverse.includes(tweet)) diverse.push(tweet);
  });

  return diverse.slice(0, Math.max(1, Number(visibleCount) || 4));
}

function FanTweetCard({ tweet, compact = false }) {
  return (
    <article
      className={`an-fan-tweet${compact ? " is-compact" : ""}`}
      data-award-key={tweet.awardKey || ""}
      data-tone={tweet.tone || ""}
    >
      <div className="an-fan-tweet__avatar">
        {tweet.fan.avatarSrc ? (
          <img src={tweet.fan.avatarSrc} alt="" loading="lazy" />
        ) : (
          <span className="an-fan-tweet__initials">{fanInitials(tweet.fan.displayName)}</span>
        )}
      </div>

      <div className="an-fan-tweet__content">
        <div className="an-fan-tweet__top">
          <strong className="an-fan-tweet__name">{tweet.fan.displayName}</strong>
          <span className="an-fan-tweet__handle">{tweet.fan.handle}</span>
          <span className="an-fan-tweet__time">{tweet.createdAtLabel}</span>
        </div>

        <p className="an-fan-tweet__text">{tweet.text}</p>

        <div className="an-fan-tweet__context">
          {tweet.awardLabel ? <span className="an-fan-tweet__tag">{tweet.awardLabel}</span> : null}
          {tweet.context?.topStat ? (
            <span className="an-fan-tweet__tag is-muted">{tweet.context.topStat}</span>
          ) : null}
        </div>

        <div className="an-fan-tweet__metrics" aria-label="Engagement metrics">
          <span className="an-fan-tweet__metric">
            <span className="an-fan-tweet__metric-value">{tweet.metrics.likes}</span>
            <span className="an-fan-tweet__metric-label">likes</span>
          </span>
        </div>
      </div>
    </article>
  );
}

export default function FanReactionFeed({
  sessionId: sessionIdProp,
  awards = null,
  eventData = {},
  franchiseState = {},
  endpoint = "",
  eventType = "awards",
  context = null,
  reactions = null,
  awardKey = "",
  placement = "bottom-left",
  enabled = true,
  intervalMs = 5500,
  maxTweets = 18,
  visibleCount = 4,
  className = "",
  feedLabel = "Fan Feed",
  feedSubLabel = "Awards Night",
  onEnabledChange,
}) {
  const [tweets, setTweets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [batchIndex, setBatchIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const warnedRef = useRef(false);

  const sessionId = sessionIdProp || getFranchiseSessionId() || "";
  const seed = useMemo(() => {
    const y = franchiseState?.season_year || franchiseState?.seasonYear;
    return y ? `${y}-awards-night` : "awards-night";
  }, [franchiseState]);

  const localFallbackTweets = useMemo(
    () =>
      buildLocalTweets({
        awards,
        franchiseState,
        eventData,
        maxTweets,
        seed,
        awardKey,
      }),
    [awards, franchiseState, eventData, maxTweets, seed, awardKey]
  );

  useEffect(() => {
    if (!enabled) {
      setTweets([]);
      setLoading(false);
      if (typeof onEnabledChange === "function") onEnabledChange(false);
      return undefined;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);

      if (Array.isArray(reactions) && reactions.length) {
        if (!cancelled) {
          setTweets(reactions.map(normalizeTweet));
          setBatchIndex(0);
          setLoading(false);
          if (typeof onEnabledChange === "function") onEnabledChange(true);
        }
        return;
      }

      if (!sessionId) {
        if (!cancelled) {
          setTweets(localFallbackTweets);
          setBatchIndex(0);
          setLoading(false);
          if (typeof onEnabledChange === "function") onEnabledChange(localFallbackTweets.length > 0);
        }
        return;
      }

      try {
        const url = buildEndpoint(sessionId, { endpoint, maxTweets, seed, eventType, awardKey });
        const { data } = await api.get(url, { timeout: 8000 });
        const incoming = Array.isArray(data?.tweets) ? data.tweets.map(normalizeTweet) : [];
        if (!cancelled) {
          setTweets(incoming.length ? incoming : localFallbackTweets);
          setBatchIndex(0);
          if (typeof onEnabledChange === "function") {
            onEnabledChange((incoming.length ? incoming : localFallbackTweets).length > 0);
          }
        }
      } catch (error) {
        if (DEV && !warnedRef.current) {
          console.warn("[FanReactionFeed] backend feed unavailable, using local fallback.", error);
          warnedRef.current = true;
        }
        if (!cancelled) {
          setTweets(localFallbackTweets);
          setBatchIndex(0);
          if (typeof onEnabledChange === "function") onEnabledChange(localFallbackTweets.length > 0);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [
    enabled,
    sessionId,
    endpoint,
    maxTweets,
    seed,
    eventType,
    awardKey,
    reactions,
    localFallbackTweets,
    onEnabledChange,
  ]);

  const visibleTweets = useMemo(
    () => pickVisibleTweets(tweets, { awardKey, visibleCount }),
    [tweets, awardKey, visibleCount]
  );

  const rotateBatches = visibleCount > 1 && intervalMs > 0 && tweets.length > visibleCount;

  useEffect(() => {
    if (!enabled || paused || !rotateBatches) return undefined;
    const timer = window.setInterval(() => {
      setBatchIndex((index) => (index + 1) % Math.ceil(tweets.length / visibleCount));
    }, Math.max(4000, Number(intervalMs) || 5500));
    return () => window.clearInterval(timer);
  }, [enabled, paused, rotateBatches, tweets.length, visibleCount, intervalMs]);

  const displayTweets = useMemo(() => {
    if (!rotateBatches) return visibleTweets;
    const start = batchIndex * visibleCount;
    return tweets.slice(start, start + visibleCount);
  }, [rotateBatches, visibleTweets, batchIndex, visibleCount, tweets]);

  const handleMouseEnter = useCallback(() => setPaused(true), []);
  const handleMouseLeave = useCallback(() => setPaused(false), []);
  const handleFocus = useCallback(() => setPaused(true), []);
  const handleBlur = useCallback((e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setPaused(false);
  }, []);

  if (!enabled) return null;

  const placementClass =
    placement === "bottom-left"
      ? "an-social-feed--bottom-left"
      : placement === "bottom-center"
        ? "an-social-feed--bottom-center"
        : `an-social-feed--${placement}`;

  return (
    <div
      className={`an-social-feed ${placementClass} ${className}`.trim()}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleFocus}
      onBlur={handleBlur}
      tabIndex={0}
      aria-live="polite"
      aria-label="Fan reaction feed"
    >
      <div className="an-social-feed__shell">
        <div className="an-social-feed__header">
          <span className="an-social-feed__dot" aria-hidden="true" />
          <span className="an-social-feed__label">{feedLabel}</span>
          <span className="an-social-feed__sub">{feedSubLabel}</span>
        </div>

        <div className="an-social-feed__stack an-social-feed__stack--multi">
          {loading ? (
            <FanTweetCard
              tweet={{
                id: "loading",
                fan: { displayName: "Loading feed", handle: "", avatarSrc: "", persona: "fan", market: "" },
                text: "Fans are lining up to react…",
                createdAtLabel: "now",
                metrics: { likes: 0 },
                awardLabel: "",
                context: {},
              }}
              compact
            />
          ) : (
            displayTweets.map((tweet) => <FanTweetCard key={tweet.id} tweet={tweet} compact />)
          )}
        </div>
      </div>
    </div>
  );
}
