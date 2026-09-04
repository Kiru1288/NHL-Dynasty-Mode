import React, { useMemo, useState, useCallback, useEffect, useRef } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { getFranchiseSessionId } from "../services/api";
import {
  activeBreakingAlerts,
  breakingAlertKey,
  readDismissedBreakingKeys,
  writeDismissedBreakingKeys,
} from "../utils/breakingAlerts";
import {
  resolvePlayerMeeting,
  startPlayerMeeting,
  advancePlayerMeeting,
  getPlayerMeetingDetail,
  getSocialFeed,
} from "../services/franchiseService";
import BurnerPanel from "../components/franchise/social/BurnerPanel";
import { collectLockerPulse, buildHubStoryTicker, isRoutineLeagueTrade } from "../utils/lockerRoomPulse";

/*
  StorylinesScreen — franchise narrative command center.

  Data rules (unchanged): read franchiseState only. No fabricated storylines,
  no invented metrics. Every number on screen traces back to a backend field.

  Presentation: broadcast control room. Lead story gets the stage, decisions
  get weight, heat is something you feel before you read it, and opening a
  story opens a case file — not an accordion row.
*/

const SORT_OPTIONS = [
  { id: "decisions", label: "Decisions First" },
  { id: "heat", label: "Heat" },
  { id: "priority", label: "Priority" },
  { id: "latest", label: "Latest" },
];

const CATEGORY_META = {
  performance: { glyph: "◆", label: "Performance", accent: "#13d8e7" },
  star_underperforming: { glyph: "◆", label: "Performance", accent: "#13d8e7" },
  rookie_breakout: { glyph: "◆", label: "Performance", accent: "#13d8e7" },
  hot_streak: { glyph: "◆", label: "Performance", accent: "#13d8e7" },
  injury: { glyph: "✚", label: "Injury", accent: "#ff606d" },
  legal_trouble: { glyph: "§", label: "Conduct", accent: "#ff8a4c" },
  trade: { glyph: "⇄", label: "Trade Wire", accent: "#c992ff" },
  rumor: { glyph: "⇄", label: "Trade Wire", accent: "#c992ff" },
  draft: { glyph: "★", label: "Draft", accent: "#e9a83c" },
  goalie: { glyph: "◎", label: "Crease", accent: "#8ab4ff" },
  contract: { glyph: "$", label: "Contract", accent: "#52df94" },
  team_crisis: { glyph: "!", label: "Team Crisis", accent: "#ff8a4c" },
  rivalry: { glyph: "⚔", label: "Rivalry", accent: "#ff606d" },
  decision: { glyph: "◈", label: "GM Decision", accent: "#e9a83c" },
  league: { glyph: "◉", label: "League", accent: "#8ab4ff" },
  locker_room: { glyph: "◍", label: "Locker Room", accent: "#7ee0b0" },
  personal_life: { glyph: "♡", label: "Off ice", accent: "#d4a0c8" },
  business: { glyph: "$", label: "Business", accent: "#52df94" },
  management: { glyph: "▣", label: "Front Office", accent: "#e9a83c" },
  storyline: { glyph: "◉", label: "League", accent: "#8096a8" },
};

const FILTERS = [
  { id: "all", label: "All" },
  { id: "breaking", label: "Breaking" },
  { id: "major", label: "Major" },
  { id: "team", label: "Team" },
  { id: "league", label: "League" },
  { id: "player", label: "Player" },
  { id: "life", label: "Off ice" },
  { id: "media_buzz", label: "Buzz" },
];

const TRADE_FILTERS = [
  { id: "all", label: "All" },
  { id: "breaking", label: "Breaking" },
  { id: "major", label: "Major" },
  { id: "rumors", label: "Rumors" },
  { id: "team", label: "Your club" },
  { id: "league", label: "League" },
];

const FILTER_EMPTY = {
  breaking: "No breaking beats on file — sim a few days to fill the wire.",
  major: "No major developments on file.",
  rumors: "Trade wire is calm — for now.",
  team: "Nothing filed on your team today.",
  league: "League desk is quiet.",
  player: "No player-specific beats yet.",
  life: "No off-ice or family beats filed yet.",
  media_buzz: "Media buzz is flat right now.",
};

const DEPARTMENTS = [
  { id: "front_page", label: "Stories", glyph: "◉" },
  { id: "trade_desk", label: "Trades", glyph: "⇄" },
  { id: "player_meetings", label: "Meetings", glyph: "◫" },
  { id: "locker_room", label: "Room & Life", glyph: "◍" },
  { id: "consequences", label: "Fallout", glyph: "⚠" },
  { id: "social", label: "Social", glyph: "◈" },
  { id: "insiders", label: "Insiders", glyph: "◇" },
  { id: "press_room", label: "Press Room", glyph: "▤" },
  { id: "archive", label: "Archive", glyph: "▥" },
];

const PRIORITY_RANK = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };

const DETAIL_TABS = [
  { id: "details", label: "Case Notes" },
  { id: "related", label: "Related" },
  { id: "rumors", label: "Rumor Mill" },
  { id: "history", label: "History" },
];

/* ------------------------------------------------------------------ */
/* primitives                                                          */
/* ------------------------------------------------------------------ */

function asArray(v) {
  return Array.isArray(v) ? v : [];
}
function asObject(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}
function str(v, fallback = "") {
  if (v === null || v === undefined) return fallback;
  return String(v);
}
function userTeamId(state) {
  return str(state?.user_team_id || state?.userTeamId || state?.team_id || "");
}
function teamLabel(state) {
  return str(
    state?.user_team_name || state?.team_name || state?.franchise_team_name || "Your Team"
  );
}
function calendarLabel(state) {
  return str(
    state?.calendar_iso ||
      state?.current_date_iso ||
      state?.date_iso ||
      state?.calendar_day_label ||
      "—"
  );
}

function prettyDate(iso) {
  const s = str(iso);
  if (!/^\d{4}-\d{2}-\d{2}/.test(s)) return s || "—";
  const d = new Date(s.slice(0, 10));
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function resolveCategoryKey(story) {
  const cat = str(story?.category || story?.type || "").toLowerCase();
  if (story?.requiresAction) return "decision";
  if (cat.includes("legal") || cat === "legal_trouble") return "legal_trouble";
  if (cat === "injury" || /injur/i.test(cat)) return "injury";
  if (/draft|prospect/.test(cat)) return "draft";
  if (/goalie|goaltender/.test(cat)) return "goalie";
  if (/trade|rumor|contract/.test(cat)) return /contract/.test(cat) ? "contract" : "trade";
  if (/locker|belong|role/.test(cat)) return "locker_room";
  if (cat === "personal_life" || /life/.test(cat) || String(story?.causeType || story?.raw?.cause_type || "").toUpperCase().includes("LIFE")) {
    return "personal_life";
  }
  if (/business|agent/.test(cat)) return "business";
  if (/coach|gm|captain|ahl/.test(cat)) return "league";
  if (/crisis|collapse|skid/.test(cat)) return "team_crisis";
  if (/rival/.test(cat)) return "rivalry";
  if (/performance|underperform|breakout|streak|star|rookie/.test(cat)) return "performance";
  return cat || "storyline";
}
function categoryMeta(story) {
  const key = resolveCategoryKey(story);
  return CATEGORY_META[key] || CATEGORY_META.storyline;
}

function parseStoryDate(raw) {
  const s = str(raw?.calendar_iso || raw?.date || "");
  if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
    const d = new Date(s.slice(0, 10));
    return Number.isNaN(d.getTime()) ? null : d;
  }
  const n = Number(raw?.calendar_day ?? raw?.date);
  if (Number.isFinite(n) && n > 1000) {
    const d = new Date(n);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  return null;
}

function storyAgeLabel(story, todayIso) {
  const sd = parseStoryDate(story?.raw || story);
  const td = todayIso ? new Date(String(todayIso).slice(0, 10)) : null;
  if (!sd || !td || Number.isNaN(td.getTime())) return story.date || "—";
  const days = Math.floor((td - sd) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  if (days < 14) return "Last week";
  return `${Math.floor(days / 7)}w ago`;
}

function storyFreshnessClass(story, todayIso) {
  const sd = parseStoryDate(story?.raw || story);
  const td = todayIso ? new Date(String(todayIso).slice(0, 10)) : null;
  if (!sd || !td) return "";
  const days = Math.floor((td - sd) / 86400000);
  if (days <= 0) return "is-fresh";
  if (days >= 7) return "is-stale";
  return "";
}

function playerInitials(name) {
  const parts = str(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function arcStage(story) {
  const st = str(story.arcStatus || story.status || "active").toLowerCase();
  if (st === "resolved") return "Resolved";
  if (story.repeatCount > 1 || story.escalatedFrom) return "Escalating";
  if (story.repeatCount === 1) return "Developing";
  return "Active";
}

function deriveFollowUp(story) {
  if (story.followUp) return story.followUp;
  const key = resolveCategoryKey(story);
  if (story.gamesRemaining > 0) {
    return `Player sidelined — projected return ${story.returnEstimate || `in ${story.gamesRemaining} games`}.`;
  }
  if (key === "injury") return "Monitor recovery timeline and lineup availability.";
  if (key === "legal_trouble") return "Monitor league investigation; situation may escalate without resolution.";
  if (key === "performance") return "Watch next game — production trend may confirm or reverse this story.";
  if (key === "trade") return "Trade chatter may intensify if performance does not improve.";
  if (key === "draft") return "Scouts will update board ranks as season progresses.";
  return "Monitor upcoming games for follow-up beats.";
}

function effectPillClass(val) {
  const n = Number(val);
  if (Number.isNaN(n)) return "neutral";
  if (n > 0) return "pos";
  if (n < 0) return "neg";
  return "neutral";
}
function formatEffectLabel(key) {
  return str(key).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatMeetingStat(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return String(Math.round(n));
}

function isPercentStatField(field) {
  const f = str(field).toLowerCase();
  return /morale|trust|satisfaction|confidence|belonging|energy|focus|stress|respect|loyalty|communication|honesty|grievance|coach/.test(f);
}

function formatMeetingReceiptLines(receipts) {
  const raw =
    receipts && typeof receipts === "object" && receipts.receipts && !receipts.profiles
      ? receipts.receipts
      : receipts;
  if (!raw || typeof raw !== "object") return [];
  const lines = [];
  const fmtVal = (field, val) => (isPercentStatField(field) ? formatMeetingStat(val) : val);
  asArray(raw.profiles).forEach((r) => {
    const label = formatEffectLabel(r.field || "profile");
    if (r.before != null && r.after != null) {
      const delta = Number(r.delta);
      const sign = Number.isFinite(delta) && delta > 0 ? "+" : "";
      const before = fmtVal(r.field, r.before);
      const after = fmtVal(r.field, r.after);
      lines.push(
        `${label}: ${before} → ${after}${Number.isFinite(delta) ? ` (${sign}${isPercentStatField(r.field) ? formatMeetingStat(delta) : delta})` : ""}`
      );
    } else if (r.delta != null) {
      const delta = Number(r.delta);
      lines.push(`${label}: ${delta > 0 ? "+" : ""}${isPercentStatField(r.field) ? formatMeetingStat(delta) : delta}`);
    }
  });
  asArray(raw.team).forEach((r) => {
    lines.push(
      `Room ${formatEffectLabel(r.field)}: ${fmtVal(r.field, r.before)} → ${fmtVal(r.field, r.after)}`
    );
  });
  asArray(raw.relationships).forEach((r) => {
    if (r.summary) lines.push(r.summary);
    else if (r.field)
      lines.push(`${formatEffectLabel(r.field)}: ${fmtVal(r.field, r.before)} → ${fmtVal(r.field, r.after)}`);
  });
  asArray(raw.attributes).forEach((r) => {
    if (r.field) lines.push(`${formatEffectLabel(r.field)}: ${r.before} → ${r.after}`);
  });
  asArray(raw.readiness).forEach((r) => {
    if (r.ovr_delta) lines.push(`Readiness: ${r.ovr_delta > 0 ? "+" : ""}${r.ovr_delta} OVR (${r.reason || "meeting"})`);
  });
  if (raw.promise_id) lines.push("Promise logged with the player.");
  return lines;
}

function parseMeetingReceipts(receipts) {
  const raw =
    receipts && typeof receipts === "object" && receipts.receipts && !receipts.profiles
      ? receipts.receipts
      : receipts;
  if (!raw || typeof raw !== "object") return [];
  const cards = [];
  const push = (group, label, before, after, delta) => {
    if (before == null && after == null && delta == null) return;
    const d = Number(delta);
    const tone = Number.isFinite(d) ? (d > 0 ? "pos" : d < 0 ? "neg" : "neutral") : "neutral";
    cards.push({
      group,
      label,
      before: before != null ? (isPercentStatField(label) ? formatMeetingStat(before) : before) : null,
      after: after != null ? (isPercentStatField(label) ? formatMeetingStat(after) : after) : null,
      delta: Number.isFinite(d)
        ? `${d > 0 ? "+" : ""}${isPercentStatField(label) ? formatMeetingStat(d) : d}`
        : null,
      tone,
    });
  };
  asArray(raw.profiles).forEach((r) =>
    push("Player", formatEffectLabel(r.field || "profile"), r.before, r.after, r.delta)
  );
  asArray(raw.team).forEach((r) =>
    push("Locker room", formatEffectLabel(r.field || "room"), r.before, r.after, r.delta)
  );
  asArray(raw.relationships).forEach((r) => {
    if (r.summary) cards.push({ group: "Relationship", label: r.summary, tone: "neutral" });
    else push("Relationship", formatEffectLabel(r.field || "trust"), r.before, r.after, r.delta);
  });
  asArray(raw.attributes).forEach((r) => {
    const label = formatEffectLabel(r.attribute || r.field || "attribute");
    push("Attributes", label, r.before, r.after, r.delta);
  });
  asArray(raw.potential).forEach((r) => {
    push("Potential", r.reason || "Development", r.before, r.after, r.delta);
  });
  asArray(raw.reporter).forEach((r) => {
    push("Media", formatEffectLabel(r.field || "reporter"), r.before, r.after, r.delta);
  });
  asArray(raw.media).forEach((r) => {
    push("Media", formatEffectLabel(r.field || "heat"), r.before, r.after, r.delta);
  });
  asArray(raw.readiness).forEach((r) => {
    if (r.ovr_delta)
      cards.push({
        group: "Readiness",
        label: r.reason || "Meeting impact",
        delta: `${r.ovr_delta > 0 ? "+" : ""}${r.ovr_delta} OVR`,
        tone: Number(r.ovr_delta) > 0 ? "pos" : "neg",
      });
  });
  if (raw.promise_id) cards.push({ group: "Promise", label: "New promise logged with player", tone: "neutral" });
  return cards;
}

function meetingKindLabel(kind) {
  const key = str(kind).toUpperCase();
  const map = {
    PLAYER_MEETING_REQUEST: "Requested a private meeting",
    REQUEST_MORE_ICE: "Wants more ice time",
    REQUEST_PP_TIME: "Wants power-play time",
    REQUEST_STARTING_ROLE: "Wants a defined starting role",
    CONTRACT_CLARITY_REQUEST: "Needs contract clarity",
    DEVELOPMENT_MEETING: "Development check-in",
    WINNING_CONCERN: "Concerned about team direction",
  };
  return map[key] || formatEffectLabel(kind || "Player concern");
}

function MeetingStatBar({ label, value, tone = "neutral" }) {
  const n = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  return (
    <div className={`sl-pm-stat sl-pm-stat--${tone}`}>
      <div className="sl-pm-stat__head">
        <span>{label}</span>
        <strong>{n}</strong>
      </div>
      <div className="sl-pm-stat__track">
        <div className="sl-pm-stat__fill" style={{ width: `${n}%` }} />
      </div>
    </div>
  );
}

function MeetingRelationshipPanel({ relationship }) {
  const rel = relationship || {};
  if (!rel.label && rel.morale == null) return null;
  const tone =
    rel.label === "Strong" || rel.label === "Good"
      ? "good"
      : rel.label === "Strained" || rel.label === "Broken"
        ? "hot"
        : "warm";
  return (
    <div className="sl-pm-rel-panel">
      <div className="sl-pm-rel-panel__head">
        <span>Relationship snapshot</span>
        <strong className={`sl-pm-rel is-${relToneClass(rel.tone)}`}>{str(rel.label || "Neutral")}</strong>
      </div>
      <div className="sl-pm-rel-panel__grid">
        <MeetingStatBar label="Morale" value={rel.morale} tone={tone} />
        <MeetingStatBar label="GM trust" value={rel.gm_trust} tone={tone} />
        <MeetingStatBar label="Role satisfaction" value={rel.role_satisfaction} tone={tone} />
      </div>
      {rel.detail ? <p className="sl-pm-rel-panel__note">{str(rel.detail)}</p> : null}
    </div>
  );
}

function MeetingCausePanel({ reasons = [], title = "Why this meeting" }) {
  if (!reasons.length) return null;
  return (
    <section className="sl-pm-cause">
      <h4>{title}</h4>
      <ul>
        {reasons.map((r, i) => (
          <li key={`${str(r.code)}-${i}`}>
            <span>{str(r.label)}</span>
            {r.value != null ? <em>{formatMeetingStat(r.value)}</em> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function MeetingEffectCards({ receipts }) {
  const cards = parseMeetingReceipts(receipts);
  if (!cards.length) return null;
  return (
    <section className="sl-pm-effects">
      <h4>What changed</h4>
      <div className="sl-pm-effects__grid">
        {cards.map((card, i) => (
          <article key={i} className={`sl-pm-effect sl-pm-effect--${card.tone || "neutral"}`}>
            <span className="sl-pm-effect__group">{str(card.group)}</span>
            <strong>{str(card.label)}</strong>
            {card.before != null && card.after != null ? (
              <p>
                {card.before} → {card.after}
                {card.delta ? <em>{card.delta}</em> : null}
              </p>
            ) : card.delta ? (
              <p>
                <em>{card.delta}</em>
              </p>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function MeetingOutcomePanel({ outcome, onDismiss, kicker = "Meeting resolved" }) {
  if (!outcome) return null;
  const rel = outcome.relationship || outcome.history?.relationship_snapshot;
  const cards = parseMeetingReceipts(outcome.receipts);
  const hasEffects = cards.length > 0;
  const negCount = cards.filter((c) => c.tone === "negative").length;
  const posCount = cards.filter((c) => c.tone === "positive").length;
  const outcomeTone = negCount > posCount ? "negative" : posCount > 0 ? "positive" : "neutral";
  return (
    <div className={`sl-meeting-outcome sl-meeting-outcome--cinematic sl-meeting-outcome--${outcomeTone}`}>
      <div className="sl-meeting-outcome__glow" aria-hidden />
      <p className="sl-meeting-outcome__kicker">{kicker}</p>
      <h4>{str(outcome.message || outcome.choice_label || "Conversation recorded")}</h4>
      {outcome.summary ? <p className="sl-meeting-outcome__summary">{str(outcome.summary)}</p> : null}
      <MeetingRelationshipPanel relationship={rel} />
      <MeetingEffectCards receipts={outcome.receipts} />
      {!hasEffects && !rel?.label && outcome.summary ? (
        <p className="sl-meeting-outcome__note">Check the front page for follow-up coverage.</p>
      ) : null}
      {onDismiss ? (
        <button type="button" className="sl-back" onClick={onDismiss}>
          Continue
        </button>
      ) : null}
    </div>
  );
}

function deriveDataPassBreakdown(story) {
  const raw = story?.raw || story;
  const effects = asObject(story?.effects);
  const entries = Object.entries(effects);
  const isDataPass = str(raw?.source) === "data_storyline_engine";
  if (!isDataPass && !entries.length) return null;
  const ev = asObject(story?.evidence);
  let triggerStat = "";
  let triggerDelta = null;
  if (ev.points != null && ev.expected_points != null) {
    triggerStat = "points vs expected";
    triggerDelta = Number(ev.points) - Number(ev.expected_points);
  } else if (ev.points_per_game != null) {
    triggerStat = "points per game";
    triggerDelta = Number(ev.points_per_game);
  } else if (story.cause) {
    triggerStat = "trigger";
    triggerDelta = story.cause;
  }
  const [effectType, magnitude] = entries.sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))[0] || ["", 0];
  if (!triggerStat && !effectType) return null;
  return { triggerStat, triggerDelta, effectType, magnitude };
}

function StoryEffectBreakdown({ story, compact = false }) {
  const [open, setOpen] = useState(false);
  const breakdown = useMemo(() => deriveDataPassBreakdown(story), [story]);
  if (!breakdown) return null;
  const { triggerStat, triggerDelta, effectType, magnitude } = breakdown;
  return (
    <div className={`sl-effect-breakdown${compact ? " is-compact" : ""}`} onClick={(e) => e.stopPropagation()}>
      <button type="button" className="sl-effect-breakdown__toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide breakdown" : "Show breakdown"}
      </button>
      {open ? (
        <div className="sl-effect-breakdown__body">
          {triggerStat ? (
            <p>
              <em>Trigger</em>{" "}
              {triggerStat}
              {typeof triggerDelta === "number" && Number.isFinite(triggerDelta)
                ? ` Δ${triggerDelta > 0 ? "+" : ""}${triggerDelta.toFixed(1)}`
                : triggerDelta
                  ? `: ${triggerDelta}`
                  : ""}
            </p>
          ) : null}
          {effectType ? (
            <p>
              <em>Effect</em> {formatEffectLabel(effectType)} {Number(magnitude) > 0 ? "+" : ""}
              {magnitude}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function LockerRoomDashboard({ narrativeUniverse, franchiseState, stories = [] }) {
  const hallmark = asObject(narrativeUniverse?.hallmark_panels);
  const culture = asObject(hallmark.room_pulse || narrativeUniverse?.locker_room?.culture);
  const risks = asArray(hallmark.character_risks);
  const leaders = asArray(hallmark.unheralded_leaders);
  const pulse = collectLockerPulse(franchiseState, { limit: 12 });
  const uid = str(franchiseState?.user_team_id);
  const roomWire = stories.filter(
    (s) =>
      s.categoryKey === "personal_life" ||
      s.categoryKey === "locker_room" ||
      /locker|life|room/i.test(str(s.category))
  );
  const lifeStories = pulse.lifeStories.length
    ? pulse.lifeStories
    : roomWire.filter((s) => s.categoryKey === "personal_life").slice(0, 10);
  const recentUniverse = asArray(narrativeUniverse?.recent_universe_events)
    .filter((ev) => !uid || str(ev?.team_id) === uid)
    .slice(-8)
    .reverse();
  const hasCultureData = Object.keys(culture).length > 0;
  const pulseParts = [culture.unity, culture.confidence, culture.belonging].filter((v) => v != null && v !== "");
  const pulseScore = pulseParts.length
    ? Math.round(pulseParts.reduce((sum, v) => sum + Number(v), 0) / pulseParts.length)
    : null;
  const clubLabel = teamLabel(franchiseState);

  if (!hasCultureData && !risks.length && !leaders.length && !pulse.people.length && !lifeStories.length && !recentUniverse.length) {
    return (
      <EmptyPanel
        kicker="Locker room"
        title="No culture data yet"
        body="Advance the calendar to sync locker-room metrics. If you just completed a meeting, refresh or sim a day to reload narrative data."
      />
    );
  }

  return (
    <div className="sl-locker-dash sl-locker-dash--cinematic">
      <header className="sl-locker-dash__head">
        <p className="sl-room__kicker">Culture · {clubLabel}</p>
        <h2>Room and life</h2>
        <p className="sl-room__sub">Locker-room pulse, character profiles, and off-ice beats tied to your roster.</p>
      </header>
      <div className="sl-locker-dash__pulse">
        {pulseScore != null ? (
          <div className="sl-locker-dash__gauge" style={{ "--pulse": `${pulseScore}%` }}>
            <strong>{pulseScore}</strong>
            <span>Room pulse</span>
          </div>
        ) : null}
        <div className="sl-locker-dash__culture">
          {Object.entries(culture).map(([key, val]) => (
            <div key={key} className="sl-num">
              <span>{formatEffectLabel(key)}</span>
              <strong>{Math.round(Number(val) || 0)}</strong>
            </div>
          ))}
        </div>
      </div>

      {lifeStories.length || roomWire.length ? (
        <section className="sl-locker-dash__section">
          <h3>Story beats</h3>
          <div className="sl-locker-dash__cards sl-locker-dash__cards--wire">
            {(lifeStories.length ? lifeStories : roomWire.slice(0, 8)).map((row) => (
              <article key={str(row.id || row.headline)} className="sl-locker-card sl-locker-card--story">
                <span className="sl-locker-card__tag">Off ice</span>
                <strong>{str(row.playerName || row.headline)}</strong>
                <p>{str(row.summary || row.headline)}</p>
              </article>
            ))}
          </div>
        </section>
      ) : (
        <section className="sl-locker-dash__section">
          <h3>Story beats</h3>
          <p className="sl-muted">No life or locker-room stories on the wire yet. After a 30-day sim, restart the backend and refresh — stories backfill automatically.</p>
        </section>
      )}

      {recentUniverse.length ? (
        <section className="sl-locker-dash__section">
          <h3>Recent triggers</h3>
          <div className="sl-locker-dash__cards sl-locker-dash__cards--wire">
            {recentUniverse.map((ev) => (
              <article key={str(ev.id || ev.kind)} className="sl-locker-card sl-locker-card--trigger">
                <span className="sl-locker-card__tag">{formatEffectLabel(ev.kind || ev.type || "event")}</span>
                <strong>{str(ev.player_name || ev.headline || "Roster event")}</strong>
                <p>{str(ev.summary || ev.headline || ev.description || "A character or room trigger fired.")}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {pulse.people.length ? (
        <section className="sl-locker-dash__section">
          <h3>Who they are</h3>
          <div className="sl-locker-dash__cards">
            {pulse.people.slice(0, 12).map((row) => (
              <article key={row.playerId || row.name} className="sl-locker-card sl-locker-card--person">
                <PlayerHeadshot player={{ id: row.playerId, player_id: row.playerId }} size={48} />
                <div className="sl-locker-card__body">
                  <strong>{row.name}</strong>
                  <span className="sl-locker-card__meta">
                    {row.position ? `${row.position} · ` : ""}
                    {row.character || "Profile"}
                  </span>
                  <p>{row.line}</p>
                  <div className="sl-locker-card__chips">
                    {row.chips.map((chip) => (
                      <em key={chip} className="sl-niche-badge">{chip}</em>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
      {risks.length ? (
        <section className="sl-locker-dash__section">
          <h3>Character risks</h3>
          <div className="sl-locker-dash__cards">
            {risks.map((row) => (
              <article key={str(row.player_id)} className="sl-locker-card">
                <strong>{str(row.player_name || "Player")}</strong>
                <span>Disruption risk {Math.round(Number(row.disruption_risk) || 0)}</span>
                {asArray(row.niche_abilities).map((n) => (
                  <em key={str(n.id)} className="sl-niche-badge">{str(n.label || n.id)}</em>
                ))}
              </article>
            ))}
          </div>
        </section>
      ) : null}
      {leaders.length ? (
        <section className="sl-locker-dash__section">
          <h3>Unheralded leaders</h3>
          <div className="sl-locker-dash__cards">
            {leaders.map((row) => (
              <article key={str(row.player_id)} className="sl-locker-card">
                <strong>{str(row.player_name || "Player")}</strong>
                <span>Room value {Math.round(Number(row.room_value) || 0)}</span>
                {asArray(row.niche_abilities).map((n) => (
                  <em key={str(n.id)} className="sl-niche-badge">{str(n.label || n.id)}</em>
                ))}
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

function ConsequencesPanel({ narrativeUniverse, stories, onOpenStory }) {
  const sanctions = asArray(narrativeUniverse?.team_sanctions).filter((s) => s.active !== false);
  const capPenalties = asObject(narrativeUniverse?.cap_penalties);
  const forfeited = asArray(narrativeUniverse?.forfeited_picks);
  const availability = asObject(narrativeUniverse?.player_availability);
  const budget = asObject(narrativeUniverse?.major_event_budget);

  if (!sanctions.length) {
    return (
      <EmptyPanel
        kicker="League fallout"
        title="No active sanctions"
        body="Major conduct incidents and cap violations will appear here with fines, pick forfeitures, and player availability."
      />
    );
  }

  return (
    <div className="sl-consequences">
      {budget.target ? (
        <p className="sl-muted">
          Major events this season: {budget.generated || 0} / {budget.target}
        </p>
      ) : null}
      {sanctions.map((s) => {
        const sid = str(s.id);
        const capHit = capPenalties[str(s.team_id)] || s.cap_penalty_m;
        const picks = forfeited.filter((p) => str(p.source_event_id) === str(s.source_event_id));
        const linkedStory = stories.find(
          (st) => str(st.causeEventId) === str(s.source_event_id) || str(st.raw?.cause_event_id) === str(s.source_event_id)
        );
        const playerRows = Object.entries(availability).filter(([, row]) =>
          str(row?.source_event_id) === str(s.source_event_id)
        );
        return (
          <article key={sid} className="sl-consequence-card">
            <header>
              <strong>{formatEffectLabel(s.event_type || "sanction")}</strong>
              <span className={`sl-status is-${s.active ? "active" : "resolved"}`}>
                {s.active ? "Active" : "Resolved"}
              </span>
            </header>
            {s.season ? <p>Season {s.season}</p> : null}
            {Number(s.fine_m) > 0 ? <p>Fine: ${Number(s.fine_m).toFixed(2)}M</p> : null}
            {Number(capHit) > 0 ? <p>Cap penalty: ${Number(capHit).toFixed(2)}M</p> : null}
            {picks.length ? (
              <p>
                Forfeited picks:{" "}
                {picks.map((p) => `${p.draft_year} R${p.round}`).join(", ")}
              </p>
            ) : null}
            {playerRows.map(([pid, row]) => (
              <p key={pid}>
                {str(row.player_name || pid)} — {str(row.status || "unavailable")}
              </p>
            ))}
            {linkedStory ? (
              <button type="button" className="sl-link" onClick={() => onOpenStory(linkedStory.id)}>
                View originating story →
              </button>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function sortStories(list, sortId) {
  const copy = [...list];
  const cmp = (a, b) => {
    if (sortId === "decisions") {
      const da = a.requiresAction ? 1 : 0;
      const db = b.requiresAction ? 1 : 0;
      if (da !== db) return db - da;
      if (b.priorityRank !== a.priorityRank) return b.priorityRank - a.priorityRank;
      if (b.heat !== a.heat) return b.heat - a.heat;
      return str(b.date).localeCompare(str(a.date));
    }
    if (sortId === "heat") {
      if (b.heat !== a.heat) return b.heat - a.heat;
      return b.priorityRank - a.priorityRank;
    }
    if (sortId === "priority") {
      if (b.priorityRank !== a.priorityRank) return b.priorityRank - a.priorityRank;
      return b.heat - a.heat;
    }
    return str(b.date).localeCompare(str(a.date));
  };
  copy.sort(cmp);
  return copy;
}

function matchesSearch(story, q) {
  if (!q) return true;
  const needle = q.toLowerCase();
  const hay = [story.headline, story.summary, story.playerName, story.teamName, story.category, story.type]
    .join(" ")
    .toLowerCase();
  return hay.includes(needle);
}

function normalizeStory(raw, idx, state) {
  const uid = userTeamId(state);
  const tid = str(raw?.team_id || raw?.team || "");
  const headline = str(raw?.headline || raw?.title || "").trim();
  const type = str(raw?.type || raw?.category || "storyline").toLowerCase();
  const category = str(raw?.category || type);
  const priority = str(raw?.priority || "MEDIUM").toUpperCase();
  const evidence = asObject(raw?.evidence);
  const effects = asObject(raw?.effects);
  const id = str(raw?.id || raw?.storyline_id || `story-${idx}-${headline.slice(0, 24)}`);

  return {
    id,
    storylineId: str(raw?.storyline_id || raw?.id || id),
    raw,
    headline: headline || "Untitled storyline",
    summary: str(raw?.short_summary || raw?.summary || raw?.description || raw?.details || raw?.text || ""),
    description: str(raw?.description || raw?.details || raw?.summary || ""),
    cause: str(raw?.cause || ""),
    causeType: str(raw?.cause_type || ""),
    causeEventId: str(raw?.cause_event_id || ""),
    sourceLabel: str(raw?.source_label || raw?.source || ""),
    culpritPlayerId: str(raw?.culprit_player_id || ""),
    culpritPlayerName: str(raw?.culprit_player_name || raw?.player_name || ""),
    affectedPlayerIds: asArray(raw?.affected_player_ids),
    recoveryConditions: asArray(raw?.recovery_conditions),
    userVisibleExplanation: str(raw?.user_visible_explanation || ""),
    baseOverall: raw?.base_overall,
    effectiveOverall: raw?.effective_overall,
    impactReason: str(raw?.impact_reason || ""),
    effectSummary: str(raw?.effect_summary || ""),
    effects,
    evidence,
    type,
    category,
    priority,
    priorityRank: PRIORITY_RANK[priority] || 2,
    tone: str(raw?.tone || "neutral"),
    heat: Number(raw?.heat) || 0,
    credibility: Number(raw?.credibility) || 0,
    status: str(raw?.status || "active"),
    source: str(raw?.source || ""),
    date: str(raw?.calendar_iso || raw?.date || ""),
    teamId: tid,
    teamName: str(raw?.team_name || raw?.team_abbrev || tid),
    playerId: str(raw?.player_id || ""),
    playerName: str(raw?.player_name || asArray(raw?.players)[0] || ""),
    playerOverall: raw?.player_overall,
    isUserTeam: Boolean(tid && uid && tid === uid),
    requiresAction: Boolean(raw?.requires_action),
    actionOptions: asArray(raw?.action_options),
    escalatedFrom: str(raw?.escalated_from || ""),
    repeatCount: Number(raw?.repeat_count) || 0,
    gamesRemaining: Number(raw?.games_remaining) || 0,
    returnEstimate: str(raw?.return_estimate || ""),
    returnDate: str(raw?.return_date || ""),
    overallBefore: raw?.overall_before,
    overallAfter: raw?.overall_after,
    overallDelta: raw?.overall_delta,
    severity: str(raw?.severity || "minor"),
    eventTier: str(raw?.event_tier || "minor"),
    effectSummary: str(raw?.effect_summary || ""),
    impactReason: str(raw?.impact_reason || ""),
    impactLines: asArray(raw?.impact_lines),
    followUp: str(raw?.follow_up || ""),
    arcStatus: str(raw?.arc_status || raw?.status || "active"),
    arcId: str(raw?.arc_id || ""),
    beatId: str(raw?.beat_id || ""),
    beatIndex: Number(raw?.beat_index) || 0,
    arcPhase: str(raw?.arc_phase || ""),
    knowledgeType: str(raw?.knowledge_type || ""),
    narrativeAngle: str(raw?.narrative_angle || ""),
    publicKnowledgeLevel: str(raw?.public_knowledge_level || ""),
    gmKnowsMore: Boolean(raw?.gm_knows_more),
    marketTone: str(raw?.market_tone || ""),
    marketDescriptor: str(raw?.market_descriptor || ""),
    breakingLevel: str(raw?.breaking_level || ""),
    pressConferenceId: str(raw?.press_conference_id || ""),
    reporterId: str(raw?.reporter_id || ""),
    reporterName: str(raw?.reporter_name || ""),
    outletName: str(raw?.outlet_name || ""),
    worldEventId: str(raw?.world_event_id || raw?.cause_event_id || ""),
    playerPosition: str(raw?.player_position || ""),
    incidentId: str(raw?.incident_id || ""),
    eligibleToPlay: raw?.eligible_to_play,
    teamCanOverride: Boolean(raw?.team_can_override),
    allegationNote: str(raw?.allegation_note || ""),
    informationStatus: str(raw?.information_status || ""),
    legalStatus: str(raw?.legal_status || ""),
    leagueStatus: str(raw?.league_status || ""),
    teamStatus: str(raw?.team_status || ""),
    conductModel: str(raw?.conduct_model || ""),
    dressBacklashRisk: Number(raw?.dress_backlash_risk) || 0,
    incidentFamily: str(raw?.incident_family || ""),
    fromTeamName: str(raw?.from_team_name || ""),
    toTeamName: str(raw?.to_team_name || ""),
    fromTeamAbbrev: str(raw?.from_team_abbrev || ""),
    toTeamAbbrev: str(raw?.to_team_abbrev || ""),
    relatedTeams: asArray(raw?.related_teams || raw?.teams),
    tradeTeams: asArray(raw?.teams),
    tradeValue: asObject(raw?.trade_value),
    tradeCategory: str(raw?.trade_type_label || raw?.trade_category || ""),
    tradeReason: str(raw?.reason_text || raw?.story_report || ""),
    triggerReason: str(raw?.trigger_reason || raw?.trigger_context?.reason_text || ""),
    triggerReasons: asArray(raw?.trigger_reasons || raw?.trigger_context?.reason_lines),
    triggerContext: asObject(raw?.trigger_context),
    categoryKey: "",
  };
}

function finalizeStory(story, state) {
  story.categoryKey = resolveCategoryKey(story);
  story.ageLabel = storyAgeLabel(story, calendarLabel(state));
  story.freshness = storyFreshnessClass(story, calendarLabel(state));
  return story;
}

function collectStories(state) {
  const rows = asArray(state?.storyline_events);
  if (!rows.length) {
    if (process.env.NODE_ENV === "development") {
      if (!state?.storyline_events) console.warn("[Storylines] storyline_events missing from franchise state");
    }
    return [];
  }
  const seen = new Set();
  const out = [];
  rows.forEach((raw, idx) => {
    const s = finalizeStory(normalizeStory(raw, idx, state), state);
    if (seen.has(s.id)) return;
    seen.add(s.id);
    out.push(s);
  });
  out.sort((a, b) => {
    if (a.requiresAction !== b.requiresAction) return a.requiresAction ? -1 : 1;
    if (b.priorityRank !== a.priorityRank) return b.priorityRank - a.priorityRank;
    if (b.heat !== a.heat) return b.heat - a.heat;
    return 0;
  });
  return out;
}

function buildChoicesMap(state) {
  const map = new Map();
  asArray(state?.storyline_choices).forEach((row) => {
    const sid = str(row?.storyline_id || row?.decision_id || row?.id);
    if (sid) map.set(sid, row);
  });
  return map;
}

function matchesFilter(story, filter) {
  if (filter === "all") return true;
  if (filter === "major") return story.priorityRank >= 3 || story.requiresAction || isBreakingStory(story);
  if (filter === "breaking") return isBreakingStory(story);
  if (filter === "rumors") return isRumourStory(story);
  if (filter === "team") return story.isUserTeam;
  if (filter === "league") return !story.isUserTeam;
  if (filter === "player") return Boolean(story.playerName);
  if (filter === "life") return story.categoryKey === "personal_life" || /life/i.test(str(story.category));
  if (filter === "media_buzz") return Number(story.heat) >= 40;
  return true;
}

function priorityClass(priority) {
  const p = String(priority || "").toUpperCase();
  if (p === "CRITICAL") return "critical";
  if (p === "HIGH") return "high";
  if (p === "LOW") return "low";
  return "medium";
}

function heatLabel(heat) {
  const n = Number(heat);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 20) return "Quiet";
  if (n < 45) return "Building";
  if (n < 75) return "Hot";
  return "Boiling";
}
function heatTier(heat) {
  const n = Number(heat) || 0;
  if (n >= 75) return "boiling";
  if (n >= 45) return "hot";
  if (n >= 20) return "warm";
  return "cool";
}
function formatCount(n) {
  const v = Number(n) || 0;
  if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
  return String(v);
}
function credibilityLabel(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 30) return "Speculation";
  if (n < 50) return "Early chatter";
  if (n < 75) return "Credible";
  return "Strongly sourced";
}
function knowledgeLevelLabel(level) {
  const s = str(level).toLowerCase();
  if (!s) return null;
  if (s === "confirmed") return "Confirmed";
  if (s === "widely_reported") return "Strongly sourced";
  if (s === "rumour") return "Developing rumour";
  if (s === "chatter") return "Early chatter";
  return s.replace(/_/g, " ");
}

function isBreakingStory(story) {
  const p = String(story?.priority || "").toUpperCase();
  const heat = Number(story?.heat) || 0;
  const level = str(story?.breakingLevel).toLowerCase();
  if (level === "breaking" || level === "league_defining") return true;
  if (level === "developing") return false;
  if (p === "CRITICAL" || Boolean(story?.requiresAction)) return true;
  return heat >= 78;
}

function storyScore(story) {
  const heat = Number(story?.heat);
  if (Number.isFinite(heat) && heat > 0) return Math.max(1, Math.min(99, Math.round(heat)));
  const base = { CRITICAL: 90, HIGH: 72, MEDIUM: 50, LOW: 26 };
  return base[str(story?.priority).toUpperCase()] || 50;
}
function scoreTone(score) {
  if (score >= 85) return "crit";
  if (score >= 65) return "high";
  if (score >= 40) return "mid";
  return "low";
}

function isRumourStory(story) {
  return isTradeDeskStory(story);
}

function isTradeDeskStory(story) {
  if (!story) return false;
  const key = str(story.categoryKey || story.category || story.type).toLowerCase();
  if (["personal_life", "locker_room", "injury", "performance"].includes(key)) return false;
  const cause = str(story.causeType || story.raw?.cause_type).toUpperCase();
  if (/TRADE_DEMAND|TRADE_REJECTED|TRADE_PROPOSAL|CULPRIT_TRADED|TRADE_ATTEMPTED/.test(cause)) return true;
  if (key === "trade" || key === "rumor") return true;
  const hay = `${story.category || ""} ${story.type || ""} ${story.headline || ""}`.toLowerCase();
  if (/\bacquires\b|\btraded to\b|trade rumour|trade rumor|trade wire/.test(hay)) return true;
  return /\btrade\b/.test(hay);
}

function parseIsoDate(iso) {
  const s = str(iso);
  if (!/^\d{4}-\d{2}-\d{2}/.test(s)) return null;
  const d = new Date(s.slice(0, 10));
  return Number.isNaN(d.getTime()) ? null : d;
}

function socialPostTimestamp(post) {
  const d = parseIsoDate(post?.calendar_iso || post?.created_at || post?.date);
  return d ? d.getTime() : 0;
}

/** Drop stale social posts — keep only recent timeline (default: last 2 days). */
function filterRecentSocialItems(items, currentIso, maxAgeDays = 2) {
  const today = parseIsoDate(currentIso);
  if (!today) return items;
  const cutoff = today.getTime() - maxAgeDays * 86400000;
  return items.filter((item) => {
    const ts = socialPostTimestamp(item);
    if (!ts) return true;
    return ts >= cutoff;
  });
}

function sortSocialItemsDesc(items) {
  return [...items].sort((a, b) => socialPostTimestamp(b) - socialPostTimestamp(a));
}

function isBrokenSocialPost(text) {
  const raw = str(text);
  if (!raw || raw.length < 8) return true;
  const lower = raw.toLowerCase();
  if (lower.includes("the player")) return true;
  if (/\(\s*0\s*ovr\s*\)/i.test(raw)) return true;
  if (/0 points in 0 games|through 0 gp|0 starts|0\.00 ppg through 0/i.test(lower)) return true;
  if (/\{[a-z_]+\}/.test(raw)) return true;
  return false;
}

function buildSocialPosts(stories, narrativeUniverse, { currentIso = null, maxAgeDays = 2 } = {}) {
  const backendPosts = sortSocialItemsDesc(
    filterRecentSocialItems(asArray(narrativeUniverse?.social_posts), currentIso, maxAgeDays)
  ).filter((p) => !isBrokenSocialPost(p?.text));
  if (backendPosts.length) {
    return backendPosts.slice(0, 40).map((p, idx) => ({
      id: str(p.id || `post-${idx}`),
      handle: str(p.handle || `@User${idx}`),
      name: str(p.author_name || p.name || "Hockey Fan"),
      verified: Boolean(p.verified),
      isAgent: str(p.author_type || "") === "agent",
      agency: str(p.agency || ""),
      age: storyAgeLabel({ raw: p }, currentIso) || prettyDate(p.calendar_iso) || "—",
      text: str(p.text || ""),
      related: str(p.related_headline || ""),
      heat: heatLabel(p.heat),
      cred: p.knowledge_type ? str(p.knowledge_type).replace(/_/g, " ") : null,
      storyId: str(p.storyline_id || ""),
      likes: p.likes,
      reposts: p.reposts,
      replies: p.replies,
      outlet: str(p.outlet || ""),
    }));
  }
  return (Array.isArray(stories) ? stories : []).slice(0, 24).map((s, idx) => {
    const isInsider = /trade|rumor|contract|market/i.test(`${s.type} ${s.category} ${s.headline}`);
    const handle = isInsider
      ? str(s.sourceLabel || s.source || "InsiderDesk").replace(/\s+/g, "")
      : str(s.playerName || s.teamName || "HockeyFan").replace(/\s+/g, "");
    return {
      id: s.id || `post-${idx}`,
      handle: `@${handle}`,
      name: isInsider ? str(s.sourceLabel || s.source || "League Insider") : str(s.playerName || s.teamName || "Fan"),
      verified: isInsider || Boolean(s.playerName),
      age: s.ageLabel || "—",
      text: s.summary || s.headline,
      related: s.headline !== (s.summary || "") ? s.headline : "",
      heat: heatLabel(s.heat),
      cred: credibilityLabel(s.credibility),
      storyId: s.id,
    };
  });
}

function buildRedditThreads(threads, subFilter = "all", { currentIso = null, maxAgeDays = 2 } = {}) {
  const rows = sortSocialItemsDesc(
    filterRecentSocialItems(asArray(threads), currentIso, maxAgeDays)
  ).filter((t) => !isBrokenSocialPost(t?.body) && !isBrokenSocialPost(t?.title));
  const filtered =
    subFilter === "all"
      ? rows
      : rows.filter((t) => str(t.subreddit).toLowerCase() === str(subFilter).toLowerCase());
  return filtered.slice(0, 40).map((t, idx) => ({
    id: str(t.thread_id || `thread-${idx}`),
    subreddit: str(t.subreddit || "r/hockey"),
    title: str(t.title || "Thread"),
    author: str(t.op_author || "u/fan"),
    archetype: str(t.op_archetype || ""),
    flair: str(t.flair || "Discussion"),
    body: str(t.body || ""),
    upvotes: Number(t.upvotes) || 0,
    upvoteRatio: Number(t.upvote_ratio) || 0.75,
    commentCount: Number(t.comment_count) || 0,
    controversial: Number(t.upvote_ratio) < 0.7,
    storyId: str(t.storyline_id || ""),
    playerId: str(t.player_id || ""),
    playerName: str(t.player_name || ""),
    knowledgeType: str(t.knowledge_type || ""),
    comments: asArray(t.top_comments).map((c, ci) => ({
      id: `c-${idx}-${ci}`,
      author: str(c.author || "u/fan"),
      text: str(c.text || ""),
      upvotes: Number(c.upvotes) || 0,
      isRival: Boolean(c.is_rival),
    })),
    heat: heatLabel(t.heat),
    createdAt: str(t.created_at || "—"),
  }));
}

function fanPulseTrend(pulse) {
  const rows = asArray(pulse).slice(-8);
  if (!rows.length) return { net: 0, label: "Flat" };
  const net = rows.reduce((s, r) => s + Number(r.delta || 0), 0);
  if (net > 0.15) return { net, label: "Rising" };
  if (net < -0.15) return { net, label: "Cooling" };
  return { net, label: "Split" };
}

function collectDossiers(narrativeUniverse) {
  const direct = asArray(narrativeUniverse?.player_dossiers);
  if (direct.length) return direct;
  const human = asObject(narrativeUniverse?.human_dossiers);
  const fromHuman = Object.values(human).filter(Boolean);
  if (fromHuman.length) return fromHuman;
  return asArray(narrativeUniverse?.players).map((p) => ({
    player_id: p.player_id,
    player_name: p.player_name,
    identity: p.identity || {},
    wants: p.top_concerns || [],
    trusts: p.trusts || {},
    remembers: asArray(p.memories).slice(-6),
    reputation: asArray(p.reputation_tags).length ? p.reputation_tags : asArray(p.personality_tags),
    personality_tags: asArray(p.personality_tags),
    niches: asArray(p.niche_abilities).map((n) => n.label).filter(Boolean),
    overall: p.overall,
    position: p.position,
  }));
}

function collectArcTimeline(stories, selected, narrativeUniverse) {
  if (!selected) return [];
  const arcId = str(selected.arcId || selected.storylineId || selected.raw?.storyline_id || "");
  const arcs = asArray(narrativeUniverse?.story_arcs);
  const arc = arcs.find((a) => str(a.arc_id) === arcId);
  if (arc && asArray(arc.beats).length) {
    return arc.beats.map((beat) => ({
      id: beat.beat_id,
      date: beat.calendar_iso || "—",
      headline: beat.headline || beat.summary || "Update",
      summary: beat.summary || "",
      knowledgeType: beat.knowledge_type,
    }));
  }
  if (!arcId) return [];
  return stories
    .filter((s) => str(s.arcId || s.storylineId || s.raw?.storyline_id || "") === arcId)
    .sort((a, b) => str(a.date).localeCompare(str(b.date)));
}

function playBreakingSting(level) {
  if (typeof window === "undefined") return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const big = level === "league_defining";
    const now = ctx.currentTime;

    const sub = ctx.createOscillator();
    const subGain = ctx.createGain();
    sub.type = "sine";
    sub.frequency.setValueAtTime(big ? 110 : 90, now);
    sub.frequency.exponentialRampToValueAtTime(big ? 52 : 46, now + 0.5);
    subGain.gain.setValueAtTime(0.09, now);
    subGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.75);
    sub.connect(subGain);
    subGain.connect(ctx.destination);
    sub.start(now);
    sub.stop(now + 0.8);

    const stab = ctx.createOscillator();
    const stabGain = ctx.createGain();
    stab.type = big ? "sawtooth" : "square";
    stab.frequency.setValueAtTime(big ? 880 : 660, now);
    stabGain.gain.setValueAtTime(0.035, now);
    stabGain.gain.exponentialRampToValueAtTime(0.0001, now + (big ? 0.45 : 0.28));
    stab.connect(stabGain);
    stabGain.connect(ctx.destination);
    stab.start(now);
    stab.stop(now + 0.5);
  } catch {
    /* optional audio */
  }
}

function parseTradeTeams(story) {
  const fromA = str(story.fromTeamAbbrev || "");
  const toA = str(story.toTeamAbbrev || "");
  if (fromA && toA) return [fromA, toA];
  const related = asArray(story.relatedTeams);
  if (related.length >= 2) {
    const a = str(related[0]?.abbrev || related[0]?.team_abbrev || related[0] || "");
    const b = str(related[1]?.abbrev || related[1]?.team_abbrev || related[1] || "");
    if (a && b) return [a.slice(0, 4).toUpperCase(), b.slice(0, 4).toUpperCase()];
  }
  const m = String(story.headline || "").match(/\b([A-Z]{2,4})\b.*?\b([A-Z]{2,4})\b/);
  if (m) return [m[1], m[2]];
  const named = String(story.headline || "").match(/([A-Za-z .]+)\s+(?:acquires|trades|sends|gets)\s+.+?\s+(?:from|to)\s+([A-Za-z .]+)/i);
  if (named) {
    const left = named[1].trim().split(/\s+/).slice(-1)[0].slice(0, 3).toUpperCase();
    const right = named[2].trim().split(/\s+/).slice(0, 1)[0].slice(0, 3).toUpperCase();
    if (left && right && left !== right) return [left, right];
  }
  return [];
}

/* ------------------------------------------------------------------ */
/* presentational pieces                                               */
/* ------------------------------------------------------------------ */

function HeatRing({ value, size = 64, label = true }) {
  const v = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  const tier = heatTier(v);
  const r = size / 2 - 5;
  const c = 2 * Math.PI * r;
  const dash = (v / 100) * c;
  return (
    <div className={`sl-ring sl-ring--${tier}`} style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} className="sl-ring__track" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          className="sl-ring__fill"
          strokeDasharray={`${dash} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="sl-ring__center">
        <strong style={{ fontSize: Math.round(size * 0.3) }}>{v}</strong>
        {label ? <span>heat</span> : null}
      </div>
    </div>
  );
}

function HeatSpark({ value }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  const tier = heatTier(v);
  return (
    <div className={`sl-spark sl-spark--${tier}`} title={`Heat ${Math.round(v)}`}>
      <span style={{ width: `${v}%` }} />
    </div>
  );
}

function ScoreBadge({ score, size = "md" }) {
  return (
    <div className={`sl-score sl-score--${scoreTone(score)} sl-score--${size}`}>
      <strong>{score}</strong>
    </div>
  );
}

function StatusPill({ story }) {
  const stage = arcStage(story);
  if (stage === "Active") return null;
  return (
    <span className={`sl-status-pill sl-status-pill--${stage.toLowerCase()}`}>
      {stage === "Escalating" ? <em aria-hidden>▲</em> : null}
      {stage}
    </span>
  );
}

function CategoryTag({ story, size = "sm" }) {
  const meta = categoryMeta(story);
  return (
    <span
      className={`sl-cat sl-cat--${size}`}
      style={{ color: meta.accent, borderColor: `${meta.accent}55`, background: `${meta.accent}14` }}
    >
      <em aria-hidden>{meta.glyph}</em>
      {meta.label}
    </span>
  );
}

function StoryFace({ story, size = 48 }) {
  const abbr = str(story?.teamName || "TEAM").slice(0, 4).toUpperCase();
  const logo =
    resolveFranchiseTeamLogo(
      { team_id: story?.teamId, team_name: story?.teamName, team_abbrev: abbr, abbrev: abbr },
      story?.teamName
    ) || "";
  return (
    <div className="sl-face" style={{ width: size, height: size }}>
      {story?.playerName ? (
        <PlayerHeadshot
          player={{
            name: story.playerName,
            position: story.playerPosition,
            overall: story.playerOverall,
            team_abbrev: abbr,
            team_name: story.teamName,
            ...(asObject(story.raw) || {}),
          }}
          size={size}
        />
      ) : logo ? (
        <img src={logo} alt="" />
      ) : (
        <span>{playerInitials(story?.playerName || abbr)}</span>
      )}
    </div>
  );
}

function TeamMark({ abbrev, name, size = 34 }) {
  const abbr = str(abbrev || name || "TM").slice(0, 4).toUpperCase();
  const logo = resolveFranchiseTeamLogo({ team_abbrev: abbr, abbrev: abbr, team_name: name }, name || abbr) || "";
  return (
    <div className="sl-teammark" style={{ width: size, height: size }}>
      {logo ? <img src={logo} alt="" /> : <strong>{abbr}</strong>}
    </div>
  );
}

function ConductChannels({ story }) {
  const has =
    story.allegationNote ||
    story.informationStatus ||
    story.legalStatus ||
    story.leagueStatus ||
    story.teamStatus ||
    story.eligibleToPlay != null;
  if (!has) return null;
  const eligible = story.eligibleToPlay == null ? null : story.eligibleToPlay ? "Eligible to dress" : "Cannot dress";
  return (
    <section className="sl-conduct">
      <h4>Conduct desk</h4>
      {story.allegationNote ? <p className="sl-conduct__note">{story.allegationNote}</p> : null}
      <div className="sl-conduct__grid">
        {story.informationStatus ? (
          <div><span>Information</span><strong>{formatEffectLabel(story.informationStatus)}</strong></div>
        ) : null}
        {story.legalStatus ? (
          <div><span>Legal</span><strong>{formatEffectLabel(story.legalStatus)}</strong></div>
        ) : null}
        {story.leagueStatus ? (
          <div><span>League</span><strong>{formatEffectLabel(story.leagueStatus)}</strong></div>
        ) : null}
        {story.teamStatus ? (
          <div><span>Team</span><strong>{formatEffectLabel(story.teamStatus)}</strong></div>
        ) : null}
        {eligible ? (
          <div><span>Availability</span><strong>{eligible}</strong></div>
        ) : null}
      </div>
    </section>
  );
}

function TradeSwap({ story }) {
  const pair = parseTradeTeams(story);
  if (pair.length < 2) return null;
  const [a, b] = pair;
  return (
    <div className="sl-swap" aria-label="Trade parties">
      <div className="sl-swap__side">
        <TeamMark abbrev={a} name={story.fromTeamName} size={42} />
        <span>{story.fromTeamName || a}</span>
      </div>
      <div className="sl-swap__mid">
        <em aria-hidden>⇄</em>
        <span>Trade wire</span>
      </div>
      <div className="sl-swap__side">
        <TeamMark abbrev={b} name={story.toTeamName} size={42} />
        <span>{story.toTeamName || b}</span>
      </div>
    </div>
  );
}

function ArcSpine({ beats, fallbackStory }) {
  const nodes =
    beats.length > 1
      ? beats
      : fallbackStory
      ? [
          {
            id: "origin",
            date: fallbackStory.date || fallbackStory.ageLabel,
            headline: "First reported",
            summary: fallbackStory.summary || "",
          },
        ]
      : [];
  if (!nodes.length) return null;
  return (
    <section className="sl-spine">
      <h4>How this developed</h4>
      <ol className="sl-spine__list">
        {nodes.map((n, idx) => (
          <li key={n.id || idx} className={idx === nodes.length - 1 ? "is-latest" : ""}>
            <span className="sl-spine__dot" aria-hidden />
            <time>{prettyDate(n.date)}</time>
            <strong>{n.headline}</strong>
            {n.summary ? <p>{n.summary}</p> : null}
          </li>
        ))}
        <li className="is-next">
          <span className="sl-spine__dot sl-spine__dot--ghost" aria-hidden />
          <time>Next</time>
          <strong>Situation still developing</strong>
        </li>
      </ol>
    </section>
  );
}

function PressureBars({ org }) {
  if (!org) return null;
  const rows = [
    ["owner_confidence", "Owner confidence", false],
    ["fan_approval", "Fan approval", false],
    ["media_heat", "Media pressure", true],
    ["sponsor_confidence", "Sponsor confidence", false],
  ].filter(([key]) => org[key] != null);
  if (!rows.length) return null;
  return (
    <div className="sl-bars">
      {rows.map(([key, label, inverted]) => {
        const pct = Math.round(Number(org[key]) * 100);
        const tone = inverted
          ? pct >= 70
            ? "hot"
            : pct >= 40
            ? "warm"
            : "good"
          : pct >= 60
          ? "good"
          : pct >= 35
          ? "warm"
          : "hot";
        return (
          <div key={key} className="sl-bar">
            <div className="sl-bar__label">
              <span>{label}</span>
              <strong>{pct}</strong>
            </div>
            <div className="sl-bar__track">
              <div
                className={`sl-bar__fill sl-bar__fill--${tone}`}
                style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
              />
            </div>
          </div>
        );
      })}
      {org.revenue_modifier != null ? (
        <p className="sl-bar__foot">Revenue modifier ×{Number(org.revenue_modifier).toFixed(2)}</p>
      ) : null}
    </div>
  );
}

function DossierCard({ dossier, compact = false }) {
  if (!dossier) return null;
  const ident = asObject(dossier.identity);
  const trusts = asObject(dossier.trusts);
  const wants = asArray(dossier.wants);
  const remembers = asArray(dossier.remembers);
  const tags = asArray(dossier.reputation).length ? asArray(dossier.reputation) : asArray(dossier.personality_tags);
  const birth = [ident.birth_city, ident.birth_country].filter(Boolean).join(", ");
  const draft = ident.draft_year
    ? `${ident.draft_year} R${ident.draft_round || "—"} P${ident.draft_pick || "—"}`
    : "";
  return (
    <article className={`sl-dossier${compact ? " is-compact" : ""}`}>
      <div className="sl-dossier__head">
        <strong>{str(dossier.player_name || ident.name || "Player")}</strong>
        <span>
          {str(dossier.position || ident.position || "")}
          {dossier.overall != null ? ` · ${Math.round(Number(dossier.overall))} OVR` : ""}
          {ident.age ? ` · ${ident.age}` : ""}
        </span>
      </div>
      {birth || draft ? (
        <p className="sl-dossier__ident">
          {birth || "Identity on file"}
          {draft ? ` · Draft ${draft}` : ""}
        </p>
      ) : null}
      {tags.length ? (
        <div className="sl-dossier__tags">
          {tags.slice(0, 5).map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>
      ) : null}
      <div className="sl-dossier__grid">
        <div>
          <h4>Wants</h4>
          {wants.length ? (
            wants.slice(0, 3).map((want) => (
              <p key={want.id || want.label}>
                {str(want.label || want.id)}
                {want.pressure != null ? ` · pressure ${want.pressure}` : ""}
              </p>
            ))
          ) : (
            <p>No live concerns published.</p>
          )}
        </div>
        <div>
          <h4>Trusts</h4>
          {Object.keys(trusts).length ? (
            Object.entries(trusts).slice(0, 4).map(([key, val]) => (
              <p key={key}>
                {formatEffectLabel(key)} · {Math.round(Number(val) || 0)}
              </p>
            ))
          ) : (
            <p>Trust ledger still forming.</p>
          )}
        </div>
      </div>
      <div className="sl-dossier__mem">
        <h4>Remembers</h4>
        {remembers.length ? (
          remembers.slice(-4).reverse().map((mem, idx) => (
            <p key={mem.id || idx}>{str(mem.summary || mem.kind || "A private beat")}</p>
          ))
        ) : (
          <p>No memories on the public desk.</p>
        )}
      </div>
    </article>
  );
}

function EmptyPanel({ kicker, title, body }) {
  return (
    <div className="sl-empty">
      <div className="sl-empty__mark" aria-hidden>◉</div>
      <p className="sl-empty__kicker">{kicker}</p>
      <h2>{title}</h2>
      <p className="sl-empty__body">{body}</p>
    </div>
  );
}

function relToneClass(tone) {
  if (tone === "positive") return "is-strong";
  if (tone === "negative") return "is-strained";
  return "is-neutral";
}

function inferChoiceTone(choice = {}) {
  const explicit = str(choice.tone || choice.valence || "").toLowerCase();
  if (explicit && explicit !== "neutral") return explicit;
  const id = str(choice.id || "").toLowerCase();
  const label = str(choice.label || "").toLowerCase();
  const hay = `${id} ${label}`;
  if (/no_comment|deny|firm|hold|cold|challenge|accountability|platoon|deflect|defer|wait|neither/.test(hay)) {
    return "firm";
  }
  if (/support|promise|commit|honest|transparent|welcome|praise|listen|acknowledge|diplomatic|agree/.test(hay)) {
    return "supportive";
  }
  if (/trade|rumor|volatile|changes|overhaul/.test(hay)) return "volatile";
  if (/conditional|cautious|performance|earn/.test(hay)) return "cautious";
  if (/deflect|neither_confirm/.test(hay)) return "deflect";
  return "neutral";
}

function choiceToneClass(tone) {
  const t = str(tone).toLowerCase();
  if (["supportive", "support_player", "diplomatic", "positive", "encouraging"].includes(t)) return "supportive";
  if (["firm", "cold", "negative", "deny", "deflect", "no_comment"].includes(t)) return "firm";
  if (["volatile", "risky", "trade", "hostile"].includes(t)) return "volatile";
  if (["cautious", "conditional", "neutral"].includes(t)) return "cautious";
  return "neutral";
}

function DialogueBubble({ line, index = 0, variant = "default" }) {
  const speaker = str(line?.speaker || "");
  const isGm = speaker === "GM" || speaker === "You";
  const isReporter = /reporter|media|press/i.test(speaker);
  const role = isGm ? "gm" : isReporter ? "reporter" : "player";
  return (
    <div
      className={`sl-bubble sl-bubble--${role} sl-bubble--${variant}`}
      style={{ animationDelay: `${index * 110}ms` }}
    >
      <div className="sl-bubble__tail" aria-hidden />
      <em className="sl-bubble__speaker">{isGm ? "You" : speaker || "Speaker"}</em>
      <p>{str(line?.text)}</p>
    </div>
  );
}

function ResponseChoiceButton({
  choice,
  response,
  className = "sl-choice",
  disabled = false,
  busy = false,
  onClick,
  children,
}) {
  const src = choice || response || {};
  const tone = inferChoiceTone(src);
  const toneClass = choiceToneClass(tone);
  return (
    <button
      type="button"
      className={`${className} sl-choice--tone-${toneClass}`}
      disabled={disabled}
      onClick={onClick}
    >
      <span className="sl-choice__tone-bar" aria-hidden />
      {children || (
        <>
          <strong>{str(src.label)}</strong>
          {src.tone ? <em className={`sl-choice__tone sl-choice__tone--${toneClass}`}>{str(src.tone).replace(/_/g, " ")}</em> : null}
          {src.detail || src.description ? <span>{str(src.detail || src.description)}</span> : null}
          {src.effect_preview ? <span className="sl-choice__effects">{src.effect_preview}</span> : null}
        </>
      )}
      {busy ? <em>Working…</em> : null}
    </button>
  );
}

function extractTradeBoard(story) {
  const raw = story?.raw || story;
  const teams = asArray(raw?.teams);
  const tv = asObject(raw?.trade_value);
  if (teams.length >= 2) {
    const left = teams[0];
    const right = teams[1];
    return {
      left,
      right,
      leftValue: Number(left?.trade_value ?? tv?.left_value ?? 0),
      rightValue: Number(right?.trade_value ?? tv?.right_value ?? 0),
      reason: str(raw?.reason_text || story?.effectSummary || raw?.story_report || ""),
      category: str(raw?.trade_type_label || raw?.trade_category || ""),
    };
  }
  const pair = parseTradeTeams(story);
  if (pair.length < 2) return null;
  return {
    left: { abbreviation: pair[0], display_name: story.fromTeamName || pair[0], acquired_assets: [] },
    right: { abbreviation: pair[1], display_name: story.toTeamName || pair[1], acquired_assets: [] },
    leftValue: Number(tv?.left_value || 0),
    rightValue: Number(tv?.right_value || 0),
    reason: str(raw?.reason_text || story?.summary || ""),
    category: str(raw?.trade_type_label || ""),
  };
}

function TradeAssetChip({ asset, compact = false }) {
  const name = str(asset?.display_name || asset?.player_name || asset?.name || "Asset");
  const ovr = asset?.ovr != null ? Math.round(Number(asset.ovr)) : null;
  const tv = asset?.trade_value != null ? Number(asset.trade_value) : null;
  const pos = str(asset?.position || asset?.role_line || "");
  const isPick = str(asset?.asset_type).toLowerCase().includes("pick");
  return (
    <div className={`sl-trade-asset${compact ? " is-compact" : ""}${isPick ? " is-pick" : ""}`}>
      <div className="sl-trade-asset__head">
        <strong>{name}</strong>
        {ovr != null && !isPick ? <span className="sl-trade-asset__ovr">{ovr} OVR</span> : null}
      </div>
      <div className="sl-trade-asset__meta">
        {pos ? <em>{pos}</em> : null}
        {tv != null ? <span className="sl-trade-asset__tv">TV {tv.toFixed(1)}</span> : null}
        {asset?.cap_hit_m != null ? <span>${Number(asset.cap_hit_m).toFixed(2)}M</span> : null}
      </div>
      {tv != null && !compact ? (
        <div className="sl-trade-asset__bar" aria-hidden>
          <span style={{ width: `${Math.min(100, (tv / 85) * 100)}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function TradeValueMeter({ leftValue = 0, rightValue = 0, leftAbbr = "", rightAbbr = "" }) {
  const lv = Math.max(0, Number(leftValue) || 0);
  const rv = Math.max(0, Number(rightValue) || 0);
  const total = Math.max(lv + rv, 1);
  const leftPct = Math.round((lv / total) * 100);
  const rightPct = 100 - leftPct;
  const delta = Math.abs(lv - rv);
  const fair = delta <= Math.max(4, total * 0.08);
  return (
    <div className="sl-trade-meter" aria-label="Trade value comparison">
      <div className="sl-trade-meter__labels">
        <span>{leftAbbr || "A"} · {lv ? lv.toFixed(1) : "—"}</span>
        <em>{fair ? "Balanced deal" : lv > rv ? `${leftAbbr || "Left"} wins value` : `${rightAbbr || "Right"} wins value`}</em>
        <span>{rightAbbr || "B"} · {rv ? rv.toFixed(1) : "—"}</span>
      </div>
      <div className="sl-trade-meter__track">
        <span className="sl-trade-meter__left" style={{ width: `${leftPct}%` }} />
        <span className="sl-trade-meter__right" style={{ width: `${rightPct}%` }} />
      </div>
    </div>
  );
}

function TradeSummaryPanel({ story, compact = false }) {
  const board = extractTradeBoard(story);
  if (!board) return <TradeSwap story={story} />;
  const leftAbbr = str(board.left?.abbreviation || board.left?.team_abbrev || "").slice(0, 4).toUpperCase();
  const rightAbbr = str(board.right?.abbreviation || board.right?.team_abbrev || "").slice(0, 4).toUpperCase();
  const leftAssets = asArray(board.left?.acquired_assets);
  const rightAssets = asArray(board.right?.acquired_assets);
  return (
    <section className={`sl-trade-board${compact ? " is-compact" : ""}`} aria-label="Trade summary">
      <div className="sl-trade-board__glow" aria-hidden />
      {board.category ? <p className="sl-trade-board__kicker">{board.category}</p> : null}
      <div className="sl-trade-board__teams">
        <div className="sl-trade-board__side">
          <TeamMark abbrev={leftAbbr} name={board.left?.display_name} size={compact ? 34 : 46} />
          <strong>{str(board.left?.display_name || leftAbbr)}</strong>
          <span>Receives</span>
          <div className="sl-trade-board__assets">
            {leftAssets.length ? leftAssets.map((a, i) => <TradeAssetChip key={i} asset={a} compact={compact} />) : (
              <p className="sl-muted">Assets undisclosed</p>
            )}
          </div>
        </div>
        <div className="sl-trade-board__mid" aria-hidden>
          <em>⇄</em>
          <span>Value check</span>
        </div>
        <div className="sl-trade-board__side">
          <TeamMark abbrev={rightAbbr} name={board.right?.display_name} size={compact ? 34 : 46} />
          <strong>{str(board.right?.display_name || rightAbbr)}</strong>
          <span>Receives</span>
          <div className="sl-trade-board__assets">
            {rightAssets.length ? rightAssets.map((a, i) => <TradeAssetChip key={i} asset={a} compact={compact} />) : (
              <p className="sl-muted">Assets undisclosed</p>
            )}
          </div>
        </div>
      </div>
      {(board.leftValue > 0 || board.rightValue > 0) ? (
        <TradeValueMeter
          leftValue={board.leftValue}
          rightValue={board.rightValue}
          leftAbbr={leftAbbr}
          rightAbbr={rightAbbr}
        />
      ) : null}
      {board.reason && !compact ? <p className="sl-trade-board__reason">{board.reason}</p> : null}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* story card + lead story                                             */
/* ------------------------------------------------------------------ */

function StoryCard({ story, socialCount, onOpen, index }) {
  const breaking = isBreakingStory(story);
  const tier = heatTier(story.heat);
  return (
    <button
      type="button"
      className={`sl-card sl-card--${tier}${breaking ? " is-breaking" : ""}${story.isUserTeam ? " is-ours" : ""} ${story.freshness || ""}`}
      style={{ animationDelay: `${Math.min(index, 12) * 28}ms` }}
      onClick={() => onOpen(story.id)}
    >
      <span className="sl-card__rail" aria-hidden />
      <div className="sl-card__top">
        <CategoryTag story={story} />
        {story.requiresAction ? <span className="sl-card__decision">On your desk</span> : null}
        <StatusPill story={story} />
        <span className="sl-card__age">{story.ageLabel || "—"}</span>
      </div>
      <div className="sl-card__body">
        <StoryFace story={story} size={46} />
        <div className="sl-card__text">
          <h3>{story.headline}</h3>
          {story.summary ? <p>{story.summary}</p> : null}
          {story.triggerReasons?.length ? (
            <MeetingCausePanel reasons={story.triggerReasons} title="Why this fired" />
          ) : story.triggerReason ? (
            <p className="sl-card__trigger">{story.triggerReason}</p>
          ) : null}
          {isTradeDeskStory(story) ? <TradeSummaryPanel story={story} compact /> : null}
          <StoryEffectBreakdown story={story} compact />
        </div>
      </div>
      <div className="sl-card__foot">
        <HeatSpark value={story.heat} />
        <div className="sl-card__stats">
          {heatLabel(story.heat) ? <span className="sl-card__heat">{heatLabel(story.heat)}</span> : null}
          {story.isUserTeam ? <span className="sl-card__ours">Your club</span> : null}
          {socialCount ? <span className="sl-card__social">{formatCount(socialCount)} posts</span> : null}
          <span className="sl-card__open">Open file →</span>
        </div>
      </div>
    </button>
  );
}

function LeadStory({ story, socialCount, onOpen, choiceOptions, onResolve, busyChoice }) {
  if (!story) return null;
  const meta = categoryMeta(story);
  const tier = heatTier(story.heat);
  return (
    <section className={`sl-lead sl-lead--${tier}`} aria-label="Lead story">
      <div className="sl-lead__glow" aria-hidden style={{ background: `radial-gradient(circle at 22% 40%, ${meta.accent}22, transparent 62%)` }} />
      <div className="sl-lead__portrait">
        <StoryFace story={story} size={132} />
        <HeatRing value={story.heat} size={72} />
      </div>
      <div className="sl-lead__main">
        <div className="sl-lead__kickers">
          <span className="sl-lead__flag">{story.requiresAction ? "Awaiting your call" : "Lead story"}</span>
          <CategoryTag story={story} size="md" />
          <StatusPill story={story} />
          <span className="sl-lead__age">{story.ageLabel || "—"}</span>
        </div>
        <h2 className="sl-lead__headline">{story.headline}</h2>
        {story.summary ? <p className="sl-lead__summary">{story.summary}</p> : null}
        {story.triggerReasons?.length ? (
          <MeetingCausePanel reasons={story.triggerReasons} title="Why this story fired" />
        ) : null}
        <StoryEffectBreakdown story={story} />
        <div className="sl-lead__meta">
          {story.playerName ? <span><em>Subject</em>{story.playerName}</span> : null}
          {story.teamName ? <span><em>Club</em>{story.teamName}</span> : null}
          {credibilityLabel(story.credibility) ? (
            <span><em>Sourcing</em>{credibilityLabel(story.credibility)}</span>
          ) : null}
          {socialCount ? <span><em>Chatter</em>{formatCount(socialCount)} posts</span> : null}
        </div>

        {choiceOptions.length ? (
          <div className="sl-lead__choices">
            {choiceOptions.slice(0, 3).map((opt) => {
              const busy = busyChoice === `${story.storylineId}:${opt.id}`;
              return (
                <button
                  key={opt.id}
                  type="button"
                  className="sl-choice sl-choice--lead"
                  disabled={Boolean(busyChoice)}
                  onClick={() => onResolve(story.storylineId, opt.id)}
                >
                  <strong>{opt.label}</strong>
                  {opt.effect_summary ? <span>{opt.effect_summary}</span> : null}
                  {busy ? <em>Applying…</em> : null}
                </button>
              );
            })}
          </div>
        ) : null}

        <button type="button" className="sl-lead__open" onClick={() => onOpen(story.id)}>
          Open the full case file →
        </button>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* player meetings                                                     */
/* ------------------------------------------------------------------ */

function PlayerMeetingsPanel({
  meetingsPayload,
  busy,
  onResolvePlayerRequest,
  onStartMeeting,
  onAdvanceMeeting,
  onRefresh,
  initialPlayerId,
}) {
  const [view, setView] = useState(initialPlayerId ? "player" : "home");
  const [playerTab, setPlayerTab] = useState("talk");
  const [selectedPlayerId, setSelectedPlayerId] = useState(initialPlayerId || null);
  const [playerDetail, setPlayerDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailCacheRef = useRef(new Map());
  const [activeMeeting, setActiveMeeting] = useState(null);
  const [notice, setNotice] = useState("");
  const [lastOutcome, setLastOutcome] = useState(null);
  const [addressedPlayerIds, setAddressedPlayerIds] = useState(() => new Set());
  const [rosterQuery, setRosterQuery] = useState("");

  const markPlayerAddressed = useCallback((playerId) => {
    const pid = str(playerId);
    if (!pid) return;
    setAddressedPlayerIds((prev) => {
      const next = new Set(prev);
      next.add(pid);
      return next;
    });
    detailCacheRef.current.delete(pid);
  }, []);

  useEffect(() => {
    if (initialPlayerId) {
      setSelectedPlayerId(initialPlayerId);
      setView("player");
    }
  }, [initialPlayerId]);

  const loadPlayerDetail = useCallback((playerId, { background = false } = {}) => {
    if (!playerId) return undefined;
    const cached = detailCacheRef.current.get(playerId);
    if (cached && !background) {
      setPlayerDetail(cached);
      setDetailLoading(false);
    } else if (!background) {
      setDetailLoading(!cached);
    }
    let cancelled = false;
    getPlayerMeetingDetail(playerId)
      .then((res) => {
        if (cancelled) return;
        const detail = res?.detail || null;
        if (detail) detailCacheRef.current.set(playerId, detail);
        setPlayerDetail(detail);
        setDetailLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          if (!cached) setPlayerDetail(null);
          setDetailLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (view !== "player" || !selectedPlayerId) {
      setPlayerDetail(null);
      setDetailLoading(false);
      return undefined;
    }
    return loadPlayerDetail(selectedPlayerId);
  }, [view, selectedPlayerId, loadPlayerDetail]);

  const roster = asArray(meetingsPayload?.roster);
  const needs = asArray(meetingsPayload?.needs_attention).filter(
    (row) => !addressedPlayerIds.has(str(row.player_id))
  );
  const requests = asArray(meetingsPayload?.player_requests).filter(
    (req) => !addressedPlayerIds.has(str(req.player_id || req.actor_id))
  );
  const promises = asObject(meetingsPayload?.promises);
  const history = asArray(meetingsPayload?.history);
  const selected = roster.find((r) => str(r.player_id) === str(selectedPlayerId)) || null;

  const visibleRoster = useMemo(() => {
    const q = rosterQuery.trim().toLowerCase();
    if (!q) return roster;
    return roster.filter((r) =>
      `${str(r.player_name)} ${str(r.position)} ${str(r.relationship?.label)}`.toLowerCase().includes(q)
    );
  }, [roster, rosterQuery]);

  const handleStart = useCallback(
    async (playerId, interactionType) => {
      setNotice("");
      try {
        const res = await onStartMeeting(playerId, interactionType);
        setActiveMeeting(res?.meeting || null);
        setView("meeting");
      } catch (err) {
        setNotice(err?.message || "Could not start meeting.");
      }
    },
    [onStartMeeting]
  );

  const handleResolveRequest = useCallback(
    async (interactionId, choiceId) => {
      setNotice("");
      setLastOutcome(null);
      try {
        const res = await onResolvePlayerRequest(interactionId, choiceId);
        const gm = res?.state?.last_gm_result || res?.last_gm_result || {};
        setLastOutcome({
          message: res?.message || gm.headline || "Meeting resolved.",
          summary: res?.effect_summary || gm.summary,
          receipts: res?.receipts,
          relationship: res?.relationship,
          choice_label: res?.choice_label,
        });
        markPlayerAddressed(res?.interaction?.player_id || res?.interaction?.actor_id);
        setActiveMeeting(null);
        setView("home");
        onRefresh?.();
      } catch (err) {
        setNotice(err?.message || "Could not resolve meeting.");
      }
    },
    [onResolvePlayerRequest, onRefresh, markPlayerAddressed]
  );

  const handleAdvance = useCallback(
    async (meetingId, choiceId) => {
      setNotice("");
      setLastOutcome(null);
      try {
        const res = await onAdvanceMeeting(meetingId, choiceId);
        const gm = res?.state?.last_gm_result || res?.last_gm_result || {};
        setLastOutcome({
          message: res?.message || gm.headline || res?.history?.choice_label || "Conversation recorded.",
          summary: res?.effect_summary || gm.summary,
          receipts: res?.receipts,
          relationship: res?.relationship,
          history: res?.history,
          choice_label: res?.choice_label || res?.history?.choice_label,
        });
        markPlayerAddressed(activeMeeting?.player_id);
        setActiveMeeting(null);
        setView("player");
        onRefresh?.();
      } catch (err) {
        setNotice(err?.message || "Could not complete meeting.");
      }
    },
    [onAdvanceMeeting, onRefresh, markPlayerAddressed, activeMeeting?.player_id]
  );

  if (!meetingsPayload || !roster.length) {
    return (
      <EmptyPanel
        kicker="GM office · private"
        title="The door is closed"
        body="Advance the calendar to sync roster relationships and open meeting availability."
      />
    );
  }

  if (view === "meeting" && activeMeeting) {
    const dialogue = asArray(activeMeeting.dialogue);
    return (
      <div className="sl-room sl-room--cinematic">
        <div className="sl-room__vignette" aria-hidden />
        <header className="sl-room__head sl-room__head--meet">
          <button type="button" className="sl-back" onClick={() => { setActiveMeeting(null); setView("player"); }}>
            ← Leave room
          </button>
          <div className="sl-pm-hero">
            <PlayerHeadshot
              player={{ id: str(activeMeeting.player_id), player_id: str(activeMeeting.player_id) }}
              size={88}
            />
            <div>
              <p className="sl-room__kicker">Private meeting · door closed</p>
              <h2>{str(activeMeeting.player_name)}</h2>
              <span className="sl-room__sub">{str(activeMeeting.title)}</span>
            </div>
          </div>
        </header>

        <MeetingCausePanel reasons={asArray(activeMeeting.trigger_reasons)} title="What brought you here" />
        <MeetingRelationshipPanel relationship={activeMeeting.relationship} />

        <div className="sl-dialogue sl-dialogue--cinematic">
          {dialogue.map((line, i) => (
            <DialogueBubble key={i} line={line} index={i} variant="meeting" />
          ))}
        </div>

        {activeMeeting.ovr_explanation?.factors?.length ? (
          <div className="sl-ovr sl-ovr--cinematic">
            <h4>{str(activeMeeting.ovr_explanation.headline)}</h4>
            <ul>
              {activeMeeting.ovr_explanation.factors.map((f, i) => (
                <li key={i}>{str(f.text)}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <section className="sl-pm-decision">
          <h4>Your response</h4>
          <div className="sl-choices sl-choices--cinematic">
            {asArray(activeMeeting.choices).map((c) => (
              <ResponseChoiceButton
                key={str(c.id)}
                choice={c}
                className="sl-choice sl-choice--cinematic"
                disabled={busy}
                onClick={() => handleAdvance(str(activeMeeting.id), str(c.id))}
              />
            ))}
          </div>
        </section>
        <MeetingOutcomePanel outcome={lastOutcome} onDismiss={() => setLastOutcome(null)} />
        {notice ? <p className="sl-notice">{notice}</p> : null}
      </div>
    );
  }

  if (view === "player" && selected) {
    const avail = asArray(playerDetail?.available_interactions);
    const rel = playerDetail?.relationship || selected.relationship || {};
    const openRequests = requests.filter((r) => str(r.player_id || r.actor_id) === str(selected.player_id));
    const attentionReasons = asArray(selected.attention_reasons);
    return (
      <div className="sl-room sl-room--cinematic">
        <div className="sl-room__vignette" aria-hidden />
        <header className="sl-room__head sl-room__head--meet">
          <button type="button" className="sl-back" onClick={() => { setView("home"); setSelectedPlayerId(null); }}>
            ← Roster
          </button>
          <div className="sl-pm-hero">
            <PlayerHeadshot player={{ id: str(selected.player_id), player_id: str(selected.player_id) }} size={88} />
            <div>
              <p className="sl-room__kicker">Player file · private office</p>
              <h2>{str(selected.player_name)}</h2>
              <p className="sl-pm-identity__line">
                {str(selected.position)} · {selected.age} · OVR {selected.overall}
                {selected.readiness_delta
                  ? ` (${selected.readiness_delta > 0 ? "+" : ""}${selected.readiness_delta})`
                  : ""}
              </p>
            </div>
          </div>
        </header>

        <MeetingRelationshipPanel relationship={rel} />
        {attentionReasons.length ? (
          <MeetingCausePanel reasons={attentionReasons} title="Why he needs attention" />
        ) : null}

        {openRequests.map((req) => (
          <article key={str(req.id)} className="sl-request sl-request--cinematic">
            <span className="sl-request__flag">He asked for this meeting</span>
            <MeetingCausePanel
              reasons={[{ code: "kind", label: meetingKindLabel(req.kind) }]}
              title="Trigger"
            />
            <h3>{str(req.title || "Player requested a meeting")}</h3>
            <p>{str(req.summary)}</p>
            {asArray(req.dialogue).slice(0, 1).map((d, i) => (
              <blockquote key={i}>{str(d.text)}</blockquote>
            ))}
            <div className="sl-choices sl-choices--cinematic">
              {asArray(req.choices).map((c) => (
                <ResponseChoiceButton
                  key={str(c.id)}
                  choice={c}
                  className="sl-choice sl-choice--cinematic"
                  disabled={busy}
                  onClick={() => handleResolveRequest(str(req.id), str(c.id))}
                />
              ))}
            </div>
          </article>
        ))}

        <nav className="sl-subtabs">
          {["talk", "promises", "history"].map((tab) => (
            <button
              key={tab}
              type="button"
              className={playerTab === tab ? "is-active" : ""}
              onClick={() => setPlayerTab(tab)}
            >
              {tab === "talk" ? "Talk" : tab === "promises" ? "Promises" : "History"}
            </button>
          ))}
        </nav>

        {playerTab === "promises" ? (
          <div className="sl-stack">
            {asArray(playerDetail?.promises || promises.active)
              .filter((p) => str(p.player_id) === str(selected.player_id))
              .map((p) => (
                <div key={str(p.id || p.type)} className="sl-promise">
                  <strong>{str(p.description || p.type)}</strong>
                  <span>{p.games_remaining != null ? `${p.games_remaining} games left` : "Active"}</span>
                </div>
              ))}
            {!asArray(playerDetail?.promises || promises.active).filter(
              (p) => str(p.player_id) === str(selected.player_id)
            ).length ? (
              <p className="sl-muted">Nothing promised to this player yet.</p>
            ) : null}
          </div>
        ) : playerTab === "history" ? (
          <div className="sl-stack">
            {asArray(playerDetail?.history || history)
              .filter((h) => str(h.player_id) === str(selected.player_id))
              .map((h) => (
                <div key={str(h.id)} className="sl-histrow">
                  <time>{str(h.calendar_iso || h.calendar_day)}</time>
                  <div>
                    <strong>{formatEffectLabel(h.interaction_type)}</strong>
                    <p>{str(h.choice_label || h.choice_id)}</p>
                    {formatMeetingReceiptLines(h.receipts).map((line, i) => (
                      <p key={i} className="sl-muted sl-histrow__effect">{line}</p>
                    ))}
                  </div>
                </div>
              ))}
            {!asArray(playerDetail?.history || history).filter(
              (h) => str(h.player_id) === str(selected.player_id)
            ).length ? (
              <p className="sl-muted">No recorded conversations yet.</p>
            ) : null}
          </div>
        ) : (
          <div className="sl-topics">
            <MeetingOutcomePanel outcome={lastOutcome} onDismiss={() => setLastOutcome(null)} />
            {playerDetail?.ovr_explanation?.factors?.length ? (
              <div className="sl-ovr">
                <h4>{str(playerDetail.ovr_explanation.headline)}</h4>
                <ul>
                  {playerDetail.ovr_explanation.factors.map((f, i) => (
                    <li key={i}>{str(f.text)}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {avail.length ? (
              Object.entries(
                avail.reduce((acc, row) => {
                  const cat = str(row.category_label || row.category);
                  if (!acc[cat]) acc[cat] = [];
                  acc[cat].push(row);
                  return acc;
                }, {})
              ).map(([cat, items]) => (
                <section key={cat}>
                  <h4>{cat}</h4>
                  <div className="sl-topic-grid">
                    {items.map((item) => (
                      <button
                        key={str(item.id)}
                        type="button"
                        className="sl-topic"
                        disabled={busy}
                        onClick={() => handleStart(str(selected.player_id), str(item.id))}
                      >
                        {str(item.title)}
                      </button>
                    ))}
                  </div>
                </section>
              ))
            ) : detailLoading ? (
              <p className="sl-muted">Loading topics…</p>
            ) : (
              <p className="sl-muted">No conversation topics available right now.</p>
            )}
          </div>
        )}
        {notice ? <p className="sl-notice">{notice}</p> : null}
      </div>
    );
  }

  return (
    <div className="sl-room sl-room--cinematic">
      <div className="sl-room__vignette" aria-hidden />
      <header className="sl-room__head">
        <p className="sl-room__kicker">GM office · relationship desk</p>
        <h2>Player meetings</h2>
        <span className="sl-room__sub">
          Private conversations behind closed doors. Resolve requests and strained relationships before they become headlines.
        </span>
      </header>
      <MeetingOutcomePanel outcome={lastOutcome} onDismiss={() => setLastOutcome(null)} />
      {notice ? <p className="sl-notice">{notice}</p> : null}

      {requests.length ? (
        <section className="sl-block">
          <h3 className="sl-block__title sl-block__title--alert">
            Requests waiting <em>{requests.length}</em>
          </h3>
          {requests.map((req) => (
            <article key={str(req.id)} className="sl-request sl-request--cinematic">
              <div className="sl-request__head">
                <strong>{str(req.player_name || "Player")}</strong>
                <span>{meetingKindLabel(req.kind)}</span>
              </div>
              <h3>{str(req.title)}</h3>
              <p>{str(req.summary)}</p>
              <div className="sl-choices sl-choices--cinematic sl-choices--row">
                {asArray(req.choices).map((c) => (
                  <ResponseChoiceButton
                    key={str(c.id)}
                    choice={c}
                    className="sl-choice sl-choice--cinematic sl-choice--compact"
                    disabled={busy}
                    onClick={() => handleResolveRequest(str(req.id), str(c.id))}
                  />
                ))}
              </div>
            </article>
          ))}
        </section>
      ) : null}

      {needs.length ? (
        <section className="sl-block">
          <h3 className="sl-block__title">
            Needs attention <em>{needs.length}</em>
          </h3>
          <div className="sl-roster sl-roster--attention">
            {needs.map((row) => (
              <button
                key={str(row.player_id)}
                type="button"
                className="sl-rosterrow is-flagged sl-rosterrow--cinematic"
                onClick={() => {
                  setSelectedPlayerId(str(row.player_id));
                  setView("player");
                }}
                onMouseEnter={() => loadPlayerDetail(str(row.player_id), { background: true })}
              >
                <PlayerHeadshot player={{ id: str(row.player_id), player_id: str(row.player_id) }} size={52} />
                <div className="sl-rosterrow__main">
                  <strong>{str(row.player_name)}</strong>
                  <span>
                    {str(row.position)} · OVR {row.overall} · Morale {formatMeetingStat(row.relationship?.morale)}
                  </span>
                  <div className="sl-pm-cause-chips">
                    {asArray(row.attention_reasons).slice(0, 2).map((reason, i) => (
                      <em key={i}>{str(reason.label)}</em>
                    ))}
                  </div>
                </div>
                {row.requested_meeting ? <span className="sl-tagbadge">Requested</span> : null}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      <section className="sl-block">
        <div className="sl-block__bar">
          <h3 className="sl-block__title">Full roster <em>{roster.length}</em></h3>
          <input
            type="search"
            className="sl-input"
            placeholder="Find a player…"
            value={rosterQuery}
            onChange={(e) => setRosterQuery(e.target.value)}
          />
        </div>
        <div className="sl-roster">
          {visibleRoster.map((row) => (
            <button
              key={str(row.player_id)}
              type="button"
              className="sl-rosterrow"
              onClick={() => {
                setSelectedPlayerId(str(row.player_id));
                setView("player");
              }}
              onMouseEnter={() => loadPlayerDetail(str(row.player_id), { background: true })}
            >
              <PlayerHeadshot player={{ id: str(row.player_id), player_id: str(row.player_id) }} size={44} />
              <div className="sl-rosterrow__main">
                <strong>{str(row.player_name)}</strong>
                <span>
                  {str(row.position)} · {row.age} · OVR {row.overall}
                  {row.ovr_trend === "up" ? " ↑" : row.ovr_trend === "down" ? " ↓" : ""}
                </span>
                <em>
                  {str(row.relationship?.label)} · {str(row.agent?.name || "Agent TBD")}
                </em>
              </div>
            </button>
          ))}
          {!visibleRoster.length ? <p className="sl-muted">No players match that search.</p> : null}
        </div>
      </section>

      {asArray(promises.active).length ? (
        <section className="sl-block">
          <h3 className="sl-block__title">Active promises <em>{asArray(promises.active).length}</em></h3>
          <div className="sl-stack">
            {promises.active.map((p) => (
              <div key={str(p.id)} className="sl-promise">
                <strong>{str(p.description || p.type)}</strong>
                <span>{p.games_remaining != null ? `${p.games_remaining}g remaining` : "Open"}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

const SEVERITY_LABELS = {
  crisis: "Crisis",
  major: "Major",
  mid: "Notable",
  minor: "Minor",
};

function StoryImpactReport({ report }) {
  const recent = asArray(report?.recent_user_stories);
  const modifiers = asArray(report?.active_player_modifiers);
  if (!recent.length && !modifiers.length) return null;

  return (
    <section className="sl-impact-report">
      <div className="sl-impact-report__head">
        <h3>Impact report</h3>
        <span className="sl-impact-report__sub">Verified rating + room effects from recent beats</span>
      </div>
      {modifiers.length ? (
        <div className="sl-impact-report__mods">
          <h4>Active OVR modifiers</h4>
          <div className="sl-impact-report__mod-grid">
            {modifiers.slice(0, 6).map((row) => {
              const delta = Number(row.overall_delta) || 0;
              return (
                <div key={str(row.player_id)} className={`sl-impact-report__mod ${delta < 0 ? "is-neg" : delta > 0 ? "is-pos" : ""}`}>
                  <strong>{str(row.player_name)}</strong>
                  <span>
                    {row.base_ovr ?? "—"} → {row.effective_ovr ?? "—"}
                    {delta ? ` (${delta > 0 ? "+" : ""}${delta})` : ""}
                  </span>
                  {asArray(row.modifiers).slice(0, 1).map((m, i) => (
                    <em key={i}>{str(m.reason || m.source || "Storyline")}</em>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
      {recent.length ? (
        <div className="sl-impact-report__recent">
          <h4>Recent story impacts</h4>
          <ul className="sl-impact-report__list">
            {recent.slice(0, 8).map((row) => {
              const delta = Number(row.overall_delta);
              const sev = SEVERITY_LABELS[row.severity] || row.severity || "Story";
              return (
                <li key={str(row.storyline_id || row.headline)}>
                  <div className="sl-impact-report__row-top">
                    <span className={`sl-impact-report__sev sl-impact-report__sev--${str(row.severity || "minor")}`}>{sev}</span>
                    <strong>{str(row.headline)}</strong>
                  </div>
                  <span className="sl-impact-report__meta">
                    {str(row.player_name || "Team")} · {prettyDate(row.calendar_iso) || "Recent"}
                    {Number.isFinite(delta) && delta !== 0 ? ` · OVR ${delta > 0 ? "+" : ""}${delta}` : ""}
                  </span>
                  {row.effect_summary ? <em>{str(row.effect_summary)}</em> : null}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* main screen                                                         */
/* ------------------------------------------------------------------ */

export default function StorylinesScreen() {
  const {
    franchiseState,
    onResolveStorylineChoice,
    setScreen,
    refreshFranchise,
    hydrateFranchiseNarrative,
    mergeFranchiseState,
    pendingMeetingPlayerId,
    setPendingMeetingPlayerId,
    pendingSocialNav,
    setPendingSocialNav,
  } = useGameUI();

  useEffect(() => {
    hydrateFranchiseNarrative?.({ force: true });
  }, [franchiseState?.narrative_revision, franchiseState?.session_id, hydrateFranchiseNarrative]);

  const [department, setDepartment] = useState(
    pendingMeetingPlayerId ? "player_meetings" : pendingSocialNav ? "social" : "front_page"
  );
  const [socialSubTab, setSocialSubTab] = useState(pendingSocialNav?.subTab || "puckr");
  const [redditSubFilter, setRedditSubFilter] = useState(pendingSocialNav?.subreddit || "all");
  const [expandedThreadId, setExpandedThreadId] = useState(null);
  const [liveSocialFeed, setLiveSocialFeed] = useState(null);
  const [meetingBusy, setMeetingBusy] = useState(false);
  const [filter, setFilter] = useState("all");
  const [sortId, setSortId] = useState("decisions");
  const [search, setSearch] = useState("");
  const [openCaseId, setOpenCaseId] = useState(null);
  const [activeTab, setActiveTab] = useState("details");
  const [busyChoice, setBusyChoice] = useState("");
  const [actionNotice, setActionNotice] = useState("");
  const [pressOutcome, setPressOutcome] = useState(null);
  const caseRef = useRef(null);

  const sessionId = str(franchiseState?.session_id || getFranchiseSessionId() || "anon");
  const [dismissedBreaking, setDismissedBreaking] = useState(() => readDismissedBreakingKeys(sessionId));

  useEffect(() => {
    setDismissedBreaking(readDismissedBreakingKeys(sessionId));
  }, [sessionId]);

  const dismissBreakingAlerts = useCallback(
    (alerts) => {
      const list = Array.isArray(alerts) ? alerts : [];
      if (!list.length) return;
      setDismissedBreaking((prev) => {
        const next = new Set(prev);
        list.forEach((alert) => {
          const key = breakingAlertKey(alert);
          if (key) next.add(key);
        });
        writeDismissedBreakingKeys(sessionId, next);
        return next;
      });
    },
    [sessionId]
  );

  const stories = useMemo(
    () => collectStories(franchiseState),
    [franchiseState?.storyline_events, franchiseState?.stats_revision, franchiseState?.narrative_revision]
  );
  const narrativeStories = useMemo(() => stories.filter((s) => !isTradeDeskStory(s)), [stories]);
  const tradeStories = useMemo(() => stories.filter((s) => isTradeDeskStory(s)), [stories]);
  const deskStories = department === "trade_desk" ? tradeStories : narrativeStories;
  const deskFilters = department === "trade_desk" ? TRADE_FILTERS : FILTERS;
  const choicesMap = useMemo(
    () => buildChoicesMap(franchiseState),
    [franchiseState?.storyline_choices, franchiseState?.pending_decisions]
  );
  const storyImpactReport = useMemo(
    () => franchiseState?.narrative_summary?.story_impact_report || {},
    [franchiseState?.narrative_summary?.story_impact_report, franchiseState?.narrative_revision]
  );

  const openStory = useCallback((id) => {
    if (!id) return;
    setOpenCaseId(id);
    setActiveTab("details");
    if (typeof window !== "undefined") {
      window.requestAnimationFrame(() => {
        caseRef.current?.scrollIntoView?.({ block: "start", behavior: "smooth" });
      });
    }
  }, []);

  const closeStory = useCallback(() => setOpenCaseId(null), []);

  const filterCounts = useMemo(() => {
    const counts = { all: deskStories.length };
    deskFilters.forEach((f) => {
      if (f.id === "all") return;
      counts[f.id] = deskStories.filter((s) => matchesFilter(s, f.id)).length;
    });
    return counts;
  }, [deskStories, deskFilters]);

  const filtered = useMemo(() => {
    const base = deskStories.filter((s) => matchesFilter(s, filter) && matchesSearch(s, search));
    return sortStories(base, sortId);
  }, [deskStories, filter, search, sortId]);

  const pendingDecisions = useMemo(
    () => deskStories.filter((s) => s.requiresAction || choicesMap.has(s.storylineId) || choicesMap.has(s.id)),
    [deskStories, choicesMap]
  );
  const yourTeamCount = deskStories.filter((s) => s.isUserTeam).length;

  const orgPressure = asObject(franchiseState?.conduct_org_pressure);
  const userOrg =
    orgPressure[userTeamId(franchiseState)] || orgPressure[str(franchiseState?.user_team_id || "")] || null;

  const narrativeUniverse = asObject(franchiseState?.narrative_universe);
  const playerMeetingsPayload = asObject(narrativeUniverse?.player_meetings);
  const pressQueue = asArray(narrativeUniverse?.press_conference_queue).filter((p) =>
    ["pending", "in_progress"].includes(str(p?.status))
  );
  const narrativeEras = asArray(narrativeUniverse?.narrative_eras);
  const narrativeArchive = asArray(narrativeUniverse?.narrative_archive);
  const userMarket = asObject(narrativeUniverse?.user_market_profile);
  const knowledgeGraph = asArray(narrativeUniverse?.knowledge_graph);
  const insiderItems = asArray(narrativeUniverse?.insider_items).length
    ? asArray(narrativeUniverse.insider_items)
    : knowledgeGraph;
  const beatWriters = asArray(narrativeUniverse?.beat_writers).length
    ? asArray(narrativeUniverse.beat_writers)
    : asArray(narrativeUniverse?.reporters);
  const playerDossiers = useMemo(() => collectDossiers(narrativeUniverse), [narrativeUniverse]);
  const breakingAlerts = asArray(narrativeUniverse?.breaking_alerts);
  const pendingBreaking = useMemo(
    () => activeBreakingAlerts(breakingAlerts, dismissedBreaking),
    [breakingAlerts, dismissedBreaking]
  );
  const activeBreaking = pendingBreaking[0] || null;

  useEffect(() => {
    if (!activeBreaking?.level) return;
    playBreakingSting(activeBreaking.level);
  }, [activeBreaking?.storyline_id, activeBreaking?.headline, activeBreaking?.level]);

  useEffect(() => {
    if (pendingSocialNav) {
      setDepartment("social");
      if (pendingSocialNav.subTab) setSocialSubTab(pendingSocialNav.subTab);
      if (pendingSocialNav.subreddit) setRedditSubFilter(pendingSocialNav.subreddit);
      if (pendingSocialNav.threadId) setExpandedThreadId(pendingSocialNav.threadId);
      setPendingSocialNav(null);
    }
  }, [pendingSocialNav, setPendingSocialNav]);

  useEffect(() => {
    if (department !== "social") return undefined;
    let cancelled = false;
    (async () => {
      try {
        const feed = await getSocialFeed(sessionId);
        if (!cancelled) setLiveSocialFeed(feed);
      } catch {
        if (!cancelled) setLiveSocialFeed(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [department, sessionId, franchiseState?.calendar_idx]);

  const currentCalendarIso = calendarLabel(franchiseState);

  const socialPosts = useMemo(() => {
    const opts = { currentIso: currentCalendarIso, maxAgeDays: 2 };
    const puckr = asArray(liveSocialFeed?.puckr);
    if (puckr.length) return buildSocialPosts(stories, { social_posts: puckr }, opts);
    return buildSocialPosts(stories, narrativeUniverse, opts);
  }, [stories, narrativeUniverse, liveSocialFeed, currentCalendarIso]);

  const redditThreads = useMemo(() => {
    const opts = { currentIso: currentCalendarIso, maxAgeDays: 2 };
    const icehole = asArray(liveSocialFeed?.icehole);
    const source = icehole.length ? icehole : asArray(narrativeUniverse?.reddit_threads);
    return buildRedditThreads(source, redditSubFilter, opts);
  }, [liveSocialFeed, narrativeUniverse, redditSubFilter, currentCalendarIso]);

  const redditPulse = useMemo(
    () => fanPulseTrend(narrativeUniverse?.reddit_engagement_pulse),
    [narrativeUniverse]
  );

  const userSubreddit = useMemo(() => {
    const label = str(userMarket?.label || franchiseState?.user_team_name || "Team");
    const slug = label.replace(/[^A-Za-z0-9]+/g, "").slice(-12) || "Team";
    return `r/${slug}`;
  }, [userMarket, franchiseState]);

  const redditSubPills = useMemo(() => ["all", userSubreddit, "r/hockey"], [userSubreddit]);

  const socialCountByStory = useMemo(() => {
    const map = new Map();
    const posts = [
      ...asArray(narrativeUniverse?.social_posts),
      ...asArray(narrativeUniverse?.twitter_feed),
      ...asArray(narrativeUniverse?.social_feed),
    ];
    posts.forEach((p) => {
      const sid = str(p?.storyline_id || "");
      if (!sid) return;
      map.set(sid, (map.get(sid) || 0) + 1);
    });
    return map;
  }, [narrativeUniverse]);

  const socialCountFor = useCallback(
    (s) => (s ? socialCountByStory.get(s.storylineId) || socialCountByStory.get(s.id) || 0 : 0),
    [socialCountByStory]
  );

  const openCase = useMemo(
    () => (openCaseId ? stories.find((s) => s.id === openCaseId) || null : null),
    [openCaseId, stories]
  );

  const leadStory = useMemo(() => {
    if (!filtered.length) return null;
    return filtered[0];
  }, [filtered]);

  const gridStories = useMemo(() => {
    if (!leadStory) return filtered;
    return filtered.filter((s) => s.id !== leadStory.id);
  }, [filtered, leadStory]);

  const tickerItems = useMemo(() => {
    const pool = department === "trade_desk" ? tradeStories : narrativeStories;
    const fromHub = buildHubStoryTicker(
      { ...franchiseState, storyline_events: pool.map((s) => s.raw || s) },
      { limit: 16 }
    );
    if (fromHub.length) {
      const byId = new Map(pool.map((s) => [s.id, s]));
      return fromHub.map((row) => byId.get(row.id) || { id: row.id, headline: row.headline, categoryKey: "" });
    }
    return pool.slice(0, 14);
  }, [department, narrativeStories, tradeStories, franchiseState]);

  const selected = openCase;
  const selectedDossier = playerDossiers.find((d) => str(d.player_id) === str(selected?.playerId)) || null;

  const relatedStories = useMemo(() => {
    if (!selected) return [];
    return stories
      .filter((s) => s.id !== selected.id)
      .map((s) => {
        let score = 0;
        if (selected.playerId && s.playerId === selected.playerId) score += 100;
        if (selected.teamId && s.teamId === selected.teamId) score += 60;
        if (selected.storylineId && s.escalatedFrom === selected.storylineId) score += 40;
        if (s.categoryKey === selected.categoryKey) score += 20;
        return { s, score };
      })
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .map((r) => r.s);
  }, [selected, stories]);

  const leagueRumours = useMemo(
    () => stories.filter((s) => isRumourStory(s) && s.id !== selected?.id).slice(0, 10),
    [stories, selected]
  );

  const selectedChoice = selected ? choicesMap.get(selected.storylineId) || choicesMap.get(selected.id) : null;

  const handleResolve = useCallback(
    async (storylineId, choiceId) => {
      if (!onResolveStorylineChoice) return;
      setBusyChoice(`${storylineId}:${choiceId}`);
      try {
        const res = await onResolveStorylineChoice(storylineId, choiceId);
        const result = res?.state?.last_gm_result || {};
        setActionNotice(str(result.headline || result.summary || "Decision recorded. Check the wire for fallout."));
      } catch (err) {
        setActionNotice(err?.message || "That choice did not land. Try again.");
      } finally {
        setBusyChoice("");
      }
    },
    [onResolveStorylineChoice]
  );

  const handlePressResponse = useCallback(
    async (pressItem, questionId, responseId) => {
      if (!onResolveStorylineChoice || !pressItem) return;
      const choiceId = `${questionId}:${responseId}`;
      const storylineId = str(pressItem.storyline_id || pressItem.id || pressItem.storylineId);
      setBusyChoice(`${storylineId}:${choiceId}`);
      setPressOutcome(null);
      try {
        const res = await onResolveStorylineChoice(storylineId, choiceId);
        const result = res?.state?.last_gm_result || {};
        setPressOutcome({
          message: str(result.headline || "You addressed the media."),
          summary: str(result.summary || result.effect_summary || ""),
          receipts: result.receipts,
          choice_label: result.choice_label || result.response_label,
        });
        setActionNotice("");
      } catch (err) {
        setActionNotice(err?.message || "The press room did not take that answer.");
      } finally {
        setBusyChoice("");
      }
    },
    [onResolveStorylineChoice]
  );

  const mergeMeetingState = useCallback(
    async (res) => {
      const next = res?.state;
      if (next && typeof next === "object") {
        mergeFranchiseState?.(next);
        await hydrateFranchiseNarrative?.({ force: true });
        return true;
      }
      return false;
    },
    [mergeFranchiseState, hydrateFranchiseNarrative]
  );

  const handleMeetingRefresh = useCallback(async () => {
    await refreshFranchise?.();
  }, [refreshFranchise]);

  const handleResolvePlayerMeeting = useCallback(
    async (interactionId, choiceId) => {
      setMeetingBusy(true);
      try {
        const res = await resolvePlayerMeeting(interactionId, choiceId);
        if (!mergeMeetingState(res)) await refreshFranchise?.();
        if (pendingMeetingPlayerId) setPendingMeetingPlayerId?.(null);
        return res;
      } finally {
        setMeetingBusy(false);
      }
    },
    [mergeMeetingState, refreshFranchise, pendingMeetingPlayerId, setPendingMeetingPlayerId]
  );

  const handleStartPlayerMeeting = useCallback(
    async (playerId, interactionType) => {
      setMeetingBusy(true);
      try {
        const res = await startPlayerMeeting(playerId, interactionType);
        if (!mergeMeetingState(res)) await refreshFranchise?.();
        return res;
      } finally {
        setMeetingBusy(false);
      }
    },
    [mergeMeetingState, refreshFranchise]
  );

  const handleAdvancePlayerMeeting = useCallback(
    async (meetingId, choiceId) => {
      setMeetingBusy(true);
      try {
        const res = await advancePlayerMeeting(meetingId, choiceId);
        if (!mergeMeetingState(res)) await refreshFranchise?.();
        return res;
      } finally {
        setMeetingBusy(false);
      }
    },
    [mergeMeetingState, refreshFranchise]
  );

  const meetingAlertCount =
    asArray(playerMeetingsPayload?.player_requests).length + asArray(playerMeetingsPayload?.needs_attention).length;

  const hasBackend = Array.isArray(franchiseState?.storyline_events);
  const filterEmptyMsg = FILTER_EMPTY[filter];
  const arcTimeline = useMemo(
    () => collectArcTimeline(stories, selected, narrativeUniverse),
    [stories, selected, narrativeUniverse]
  );

  const keyFactors = useMemo(() => {
    if (!selected) return [];
    const items = [];
    if (selected.impactReason) items.push(selected.impactReason);
    if (selected.cause) items.push(selected.cause);
    asArray(selected.recoveryConditions).forEach((r) => items.push(r));
    Object.entries(selected.evidence || {})
      .slice(0, 3)
      .forEach(([k, v]) => items.push(`${formatEffectLabel(k)}: ${v}`));
    if (!items.length) items.push(deriveFollowUp(selected));
    return items.slice(0, 6);
  }, [selected]);

  const parties = useMemo(() => {
    if (!selected) return [];
    const list = [];
    if (selected.playerName) list.push({ label: "Player", name: selected.playerName });
    if (selected.teamName) list.push({ label: "Team", name: selected.teamName });
    if (selected.culpritPlayerName && selected.culpritPlayerName !== selected.playerName)
      list.push({ label: "Central figure", name: selected.culpritPlayerName });
    if (selected.fromTeamName) list.push({ label: "From", name: selected.fromTeamName });
    if (selected.toTeamName) list.push({ label: "To", name: selected.toTeamName });
    if (selected.reporterName)
      list.push({
        label: "Reporter",
        name: `${selected.reporterName}${selected.outletName ? ` (${selected.outletName})` : ""}`,
      });
    if (selected.sourceLabel && !selected.reporterName) list.push({ label: "Source", name: selected.sourceLabel });
    return list;
  }, [selected]);

  const infoRows = useMemo(() => {
    if (!selected) return [];
    const rows = [];
    if (selected.informationStatus) rows.push(["Information status", formatEffectLabel(selected.informationStatus)]);
    if (credibilityLabel(selected.credibility)) rows.push(["Evidence strength", credibilityLabel(selected.credibility)]);
    if (knowledgeLevelLabel(selected.publicKnowledgeLevel))
      rows.push(["Public visibility", knowledgeLevelLabel(selected.publicKnowledgeLevel)]);
    if (selected.sourceLabel) rows.push(["Leaked by", selected.sourceLabel]);
    return rows;
  }, [selected]);

  const choiceOptions = selected
    ? asArray(selectedChoice?.action_options).length
      ? selectedChoice.action_options
      : selected.actionOptions
    : [];

  const leadChoice = leadStory
    ? choicesMap.get(leadStory.storylineId) || choicesMap.get(leadStory.id) || null
    : null;
  const leadChoiceOptions = leadStory
    ? asArray(leadChoice?.action_options).length
      ? leadChoice.action_options
      : asArray(leadStory.actionOptions)
    : [];

  return (
    <div className="nhlcal-sl-root">
      <style>{`
        .nhlcal-sl-root {
          --bg-deep: #030b13;
          --panel: rgba(10, 26, 40, 0.82);
          --panel-2: rgba(7, 19, 30, 0.78);
          --panel-3: rgba(5, 14, 23, 0.94);
          --line: rgba(150, 214, 235, 0.13);
          --line-2: rgba(150, 214, 235, 0.24);
          --line-strong: rgba(73, 231, 240, 0.55);
          --text: #eaf7fc;
          --muted: #7e94a6;
          --muted-2: #9fb4c4;
          --cyan: #16dcea;
          --cyan-dim: rgba(22, 220, 234, 0.14);
          --gold: #e9a83c;
          --brass: #c9a227;
          --gold-dim: rgba(233, 168, 60, 0.14);
          --green: #52df94;
          --red: #ff5f6d;
          --red-dim: rgba(255, 95, 109, 0.14);
          --ember: #ff8a4c;
          --purple: #c992ff;

          position: relative;
          height: 100dvh;
          max-height: 100dvh;
          overflow: hidden;
          display: flex;
          flex-direction: column;
          width: 100%;
          background:
            radial-gradient(ellipse 70% 50% at 18% -6%, rgba(22, 220, 234, 0.13), transparent 60%),
            radial-gradient(ellipse 50% 40% at 92% 4%, rgba(233, 168, 60, 0.09), transparent 60%),
            radial-gradient(ellipse 80% 60% at 50% 108%, rgba(20, 60, 92, 0.35), transparent 65%),
            linear-gradient(178deg, #071726 0%, #030b13 55%, #01060b 100%);
          color: var(--text);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .sl-atmos {
          position: absolute; inset: 0; pointer-events: none; z-index: 0;
          background-image:
            repeating-linear-gradient(0deg, rgba(255,255,255,.014) 0 1px, transparent 1px 3px),
            repeating-linear-gradient(90deg, rgba(150,214,235,.03) 0 1px, transparent 1px 96px);
          mask-image: radial-gradient(ellipse 90% 80% at 50% 40%, #000 40%, transparent 100%);
          opacity: .7;
        }
        .sl-atmos::after {
          content: ""; position: absolute; inset: 0;
          box-shadow: inset 0 0 220px 60px rgba(0,0,0,.55);
        }

        .sl-app {
          position: relative; z-index: 1;
          width: min(1560px, 100%);
          margin: 0 auto;
          padding: 14px 20px 40px;
          display: flex; flex-direction: column; gap: 14px;
          flex: 1; min-height: 0;
          overflow-y: auto; overflow-x: hidden; overscroll-behavior: contain;
        }
        .sl-app::-webkit-scrollbar { width: 10px; }
        .sl-app::-webkit-scrollbar-thumb { background: rgba(150,214,235,.16); border-radius: 8px; }
        .sl-app::-webkit-scrollbar-track { background: transparent; }

        /* ------------- command bar ------------- */
        .sl-command {
          display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
          padding: 12px 16px;
          border: 1px solid var(--line);
          border-radius: 12px;
          background:
            linear-gradient(120deg, rgba(22,220,234,.06), transparent 42%),
            linear-gradient(180deg, rgba(12,32,48,.92), rgba(6,17,27,.92));
          box-shadow: 0 14px 34px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.04);
        }
        .sl-command__crest { width: 46px; height: 46px; display: grid; place-items: center; flex-shrink: 0;
          border: 1px solid var(--line-2); border-radius: 10px; background: rgba(255,255,255,.03); overflow: hidden; }
        .sl-command__crest img { width: 100%; height: 100%; object-fit: contain; padding: 4px; }
        .sl-command__crest strong { font-size: 13px; font-weight: 900; color: var(--cyan); }
        .sl-command__id { min-width: 0; }
        .sl-command__eyebrow { margin: 0; font-size: 10px; font-weight: 900; letter-spacing: .22em; text-transform: uppercase; color: var(--cyan); }
        .sl-command__id h1 { margin: 2px 0 3px; font-size: 25px; font-weight: 900; letter-spacing: .07em; text-transform: uppercase; line-height: 1; }
        .sl-command__sub { margin: 0; font-size: 11.5px; font-weight: 700; color: var(--muted-2); }
        .sl-command__stats { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
        .sl-stat { border: 1px solid var(--line); border-radius: 9px; padding: 6px 12px; min-width: 76px;
          background: rgba(255,255,255,.02); text-align: left; }
        .sl-stat strong { display: block; font-size: 19px; font-weight: 900; line-height: 1.05; letter-spacing: .01em; }
        .sl-stat span { display: block; font-size: 9.5px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-top: 2px; }
        .sl-stat--alert { border-color: rgba(233,168,60,.45); background: var(--gold-dim); }
        .sl-stat--alert strong { color: var(--gold); }
        .sl-stat--ours strong { color: var(--cyan); }
        .sl-command__nav { display: flex; gap: 8px; }
        .sl-command__nav button {
          height: 34px; padding: 0 15px; border-radius: 8px; cursor: pointer;
          border: 1px solid var(--line-2); background: rgba(14,36,52,.8); color: var(--text);
          font-size: 11px; font-weight: 900; letter-spacing: .09em; text-transform: uppercase;
          transition: border-color .15s ease, background .15s ease, transform .12s ease;
        }
        .sl-command__nav button:hover { border-color: var(--line-strong); background: rgba(22,220,234,.12); transform: translateY(-1px); }

        /* ------------- ticker ------------- */
        .sl-ticker {
          display: flex; align-items: stretch; overflow: hidden;
          border: 1px solid var(--line); border-radius: 9px;
          background: linear-gradient(90deg, rgba(255,95,109,.1), rgba(6,17,27,.9) 26%);
        }
        .sl-ticker__flag {
          flex-shrink: 0; display: flex; align-items: center; gap: 7px; padding: 0 14px;
          font-size: 10px; font-weight: 900; letter-spacing: .16em; text-transform: uppercase; color: #ffb4bb;
          border-right: 1px solid rgba(255,95,109,.3); background: rgba(255,95,109,.1);
        }
        .sl-ticker__dot { width: 7px; height: 7px; border-radius: 50%; background: var(--red); animation: slPulse 1.6s ease-in-out infinite; }
        .sl-ticker__viewport { overflow: hidden; flex: 1; position: relative; }
        .sl-ticker__track { display: flex; gap: 34px; padding: 9px 0; white-space: nowrap; width: max-content;
          animation: slTicker 46s linear infinite; }
        .sl-ticker:hover .sl-ticker__track { animation-play-state: paused; }
        .sl-ticker__item { display: inline-flex; align-items: center; gap: 9px; font-size: 12px; font-weight: 700; color: var(--muted-2);
          background: none; border: 0; cursor: pointer; padding: 0; }
        .sl-ticker__item:hover { color: var(--text); }
        .sl-ticker__item i { font-style: normal; font-size: 9.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
        .sl-ticker__sep { color: rgba(150,214,235,.25); }

        /* ------------- departments ------------- */
        .sl-depts { display: flex; gap: 6px; flex-wrap: wrap; padding: 5px; border: 1px solid var(--line);
          border-radius: 11px; background: rgba(6,17,27,.6); }
        .sl-depts button {
          display: inline-flex; align-items: center; gap: 7px; position: relative;
          height: 34px; padding: 0 15px; border-radius: 8px; cursor: pointer;
          border: 1px solid transparent; background: transparent; color: rgba(234,247,252,.6);
          font-size: 11px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase;
          transition: color .15s ease, background .15s ease, border-color .15s ease;
        }
        .sl-depts button em { font-style: normal; font-size: 12px; opacity: .8; }
        .sl-depts button:hover { color: var(--text); background: rgba(255,255,255,.04); }
        .sl-depts button.is-active { color: #041018; background: linear-gradient(180deg, #2ee6f0, #12b9c9);
          border-color: rgba(22,220,234,.4); box-shadow: 0 4px 14px rgba(22,220,234,.22); }
        .sl-depts .sl-dept-count {
          min-width: 17px; height: 17px; padding: 0 5px; border-radius: 9px; display: grid; place-items: center;
          font-size: 9.5px; font-weight: 900; background: var(--red); color: #290308;
        }
        .sl-depts button.is-active .sl-dept-count { background: #062028; color: #7fe9f2; }

        .sl-market {
          display: flex; align-items: center; gap: 10px; margin: 0; padding: 8px 14px;
          border: 1px solid rgba(201,162,39,.28); border-left: 3px solid var(--brass); border-radius: 0 8px 8px 0;
          background: linear-gradient(90deg, rgba(201,162,39,.12), rgba(6,17,27,.5));
          font-size: 11.5px; font-weight: 800; color: #f0cf93;
        }
        .sl-market em { font-style: normal; font-size: 9.5px; letter-spacing: .16em; text-transform: uppercase; color: var(--brass); }
        .sl-action-notice {
          display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
          margin: 0 0 10px; padding: 10px 14px; border-radius: 10px;
          border: 1px solid rgba(46,230,240,.28); background: rgba(10,42,52,.72);
          color: #d8f7fb; font-size: 13px; font-weight: 700; line-height: 1.35;
        }
        .sl-action-notice button {
          border: 0; background: transparent; color: #7fe9f2; cursor: pointer; font-weight: 900;
        }

        /* ------------- lead story ------------- */
        .sl-lead {
          position: relative; overflow: hidden;
          display: grid; grid-template-columns: 168px minmax(0,1fr); gap: 22px; align-items: center;
          padding: 22px 26px; border: 1px solid var(--line-2); border-radius: 14px;
          background: linear-gradient(135deg, rgba(13,36,53,.95), rgba(5,15,24,.95));
          box-shadow: 0 22px 52px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.05);
          animation: slRise .38s cubic-bezier(.2,.7,.3,1) both;
        }
        .sl-lead::before {
          content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
          background: linear-gradient(180deg, var(--cyan), rgba(22,220,234,.1));
        }
        .sl-lead--hot::before { background: linear-gradient(180deg, var(--gold), rgba(233,168,60,.1)); }
        .sl-lead--boiling::before { background: linear-gradient(180deg, var(--red), rgba(255,95,109,.1)); }
        .sl-lead--boiling { border-color: rgba(255,95,109,.32); }
        .sl-lead__glow { position: absolute; inset: 0; pointer-events: none; }
        .sl-lead__portrait { position: relative; z-index: 1; display: grid; justify-items: center; gap: 12px; }
        .sl-lead__main { position: relative; z-index: 1; min-width: 0; }
        .sl-lead__kickers { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
        .sl-lead__flag { font-size: 9.5px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase;
          color: #041018; background: linear-gradient(90deg, #ffd27a, var(--gold)); padding: 4px 10px; border-radius: 4px; }
        .sl-lead__age { font-size: 10.5px; font-weight: 800; color: var(--muted); margin-left: auto; }
        .sl-lead__headline { margin: 0 0 10px; font-size: clamp(21px, 2.5vw, 31px); line-height: 1.14; font-weight: 800; letter-spacing: -.01em; }
        .sl-lead__summary { margin: 0 0 14px; font-size: 14px; line-height: 1.55; color: rgba(234,247,252,.8); max-width: 76ch; }
        .sl-lead__meta { display: flex; flex-wrap: wrap; gap: 22px; margin-bottom: 16px; }
        .sl-lead__meta span { font-size: 12.5px; font-weight: 800; }
        .sl-lead__meta em { display: block; font-style: normal; font-size: 9.5px; font-weight: 900; letter-spacing: .14em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 3px; }
        .sl-lead__choices { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
        .sl-lead__open { border: 0; background: none; cursor: pointer; padding: 0;
          color: var(--cyan); font-size: 12px; font-weight: 900; letter-spacing: .07em; text-transform: uppercase; }
        .sl-lead__open:hover { text-decoration: underline; }
        @media (max-width: 860px) { .sl-lead { grid-template-columns: 1fr; } .sl-lead__portrait { justify-items: start; grid-auto-flow: column; } }

        /* ------------- heat ring / spark ------------- */
        .sl-ring { position: relative; display: grid; place-items: center; flex-shrink: 0; }
        .sl-ring svg { position: absolute; inset: 0; }
        .sl-ring__track { fill: none; stroke: rgba(150,214,235,.12); stroke-width: 5; }
        .sl-ring__fill { fill: none; stroke-width: 5; stroke-linecap: round; transition: stroke-dasharray .6s cubic-bezier(.2,.8,.3,1); }
        .sl-ring--cool .sl-ring__fill { stroke: #4fd3e6; }
        .sl-ring--warm .sl-ring__fill { stroke: var(--cyan); }
        .sl-ring--hot .sl-ring__fill { stroke: var(--gold); }
        .sl-ring--boiling .sl-ring__fill { stroke: var(--red); }
        .sl-ring--boiling { animation: slEmber 2.4s ease-in-out infinite; }
        .sl-ring__center { position: relative; text-align: center; line-height: 1; }
        .sl-ring__center strong { display: block; font-weight: 900; }
        .sl-ring__center span { display: block; font-size: 8.5px; font-weight: 900; letter-spacing: .16em;
          text-transform: uppercase; color: var(--muted); margin-top: 3px; }

        .sl-spark { height: 4px; border-radius: 3px; background: rgba(150,214,235,.1); overflow: hidden; flex: 1; min-width: 60px; }
        .sl-spark span { display: block; height: 100%; border-radius: 3px; animation: slGrow .6s cubic-bezier(.2,.8,.3,1) both; }
        .sl-spark--cool span { background: #3aa8c4; }
        .sl-spark--warm span { background: var(--cyan); }
        .sl-spark--hot span { background: linear-gradient(90deg, var(--gold), var(--ember)); }
        .sl-spark--boiling span { background: linear-gradient(90deg, var(--ember), var(--red)); }

        /* ------------- category tag / pills ------------- */
        .sl-cat { display: inline-flex; align-items: center; gap: 5px; border: 1px solid; border-radius: 5px;
          padding: 3px 8px; font-size: 9.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
        .sl-cat--md { font-size: 10.5px; padding: 4px 10px; }
        .sl-cat em { font-style: normal; font-size: 10px; }

        .sl-status-pill { display: inline-flex; align-items: center; gap: 4px; font-size: 9.5px; font-weight: 900;
          letter-spacing: .1em; text-transform: uppercase; border-radius: 4px; padding: 3px 8px; }
        .sl-status-pill em { font-style: normal; font-size: 8px; }
        .sl-status-pill--escalating { color: #ff9aa2; background: rgba(255,95,109,.13); border: 1px solid rgba(255,95,109,.4); animation: slPulse 2.2s ease-in-out infinite; }
        .sl-status-pill--developing { color: #ffcd7a; background: rgba(233,168,60,.13); border: 1px solid rgba(233,168,60,.35); }
        .sl-status-pill--resolved { color: #8ef0b8; background: rgba(82,223,148,.12); border: 1px solid rgba(82,223,148,.35); }

        .sl-score { display: grid; place-items: center; border: 1px solid; border-radius: 9px; font-weight: 900; }
        .sl-score--sm { width: 32px; height: 32px; }
        .sl-score--md { width: 40px; height: 40px; }
        .sl-score--lg { width: 58px; height: 58px; }
        .sl-score--crit { background: rgba(255,95,109,.16); border-color: rgba(255,95,109,.55); color: #ff9aa2; }
        .sl-score--high { background: rgba(233,168,60,.16); border-color: rgba(233,168,60,.5); color: #ffcd7a; }
        .sl-score--mid { background: rgba(22,220,234,.12); border-color: rgba(22,220,234,.4); color: var(--cyan); }
        .sl-score--low { background: rgba(150,214,235,.07); border-color: rgba(150,214,235,.22); color: var(--muted); }

        /* ------------- faces ------------- */
        .sl-face { position: relative; border-radius: 12px; overflow: hidden; flex-shrink: 0;
          border: 1px solid var(--line-2); background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(0,0,0,.2));
          display: grid; place-items: center; }
        .sl-face img { width: 100%; height: 100%; object-fit: cover; }
        .sl-face > span { font-size: 15px; font-weight: 900; color: var(--cyan); letter-spacing: .04em; }
        .sl-teammark { display: grid; place-items: center; border-radius: 8px; border: 1px solid var(--line-2);
          background: rgba(255,255,255,.03); overflow: hidden; flex-shrink: 0; }
        .sl-teammark img { width: 100%; height: 100%; object-fit: contain; padding: 3px; }
        .sl-teammark strong { font-size: 11px; font-weight: 900; color: var(--cyan); }

        /* ------------- toolbar ------------- */
        .sl-toolbar { display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }
        .sl-chips { display: flex; gap: 5px; flex-wrap: wrap; }
        .sl-chip { display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 13px; cursor: pointer;
          border: 1px solid var(--line); border-radius: 999px; background: rgba(10,28,42,.6); color: rgba(234,247,252,.7);
          font-size: 10.5px; font-weight: 900; letter-spacing: .07em; text-transform: uppercase;
          transition: border-color .15s ease, color .15s ease, background .15s ease; }
        .sl-chip:hover { color: var(--text); border-color: var(--line-2); }
        .sl-chip.is-active { border-color: var(--line-strong); background: var(--cyan-dim); color: var(--text); }
        .sl-chip b { font-weight: 800; opacity: .55; }
        .sl-tools { display: flex; gap: 8px; align-items: center; }
        .sl-input { height: 32px; padding: 0 12px; width: 210px; border-radius: 8px; border: 1px solid var(--line);
          background: rgba(6,18,29,.86); color: var(--text); font-size: 12px; font-weight: 700; }
        .sl-input:focus { outline: none; border-color: var(--line-strong); }
        .sl-tools select { height: 32px; padding: 0 10px; border-radius: 8px; border: 1px solid var(--line);
          background: rgba(12,32,48,.9); color: var(--text); font-size: 11px; font-weight: 800; cursor: pointer; }

        /* ------------- story grid ------------- */
        .sl-gridhead { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
          padding-bottom: 8px; border-bottom: 1px solid var(--line); }
        .sl-gridhead h3 { margin: 0; font-size: 10.5px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase; color: var(--muted); }
        .sl-gridhead span { font-size: 10.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; color: var(--cyan); }

        .sl-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }
        .sl-card {
          position: relative; overflow: hidden; text-align: left; cursor: pointer;
          display: flex; flex-direction: column; gap: 11px;
          padding: 13px 15px 13px 18px; border: 1px solid var(--line); border-radius: 12px;
          background: linear-gradient(160deg, rgba(11,29,44,.86), rgba(5,15,24,.9));
          color: inherit;
          animation: slRise .34s cubic-bezier(.2,.7,.3,1) both;
          transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        }
        .sl-card:hover { transform: translateY(-2px); border-color: var(--line-strong); box-shadow: 0 14px 30px rgba(0,0,0,.4); }
        .sl-card__rail { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }
        .sl-card--cool .sl-card__rail { background: linear-gradient(180deg, #3aa8c4, rgba(58,168,196,.15)); }
        .sl-card--warm .sl-card__rail { background: linear-gradient(180deg, var(--cyan), rgba(22,220,234,.15)); }
        .sl-card--hot .sl-card__rail { background: linear-gradient(180deg, var(--gold), rgba(233,168,60,.15)); }
        .sl-card--boiling .sl-card__rail { background: linear-gradient(180deg, var(--red), rgba(255,95,109,.15)); }
        .sl-card.is-breaking { border-color: rgba(255,95,109,.28); }
        .sl-card.is-ours { background: linear-gradient(160deg, rgba(14,38,56,.9), rgba(5,15,24,.9)); }
        .sl-card.is-ours::after { content: ""; position: absolute; right: 0; top: 0; width: 42px; height: 42px;
          background: linear-gradient(225deg, rgba(22,220,234,.16), transparent 60%); pointer-events: none; }
        .sl-card.is-stale { opacity: .78; }
        .sl-card__top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .sl-card__decision { font-size: 9.5px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase;
          color: #041018; background: linear-gradient(90deg, #ffd27a, var(--gold)); padding: 3px 8px; border-radius: 4px; }
        .sl-card__age { margin-left: auto; font-size: 10px; font-weight: 800; color: var(--muted); white-space: nowrap; }
        .sl-card__body { display: flex; gap: 12px; align-items: flex-start; }
        .sl-card__text { min-width: 0; }
        .sl-card__text h3 { margin: 0 0 5px; font-size: 14.5px; line-height: 1.3; font-weight: 700;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .sl-card__text p { margin: 0; font-size: 12px; line-height: 1.42; color: var(--muted-2);
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .sl-card__foot { display: flex; align-items: center; gap: 12px; }
        .sl-card__stats { display: flex; align-items: center; gap: 10px; flex-shrink: 0;
          font-size: 10px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
        .sl-card__heat { color: var(--muted-2); }
        .sl-card__ours { color: var(--cyan); }
        .sl-card__social { color: var(--purple); }
        .sl-card__open { color: var(--cyan); opacity: 0; transition: opacity .16s ease; }
        .sl-card:hover .sl-card__open { opacity: 1; }

        /* ------------- case file ------------- */
        .sl-case { display: grid; grid-template-columns: minmax(0,1fr) 322px; gap: 16px; align-items: start;
          animation: slRise .32s cubic-bezier(.2,.7,.3,1) both; }
        @media (max-width: 1060px) { .sl-case { grid-template-columns: 1fr; } }
        .sl-case__main { border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
          background: linear-gradient(180deg, rgba(10,27,41,.9), rgba(5,14,23,.92)); }
        .sl-case__hero { position: relative; display: flex; gap: 18px; align-items: flex-start; padding: 20px 22px;
          border-bottom: 1px solid var(--line);
          background: linear-gradient(120deg, rgba(22,220,234,.05), transparent 55%); }
        .sl-case__hero-main { flex: 1; min-width: 0; }
        .sl-case__crumbs { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
        .sl-back { border: 1px solid var(--line-2); border-radius: 7px; background: rgba(255,255,255,.03); cursor: pointer;
          color: var(--cyan); font-size: 10.5px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; padding: 6px 11px; }
        .sl-back:hover { border-color: var(--line-strong); background: var(--cyan-dim); }
        .sl-case__title { margin: 0 0 10px; font-size: clamp(19px, 2.2vw, 27px); line-height: 1.16; font-weight: 800; }
        .sl-case__byline { display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; font-weight: 800; color: var(--muted); margin-bottom: 12px; }
        .sl-case__byline b { color: var(--muted-2); font-weight: 800; }
        .sl-case__lede { margin: 0 0 6px; font-size: 14px; line-height: 1.6; color: rgba(234,247,252,.86); }
        .sl-case__body { padding: 18px 22px 22px; }
        .sl-case__section { margin-bottom: 20px; }
        .sl-case__prose { margin: 0 0 14px; font-size: 13.5px; line-height: 1.6; color: rgba(234,247,252,.78); }

        .sl-swap { display: flex; align-items: center; justify-content: center; gap: 26px; margin: 14px 0;
          padding: 14px; border: 1px solid rgba(201,146,255,.24); border-radius: 10px; background: rgba(201,146,255,.06); }
        .sl-swap__side { display: grid; justify-items: center; gap: 6px; min-width: 84px; }
        .sl-swap__side span { font-size: 11px; font-weight: 900; letter-spacing: .05em; text-align: center; }
        .sl-swap__mid { display: grid; justify-items: center; gap: 3px; }
        .sl-swap__mid em { font-style: normal; font-size: 20px; font-weight: 900; color: var(--purple); }
        .sl-swap__mid span { font-size: 9px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }

        .sl-conduct { border: 1px solid rgba(255,138,76,.26); border-left: 3px solid var(--ember); border-radius: 0 10px 10px 0;
          background: rgba(255,138,76,.05); padding: 13px 16px; margin: 14px 0; }
        .sl-conduct h4 { margin: 0 0 8px; font-size: 10px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--ember); }
        .sl-conduct__note { margin: 0 0 10px; font-size: 12.5px; line-height: 1.5; color: rgba(234,247,252,.85); }
        .sl-conduct__grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 10px; }
        .sl-conduct__grid > div span { display: block; font-size: 9px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 3px; }
        .sl-conduct__grid > div strong { font-size: 12px; font-weight: 800; }

        .sl-spine { margin: 18px 0; }
        .sl-spine h4, .sl-case__section h4 { margin: 0 0 12px; font-size: 10px; font-weight: 900; letter-spacing: .16em;
          text-transform: uppercase; color: var(--cyan); }
        .sl-spine__list { list-style: none; margin: 0; padding: 0 0 0 22px; position: relative; }
        .sl-spine__list::before { content: ""; position: absolute; left: 5px; top: 6px; bottom: 14px; width: 1px;
          background: linear-gradient(180deg, var(--cyan), rgba(150,214,235,.12)); }
        .sl-spine__list li { position: relative; padding-bottom: 16px; }
        .sl-spine__dot { position: absolute; left: -21px; top: 4px; width: 11px; height: 11px; border-radius: 50%;
          background: #061a26; border: 2px solid var(--cyan); }
        .sl-spine__list li.is-latest .sl-spine__dot { border-color: var(--red); box-shadow: 0 0 0 4px rgba(255,95,109,.13); }
        .sl-spine__dot--ghost { border-color: rgba(150,214,235,.3); }
        .sl-spine__list li.is-next { opacity: .55; padding-bottom: 0; }
        .sl-spine__list time { display: block; font-size: 9.5px; font-weight: 900; letter-spacing: .1em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 3px; }
        .sl-spine__list strong { display: block; font-size: 13px; line-height: 1.35; font-weight: 700; }
        .sl-spine__list p { margin: 4px 0 0; font-size: 12px; line-height: 1.45; color: var(--muted-2); }

        .sl-tabs { display: flex; gap: 24px; border-bottom: 1px solid var(--line); margin: 18px 0 0; }
        .sl-tabs button { border: 0; background: none; cursor: pointer; padding: 9px 0; margin-bottom: -1px;
          border-bottom: 2px solid transparent; color: var(--muted);
          font-size: 10.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
        .sl-tabs button:hover { color: var(--muted-2); }
        .sl-tabs button.is-active { color: var(--text); border-bottom-color: var(--cyan); }
        .sl-tabpanel { padding-top: 18px; }

        .sl-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap: 20px; }
        .sl-cols h4 { margin: 0 0 10px; font-size: 10px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); }
        .sl-kv { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0;
          border-bottom: 1px solid rgba(150,214,235,.08); font-size: 12px; }
        .sl-kv span:first-child { color: var(--muted); font-weight: 700; }
        .sl-kv span:last-child { font-weight: 800; text-align: right; }
        .sl-factors { list-style: none; margin: 0; padding: 0; display: grid; gap: 9px; }
        .sl-factors li { position: relative; padding-left: 15px; font-size: 12.5px; line-height: 1.45; color: rgba(234,247,252,.85); }
        .sl-factors li::before { content: "▸"; position: absolute; left: 0; color: var(--gold); font-size: 10px; top: 2px; }
        .sl-nums { display: grid; grid-template-columns: repeat(auto-fit, minmax(78px,1fr)); gap: 8px; margin-top: 12px; }
        .sl-num { border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; background: rgba(255,255,255,.02); }
        .sl-num strong { display: block; font-size: 18px; font-weight: 900; line-height: 1; margin-bottom: 4px; }
        .sl-num span { font-size: 9px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }

        .sl-linklist { display: grid; gap: 8px; }
        .sl-linklist button { display: grid; gap: 4px; text-align: left; cursor: pointer; padding: 11px 13px;
          border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.02); color: inherit;
          transition: border-color .15s ease, transform .12s ease; }
        .sl-linklist button:hover { border-color: var(--line-strong); transform: translateX(2px); }
        .sl-linklist strong { font-size: 12.5px; line-height: 1.35; font-weight: 700; }
        .sl-linklist em { font-style: normal; font-size: 10px; font-weight: 800; color: var(--muted); }

        .sl-case__foot { display: flex; justify-content: space-between; gap: 12px; margin-top: 20px; padding-top: 12px;
          border-top: 1px solid var(--line); font-size: 10px; font-weight: 800; color: var(--muted); }

        /* ------------- rail ------------- */
        .sl-rail { display: grid; gap: 12px; position: sticky; top: 0; }
        .sl-panel { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
          background: linear-gradient(180deg, rgba(10,27,41,.86), rgba(5,14,23,.9)); }
        .sl-panel h3 { margin: 0 0 12px; font-size: 10px; font-weight: 900; letter-spacing: .16em;
          text-transform: uppercase; color: var(--cyan); }
        .sl-panel--desk { border-color: rgba(201,162,39,.34);
          background: linear-gradient(180deg, rgba(38,29,12,.75), rgba(9,17,24,.92)); }
        .sl-panel--desk h3 { color: var(--brass); }

        .sl-bars { display: grid; gap: 12px; }
        .sl-bar__label { display: flex; justify-content: space-between; margin-bottom: 5px;
          font-size: 10px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
        .sl-bar__label strong { color: var(--text); font-size: 11.5px; }
        .sl-bar__track { height: 6px; border-radius: 4px; background: rgba(150,214,235,.1); overflow: hidden; }
        .sl-bar__fill { height: 100%; border-radius: 4px; animation: slGrow .7s cubic-bezier(.2,.8,.3,1) both; }
        .sl-bar__fill--good { background: linear-gradient(90deg, #2f9c68, var(--green)); }
        .sl-bar__fill--warm { background: linear-gradient(90deg, #a5731c, var(--gold)); }
        .sl-bar__fill--hot { background: linear-gradient(90deg, #b03744, var(--red)); }
        .sl-bar__foot { margin: 4px 0 0; font-size: 10.5px; font-weight: 700; color: var(--muted); }

        .sl-effects { display: grid; gap: 4px; }
        .sl-effect { display: flex; justify-content: space-between; align-items: center; gap: 10px;
          padding: 8px 0; border-bottom: 1px solid rgba(150,214,235,.07); font-size: 11.5px; font-weight: 800; }
        .sl-effect:last-child { border-bottom: 0; }
        .sl-effect span { color: var(--muted-2); }
        .sl-effect strong { font-variant-numeric: tabular-nums; }
        .sl-effect.pos strong { color: var(--green); }
        .sl-effect.neg strong { color: var(--red); }
        .sl-effect.neutral strong { color: var(--muted); }

        /* ------------- choices ------------- */
        .sl-choices { display: grid; gap: 9px; }
        .sl-choices--row { grid-auto-flow: column; grid-auto-columns: minmax(0,1fr); }
        @media (max-width: 700px) { .sl-choices--row { grid-auto-flow: row; } }
        .sl-choice {
          position: relative; overflow: hidden; text-align: left; cursor: pointer; color: var(--text);
          display: block; width: 100%; padding: 12px 14px; border-radius: 10px;
          border: 1px solid var(--line-2);
          background: linear-gradient(180deg, rgba(22,220,234,.07), rgba(255,255,255,.015));
          transition: border-color .15s ease, background .15s ease, transform .12s ease;
        }
        .sl-choice::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
          background: var(--cyan); opacity: .55; }
        .sl-choice:hover:not(:disabled) { border-color: var(--line-strong); transform: translateY(-1px);
          background: linear-gradient(180deg, rgba(22,220,234,.16), rgba(255,255,255,.03)); }
        .sl-choice:disabled { opacity: .5; cursor: not-allowed; }
        .sl-choice strong { display: block; font-size: 12px; font-weight: 900; letter-spacing: .04em;
          text-transform: uppercase; margin-bottom: 4px; }
        .sl-choice span { display: block; font-size: 11.5px; font-weight: 600; line-height: 1.45; color: var(--muted-2); }
        .sl-choice em { display: block; margin-top: 6px; font-style: normal; font-size: 10px; font-weight: 900;
          letter-spacing: .1em; text-transform: uppercase; color: var(--cyan); }
        .sl-choice--lead { flex: 1 1 220px; width: auto;
          border-color: rgba(201,162,39,.4); background: linear-gradient(180deg, rgba(233,168,60,.13), rgba(255,255,255,.02)); }
        .sl-choice--lead::before { background: var(--gold); }
        .sl-choice--lead:hover:not(:disabled) { border-color: rgba(233,168,60,.75);
          background: linear-gradient(180deg, rgba(233,168,60,.24), rgba(255,255,255,.04)); }
        .sl-choice--cinematic {
          border-color: rgba(201,162,39,.28);
          background: linear-gradient(180deg, rgba(233,168,60,.08), rgba(255,255,255,.02));
        }
        .sl-choice--cinematic::before { background: var(--gold); opacity: .75; }
        .sl-choice--cinematic:hover:not(:disabled) {
          border-color: rgba(233,168,60,.55);
          background: linear-gradient(180deg, rgba(233,168,60,.16), rgba(255,255,255,.04));
        }
        .sl-rosterrow--cinematic { min-height: 74px; }
        .sl-choice--compact { padding: 9px 12px; }
        .sl-choice--compact strong { margin-bottom: 0; }
        .sl-choice__tone-bar {
          position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 10px 0 0 10px;
          background: var(--cyan); opacity: .85;
        }
        .sl-choice--tone-supportive {
          border-color: rgba(82,223,148,.35);
          background: linear-gradient(180deg, rgba(82,223,148,.12), rgba(255,255,255,.02));
        }
        .sl-choice--tone-supportive .sl-choice__tone-bar { background: var(--green); }
        .sl-choice--tone-firm {
          border-color: rgba(255,95,109,.35);
          background: linear-gradient(180deg, rgba(255,95,109,.1), rgba(255,255,255,.02));
        }
        .sl-choice--tone-firm .sl-choice__tone-bar { background: var(--red); }
        .sl-choice--tone-volatile {
          border-color: rgba(255,138,76,.4);
          background: linear-gradient(180deg, rgba(255,138,76,.12), rgba(255,255,255,.02));
        }
        .sl-choice--tone-volatile .sl-choice__tone-bar { background: var(--ember); }
        .sl-choice--tone-cautious {
          border-color: rgba(233,168,60,.35);
          background: linear-gradient(180deg, rgba(233,168,60,.1), rgba(255,255,255,.02));
        }
        .sl-choice--tone-cautious .sl-choice__tone-bar { background: var(--gold); }
        .sl-choice--tone-deflect {
          border-color: rgba(201,146,255,.35);
          background: linear-gradient(180deg, rgba(201,146,255,.1), rgba(255,255,255,.02));
        }
        .sl-choice--tone-deflect .sl-choice__tone-bar { background: var(--purple); }
        .sl-choice--press { animation: slRise .35s cubic-bezier(.2,.7,.3,1) both; }
        .sl-choice__effects {
          display: block; margin-top: 6px; font-size: 10px; font-weight: 800; letter-spacing: .04em;
          color: var(--cyan); opacity: .9;
        }

        /* ------------- dialogue bubbles ------------- */
        .sl-bubble {
          position: relative; max-width: min(720px, 92%);
          padding: 12px 14px 12px 16px; border-radius: 14px;
          border: 1px solid var(--line);
          background: rgba(255,255,255,.03);
          animation: slBubbleIn .45s cubic-bezier(.2,.7,.3,1) both;
        }
        .sl-bubble__speaker {
          display: block; font-style: normal; font-size: 9px; font-weight: 900;
          letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px; color: var(--muted);
        }
        .sl-bubble p { margin: 0; font-size: 15px; line-height: 1.6; }
        .sl-bubble--gm {
          margin-left: auto; border-color: rgba(233,168,60,.35);
          background: linear-gradient(180deg, rgba(233,168,60,.14), rgba(8,18,28,.9));
          box-shadow: 0 10px 28px rgba(0,0,0,.22);
        }
        .sl-bubble--gm .sl-bubble__speaker { color: var(--gold); }
        .sl-bubble--player {
          margin-right: auto; border-color: rgba(22,220,234,.28);
          background: linear-gradient(180deg, rgba(22,220,234,.1), rgba(8,18,28,.88));
        }
        .sl-bubble--player .sl-bubble__speaker { color: var(--cyan); }
        .sl-bubble--reporter {
          margin-right: auto; border-color: rgba(201,146,255,.28);
          background: linear-gradient(180deg, rgba(201,146,255,.08), rgba(8,18,28,.88));
        }
        .sl-bubble--reporter .sl-bubble__speaker { color: var(--purple); }
        @keyframes slBubbleIn {
          from { opacity: 0; transform: translateY(10px) scale(.98); filter: blur(2px); }
          to { opacity: 1; transform: none; filter: none; }
        }

        /* ------------- trade board ------------- */
        .sl-trade-board {
          position: relative; overflow: hidden; margin: 14px 0; padding: 16px;
          border: 1px solid rgba(201,146,255,.28); border-radius: 14px;
          background: linear-gradient(180deg, rgba(201,146,255,.08), rgba(5,14,23,.92));
        }
        .sl-trade-board.is-compact { margin: 10px 0 0; padding: 10px 12px; }
        .sl-trade-board__glow {
          position: absolute; inset: -30% auto auto 50%; width: 280px; height: 280px; transform: translateX(-50%);
          background: radial-gradient(circle, rgba(201,146,255,.12), transparent 70%); pointer-events: none;
        }
        .sl-trade-board__kicker {
          margin: 0 0 10px; font-size: 9px; font-weight: 900; letter-spacing: .16em;
          text-transform: uppercase; color: var(--purple);
        }
        .sl-trade-board__teams {
          display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: start;
        }
        @media (max-width: 760px) { .sl-trade-board__teams { grid-template-columns: 1fr; } }
        .sl-trade-board__side { display: grid; gap: 8px; justify-items: center; text-align: center; }
        .sl-trade-board__side strong { font-size: 13px; }
        .sl-trade-board__side > span { font-size: 9px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
        .sl-trade-board__mid { display: grid; justify-items: center; gap: 4px; padding-top: 18px; }
        .sl-trade-board__mid em { font-style: normal; font-size: 22px; color: var(--purple); }
        .sl-trade-board__mid span { font-size: 9px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
        .sl-trade-board__assets { width: 100%; display: grid; gap: 6px; }
        .sl-trade-board__reason { margin: 12px 0 0; font-size: 12px; line-height: 1.5; color: var(--muted-2); }
        .sl-trade-asset {
          text-align: left; padding: 8px 10px; border-radius: 9px;
          border: 1px solid rgba(150,214,235,.16); background: rgba(255,255,255,.03);
        }
        .sl-trade-asset.is-pick { border-color: rgba(233,168,60,.25); }
        .sl-trade-asset__head { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
        .sl-trade-asset__head strong { font-size: 12px; line-height: 1.3; }
        .sl-trade-asset__ovr {
          flex-shrink: 0; font-size: 10px; font-weight: 900; letter-spacing: .06em;
          color: #041018; background: linear-gradient(180deg, #2ee6f0, #12b9c9); padding: 2px 6px; border-radius: 4px;
        }
        .sl-trade-asset__meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; font-size: 10px; font-weight: 800; color: var(--muted); }
        .sl-trade-asset__meta em { font-style: normal; text-transform: uppercase; letter-spacing: .06em; }
        .sl-trade-asset__tv { color: var(--gold); }
        .sl-trade-asset__bar { height: 4px; border-radius: 999px; background: rgba(150,214,235,.12); margin-top: 6px; overflow: hidden; }
        .sl-trade-asset__bar span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #7b4fd6, var(--purple)); }
        .sl-trade-meter { margin-top: 12px; }
        .sl-trade-meter__labels {
          display: flex; justify-content: space-between; gap: 8px; align-items: center;
          font-size: 10px; font-weight: 800; color: var(--muted-2); margin-bottom: 6px;
        }
        .sl-trade-meter__labels em { font-style: normal; color: var(--gold); letter-spacing: .04em; text-transform: uppercase; font-size: 9px; }
        .sl-trade-meter__track { display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: rgba(150,214,235,.1); }
        .sl-trade-meter__left { background: linear-gradient(90deg, #2f9c68, var(--green)); }
        .sl-trade-meter__right { background: linear-gradient(90deg, #7b4fd6, var(--purple)); }
        .sl-press__question {
          margin: 0 0 12px; font-size: 14px; line-height: 1.55; font-style: italic;
          color: rgba(234,247,252,.9); padding: 12px 14px; border-radius: 12px;
          border: 1px solid rgba(22,220,234,.2); background: rgba(22,220,234,.05);
        }

        /* ------------- decision desk strip ------------- */
        .sl-desk { border: 1px solid rgba(201,162,39,.3); border-radius: 12px; padding: 12px 16px;
          background: linear-gradient(100deg, rgba(201,162,39,.11), rgba(6,17,27,.7) 55%); }
        .sl-desk__head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .sl-desk__head h3 { margin: 0; font-size: 10.5px; font-weight: 900; letter-spacing: .16em; text-transform: uppercase; color: var(--brass); }
        .sl-desk__count { font-size: 9.5px; font-weight: 900; letter-spacing: .1em; color: #2a1f06;
          background: var(--gold); padding: 3px 8px; border-radius: 4px; }
        .sl-desk__list { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap: 9px; }
        .sl-desk__item { display: flex; align-items: center; gap: 10px; text-align: left; cursor: pointer;
          padding: 9px 12px; border-radius: 9px; border: 1px solid rgba(201,162,39,.22);
          background: rgba(255,255,255,.02); color: inherit; transition: border-color .15s ease, transform .12s ease; }
        .sl-desk__item:hover { border-color: rgba(233,168,60,.6); transform: translateY(-1px); }
        .sl-desk__item strong { display: block; font-size: 12.5px; line-height: 1.3; font-weight: 700;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
        .sl-desk__item span { display: block; font-size: 10px; font-weight: 800; color: var(--muted); margin-top: 2px; }
        .sl-impact-report { border: 1px solid rgba(126,224,176,.22); border-radius: 12px; padding: 14px 16px;
          margin-bottom: 14px; background: linear-gradient(180deg, rgba(18,32,28,.92), rgba(10,16,14,.88)); }
        .sl-impact-report__head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
        .sl-impact-report__head h3 { margin: 0; font-size: 11px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: #7ee0b0; }
        .sl-impact-report__sub { font-size: 10px; color: var(--muted); }
        .sl-impact-report h4 { margin: 0 0 8px; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
        .sl-impact-report__mod-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr)); gap: 8px; margin-bottom: 12px; }
        .sl-impact-report__mod { border: 1px solid rgba(255,255,255,.08); border-radius: 8px; padding: 8px 10px; background: rgba(0,0,0,.18); }
        .sl-impact-report__mod strong { display: block; font-size: 12px; }
        .sl-impact-report__mod span { display: block; font-size: 11px; color: #d8e8f0; margin-top: 2px; }
        .sl-impact-report__mod em { display: block; font-size: 10px; color: var(--muted); margin-top: 4px; font-style: normal; }
        .sl-impact-report__mod.is-neg { border-color: rgba(255,96,109,.35); }
        .sl-impact-report__mod.is-pos { border-color: rgba(82,223,148,.35); }
        .sl-impact-report__list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
        .sl-impact-report__row-top { display: flex; align-items: center; gap: 8px; }
        .sl-impact-report__row-top strong { font-size: 12.5px; line-height: 1.3; }
        .sl-impact-report__sev { font-size: 9px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; }
        .sl-impact-report__sev--crisis { background: rgba(255,60,80,.2); color: #ff8090; }
        .sl-impact-report__sev--major { background: rgba(255,138,76,.18); color: #ffb080; }
        .sl-impact-report__sev--mid { background: rgba(233,168,60,.16); color: #e9c070; }
        .sl-impact-report__sev--minor { background: rgba(128,150,168,.16); color: #9eb0c0; }
        .sl-impact-report__meta { display: block; font-size: 10px; color: var(--muted); margin-top: 2px; }
        .sl-impact-report__list em { display: block; font-size: 10.5px; color: #b8c8d4; margin-top: 3px; font-style: normal; }

        /* ------------- dossier ------------- */
        .sl-dossier { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
          background: rgba(255,255,255,.02); margin: 14px 0; }
        .sl-dossier__head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 4px; }
        .sl-dossier__head strong { font-size: 13.5px; font-weight: 800; }
        .sl-dossier__head span { font-size: 11px; font-weight: 800; color: var(--muted); }
        .sl-dossier__ident { margin: 0 0 8px; font-size: 11.5px; color: var(--muted); }
        .sl-dossier h4 { margin: 10px 0 5px; font-size: 9px; font-weight: 900; letter-spacing: .14em;
          text-transform: uppercase; color: var(--cyan); }
        .sl-dossier p { margin: 0 0 3px; font-size: 11.5px; line-height: 1.4; color: var(--muted-2); }
        .sl-dossier__tags { display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
        .sl-dossier__tags span { font-size: 9px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase;
          border: 1px solid var(--line-2); border-radius: 4px; padding: 3px 7px; color: var(--muted-2); }
        .sl-dossier__grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .sl-dossier.is-compact { margin: 0; }

        /* ------------- social ------------- */
        .sl-two { display: grid; grid-template-columns: minmax(0,1fr) minmax(250px,320px); gap: 16px; align-items: start; }
        @media (max-width: 1000px) { .sl-two { grid-template-columns: 1fr; } }
        .sl-subtabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; padding: 4px;
          border: 1px solid var(--line); border-radius: 10px; background: rgba(6,17,27,.6); width: fit-content; }
        .sl-subtabs button { border: 0; border-radius: 7px; background: transparent; cursor: pointer;
          padding: 8px 15px; color: var(--muted); font-size: 10.5px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; }
        .sl-subtabs button:hover { color: var(--text); background: rgba(255,255,255,.04); }
        .sl-subtabs button.is-active { color: #041018; background: linear-gradient(180deg, #2ee6f0, #12b9c9); }
        .sl-feed { display: grid; gap: 10px; }
        .sl-post { text-align: left; cursor: pointer; color: inherit; padding: 12px 14px;
          border: 1px solid var(--line); border-radius: 11px; background: rgba(10,27,41,.7);
          transition: border-color .15s ease, transform .12s ease;
          animation: slRise .3s cubic-bezier(.2,.7,.3,1) both; }
        .sl-post:hover { border-color: var(--line-strong); transform: translateY(-1px); }
        .sl-post__head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 7px; }
        .sl-post__avatar { width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; flex-shrink: 0;
          background: linear-gradient(160deg, rgba(22,220,234,.22), rgba(201,146,255,.18));
          font-size: 10.5px; font-weight: 900; color: #eaf7fc; align-self: center; }
        .sl-post__head strong { font-size: 12.5px; font-weight: 800; }
        .sl-post__head span { font-size: 11px; font-weight: 700; color: var(--muted); }
        .sl-post__head em { margin-left: auto; font-style: normal; font-size: 10px; font-weight: 800; color: var(--muted); }
        .sl-post__verified { color: var(--cyan); font-size: 11px; }
        .sl-post p { margin: 0; font-size: 13px; line-height: 1.5; }
        .sl-post__meta { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 9px; padding-top: 9px;
          border-top: 1px solid rgba(150,214,235,.08); font-size: 10px; font-weight: 800; color: var(--muted); }
        .sl-post__related { color: var(--gold); letter-spacing: .04em; text-transform: uppercase; }

        .sl-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
        .sl-pills button { border: 1px solid var(--line); border-radius: 999px; background: transparent; cursor: pointer;
          padding: 5px 12px; color: var(--muted); font-size: 10px; font-weight: 900; letter-spacing: .06em; }
        .sl-pills button.is-active { border-color: rgba(233,168,60,.55); color: var(--gold); background: var(--gold-dim); }

        .sl-thread { width: 100%; text-align: left; cursor: pointer; color: inherit; padding: 12px 14px;
          border: 1px solid var(--line); border-radius: 11px; background: rgba(10,27,41,.7);
          transition: border-color .15s ease; }
        .sl-thread:hover { border-color: var(--line-strong); }
        .sl-thread.is-controversial { border-color: rgba(255,138,76,.32); background: rgba(255,110,50,.05); }
        .sl-thread__meta { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 7px;
          font-size: 10px; font-weight: 800; color: var(--muted); }
        .sl-thread__flair { color: var(--gold); letter-spacing: .06em; text-transform: uppercase; }
        .sl-thread h4 { margin: 0 0 6px; font-size: 13.5px; line-height: 1.35; font-weight: 700; }
        .sl-thread p { margin: 0; font-size: 12px; line-height: 1.45; color: var(--muted-2); }
        .sl-comments { display: grid; gap: 9px; margin: 9px 0 0; padding: 11px 14px;
          border: 1px solid var(--line); border-top: 0; border-radius: 0 0 11px 11px; background: rgba(4,12,20,.6); }
        .sl-comment { font-size: 12px; line-height: 1.45; color: var(--muted-2); padding-left: 10px;
          border-left: 2px solid rgba(150,214,235,.15); }
        .sl-comment em { display: block; font-style: normal; font-size: 10px; font-weight: 900; color: var(--cyan); margin-bottom: 3px; }
        .sl-comment.is-rival { border-left-color: rgba(255,95,109,.4); }

        .sl-pulse { border: 1px solid rgba(22,220,234,.22); border-radius: 10px; padding: 12px 14px;
          background: rgba(22,220,234,.05); margin-bottom: 12px; }
        .sl-pulse span { font-size: 9.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
        .sl-pulse strong { display: block; font-size: 22px; font-weight: 900; margin: 5px 0 3px; }
        .sl-pulse p { margin: 0; font-size: 11px; color: var(--muted); font-weight: 700; }

        .sl-trend { display: flex; align-items: center; gap: 10px; padding: 8px 0;
          border-bottom: 1px solid rgba(150,214,235,.07); font-size: 11.5px; font-weight: 800; }
        .sl-trend:last-child { border-bottom: 0; }
        .sl-trend b { width: 18px; font-size: 12px; font-weight: 900; color: var(--muted); }
        .sl-trend span { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted-2); }
        .sl-trend em { font-style: normal; font-size: 10px; font-weight: 900; letter-spacing: .06em;
          text-transform: uppercase; color: var(--gold); }

        /* ------------- insiders ------------- */
        .sl-insider { text-align: left; cursor: pointer; color: inherit; padding: 12px 14px;
          border: 1px solid var(--line); border-radius: 11px; background: rgba(10,27,41,.7);
          transition: border-color .15s ease, transform .12s ease; }
        .sl-insider:hover { border-color: var(--line-strong); transform: translateX(2px); }
        .sl-insider__head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 5px; }
        .sl-insider__head strong { font-size: 13px; line-height: 1.35; font-weight: 700; }
        .sl-insider__head em { flex-shrink: 0; font-style: normal; font-size: 9.5px; font-weight: 900;
          letter-spacing: .1em; text-transform: uppercase; color: var(--gold); }
        .sl-insider p { margin: 0 0 9px; font-size: 12.5px; line-height: 1.45; color: var(--muted-2); }
        .sl-insider__meta { display: flex; gap: 12px; flex-wrap: wrap; font-size: 9.5px; font-weight: 900;
          letter-spacing: .07em; text-transform: uppercase; color: var(--cyan); }

        /* ------------- press room ------------- */
        .sl-press { border: 1px solid var(--line); border-radius: 12px; margin-bottom: 14px; overflow: hidden;
          background: linear-gradient(180deg, rgba(10,27,41,.9), rgba(5,14,23,.92)); }
        .sl-press__head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 14px 18px;
          border-bottom: 1px solid var(--line); background: linear-gradient(90deg, rgba(22,220,234,.06), transparent 60%); }
        .sl-press__head strong { font-size: 15px; font-weight: 800; }
        .sl-press__head span { font-size: 10.5px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
        .sl-press__mics { margin-left: auto; font-size: 9.5px; font-weight: 900; letter-spacing: .14em;
          text-transform: uppercase; color: var(--red); }
        .sl-press__body { padding: 16px 18px; }
        .sl-press__summary { margin: 0 0 14px; font-size: 13px; line-height: 1.55; color: var(--muted-2); }
        .sl-press__q { padding: 14px 0; border-top: 1px solid var(--line); }
        .sl-press__q:first-child { border-top: 0; padding-top: 0; }
        .sl-press__reporter { display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
          font-size: 10px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; color: var(--cyan); }
        .sl-press__reporter i { font-style: normal; width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); }

        /* ------------- archive ------------- */
        .sl-era { border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; margin-bottom: 14px;
          background: linear-gradient(180deg, rgba(10,27,41,.86), rgba(5,14,23,.9)); }
        .sl-era__head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px;
          padding-bottom: 10px; border-bottom: 1px solid var(--line); }
        .sl-era__head h3 { margin: 0; font-size: 16px; font-weight: 800; letter-spacing: .02em; }
        .sl-era__head span { font-size: 10px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
        .sl-era__themes { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
        .sl-era__themes span { font-size: 9.5px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase;
          border: 1px solid rgba(201,162,39,.3); color: var(--brass); border-radius: 4px; padding: 4px 8px; }
        .sl-era__stories { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px,1fr)); gap: 8px; }
        .sl-era__stories button { text-align: left; cursor: pointer; color: inherit; padding: 10px 12px;
          border: 1px solid transparent; border-radius: 8px; background: rgba(255,255,255,.02); }
        .sl-era__stories button:hover { border-color: var(--line-2); background: rgba(22,220,234,.06); }
        .sl-era__stories strong { display: block; font-size: 12.5px; line-height: 1.35; margin-bottom: 3px; }
        .sl-era__stories em { font-style: normal; font-size: 10px; font-weight: 800; color: var(--muted); }

        /* ------------- meetings room ------------- */
        .sl-room { position: relative; border: 1px solid var(--line); border-radius: 14px; padding: 20px 22px;
          background: linear-gradient(180deg, rgba(10,27,41,.86), rgba(5,14,23,.9)); }
        .sl-room--cinematic {
          overflow: hidden;
          border-color: rgba(201,162,39,.28);
          background:
            radial-gradient(120% 80% at 50% -10%, rgba(233,168,60,.12), transparent 55%),
            linear-gradient(180deg, rgba(12,22,34,.96), rgba(4,10,18,.98));
          box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 18px 48px rgba(0,0,0,.35);
        }
        .sl-room__vignette {
          pointer-events: none; position: absolute; inset: 0; border-radius: inherit;
          background: radial-gradient(ellipse at center, transparent 35%, rgba(0,0,0,.45) 100%);
        }
        .sl-room__head { margin-bottom: 18px; position: relative; z-index: 1; }
        .sl-room__head--meet { margin-bottom: 22px; }
        .sl-room__head .sl-back { margin-bottom: 12px; }
        .sl-room__kicker { margin: 0 0 4px; font-size: 9.5px; font-weight: 900; letter-spacing: .18em;
          text-transform: uppercase; color: var(--brass); }
        .sl-room__head h2 { margin: 0 0 5px; font-size: 22px; font-weight: 800; letter-spacing: .01em; }
        .sl-room__sub { font-size: 12.5px; color: var(--muted-2); }

        .sl-pm-hero { display: flex; gap: 18px; align-items: center; }
        .sl-pm-hero h2 { margin: 0 0 4px; font-size: 24px; }

        .sl-pm-rel-panel {
          position: relative; z-index: 1; margin-bottom: 16px; padding: 14px 16px;
          border: 1px solid rgba(201,162,39,.22); border-radius: 12px;
          background: linear-gradient(180deg, rgba(233,168,60,.07), rgba(255,255,255,.02));
        }
        .sl-pm-rel-panel__head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
        .sl-pm-rel-panel__head span { font-size: 9.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
        .sl-pm-rel-panel__grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .sl-pm-rel-panel__note { margin: 10px 0 0; font-size: 12px; line-height: 1.5; color: var(--muted-2); }

        .sl-pm-stat__head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
        .sl-pm-stat__head span { font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
        .sl-pm-stat__head strong { font-size: 18px; font-weight: 900; font-variant-numeric: tabular-nums; }
        .sl-pm-stat__track { height: 6px; border-radius: 999px; background: rgba(150,214,235,.1); overflow: hidden; }
        .sl-pm-stat__fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2f9c68, var(--green)); }
        .sl-pm-stat--hot .sl-pm-stat__fill { background: linear-gradient(90deg, #b03744, var(--red)); }
        .sl-pm-stat--warm .sl-pm-stat__fill { background: linear-gradient(90deg, #a5731c, var(--gold)); }

        .sl-pm-cause {
          position: relative; z-index: 1; margin-bottom: 16px; padding: 12px 14px;
          border: 1px solid rgba(22,220,234,.18); border-radius: 10px; background: rgba(22,220,234,.04);
        }
        .sl-pm-cause h4 { margin: 0 0 8px; font-size: 9.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); }
        .sl-pm-cause ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
        .sl-pm-cause li { display: flex; justify-content: space-between; gap: 10px; font-size: 12.5px; color: rgba(234,247,252,.9); }
        .sl-pm-cause li em { font-style: normal; font-weight: 900; color: var(--gold); font-variant-numeric: tabular-nums; }
        .sl-pm-cause-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
        .sl-pm-cause-chips em { font-style: normal; font-size: 9.5px; font-weight: 800; letter-spacing: .04em;
          text-transform: uppercase; color: var(--cyan); border: 1px solid rgba(22,220,234,.22); padding: 2px 7px; border-radius: 999px; }

        .sl-pm-effects { position: relative; z-index: 1; margin-top: 14px; }
        .sl-pm-effects h4 { margin: 0 0 10px; font-size: 9.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); }
        .sl-pm-effects__grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
        .sl-pm-effect {
          padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px;
          background: rgba(255,255,255,.02);
        }
        .sl-pm-effect__group { display: block; font-size: 9px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
        .sl-pm-effect strong { display: block; font-size: 12.5px; line-height: 1.35; margin-bottom: 4px; }
        .sl-pm-effect p { margin: 0; font-size: 12px; color: var(--muted-2); }
        .sl-pm-effect p em { font-style: normal; font-weight: 900; margin-left: 6px; }
        .sl-pm-effect--pos { border-color: rgba(82,223,148,.28); background: rgba(82,223,148,.05); }
        .sl-pm-effect--pos p em { color: var(--green); }
        .sl-pm-effect--neg { border-color: rgba(255,95,109,.28); background: rgba(255,95,109,.05); }
        .sl-pm-effect--neg p em { color: var(--red); }

        .sl-pm-decision { position: relative; z-index: 1; margin-top: 18px; }
        .sl-pm-decision h4 { margin: 0 0 10px; font-size: 9.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--brass); }

        .sl-meeting-outcome--cinematic {
          position: relative; z-index: 1; overflow: hidden; margin: 16px 0;
          padding: 16px 18px; border: 1px solid rgba(82,223,148,.28); border-radius: 12px;
          background: linear-gradient(180deg, rgba(82,223,148,.08), rgba(5,14,23,.7));
        }
        .sl-meeting-outcome__glow {
          position: absolute; inset: -40% auto auto -20%; width: 220px; height: 220px; border-radius: 50%;
          background: radial-gradient(circle, rgba(126,224,176,.18), transparent 70%);
        }
        .sl-meeting-outcome__kicker { margin: 0 0 6px; font-size: 9.5px; font-weight: 900; letter-spacing: .16em; text-transform: uppercase; color: var(--green); }
        .sl-meeting-outcome h4 { margin: 0 0 8px; font-size: 18px; line-height: 1.35; font-weight: 800; }
        .sl-meeting-outcome--negative {
          border-color: rgba(255,95,109,.35);
          background: linear-gradient(180deg, rgba(255,95,109,.1), rgba(5,14,23,.7));
        }
        .sl-meeting-outcome--negative .sl-meeting-outcome__kicker { color: var(--red); }
        .sl-meeting-outcome--positive {
          border-color: rgba(82,223,148,.28);
          background: linear-gradient(180deg, rgba(82,223,148,.08), rgba(5,14,23,.7));
        }
        .sl-meeting-outcome--positive .sl-meeting-outcome__kicker { color: var(--green); }
        .sl-meeting-outcome__summary { margin: 0 0 12px; font-size: 13px; line-height: 1.55; color: var(--muted-2); }
        .sl-choice-empty { font-size: 12px; color: var(--muted); text-align: center; padding: 10px 0; }
        .sl-block { margin-bottom: 22px; }
        .sl-block__bar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }
        .sl-block__title { display: flex; align-items: center; gap: 9px; margin: 0 0 10px;
          font-size: 10px; font-weight: 900; letter-spacing: .16em; text-transform: uppercase; color: var(--cyan); }
        .sl-block__bar .sl-block__title { margin: 0; }
        .sl-block__title em { font-style: normal; font-size: 9.5px; padding: 2px 7px; border-radius: 999px;
          background: rgba(22,220,234,.15); color: var(--cyan); }
        .sl-block__title--alert { color: var(--gold); }
        .sl-block__title--alert em { background: var(--gold); color: #2a1f06; }

        .sl-roster { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 8px; }
        .sl-rosterrow { display: flex; align-items: center; gap: 11px; width: 100%; text-align: left; cursor: pointer;
          padding: 9px 12px; border: 1px solid var(--line); border-radius: 10px; color: inherit;
          background: rgba(255,255,255,.02); transition: border-color .15s ease, transform .12s ease; }
        .sl-rosterrow:hover { border-color: var(--line-strong); transform: translateY(-1px); }
        .sl-rosterrow.is-flagged { border-color: rgba(233,168,60,.32); background: rgba(233,168,60,.05); }
        .sl-rosterrow__main { flex: 1; min-width: 0; }
        .sl-rosterrow__main strong { display: block; font-size: 13px; font-weight: 700; }
        .sl-rosterrow__main span { display: block; font-size: 11px; font-weight: 700; color: var(--muted); margin-top: 1px; }
        .sl-rosterrow__main em { display: block; font-style: normal; font-size: 10.5px; font-weight: 800; color: var(--cyan); margin-top: 2px; }
        .sl-tagbadge { flex-shrink: 0; font-size: 8.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase;
          color: var(--gold); border: 1px solid rgba(233,168,60,.4); padding: 3px 7px; border-radius: 4px; }

        .sl-pm-identity { display: flex; gap: 16px; align-items: center; }
        .sl-pm-identity h2 { margin: 0 0 4px; }
        .sl-pm-identity__line { margin: 0 0 6px; font-size: 12.5px; font-weight: 800; color: var(--muted-2); }
        .sl-pm-rel { margin: 0; font-size: 12.5px; }
        .sl-pm-rel em { font-style: normal; font-weight: 900; letter-spacing: .04em; }
        .sl-pm-rel.is-strong { color: #7ee0b0; }
        .sl-pm-rel.is-strained { color: var(--ember); }
        .sl-pm-rel.is-neutral { color: var(--muted-2); }

        .sl-request { border: 1px solid rgba(233,168,60,.3); border-left: 3px solid var(--gold); border-radius: 0 10px 10px 0;
          background: rgba(233,168,60,.05); padding: 14px 16px; margin-bottom: 12px; position: relative; z-index: 1; }
        .sl-request--cinematic {
          background: linear-gradient(135deg, rgba(233,168,60,.08), rgba(5,14,23,.4));
          box-shadow: 0 10px 28px rgba(0,0,0,.18);
        }
        .sl-request__flag { display: inline-block; margin-bottom: 8px; font-size: 9px; font-weight: 900;
          letter-spacing: .14em; text-transform: uppercase; color: #2a1f06; background: var(--gold); padding: 3px 8px; border-radius: 4px; }
        .sl-request__head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
        .sl-request__head strong { font-size: 14px; font-weight: 800; }
        .sl-request__head span { font-size: 11.5px; font-weight: 800; color: var(--muted); }
        .sl-request h3 { margin: 0 0 7px; font-size: 14px; font-weight: 800; }
        .sl-request p { margin: 0 0 8px; font-size: 12.5px; line-height: 1.5; color: rgba(234,247,252,.85); }
        .sl-request blockquote { margin: 0 0 10px; padding-left: 13px; border-left: 2px solid rgba(233,168,60,.5);
          font-size: 13.5px; line-height: 1.55; font-style: italic; color: #f3dcae; }

        .sl-subtabs.sl-pm { margin: 16px 0 14px; }
        .sl-dialogue { display: grid; gap: 14px; margin-bottom: 18px; position: relative; z-index: 1; }
        .sl-dialogue--cinematic { padding: 14px 0 6px; }
        .sl-dialogue__line { padding-left: 14px; border-left: 2px solid var(--line-2);
          animation: slRise .3s cubic-bezier(.2,.7,.3,1) both; }
        .sl-dialogue__line.is-gm { border-left-color: var(--gold); }
        .sl-dialogue__line.is-player { border-left-color: var(--cyan); }
        .sl-dialogue__line em { display: block; font-style: normal; font-size: 9.5px; font-weight: 900;
          letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: 5px; }
        .sl-dialogue__line p { margin: 0; font-size: 15px; line-height: 1.65; }
        .sl-dialogue__line.is-player p { color: rgba(234,247,252,.94); font-size: 16px; }

        .sl-ovr { border: 1px solid rgba(22,220,234,.22); border-radius: 10px; background: rgba(22,220,234,.05);
          padding: 12px 14px; margin-bottom: 14px; position: relative; z-index: 1; }
        .sl-ovr--cinematic { border-color: rgba(22,220,234,.3); background: linear-gradient(180deg, rgba(22,220,234,.08), rgba(255,255,255,.02)); }
        .sl-ovr h4 { margin: 0 0 8px; font-size: 10px; font-weight: 900; letter-spacing: .12em;
          text-transform: uppercase; color: var(--cyan); }
        .sl-ovr ul { margin: 0; padding-left: 18px; font-size: 12.5px; line-height: 1.55; color: var(--muted-2); }

        .sl-topics section { margin-bottom: 18px; }
        .sl-topics h4 { margin: 0 0 9px; font-size: 9.5px; font-weight: 900; letter-spacing: .14em;
          text-transform: uppercase; color: var(--muted); }
        .sl-topic-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px,1fr)); gap: 7px; }
        .sl-topic { text-align: left; cursor: pointer; color: inherit; padding: 10px 12px;
          border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.02);
          font-size: 12.5px; font-weight: 700; transition: border-color .15s ease, background .15s ease; }
        .sl-topic:hover:not(:disabled) { border-color: var(--line-strong); background: var(--cyan-dim); }
        .sl-topic:disabled { opacity: .5; cursor: not-allowed; }

        .sl-stack { display: grid; gap: 7px; }
        .sl-promise { display: flex; justify-content: space-between; align-items: center; gap: 12px;
          padding: 10px 13px; border: 1px solid var(--line); border-left: 3px solid var(--green); border-radius: 0 8px 8px 0;
          background: rgba(82,223,148,.04); font-size: 12.5px; }
        .sl-promise strong { font-weight: 700; }
        .sl-promise span { font-size: 10.5px; font-weight: 800; color: var(--muted); white-space: nowrap; }
        .sl-histrow { display: grid; grid-template-columns: 96px 1fr; gap: 12px; padding: 10px 0;
          border-bottom: 1px solid rgba(150,214,235,.08); }
        .sl-histrow time { font-size: 10.5px; font-weight: 800; color: var(--muted); }
        .sl-histrow strong { display: block; font-size: 12.5px; font-weight: 700; }
        .sl-histrow p { margin: 3px 0 0; font-size: 11.5px; color: var(--muted-2); }
        .sl-notice { margin: 12px 0 0; font-size: 12.5px; font-weight: 700; color: var(--cyan); }
        .sl-muted { color: var(--muted); font-size: 12.5px; margin: 0; }

        /* ------------- empty ------------- */
        .sl-empty { display: grid; justify-items: center; text-align: center; gap: 6px; padding: 46px 24px;
          border: 1px solid var(--line); border-radius: 14px;
          background: linear-gradient(180deg, rgba(9,24,37,.6), rgba(4,12,20,.6)); }
        .sl-empty__mark { font-size: 30px; color: rgba(22,220,234,.35); margin-bottom: 6px; animation: slPulse 3.4s ease-in-out infinite; }
        .sl-empty__kicker { margin: 0; font-size: 9.5px; font-weight: 900; letter-spacing: .2em; text-transform: uppercase; color: var(--cyan); }
        .sl-empty h2 { margin: 2px 0 4px; font-size: 17px; font-weight: 800; letter-spacing: .02em; }
        .sl-empty__body { margin: 0; max-width: 46ch; font-size: 12.5px; line-height: 1.6; color: var(--muted); }

        /* ------------- breaking ------------- */
        .sl-breaking { position: fixed; inset: 0; z-index: 12000; display: grid; place-items: center; padding: 24px;
          background: rgba(1,5,10,.82); backdrop-filter: blur(6px); cursor: pointer;
          animation: slFade .2s ease both; }
        .sl-breaking__card { position: relative; overflow: hidden; cursor: default; width: min(600px, 100%);
          border: 1px solid rgba(255,95,109,.5); border-top: 4px solid var(--red); border-radius: 4px;
          background: linear-gradient(165deg, rgba(46,9,14,.98), rgba(7,19,30,.99));
          box-shadow: 0 30px 80px rgba(0,0,0,.6), 0 0 0 1px rgba(255,95,109,.14);
          animation: slSlam .34s cubic-bezier(.16,.9,.3,1) both; }
        .sl-breaking__strip { display: flex; align-items: center; gap: 9px; padding: 10px 20px;
          border-bottom: 1px solid rgba(255,95,109,.28); background: rgba(255,95,109,.1); }
        .sl-breaking__strip i { font-style: normal; width: 8px; height: 8px; border-radius: 50%; background: var(--red);
          animation: slPulse 1.2s ease-in-out infinite; }
        .sl-breaking__strip strong { font-size: 10.5px; font-weight: 900; letter-spacing: .2em; text-transform: uppercase; color: #ffb4bb; }
        .sl-breaking__strip span { margin-left: auto; font-size: 10px; font-weight: 800; color: rgba(255,180,187,.7); }
        .sl-breaking__body { padding: 20px 22px 22px; }
        .sl-breaking__body h2 { margin: 0 0 10px; font-size: 21px; line-height: 1.3; font-weight: 800; }
        .sl-breaking__body p { margin: 0 0 18px; font-size: 13.5px; line-height: 1.6; color: var(--muted-2); }
        .sl-breaking__actions { display: flex; gap: 9px; flex-wrap: wrap; }
        .sl-breaking__actions button { cursor: pointer; padding: 10px 16px; border-radius: 8px;
          border: 1px solid var(--line-2); background: rgba(255,255,255,.04); color: var(--text);
          font-size: 11.5px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; }
        .sl-breaking__actions button:hover { border-color: var(--line-strong); }
        .sl-breaking__actions button.is-primary { border-color: rgba(22,220,234,.5);
          background: linear-gradient(180deg, #2ee6f0, #12b9c9); color: #041018; }

        /* ------------- debug ------------- */
        .sl-effect-breakdown { margin-top: 8px; font-size: 12px; }
        .sl-effect-breakdown__toggle { background: none; border: none; color: var(--accent); cursor: pointer; padding: 0; font-size: 12px; }
        .sl-effect-breakdown__body { margin-top: 6px; padding: 8px 10px; border-radius: 8px; background: rgba(255,255,255,0.04); }
        .sl-effect-breakdown__body p { margin: 4px 0; }
        .sl-meeting-outcome { margin: 12px 0; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px; background: rgba(126,224,176,0.06); }
        .sl-meeting-outcome__list { margin: 8px 0 0; padding-left: 18px; font-size: 13px; }
        .sl-locker-dash__pulse { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
        .sl-locker-dash__gauge { width: 120px; height: 120px; border-radius: 50%; border: 4px solid var(--accent); display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .sl-locker-dash__culture { display: flex; flex-wrap: wrap; gap: 8px; flex: 1; }
        .sl-locker-dash__cards { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
        .sl-locker-dash--cinematic .sl-locker-dash__head { margin-bottom: 16px; }
        .sl-locker-dash--cinematic .sl-room__sub { display: block; margin-top: 6px; font-size: 12.5px; color: var(--muted-2); }
        .sl-locker-card--person { display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; }
        .sl-locker-card__body { flex: 1; min-width: 0; }
        .sl-locker-card__meta { display: block; font-size: 10.5px; font-weight: 800; color: var(--muted); margin: 2px 0 6px; }
        .sl-locker-card__body p { margin: 0 0 8px; font-size: 12px; line-height: 1.45; color: rgba(234,247,252,.88); }
        .sl-locker-card__chips { display: flex; flex-wrap: wrap; gap: 6px; }
        .sl-locker-card--story, .sl-locker-card--trigger {
          border-color: rgba(22,220,234,.22); background: linear-gradient(180deg, rgba(22,220,234,.06), rgba(255,255,255,.02));
        }
        .sl-locker-card__tag { display: inline-block; margin-bottom: 6px; font-size: 9px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--cyan); }
        .sl-locker-card--story p, .sl-locker-card--trigger p { margin: 4px 0 0; font-size: 12px; line-height: 1.45; color: var(--muted-2); }
        .sl-locker-dash__cards--wire { grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
        .sl-locker-card { padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; display: flex; flex-direction: column; gap: 4px; }
        .sl-niche-badge { font-size: 11px; font-style: normal; padding: 2px 8px; border-radius: 999px; background: rgba(126,224,176,0.15); width: fit-content; }
        .sl-consequences { display: grid; gap: 12px; }
        .sl-consequence-card { padding: 14px; border: 1px solid var(--line); border-radius: 10px; }
        .sl-consequence-card header { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
        .sl-debug { border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-top: 14px; }
        .sl-debug summary { cursor: pointer; font-size: 11px; font-weight: 800; color: var(--muted); }
        .sl-debug pre { font-size: 11px; overflow: auto; max-height: 240px; color: var(--muted-2); }

        /* ------------- keyframes ------------- */
        @keyframes slRise { from { opacity: 0; transform: translateY(9px); } to { opacity: 1; transform: none; } }
        @keyframes slFade { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slSlam {
          0% { opacity: 0; transform: translateY(-24px) scale(.965); }
          60% { opacity: 1; transform: translateY(3px) scale(1.004); }
          100% { opacity: 1; transform: none; }
        }
        @keyframes slGrow { from { transform: scaleX(0); transform-origin: left; } to { transform: scaleX(1); } }
        @keyframes slPulse { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
        @keyframes slEmber {
          0%, 100% { filter: drop-shadow(0 0 0 rgba(255,95,109,0)); }
          50% { filter: drop-shadow(0 0 7px rgba(255,95,109,.45)); }
        }
        @keyframes slTicker { from { transform: translateX(0); } to { transform: translateX(-50%); } }

        @media (prefers-reduced-motion: reduce) {
          .nhlcal-sl-root *, .nhlcal-sl-root *::before, .nhlcal-sl-root *::after {
            animation: none !important; transition: none !important;
          }
        }
      `}</style>

      <div className="sl-atmos" aria-hidden />

      {activeBreaking ? (
        <div
          className="sl-breaking"
          role="dialog"
          aria-label="Breaking news"
          onClick={() => dismissBreakingAlerts(pendingBreaking)}
        >
          <div className="sl-breaking__card" onClick={(e) => e.stopPropagation()}>
            <div className="sl-breaking__strip">
              <i aria-hidden />
              <strong>Breaking · {str(activeBreaking.level || "major").replace(/_/g, " ")}</strong>
              {pendingBreaking.length > 1 ? <span>{pendingBreaking.length} alerts queued</span> : null}
            </div>
            <div className="sl-breaking__body">
              <h2>{str(activeBreaking.headline || "Major league development")}</h2>
              {activeBreaking.summary ? <p>{activeBreaking.summary}</p> : null}
              <div className="sl-breaking__actions">
                <button
                  type="button"
                  className="is-primary"
                  onClick={() => {
                    const storyKey = str(activeBreaking.storyline_id || "");
                    if (storyKey) {
                      const match = stories.find((s) => str(s.storylineId) === storyKey || str(s.id) === storyKey);
                      if (match) {
                        setDepartment(isTradeDeskStory(match) ? "trade_desk" : "front_page");
                        openStory(match.id);
                      } else {
                        setDepartment("front_page");
                      }
                    } else {
                      setDepartment("front_page");
                    }
                    dismissBreakingAlerts(pendingBreaking);
                  }}
                >
                  Open the story
                </button>
                <button type="button" onClick={() => dismissBreakingAlerts(pendingBreaking)}>
                  Dismiss{pendingBreaking.length > 1 ? " all" : ""}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <div className="sl-app">
        {/* ---------- command bar ---------- */}
        <header className="sl-command">
          <div className="sl-command__crest">
            {(() => {
              const logo =
                resolveFranchiseTeamLogo(
                  { team_id: userTeamId(franchiseState), team_name: teamLabel(franchiseState) },
                  teamLabel(franchiseState)
                ) || "";
              return logo ? <img src={logo} alt="" /> : <strong>{playerInitials(teamLabel(franchiseState))}</strong>;
            })()}
          </div>
          <div className="sl-command__id">
            <p className="sl-command__eyebrow">Franchise newsroom</p>
            <h1>Storylines</h1>
            <p className="sl-command__sub">
              {prettyDate(calendarLabel(franchiseState))} · {teamLabel(franchiseState)}
            </p>
          </div>
          <div className="sl-command__stats">
            <div className="sl-stat">
              <strong>{narrativeStories.length}</strong>
              <span>Stories</span>
            </div>
            <div className="sl-stat">
              <strong>{tradeStories.length}</strong>
              <span>Trades</span>
            </div>
            <div className="sl-stat sl-stat--ours">
              <strong>{yourTeamCount}</strong>
              <span>Your club</span>
            </div>
            {pendingDecisions.length ? (
              <div className="sl-stat sl-stat--alert">
                <strong>{pendingDecisions.length}</strong>
                <span>On your desk</span>
              </div>
            ) : null}
          </div>
          <nav className="sl-command__nav" aria-label="Navigation">
            <button type="button" onClick={() => setScreen?.(SCREENS.CALENDAR)}>Calendar</button>
            <button type="button" onClick={() => setScreen?.(SCREENS.HUB)}>Hub</button>
          </nav>
        </header>

        {/* ---------- ticker ---------- */}
        {tickerItems.length ? (
          <div className="sl-ticker" aria-label="League wire">
            <div className="sl-ticker__flag">
              <span className="sl-ticker__dot" aria-hidden />
              {department === "trade_desk" ? "Trade wire" : "Story wire"}
            </div>
            <div className="sl-ticker__viewport">
              <div className="sl-ticker__track">
                {[...tickerItems, ...tickerItems].map((s, i) => (
                  <button
                    key={`${s.id}-${i}`}
                    type="button"
                    className="sl-ticker__item"
                    onClick={() => {
                      setDepartment(isTradeDeskStory(s) ? "trade_desk" : "front_page");
                      setFilter("all");
                      openStory(s.id);
                    }}
                  >
                    <i style={{ color: categoryMeta(s).accent }}>{categoryMeta(s).label}</i>
                    {s.headline}
                    <span className="sl-ticker__sep">◆</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : null}

        {/* ---------- departments ---------- */}
        <nav className="sl-depts" aria-label="Media departments">
          {DEPARTMENTS.map((d) => {
            let count = 0;
            if (d.id === "front_page") count = narrativeStories.length;
            if (d.id === "trade_desk") count = tradeStories.length;
            if (d.id === "player_meetings") count = meetingAlertCount;
            if (d.id === "consequences") count = asArray(narrativeUniverse?.team_sanctions).filter((s) => s.active !== false).length;
            if (d.id === "press_room") count = pressQueue.length;
            if (d.id === "locker_room") {
              const pulse = collectLockerPulse(franchiseState, { limit: 24 });
              const roomStories = stories.filter(
                (s) => s.categoryKey === "personal_life" || s.categoryKey === "locker_room"
              ).length;
              const triggers = asArray(narrativeUniverse?.recent_universe_events).length;
              count = pulse.lifeStories.length + roomStories + Math.min(triggers, 8);
            }
            return (
              <button
                key={d.id}
                type="button"
                className={department === d.id ? "is-active" : ""}
                onClick={() => {
                  setDepartment(d.id);
                  setFilter("all");
                  setOpenCaseId(null);
                }}
              >
                <em aria-hidden>{d.glyph}</em>
                {d.label}
                {count > 0 ? <span className="sl-dept-count">{count}</span> : null}
              </button>
            );
          })}
        </nav>

        {actionNotice ? (
          <div className="sl-action-notice" role="status">
            <span>{actionNotice}</span>
            <button type="button" onClick={() => setActionNotice("")}>Dismiss</button>
          </div>
        ) : null}

        {userMarket?.label && department === "trade_desk" ? (
          <p className="sl-market">
            <em>Market</em>
            {userMarket.label} · {userMarket.descriptor || userMarket.tone || "High scrutiny"}
            {userMarket.pressure_mult ? ` · pressure ×${Number(userMarket.pressure_mult).toFixed(2)}` : ""}
          </p>
        ) : null}

        {/* ================= CONTENT ================= */}

        {!hasBackend ? (
          <EmptyPanel
            kicker="League wire · idle"
            title="No coverage yet"
            body="Advance the calendar and the newsroom will start filing from backend storylines."
          />
        ) : department === "player_meetings" ? (
          <PlayerMeetingsPanel
            meetingsPayload={playerMeetingsPayload}
            busy={meetingBusy}
            onResolvePlayerRequest={handleResolvePlayerMeeting}
            onStartMeeting={handleStartPlayerMeeting}
            onAdvanceMeeting={handleAdvancePlayerMeeting}
            onRefresh={handleMeetingRefresh}
            initialPlayerId={pendingMeetingPlayerId}
          />
        ) : department === "locker_room" ? (
          <LockerRoomDashboard narrativeUniverse={narrativeUniverse} franchiseState={franchiseState} stories={stories} />
        ) : department === "consequences" ? (
          <ConsequencesPanel narrativeUniverse={narrativeUniverse} stories={stories} onOpenStory={openStory} />
        ) : department === "social" ? (
          <div className="sl-two">
            <div>
              <div className="sl-subtabs">
                {[
                  { id: "puckr", label: "Puckr" },
                  { id: "icehole", label: "IceHole" },
                  { id: "burner", label: "Burner" },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={socialSubTab === tab.id ? "is-active" : ""}
                    onClick={() => setSocialSubTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {socialSubTab === "puckr" ? (
                <div className="sl-feed">
                  {socialPosts.length ? (
                    socialPosts.map((post, i) => (
                      <button
                        key={post.id}
                        type="button"
                        className="sl-post"
                        style={{ animationDelay: `${Math.min(i, 10) * 24}ms` }}
                        onClick={() => {
                          if (post.storyId) {
                            const match = stories.find((s) => s.id === post.storyId || s.storylineId === post.storyId);
                            if (match) {
                              setDepartment(isTradeDeskStory(match) ? "trade_desk" : "front_page");
                              setFilter("all");
                            } else {
                              setDepartment("front_page");
                            }
                            openStory(post.storyId);
                          }
                        }}
                      >
                        <div className="sl-post__head">
                          <span className="sl-post__avatar" aria-hidden>{playerInitials(post.name)}</span>
                          <strong>{post.name}</strong>
                          {post.verified ? <span className="sl-post__verified">✓</span> : null}
                          <span>{post.handle}</span>
                          {post.isAgent ? <span>· agent</span> : null}
                          <em>{post.age}</em>
                        </div>
                        <p>{post.text}</p>
                        <div className="sl-post__meta">
                          {post.related && post.related !== post.text && !post.text.includes(post.related) ? (
                            <span className="sl-post__related">{post.related}</span>
                          ) : null}
                          {post.cred ? <span>{post.cred}</span> : null}
                          {post.likes != null ? (
                            <>
                              <span>{Number(post.replies || 0).toLocaleString()} replies</span>
                              <span>{Number(post.reposts || 0).toLocaleString()} reposts</span>
                              <span>{Number(post.likes || 0).toLocaleString()} likes</span>
                            </>
                          ) : null}
                        </div>
                      </button>
                    ))
                  ) : (
                    <EmptyPanel
                      kicker="Puckr · quiet"
                      title="Nothing in the last 48 hours"
                      body="Only posts from the past two franchise days appear here. Advance the calendar or trigger storylines to refresh the timeline."
                    />
                  )}
                </div>
              ) : null}

              {socialSubTab === "icehole" ? (
                <>
                  <div className="sl-pills">
                    {redditSubPills.map((pill) => (
                      <button
                        key={pill}
                        type="button"
                        className={redditSubFilter === pill ? "is-active" : ""}
                        onClick={() => setRedditSubFilter(pill)}
                      >
                        {pill}
                      </button>
                    ))}
                  </div>
                  <div className="sl-feed">
                    {redditThreads.length ? (
                      redditThreads.map((thread) => (
                        <div key={thread.id}>
                          <button
                            type="button"
                            className={`sl-thread${thread.controversial ? " is-controversial" : ""}`}
                            onClick={() => setExpandedThreadId((prev) => (prev === thread.id ? null : thread.id))}
                          >
                            <div className="sl-thread__meta">
                              <span>{thread.subreddit}</span>
                              <span className="sl-thread__flair">{thread.flair}</span>
                              <span>{thread.upvotes.toLocaleString()} ↑</span>
                              <span>{Math.round(thread.upvoteRatio * 100)}% upvoted</span>
                              <span>{thread.commentCount} comments</span>
                            </div>
                            <h4>{thread.title}</h4>
                            <p>{thread.body}</p>
                            <div className="sl-thread__meta" style={{ marginTop: 8, marginBottom: 0 }}>
                              <span>{thread.author}</span>
                              <span>{thread.createdAt}</span>
                            </div>
                          </button>
                          {expandedThreadId === thread.id && thread.comments.length ? (
                            <div className="sl-comments">
                              {thread.comments.map((c) => (
                                <div key={c.id} className={`sl-comment${c.isRival ? " is-rival" : ""}`}>
                                  <em>
                                    {c.author}
                                    {c.isRival ? " · rival fan" : ""} · {c.upvotes}↑
                                  </em>
                                  {c.text}
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ))
                    ) : (
                      <EmptyPanel
                        kicker="IceHole · quiet"
                        title="No threads yet"
                        body="Heated storylines spawn fan threads once league heat builds."
                      />
                    )}
                  </div>
                </>
              ) : null}

              {socialSubTab === "burner" ? (
                <BurnerPanel
                  sessionId={sessionId}
                  marketProfiles={narrativeUniverse?.market_profiles}
                  defaultMarketKey={userMarket?.market_key}
                  onPosted={() => refreshFranchise?.()}
                />
              ) : null}
            </div>

            <aside className="sl-rail">
              {socialSubTab === "icehole" ? (
                <div className="sl-panel">
                  <div className="sl-pulse">
                    <span>Fan pulse · IceHole</span>
                    <strong>{redditPulse.label}</strong>
                    <p>Net sentiment delta {redditPulse.net.toFixed(2)}</p>
                  </div>
                  <h3>Hot threads</h3>
                  <div className="sl-effects">
                    {redditThreads.slice(0, 6).map((t, i) => (
                      <div key={t.id} className="sl-trend">
                        <b>{i + 1}</b>
                        <span>{t.title}</span>
                        <em>{t.upvotes.toLocaleString()}↑</em>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="sl-panel">
                  <h3>Trending now</h3>
                  <div className="sl-effects">
                    {stories
                      .filter((s) => Number(s.heat) > 0 && !isRoutineLeagueTrade(s.raw || s, userTeamId(franchiseState)))
                      .sort((a, b) => {
                        const lb = str(b.breakingLevel) ? 1 : 0;
                        const la = str(a.breakingLevel) ? 1 : 0;
                        if (lb !== la) return lb - la;
                        return Number(b.heat) - Number(a.heat);
                      })
                      .slice(0, 8)
                      .map((s, i) => (
                        <div key={s.id} className="sl-trend">
                          <b>{i + 1}</b>
                          <span>{s.playerName || s.teamName || s.headline}</span>
                          <em>{heatLabel(s.heat)}</em>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </aside>
          </div>
        ) : department === "insiders" ? (
          <div className="sl-two">
            <div className="sl-feed">
              {insiderItems.length ? (
                insiderItems
                  .slice()
                  .reverse()
                  .slice(0, 48)
                  .map((item, idx) => {
                    const sid = str(item.storyline_id || item.world_event_id || idx);
                    const match = stories.find((s) => str(s.storylineId) === sid || str(s.id) === sid);
                    return (
                      <button
                        key={sid}
                        type="button"
                        className="sl-insider"
                        onClick={() => {
                          if (match) {
                            setDepartment(isTradeDeskStory(match) ? "trade_desk" : "front_page");
                            setFilter("all");
                            openStory(match.id);
                          }
                        }}
                      >
                        <div className="sl-insider__head">
                          <strong>{str(item.headline || match?.headline || "Desk note")}</strong>
                          <em>{knowledgeLevelLabel(item.public_knowledge_level)}</em>
                        </div>
                        <p>{str(item.summary || match?.summary || "")}</p>
                        <div className="sl-insider__meta">
                          <span>{str(item.reporter_name || item.source_label || "Insider")}</span>
                          {item.outlet_name ? <span>{item.outlet_name}</span> : null}
                          <span>{str(item.knowledge_type || "report").replace(/_/g, " ")}</span>
                          {item.player_name ? <span>{item.player_name}</span> : null}
                          {item.calendar_iso ? <span>{prettyDate(item.calendar_iso)}</span> : null}
                        </div>
                      </button>
                    );
                  })
              ) : (
                <EmptyPanel
                  kicker="Insiders · quiet"
                  title="No private layers yet"
                  body="Rumors, claims, and confirmed facts land here as the knowledge graph fills in."
                />
              )}
            </div>
            <aside className="sl-rail">
              {beatWriters.length ? (
                <div className="sl-panel">
                  <h3>Beat desks</h3>
                  <div className="sl-effects">
                    {beatWriters.slice(0, 10).map((writer) => (
                      <div key={str(writer.id || writer.name)} className="sl-effect">
                        <span>{str(writer.name)}</span>
                        <strong>{str(writer.specialty || writer.role || writer.outlet)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="sl-panel">
                <h3>Player dossiers</h3>
                {playerDossiers.length ? (
                  playerDossiers.slice(0, 8).map((dossier) => (
                    <DossierCard key={str(dossier.player_id || dossier.player_name)} dossier={dossier} compact />
                  ))
                ) : (
                  <p className="sl-choice-empty">Dossiers publish after the next calendar tick.</p>
                )}
              </div>
            </aside>
          </div>
        ) : department === "press_room" ? (
          <div>
            {pressOutcome ? (
              <MeetingOutcomePanel
                outcome={pressOutcome}
                kicker="Press conference"
                onDismiss={() => setPressOutcome(null)}
              />
            ) : null}
            {pressQueue.length ? (
              pressQueue.map((press) => (
                <article key={str(press.id)} className="sl-press">
                  <div className="sl-press__head">
                    <strong>{str(press.headline || "Media availability scheduled")}</strong>
                    {press.player_name ? <span>{press.player_name}</span> : null}
                    <span className="sl-press__mics">
                      {heatLabel(press.heat) || "Room is filling"}
                    </span>
                  </div>
                  <div className="sl-press__body">
                    {press.summary ? <p className="sl-press__summary">{press.summary}</p> : null}
                    {press.context?.record ? (
                      <p className="sl-press__record">
                        Team record: <strong>{str(press.context.record)}</strong>
                        {press.context.league_rank ? (
                          <span> · Rank #{Number(press.context.league_rank)}</span>
                        ) : null}
                      </p>
                    ) : null}
                    {asArray(press.context_triggers).length ? (
                      <div className="sl-press__context" aria-label="Active story triggers">
                        {asArray(press.context_triggers).map((t) => (
                          <span key={str(t.code)} className="sl-press__trigger" title={str(t.label)}>
                            ✓ {str(t.label)}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    {asArray(press.questions).map((q) => {
                      const answeredQuestions = new Set(asArray(press.answered_questions).map((id) => str(id)));
                      const questionAnswered = answeredQuestions.has(str(q.id));
                      return (
                      <div key={str(q.id)} className={`sl-press__q${questionAnswered ? " sl-press__q--done" : ""}`}>
                        <div className="sl-press__reporter">
                          <i aria-hidden />
                          {str(q.reporter_name || "Reporter")}
                          {q.outlet ? ` · ${q.outlet}` : ""}
                          {questionAnswered ? <em className="sl-press__answered">Answered</em> : null}
                        </div>
                        {asArray(q.context_tags).length ? (
                          <div className="sl-press__q-tags">
                            {asArray(q.context_tags).map((tag) => (
                              <span key={str(tag)} className="sl-press__q-tag">
                                ✓ {str(tag).replace(/_/g, " ")}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        <p className="sl-press__question">{str(q.question || "")}</p>
                        <div className="sl-choices sl-choices--press">
                          {asArray(q.responses).map((resp) => {
                            const sid = str(press.storyline_id || press.id);
                            const choiceId = `${str(q.id)}:${str(resp.id)}`;
                            const busy = busyChoice === `${sid}:${choiceId}`;
                            return (
                              <ResponseChoiceButton
                                key={resp.id}
                                response={resp}
                                className="sl-choice sl-choice--press"
                                disabled={Boolean(busyChoice) || questionAnswered}
                                busy={busy}
                                onClick={() => handlePressResponse(press, str(q.id), str(resp.id))}
                              />
                            );
                          })}
                        </div>
                      </div>
                    );})}
                  </div>
                </article>
              ))
            ) : (
              <EmptyPanel
                kicker="Press room · clear"
                title="No scheduled availability"
                body="When heat builds around your club, reporters will queue questions for your next media session."
              />
            )}
          </div>
        ) : department === "archive" ? (
          <div>
            {narrativeEras.length ? (
              narrativeEras
                .slice()
                .reverse()
                .map((era) => (
                  <article key={str(era.season)} className="sl-era">
                    <div className="sl-era__head">
                      <h3>{str(era.label || `Season ${era.season}`)}</h3>
                      <span>{Number(era.story_count || 0)} archived beats</span>
                    </div>
                    {asArray(era.themes).length ? (
                      <div className="sl-era__themes">
                        {era.themes.map((theme) => (
                          <span key={theme}>{theme}</span>
                        ))}
                      </div>
                    ) : null}
                    <div className="sl-era__stories">
                      {asArray(era.top_stories).map((story, idx) => (
                        <button
                          key={str(story.storyline_id || story.headline || idx)}
                          type="button"
                          onClick={() => {
                            const match = stories.find(
                              (s) => str(s.storylineId) === str(story.storyline_id) || s.headline === story.headline
                            );
                            if (match) {
                              setDepartment(isTradeDeskStory(match) ? "trade_desk" : "front_page");
                              setFilter("all");
                              openStory(match.id);
                            }
                          }}
                        >
                          <strong>{str(story.headline || "Archived beat")}</strong>
                          <em>
                            {str(story.category || "storyline")}
                            {story.heat != null ? ` · heat ${Math.round(Number(story.heat))}` : ""}
                            {story.calendar_iso ? ` · ${prettyDate(story.calendar_iso)}` : ""}
                          </em>
                        </button>
                      ))}
                    </div>
                  </article>
                ))
            ) : narrativeArchive.length ? (
              <article className="sl-era">
                <div className="sl-era__head">
                  <h3>League archive</h3>
                  <span>{narrativeArchive.length} beats on file</span>
                </div>
                <div className="sl-era__stories">
                  {narrativeArchive
                    .slice()
                    .reverse()
                    .slice(0, 24)
                    .map((story, idx) => (
                      <button
                        key={str(story.storyline_id || story.headline || idx)}
                        type="button"
                        onClick={() => {
                          const match = stories.find((s) => str(s.storylineId) === str(story.storyline_id));
                          if (match) {
                            setDepartment(isTradeDeskStory(match) ? "trade_desk" : "front_page");
                            setFilter("all");
                            openStory(match.id);
                          }
                        }}
                      >
                        <strong>{str(story.headline || "Archived beat")}</strong>
                        <em>{prettyDate(story.calendar_iso) || str(story.season || "—")}</em>
                      </button>
                    ))}
                </div>
              </article>
            ) : (
              <EmptyPanel
                kicker="Archive · empty"
                title="No sealed eras yet"
                body="Completed seasons are preserved here — themes, top stories, and the beats that defined them."
              />
            )}
          </div>
        ) : (department === "front_page" || department === "trade_desk") && deskStories.length === 0 && !openCase ? (
          <EmptyPanel
            kicker={department === "trade_desk" ? "Trade desk" : "Story desk"}
            title={department === "trade_desk" ? "No trades on the wire" : "No stories filed yet"}
            body={
              department === "trade_desk"
                ? "Completed deals and trade rumors live here. Narrative beats sit on the Stories desk."
                : "Locker-room, life, injury, and on-ice beats file here. Deals sit on the Trades desk. Advance a few days to fill this tray."
            }
          />
        ) : stories.length === 0 ? (
          <EmptyPanel
            kicker="League wire · idle"
            title="Wire standing by"
            body="No active storylines on file. Coverage appears as the season generates league beats."
          />
        ) : openCase ? (
          /* ================= CASE FILE ================= */
          <div className="sl-case" ref={caseRef}>
            <div className="sl-case__main">
              <div className="sl-case__hero">
                <div style={{ display: "grid", gap: 12, justifyItems: "center" }}>
                  <StoryFace story={openCase} size={104} />
                  <HeatRing value={openCase.heat} size={62} />
                </div>
                <div className="sl-case__hero-main">
                  <div className="sl-case__crumbs">
                    <button type="button" className="sl-back" onClick={closeStory}>
                      ← {department === "trade_desk" ? "Trades" : "Stories"}
                    </button>
                    <CategoryTag story={openCase} size="md" />
                    {openCase.requiresAction ? <span className="sl-card__decision">Decision required</span> : null}
                    <StatusPill story={openCase} />
                    <ScoreBadge score={storyScore(openCase)} size="sm" />
                  </div>
                  <h2 className="sl-case__title">{openCase.headline}</h2>
                  <div className="sl-case__byline">
                    {openCase.reporterName || openCase.sourceLabel ? (
                      <span>
                        Filed by <b>{openCase.reporterName || openCase.sourceLabel}</b>
                        {openCase.outletName ? ` · ${openCase.outletName}` : ""}
                      </span>
                    ) : null}
                    <span>{prettyDate(openCase.date)} · {openCase.ageLabel}</span>
                    {credibilityLabel(openCase.credibility) ? (
                      <span>Sourcing: <b>{credibilityLabel(openCase.credibility)}</b></span>
                    ) : null}
                    {socialCountFor(openCase) ? (
                      <span>{formatCount(socialCountFor(openCase))} social posts</span>
                    ) : null}
                  </div>
                  {openCase.summary ? <p className="sl-case__lede">{openCase.summary}</p> : null}
                </div>
              </div>

              <div className="sl-case__body">
                {openCase.triggerReasons?.length ? (
                  <MeetingCausePanel reasons={openCase.triggerReasons} title="Why this story fired" />
                ) : openCase.triggerReason ? (
                  <section className="sl-case__section">
                    <h4>Why this story fired</h4>
                    <p className="sl-case__prose">{openCase.triggerReason}</p>
                  </section>
                ) : null}
                {openCase.description && openCase.description !== openCase.summary ? (
                  <p className="sl-case__prose">{openCase.description}</p>
                ) : null}

                {isRumourStory(openCase) || isTradeDeskStory(openCase) ? (
                  <TradeSummaryPanel story={openCase} />
                ) : null}
                <ConductChannels story={openCase} />
                {selectedDossier ? <DossierCard dossier={selectedDossier} /> : null}

                <ArcSpine beats={arcTimeline} fallbackStory={openCase} />

                <nav className="sl-tabs">
                  {DETAIL_TABS.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      className={activeTab === t.id ? "is-active" : ""}
                      onClick={() => setActiveTab(t.id)}
                    >
                      {t.label}
                    </button>
                  ))}
                </nav>

                <div className="sl-tabpanel">
                  {activeTab === "details" ? (
                    <div className="sl-cols">
                      <div>
                        <h4>Information</h4>
                        {infoRows.length ? (
                          infoRows.map(([label, val]) => (
                            <div key={label} className="sl-kv">
                              <span>{label}</span>
                              <span>{val}</span>
                            </div>
                          ))
                        ) : (
                          <p className="sl-muted">No sourcing details on file.</p>
                        )}
                        {Object.keys(openCase.evidence || {}).length ? (
                          <div className="sl-nums">
                            {Object.entries(openCase.evidence)
                              .slice(0, 4)
                              .map(([k, v]) => (
                                <div key={k} className="sl-num" title={`${formatEffectLabel(k)}: ${v}`}>
                                  <strong>{String(v)}</strong>
                                  <span>{formatEffectLabel(k)}</span>
                                </div>
                              ))}
                          </div>
                        ) : null}
                      </div>

                      <div>
                        <h4>Parties involved</h4>
                        {parties.length ? (
                          parties.map((p) => (
                            <div key={p.label} className="sl-kv">
                              <span>{p.label}</span>
                              <span>{p.name}</span>
                            </div>
                          ))
                        ) : (
                          <p className="sl-muted">No named parties on file.</p>
                        )}
                      </div>

                      <div>
                        <h4>Key factors</h4>
                        <ul className="sl-factors">
                          {keyFactors.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                        {openCase.effectSummary ? (
                          <p className="sl-muted" style={{ marginTop: 12 }}>{openCase.effectSummary}</p>
                        ) : null}
                      </div>
                    </div>
                  ) : null}

                  {activeTab === "related" ? (
                    <div className="sl-linklist">
                      {relatedStories.length ? (
                        relatedStories.slice(0, 8).map((r) => (
                          <button key={r.id} type="button" onClick={() => openStory(r.id)}>
                            <CategoryTag story={r} />
                            <strong>{r.headline}</strong>
                            <em>{r.ageLabel || "—"}{heatLabel(r.heat) ? ` · ${heatLabel(r.heat)}` : ""}</em>
                          </button>
                        ))
                      ) : (
                        <p className="sl-muted">No related coverage yet.</p>
                      )}
                    </div>
                  ) : null}

                  {activeTab === "rumors" ? (
                    <div className="sl-linklist">
                      {leagueRumours.length ? (
                        leagueRumours.map((r) => (
                          <button key={r.id} type="button" onClick={() => openStory(r.id)}>
                            <em>{r.playerName || r.teamName || "League"}</em>
                            <strong>{r.headline}</strong>
                            <em>
                              {heatLabel(r.heat) ? `Heat: ${heatLabel(r.heat)}` : ""}
                              {credibilityLabel(r.credibility) ? ` · ${credibilityLabel(r.credibility)}` : ""}
                            </em>
                          </button>
                        ))
                      ) : (
                        <p className="sl-muted">Trade wire is quiet.</p>
                      )}
                    </div>
                  ) : null}

                  {activeTab === "history" ? (
                    <div>
                      {arcTimeline.length ? (
                        <ol className="sl-spine__list" style={{ paddingLeft: 22 }}>
                          {arcTimeline.map((beat, i) => (
                            <li key={beat.id || i} className={i === arcTimeline.length - 1 ? "is-latest" : ""}>
                              <span className="sl-spine__dot" aria-hidden />
                              <time>{prettyDate(beat.date)}</time>
                              <strong>{beat.headline}</strong>
                              {beat.summary ? <p>{beat.summary}</p> : null}
                            </li>
                          ))}
                        </ol>
                      ) : (
                        <p className="sl-muted">No prior beats on file for this story.</p>
                      )}
                      {openCase.repeatCount > 0 ? (
                        <p className="sl-muted" style={{ marginTop: 10 }}>
                          Beat #{openCase.repeatCount + 1}
                          {openCase.escalatedFrom ? ` · escalated from ${openCase.escalatedFrom}` : ""}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                </div>

                <section className="sl-case__section" style={{ marginTop: 22, paddingTop: 16, borderTop: "1px solid var(--line)" }}>
                  <h4>What comes next</h4>
                  <p className="sl-case__prose" style={{ margin: 0 }}>{deriveFollowUp(openCase)}</p>
                </section>

                <div className="sl-case__foot">
                  <span>Last updated {prettyDate(openCase.date) || openCase.ageLabel || "—"}</span>
                  <span>ID {openCase.storylineId || openCase.id}</span>
                </div>
              </div>
            </div>

            <aside className="sl-rail">
              <div className={`sl-panel${choiceOptions.length ? " sl-panel--desk" : ""}`}>
                <h3>{choiceOptions.length ? "Your call" : "GM decisions"}</h3>
                {choiceOptions.length ? (
                  <div className="sl-choices">
                    {choiceOptions.map((opt) => {
                      const busy = busyChoice === `${openCase.storylineId}:${opt.id}`;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          className="sl-choice sl-choice--lead"
                          style={{ width: "100%", flex: "unset" }}
                          disabled={Boolean(busyChoice)}
                          onClick={() => handleResolve(selectedChoice?.storyline_id || openCase.storylineId, opt.id)}
                        >
                          <strong>{opt.label}</strong>
                          {opt.effect_summary ? <span>{opt.effect_summary}</span> : null}
                          {busy ? <em>Applying…</em> : null}
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <p className="sl-choice-empty">Nothing to decide here. The story develops on its own.</p>
                )}
              </div>

              {Object.keys(openCase.effects || {}).length ? (
                <div className="sl-panel">
                  <h3>Potential effects</h3>
                  <div className="sl-effects">
                    {Object.entries(openCase.effects)
                      .slice(0, 8)
                      .map(([k, v]) => (
                        <div key={k} className={`sl-effect ${effectPillClass(v)}`}>
                          <span>{formatEffectLabel(k)}</span>
                          <strong>
                            {Number(v) > 0 ? "+" : ""}
                            {String(v)}
                          </strong>
                        </div>
                      ))}
                  </div>
                </div>
              ) : null}

              <div className="sl-panel">
                <h3>Organizational pressure</h3>
                {userOrg ? (
                  <PressureBars org={userOrg} />
                ) : (
                  <p className="sl-choice-empty">No pressure readings on file.</p>
                )}
              </div>

              {openCase.gmKnowsMore || knowledgeLevelLabel(openCase.publicKnowledgeLevel) ? (
                <div className="sl-panel">
                  <h3>Knowledge layers</h3>
                  {openCase.gmKnowsMore ? (
                    <p style={{ margin: "0 0 8px", fontSize: 12.5, fontWeight: 700, color: "#8ef0b8" }}>
                      You know more than the public sees.
                    </p>
                  ) : null}
                  {knowledgeLevelLabel(openCase.publicKnowledgeLevel) ? (
                    <div className="sl-kv" style={{ borderBottom: 0 }}>
                      <span>Public knowledge</span>
                      <span>{knowledgeLevelLabel(openCase.publicKnowledgeLevel)}</span>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </aside>
          </div>
        ) : (
          /* ================= NEWSROOM ================= */
          <>
            {department === "front_page" ? <StoryImpactReport report={storyImpactReport} /> : null}
            {leadStory ? (
              <LeadStory
                story={leadStory}
                socialCount={socialCountFor(leadStory)}
                onOpen={openStory}
                choiceOptions={leadChoiceOptions}
                onResolve={handleResolve}
                busyChoice={busyChoice}
              />
            ) : null}

            {pendingDecisions.length > 1 ? (
              <section className="sl-desk">
                <div className="sl-desk__head">
                  <h3>On your desk</h3>
                  <span className="sl-desk__count">{pendingDecisions.length}</span>
                </div>
                <div className="sl-desk__list">
                  {pendingDecisions.slice(0, 6).map((d) => (
                    <button key={d.id} type="button" className="sl-desk__item" onClick={() => openStory(d.id)}>
                      <StoryFace story={d} size={34} />
                      <div style={{ minWidth: 0 }}>
                        <strong>{d.headline}</strong>
                        <span>{d.teamName || "League"} · {d.ageLabel}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </section>
            ) : null}

            <div className="sl-toolbar">
              <div className="sl-chips">
                {deskFilters.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`sl-chip ${filter === f.id ? "is-active" : ""}`}
                    onClick={() => setFilter(f.id)}
                  >
                    {f.label}
                    <b>{filterCounts[f.id] ?? 0}</b>
                  </button>
                ))}
              </div>
              <div className="sl-tools">
                <input
                  type="search"
                  className="sl-input"
                  placeholder="Search player, team, headline…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select value={sortId} onChange={(e) => setSortId(e.target.value)} aria-label="Sort stories">
                  {SORT_OPTIONS.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="sl-gridhead">
              <h3>The wire</h3>
              <span>{filtered.length} filed</span>
            </div>

            {filtered.length === 0 ? (
              <EmptyPanel
                kicker="Desk · empty"
                title="Nothing matches"
                body={filterEmptyMsg || "No stories match this filter or search."}
              />
            ) : (
              <div className="sl-grid">
                {gridStories.map((s, i) => (
                  <StoryCard
                    key={s.id}
                    story={s}
                    index={i}
                    socialCount={socialCountFor(s)}
                    onOpen={openStory}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {process.env.NODE_ENV === "development" && !hasBackend ? (
          <details className="sl-debug">
            <summary>Storyline debug (dev)</summary>
            <pre>
              {JSON.stringify(
                {
                  has_storyline_events: Array.isArray(franchiseState?.storyline_events),
                  state_keys: Object.keys(franchiseState || {}),
                },
                null,
                2
              )}
            </pre>
          </details>
        ) : null}
        {process.env.NODE_ENV === "development" && franchiseState?.storyline_debug ? (
          <details className="sl-debug">
            <summary>Storyline debug (dev)</summary>
            <pre>{JSON.stringify(franchiseState.storyline_debug, null, 2)}</pre>
          </details>
        ) : null}
      </div>
    </div>
  );
}
