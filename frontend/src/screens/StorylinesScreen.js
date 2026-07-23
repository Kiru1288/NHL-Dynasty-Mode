import React, { useMemo, useState, useCallback, useEffect, useRef } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";

/*
  StorylinesScreen — backend-driven news hub.
  Rules: read franchiseState only; no fake storylines or invented metrics.
*/

const SORT_OPTIONS = [
  { id: "decisions", label: "Decisions First" },
  { id: "heat", label: "Heat" },
  { id: "priority", label: "Priority" },
  { id: "latest", label: "Latest" },
];

const LEGAL_GROUP_THRESHOLD = 5;

const CATEGORY_META = {
  performance: { icon: "◆", label: "Performance", accent: "#13d8e7" },
  star_underperforming: { icon: "◆", label: "Performance", accent: "#13d8e7" },
  rookie_breakout: { icon: "◆", label: "Performance", accent: "#13d8e7" },
  hot_streak: { icon: "◆", label: "Performance", accent: "#13d8e7" },
  injury: { icon: "+", label: "Injury", accent: "#ff606d" },
  legal_trouble: { icon: "§", label: "League News", accent: "#8ab4ff" },
  trade: { icon: "⇄", label: "Trade Rumor", accent: "#c992ff" },
  rumor: { icon: "⇄", label: "Trade Rumor", accent: "#c992ff" },
  draft: { icon: "★", label: "Draft", accent: "#e9a83c" },
  goalie: { icon: "◎", label: "Goalie", accent: "#8ab4ff" },
  contract: { icon: "$", label: "Contract", accent: "#52df94" },
  team_crisis: { icon: "!", label: "Team Crisis", accent: "#ff8a4c" },
  rivalry: { icon: "⚔", label: "Rivalry", accent: "#ff606d" },
  decision: { icon: "?", label: "GM Decision", accent: "#e9a83c" },
  league: { icon: "◉", label: "League News", accent: "#8ab4ff" },
  storyline: { icon: "◉", label: "League News", accent: "#8096a8" },
};

const FILTER_EMPTY = {
  draft: "No draft movement yet. Scouts are still watching.",
  injuries: "Medical desk is quiet.",
  rumors: "Trade wire is calm — for now.",
  decisions: "Front office is clear. No GM response needed.",
};

const FILTERS = [
  { id: "all", label: "All" },
  { id: "your_team", label: "Your Team" },
  { id: "league", label: "League" },
  { id: "rumors", label: "Rumors" },
  { id: "injuries", label: "Injuries" },
  { id: "draft", label: "Draft" },
  { id: "decisions", label: "Decisions" },
];

const PRIORITY_RANK = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };

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
    state?.user_team_name ||
      state?.team_name ||
      state?.franchise_team_name ||
      "Your Team"
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

function resolveCategoryKey(story) {
  const cat = str(story?.category || story?.type || "").toLowerCase();
  if (story?.requiresAction) return "decision";
  if (cat.includes("legal") || cat === "legal_trouble") return "legal_trouble";
  if (cat === "injury" || /injur/i.test(cat)) return "injury";
  if (/draft|prospect/.test(cat)) return "draft";
  if (/goalie|goaltender/.test(cat)) return "goalie";
  if (/trade|rumor|contract/.test(cat)) return /contract/.test(cat) ? "contract" : "trade";
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
  if (days === 1) return "1d ago";
  if (days < 7) return `${days}d ago`;
  if (days < 14) return "Week-old";
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

function groupFeedItems(stories, expandedGroups) {
  const legal = stories.filter((s) => resolveCategoryKey(s) === "legal_trouble");
  const rest = stories.filter((s) => resolveCategoryKey(s) !== "legal_trouble");
  if (legal.length < LEGAL_GROUP_THRESHOLD) {
    return stories.map((s) => ({ kind: "story", story: s }));
  }
  const groupKey = "legal_conduct";
  const expanded = expandedGroups.has(groupKey);
  const items = [];
  if (!expanded) {
    items.push({
      kind: "group",
      groupKey,
      label: "Legal / Conduct Updates",
      count: legal.length,
      preview: legal[0],
      stories: legal,
    });
    rest.forEach((s) => items.push({ kind: "story", story: s }));
    return items;
  }
  legal.forEach((s) => items.push({ kind: "story", story: s }));
  rest.forEach((s) => items.push({ kind: "story", story: s }));
  return items;
}

function matchesSearch(story, q) {
  if (!q) return true;
  const needle = q.toLowerCase();
  const hay = [
    story.headline,
    story.summary,
    story.playerName,
    story.teamName,
    story.category,
    story.type,
  ]
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
  const id = str(
    raw?.id || raw?.storyline_id || `story-${idx}-${headline.slice(0, 24)}`
  );

  return {
    id,
    storylineId: str(raw?.storyline_id || raw?.id || id),
    raw,
    headline: headline || "Untitled storyline",
    summary: str(
      raw?.short_summary ||
        raw?.summary ||
        raw?.description ||
        raw?.details ||
        raw?.text ||
        ""
    ),
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
    followUp: str(raw?.follow_up || ""),
    arcStatus: str(raw?.arc_status || raw?.status || "active"),
    playerPosition: str(raw?.player_position || ""),
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
      const missing = !state?.storyline_events;
      if (missing) console.warn("[Storylines] storyline_events missing from franchise state");
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

function evidenceLine(evidence) {
  const keys = Object.keys(evidence || {});
  if (!keys.length) return "";
  const parts = [];
  const labels = {
    games_played: "GP",
    goals: "G",
    assists: "A",
    points: "P",
    expected_points: "Exp P",
    overall: "OVR",
    cap_hit: "Cap",
    team_record: "Rec",
    save_pct: "SV%",
    gaa: "GAA",
    points_per_game: "P/GP",
  };
  keys.slice(0, 8).forEach((k) => {
    const label = labels[k] || k.replace(/_/g, " ");
    parts.push(`${label}: ${evidence[k]}`);
  });
  return parts.join(" · ");
}

function matchesFilter(story, filter, choicesMap) {
  if (filter === "all") return true;
  if (filter === "your_team") return story.isUserTeam;
  if (filter === "league") return !story.isUserTeam && story.category !== "draft";
  if (filter === "rumors") {
    const t = `${story.type} ${story.headline} ${story.category}`;
    return /rumor|trade|contract|market/i.test(t);
  }
  if (filter === "injuries") {
    return story.category === "injury" || /injur/i.test(`${story.type} ${story.headline}`);
  }
  if (filter === "draft") return story.category === "draft" || /prospect|draft/i.test(story.type);
  if (filter === "decisions") {
    return story.requiresAction || choicesMap.has(story.storylineId) || choicesMap.has(story.id);
  }
  return true;
}

function priorityClass(priority) {
  const p = String(priority || "").toUpperCase();
  if (p === "CRITICAL") return "critical";
  if (p === "HIGH") return "high";
  return "";
}

function LeagueWireTicker({ headlines }) {
  if (!headlines.length) {
    return (
      <div className="sl-wire sl-wire--empty">
        <span className="sl-wire-label">League Wire</span>
        <span>No league wire updates yet.</span>
      </div>
    );
  }
  const text = headlines.map((h) => h.headline).join("  ◆  ");
  return (
    <div className="sl-wire" title="League Wire — hover to pause">
      <span className="sl-wire-label">League Wire</span>
      <div className="sl-wire-track">
        <span className="sl-wire-text">{text}</span>
        <span className="sl-wire-text" aria-hidden>
          {text}
        </span>
      </div>
    </div>
  );
}

function StoryGroupCard({ group, onExpand, onSelectStory }) {
  return (
    <div className="sl-group-card">
      <button type="button" className="sl-group-head" onClick={() => onExpand(group.groupKey)}>
        <span className="sl-cat-icon" style={{ color: "#8ab4ff" }}>
          §
        </span>
        <div>
          <strong>{group.label}</strong>
          <p>{group.count} stories — click to expand</p>
        </div>
        <em>{group.count}</em>
      </button>
      {group.preview ? (
        <button type="button" className="sl-group-preview" onClick={() => onSelectStory(group.preview.id)}>
          Latest: {group.preview.headline}
        </button>
      ) : null}
    </div>
  );
}

function StoryCard({ story, selected, onSelect, todayIso }) {
  const meta = categoryMeta(story);
  const evLine = evidenceLine(story.evidence);
  const fresh = story.freshness || storyFreshnessClass(story, todayIso);
  const age = story.ageLabel || storyAgeLabel(story, todayIso);

  return (
    <button
      type="button"
      className={`sl-card ${fresh} ${selected ? "is-selected" : ""}`}
      onClick={() => onSelect(story.id)}
      style={{ "--cat-accent": meta.accent }}
    >
      <div className="sl-card-rail" aria-hidden />
      <div className="sl-card-inner">
        <div className="sl-card-top">
          <span className="sl-cat-chip">
            <span className="sl-cat-icon">{meta.icon}</span>
            {meta.label}
          </span>
          <span className="sl-age">{age}</span>
          <em className={priorityClass(story.priority)}>{story.priority}</em>
          {selected ? <span className="sl-selected-chip">Selected</span> : null}
        </div>
        <h3 className="sl-card-headline">{story.headline}</h3>
        {story.summary ? <p className="sl-card-summary">{story.summary.slice(0, 140)}{story.summary.length > 140 ? "…" : ""}</p> : null}
        <div className="sl-card-chips">
          {evLine ? <span className="sl-chip sl-chip--ev">{evLine}</span> : null}
          {story.effectSummary ? <span className="sl-chip">{story.effectSummary}</span> : null}
          {story.gamesRemaining > 0 ? (
            <span className="sl-chip sl-chip--return">
              OUT {story.gamesRemaining}G · {story.returnEstimate || "TBD"}
            </span>
          ) : null}
          {story.overallDelta != null && Number(story.overallDelta) !== 0 ? (
            <span className="sl-chip sl-chip--ovr">
              OVR {story.overallBefore}→{story.overallAfter} ({Number(story.overallDelta) > 0 ? "+" : ""}
              {story.overallDelta})
            </span>
          ) : null}
        </div>
        <div className="sl-card-foot">
          <span>{story.playerName || story.teamName || "—"}</span>
          <span className="sl-card-badges">
            {story.requiresAction ? <span className="sl-decision-badge">Decision</span> : null}
            {story.heat > 0 ? <span className="sl-heat-badge">Heat {story.heat}</span> : null}
            <span className="sl-open-hint">Open →</span>
          </span>
        </div>
      </div>
    </button>
  );
}

function PlayerIdentityCard({ story }) {
  if (!story.playerName && !story.teamName) return null;
  return (
    <div className="sl-identity-card">
      {story.playerName ? (
        <div className="sl-avatar">{playerInitials(story.playerName)}</div>
      ) : (
        <div className="sl-avatar sl-avatar--team">{str(story.teamName).slice(0, 3).toUpperCase()}</div>
      )}
      <div>
        <strong>{story.playerName || story.teamName}</strong>
        <p>
          {[story.playerPosition, story.teamName, story.playerOverall != null ? `${story.playerOverall} OVR` : ""]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
    </div>
  );
}

function EffectPills({ effects }) {
  const entries = Object.entries(effects || {});
  if (!entries.length) return null;
  return (
    <div className="sl-effect-pills">
      {entries.map(([k, v]) => (
        <span key={k} className={`sl-effect-pill ${effectPillClass(v)}`}>
          {formatEffectLabel(k)} {Number(v) > 0 ? "+" : ""}
          {v}
        </span>
      ))}
    </div>
  );
}

function DetailPanel({ story, choiceRow, onResolve, busyId, relatedStories }) {
  if (!story) {
    return (
      <div className="sl-detail sl-detail--empty">
        <div className="sl-empty-hero">
          <p className="sl-empty-kicker">News Desk</p>
          <h3>Select a story</h3>
          <p>Pick a headline from the feed to read evidence, franchise effects, and GM options.</p>
        </div>
      </div>
    );
  }

  const meta = categoryMeta(story);
  const options = asArray(choiceRow?.action_options).length
    ? choiceRow.action_options
    : story.actionOptions;

  return (
    <div className="sl-detail sl-detail--active" key={story.id}>
      <div className="sl-feature-head" style={{ "--cat-accent": meta.accent }}>
        <div className="sl-feature-badges">
          <span className="sl-cat-chip">
            <span className="sl-cat-icon">{meta.icon}</span>
            {meta.label}
          </span>
          <span className={`sl-detail-badge ${priorityClass(story.priority)}`}>{story.priority}</span>
          {story.heat > 0 ? <span className="sl-heat">Heat {story.heat}</span> : null}
          <span className="sl-arc-badge">{arcStage(story)}</span>
        </div>
        <h2 className="sl-detail-headline">{story.headline}</h2>
        <p className="sl-feature-sub">
          {story.sourceLabel ? `${story.sourceLabel} · ` : ""}
          {story.ageLabel || "—"} · {story.teamName || story.playerName || "League"}
          {story.gamesRemaining > 0 ? ` · Return ${story.returnEstimate || `in ${story.gamesRemaining}G`}` : ""}
        </p>
      </div>

      {story.cause || story.userVisibleExplanation ? (
        <section className="sl-section sl-section--highlight">
          <h4>Trigger / Cause</h4>
          <p>{story.userVisibleExplanation || story.cause}</p>
          {story.causeType ? (
            <p className="sl-muted-box" style={{ marginTop: 8 }}>
              Cause type: {story.causeType.replace(/_/g, " ")}
            </p>
          ) : null}
        </section>
      ) : null}

      {(story.culpritPlayerName || story.culpritPlayerId) ? (
        <section className="sl-section">
          <h4>Culprit</h4>
          <p>{story.culpritPlayerName || story.culpritPlayerId}</p>
        </section>
      ) : null}

      {story.summary ? <p className="sl-detail-lead">{story.summary}</p> : null}
      {story.description && story.description !== story.summary ? (
        <p className="sl-detail-desc">{story.description}</p>
      ) : null}

      <PlayerIdentityCard story={story} />

      {(story.overallBefore != null || story.gamesRemaining > 0) && (
        <section className="sl-section sl-section--highlight">
          <h4>Availability Impact</h4>
          <div className="sl-impact-row">
            {story.gamesRemaining > 0 ? (
              <span className="sl-impact-pill neg">
                OUT {story.gamesRemaining} games · {story.returnEstimate || "TBD"}
                {story.returnDate ? ` (${story.returnDate})` : ""}
              </span>
            ) : null}
            {story.overallDelta != null && Number(story.overallDelta) !== 0 ? (
              <span className="sl-impact-pill neg">
                {story.baseOverall != null ? (
                  <>
                    Effective OVR {story.baseOverall} → {story.effectiveOverall ?? story.overallAfter} (
                    {Number(story.overallDelta) > 0 ? "+" : ""}
                    {story.overallDelta})
                  </>
                ) : (
                  <>
                    OVR {story.overallBefore} → {story.overallAfter} ({Number(story.overallDelta) > 0 ? "+" : ""}
                    {story.overallDelta})
                  </>
                )}
              </span>
            ) : null}
            {story.impactReason ? (
              <span className="sl-impact-pill neg">{story.impactReason}</span>
            ) : null}
          </div>
        </section>
      )}

      {story.recoveryConditions?.length ? (
        <section className="sl-section">
          <h4>Recovery</h4>
          <ul className="sl-related-list">
            {story.recoveryConditions.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="sl-section">
        <h4>Data Evidence</h4>
        {Object.keys(story.evidence || {}).length ? (
          <dl className="sl-evidence-grid">
            {Object.entries(story.evidence).map(([k, v]) => (
              <div key={k} className="sl-evidence-row">
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="sl-muted-box">Backend evidence not provided for this story.</p>
        )}
      </section>

      <section className="sl-section">
        <h4>Franchise Effects</h4>
        {story.effectSummary ? <p>{story.effectSummary}</p> : null}
        <EffectPills effects={story.effects} />
        {!story.effectSummary && !Object.keys(story.effects || {}).length ? (
          <p className="sl-muted-box">No sim effects reported for this story.</p>
        ) : null}
      </section>

      {options.length ? (
        <section className="sl-section">
          <h4>GM Response Options</h4>
          <div className="sl-choice-grid">
            {options.map((opt) => {
              const busy = busyId === `${story.storylineId}:${opt.id}`;
              return (
                <button
                  key={opt.id}
                  type="button"
                  className="nhlcal-storyline-choice-button"
                  disabled={Boolean(busyId)}
                  onClick={() => onResolve?.(choiceRow?.storyline_id || story.storylineId, opt.id)}
                >
                  <strong>{opt.label}</strong>
                  {opt.effect_summary ? <span>{opt.effect_summary}</span> : null}
                  {busy ? <em>Applying…</em> : null}
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      <section className="sl-section">
        <h4>What Happens Next</h4>
        <p>{deriveFollowUp(story)}</p>
      </section>

      {relatedStories?.length ? (
        <section className="sl-section">
          <h4>Related Stories</h4>
          <ul className="sl-related-list">
            {relatedStories.slice(0, 4).map((r) => (
              <li key={r.id}>{r.headline}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {story.repeatCount > 0 ? (
        <section className="sl-section sl-section--muted">
          <h4>Story Arc</h4>
          <p>
            Beat #{story.repeatCount + 1}
            {story.escalatedFrom ? ` · escalated from ${story.escalatedFrom}` : ""} — stage: {arcStage(story)}
          </p>
        </section>
      ) : null}
    </div>
  );
}

export default function StorylinesScreen() {
  const { franchiseState, onResolveStorylineChoice, setScreen } = useGameUI();
  const [filter, setFilter] = useState("all");
  const [sortId, setSortId] = useState("decisions");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [busyChoice, setBusyChoice] = useState("");
  const [expandedGroups, setExpandedGroups] = useState(() => new Set());
  const [heatPulse, setHeatPulse] = useState(false);
  const prevHeatRef = useRef(0);
  const todayIso = calendarLabel(franchiseState);

  const stories = useMemo(() => collectStories(franchiseState), [franchiseState]);
  const choicesMap = useMemo(() => buildChoicesMap(franchiseState), [franchiseState]);

  const filterCounts = useMemo(() => {
    const counts = { all: stories.length };
    FILTERS.forEach((f) => {
      if (f.id === "all") return;
      counts[f.id] = stories.filter((s) => matchesFilter(s, f.id, choicesMap)).length;
    });
    return counts;
  }, [stories, choicesMap]);

  const filtered = useMemo(() => {
    const base = stories.filter((s) => matchesFilter(s, filter, choicesMap) && matchesSearch(s, search));
    return sortStories(base, sortId);
  }, [stories, filter, choicesMap, search, sortId]);

  const feedItems = useMemo(
    () => groupFeedItems(filtered, expandedGroups),
    [filtered, expandedGroups]
  );

  const pendingDecisions = useMemo(() => {
    return stories.filter(
      (s) => s.requiresAction || choicesMap.has(s.storylineId) || choicesMap.has(s.id)
    );
  }, [stories, choicesMap]);

  const topPending = pendingDecisions[0] || null;
  const topStory = stories[0] || null;
  const highestHeat = stories.reduce((m, s) => Math.max(m, s.heat || 0), 0);
  const yourTeamCount = stories.filter((s) => s.isUserTeam).length;
  const topHeatLabel = topStory ? resolveCategoryKey(topStory).replace(/_/g, " ") : "—";

  useEffect(() => {
    if (highestHeat !== prevHeatRef.current && prevHeatRef.current > 0) {
      setHeatPulse(true);
      const t = window.setTimeout(() => setHeatPulse(false), 900);
      return () => window.clearTimeout(t);
    }
    prevHeatRef.current = highestHeat;
    return undefined;
  }, [highestHeat]);

  const wireHeadlines = useMemo(() => stories.slice(0, 10), [stories]);

  const narrativeLine = useMemo(() => {
    return `${stories.length} active stories · ${pendingDecisions.length} GM decisions · top heat: ${topHeatLabel} · ${yourTeamCount} your-team stories`;
  }, [stories.length, pendingDecisions.length, topHeatLabel, yourTeamCount]);

  const selected =
    filtered.find((s) => s.id === selectedId) ||
    stories.find((s) => s.id === selectedId) ||
    filtered[0] ||
    null;

  const relatedStories = useMemo(() => {
    if (!selected) return [];
    const key = selected.categoryKey || resolveCategoryKey(selected);
    return stories.filter(
      (s) =>
        s.id !== selected.id &&
        (s.categoryKey === key ||
          s.teamId === selected.teamId ||
          s.playerId === selected.playerId)
    );
  }, [selected, stories]);

  const selectedChoice = selected
    ? choicesMap.get(selected.storylineId) || choicesMap.get(selected.id)
    : null;

  const handleResolve = useCallback(
    async (storylineId, choiceId) => {
      if (!onResolveStorylineChoice) return;
      setBusyChoice(`${storylineId}:${choiceId}`);
      try {
        await onResolveStorylineChoice(storylineId, choiceId);
      } finally {
        setBusyChoice("");
      }
    },
    [onResolveStorylineChoice]
  );

  const toggleGroup = useCallback((key) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const hasBackend = Array.isArray(franchiseState?.storyline_events);
  const filterEmptyMsg = FILTER_EMPTY[filter];

  return (
    <div className="nhlcal-sl-root">
      <style>{`
        .nhlcal-sl-root {
          --bg: #04101a;
          --bg-2: #061522;
          --panel: rgba(9, 25, 38, 0.94);
          --line: rgba(156, 218, 236, 0.14);
          --line-strong: rgba(73, 231, 240, 0.5);
          --text: #e9f7fb;
          --muted: #8096a8;
          --cyan: #13d8e7;
          --cyan-soft: rgba(19, 216, 231, 0.13);
          --gold: #e9a83c;
          --gold-soft: rgba(233, 168, 60, 0.14);
          --green: #52df94;
          --green-soft: rgba(82, 223, 148, 0.13);
          --red: #ff606d;
          --red-soft: rgba(255, 96, 109, 0.13);
          --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);

          min-height: 100%;
          width: 100%;
          background:
            radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
            radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
            linear-gradient(180deg, #06131f 0%, #020a11 100%);
          color: var(--text);
          display: flex;
          flex-direction: column;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .nhlcal-sl-root *,
        .nhlcal-sl-root *::before,
        .nhlcal-sl-root *::after {
          box-sizing: border-box;
        }

        .nhlcal-sl-root button {
          font-family: inherit;
        }

        .nhlcal-sl-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-height: 0;
          padding: 10px 14px 20px;
        }

        .nhlcal-sl-topbar {
          flex: 0 0 auto;
          display: grid;
          grid-template-columns: minmax(180px, 1fr) auto auto;
          align-items: center;
          gap: 12px;
          padding: 4px 0 10px;
        }

        .nhlcal-sl-kicker {
          margin: 0 0 4px;
          color: var(--cyan);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.13em;
          text-transform: uppercase;
        }

        .nhlcal-sl-topbar h1 {
          margin: 0;
          font-size: clamp(20px, 2vw, 28px);
          letter-spacing: 0.08em;
          text-transform: uppercase;
          text-shadow: 0 0 24px rgba(19, 216, 231, 0.12);
        }

        .nhlcal-sl-sub {
          margin: 4px 0 0;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
        }

        .nhlcal-sl-stat-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          border: 1px solid var(--line);
          background: rgba(8, 23, 35, 0.86);
          border-radius: 8px;
          overflow: hidden;
          box-shadow: var(--shadow);
        }

        .nhlcal-sl-stat-pill {
          min-height: 48px;
          padding: 8px 12px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 2px;
          border-right: 1px solid rgba(156, 218, 236, 0.08);
          background:
            linear-gradient(180deg, rgba(18, 42, 61, 0.45), rgba(6, 20, 31, 0.34)),
            radial-gradient(circle at 100% 0%, rgba(19, 216, 231, 0.05), transparent 52%);
        }

        .nhlcal-sl-stat-pill:last-child {
          border-right: 0;
        }

        .nhlcal-sl-stat-pill span {
          color: var(--muted);
          font-size: 9px;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .nhlcal-sl-stat-pill strong {
          color: var(--gold);
          font-size: 18px;
          font-weight: 1000;
          line-height: 1;
        }

        .nhlcal-sl-nav {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }

        .nhlcal-sl-nav button {
          height: 34px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(14, 35, 50, 0.9);
          color: rgba(233, 247, 251, 0.82);
          padding: 0 14px;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          cursor: pointer;
          transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }

        .nhlcal-sl-nav button:hover {
          border-color: var(--line-strong);
          color: var(--text);
          background: rgba(19, 216, 231, 0.11);
          transform: translateY(-1px);
        }

        .nhlcal-sl-alert {
          margin-top: 10px;
          border: 1px solid rgba(255, 96, 109, 0.42);
          background:
            radial-gradient(circle at 0% 0%, rgba(255, 96, 109, 0.16), transparent 38%),
            rgba(8, 23, 35, 0.94);
          border-radius: 12px;
          padding: 14px 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
        }

        .nhlcal-sl-alert.is-clear {
          border-color: rgba(82, 223, 148, 0.35);
          background:
            radial-gradient(circle at 0% 0%, rgba(82, 223, 148, 0.12), transparent 38%),
            rgba(8, 23, 35, 0.94);
        }

        .nhlcal-sl-alert strong {
          display: block;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          font-size: 12px;
          margin-bottom: 4px;
        }

        .nhlcal-sl-alert p {
          margin: 0;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
          line-height: 1.4;
        }

        .nhlcal-sl-alert-btn {
          height: 34px;
          border: 1px solid rgba(255, 255, 255, 0.16);
          background: rgba(255, 255, 255, 0.08);
          color: var(--text);
          border-radius: 10px;
          padding: 0 14px;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          cursor: pointer;
          white-space: nowrap;
        }

        .nhlcal-sl-alert-btn:hover {
          border-color: var(--line-strong);
          background: rgba(19, 216, 231, 0.11);
        }

        .sl-risk-label {
          display: inline-block;
          margin-top: 6px;
          padding: 2px 8px;
          border-radius: 999px;
          background: var(--red-soft);
          color: var(--red);
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .nhlcal-sl-filters {
          margin-top: 12px;
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .nhlcal-sl-filters button {
          height: 30px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(14, 35, 50, 0.9);
          color: rgba(233, 247, 251, 0.72);
          padding: 0 12px;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          cursor: pointer;
        }

        .nhlcal-sl-filters button.is-active,
        .nhlcal-sl-filters button:hover {
          border-color: var(--line-strong);
          color: var(--text);
          background: rgba(19, 216, 231, 0.11);
        }

        .nhlcal-sl-grid {
          margin-top: 12px;
          display: grid;
          grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.1fr);
          gap: 12px;
          flex: 1;
          min-height: 0;
        }

        .nhlcal-sl-panel {
          border: 1px solid var(--line);
          border-radius: 12px;
          background:
            linear-gradient(180deg, rgba(10, 30, 45, 0.94), rgba(5, 18, 29, 0.94)),
            radial-gradient(circle at 90% 0%, rgba(19, 216, 231, 0.07), transparent 38%);
          box-shadow: var(--shadow);
          display: flex;
          flex-direction: column;
          min-height: 420px;
          max-height: calc(100vh - 280px);
          overflow: hidden;
        }

        .nhlcal-sl-panel-head {
          flex: 0 0 auto;
          padding: 14px 16px 8px;
          border-bottom: 1px solid rgba(156, 218, 236, 0.08);
        }

        .nhlcal-sl-panel-head p {
          margin: 0;
          color: var(--cyan);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.13em;
          text-transform: uppercase;
        }

        .nhlcal-sl-panel-head h2 {
          margin: 4px 0 0;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .nhlcal-sl-scroll {
          flex: 1;
          overflow: auto;
          padding: 10px;
          scrollbar-width: thin;
          scrollbar-color: rgba(110, 173, 191, 0.35) rgba(4, 16, 26, 0.72);
        }

        .nhlcal-sl-scroll::-webkit-scrollbar {
          width: 8px;
        }

        .nhlcal-sl-scroll::-webkit-scrollbar-track {
          background: rgba(4, 16, 26, 0.72);
          border-radius: 999px;
        }

        .nhlcal-sl-scroll::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, rgba(19, 216, 231, 0.34), rgba(110, 173, 191, 0.28));
          border-radius: 999px;
          border: 2px solid rgba(4, 16, 26, 0.72);
        }

        .nhlcal-sl-feed-list {
          display: grid;
          gap: 9px;
        }

        .sl-card {
          width: 100%;
          text-align: left;
          padding: 0;
          cursor: pointer;
          color: inherit;
          display: flex;
          gap: 0;
          overflow: hidden;
          border: 1px solid rgba(156, 218, 236, 0.09);
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.025);
          transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        }

        .sl-card:hover {
          border-color: rgba(19, 216, 231, 0.35);
          transform: translateY(-1px);
        }

        .sl-card.is-selected {
          border-color: rgba(19, 216, 231, 0.55);
          box-shadow:
            inset 0 0 0 1px rgba(19, 216, 231, 0.35),
            0 0 20px rgba(19, 216, 231, 0.15);
        }

        .sl-card-tone {
          width: 4px;
          flex-shrink: 0;
        }

        .tone-pos { background: var(--green); }
        .tone-neg { background: var(--red); }
        .tone-mix { background: var(--gold); }
        .tone-neutral { background: var(--cyan); }

        .sl-card-inner {
          flex: 1;
          min-width: 0;
          padding: 9px 10px 9px 8px;
        }

        .nhlcal-storyline-topline {
          display: grid;
          grid-template-columns: 44px minmax(0, 1fr) auto;
          gap: 8px;
          align-items: center;
        }

        .nhlcal-storyline-topline span {
          color: var(--muted);
          font-size: 10px;
          font-weight: 900;
        }

        .nhlcal-storyline-topline strong {
          min-width: 0;
          color: var(--text);
          font-size: 11px;
          font-weight: 1000;
          line-height: 1.25;
        }

        .nhlcal-storyline-topline em {
          border-radius: 999px;
          padding: 3px 6px;
          background: rgba(255, 255, 255, 0.04);
          color: var(--muted);
          font-size: 8px;
          font-weight: 1000;
          font-style: normal;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .nhlcal-subtext {
          margin-top: 5px;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
          line-height: 1.35;
        }

        .nhlcal-storyline-topline em.critical {
          color: var(--red);
          background: var(--red-soft);
        }

        .nhlcal-storyline-topline em.high {
          color: var(--gold);
          background: var(--gold-soft);
        }

        .sl-evidence-chip {
          margin: 6px 0 0;
          color: var(--cyan);
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 10px;
          font-weight: 800;
          line-height: 1.35;
        }

        .sl-effect-chip {
          margin: 4px 0 0;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
        }

        .sl-card-foot {
          margin-top: 8px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 8px;
          font-size: 10px;
          font-weight: 800;
          color: var(--muted);
        }

        .sl-card-badges {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }

        .sl-decision-badge {
          padding: 2px 7px;
          border-radius: 999px;
          background: var(--red-soft);
          color: var(--red);
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-heat-badge {
          padding: 2px 7px;
          border-radius: 999px;
          background: var(--gold-soft);
          color: var(--gold);
          font-size: 8px;
          font-weight: 1000;
        }

        .sl-detail {
          padding: 16px 18px;
        }

        .sl-detail--empty {
          min-height: 200px;
          display: grid;
          place-items: center;
          color: var(--muted);
          font-size: 12px;
          font-weight: 800;
          text-align: center;
          padding: 24px;
        }

        .sl-detail-meta {
          display: flex;
          gap: 8px;
          align-items: center;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }

        .sl-detail-badge {
          padding: 3px 8px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.04);
          color: var(--muted);
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-detail-badge.critical {
          color: var(--red);
          background: var(--red-soft);
        }

        .sl-detail-badge.high {
          color: var(--gold);
          background: var(--gold-soft);
        }

        .sl-detail-headline {
          margin: 0 0 10px;
          font-size: clamp(18px, 2vw, 24px);
          line-height: 1.2;
          letter-spacing: 0.02em;
        }

        .sl-detail-lead {
          margin: 0 0 8px;
          color: rgba(233, 247, 251, 0.88);
          font-size: 12px;
          font-weight: 800;
          line-height: 1.5;
        }

        .sl-detail-desc {
          margin: 0 0 12px;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
          line-height: 1.5;
        }

        .sl-section {
          margin-top: 14px;
          padding-top: 12px;
          border-top: 1px solid rgba(156, 218, 236, 0.09);
        }

        .sl-section h4 {
          margin: 0 0 8px;
          color: var(--cyan);
          font-size: 11px;
          font-weight: 1000;
          letter-spacing: 0.13em;
          text-transform: uppercase;
        }

        .sl-section p {
          margin: 0;
          color: rgba(233, 247, 251, 0.86);
          font-size: 12px;
          font-weight: 800;
          line-height: 1.45;
        }

        .sl-section--muted p {
          color: var(--muted);
          font-style: italic;
        }

        .sl-evidence-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px 14px;
          margin: 0;
        }

        .sl-evidence-row dt {
          margin: 0;
          color: var(--muted);
          font-size: 9px;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .sl-evidence-row dd {
          margin: 3px 0 0;
          color: var(--text);
          font-size: 15px;
          font-weight: 1000;
        }

        .sl-effect-list {
          margin: 8px 0 0;
          padding-left: 18px;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
        }

        .sl-choice-grid {
          display: grid;
          gap: 8px;
          margin-top: 8px;
        }

        .nhlcal-storyline-choice-button {
          width: 100%;
          min-height: 44px;
          border: 1px solid var(--line);
          border-radius: 10px;
          background: rgba(19, 216, 231, 0.07);
          color: var(--text);
          padding: 10px 12px;
          text-align: left;
          cursor: pointer;
        }

        .nhlcal-storyline-choice-button:hover:not(:disabled) {
          border-color: var(--line-strong);
          background: rgba(19, 216, 231, 0.14);
        }

        .nhlcal-storyline-choice-button:disabled {
          cursor: not-allowed;
          opacity: 0.58;
        }

        .nhlcal-storyline-choice-button strong {
          display: block;
          color: var(--cyan);
          font-size: 11px;
          font-weight: 1000;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 4px;
        }

        .nhlcal-storyline-choice-button span {
          display: block;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
        }

        .nhlcal-sl-empty {
          margin: 48px auto;
          max-width: 520px;
          padding: 24px;
          text-align: center;
          border: 1px solid var(--line);
          border-radius: 12px;
          background: rgba(8, 23, 35, 0.86);
          box-shadow: var(--shadow);
        }

        .nhlcal-sl-empty h2 {
          margin: 0 0 8px;
          font-size: 18px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .nhlcal-sl-empty p {
          margin: 0;
          color: var(--muted);
          font-size: 12px;
          font-weight: 800;
          line-height: 1.5;
        }

        .sl-heat {
          color: var(--gold);
          font-size: 10px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-wire {
          margin-top: 8px;
          display: flex;
          align-items: center;
          gap: 10px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(8, 23, 35, 0.72);
          padding: 6px 10px;
          overflow: hidden;
        }

        .sl-wire-label {
          flex: 0 0 auto;
          color: var(--cyan);
          font-size: 9px;
          font-weight: 1000;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .sl-wire-track {
          display: flex;
          overflow: hidden;
          min-width: 0;
        }

        .sl-wire:hover .sl-wire-text {
          animation-play-state: paused;
        }

        .sl-wire-text {
          white-space: nowrap;
          padding-right: 48px;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
          animation: sl-wire-scroll 42s linear infinite;
        }

        @keyframes sl-wire-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }

        .sl-narrative {
          margin-top: 8px;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
        }

        .nhlcal-sl-alert.is-urgent {
          animation: sl-pulse-urgent 2.4s ease-in-out infinite;
        }

        @keyframes sl-pulse-urgent {
          0%, 100% { box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24); }
          50% { box-shadow: 0 0 28px rgba(255, 96, 109, 0.28); }
        }

        .nhlcal-sl-stat-pill strong.pulse-once {
          animation: sl-heat-pulse 0.9s ease;
        }

        @keyframes sl-heat-pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.12); color: var(--cyan); }
        }

        .sl-toolbar {
          margin-top: 10px;
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          align-items: center;
        }

        .sl-search {
          flex: 1;
          min-width: 180px;
          height: 32px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(8, 23, 35, 0.86);
          color: var(--text);
          padding: 0 12px;
          font-size: 11px;
          font-weight: 800;
        }

        .sl-sort select {
          height: 32px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(14, 35, 50, 0.9);
          color: var(--text);
          padding: 0 10px;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
        }

        .nhlcal-sl-filters button.has-urgent {
          border-color: rgba(255, 96, 109, 0.45);
        }

        .nhlcal-sl-filters .count {
          margin-left: 4px;
          opacity: 0.72;
        }

        .sl-card {
          --cat-accent: var(--cyan);
          animation: sl-card-in 0.35s ease both;
        }

        @keyframes sl-card-in {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .sl-card-rail {
          width: 4px;
          flex-shrink: 0;
          background: var(--cat-accent);
        }

        .sl-card.is-selected {
          border-color: rgba(19, 216, 231, 0.65);
          box-shadow: inset 0 0 0 1px rgba(19, 216, 231, 0.35), 0 0 24px rgba(19, 216, 231, 0.18);
          background: rgba(19, 216, 231, 0.06);
        }

        .sl-card.is-fresh { opacity: 1; }
        .sl-card.is-stale { opacity: 0.72; }

        .sl-card-top {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 6px;
          margin-bottom: 6px;
        }

        .sl-cat-chip {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 2px 7px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.04);
          color: var(--cat-accent, var(--cyan));
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-cat-icon { font-size: 10px; }

        .sl-age {
          color: var(--muted);
          font-size: 9px;
          font-weight: 800;
        }

        .sl-selected-chip {
          padding: 2px 6px;
          border-radius: 999px;
          background: var(--cyan-soft);
          color: var(--cyan);
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-card-headline {
          margin: 0 0 6px;
          font-size: 12px;
          font-weight: 1000;
          line-height: 1.3;
          text-align: left;
        }

        .sl-card-summary {
          margin: 0 0 6px;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
          line-height: 1.35;
          text-align: left;
        }

        .sl-card-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-bottom: 6px;
        }

        .sl-chip {
          padding: 2px 6px;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.04);
          color: var(--muted);
          font-size: 8px;
          font-weight: 800;
        }

        .sl-chip--ev { color: var(--cyan); }
        .sl-chip--return { color: var(--gold); }
        .sl-chip--ovr { color: var(--red); }

        .sl-open-hint {
          color: var(--cyan);
          font-size: 8px;
          font-weight: 1000;
          opacity: 0;
          transition: opacity 0.2s ease;
        }

        .sl-card:hover .sl-open-hint { opacity: 1; }

        .sl-group-card {
          border: 1px solid rgba(138, 180, 255, 0.2);
          border-radius: 10px;
          background: rgba(138, 180, 255, 0.06);
          overflow: hidden;
        }

        .sl-group-head {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 12px;
          border: 0;
          background: transparent;
          color: inherit;
          cursor: pointer;
          text-align: left;
        }

        .sl-group-head strong {
          display: block;
          font-size: 11px;
          text-transform: uppercase;
        }

        .sl-group-head p {
          margin: 2px 0 0;
          color: var(--muted);
          font-size: 9px;
          font-weight: 800;
        }

        .sl-group-head em {
          margin-left: auto;
          padding: 3px 8px;
          border-radius: 999px;
          background: rgba(138, 180, 255, 0.14);
          color: #8ab4ff;
          font-style: normal;
          font-size: 10px;
          font-weight: 1000;
        }

        .sl-group-preview {
          width: 100%;
          border: 0;
          border-top: 1px solid rgba(156, 218, 236, 0.08);
          background: rgba(0, 0, 0, 0.12);
          color: var(--muted);
          padding: 8px 12px;
          font-size: 9px;
          font-weight: 800;
          text-align: left;
          cursor: pointer;
        }

        .sl-feature-head {
          margin: -4px -4px 12px;
          padding: 14px;
          border-radius: 10px;
          border: 1px solid rgba(156, 218, 236, 0.12);
          background:
            linear-gradient(135deg, rgba(19, 216, 231, 0.08), transparent 55%),
            rgba(255, 255, 255, 0.02);
          border-left: 3px solid var(--cat-accent, var(--cyan));
        }

        .sl-feature-badges {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 8px;
        }

        .sl-feature-sub {
          margin: 6px 0 0;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
        }

        .sl-identity-card {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 12px;
          border: 1px solid var(--line);
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.03);
          margin-bottom: 12px;
        }

        .sl-avatar {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          display: grid;
          place-items: center;
          background: var(--cyan-soft);
          color: var(--cyan);
          font-size: 13px;
          font-weight: 1000;
        }

        .sl-avatar--team { background: var(--gold-soft); color: var(--gold); }

        .sl-identity-card strong {
          display: block;
          font-size: 13px;
        }

        .sl-identity-card p {
          margin: 2px 0 0;
          color: var(--muted);
          font-size: 10px;
          font-weight: 800;
        }

        .sl-muted-box {
          margin: 0;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px dashed rgba(156, 218, 236, 0.16);
          color: var(--muted);
          font-size: 11px;
          font-style: italic;
        }

        .sl-effect-pills {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-top: 8px;
        }

        .sl-effect-pill {
          padding: 4px 8px;
          border-radius: 999px;
          font-size: 9px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-effect-pill.pos { background: var(--green-soft); color: var(--green); }
        .sl-effect-pill.neg { background: var(--red-soft); color: var(--red); }
        .sl-effect-pill.neutral { background: rgba(255, 255, 255, 0.05); color: var(--muted); }

        .sl-impact-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .sl-impact-pill {
          padding: 6px 10px;
          border-radius: 8px;
          font-size: 10px;
          font-weight: 900;
        }
        .sl-impact-pill.neg { background: var(--red-soft); color: var(--red); }

        .sl-arc-badge {
          padding: 2px 7px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.05);
          color: var(--muted);
          font-size: 8px;
          font-weight: 1000;
          text-transform: uppercase;
        }

        .sl-related-list {
          margin: 0;
          padding-left: 16px;
          color: var(--muted);
          font-size: 11px;
          font-weight: 800;
        }

        .sl-empty-hero h3 {
          margin: 0 0 8px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .sl-empty-kicker {
          color: var(--cyan);
          font-size: 10px;
          font-weight: 1000;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .sl-detail--active {
          animation: sl-detail-in 0.28s ease;
        }

        @keyframes sl-detail-in {
          from { opacity: 0.4; }
          to { opacity: 1; }
        }

        @media (max-width: 900px) {
          .nhlcal-sl-topbar {
            grid-template-columns: 1fr;
          }

          .nhlcal-sl-grid {
            grid-template-columns: 1fr;
          }

          .nhlcal-sl-panel {
            max-height: none;
          }

          .nhlcal-sl-stat-strip {
            grid-template-columns: 1fr;
          }

          .nhlcal-sl-stat-pill {
            border-right: 0;
            border-bottom: 1px solid rgba(156, 218, 236, 0.08);
          }
        }
      `}</style>

      <div className="nhlcal-sl-main">
        <header className="nhlcal-sl-topbar">
          <div>
            <p className="nhlcal-sl-kicker">Franchise News Desk</p>
            <h1>Storylines</h1>
            <p className="nhlcal-sl-sub">
              {teamLabel(franchiseState)} · {calendarLabel(franchiseState)}
            </p>
          </div>

          <section className="nhlcal-sl-stat-strip" aria-label="Newsroom status">
            <div className="nhlcal-sl-stat-pill">
              <span>Active Stories</span>
              <strong>{stories.length}</strong>
            </div>
            <div className="nhlcal-sl-stat-pill">
              <span>GM Decisions</span>
              <strong>{pendingDecisions.length}</strong>
            </div>
            <div className="nhlcal-sl-stat-pill">
              <span>Top Heat</span>
              <strong className={heatPulse ? "pulse-once" : ""}>{highestHeat || topHeatLabel}</strong>
            </div>
            <div className="nhlcal-sl-stat-pill">
              <span>Your Team</span>
              <strong>{yourTeamCount}</strong>
            </div>
          </section>

          <nav className="nhlcal-sl-nav" aria-label="Navigation">
            <button type="button" onClick={() => setScreen?.(SCREENS.CALENDAR)}>
              Calendar
            </button>
            <button type="button" onClick={() => setScreen?.(SCREENS.HUB)}>
              Hub
            </button>
          </nav>
        </header>

        <LeagueWireTicker headlines={wireHeadlines} />
        <p className="sl-narrative">{narrativeLine}</p>

        {topPending ? (
          <div className="nhlcal-sl-alert is-urgent">
            <div>
              <strong>GM Response Required · {pendingDecisions.length} pending</strong>
              <p>{topPending.headline}</p>
              <span className="sl-risk-label">High priority · decision needed</span>
            </div>
            <button type="button" className="nhlcal-sl-alert-btn" onClick={() => setSelectedId(topPending.id)}>
              Review Decision
            </button>
          </div>
        ) : (
          <div className="nhlcal-sl-alert is-clear">
            <div>
              <strong>Front office clear — no GM decisions pending</strong>
              <p>{topStory ? `Lead story: ${topStory.headline}` : "Advance the season to generate stories from real stats."}</p>
            </div>
          </div>
        )}

        {!hasBackend ? (
          <div className="nhlcal-sl-empty">
            <h2>Storyline backend data missing</h2>
            <p>The franchise state did not include storyline_events. Check /api/franchise/state export.</p>
          </div>
        ) : stories.length === 0 ? (
          <div className="nhlcal-sl-empty">
            <h2>No storylines yet</h2>
            <p>
              Advance the season and the backend will generate stories from real player stats, standings,
              injuries, and league events.
            </p>
          </div>
        ) : (
          <>
            <div className="nhlcal-sl-filters">
              {FILTERS.map((f) => {
                const urgent = f.id === "decisions" && pendingDecisions.length > 0;
                return (
                  <button
                    key={f.id}
                    type="button"
                    className={`${filter === f.id ? "is-active" : ""} ${urgent ? "has-urgent" : ""}`}
                    onClick={() => setFilter(f.id)}
                  >
                    {f.label}
                    <span className="count">({filterCounts[f.id] ?? 0})</span>
                  </button>
                );
              })}
            </div>

            <div className="sl-toolbar">
              <input
                type="search"
                className="sl-search"
                placeholder="Search player, team, headline…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <label className="sl-sort">
                <select value={sortId} onChange={(e) => setSortId(e.target.value)}>
                  {SORT_OPTIONS.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="nhlcal-sl-grid">
              <div className="nhlcal-sl-panel">
                <div className="nhlcal-sl-panel-head">
                  <p>News Feed</p>
                  <h2>{filtered.length} stories</h2>
                </div>
                <div className="nhlcal-sl-scroll">
                  <div className="nhlcal-sl-feed-list">
                    {filtered.length === 0 ? (
                      <p className="sl-muted-box">{filterEmptyMsg || "No stories match this filter."}</p>
                    ) : (
                      feedItems.map((item, idx) =>
                        item.kind === "group" ? (
                          <StoryGroupCard
                            key={item.groupKey}
                            group={item}
                            onExpand={toggleGroup}
                            onSelectStory={setSelectedId}
                          />
                        ) : (
                          <StoryCard
                            key={item.story.id}
                            story={item.story}
                            selected={selected?.id === item.story.id}
                            onSelect={setSelectedId}
                            todayIso={todayIso}
                          />
                        )
                      )
                    )}
                  </div>
                </div>
              </div>

              <div className="nhlcal-sl-panel">
                <div className="nhlcal-sl-panel-head">
                  <p>Story Detail</p>
                  <h2>{selected?.headline ? "Top Story" : "Pick a story"}</h2>
                </div>
                <div className="nhlcal-sl-scroll">
                  <DetailPanel
                    story={selected}
                    choiceRow={selectedChoice}
                    onResolve={handleResolve}
                    busyId={busyChoice}
                    relatedStories={relatedStories}
                  />
                </div>
              </div>
            </div>
          </>
        )}
        {process.env.NODE_ENV === "development" && franchiseState?.storyline_debug ? (
          <details className="sl-section" style={{ margin: "1rem", padding: "0.75rem", opacity: 0.85 }}>
            <summary>Storyline debug (dev)</summary>
            <pre style={{ fontSize: 11, overflow: "auto", maxHeight: 240 }}>
              {JSON.stringify(franchiseState.storyline_debug, null, 2)}
            </pre>
          </details>
        ) : null}
      </div>
    </div>
  );
}
