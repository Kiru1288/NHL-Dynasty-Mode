import React, { useMemo, useState, useCallback, useEffect } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import PlayerHeadshot from "../components/PlayerHeadshot";

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
  { id: "all", label: "Latest" },
  { id: "your_team", label: "Your Team" },
  { id: "rumors", label: "Rumours" },
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

function heatLabel(heat) {
  const n = Number(heat);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 20) return "Quiet";
  if (n < 45) return "Building";
  if (n < 75) return "Hot";
  return "Boiling";
}

function credibilityLabel(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 30) return "Speculation";
  if (n < 50) return "Early chatter";
  if (n < 75) return "Credible";
  return "Strongly sourced";
}

function isRumourStory(story) {
  return /trade|rumor|contract|market/i.test(`${story.type} ${story.category} ${story.headline}`);
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

function LeagueWireTicker({ headlines }) {
  if (!headlines.length) return null;
  const wireText = headlines.map((h) => h.headline).join("  ◆  ");
  return (
    <div className="sl-breaking" title="Breaking wire — hover to pause">
      <span className="sl-breaking-label">BREAKING</span>
      <div className="sl-breaking-track">
        <span className="sl-breaking-text">{wireText}</span>
        <span className="sl-breaking-text" aria-hidden>
          {wireText}
        </span>
      </div>
    </div>
  );
}

function TeamOrPlayerIdentity({ story }) {
  if (!story?.playerName && !story?.teamName) return null;
  const abbr = str(story.teamName || "TEAM").slice(0, 4).toUpperCase();
  const logo =
    resolveFranchiseTeamLogo(
      { team_id: story.teamId, team_name: story.teamName, team_abbrev: abbr, abbrev: abbr },
      story.teamName
    ) || "";
  return (
    <div className="sl-identity-inline">
      <div className="sl-identity-logo">
        {story.playerName ? (
          <PlayerHeadshot
            player={{
              name: story.playerName,
              position: story.playerPosition,
              overall: story.playerOverall,
              team_abbrev: abbr,
              team_name: story.teamName,
              ...(asObject(story.raw) || {}),
            }}
            size="md"
          />
        ) : logo ? (
          <img src={logo} alt="" />
        ) : (
          <span>{playerInitials(story.playerName || abbr)}</span>
        )}
      </div>
      <div>
        <strong>{story.playerName || story.teamName}</strong>
        <p>
          {[
            story.playerPosition,
            story.teamName,
            story.playerOverall != null && Number(story.playerOverall) > 0 ? `${story.playerOverall} OVR` : "",
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
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
  const eligible =
    story.eligibleToPlay == null ? null : story.eligibleToPlay ? "Eligible to dress" : "Cannot dress";
  return (
    <section className="sl-article-section">
      <h4>Conduct Desk</h4>
      {story.allegationNote ? <p>{story.allegationNote}</p> : null}
      <div className="sl-conduct-grid">
        {story.informationStatus ? <span>Info: {formatEffectLabel(story.informationStatus)}</span> : null}
        {story.legalStatus ? <span>Legal: {formatEffectLabel(story.legalStatus)}</span> : null}
        {story.leagueStatus ? <span>League: {formatEffectLabel(story.leagueStatus)}</span> : null}
        {story.teamStatus ? <span>Team: {formatEffectLabel(story.teamStatus)}</span> : null}
        {eligible ? <span>{eligible}</span> : null}
      </div>
    </section>
  );
}

function TradeSwap({ story }) {
  const pair = parseTradeTeams(story);
  if (pair.length < 2) return null;
  const [a, b] = pair;
  const logoA = resolveFranchiseTeamLogo({ team_abbrev: a, abbrev: a }, a) || "";
  const logoB = resolveFranchiseTeamLogo({ team_abbrev: b, abbrev: b }, b) || "";
  return (
    <div className="sl-trade-swap" aria-label="Trade parties">
      <div className="sl-trade-side">
        {logoA ? <img src={logoA} alt="" /> : <strong>{a}</strong>}
        <span>{a}</span>
      </div>
      <em>⇄</em>
      <div className="sl-trade-side">
        {logoB ? <img src={logoB} alt="" /> : <strong>{b}</strong>}
        <span>{b}</span>
      </div>
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

  const pendingDecisions = useMemo(() => {
    return stories.filter(
      (s) => s.requiresAction || choicesMap.has(s.storylineId) || choicesMap.has(s.id)
    );
  }, [stories, choicesMap]);

  const topPending = pendingDecisions[0] || null;
  const yourTeamCount = stories.filter((s) => s.isUserTeam).length;
  const orgPressure = asObject(franchiseState?.conduct_org_pressure);
  const userOrg =
    orgPressure[userTeamId(franchiseState)] ||
    orgPressure[str(franchiseState?.user_team_id || "")] ||
    null;

  const wireHeadlines = useMemo(() => stories.slice(0, 10), [stories]);

  const selected =
    filtered.find((s) => s.id === selectedId) ||
    stories.find((s) => s.id === selectedId) ||
    filtered[0] ||
    null;

  useEffect(() => {
    if (!filtered.length) {
      if (selectedId != null) setSelectedId(null);
      return;
    }
    if (!filtered.some((s) => s.id === selectedId)) {
      setSelectedId(filtered[0].id);
    }
  }, [filtered, selectedId]);

  const relatedStories = useMemo(() => {
    if (!selected) return [];
    const scored = stories
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
      .sort((a, b) => b.score - a.score);
    return scored.map((r) => r.s);
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

  const hasBackend = Array.isArray(franchiseState?.storyline_events);
  const filterEmptyMsg = FILTER_EMPTY[filter];
  const leadStory = selected || filtered[0] || stories[0] || null;
  const statusLine = `${stories.length} active${yourTeamCount ? ` · ${yourTeamCount} involving your team` : ""}${
    pendingDecisions.length ? ` · ${pendingDecisions.length} decisions` : ""
  }`;

  const latestStories = filtered.filter((s) => s.id !== leadStory?.id).slice(0, 12);
  const yourTeamStories = filtered.filter((s) => s.isUserTeam && s.id !== leadStory?.id).slice(0, 6);
  const aroundLeagueStories = filtered.filter((s) => !s.isUserTeam && s.id !== leadStory?.id).slice(0, 6);
  const rumourStories = filtered
    .filter((s) => s.id !== leadStory?.id && /trade|rumor|contract|market/i.test(`${s.type} ${s.category} ${s.headline}`))
    .slice(0, 6);
  const hotStories = filtered
    .filter((s) => s.id !== leadStory?.id && Number(s.heat) > 0)
    .sort((a, b) => Number(b.heat || 0) - Number(a.heat || 0))
    .slice(0, 5);

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

          /* The newsroom occupies the full workspace so a quiet wire does not
             leave half the viewport as dead black space. */
          min-height: 100vh;
          width: 100%;
          background:
            radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
            radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
            linear-gradient(180deg, #06131f 0%, #020a11 100%);
          color: var(--text);
          display: flex;
          flex-direction: column;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          overflow-x: hidden;
        }

        .nhlcal-sl-main {
          width: min(1400px, 100%);
          margin: 0 auto;
          padding: 12px 16px 30px;
          /* The page column owns the full workspace height so the wire board
             extends to the bottom rule instead of floating in dead space. */
          flex: 1 1 auto;
          min-height: 0;
          display: flex;
          flex-direction: column;
        }

        .nhlcal-sl-topbar {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          flex-wrap: wrap;
          gap: 12px;
          border-bottom: 1px solid rgba(156, 218, 236, 0.14);
          padding-bottom: 10px;
        }

        .nhlcal-sl-kicker { margin: 0; color: var(--cyan); font-size: 11px; font-weight: 1000; letter-spacing: .12em; text-transform: uppercase; }
        .nhlcal-sl-topbar h1 { margin: 2px 0; font-size: 28px; letter-spacing: .06em; text-transform: uppercase; }
        .nhlcal-sl-sub { margin: 0; color: var(--muted); font-size: 11px; font-weight: 800; }
        .sl-status-line { margin-top: 8px; color: var(--muted); font-size: 11px; font-weight: 800; }

        .nhlcal-sl-nav { display: flex; gap: 8px; }
        .nhlcal-sl-nav button {
          height: 32px; border: 1px solid var(--line); border-radius: 8px; background: rgba(14, 35, 50, 0.9);
          color: rgba(233,247,251,.9); padding: 0 12px; font-size: 11px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase;
        }

        .sl-breaking { margin-top: 12px; display: flex; align-items: stretch; gap: 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: rgba(6, 21, 34, 0.72); overflow: hidden; min-height: 32px; }
        .sl-breaking-label { flex: 0 0 auto; display: flex; align-items: center; padding: 0 12px; color: var(--ops-injury, #ff606d); background: rgba(255, 96, 109, 0.1); font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase; border-right: 1px solid rgba(255, 96, 109, 0.22); }
        .sl-breaking-track { min-width: 0; flex: 1; display: flex; overflow: hidden; align-items: center; }
        .sl-breaking-text { white-space: nowrap; color: var(--ops-text, #e9f7fb); font-size: 0.8125rem; font-weight: 700; padding: 0 12px; animation: sl-wire-scroll 45s linear infinite; }
        .sl-breaking:hover .sl-breaking-text { animation-play-state: paused; }

        .sl-frontoffice-alert {
          margin-top: 10px; width: 100%; text-align: left;
          border: 1px solid rgba(255, 96, 109, 0.55);
          border-left: 4px solid var(--ops-injury, #ff606d);
          background: linear-gradient(90deg, rgba(255, 96, 109, 0.14), rgba(6, 21, 34, 0.88));
          color: var(--text); padding: 10px 12px; display: grid; gap: 4px;
          border-radius: var(--radius-control, 6px);
          box-shadow: inset 0 0 0 1px rgba(255, 96, 109, 0.08);
        }
        .sl-frontoffice-alert span { color: #ff8d97; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase; }
        .sl-frontoffice-alert strong { font-size: 0.95rem; line-height: 1.3; font-weight: 800; }
        .sl-frontoffice-alert em { font-size: 0.72rem; color: #ffc9cf; font-style: normal; font-weight: 700; }

        .sl-toolbar { margin-top: 12px; display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; }
        .nhlcal-sl-filters { display: flex; gap: 10px; overflow-x: auto; }
        .nhlcal-sl-filters button { border: 0; border-bottom: 2px solid transparent; background: transparent; color: var(--muted); font-size: 11px; font-weight: 900; text-transform: uppercase; padding: 6px 0; white-space: nowrap; }
        .nhlcal-sl-filters button.is-active { color: var(--text); border-bottom-color: var(--cyan); }
        .nhlcal-sl-filters button.has-urgent { color: #ff9ea7; }
        .nhlcal-sl-filters .count { opacity: .65; margin-left: 4px; }
        .sl-search { height: 32px; border: 1px solid var(--line); border-radius: 6px; background: rgba(8, 23, 35, 0.86); color: var(--text); padding: 0 10px; font-size: 12px; font-weight: 700; }
        .sl-sort select { height: 32px; border: 1px solid var(--line); border-radius: 6px; background: rgba(14,35,50,.9); color: var(--text); font-size: 11px; font-weight: 800; padding: 0 10px; }

        .sl-lead { margin-top: 12px; border-top: 2px solid var(--cyan); border-bottom: 1px solid var(--line); padding: 12px 0 14px; background: transparent; border-radius: 0; }
        .sl-lead-top { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; color: var(--muted); font-size: 0.6875rem; font-weight: 900; text-transform: uppercase; letter-spacing: 0.1em; }
        .sl-lead h2 { margin: 8px 0 10px; font-size: clamp(1.35rem, 2.8vw, 2rem); line-height: 1.08; font-weight: 800; letter-spacing: 0.02em; text-transform: none; font-family: var(--font-editorial, Inter, serif); }
        .sl-lead-summary { margin: 0; font-size: var(--type-body-size, 0.875rem); color: rgba(233,247,251,.88); line-height: 1.45; }
        .sl-lead-signals { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
        .sl-lead-signals span { font-size: 0.6875rem; font-weight: 900; color: var(--gold); background: var(--gold-soft); padding: 3px 8px; border-radius: var(--radius-ops, 2px); letter-spacing: 0.08em; text-transform: uppercase; }

        .sl-layout { margin-top: 14px; display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(280px, 1fr); gap: 16px; align-items: start; }
        .sl-section-block { border-top: 1px solid rgba(156,218,236,.12); padding-top: 10px; margin-top: 10px; }
        /* Wire slug: every section is a filed desk on the league wire. */
        .sl-section-block h3 { position: relative; margin: 0 0 8px; padding-left: 16px; font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: var(--cyan); }
        .sl-section-block h3::before { content: ""; position: absolute; left: 0; top: 50%; transform: translateY(-50%); width: 10px; height: 2px; background: var(--cyan); }
        .sl-row { width: 100%; text-align: left; border: 0; border-bottom: 1px solid rgba(156,218,236,.12); background: transparent; color: inherit; padding: 10px 0; display: grid; gap: 4px; }
        .sl-row.is-active { border-left: 3px solid var(--cyan); padding-left: 10px; margin-left: -10px; }
        .sl-row-meta { display: flex; gap: 10px; align-items: center; font-size: var(--type-wire-ts-size, 0.72rem); color: var(--muted); font-weight: 900; text-transform: uppercase; letter-spacing: 0.08em; font-family: var(--font-mono-data, monospace); }
        .sl-row h4 { margin: 0; font-size: clamp(0.875rem, 1.1vw, 1rem); line-height: 1.35; font-weight: 700; }
        .sl-row p { margin: 0; font-size: 0.8125rem; color: var(--muted); line-height: 1.35; }
        .sl-row-foot { font-size: 0.72rem; color: #9ec0d3; font-weight: 700; }
        /* Urgency reads as an editorial filing stamp, not coloured text alone. */
        .sl-row-urgency { color: var(--gold); border: 1px solid rgba(233,168,60,.42); border-radius: 2px; padding: 0 4px; letter-spacing: .12em; }

        .sl-side-section { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: rgba(6, 21, 34, 0.55); padding: 10px 0; margin-bottom: 12px; border-radius: 0; }
        .sl-side-section h3 { margin: 0 0 8px; padding: 0 10px; font-size: 0.6875rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan); font-weight: 900; }
        .sl-r-item { width: 100%; border: 0; border-bottom: 1px solid rgba(156,218,236,.12); background: transparent; text-align: left; color: inherit; padding: 8px 0; }
        .sl-r-item:last-child { border-bottom: 0; }
        .sl-r-item strong { display: block; font-size: 13px; line-height: 1.3; margin-bottom: 3px; }
        .sl-r-item p { margin: 0; color: var(--muted); font-size: 11px; }
        .sl-r-item em { display: block; margin-top: 3px; color: #d9b873; font-size: 11px; font-style: normal; font-weight: 800; }

        /* Selected story is a filed page: squared edge with a gold spine. */
        .sl-article { margin-top: 14px; border: 1px solid var(--line); border-left: 2px solid var(--gold); border-radius: 2px; background: rgba(7,20,31,.86); padding: 14px; }
        .sl-article-meta { display: flex; gap: 8px; font-size: 11px; font-weight: 900; text-transform: uppercase; color: var(--muted); }
        .sl-article h2 { margin: 8px 0 8px; font-size: clamp(24px, 2.4vw, 34px); line-height: 1.12; }
        .sl-article-standfirst { margin: 0 0 12px; color: rgba(233,247,251,.9); font-size: 14px; line-height: 1.5; }
        .sl-article-section { margin-top: 12px; border-top: 1px solid rgba(156,218,236,.12); padding-top: 10px; }
        .sl-article-section h4 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--cyan); }
        .sl-article-section p { margin: 0; font-size: 13px; line-height: 1.45; color: rgba(233,247,251,.88); }
        .sl-numbers { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; }
        .sl-num { border: 1px solid rgba(156,218,236,.14); border-radius: 6px; padding: 8px; }
        .sl-num strong { display: block; font-size: 19px; line-height: 1; }
        .sl-num span { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
        .sl-impact-list { margin: 8px 0 0; padding: 0; list-style: none; display: grid; gap: 6px; }
        .sl-impact-list li { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; font-weight: 800; }
        .sl-impact-list li.pos strong { color: var(--green); }
        .sl-impact-list li.neg strong { color: var(--red); }

        .sl-related-coverage { display: grid; gap: 8px; }
        .sl-related-coverage button { border: 1px solid rgba(156,218,236,.16); border-radius: 6px; background: rgba(255,255,255,.02); color: inherit; text-align: left; padding: 8px; }
        .sl-related-coverage span { display: block; font-size: 11px; font-weight: 900; text-transform: uppercase; color: var(--muted); }
        .sl-related-coverage strong { display: block; font-size: 13px; line-height: 1.3; margin: 3px 0; }
        .sl-related-coverage em { color: var(--muted); font-size: 11px; font-style: normal; }

        .sl-identity-inline { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .sl-identity-logo { width: 44px; height: 44px; border-radius: 8px; border: 1px solid rgba(156,218,236,.2); overflow: hidden; display: grid; place-items: center; background: rgba(255,255,255,.03); }
        .sl-identity-logo img { width: 100%; height: 100%; object-fit: contain; }
        .sl-identity-logo span { font-size: 13px; font-weight: 1000; color: var(--cyan); }
        .sl-identity-inline strong { display: block; font-size: 15px; }
        .sl-identity-inline p { margin: 2px 0 0; color: var(--muted); font-size: 11px; font-weight: 800; }

        /* An empty wire still reads as a newsroom page: masthead rule, column
           rules, and a stated reason for the silence. */
        .sl-empty-page {
          display: grid;
          align-content: start;
          gap: 5px;
          margin: 16px 0 0;
          flex: 1 1 auto;
          min-height: 44vh;
          max-width: none;
          text-align: left;
          border: 1px solid var(--line);
          border-top: 2px solid var(--cyan);
          border-radius: 0;
          background:
            repeating-linear-gradient(
              90deg,
              transparent 0 calc(33.333% - 1px),
              rgba(156, 218, 236, 0.07) calc(33.333% - 1px) 33.333%
            );
          padding: 16px 20px;
        }
        .sl-empty-page .sl-wire-kicker { margin: 0 0 6px; font-size: 0.6875rem; font-weight: 900; letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan); }
        .sl-empty-page h2 { margin: 0 0 6px; font-size: 0.95rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 800; }
        .sl-empty-page p { margin: 0; color: var(--muted); font-size: 0.8125rem; line-height: 1.45; }

        @keyframes sl-wire-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
        @media (max-width: 1000px) {
          .sl-layout { grid-template-columns: 1fr; }
          .sl-toolbar { grid-template-columns: 1fr; }
          .sl-numbers { grid-template-columns: repeat(2, minmax(0,1fr)); }
        }

        .sl-conduct-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
        .sl-conduct-grid span { font-size: 11px; font-weight: 800; color: #c7dde8; border: 1px solid rgba(156,218,236,.16); border-radius: 4px; padding: 4px 7px; }
        .sl-org-pressure { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
        .sl-org-pressure span { font-size: 11px; font-weight: 800; color: #ffc98a; border: 1px solid rgba(233,168,60,.22); border-radius: 4px; padding: 4px 7px; }
        .sl-trade-swap { display: flex; align-items: center; justify-content: center; gap: 16px; margin: 10px 0 12px; padding: 10px; border: 1px solid rgba(138, 180, 255, .25); border-radius: 8px; background: rgba(138, 180, 255, .06); }
        .sl-trade-side { display: grid; justify-items: center; gap: 4px; min-width: 64px; }
        .sl-trade-side img { width: 40px; height: 40px; object-fit: contain; }
        .sl-trade-side span { font-size: 11px; font-weight: 900; letter-spacing: .06em; }
        .sl-trade-swap em { font-style: normal; color: var(--ops-info, #8ab4ff); font-size: 18px; font-weight: 1000; }
        .sl-choice-grid { display: grid; gap: 8px; margin-top: 8px; }
        .nhlcal-storyline-choice-button {
          width: 100%; min-height: 44px; border: 1px solid var(--line); border-radius: 8px;
          background: rgba(19, 216, 231, 0.07); color: var(--text); padding: 10px 12px; text-align: left; cursor: pointer;
        }
        .nhlcal-storyline-choice-button:hover:not(:disabled) { border-color: var(--line-strong); background: rgba(19, 216, 231, 0.14); }
        .nhlcal-storyline-choice-button:disabled { opacity: .58; cursor: not-allowed; }
        .nhlcal-storyline-choice-button strong { display: block; color: var(--cyan); font-size: 11px; font-weight: 1000; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
        .nhlcal-storyline-choice-button span { display: block; color: var(--muted); font-size: 11px; font-weight: 800; }
        .sl-identity-inline .player-headshot { width: 44px; height: 44px; }

        @media (prefers-reduced-motion: reduce) {
          .sl-breaking-text { animation: none !important; }
          .nhlcal-sl-root * { transition: none !important; animation: none !important; }
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
            <p className="sl-status-line">{statusLine}</p>
          </div>

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

        {topPending ? (
          <button type="button" className="sl-frontoffice-alert" onClick={() => setSelectedId(topPending.id)}>
            <span>Front Office Alert</span>
            <strong>{topPending.headline}</strong>
            <em>Review decision →</em>
          </button>
        ) : null}

        {!hasBackend ? (
          <div className="sl-empty-page">
            <p className="sl-wire-kicker">League Wire · Idle</p>
            <h2>No Coverage Yet</h2>
            <p>Advance the calendar to populate the newsroom wire from backend storylines.</p>
          </div>
        ) : stories.length === 0 ? (
          <div className="sl-empty-page">
            <p className="sl-wire-kicker">League Wire · Idle</p>
            <h2>Wire Standing By</h2>
            <p>No active storylines on file. Coverage will appear as the season generates league beats.</p>
          </div>
        ) : (
          <>
            {leadStory ? (
              <section className="sl-lead">
                <div className="sl-lead-top">
                  <span>{categoryMeta(leadStory).label}</span>
                  <span>{leadStory.ageLabel || "—"}</span>
                  <span>{leadStory.teamName || leadStory.playerName || "League"}</span>
                </div>
                <h2>{leadStory.headline}</h2>
                {leadStory.summary ? <p className="sl-lead-summary">{leadStory.summary}</p> : null}
                <div className="sl-lead-signals">
                  {heatLabel(leadStory.heat) ? (
                    <span title={`Heat ${leadStory.heat}`}>Heat: {heatLabel(leadStory.heat)}</span>
                  ) : null}
                  {credibilityLabel(leadStory.credibility) ? (
                    <span title={`Credibility ${leadStory.credibility}`}>
                      Source: {credibilityLabel(leadStory.credibility)}
                    </span>
                  ) : null}
                  {leadStory.requiresAction ? <span>Decision Required</span> : null}
                </div>
              </section>
            ) : null}

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

            {filtered.length === 0 ? (
              <div className="sl-empty-page">
                <p className="sl-wire-kicker">Section Empty</p>
                <h2>{filterEmptyMsg ? "No Matches" : "No Stories"}</h2>
                <p>{filterEmptyMsg || "Try another filter or search term."}</p>
              </div>
            ) : (
              <div className="sl-layout">
                <div>
                  <section className="sl-section-block">
                    <h3>Latest Stories</h3>
                    {latestStories.map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        className={`sl-row ${selected?.id === s.id ? "is-active" : ""}`}
                        onClick={() => setSelectedId(s.id)}
                      >
                        <div className="sl-row-meta">
                          <span style={{ color: categoryMeta(s).accent }}>{categoryMeta(s).label}</span>
                          <span>{s.ageLabel || "—"}</span>
                          {s.requiresAction ? <span className="sl-row-urgency">Decision</span> : null}
                        </div>
                        <h4>{s.headline}</h4>
                        {s.summary ? <p>{s.summary.slice(0, 130)}{s.summary.length > 130 ? "…" : ""}</p> : null}
                        <div className="sl-row-foot">{s.playerName || s.teamName || "League"}</div>
                      </button>
                    ))}
                  </section>

                  {yourTeamStories.length ? (
                    <section className="sl-section-block">
                      <h3>Your Team</h3>
                      {yourTeamStories.map((s) => (
                        <button key={s.id} type="button" className="sl-row" onClick={() => setSelectedId(s.id)}>
                          <div className="sl-row-meta">
                            <span style={{ color: categoryMeta(s).accent }}>{categoryMeta(s).label}</span>
                            <span>{s.ageLabel || "—"}</span>
                          </div>
                          <h4>{s.headline}</h4>
                        </button>
                      ))}
                    </section>
                  ) : null}

                  {aroundLeagueStories.length ? (
                    <section className="sl-section-block">
                      <h3>Around the League</h3>
                      {aroundLeagueStories.map((s) => (
                        <button key={s.id} type="button" className="sl-row" onClick={() => setSelectedId(s.id)}>
                          <div className="sl-row-meta">
                            <span style={{ color: categoryMeta(s).accent }}>{categoryMeta(s).label}</span>
                            <span>{s.ageLabel || "—"}</span>
                          </div>
                          <h4>{s.headline}</h4>
                        </button>
                      ))}
                    </section>
                  ) : null}
                </div>

                <aside>
                  {rumourStories.length ? (
                    <section className="sl-side-section">
                      <h3>Rumour Mill</h3>
                      {rumourStories.map((s) => (
                        <button key={s.id} type="button" className="sl-r-item" onClick={() => setSelectedId(s.id)}>
                          <strong>{s.headline}</strong>
                          <p>{s.playerName || s.teamName || "League"}</p>
                          {heatLabel(s.heat) ? <em title={`Heat ${s.heat}`}>Heat: {heatLabel(s.heat)}</em> : null}
                          {credibilityLabel(s.credibility) ? (
                            <em title={`Credibility ${s.credibility}`}>Source: {credibilityLabel(s.credibility)}</em>
                          ) : null}
                        </button>
                      ))}
                    </section>
                  ) : null}

                  {hotStories.length ? (
                    <section className="sl-side-section">
                      <h3>Most Discussed</h3>
                      {hotStories.map((s) => (
                        <button key={s.id} type="button" className="sl-r-item" onClick={() => setSelectedId(s.id)}>
                          <strong>{s.headline}</strong>
                          <p>
                            {s.playerName || s.teamName || "League"}
                            {heatLabel(s.heat) ? ` · Heat: ${heatLabel(s.heat)}` : ""}
                          </p>
                        </button>
                      ))}
                    </section>
                  ) : null}

                  {userOrg ? (
                    <section className="sl-side-section">
                      <h3>Org Pressure</h3>
                      <div className="sl-org-pressure">
                        {userOrg.owner_confidence != null ? (
                          <span>Owner {Math.round(Number(userOrg.owner_confidence) * 100)}</span>
                        ) : null}
                        {userOrg.fan_approval != null ? (
                          <span>Fans {Math.round(Number(userOrg.fan_approval) * 100)}</span>
                        ) : null}
                        {userOrg.media_heat != null ? (
                          <span>Media {Math.round(Number(userOrg.media_heat) * 100)}</span>
                        ) : null}
                        {userOrg.sponsor_confidence != null ? (
                          <span>Sponsors {Math.round(Number(userOrg.sponsor_confidence) * 100)}</span>
                        ) : null}
                        {userOrg.revenue_modifier != null ? (
                          <span>Rev {Number(userOrg.revenue_modifier).toFixed(2)}×</span>
                        ) : null}
                      </div>
                    </section>
                  ) : null}

                  {pendingDecisions.length ? (
                    <section className="sl-side-section">
                      <h3>GM Decisions</h3>
                      {pendingDecisions.slice(0, 4).map((s) => (
                        <button key={s.id} type="button" className="sl-r-item" onClick={() => setSelectedId(s.id)}>
                          <strong>{s.headline}</strong>
                          <p>{s.playerName || s.teamName || "League"}</p>
                        </button>
                      ))}
                    </section>
                  ) : null}
                </aside>
              </div>
            )}

            {selected ? (
              <article className="sl-article">
                <div className="sl-article-meta">
                  <span style={{ color: categoryMeta(selected).accent }}>{categoryMeta(selected).label}</span>
                  <span>{selected.ageLabel || "—"}</span>
                </div>
                <h2>{selected.headline}</h2>
                {selected.summary ? <p className="sl-article-standfirst">{selected.summary}</p> : null}
                <TeamOrPlayerIdentity story={selected} />
                {isRumourStory(selected) ? <TradeSwap story={selected} /> : null}
                {selected.description && selected.description !== selected.summary ? (
                  <p className="sl-article-standfirst">{selected.description}</p>
                ) : null}
                <ConductChannels story={selected} />

                {(selected.userVisibleExplanation || selected.cause) ? (
                  <section className="sl-article-section">
                    <h4>How We Got Here</h4>
                    <p>{selected.userVisibleExplanation || selected.cause}</p>
                  </section>
                ) : null}

                {(selected.gamesRemaining > 0 || selected.impactReason || (selected.overallDelta != null && Number(selected.overallDelta) !== 0)) ? (
                  <section className="sl-article-section">
                    <h4>Lineup Impact</h4>
                    {selected.gamesRemaining > 0 ? (
                      <p>
                        Out {selected.gamesRemaining} games
                        {selected.returnEstimate ? ` · ${selected.returnEstimate}` : ""}
                        {selected.returnDate ? ` (${selected.returnDate})` : ""}
                      </p>
                    ) : null}
                    {selected.overallDelta != null && Number(selected.overallDelta) !== 0 ? (
                      <p>
                        Temporary readiness {selected.baseOverall ?? selected.overallBefore} → {selected.effectiveOverall ?? selected.overallAfter} ({Number(selected.overallDelta) > 0 ? "+" : ""}{selected.overallDelta})
                      </p>
                    ) : null}
                    {selected.impactReason ? <p>{selected.impactReason}</p> : null}
                  </section>
                ) : null}
                {selected.recoveryConditions?.length ? (
                  <section className="sl-article-section">
                    <h4>Road Back</h4>
                    <ul className="sl-impact-list">
                      {selected.recoveryConditions.map((r) => (
                        <li key={r}><span>{r}</span><strong /></li>
                      ))}
                    </ul>
                  </section>
                ) : null}
                {selected.culpritPlayerName ? (
                  <section className="sl-article-section">
                    <h4>How We Got Here</h4>
                    <p>Central figure: {selected.culpritPlayerName}</p>
                  </section>
                ) : null}

                {Object.keys(selected.evidence || {}).length ? (
                  <section className="sl-article-section">
                    <h4>By the Numbers</h4>
                    <div className="sl-numbers">
                      {Object.entries(selected.evidence)
                        .slice(0, 4)
                        .map(([k, v]) => (
                          <div key={k} className="sl-num" title={`${formatEffectLabel(k)}: ${v}`}>
                            <strong>{String(v)}</strong>
                            <span>{formatEffectLabel(k)}</span>
                          </div>
                        ))}
                    </div>
                  </section>
                ) : null}

                {(selected.effectSummary || Object.keys(selected.effects || {}).length) ? (
                  <section className="sl-article-section">
                    <h4>Why It Matters</h4>
                    {selected.effectSummary ? <p>{selected.effectSummary}</p> : null}
                    {Object.keys(selected.effects || {}).length ? (
                      <ul className="sl-impact-list">
                        {Object.entries(selected.effects)
                          .slice(0, 3)
                          .map(([k, v]) => (
                            <li key={k} className={effectPillClass(v)}>
                              <span>{formatEffectLabel(k)}</span>
                              <strong>
                                {Number(v) > 0 ? "+" : ""}
                                {String(v)}
                              </strong>
                            </li>
                          ))}
                      </ul>
                    ) : null}
                  </section>
                ) : null}

                <section className="sl-article-section">
                  <h4>What Comes Next</h4>
                  <p>{deriveFollowUp(selected)}</p>
                </section>

                {relatedStories.length ? (
                  <section className="sl-article-section">
                    <h4>Related Coverage</h4>
                    <div className="sl-related-coverage">
                      {relatedStories.slice(0, 3).map((r) => (
                        <button key={r.id} type="button" onClick={() => setSelectedId(r.id)}>
                          <span>{categoryMeta(r).label}</span>
                          <strong>{r.headline}</strong>
                          <em>{r.ageLabel || "—"}</em>
                        </button>
                      ))}
                    </div>
                  </section>
                ) : null}

                {asArray(selectedChoice?.action_options).length || asArray(selected.actionOptions).length ? (
                  <section className="sl-article-section">
                    <h4>Front Office Decision</h4>
                    <div className="sl-choice-grid">
                      {(asArray(selectedChoice?.action_options).length ? selectedChoice.action_options : selected.actionOptions).map((opt) => {
                        const busy = busyChoice === `${selected.storylineId}:${opt.id}`;
                        return (
                          <button
                            key={opt.id}
                            type="button"
                            className="nhlcal-storyline-choice-button"
                            disabled={Boolean(busyChoice)}
                            onClick={() => handleResolve(selectedChoice?.storyline_id || selected.storylineId, opt.id)}
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

                {selected.repeatCount > 0 ? (
                  <section className="sl-article-section">
                    <h4>Developing Story</h4>
                    <p>
                      Beat #{selected.repeatCount + 1}
                      {selected.escalatedFrom ? ` · escalated from ${selected.escalatedFrom}` : ""}
                    </p>
                  </section>
                ) : null}
              </article>
            ) : null}
          </>
        )}
        {process.env.NODE_ENV === "development" && !hasBackend ? (
          <details className="sl-article-section" style={{ marginTop: 12 }}>
            <summary>Storyline debug (dev)</summary>
            <pre style={{ fontSize: 11, overflow: "auto", maxHeight: 240 }}>
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
          <details className="sl-article-section" style={{ marginTop: 12 }}>
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
