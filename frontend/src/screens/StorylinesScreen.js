import React, { useMemo, useState, useCallback, useEffect } from "react";
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

/*
  StorylinesScreen — backend-driven news hub.
  Rules: read franchiseState only; no fake storylines or invented metrics.
  UI language: game HUD (feed / case file / impact rail), not a news website.
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
  locker_room: { icon: "◎", label: "Locker Room", accent: "#7ee0b0" },
  business: { icon: "$", label: "Business", accent: "#52df94" },
  management: { icon: "▣", label: "Front Office", accent: "#e9a83c" },
  storyline: { icon: "◉", label: "League News", accent: "#8096a8" },
};

const FILTERS = [
  { id: "all", label: "All" },
  { id: "major", label: "Major" },
  { id: "rumors", label: "Rumors" },
  { id: "team", label: "Team" },
  { id: "league", label: "League" },
  { id: "player", label: "Player" },
  { id: "media_buzz", label: "Media Buzz" },
];

const FILTER_EMPTY = {
  major: "No major developments on file.",
  rumors: "Trade wire is calm — for now.",
  team: "Nothing filed on your team today.",
  league: "League desk is quiet.",
  player: "No player-specific beats yet.",
  media_buzz: "Media buzz is flat right now.",
};

const DEPARTMENTS = [
  { id: "front_page", label: "Newsroom" },
  { id: "social", label: "Social" },
  { id: "insiders", label: "Insiders" },
  { id: "press_room", label: "Press Room" },
  { id: "archive", label: "Archive" },
];

const PRIORITY_RANK = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
const DETAIL_TABS = [
  { id: "details", label: "Details" },
  { id: "related", label: "Related Coverage" },
  { id: "rumors", label: "Rumor Mill" },
  { id: "history", label: "History" },
];

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

function resolveCategoryKey(story) {
  const cat = str(story?.category || story?.type || "").toLowerCase();
  if (story?.requiresAction) return "decision";
  if (cat.includes("legal") || cat === "legal_trouble") return "legal_trouble";
  if (cat === "injury" || /injur/i.test(cat)) return "injury";
  if (/draft|prospect/.test(cat)) return "draft";
  if (/goalie|goaltender/.test(cat)) return "goalie";
  if (/trade|rumor|contract/.test(cat)) return /contract/.test(cat) ? "contract" : "trade";
  if (/locker|belong|role/.test(cat)) return "locker_room";
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
  if (filter === "major") return story.priorityRank >= 3 || story.requiresAction;
  if (filter === "rumors") return isRumourStory(story);
  if (filter === "team") return story.isUserTeam;
  if (filter === "league") return !story.isUserTeam;
  if (filter === "player") return Boolean(story.playerName);
  if (filter === "media_buzz") return Number(story.heat) >= 40;
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
function heatTier(heat) {
  const n = Number(heat) || 0;
  if (n >= 70) return "hot";
  if (n >= 40) return "warm";
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
  return p === "CRITICAL" || heat >= 75 || story?.requiresAction;
}

// --- score badge: a visual read of real heat/priority data, not a fabricated stat ---
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

function buildSocialPosts(stories, narrativeUniverse) {
  const backendPosts = asArray(narrativeUniverse?.social_posts);
  if (backendPosts.length) {
    return backendPosts.slice(0, 40).map((p, idx) => ({
      id: str(p.id || `post-${idx}`),
      handle: str(p.handle || `@User${idx}`),
      name: str(p.author_name || p.name || "Hockey Fan"),
      verified: Boolean(p.verified),
      isAgent: str(p.author_type || "") === "agent",
      agency: str(p.agency || ""),
      age: str(p.calendar_iso || "—"),
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



function collectDossiers(narrativeUniverse) {
  const direct = asArray(narrativeUniverse?.player_dossiers);
  if (direct.length) return direct;
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

function DossierCard({ dossier }) {
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
    <article className="sl-dossier">
      <div className="sl-dossier__head">
        <strong>{str(dossier.player_name || ident.name || "Player")}</strong>
        <span>
          {str(dossier.position || ident.position || "")}
          {dossier.overall != null ? ` · ${Math.round(Number(dossier.overall))}` : ""}
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
            Object.entries(trusts).map(([key, val]) => (
              <p key={key}>
                {formatEffectLabel(key)} · {Math.round(Number(val) || 0)}
              </p>
            ))
          ) : (
            <p>Trust ledger still forming.</p>
          )}
        </div>
      </div>
      <div>
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

function isRumourStory(story) {
  return /trade|rumor|contract|market/i.test(`${story.type} ${story.category} ${story.headline}`);
}

function playBreakingSting(level) {
  if (typeof window === "undefined") return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = level === "league_defining" ? "sawtooth" : "square";
    osc.frequency.value = level === "league_defining" ? 880 : 660;
    gain.gain.value = 0.04;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + (level === "league_defining" ? 0.45 : 0.28));
    osc.stop(ctx.currentTime + 0.5);
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

// ---- small presentational pieces ----

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
      {stage}
      {stage === "Escalating" ? <em aria-hidden>▲</em> : null}
    </span>
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
  const eligible = story.eligibleToPlay == null ? null : story.eligibleToPlay ? "Eligible to dress" : "Cannot dress";
  return (
    <section className="sl-detail-block">
      <h4>Conduct Desk</h4>
      {story.allegationNote ? <p>{story.allegationNote}</p> : null}
      <div className="sl-chip-grid">
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

function DevelopmentTimeline({ beats, fallbackStory }) {
  const nodes =
    beats.length > 1
      ? beats
      : fallbackStory
      ? [{ id: "origin", date: fallbackStory.date || fallbackStory.ageLabel, headline: "First reported", summary: "" }]
      : [];
  if (!nodes.length) return null;
  return (
    <section className="sl-timeline">
      <h4>Development Timeline</h4>
      <div className="sl-timeline-track">
        {nodes.map((n, idx) => (
          <div key={n.id || idx} className={`sl-timeline-node${idx === nodes.length - 1 ? " is-latest" : ""}`}>
            <span className="sl-timeline-dot" />
            <time>{n.date || "—"}</time>
            <strong>{n.headline}</strong>
            {n.summary ? <p>{n.summary}</p> : null}
          </div>
        ))}
        <div className="sl-timeline-node sl-timeline-node--next">
          <span className="sl-timeline-dot sl-timeline-dot--ghost" />
          <time>What's next?</time>
          <strong>Situation developing</strong>
        </div>
      </div>
    </section>
  );
}

function OrgPressureBars({ org }) {
  if (!org) return null;
  const rows = [
    ["owner_confidence", "Owner Confidence"],
    ["fan_approval", "Fan Approval"],
    ["media_heat", "Media Pressure"],
    ["sponsor_confidence", "Sponsor Confidence"],
  ].filter(([key]) => org[key] != null);
  if (!rows.length) return null;
  return (
    <div className="sl-bars">
      {rows.map(([key, label]) => {
        const pct = Math.round(Number(org[key]) * 100);
        const tone = key === "media_heat" ? (pct >= 70 ? "hot" : "warm") : pct >= 60 ? "good" : pct >= 35 ? "warm" : "hot";
        return (
          <div key={key} className="sl-bar-row">
            <div className="sl-bar-label">
              <span>{label}</span>
              <strong>{pct}/100</strong>
            </div>
            <div className="sl-bar-track">
              <div className={`sl-bar-fill sl-bar-fill--${tone}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
            </div>
          </div>
        );
      })}
      {org.revenue_modifier != null ? (
        <p className="sl-bar-footnote">Revenue modifier ×{Number(org.revenue_modifier).toFixed(2)}</p>
      ) : null}
    </div>
  );
}

export default function StorylinesScreen() {
  const { franchiseState, onResolveStorylineChoice, setScreen } = useGameUI();
  const [department, setDepartment] = useState("front_page");
  const [filter, setFilter] = useState("all");
  const [sortId, setSortId] = useState("decisions");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [activeTab, setActiveTab] = useState("details");
  const [busyChoice, setBusyChoice] = useState("");
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

  const openStory = useCallback((id) => {
    if (!id) return;
    setSelectedId(id);
    setExpandedId(id);
    setFilter("all");
    setSearch("");
  }, []);

  const toggleRow = useCallback(
    (id) => {
      setExpandedId((prev) => (prev === id ? null : id));
      setSelectedId(id);
    },
    []
  );

  const stories = useMemo(() => collectStories(franchiseState), [franchiseState]);
  const choicesMap = useMemo(() => buildChoicesMap(franchiseState), [franchiseState]);

  const filterCounts = useMemo(() => {
    const counts = { all: stories.length };
    FILTERS.forEach((f) => {
      if (f.id === "all") return;
      counts[f.id] = stories.filter((s) => matchesFilter(s, f.id)).length;
    });
    return counts;
  }, [stories]);

  const filtered = useMemo(() => {
    const base = stories.filter((s) => matchesFilter(s, filter) && matchesSearch(s, search));
    return sortStories(base, sortId);
  }, [stories, filter, search, sortId]);

  const pendingDecisions = useMemo(
    () => stories.filter((s) => s.requiresAction || choicesMap.has(s.storylineId) || choicesMap.has(s.id)),
    [stories, choicesMap]
  );
  const topPending = pendingDecisions[0] || null;
  const yourTeamCount = stories.filter((s) => s.isUserTeam).length;

  const orgPressure = asObject(franchiseState?.conduct_org_pressure);
  const userOrg =
    orgPressure[userTeamId(franchiseState)] || orgPressure[str(franchiseState?.user_team_id || "")] || null;

  const narrativeUniverse = asObject(franchiseState?.narrative_universe);
  const pressQueue = asArray(narrativeUniverse?.press_conference_queue).filter((p) => str(p?.status) === "pending");
  const narrativeEras = asArray(narrativeUniverse?.narrative_eras);
  const narrativeArchive = asArray(narrativeUniverse?.narrative_archive);
  const userMarket = asObject(narrativeUniverse?.user_market_profile);
  const agentRoster = asArray(narrativeUniverse?.agents);
  const agentRelationships = asObject(narrativeUniverse?.agent_relationships);
  const knowledgeGraph = asArray(narrativeUniverse?.knowledge_graph);
  const insiderItems = asArray(narrativeUniverse?.insider_items).length
    ? asArray(narrativeUniverse.insider_items)
    : knowledgeGraph;
  const beatWriters = asArray(narrativeUniverse?.beat_writers).length
    ? asArray(narrativeUniverse.beat_writers)
    : asArray(narrativeUniverse?.reporters);
  const playerDossiers = useMemo(() => collectDossiers(narrativeUniverse), [narrativeUniverse]);
  const breakingAlerts = asArray(narrativeUniverse?.breaking_alerts);
  const pendingBreaking = useMemo(() => activeBreakingAlerts(breakingAlerts, dismissedBreaking), [breakingAlerts, dismissedBreaking]);
  const activeBreaking = pendingBreaking[0] || null;

  useEffect(() => {
    if (!activeBreaking?.level) return;
    playBreakingSting(activeBreaking.level);
  }, [activeBreaking?.storyline_id, activeBreaking?.headline, activeBreaking?.level]);

  const socialPosts = useMemo(() => buildSocialPosts(stories, narrativeUniverse), [stories, narrativeUniverse]);
  const socialCountByStory = useMemo(() => {
    const map = new Map();
    asArray(narrativeUniverse?.social_posts).forEach((p) => {
      const sid = str(p?.storyline_id || "");
      if (!sid) return;
      map.set(sid, (map.get(sid) || 0) + 1);
    });
    return map;
  }, [narrativeUniverse]);

  const selected = filtered.find((s) => s.id === selectedId) || stories.find((s) => s.id === selectedId) || filtered[0] || null;
  const selectedDossier = playerDossiers.find((d) => str(d.player_id) === str(selected?.playerId)) || null;

  useEffect(() => {
    if (!filtered.length) {
      if (selectedId != null) setSelectedId(null);
      return;
    }
    if (!filtered.some((s) => s.id === selectedId)) setSelectedId(filtered[0].id);
  }, [filtered, selectedId]);

  useEffect(() => {
    setActiveTab("details");
  }, [selectedId]);

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

  const leagueRumours = useMemo(() => stories.filter((s) => isRumourStory(s) && s.id !== selected?.id).slice(0, 10), [stories, selected]);

  const selectedChoice = selected ? choicesMap.get(selected.storylineId) || choicesMap.get(selected.id) : null;

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

  const handlePressResponse = useCallback(
    async (pressItem, questionId, responseId) => {
      if (!onResolveStorylineChoice || !pressItem) return;
      const choiceId = `${questionId}:${responseId}`;
      const storylineId = str(pressItem.storyline_id || pressItem.storylineId);
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
  const arcTimeline = useMemo(() => collectArcTimeline(stories, selected, narrativeUniverse), [stories, selected, narrativeUniverse]);
  const statusLine = `${stories.length} active${yourTeamCount ? ` · ${yourTeamCount} involving your team` : ""}${
    pendingDecisions.length ? ` · ${pendingDecisions.length} decisions pending` : ""
  }`;

  // key factors + information + parties are all pulled from real fields only — nothing invented
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
      list.push({ label: "Central Figure", name: selected.culpritPlayerName });
    if (selected.fromTeamName) list.push({ label: "From", name: selected.fromTeamName });
    if (selected.toTeamName) list.push({ label: "To", name: selected.toTeamName });
    if (selected.reporterName) list.push({ label: "Reporter", name: `${selected.reporterName}${selected.outletName ? ` (${selected.outletName})` : ""}` });
    if (selected.sourceLabel && !selected.reporterName) list.push({ label: "Source", name: selected.sourceLabel });
    return list;
  }, [selected]);

  const infoRows = useMemo(() => {
    if (!selected) return [];
    const rows = [];
    if (selected.informationStatus) rows.push(["Information Status", formatEffectLabel(selected.informationStatus)]);
    if (credibilityLabel(selected.credibility)) rows.push(["Evidence Strength", credibilityLabel(selected.credibility)]);
    if (knowledgeLevelLabel(selected.publicKnowledgeLevel)) rows.push(["Public Visibility", knowledgeLevelLabel(selected.publicKnowledgeLevel)]);
    if (selected.sourceLabel) rows.push(["Leaked By", selected.sourceLabel]);
    return rows;
  }, [selected]);

  const choiceOptions = selected
    ? (asArray(selectedChoice?.action_options).length ? selectedChoice.action_options : selected.actionOptions)
    : [];

  return (
    <div className="nhlcal-sl-root">
      <style>{`
        .nhlcal-sl-root {
          --bg: #04101a;
          --panel: rgba(9, 25, 38, 0.94);
          --panel-2: rgba(6, 18, 29, 0.9);
          --line: rgba(156, 218, 236, 0.14);
          --line-strong: rgba(73, 231, 240, 0.5);
          --text: #e9f7fb;
          --muted: #8096a8;
          --cyan: #13d8e7;
          --cyan-soft: rgba(19, 216, 231, 0.13);
          --gold: #e9a83c;
          --gold-soft: rgba(233, 168, 60, 0.14);
          --green: #52df94;
          --red: #ff606d;
          --red-soft: rgba(255, 96, 109, 0.13);
          --purple: #c992ff;

          min-height: 100vh;
          width: 100%;
          background:
            radial-gradient(circle at 20% 0%, rgba(19, 216, 231, 0.1), transparent 32%),
            radial-gradient(circle at 92% 12%, rgba(233, 168, 60, 0.07), transparent 24%),
            linear-gradient(180deg, #06131f 0%, #020a11 100%);
          color: var(--text);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .sl-app { width: min(1480px, 100%); margin: 0 auto; padding: 14px 18px 32px; display: flex; flex-direction: column; gap: 12px; }

        /* ---------- top bar ---------- */
        .sl-topbar { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px; border-bottom: 1px solid var(--line); padding-bottom: 10px; }
        .sl-eyebrow { margin: 0; font-size: 10.5px; font-weight: 900; letter-spacing: .16em; text-transform: uppercase; color: var(--cyan); }
        .sl-topbar h1 { margin: 3px 0; font-size: 26px; font-weight: 900; letter-spacing: .05em; text-transform: uppercase; }
        .sl-topbar-sub { margin: 0; color: var(--muted); font-size: 11.5px; font-weight: 800; }
        .sl-topbar-nav { display: flex; gap: 8px; }
        .sl-topbar-nav button { height: 32px; border: 1px solid var(--line); border-radius: 6px; background: rgba(14,35,50,.85); color: var(--text); padding: 0 14px; font-size: 11px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; }
        .sl-topbar-nav button:hover { border-color: var(--line-strong); }

        /* ---------- department pills ---------- */
        .sl-departments { display: flex; gap: 6px; flex-wrap: wrap; }
        .sl-departments button { height: 30px; border: 1px solid var(--line); border-radius: 6px; background: rgba(14,35,50,.6); color: rgba(233,247,251,.78); padding: 0 12px; font-size: 11px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; }
        .sl-departments button.is-active { border-color: var(--line-strong); background: var(--cyan-soft); color: var(--text); }
        .sl-departments button.has-alert { border-color: rgba(255,96,109,.5); color: #ffb4bb; }

        /* ---------- filter bar ---------- */
        .sl-filterbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 8px; background: rgba(7,20,31,.7); padding: 8px 10px; }
        .sl-filter-chips { display: flex; gap: 6px; flex-wrap: wrap; }
        .sl-chip { height: 30px; border: 1px solid var(--line); border-radius: 999px; background: rgba(14,35,50,.7); color: rgba(233,247,251,.8); padding: 0 12px; font-size: 11px; font-weight: 900; letter-spacing: .04em; text-transform: uppercase; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
        .sl-chip.is-active { border-color: var(--line-strong); background: var(--cyan-soft); color: var(--text); }
        .sl-chip .sl-chip-count { opacity: .65; font-weight: 700; }
        .sl-filter-tools { display: flex; gap: 8px; align-items: center; }
        .sl-search { height: 30px; border: 1px solid var(--line); border-radius: 6px; background: rgba(8,23,35,.86); color: var(--text); padding: 0 10px; font-size: 12px; font-weight: 700; width: 190px; }
        .sl-sort select { height: 30px; border: 1px solid var(--line); border-radius: 6px; background: rgba(14,35,50,.9); color: var(--text); font-size: 11px; font-weight: 800; padding: 0 8px; }

        .sl-frontoffice-alert {
          width: 100%; text-align: left; border: 1px solid rgba(255,96,109,.5); border-left: 4px solid var(--red);
          background: linear-gradient(90deg, rgba(255,96,109,.14), rgba(6,21,34,.85)); color: var(--text);
          padding: 9px 12px; border-radius: 6px; display: flex; align-items: center; gap: 10px; cursor: pointer;
        }
        .sl-frontoffice-alert span { color: #ff8d97; font-size: 10.5px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
        .sl-frontoffice-alert strong { font-size: 13px; font-weight: 800; flex: 1; }
        .sl-frontoffice-alert em { font-size: 11px; color: #ffc9cf; font-style: normal; font-weight: 800; }

        /* ---------- single-stream feed ---------- */
        .sl-stream { border: 1px solid var(--line); border-radius: 10px; background: var(--panel-2); overflow: hidden; }
        .sl-stream-head { display: flex; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid var(--line); font-size: 10.5px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
        .sl-stream-head span:last-child { color: var(--cyan); }
        .sl-feed-empty { padding: 24px 16px; color: var(--muted); font-size: 12px; text-align: center; }

        /* story row */
        .sl-row { border-bottom: 1px solid rgba(156,218,236,.1); }
        .sl-row:last-child { border-bottom: 0; }
        .sl-row.is-open { background: rgba(19,216,231,.04); }
        .sl-row__header { width: 100%; display: grid; grid-template-columns: 4px 1fr auto; align-items: stretch; gap: 12px; text-align: left; border: 0; background: transparent; color: inherit; padding: 12px 14px 12px 0; cursor: pointer; }
        .sl-row__header:hover { background: rgba(19,216,231,.05); }
        .sl-row__heatbar { border-radius: 3px; align-self: stretch; min-height: 100%; }
        .sl-row__heatbar--cool { background: linear-gradient(180deg, var(--cyan), rgba(19,216,231,.35)); }
        .sl-row__heatbar--warm { background: linear-gradient(180deg, var(--gold), rgba(233,168,60,.4)); }
        .sl-row__heatbar--hot { background: linear-gradient(180deg, var(--red), rgba(255,96,109,.4)); }
        .sl-row__main { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
        .sl-row__topline { display: flex; align-items: center; gap: 8px; font-size: 10px; font-weight: 900; text-transform: uppercase; letter-spacing: .06em; }
        .sl-row__decision { color: var(--gold); }
        .sl-row__headline { margin: 0; font-size: 14px; line-height: 1.32; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .sl-row.is-open .sl-row__headline { white-space: normal; }
        .sl-row__subline { display: flex; gap: 10px; font-size: 11px; color: var(--muted); font-weight: 700; }
        .sl-row__pulse { color: var(--cyan); }
        .sl-row__meta { display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between; gap: 4px; flex-shrink: 0; }
        .sl-row__date { font-size: 10px; color: var(--muted); font-weight: 800; white-space: nowrap; }
        .sl-row__chevron { font-size: 10px; color: var(--muted); }
        .sl-row__body { border-top: 1px solid var(--line); background: var(--panel); padding: 16px; }
        .sl-row__layout { display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 16px; align-items: start; }
        @media (max-width: 900px) { .sl-row__layout { grid-template-columns: 1fr; } }
        .sl-row__impact { display: grid; gap: 12px; }

        /* score badge */
        .sl-score { width: 38px; height: 38px; border-radius: 8px; display: grid; place-items: center; font-weight: 900; border: 1px solid; align-self: start; }
        .sl-score--sm { width: 30px; height: 30px; }
        .sl-score--lg { width: 56px; height: 56px; border-radius: 10px; }
        .sl-score strong { font-size: 14px; }
        .sl-score--lg strong { font-size: 20px; }
        .sl-score--crit { background: rgba(255,96,109,.16); border-color: rgba(255,96,109,.55); color: #ff9aa2; }
        .sl-score--high { background: rgba(233,168,60,.16); border-color: rgba(233,168,60,.5); color: #ffcd7a; }
        .sl-score--mid { background: rgba(19,216,231,.12); border-color: rgba(19,216,231,.4); color: var(--cyan); }
        .sl-score--low { background: rgba(156,218,236,.08); border-color: rgba(156,218,236,.22); color: var(--muted); }

        .sl-status-pill { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; border-radius: 4px; padding: 3px 8px; }
        .sl-status-pill--escalating { color: #ff9aa2; background: rgba(255,96,109,.12); border: 1px solid rgba(255,96,109,.4); }
        .sl-status-pill--developing { color: #ffcd7a; background: rgba(233,168,60,.12); border: 1px solid rgba(233,168,60,.35); }
        .sl-status-pill--resolved { color: #8ef0b8; background: rgba(82,223,148,.12); border: 1px solid rgba(82,223,148,.35); }

        /* detail column */
        .sl-detail-head { display: flex; gap: 12px; align-items: flex-start; }
        .sl-detail-headmeta { flex: 1; min-width: 0; }
        .sl-detail-eyebrow { display: flex; align-items: center; gap: 10px; font-size: 11px; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 4px; }
        .sl-detail-head h2 { margin: 4px 0 8px; font-size: clamp(18px, 2.2vw, 26px); font-weight: 800; line-height: 1.15; }
        .sl-detail-source { display: flex; gap: 10px; flex-wrap: wrap; font-size: 11px; color: var(--muted); font-weight: 800; margin-bottom: 12px; }
        .sl-detail-summary { margin: 0 0 10px; font-size: 13.5px; line-height: 1.5; color: rgba(233,247,251,.9); }

        .sl-identity-inline { display: flex; gap: 10px; align-items: center; margin: 10px 0; }
        .sl-identity-logo { width: 46px; height: 46px; border-radius: 8px; border: 1px solid rgba(156,218,236,.2); overflow: hidden; display: grid; place-items: center; background: rgba(255,255,255,.03); }
        .sl-identity-logo img { width: 100%; height: 100%; object-fit: contain; }
        .sl-identity-logo span { font-size: 13px; font-weight: 1000; color: var(--cyan); }
        .sl-identity-inline strong { display: block; font-size: 15px; }
        .sl-identity-inline p { margin: 2px 0 0; color: var(--muted); font-size: 11px; font-weight: 800; }

        .sl-trade-swap { display: flex; align-items: center; justify-content: center; gap: 16px; margin: 10px 0 12px; padding: 10px; border: 1px solid rgba(138,180,255,.25); border-radius: 8px; background: rgba(138,180,255,.06); }
        .sl-trade-side { display: grid; justify-items: center; gap: 4px; min-width: 64px; }
        .sl-trade-side img { width: 40px; height: 40px; object-fit: contain; }
        .sl-trade-side span { font-size: 11px; font-weight: 900; letter-spacing: .06em; }
        .sl-trade-swap em { font-style: normal; color: #8ab4ff; font-size: 18px; font-weight: 1000; }

        /* development timeline */
        .sl-timeline { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
        .sl-timeline h4 { margin: 0 0 12px; font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--cyan); }
        .sl-timeline-track { display: flex; gap: 14px; overflow-x: auto; padding-bottom: 4px; }
        .sl-timeline-node { position: relative; flex: 0 0 150px; padding-top: 14px; border-top: 2px solid rgba(19,216,231,.4); }
        .sl-timeline-node.is-latest { border-top-color: var(--red); }
        .sl-timeline-node--next { border-top-color: rgba(156,218,236,.2); opacity: .6; }
        .sl-timeline-dot { position: absolute; top: -5px; left: 0; width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); }
        .sl-timeline-node.is-latest .sl-timeline-dot { background: var(--red); }
        .sl-timeline-dot--ghost { background: var(--muted); }
        .sl-timeline-node time { display: block; font-size: 10px; font-weight: 900; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; margin-bottom: 3px; }
        .sl-timeline-node strong { display: block; font-size: 12px; line-height: 1.3; }
        .sl-timeline-node p { margin: 3px 0 0; font-size: 11px; color: var(--muted); line-height: 1.3; }

        /* tabs */
        .sl-tabs { display: flex; gap: 4px; margin-top: 16px; border-bottom: 1px solid var(--line); }
        .sl-tabs button { border: 0; border-bottom: 2px solid transparent; background: transparent; color: var(--muted); font-size: 11px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; padding: 8px 4px; margin-right: 16px; cursor: pointer; }
        .sl-tabs button.is-active { color: var(--text); border-bottom-color: var(--cyan); }
        .sl-tab-panel { padding-top: 14px; }

        .sl-detail-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 16px; }
        @media (max-width: 800px) { .sl-detail-grid { grid-template-columns: 1fr; } }
        .sl-detail-block h4 { margin: 0 0 8px; font-size: 10.5px; text-transform: uppercase; letter-spacing: .1em; color: var(--cyan); font-weight: 900; }
        .sl-info-row { display: flex; justify-content: space-between; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(156,218,236,.1); font-size: 12px; }
        .sl-info-row span:first-child { color: var(--muted); font-weight: 700; }
        .sl-info-row span:last-child { font-weight: 800; }
        .sl-party-row { display: flex; justify-content: space-between; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(156,218,236,.1); font-size: 12px; }
        .sl-party-row span:first-child { color: var(--muted); font-weight: 700; }
        .sl-key-factors { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
        .sl-key-factors li { font-size: 12px; line-height: 1.4; padding-left: 14px; position: relative; color: rgba(233,247,251,.88); }
        .sl-key-factors li::before { content: "•"; position: absolute; left: 0; color: var(--gold); }

        .sl-chip-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
        .sl-chip-grid span { font-size: 11px; font-weight: 800; color: #c7dde8; border: 1px solid rgba(156,218,236,.16); border-radius: 4px; padding: 4px 7px; }

        .sl-numbers { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin-top: 6px; }
        .sl-num { border: 1px solid rgba(156,218,236,.14); border-radius: 6px; padding: 8px; }
        .sl-num strong { display: block; font-size: 18px; line-height: 1; }
        .sl-num span { color: var(--muted); font-size: 10.5px; font-weight: 800; text-transform: uppercase; }

        .sl-related-list, .sl-rumor-list, .sl-history-list { display: grid; gap: 8px; }
        .sl-related-list button, .sl-rumor-list button { border: 1px solid rgba(156,218,236,.16); border-radius: 6px; background: rgba(255,255,255,.02); color: inherit; text-align: left; padding: 10px; cursor: pointer; display: grid; gap: 3px; }
        .sl-related-list button:hover, .sl-rumor-list button:hover { border-color: var(--line-strong); }
        .sl-related-list span, .sl-rumor-list span { font-size: 10px; font-weight: 900; text-transform: uppercase; color: var(--muted); }
        .sl-related-list strong, .sl-rumor-list strong { font-size: 12.5px; line-height: 1.3; }
        .sl-related-list em, .sl-rumor-list em { font-size: 10.5px; color: var(--muted); font-style: normal; }

        .sl-history-list li { display: grid; grid-template-columns: 78px 1fr; gap: 10px; padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid rgba(156,218,236,.1); list-style: none; }
        .sl-history-list time { font-size: 10.5px; font-weight: 800; color: var(--muted); }
        .sl-history-list strong { display: block; font-size: 12.5px; }
        .sl-history-list p { margin: 3px 0 0; font-size: 11.5px; color: var(--muted); line-height: 1.35; }

        .sl-detail-footer { display: flex; justify-content: space-between; margin-top: 16px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 10.5px; color: var(--muted); font-weight: 700; }

        /* impact rail */
        .sl-impact-panel { border: 1px solid var(--line); border-radius: 10px; background: var(--panel-2); padding: 12px 14px; }
        .sl-impact-panel h3 { margin: 0 0 10px; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--cyan); font-weight: 900; }
        .sl-impact-panel h3.is-title { color: var(--text); font-size: 13px; letter-spacing: .06em; }

        .sl-bars { display: grid; gap: 10px; }
        .sl-bar-label { display: flex; justify-content: space-between; font-size: 10.5px; font-weight: 800; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }
        .sl-bar-label strong { color: var(--text); }
        .sl-bar-track { height: 6px; border-radius: 4px; background: rgba(156,218,236,.1); overflow: hidden; }
        .sl-bar-fill { height: 100%; border-radius: 4px; }
        .sl-bar-fill--good { background: var(--green); }
        .sl-bar-fill--warm { background: var(--gold); }
        .sl-bar-fill--hot { background: var(--red); }
        .sl-bar-footnote { margin: 4px 0 0; font-size: 10.5px; color: var(--muted); font-weight: 700; }

        .sl-effects-grid { display: grid; gap: 6px; }
        .sl-effect-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 11.5px; font-weight: 800; padding: 6px 0; border-bottom: 1px solid rgba(156,218,236,.08); }
        .sl-effect-row.pos strong { color: var(--green); }
        .sl-effect-row.neg strong { color: var(--red); }

        .sl-decisions { display: grid; gap: 8px; }
        .sl-decision-btn { width: 100%; text-align: left; border: 1px solid var(--line); border-radius: 8px; background: rgba(19,216,231,.06); color: var(--text); padding: 10px 12px; cursor: pointer; display: flex; gap: 10px; align-items: flex-start; }
        .sl-decision-btn:hover:not(:disabled) { border-color: var(--line-strong); background: rgba(19,216,231,.13); }
        .sl-decision-btn:disabled { opacity: .55; cursor: not-allowed; }
        .sl-decision-btn__dot { width: 8px; height: 8px; border-radius: 50%; border: 2px solid var(--cyan); margin-top: 4px; flex-shrink: 0; }
        .sl-decision-btn strong { display: block; font-size: 12.5px; font-weight: 900; text-transform: uppercase; letter-spacing: .03em; margin-bottom: 3px; }
        .sl-decision-btn span { display: block; font-size: 11px; color: var(--muted); font-weight: 700; line-height: 1.35; }
        .sl-decision-btn em { display: block; margin-top: 4px; color: var(--cyan); font-size: 10.5px; font-style: normal; font-weight: 800; }
        .sl-decision-empty { font-size: 12px; color: var(--muted); text-align: center; padding: 12px 0; }

        /* empty state */
        .sl-empty-panel { border: 1px solid var(--line); border-top: 2px solid var(--cyan); border-radius: 10px; background: rgba(7,20,31,.6); padding: 26px 20px; text-align: center; }
        .sl-empty-panel p.sl-kicker { margin: 0 0 6px; font-size: 10.5px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); }
        .sl-empty-panel h2 { margin: 0 0 6px; font-size: 15px; letter-spacing: .04em; text-transform: uppercase; font-weight: 800; }
        .sl-empty-panel p { margin: 0; color: var(--muted); font-size: 12.5px; }

        /* social */
        .sl-social-layout { display: grid; grid-template-columns: minmax(0,1fr) minmax(240px,300px); gap: 14px; }
        .sl-social-feed { display: grid; gap: 10px; }
        .sl-social-post { border: 1px solid var(--line); border-radius: 8px; background: var(--panel-2); padding: 10px 12px; text-align: left; color: inherit; cursor: pointer; }
        .sl-social-post:hover { border-color: var(--line-strong); }
        .sl-social-post__head { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; margin-bottom: 6px; }
        .sl-social-post__head strong { font-size: 12.5px; }
        .sl-social-post__head span { color: var(--muted); font-size: 11px; font-weight: 700; }
        .sl-social-post__head em { margin-left: auto; font-style: normal; font-size: 10.5px; color: var(--muted); }
        .sl-social-post p { margin: 0; font-size: 12.5px; line-height: 1.4; }
        .sl-social-post__related { margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(156,218,236,.1); font-size: 10.5px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--gold); }

        /* press room */
        .sl-press-card { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); padding: 14px 16px; margin-bottom: 12px; }
        .sl-press-card__head { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin-bottom: 8px; }
        .sl-press-card__head strong { font-size: 14px; }
        .sl-press-card__head span { color: var(--muted); font-size: 11px; font-weight: 800; }
        .sl-press-question { border-top: 1px solid var(--line); padding-top: 10px; margin-top: 10px; }
        .sl-press-question p { margin: 0 0 8px; font-size: 12.5px; line-height: 1.4; }
        .sl-press-question em { display: block; margin-bottom: 8px; font-style: normal; font-size: 11px; font-weight: 800; color: var(--cyan); }

        /* archive */
        .sl-era-card { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); padding: 14px 16px; margin-bottom: 12px; }
        .sl-era-card__head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
        .sl-era-card__head h3 { margin: 0; font-size: 14px; }
        .sl-era-themes { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
        .sl-era-themes span { font-size: 10px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; border: 1px solid rgba(156,218,236,.18); border-radius: 4px; padding: 3px 7px; color: var(--muted); }
        .sl-era-stories { display: grid; gap: 6px; }
        .sl-era-stories button { text-align: left; border: 1px solid transparent; border-radius: 6px; background: rgba(255,255,255,.02); padding: 8px 10px; cursor: pointer; color: var(--text); }
        .sl-era-stories button:hover { border-color: var(--line); background: rgba(19,216,231,.06); }
        .sl-era-stories strong { display: block; font-size: 12.5px; margin-bottom: 2px; }
        .sl-era-stories em { font-style: normal; font-size: 10.5px; color: var(--muted); }

        /* breaking overlay */
        .sl-breaking-overlay { position: fixed; inset: 0; z-index: 12000; display: grid; place-items: start center; padding: 24px 16px; background: rgba(2,8,14,.72); backdrop-filter: blur(4px); cursor: pointer; }
        .sl-breaking-card { width: min(560px,100%); border: 1px solid rgba(255,96,109,.55); border-top: 3px solid #ff606d; background: linear-gradient(180deg, rgba(40,8,12,.98), rgba(9,25,38,.98)); padding: 16px 18px; box-shadow: 0 18px 48px rgba(0,0,0,.45); cursor: default; }
        .sl-breaking-card__kicker { margin: 0 0 6px; font-size: 11px; font-weight: 1000; letter-spacing: .14em; text-transform: uppercase; color: #ff606d; }
        .sl-breaking-card h2 { margin: 0 0 8px; font-size: 1.05rem; line-height: 1.35; }
        .sl-breaking-card p { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }
        .sl-breaking-card__actions { display: flex; gap: 8px; flex-wrap: wrap; }
        .sl-breaking-card__actions button { border: 1px solid var(--line); border-radius: 6px; background: rgba(19,216,231,.1); color: var(--text); padding: 8px 12px; font-size: 12px; font-weight: 800; cursor: pointer; }
        .sl-breaking-card__actions button.is-primary { border-color: var(--line-strong); background: rgba(19,216,231,.22); color: var(--cyan); }

        .sl-market-banner { margin: 0; padding: 8px 12px; border: 1px solid rgba(233,168,60,.25); border-radius: 6px; background: rgba(233,168,60,.08); font-size: 12px; font-weight: 800; color: #ffc98a; }

        .sl-insiders-layout { display: grid; grid-template-columns: minmax(0,1fr) minmax(260px,340px); gap: 14px; }
        .sl-insider-feed { display: grid; gap: 8px; }
        .sl-insider-row { text-align: left; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-2); padding: 10px 12px; color: inherit; cursor: pointer; }
        .sl-insider-row:hover { border-color: var(--line-strong); }
        .sl-insider-row__head { display: flex; gap: 8px; align-items: baseline; justify-content: space-between; margin-bottom: 4px; }
        .sl-insider-row__head strong { font-size: 13px; }
        .sl-insider-row__head em { font-style: normal; font-size: 10.5px; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; color: var(--gold); }
        .sl-insider-row p { margin: 0 0 8px; font-size: 12.5px; color: var(--muted); line-height: 1.4; }
        .sl-insider-row__meta { display: flex; flex-wrap: wrap; gap: 8px; font-size: 10.5px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--cyan); }
        .sl-insider-rail { display: grid; gap: 10px; align-content: start; }
        .sl-insider-rail h3 { margin: 0; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
        .sl-dossier { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 10px 12px; }
        .sl-dossier__head { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 4px; }
        .sl-dossier__head strong { font-size: 13px; }
        .sl-dossier__head span, .sl-dossier__ident, .sl-dossier p { margin: 0; font-size: 11.5px; color: var(--muted); }
        .sl-dossier h4 { margin: 8px 0 4px; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--cyan); }
        .sl-dossier__tags { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
        .sl-dossier__tags span { font-size: 10px; font-weight: 900; letter-spacing: .04em; text-transform: uppercase; border: 1px solid rgba(156,218,236,.18); border-radius: 4px; padding: 2px 6px; }
        .sl-dossier__grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

        @media (prefers-reduced-motion: reduce) { .nhlcal-sl-root * { transition: none !important; animation: none !important; } }
      `}</style>

      <div className="sl-app">
        {activeBreaking ? (
          <div className="sl-breaking-overlay" role="dialog" aria-label="Breaking news" onClick={() => dismissBreakingAlerts(pendingBreaking)}>
            <div className="sl-breaking-card" onClick={(e) => e.stopPropagation()}>
              <p className="sl-breaking-card__kicker">
                Breaking · {str(activeBreaking.level || "major").replace(/_/g, " ")}
                {pendingBreaking.length > 1 ? ` · ${pendingBreaking.length} alerts` : ""}
              </p>
              <h2>{str(activeBreaking.headline || "Major league development")}</h2>
              {activeBreaking.summary ? <p>{activeBreaking.summary}</p> : null}
              <div className="sl-breaking-card__actions">
                <button
                  type="button"
                  className="is-primary"
                  onClick={() => {
                    const storyKey = str(activeBreaking.storyline_id || "");
                    if (storyKey) {
                      const match = stories.find((s) => str(s.storylineId) === storyKey || str(s.id) === storyKey);
                      if (match) openStory(match.id);
                    }
                    setDepartment("front_page");
                    dismissBreakingAlerts(pendingBreaking);
                  }}
                >
                  Open story
                </button>
                <button type="button" onClick={() => dismissBreakingAlerts(pendingBreaking)}>
                  Dismiss{pendingBreaking.length > 1 ? " all" : ""}
                </button>
              </div>
            </div>
          </div>
        ) : null}

        <header className="sl-topbar">
          <div>
            <p className="sl-eyebrow">Franchise Newsroom</p>
            <h1>Storylines</h1>
            <p className="sl-topbar-sub">
              {calendarLabel(franchiseState)} · {teamLabel(franchiseState)} · {statusLine}
            </p>
          </div>
          <nav className="sl-topbar-nav" aria-label="Navigation">
            <button type="button" onClick={() => setScreen?.(SCREENS.CALENDAR)}>Calendar</button>
            <button type="button" onClick={() => setScreen?.(SCREENS.HUB)}>Hub</button>
          </nav>
        </header>

        <nav className="sl-departments" aria-label="Media departments">
          {DEPARTMENTS.map((d) => {
            const alert =
              (d.id === "press_room" && pressQueue.length > 0)
              || (d.id === "archive" && narrativeEras.length > 0)
              || (d.id === "insiders" && (insiderItems.length > 0 || playerDossiers.length > 0));
            return (
              <button
                key={d.id}
                type="button"
                className={`${department === d.id ? "is-active" : ""}${alert ? " has-alert" : ""}`}
                onClick={() => setDepartment(d.id)}
              >
                {d.label}
              </button>
            );
          })}
        </nav>

        {userMarket?.label && (department === "front_page") ? (
          <p className="sl-market-banner">
            {userMarket.label} market · {userMarket.descriptor || userMarket.tone || "High scrutiny"}
            {userMarket.pressure_mult ? ` · pressure ×${Number(userMarket.pressure_mult).toFixed(2)}` : ""}
          </p>
        ) : null}

        {!hasBackend ? (
          <div className="sl-empty-panel">
            <p className="sl-kicker">League Wire · Idle</p>
            <h2>No Coverage Yet</h2>
            <p>Advance the calendar to populate the newsroom from backend storylines.</p>
          </div>
        ) : department === "social" ? (
          <div className="sl-social-layout">
            <div className="sl-social-feed">
              {socialPosts.length ? (
                socialPosts.map((post) => (
                  <button key={post.id} type="button" className="sl-social-post" onClick={() => { if (post.storyId) { openStory(post.storyId); setDepartment("front_page"); } }}>
                    <div className="sl-social-post__head">
                      <strong>
                        {post.name}
                        {post.verified ? " ✓" : ""}
                        {post.isAgent ? " · Agent" : ""}
                      </strong>
                      <span>{post.handle}</span>
                      <em>{post.age}</em>
                    </div>
                    <p>{post.text}</p>
                    {post.related ? <div className="sl-social-post__related">Related · {post.related}</div> : null}
                    {post.cred ? <div className="sl-social-post__related">{post.cred}</div> : null}
                    {post.likes != null ? (
                      <div className="sl-social-post__related">
                        {Number(post.replies || 0).toLocaleString()} replies · {Number(post.reposts || 0).toLocaleString()} reposts · {Number(post.likes || 0).toLocaleString()} likes
                      </div>
                    ) : null}
                  </button>
                ))
              ) : (
                <div className="sl-empty-panel">
                  <p className="sl-kicker">Social desk · Quiet</p>
                  <h2>No posts yet</h2>
                  <p>Hockey Twitter wakes up as storylines generate across the league.</p>
                </div>
              )}
            </div>
            <aside className="sl-impact-panel">
              <h3>Trending</h3>
              <div className="sl-effects-grid">
                {stories
                  .filter((s) => Number(s.heat) > 0)
                  .sort((a, b) => Number(b.heat) - Number(a.heat))
                  .slice(0, 6)
                  .map((s, i) => (
                    <div key={s.id} className="sl-effect-row">
                      <span>{i + 1}. {s.playerName || s.teamName || s.headline?.slice(0, 26)}</span>
                      <strong>{heatLabel(s.heat)}</strong>
                    </div>
                  ))}
              </div>
            </aside>
          </div>
        ) : department === "insiders" ? (
          <div className="sl-insiders-layout">
            <div className="sl-insider-feed">
              {insiderItems.length ? (
                insiderItems.slice().reverse().slice(0, 48).map((item, idx) => {
                  const sid = str(item.storyline_id || item.world_event_id || idx);
                  const match = stories.find((s) => str(s.storylineId) === sid || str(s.id) === sid);
                  return (
                    <button
                      key={sid}
                      type="button"
                      className="sl-insider-row"
                      onClick={() => {
                        if (match) {
                          openStory(match.id);
                          setDepartment("front_page");
                        }
                      }}
                    >
                      <div className="sl-insider-row__head">
                        <strong>{str(item.headline || match?.headline || "Desk note")}</strong>
                        <em>{knowledgeLevelLabel(item.public_knowledge_level)}</em>
                      </div>
                      <p>{str(item.summary || match?.summary || "")}</p>
                      <div className="sl-insider-row__meta">
                        <span>{str(item.reporter_name || item.source_label || "Insider")}</span>
                        {item.outlet_name ? <span>{item.outlet_name}</span> : null}
                        <span>{str(item.knowledge_type || "report").replace(/_/g, " ")}</span>
                        {item.player_name ? <span>{item.player_name}</span> : null}
                        {item.calendar_iso ? <span>{item.calendar_iso}</span> : null}
                      </div>
                    </button>
                  );
                })
              ) : (
                <div className="sl-empty-panel">
                  <p className="sl-kicker">Insiders · Quiet</p>
                  <h2>No private layers yet</h2>
                  <p>Rumors, claims, and confirmed facts land here as the knowledge graph fills in.</p>
                </div>
              )}
            </div>
            <aside className="sl-insider-rail">
              <h3>Beat desks</h3>
              <div className="sl-effects-grid">
                {beatWriters.slice(0, 10).map((writer) => (
                  <div key={str(writer.id || writer.name)} className="sl-effect-row">
                    <span>{str(writer.name)}</span>
                    <strong>{str(writer.specialty || writer.role || writer.outlet)}</strong>
                  </div>
                ))}
              </div>
              <h3>Player dossiers</h3>
              {playerDossiers.length ? (
                playerDossiers.slice(0, 8).map((dossier) => (
                  <DossierCard key={str(dossier.player_id || dossier.player_name)} dossier={dossier} />
                ))
              ) : (
                <p className="sl-decision-empty">Roster beings publish here after the next calendar tick.</p>
              )}
            </aside>
          </div>
        ) : department === "press_room" ? (
          <div>
            {pressQueue.length ? (
              pressQueue.map((press) => (
                <article key={str(press.id)} className="sl-press-card">
                  <div className="sl-press-card__head">
                    <strong>{str(press.headline || "Media availability scheduled")}</strong>
                    <span>{heatLabel(press.heat) || "Press heat rising"}</span>
                    {press.player_name ? <span>{press.player_name}</span> : null}
                  </div>
                  {press.summary ? <p>{press.summary}</p> : null}
                  {asArray(press.questions).map((q) => (
                    <div key={str(q.id)} className="sl-press-question">
                      <em>{str(q.reporter_name || "Reporter")}{q.outlet ? ` · ${q.outlet}` : ""}</em>
                      <p>{str(q.question || "")}</p>
                      <div className="sl-decisions">
                        {asArray(q.responses).map((resp) => {
                          const sid = str(press.storyline_id);
                          const choiceId = `${str(q.id)}:${str(resp.id)}`;
                          const busy = busyChoice === `${sid}:${choiceId}`;
                          return (
                            <button
                              key={resp.id}
                              type="button"
                              className="sl-decision-btn"
                              disabled={Boolean(busyChoice)}
                              onClick={() => handlePressResponse(press, str(q.id), str(resp.id))}
                            >
                              <span className="sl-decision-btn__dot" />
                              <span>
                                <strong>{resp.label}</strong>
                                {resp.description ? <span>{resp.description}</span> : null}
                                {busy ? <em>Answering…</em> : null}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </article>
              ))
            ) : (
              <div className="sl-empty-panel">
                <p className="sl-kicker">Press room · Clear</p>
                <h2>No scheduled availability</h2>
                <p>When heat builds around your team, reporters will queue questions for your next media session.</p>
              </div>
            )}
          </div>
        ) : department === "archive" ? (
          <div>
            {narrativeEras.length ? (
              narrativeEras.slice().reverse().map((era) => (
                <article key={str(era.season)} className="sl-era-card">
                  <div className="sl-era-card__head">
                    <h3>{str(era.label || `Season ${era.season}`)}</h3>
                    <span>{Number(era.story_count || 0)} archived beats</span>
                  </div>
                  {asArray(era.themes).length ? (
                    <div className="sl-era-themes">
                      {era.themes.map((theme) => (
                        <span key={theme}>{theme}</span>
                      ))}
                    </div>
                  ) : null}
                  <div className="sl-era-stories">
                    {asArray(era.top_stories).map((story, idx) => (
                      <button
                        key={str(story.storyline_id || story.headline || idx)}
                        type="button"
                        onClick={() => {
                          const match = stories.find(
                            (s) => str(s.storylineId) === str(story.storyline_id) || s.headline === story.headline
                          );
                          if (match) {
                            openStory(match.id);
                            setDepartment("front_page");
                          }
                        }}
                      >
                        <strong>{str(story.headline || "Archived beat")}</strong>
                        <em>
                          {str(story.category || "storyline")}
                          {story.heat != null ? ` · heat ${story.heat}` : ""}
                          {story.calendar_iso ? ` · ${story.calendar_iso}` : ""}
                        </em>
                      </button>
                    ))}
                  </div>
                </article>
              ))
            ) : narrativeArchive.length ? (
              <article className="sl-era-card">
                <div className="sl-era-card__head">
                  <h3>League archive</h3>
                  <span>{narrativeArchive.length} beats on file</span>
                </div>
                <div className="sl-era-stories">
                  {narrativeArchive.slice().reverse().slice(0, 24).map((story, idx) => (
                    <button
                      key={str(story.storyline_id || story.headline || idx)}
                      type="button"
                      onClick={() => {
                        const match = stories.find((s) => str(s.storylineId) === str(story.storyline_id));
                        if (match) {
                          openStory(match.id);
                          setDepartment("front_page");
                        }
                      }}
                    >
                      <strong>{str(story.headline || "Archived beat")}</strong>
                      <em>{str(story.calendar_iso || story.season || "—")}</em>
                    </button>
                  ))}
                </div>
              </article>
            ) : (
              <div className="sl-empty-panel">
                <p className="sl-kicker">Archive · Empty</p>
                <h2>No sealed eras yet</h2>
                <p>Completed seasons are preserved here — themes, top stories, and defining beats.</p>
              </div>
            )}
          </div>
        ) : stories.length === 0 ? (
          <div className="sl-empty-panel">
            <p className="sl-kicker">League Wire · Idle</p>
            <h2>Wire Standing By</h2>
            <p>No active storylines on file. Coverage will appear as the season generates league beats.</p>
          </div>
        ) : (
          <>
            {topPending ? (
              <button type="button" className="sl-frontoffice-alert" onClick={() => openStory(topPending.id)}>
                <span>Front Office Alert</span>
                <strong>{topPending.headline}</strong>
                <em>Review decision →</em>
              </button>
            ) : null}

            <div className="sl-filterbar">
              <div className="sl-filter-chips">
                {FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`sl-chip ${filter === f.id ? "is-active" : ""}`}
                    onClick={() => setFilter(f.id)}
                  >
                    {f.label}
                    <span className="sl-chip-count">({filterCounts[f.id] ?? 0})</span>
                  </button>
                ))}
              </div>
              <div className="sl-filter-tools">
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
                      <option key={o.id} value={o.id}>{o.label}</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            <div className="sl-stream">
              <div className="sl-stream-head">
                <span>Story Feed</span>
                <span>{filtered.length} active</span>
              </div>
              {filtered.length === 0 ? (
                <div className="sl-feed-empty">{filterEmptyMsg || "No matching stories."}</div>
              ) : (
                filtered.map((s) => {
                  const isOpen = expandedId === s.id;
                  const socialCount = socialCountByStory.get(s.storylineId) || socialCountByStory.get(s.id) || 0;
                  return (
                    <article key={s.id} className={`sl-row${isOpen ? " is-open" : ""}`}>
                      <button type="button" className="sl-row__header" onClick={() => toggleRow(s.id)} aria-expanded={isOpen}>
                        <span className={`sl-row__heatbar sl-row__heatbar--${heatTier(s.heat)}`} aria-hidden="true" />
                        <span className="sl-row__main">
                          <span className="sl-row__topline">
                            <span className="sl-row__category" style={{ color: categoryMeta(s).accent }}>{categoryMeta(s).label}</span>
                            {s.requiresAction ? <span className="sl-row__decision">Decision</span> : null}
                            <StatusPill story={s} />
                          </span>
                          <h4 className="sl-row__headline">{s.headline}</h4>
                          <span className="sl-row__subline">
                            {heatLabel(s.heat) ? <span>Heat: {Math.round(Number(s.heat) || 0)}</span> : null}
                            {socialCount ? <span className="sl-row__pulse">↙ {formatCount(socialCount)} social posts</span> : null}
                          </span>
                        </span>
                        <span className="sl-row__meta">
                          <span className="sl-row__date">{s.ageLabel || "—"}</span>
                          <span className="sl-row__chevron">{isOpen ? "▼" : "▶"}</span>
                        </span>
                      </button>

                      {isOpen && selected ? (
                        <div className="sl-row__body">
                          <div className="sl-row__layout">
                            <div className="sl-row__detail">
                              <div className="sl-detail-head">
                                <ScoreBadge score={storyScore(selected)} size="lg" />
                                <div className="sl-detail-headmeta">
                                  <div className="sl-detail-source">
                                    {selected.reporterName || selected.sourceLabel ? (
                                      <span>Source: {selected.reporterName || selected.sourceLabel}{selected.outletName ? ` (${selected.outletName})` : ""}</span>
                                    ) : null}
                                    {credibilityLabel(selected.credibility) ? <span>Credibility: {credibilityLabel(selected.credibility)}</span> : null}
                                    <span>{selected.ageLabel || "—"}</span>
                                  </div>
                                  {selected.summary ? <p className="sl-detail-summary">{selected.summary}</p> : null}
                                </div>
                              </div>

                              <TeamOrPlayerIdentity story={selected} />
                              {selectedDossier ? <DossierCard dossier={selectedDossier} /> : null}
                              {isRumourStory(selected) ? <TradeSwap story={selected} /> : null}
                              {selected.description && selected.description !== selected.summary ? (
                                <p className="sl-detail-summary">{selected.description}</p>
                              ) : null}
                              <ConductChannels story={selected} />

                              <DevelopmentTimeline beats={arcTimeline} fallbackStory={selected} />

                              <nav className="sl-tabs">
                                {DETAIL_TABS.map((t) => (
                                  <button key={t.id} type="button" className={activeTab === t.id ? "is-active" : ""} onClick={() => setActiveTab(t.id)}>
                                    {t.label}
                                  </button>
                                ))}
                              </nav>

                              <div className="sl-tab-panel">
                                {activeTab === "details" ? (
                                  <div className="sl-detail-grid">
                                    <div className="sl-detail-block">
                                      <h4>Information</h4>
                                      {infoRows.length ? (
                                        infoRows.map(([label, val]) => (
                                          <div key={label} className="sl-info-row">
                                            <span>{label}</span>
                                            <span>{val}</span>
                                          </div>
                                        ))
                                      ) : (
                                        <p style={{ color: "var(--muted)", fontSize: 12 }}>No sourcing details on file.</p>
                                      )}
                                      {Object.keys(selected.evidence || {}).length ? (
                                        <div className="sl-numbers">
                                          {Object.entries(selected.evidence).slice(0, 4).map(([k, v]) => (
                                            <div key={k} className="sl-num" title={`${formatEffectLabel(k)}: ${v}`}>
                                              <strong>{String(v)}</strong>
                                              <span>{formatEffectLabel(k)}</span>
                                            </div>
                                          ))}
                                        </div>
                                      ) : null}
                                    </div>

                                    <div className="sl-detail-block">
                                      <h4>Parties Involved</h4>
                                      {parties.length ? (
                                        parties.map((p) => (
                                          <div key={p.label} className="sl-party-row">
                                            <span>{p.label}</span>
                                            <span>{p.name}</span>
                                          </div>
                                        ))
                                      ) : (
                                        <p style={{ color: "var(--muted)", fontSize: 12 }}>No named parties on file.</p>
                                      )}
                                    </div>

                                    <div className="sl-detail-block">
                                      <h4>Key Factors</h4>
                                      <ul className="sl-key-factors">
                                        {keyFactors.map((f, i) => (
                                          <li key={i}>{f}</li>
                                        ))}
                                      </ul>
                                      {(selected.effectSummary || Object.keys(selected.effects || {}).length) ? (
                                        <div style={{ marginTop: 10 }}>
                                          {selected.effectSummary ? <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>{selected.effectSummary}</p> : null}
                                        </div>
                                      ) : null}
                                    </div>
                                  </div>
                                ) : null}

                                {activeTab === "related" ? (
                                  <div className="sl-related-list">
                                    {relatedStories.length ? (
                                      relatedStories.slice(0, 8).map((r) => (
                                        <button key={r.id} type="button" onClick={() => openStory(r.id)}>
                                          <span style={{ color: categoryMeta(r).accent }}>{categoryMeta(r).label}</span>
                                          <strong>{r.headline}</strong>
                                          <em>{r.ageLabel || "—"}</em>
                                        </button>
                                      ))
                                    ) : (
                                      <p style={{ color: "var(--muted)", fontSize: 12 }}>No related coverage yet.</p>
                                    )}
                                  </div>
                                ) : null}

                                {activeTab === "rumors" ? (
                                  <div className="sl-rumor-list">
                                    {leagueRumours.length ? (
                                      leagueRumours.map((r) => (
                                        <button key={r.id} type="button" onClick={() => openStory(r.id)}>
                                          <span>{r.playerName || r.teamName || "League"}</span>
                                          <strong>{r.headline}</strong>
                                          <em>
                                            {heatLabel(r.heat) ? `Heat: ${heatLabel(r.heat)}` : ""}
                                            {credibilityLabel(r.credibility) ? ` · ${credibilityLabel(r.credibility)}` : ""}
                                          </em>
                                        </button>
                                      ))
                                    ) : (
                                      <p style={{ color: "var(--muted)", fontSize: 12 }}>Trade wire is quiet.</p>
                                    )}
                                  </div>
                                ) : null}

                                {activeTab === "history" ? (
                                  <ul className="sl-history-list">
                                    {arcTimeline.length ? (
                                      arcTimeline.map((beat) => (
                                        <li key={beat.id}>
                                          <time>{beat.date || "—"}</time>
                                          <div>
                                            <strong>{beat.headline}</strong>
                                            {beat.summary ? <p>{beat.summary}</p> : null}
                                          </div>
                                        </li>
                                      ))
                                    ) : (
                                      <p style={{ color: "var(--muted)", fontSize: 12 }}>No prior beats on file for this story.</p>
                                    )}
                                    {selected.repeatCount > 0 ? (
                                      <p style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>
                                        Beat #{selected.repeatCount + 1}
                                        {selected.escalatedFrom ? ` · escalated from ${selected.escalatedFrom}` : ""}
                                      </p>
                                    ) : null}
                                  </ul>
                                ) : null}
                              </div>

                              <section className="sl-detail-block" style={{ marginTop: 14, borderTop: "1px solid var(--line)", paddingTop: 12 }}>
                                <h4>What Comes Next</h4>
                                <p style={{ fontSize: 12.5, color: "rgba(233,247,251,.85)" }}>{deriveFollowUp(selected)}</p>
                              </section>

                              <div className="sl-detail-footer">
                                <span>Last updated: {selected.date || selected.ageLabel || "—"}</span>
                                <span>Story ID: {selected.storylineId || selected.id}</span>
                              </div>
                            </div>

                            <div className="sl-row__impact">
                              <div className="sl-impact-panel">
                                <h3 className="is-title">Story Impact</h3>
                                {userOrg ? (
                                  <>
                                    <h3 style={{ marginTop: 4 }}>Organizational Pressure</h3>
                                    <OrgPressureBars org={userOrg} />
                                  </>
                                ) : (
                                  <p style={{ fontSize: 12, color: "var(--muted)" }}>No organizational pressure data on file.</p>
                                )}
                              </div>

                              {selected && Object.keys(selected.effects || {}).length ? (
                                <div className="sl-impact-panel">
                                  <h3>Potential Effects</h3>
                                  <div className="sl-effects-grid">
                                    {Object.entries(selected.effects).slice(0, 6).map(([k, v]) => (
                                      <div key={k} className={`sl-effect-row ${effectPillClass(v)}`}>
                                        <span>{formatEffectLabel(k)}</span>
                                        <strong>{Number(v) > 0 ? "+" : ""}{String(v)}</strong>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}

                              <div className="sl-impact-panel">
                                <h3>GM Decisions</h3>
                                {choiceOptions.length ? (
                                  <div className="sl-decisions">
                                    {choiceOptions.map((opt) => {
                                      const busy = busyChoice === `${selected.storylineId}:${opt.id}`;
                                      return (
                                        <button
                                          key={opt.id}
                                          type="button"
                                          className="sl-decision-btn"
                                          disabled={Boolean(busyChoice)}
                                          onClick={() => handleResolve(selectedChoice?.storyline_id || selected.storylineId, opt.id)}
                                        >
                                          <span className="sl-decision-btn__dot" />
                                          <span>
                                            <strong>{opt.label}</strong>
                                            {opt.effect_summary ? <span>{opt.effect_summary}</span> : null}
                                            {busy ? <em>Applying…</em> : null}
                                          </span>
                                        </button>
                                      );
                                    })}
                                  </div>
                                ) : (
                                  <p className="sl-decision-empty">No action needed on this story.</p>
                                )}
                              </div>

                              {selected?.gmKnowsMore || knowledgeLevelLabel(selected?.publicKnowledgeLevel) ? (
                                <div className="sl-impact-panel">
                                  <h3>Knowledge Layers</h3>
                                  {selected.gmKnowsMore ? (
                                    <p style={{ fontSize: 12, color: "#8ef0b8", marginBottom: 6 }}>You know more than the public sees.</p>
                                  ) : null}
                                  {knowledgeLevelLabel(selected.publicKnowledgeLevel) ? (
                                    <p style={{ fontSize: 12, color: "var(--muted)" }}>Public knowledge: {knowledgeLevelLabel(selected.publicKnowledgeLevel)}</p>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </article>
                  );
                })
              )}
            </div>
          </>
        )}

        {process.env.NODE_ENV === "development" && !hasBackend ? (
          <details className="sl-detail-block" style={{ marginTop: 12 }}>
            <summary>Storyline debug (dev)</summary>
            <pre style={{ fontSize: 11, overflow: "auto", maxHeight: 240 }}>
              {JSON.stringify({ has_storyline_events: Array.isArray(franchiseState?.storyline_events), state_keys: Object.keys(franchiseState || {}) }, null, 2)}
            </pre>
          </details>
        ) : null}
        {process.env.NODE_ENV === "development" && franchiseState?.storyline_debug ? (
          <details className="sl-detail-block" style={{ marginTop: 12 }}>
            <summary>Storyline debug (dev)</summary>
            <pre style={{ fontSize: 11, overflow: "auto", maxHeight: 240 }}>{JSON.stringify(franchiseState.storyline_debug, null, 2)}</pre>
          </details>
        ) : null}
      </div>
    </div>
  );
}