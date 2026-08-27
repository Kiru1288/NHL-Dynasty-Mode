import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useGameUI } from "../../game/GameUIContext";
import {
  acceptEntryDraftTrade,
  completeEntryDraft,
  simEntryDraftRound,
  simEntryDraftToUserPick,
  startEntryDraft,
  submitCpuDraftPick,
  submitDraftPick,
} from "../../services/franchiseService";
import { getTeamAbbreviation, getTeamLogoSrc } from "../../utils/teamLogos";
import { flagApiUrl, resolveCountryCode } from "../../utils/countryFlags";
import FanReactionFeed from "../../components/franchise/social/FanReactionFeed";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { buildDraftFanTweets, buildDraftPickReactionTweet } from "../awardsNight/awardHelpers";
import {
  buildCinematicCss,
  formatPick,
  getPlayerName,
  getPlayerPosition,
  pickFranchiseData,
  safeArray,
} from "../shared/eventHelpers";
import "./EntryDraft.css";

function ProspectFlag({ country, code, size = 16 }) {
  const src = flagApiUrl(code || country, 32);
  const iso = resolveCountryCode(code || country);
  if (!src) {
    return <span className={`${PREFIX}-flag ${PREFIX}-flag--empty`} aria-hidden="true" />;
  }
  return (
    <img
      className={`${PREFIX}-flag`}
      src={src}
      alt={iso ? `${iso} flag` : "Country flag"}
      width={size}
      height={Math.round(size * 0.75)}
      loading="lazy"
      onError={(e) => {
        e.currentTarget.style.visibility = "hidden";
      }}
    />
  );
}

const PREFIX = "edraft";
const ROUNDS = 7;

const LOADING_COPY = {
  start: "Preparing…",
  cpu: "Advancing pick…",
  simNext: "Advancing pick…",
  simUser: "Simulating to your pick…",
  simRound: "Simulating round…",
  submit: "Submitting selection…",
  complete: "Completing draft…",
};

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function draftYearLabel(draft) {
  return draft?.draft_year || draft?.draftYear || "";
}

function getId(p) {
  return p?.key || p?.prospect_id || p?.player_id || p?.id || `${getPlayerName(p)}-${p?.rank}`;
}

function num(v, fallback = null) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function getRank(p) {
  const n = num(p?.rank ?? p?.final_rank ?? p?.public_rank ?? p?.draft_rank);
  return n;
}

function getPublicRankAtPick(p) {
  return num(p?.final_rank ?? p?.rank ?? p?.public_rank);
}

function getPreseasonRank(p) {
  return num(p?.preseason_rank);
}

function getTeamRank(p) {
  return num(p?.team_board_rank ?? p?.internal_rank ?? p?.team_rank);
}

function getPickValueDelta(p) {
  const pub = getPublicRankAtPick(p);
  const slot = num(p?.overall_pick);
  if (pub == null || slot == null) return null;
  // pickNumber - publicRank: positive = fell (value/steal), negative = early/reach
  return slot - pub;
}

function describePickMovement(p) {
  const delta = getPickValueDelta(p);
  if (delta == null) return null;
  const abs = Math.abs(delta);
  if (delta >= 12) return { label: `Fell ${abs} spots`, short: `↓${abs}`, tone: "steal", tag: "Steal" };
  if (delta >= 5) return { label: `Slid ${abs} spots`, short: `↓${abs}`, tone: "safe", tag: "Value" };
  if (delta <= -15) return { label: `Reached ${abs} early`, short: `↑${abs}`, tone: "reach", tag: "Reach" };
  if (delta <= -8) return { label: `Taken ${abs} early`, short: `↑${abs}`, tone: "early", tag: "Early" };
  if (Math.abs(delta) <= 4) return { label: "On board", short: "On board", tone: "default", tag: null };
  if (delta > 0) return { label: `Fell ${abs}`, short: `↓${abs}`, tone: "safe", tag: "Value" };
  return { label: `Early by ${abs}`, short: `↑${abs}`, tone: "early", tag: "Early" };
}

function getPickDisplayTag(p) {
  // Prefer backend public-board selection_label snapshot — never invent from user board.
  const backendLabel = String(p?.selection_label || p?.pick_classification || "").trim();
  const tags = safeArray(p?.pick_tags);
  const labelMap = {
    Steal: "steal",
    Value: "safe",
    Expected: "default",
    Early: "early",
    Reach: "reach",
    "Off Board": "offboard",
    "Off the Board": "offboard",
    BPA: "bpa",
    "Need Fit": "need",
    "Need Pick": "need",
  };
  if (backendLabel && labelMap[backendLabel] !== undefined) {
    return { tag: backendLabel === "Off the Board" ? "Off Board" : backendLabel, tone: labelMap[backendLabel] };
  }
  if (p?.was_off_board || tags.includes("Off Board") || tags.includes("Off the Board")) {
    return { tag: "Off Board", tone: "offboard" };
  }
  if (p?.was_steal || tags.includes("Steal")) return { tag: "Steal", tone: "steal" };
  if (p?.was_value || tags.includes("Value")) return { tag: "Value", tone: "safe" };
  if (p?.was_reach || tags.includes("Reach")) return { tag: "Reach", tone: "reach" };
  if (p?.was_early || tags.includes("Early")) return { tag: "Early", tone: "early" };
  if (p?.was_expected || tags.includes("Expected")) return { tag: "Expected", tone: "default" };
  if (p?.was_bpa || tags.includes("BPA")) return { tag: "BPA", tone: "bpa" };
  if (p?.was_team_need || tags.includes("Need Fit") || tags.includes("Need Pick")) {
    return { tag: "Need", tone: "need" };
  }
  // Fallback only when backend sent no label — use public delta, never user board.
  const movement = describePickMovement(p);
  if (movement?.tag) {
    const tone =
      movement.tag === "Steal" ? "steal" :
      movement.tag === "Reach" ? "reach" :
      movement.tag === "Early" ? "early" :
      movement.tag === "Value" ? "safe" : "default";
    return { tag: movement.tag, tone };
  }
  return { tag: null, tone: "default" };
}

function enrichPickFromBoard(pick, boardEntries) {
  if (!pick) return null;
  const pid = String(pick.prospect_id || pick.key || "");
  const fromBoard = safeArray(boardEntries).find((e) => String(e.key || e.prospect_id) === pid);
  return { ...(fromBoard || {}), ...pick };
}

function getRisk(p) {
  const v = p?.risk ?? p?.risk_level ?? p?.draft_risk;
  if (v == null || v === "") return null;
  return String(v);
}

function getConfidence(p) {
  const raw = p?.scouting_confidence ?? p?.confidence ?? p?.scout_confidence;
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function getConfidenceRange(p) {
  const lo = num(p?.confidence_low ?? p?.scouting_confidence_low);
  const hi = num(p?.confidence_high ?? p?.scouting_confidence_high);
  if (lo != null && hi != null) return [lo, hi];
  return null;
}

function formatConfidence(p) {
  const range = getConfidenceRange(p);
  if (range) return `${Math.round(range[0])}–${Math.round(range[1])}%`;
  const c = getConfidence(p);
  if (c == null) return null;
  return `${Math.round(c)}%`;
}

function getTeamFit(p) {
  const raw = p?.team_fit_score ?? p?.fit_score ?? p?.need_fit_score ?? p?.organizational_fit;
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function normalizeStockLabel(p) {
  const raw = p?.stock_label || p?.stock_movement_label || p?.stock;
  if (raw == null || raw === "") {
    const move = num(p?.stock_movement ?? p?.movement);
    if (move == null) return null;
    if (move > 0) return "Rising";
    if (move < 0) return "Falling";
    return "Stable";
  }
  const label = String(raw).toLowerCase();
  if (label.includes("rise") || label.includes("up") || label.includes("trend")) return "Rising";
  if (label.includes("fall") || label.includes("down")) return "Falling";
  if (label.includes("hold") || label.includes("stable") || label.includes("flat")) return "Stable";
  return "Stable";
}

function getDefiningTrait(p) {
  if (!p) return null;
  const dossier = p.dossier || {};
  const v =
    dossier?.player_comparison?.archetype ||
    p.projected_role ||
    p.expected_role ||
    p.rights_card?.expected_role ||
    p.scouting_trait ||
    p.player_type ||
    p.archetype ||
    p.style;
  return v ? String(v) : null;
}

function getScoutedPotentialLabel(p) {
  if (!p) return null;
  const d = p.dossier?.potential_range || p.potential_range;
  if (d && typeof d === "object" && !Array.isArray(d) && d.low != null && d.high != null) {
    return `${Math.round(d.low)}–${Math.round(d.high)}`;
  }
  const range = safeArray(p.scouted_potential_range || (Array.isArray(d) ? d : null));
  if (range.length >= 2 && Number.isFinite(Number(range[0])) && Number.isFinite(Number(range[1]))) {
    return `${Math.round(range[0])}–${Math.round(range[1])}`;
  }
  if (p.scouted_potential != null && p.scouted_potential !== "") return String(p.scouted_potential);
  if (p.potential_label) return String(p.potential_label);
  if (p.potential_tier) return String(p.potential_tier);
  return null;
}

function formatCurrentOvr(p) {
  if (!p) return null;
  if (p.ovr_revealed && p.true_ovr != null && Number.isFinite(Number(p.true_ovr))) {
    return String(Math.round(Number(p.true_ovr)));
  }
  const range = p.current_ovr_range;
  if (Array.isArray(range) && range.length >= 2) {
    return `${Math.round(range[0])}–${Math.round(range[1])}`;
  }
  if (p.scouted_overall_estimate != null) return `~${Math.round(Number(p.scouted_overall_estimate))}`;
  if (p.current_ovr_estimate != null) return `~${Math.round(Number(p.current_ovr_estimate))}`;
  return null;
}

function formatNhlReadiness(p) {
  // NHL Readiness = ability to contribute NHL games NOW (0–100 score or label).
  // Distinct from ETA (years until arrival).
  const ready = p?.nhl_readiness ?? p?.dossier?.nhl_readiness ?? p?.dossier?.readiness;
  if (ready != null && ready !== "") {
    if (typeof ready === "object") {
      if (ready.label) return String(ready.label);
      if (ready.score != null && Number.isFinite(Number(ready.score))) {
        const s = Number(ready.score);
        if (s >= 78) return "NHL Ready";
        if (s >= 68) return "Close";
        if (s >= 55) return "Developing";
        if (s >= 40) return "Long-term project";
        return "At Risk";
      }
    }
    const n = Number(ready);
    if (Number.isFinite(n)) {
      // 0–100 readiness score from backend development engine.
      if (n > 1.5 && n <= 100) {
        if (n >= 78) return "NHL Ready";
        if (n >= 68) return "Close";
        if (n >= 55) return "Developing";
        if (n >= 40) return "Long-term project";
        return "At Risk";
      }
      // Legacy: small integers mistakenly stored as years.
      if (n <= 0) return "NHL Ready";
      if (n <= 8) return `${Math.round(n)} Years Away`;
    }
    const s = String(ready);
    if (/ready|close|develop|project|risk/i.test(s)) return s;
  }
  const label = p?.dossier?.readinessLabel || p?.readinessLabel;
  if (label) return String(label);
  return null;
}

function formatNhlArrivalEta(p) {
  // Years until NHL arrival — not peak potential, not readiness score.
  const etaObj = p?.dossier?.eta;
  if (etaObj && typeof etaObj === "object") {
    if (etaObj.years === 0 || String(etaObj.label || "").toLowerCase() === "now") return "NHL Ready / Now";
    if (etaObj.years != null) return `${etaObj.years} Years Away`;
    if (etaObj.label) return String(etaObj.label);
  }
  const years = Number(p?.nhl_eta_years ?? p?.nhl_eta ?? p?.rights_card?.eta);
  if (Number.isFinite(years)) {
    if (years <= 0) return "NHL Ready / Now";
    return `${Math.round(years)} Years Away`;
  }
  if (p?.nhl_eta_label) return String(p.nhl_eta_label);
  return null;
}

function formatNhlPotentialEta(p) {
  // NHL ETA = how soon they reach their projected ceiling / potential.
  const peak = p?.potential_eta ?? p?.dossier?.potential_eta ?? p?.dossier?.peak_eta;
  if (peak != null && peak !== "") {
    if (typeof peak === "object") {
      if (peak.label) return String(peak.label);
      if (peak.years != null) return `${peak.years}y to peak`;
    }
    return String(peak);
  }
  const etaObj = p?.dossier?.eta;
  if (etaObj && typeof etaObj === "object" && etaObj.peak_years != null) {
    return `${etaObj.peak_years}y to peak`;
  }
  // Fall back: readiness years + 2–4 as a soft peak window when only one ETA exists.
  const readyYears = Number(
    (typeof p?.dossier?.eta === "object" ? p.dossier.eta.years : null) ??
      p?.nhl_eta_years ??
      p?.nhl_eta
  );
  if (Number.isFinite(readyYears) && readyYears >= 0) {
    const peakY = Math.max(readyYears + 2, Math.min(8, readyYears + 4));
    return `${peakY}y to peak`;
  }
  return getNhlEta(p);
}

function isPhilosophyNeedLabel(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("upside") || s.includes("philosophy") || s.includes("bpa")) return true;
  if (s.includes("high") && s.includes("swing")) return true;
  return false;
}

function AttrBar({ label, value, confidence, rangeLow, rangeHigh }) {
  if (value == null || !Number.isFinite(Number(value))) {
    if (rangeLow == null && rangeHigh == null) return null;
  }
  const conf = Number(confidence);
  const uncertain = Number.isFinite(conf) && conf < 45;
  const v = value != null && Number.isFinite(Number(value))
    ? Math.max(0, Math.min(100, Math.round(Number(value))))
    : null;
  let display = "Unavailable";
  if (rangeLow != null && rangeHigh != null) {
    display = `${rangeLow}–${rangeHigh}`;
  } else if (uncertain) {
    display = "Low confidence";
  } else if (v != null) {
    display = String(v);
  }
  return (
    <div className={`${PREFIX}-attr-bar${uncertain ? " is-uncertain" : ""}`}>
      <div className={`${PREFIX}-attr-bar-head`}>
        <span>{label}</span>
        <strong className={`${PREFIX}-tabular`}>{display}</strong>
      </div>
      <div className={`${PREFIX}-attr-track`}>
        <i style={{ width: `${v != null ? v : 0}%` }} />
      </div>
    </div>
  );
}

function StockSpark({ history, preseason, current }) {
  const rows = safeArray(history);
  const points = [];
  if (preseason != null) points.push({ label: "Preseason", rank: preseason });
  rows.forEach((h) => {
    const r = num(h.rank ?? h.public_rank);
    if (r != null) points.push({
      label: h.date || h.stage || h.event_source || "Update",
      rank: r,
      reason: h.reason || h.stock_label || null,
    });
  });
  if (current != null && (!points.length || points[points.length - 1].rank !== current)) {
    points.push({ label: "Now", rank: current });
  }
  if (points.length < 2) return null;
  const maxR = Math.max(...points.map((p) => p.rank));
  const minR = Math.min(...points.map((p) => p.rank));
  const span = Math.max(1, maxR - minR);
  return (
    <div className={`${PREFIX}-stock-spark`}>
      <div className={`${PREFIX}-stock-spark-chart`} aria-hidden="true">
        {points.map((pt, i) => (
          <span
            key={`${pt.label}-${i}`}
            title={`#${pt.rank} · ${pt.label}`}
            style={{ height: `${18 + ((maxR - pt.rank) / span) * 42}%` }}
          />
        ))}
      </div>
      <ul className={`${PREFIX}-stock-spark-list`}>
        {points.map((pt, i) => (
          <li key={`${pt.label}-li-${i}`}>
            <strong className={`${PREFIX}-tabular`}>#{pt.rank}</strong>
            <span>{pt.label}</span>
            {pt.reason ? <em>{pt.reason}</em> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

function getNhlEta(p) {
  const value = p?.nhl_eta ?? p?.eta ?? p?.nhl_eta_years ?? p?.development_eta ?? p?.rights_card?.eta;
  if (value == null || value === "") return null;
  if (typeof value === "object") {
    if (value.label === "Now" || value.years === 0) return "NHL Ready";
    if (value.label) return String(value.label);
    if (value.years != null && value.years !== "") return `${value.years}Y`;
    return null;
  }
  return String(value);
}

function getDevelopmentPath(p) {
  return p?.development_path || null;
}

function getComparable(p) {
  const v = p?.comparable_player || p?.player_comparable || p?.style_comparable;
  return v ? String(v) : null;
}

function getScoutSummary(p) {
  return p?.scout_summary || p?.scout_quote || p?.summary || null;
}

function getBackendWhyWorks(p) {
  return p?.why_pick_makes_sense || p?.why_this_pick_makes_sense || null;
}

function getBackendWhyFails(p) {
  return p?.why_pick_could_fail || p?.bust_reason || null;
}

function shortTeamName(name) {
  if (!name) return "";
  const s = String(name);
  if (s.includes(" ")) {
    const parts = s.trim().split(/\s+/);
    return parts[parts.length - 1];
  }
  return s;
}

function teamAbbrev(teamId, teamName) {
  return getTeamAbbreviation({ team_id: teamId, name: teamName, id: teamId }) ||
    String(teamName || teamId || "").slice(0, 3).toUpperCase();
}

function needLabel(n) {
  return n?.category || n?.position || n?.need || n?.label || null;
}

function getUpcomingOrder(draftOrder, completedCount, userTeamId, windowSize = 5) {
  const order = safeArray(draftOrder);
  const upcoming = order.slice(completedCount + 1);
  const next = upcoming.slice(0, windowSize);
  const userNext = upcoming.find((s) => String(s.team_id) === String(userTeamId));
  if (userNext && !next.some((s) => s.overall_pick === userNext.overall_pick)) {
    return [...next, userNext];
  }
  return next;
}

function TeamLogo({ teamId, teamName, size = "md" }) {
  const src = getTeamLogoSrc({ team_id: teamId, name: teamName });
  if (!src) {
    const label = teamAbbrev(teamId, teamName) || "TM";
    return <span className={`${PREFIX}-logo ${size} ${PREFIX}-logo-fallback`}>{label}</span>;
  }
  return (
    <span className={`${PREFIX}-logo ${size}`}>
      <img src={src} alt="" loading="lazy" />
    </span>
  );
}

function PickBadge({ tag, tone }) {
  if (!tag) return null;
  const resolved =
    tone ||
    (tag === "Steal" ? "steal" :
    tag === "Reach" ? "reach" :
    tag === "Early" ? "early" :
    tag === "BPA" ? "bpa" :
    tag === "Need" || tag === "Need Pick" ? "need" :
    tag === "Off Board" || tag === "Off the Board" ? "offboard" :
    tag === "Safe" || tag === "Value" ? "safe" :
    tag === "Goalie" || tag === "Goalie Gamble" ? "goalie" :
    tag === "Risk" ? "risk" : "default");
  return <span className={`${PREFIX}-pick-badge tone-${resolved}`}>{tag}</span>;
}

function DraftLogPanel({
  completed,
  selectedPick,
  onSelectPick,
  userTeamId,
  autoFollow,
  onToggleAutoFollow,
}) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const scrollRef = useRef(null);
  const rows = safeArray(completed);
  const empty = !rows.length;

  const filtered = useMemo(() => {
    let list = rows.slice().reverse();
    if (filter === "mine") list = list.filter((p) => String(p.team_id) === String(userTeamId));
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((p) => {
        const name = String(p.prospect_name || getPlayerName(p) || "").toLowerCase();
        const team = String(p.team_name || "").toLowerCase();
        return name.includes(q) || team.includes(q) || String(p.overall_pick).includes(q);
      });
    }
    return list;
  }, [rows, filter, search, userTeamId]);

  const grouped = useMemo(() => {
    const out = [];
    let lastRound = null;
    for (const p of filtered) {
      if (p.round !== lastRound) {
        out.push({ type: "round", round: p.round, key: `round-${p.round}-${p.overall_pick}` });
        lastRound = p.round;
      }
      out.push({ type: "pick", pick: p, key: `pick-${p.overall_pick}` });
    }
    return out;
  }, [filtered]);

  useEffect(() => {
    if (!autoFollow || empty || !scrollRef.current) return;
    scrollRef.current.scrollTop = 0;
  }, [rows.length, autoFollow, empty]);

  return (
    <aside className={`${PREFIX}-log${empty ? " is-empty" : ""}`}>
      <div className={`${PREFIX}-section-head`}>
        <h3 className={`${PREFIX}-section-title`}>Draft Log</h3>
        {!empty ? (
          <button
            type="button"
            className={`${PREFIX}-text-btn`}
            onClick={onToggleAutoFollow}
            aria-pressed={autoFollow}
          >
            {autoFollow ? "Pause follow" : "Follow"}
          </button>
        ) : null}
      </div>

      {empty ? (
        <p className={`${PREFIX}-log-empty`}>No picks yet</p>
      ) : (
        <>
          <div className={`${PREFIX}-log-tools`}>
            <div className={`${PREFIX}-seg`}>
              <button type="button" className={filter === "all" ? "is-active" : ""} onClick={() => setFilter("all")}>
                All Picks
              </button>
              <button type="button" className={filter === "mine" ? "is-active" : ""} onClick={() => setFilter("mine")}>
                My Picks
              </button>
            </div>
            {rows.length >= 20 ? (
              <input
                type="search"
                className={`${PREFIX}-input`}
                placeholder="Search picks"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search draft log"
              />
            ) : null}
          </div>

          <div className={`${PREFIX}-log-scroll`} ref={scrollRef}>
            {grouped.map((row) => {
              if (row.type === "round") {
                return (
                  <div key={row.key} className={`${PREFIX}-log-round`}>
                    Round {row.round}
                  </div>
                );
              }
              const p = row.pick;
              const { tag, tone } = getPickDisplayTag(p);
              const selected = selectedPick?.overall_pick === p.overall_pick;
              const abbr = teamAbbrev(p.team_id, p.team_name);
              return (
                <button
                  type="button"
                  key={row.key}
                  className={`${PREFIX}-log-row${selected ? " is-selected" : ""}`}
                  onClick={() => onSelectPick?.(p)}
                >
                  <span className={`${PREFIX}-log-num`}>#{p.overall_pick}</span>
                  <TeamLogo teamId={p.team_id} teamName={p.team_name} size="sm" />
                  <span className={`${PREFIX}-log-body`}>
                    <strong>{p.prospect_name || getPlayerName(p)}</strong>
                    <span>
                      {p.position || getPlayerPosition(p)} · {abbr}
                      {p.is_traded && p.via_team_name ? ` · via ${teamAbbrev(p.via_team_id, p.via_team_name)}` : ""}
                      {p.public_rank_at_pick != null || p.final_rank != null
                        ? ` · Public #${p.public_rank_at_pick ?? p.final_rank}`
                        : ""}
                    </span>
                  </span>
                  {tag ? <PickBadge tag={tag} tone={tone} /> : null}
                </button>
              );
            })}
          </div>
        </>
      )}
    </aside>
  );
}

function OrderStrip({ upcoming, userTeamId }) {
  // Scrollable on-deck so the GM can see the next wave of picks, not just three.
  const next = safeArray(upcoming).slice(0, 16);
  if (!next.length) return null;

  return (
    <div className={`${PREFIX}-order-strip`}>
      <span className={`${PREFIX}-order-label`}>On deck</span>
      <div className={`${PREFIX}-order-track`} aria-label="Upcoming picks">
        {next.map((slot, idx) => {
          const isUser = String(slot.team_id) === String(userTeamId);
          const abbr = teamAbbrev(slot.team_id, slot.team_name);
          return (
            <div
              key={slot.overall_pick}
              className={`${PREFIX}-order-item${isUser ? " is-user" : ""}${idx === 0 ? " is-next" : ""}`}
              title={slot.team_name || abbr}
            >
              <span className={`${PREFIX}-order-num`}>#{slot.overall_pick}</span>
              <TeamLogo teamId={slot.team_id} teamName={slot.team_name} size="sm" />
              <div className={`${PREFIX}-order-meta`}>
                <span className={`${PREFIX}-order-abbr`}>{abbr}</span>
                {slot.is_traded && (slot.via_team_name || slot.via_team_id) ? (
                  <span className={`${PREFIX}-order-via`}>
                    <TeamLogo
                      teamId={slot.via_team_id || slot.original_owner_team_id}
                      teamName={slot.via_team_name || slot.original_owner_team_name}
                      size="xs"
                    />
                    via {teamAbbrev(slot.via_team_id, slot.via_team_name)}
                  </span>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TeamBoardPanel({
  available,
  needs,
  philosophy,
  selectedId,
  onSelectProspect,
  density,
  onDensityChange,
  pinnedIds = [],
}) {
  const [search, setSearch] = useState("");
  const [posFilter, setPosFilter] = useState("all");
  const [sortBy, setSortBy] = useState("backend");
  const [showMoreNeeds, setShowMoreNeeds] = useState(false);
  const listRef = useRef(null);
  const scrollTopRef = useRef(0);

  const needRows = safeArray(needs);
  const hasMoreNeeds = needRows.filter((n) => !isPhilosophyNeedLabel(needLabel(n))).length > 3;
  const philLabel = philosophy?.label || philosophy?.name || null;
  const philText = philosophy?.description || philosophy?.summary || null;

  const filtered = useMemo(() => {
    let rows = safeArray(available).slice();
    if (posFilter === "C") rows = rows.filter((p) => getPlayerPosition(p) === "C");
    else if (posFilter === "LW") rows = rows.filter((p) => getPlayerPosition(p) === "LW");
    else if (posFilter === "RW") rows = rows.filter((p) => getPlayerPosition(p) === "RW");
    else if (posFilter === "D") rows = rows.filter((p) => getPlayerPosition(p) === "D");
    else if (posFilter === "G") rows = rows.filter((p) => getPlayerPosition(p) === "G");

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((p) => String(getPlayerName(p) || "").toLowerCase().includes(q));
    }

    if (sortBy === "public") {
      rows.sort((a, b) => (getRank(a) ?? 9999) - (getRank(b) ?? 9999));
    } else if (sortBy === "team") {
      rows.sort((a, b) => (getTeamRank(a) ?? 9999) - (getTeamRank(b) ?? 9999));
    } else if (sortBy === "position") {
      rows.sort((a, b) => String(getPlayerPosition(a)).localeCompare(String(getPlayerPosition(b))));
    } else if (sortBy === "confidence") {
      rows.sort((a, b) => (getConfidence(b) ?? -1) - (getConfidence(a) ?? -1));
    }
    return rows;
  }, [available, posFilter, search, sortBy]);

  const onSelect = useCallback((p) => {
    if (listRef.current) scrollTopRef.current = listRef.current.scrollTop;
    onSelectProspect?.(p);
  }, [onSelectProspect]);

  // Keep the selected row visible when selection changes (including external changes,
  // e.g. after filtering/sorting or a pick submission). Uses "nearest" so the list
  // does not jump when the row is already on screen.
  useEffect(() => {
    if (!listRef.current || !selectedId) return;
    const el = listRef.current.querySelector('[data-selected="true"]');
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ block: "nearest" });
    }
  }, [selectedId, filtered]);

  return (
    <aside className={`${PREFIX}-board`}>
      <div className={`${PREFIX}-board-sticky`}>
        <div className={`${PREFIX}-section-head`}>
          <div>
            <h3 className={`${PREFIX}-section-title`}>Best Available</h3>
            <p className={`${PREFIX}-section-meta`}>
              <span className={`${PREFIX}-tabular`}>{filtered.length}</span> available
            </p>
          </div>
          <div className={`${PREFIX}-seg`}>
            <button
              type="button"
              className={density === "compact" ? "is-active" : ""}
              onClick={() => onDensityChange("compact")}
            >
              Compact
            </button>
            <button
              type="button"
              className={density === "detailed" ? "is-active" : ""}
              onClick={() => onDensityChange("detailed")}
            >
              Detailed
            </button>
          </div>
        </div>

        {needRows.length || philLabel ? (
          <div className={`${PREFIX}-needs-line`}>
            {needRows.filter((n) => !isPhilosophyNeedLabel(needLabel(n))).length ? (
              <span>
                Needs: {needRows
                  .map(needLabel)
                  .filter((n) => n && !isPhilosophyNeedLabel(n))
                  .slice(0, showMoreNeeds ? 99 : 3)
                  .join(", ")}
                {hasMoreNeeds && !showMoreNeeds ? (
                  <>
                    {" "}
                    <button type="button" className={`${PREFIX}-text-btn`} onClick={() => setShowMoreNeeds(true)}>
                      More
                    </button>
                  </>
                ) : null}
              </span>
            ) : null}
            {philLabel ? (
              <span className={`${PREFIX}-phil-text`} title={philText || undefined}>
                Strategy: {philLabel}
              </span>
            ) : null}
          </div>
        ) : null}

        <div className={`${PREFIX}-board-search`}>
          <input
            type="search"
            className={`${PREFIX}-input`}
            placeholder="Search prospects"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search prospects"
          />
        </div>

        <div className={`${PREFIX}-board-tools`}>
          <div className={`${PREFIX}-seg ${PREFIX}-pos-seg`}>
            {["all", "C", "LW", "RW", "D", "G"].map((f) => (
              <button
                key={f}
                type="button"
                className={posFilter === f ? "is-active" : ""}
                aria-pressed={posFilter === f}
                onClick={() => setPosFilter(f)}
              >
                {f === "all" ? "All" : f}
              </button>
            ))}
          </div>
          <label className={`${PREFIX}-sort-label`}>
            Sort
            <select
              className={`${PREFIX}-select`}
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="backend">Board order</option>
              <option value="public">Public rank</option>
              <option value="team">Team rank</option>
              <option value="position">Position</option>
              <option value="confidence">Confidence</option>
            </select>
          </label>
        </div>

        <div className={`${PREFIX}-board-cols ${density === "detailed" ? "is-detailed" : ""}`} aria-hidden="true">
          <span>Public</span>
          <span>Team</span>
          <span>Prospect</span>
          <span>Pos</span>
          <span>OVR</span>
          <span>Pot</span>
          <span>Risk</span>
          <span>Conf</span>
          <span>Stock</span>
        </div>
      </div>

      <div className={`${PREFIX}-board-scroll`} ref={listRef} role="listbox" aria-label="Best available prospects">
        {!filtered.length ? (
          <p className={`${PREFIX}-muted`}>No prospects match.</p>
        ) : (
          filtered.map((p) => {
            const pub = getRank(p);
            const team = getTeamRank(p);
            const risk = getRisk(p);
            const conf = formatConfidence(p);
            const stock = normalizeStockLabel(p);
            const trait = getDefiningTrait(p);
            const pot = getScoutedPotentialLabel(p);
            const ovr = formatCurrentOvr(p);
            const reason = p.stock_reason || p.franchise_tier?.label || null;
            const selected = selectedId === getId(p);
            const pinned = safeArray(pinnedIds).includes(getId(p));

            return (
              <button
                type="button"
                key={getId(p)}
                role="option"
                aria-selected={selected}
                data-selected={selected ? "true" : undefined}
                className={`${PREFIX}-board-row ${density === "detailed" ? "is-detailed" : "is-compact"}${selected ? " is-selected" : ""}`}
                onClick={() => onSelect(p)}
                title={reason || undefined}
              >
                <span className={`${PREFIX}-col-rank ${PREFIX}-tabular`}>{pub != null ? `#${pub}` : ""}</span>
                <span className={`${PREFIX}-col-rank ${PREFIX}-tabular is-team`}>{team != null ? `#${team}` : ""}</span>
                <span className={`${PREFIX}-col-prospect`}>
                  <strong>{getPlayerName(p)}{pinned ? " · Pin" : ""}</strong>
                  {trait ? <em>{trait}</em> : null}
                  {density === "detailed" && reason ? <span>{reason}</span> : null}
                  {p.scout_favorite ? <span className={`${PREFIX}-fav`}>Scout favorite</span> : null}
                </span>
                <span className={`${PREFIX}-col-pos`}>{getPlayerPosition(p)}</span>
                <span className={`${PREFIX}-col-ovr ${PREFIX}-tabular`}>{ovr || ""}</span>
                <span className={`${PREFIX}-col-pot ${PREFIX}-tabular`}>{pot || ""}</span>
                <span className={`${PREFIX}-col-risk`}>{risk || ""}</span>
                <span className={`${PREFIX}-col-conf ${PREFIX}-tabular`}>{conf || ""}</span>
                <span className={`${PREFIX}-col-stock ${stock === "Rising" ? "up" : stock === "Falling" ? "down" : ""}`}>
                  {stock || ""}
                </span>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}

function ProspectDossier({
  prospect,
  onClose,
  onDraft,
  onCompare,
  compareIds,
  isUserPick,
  loading,
  shortlistIds,
  onToggleShortlist,
  onPinBoard,
}) {
  const [tab, setTab] = useState("overview");
  const [confirm, setConfirm] = useState(false);
  const closeBtnRef = useRef(null);

  useEffect(() => {
    closeBtnRef.current?.focus();
    setTab("overview");
    setConfirm(false);
  }, [prospect]);

  if (!prospect) return null;

  const d = prospect.dossier || {};
  const pub = getRank(prospect);
  const pre = getPreseasonRank(prospect) ?? num(prospect.rank_prev ?? prospect.previous_rank);
  const tb = getTeamRank(prospect);
  const move = num(prospect.stock_change ?? prospect.stock_delta ?? prospect.rank_change);
  const ovr = formatCurrentOvr(prospect);
  const pot = getScoutedPotentialLabel(prospect);
  const conf = formatConfidence(prospect);
  const confNum = getConfidence(prospect);
  const readiness = formatNhlReadiness(prospect);
  const archetype = getDefiningTrait(prospect) || d?.player_comparison?.archetype || null;
  const strengths = safeArray(d.strengths).filter(Boolean).slice(0, 5);
  const concerns = safeArray(d.concerns).filter(Boolean).slice(0, 5);
  const stats = d.stats || {};
  const projection = d.projection || {};
  const potentialBlock = d.potential || {};
  const teamFit = d.team_fit || {};
  const comparison = d.player_comparison || {};
  const eta = d.eta || {};
  const rights = prospect.rights_card || {};
  const notes = safeArray(prospect.scouting_event_notes);
  const stockReason = prospect.stock_reason || prospect.weekly_stock_reason || null;
  const isGoaliePos = getPlayerPosition(prospect) === "G";
  const attrs = isGoaliePos
    ? [
        ["Reflexes", prospect.reflexes ?? prospect.reflex_score],
        ["Rebound control", prospect.rebound_control ?? prospect.rebound_score],
        ["Positioning", prospect.positioning ?? prospect.positioning_score],
        ["Glove", prospect.glove ?? prospect.glove_score],
        ["Athleticism", prospect.athleticism ?? prospect.athletic_score],
        ["Mental", prospect.mental_toughness ?? prospect.poise],
      ]
    : [
        ["Skating", prospect.skating ?? prospect.skating_score ?? prospect.speed],
        ["Shooting", prospect.shot ?? prospect.shooting ?? prospect.shot_score],
        ["Passing", prospect.passing ?? prospect.pass_score ?? prospect.playmaking],
        ["Hockey IQ", prospect.hockey_iq ?? prospect.iq ?? prospect.processing],
        ["Defence", prospect.defense ?? prospect.defensive_game ?? prospect.defensive_score],
        ["Physicality", prospect.physicality ?? prospect.physical ?? prospect.strength],
        ["Puck skills", prospect.puck_skills ?? prospect.hands ?? prospect.puck_skill],
      ];
  const visibleAttrs = attrs.filter(([, v]) => v != null && Number.isFinite(Number(v)));
  const pid = getId(prospect);
  const shortlisted = safeArray(shortlistIds).includes(pid);

  return (
    <div className={`${PREFIX}-dossier-backdrop`} role="presentation" onClick={onClose}>
      <article
        className={`${PREFIX}-dossier`}
        role="dialog"
        aria-modal="true"
        aria-label={`${getPlayerName(prospect)} prospect dossier`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className={`${PREFIX}-dossier-head`}>
          <div className={`${PREFIX}-dossier-identity`}>
            <div className={`${PREFIX}-dossier-avatar`} aria-hidden="true">
              <PlayerHeadshot
                player={prospect}
                size="sm"
                style={{ "--size": "45px" }}
              />
            </div>
            <div>
              <h2>{getPlayerName(prospect)}</h2>
              <p>
                {[
                  getPlayerPosition(prospect),
                  prospect.handedness || prospect.shoots || prospect.shot_side,
                  d.team || prospect.team_name || prospect.team,
                  d.league || prospect.league || prospect.league_name,
                  d.nationality || prospect.nationality || prospect.country,
                ].filter(Boolean).join(" · ")}
              </p>
              <p className={`${PREFIX}-dossier-vitals`}>
                {[
                  prospect.height || d.height,
                  prospect.weight || d.weight ? `${prospect.weight || d.weight} lbs` : null,
                  (prospect.age || d.age) != null ? `Age ${prospect.age || d.age}` : null,
                ].filter(Boolean).join(" · ")}
              </p>
            </div>
          </div>
          <div className={`${PREFIX}-dossier-ranks`}>
            {pub != null ? (
              <div>
                <span>Public</span>
                <strong className={`${PREFIX}-tabular`}>#{pub}</strong>
                {pre != null && pre !== pub ? (
                  <em className={`${PREFIX}-tabular`}>was #{pre}</em>
                ) : null}
              </div>
            ) : null}
            {tb != null ? (
              <div>
                <span>Your board</span>
                <strong className={`${PREFIX}-tabular`}>#{tb}</strong>
              </div>
            ) : null}
            {move != null && move !== 0 ? (
              <div>
                <span>Movement</span>
                <strong className={move > 0 ? "up" : "down"}>
                  {move > 0 ? `↑${move}` : `↓${Math.abs(move)}`}
                </strong>
              </div>
            ) : null}
            {pot ? (
              <div>
                <span>Projected range</span>
                <strong>{pot}</strong>
              </div>
            ) : null}
          </div>
          <button type="button" className={`${PREFIX}-ghost-btn`} ref={closeBtnRef} onClick={onClose}>
            Close
          </button>
        </header>

        <div className={`${PREFIX}-dossier-tabs`} role="tablist">
          {[
            ["overview", "Overview"],
            ["scouting", "Scouting"],
            ["production", "Production"],
            ["fit", "Team Fit"],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              className={tab === id ? "is-active" : ""}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className={`${PREFIX}-dossier-body`}>
          {tab === "overview" ? (
            <>
              <section className={`${PREFIX}-dossier-snapshot`}>
                <h3>Scouting snapshot</h3>
                <div className={`${PREFIX}-snapshot-grid`}>
                  {ovr ? <div><span>Current OVR</span><strong>{ovr}</strong></div> : null}
                  {pot ? <div><span>Potential range</span><strong>{pot}</strong></div> : null}
                  {conf ? <div><span>Scouting confidence</span><strong>{conf}</strong></div> : null}
                  {readiness ? <div><span>NHL readiness</span><strong>{readiness}</strong></div> : null}
                  {archetype ? <div><span>Archetype</span><strong>{archetype}</strong></div> : null}
                </div>
                {stockReason ? <p className={`${PREFIX}-dossier-note`}>{stockReason}</p> : null}
                {d.micro_summary ? <p className={`${PREFIX}-dossier-note`}>{d.micro_summary}</p> : null}
              </section>

              {(strengths.length || concerns.length) ? (
                <section className={`${PREFIX}-dossier-split`}>
                  {strengths.length ? (
                    <div>
                      <h3>Strengths</h3>
                      <ul>{strengths.map((s) => <li key={s}>{s}</li>)}</ul>
                    </div>
                  ) : null}
                  {concerns.length ? (
                    <div>
                      <h3>Concerns</h3>
                      <ul>{concerns.map((s) => <li key={s}>{s}</li>)}</ul>
                    </div>
                  ) : null}
                </section>
              ) : null}

              {visibleAttrs.length ? (
                <section>
                  <h3>Attributes</h3>
                  <div className={`${PREFIX}-attr-grid`}>
                    {visibleAttrs.map(([label, value]) => (
                      <AttrBar key={label} label={label} value={value} confidence={confNum} />
                    ))}
                  </div>
                </section>
              ) : null}

              {(projection.label || potentialBlock.label || comparison.archetype || eta.label) ? (
                <section>
                  <h3>Projection</h3>
                  <div className={`${PREFIX}-sheet-facts`}>
                    {projection.label ? <span>Likely: {projection.label}</span> : null}
                    {potentialBlock.label ? <span>Ceiling view: {potentialBlock.label}</span> : null}
                    {eta.label || readiness ? <span>Timeline: {eta.label || readiness}</span> : null}
                    {getRisk(prospect) ? <span>Bust risk: {getRisk(prospect)}</span> : null}
                    {comparison.archetype ? <span>Style: {comparison.archetype}</span> : null}
                    {pot ? <span>Peak OVR est.: {pot}</span> : null}
                  </div>
                </section>
              ) : null}

              <StockSpark
                history={prospect.stock_history}
                preseason={pre}
                current={pub}
              />
            </>
          ) : null}

          {tab === "scouting" ? (
            <>
              {d.character_read ? (
                <section>
                  <h3>Character read</h3>
                  <div className={`${PREFIX}-sheet-facts`}>
                    <span>Headline: {d.character_read.headline || "Mixed reports"}</span>
                    {d.character_read.confidence ? (
                      <span>Scout confidence: {d.character_read.confidence}%</span>
                    ) : null}
                  </div>
                  {(d.character_read.traits || []).length ? (
                    <div className={`${PREFIX}-attr-grid`}>
                      {d.character_read.traits.map((trait) => (
                        <div key={trait.label} className={`${PREFIX}-attr-row`}>
                          <span>{trait.label}</span>
                          <strong>{trait.tier || "Unknown"}</strong>
                          {trait.confidence ? (
                            <em>{trait.confidence}% conf.</em>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {d.character_read.interview_notes ? (
                    <p className={`${PREFIX}-dossier-note`}>{d.character_read.interview_notes}</p>
                  ) : null}
                </section>
              ) : null}
              {getScoutSummary(prospect) || d.micro_summary ? (
                <section>
                  <h3>Scout report</h3>
                  <p>{getScoutSummary(prospect) || d.micro_summary}</p>
                </section>
              ) : null}
              {notes.length ? (
                <section>
                  <h3>Staff notes</h3>
                  <ul className={`${PREFIX}-dossier-notes`}>
                    {notes.slice(0, 8).map((n, i) => <li key={i}>{typeof n === "string" ? n : n.text || n.note}</li>)}
                  </ul>
                </section>
              ) : null}
              {(prospect.combine_status || prospect.interview_notes || rights.expected_role) ? (
                <section>
                  <h3>Combine / character</h3>
                  <div className={`${PREFIX}-sheet-facts`}>
                    {prospect.combine_status ? <span>Combine: {prospect.combine_status}</span> : null}
                    {prospect.interview_notes ? <span>{prospect.interview_notes}</span> : null}
                    {rights.expected_role ? <span>Role path: {rights.expected_role}</span> : null}
                    {rights.rights_status ? <span>Rights: {rights.rights_status}</span> : null}
                  </div>
                </section>
              ) : null}
              {!getScoutSummary(prospect) && !notes.length && !d.micro_summary ? (
                <p className={`${PREFIX}-muted`}>No detailed scout write-up available yet.</p>
              ) : null}
            </>
          ) : null}

          {tab === "production" ? (
            <>
              <section>
                <h3>Production</h3>
                <div className={`${PREFIX}-prod-grid`}>
                  {stats.games != null ? <div><span>GP</span><strong className={`${PREFIX}-tabular`}>{stats.games}</strong></div> : null}
                  {stats.goals != null ? <div><span>G</span><strong className={`${PREFIX}-tabular`}>{stats.goals}</strong></div> : null}
                  {stats.assists != null ? <div><span>A</span><strong className={`${PREFIX}-tabular`}>{stats.assists}</strong></div> : null}
                  {stats.points != null ? <div><span>P</span><strong className={`${PREFIX}-tabular`}>{stats.points}</strong></div> : null}
                  {stats.ppg != null ? <div><span>PPG</span><strong className={`${PREFIX}-tabular`}>{stats.ppg}</strong></div> : null}
                  {prospect.primary_points != null ? <div><span>Primary</span><strong className={`${PREFIX}-tabular`}>{prospect.primary_points}</strong></div> : null}
                  {prospect.shots != null ? <div><span>Shots</span><strong className={`${PREFIX}-tabular`}>{prospect.shots}</strong></div> : null}
                  {prospect.shooting_pct != null ? <div><span>SH%</span><strong className={`${PREFIX}-tabular`}>{prospect.shooting_pct}</strong></div> : null}
                  {prospect.production_adjusted_score != null ? (
                    <div><span>League-adj</span><strong className={`${PREFIX}-tabular`}>{Math.round(Number(prospect.production_adjusted_score))}</strong></div>
                  ) : null}
                </div>
              </section>
              {d.competition ? (
                <section>
                  <h3>Competition context</h3>
                  <p>
                    {[d.competition.label, d.competition.comp_label, prospect.quality_of_teammates != null ? `Linemates ${prospect.quality_of_teammates}` : null]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                </section>
              ) : null}
              {!stats.games && stats.points == null ? (
                <p className={`${PREFIX}-muted`}>No production sample available.</p>
              ) : null}
            </>
          ) : null}

          {tab === "fit" ? (
            <>
              {teamFit && (teamFit.label || teamFit.summary || teamFit.score != null) ? (
                <section>
                  <h3>Team fit</h3>
                  {teamFit.label ? <p>{teamFit.label}</p> : null}
                  {teamFit.summary ? <p>{teamFit.summary}</p> : null}
                  {teamFit.score != null ? <p>Fit score: {Math.round(Number(teamFit.score))}</p> : null}
                </section>
              ) : null}
              {rights.development_path || getDevelopmentPath(prospect) ? (
                <section>
                  <h3>Development path</h3>
                  <p>{rights.development_path || getDevelopmentPath(prospect)}</p>
                </section>
              ) : null}
              {getTeamFit(prospect) != null ? (
                <p>Organizational fit: {Math.round(getTeamFit(prospect))}</p>
              ) : null}
              {!teamFit?.label && !rights.development_path && getTeamFit(prospect) == null ? (
                <p className={`${PREFIX}-muted`}>No team-fit report for this club yet.</p>
              ) : null}
            </>
          ) : null}
        </div>

        <footer className={`${PREFIX}-dossier-actions`}>
          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => onToggleShortlist?.(prospect)}>
            {shortlisted ? "Remove shortlist" : "Add to shortlist"}
          </button>
          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => onCompare?.(prospect)}>
            {compareIds?.includes(pid) ? "In Compare" : "Compare prospects"}
          </button>
          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => onPinBoard?.(prospect)}>
            Pin to draft board
          </button>
          {isUserPick ? (
            confirm ? (
              <>
                <span className={`${PREFIX}-confirm-q`}>Confirm selection?</span>
                <button type="button" className={`${PREFIX}-cta-btn`} disabled={loading} onClick={() => onDraft?.(prospect)}>
                  {loading ? LOADING_COPY.submit : `Select ${getPlayerName(prospect)}`}
                </button>
                <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => setConfirm(false)}>Cancel</button>
              </>
            ) : (
              <button type="button" className={`${PREFIX}-cta-btn`} disabled={loading} onClick={() => setConfirm(true)}>
                Select {getPlayerName(prospect)}
              </button>
            )
          ) : null}
          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={onClose}>Close dossier</button>
        </footer>
      </article>
    </div>
  );
}

function prospectAttrRows(prospect) {
  const isGoaliePos = getPlayerPosition(prospect) === "G";
  const attrs = isGoaliePos
    ? [
        ["Reflexes", prospect.reflexes ?? prospect.reflex_score],
        ["Rebound control", prospect.rebound_control ?? prospect.rebound_score],
        ["Positioning", prospect.positioning ?? prospect.positioning_score],
        ["Glove", prospect.glove ?? prospect.glove_score],
        ["Athleticism", prospect.athleticism ?? prospect.athletic_score],
        ["Mental", prospect.mental_toughness ?? prospect.poise],
      ]
    : [
        ["Skating", prospect.skating ?? prospect.skating_score ?? prospect.speed],
        ["Shooting", prospect.shot ?? prospect.shooting ?? prospect.shot_score],
        ["Passing", prospect.passing ?? prospect.pass_score ?? prospect.playmaking],
        ["Hockey IQ", prospect.hockey_iq ?? prospect.iq ?? prospect.processing],
        ["Defence", prospect.defense ?? prospect.defensive_game ?? prospect.defensive_score],
        ["Physicality", prospect.physicality ?? prospect.physical ?? prospect.strength],
        ["Puck skills", prospect.puck_skills ?? prospect.hands ?? prospect.puck_skill],
      ];
  return attrs.filter(([, v]) => v != null && Number.isFinite(Number(v)));
}

// Inline scouting dossier that anchors the centre stage (concept-art layout).
// Every sub-panel is gated on real backend data — nothing renders as a placeholder.
function ProspectStageCard({ prospect, boardContext, currentPickOverall, onFullReport }) {
  if (!prospect) {
    return (
      <div className={`${PREFIX}-pcard ${PREFIX}-pcard--empty`}>
        <div className={`${PREFIX}-pcard-empty-inner`}>
          <span className={`${PREFIX}-pcard-empty-mark`} aria-hidden="true">◆</span>
          <p>Select a prospect from Best Available to open the scouting dossier.</p>
        </div>
      </div>
    );
  }

  const d = prospect.dossier || {};
  const name = getPlayerName(prospect);
  const pos = getPlayerPosition(prospect);
  const shoots = prospect.handedness || prospect.shoots || prospect.shot_side || null;
  const nationality = d.nationality || prospect.nationality || prospect.country || null;
  const natCode = prospect.country_code || prospect.nat || prospect.nationality_code || nationality;
  const league = d.league || prospect.league || prospect.league_name || null;
  const team = d.team || prospect.team_name || prospect.team || null;
  const height = prospect.height || d.height || null;
  const weight = (prospect.weight || d.weight) ? `${prospect.weight || d.weight} lbs` : null;
  const ageRaw = prospect.age ?? d.age;
  const age = ageRaw != null ? `Age ${ageRaw}` : null;
  const pub = getRank(prospect);
  const conf = getConfidence(prospect);
  const summary = getScoutSummary(prospect) || d.micro_summary || null;

  const stats = d.stats || {};
  const projection = d.projection || {};
  const comparison = d.player_comparison || {};
  const archetype = comparison.archetype || getDefiningTrait(prospect) || null;

  // Evidence-based strengths / weaknesses (title + supporting fact). Fall back to
  // the plain string lists the backend also provides — never invented client-side.
  const strengthItems = safeArray(d.strengthsEvidence).length
    ? safeArray(d.strengthsEvidence).slice(0, 4)
    : safeArray(d.strengths).filter(Boolean).slice(0, 4).map((s) => ({ title: null, fact: String(s) }));
  const weakItems = safeArray(d.weaknessesEvidence).length
    ? safeArray(d.weaknessesEvidence).slice(0, 3)
    : safeArray(d.concerns).filter(Boolean).slice(0, 3).map((s) => ({ title: null, fact: String(s) }));

  // Current ability and projected peak are kept strictly separate.
  const curLow = num(d.overallRangeLow ?? (Array.isArray(prospect.current_ovr_range) ? prospect.current_ovr_range[0] : null));
  const curHigh = num(d.overallRangeHigh ?? (Array.isArray(prospect.current_ovr_range) ? prospect.current_ovr_range[1] : null));
  const potLow = num(d.scoutedPotentialLow);
  const potHigh = num(d.scoutedPotentialHigh);
  const curLabel = curLow != null && curHigh != null
    ? (curLow === curHigh ? `${curLow}` : `${curLow}–${curHigh}`)
    : formatCurrentOvr(prospect);
  const potLabel = potLow != null && potHigh != null
    ? (potLow === potHigh ? `${potLow}` : `${potLow}–${potHigh}`)
    : getScoutedPotentialLabel(prospect);
  const role = projection.label || null;
  const eta = d.eta || {};
  const readinessLabel = formatNhlReadiness(prospect);
  const arrivalLabel =
    formatNhlArrivalEta(prospect) ||
    (eta.years === 0 || String(eta.label || "").toLowerCase() === "now"
      ? "NHL Ready / Now"
      : eta.years != null
        ? `${eta.years} Years Away`
        : eta.label || null);
  const potentialEtaLabel = formatNhlPotentialEta(prospect);
  const risk = (d.potential && d.potential.risk) || getRisk(prospect) || null;
  const projectionConf = num(projection.confidence) ?? conf;
  const devPath = prospect.rights_card?.development_path || getDevelopmentPath(prospect) || null;

  const curMid = curLow != null && curHigh != null ? (curLow + curHigh) / 2 : num(formatCurrentOvr(prospect));
  const peakMid = potLow != null && potHigh != null ? (potLow + potHigh) / 2 : num(getScoutedPotentialLabel(prospect));
  const growth =
    curMid != null && peakMid != null ? Math.max(0, Math.round(peakMid - curMid)) : null;
  const starCount =
    peakMid != null
      ? peakMid >= 92
        ? 5
        : peakMid >= 88
          ? 4
          : peakMid >= 84
            ? 3
            : peakMid >= 78
              ? 2
              : 1
      : 3;
  const gradeBanner =
    peakMid != null
      ? peakMid >= 92
        ? "ELITE NHL PROSPECT"
        : peakMid >= 88
          ? "HIGH-END NHL PROSPECT"
          : peakMid >= 82
            ? "NHL PROSPECT"
            : "DEVELOPMENTAL PROSPECT"
      : "NHL PROSPECT";
  const talentGrade =
    peakMid != null
      ? peakMid >= 92
        ? "A+"
        : peakMid >= 88
          ? "A"
          : peakMid >= 84
            ? "A-"
            : peakMid >= 80
              ? "B+"
              : "B"
      : "—";

  let decision = null;
  let reachSpots = null;
  if (pub != null && currentPickOverall != null) {
    const delta = currentPickOverall - pub;
    reachSpots = delta;
    if (delta >= 8) decision = "Value";
    else if (delta <= -12) decision = "Reach";
    else decision = "Expected";
  }

  const flagIso = resolveCountryCode(natCode || nationality);
  const flagSrc = flagIso ? `https://flagcdn.com/w320/${flagIso.toLowerCase()}.png` : flagApiUrl(natCode || nationality, 64);
  const oneLiner = [
    ageRaw != null ? `${ageRaw}-year-old` : null,
    nationality || null,
    pos || null,
    shoots ? `${shoots} shot` : null,
    height || weight || null,
    league ? `Playing ${league}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const riskPct =
    String(risk || "").toLowerCase().includes("high")
      ? 78
      : String(risk || "").toLowerCase().includes("med")
        ? 48
        : String(risk || "").toLowerCase().includes("low")
          ? 22
          : 40;

  return (
    <div className={`${PREFIX}-pcard ${PREFIX}-pcard--hero`}>
      <header className={`${PREFIX}-pcard-banner`}>
        <span className={`${PREFIX}-pcard-stars`} aria-hidden="true">
          {"★".repeat(starCount)}{"☆".repeat(Math.max(0, 5 - starCount))}
        </span>
        <strong>{gradeBanner}</strong>
        {role ? <em>{role}</em> : null}
      </header>

      <div className={`${PREFIX}-pcard-hero-head`}>
        {flagSrc ? (
          <img className={`${PREFIX}-pcard-hero-flag`} src={flagSrc} alt="" loading="lazy" />
        ) : null}
        <div>
          <h2 className={`${PREFIX}-pcard-name`}>{name}</h2>
          <p className={`${PREFIX}-pcard-pos`}>
            {[pos, age, league || team].filter(Boolean).join(" · ")}
          </p>
        </div>
        {currentPickOverall != null ? (
          <span className={`${PREFIX}-pcard-pickbadge`}>#{currentPickOverall} Overall</span>
        ) : (
          <span className={`${PREFIX}-pcard-pickbadge`}>EARLY PICK</span>
        )}
      </div>

      {oneLiner ? <p className={`${PREFIX}-pcard-oneliner`}>{oneLiner}</p> : null}

      <section className={`${PREFIX}-pcard-growth`} aria-label="Overall growth">
        <div className={`${PREFIX}-pcard-growth-nums`}>
          <div>
            <span>Current OVR</span>
            <strong className={`${PREFIX}-tabular`}>{curLabel || "—"}</strong>
          </div>
          <div className={`${PREFIX}-pcard-growth-arrow`} aria-hidden="true">
            <i />
            {growth != null ? <em>+{growth} OVR Growth Potential</em> : null}
          </div>
          <div>
            <span>Projected Ceiling</span>
            <strong className={`${PREFIX}-tabular`}>{potLabel || "—"}</strong>
          </div>
        </div>
        <div className={`${PREFIX}-pcard-grade-row`}>
          <div>
            <span>NHL Projection</span>
            <strong>{"★".repeat(starCount)}{"☆".repeat(Math.max(0, 5 - starCount))}</strong>
          </div>
          <div>
            <span>Talent Grade</span>
            <strong>{talentGrade}</strong>
          </div>
          <div>
            <span>NHL Ready</span>
            <strong>{readinessLabel || "—"}</strong>
          </div>
          <div>
            <span>Peak ETA</span>
            <strong>{potentialEtaLabel || "—"}</strong>
          </div>
        </div>
      </section>

      {summary ? (
        <section className={`${PREFIX}-pcard-why`}>
          <h4>Why We Picked Him</h4>
          <p>&ldquo;{summary}&rdquo;</p>
        </section>
      ) : null}

      <div className={`${PREFIX}-pcard-sw`}>
        <div className={`${PREFIX}-pcard-strengths`}>
          <h4>Strengths</h4>
          {strengthItems.length ? (
            <ul>
              {strengthItems.map((s, i) => (
                <li key={`str-${i}`}>{s.title || s.fact}</li>
              ))}
            </ul>
          ) : (
            <p className={`${PREFIX}-pcard-none`}>No report</p>
          )}
        </div>
        <div className={`${PREFIX}-pcard-weak`}>
          <h4>Risks</h4>
          {weakItems.length ? (
            <ul>
              {weakItems.map((s, i) => (
                <li key={`weak-${i}`}>{s.title || s.fact}</li>
              ))}
            </ul>
          ) : (
            <p className={`${PREFIX}-pcard-none`}>No report</p>
          )}
          {risk ? (
            <div className={`${PREFIX}-pcard-riskbar`}>
              <span>Risk</span>
              <div className={`${PREFIX}-pcard-conf-track`}>
                <i style={{ width: `${riskPct}%` }} />
              </div>
              <strong>{String(risk).toUpperCase()}</strong>
            </div>
          ) : null}
        </div>
      </div>

      <section className={`${PREFIX}-pcard-timeline`}>
        <h4>Development Path</h4>
        <p className={`${PREFIX}-pcard-timeline-line`}>
          {[league || "Junior", "AHL", "NHL"].join("  →  ")}
          {devPath ? <em> · {String(devPath).replace(/_/g, " ")}</em> : null}
        </p>
        <p className={`${PREFIX}-pcard-timeline-meta`}>
          {[
            readinessLabel ? `Readiness: ${readinessLabel}` : null,
            arrivalLabel ? `NHL debut: ${arrivalLabel}` : null,
            potentialEtaLabel ? `Ceiling: ${potentialEtaLabel}` : null,
          ]
            .filter(Boolean)
            .join(" · ") || "Path updates as scouting deepens"}
        </p>
      </section>

      <section className={`${PREFIX}-pcard-public`}>
        <h4>Public Opinion</h4>
        <div className={`${PREFIX}-pcard-public-grid`}>
          {pub != null ? <div><span>Draft Rank</span><strong>#{pub}</strong></div> : null}
          {currentPickOverall != null ? (
            <div><span>Selected</span><strong>#{currentPickOverall}</strong></div>
          ) : null}
          {reachSpots != null ? (
            <div>
              <span>Reach</span>
              <strong>
                {reachSpots > 0 ? `▼ ${reachSpots} spots` : reachSpots < 0 ? `▲ ${Math.abs(reachSpots)} spots` : "On the board"}
              </strong>
            </div>
          ) : null}
          {projectionConf != null ? (
            <div>
              <span>Scout Confidence</span>
              <div className={`${PREFIX}-pcard-conf-track`}>
                <i style={{ width: `${Math.max(0, Math.min(100, Math.round(projectionConf)))}%` }} />
              </div>
              <strong>{Math.round(projectionConf)}%</strong>
            </div>
          ) : null}
        </div>
        {decision ? (
          <p className={`${PREFIX}-pcard-decision-note`}>
            {decision === "Reach"
              ? "Your club believed he was worth taking despite the slide / board gap."
              : decision === "Value"
                ? "Falls into a value pocket versus the public board."
                : "Aligns with consensus board ranking."}
          </p>
        ) : null}
      </section>

      {(archetype || comparison.ceiling || comparison.floor || comparison.current) ? (
        <section className={`${PREFIX}-pcard-comps`}>
          <h4>Player Comparison</h4>
          <div className={`${PREFIX}-pcard-comp-grid`}>
            {(comparison.current || archetype) ? (
              <div><span>Current</span><strong>{comparison.current || archetype}</strong></div>
            ) : null}
            {comparison.ceiling ? <div><span>Ceiling</span><strong>{comparison.ceiling}</strong></div> : null}
            {comparison.floor ? <div><span>Floor</span><strong>{comparison.floor}</strong></div> : null}
          </div>
        </section>
      ) : null}

      <footer className={`${PREFIX}-pcard-foot`}>
        <button type="button" className={`${PREFIX}-text-btn`} onClick={() => onFullReport?.(prospect)}>
          Full report
        </button>
      </footer>
    </div>
  );
}

function DraftNightTweetCard({ tweet }) {
  if (!tweet) return null;
  const fan = tweet.fan || {};
  const metrics = tweet.metrics || {};
  const isHomage = Boolean(tweet.context?.isHomage || (tweet.context?.tags || []).includes?.("lupulHomage"));
  const initials = String(fan.displayName || "RW")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0] || "")
    .join("")
    .toUpperCase() || "RW";

  return (
    <article className={`edraft-pdsheet-tweet${isHomage ? " is-homage" : ""}`}>
      <div className="edraft-pdsheet-tweet-avatar" aria-hidden="true">
        {fan.avatarSrc ? (
          <img src={fan.avatarSrc} alt="" loading="lazy" />
        ) : (
          <span>{initials}</span>
        )}
      </div>
      <div className="edraft-pdsheet-tweet-body">
        <div className="edraft-pdsheet-tweet-meta">
          <strong>{fan.displayName || "Rink Watcher"}</strong>
          <span>{fan.handle || "@draftfloor"}</span>
          <span>{tweet.createdAtLabel || "now"}</span>
        </div>
        <p>{tweet.text}</p>
        <div className="edraft-pdsheet-tweet-foot">
          {tweet.awardLabel ? <span>{tweet.awardLabel}</span> : null}
          {tweet.context?.selectionLabel ? <span>{tweet.context.selectionLabel}</span> : null}
          {isHomage ? <span className="is-homage-tag">Floor Homage</span> : null}
          {metrics.likes != null ? <span>{metrics.likes} likes</span> : null}
        </div>
      </div>
    </article>
  );
}

function PickDetailSheet({ pick, reactionTweet, onClose }) {
  useEffect(() => {
    if (!pick) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pick, onClose]);

  if (!pick) return null;

  const name = pick.prospect_name || getPlayerName(pick);
  const pos = pick.position || getPlayerPosition(pick);
  const league = pick.league || pick.league_name || null;
  const nationality = pick.nationality || null;
  const natCode = pick.country_code || pick.nat || nationality;
  const teamName = pick.team_name || null;
  const overall = num(pick.overall_pick);
  const round = num(pick.round);
  const pickInRound = num(pick.pick_in_round);
  const pub = getPublicRankAtPick(pick);
  const pre = getPreseasonRank(pick);
  const movement = describePickMovement(pick);
  const { tag, tone } = getPickDisplayTag(pick);
  const risk = pick.risk_score ?? getRisk(pick);
  const conf = formatConfidence(pick);
  const readiness = pick.nhl_readiness || formatNhlReadiness(pick);
  const eta = formatNhlPotentialEta(pick) || getNhlEta(pick);
  const role = pick.player_type || getDefiningTrait(pick);
  const potGrade = pick.potential_grade || null;
  const floor = num(pick.floor_grade);
  const ceiling = num(pick.ceiling_grade);
  const age = num(pick.age);
  const eligYear = num(pick.draft_eligibility_year);
  const shoots = pick.shoots || pick.handedness || null;
  const stockLabel = normalizeStockLabel(pick);
  const stockReason = pick.stock_reason || null;
  const valueDelta = getPickValueDelta(pick);
  const tags = safeArray(pick.pick_tags).filter(Boolean);
  const rights = pick.rights_card || {};
  const viaLine = pick.is_traded && pick.via_team_name ? `via ${pick.via_team_name}` : null;

  const profileStats = [
    age != null ? { label: "Age", value: age } : null,
    eligYear != null ? { label: "Draft Eligibility", value: eligYear } : null,
    shoots ? { label: "Shoots", value: shoots } : null,
    pos ? { label: "Position", value: pos } : null,
    league ? { label: "League", value: league } : null,
    nationality ? { label: "Nationality", value: nationality } : null,
  ].filter(Boolean);

  const projStats = [
    floor != null ? { label: "Current OVR", value: Math.round(floor) } : null,
    ceiling != null ? { label: "Projected Ceiling", value: Math.round(ceiling), peak: true } : null,
    potGrade ? { label: "Talent Grade", value: potGrade } : null,
    role ? { label: "Projected Role", value: role } : null,
    readiness ? { label: "NHL Readiness", value: readiness } : null,
    eta ? { label: "NHL ETA", value: eta } : null,
    conf ? { label: "Scout Confidence", value: conf } : null,
    risk ? { label: "Risk", value: risk } : null,
  ].filter(Boolean);

  const decisionStats = [
    overall != null ? { label: "Overall Pick", value: `#${overall}` } : null,
    round != null ? { label: "Round", value: pickInRound != null ? `${round} · #${pickInRound}` : `${round}` } : null,
    pub != null ? { label: "Public Board", value: `#${pub}` } : null,
    pre != null ? { label: "Preseason", value: `#${pre}` } : null,
    valueDelta != null ? { label: "Value vs Slot", value: `${valueDelta > 0 ? "+" : ""}${valueDelta}` } : null,
    movement?.label ? { label: "Movement", value: movement.label } : null,
  ].filter(Boolean);

  const rightsStats = [
    rights.rights_through != null ? { label: "Rights Through", value: rights.rights_through } : null,
    rights.rights_status ? { label: "Rights Status", value: rights.rights_status } : null,
    rights.rights_type ? { label: "Rights Type", value: rights.rights_type } : null,
    rights.returning_to || pick.development_path ? { label: "Returning To", value: rights.returning_to || pick.development_path } : null,
    rights.expected_role ? { label: "Org Role", value: rights.expected_role } : null,
    rights.organizational_status ? { label: "Org Status", value: rights.organizational_status } : null,
    rights.elc_decision ? { label: "ELC", value: rights.elc_decision } : null,
    (rights.eta != null) ? { label: "Rights ETA", value: `${rights.eta}y` } : null,
    rights.rights_signing_deadline ? { label: "Signing Deadline", value: rights.rights_signing_deadline } : null,
  ].filter(Boolean);

  const flagIso = resolveCountryCode(natCode || nationality);
  const flagSrc = flagIso ? `https://flagcdn.com/w160/${flagIso.toLowerCase()}.png` : null;

  return createPortal(
    <div className="edraft-pdsheet-backdrop" role="presentation" onClick={onClose}>
      <div
        className="edraft-pdsheet"
        role="dialog"
        aria-modal="true"
        aria-label={`${name} draft profile`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="edraft-pdsheet-head">
          <div className="edraft-pdsheet-id">
            {flagSrc ? (
              <img className="edraft-pdsheet-flag" src={flagSrc} alt={nationality ? `${nationality} flag` : ""} loading="lazy" />
            ) : null}
            <div className="edraft-pdsheet-idtext">
              <h3>{name}</h3>
              <p>{[pos, league, nationality].filter(Boolean).join(" · ") || "—"}</p>
              <p className="edraft-pdsheet-pickline">
                {[overall != null ? `Pick #${overall}` : null, teamName, viaLine].filter(Boolean).join(" · ")}
              </p>
            </div>
          </div>
          <div className="edraft-pdsheet-head-right">
            {tag ? <PickBadge tag={tag} tone={tone} /> : null}
            <button type="button" className="edraft-pdsheet-close" onClick={onClose} aria-label="Close">×</button>
          </div>
        </header>

        <div className="edraft-pdsheet-body">
          {reactionTweet ? (
            <section className="edraft-pdsheet-sec">
              <h4>Draft Night Reaction</h4>
              <DraftNightTweetCard tweet={reactionTweet} />
            </section>
          ) : null}

          {profileStats.length ? (
            <section className="edraft-pdsheet-sec">
              <h4>Player Profile</h4>
              <div className="edraft-pdsheet-grid">
                {profileStats.map((s) => (
                  <div key={s.label} className="edraft-pdsheet-stat">
                    <span>{s.label}</span>
                    <strong>{s.value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {projStats.length ? (
            <section className="edraft-pdsheet-sec">
              <h4>Scouting &amp; Projection</h4>
              <div className="edraft-pdsheet-grid">
                {projStats.map((s) => (
                  <div key={s.label} className={`edraft-pdsheet-stat${s.peak ? " is-peak" : ""}`}>
                    <span>{s.label}</span>
                    <strong>{s.value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {decisionStats.length ? (
            <section className="edraft-pdsheet-sec">
              <h4>Draft Value</h4>
              <div className="edraft-pdsheet-grid">
                {decisionStats.map((s) => (
                  <div key={s.label} className="edraft-pdsheet-stat">
                    <span>{s.label}</span>
                    <strong>{s.value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {pick.pick_reason ? (
            <section className="edraft-pdsheet-sec">
              <h4>Why This Pick</h4>
              <p>{pick.pick_reason}</p>
            </section>
          ) : null}

          {(getBackendWhyWorks(pick) || getBackendWhyFails(pick)) ? (
            <section className="edraft-pdsheet-sec">
              <h4>Analysis</h4>
              {getBackendWhyWorks(pick) ? <p className="edraft-pdsheet-pos">{getBackendWhyWorks(pick)}</p> : null}
              {getBackendWhyFails(pick) ? <p className="edraft-pdsheet-neg">{getBackendWhyFails(pick)}</p> : null}
            </section>
          ) : null}

          {(stockLabel || stockReason) ? (
            <section className="edraft-pdsheet-sec">
              <h4>Stock Trend</h4>
              <p>{[stockLabel, stockReason].filter(Boolean).join(" — ")}</p>
            </section>
          ) : null}

          {rightsStats.length ? (
            <section className="edraft-pdsheet-sec">
              <h4>Rights &amp; Development</h4>
              <div className="edraft-pdsheet-grid">
                {rightsStats.map((s) => (
                  <div key={s.label} className="edraft-pdsheet-stat">
                    <span>{s.label}</span>
                    <strong>{s.value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ) : null}

          {tags.length ? (
            <div className="edraft-pdsheet-tags">
              {tags.slice(0, 6).map((t) => <span key={t}>{t}</span>)}
            </div>
          ) : null}
        </div>
      </div>
    </div>,
    document.body
  );
}

function ComparisonTable({ prospects }) {
  const rows = safeArray(prospects).slice(0, 3);
  if (!rows.length) {
    return <div className={`${PREFIX}-compare-empty`}>Add up to 3 prospects to compare.</div>;
  }

  const compareRows = [
    ["Public rank", (p) => (getRank(p) != null ? `#${getRank(p)}` : null)],
    ["Team rank", (p) => (getTeamRank(p) != null ? `#${getTeamRank(p)}` : null)],
    ["Position", (p) => getPlayerPosition(p) || null],
    ["League", (p) => p.league || p.league_name || null],
    ["Risk", (p) => getRisk(p)],
    ["Confidence", (p) => formatConfidence(p)],
    ["Team fit", (p) => (getTeamFit(p) != null ? String(Math.round(getTeamFit(p))) : null)],
    ["NHL ETA", (p) => getNhlEta(p)],
    ["Comparable", (p) => getComparable(p)],
    ["Scout summary", (p) => getScoutSummary(p)],
  ].filter(([, render]) => rows.some((p) => render(p)));

  return (
    <div className={`${PREFIX}-compare-table-wrap`}>
      <table className={`${PREFIX}-compare-table`}>
        <thead>
          <tr>
            <th>Point</th>
            {rows.map((p) => (
              <th key={getId(p)}>
                <strong>{getPlayerName(p)}</strong>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {compareRows.map(([label, render]) => (
            <tr key={label}>
              <td>{label}</td>
              {rows.map((p) => (
                <td key={`${getId(p)}-${label}`}>{render(p) || ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UserPickModal({ open, prospects, compareIds, onCompare, onDraft, onClose, loading, currentPick, draft, onTradeDown }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("overview");
  const [confirm, setConfirm] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const filtered = useMemo(() => {
    let rows = safeArray(prospects);
    if (filter === "C") rows = rows.filter((p) => getPlayerPosition(p) === "C");
    else if (filter === "D") rows = rows.filter((p) => getPlayerPosition(p) === "D");
    else if (filter === "G") rows = rows.filter((p) => getPlayerPosition(p) === "G");
    else if (filter === "W") rows = rows.filter((p) => ["LW", "RW", "W"].includes(getPlayerPosition(p)));
    else if (filter === "rising") {
      rows = rows.filter((p) => num(p.rank_movement ?? p.stock_delta, 0) > 0);
    } else if (filter === "falling") {
      rows = rows.filter((p) => num(p.rank_movement ?? p.stock_delta, 0) < 0);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((p) => {
        const name = String(getPlayerName(p) || "").toLowerCase();
        const league = String(p.league || p.dossier?.league || "").toLowerCase();
        const country = String(p.nationality || p.country_code || "").toLowerCase();
        return name.includes(q) || league.includes(q) || country.includes(q) || String(getPlayerPosition(p)).toLowerCase().includes(q);
      });
    }
    return rows;
  }, [prospects, filter, search]);

  const detail = selected || filtered[0] || null;
  const dossier = detail?.dossier || detail || {};
  const compareProspects = safeArray(prospects).filter((p) => compareIds.includes(getId(p))).slice(0, 4);
  const strengths = safeArray(dossier.strengthsEvidence?.length ? dossier.strengthsEvidence : dossier.strengths).slice(0, 3);
  const concerns = safeArray(dossier.weaknessesEvidence?.length ? dossier.weaknessesEvidence : dossier.concerns).slice(0, 3);
  const fit = dossier.teamFit || dossier.team_fit || null;

  useEffect(() => {
    if (!open) {
      setConfirm(false);
      setSubmitting(false);
      setTab("overview");
    }
  }, [open]);

  if (!open) return null;

  const handleDraft = async () => {
    if (!detail || submitting || loading) return;
    setSubmitting(true);
    try {
      await onDraft(detail);
    } finally {
      setSubmitting(false);
      setConfirm(false);
    }
  };

  const ovrBand = () => {
    const lo = dossier.overallRangeLow ?? detail?.overall_range_low;
    const hi = dossier.overallRangeHigh ?? detail?.overall_range_high;
    if (lo != null && hi != null) return `${lo}–${hi}`;
    return formatCurrentOvr(detail) || "Unavailable";
  };
  const potBand = () => {
    if (dossier.ceilingHidden || detail?.ceiling_hidden) {
      return "Fogged";
    }
    const lo = dossier.scoutedPotentialLow;
    const hi = dossier.scoutedPotentialHigh;
    if (lo != null && hi != null) return `${lo}–${hi}`;
    const pr = dossier.potential_range || detail?.potential_range;
    if (Array.isArray(pr) && pr.length >= 2) return `${pr[0]}–${pr[1]}`;
    return "Unavailable";
  };

  return (
    <div className={`${PREFIX}-modal-backdrop`} role="dialog" aria-modal="true" aria-label="Make selection">
      <div className={`${PREFIX}-modal ${PREFIX}-modal--decision`}>
        <header>
          <div>
            <h2>Make Selection</h2>
            <p>
              {[
                formatPick(currentPick?.overall_pick || draft?.overall_pick),
                draft?.current_team_name,
                currentPick?.round ? `Rd ${currentPick.round}` : null,
              ].filter(Boolean).join(" · ")}
            </p>
          </div>
          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={onClose} aria-label="Close">Close</button>
          {typeof onTradeDown === "function" ? (
            <button
              type="button"
              className={`${PREFIX}-ghost-btn`}
              onClick={() => {
                onClose?.();
                onTradeDown();
              }}
            >
              Trade Down
            </button>
          ) : null}
        </header>

        <div className={`${PREFIX}-modal-filters`}>
          {["all", "C", "W", "D", "G", "rising", "falling"].map((f) => (
            <button key={f} type="button" className={filter === f ? "is-active" : ""} onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f === "rising" ? "Rising" : f === "falling" ? "Falling" : f}
            </button>
          ))}
          <input
            type="search"
            className={`${PREFIX}-input`}
            placeholder="Search name, league, country"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search prospects"
          />
        </div>

        <div className={`${PREFIX}-modal-body`}>
          <div className={`${PREFIX}-modal-list`} role="listbox" aria-label="Available prospects">
            {filtered.map((p) => {
              const d = p.dossier || {};
              const move = num(p.rank_movement ?? p.stock_delta ?? d.rankMovement);
              return (
                <button
                  type="button"
                  key={getId(p)}
                  role="option"
                  aria-selected={getId(selected || detail) === getId(p)}
                  className={`${PREFIX}-modal-row${getId(selected || detail) === getId(p) ? " is-selected" : ""}`}
                  onClick={() => { setSelected(p); setConfirm(false); setTab("overview"); }}
                >
                  <ProspectFlag country={p.nationality || d.nationality} code={p.country_code || d.country_code || d.countryCode} />
                  <span className={`${PREFIX}-tabular`}>#{getRank(p) ?? "—"}</span>
                  {getTeamRank(p) != null ? <span className={`${PREFIX}-tabular ${PREFIX}-muted`}>T{getTeamRank(p)}</span> : <span />}
                  <strong>{getPlayerName(p)}</strong>
                  <span>{getPlayerPosition(p)}</span>
                  <span className={`${PREFIX}-modal-row-league`}>{p.league || d.league || d.club || ""}</span>
                  {move != null && move !== 0 ? (
                    <span className={`${PREFIX}-stock ${move > 0 ? "is-up" : "is-down"}`}>
                      {move > 0 ? `+${move}` : move}
                    </span>
                  ) : <span />}
                </button>
              );
            })}
          </div>

          <div className={`${PREFIX}-modal-detail`}>
            {detail ? (
              <>
                <div className={`${PREFIX}-modal-detail-head`}>
                  <ProspectFlag country={detail.nationality || dossier.nationality} code={detail.country_code || dossier.countryCode} size={22} />
                  <div>
                    <h3>{getPlayerName(detail)}</h3>
                    <p>
                      {[
                        getPlayerPosition(detail),
                        detail.handedness || dossier.handedness,
                        detail.age || dossier.age ? `Age ${detail.age || dossier.age}` : null,
                        dossier.heightDisplay || detail.height,
                        detail.weight ? `${detail.weight} lb` : null,
                      ].filter(Boolean).join(" · ")}
                    </p>
                    <p className={`${PREFIX}-muted`}>
                      {[dossier.club || detail.team, dossier.league || detail.league].filter(Boolean).join(" · ")}
                    </p>
                  </div>
                </div>

                <div className={`${PREFIX}-modal-tabs`} role="tablist">
                  {["overview", "stats", "projection", "fit", "compare"].map((t) => (
                    <button
                      key={t}
                      type="button"
                      role="tab"
                      aria-selected={tab === t}
                      className={tab === t ? "is-active" : ""}
                      onClick={() => setTab(t)}
                    >
                      {t === "fit" ? "Team Fit" : t[0].toUpperCase() + t.slice(1)}
                    </button>
                  ))}
                </div>

                <div className={`${PREFIX}-modal-detail-body`}>
                  {tab === "overview" ? (
                    <>
                      <div className={`${PREFIX}-sheet-facts`}>
                        {getRank(detail) != null ? <span>Public #{getRank(detail)}</span> : null}
                        {getTeamRank(detail) != null ? <span>Team #{getTeamRank(detail)}</span> : null}
                        {dossier.preseasonRank || detail.preseason_rank ? (
                          <span>Preseason #{dossier.preseasonRank || detail.preseason_rank}</span>
                        ) : null}
                        {formatConfidence(detail) ? <span>Conf {formatConfidence(detail)}</span> : null}
                        {getRisk(detail) ? <span>Risk {getRisk(detail)}</span> : null}
                      </div>
                      <div className={`${PREFIX}-ability-grid`}>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Current ability</span>
                          <strong>{ovrBand()}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Projected ceiling</span>
                          <strong>{potBand()}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>NHL probability</span>
                          <strong>
                            {dossier.nhlProbability != null
                              ? `${Math.round(dossier.nhlProbability)}%`
                              : dossier.potential?.probability != null
                                ? `${Math.round(dossier.potential.probability)}%`
                                : "Unavailable"}
                          </strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Profile</span>
                          <strong>{dossier.developmentProfile || "Unavailable"}</strong>
                        </div>
                      </div>
                      {(strengths.length || concerns.length) ? (
                        <div className={`${PREFIX}-evidence-cols`}>
                          {strengths.length ? (
                            <div>
                              <h4>Strengths</h4>
                              <ul>
                                {strengths.map((s, i) => (
                                  <li key={i}>{typeof s === "string" ? s : `${s.title} — ${s.fact}`}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                          {concerns.length ? (
                            <div>
                              <h4>Concerns</h4>
                              <ul>
                                {concerns.map((s, i) => (
                                  <li key={i}>{typeof s === "string" ? s : `${s.title} — ${s.fact}`}</li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                        </div>
                      ) : (
                        <p className={`${PREFIX}-muted`}>No evidence-based notes available.</p>
                      )}
                    </>
                  ) : null}

                  {tab === "stats" ? (
                    <div className={`${PREFIX}-stats-block`}>
                      {dossier.stats || detail.gp != null ? (
                        <>
                          <div className={`${PREFIX}-ability-grid`}>
                            <div>
                              <span className={`${PREFIX}-ability-label`}>Games</span>
                              <strong>{dossier.stats?.games ?? detail.gp ?? "—"}</strong>
                            </div>
                            <div>
                              <span className={`${PREFIX}-ability-label`}>G–A–P</span>
                              <strong>
                                {dossier.stats?.goals ?? detail.goals ?? "—"}–
                                {dossier.stats?.assists ?? detail.assists ?? "—"}–
                                {dossier.stats?.points ?? detail.points ?? "—"}
                              </strong>
                            </div>
                            <div>
                              <span className={`${PREFIX}-ability-label`}>P/GP</span>
                              <strong>
                                {dossier.stats?.ppg != null
                                  ? Number(dossier.stats.ppg).toFixed(2)
                                  : (num(detail.gp) > 0 && num(detail.points) != null
                                    ? (num(detail.points) / num(detail.gp)).toFixed(2)
                                    : "—")}
                              </strong>
                            </div>
                            <div>
                              <span className={`${PREFIX}-ability-label`}>League</span>
                              <strong>{dossier.league || detail.league || "—"}</strong>
                            </div>
                            {dossier.analytics?.war != null || dossier.stats?.analytics?.war != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>WAR</span>
                                <strong>
                                  {Number(dossier.analytics?.war ?? dossier.stats?.analytics?.war).toFixed(2)}
                                </strong>
                              </div>
                            ) : null}
                            {dossier.analytics?.shot_rate != null || dossier.stats?.analytics?.shot_rate != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>Shots/G</span>
                                <strong>
                                  {Number(dossier.analytics?.shot_rate ?? dossier.stats?.analytics?.shot_rate).toFixed(2)}
                                </strong>
                              </div>
                            ) : null}
                            {dossier.analytics?.shooting_pct != null || dossier.stats?.analytics?.shooting_pct != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>SH%</span>
                                <strong>
                                  {Number(dossier.analytics?.shooting_pct ?? dossier.stats?.analytics?.shooting_pct).toFixed(1)}%
                                </strong>
                              </div>
                            ) : null}
                            {dossier.analytics?.primary_points != null || dossier.stats?.analytics?.primary_points != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>Primary P</span>
                                <strong>
                                  {dossier.analytics?.primary_points ?? dossier.stats?.analytics?.primary_points}
                                </strong>
                              </div>
                            ) : null}
                            {dossier.competition?.adjustment != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>Translation</span>
                                <strong>
                                  {dossier.competition?.label
                                    || (Number(dossier.competition.adjustment) >= 1.1
                                      ? "Strong"
                                      : Number(dossier.competition.adjustment) >= 0.85
                                        ? "Avg"
                                        : "Risk")}
                                </strong>
                              </div>
                            ) : null}
                            {dossier.stats?.toi != null || dossier.analytics?.toi != null ? (
                              <div>
                                <span className={`${PREFIX}-ability-label`}>Est. TOI</span>
                                <strong>{Number(dossier.stats?.toi ?? dossier.analytics?.toi).toFixed(1)}</strong>
                              </div>
                            ) : null}
                          </div>
                          {dossier.sampleNote || dossier.translationNote ? (
                            <p className={`${PREFIX}-muted`}>
                              {[dossier.sampleNote, dossier.translationNote].filter(Boolean).join(" · ")}
                            </p>
                          ) : null}
                        </>
                      ) : (
                        <p className={`${PREFIX}-muted`}>Season stats unavailable.</p>
                      )}
                      {dossier.playoffStats?.games ? (
                        <p>
                          Playoffs: {dossier.playoffStats.games} GP · {dossier.playoffStats.points ?? "—"} P
                          {dossier.playoffStats.ppg != null ? ` · ${dossier.playoffStats.ppg} PPG` : ""}
                        </p>
                      ) : null}
                      {dossier.wjcStats?.played || dossier.wjcStats?.games ? (
                        <p>
                          WJC: {dossier.wjcStats.games} GP · {dossier.wjcStats.points ?? "—"} P
                          {dossier.wjcStats.ppg != null ? ` · ${dossier.wjcStats.ppg} PPG` : ""}
                          {dossier.wjcStats.team ? ` · ${dossier.wjcStats.team}` : ""}
                        </p>
                      ) : null}
                    </div>
                  ) : null}

                  {tab === "projection" ? (
                    <div className={`${PREFIX}-stats-block`}>
                      <div className={`${PREFIX}-ability-grid`}>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Current ability</span>
                          <strong>{ovrBand()}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Ceiling</span>
                          <strong>
                            {dossier.ceilingHidden
                              ? (dossier.ceilingHint ? "Fogged" : "Ungraded")
                              : potBand()}
                          </strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>
                            {dossier.ceilingHidden ? "NHL odds (prod.)" : "NHL probability"}
                          </span>
                          <strong>
                            {dossier.nhlProbability != null
                              ? `${Math.round(dossier.nhlProbability)}%`
                              : "Unavailable"}
                          </strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Ceiling likelihood</span>
                          <strong>{dossier.ceilingLikelihood || "Unavailable"}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Volatility</span>
                          <strong>{dossier.developmentVolatility || "Unavailable"}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>ETA</span>
                          <strong>
                            {dossier.estimatedNhlArrival
                              || dossier.eta?.label
                              || formatNhlArrivalEta(detail)
                              || getNhlEta(detail)
                              || "Unavailable"}
                          </strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Readiness</span>
                          <strong>
                            {formatNhlReadiness(detail)
                              || dossier.readinessLabel
                              || "Unavailable"}
                          </strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>
                            Role{dossier.projection?.based_on === "floor" ? " (floor)" : ""}
                          </span>
                          <strong>{dossier.projection?.label || "Unavailable"}</strong>
                        </div>
                        <div>
                          <span className={`${PREFIX}-ability-label`}>Profile</span>
                          <strong>{dossier.developmentProfile || "Unavailable"}</strong>
                        </div>
                      </div>
                      {dossier.ceilingHidden ? (
                        <p className={`${PREFIX}-muted`}>
                          {dossier.ceilingHint
                            || "True ceiling is ungraded for late-round attention — use production, age, size, and league context."}
                        </p>
                      ) : null}
                      {safeArray(dossier.projectionNotes).map((n, i) => (
                        <p key={i}>{typeof n === "string" ? n : `${n.title} — ${n.fact}`}</p>
                      ))}
                    </div>
                  ) : null}

                  {tab === "fit" ? (
                    <div className={`${PREFIX}-stats-block`}>
                      {fit ? (
                        <>
                          <p>{fit.label || "Fit"}{fit.score != null ? ` · ${Math.round(fit.score)}` : ""}</p>
                          {safeArray(fit.reasons || fit.fit_strengths).map((r, i) => (
                            <p key={`r-${i}`}>{r}</p>
                          ))}
                          {safeArray(fit.fit_concerns).map((r, i) => (
                            <p key={`c-${i}`} className={`${PREFIX}-muted`}>{r}</p>
                          ))}
                        </>
                      ) : (
                        <p className={`${PREFIX}-muted`}>Team fit unavailable.</p>
                      )}
                    </div>
                  ) : null}

                  {tab === "compare" ? (
                    <ComparisonTable prospects={compareProspects.length ? compareProspects : [detail]} />
                  ) : null}
                </div>

                <div className={`${PREFIX}-sheet-actions`}>
                  <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => onCompare(detail)}>
                    {compareIds.includes(getId(detail)) ? "Remove Compare" : "Add to Compare"}
                  </button>
                  {confirm ? (
                    <>
                      <span className={`${PREFIX}-confirm-q`}>Confirm selection?</span>
                      <button
                        type="button"
                        className={`${PREFIX}-cta-btn`}
                        disabled={loading || submitting}
                        onClick={handleDraft}
                      >
                        {loading || submitting ? LOADING_COPY.submit : `Draft ${getPlayerName(detail)}`}
                      </button>
                      <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => setConfirm(false)}>Back</button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className={`${PREFIX}-cta-btn`}
                      disabled={loading || submitting}
                      onClick={() => setConfirm(true)}
                    >
                      Draft {getPlayerName(detail)}
                    </button>
                  )}
                </div>
              </>
            ) : (
              <p className={`${PREFIX}-muted`}>Select a prospect.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CpuReveal({ pick, onDone, fast = false }) {
  useEffect(() => {
    const r = Number(pick?.round) || 1;
    const pace = fast ? 400 : r === 1 ? 2200 : r <= 3 ? 1400 : 700;
    const t = window.setTimeout(onDone, pace);
    return () => window.clearTimeout(t);
  }, [pick, onDone, fast]);

  if (!pick) return null;
  const pub = getPublicRankAtPick(pick);
  const movement = describePickMovement(pick);
  const { tag, tone } = getPickDisplayTag(pick);

  return (
    <div className={`${PREFIX}-cpu-reveal`}>
      <p className={`${PREFIX}-cpu-label`}>With the {formatPick(pick.overall_pick)} pick</p>
      <TeamLogo teamId={pick.team_id} teamName={pick.team_name} size="lg" />
      <h2>{pick.team_name}</h2>
      {pick.is_traded && (pick.via_team_name || pick.via_team_id) ? (
        <p className={`${PREFIX}-via-note`}>
          <TeamLogo
            teamId={pick.via_team_id || pick.original_owner_team_id}
            teamName={pick.via_team_name || pick.original_owner_team_name}
            size="xs"
          />
          via {teamAbbrev(pick.via_team_id, pick.via_team_name)}
        </p>
      ) : null}
      <p className={`${PREFIX}-cpu-selects`}>selects</p>
      <h1>{pick.prospect_name}</h1>
      <div className={`${PREFIX}-cpu-meta`}>
        {pick.position ? <span>{pick.position}</span> : null}
        {pick.league ? <span>{pick.league}</span> : null}
        {pub != null ? <span>Listed #{pub}</span> : null}
        {movement ? <span>{movement.label}</span> : null}
        {tag ? <PickBadge tag={tag} tone={tone} /> : null}
      </div>
      {pick.pick_reason ? <p className={`${PREFIX}-cpu-reason`}>{pick.pick_reason}</p> : null}
    </div>
  );
}

function RoundRecapPanel({ recap, onContinue }) {
  if (!recap) return null;
  return (
    <div className={`${PREFIX}-round-recap`}>
      <h3>Round {recap.round} Recap</h3>
      {recap.headline ? <p>{recap.headline}</p> : null}
      {recap.user_picks?.length ? (
        <div>
          {recap.user_picks.map((p, i) => (
            <p key={i}>{formatPick(p.overall_pick)} {p.prospect_name}</p>
          ))}
        </div>
      ) : null}
      <button type="button" className={`${PREFIX}-cta-btn`} onClick={onContinue}>Continue Draft</button>
    </div>
  );
}

function TradePanel({ draft, onClose, onAccept, accepting = false }) {
  const offers = safeArray(draft?.trade_offers || draft?.draft_day_trade_offers || draft?.pick_trade_offers);
  return (
    <div className={`${PREFIX}-modal-backdrop`} role="dialog" aria-modal="true" aria-label="Trade down">
      <div className={`${PREFIX}-trade-panel`}>
        <header>
          <h3>Trade Down</h3>
          <button
            type="button"
            className={`${PREFIX}-ghost-btn`}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onClose?.();
            }}
          >
            Close
          </button>
        </header>
        {!offers.length ? (
          <p className={`${PREFIX}-muted`}>
            No clubs are paying to climb right now. Only teams with a real board
            priority still on the board will bid — keep the pick or make your selection.
          </p>
        ) : (
          offers.map((offer, i) => {
            const candidates = safeArray(offer.target_candidates);
            const incoming = safeArray(
              offer.incoming_assets?.length
                ? offer.incoming_assets
                : String(offer.assets_in || "")
                    .split(/\s*[·+]\s*/)
                    .map((s) => s.trim())
                    .filter(Boolean)
            );
            const outgoing = safeArray(
              offer.outgoing_assets?.length
                ? offer.outgoing_assets
                : [offer.on_clock_overall_pick ? `#${offer.on_clock_overall_pick} pick` : "On-clock pick"]
            );
            return (
              <article key={i} className={`${PREFIX}-trade-offer`}>
                <strong>{offer.team_name || offer.from_team_name || "Team"}</strong>
                <div className={`${PREFIX}-trade-offer-block`}>
                  <span className={`${PREFIX}-trade-offer-label`}>You send</span>
                  <ul>
                    {outgoing.map((asset, idx) => (
                      <li key={`out-${idx}`}>{typeof asset === "string" ? asset : String(asset)}</li>
                    ))}
                  </ul>
                </div>
                <div className={`${PREFIX}-trade-offer-block`}>
                  <span className={`${PREFIX}-trade-offer-label`}>You receive</span>
                  {incoming.length ? (
                    <ul>
                      {incoming.map((asset, idx) => (
                        <li key={`in-${idx}`}>{typeof asset === "string" ? asset : String(asset)}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className={`${PREFIX}-muted`}>No package attached</p>
                  )}
                </div>
                <div className={`${PREFIX}-trade-offer-block`}>
                  <span className={`${PREFIX}-trade-offer-label`}>Rumored targets</span>
                  {candidates.length ? (
                    <ul>
                      {candidates.map((c, idx) => (
                        <li key={c.prospect_id || idx}>
                          {c.name || "Prospect"}
                          {c.position ? ` (${c.position})` : ""}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className={`${PREFIX}-muted`}>Unknown</p>
                  )}
                  <p className={`${PREFIX}-muted`}>Only one name is their true priority.</p>
                </div>
                <div className={`${PREFIX}-trade-offer-meta`}>
                  {offer.partner_overall_pick ? <span>Move to #{offer.partner_overall_pick}</span> : null}
                  {offer.slot_value_gap != null ? (
                    <span>Chart gap {Number(offer.slot_value_gap).toFixed(1)}</span>
                  ) : null}
                  {offer.value || offer.value_grade ? <span>{offer.value || offer.value_grade}</span> : null}
                </div>
                <button
                  type="button"
                  className={`${PREFIX}-cta-btn`}
                  disabled={accepting || !incoming.length}
                  onClick={() => onAccept?.(offer)}
                >
                  {accepting ? "Accepting…" : "Accept & trade down"}
                </button>
              </article>
            );
          })
        )}
      </div>
    </div>
  );
}

function buildDraftGrade(userPicks) {
  const picks = safeArray(userPicks);
  if (!picks.length) return null;
  let score = 75;
  for (const p of picks) {
    if (p.pick_classification === "Steal" || p.was_steal) score += 5;
    if (p.pick_classification === "Reach" || p.was_reach) score -= 5;
    if (p.pick_classification === "BPA") score += 3;
    if (p.pick_classification === "Need Pick") score += 2;
    if (p.pick_classification === "Goalie Gamble") score -= 2;
  }
  score = Math.max(0, Math.min(100, score));
  if (score >= 94) return "A+";
  if (score >= 90) return "A";
  if (score >= 86) return "A-";
  if (score >= 82) return "B+";
  if (score >= 78) return "B";
  if (score >= 74) return "B-";
  if (score >= 70) return "C+";
  if (score >= 65) return "C";
  if (score >= 60) return "C-";
  if (score >= 50) return "D";
  return "F";
}

function clip15(text) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean);
  return words.slice(0, 15).join(" ");
}

const RECAP_POS_COLORS = {
  C: "#4fb0ff",
  LW: "#6ce5b0",
  RW: "#ffd166",
  D: "#c792ea",
  G: "#ff7b72",
  W: "#6ce5b0",
};

// Value vs PUBLIC board only (item 19). Positive = fell to you; negative = reach.
function pubDelta(pick) {
  let d = pick?.public_rank_delta;
  if (d == null) d = pick?.public_board_delta;
  if (d == null) {
    const o = Number(pick?.overall_pick || 0);
    const f = Number(pick?.final_rank || 0);
    d = o && f ? o - f : null;
  }
  return d == null || Number.isNaN(Number(d)) ? null : Math.round(Number(d));
}

function fmtSigned(v, digits = 0) {
  if (v == null || Number.isNaN(Number(v))) return null;
  const n = Number(v);
  const s = digits ? n.toFixed(digits) : String(Math.round(n));
  return n > 0 ? `+${s}` : s;
}

function numOrDash(v) {
  return v == null || v === "" || Number.isNaN(Number(v)) ? "—" : Math.round(Number(v));
}

function ovrVal(p) {
  return numOrDash(p?.floor_grade ?? p?.true_ovr ?? p?.scouted_ovr);
}

function potVal(p) {
  const v = p?.ceiling_grade ?? p?.potential_score;
  if (v != null && !Number.isNaN(Number(v))) return Math.round(Number(v));
  return p?.potential_grade || "—";
}

function projVal(p) {
  const raw = p?.player_type || p?.development_path || p?.nhl_readiness;
  if (raw) return String(raw);
  return p?.nhl_eta != null ? `${p.nhl_eta}y` : "—";
}

// Deterministic per-team accent derived from the team identity (styling only).
function teamAccent(team) {
  const abbr = String(getTeamAbbreviation(team) || "").toUpperCase();
  if (!abbr) return "var(--edraft-gold)";
  let hash = 0;
  for (let i = 0; i < abbr.length; i += 1) hash = abbr.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 68% 54%)`;
}

function RecapMetric({ label, value, hint, tone = "neutral" }) {
  return (
    <div className={`${PREFIX}-rc-metric ${PREFIX}-rc-metric--${tone}`} title={hint || label}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function StorylineItem({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      type="button"
      className={`${PREFIX}-rc-story ${open ? "is-open" : ""}`}
      onClick={() => setOpen((o) => !o)}
      title={open ? "Collapse" : "Expand"}
    >
      <span className={`${PREFIX}-rc-story-dot`} aria-hidden="true" />
      <span className={`${PREFIX}-rc-story-text`}>{text}</span>
      <span className={`${PREFIX}-rc-story-caret`} aria-hidden="true">{open ? "−" : "+"}</span>
    </button>
  );
}

function RecapValueRow({ pick, tone, onOpen }) {
  const d = pubDelta(pick);
  const abbr = teamAbbrev(pick.team_id, pick.team_name);
  return (
    <div
      className={`${PREFIX}-rc-row ${PREFIX}-rc-row--click`}
      role="button"
      tabIndex={0}
      title={`${pick.prospect_name} — view full profile`}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen?.(); } }}
    >
      <span className={`${PREFIX}-rc-row-main`}>
        {abbr ? <b className={`${PREFIX}-rc-row-team`}>{abbr}</b> : null} {formatPick(pick.overall_pick)} {pick.prospect_name}
      </span>
      <span className={`${PREFIX}-rc-row-side`}>
        {pick.final_rank != null ? <span className={`${PREFIX}-rc-row-board`} title="Public board rank">Brd #{pick.final_rank}</span> : null}
        {d != null ? <span className={`${PREFIX}-rc-row-delta ${tone}`}>{fmtSigned(d)}</span> : null}
      </span>
    </div>
  );
}

function RecapShow({ draft, completed, userPicks }) {
  const r = draft?.recap || {};
  const [profilePick, setProfilePick] = useState(null);
  const picks = safeArray(userPicks).slice().sort((a, b) => Number(a.overall_pick || 0) - Number(b.overall_pick || 0));

  const teamId = picks[0]?.team_id ?? picks[0]?.team;
  const teamName = picks[0]?.team_name;
  const teamObj = { id: teamId, team_id: teamId, name: teamName, team_name: teamName };
  const logo = getTeamLogoSrc(teamObj);
  const accent = teamAccent(teamObj);

  const grade = r.user_grade || draft?.user_draft_grade || buildDraftGrade(picks) || "—";
  const score = r.user_grade_score;
  const summary = r.grade_summary ? clip15(r.grade_summary) : "";
  const posBreakdown = r.user_position_breakdown || r.position_breakdown || {};

  const steals = safeArray(r.best_steals || safeArray(completed).filter((p) => p.was_steal)).slice(0, 5);
  const reaches = safeArray(r.biggest_reaches || safeArray(completed).filter((p) => p.was_reach)).slice(0, 5);
  const headlines = safeArray(r.headlines).slice(0, 6);
  const needsReport = safeArray(r.needs_report);
  const stillOnBoard = safeArray(r.still_on_board);
  const bestId = String(r.best_user_pick?.prospect_id ?? "");

  const avgRaw = r.user_avg_value_delta != null
    ? Number(r.user_avg_value_delta)
    : (picks.length ? picks.reduce((s, p) => s + (pubDelta(p) || 0), 0) / picks.length : 0);
  const avgDisplay = fmtSigned(avgRaw, 1) ?? "0.0";

  const dnaEntries = Object.entries(posBreakdown).filter(([, v]) => Number(v) > 0);
  const dnaTotal = dnaEntries.reduce((s, [, v]) => s + Number(v || 0), 0);

  return (
    <section className={`${PREFIX}-recap-show`} style={{ "--rc-accent": accent }}>
      {/* Top band: grade focal + metrics + DNA + needs */}
      <div className={`${PREFIX}-rc-top`}>
        {/* 7-9: grade focal point with team logo + accent, no decorative stars */}
        <div className={`${PREFIX}-rc-grade`}>
          {logo ? <img className={`${PREFIX}-rc-logo`} src={logo} alt={teamName || "Team"} /> : null}
          <div className={`${PREFIX}-rc-grade-face`}>
            <span className={`${PREFIX}-rc-grade-kicker`}>Draft Grade</span>
            <strong className={`${PREFIX}-rc-grade-letter`}>{grade}</strong>
            <div className={`${PREFIX}-rc-grade-meta`}>
              {score != null ? <span title="Weighted grade score">{score}/100</span> : null}
              {r.user_class_rank ? <span title="Rank vs the other 31 clubs">{r.user_class_rank}</span> : null}
            </div>
          </div>
          {summary ? <p className={`${PREFIX}-rc-grade-sum`}>{summary}</p> : null}
        </div>

        <div className={`${PREFIX}-rc-side`}>
          {/* 10-12: prioritized metrics with tooltips + backend avg fallback */}
          <div className={`${PREFIX}-rc-metrics`}>
            <RecapMetric label="Picks" value={r.user_pick_count ?? picks.length} hint="Total selections you made" />
            <RecapMetric label="Steals" value={r.user_steal_count ?? 0} tone="good" hint="Fell 10+ spots past their public board rank" />
            <RecapMetric label="Reaches" value={r.user_reach_count ?? 0} tone="bad" hint="Taken 10+ spots above their public board rank" />
            <RecapMetric label="Value Adds" value={r.user_value_count ?? 0} tone="good" hint="Solid value relative to the public board" />
            <RecapMetric label="Needs Filled" value={r.user_need_count ?? 0} hint="Picks that addressed a roster need" />
            <RecapMetric label="Avg Value" value={avgDisplay} tone={avgRaw >= 0 ? "good" : "bad"} hint="Average public-board value gained per pick" />
          </div>

          <div className={`${PREFIX}-rc-midrow`}>
            {/* 13-14: readable Draft DNA with counts on the bar */}
            <div className={`${PREFIX}-rc-card`}>
              <div className={`${PREFIX}-rc-card-head`}><h4>Draft DNA</h4><span>{dnaTotal} picks</span></div>
              {dnaTotal ? (
                <>
                  <div className={`${PREFIX}-rc-dna-track`}>
                    {dnaEntries.map(([pos, v]) => {
                      const w = (Number(v) / dnaTotal) * 100;
                      return (
                        <div
                          key={pos}
                          className={`${PREFIX}-rc-dna-seg`}
                          style={{ width: `${w}%`, background: RECAP_POS_COLORS[pos] || "#9cb2c4" }}
                          title={`${pos}: ${v}`}
                        >
                          {w >= 12 ? <span>{pos} {v}</span> : null}
                        </div>
                      );
                    })}
                  </div>
                  <div className={`${PREFIX}-rc-dna-legend`}>
                    {dnaEntries.map(([pos, v]) => (
                      <span key={pos}><i style={{ background: RECAP_POS_COLORS[pos] || "#9cb2c4" }} />{pos} {v}</span>
                    ))}
                  </div>
                </>
              ) : (
                <p className={`${PREFIX}-rc-empty`}>No positional data.</p>
              )}
            </div>

            {/* 15: real roster-need results or a clear empty state */}
            <div className={`${PREFIX}-rc-card`}>
              <div className={`${PREFIX}-rc-card-head`}><h4>Needs Report</h4><span>{needsReport.length} tracked</span></div>
              {needsReport.length ? (
                <ul className={`${PREFIX}-rc-needs`}>
                  {needsReport.slice(0, 4).map((n, i) => (
                    <li key={i} className={n.filled ? "is-filled" : "is-open"} title={n.detail || n.category}>
                      <span className={`${PREFIX}-rc-need-mark`} aria-hidden="true">{n.filled ? "✓" : "•"}</span>
                      <span className={`${PREFIX}-rc-need-cat`}>{n.category}</span>
                      <span className={`${PREFIX}-rc-need-state`}>{n.filled ? "Addressed" : "Open"}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={`${PREFIX}-rc-empty`}>No pressing roster needs — depth is healthy.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 16-18, 20: wide draft class table, compact columns, highlighted best pick, value delta */}
      <div className={`${PREFIX}-rc-class`}>
        <div className={`${PREFIX}-rc-card-head`}><h4>Your Draft Class</h4><span>Click a player for their full profile</span></div>
        <div className={`${PREFIX}-rc-table`}>
          <div className={`${PREFIX}-rc-thead`}>
            <span>Rd</span><span>Pick</span><span className={`${PREFIX}-rc-c-name`}>Player</span>
            <span>Pos</span><span>OVR</span><span>POT</span><span className={`${PREFIX}-rc-c-proj`}>Projection</span><span>Value</span><span aria-hidden="true" />
          </div>
          <div className={`${PREFIX}-rc-tbody`}>
            {picks.length ? picks.map((p, i) => {
              const d = pubDelta(p);
              const isBest = bestId && String(p.prospect_id) === bestId;
              return (
                <div
                  key={i}
                  className={`${PREFIX}-rc-trow ${PREFIX}-rc-trow--click ${isBest ? "is-best" : ""}`}
                  role="button"
                  tabIndex={0}
                  title={`${p.prospect_name} — view full profile`}
                  onClick={() => setProfilePick(p)}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setProfilePick(p); } }}
                >
                  <span>{p.round}</span>
                  <span>{formatPick(p.overall_pick)}</span>
                  <span className={`${PREFIX}-rc-c-name`}>
                    {isBest ? <span className={`${PREFIX}-rc-best-mark`} aria-hidden="true">★</span> : null}
                    {p.prospect_name}
                  </span>
                  <span>{p.position || "—"}</span>
                  <span>{ovrVal(p)}</span>
                  <span>{potVal(p)}</span>
                  <span className={`${PREFIX}-rc-c-proj`}>{projVal(p)}</span>
                  <span className={d == null ? "" : d > 0 ? "pos" : d < 0 ? "neg" : ""}>{d == null ? "—" : fmtSigned(d)}</span>
                  <span className={`${PREFIX}-rc-trow-caret`} aria-hidden="true">›</span>
                </div>
              );
            }) : (
              <p className={`${PREFIX}-rc-empty`}>No picks were made.</p>
            )}
          </div>
        </div>
      </div>

      {/* Bottom band: steals / reaches / storylines / still on board */}
      <div className={`${PREFIX}-rc-bottom`}>
        <div className={`${PREFIX}-rc-card`}>
          <div className={`${PREFIX}-rc-card-head`}><h4>Best Steals</h4><span>Pick · Board</span></div>
          {steals.length ? steals.map((p, i) => (
            <RecapValueRow key={i} pick={p} tone="pos" onOpen={() => setProfilePick(p)} />
          )) : <p className={`${PREFIX}-rc-empty`}>No steals — the board stayed honest.</p>}
        </div>

        <div className={`${PREFIX}-rc-card`}>
          <div className={`${PREFIX}-rc-card-head`}><h4>Biggest Reaches</h4><span>Pick · Board</span></div>
          {reaches.length ? reaches.map((p, i) => (
            <RecapValueRow key={i} pick={p} tone="neg" onOpen={() => setProfilePick(p)} />
          )) : <p className={`${PREFIX}-rc-empty`}>No reaches — a disciplined night.</p>}
        </div>

        {/* 22: expandable one-line storylines */}
        <div className={`${PREFIX}-rc-card`}>
          <div className={`${PREFIX}-rc-card-head`}><h4>Class Storylines</h4></div>
          {headlines.length ? headlines.map((h, i) => (
            <StorylineItem key={i} text={h} />
          )) : <p className={`${PREFIX}-rc-empty`}>A quiet draft night.</p>}
        </div>

        {/* 23: projected range vs actual availability */}
        <div className={`${PREFIX}-rc-card`}>
          <div className={`${PREFIX}-rc-card-head`}><h4>Still On The Board</h4></div>
          {stillOnBoard.length ? stillOnBoard.slice(0, 5).map((e, i) => (
            <p key={i} className={`${PREFIX}-rc-row`} title={`Projected ${e.projected} · ${e.status}`}>
              <span className={`${PREFIX}-rc-row-main`}>{e.name}{e.position ? ` · ${e.position}` : ""}</span>
              <span className={`${PREFIX}-rc-row-proj`}>{e.projected} → <em>{e.status}</em></span>
            </p>
          )) : <p className={`${PREFIX}-rc-empty`}>The board cleared out.</p>}
        </div>
      </div>

      {profilePick ? (
        <PickDetailSheet
          pick={profilePick}
          reactionTweet={buildDraftPickReactionTweet(profilePick, {
            seed: `${draft?.draft_year || "draft"}-recap-pick`,
          })}
          onClose={() => setProfilePick(null)}
        />
      ) : null}
    </section>
  );
}

export default function EntryDraftMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const { setFranchiseState } = useGameUI();
  const css = buildCinematicCss(PREFIX);

  const rawDraft = pickFranchiseData(franchiseState, eventData, ["draft", "offseason.draft"]);
  const [draft, setDraft] = useState(rawDraft || {});
  const [stage, setStage] = useState(() => {
    if (rawDraft?.draft_completed) return "recap";
    if (rawDraft?.draft_started) return "live";
    return "intro";
  });

  const [loadingOp, setLoadingOp] = useState(null);
  const [userModalOpen, setUserModalOpen] = useState(false);
  const [cpuReveal, setCpuReveal] = useState(null);
  const [batchSummary, setBatchSummary] = useState(null);
  const [batchMeta, setBatchMeta] = useState(null);
  const [roundRecapView, setRoundRecapView] = useState(null);
  const [simMode] = useState("manual");
  const [compareIds, setCompareIds] = useState([]);
  const [error, setError] = useState("");
  const [tradePanelOpen, setTradePanelOpen] = useState(false);
  const [tradeAccepting, setTradeAccepting] = useState(false);
  const [selectedCompletedPick, setSelectedCompletedPick] = useState(null);
  const [selectedAvailable, setSelectedAvailable] = useState(null);
  const [dossierOpen, setDossierOpen] = useState(false);
  const [density, setDensity] = useState("compact");
  const [autoFollowLog, setAutoFollowLog] = useState(true);
  const [shortlistIds, setShortlistIds] = useState([]);
  const [pinnedIds, setPinnedIds] = useState([]);

  const simLock = useRef(false);
  const lastRoundRef = useRef(1);
  // When the user closes Make Selection while still on the clock, remember that
  // pick so the auto-open effect does not immediately reopen the modal.
  const userModalDismissedPickRef = useRef(null);
  const loading = Boolean(loadingOp);

  useEffect(() => {
    if (rawDraft && Object.keys(rawDraft).length) setDraft(rawDraft);
  }, [rawDraft]);

  const completed = safeArray(draft.completed_picks || draft.draft_results);
  const available = safeArray(draft.available_prospects);
  const currentPick = draft.current_pick;
  const currentRound = Number(currentPick?.round || draft.current_round || 1);
  const isUserPick = Boolean(draft.is_user_pick);
  const draftDone = Boolean(draft.draft_completed);
  const uid = String(franchiseState?.user_team_id || "");
  const userPicks = useMemo(() => completed.filter((p) => String(p.team_id) === uid), [completed, uid]);
  const ticker = useMemo(() => completed.slice(-8).reverse(), [completed]);
  const roundRecaps = draft.round_recaps || {};
  const orderNote = draft.draft_order_note;
  const storylines = safeArray(draft.storylines);
  const publicBoard = safeArray(draft.public_draft_board || draft.draft_class_rankings?.entries).slice(0, 12);
  const needs = safeArray(draft.team_needs);
  const phil = draft.team_philosophy || {};
  const draftOrder = safeArray(draft.draft_order);
  const totalPicks = draft.total_picks || 224;
  const overallNow = currentPick?.overall_pick || draft.overall_pick;
  const tradeOffers = safeArray(draft?.trade_offers || draft?.draft_day_trade_offers || draft?.pick_trade_offers);

  const upcomingPicks = useMemo(
    () => getUpcomingOrder(draftOrder, completed.length, uid, 5),
    [draftOrder, completed.length, uid]
  );

  const userNextPick = useMemo(() => {
    const upcoming = draftOrder.slice(completed.length + (draftDone ? 0 : 1));
    return upcoming.find((s) => String(s.team_id) === uid) || null;
  }, [draftOrder, completed.length, uid, draftDone]);

  const positionalNeeds = useMemo(
    () => needs.map(needLabel).filter((n) => n && !isPhilosophyNeedLabel(n)),
    [needs]
  );
  const primaryNeed = positionalNeeds[0] || null;
  const otherNeeds = positionalNeeds.slice(1, 3);

  const stageProspect = selectedAvailable || available[0] || null;

  const stageProspectContext = useMemo(() => {
    if (!stageProspect) return null;
    const top = available[0];
    const isBestAvailable = top ? getId(top) === getId(stageProspect) : false;
    return {
      // "Best Available" when showing the top board prospect by default,
      // "Highlighted Prospect" once the user has actively picked a row.
      label: selectedAvailable ? "Highlighted Prospect" : "Best Available",
      isBestAvailable,
      publicRank: getRank(stageProspect),
      teamRank: getTeamRank(stageProspect),
    };
  }, [stageProspect, available, selectedAvailable]);

  const draftSocialTweets = useMemo(() => {
    if (!completed.length) return [];
    const season = franchiseState?.season_year || franchiseState?.seasonYear || draft?.draft_year || "draft";
    return buildDraftFanTweets(completed, {
      maxTweets: 28,
      seed: `${season}-entry-draft`,
      draftContext: {
        franchiseSeed: String(franchiseState?.seed || franchiseState?.franchise_seed || season),
        primaryNeed,
        teamNeeds: needs,
        recentPositions: completed.slice(-5).map((p) => String(p.position || "").toUpperCase()),
      },
    });
  }, [completed, franchiseState, draft?.draft_year, primaryNeed, needs]);

  const selectedPickReaction = useMemo(() => {
    if (!selectedCompletedPick) return null;
    const overall = Number(selectedCompletedPick.overall_pick);
    const fromFeed = draftSocialTweets.find(
      (t) => Number(t?.context?.overallPick) === overall
        || String(t?.context?.winnerLabel || "").toLowerCase()
          === String(selectedCompletedPick.prospect_name || "").toLowerCase()
    );
    if (fromFeed) return fromFeed;
    const season = franchiseState?.season_year || franchiseState?.seasonYear || draft?.draft_year || "draft";
    return buildDraftPickReactionTweet(selectedCompletedPick, {
      seed: `${season}-entry-draft`,
      draftContext: {
        franchiseSeed: String(franchiseState?.seed || franchiseState?.franchise_seed || season),
        primaryNeed,
        teamNeeds: needs,
      },
    });
  }, [selectedCompletedPick, draftSocialTweets, franchiseState, draft?.draft_year, primaryNeed, needs]);

  const applyResponse = useCallback((res, { batch = false } = {}) => {
    if (res?.state) setFranchiseState(res.state);
    if (res?.draft) setDraft(res.draft);
    else if (res?.state?.draft) setDraft(res.state.draft);
    if (batch && safeArray(res?.simulated_picks).length) {
      setBatchSummary(res.simulated_picks);
      setBatchMeta(res.batch_summary || null);
      setCpuReveal(null);
    }
    if (res?.batch_summary) setBatchMeta(res.batch_summary);
    if (res?.draft?.recap) setDraft((d) => ({ ...d, recap: res.draft.recap }));
    if (res?.recap) setDraft((d) => ({ ...d, recap: res.recap }));
  }, [setFranchiseState]);

  const acceptTradeOffer = useCallback(async (offer) => {
    setTradeAccepting(true);
    setError("");
    try {
      const res = await acceptEntryDraftTrade(offer);
      applyResponse(res);
      setTradePanelOpen(false);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Trade failed");
    } finally {
      setTradeAccepting(false);
    }
  }, [applyResponse]);

  const handleStart = useCallback(async () => {
    setLoadingOp("start");
    setError("");
    try {
      const res = draft.draft_started ? { draft } : await startEntryDraft();
      applyResponse(res);
      setStage("board");
      window.setTimeout(() => setStage("live"), 1600);
    } catch (e) {
      setError(e?.message || "Failed to start draft");
    } finally {
      setLoadingOp(null);
    }
  }, [applyResponse, draft]);

  const runCpuPick = useCallback(async () => {
    if (simLock.current) return;
    simLock.current = true;
    setLoadingOp("cpu");
    setError("");
    try {
      const res = await submitCpuDraftPick();
      setCpuReveal(res?.pick_result);
      applyResponse(res);
    } catch (e) {
      setError(e?.message || "CPU pick failed");
      simLock.current = false;
    } finally {
      setLoadingOp(null);
    }
  }, [applyResponse]);

  const simToUser = useCallback(async (batch = false) => {
    if (simLock.current || draftDone) return;
    if (isUserPick) {
      setError("You are on the clock. Make a selection or confirm Sim Round.");
      return;
    }
    simLock.current = true;
    setLoadingOp("simUser");
    setError("");
    setBatchSummary(null);
    setBatchMeta(null);
    try {
      const res = await simEntryDraftToUserPick();
      applyResponse(res, { batch });
      const simulated = safeArray(res?.simulated_picks);
      if (batch || simulated.length > 3) {
        simLock.current = false;
        setLoadingOp(null);
        return;
      }
      if (simulated.length) {
        setCpuReveal(simulated[simulated.length - 1]);
      } else if (res?.draft?.is_user_pick) {
        userModalDismissedPickRef.current = null;
        setUserModalOpen(true);
        simLock.current = false;
        setLoadingOp(null);
      }
    } catch (e) {
      setError(e?.message || "Sim failed");
      simLock.current = false;
    } finally {
      setLoadingOp(null);
    }
  }, [applyResponse, draftDone, isUserPick]);

  const simRound = useCallback(async () => {
    if (simLock.current || draftDone) return;
    const picksUntil = num(draft.picks_until_user, 999) ?? 999;
    let warn = "Simulate the rest of this round?";
    if (isUserPick) {
      warn = "You are on the clock. Sim Round will auto-select for you. Continue?";
    } else if (picksUntil > 0 && picksUntil <= 32) {
      warn = `Your pick is ${picksUntil} away. Sim Round may pass it. Continue?`;
    }
    if (!window.confirm(warn)) return;
    simLock.current = true;
    setLoadingOp("simRound");
    setError("");
    try {
      const res = await simEntryDraftRound();
      applyResponse(res, { batch: true });
    } catch (e) {
      setError(e?.message || "Sim round failed");
    } finally {
      setLoadingOp(null);
      simLock.current = false;
    }
  }, [applyResponse, draft.picks_until_user, isUserPick, draftDone]);

  const simFullDraft = useCallback(async () => {
    if (simLock.current || draftDone) return;
    if (!window.confirm("Complete the entire remaining draft? This may skip your picks.")) return;
    simLock.current = true;
    setLoadingOp("complete");
    setError("");
    try {
      const res = await completeEntryDraft();
      applyResponse(res, { batch: true });
      setStage("recap");
    } catch (e) {
      setError(e?.message || "Complete draft failed");
    } finally {
      setLoadingOp(null);
      simLock.current = false;
    }
  }, [applyResponse, draftDone]);

  const handleUserDraft = useCallback(async (prospect) => {
    if (simLock.current) return;
    simLock.current = true;
    const pid = getId(prospect);
    setLoadingOp("submit");
    setError("");
    try {
      const res = await submitDraftPick({
        player_id: pid,
        pick_round: currentPick?.round || currentRound,
        pick_overall: currentPick?.overall_pick || draft.overall_pick,
        request_id: `${draft?.draft_id || "draft"}:${currentPick?.pick_id || currentPick?.overall_pick || draft.overall_pick}:${pid}`,
      });
      applyResponse(res);
      setUserModalOpen(false);
      userModalDismissedPickRef.current = null;
      setSelectedAvailable(null);
      simLock.current = false;
    } catch (e) {
      setError(e?.message || "Pick failed");
      simLock.current = false;
    } finally {
      setLoadingOp(null);
    }
  }, [applyResponse, currentPick, currentRound, draft.overall_pick]);

  const simCurrentPick = useCallback(async () => {
    if (draftDone || loading || simLock.current) return;
    if (isUserPick) {
      if (!window.confirm("Auto-select the top available prospect for your pick?")) return;
      const best = available[0];
      if (!best) {
        setError("No available prospect to auto-pick.");
        return;
      }
      await handleUserDraft(best);
      return;
    }
    setLoadingOp("simNext");
    await runCpuPick();
  }, [draftDone, loading, isUserPick, available, handleUserDraft, runCpuPick]);

  useEffect(() => {
    if (stage !== "live" || draftDone || loading || cpuReveal || userModalOpen || roundRecapView) return;
    if (isUserPick) {
      const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
      // Honor an explicit Close while still on this same pick.
      if (pickKey && userModalDismissedPickRef.current === pickKey) return;
      if (simMode === "auto_user") {
        simCurrentPick();
        return;
      }
      setUserModalOpen(true);
      return;
    }
    userModalDismissedPickRef.current = null;
    if (simMode === "manual") return;
    if (!simLock.current) simToUser(simMode === "fast");
  }, [stage, draftDone, isUserPick, loading, cpuReveal, userModalOpen, roundRecapView, simToUser, simMode, simCurrentPick, currentPick?.overall_pick, draft.overall_pick]);

  useEffect(() => {
    const prev = lastRoundRef.current;
    if (currentRound > prev && roundRecaps[String(prev)]) {
      setRoundRecapView(roundRecaps[String(prev)]);
    }
    lastRoundRef.current = currentRound;
  }, [currentRound, roundRecaps]);

  useEffect(() => {
    if (draftDone && stage === "live") setStage("recap");
  }, [draftDone, stage]);

  const handleCpuRevealDone = useCallback(() => {
    setCpuReveal(null);
    simLock.current = false;
  }, []);

  const onCompare = useCallback((p) => {
    const id = getId(p);
    setCompareIds((prev) => (
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-3)
    ));
  }, []);

  const statusLine = !draftDone
    ? `Round ${Math.min(currentRound, ROUNDS)} of ${ROUNDS} · Pick ${overallNow || "—"} · ${completed.length}/${totalPicks}`
    : `Draft complete · ${completed.length} picks`;

  const clockTimer = draft.clock_seconds ?? draft.pick_clock ?? currentPick?.clock_seconds ?? currentPick?.time_remaining;

  return (
    <section className={`${PREFIX}-root ${PREFIX}-broadcast`}>
      <style>{css}</style>

      <header className={`${PREFIX}-topbar`}>
        <button type="button" onClick={onBack} className={`${PREFIX}-ghost-btn ${PREFIX}-back-btn`}>
          <span aria-hidden="true">←</span> Back to Hub
        </button>
        <div className={`${PREFIX}-topbar-main`}>
          <div className={`${PREFIX}-insignia`} aria-hidden="true">
            NHL
            <span>DRAFT</span>
          </div>
          <div>
            <h1 className={`${PREFIX}-page-title`}>NHL Entry Draft</h1>
            {(seasonLabel(franchiseState) || draftYearLabel(draft)) ? (
              <span className={`${PREFIX}-season`}>
                {[seasonLabel(franchiseState), draftYearLabel(draft)].filter(Boolean).join(" ")}
              </span>
            ) : null}
          </div>
        </div>
        {(stage === "live" || stage === "recap") ? (
          <span className={`${PREFIX}-topbar-status ${PREFIX}-tabular`}>{statusLine}</span>
        ) : (
          <span className={`${PREFIX}-topbar-spacer`} aria-hidden="true" />
        )}
      </header>

      {stage === "intro" && (
        <main className={`${PREFIX}-intro`}>
          <div className={`${PREFIX}-ceremony-mark`}>
            <div className={`${PREFIX}-ceremony-seal`} aria-hidden="true">
              {draftYearLabel(draft) || "NHL"}
            </div>
            <p className={`${PREFIX}-ceremony-kicker`}>Entry Draft Ceremony</p>
          </div>
          <h1 className={`${PREFIX}-title`}>NHL Entry Draft</h1>
          {(draft.location || draft.class_strength) ? (
            <p className={`${PREFIX}-subtitle`}>
              {[draft.location, draft.class_strength].filter(Boolean).join(" · ")}
            </p>
          ) : null}
          {orderNote ? <p className={`${PREFIX}-order-note`}>{orderNote}</p> : null}
          {storylines.length ? (
            <div className={`${PREFIX}-storylines`}>
              {storylines.map((s, i) => <span key={i}>{s}</span>)}
            </div>
          ) : null}
          {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
          <button type="button" className={`${PREFIX}-cta-btn`} disabled={loading} onClick={handleStart}>
            {loadingOp === "start" ? LOADING_COPY.start : draft.draft_started ? "Enter Draft Floor" : "Start Draft"}
          </button>
        </main>
      )}

      {stage === "board" && (
        <main className={`${PREFIX}-board-reveal`}>
          <h2>Draft Board Reveal</h2>
          {draft.class_strength ? <p>{draft.class_strength}</p> : null}
          <div className={`${PREFIX}-board-grid`}>
            {publicBoard.map((p) => (
              <div key={getId(p)} className={`${PREFIX}-reveal-card`}>
                <span className={`${PREFIX}-tabular`}>#{getRank(p) ?? ""}</span>
                <strong>{getPlayerName(p)}</strong>
                <span>{getPlayerPosition(p)}</span>
              </div>
            ))}
          </div>
        </main>
      )}

      {stage === "live" && (
        <main className={`${PREFIX}-live-floor`}>
          {loading && !cpuReveal ? (
            <div className={`${PREFIX}-batch-overlay`}>{LOADING_COPY[loadingOp] || "Working…"}</div>
          ) : null}

          {cpuReveal ? (
            <CpuReveal pick={cpuReveal} onDone={handleCpuRevealDone} fast={simMode === "fast"} />
          ) : null}

          {roundRecapView ? (
            <RoundRecapPanel recap={roundRecapView} onContinue={() => setRoundRecapView(null)} />
          ) : null}

          {batchSummary?.length ? (
            <div className={`${PREFIX}-batch-summary`}>
              <strong>{batchMeta?.picks_made ?? batchSummary.length} picks simmed</strong>
              {batchMeta?.biggest_steal ? (
                <span>Steal: {batchMeta.biggest_steal.prospect_name}</span>
              ) : null}
              {batchMeta?.biggest_reach ? <span>Reach: {batchMeta.biggest_reach.prospect_name}</span> : null}
              <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => { setBatchSummary(null); setBatchMeta(null); }}>
                Dismiss
              </button>
            </div>
          ) : null}

          <div className={`${PREFIX}-floor-grid${!completed.length ? " log-collapsed" : ""}`}>
            <DraftLogPanel
              completed={completed}
              selectedPick={selectedCompletedPick}
              userTeamId={uid}
              autoFollow={autoFollowLog}
              onToggleAutoFollow={() => setAutoFollowLog((v) => !v)}
              onSelectPick={(p) => {
                const enriched = enrichPickFromBoard(
                  p,
                  draft.public_draft_board || draft.draft_class_rankings?.entries
                );
                setSelectedCompletedPick((prev) => (
                  prev?.overall_pick === p.overall_pick ? null : enriched
                ));
                setSelectedAvailable(null);
              }}
            />

            {!roundRecapView ? (
              <section className={`${PREFIX}-stage${isUserPick ? " is-user" : ""}`}>
                {!draftDone ? (
                  <>
                    {/* pickHeader — one compact band leading into the prospect workspace */}
                    <header className={`${PREFIX}-pickhead`}>
                      <div className={`${PREFIX}-pickhead-lead`}>
                        <span className={`${PREFIX}-pickhead-num ${PREFIX}-tabular`}>#{overallNow}</span>
                        <TeamLogo teamId={draft.current_team_id} teamName={draft.current_team_name} size="lg" />
                      </div>
                      <div className={`${PREFIX}-pickhead-identity`}>
                        <span className={`${PREFIX}-pickhead-state${isUserPick ? " is-user" : ""}`}>
                          {isUserPick ? "You are selecting" : "CPU selecting"}
                        </span>
                        <h2 className={`${PREFIX}-pickhead-team`}>{draft.current_team_name || "Team"}</h2>
                        <div className={`${PREFIX}-pickhead-meta`}>
                          {primaryNeed ? (
                            <span className={`${PREFIX}-pickhead-need`}>
                              Needs: {primaryNeed}{otherNeeds.length ? `, ${otherNeeds.join(", ")}` : ""}
                            </span>
                          ) : null}
                          {draft.is_traded_pick && (draft.via_team_name || draft.via_team_id) ? (
                            <span className={`${PREFIX}-pickhead-via`}>
                              <TeamLogo
                                teamId={draft.via_team_id || draft.original_owner_team_id}
                                teamName={draft.via_team_name || draft.original_owner_team_name}
                                size="xs"
                              />
                              via {teamAbbrev(draft.via_team_id, draft.via_team_name)}
                            </span>
                          ) : null}
                          {userNextPick ? (
                            <span>Your next: #{userNextPick.overall_pick}{userNextPick.round ? ` · Rd ${userNextPick.round}` : ""}</span>
                          ) : null}
                          {clockTimer != null && Number.isFinite(Number(clockTimer)) ? (
                            <span className={`${PREFIX}-pickhead-clock ${PREFIX}-tabular`}>{Math.round(Number(clockTimer))}s</span>
                          ) : null}
                        </div>
                      </div>
                      <div className={`${PREFIX}-pickhead-ondeck`}>
                        <OrderStrip upcoming={upcomingPicks} userTeamId={uid} />
                      </div>
                    </header>

                    {/* prospect — highlighted prospect workspace fills the middle track */}
                    <div className={`${PREFIX}-stage-prospect`}>
                      <ProspectStageCard
                        prospect={stageProspect}
                        boardContext={stageProspectContext}
                        currentPickOverall={overallNow}
                        onFullReport={(p) => {
                          setSelectedAvailable(p);
                          setSelectedCompletedPick(null);
                          setDossierOpen(true);
                        }}
                      />
                    </div>

                    {/* reactions — Award Show Twitter universe reused for the draft floor */}
                    <div className={`${PREFIX}-social-lane`}>
                      <FanReactionFeed
                        enabled={completed.length > 0}
                        reactions={draftSocialTweets}
                        eventType="entry_draft"
                        visibleCount={3}
                        intervalMs={6000}
                        maxTweets={28}
                        feedLabel="Draft Floor"
                        feedSubLabel="Live reactions"
                        className={`${PREFIX}-social-feed`}
                      />
                    </div>

                    {/* controls — one full-width action dock */}
                    <div className={`${PREFIX}-dock`}>
                      <div className={`${PREFIX}-dock-state`}>
                        {isUserPick
                          ? (stageProspect
                              ? `Ready: ${getPlayerName(stageProspect)}`
                              : "Select a prospect to draft")
                          : `${shortTeamName(draft.current_team_name) || "CPU"} on the clock`}
                      </div>

                      <div className={`${PREFIX}-dock-actions`}>
                        {isUserPick ? (
                          <button
                            type="button"
                            className={`${PREFIX}-cta-btn ${PREFIX}-dock-primary`}
                            disabled={loading || !stageProspect}
                            onClick={() => {
                              userModalDismissedPickRef.current = null;
                              setUserModalOpen(true);
                            }}
                          >
                            {stageProspect ? "Make Selection" : "Select a Prospect"}
                          </button>
                        ) : (
                          <button
                            type="button"
                            className={`${PREFIX}-cta-btn ${PREFIX}-dock-primary`}
                            disabled={loading || Boolean(cpuReveal)}
                            onClick={runCpuPick}
                          >
                            {loadingOp === "cpu" ? LOADING_COPY.cpu : "Advance CPU Pick"}
                          </button>
                        )}
                        <button type="button" className={`${PREFIX}-ghost-btn`} disabled={loading || draftDone} onClick={simCurrentPick}>
                          Sim Next Pick
                        </button>
                        <button type="button" className={`${PREFIX}-ghost-btn`} disabled={loading || draftDone || isUserPick} onClick={() => simToUser(true)}>
                          {loadingOp === "simUser" ? LOADING_COPY.simUser : "Sim to User"}
                        </button>
                        <button type="button" className={`${PREFIX}-ghost-btn`} disabled={loading || draftDone} onClick={simRound}>
                          {loadingOp === "simRound" ? LOADING_COPY.simRound : "Sim Round"}
                        </button>
                        <button type="button" className={`${PREFIX}-ghost-btn`} disabled={loading || draftDone} onClick={simFullDraft}>
                          {loadingOp === "complete" ? LOADING_COPY.complete : "Complete Draft"}
                        </button>
                        {isUserPick ? (
                          <button type="button" className={`${PREFIX}-ghost-btn`} onClick={() => setTradePanelOpen(true)}>
                            {tradeOffers.length ? `Trade Down (${tradeOffers.length})` : "Trade Down"}
                          </button>
                        ) : null}
                      </div>

                      {error ? <p className={`${PREFIX}-error ${PREFIX}-dock-error`}>{error}</p> : null}
                    </div>
                  </>
                ) : (
                  <div className={`${PREFIX}-complete-banner`}>
                    <h2>Draft Complete</h2>
                    <p>{completed.length} selections made</p>
                  </div>
                )}
              </section>
            ) : null}

            <TeamBoardPanel
              available={available}
              needs={needs}
              philosophy={phil}
              selectedId={selectedAvailable ? getId(selectedAvailable) : null}
              density={density}
              onDensityChange={setDensity}
              pinnedIds={pinnedIds}
              onSelectProspect={(p) => {
                setSelectedAvailable((prev) => (getId(prev) === getId(p) ? null : p));
                setSelectedCompletedPick(null);
              }}
            />
          </div>

          {selectedCompletedPick ? (
            <PickDetailSheet
              pick={selectedCompletedPick}
              reactionTweet={selectedPickReaction}
              onClose={() => setSelectedCompletedPick(null)}
            />
          ) : null}

          {dossierOpen && selectedAvailable ? (
            <ProspectDossier
              prospect={selectedAvailable}
              isUserPick={isUserPick}
              loading={loading}
              compareIds={compareIds}
              shortlistIds={shortlistIds}
              onCompare={onCompare}
              onToggleShortlist={(p) => {
                const id = getId(p);
                setShortlistIds((prev) => (
                  prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
                ));
              }}
              onPinBoard={(p) => {
                const id = getId(p);
                setPinnedIds((prev) => (
                  prev.includes(id) ? prev : [...prev, id].slice(-12)
                ));
              }}
              onClose={() => setDossierOpen(false)}
              onDraft={handleUserDraft}
            />
          ) : null}

          {tradePanelOpen ? (
            <TradePanel
              draft={draft}
              onClose={() => setTradePanelOpen(false)}
              onAccept={acceptTradeOffer}
              accepting={tradeAccepting}
            />
          ) : null}

          <div className={`${PREFIX}-ticker-bar`}>
            <div className={`${PREFIX}-ticker-track`}>
              {ticker.length ? ticker.map((p, i) => (
                <span key={`${p.overall_pick}-${i}`}>
                  #{p.overall_pick} {teamAbbrev(p.team_id, p.team_name)} — {p.prospect_name}{p.position ? `, ${p.position}` : ""}
                </span>
              )) : (
                <span>Draft floor open</span>
              )}
            </div>
          </div>
        </main>
      )}

      {stage === "recap" && (
        <main className={`${PREFIX}-recap-stage`}>
          <RecapShow draft={draft} completed={completed} userPicks={userPicks} />
          <div className={`${PREFIX}-recap-actions`}>
            <button type="button" className={`${PREFIX}-cta-btn ${PREFIX}-rc-continue`} onClick={onContinue}>
              Continue Offseason <span aria-hidden="true">→</span>
            </button>
          </div>
        </main>
      )}

      <UserPickModal
        open={userModalOpen && isUserPick && !draftDone && !tradePanelOpen}
        prospects={available}
        compareIds={compareIds}
        currentPick={currentPick}
        draft={draft}
        onCompare={onCompare}
        onDraft={handleUserDraft}
        onClose={() => {
          const pickKey = Number(currentPick?.overall_pick || draft.overall_pick || 0) || 0;
          if (pickKey) userModalDismissedPickRef.current = pickKey;
          setUserModalOpen(false);
        }}
        onTradeDown={() => setTradePanelOpen(true)}
        loading={loading}
      />
    </section>
  );
}
