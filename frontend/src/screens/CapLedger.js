import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import {
  getContractOffice,
  reSignContract,
  qualifyRfa,
  releaseRfaRights,
  buyoutContract,
  waiveContract,
  buryContract,
  signFreeAgent,
  getFreeAgentDetail,
  submitOfferSheet,
  matchOfferSheet,
  declineOfferSheet,
  fileArbitration,
  settleArbitration,
  getRosterMoves,
  moveRosterPlayer,
} from "../services/franchiseService";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { ContractStrip, StatusSeal } from "../components/franchise/commandVisuals";
import "./CapLedger.css";

const NHL_ROSTER_LIMIT = 23;
const AHL_ROSTER_LIMIT = 24;
const ECHL_ROSTER_LIMIT = 18;

function resolveUserOrganization(franchiseState, teamId) {
  const orgs = safeArray(franchiseState?.roster_browser?.organizations);
  const tid = String(teamId || franchiseState?.team?.id || franchiseState?.team_id || "");
  return orgs.find((org) => String(org.team_id) === tid) || orgs[0] || null;
}

function buildRosterSlotSummary(franchiseState, snap, teamId) {
  const org = resolveUserOrganization(franchiseState, teamId);
  const nhlUsed = safeNum(
    snap?.active_roster_count,
    Array.isArray(org?.nhl) ? org.nhl.length : 0,
  );
  const ahlUsed = Array.isArray(org?.ahl) ? org.ahl.length : 0;
  const echlUsed = Array.isArray(org?.echl) ? org.echl.length : 0;
  // Backend org SPC total (assignment-agnostic). Prefer nhl_spcs_used alias, then contract_slots_used.
  const spcUsedRaw =
    snap?.nhl_spcs_used ??
    snap?.contract_slots_used ??
    franchiseState?.contract_slots?.used ??
    franchiseState?.team?.contract_slots_used;
  const spcUsed = Number.isFinite(Number(spcUsedRaw)) ? Number(spcUsedRaw) : null;
  const spcLimit = safeNum(
    snap?.nhl_spcs_limit ?? snap?.contract_slots_limit,
    50,
  );

  return {
    nhl: { used: nhlUsed, limit: NHL_ROSTER_LIMIT },
    ahl: { used: ahlUsed, limit: AHL_ROSTER_LIMIT },
    echl: { used: echlUsed, limit: ECHL_ROSTER_LIMIT },
    spc: spcUsed == null ? null : { used: spcUsed, limit: spcLimit },
    compact:
      spcUsed == null
        ? `NHL ${nhlUsed}/${NHL_ROSTER_LIMIT}`
        : `NHL SPCs ${spcUsed}/${spcLimit}`,
    subline:
      spcUsed == null
        ? `AHL ${ahlUsed}/${AHL_ROSTER_LIMIT} · ECHL ${echlUsed}/${ECHL_ROSTER_LIMIT}`
        : `NHL ${nhlUsed}/${NHL_ROSTER_LIMIT} · AHL ${ahlUsed}/${AHL_ROSTER_LIMIT} · ECHL ${echlUsed}/${ECHL_ROSTER_LIMIT}`,
    full:
      spcUsed == null
        ? `NHL ${nhlUsed}/${NHL_ROSTER_LIMIT} · AHL ${ahlUsed}/${AHL_ROSTER_LIMIT} · ECHL ${echlUsed}/${ECHL_ROSTER_LIMIT}`
        : `NHL SPCs ${spcUsed}/${spcLimit} · NHL ${nhlUsed}/${NHL_ROSTER_LIMIT} · AHL ${ahlUsed}/${AHL_ROSTER_LIMIT} · ECHL ${echlUsed}/${ECHL_ROSTER_LIMIT}`,
  };
}

const TABS = [
  { id: "ledger", label: "Board" },
  { id: "freeAgents", label: "Free Agents" },
  { id: "rfa", label: "RFA / Sheets" },
  { id: "cap", label: "Cap" },
];

const FILTERS = [
  { id: "all", label: "All" },
  { id: "expiring", label: "Exp" },
  { id: "rfa", label: "RFA" },
  { id: "core", label: "Core" },
  { id: "bad", label: "Risk" },
  { id: "bargain", label: "Deal" },
  { id: "clause", label: "NTC" },
];

function safeArray(v) {
  return Array.isArray(v) ? v : [];
}

function safeNum(v, fb = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fb;
}

function safeText(v, fb = "—") {
  if (v === null || v === undefined || v === "") return fb;
  return String(v);
}

function formatMoneyM(v) {
  const n = safeNum(v, NaN);
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "$0";
  return `$${n.toFixed(1)}M`;
}

function formatPct(used, limit) {
  const capLimit = safeNum(limit, 0);
  if (!capLimit) return "—";
  return `${((safeNum(used) / capLimit) * 100).toFixed(1)}%`;
}

function getCapSnap(data) {
  return data?.cap_snapshot || data?.team_cap || {};
}

function valueKey(value, tags = []) {
  const text = `${value || ""} ${safeArray(tags).join(" ")}`.toLowerCase();

  if (text.includes("casualty") || text.includes("bad") || text.includes("overpaid") || text.includes("heavy")) {
    return "bad";
  }

  if (text.includes("bargain") || text.includes("deal") || text.includes("elc")) {
    return "bargain";
  }

  if (text.includes("core")) {
    return "core";
  }

  return "neutral";
}

function chipClass(label) {
  const t = String(label || "").toLowerCase();

  if (t.includes("bad") || t.includes("over") || t.includes("risk") || t.includes("casualty") || t.includes("buyout")) {
    return "cap-chip cap-chip-danger";
  }

  if (t.includes("bargain") || t.includes("deal") || t.includes("core") || t.includes("elc")) {
    return "cap-chip cap-chip-good";
  }

  if (t.includes("clause") || t.includes("ntc") || t.includes("nmc") || t.includes("heavy") || t.includes("expiring") || t.includes("rfa")) {
    return "cap-chip cap-chip-warning";
  }

  return "cap-chip";
}

function getStatusLabel(row) {
  const expiry = safeText(row?.expiry_status, "");
  const clause = safeText(row?.clause_label, "");

  if (clause && clause !== "None" && clause !== "—") return clause;
  if (expiry && expiry !== "—") return expiry;

  const years = safeNum(row?.years_remaining ?? row?.yearsRemaining, 0);
  if (years <= 1) return "Exp";
  return "Signed";
}

function getValueLabel(row) {
  const tags = safeArray(row?.tags);
  const value = safeText(row?.contract_value_score, "");

  if (tags.some((t) => /casualty|bad|overpaid|heavy/i.test(t))) return "Risk";
  if (tags.some((t) => /bargain|deal|elc/i.test(t))) return "Deal";
  if (value && value !== "—") return value;

  return "Fair";
}

function buildHeadshotPlayer(row) {
  const ovr = safeNum(row?.overall ?? row?.ovr, 0);

  return ensurePlayerHeadshotFields({
    ...row,
    ovr: ovr / 99,
    position: row?.position,
    age: row?.age,
  });
}

function ContractBoardRow({ row, onSelect, isSelected = false }) {
  const ovr = safeNum(row.overall ?? row.ovr);
  const aav = safeNum(row.aav_m ?? row.aav);
  const yrs = safeNum(row.years_remaining ?? row.yearsRemaining);
  const tags = safeArray(row.tags);
  const status = getStatusLabel(row);
  const value = getValueLabel(row);
  const stripe = valueKey(value, tags);
  const player = buildHeadshotPlayer(row);
  const expiry = String(row?.expiry_status || "").toUpperCase();
  const daysLeft = Number(row?.days_to_expiry ?? row?.daysToExpiry ?? row?.ufa_days);
  const isExpiring = yrs <= 1 || /UFA|RFA|EXP/i.test(status);
  const sealTone = /UFA/.test(expiry) || /UFA/.test(status)
    ? "ufa"
    : /RFA/.test(expiry) || /RFA/.test(status)
      ? "rfa"
      : /ELC|ENTRY/.test(status) || tags.some((t) => /elc/i.test(String(t)))
        ? "elc"
        : /NTC|NMC/.test(status)
          ? "ntc"
          : "neutral";

  return (
    <button
      type="button"
      className={`cap-contract-card cap-contract-row cap-stripe-${stripe}${isSelected ? " is-selected" : ""}${isExpiring ? " is-expiring" : ""}`}
      onClick={() => onSelect(row)}
    >
      <span className="cap-contract-player">
        <PlayerHeadshot player={player} size="sm" className="cap-card-headshot" />

        <span className="cap-contract-meta cap-contract-row__identity">
          <strong className="cap-contract-row__name">{safeText(row.name, "Unnamed Player")}</strong>
          <em className="cap-contract-row__sub">
            {safeText(row.position)} · {row.age ? `${row.age}` : "—"}
          </em>
        </span>
      </span>

      <span className="cap-contract-row__cell cap-contract-row__pos">{safeText(row.position)}</span>

      <span className="cap-contract-row__cell cap-contract-row__ovr">
        <strong>{ovr || "—"}</strong>
        <em>OVR</em>
      </span>

      <span className="cap-contract-row__cell">
        <ContractStrip aav={aav} years={yrs} />
        <em>Deal</em>
      </span>

      <span className="cap-contract-row__tags">
        <StatusSeal label={status} tone={sealTone} />
        <span className={chipClass(value)}>{value}</span>
        {isExpiring && Number.isFinite(daysLeft) ? (
          <span className="cap-ufa-count">{expiry || "UFA"} IN {Math.max(0, Math.round(daysLeft))} DAYS</span>
        ) : null}
      </span>
    </button>
  );
}

function ContractBoardHeader() {
  return (
    <div className="cap-board-head">
      <span>Player</span>
      <span>Pos</span>
      <span>OVR</span>
      <span>AAV</span>
      <span>Term</span>
      <span>Status</span>
    </div>
  );
}

function LedgerTab({ data, onSelect, selectedId }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");

  const rows = useMemo(() => {
    let list = [...safeArray(data.contracts)];

    if (search.trim()) {
      const q = search.trim().toLowerCase();

      list = list.filter((r) => {
        const haystack = [
          r.name,
          r.position,
          r.expiry_status,
          r.contract_value_score,
          r.clause_label,
          ...safeArray(r.tags),
        ]
          .map((v) => safeText(v, "").toLowerCase())
          .join(" ");

        return haystack.includes(q);
      });
    }

    if (filter === "expiring") {
      list = list.filter((r) => safeNum(r.years_remaining ?? r.yearsRemaining, 99) <= 1);
    }

    if (filter === "rfa") {
      list = list.filter((r) => String(r.expiry_status || "").toUpperCase().includes("RFA"));
    }

    if (filter === "bad") {
      list = list.filter((r) => {
        const text = `${r.contract_value_score || ""} ${safeArray(r.tags).join(" ")}`.toLowerCase();
        return /bad|casualty|heavy|overpaid|risk/.test(text);
      });
    }

    if (filter === "bargain") {
      list = list.filter((r) => {
        const text = `${r.contract_value_score || ""} ${safeArray(r.tags).join(" ")}`.toLowerCase();
        return /bargain|deal|elc/.test(text);
      });
    }

    if (filter === "clause") {
      list = list.filter((r) => r.clause_label && r.clause_label !== "None");
    }

    if (filter === "core") {
      list = list.filter((r) => safeNum(r.overall ?? r.ovr) >= 85);
    }

    return list.sort((a, b) => {
      const aavDiff = safeNum(b.aav_m ?? b.aav) - safeNum(a.aav_m ?? a.aav);
      if (aavDiff !== 0) return aavDiff;
      return safeNum(b.overall ?? b.ovr) - safeNum(a.overall ?? a.ovr);
    });
  }, [data.contracts, filter, search]);

  return (
    <section className="cap-office-panel cap-board-panel">
      <div className="cap-ledger-toolbar cap-board-toolbar">
        <input
          className="cap-search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search player"
        />

        <div className="cap-filter-pills">
          {FILTERS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`cap-filter-pill ${filter === item.id ? "is-active" : ""}`}
              onClick={() => setFilter(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <ContractBoardHeader />

      <div className="cap-contract-list cap-board-list">
        {rows.length ? (
          rows.map((row) => (
            <ContractBoardRow
              key={row.player_id || row.id || `${row.name}-${row.position}`}
              row={row}
              onSelect={onSelect}
              isSelected={selectedId === (row.player_id || row.id)}
            />
          ))
        ) : (
          <div className="cap-empty-state">No matches.</div>
        )}
      </div>
    </section>
  );
}

function ContractRosterMoves({ row, onMoved }) {
  const playerId = String(row?.player_id || row?.id || "");
  const [moves, setMoves] = useState([]);
  const [meta, setMeta] = useState({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    let cancelled = false;
    setError("");
    setNote("");
    setMoves([]);
    if (!playerId || row?.own_ufa || row?.contract_status === "own_ufa") {
      return undefined;
    }
    getRosterMoves(playerId)
      .then((data) => {
        if (cancelled) return;
        setMeta(data || {});
        setMoves(Array.isArray(data?.actions) ? data.actions : []);
        if (data && data.ok === false && data.reason) setError(String(data.reason));
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message || "Could not load roster moves");
      });
    return () => {
      cancelled = true;
    };
  }, [playerId, row?.own_ufa, row?.contract_status]);

  if (!playerId || row?.own_ufa || row?.contract_status === "own_ufa") {
    return null;
  }

  const runMove = async (action) => {
    setBusy(action);
    setError("");
    setNote("");
    try {
      let result = await moveRosterPlayer({ player_id: playerId, action });
      if (!result?.ok && result?.requires_waivers) {
        const ok = window.confirm(
          `${row.name || "Player"} requires waivers to leave the NHL roster. Place on waivers and send down?`
        );
        if (!ok) {
          setError("Waivers required — move cancelled");
          return;
        }
        result = await moveRosterPlayer({
          player_id: playerId,
          action,
          confirm_waivers: true,
        });
      }
      if (!result?.ok) {
        setError(result?.reason || "Move failed");
        return;
      }
      setNote(result.moved || "Move completed");
      setMoves(Array.isArray(result.available_moves) ? result.available_moves : []);
      if (typeof onMoved === "function") onMoved(result);
    } catch (err) {
      setError(err?.message || "Move failed");
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="cap-action-panel__moves">
      <h4>Call up / Send down</h4>
      <p className="cap-action-panel__moves-meta">
        Location: {meta.location || "—"}
        {meta.nhl_gp != null ? ` · NHL GP ${meta.nhl_gp}` : ""}
        {meta.waiver_exempt ? " · Waiver exempt" : ""}
      </p>
      <div className="cap-action-panel__actions">
        {moves.length ? (
          moves.map((m) => (
            <button
              key={m.id}
              type="button"
              className="cap-action-btn"
              disabled={Boolean(busy) || m.enabled === false}
              title={m.reason || m.note || ""}
              onClick={() => runMove(m.id)}
            >
              {busy === m.id ? "…" : m.label}
              {m.requires_waivers ? " (waivers)" : ""}
            </button>
          ))
        ) : (
          <span className="cap-chip">{error || "No assignment moves for this player"}</span>
        )}
      </div>
      {note ? <p className="cap-action-panel__moves-note">{note}</p> : null}
      {error && moves.length ? <p className="cap-action-panel__moves-error">{error}</p> : null}
    </div>
  );
}

function ContractActionPanel({ row, onClose, onAction, busy, onRosterMoved }) {
  if (!row) {
    return (
      <section className="cap-action-panel cap-action-panel--empty">
        <p>Select a player to re-sign, buy out, call up / send down, or manage their contract.</p>
      </section>
    );
  }

  const ext = row.extension_estimate || {};
  const buy = row.buyout_estimate || {};
  const ovr = safeNum(row.overall ?? row.ovr);
  const aav = safeNum(row.aav_m ?? row.aav);
  const yrs = safeNum(row.years_remaining ?? row.yearsRemaining);
  const player = buildHeadshotPlayer(row);
  const displayTags = safeArray(row.tags).filter(
    (tag) => !/casualty|cap casualty|trade pressure/i.test(String(tag)),
  );

  const actions = [];
  if (row.can_negotiate) actions.push({ id: "re-sign", label: "Re-sign" });
  if (row.can_qualify) actions.push({ id: "qualify-rfa", label: "Qualify RFA" });
  if (row.can_release_rights) actions.push({ id: "release-rights", label: "Release Rights" });
  if (row.can_file_arbitration || (row.arbitration_eligible && !row.arbitration_filed)) {
    actions.push({ id: "arbitration-file", label: "File Arbitration" });
  }
  if (row.arbitration_filed && !row.award_aav_m) {
    actions.push({ id: "arbitration-settle", label: "Settle Arbitration" });
  }
  if (row.offer_sheet_pending || row.pending_offer_sheet) {
    actions.push({ id: "match-offer-sheet", label: "Match Offer Sheet" });
    actions.push({ id: "decline-offer-sheet", label: "Decline Offer Sheet" });
  }
  if (row.can_buyout) actions.push({ id: "buyout", label: "Buyout" });
  if (row.can_waive) actions.push({ id: "waive", label: "Waive" });
  if (row.can_bury || row.can_waive) actions.push({ id: "bury", label: "Bury" });

  return (
    <section className="cap-action-panel">
      <button type="button" className="cap-action-panel__close" onClick={onClose} aria-label="Clear selection">
        ×
      </button>

      <div className="cap-action-panel__player">
        <PlayerHeadshot player={player} size="md" className="cap-action-panel__headshot" />
        <div>
          <h3>{safeText(row.name, "Unnamed Player")}</h3>
          <p>
            {safeText(row.position)} · {row.age || "—"} · OVR {ovr || "—"}
          </p>
        </div>
      </div>

      <div className="cap-action-panel__stats">
        <div>
          <span>AAV</span>
          <strong>{formatMoneyM(aav)}</strong>
        </div>
        <div>
          <span>Term</span>
          <strong>{yrs ? `${yrs} yr` : "—"}</strong>
        </div>
        <div>
          <span>Status</span>
          <strong>{getStatusLabel(row)}</strong>
        </div>
        {ext.likelyAav ? (
          <div>
            <span>Extension</span>
            <strong>{formatMoneyM(ext.likelyAav)} × {ext.likelyTerm}yr</strong>
          </div>
        ) : null}
        {buy.totalCost != null ? (
          <div>
            <span>Buyout</span>
            <strong>{formatMoneyM(buy.totalCost)}</strong>
          </div>
        ) : null}
      </div>

      {displayTags.length || (row.clause_label && row.clause_label !== "None") ? (
        <div className="cap-action-panel__tags">
          {row.clause_label && row.clause_label !== "None" ? (
            <span className={chipClass(row.clause_label)}>{row.clause_label}</span>
          ) : null}
          {displayTags.slice(0, 4).map((tag) => (
            <span key={tag} className={chipClass(tag)}>{tag}</span>
          ))}
        </div>
      ) : null}

      <div className="cap-action-panel__actions">
        {actions.length ? (
          actions.map((a) => (
            <button
              key={a.id}
              type="button"
              className={`cap-action-btn${a.id === "buyout" ? " cap-action-btn--danger" : ""}`}
              disabled={busy}
              onClick={() => onAction(a.id, row)}
            >
              {a.label}
            </button>
          ))
        ) : (
          <span className="cap-chip">No contract actions</span>
        )}
      </div>

      <ContractRosterMoves row={row} onMoved={onRosterMoved} />
    </section>
  );
}

const STOCK_ICON = { breakout: "↑↑", rising: "↑", falling: "↓", stable: "→" };
const STOCK_WORD = { breakout: "Breakout", rising: "Rising", falling: "Falling", stable: "Stable" };

function formatSvPct(v) {
  const n = safeNum(v, NaN);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(3).replace(/^0/, "");
}

function formatGaa(v) {
  const n = safeNum(v, NaN);
  return Number.isFinite(n) ? n.toFixed(2) : "—";
}

function isGoalieRow(row) {
  const s = row?.season_stats || {};
  return Boolean(s.is_goalie) || String(row?.position || "").toUpperCase() === "G";
}

function primaryStatValue(row) {
  const s = row?.season_stats || {};
  if (isGoalieRow(row)) return formatSvPct(s.save_pct);
  return s.points != null ? String(s.points) : "—";
}

function StockTrend({ row, showLabel = false }) {
  const dir = row?.stock_direction || "stable";
  const icon = STOCK_ICON[dir] || "→";
  const word = STOCK_WORD[dir] || "Stable";
  return (
    <span className={`cap-fa-trend cap-fa-trend--${dir}`} title={row?.stock_reason || word}>
      <span aria-hidden="true">{icon}</span>
      {showLabel ? <em>{word}</em> : null}
    </span>
  );
}

function FreeAgentHeader() {
  return (
    <div className="cap-board-head cap-fa-head">
      <span>Player</span>
      <span title="Position">Pos</span>
      <span title="Age">Age</span>
      <span title="Current league the player is featuring in">League</span>
      <span title="Projected games played this season">GP</span>
      <span title="Projected points (skaters) / save % (goalies) — from attributes, not game results">Pts/SV%</span>
      <span title="Overall rating">OVR</span>
      <span title="Market stock trend">Trend</span>
      <span title="Asking average annual value">Ask</span>
    </div>
  );
}

function FreeAgentRow({ row, onSelect, isSelected = false }) {
  const ovr = safeNum(row.overall ?? row.ovr);
  const s = row.season_stats || {};
  const goalie = isGoalieRow(row);
  const player = buildHeadshotPlayer(row);

  return (
    <button
      type="button"
      className={`cap-contract-card cap-contract-row cap-fa-row${isSelected ? " is-selected" : ""}`}
      onClick={() => onSelect(row)}
    >
      <span className="cap-contract-player">
        <PlayerHeadshot player={player} size="sm" className="cap-card-headshot" />
        <span className="cap-contract-meta cap-contract-row__identity">
          <strong className="cap-contract-row__name">{safeText(row.name, "Unnamed Player")}</strong>
          <em className="cap-contract-row__sub">{safeText(row.role || row.position)}</em>
        </span>
      </span>

      <span className="cap-contract-row__cell cap-contract-row__pos">{safeText(row.position)}</span>
      <span className="cap-contract-row__cell">{row.age || "—"}</span>
      <span className="cap-contract-row__cell cap-fa-league" title={row.current_team || row.current_league || ""}>
        {safeText(row.current_league, "—")}
      </span>
      <span className="cap-contract-row__cell">{s.gp != null ? s.gp : "—"}</span>
      <span className="cap-contract-row__cell">
        <strong>{primaryStatValue(row)}</strong>
        <em>{goalie ? "SV%" : "PTS"}</em>
      </span>
      <span className="cap-contract-row__cell cap-contract-row__ovr">
        <strong>{ovr || "—"}</strong>
        <em>OVR</em>
      </span>
      <span className="cap-contract-row__cell"><StockTrend row={row} /></span>
      <span className="cap-contract-row__cell">
        <strong>{formatMoneyM(row.asking_aav ?? row.askingAav)}</strong>
        <em>{row.asking_term ? `${row.asking_term}yr` : ""}</em>
      </span>
    </button>
  );
}

const FA_POS = [["all", "All"], ["C", "C"], ["LW", "LW"], ["RW", "RW"], ["D", "D"], ["G", "G"]];
const FA_AGE = [["all", "Any age"], ["u23", "23 & under"], ["24-29", "24–29"], ["30+", "30+"]];
const FA_OVR = [["all", "Any OVR"], ["80", "80+"], ["70", "70–79"], ["u70", "Under 70"]];

function FreeAgentsTab({ data, onSelect, selectedId }) {
  const [search, setSearch] = useState("");
  const [pos, setPos] = useState("all");
  const [age, setAge] = useState("all");
  const [ovrBand, setOvrBand] = useState("all");

  const rows = useMemo(() => {
    let list = [...safeArray(data.free_agents)];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((r) =>
        [r.name, r.position, r.current_league, r.role]
          .map((v) => safeText(v, "").toLowerCase())
          .join(" ")
          .includes(q),
      );
    }

    if (pos !== "all") list = list.filter((r) => String(r.position || "").toUpperCase() === pos);

    if (age !== "all") {
      list = list.filter((r) => {
        const a = safeNum(r.age);
        if (age === "u23") return a <= 23;
        if (age === "24-29") return a >= 24 && a <= 29;
        return a >= 30;
      });
    }

    if (ovrBand !== "all") {
      list = list.filter((r) => {
        const o = safeNum(r.overall ?? r.ovr);
        if (ovrBand === "80") return o >= 80;
        if (ovrBand === "70") return o >= 70 && o < 80;
        return o < 70;
      });
    }

    // Default view leads with skaters (sorted by OVR); goalies fall to the bottom so the
    // list isn't goalie-flooded. Filtering position to "G" surfaces goalies directly.
    return list.sort((a, b) => {
      const ga = String(a.position || "").toUpperCase() === "G" ? 1 : 0;
      const gb = String(b.position || "").toUpperCase() === "G" ? 1 : 0;
      if (ga !== gb) return ga - gb;
      return safeNum(b.overall ?? b.ovr) - safeNum(a.overall ?? a.ovr);
    });
  }, [data.free_agents, search, pos, age, ovrBand]);

  return (
    <section className="cap-office-panel cap-board-panel cap-fa-panel">
      <div className="cap-ledger-toolbar cap-board-toolbar cap-fa-toolbar">
        <input
          className="cap-search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search free agents"
        />
        <select className="cap-fa-select" value={pos} onChange={(e) => setPos(e.target.value)} title="Filter by position">
          {FA_POS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <select className="cap-fa-select" value={age} onChange={(e) => setAge(e.target.value)} title="Filter by age">
          {FA_AGE.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <select className="cap-fa-select" value={ovrBand} onChange={(e) => setOvrBand(e.target.value)} title="Filter by overall">
          {FA_OVR.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      </div>

      <FreeAgentHeader />

      <div className="cap-contract-list cap-board-list cap-fa-list">
        {rows.length ? (
          rows.map((row) => (
            <FreeAgentRow
              key={row.player_id || row.id || row.name}
              row={row}
              onSelect={onSelect}
              isSelected={selectedId === (row.player_id || row.id)}
            />
          ))
        ) : (
          <div className="cap-empty-state">No free agents match.</div>
        )}
      </div>
    </section>
  );
}

function FreeAgentDetailPanel({ row, detail, loading, onClose, onSign, busy, slots, capSpace }) {
  if (!row) {
    return (
      <section className="cap-action-panel cap-action-panel--empty">
        <p>Select a free agent to view projected-season stats and sign him.</p>
      </section>
    );
  }

  // Header comes from the light list row immediately; heavy stats load on demand.
  const fa = detail?.free_agent || null;
  const ovr = safeNum(row.overall ?? row.ovr);
  const ask = safeNum(row.asking_aav ?? row.askingAav);
  const term = safeNum(row.asking_term ?? row.askingTerm);
  const goalie = isGoalieRow(fa || row);
  const s = fa?.season_stats || null;
  const prev = fa?.previous_season_stats || null;
  const player = buildHeadshotPlayer(row);
  const openSlots = safeNum(detail?.open_contract_slots ?? slots?.open, NaN);
  const effCap = Number.isFinite(safeNum(detail?.cap_space_m, NaN)) ? safeNum(detail.cap_space_m) : capSpace;
  const spaceOk = Number.isFinite(effCap) ? ask <= effCap : true;
  const slotOk = Number.isFinite(openSlots) ? openSlots > 0 : true;
  const eligible = spaceOk && slotOk;
  const blocker = !spaceOk ? "Not enough cap space" : (!slotOk ? "No open contract slot" : null);

  const capAfter = Number.isFinite(effCap) ? effCap - ask : null;

  return (
    <section className="cap-action-panel cap-fa-detail">
      <button type="button" className="cap-action-panel__close" onClick={onClose} aria-label="Clear selection">
        ×
      </button>

      <div className="cap-fa-detail__id">
        <div className="cap-fa-detail__idhead">
          <PlayerHeadshot player={player} size="md" className="cap-action-panel__headshot" />
          <div className="cap-fa-detail__idtext">
            <h3>{safeText(row.name, "Unnamed Player")}</h3>
            <p>
              {safeText(row.position)} · {row.age || "—"} · OVR {ovr || "—"}
              {row.potential ? ` · POT ${row.potential}` : ""}
            </p>
            <p className="cap-fa-detail__where">
              {safeText(row.current_team || row.current_league, "Unsigned")}
              {row.current_team && row.current_league ? ` · ${row.current_league}` : ""}
            </p>
          </div>
        </div>

        <div className="cap-fa-detail__trendline">
          <StockTrend row={fa || row} showLabel />
          {fa?.stock_reason ? <span className="cap-fa-detail__reason">{fa.stock_reason}</span> : null}
          <span
            className="cap-fa-proj-tag"
            title="Projected from the player's attributes and role — not live game-ledger results."
          >
            Projected
          </span>
        </div>
      </div>

      <div className="cap-fa-detail__body">
        <div className="cap-fa-detail__block">
          <h4 className="cap-fa-detail__blocktitle">
            {fa?.season ? `${fa.season} season` : "This season"} · projected
          </h4>
          {loading || !s ? (
            <div className="cap-fa-detail__loading">{loading ? "Loading stats…" : "No projection yet"}</div>
          ) : (
            <>
              <div className="cap-action-panel__stats cap-fa-detail__stats">
                {goalie ? (
                  <>
                    <div><span>GP</span><strong>{s.gp ?? "—"}</strong></div>
                    <div><span>W</span><strong>{s.wins ?? "—"}</strong></div>
                    <div><span title="Save percentage">SV%</span><strong>{formatSvPct(s.save_pct)}</strong></div>
                    <div><span title="Goals-against average">GAA</span><strong>{formatGaa(s.gaa)}</strong></div>
                    <div><span title="Shutouts">SO</span><strong>{s.shutouts ?? "—"}</strong></div>
                  </>
                ) : (
                  <>
                    <div><span>GP</span><strong>{s.gp ?? "—"}</strong></div>
                    <div><span>G</span><strong>{s.goals ?? "—"}</strong></div>
                    <div><span>A</span><strong>{s.assists ?? "—"}</strong></div>
                    <div><span>PTS</span><strong>{s.points ?? "—"}</strong></div>
                    <div><span title="Points per game">P/GP</span><strong>{s.ppg ?? "—"}</strong></div>
                  </>
                )}
              </div>
              {prev ? (
                <p className="cap-fa-detail__prev">
                  Last season ({safeText(fa.previous_season_league, "—")}):{" "}
                  {goalie
                    ? `${prev.gp || 0} GP · ${formatSvPct(prev.save_pct)} SV%`
                    : `${prev.gp || 0} GP · ${prev.points || 0} PTS`}
                </p>
              ) : null}
            </>
          )}
        </div>

        <div className="cap-fa-detail__block">
          <h4 className="cap-fa-detail__blocktitle">Contract & fit</h4>
          <div className="cap-action-panel__stats cap-fa-detail__terms">
            <div><span>Role</span><strong>{safeText(fa?.role || row.role)}</strong></div>
            <div><span title="Asking average annual value">Ask AAV</span><strong>{formatMoneyM(ask)}</strong></div>
            <div><span>Term</span><strong>{term ? `${term} yr` : "—"}</strong></div>
            <div><span title="Projected cap space after this signing">Cap After</span><strong>{capAfter != null ? formatMoneyM(capAfter) : "—"}</strong></div>
            <div><span title="Open contract slots">Slots</span><strong>{Number.isFinite(openSlots) ? `${openSlots} open` : "—"}</strong></div>
          </div>
          {fa ? (
            <p className="cap-fa-detail__risk">
              <span>Risk</span> {safeText(fa.risk)} <span>·</span> <span>Fit</span> {safeText(fa.fit)}
            </p>
          ) : null}
        </div>
      </div>

      <div className="cap-fa-detail__cta">
        <button
          type="button"
          className="cap-action-btn cap-action-btn--sign"
          disabled={busy || !eligible}
          onClick={() => onSign(row)}
          title={blocker || "Offer the player his asking terms"}
        >
          {eligible ? "Sign Player" : (blocker || "Unavailable")}
        </button>
        {blocker ? <span className="cap-fa-detail__blocker">{blocker}</span> : null}
      </div>
    </section>
  );
}

function CapTile({ label, value, danger = false }) {
  return (
    <article className={`cap-ledger-stat-card cap-tile ${danger ? "is-danger" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function RfaSheetsTab({
  data,
  onSelect,
  selectedId,
  sheetDraft,
  setSheetDraft,
  onFileSheet,
  onResolveSheet,
  busy,
}) {
  const ownRfa = safeArray(data?.rfa_rights);
  const targets = safeArray(data?.offer_sheet_targets);
  const pending = safeArray(data?.pending_offer_sheets);
  const selected =
    targets.find((r) => String(r.player_id) === String(selectedId)) ||
    ownRfa.find((r) => String(r.player_id) === String(selectedId)) ||
    null;

  return (
    <section className="cap-board-panel">
      <div className="cap-board-toolbar">
        <h2>RFA desk</h2>
        <p className="cap-empty-state" style={{ margin: 0, padding: 0 }}>
          Qualify your RFAs, settle arbitration, match sheets against you, or file sheets on other clubs.
        </p>
      </div>

      {pending.length ? (
        <div className="cap-contract-list" style={{ marginBottom: "0.75rem" }}>
          <h3 style={{ fontSize: "0.75rem", margin: "0 0 0.35rem" }}>Pending offer sheets</h3>
          {pending.map((sheet) => (
            <div key={`${sheet.player_id}-${sheet.offering_team_id}`} className="cap-fa-row" style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
              <strong>{safeText(sheet.name || sheet.player_id)}</strong>
              <span>{formatMoneyM(sheet.aav_m)} × {sheet.years || 1}yr</span>
              <span>{safeText(sheet.compensation_label || sheet.compensation_tier)}</span>
              {sheet.days_remaining != null || sheet.expires_day != null ? (
                <span className="cap-chip cap-chip--warn">
                  Match in {Math.max(0, Number(sheet.days_remaining ?? 0))}d
                </span>
              ) : null}
              <button type="button" className="cap-action-btn" disabled={busy} onClick={() => onResolveSheet("match", sheet)}>
                Match
              </button>
              <button type="button" className="cap-action-btn cap-action-btn--danger" disabled={busy} onClick={() => onResolveSheet("decline", sheet)}>
                Decline
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <div className="cap-board-head" style={{ gridTemplateColumns: "2fr 0.5fr 0.5fr 0.8fr 1fr" }}>
        <span>Your RFA rights</span>
        <span>Pos</span>
        <span>OVR</span>
        <span>QO</span>
        <span>Status</span>
      </div>
      <div className="cap-contract-list cap-board-list" style={{ maxHeight: "28vh", marginBottom: "0.85rem" }}>
        {ownRfa.length ? (
          ownRfa.map((row) => (
            <button
              key={row.player_id}
              type="button"
              className={`cap-fa-row ${selectedId === row.player_id ? "is-selected" : ""}`}
              style={{ display: "grid", gridTemplateColumns: "2fr 0.5fr 0.5fr 0.8fr 1fr", width: "100%", textAlign: "left" }}
              onClick={() => onSelect(row)}
            >
              <span>{safeText(row.name)}</span>
              <span>{safeText(row.position)}</span>
              <span>{safeNum(row.overall)}</span>
              <span>{formatMoneyM(row.qualifying_offer_aav_m)}</span>
              <span>
                {row.arbitration_filed ? "Arb filed" : row.offer_sheet_pending ? "Sheet pending" : "Rights"}
              </span>
            </button>
          ))
        ) : (
          <div className="cap-empty-state">No RFA rights on file.</div>
        )}
      </div>

      <div className="cap-board-head" style={{ gridTemplateColumns: "2fr 0.5fr 0.5fr 0.8fr 1.2fr" }}>
        <span>Offer-sheet targets</span>
        <span>Pos</span>
        <span>OVR</span>
        <span>QO</span>
        <span>Comp @ suggest</span>
      </div>
      <div className="cap-contract-list cap-board-list" style={{ maxHeight: "28vh" }}>
        {targets.length ? (
          targets.map((row) => (
            <button
              key={`${row.rights_team_id}-${row.player_id}`}
              type="button"
              className={`cap-fa-row ${selectedId === row.player_id ? "is-selected" : ""}`}
              style={{ display: "grid", gridTemplateColumns: "2fr 0.5fr 0.5fr 0.8fr 1.2fr", width: "100%", textAlign: "left" }}
              onClick={() => {
                onSelect(row);
                setSheetDraft({
                  aav_m: String(row.suggested_aav_m || row.qualifying_offer_aav_m || 1),
                  years: "4",
                });
              }}
            >
              <span>{safeText(row.name)} <em style={{ opacity: 0.65 }}>{safeText(row.rights_team_id)}</em></span>
              <span>{safeText(row.position)}</span>
              <span>{safeNum(row.overall)}</span>
              <span>{formatMoneyM(row.qualifying_offer_aav_m)}</span>
              <span>{safeText(row.compensation_preview?.label)}</span>
            </button>
          ))
        ) : (
          <div className="cap-empty-state">No other-club RFAs available for offer sheets.</div>
        )}
      </div>

      {selected?.offer_sheet_eligible && selected?.rights_team_id ? (
        <div className="cap-action-panel" style={{ marginTop: "0.75rem" }}>
          <h3>File offer sheet — {safeText(selected.name)}</h3>
          <div className="cap-action-panel__stats">
            <label>
              <span>AAV ($M)</span>
              <input
                type="number"
                step="0.025"
                value={sheetDraft.aav_m}
                onChange={(e) => setSheetDraft((d) => ({ ...d, aav_m: e.target.value }))}
              />
            </label>
            <label>
              <span>Years</span>
              <input
                type="number"
                min="1"
                max="8"
                value={sheetDraft.years}
                onChange={(e) => setSheetDraft((d) => ({ ...d, years: e.target.value }))}
              />
            </label>
          </div>
          <button
            type="button"
            className="cap-action-btn"
            disabled={busy}
            onClick={() => onFileSheet(selected, sheetDraft)}
          >
            Submit offer sheet
          </button>
        </div>
      ) : null}
    </section>
  );
}

function CapTab({ data, slotSummary }) {
  const snap = getCapSnap(data);
  const used = safeNum(snap.total_cap_hit_m);
  const limit = safeNum(snap.upper_limit_m);
  const space = safeNum(snap.usable_cap_space_m, limit - used);
  const pct = limit > 0 ? (used / limit) * 100 : 0;
  const slots = slotSummary || buildRosterSlotSummary(null, snap, data?.team_id);

  const meterClass =
    pct >= 100 ? "is-danger" :
    pct >= 95 ? "is-tight" :
    pct >= 85 ? "is-warning" :
    "is-comfortable";

  const tiles = [
    ["Buyout", formatMoneyM(snap.buyout_cap_hit_m)],
    ["Dead", formatMoneyM(snap.other_dead_cap_m || 0)],
    ["Buried", formatMoneyM(snap.buried_cap_hit_m)],
    ["Retained", formatMoneyM(snap.retained_salary_m)],
    ["NHL SPCs", slots.spc ? `${slots.spc.used}/${slots.spc.limit}` : "—"],
    ["NHL", `${slots.nhl.used}/${slots.nhl.limit}`],
    ["AHL", `${slots.ahl.used}/${slots.ahl.limit}`],
    ["ECHL", `${slots.echl.used}/${slots.echl.limit}`],
    ["Next Yr", formatMoneyM(snap.projected_next_year_upper_limit_m)],
  ];

  return (
    <section className="cap-office-panel cap-pressure-panel">
      <div className={`cap-meter-board ${space < 0 ? "is-over" : ""}`}>
        <div className="cap-meter-board__main">
          <p className="cap-meter-label">{space < 0 ? "OVER" : "SPACE"}</p>
          <h2>{formatMoneyM(Math.abs(space))}</h2>
          <p className="cap-meter-pct">
            {formatMoneyM(used)} / {formatMoneyM(limit)} · {formatPct(used, limit)}
          </p>
        </div>

        <div className="cap-bar cap-bar-hero">
          <div
            className={`cap-bar-fill ${meterClass}`}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </div>
      </div>

      <div className="cap-ledger-grid cap-cap-grid">
        {tiles.map(([label, value]) => (
          <CapTile key={label} label={label} value={value} />
        ))}
      </div>

      {safeArray(snap.warnings).length ? (
        <div className="cap-ledger-attention cap-warning-strip">
          {safeArray(snap.warnings).slice(0, 5).map((w) => (
            <span key={w} className={chipClass(w)}>{w}</span>
          ))}
        </div>
      ) : (
        <div className="cap-ledger-attention cap-warning-strip">
          <span className="cap-chip cap-chip-good">Clean</span>
        </div>
      )}
    </section>
  );
}

function OfficeTopBar({ team, snap, slotSummary, teamLogo, onBack }) {
  const space = safeNum(snap.usable_cap_space_m ?? team.cap_space);
  const hit = safeNum(snap.total_cap_hit_m ?? team.cap_hit);
  const slots = slotSummary || { compact: "—", subline: "" };

  return (
    <header
      className={`cap-ledger-hero cap-office-topbar ${teamLogo ? "has-logo" : ""}`}
      style={teamLogo ? { "--team-logo-url": `url("${teamLogo}")` } : undefined}
    >
      <div className="cap-office-team">
        <button type="button" className="cap-office-back" onClick={onBack}>
          ←
        </button>

        {teamLogo ? (
          <img className="cap-office-logo" src={teamLogo} alt="" />
        ) : (
          <span className="cap-office-logo cap-office-logo-fallback">
            {safeText(team.abbr || team.abbreviation || team.name, "TM").slice(0, 3)}
          </span>
        )}

        <div className="cap-ledger-hero-main">
          <p className="cap-office-kicker">Contract Office</p>
          <h1>{team.name || "Franchise"}</h1>
        </div>
      </div>

      <div className="cap-ledger-hero-stats cap-hero-trio cap-office-metrics">
        <div className={space < 0 ? "is-danger" : ""}>
          <span>{space < 0 ? "Over" : "Space"}</span>
          <strong>{formatMoneyM(Math.abs(space))}</strong>
        </div>

        <div>
          <span>Hit</span>
          <strong>{formatMoneyM(hit)}</strong>
          <div className="cap-payroll-meter" aria-hidden="true">
            <i style={{ width: `${Math.max(4, Math.min(100, (hit / Math.max(hit + Math.max(space, 0), 1)) * 100))}%` }} />
          </div>
        </div>

        <div className="cap-metric-roster">
          <span>Roster</span>
          <strong>{slots.compact}</strong>
          {slots.subline ? <em>{slots.subline}</em> : null}
        </div>
      </div>
    </header>
  );
}

export default function CapLedger() {
  const {
    franchiseState,
    setScreen,
    setCapLedgerTab,
    capLedgerTab,
    refreshFranchise,
    openFranchiseEvent,
    onReopenOffseasonStage,
  } = useGameUI();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [faDetail, setFaDetail] = useState(null);
  const [faDetailLoading, setFaDetailLoading] = useState(false);
  const [sheetDraft, setSheetDraft] = useState({ aav_m: "2.000", years: "4" });

  const tab =
    capLedgerTab === "salaryCap"
      ? "cap"
      : capLedgerTab === "decisions"
        ? "ledger"
      : capLedgerTab === "freeAgency" || capLedgerTab === "freeAgents"
        ? "freeAgents"
      : capLedgerTab === "rfa" || capLedgerTab === "offerSheets"
        ? "rfa"
      : capLedgerTab === "contracts"
        ? "ledger"
        : capLedgerTab === "cap"
          ? "cap"
          : capLedgerTab || "ledger";

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const payload = await getContractOffice();
      setData(payload);
    } catch (e) {
      setError(String(e?.message || "Contract office unavailable"));

      const snap = franchiseState?.team?.cap_snapshot;

      if (snap) {
        setData({
          ok: true,
          cap_snapshot: snap,
          team: franchiseState.team,
          contracts: [],
          expiring: [],
          rfa_rights: [],
          buyout_candidates: [],
          cap_casualty_candidates: [],
        });
      }
    } finally {
      setLoading(false);
    }
  }, [franchiseState?.stats_revision, franchiseState?.session_id]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Lazily load the heavy free-agent detail only for the selected player (the list payload
  // stays light). Cancels stale responses when the selection changes.
  const selectedFaId = tab === "freeAgents" ? (selected?.player_id || selected?.id) : null;
  useEffect(() => {
    if (!selectedFaId) {
      setFaDetail(null);
      setFaDetailLoading(false);
      return undefined;
    }
    let cancelled = false;
    setFaDetail(null);
    setFaDetailLoading(true);
    getFreeAgentDetail(selectedFaId)
      .then((res) => {
        if (!cancelled) setFaDetail(res && res.ok ? res : null);
      })
      .catch(() => {
        if (!cancelled) setFaDetail(null);
      })
      .finally(() => {
        if (!cancelled) setFaDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedFaId]);

  const snap = getCapSnap(data || {});
  const team = data?.team || franchiseState?.team || {};
  const teamLogo = resolveFranchiseTeamLogo(team, team.name);
  const slotSummary = useMemo(
    () => buildRosterSlotSummary(franchiseState, snap, data?.team_id || team?.id),
    [franchiseState, snap, data?.team_id, team?.id],
  );

  const handleAction = async (action, row) => {
    const pid = row.player_id || row.id;

    if (!pid) {
      setError("Missing player id");
      return;
    }

    setBusy(true);
    setError("");

    try {
      let result;
      const base = { player_id: pid };

      if (action === "re-sign") {
        const aav = safeNum(row.aav_m) * 1.05 || 0.95;
        const years = Math.max(1, safeNum(row.years_remaining, 1));

        result = await reSignContract({
          ...base,
          aav_m: aav,
          years,
          context: "re_sign",
        });
      } else if (action === "qualify-rfa") {
        result = await qualifyRfa(base);
      } else if (action === "release-rights") {
        result = await releaseRfaRights(base);
      } else if (action === "buyout") {
        result = await buyoutContract(base);
      } else if (action === "waive") {
        result = await waiveContract(base);
      } else if (action === "bury") {
        result = await buryContract(base);
      } else if (action === "arbitration-file") {
        result = await fileArbitration({
          ...base,
          player_ask_m: safeNum(row.player_ask_m || row.requested_cap_hit || row.qualifying_offer_aav_m, 1) * 1.15,
        });
      } else if (action === "arbitration-settle") {
        result = await settleArbitration(base);
      } else if (action === "match-offer-sheet") {
        result = await matchOfferSheet(base);
      } else if (action === "decline-offer-sheet") {
        result = await declineOfferSheet(base);
      }

      if (result?.office) {
        setData(result.office);
      } else {
        await loadData();
      }

      if (!result?.ok && result?.reason) {
        setError(result.reason);
      }

      setSelected(null);
    } catch (e) {
      setError(String(e?.message || "Action failed"));
    } finally {
      setBusy(false);
    }
  };

  const handleFileOfferSheet = async (row, draft) => {
    const pid = row.player_id || row.id;
    if (!pid || !row.rights_team_id) {
      setError("Missing offer-sheet target");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await submitOfferSheet({
        player_id: pid,
        rights_team_id: row.rights_team_id,
        aav_m: safeNum(draft.aav_m, 2),
        years: Math.max(1, safeNum(draft.years, 4)),
      });
      if (result?.office) setData(result.office);
      else await loadData();
      if (!result?.ok && result?.reason) setError(result.reason);
      else setSelected(null);
    } catch (e) {
      setError(String(e?.message || "Offer sheet failed"));
    } finally {
      setBusy(false);
    }
  };

  const handleResolveSheet = async (decision, sheet) => {
    const pid = sheet.player_id;
    if (!pid) return;
    setBusy(true);
    setError("");
    try {
      const result =
        decision === "match"
          ? await matchOfferSheet({ player_id: pid })
          : await declineOfferSheet({ player_id: pid });
      if (result?.office) setData(result.office);
      else await loadData();
      if (!result?.ok && result?.reason) setError(result.reason);
    } catch (e) {
      setError(String(e?.message || "Offer sheet resolution failed"));
    } finally {
      setBusy(false);
    }
  };

  const handleSign = async (row) => {
    const pid = row.player_id || row.id;

    if (!pid) {
      setError("Missing player id");
      return;
    }

    setBusy(true);
    setError("");

    try {
      const result = await signFreeAgent({
        player_id: pid,
        aav_m: safeNum(row.asking_aav ?? row.askingAav),
        years: Math.max(1, safeNum(row.asking_term ?? row.askingTerm, 1)),
      });

      if (result?.office) {
        setData(result.office);
      } else {
        await loadData();
      }

      if (!result?.ok) {
        setError(result?.reason || "The player rejected the offer.");
      } else {
        setSelected(null);
      }
    } catch (e) {
      setError(String(e?.message || "Signing failed"));
    } finally {
      setBusy(false);
    }
  };

  const selectTab = (id) => {
    setSelected(null);
    setCapLedgerTab(id);
  };

  const handleBack = () => {
    if (typeof setScreen === "function") {
      setScreen(SCREENS.HUB);
    }
  };

  return (
    <div className="game-screen cap-ledger-screen cap-office-screen">
      <OfficeTopBar
        team={team}
        snap={snap}
        slotSummary={slotSummary}
        teamLogo={teamLogo}
        onBack={handleBack}
      />

      {error ? (
        <div className="cap-empty-state cap-error">
          {error}
        </div>
      ) : null}

      <nav className="cap-ledger-tabs cap-office-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`cap-ledger-tab ${tab === t.id ? "cap-ledger-tab-active" : ""}`}
            onClick={() => selectTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {loading ? (
        <div className="cap-empty-state">Loading…</div>
      ) : null}

      {!loading && data ? (
        <main className="cap-ledger-main cap-office-main">
          {tab === "ledger" ? (
            <div className="cap-board-layout">
              <LedgerTab
                data={data}
                onSelect={setSelected}
                selectedId={selected?.player_id || selected?.id}
              />
              <ContractActionPanel
                row={selected}
                onClose={() => setSelected(null)}
                onAction={handleAction}
                busy={busy}
                onRosterMoved={() => loadData()}
              />
            </div>
          ) : null}

          {tab === "freeAgents" ? (
            <div className="cap-board-layout">
              <div className="cap-ledger-empty" style={{ padding: "2rem", maxWidth: 520 }}>
                <h3 style={{ marginTop: 0 }}>Free Agency Wire</h3>
                <p>
                  Free agency runs on the same offseason Free Agency Wire used in the
                  timeline — bids, Sim Day, and the signing desk.
                </p>
                <button
                  type="button"
                  className="cap-ledger-primary-btn"
                  disabled={busy}
                  onClick={() => {
                    if (typeof setScreen === "function") setScreen(SCREENS.FREE_AGENCY);
                  }}
                >
                  Open Free Agency Wire
                </button>
              </div>
            </div>
          ) : null}

          {tab === "rfa" ? (
            <div className="cap-board-layout">
              <RfaSheetsTab
                data={data}
                onSelect={setSelected}
                selectedId={selected?.player_id || selected?.id}
                sheetDraft={sheetDraft}
                setSheetDraft={setSheetDraft}
                onFileSheet={handleFileOfferSheet}
                onResolveSheet={handleResolveSheet}
                busy={busy}
              />
              <ContractActionPanel
                row={selected && !selected.rights_team_id ? selected : null}
                onClose={() => setSelected(null)}
                onAction={handleAction}
                busy={busy}
                onRosterMoved={() => loadData()}
              />
            </div>
          ) : null}

          {tab === "cap" ? (
            <CapTab data={data} slotSummary={slotSummary} />
          ) : null}
        </main>
      ) : null}
    </div>
  );
}