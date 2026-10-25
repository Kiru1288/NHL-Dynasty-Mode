import { gameConfirm } from "../components/common/gameConfirm";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getContractOffice, qualifyRfa, releaseRfaRights, buyoutContract, waiveContract, buryContract, submitOfferSheet, matchOfferSheet, declineOfferSheet, fileArbitration, settleArbitration, getRosterMoves, moveRosterPlayer } from "../services/franchiseService";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import PlayerHeadshot from "../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import CapContractNegotiation from "../components/contracts/CapContractNegotiation";

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
  { id: "ledger", label: "Contracts" },
  { id: "rfa", label: "RFAs & offer sheets" },
  { id: "cap", label: "Cap breakdown" },
];

const FILTERS = [
  { id: "all", label: "All" },
  { id: "expiring", label: "Expiring" },
  { id: "rfa", label: "RFA" },
  { id: "core", label: "Core 85+" },
  { id: "bargain", label: "Bargains" },
  { id: "bad", label: "Overpaid" },
  { id: "clause", label: "Clauses" },
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
  return `$${n.toFixed(2)}M`;
}

function getCapSnap(data) {
  const snap = { ...(data?.cap_snapshot || data?.team_cap || {}) };
  const used = safeNum(snap.total_cap_hit_m);
  const limit = safeNum(snap.upper_limit_m);
  const team = data?.team || {};
  snap.usable_cap_space_m = safeNum(
    snap.usable_cap_space_m,
    safeNum(team.cap_space, safeNum(team.cap_space_m, limit - used)),
  );
  return snap;
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

function getStatusLabel(row) {
  const clause = safeText(row?.clause_label, "");
  const years = safeNum(row?.years_remaining ?? row?.yearsRemaining, 0);

  if (row?.extension_signed) return "EXT";
  if (years > 1) {
    if (clause && clause !== "None" && clause !== "—") return clause;
    return "Signed";
  }
  const expiry = safeText(row?.expiry_status, "");
  if (expiry && expiry !== "—") return expiry;
  if (clause && clause !== "None" && clause !== "—") return clause;
  return "Exp";
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

function ovrPopTone(ovr) {
  const n = safeNum(ovr, 0);
  if (n >= 92) return "gold";
  if (n >= 86) return "cyan";
  if (n >= 80) return "blue";
  return "muted";
}

function contractEndLabel(row) {
  const yrs = safeNum(row?.years_remaining ?? row?.yearsRemaining, 0);
  const endYear = row?.expiry_year ?? row?.expiryYear;
  if (yrs <= 0) return "—";
  return endYear ? `${yrs} yr${yrs === 1 ? "" : "s"} · thru ${endYear}` : `${yrs} yr${yrs === 1 ? "" : "s"}`;
}

function contractStatusTone(row) {
  const status = getStatusLabel(row);
  const expiry = String(row?.expiry_status || "").toUpperCase();
  const tags = safeArray(row?.tags);
  if (/UFA/.test(expiry) || /UFA/.test(status)) return "ufa";
  if (/RFA/.test(expiry) || /RFA/.test(status)) return "rfa";
  if (/ELC|ENTRY/.test(status) || tags.some((t) => /elc/i.test(String(t)))) return "elc";
  if (/NTC|NMC/.test(status)) return "ntc";
  return "neutral";
}

function isExpiringRow(row) {
  const yrs = safeNum(row?.years_remaining ?? row?.yearsRemaining, 99);
  return !row?.extension_signed && (yrs <= 1 || row?.contract_status === "expiring" || row?.pending_july1_expiry);
}

function clauseOf(row) {
  const c = safeText(row?.clause_label, "");
  return c && c !== "None" && c !== "—" ? c : "";
}

function ContractBoardRow({ row, onSelect, isSelected = false }) {
  const ovr = safeNum(row.overall ?? row.ovr);
  const aav = safeNum(row.aav_m ?? row.aav);
  const value = getValueLabel(row);
  // Tone follows the label shown, not the raw tag soup (which can say "Bad Deal" and "Bargain" at once).
  const valueTone = valueKey(value);
  const player = buildHeadshotPlayer(row);
  const expiring = isExpiringRow(row);
  const expiry = String(row?.expiry_status || "").toUpperCase();
  const daysLeft = Number(row?.days_to_expiry ?? row?.daysToExpiry ?? row?.ufa_days);
  const clause = clauseOf(row);
  const tone = contractStatusTone(row);

  // One row, no boxes: who · rating · money · how long · what happens next.
  return (
    <button
      type="button"
      className={`cl-row${isSelected ? " is-selected" : ""}${expiring ? " is-expiring" : ""}`}
      onClick={() => onSelect(row)}
      aria-pressed={isSelected}
    >
      <span className="cl-row__who">
        <PlayerHeadshot player={player} size="sm" className="cl-row__face" />
        <span className="cl-row__name">
          <strong>{safeText(row.name, "Unnamed Player")}</strong>
          <em>{safeText(row.position)} · {row.age ? `${row.age}` : "—"}</em>
        </span>
      </span>
      <span className={`cl-row__ovr tone-${ovrPopTone(ovr)}`}>{ovr || "—"}</span>
      <span className="cl-row__aav">{formatMoneyM(aav)}</span>
      <span className="cl-row__term">
        {row.extension_signed && row.extension_aav_m ? (
          <>
            <b>{contractEndLabel(row)}</b>
            <small>Ext. {formatMoneyM(row.extension_aav_m)}{row.extension_years ? ` × ${row.extension_years}` : ""}</small>
          </>
        ) : (
          <b>{contractEndLabel(row)}</b>
        )}
      </span>
      <span className="cl-row__status">
        {expiring ? (
          <span className={`cl-tag is-${tone}`}>
            {expiry || (Number(row?.age) > 0 && Number(row?.age) < 27 ? "RFA" : "UFA")}{Number.isFinite(daysLeft) ? ` in ${Math.max(0, Math.round(daysLeft))}d` : ""}
          </span>
        ) : tone === "elc" ? (
          <span className="cl-tag is-elc">ELC</span>
        ) : null}
        {clause ? <span className="cl-tag is-ntc">{clause}</span> : null}
        {valueTone !== "neutral" ? <span className={`cl-tag is-${valueTone}`}>{value}</span> : null}
      </span>
    </button>
  );
}

const SORTS = {
  aav: { label: "Cap hit", fn: (a, b) => safeNum(b.aav_m ?? b.aav) - safeNum(a.aav_m ?? a.aav) },
  ovr: { label: "OVR", fn: (a, b) => safeNum(b.overall ?? b.ovr) - safeNum(a.overall ?? a.ovr) },
  term: { label: "Term", fn: (a, b) => safeNum(a.years_remaining ?? a.yearsRemaining, 99) - safeNum(b.years_remaining ?? b.yearsRemaining, 99) },
  name: { label: "Player", fn: (a, b) => safeText(a.name, "").localeCompare(safeText(b.name, "")) },
};

function ContractBoardHeader({ sort, onSort }) {
  const col = (key, className) => (
    <button
      type="button"
      className={`cl-head__col ${className}${sort === key ? " is-sorted" : ""}`}
      onClick={() => onSort(key)}
      aria-pressed={sort === key}
    >
      {SORTS[key].label}
    </button>
  );
  return (
    <div className="cl-head" role="row">
      {col("name", "cl-head__who")}
      {col("ovr", "cl-head__ovr")}
      {col("aav", "cl-head__aav")}
      {col("term", "cl-head__term")}
      <span className="cl-head__col cl-head__status">Flags</span>
    </div>
  );
}

function matchesFilter(r, filter) {
  const text = `${r.contract_value_score || ""} ${safeArray(r.tags).join(" ")}`.toLowerCase();
  if (filter === "expiring") return !r.extension_signed && safeNum(r.years_remaining ?? r.yearsRemaining, 99) <= 1;
  if (filter === "rfa") return String(r.expiry_status || "").toUpperCase().includes("RFA");
  if (filter === "bad") return /bad|casualty|heavy|overpaid|risk/.test(text);
  if (filter === "bargain") return /bargain|deal|elc/.test(text);
  if (filter === "clause") return Boolean(r.clause_label && r.clause_label !== "None");
  if (filter === "core") return safeNum(r.overall ?? r.ovr) >= 85;
  return true;
}

function LedgerTab({ data, onSelect, selectedId }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("aav");
  const all = safeArray(data.contracts);

  const counts = useMemo(
    () => Object.fromEntries(FILTERS.map((f) => [f.id, all.filter((r) => matchesFilter(r, f.id)).length])),
    [all],
  );

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return all
      .filter((r) => matchesFilter(r, filter))
      .filter((r) => {
        if (!q) return true;
        return [r.name, r.position, r.expiry_status, r.contract_value_score, r.clause_label, ...safeArray(r.tags)]
          .map((v) => safeText(v, "").toLowerCase())
          .join(" ")
          .includes(q);
      })
      .sort((a, b) => SORTS[sort].fn(a, b) || SORTS.ovr.fn(a, b));
  }, [all, filter, search, sort]);

  return (
    <section className="cl-board" aria-label="Contracts">
      <div className="cl-board__toolbar">
        <div className="cl-filters" role="tablist" aria-label="Filter contracts">
          {FILTERS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={filter === item.id}
              className={`cl-filter${filter === item.id ? " is-active" : ""}`}
              onClick={() => setFilter(item.id)}
              disabled={item.id !== "all" && !counts[item.id]}
            >
              {item.label} <span>{counts[item.id] ?? 0}</span>
            </button>
          ))}
        </div>
        <input
          className="cl-search"
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search players"
          aria-label="Search players"
        />
      </div>

      <ContractBoardHeader sort={sort} onSort={setSort} />

      <div className="cl-board__list">
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
          <p className="cl-empty">No contracts match{search ? ` “${search}”` : " this filter"}.</p>
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
        const ok = await gameConfirm(
          `${row.name || "Player"} requires waivers to leave the NHL roster. Place on waivers and send down?`,
          { title: "Waivers required", confirmLabel: "Place on waivers" }
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
    <div className="cl-moves">
      <h4>Assignment</h4>
      <p className="cl-moves__meta">
        Location: {meta.location || "—"}
        {meta.nhl_gp != null ? ` · NHL GP ${meta.nhl_gp}` : ""}
        {meta.waiver_exempt ? " · Waiver exempt" : ""}
      </p>
      <div className="cl-panel__actions">
        {moves.length ? (
          moves.map((m) => (
            <button
              key={m.id}
              type="button"
              className="cl-btn"
              disabled={Boolean(busy) || m.enabled === false}
              title={m.reason || m.note || ""}
              onClick={() => runMove(m.id)}
            >
              {busy === m.id ? "…" : m.label}
              {m.requires_waivers ? " (waivers)" : ""}
            </button>
          ))
        ) : (
          <p className="cl-panel__note">{error || "No assignment moves for this player."}</p>
        )}
      </div>
      {note ? <p className="cl-moves__note">{note}</p> : null}
      {error && moves.length ? <p className="cl-moves__error">{error}</p> : null}
    </div>
  );
}

const CONTRACT_ACTION_ID_MAP = {
  negotiate_extension: "re-sign",
  negotiate: "re-sign",
  qualify_rfa: "qualify-rfa",
  walk_away: "release-rights",
  buyout: "buyout",
  waive: "waive",
  bury: "bury",
  arbitration_file: "arbitration-file",
  arbitration_settle: "arbitration-settle",
  match_offer_sheet: "match-offer-sheet",
  decline_offer_sheet: "decline-offer-sheet",
};

function mapContractActionId(id) {
  const key = String(id || "");
  return CONTRACT_ACTION_ID_MAP[key] || key;
}

const NEGOTIATE_ACTION_IDS = new Set(["re-sign", "negotiate", "negotiate-extension"]);

function isNegotiateAction(actionId) {
  const id = String(actionId || "").toLowerCase();
  return NEGOTIATE_ACTION_IDS.has(id) || id === "negotiate_extension";
}

function buildContractPanelActions(row) {
  const apiActions = safeArray(row.available_actions).filter(
    (a) => a && a.enabled !== false && a.id !== "view_dossier",
  );
  if (apiActions.length) {
    return apiActions.map((a) => ({
      id: mapContractActionId(a.id),
      label: safeText(a.label, "Action"),
    }));
  }

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
  return actions;
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
    <section className="cl-board cl-rfa" aria-label="RFAs and offer sheets">
      {pending.length ? (
        <div className="cl-rfa__group is-urgent">
          <h3>Offer sheets on your players <span>{pending.length}</span></h3>
          {pending.map((sheet) => (
            <div key={`${sheet.player_id}-${sheet.offering_team_id}`} className="cl-rfa__sheet">
              <strong>{safeText(sheet.name || sheet.player_id)}</strong>
              <span>{formatMoneyM(sheet.aav_m)} × {sheet.years || 1} yrs</span>
              <span>{safeText(sheet.compensation_label || sheet.compensation_tier)}</span>
              {sheet.days_remaining != null || sheet.expires_day != null ? (
                <span className="cl-tag is-rfa">Match in {Math.max(0, Number(sheet.days_remaining ?? 0))}d</span>
              ) : null}
              <button type="button" className="cl-btn cl-btn--primary" disabled={busy} onClick={() => onResolveSheet("match", sheet)}>
                Match
              </button>
              <button type="button" className="cl-btn cl-btn--danger" disabled={busy} onClick={() => onResolveSheet("decline", sheet)}>
                Decline
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <div className="cl-rfa__group">
        <h3>Your RFA rights <span>{ownRfa.length}</span></h3>
        <div className="cl-rfa__head"><span>Player</span><span>OVR</span><span>Qualifying offer</span><span>Status</span></div>
        <div className="cl-rfa__list">
          {ownRfa.length ? (
            ownRfa.map((row) => (
              <button
                key={row.player_id}
                type="button"
                className={`cl-rfa__row${selectedId === row.player_id ? " is-selected" : ""}`}
                aria-pressed={selectedId === row.player_id}
                onClick={() => onSelect(row)}
              >
                <span className="cl-rfa__name"><strong>{safeText(row.name)}</strong><em>{safeText(row.position)}</em></span>
                <span>{safeNum(row.overall) || "—"}</span>
                <span>{formatMoneyM(row.qualifying_offer_aav_m)}</span>
                <span>{row.arbitration_filed ? "Arbitration filed" : row.offer_sheet_pending ? "Sheet pending" : "Rights held"}</span>
              </button>
            ))
          ) : (
            <p className="cl-empty">No RFA rights on file.</p>
          )}
        </div>
      </div>

      <div className="cl-rfa__group">
        <h3>Offer-sheet targets <span>{targets.length}</span></h3>
        <div className="cl-rfa__head"><span>Player</span><span>OVR</span><span>Qualifying offer</span><span>Compensation</span></div>
        <div className="cl-rfa__list">
          {targets.length ? (
            targets.map((row) => (
              <button
                key={`${row.rights_team_id}-${row.player_id}`}
                type="button"
                className={`cl-rfa__row${selectedId === row.player_id ? " is-selected" : ""}`}
                aria-pressed={selectedId === row.player_id}
                onClick={() => {
                  onSelect(row);
                  setSheetDraft({
                    aav_m: String(row.suggested_aav_m || row.qualifying_offer_aav_m || 1),
                    years: "4",
                  });
                }}
              >
                <span className="cl-rfa__name"><strong>{safeText(row.name)}</strong><em>{safeText(row.position)} · {safeText(row.rights_team_id)}</em></span>
                <span>{safeNum(row.overall) || "—"}</span>
                <span>{formatMoneyM(row.qualifying_offer_aav_m)}</span>
                <span>{safeText(row.compensation_preview?.label)}</span>
              </button>
            ))
          ) : (
            <p className="cl-empty">No other-club RFAs are open to offer sheets.</p>
          )}
        </div>
      </div>

      {selected?.offer_sheet_eligible && selected?.rights_team_id ? (
        <form
          className="cl-rfa__form"
          onSubmit={(e) => {
            e.preventDefault();
            onFileSheet(selected, sheetDraft);
          }}
        >
          <h3>Offer sheet · {safeText(selected.name)}</h3>
          <label>
            <span>Cap hit ($M)</span>
            <input
              type="number"
              step="0.025"
              min="0"
              value={sheetDraft.aav_m}
              onChange={(e) => setSheetDraft((d) => ({ ...d, aav_m: e.target.value }))}
            />
          </label>
          <label>
            <span>Years</span>
            <input
              type="number"
              min="1"
              max={Math.max(1, Number(data?.signing_bonus?.max_term_ufa || 6))}
              value={sheetDraft.years}
              onChange={(e) => setSheetDraft((d) => ({ ...d, years: e.target.value }))}
            />
          </label>
          <button type="submit" className="cl-btn cl-btn--primary" disabled={busy}>
            Submit offer sheet
          </button>
        </form>
      ) : null}
    </section>
  );
}

function ContractActionPanel({
  row,
  onClose,
  onAction,
  busy,
  onRosterMoved,
  onOpenNegotiate,
  summary = null,
}) {
  if (!row) {
    return (
      <aside className="cl-panel cl-panel--empty">
        <h3>Pick a contract</h3>
        <p>Select a player to re-sign, buy out, waive, or call up / send down.</p>
        {summary ? (
          <dl className="cl-facts">
            <div><dt>Expiring this year</dt><dd>{summary.expiring}</dd></div>
            <div><dt>Pending RFAs</dt><dd>{summary.rfa}</dd></div>
            <div><dt>With a clause</dt><dd>{summary.clause}</dd></div>
          </dl>
        ) : null}
      </aside>
    );
  }

  const ext = row.extension_estimate || {};
  const buy = row.buyout_estimate || {};
  const ovr = safeNum(row.overall ?? row.ovr);
  const pot = safeNum(row.potential ?? row.pot);
  const aav = safeNum(row.aav_m ?? row.aav);
  const valueScore = safeText(row.contract_value_score, "—");
  const player = buildHeadshotPlayer(row);
  const clause = clauseOf(row);
  const displayTags = safeArray(row.tags).filter(
    (tag) => !/casualty|cap casualty|trade pressure/i.test(String(tag)) && String(tag) !== clause,
  );
  const actions = buildContractPanelActions(row);
  const primary = actions.find((a) => isNegotiateAction(a.id)) || null;
  const secondary = actions.filter((a) => a !== primary);
  const ineligible = safeText(row.ineligible_reason, "");

  const facts = [
    ["Status", getStatusLabel(row)],
    ["Value", valueScore],
    clause ? ["Clause", clause] : null,
    ext.likelyAav ? ["Projected deal", `${formatMoneyM(ext.likelyAav)} × ${ext.likelyTerm} yrs`] : null,
    row.extension_aav_m
      ? ["Extension", `${formatMoneyM(row.extension_aav_m)}${row.extension_years_remaining ? ` × ${row.extension_years_remaining} yrs` : ""}`]
      : null,
    buy.totalCost != null ? ["Buyout cost", formatMoneyM(buy.totalCost)] : null,
  ].filter(Boolean);

  const runAction = (a) => {
    if (isNegotiateAction(a.id)) {
      onOpenNegotiate?.(row);
      return;
    }
    onAction(a.id, row);
  };

  return (
    <aside className="cl-panel" aria-label={`${safeText(row.name, "Player")} contract`}>
      <button type="button" className="cl-panel__close" onClick={onClose} aria-label="Clear selection">×</button>

      <header className="cl-panel__who">
        <PlayerHeadshot player={player} size="md" className="cl-panel__face" />
        <div>
          <h3>{safeText(row.name, "Unnamed Player")}</h3>
          <p>
            {safeText(row.position)} · Age {row.age || "—"}
            {row.sweater_number || row.jersey ? ` · #${row.sweater_number || row.jersey}` : ""}
          </p>
          <p className="cl-panel__ratings">
            <span className={`tone-${ovrPopTone(ovr)}`}>{ovr || "—"} <small>OVR</small></span>
            {pot ? <span className={pot >= 88 ? "tone-gold" : ""}>{pot} <small>POT</small></span> : null}
          </p>
        </div>
      </header>

      <div className="cl-panel__deal">
        <strong>{formatMoneyM(aav)}</strong>
        <span>{contractEndLabel(row)}</span>
      </div>

      <dl className="cl-facts">
        {facts.map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
        ))}
      </dl>

      {displayTags.length ? (
        <div className="cl-panel__tags">
          {displayTags.map((tag) => (
            <span key={tag} className={`cl-tag is-${valueKey(tag, [tag])}`}>{tag}</span>
          ))}
        </div>
      ) : null}

      <div className="cl-panel__actions">
        {primary ? (
          <button type="button" className="cl-btn cl-btn--primary" disabled={busy} onClick={() => runAction(primary)}>
            {primary.label}
          </button>
        ) : null}
        {secondary.map((a) => (
          <button
            key={a.id}
            type="button"
            className={`cl-btn${a.id === "buyout" || a.id === "release-rights" ? " cl-btn--danger" : ""}`}
            disabled={busy}
            onClick={() => runAction(a)}
          >
            {a.label}
          </button>
        ))}
        {!actions.length ? (
          <p className="cl-panel__note">{ineligible || "No contract actions for this player right now."}</p>
        ) : null}
      </div>

      <ContractRosterMoves row={row} onMoved={onRosterMoved} />
    </aside>
  );
}

function CapTab({ data, slotSummary }) {
  const snap = getCapSnap(data);
  const slots = slotSummary || buildRosterSlotSummary(null, snap, data?.team_id);
  const groups = [
    {
      title: "Dead money",
      rows: [
        ["Buyouts", formatMoneyM(snap.buyout_cap_hit_m)],
        ["Other dead cap", formatMoneyM(snap.other_dead_cap_m || 0)],
        ["Buried in minors", formatMoneyM(snap.buried_cap_hit_m)],
        ["Retained salary", formatMoneyM(snap.retained_salary_m)],
      ],
    },
    {
      title: "Roster limits",
      rows: [
        ["NHL contracts", slots.spc ? `${slots.spc.used} / ${slots.spc.limit}` : "—"],
        ["NHL roster", `${slots.nhl.used} / ${slots.nhl.limit}`],
        ["AHL roster", `${slots.ahl.used} / ${slots.ahl.limit}`],
        ["ECHL roster", `${slots.echl.used} / ${slots.echl.limit}`],
      ],
    },
    {
      title: "Next season",
      rows: [["Projected ceiling", formatMoneyM(snap.projected_next_year_upper_limit_m)]],
    },
  ];
  const warnings = safeArray(snap.warnings).slice(0, 5);

  return (
    <section className="cl-cap" aria-label="Cap breakdown">
      {groups.map((g) => (
        <div key={g.title} className="cl-cap__group">
          <h3>{g.title}</h3>
          <dl className="cl-facts">
            {g.rows.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
        </div>
      ))}
      <div className="cl-cap__group">
        <h3>Warnings</h3>
        {warnings.length ? (
          <div className="cl-panel__tags">
            {warnings.map((w) => <span key={w} className="cl-tag is-bad">{w}</span>)}
          </div>
        ) : (
          <p className="cl-panel__note">No cap warnings. The books are clean.</p>
        )}
      </div>
    </section>
  );
}

function OfficeTopBar({ team, snap, slotSummary, teamLogo, onBack }) {
  const space = safeNum(snap.usable_cap_space_m ?? team.cap_space);
  const hit = safeNum(snap.total_cap_hit_m ?? team.cap_hit);
  const limit = safeNum(snap.upper_limit_m, hit + Math.max(space, 0));
  const pct = limit > 0 ? Math.min(100, Math.max(0, (hit / limit) * 100)) : 0;
  const tone = space < 0 ? "over" : pct >= 97 ? "tight" : pct >= 90 ? "warm" : "ok";
  const slots = slotSummary || null;

  // The page's one hero number is cap space; hit and ceiling explain it.
  return (
    <header className="cl-top">
      <div className="cl-top__team">
        <button type="button" className="cl-back" onClick={onBack} aria-label="Back to hub">←</button>
        {teamLogo ? (
          <img className="cl-top__logo" src={teamLogo} alt="" />
        ) : (
          <span className="cl-top__logo cl-top__logo--text">{safeText(team.abbr || team.abbreviation || team.name, "TM").slice(0, 3)}</span>
        )}
        <div>
          <h1>Contracts</h1>
          <p>{team.name || "Franchise"}</p>
        </div>
      </div>

      <div className={`cl-meter is-${tone}`}>
        <div className="cl-meter__space">
          <strong>{formatMoneyM(Math.abs(space))}</strong>
          <span>{space < 0 ? "over the cap" : "cap space"}</span>
        </div>
        <div className="cl-meter__bar" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} aria-label="Cap used">
          <i style={{ width: `${pct}%` }} />
        </div>
        <div className="cl-meter__scale">
          <span>Hit <b>{formatMoneyM(hit)}</b></span>
          <span>Ceiling <b>{formatMoneyM(limit)}</b></span>
        </div>
      </div>

      {slots ? (
        <dl className="cl-top__slots">
          {slots.spc ? <div><dt>Contracts</dt><dd>{slots.spc.used}/{slots.spc.limit}</dd></div> : null}
          <div><dt>NHL</dt><dd>{slots.nhl.used}/{slots.nhl.limit}</dd></div>
          <div><dt>AHL</dt><dd>{slots.ahl.used}/{slots.ahl.limit}</dd></div>
          <div><dt>ECHL</dt><dd>{slots.echl.used}/{slots.echl.limit}</dd></div>
        </dl>
      ) : null}
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
    
    
    pendingPromiseNav,
    setPendingPromiseNav,
  } = useGameUI();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [sheetDraft, setSheetDraft] = useState({ aav_m: "2.000", years: "4" });
  const [negotiateRow, setNegotiateRow] = useState(null);

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

  const loadData = useCallback(async (opts = {}) => {
    const silent = Boolean(opts.silent);
    if (!silent) {
      setLoading(true);
    }
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
      if (!silent) {
        setLoading(false);
      }
    }
  }, [franchiseState?.stats_revision, franchiseState?.session_id]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Free agency lives on the wire screen; old deep links to the tab go there.
  useEffect(() => {
    if (tab === "freeAgents" && typeof setScreen === "function") setScreen(SCREENS.FREE_AGENCY);
  }, [tab, setScreen]);

  const snap = getCapSnap(data || {});
  const team = data?.team || franchiseState?.team || {};
  const seasonLabel = data?.season ? String(data.season) : "";

  useEffect(() => {
    if (!selected && !negotiateRow) return;
    if (!selected && negotiateRow) {
      setNegotiateRow(null);
    }
  }, [selected, negotiateRow]);
  const teamLogo = resolveFranchiseTeamLogo(team, team.name);
  const slotSummary = useMemo(
    () => buildRosterSlotSummary(franchiseState, snap, data?.team_id || team?.id),
    [franchiseState, snap, data?.team_id, team?.id],
  );

  const boardSummary = useMemo(() => {
    const list = safeArray(data?.contracts);
    return {
      expiring: list.filter(isExpiringRow).length,
      rfa: list.filter((r) => String(r.expiry_status || "").toUpperCase().includes("RFA")).length,
      clause: list.filter((r) => clauseOf(r)).length,
    };
  }, [data?.contracts]);

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

      if (isNegotiateAction(action)) {
        setBusy(false);
        return;
      }
      if (action === "qualify-rfa") {
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

      if (result?.ok) {
        setSelected(null);
      }
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
    <div className="game-screen cap-ledger-screen cap-office-screen cl-screen">
      <OfficeTopBar
        team={team}
        snap={snap}
        slotSummary={slotSummary}
        teamLogo={teamLogo}
        onBack={handleBack}
      />

      {pendingPromiseNav?.screen === "cap_ledger" ? (
        <p className="cl-banner is-promise">
          {pendingPromiseNav.label || "The contract talk is waiting on this desk."}
          <button type="button" onClick={() => setPendingPromiseNav?.(null)}>Dismiss</button>
        </p>
      ) : null}

      {note ? (
        <p className="cl-banner is-note" role="status">
          {note}
          <button type="button" onClick={() => setNote("")} aria-label="Dismiss">×</button>
        </p>
      ) : null}

      {error ? (
        <p className="cl-banner is-error" role="alert">
          {error}
          <button type="button" onClick={() => setError("")} aria-label="Dismiss">×</button>
        </p>
      ) : null}

      <nav className="cl-tabs" aria-label="Contract office sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? "is-active" : ""}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => selectTab(t.id)}
          >
            {t.label}
          </button>
        ))}
        {/* Free agency is its own screen (the wire); say so instead of posing as a tab. */}
        <button
          type="button"
          className="cl-tabs__link"
          onClick={() => typeof setScreen === "function" && setScreen(SCREENS.FREE_AGENCY)}
        >
          Free agency wire →
        </button>
      </nav>

      {loading && !negotiateRow ? (
        <p className="cl-empty">Loading contracts…</p>
      ) : null}

      {data && (!loading || negotiateRow) ? (
        <main className="cl-main">
          {tab === "ledger" ? (
            <div className={`cl-split${negotiateRow ? " is-negotiating" : ""}`}>
              <LedgerTab
                data={data}
                onSelect={(row) => {
                  setSelected(row);
                  setNegotiateRow(null);
                }}
                selectedId={selected?.player_id || selected?.id}
              />
              {negotiateRow ? (
                <section className="cl-nego" aria-label="Contract negotiation">
                  <CapContractNegotiation
                    row={negotiateRow}
                    capSnapshot={snap}
                    nextYearProjection={data?.nextYearProjection || {}}
                    signingBonusElig={data?.signing_bonus || {}}
                    seasonLabel={seasonLabel}
                    busy={busy}
                    onBusy={(v) => setBusy(Boolean(v))}
                    onBack={() => setNegotiateRow(null)}
                    onResult={(result, meta) => {
                      if (meta?.preview) {
                        if (!result?.ok && result?.reason) setError(result.reason);
                        return;
                      }
                      const st = String(result?.status || "").toLowerCase();
                      if (result?.office) setData(result.office);
                      else if (result?.re_sign || result?.contracts) {
                        setData((prev) =>
                          prev
                            ? {
                                ...prev,
                                re_sign: result.re_sign || result.contracts,
                                contracts: result.contracts || result.re_sign,
                              }
                            : prev,
                        );
                      } else if (meta?.simDay || st === "pending" || st === "countered" || st === "rejected") {
                        loadData({ silent: true });
                      } else if (result?.ok) {
                        loadData({ silent: true });
                      }
                      const spoken =
                        result?.player_response?.feedback ||
                        result?.evaluation?.agent_dialogue ||
                        result?.message ||
                        result?.reason ||
                        "";
                      if (st === "countered" || st === "rejected" || st === "pending") {
                        setError("");
                        setNote(
                          spoken ||
                            (st === "pending"
                              ? "Offer is on the table. Sim a day to hear back."
                              : st === "countered"
                                ? "He countered. The new number is on the desk."
                                : "He turned the offer down."),
                        );
                        if (typeof refreshFranchise === "function") refreshFranchise();
                        return;
                      }
                      if (!result?.ok && result?.reason) setError(result.reason);
                      else if (result?.ok && (meta?.signed || st === "accepted")) {
                        const who = negotiateRow?.name || "Player";
                        const hit = result?.final_cap_hit ?? result?.evaluation?.aav_m;
                        setError("");
                        setNote(
                          result?.message ||
                            `${who} signed${hit != null ? ` at $${Number(hit).toFixed(2)}M` : ""}.`,
                        );
                        setNegotiateRow(null);
                        setSelected(null);
                        if (typeof refreshFranchise === "function") refreshFranchise();
                      }
                    }}
                  />
                </section>
              ) : (
                <ContractActionPanel
                  row={selected}
                  onClose={() => setSelected(null)}
                  onAction={handleAction}
                  busy={busy}
                  onOpenNegotiate={setNegotiateRow}
                  onRosterMoved={() => loadData()}
                  summary={boardSummary}
                />
              )}
            </div>
          ) : null}

          {tab === "rfa" ? (
            <div className="cl-split">
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