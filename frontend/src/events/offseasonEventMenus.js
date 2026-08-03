import React from "react";
import CinematicEventShell from "./shared/CinematicEventShell";
import AwardsNight from "./awardsNight/AwardsNight";
import DraftLotteryNight from "./draftLottery/DraftLotteryNight";
import ProspectDevelopmentMenu from "./prospectDevelopment/ProspectDevelopmentMenu";
import RetirementsBoard from "./retirements/RetirementsBoard";
import CapReportBoard from "./salaryCap/CapReportBoard";
import PlayerHeadshot from "../components/PlayerHeadshot";
import TeamLogoBadge from "../components/ui/TeamLogoBadge";
import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import {
  getContractOffice,
  reSignContract,
  qualifyRfa,
  releaseRfaRights,
  evaluateContractOffer,
  prospectRightsDecision,
  signElcContract,
  previewElcOffer,
  submitElcOffer,
  fileArbitration,
  settleArbitration,
  matchOfferSheet,
  declineOfferSheet,
  advanceFreeAgencyDay,
  advanceContractNegotiationDay,
  signFreeAgent,
  getFreeAgentDetail,
  getFreeAgencyDesk,
} from "../services/franchiseService";
import {
  firstDefined,
  formatMoney,
  formatPick,
  getPlayerName,
  getPlayerOverall,
  getPlayerPosition,
  getTeamName,
  pickFranchiseData,
  safeArray,
} from "./shared/eventHelpers";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function Card({ prefix, rank, title, details, active }) {
  const p = prefix;
  return (
    <article className={`${p}-card ${active ? "is-active" : ""}`}>
      {rank != null ? <div className={`${p}-card-rank`}>{rank}</div> : null}
      <div className={`${p}-card-body`}>
        <strong>{title}</strong>
        <div className={`${p}-card-details`}>
          {safeArray(details).map((d, i) => (
            <span key={i}>{d}</span>
          ))}
        </div>
      </div>
    </article>
  );
}

/** Accept / deny deal popup for re-sign, prospect rights, and free agency. */
function DealOutcomeModal({ prefix, deal, onClose }) {
  if (!deal) return null;
  const tone = deal.tone === "accept" ? "is-accept" : deal.tone === "deny" ? "is-deny" : "is-pending";
  return (
    <div className={`${prefix}-deal-overlay`} role="dialog" aria-modal="true" aria-labelledby={`${prefix}-deal-title`}>
      <div className={`${prefix}-deal-modal ${tone}`}>
        <p className={`${prefix}-deal-kicker`}>{deal.kicker || "Contract desk"}</p>
        <h3 id={`${prefix}-deal-title`}>{deal.title}</h3>
        {deal.player ? <p className={`${prefix}-deal-player`}>{deal.player}</p> : null}
        <p className={`${prefix}-deal-body`}>{deal.body}</p>
        {deal.terms ? <p className={`${prefix}-deal-terms`}>{deal.terms}</p> : null}
        <button type="button" className={`${prefix}-cta-btn`} onClick={onClose}>
          {deal.cta || "Got it"}
        </button>
      </div>
    </div>
  );
}

function formatDealTerms(result) {
  const aav = result?.aav_m ?? result?.contract?.aav_m ?? result?.signed_aav_m;
  const years = result?.years ?? result?.contract?.years ?? result?.signed_years;
  if (aav == null || years == null) return null;
  return `${formatMoney(aav)} × ${years}y`;
}

function buildDealOutcome(result, { playerName, contextLabel }) {
  if (!result) return null;
  const status = String(result.status || result.decision?.status || "").toLowerCase();
  const reason = String(result.reason || result.user_message || "").toLowerCase();
  const name = playerName || "Player";
  const terms = formatDealTerms(result);
  const feedback =
    result?.player_response?.feedback ||
    result?.decision?.message ||
    result?.acceptance?.message ||
    result?.message ||
    result?.reason ||
    "";

  if (
    status === "accepted" ||
    result.signed === true ||
    (result.ok && ["signed", "elc_signed", "rights_retained"].includes(status))
  ) {
    return {
      tone: "accept",
      kicker: contextLabel || "Deal desk",
      title: "Deal accepted",
      player: name,
      body: feedback || `${name} accepted your offer.`,
      terms,
      cta: "Continue",
    };
  }

  if (
    status === "rejected" ||
    status === "declined" ||
    reason === "prospect_declined" ||
    result?.decision?.accepted === false
  ) {
    return {
      tone: "deny",
      kicker: contextLabel || "Deal desk",
      title: "Deal declined",
      player: name,
      body: feedback || `${name} turned down the offer.`,
      terms,
      cta: "Close",
    };
  }

  if (status === "countered") {
    const counter = result?.player_response || result?.evaluation?.counter_offer || {};
    const cAav = counter.counter_cap_hit ?? counter.aav_m;
    const cYears = counter.counter_term ?? counter.years;
    return {
      tone: "deny",
      kicker: contextLabel || "Deal desk",
      title: "Offer rejected — counter",
      player: name,
      body: feedback || `${name} rejected your terms and sent a counter.`,
      terms:
        cAav != null && cYears != null
          ? `Counter: ${formatMoney(cAav)} × ${cYears}y`
          : terms,
      cta: "Review counter",
    };
  }

  return null;
}

export function AwardsEventMenu(props) {
  return <AwardsNight {...props} />;
}


export function RetirementsEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const raw = pickFranchiseData(franchiseState, eventData, ["retirements", "offseason.retirements"]);

  return (
    <RetirementsBoard
      franchiseState={franchiseState}
      retirees={raw}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}

export function DraftLotteryEventMenu(props) {
  return <DraftLotteryNight {...props} />;
}

export function DraftEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "draft";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "draft",
    "draft_board",
    "prospects",
    "offseason.draft",
  ]);
  const prospects = safeArray(raw?.prospects || raw?.board || raw);
  const userPick = raw?.current_pick || raw?.user_picks?.[0];
  const uid = String(franchiseState?.user_team_id || "");

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="DRAFT FLOOR"
      titleVariant="floor"
      seasonLabel={seasonLabel(franchiseState)}
      title="DRAFT FLOOR"
      eyebrow="Prospects on the board"
      ctaLabel="Continue to Re-Sign Phase"
      tickerItems={["DRAFT FLOOR", "ON THE CLOCK", "NEXT GENERATION", "FRANCHISE PICKS"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        userPick ? (
          <>
            <p className={`${prefix}-hero-name`}>Your pick #{userPick.pick ?? "—"}</p>
            <div className={`${prefix}-meta`}>
              <span className={`${prefix}-chip`}>{userPick.team_name || "On the clock"}</span>
            </div>
          </>
        ) : (
          <p className={`${prefix}-empty`}>Draft board loading</p>
        )
      }
      railTitle="Prospect Board"
      railContent={
        prospects.length ? (
          prospects.slice(0, 24).map((p, i) => (
            <Card
              key={i}
              prefix={prefix}
              rank={formatPick(p.rank ?? p.overall ?? i + 1)}
              title={getPlayerName(p)}
              details={[
                getPlayerPosition(p),
                p.age != null ? `Age ${p.age}` : null,
                p.league || p.league_name,
                p.potential != null ? `Pot ${p.potential}` : null,
                p.scout_grade || p.grade,
                p.risk,
              ].filter(Boolean)}
            />
          ))
        ) : (
          <p className={`${prefix}-empty`}>No prospects on board</p>
        )
      }
    />
  );
}

export function ReSignEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "resign";
  const initial = pickFranchiseData(franchiseState, eventData, [
    "contracts",
    "re_sign",
    "offseason.expiring_contracts",
    "pending_free_agents",
  ]) || {};

  const [payload, setPayload] = React.useState(initial);
  const [loading, setLoading] = React.useState(!safeArray(initial?.contracts).length);
  const [filter, setFilter] = React.useState("all");
  const [sortKey, setSortKey] = React.useState("overall");
  const [sortDir, setSortDir] = React.useState("desc");
  const [selectedId, setSelectedId] = React.useState(null);
  const [panelMode, setPanelMode] = React.useState("detail"); // detail | negotiate
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [response, setResponse] = React.useState(null);
  const [offerAav, setOfferAav] = React.useState("");
  const [offerYears, setOfferYears] = React.useState("2");
  const [offerNtc, setOfferNtc] = React.useState(false);
  const [offerNmc, setOfferNmc] = React.useState(false);
  const [offerTwoWay, setOfferTwoWay] = React.useState(false);
  const [offerBonus, setOfferBonus] = React.useState("0");
  const [offerContractType, setOfferContractType] = React.useState("nhl_one_way");
  const [dayFlash, setDayFlash] = React.useState(null);
  const [dealPopup, setDealPopup] = React.useState(null);
  const rowRefs = React.useRef({});
  const tableWrapRef = React.useRef(null);

  const applyOfficeOrResign = React.useCallback((office, resign) => {
    if (resign && (resign.contracts || resign.expiring_contracts)) {
      setPayload((prev) => ({
        ...resign,
        // Prefer live office snapshot when both are present — resign blob alone
        // can lag if office was rebuilt after the signing.
        cap_snapshot:
          office?.cap_snapshot ||
          office?.team_cap ||
          resign.cap_snapshot ||
          prev?.cap_snapshot,
        contract_slots: office?.contract_slots || resign.contract_slots || prev?.contract_slots,
      }));
      return;
    }
    if (!office) return;
    const contracts = safeArray(office.contracts);
    const expiring = safeArray(office.expiring);
    const rfa = safeArray(office.rfa_rights);
    setPayload((prev) => ({
      ...prev,
      version: 3,
      contracts,
      expiring_contracts: expiring,
      rfa_rights: rfa.map((row) => {
        const sheets = safeArray(office.pending_offer_sheets);
        const hit = sheets.find((s) => String(s.player_id) === String(row.player_id));
        return hit
          ? {
              ...row,
              offer_sheet_pending: true,
              pending_offer_sheet: hit,
              offer_sheet_aav_m: hit.aav_m,
              offer_sheet_compensation: hit.compensation_label || hit.compensation_tier,
            }
          : row;
      }),
      pending_offer_sheets: office.pending_offer_sheets || [],
      cap_snapshot: office.cap_snapshot || office.team_cap || prev.cap_snapshot,
      contract_slots: office.contract_slots || prev.contract_slots,
      summary: office.summary || prev.summary,
      grouped: {
        pending_ufa: expiring.filter((r) => String(r.expiry_status || "").toUpperCase() === "UFA"),
        pending_rfa: [
          ...expiring.filter((r) => String(r.expiry_status || "").toUpperCase() === "RFA"),
          ...rfa,
        ],
        ...(prev.grouped || {}),
      },
      pending_decisions: [
        ...expiring,
        ...rfa,
      ],
      can_continue: prev.can_continue !== false,
      warning_reasons: prev.warning_reasons || [],
    }));
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    async function hydrate() {
      const hasRows = safeArray(initial?.contracts).length > 0;
      if (hasRows) {
        setLoading(false);
        return;
      }
      setLoading(true);
      try {
        const office = await getContractOffice();
        if (!cancelled) applyOfficeOrResign(office, null);
      } catch (e) {
        if (!cancelled) setError(String(e?.message || "Contract office unavailable"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    hydrate();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    if (panelMode !== "negotiate" || !selected?.player_id || busy) return undefined;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const result = await evaluateContractOffer({
          player_id: selected.player_id,
          aav_m: Number(offerAav) || 0,
          years: Math.max(1, parseInt(offerYears, 10) || 1),
          ntc: offerNtc,
          nmc: offerNmc,
          two_way: offerTwoWay,
          signing_bonus_m: Number(offerBonus) || 0,
          contract_category: offerContractType || "nhl_one_way",
          context: "re_sign",
        });
        if (!cancelled && result?.evaluation) setResponse(result);
      } catch {
        /* preview is best-effort */
      }
    }, 220);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panelMode, selectedId, offerAav, offerYears, offerNtc, offerNmc, offerBonus, offerContractType]);

  const contracts = safeArray(payload?.contracts);
  const expiring = safeArray(payload?.expiring_contracts || payload?.expiring);
  const grouped = payload?.grouped || {};
  const ufa = safeArray(grouped.pending_ufa);
  const rfa = safeArray(grouped.pending_rfa);
  const pending = safeArray(
    payload?.pending_decisions?.length
      ? payload.pending_decisions
      : [...ufa, ...rfa, ...expiring]
  );
  const summary = payload?.summary || {};
  const cap = payload?.cap_snapshot || franchiseState?.team?.cap || franchiseState?.salary_cap || {};
  const slots = payload?.contract_slots || {};
  const warnings = safeArray(payload?.warning_reasons);
  const blocking = safeArray(payload?.blocking_decisions || payload?.blocking_reasons);
  const canContinue = payload?.can_continue !== false && !safeArray(payload?.blocking_decisions).length;

  const selected =
    contracts.find((r) => String(r.player_id) === String(selectedId)) ||
    pending.find((r) => String(r.player_id) === String(selectedId)) ||
    null;

    React.useEffect(() => {
    if (!selected) return;
    const ask =
      selected.player_ask_aav_m ??
      selected.requested_cap_hit ??
      selected.qualifying_offer_aav_m ??
      selected.previous_aav_m ??
      selected.aav_m ??
      1;
    const years =
      selected.requested_term ??
      (selected.contract_status === "rfa_rights" ? 1 : selected.extension_estimate?.likelyTerm) ??
      2;
    setOfferAav(String(Number(ask).toFixed(3)));
    setOfferYears(String(years));
    setOfferNtc(Boolean(selected.clause_ask === "NTC" || selected.clause_ask === "NMC"));
    setOfferNmc(Boolean(selected.clause_ask === "NMC"));
    setOfferTwoWay(Boolean(selected.two_way));
    setOfferBonus("0");
    const legal = safeArray(selected.legal_contract_types).find((t) => t.enabled !== false);
    setOfferContractType(legal?.id || (selected.two_way ? "nhl_two_way" : "nhl_one_way"));
    setResponse(null);
    setError("");
  }, [selectedId]); // eslint-disable-line react-hooks/exhaustive-deps

  const filterCounts = React.useMemo(() => {
    const rows = contracts;
    return {
      all: rows.length,
      expiring: rows.filter((r) => intOr(r.years_remaining, 99) <= 1 || r.contract_status === "expiring").length,
      ufa: rows.filter((r) => String(r.expiry_status || r.expiry_type || "").toUpperCase() === "UFA").length,
      rfa: rows.filter(
        (r) =>
          String(r.expiry_status || r.expiry_type || "").toUpperCase() === "RFA" ||
          r.contract_status === "rfa_rights"
      ).length,
      signed: rows.filter((r) => intOr(r.years_remaining, 0) > 1 && r.contract_status !== "rfa_rights").length,
      extension: rows.filter((r) => r.extension_eligible === true).length,
      minors: rows.filter((r) => r.in_minors || String(r.role || "").toLowerCase().includes("minor")).length,
      unsigned: rows.filter((r) => r.contract_status === "rfa_rights" || r.qualifying_offer_eligible).length,
    };
  }, [contracts]);

  const visibleRows = React.useMemo(() => {
    let rows = [...contracts];
    const phaseStatus = (r) => String(r.phase_status || r.negotiation_status || "").toLowerCase();
    const isPhaseAccepted = (r) => phaseStatus(r) === "accepted";
    const isPhaseReleased = (r) => phaseStatus(r) === "released" || phaseStatus(r) === "lapsed";
    const isPhaseRejected = (r) => phaseStatus(r) === "rejected";
    const wasExpiring = (r) =>
      intOr(r.years_remaining, 99) <= 1 ||
      r.own_ufa === true ||
      r.contract_status === "own_ufa" ||
      r.contract_status === "expiring" ||
      r.contract_status === "rfa_rights" ||
      r.contract_status === "released" ||
      isPhaseAccepted(r) ||
      isPhaseReleased(r) ||
      isPhaseRejected(r) ||
      ["pending", "countered", "open"].includes(phaseStatus(r));

    if (filter === "expiring") {
      rows = rows.filter((r) => wasExpiring(r));
    } else if (filter === "ufa") {
      rows = rows.filter(
        (r) =>
          String(r.expiry_status || r.expiry_type || "").toUpperCase() === "UFA" ||
          (isPhaseAccepted(r) && String(r.expiry_status || "").toUpperCase() === "UFA") ||
          (isPhaseRejected(r) && String(r.expiry_status || "").toUpperCase() === "UFA")
      );
    } else if (filter === "rfa") {
      rows = rows.filter(
        (r) =>
          String(r.expiry_status || r.expiry_type || "").toUpperCase() === "RFA" ||
          r.contract_status === "rfa_rights" ||
          r.contract_status === "released" ||
          (isPhaseAccepted(r) && String(r.expiry_status || "").toUpperCase() === "RFA") ||
          isPhaseReleased(r)
      );
    } else if (filter === "signed") {
      rows = rows.filter(
        (r) =>
          isPhaseAccepted(r) ||
          (intOr(r.years_remaining, 0) > 1 && r.contract_status !== "rfa_rights" && r.contract_status !== "released")
      );
    } else if (filter === "extension") {
      // Current free-agency / re-sign class only — not players still owed another season.
      rows = rows.filter((r) => r.extension_eligible === true);
    } else if (filter === "minors") {
      rows = rows.filter((r) => r.in_minors || String(r.tags || []).includes("Minor"));
    } else if (filter === "unsigned") {
      rows = rows.filter((r) => r.contract_status === "rfa_rights" || r.qualifying_offer_eligible);
    }

    const dir = sortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      const av = sortValue(a, sortKey);
      const bv = sortValue(b, sortKey);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return String(a.name || "").localeCompare(String(b.name || ""));
    });
    return rows;
  }, [contracts, filter, sortKey, sortDir]);

  const selectPlayer = (id, { openNegotiate = false } = {}) => {
    setSelectedId(String(id));
    setPanelMode(openNegotiate ? "negotiate" : "detail");
    setResponse(null);
    setError("");
    requestAnimationFrame(() => {
      const node = rowRefs.current[String(id)];
      if (node?.scrollIntoView) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  };

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir(key === "name" ? "asc" : "desc");
    }
  };

  const capSpace = Number(
    cap.usable_cap_space_m ?? cap.cap_space ?? franchiseState?.team?.cap_space ?? 0
  );
  const offerAavNum = Number(offerAav) || 0;
  const offerYearsNum = Math.max(1, parseInt(offerYears, 10) || 1);
  const projectedSpace = Number.isFinite(capSpace) ? capSpace - offerAavNum : null;
  const askAav = Number(
    selected?.player_ask_aav_m ??
      selected?.requested_cap_hit ??
      selected?.qualifying_offer_aav_m ??
      selected?.previous_aav_m ??
      0
  );
  const askYears = Number(selected?.requested_term ?? selected?.extension_estimate?.likelyTerm ?? 0);

  const runAction = async (actionFn, { announceDeal = false } = {}) => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await actionFn();
      if (result?.re_sign || result?.contracts?.contracts) {
        applyOfficeOrResign(result.office, result.re_sign || result.contracts);
      } else if (result?.office) {
        applyOfficeOrResign(result.office, null);
      }
      if (result?.status === "accepted" || result?.ok) {
        setResponse(result);
        if (announceDeal) {
          const popup = buildDealOutcome(result, {
            playerName: getPlayerName(selected),
            contextLabel: "Player re-sign",
          });
          if (popup) setDealPopup(popup);
        }
        if (result?.status === "pending") {
          setPanelMode("negotiate");
        } else if (result?.ok && result?.status !== "evaluated" && result?.status !== "pending") {
          // Keep the resolved player selected so Accepted / Released stays on the desk.
          setPanelMode("detail");
          if (selected?.player_id) setSelectedId(String(selected.player_id));
        }
      } else if (result?.status === "countered" || result?.status === "rejected") {
        setResponse(result);
        if (announceDeal) {
          const popup = buildDealOutcome(result, {
            playerName: getPlayerName(selected),
            contextLabel: "Player re-sign",
          });
          if (popup) setDealPopup(popup);
        }
        setPanelMode("negotiate");
        if (selected?.player_id) setSelectedId(String(selected.player_id));
      } else if (!result?.ok) {
        setError(result?.reason || "Action failed");
        setResponse(result);
        if (announceDeal && (result?.status === "rejected" || result?.reason)) {
          const popup = buildDealOutcome(
            { ...result, status: result.status || "rejected" },
            {
              playerName: getPlayerName(selected),
              contextLabel: "Player re-sign",
            }
          );
          if (popup) setDealPopup(popup);
        }
      }
      return result;
    } catch (e) {
      setError(String(e?.message || "Action failed"));
      return null;
    } finally {
      setBusy(false);
    }
  };

  const submitOffer = () => {
    if (!selected?.player_id) return;
    const category = offerContractType || "nhl_one_way";
    const isNhl = String(category).startsWith("nhl");
    return runAction(
      () =>
        reSignContract({
          player_id: selected.player_id,
          aav_m: offerAavNum,
          years: offerYearsNum,
          ntc: offerNtc,
          nmc: offerNmc,
          two_way: category === "nhl_two_way" || offerTwoWay,
          signing_bonus_m: Number(offerBonus) || 0,
          contract_category: category,
          contract_type: isNhl ? undefined : category,
          context: "re_sign",
        }),
      { announceDeal: true }
    );
  };

  const previewOffer = () => {
    if (!selected?.player_id) return;
    const category = offerContractType || "nhl_one_way";
    return runAction(() =>
      evaluateContractOffer({
        player_id: selected.player_id,
        aav_m: offerAavNum,
        years: offerYearsNum,
        ntc: offerNtc,
        nmc: offerNmc,
        two_way: category === "nhl_two_way" || offerTwoWay,
        signing_bonus_m: Number(offerBonus) || 0,
        contract_category: category,
        context: "re_sign",
      })
    );
  };

  const acceptCounter = () => {
    const counter = response?.player_response || response?.evaluation?.counter_offer || {};
    const aav = counter.counter_cap_hit ?? counter.aav_m;
    const years = counter.counter_term ?? counter.years;
    if (aav == null || years == null || !selected?.player_id) return;
    setOfferAav(String(aav));
    setOfferYears(String(years));
    if (counter.counter_ntc != null || counter.ntc != null) {
      setOfferNtc(Boolean(counter.counter_ntc ?? counter.ntc));
    }
    if (counter.counter_nmc != null || counter.nmc != null) {
      setOfferNmc(Boolean(counter.counter_nmc ?? counter.nmc));
    }
    if (counter.counter_signing_bonus_m != null || counter.signing_bonus_m != null) {
      setOfferBonus(String(counter.counter_signing_bonus_m ?? counter.signing_bonus_m ?? 0));
    }
    return runAction(
      () =>
        reSignContract({
          player_id: selected.player_id,
          aav_m: Number(aav),
          years: Number(years),
          ntc: Boolean(counter.counter_ntc ?? counter.ntc ?? offerNtc),
          nmc: Boolean(counter.counter_nmc ?? counter.nmc ?? offerNmc),
          signing_bonus_m: Number(
            counter.counter_signing_bonus_m ?? counter.signing_bonus_m ?? offerBonus ?? 0
          ),
          two_way: offerTwoWay,
          context: "re_sign",
        }),
      { announceDeal: true }
    );
  };

  const simNegotiationDay = async () => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await advanceContractNegotiationDay(1);
      if (result?.re_sign || result?.contracts) {
        applyOfficeOrResign(null, result.re_sign || result.contracts);
      }
      const signed = safeArray(result?.signed);
      setDayFlash(
        signed.length
          ? `Day advanced — ${signed.map((s) => s.name || s.player_id).join(", ")} signed`
          : `Day ${result?.own_fa_window?.day ?? "—"} / ${result?.own_fa_window?.days_total ?? 6} — no signatures yet`
      );
      setResponse(result);
    } catch (e) {
      setError(String(e?.message || "Could not advance day"));
    } finally {
      setBusy(false);
    }
  };

  const continueDisabled = busy || !canContinue;
  const continueTitle = continueDisabled
    ? safeArray(payload?.blocking_reasons)[0] ||
      (blocking[0]?.message || blocking[0]) ||
      "Resolve required RFA decisions before continuing"
    : undefined;

  const selectedOvr = selected?.overall ?? selected?.ovr ?? getPlayerOverall(selected);
  const selectedInterest = resignInterestLabel(selected);
  const selectedArchetype = resignArchetype(selected);
  const offerDiff = askAav ? offerAavNum - askAav : 0;
  const offerDiffLabel =
    !askAav
      ? "No ask on file"
      : Math.abs(offerDiff) < 0.05
        ? "Perfect Match"
        : offerDiff > 0
          ? `+${formatMoney(offerDiff)} above ask`
          : `${formatMoney(offerDiff)} below ask`;
  const sliderMax = Math.max(
    12,
    askAav * 1.4 || 0,
    offerAavNum || 0,
    Number(selected?.aav_m || 0) * 1.8 || 0
  );
  const sliderMin = 0.775;
  const agentBits = resignAgentBits(selected, askAav, askYears);
  const decisionCount = summary.pendingDecisions ?? pending.length;
  const ufaCount = summary.ufaCount ?? filterCounts.ufa;
  const rfaCount = summary.rfaCount ?? filterCounts.rfa;
  const ownWindow = payload?.own_fa_window || {};
  const bonusElig = payload?.signing_bonus || {};
  const bonusAllowed = Boolean(bonusElig.eligible);
  const bonusMaxPct = Number(bonusElig.max_bonus_pct || 0);
  const offerBonusNum = Number(offerBonus) || 0;
  const negoInterest =
    Number(
      response?.evaluation?.interest ??
        response?.player_response?.interest ??
        selected?.pending_offer?.interest ??
        0
    ) || 0;
  const negoFeedback =
    response?.player_response?.feedback ||
    response?.evaluation?.reason ||
    (selected?.pending_offer
      ? `Offer pending — ${selected.pending_offer.days_remaining ?? "?"} day(s) left`
      : null);
  const slotUsed =
    slots.used != null
      ? slots.used
      : slots.limit != null && slots.open != null
        ? slots.limit - slots.open
        : null;

  return (
    <>
    <DealOutcomeModal prefix={prefix} deal={dealPopup} onClose={() => setDealPopup(null)} />
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="RE-SIGNING DESK"
      phaseStyle="text"
      titleVariant="ledger"
      rootClassName={`${prefix}-root--ledger`}
      seasonLabel={seasonLabel(franchiseState)}
      title="Negotiation Desk"
      hideTitle
      hideEyebrow
      ctaLabel="Open Free Agency"
      hideTicker
      persistRevealKey="resign-revealed"
      footerAlign="split"
      onContinue={continueDisabled ? undefined : onContinue}
      onBack={onBack}
      railTitle="Queue"
      railHint={null}
      heroContent={
        <div className={`${prefix}-workspace`}>
          <div className={`${prefix}-office-bar`} aria-label="Contract office summary">
            <strong className={`${prefix}-office-count`}>
              {decisionCount} <span>decisions</span>
            </strong>
            <span className={`${prefix}-office-meta`}>
              {ufaCount} UFA · {rfaCount} RFA
            </span>
            <span className={`${prefix}-office-cap`}>{formatMoney(capSpace)} available</span>
            <span className={`${prefix}-office-window`}>
              Exclusive {ownWindow.day ?? 0}/{ownWindow.days_total ?? 6}
              {ownWindow.days_remaining != null ? ` · ${ownWindow.days_remaining}d left` : ""}
            </span>
            <button
              type="button"
              className={`${prefix}-sim-day-btn`}
              disabled={busy || Boolean(ownWindow.complete)}
              onClick={simNegotiationDay}
            >
              Sim Day
            </button>
            {slotUsed != null && slots.limit != null ? (
              <span className={`${prefix}-office-slots`}>
                {slotUsed}/{slots.limit} contracts
              </span>
            ) : null}
            {dayFlash ? <em className={`${prefix}-office-flash`}>{dayFlash}</em> : null}
            {warnings[0] ? <em className={`${prefix}-office-warn`}>{warnings[0]}</em> : null}
          </div>
          <div className={`${prefix}-filter-row`}>
            {[
              ["all", "All"],
              ["expiring", "Expiring"],
              ["ufa", "UFAs"],
              ["rfa", "RFAs"],
              ["extension", "Extensions"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={`${prefix}-filter-chip${filter === id ? " is-active" : ""}`}
                onClick={() => setFilter(id)}
              >
                {label}
                <em>{filterCounts[id] ?? ""}</em>
              </button>
            ))}
          </div>

          <div className={`${prefix}-table-layout`}>
            <div className={`${prefix}-table-wrap`} ref={tableWrapRef}>
              {loading ? (
                <p className={`${prefix}-empty`}>Loading organization contracts…</p>
              ) : visibleRows.length === 0 ? (
                <p className={`${prefix}-empty`}>No players match this filter.</p>
              ) : (
                <table className={`${prefix}-table`}>
                  <thead>
                    <tr>
                      <th onClick={() => toggleSort("name")}>Player</th>
                      <th>Pos</th>
                      <th onClick={() => toggleSort("age")}>Age</th>
                      <th onClick={() => toggleSort("overall")}>OVR</th>
                      <th onClick={() => toggleSort("status")}>Status</th>
                      <th onClick={() => toggleSort("cap")}>Cap</th>
                      <th onClick={() => toggleSort("expiry")}>Expiry</th>
                      <th onClick={() => toggleSort("interest")}>Interest</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row) => {
                      const id = String(row.player_id || "");
                      const active = id === String(selectedId);
                      const headshotPlayer = ensurePlayerHeadshotFields({
                        id,
                        player_id: id,
                        name: row.name,
                        position: row.position,
                        age: row.age,
                      });
                      const phase = String(row.phase_status || row.negotiation_status || "").toLowerCase();
                      const status =
                        phase === "accepted"
                          ? "Accepted"
                          : phase === "rejected"
                            ? "Rejected"
                            : phase === "countered"
                              ? "Countered"
                              : phase === "pending"
                                ? "Pending"
                                : phase === "released"
                                  ? "Released"
                                  : phase === "lapsed"
                                    ? "Lapsed"
                                    : row.contract_status === "rfa_rights"
                                      ? "RFA Rights"
                                      : row.expiry_status || row.expiry_type || row.contract_status || "—";
                      const statusTone =
                        phase === "accepted"
                          ? "accepted"
                          : phase === "rejected"
                            ? "rejected"
                            : phase === "released"
                              ? "released"
                              : phase === "lapsed"
                                ? "lapsed"
                                : phase === "countered"
                                  ? "countered"
                                  : phase === "pending"
                                    ? "pending"
                                    : "";
                      const terminal = Boolean(row.phase_terminal) || ["accepted", "released", "lapsed"].includes(phase);
                      const primaryAction = terminal
                        ? null
                        : safeArray(row.available_actions).find((a) =>
                            ["negotiate_extension", "qualify_rfa"].includes(a.id)
                          );
                      const ovr = row.overall ?? row.ovr ?? getPlayerOverall(row);
                      const interest = resignInterestLabel(row);
                      return (
                        <tr
                          key={id}
                          ref={(node) => {
                            if (id) rowRefs.current[id] = node;
                          }}
                          className={[
                            active ? "is-selected" : "",
                            primaryAction ? "is-actionable" : "",
                          ]
                            .filter(Boolean)
                            .join(" ")}
                          onClick={() => selectPlayer(id)}
                        >
                          <td>
                            <div className={`${prefix}-player-cell`}>
                              <span className={`${prefix}-shot-frame`}>
                                <PlayerHeadshot player={headshotPlayer} size="sm" />
                              </span>
                              <strong>{row.name || getPlayerName(row)}</strong>
                            </div>
                          </td>
                          <td>{row.position || getPlayerPosition(row)}</td>
                          <td>{row.age ?? "—"}</td>
                          <td>
                            <span className={`${prefix}-ovr tone-${resignOvrTone(ovr)}`}>
                              {ovr ?? "—"}
                            </span>
                          </td>
                          <td>
                            <span className={`${prefix}-status-chip${statusTone ? ` tone-${statusTone}` : ""}`}>
                              {status}
                            </span>
                          </td>
                          <td>{formatMoney(row.aav_m ?? row.cap_hit_m ?? row.current_cap_hit)}</td>
                          <td>
                            {[
                              row.expiry_year,
                              row.years_remaining != null ? `${row.years_remaining}y` : null,
                            ]
                              .filter(Boolean)
                              .join(" · ") || "—"}
                          </td>
                          <td>
                            <span
                              className={`${prefix}-interest-pill tone-${resignInterestTone(interest)}`}
                            >
                              {interest}
                            </span>
                          </td>
                          <td>
                            {primaryAction ? (
                              <button
                                type="button"
                                className={`${prefix}-row-action`}
                                disabled={busy}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  selectPlayer(id, { openNegotiate: true });
                                }}
                              >
                                {primaryAction.id === "qualify_rfa" ? "Qualify" : "Negotiate"}
                              </button>
                            ) : (
                              <span className={`${prefix}-muted`}>—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>

            <aside className={`${prefix}-detail-panel is-focal`}>
              {!selected ? (
                <p className={`${prefix}-empty`}>Select a player to open the desk.</p>
              ) : (
                <>
                  <div className={`${prefix}-identity`}>
                    <div className={`${prefix}-identity-shot`}>
                      <PlayerHeadshot
                        player={ensurePlayerHeadshotFields({
                          id: selected.player_id,
                          player_id: selected.player_id,
                          name: selected.name,
                          position: selected.position,
                          age: selected.age,
                        })}
                        size="lg"
                      />
                    </div>
                    <div className={`${prefix}-identity-copy`}>
                      <span className={`${prefix}-ovr-badge tone-${resignOvrTone(selectedOvr)}`}>
                        {selectedOvr ?? "—"} <em>OVR</em>
                      </span>
                      <h3>{String(selected.name || getPlayerName(selected) || "").toUpperCase()}</h3>
                      <p>
                        {[
                          selected.position || getPlayerPosition(selected),
                          selected.age != null ? `Age ${selected.age}` : null,
                          selectedArchetype,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                    </div>
                  </div>

                  <div className={`${prefix}-detail-tabs`}>
                    <button
                      type="button"
                      className={panelMode === "detail" ? "is-active" : ""}
                      onClick={() => setPanelMode("detail")}
                    >
                      Contract
                    </button>
                    <button
                      type="button"
                      className={panelMode === "negotiate" ? "is-active" : ""}
                      onClick={() => setPanelMode("negotiate")}
                      disabled={
                        !safeArray(selected.available_actions).some((a) =>
                          ["negotiate_extension", "qualify_rfa"].includes(a.id)
                        )
                      }
                    >
                      Negotiate
                    </button>
                  </div>

                  {panelMode === "detail" ? (
                    <div className={`${prefix}-detail-body`}>
                      <div className={`${prefix}-dossier-grid`}>
                        <div>
                          <span>Cap hit</span>
                          <strong>
                            {formatMoney(
                              selected.aav_m ??
                                selected.previous_aav_m ??
                                selected.qualifying_offer_aav_m ??
                                selected.current_cap_hit
                            )}
                          </strong>
                        </div>
                        <div>
                          <span>Term left</span>
                          <strong>
                            {selected.contract_status === "rfa_rights"
                              ? "Rights"
                              : selected.years_remaining ?? "—"}
                          </strong>
                        </div>
                        <div>
                          <span>Expiry</span>
                          <strong>
                            {[selected.expiry_year, selected.expiry_status || selected.expiry_type]
                              .filter(Boolean)
                              .join(" · ") || "—"}
                          </strong>
                        </div>
                        <div>
                          <span>Interest</span>
                          <strong>
                            <span
                              className={`${prefix}-interest-pill tone-${resignInterestTone(selectedInterest)}`}
                            >
                              {selectedInterest}
                            </span>
                          </strong>
                        </div>
                        <div>
                          <span>Ask</span>
                          <strong>
                            {selected.player_ask_aav_m != null ||
                            selected.qualifying_offer_aav_m != null ||
                            selected.previous_aav_m != null
                              ? `${formatMoney(
                                  selected.player_ask_aav_m ??
                                    selected.qualifying_offer_aav_m ??
                                    selected.previous_aav_m
                                )} / ${selected.requested_term || (selected.can_qualify ? "1" : "—")}y`
                              : "—"}
                          </strong>
                        </div>
                        <div>
                          <span>Clauses</span>
                          <strong>{selected.clause_label || selected.clauseLabel || "None"}</strong>
                        </div>
                      </div>
                      {selected.extension_estimate?.likelyAav != null ? (
                        <p className={`${prefix}-context`}>
                          Market comp {formatMoney(selected.extension_estimate.likelyAav)} ×{" "}
                          {selected.extension_estimate.likelyTerm || "—"}y
                          {selected.extension_estimate.risk
                            ? ` · ${selected.extension_estimate.risk}`
                            : ""}
                        </p>
                      ) : null}
                      {agentBits[0] ? <p className={`${prefix}-context`}>{agentBits[0]}</p> : null}
                      <div className={`${prefix}-decision-list`}>
                        {safeArray(selected.available_actions)
                          .filter((a) => a.id !== "view_dossier")
                          .map((a) => (
                            <button
                              key={a.id}
                              type="button"
                              className={`${prefix}-decision-btn${
                                a.id === "negotiate_extension" || a.id === "qualify_rfa"
                                  ? " is-recommended"
                                  : ""
                              }`}
                              disabled={busy || a.enabled === false}
                              onClick={() => {
                                if (a.id === "negotiate_extension") setPanelMode("negotiate");
                                else if (a.id === "qualify_rfa") {
                                  runAction(() => qualifyRfa({ player_id: selected.player_id }));
                                } else if (a.id === "walk_away") {
                                  runAction(() => releaseRfaRights({ player_id: selected.player_id }));
                                }
                              }}
                            >
                              <span>{a.label}</span>
                            </button>
                          ))}
                      </div>
                      {selected.ineligible_reason ? (
                        <p className={`${prefix}-context`}>{selected.ineligible_reason}</p>
                      ) : null}
                    </div>
                  ) : (
                    <div className={`${prefix}-detail-body ${prefix}-nego-body`}>
                      <div className={`${prefix}-nego-scroll`}>
                        <div className={`${prefix}-cap-banner`}>
                          <div>
                            <span>Cap space</span>
                            <strong className="is-green">{formatMoney(capSpace)}</strong>
                          </div>
                          <div>
                            <span>After offer</span>
                            <strong
                              className={
                                projectedSpace != null && projectedSpace < 0 ? "is-red" : "is-green"
                              }
                            >
                              {projectedSpace != null ? formatMoney(projectedSpace) : "—"}
                            </strong>
                          </div>
                        </div>

                        <div className={`${prefix}-offer-compare`}>
                          <div>
                            <span>Ask</span>
                            <strong>
                              {askAav ? formatMoney(askAav) : "—"}
                              <em>{askYears ? `${askYears}y` : ""}</em>
                            </strong>
                          </div>
                          <div>
                            <span>Your offer</span>
                            <strong>
                              {formatMoney(offerAavNum)}
                              <em>{offerYearsNum}y</em>
                            </strong>
                          </div>
                        </div>
                        <p
                          className={`${prefix}-offer-diff-line ${
                            Math.abs(offerDiff) < 0.05
                              ? "is-green"
                              : offerDiff < 0
                                ? "is-warn"
                                : "is-cyan"
                          }`}
                        >
                          {offerDiffLabel}
                        </p>

                        <div className={`${prefix}-nego-meter`} aria-label="Negotiation interest">
                          <div className={`${prefix}-nego-meter-head`}>
                            <span>Deal interest</span>
                            <strong>{negoInterest ? `${Math.round(negoInterest)}` : "—"}</strong>
                          </div>
                          <div className={`${prefix}-nego-meter-track`}>
                            <span
                              className={`${prefix}-nego-meter-fill tone-${
                                negoInterest >= 88
                                  ? "instant"
                                  : negoInterest >= 62
                                    ? "good"
                                    : negoInterest >= 40
                                      ? "mid"
                                      : "bad"
                              }`}
                              style={{ width: `${Math.max(4, Math.min(100, negoInterest || 4))}%` }}
                            />
                            <i className={`${prefix}-nego-meter-mark`} style={{ left: "62%" }} title="Accept" />
                            <i
                              className={`${prefix}-nego-meter-mark is-instant`}
                              style={{ left: "88%" }}
                              title="Instant"
                            />
                          </div>
                          <p className={`${prefix}-nego-meter-note`}>
                            {negoFeedback ||
                              `${resignInterestLabel(selected)} stay interest · Prefer ${
                                selected.clause_ask || "no clause"
                              }`}
                          </p>
                        </div>

                        {(selected.can_qualify ||
                          selected.arbitration_eligible ||
                          selected.can_file_arbitration ||
                          selected.arbitration_filed ||
                          selected.offer_sheet_pending ||
                          selected.pending_offer_sheet) && (
                          <div className={`${prefix}-special-actions`}>
                            {selected.can_qualify ? (
                              <button
                                type="button"
                                className={`${prefix}-decision-btn is-recommended`}
                                disabled={busy}
                                onClick={() =>
                                  runAction(() => qualifyRfa({ player_id: selected.player_id }))
                                }
                              >
                                <span>
                                  Qualifying offer
                                  <span className={`${prefix}-decision-meta`}>
                                    {selected.qualifying_offer_aav_m != null
                                      ? formatMoney(selected.qualifying_offer_aav_m)
                                      : "QO"}
                                  </span>
                                </span>
                              </button>
                            ) : null}
                            {selected.arbitration_eligible || selected.can_file_arbitration ? (
                              <button
                                type="button"
                                className={`${prefix}-decision-btn`}
                                disabled={busy}
                                onClick={() =>
                                  runAction(() =>
                                    fileArbitration({
                                      player_id: selected.player_id,
                                      player_ask_m:
                                        Number(
                                          selected.player_ask_aav_m ||
                                            selected.requested_cap_hit ||
                                            offerAav
                                        ) || 0,
                                    })
                                  )
                                }
                              >
                                <span>File arbitration</span>
                              </button>
                            ) : null}
                            {selected.arbitration_filed ? (
                              <button
                                type="button"
                                className={`${prefix}-decision-btn is-recommended`}
                                disabled={busy}
                                onClick={() =>
                                  runAction(() =>
                                    settleArbitration({ player_id: selected.player_id })
                                  )
                                }
                              >
                                <span>Settle arbitration</span>
                              </button>
                            ) : null}
                            {selected.offer_sheet_pending || selected.pending_offer_sheet ? (
                              <>
                                <button
                                  type="button"
                                  className={`${prefix}-decision-btn is-recommended`}
                                  disabled={busy}
                                  onClick={() =>
                                    runAction(() =>
                                      matchOfferSheet({ player_id: selected.player_id })
                                    )
                                  }
                                >
                                  <span>Match offer sheet</span>
                                </button>
                                <button
                                  type="button"
                                  className={`${prefix}-decision-btn`}
                                  disabled={busy}
                                  onClick={() =>
                                    runAction(() =>
                                      declineOfferSheet({ player_id: selected.player_id })
                                    )
                                  }
                                >
                                  <span>Decline sheet</span>
                                </button>
                              </>
                            ) : null}
                          </div>
                        )}

                        <div className={`${prefix}-deal-controls`}>
                          {safeArray(selected.legal_contract_types).length ? (
                            <label className={`${prefix}-field ${prefix}-select-field`}>
                              Type
                              <div className={`${prefix}-select-wrap`}>
                                <select
                                  value={offerContractType}
                                  disabled={busy}
                                  onChange={(e) => setOfferContractType(e.target.value)}
                                >
                                  {safeArray(selected.legal_contract_types)
                                    .filter((t) => t.enabled !== false)
                                    .map((t) => (
                                      <option key={t.id} value={t.id}>
                                        {t.label}
                                      </option>
                                    ))}
                                </select>
                              </div>
                            </label>
                          ) : null}

                          <div className={`${prefix}-slider-block`}>
                            <div className={`${prefix}-slider-head`}>
                              <span>Annual salary</span>
                              <strong>{formatMoney(offerAavNum)}</strong>
                            </div>
                            <input
                              type="range"
                              className={`${prefix}-salary-slider`}
                              min={sliderMin}
                              max={sliderMax}
                              step="0.025"
                              value={Math.min(
                                sliderMax,
                                Math.max(sliderMin, offerAavNum || sliderMin)
                              )}
                              disabled={busy}
                              onChange={(e) => setOfferAav(Number(e.target.value).toFixed(3))}
                            />
                          </div>

                          <div className={`${prefix}-term-block`}>
                            <span className={`${prefix}-mini-label`}>Term</span>
                            <div
                              className={`${prefix}-term-seg`}
                              role="group"
                              aria-label="Contract term"
                            >
                              {[1, 2, 3, 4, 5, 6, 7, 8].map((y) => (
                                <button
                                  key={y}
                                  type="button"
                                  className={offerYearsNum === y ? "is-active" : ""}
                                  disabled={busy}
                                  onClick={() => setOfferYears(String(y))}
                                >
                                  {y}
                                </button>
                              ))}
                            </div>
                          </div>

                          <div className={`${prefix}-check-row is-clauses`}>
                            <label className={offerNtc ? "is-on" : ""}>
                              <input
                                type="checkbox"
                                checked={offerNtc}
                                disabled={busy || offerNmc}
                                onChange={(e) => setOfferNtc(e.target.checked)}
                              />
                              NTC
                              {selected.clause_ask === "NTC" ? <em>asked</em> : null}
                            </label>
                            <label className={offerNmc ? "is-on" : ""}>
                              <input
                                type="checkbox"
                                checked={offerNmc}
                                disabled={busy}
                                onChange={(e) => {
                                  setOfferNmc(e.target.checked);
                                  if (e.target.checked) setOfferNtc(true);
                                }}
                              />
                              NMC
                              {selected.clause_ask === "NMC" ? <em>asked</em> : null}
                            </label>
                            <label>
                              <input
                                type="checkbox"
                                checked={offerTwoWay}
                                disabled={busy}
                                onChange={(e) => setOfferTwoWay(e.target.checked)}
                              />
                              Two-way
                            </label>
                          </div>

                          <div className={`${prefix}-slider-block`}>
                            <div className={`${prefix}-slider-head`}>
                              <span>Signing bonus</span>
                              <strong>
                                {bonusAllowed ? formatMoney(offerBonusNum) : "Locked"}
                              </strong>
                            </div>
                            {bonusAllowed ? (
                              <input
                                type="range"
                                className={`${prefix}-salary-slider`}
                                min={0}
                                max={Math.max(
                                  0.25,
                                  offerAavNum * offerYearsNum * (bonusMaxPct || 0.08)
                                )}
                                step="0.025"
                                value={Math.min(
                                  offerAavNum * offerYearsNum * (bonusMaxPct || 0.08),
                                  Math.max(0, offerBonusNum)
                                )}
                                disabled={busy}
                                onChange={(e) => setOfferBonus(Number(e.target.value).toFixed(3))}
                              />
                            ) : (
                              <p className={`${prefix}-context`}>
                                {bonusElig.label ||
                                  "Signing bonuses require NHL revenue ≥ $130M"}
                              </p>
                            )}
                          </div>
                        </div>

                        {response?.player_response?.feedback || response?.evaluation ? (
                          <div className={`${prefix}-rights-callout`}>
                            <span>Response</span>
                            <strong>
                              {response.status || response.player_response?.status || "—"}
                              {response.player_response?.feedback
                                ? ` · ${response.player_response.feedback}`
                                : response.reason
                                  ? ` · ${response.reason}`
                                  : ""}
                            </strong>
                          </div>
                        ) : null}
                        {error ? <p className={`${prefix}-warn`}>{error}</p> : null}
                      </div>

                      <div className={`${prefix}-negotiate-actions`}>
                        <button
                          type="button"
                          className={`${prefix}-ghost-btn ${prefix}-preview-btn`}
                          disabled={busy}
                          onClick={previewOffer}
                        >
                          Preview
                        </button>
                        <button
                          type="button"
                          className={`${prefix}-cta-btn ${prefix}-submit-btn`}
                          disabled={busy}
                          onClick={submitOffer}
                        >
                          {busy ? "Submitting…" : "Submit Offer"}
                        </button>
                        {response?.status === "countered" ? (
                          <button
                            type="button"
                            className={`${prefix}-decision-btn is-recommended`}
                            disabled={busy}
                            onClick={acceptCounter}
                          >
                            <span>
                              Accept counter
                              <span className={`${prefix}-decision-meta`}>
                                {formatMoney(
                                  response.player_response?.counter_cap_hit ??
                                    response.evaluation?.counter_offer?.aav_m
                                )}{" "}
                                ·{" "}
                                {response.player_response?.counter_term ??
                                  response.evaluation?.counter_offer?.years}
                                y
                              </span>
                            </span>
                          </button>
                        ) : null}
                      </div>
                    </div>
                  )}
                </>
              )}
            </aside>
          </div>
          {continueDisabled ? (
            <p className={`${prefix}-context ${prefix}-continue-note`} title={continueTitle}>
              {continueTitle || "Resolve required decisions before Free Agency."}
            </p>
          ) : null}
        </div>
      }
      railContent={
        pending.length ? (
          pending.slice(0, 24).map((p, i) => {
            const id = String(p.player_id || i);
            const active = id === String(selectedId);
            const ovr = p.overall ?? p.ovr ?? getPlayerOverall(p);
            const interest = resignInterestLabel(p);
            const status =
              p.contract_status === "rfa_rights"
                ? "RFA"
                : p.expiry_status || p.expiry_type || p.rights_status || "—";
            return (
              <button
                key={id}
                type="button"
                className={`${prefix}-rail-btn`}
                onClick={() => selectPlayer(id, { openNegotiate: true })}
              >
                <article className={`${prefix}-queue-card${active ? " is-active" : ""}`}>
                  <span className={`${prefix}-ovr tone-${resignOvrTone(ovr)}`}>{ovr ?? "—"}</span>
                  <div className={`${prefix}-queue-body`}>
                    <strong>{p.name || getPlayerName(p)}</strong>
                    <span className={`${prefix}-queue-status`}>
                      {status} · {interest}
                    </span>
                    <em>{formatMoney(p.aav_m ?? p.player_ask_aav_m ?? p.current_cap_hit)}</em>
                  </div>
                </article>
              </button>
            );
          })
        ) : (
          <p className={`${prefix}-empty`}>No pending decisions</p>
        )
      }
    />
    </>
  );
}


function intOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function resignOvrTone(ovr) {
  const n = Number(ovr);
  if (n >= 90) return "gold";
  if (n >= 82) return "blue";
  return "grey";
}

function resignInterestTone(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("high")) return "high";
  if (s.includes("low")) return "low";
  return "med";
}

function resignInterestLabel(row) {
  if (!row) return "Medium";
  if (row.interest_label || row.interest_level) {
    return String(row.interest_label || row.interest_level);
  }
  if (row.stay_interest != null) {
    const n = Number(row.stay_interest);
    if (n >= 70) return "High";
    if (n < 45) return "Low";
    return "Medium";
  }
  const importance = Number(row.team_importance_score);
  if (Number.isFinite(importance)) {
    if (importance >= 0.7) return "High";
    if (importance >= 0.4) return "Medium";
    return "Low";
  }
  const ovr = Number(row.overall ?? row.ovr ?? 0);
  if (ovr >= 88) return "High";
  if (ovr >= 78) return "Medium";
  return "Low";
}

function resignArchetype(row) {
  if (!row) return "Roster Player";
  const tags = safeArray(row.tags).map(String);
  const preferred = tags.find(
    (t) => !/casualty|trade|clause|bargain|bad|fair|overpaid|ntc|nmc/i.test(t)
  );
  if (preferred) return humanizeLabel(preferred) || preferred;
  if (row.player_type || row.archetype) {
    return humanizeLabel(row.player_type || row.archetype);
  }
  const ovr = Number(row.overall ?? row.ovr ?? 0);
  const pos = String(row.position || "").toUpperCase();
  if (ovr >= 92) return pos === "G" ? "Franchise Goalie" : "Elite Playmaker";
  if (ovr >= 88) return "Core Piece";
  if (ovr >= 82) return pos === "D" ? "Top-Pair Defender" : "Top-Six Forward";
  if (ovr >= 76) return "Depth Contributor";
  return "Organizational Depth";
}

function resignSeasonBits(row) {
  if (!row) return [];
  const stats = row.season_stats || row.seasonStats || row.stats || {};
  const bits = [];
  const gp = stats.gp ?? stats.games ?? row.games_played;
  const pts = stats.points ?? stats.pts ?? row.points;
  const g = stats.goals ?? stats.g ?? row.goals;
  const a = stats.assists ?? stats.a ?? row.assists;
  if (gp != null) bits.push({ label: "GP", value: String(gp) });
  if (g != null) bits.push({ label: "G", value: String(g) });
  if (a != null) bits.push({ label: "A", value: String(a) });
  if (pts != null) bits.push({ label: "PTS", value: String(pts) });
  if (row.potential != null) bits.push({ label: "POT", value: String(row.potential) });
  if (row.contract_value_score) bits.push({ label: "Deal", value: String(row.contract_value_score) });
  return bits.slice(0, 6);
}

function resignAgentBits(row, askAav, askYears) {
  if (!row) return [];
  const lines = [];
  if (askAav) {
    lines.push(`Target AAV ${formatMoney(askAav)}${askYears ? ` over ${askYears} years` : ""}`);
  }
  const ext = row.extension_estimate || {};
  if (ext.likelyAav != null) {
    lines.push(
      `Comparable market ${formatMoney(ext.likelyAav)} × ${ext.likelyTerm || "—"}y (${ext.risk || "risk n/a"})`
    );
  }
  if (row.clause_label && row.clause_label !== "None") {
    lines.push(`Clause preference in play: ${row.clause_label}`);
  }
  const interest = resignInterestLabel(row);
  lines.push(`${interest} interest in staying with the club`);
  return lines.slice(0, 4);
}

function sortValue(row, key) {
  if (key === "name") return String(row.name || "").toLowerCase();
  if (key === "age") return intOr(row.age, 0);
  if (key === "overall") return intOr(row.overall ?? row.ovr, 0);
  if (key === "cap") return Number(row.aav_m ?? row.cap_hit_m ?? 0);
  if (key === "years") return intOr(row.years_remaining, 0);
  if (key === "expiry") return intOr(row.expiry_year, 0);
  if (key === "status") return String(row.expiry_status || row.contract_status || "");
  if (key === "interest") {
    const label = String(resignInterestLabel(row) || "").toLowerCase();
    if (label.startsWith("high")) return 3;
    if (label.startsWith("med")) return 2;
    if (label.startsWith("low")) return 1;
    return 0;
  }
  return 0;
}

export function FreeAgencyEventMenu({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
  standalone = false,
  ctaLabel: ctaLabelProp,
}) {
  const prefix = "fa";
  const phase = String(franchiseState?.phase || franchiseState?.season_phase || "").toLowerCase();
  const inOffseasonWire =
    phase === "offseason" && String(franchiseState?.offseason_stage || "") === "free_agency";
  const ctaLabel =
    ctaLabelProp ||
    (standalone || !inOffseasonWire ? "Back to Hub" : "Continue to Roster Cleanup");
  const continueHandler =
    typeof onContinue === "function"
      ? onContinue
      : standalone
        ? onBack
        : inOffseasonWire
          ? onContinue
          : onBack;
  const initialMarket =
    pickFranchiseData(franchiseState, eventData, [
      "free_agency_market",
      "offseason.free_agency_market",
    ]) ||
    franchiseState?.free_agency_market ||
    {};
  const [market, setMarket] = React.useState(initialMarket);
  const [rows, setRows] = React.useState(() => {
    const full =
      safeArray(initialMarket.free_agents).length > 0
        ? safeArray(initialMarket.free_agents)
        : safeArray(
            pickFranchiseData(franchiseState, eventData, ["free_agents", "offseason.free_agents"])
          );
    if (full.length) return full;
    return safeArray(initialMarket.major_available);
  });
  const [filter, setFilter] = React.useState("all");
  const [sortKey, setSortKey] = React.useState("overall");
  const [search, setSearch] = React.useState("");
  const [watch, setWatch] = React.useState(() => new Set());
  const [selectedId, setSelectedId] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [feedback, setFeedback] = React.useState("");
  const [response, setResponse] = React.useState(null);
  const [dealPopup, setDealPopup] = React.useState(null);
  const [faDetail, setFaDetail] = React.useState(null);
  const [faDetailLoading, setFaDetailLoading] = React.useState(false);
  const [offerAav, setOfferAav] = React.useState("");
  const [offerYears, setOfferYears] = React.useState("2");
  const [offerNtc, setOfferNtc] = React.useState(false);
  const [offerNmc, setOfferNmc] = React.useState(false);
  const [offerBonus, setOfferBonus] = React.useState("0");
  const [contractCategory, setContractCategory] = React.useState("nhl_one_way");

  // Stale empty market payloads (version stamped, 0 agents) left the Wire blank.
  // Always refresh from the desk so overseas / July 1 pools appear.
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await getFreeAgencyDesk();
        if (cancelled || !res) return;
        const next = applyMarketPayload(res);
        if (!next && safeArray(res?.free_agency_market?.free_agents).length) {
          setRows(safeArray(res.free_agency_market.free_agents));
        }
        if (res?.free_agency_market) {
          setMarket((prev) => ({ ...(prev || {}), ...res.free_agency_market }));
        }
      } catch {
        /* keep local market */
      }
    })();
    return () => {
      cancelled = true;
    };
    // Mount / stage open only — applyMarketPayload is stable enough for this pass.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const bonus = market.signing_bonus || {};
  const bonusAllowed = Boolean(bonus.eligible);
  const recent = safeArray(market.recent_league_signings);
  const news = safeArray(market.market_news);
  const capSnap = market.cap_snapshot || {};
  const capSpace = Number(
    capSnap.usable_cap_space_m ?? market.cap_space_m ?? 0
  );
  const slots = market.contract_slots || {};
  const selected =
    rows.find((p) => String(p.player_id || p.id) === String(selectedId)) || null;
  const deskPlayer = faDetail?.free_agent
    ? { ...selected, ...faDetail.free_agent }
    : selected;

  React.useEffect(() => {
    if (!selectedId) {
      setFaDetail(null);
      setFaDetailLoading(false);
      return undefined;
    }
    let cancelled = false;
    setFaDetail(null);
    setFaDetailLoading(true);
    getFreeAgentDetail(selectedId)
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
  }, [selectedId]);

  React.useEffect(() => {
    if (!selected) return;
    const ask =
      selected.askingAav ?? selected.asking_aav_m ?? selected.ask_aav_m ?? selected.asking_price;
    setOfferAav(ask != null ? String(Number(ask).toFixed(3)) : "1.000");
    setOfferYears(String(selected.askingTerm || selected.asking_term || 2));
    setOfferNtc(false);
    setOfferNmc(false);
    setOfferBonus("0");
    setError("");
    setFeedback("");
    setResponse(null);
  }, [selected?.player_id || selected?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const offerAavNum = Number(offerAav) || 0;
  const offerYearsNum = Math.max(1, parseInt(offerYears, 10) || 1);
  const offerBonusNum = Number(offerBonus) || 0;
  const projectedSpace = Number.isFinite(capSpace) ? capSpace - offerAavNum : null;
  const negoInterest = Number(response?.evaluation?.interest ?? response?.player_response?.interest ?? 0) || 0;
  const negoFeedback =
    response?.player_response?.feedback ||
    response?.evaluation?.reason ||
    response?.player_response?.clause_note ||
    null;

  React.useEffect(() => {
    if (!selected?.player_id && !selected?.id) return undefined;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const result = await evaluateContractOffer({
          player_id: selected.player_id || selected.id,
          aav_m: offerAavNum,
          years: offerYearsNum,
          ntc: offerNtc,
          nmc: offerNmc,
          signing_bonus_m: offerBonusNum,
          contract_category: contractCategory,
          two_way: contractCategory === "nhl_two_way",
          context: "ufa",
        });
        if (!cancelled && result?.evaluation) setResponse(result);
      } catch {
        /* ignore */
      }
    }, 220);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, offerAav, offerYears, offerNtc, offerNmc, offerBonus, contractCategory]);

  const filtered = React.useMemo(() => {
    let list = [...rows];
    const q = search.trim().toLowerCase();
    if (q) {
      list = list.filter((p) => {
        const blob = [
          getPlayerName(p),
          getPlayerPosition(p),
          p.previous_team,
          p.current_team,
          p.nationality,
          p.previous_team_abbrev,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return blob.includes(q);
      });
    }
    if (filter === "F") {
      list = list.filter((p) =>
        ["C", "LW", "RW", "W", "F"].includes(String(getPlayerPosition(p)).toUpperCase())
      );
    } else if (filter === "D") {
      list = list.filter((p) => ["D", "LD", "RD"].includes(String(getPlayerPosition(p)).toUpperCase()));
    } else if (filter === "G") {
      list = list.filter((p) => String(getPlayerPosition(p)).toUpperCase() === "G");
    } else if (filter === "elite") {
      list = list.filter((p) => Number(getPlayerOverall(p) || 0) >= 86);
    } else if (filter === "top6") {
      list = list.filter((p) => Number(getPlayerOverall(p) || 0) >= 82);
    } else if (filter === "cheap") {
      // Honest "cheap" = players who will actually take short money, not stars
      // with a low mirage ask / listed AAV they would never accept.
      list = list.filter((p) => {
        const ovr = Number(getPlayerOverall(p) || 0);
        const ask = Number(p.askingAav ?? p.asking_aav_m ?? p.ask_aav_m ?? 99);
        const minAccept = Number(
          p.min_acceptable_aav_m ?? p.minimum_acceptance ?? p.fair_aav_m ?? ask
        );
        const willing = Math.min(ask, minAccept);
        if (ovr >= 82) return false;
        return willing <= 2.5;
      });
    } else if (filter === "vet") {
      list = list.filter((p) => Number(p.age || 0) >= 32);
    } else if (filter === "young") {
      list = list.filter((p) => Number(p.age || 0) <= 26);
    } else if (filter === "watch") {
      list = list.filter((p) => watch.has(String(p.player_id || p.id)));
    } else if (filter === "hot") {
      list = list.filter((p) => Number(p.market_offers || p.competing_clubs || 0) >= 2);
    }
    const dir = sortKey === "age" || sortKey === "ask" ? 1 : -1;
    list.sort((a, b) => {
      const av =
        sortKey === "age"
          ? Number(a.age || 0)
          : sortKey === "ask"
            ? Number(a.askingAav ?? a.asking_aav_m ?? a.ask_aav_m ?? 0)
            : sortKey === "interest"
              ? Number(a.interest_to_user || 0)
              : Number(getPlayerOverall(a) || 0);
      const bv =
        sortKey === "age"
          ? Number(b.age || 0)
          : sortKey === "ask"
            ? Number(b.askingAav ?? b.asking_aav_m ?? b.ask_aav_m ?? 0)
            : sortKey === "interest"
              ? Number(b.interest_to_user || 0)
              : Number(getPlayerOverall(b) || 0);
      if (av === bv) return String(getPlayerName(a)).localeCompare(String(getPlayerName(b)));
      return av < bv ? -1 * (sortKey === "age" || sortKey === "ask" ? 1 : dir) : 1 * (sortKey === "age" || sortKey === "ask" ? 1 : dir);
    });
    return list;
  }, [rows, filter, search, sortKey, watch]);

  const applyMarketPayload = (result) => {
    const nextMarket = result?.free_agency_market;
    const officeSnap = result?.office?.cap_snapshot || result?.office?.team_cap;
    if (nextMarket && typeof nextMarket === "object") {
      const usable =
        officeSnap?.usable_cap_space_m ??
        nextMarket.cap_space_m ??
        nextMarket.cap_snapshot?.usable_cap_space_m;
      setMarket({
        ...nextMarket,
        ...(usable != null
          ? {
              cap_space_m: Number(usable),
              cap_snapshot: officeSnap || nextMarket.cap_snapshot || {
                ...(nextMarket.cap_snapshot || {}),
                usable_cap_space_m: Number(usable),
              },
            }
          : {}),
      });
    } else if (officeSnap) {
      setMarket((prev) => ({
        ...prev,
        cap_space_m: Number(officeSnap.usable_cap_space_m ?? prev.cap_space_m ?? 0),
        cap_snapshot: officeSnap,
      }));
    }
    const nextRows =
      safeArray(result?.free_agents).length > 0
        ? safeArray(result.free_agents)
        : safeArray(nextMarket?.free_agents);
    // Always sync the list when the desk returns an explicit free_agents array
    // (including empty after a refresh that still needs overseas top-up).
    if (Array.isArray(result?.free_agents) || Array.isArray(nextMarket?.free_agents)) {
      setRows(nextRows);
      return nextRows;
    }
    if (nextMarket?.major_available) {
      const major = safeArray(nextMarket.major_available);
      setRows(major);
      return major;
    }
    return null;
  };

  const advanceDay = async (days = 1) => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await advanceFreeAgencyDay(days);
      applyMarketPayload(result);
      const userSigned = safeArray(result?.user_resolve?.signed);
      const userRejected = safeArray(result?.user_resolve?.rejected);
      const cpuSigned = safeArray(result?.free_agency_market?.day_events?.recent_signings);
      const names = [
        ...userSigned.map((s) => s.name || s.player_id),
        ...cpuSigned.map((s) => s.name || s.player_id),
      ]
        .filter(Boolean)
        .slice(0, 4);
      setFeedback(
        `Day ${result?.day || result?.free_agency_market?.fa_market_day || "—"} · ${
          result?.free_agency_market?.market_phase_label || ""
        }${names.length ? ` · ${names.join(", ")}` : " · Market moved"}`
      );
      if (userSigned.length) {
        const first = userSigned[0];
        setDealPopup({
          tone: "accept",
          kicker: "Free agency",
          title: "Deal accepted",
          player: first.name || first.player_id,
          body: `${first.name || "Player"} accepted your pending offer.`,
          terms:
            first.aav_m != null && first.years != null
              ? `${formatMoney(first.aav_m)} × ${first.years}y`
              : null,
          cta: "Continue",
        });
      } else if (userRejected.length) {
        const first = userRejected[0];
        setDealPopup({
          tone: "deny",
          kicker: "Free agency",
          title: "Deal declined",
          player: first.name || first.player_id,
          body:
            first.feedback ||
            first.reason ||
            `${first.name || "Player"} declined your pending offer.`,
          cta: "Close",
        });
      }
      if (selectedId) {
        const stillThere = safeArray(result?.free_agency_market?.free_agents).some(
          (p) => String(p.player_id || p.id) === String(selectedId)
        );
        if (!stillThere) setSelectedId(null);
      }
    } catch (e) {
      setError(e?.message || "Could not advance market day");
    } finally {
      setBusy(false);
    }
  };

  const submitSign = async () => {
    if (!selected || busy) return;
    setBusy(true);
    setError("");
    setFeedback("");
    try {
      const pid = selected.player_id || selected.id;
      const result = await signFreeAgent({
        player_id: pid,
        aav_m: offerAavNum,
        years: offerYearsNum,
        ntc: offerNtc,
        nmc: offerNmc,
        signing_bonus_m: offerBonusNum,
        contract_category: contractCategory,
        two_way: contractCategory === "nhl_two_way",
        context: "ufa",
      });
      setResponse(result);
      applyMarketPayload(result);
      const popup = buildDealOutcome(result, {
        playerName: getPlayerName(selected),
        contextLabel: "Free agency",
      });
      if (popup) setDealPopup(popup);
      if (result?.ok && result?.status === "accepted") {
        setFeedback(`Signed ${getPlayerName(selected)}`);
        setRows((prev) => prev.filter((p) => String(p.player_id || p.id) !== String(pid)));
        setSelectedId(null);
      } else if (result?.status === "pending") {
        setFeedback(
          result?.player_response?.feedback ||
            "Offer pending — player is evaluating. Hit Sim Day to hear back."
        );
        if (!popup) {
          setDealPopup({
            tone: "pending",
            kicker: "Free agency",
            title: "Offer pending",
            player: getPlayerName(selected),
            body:
              result?.player_response?.feedback ||
              "Player is evaluating your offer. Sim Day to hear back.",
            terms: `${formatMoney(offerAavNum)} × ${offerYearsNum}y`,
            cta: "Got it",
          });
        }
        setRows((prev) =>
          prev.map((p) =>
            String(p.player_id || p.id) === String(pid)
              ? {
                  ...p,
                  decision_state: "evaluating_offers",
                  decision_reason: "Evaluating your offer",
                }
              : p
          )
        );
      } else if (result?.status === "countered" || result?.status === "rejected") {
        setFeedback(result?.player_response?.feedback || "Player countered");
      } else if (!result?.ok) {
        setError(result?.reason || result?.user_message || "Signing failed");
        if (!popup) {
          setDealPopup({
            tone: "deny",
            kicker: "Free agency",
            title: "Deal declined",
            player: getPlayerName(selected),
            body: result?.reason || result?.user_message || "Signing failed",
            cta: "Close",
          });
        }
      }
    } catch (e) {
      setError(e?.message || "Signing failed");
    } finally {
      setBusy(false);
    }
  };

  const toggleWatch = (id) => {
    setWatch((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const seasonStats =
    deskPlayer?.stats_central?.season_stats ||
    deskPlayer?.season_stats ||
    deskPlayer?.previous_season_stats ||
    {};
  const sc = deskPlayer?.stats_central || {};
  const percentiles = safeArray(sc.percentiles);
  const previousTeams = safeArray(deskPlayer?.previous_teams);
  const competing = safeArray(selected?.competing_offers);
  const interested = safeArray(selected?.interested_teams);
  const slotUsed =
    slots.used != null
      ? slots.used
      : slots.limit != null && slots.open != null
        ? slots.limit - slots.open
        : null;

  const fmtPctile = (n) => (n == null || !Number.isFinite(Number(n)) ? "—" : `${Math.round(Number(n))}th`);
  const fmtWar = (n) => (n == null || !Number.isFinite(Number(n)) ? "—" : Number(n).toFixed(2));
  const fmtStat = (n, digits = 0) => {
    if (n == null || n === "") return "—";
    const v = Number(n);
    if (!Number.isFinite(v)) return String(n);
    return digits ? v.toFixed(digits) : String(Math.round(v));
  };

  return (
    <>
    <DealOutcomeModal prefix={prefix} deal={dealPopup} onClose={() => setDealPopup(null)} />
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="FREE AGENCY WIRE"
      phaseStyle="text"
      titleVariant="market"
      rootClassName={`${prefix}-root--market`}
      seasonLabel={seasonLabel(franchiseState)}
      title="Market Board"
      hideTitle
      hideEyebrow
      hideTicker
      ctaLabel={ctaLabel}
      onContinue={continueHandler}
      onBack={onBack}
      footerAlign="split"
      railTitle="Market Wire"
      railHint={null}
      heroContent={
        <div className={`${prefix}-workspace`}>
          <div className={`${prefix}-fa-bar`}>
            <strong>{market.market_phase_label || "Open market"}</strong>
            <span>Day {market.fa_market_day ?? 0}</span>
            <span>{market.available_count ?? rows.length} available</span>
            <span className={`${prefix}-fa-cap`}>{formatMoney(capSpace)} space</span>
            <span>
              {slotUsed != null && slots.limit != null
                ? `${slotUsed}/${slots.limit} contracts`
                : `${slots.open ?? "—"} open slots`}
            </span>
            <span className={bonusAllowed ? "is-ok" : "is-bad"}>
              Bonus {bonusAllowed ? "OK" : `Locked $${Math.round(Number(bonus.floor_m) || 155)}M`}
            </span>
            <button type="button" className={`${prefix}-sim-btn`} disabled={busy} onClick={() => advanceDay(1)}>
              Sim day
            </button>
            <button type="button" className={`${prefix}-sim-btn`} disabled={busy} onClick={() => advanceDay(7)}>
              Sim week
            </button>
          </div>

          <div className={`${prefix}-fa-tools`}>
            <input
              className={`${prefix}-fa-search`}
              placeholder="Search name, team, country…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <div className={`${prefix}-fa-filters`}>
              {[
                ["all", "All"],
                ["elite", "Elite"],
                ["top6", "Top 6"],
                ["F", "Forwards"],
                ["D", "Defense"],
                ["G", "Goalies"],
                ["young", "Young"],
                ["vet", "Veterans"],
                ["cheap", "Cheap"],
                ["hot", "Bidding"],
                ["watch", "Watch"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={`${prefix}-fa-chip${filter === id ? " is-active" : ""}`}
                  onClick={() => setFilter(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <select
              className={`${prefix}-fa-sort`}
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
            >
              <option value="overall">Sort: OVR</option>
              <option value="ask">Sort: Ask</option>
              <option value="age">Sort: Age</option>
              <option value="interest">Sort: Interest in you</option>
            </select>
          </div>

          {error ? <p className={`${prefix}-warn`}>{error}</p> : null}
          {feedback ? <p className={`${prefix}-context`}>{feedback}</p> : null}

          <div className={`${prefix}-fa-layout`}>
            <div className={`${prefix}-fa-list`}>
              <p className={`${prefix}-fa-count`}>{filtered.length} free agents</p>
              {filtered.length ? (
                filtered.map((p, i) => {
                  const id = String(p.player_id || p.id || i);
                  const active = String(selectedId) === id;
                  const prevAbbr = p.previous_team_abbrev || "";
                  const prevName = p.previous_team || p.current_team || prevAbbr || "FA";
                  const logos = safeArray(p.interested_teams).slice(0, 4);
                  const interest = p.interest_to_user_label || p.market_interest || "—";
                  return (
                    <button
                      key={id}
                      type="button"
                      className={`${prefix}-fa-row${active ? " is-selected" : ""}`}
                      onClick={() => setSelectedId(id)}
                    >
                      <span className={`${prefix}-fa-prev`}>
                        <TeamLogoBadge
                          teamLogo={resolveFranchiseTeamLogo(
                            { abbrev: prevAbbr, team_abbrev: prevAbbr, name: prevName },
                            prevName
                          )}
                          teamName={prevName}
                          size={28}
                          variant="circle"
                        />
                      </span>
                      <span className={`${prefix}-fa-row-body`}>
                        <strong>{getPlayerName(p)}</strong>
                        <em>
                          {[
                            getPlayerPosition(p),
                            getPlayerOverall(p) != null ? `${getPlayerOverall(p)} OVR` : null,
                            p.age != null ? `Age ${p.age}` : null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </em>
                        <span className={`${prefix}-fa-row-meta`}>
                          {String(p.decision_state || "awaiting").replace(/_/g, " ")}
                          {" · "}
                          Interest {String(interest)}
                        </span>
                      </span>
                      <span className={`${prefix}-fa-logos`}>
                        {logos.map((t) => (
                          <TeamLogoBadge
                            key={t.team_id || t.team_abbrev}
                            teamLogo={resolveFranchiseTeamLogo(
                              {
                                abbrev: t.team_abbrev,
                                team_abbrev: t.team_abbrev,
                                name: t.team_name,
                              },
                              t.team_name || t.team_abbrev
                            )}
                            teamName={t.team_name || t.team_abbrev}
                            size={18}
                            variant="circle"
                          />
                        ))}
                      </span>
                      <span className={`${prefix}-fa-ask`}>
                        {formatMoney(p.askingAav ?? p.asking_aav_m ?? p.ask_aav_m)}
                      </span>
                      <span
                        className={`${prefix}-fa-star${watch.has(id) ? " is-on" : ""}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleWatch(id);
                        }}
                        role="presentation"
                      >
                        ★
                      </span>
                    </button>
                  );
                })
              ) : (
                <p className={`${prefix}-empty`}>
                  {market.empty_reason || "No free agents match"}
                </p>
              )}
            </div>

            <aside className={`${prefix}-fa-desk`}>
              {!selected ? (
                <p className={`${prefix}-empty`}>Select a free agent to open the signing desk.</p>
              ) : (
                <>
                  <div className={`${prefix}-fa-identity`}>
                    <div className={`${prefix}-fa-shot`}>
                      <PlayerHeadshot
                        player={ensurePlayerHeadshotFields({
                          id: deskPlayer.player_id || deskPlayer.id,
                          player_id: deskPlayer.player_id || deskPlayer.id,
                          name: getPlayerName(deskPlayer),
                          position: getPlayerPosition(deskPlayer),
                          age: deskPlayer.age,
                          nationality: deskPlayer.nationality,
                        })}
                        size="lg"
                      />
                    </div>
                    <div>
                      <span className={`${prefix}-fa-ovr`}>{getPlayerOverall(deskPlayer) ?? "—"} OVR</span>
                      <h3>{String(getPlayerName(deskPlayer) || "").toUpperCase()}</h3>
                      <p>
                        {[
                          getPlayerPosition(deskPlayer),
                          deskPlayer.age != null ? `Age ${deskPlayer.age}` : null,
                          deskPlayer.potential != null ? `POT ${deskPlayer.potential}` : null,
                          deskPlayer.role,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                      <p className={`${prefix}-fa-prev-line`}>
                        Former · {deskPlayer.previous_team || deskPlayer.current_team || deskPlayer.nhl_team || "Unsigned"}
                        {deskPlayer.nationality ? ` · ${deskPlayer.nationality}` : ""}
                      </p>
                    </div>
                  </div>

                  <div className={`${prefix}-fa-capbar`}>
                    <div>
                      <span>Cap before</span>
                      <strong>{formatMoney(capSpace)}</strong>
                    </div>
                    <div>
                      <span>After offer</span>
                      <strong className={projectedSpace != null && projectedSpace < 0 ? "is-red" : "is-green"}>
                        {projectedSpace != null ? formatMoney(projectedSpace) : "—"}
                      </strong>
                    </div>
                    <div className={`${prefix}-fa-captrack`}>
                      <span
                        style={{
                          width: `${Math.max(
                            4,
                            Math.min(100, ((offerAavNum || 0) / Math.max(capSpace || 1, 1)) * 100)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className={`${prefix}-nego-meter`}>
                    <div className={`${prefix}-nego-meter-head`}>
                      <span>Deal interest</span>
                      <strong>{negoInterest ? Math.round(negoInterest) : "—"}</strong>
                    </div>
                    <div className={`${prefix}-nego-meter-track`}>
                      <span
                        className={`${prefix}-nego-meter-fill tone-${
                          negoInterest >= 88
                            ? "instant"
                            : negoInterest >= 62
                              ? "good"
                              : negoInterest >= 40
                                ? "mid"
                                : "bad"
                        }`}
                        style={{ width: `${Math.max(4, Math.min(100, negoInterest || 4))}%` }}
                      />
                    </div>
                    <p className={`${prefix}-nego-meter-note`}>
                      {negoFeedback ||
                        `Your club interest ${selected.interest_to_user_label || "—"} (${
                          selected.interest_to_user != null ? Math.round(selected.interest_to_user) : "—"
                        }) · Sign likelihood ${
                          selected.sign_likelihood != null
                            ? `${Math.round(Number(selected.sign_likelihood) * 100)}%`
                            : "—"
                        }`}
                    </p>
                  </div>

                  {faDetailLoading ? (
                    <p className={`${prefix}-context`}>Loading Stats Central…</p>
                  ) : null}

                  <div className={`${prefix}-fa-stats`}>
                    {[
                      ["GP", seasonStats.gp],
                      ["G", seasonStats.g ?? seasonStats.goals],
                      ["A", seasonStats.a ?? seasonStats.assists],
                      ["P", seasonStats.points ?? seasonStats.pts],
                      ["WAR", sc.war ?? seasonStats.war ?? deskPlayer.war],
                      ["+/-", seasonStats.plus_minus ?? seasonStats.pm],
                    ]
                      .filter(([, v]) => v != null && v !== "")
                      .map(([k, v]) => (
                        <div key={k}>
                          <span>{k}</span>
                          <strong>{k === "WAR" ? fmtWar(v) : fmtStat(v, k === "+/-" ? 0 : 0)}</strong>
                        </div>
                      ))}
                  </div>

                  {percentiles.length ? (
                    <div className={`${prefix}-fa-pctiles`}>
                      <span className={`${prefix}-mini-label`}>
                        Stats Central · {sc.pool_size || percentiles.length} peers
                      </span>
                      {percentiles.map((row) => (
                        <div key={row.key || row.label} className={`${prefix}-fa-pctile`}>
                          <span>{row.label}</span>
                          <div className={`${prefix}-fa-pctile-track`}>
                            <i style={{ width: `${Math.max(2, Number(row.percentile) || 0)}%` }} />
                          </div>
                          <strong>{fmtPctile(row.percentile)}</strong>
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {previousTeams.length ? (
                    <div className={`${prefix}-fa-history`}>
                      <span className={`${prefix}-mini-label`}>Previous teams</span>
                      {previousTeams.slice(0, 4).map((t, i) => {
                        const st = t.stats || {};
                        const line = st.is_goalie
                          ? `${st.gp ?? "—"} GP · ${st.save_pct != null ? Number(st.save_pct).toFixed(3) : "—"} SV%`
                          : `${st.gp ?? "—"} GP · ${st.pts ?? st.points ?? "—"} PTS${
                              st.war != null ? ` · ${fmtWar(st.war)} WAR` : ""
                            }`;
                        return (
                          <div key={`${t.team || "club"}-${i}`} className={`${prefix}-fa-history-row`}>
                            <strong>{t.team || t.league || "—"}</strong>
                            <em>
                              {[t.label, t.league, t.season].filter(Boolean).join(" · ")}
                            </em>
                            <span>{line}</span>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}

                  {competing.length ? (
                    <div className={`${prefix}-fa-offers`}>
                      <span className={`${prefix}-mini-label`}>Offers received</span>
                      {competing.slice(0, 5).map((o) => (
                        <div key={`${o.team_id}-${o.aav_m}`} className={`${prefix}-fa-offer`}>
                          <TeamLogoBadge
                            teamLogo={resolveFranchiseTeamLogo(
                              {
                                abbrev: o.team_abbrev,
                                team_abbrev: o.team_abbrev,
                                name: o.team_name,
                              },
                              o.team_name || o.team_abbrev
                            )}
                            teamName={o.team_name || o.team_abbrev}
                            size={22}
                            variant="circle"
                          />
                          <strong>{o.team_abbrev || o.team_name}</strong>
                          <em>
                            {formatMoney(o.aav_m)} · {o.years}y
                            {o.is_user ? " · YOU" : ""}
                          </em>
                        </div>
                      ))}
                    </div>
                  ) : interested.length ? (
                    <div className={`${prefix}-fa-offers`}>
                      <span className={`${prefix}-mini-label`}>Clubs circling</span>
                      <div className={`${prefix}-fa-logos is-large`}>
                        {interested.map((t) => (
                          <TeamLogoBadge
                            key={t.team_id || t.team_abbrev}
                            teamLogo={resolveFranchiseTeamLogo(
                              {
                                abbrev: t.team_abbrev,
                                team_abbrev: t.team_abbrev,
                                name: t.team_name,
                              },
                              t.team_name || t.team_abbrev
                            )}
                            teamName={t.team_name || t.team_abbrev}
                            size={26}
                            variant="circle"
                          />
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className={`${prefix}-fa-controls`}>
                    <label className={`${prefix}-field`}>
                      Type
                      <div className={`${prefix}-select-wrap`}>
                        <select
                          value={contractCategory}
                          disabled={busy}
                          onChange={(e) => setContractCategory(e.target.value)}
                        >
                          <option value="nhl_one_way">NHL one-way</option>
                          <option value="nhl_two_way">NHL two-way</option>
                          <option value="ahl">AHL</option>
                          <option value="pto">PTO</option>
                        </select>
                      </div>
                    </label>
                    <div className={`${prefix}-slider-block`}>
                      <div className={`${prefix}-slider-head`}>
                        <span>Annual salary</span>
                        <strong>{formatMoney(offerAavNum)}</strong>
                      </div>
                      <input
                        type="range"
                        className={`${prefix}-salary-slider`}
                        min={0.775}
                        max={Math.max(12, offerAavNum * 1.35, Number(selected.ask_aav_m || selected.askingAav || 4) * 1.4)}
                        step="0.025"
                        value={Math.min(
                          Math.max(12, offerAavNum * 1.35, Number(selected.ask_aav_m || selected.askingAav || 4) * 1.4),
                          Math.max(0.775, offerAavNum || 0.775)
                        )}
                        disabled={busy}
                        onChange={(e) => setOfferAav(Number(e.target.value).toFixed(3))}
                      />
                    </div>
                    <div className={`${prefix}-term-block`}>
                      <span className={`${prefix}-mini-label`}>Term</span>
                      <div className={`${prefix}-term-seg`}>
                        {[1, 2, 3, 4, 5, 6, 7, 8].map((y) => (
                          <button
                            key={y}
                            type="button"
                            className={offerYearsNum === y ? "is-active" : ""}
                            disabled={busy}
                            onClick={() => setOfferYears(String(y))}
                          >
                            {y}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className={`${prefix}-check-row is-clauses`}>
                      <label className={offerNtc ? "is-on" : ""}>
                        <input
                          type="checkbox"
                          checked={offerNtc}
                          disabled={busy || offerNmc}
                          onChange={(e) => setOfferNtc(e.target.checked)}
                        />
                        NTC
                      </label>
                      <label className={offerNmc ? "is-on" : ""}>
                        <input
                          type="checkbox"
                          checked={offerNmc}
                          disabled={busy}
                          onChange={(e) => {
                            setOfferNmc(e.target.checked);
                            if (e.target.checked) setOfferNtc(true);
                          }}
                        />
                        NMC
                      </label>
                    </div>
                    <div className={`${prefix}-slider-block`}>
                      <div className={`${prefix}-slider-head`}>
                        <span>Signing bonus</span>
                        <strong>{bonusAllowed ? formatMoney(offerBonusNum) : "Locked"}</strong>
                      </div>
                      {bonusAllowed ? (
                        <input
                          type="range"
                          className={`${prefix}-salary-slider`}
                          min={0}
                          max={Math.max(0.25, offerAavNum * offerYearsNum * Number(bonus.max_bonus_pct || 0.08))}
                          step="0.025"
                          value={Math.max(0, Math.min(offerBonusNum, offerAavNum * offerYearsNum * Number(bonus.max_bonus_pct || 0.08)))}
                          disabled={busy}
                          onChange={(e) => setOfferBonus(Number(e.target.value).toFixed(3))}
                        />
                      ) : (
                        <p className={`${prefix}-context`}>
                          {bonus.label || `Signing bonuses require NHL revenue ≥ $${Math.round(Number(bonus.floor_m) || 155)}M`}
                        </p>
                      )}
                    </div>
                    <div className={`${prefix}-negotiate-actions`}>
                      <button
                        type="button"
                        className={`${prefix}-cta-btn ${prefix}-submit-btn`}
                        disabled={busy}
                        onClick={submitSign}
                      >
                        {busy ? "Working…" : "Submit Offer"}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </aside>
          </div>
        </div>
      }
      railContent={
        <>
          {(news.length ? news : recent).slice(-18).reverse().map((n, i) => (
            <p key={`${n.text || n.player_id || "wire"}-${i}`} className={`${prefix}-wire-item`}>
              {n.text ||
                [
                  n.team_name || n.team_abbrev || (n.team_id && String(n.team_id).length > 3 ? n.team_id : null) || "A club",
                  n.name || n.player_id,
                  n.aav_m != null ? `${n.aav_m}M × ${n.years || "?"}y` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
            </p>
          ))}
          {!news.length && !recent.length ? (
            <p className={`${prefix}-empty`}>
              Submit offers or Sim Day — signings and pending decisions land here.
            </p>
          ) : null}
        </>
      }
    />
    </>
  );
}

export function SalaryCapEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  return (
    <CapReportBoard
      franchiseState={franchiseState}
      eventData={eventData}
      onContinue={onContinue}
      onBack={onBack}
    />
  );
}

export function DevelopmentReportEventMenu(props) {
  return <ProspectDevelopmentMenu {...props} />;
}

export function RosterCleanupEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "cleanup";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "roster_cleanup",
    "roster_issues",
    "offseason.roster_cleanup",
  ]) || {};
  const blocking = safeArray(raw.blocking?.length ? raw.blocking : (raw.issues || []).map((m) => ({ message: m })));
  const warnings = safeArray(raw.warnings?.length ? raw.warnings : (raw.warning_messages || []).map((m) => ({ message: m })));
  const canGenerate = Boolean(raw.valid ?? franchiseState?.flags?.can_generate_next_season);
  const nhlCount = raw.nhl_roster_count;
  const nhlMin = raw.nhl_roster_min ?? 20;
  const nhlMax = raw.nhl_roster_max ?? 23;
  const minF = raw.min_forwards ?? 12;
  const minD = raw.min_defense ?? 6;
  const minG = raw.min_goalies ?? 2;
  const forwards = raw.forward_count;
  const defense = raw.defense_count;
  const goalies = raw.goalie_count;
  const irCount = raw.ir_count;
  const ltirCount = raw.ltir_count;
  const resolveRoute = String(blocking[0]?.route || "free_agency").toLowerCase();
  const resolveStage = resolveRoute.includes("re_sign") || resolveRoute.includes("resign")
    ? "re_sign"
    : "free_agency";

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="ROSTER COMPLIANCE"
      titleVariant="compliance"
      seasonLabel={seasonLabel(franchiseState)}
      title="Roster Check"
      eyebrow="Pre-season compliance sheet"
      ctaLabel={canGenerate ? "Generate Next Season" : "Resolve Blocking Issues"}
      tickerItems={["ROSTER CHECK", "COMPLIANCE", "FINAL REVIEW", "READY TO GO"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        <div className={`${prefix}-meta`}>
          <span className={`${prefix}-chip`} title="Active NHL roster (excludes buried, IR, LTIR)">
            Active NHL {nhlCount ?? "—"}/{nhlMin}–{nhlMax}
          </span>
          <span className={`${prefix}-chip`}>
            F {forwards ?? "—"}/{minF}
          </span>
          <span className={`${prefix}-chip`}>
            D {defense ?? "—"}/{minD}
          </span>
          <span className={`${prefix}-chip`}>
            G {goalies ?? "—"}/{minG}
          </span>
          {irCount != null || ltirCount != null ? (
            <span className={`${prefix}-chip`}>
              IR {irCount ?? 0} · LTIR {ltirCount ?? 0}
            </span>
          ) : null}
          <span className={`${prefix}-chip`}>Space {formatMoney(raw.cap_space_m)}</span>
          <span className={`${prefix}-chip`}>
            Slots {raw.contract_slots_used ?? "—"}/{raw.contract_slots_limit ?? "—"}
          </span>
          <span className={`${prefix}-chip ${canGenerate ? "is-ready" : "is-blocked"}`}>
            {canGenerate ? "Ready" : "Blocked"}
          </span>
        </div>
      }
      railTitle="Compliance"
      railContent={
        <>
          {blocking.length ? (
            blocking.map((item, i) => (
              <Card
                key={`b-${i}`}
                prefix={prefix}
                title={String(item.message || item)}
                details={[
                  item.route
                    ? `Resolve in ${item.route === "free_agency" ? "Free Agency" : item.route}`
                    : "Blocking",
                ].filter(Boolean)}
                active
              />
            ))
          ) : null}
          {warnings.length ? (
            warnings.map((item, i) => (
              <Card
                key={`w-${i}`}
                prefix={prefix}
                title={String(item.message || item)}
                details={["Warning"]}
              />
            ))
          ) : null}
          {!blocking.length && !warnings.length ? (
            <p className={`${prefix}-empty`}>No roster issues flagged</p>
          ) : null}
          {!canGenerate ? (
            <p className={`${prefix}-empty`}>
              Generate stays locked until blocking issues clear.
              {" "}
              Use Resolve Blocking Issues to return to {resolveStage === "re_sign" ? "Re-Sign" : "Free Agency"}.
            </p>
          ) : null}
        </>
      }
    />
  );
}

export function NextSeasonRevealEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "nextseason";
  const raw = pickFranchiseData(franchiseState, eventData, [
    "next_season",
    "calendar_summary",
  ]) || {};
  const year = raw.season_year || franchiseState?.season_year;
  const markers = safeArray(raw.calendar_markers).slice(0, 6);

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="SEASON COUNTDOWN"
      titleVariant="countdown"
      seasonLabel={raw.season_label || (year ? `${year}–${Number(year) + 1}` : "")}
      title="New Season"
      eyebrow="Calendar locked · camp opens soon"
      ctaLabel="Enter Preseason"
      tickerItems={["NEW YEAR", "CAMP OPENS", "THE CHASE RETURNS", "FRANCHISE CONTINUES"]}
      onContinue={onContinue}
      onBack={onBack}
      heroContent={
        <>
          <p className={`${prefix}-hero-name`}>{raw.season_label || year || "Next Season"}</p>
          <div className={`${prefix}-meta`}>
            {raw.opening_night ? <span className={`${prefix}-chip`}>Opening {raw.opening_night}</span> : null}
            {raw.preseason_start ? <span className={`${prefix}-chip`}>Camp {raw.preseason_start}</span> : null}
            {raw.first_opponent ? <span className={`${prefix}-chip`}>vs {raw.first_opponent}</span> : null}
            {raw.salary_cap_m != null ? <span className={`${prefix}-chip`}>Cap ${raw.salary_cap_m}M</span> : null}
            <span className={`${prefix}-chip`}>{raw.generation_status || "ready"}</span>
          </div>
        </>
      }
      railTitle="Season Preview"
      railContent={
        <>
          {raw.schedule_games != null ? (
            <Card prefix={prefix} title="Schedule Ready" details={[`${raw.schedule_games} games mapped`]} />
          ) : null}
          {markers.map((m, i) => (
            <Card
              key={m.key || i}
              prefix={prefix}
              title={m.label || m.key || "Marker"}
              details={[m.iso].filter(Boolean)}
            />
          ))}
          {!raw.schedule_games ? <p className={`${prefix}-empty`}>Generate next season to load calendar</p> : null}
        </>
      }
    />
  );
}

function draftReviewFirstId(picks) {
  const row = picks.find((p) => p?.prospect_id != null && String(p.prospect_id));
  return row ? String(row.prospect_id) : null;
}

function productionStatEntries(prod, isGoalie) {
  if (!prod || typeof prod !== "object" || prod.mode === "scouting") return [];
  if (isGoalie) {
    return [
      ["GP", prod.games],
      ["Starts", prod.starts],
      ["SV%", prod.save_percentage],
      ["GAA", prod.goals_against_average],
      ["SO", prod.shutouts],
    ].filter(([, v]) => v != null && v !== "");
  }
  return [
    ["GP", prod.games],
    ["G", prod.goals],
    ["A", prod.assists],
    ["PTS", prod.points],
    ["PPG", prod.points_per_game],
  ].filter(([, v]) => v != null && v !== "").slice(0, 5);
}

function formatProductionStat(label, value) {
  if (value == null || value === "") return null;
  if (label === "PPG" || label === "GAA") {
    const n = Number(value);
    if (Number.isFinite(n)) return n.toFixed(2);
  }
  return value;
}

function findPickName(picks, id) {
  if (!id) return null;
  return picks.find((p) => String(p.prospect_id) === String(id))?.prospect_name || null;
}

function humanizeLabel(value) {
  if (value == null || value === "") return null;
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function gradeLetter(value) {
  if (value == null || value === "") return null;
  const m = String(value).trim().match(/[A-F][+-]?/i);
  return m ? m[0].toUpperCase() : String(value).trim().slice(0, 2);
}

function gradeTone(letter) {
  const L = String(letter || "").charAt(0).toUpperCase();
  if (L === "A") return "a";
  if (L === "B") return "b";
  if (L === "C") return "c";
  if (L === "D" || L === "F") return "d";
  return "n";
}

function riskTone(risk) {
  const r = String(risk || "").toLowerCase();
  if (r.includes("high") || r.includes("excellent") || r.includes("ideal") || r.includes("strong")) return "high";
  if (r.includes("low") || r.includes("poor") || r.includes("thin")) return "low";
  if (r.includes("med") || r.includes("mid") || r.includes("moderate") || r.includes("acceptable")) return "med";
  return "neutral";
}

function confidenceWord(value) {
  if (value == null || value === "") return null;
  const raw = String(value).trim();
  const tone = riskTone(raw);
  if (tone === "high") return raw.toLowerCase().includes("excellent") ? "Excellent" : "High";
  if (tone === "low") return "Low";
  if (tone === "med") return raw.toLowerCase().includes("moderate") ? "Moderate" : "Medium";
  return humanizeLabel(raw);
}

/** Map categorical confidence/risk labels to a presentation bar width only. */
function confidencePct(value) {
  const tone = riskTone(value);
  if (tone === "high") return 86;
  if (tone === "med") return 55;
  if (tone === "low") return 28;
  return 42;
}

function valueBannerTone(verdict) {
  const v = String(verdict || "").toLowerCase();
  if (v.includes("best value") || v.includes("steal")) return "steal";
  if (v.includes("good value") || v.includes("value")) return "value";
  if (v.includes("reach") || v.includes("aggressive") || v.includes("swing")) return "reach";
  if (v.includes("expected") || v.includes("need")) return "expected";
  return "neutral";
}

function positionGroup(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "G") return "g";
  if (p === "D" || p === "LD" || p === "RD" || p === "LHD" || p === "RHD") return "d";
  if (p === "C" || p === "LW" || p === "RW" || p === "W" || p === "F") return "f";
  return "f";
}

function wordCount(text) {
  return String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function pickConciseLine(candidates, maxWords = 15) {
  const lines = candidates.filter((v) => v != null && String(v).trim() !== "").map((v) => String(v).trim());
  if (!lines.length) return null;
  const scored = lines
    .map((line) => ({ line, words: wordCount(line) }))
    .sort((a, b) => a.words - b.words);
  const preferred = scored.find((s) => s.words <= maxWords) || scored[0];
  const words = preferred.line.split(/\s+/);
  if (words.length <= maxWords) return preferred.line;
  return `${words.slice(0, maxWords).join(" ")}…`;
}

function pathStepStatus(status) {
  const s = String(status || "future").toLowerCase();
  if (s === "next" || s === "current" || s === "active") return "next";
  if (s === "projection") return "projection";
  return "future";
}

function buildPathStages(pick, plan) {
  const steps = safeArray(plan?.path_steps).slice(0, 3);
  if (steps.length) {
    return steps
      .map((step, i) => ({
        key: `${step.stage || "stage"}-${i}`,
        label: step.stage || null,
        detail: step.detail || null,
        status:
          pathStepStatus(step.status) ||
          (i === 0 ? "next" : i === steps.length - 1 ? "projection" : "future"),
      }))
      .filter((s) => s.label);
  }
  const visual = safeArray(pick?.path_visual).filter(Boolean).slice(0, 3);
  return visual.map((stage, i) => ({
    key: `${stage}-${i}`,
    label: stage,
    detail: null,
    status: i === 0 ? "next" : i === visual.length - 1 ? "projection" : "future",
  }));
}

function buildScoutSignals(pick, fit, production) {
  const pros = [];
  const cons = [];
  safeArray(fit?.opportunities).forEach((item) => {
    if (item && pros.length < 2) pros.push({ text: String(item), tone: "pro" });
  });
  safeArray(fit?.blockers).forEach((item) => {
    if (item && cons.length < 2) cons.push({ text: String(item), tone: "con" });
  });
  if (pick?.risk_reason && cons.length < 2) {
    cons.push({ text: String(pick.risk_reason), tone: "con" });
  }
  const notes = safeArray(production?.notes);
  if (!pros.length && !cons.length && notes.length) {
    return notes.slice(0, 4).map((note) => ({ text: String(note), tone: "neutral" }));
  }
  return [...pros, ...cons].slice(0, 4);
}

function MeterRow({ prefix, label, value }) {
  if (!value) return null;
  const word = confidenceWord(value);
  const tone = riskTone(value);
  const pct = confidencePct(value);
  return (
    <div className={`${prefix}-meter-row`}>
      <div className={`${prefix}-meter-head`}>
        <span>{label}</span>
        <strong>{word}</strong>
      </div>
      <div
        className={`${prefix}-meter-track`}
        role="meter"
        aria-label={`${label}: ${word}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <span className={`${prefix}-meter-fill tone-${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

const DRAFT_REVIEW_TABS = ["DRAFT REVIEW", "VALUE & FIT", "DEVELOPMENT PLAN", "NEXT: RIGHTS"];

export function DraftReviewEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "draftreview";
  const raw = pickFranchiseData(franchiseState, eventData, ["draft_review", "offseason.draft_review"]) || {};
  const picks = safeArray(raw.user_picks);
  const haul = raw.haul_summary || {};
  const [selectedId, setSelectedId] = React.useState(() => draftReviewFirstId(picks));
  const [showMore, setShowMore] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState(DRAFT_REVIEW_TABS[0]);
  const haulRef = React.useRef(null);
  const fitRef = React.useRef(null);
  const planRef = React.useRef(null);
  const rightsRef = React.useRef(null);
  const railItemRefs = React.useRef({});

  React.useEffect(() => {
    if (!picks.length) {
      setSelectedId(null);
      return;
    }
    const stillThere = picks.some((p) => String(p?.prospect_id) === String(selectedId));
    if (!stillThere) setSelectedId(draftReviewFirstId(picks));
  }, [picks, selectedId]);

  React.useEffect(() => {
    setShowMore(false);
  }, [selectedId]);

  React.useEffect(() => {
    const node = railItemRefs.current[String(selectedId)];
    if (node && typeof node.scrollIntoView === "function") {
      node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedId]);

  const pick =
    picks.find((p) => String(p?.prospect_id) === String(selectedId)) || picks[0] || null;
  const plan = pick?.development_plan || {};
  const production = pick?.production || {};
  const fit = pick?.organizational_fit || {};
  const rights = pick?.rights_card || {};
  const isGoalie = String(pick?.position || "").toUpperCase() === "G";
  const stats = productionStatEntries(production, isGoalie);
  const pathStages = buildPathStages(pick, plan);
  const scoutSignals = buildScoutSignals(pick, fit, production);
  const pros = scoutSignals.filter((s) => s.tone === "pro").slice(0, 2);
  const cons = scoutSignals.filter((s) => s.tone === "con").slice(0, 2);
  const neutralNotes = scoutSignals.filter((s) => s.tone === "neutral").slice(0, 4);

  const closestName = findPickName(picks, haul.closest_to_nhl_pick_id);
  const bestValueName = findPickName(picks, haul.best_value_pick_id);
  const pickGrade = gradeLetter(pick?.selection_grade);
  const haulGrade = haul.haul_grade || raw.user_grade || null;
  const haulLabel = haul.haul_grade_label || null;
  const haulExplain = pickConciseLine([haul.summary_line, haul.haul_grade_reason], 20);
  const pos = pick?.position || getPlayerPosition(pick);
  const posGroup = positionGroup(pos);
  const valueTone = valueBannerTone(pick?.selection_verdict || pick?.selection_grade_label);
  const ppgStat = stats.find(([label]) => label === "PPG");
  const otherStats = stats.filter(([label]) => label !== "PPG");

  const haulMetrics = [
    {
      key: "picks",
      label: "Picks",
      value: haul.total_picks != null ? String(haul.total_picks) : picks.length ? String(picks.length) : null,
    },
    haul.reaches != null && Number(haul.reaches) > 0
      ? { key: "reach", label: "Reaches", value: String(haul.reaches) }
      : null,
    bestValueName ? { key: "value", label: "Best value", value: bestValueName } : null,
    closestName ? { key: "closest", label: "Closest NHL", value: closestName } : null,
  ].filter((m) => m && m.value);

  const classChips = [
    haul.near_ready_count != null && Number(haul.near_ready_count) > 0
      ? `${haul.near_ready_count} near-ready`
      : null,
    haul.steals != null && Number(haul.steals) > 0 ? `${haul.steals} steal${Number(haul.steals) === 1 ? "" : "s"}` : null,
    haul.reaches != null && Number(haul.reaches) > 0
      ? `${haul.reaches} reach${Number(haul.reaches) === 1 ? "" : "es"}`
      : null,
    haul.long_term_projects != null && Number(haul.long_term_projects) > 0
      ? `${haul.long_term_projects} long project${Number(haul.long_term_projects) === 1 ? "" : "s"}`
      : null,
    haul.position_balance_label || null,
  ].filter(Boolean);

  const oneLineVerdict = pickConciseLine(
    [
      pick?.selection_reason,
      pick?.review_line,
      pick?.selection_verdict,
      fit?.fit_label,
      plan?.nhl_projection,
    ],
    18
  );

  const developReason = pickConciseLine(
    [plan.season_objective, plan.nhl_projection_reason, fit.environment_reason],
    18
  );
  const orgNote = pickConciseLine(
    [fit.fit_tension_note, fit.environment_reason, fit.depth_status, rights.signing_reason],
    22
  );
  const prodContext = pickConciseLine(
    [production.league_context, production.production_trend, production.headline, production.board_context],
    18
  );
  const hasDepth =
    fit.nhl_players_ahead != null ||
    fit.ahl_players_ahead != null ||
    fit.prospects_ahead != null;
  const depthMax = Math.max(
    1,
    Number(fit.nhl_players_ahead) || 0,
    Number(fit.ahl_players_ahead) || 0,
    Number(fit.prospects_ahead) || 0
  );

  const hasOrgPlan = Boolean(
    fit.need_filled ||
      fit.fit_label ||
      fit.fit_grade ||
      fit.depth_status ||
      fit.path_congestion ||
      rights.rights_status_label ||
      rights.signing_recommendation ||
      fit.environment_grade ||
      hasDepth ||
      orgNote
  );

  const advancedLines = [
    pick?.selection_reason,
    pick?.review_line,
    plan.nhl_projection_reason,
    plan.alternate_path,
    fit.fit_tension_note,
    fit.environment_reason,
    rights.signing_reason,
    rights.rights_deadline_label,
    rights.rights_status_label,
    production.board_context,
    production.league_context,
    safeArray(production.notes).join(" · ") || null,
    pick?.floor_label ? `Floor: ${pick.floor_label}` : null,
    pick?.ceiling_label ? `Ceiling: ${pick.ceiling_label}` : null,
    pick?.board_range ? `Public board: ${pick.board_range}` : null,
    pick?.selection_delta_label || null,
  ].filter((line, i, arr) => {
    if (!line) return false;
    const n = String(line).trim();
    if (!n) return false;
    if (n === oneLineVerdict || n === developReason || n === orgNote || n === prodContext) return false;
    return arr.findIndex((x) => String(x).trim() === n) === i;
  });

  const onRailKeyDown = (e, id) => {
    if (!picks.length) return;
    const ids = picks.map((p) => String(p.prospect_id));
    const idx = Math.max(0, ids.indexOf(String(selectedId)));
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      setSelectedId(ids[Math.min(ids.length - 1, idx + 1)]);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      setSelectedId(ids[Math.max(0, idx - 1)]);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedId(String(id));
    }
  };

  const focusPane = (ref) => {
    const node = ref?.current;
    if (node && typeof node.scrollIntoView === "function") {
      node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  };

  const onTickerSelect = (item) => {
    setActiveTab(item);
    if (item === "DRAFT REVIEW") focusPane(haulRef);
    else if (item === "VALUE & FIT") focusPane(fitRef);
    else if (item === "DEVELOPMENT PLAN") focusPane(planRef);
    else if (item === "NEXT: RIGHTS") focusPane(rightsRef);
  };

  const nextHero =
    plan.next_club || plan.next_destination || plan.next_destination_label || null;

  return (
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="DRAFT HAUL REVIEW"
      phaseStyle="text"
      titleVariant="review"
      seasonLabel={seasonLabel(franchiseState)}
      title="Draft Review"
      hideTitle
      hideEyebrow
      ctaLabel="Open Prospect Rights"
      tickerItems={DRAFT_REVIEW_TABS}
      tickerInteractive
      activeTicker={activeTab}
      onTickerSelect={onTickerSelect}
      persistRevealKey="draftreview-revealed"
      footerAlign="split"
      onContinue={onContinue}
      onBack={onBack}
      railTitle="Your Haul"
      railHint={picks.length ? "↑↓ to switch picks" : null}
      heroContent={
        pick ? (
          <div className={`${prefix}-workspace`}>
            <div className={`${prefix}-broadcast-strip`} aria-hidden="true">
              <span className={`${prefix}-broadcast-live`}>On air</span>
              <span className={`${prefix}-broadcast-show`}>Draft Class Review</span>
              <span className={`${prefix}-broadcast-seg`}>
                {safeArray(raw.user_picks).length || haul.total_picks || "—"} picks
              </span>
            </div>
            <div className={`${prefix}-haul-banner`} ref={haulRef} aria-label="Draft class verdict">
              <div
                className={`${prefix}-haul-banner-grade tone-${gradeTone(gradeLetter(haulGrade))}`}
                aria-label={haulGrade ? `Overall draft grade ${haulGrade}` : "Draft grade unavailable"}
              >
                <strong>{haulGrade || "—"}</strong>
                <span>Class</span>
              </div>
              <div className={`${prefix}-haul-banner-copy`}>
                <p className={`${prefix}-kicker`} style={{ margin: 0 }}>
                  Studio verdict · post-draft desk
                </p>
                {haulLabel ? <strong>{haulLabel}</strong> : null}
                {haulExplain ? <p>{haulExplain}</p> : null}
                {classChips.length ? (
                  <div className={`${prefix}-haul-chips`} aria-label="Class composition">
                    {classChips.slice(0, 4).map((chip) => (
                      <span key={chip} className={`${prefix}-haul-chip`}>
                        {chip}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
              {haulMetrics.length ? (
                <div className={`${prefix}-haul-metrics`}>
                  {haulMetrics.slice(0, 4).map((metric) => (
                    <div key={metric.key} className={`${prefix}-haul-metric-card`} title={metric.value}>
                      <span>{metric.label}</span>
                      <strong>{metric.value}</strong>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            <div className={`${prefix}-review-main`}>
              <section className={`${prefix}-draft-card`} ref={fitRef} aria-label="Selected prospect">
                <div className={`${prefix}-pos-mark pos-${posGroup}`} aria-hidden={!pos}>
                  <strong>{pos || "—"}</strong>
                  <span>{pick.round != null ? `R${pick.round}` : "Pick"}</span>
                </div>
                <div className={`${prefix}-draft-card-body`}>
                  <p className={`${prefix}-hero-kicker`}>
                    {[
                      pick.overall_pick != null ? `Overall #${pick.overall_pick}` : null,
                      pick.round != null ? `Round ${pick.round}` : null,
                      pick.nationality ? String(pick.nationality) : null,
                    ]
                      .filter(Boolean)
                      .join(" · ") || "Draft pick"}
                  </p>
                  <h2 className={`${prefix}-hero-name`}>
                    {pick.prospect_name || getPlayerName(pick)}
                  </h2>
                  <p className={`${prefix}-hero-sub`}>
                    {[
                      humanizeLabel(pick.archetype || pick.player_type),
                      pick.league,
                      pick.club,
                      pick.age != null ? `Age ${pick.age}` : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                  <p className={`${prefix}-hero-meta`}>
                    {[
                      pick.height,
                      pick.weight != null ? `${pick.weight} lb` : null,
                      pick.shoots ? `Shoots ${pick.shoots}` : null,
                      pick.board_range ? `Board ${pick.board_range}` : null,
                      pick.selection_delta_label,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                  {pick.selection_verdict || pick.selection_grade_label ? (
                    <div className={`${prefix}-value-banner tone-${valueTone}`}>
                      {pick.selection_verdict || pick.selection_grade_label}
                    </div>
                  ) : null}
                </div>
                {pickGrade || pick.selection_grade_label ? (
                  <div
                    className={`${prefix}-grade-shield tone-${gradeTone(pickGrade)}`}
                    aria-label={
                      pickGrade
                        ? `Selection grade ${pickGrade}${
                            pick.selection_grade_label ? `, ${pick.selection_grade_label}` : ""
                          }`
                        : pick.selection_grade_label
                    }
                  >
                    <strong>{pickGrade || "—"}</strong>
                    {pick.selection_grade_label ? <span>{pick.selection_grade_label}</span> : null}
                  </div>
                ) : null}
              </section>

              <div className={`${prefix}-stage-grid`}>
                <section className={`${prefix}-projection-card`} aria-label="NHL projection">
                  <p className={`${prefix}-section-label`}>Why he matters</p>
                  {plan.nhl_projection || fit.fit_label || pick.selection_verdict ? (
                    <p className={`${prefix}-projection-role`}>
                      {plan.nhl_projection || fit.fit_label || pick.selection_verdict}
                    </p>
                  ) : null}
                  {plan.eta_range ? (
                    <p className={`${prefix}-projection-eta`}>Ready in {plan.eta_range}</p>
                  ) : null}
                  <div className={`${prefix}-meter-stack`}>
                    <MeterRow
                      prefix={prefix}
                      label="Projection confidence"
                      value={plan.nhl_projection_confidence}
                    />
                    <MeterRow
                      prefix={prefix}
                      label="Development risk"
                      value={pick.risk_level}
                    />
                    <MeterRow
                      prefix={prefix}
                      label="Scouting confidence"
                      value={pick.scouting_confidence_label}
                    />
                    <MeterRow
                      prefix={prefix}
                      label="Timeline confidence"
                      value={plan.eta_confidence}
                    />
                  </div>
                </section>

                <section className={`${prefix}-prod-card`} aria-label="Production">
                  <p className={`${prefix}-section-label`}>
                    {production.mode === "scouting" ? "Scouting read" : "Junior / league production"}
                  </p>
                  {ppgStat ? (
                    <div className={`${prefix}-prod-hero`}>
                      <strong>{formatProductionStat(ppgStat[0], ppgStat[1])}</strong>
                      <span>Points per game</span>
                    </div>
                  ) : production.headline ? (
                    <p className={`${prefix}-projection-role`}>{production.headline}</p>
                  ) : null}
                  {otherStats.length ? (
                    <div className={`${prefix}-prod-stats`}>
                      {otherStats.slice(0, 4).map(([label, value]) => (
                        <p
                          key={label}
                          className={`${prefix}-prod-stat${label === "PPG" ? " is-ppg" : ""}`}
                        >
                          <span>{label}</span>
                          <strong>{formatProductionStat(label, value)}</strong>
                        </p>
                      ))}
                    </div>
                  ) : null}
                  {prodContext ? <p className={`${prefix}-prod-context`}>{prodContext}</p> : null}
                </section>

                {(nextHero || pathStages.length) ? (
                  <section
                    className={`${prefix}-roadmap-card`}
                    ref={planRef}
                    aria-label="Development roadmap"
                  >
                    <p className={`${prefix}-section-label`}>
                      Development roadmap
                      {plan.recommended_role ? ` · ${plan.recommended_role}` : ""}
                      {[plan.minutes_target, plan.special_teams_role].filter(Boolean).length
                        ? ` · ${[plan.minutes_target, plan.special_teams_role].filter(Boolean).join(" · ")}`
                        : ""}
                    </p>
                    {pathStages.length ? (
                      <div className={`${prefix}-roadmap`}>
                        {pathStages.map((step, i) => (
                          <div key={step.key} className={`${prefix}-road-step is-${step.status}`}>
                            <span className={`${prefix}-road-step-label`}>
                              {step.status === "next"
                                ? "Now"
                                : step.status === "projection"
                                  ? "NHL goal"
                                  : "Next"}
                            </span>
                            <span className={`${prefix}-road-step-title`}>{step.label}</span>
                            {step.detail ? (
                              <span className={`${prefix}-road-step-detail`}>{step.detail}</span>
                            ) : null}
                            {i === pathStages.length - 1 && plan.eta_range ? (
                              <span className={`${prefix}-road-step-eta`}>ETA {plan.eta_range}</span>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    ) : nextHero ? (
                      <p className={`${prefix}-projection-role`}>Next: {nextHero}</p>
                    ) : null}
                  </section>
                ) : null}

                {(pros.length ||
                  cons.length ||
                  neutralNotes.length ||
                  hasOrgPlan ||
                  pick.risk_reason) ? (
                  <section
                    className={`${prefix}-bottom-band`}
                    ref={rightsRef}
                    aria-label="Scouting and organization"
                  >
                    {(pros.length || neutralNotes.length > 0) ? (
                      <div className={`${prefix}-scout-col is-pro`}>
                        <h3>Biggest strengths</h3>
                        {(pros.length ? pros : neutralNotes.slice(0, 2)).map((item) => (
                          <p key={`pro-${item.text}`} className={`${prefix}-scout-item`}>
                            {item.text}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <div />
                    )}
                    {(cons.length || pick.risk_reason || neutralNotes.length > 2) ? (
                      <div className={`${prefix}-scout-col is-con`}>
                        <h3>Biggest concerns</h3>
                        {cons.length
                          ? cons.map((item) => (
                              <p key={`con-${item.text}`} className={`${prefix}-scout-item`}>
                                {item.text}
                              </p>
                            ))
                          : null}
                        {!cons.length && pick.risk_reason ? (
                          <p className={`${prefix}-scout-item`}>{pick.risk_reason}</p>
                        ) : null}
                        {!cons.length && !pick.risk_reason
                          ? neutralNotes.slice(2, 4).map((item) => (
                              <p key={`con-${item.text}`} className={`${prefix}-scout-item`}>
                                {item.text}
                              </p>
                            ))
                          : null}
                      </div>
                    ) : (
                      <div />
                    )}
                    {hasOrgPlan ? (
                      <div className={`${prefix}-org-col`}>
                        <p className={`${prefix}-section-label`}>Organization fit</p>
                        <div className={`${prefix}-org-top`}>
                          {fit.need_filled ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Need filled</span>
                              <strong>{fit.need_filled}</strong>
                            </p>
                          ) : null}
                          {fit.fit_label || fit.fit_grade ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Fit</span>
                              <strong>
                                {[fit.fit_grade, fit.fit_label].filter(Boolean).join(" · ")}
                              </strong>
                            </p>
                          ) : null}
                          {fit.depth_status || fit.path_congestion ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Opportunity</span>
                              <strong>{fit.depth_status || fit.path_congestion}</strong>
                            </p>
                          ) : null}
                          {fit.environment_grade ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Environment</span>
                              <strong>{fit.environment_grade}</strong>
                            </p>
                          ) : null}
                          {rights.rights_status_label || rights.signing_recommendation ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Rights</span>
                              <strong>
                                {[rights.rights_status_label, rights.signing_recommendation]
                                  .filter(Boolean)
                                  .join(" · ")}
                              </strong>
                            </p>
                          ) : null}
                          {rights.rights_deadline_label ? (
                            <p className={`${prefix}-org-kv`}>
                              <span>Window</span>
                              <strong>{rights.rights_deadline_label}</strong>
                            </p>
                          ) : null}
                        </div>
                        {hasDepth ? (
                          <div className={`${prefix}-depth-bars`} aria-label="Players ahead at position">
                            {[
                              { key: "nhl", label: "NHL ahead", value: fit.nhl_players_ahead },
                              { key: "ahl", label: "AHL ahead", value: fit.ahl_players_ahead },
                              { key: "pro", label: "Prospects", value: fit.prospects_ahead },
                            ]
                              .filter((row) => row.value != null)
                              .map((row) => (
                                <div key={row.key} className={`${prefix}-depth-bar-row`}>
                                  <span>{row.label}</span>
                                  <div className={`${prefix}-depth-track`}>
                                    <span
                                      className={`${prefix}-depth-fill`}
                                      style={{
                                        width: `${Math.max(
                                          8,
                                          Math.round((Number(row.value) / depthMax) * 100)
                                        )}%`,
                                      }}
                                    />
                                  </div>
                                  <em>{row.value}</em>
                                </div>
                              ))}
                          </div>
                        ) : null}
                        {orgNote ? <p className={`${prefix}-org-note`}>{orgNote}</p> : null}
                      </div>
                    ) : (
                      <div />
                    )}
                  </section>
                ) : (
                  <div ref={rightsRef} />
                )}
              </div>

              {advancedLines.length ? (
                <div className={`${prefix}-more-details`}>
                  <button
                    type="button"
                    className={`${prefix}-more-toggle`}
                    aria-expanded={showMore}
                    onClick={() => setShowMore((v) => !v)}
                  >
                    {showMore ? "Hide scouting detail" : "More scouting detail"}
                  </button>
                  {showMore ? (
                    <div className={`${prefix}-more-panel`}>
                      {[oneLineVerdict, developReason, ...advancedLines]
                        .filter(Boolean)
                        .filter((line, i, arr) => arr.indexOf(line) === i)
                        .slice(0, 8)
                        .map((line) => (
                          <p key={line}>{line}</p>
                        ))}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ) : (
          <p className={`${prefix}-empty`}>{raw.headline || "No picks recorded for your club"}</p>
        )
      }
      railContent={
        picks.length ? (
          picks.map((p) => {
            const id = String(p.prospect_id || "");
            const active = id && id === String(selectedId);
            const letter = gradeLetter(p.selection_grade);
            const pPos = p.position || getPlayerPosition(p);
            const rankLabel = [
              p.round != null ? `R${p.round}` : null,
              p.overall_pick != null ? `#${p.overall_pick}` : null,
            ]
              .filter(Boolean)
              .join(" ");
            const lineTwo = [humanizeLabel(p.archetype || p.player_type) || p.league, p.selection_verdict]
              .filter(Boolean)
              .join(" · ");
            const hoverPreview = [
              p.prospect_name || getPlayerName(p),
              letter ? `Grade ${letter}` : null,
              p.selection_verdict || null,
            ]
              .filter(Boolean)
              .join(" · ");
            return (
              <button
                key={id || p.overall_pick}
                type="button"
                className={`${prefix}-rail-btn`}
                ref={(node) => {
                  if (id) railItemRefs.current[id] = node;
                }}
                onClick={() => setSelectedId(id)}
                onKeyDown={(e) => onRailKeyDown(e, id)}
                aria-pressed={active}
                title={hoverPreview}
              >
                <article className={`${prefix}-haul-card${active ? " is-active" : ""}`}>
                  <div className={`${prefix}-haul-pos pos-${positionGroup(pPos)}`}>{pPos || "—"}</div>
                  <div className={`${prefix}-haul-rank`}>{rankLabel || "—"}</div>
                  <div className={`${prefix}-haul-body`}>
                    <strong>{p.prospect_name || getPlayerName(p)}</strong>
                    {lineTwo ? (
                      <div className={`${prefix}-card-details`}>
                        <span>{lineTwo}</span>
                      </div>
                    ) : null}
                  </div>
                  {letter ? (
                    <div
                      className={`${prefix}-haul-grade-letter tone-${gradeTone(letter)}`}
                      aria-label={`Grade ${letter}`}
                    >
                      {letter}
                    </div>
                  ) : null}
                </article>
              </button>
            );
          })
        ) : (
          <p className={`${prefix}-empty`}>No picks recorded for your club</p>
        )
      }
    />
  );
}

const PROSPECT_RIGHTS_TABS = [];

function parseContractSlots(payload) {
  const used =
    payload?.contract_slots_used != null ? Number(payload.contract_slots_used) : null;
  const limit =
    payload?.contract_slots_limit != null ? Number(payload.contract_slots_limit) : 50;
  if (used != null && !Number.isNaN(used)) {
    return { used, limit: Number.isNaN(limit) ? 50 : limit };
  }
  const raw = String(payload?.contracts || "");
  const m = raw.match(/(\d+)\s*\/\s*(\d+)/);
  if (m) return { used: Number(m[1]), limit: Number(m[2]) };
  return { used: null, limit: 50 };
}

function prospectPriorityMeta(p, { priorityIds, seasonYear }) {
  const through = p?.rights_through != null ? Number(p.rights_through) : null;
  const path = String(p?.returning_to || p?.development_environment?.path || "").toUpperCase();
  const eta = p?.eta != null ? Number(p.eta) : null;
  const rec = p?.recommended_action;
  const isPriority = priorityIds.has(String(p?.player_id || ""));
  const nearExpiry =
    through != null && seasonYear != null && through <= Number(seasonYear) + 1;
  if (isPriority || nearExpiry || rec === "sign_elc") {
    return { label: "Sign now", tone: "hot", rank: 5 };
  }
  if (rec === "allow_expire") return { label: "Rights expire soon", tone: "hot", rank: 4 };
  if (path.includes("EUROPE") || rec === "keep_europe") {
    return { label: "Leave in Europe", tone: "calm", rank: 2 };
  }
  if (path.includes("NCAA") || rec === "keep_college") {
    return { label: "NCAA path", tone: "calm", rank: 2 };
  }
  if (rec === "return_junior") return { label: "Return to junior", tone: "calm", rank: 2 };
  if (rec === "keep_unsigned" || rec === "delay") {
    return { label: eta != null && eta >= 4 ? "Defer" : "Defer", tone: "soft", rank: 1 };
  }
  if (eta != null && eta >= 4) return { label: "Long-term project", tone: "soft", rank: 1 };
  return { label: "Review", tone: "soft", rank: 0 };
}

function buildRightsTimeline(focus, seasonYear) {
  const steps = [];
  const draftYear = focus?.draft_year != null ? Number(focus.draft_year) : null;
  if (draftYear) steps.push({ key: "draft", label: "Drafted", value: String(draftYear), tone: "now" });
  safeArray(focus?.path_visual)
    .slice(0, 3)
    .forEach((step, i) => {
      steps.push({
        key: `path-${i}`,
        label: String(step),
        value: i === 0 && seasonYear != null ? String(seasonYear) : "Next",
        tone: i === 0 ? "now" : "future",
      });
    });
  if (focus?.rights_signing_deadline) {
    steps.push({
      key: "deadline",
      label: "Sign deadline",
      value: String(focus.rights_signing_deadline),
      tone: "deadline",
    });
  } else if (focus?.rights_through != null) {
    steps.push({
      key: "expire",
      label: "Rights expire",
      value: String(focus.rights_through),
      tone: "deadline",
    });
  }
  return steps.slice(0, 6);
}

function buildDecisionPackages(focus) {
  const actions = safeArray(focus?.available_actions);
  const byId = Object.fromEntries(actions.map((a) => [a.id, a]));
  const templates = safeArray(focus?.offer_templates);
  const packages = [];

  if (templates.length) {
    templates.forEach((t) => {
      packages.push({
        packageId: `elc_${t.template_id}`,
        kind: "elc",
        section: "sign",
        actionId: "sign_elc",
        templateId: t.template_id,
        title: t.label,
        blurb: t.summary,
        termYears: t.term_years,
        aavDisplay: t.aav_display,
        signingBonusDisplay: t.signing_bonus_display,
        scheduleA: t.schedule_a_display,
        scheduleB: t.schedule_b_display,
        slide: t.slide_eligible,
        action: byId.sign_elc,
        recommended: t.template_id === "standard_elc" && focus?.recommended_action === "sign_elc",
        disabled: byId.sign_elc?.enabled === false,
      });
    });
  } else if (byId.sign_elc) {
    packages.push({
      packageId: "elc_standard",
      kind: "elc",
      section: "sign",
      actionId: "sign_elc",
      templateId: "standard_elc",
      title: byId.sign_elc.label || "Offer ELC",
      blurb: byId.sign_elc.summary || "Entry-level contract",
      action: byId.sign_elc,
      recommended: focus?.recommended_action === "sign_elc",
      disabled: byId.sign_elc.enabled === false,
    });
  }

  const development = [
    "return_junior",
    "keep_college",
    "keep_europe",
    "assign_ahl",
    "invite_camp",
  ];
  development.forEach((id) => {
    const action = byId[id];
    if (!action) return;
    packages.push({
      packageId: `path_${id}`,
      kind: "path",
      section: "development",
      actionId: id,
      title: action.label || humanizeLabel(id),
      blurb: action.summary || action.blocked_reason || action.warning || "",
      action,
      recommended: action.id === focus?.recommended_action,
      disabled: action.enabled === false,
    });
  });

  ["keep_unsigned", "delay", "allow_expire"].forEach((id) => {
    const action = byId[id];
    if (!action) return;
    packages.push({
      packageId: `path_${id}`,
      kind: "path",
      section: "rights",
      actionId: id,
      title: action.label || humanizeLabel(id),
      blurb: action.summary || action.warning || "",
      action,
      recommended: action.id === focus?.recommended_action,
      disabled: action.enabled === false,
    });
  });

  return packages;
}

export function ProspectRightsEventMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const prefix = "prospectrights";
  const initial =
    pickFranchiseData(franchiseState, eventData, ["prospect_rights", "offseason.prospect_rights"]) ||
    {};
  const [payload, setPayload] = React.useState(initial);
  const [selectedId, setSelectedId] = React.useState(() =>
    safeArray(initial.prospects)[0]?.player_id != null
      ? String(safeArray(initial.prospects)[0].player_id)
      : null
  );
  const [selectedPackageId, setSelectedPackageId] = React.useState(null);
  const [preview, setPreview] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [feedback, setFeedback] = React.useState("");
  const [dealPopup, setDealPopup] = React.useState(null);
  const railItemRefs = React.useRef({});

  const prospects = safeArray(payload?.prospects);
  const priority = safeArray(payload?.recommended_signing_priority);
  const warnings = safeArray(payload?.warning_reasons);
  const seasonYear = payload?.season_year || franchiseState?.season_year || franchiseState?.seasonYear;
  const slots = parseContractSlots(payload);
  const priorityIds = React.useMemo(
    () => new Set(priority.map((p) => String(p?.player_id || "")).filter(Boolean)),
    [priority]
  );

  const sortedProspects = React.useMemo(() => {
    return [...prospects].sort((a, b) => {
      const ma = prospectPriorityMeta(a, { priorityIds, seasonYear });
      const mb = prospectPriorityMeta(b, { priorityIds, seasonYear });
      if (mb.rank !== ma.rank) return mb.rank - ma.rank;
      return (Number(a?.eta) || 99) - (Number(b?.eta) || 99);
    });
  }, [prospects, priorityIds, seasonYear]);

  React.useEffect(() => {
    if (!prospects.length) {
      setSelectedId(null);
      return;
    }
    if (!prospects.some((p) => String(p?.player_id) === String(selectedId))) {
      setSelectedId(prospects[0]?.player_id != null ? String(prospects[0].player_id) : null);
    }
  }, [prospects, selectedId]);

  const focus =
    prospects.find((p) => String(p?.player_id) === String(selectedId)) || prospects[0] || null;
  const env = focus?.development_environment || {};
  const packages = React.useMemo(() => buildDecisionPackages(focus), [focus]);
  const selectedPackage =
    packages.find((p) => p.packageId === selectedPackageId) ||
    packages.find((p) => p.recommended && !p.disabled) ||
    packages.find((p) => !p.disabled) ||
    null;
  const selectedAction = selectedPackage?.action || null;
  const isSigning = selectedPackage?.kind === "elc";
  const [assignmentPlanOverride, setAssignmentPlanOverride] = React.useState(null);
  const assignmentOptions = safeArray(focus?.legal_elc_terms?.assignment_options);
  const defaultAssignmentPlanId = assignmentOptions.find((o) => o.enabled)?.id || null;
  const effectiveAssignmentPlan = assignmentOptions.some(
    (o) => o.id === assignmentPlanOverride && o.enabled
  )
    ? assignmentPlanOverride
    : defaultAssignmentPlanId;

  React.useEffect(() => {
    setError("");
    setFeedback("");
    setPreview(null);
    setAssignmentPlanOverride(null);
    if (!focus) {
      setSelectedPackageId(null);
      return;
    }
    const pkgs = buildDecisionPackages(focus);
    setSelectedPackageId(
      pkgs.find((p) => p.recommended && !p.disabled)?.packageId ||
        pkgs.find((p) => !p.disabled)?.packageId ||
        null
    );
  }, [focus?.player_id]); // eslint-disable-line react-hooks/exhaustive-deps

  React.useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!focus?.player_id || !selectedPackage || selectedPackage.kind !== "elc") {
        setPreview(null);
        return;
      }
      try {
        const result = await previewElcOffer({
          player_id: focus.player_id,
          template_id: selectedPackage.templateId || "standard_elc",
          assignment_plan: effectiveAssignmentPlan,
        });
        if (!cancelled) setPreview(result);
      } catch (err) {
        if (!cancelled) setPreview(null);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [focus?.player_id, selectedPackage?.packageId, selectedPackage?.templateId, effectiveAssignmentPlan]);

  const offer = preview?.offer || null;
  const acceptance = preview?.acceptance || focus?.elc_acceptance_summary || null;
  const signingResult = preview?.signing_result || null;
  const slotPreview = preview?.slot_preview || null;
  const pros = safeArray(selectedAction?.pros || acceptance?.positives);
  const cons = safeArray(selectedAction?.cons || acceptance?.concerns);
  const agentWants = safeArray(acceptance?.agent_wants);
  const timeline = focus ? buildRightsTimeline(focus, seasonYear) : [];
  const ovr = focus?.overall != null ? focus.overall : getPlayerOverall(focus);
  const pot =
    focus?.potential != null
      ? focus.potential
      : focus?.potential_score != null
        ? focus.potential_score
        : focus?.pot != null
          ? focus.pot
          : null;
  const urgentCount = priority.length;

  const onRailKeyDown = (e, id) => {
    if (!sortedProspects.length) return;
    const ids = sortedProspects.map((p) => String(p.player_id));
    const idx = Math.max(0, ids.indexOf(String(selectedId)));
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      setSelectedId(ids[Math.min(ids.length - 1, idx + 1)]);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      setSelectedId(ids[Math.max(0, idx - 1)]);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedId(String(id));
    }
  };

  const applyDecision = async () => {
    if (!focus?.player_id || !selectedPackage?.actionId || selectedPackage.disabled || busy) return;
    setBusy(true);
    setError("");
    setFeedback("");
    try {
      let result;
      if (selectedPackage.kind === "elc") {
        result = await submitElcOffer({
          player_id: focus.player_id,
          template_id: selectedPackage.templateId || "standard_elc",
          assignment_plan: offer?.assignment_plan,
          development_promise: offer?.development_promise,
          term_years: offer?.term_years,
          idempotency_key: offer?.offer_id,
        });
      } else {
        result = await prospectRightsDecision({
          player_id: focus.player_id,
          action_id: selectedPackage.actionId,
        });
      }
      if (result?.prospect_rights) setPayload(result.prospect_rights);
      const playerName = focus.name || getPlayerName(focus);
      if (result?.ok && (result?.signed || selectedPackage.kind !== "elc")) {
        const assignmentFailed = selectedPackage.kind === "elc" && result?.assignment_ok === false;
        setFeedback(
          selectedPackage.kind === "elc"
            ? assignmentFailed
              ? `${playerName} signed, but the requested assignment failed: ${result?.assignment?.reason || "see roster."}`
              : `${playerName} signed — ${selectedPackage.title}.`
            : `Recorded: ${selectedPackage.title}.`
        );
        if (selectedPackage.kind === "elc" || result?.signed) {
          setDealPopup({
            tone: "accept",
            kicker: "Prospect re-sign",
            title: result?.signed || selectedPackage.kind === "elc" ? "Deal accepted" : "Decision recorded",
            player: playerName,
            body:
              selectedPackage.kind === "elc"
                ? `${playerName} signed the ${selectedPackage.title}.`
                : `Recorded: ${selectedPackage.title}.`,
            terms: offer?.aav_display || null,
            cta: "Continue",
          });
        } else {
          setDealPopup({
            tone: "accept",
            kicker: "Prospect rights",
            title: "Decision recorded",
            player: playerName,
            body: `Recorded: ${selectedPackage.title}.`,
            cta: "Continue",
          });
        }
      } else if (!result?.ok) {
        const declineReasons = safeArray(result?.decision?.reasons || result?.acceptance?.reasons);
        const counter = result?.counter_offer?.message;
        const denyBody =
          result?.reason === "prospect_declined"
            ? ["Prospect declined.", counter, ...declineReasons.slice(0, 2)].filter(Boolean).join(" ")
            : result?.validation?.user_message ||
              result?.reason ||
              result?.user_message ||
              result?.message ||
              "Decision failed";
        setError(denyBody);
        setDealPopup({
          tone: "deny",
          kicker: "Prospect re-sign",
          title: result?.reason === "prospect_declined" ? "Deal declined" : "Could not complete",
          player: playerName,
          body: denyBody,
          cta: "Close",
        });
      }
    } catch (err) {
      setError(err?.message || "Could not apply rights decision");
    } finally {
      setBusy(false);
    }
  };

  const applyLabel = isSigning
    ? busy
      ? "Submitting…"
      : `Sign ${selectedPackage?.title || "ELC"}`
    : busy
      ? "Applying…"
      : `Confirm: ${selectedPackage?.title || "Decision"}`;

  const sectionPackages = (section) => packages.filter((p) => p.section === section);

  return (
    <>
    <DealOutcomeModal prefix={prefix} deal={dealPopup} onClose={() => setDealPopup(null)} />
    <CinematicEventShell
      prefix={prefix}
      phaseLabel="RIGHTS DESK"
      phaseStyle="text"
      titleVariant="rights"
      seasonLabel={seasonLabel(franchiseState)}
      title="Prospect Rights"
      hideTitle
      hideEyebrow
      ctaLabel="Continue to Re-Sign"
      tickerItems={PROSPECT_RIGHTS_TABS}
      persistRevealKey="prospectrights-revealed"
      footerAlign="split"
      onContinue={onContinue}
      onBack={onBack}
      railTitle="Unsigned Prospects"
      railHint={prospects.length ? "↑↓ switch" : null}
      heroContent={
        focus ? (
          <div className={`${prefix}-workspace`}>
            {warnings[0] ? (
              <p className={`${prefix}-alert-banner`} role="status">
                {warnings[0]}
              </p>
            ) : null}

            <div className={`${prefix}-impact-strip`} aria-label="Organization impact">
              <div className={`${prefix}-impact-card`}>
                <span>Contracts</span>
                <strong>
                  {slotPreview
                    ? `${slotPreview.contract_slots_used}/${slotPreview.contract_slots_limit}`
                    : slots.used != null
                      ? `${slots.used}/${slots.limit}`
                      : payload?.contracts || "—"}
                </strong>
                {isSigning && slotPreview ? (
                  <em>After {slotPreview.after_signing}/{slotPreview.contract_slots_limit}</em>
                ) : null}
              </div>
              <div className={`${prefix}-impact-card${isSigning ? " is-warn" : ""}`}>
                <span>Cap</span>
                <strong>
                  {isSigning
                    ? preview?.signing_result?.cap_impact_display ||
                      offer?.aav_display ||
                      "+$950K"
                    : "—"}
                </strong>
              </div>
              <div className={`${prefix}-impact-card`}>
                <span>Reserve</span>
                <strong>{payload?.reserve_rights ?? prospects.length}</strong>
              </div>
              <div className={`${prefix}-impact-card`}>
                <span>Urgent</span>
                <strong>{urgentCount}</strong>
              </div>
              <div className={`${prefix}-impact-card`}>
                <span>Acceptance</span>
                <strong>
                  {isSigning && acceptance?.acceptance_pct != null
                    ? `${acceptance.acceptance_pct}%`
                    : isSigning
                      ? acceptance?.outlook_label || "—"
                      : "N/A"}
                </strong>
              </div>
            </div>

            <div className={`${prefix}-nego-grid`}>
              <div className={`${prefix}-nego-col`}>
                <article className={`${prefix}-player-card`}>
                  <div className={`${prefix}-player-card-top`}>
                    <div className={`${prefix}-pos-badge`}>
                      <strong>{focus.position || getPlayerPosition(focus) || "—"}</strong>
                      <span>{ovr != null ? ovr : "OVR"}</span>
                    </div>
                    <div>
                      <p className={`${prefix}-kicker`}>
                        {[
                          focus.age != null ? `Age ${focus.age}` : null,
                          focus.draft_overall_pick != null ? `Pick ${focus.draft_overall_pick}` : null,
                          focus.eta != null ? `ETA ${focus.eta}y` : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                      <h2>{focus.name || getPlayerName(focus)}</h2>
                    </div>
                  </div>
                  <div className={`${prefix}-stat-grid`}>
                    <div className={`${prefix}-stat-cell`}>
                      <span>Age</span>
                      <strong>{focus.age != null ? focus.age : "—"}</strong>
                    </div>
                    {ovr != null ? (
                      <div className={`${prefix}-stat-cell`}>
                        <span>OVR</span>
                        <strong>{ovr}</strong>
                      </div>
                    ) : (
                      <div className={`${prefix}-stat-cell`}>
                        <span>OVR</span>
                        <strong>—</strong>
                      </div>
                    )}
                    {pot != null ? (
                      <div className={`${prefix}-stat-cell`}>
                        <span>POT</span>
                        <strong>{pot}</strong>
                      </div>
                    ) : null}
                    {focus.expected_role ? (
                      <div className={`${prefix}-stat-cell`}>
                        <span>Role</span>
                        <strong title={focus.expected_role}>{focus.expected_role}</strong>
                      </div>
                    ) : null}
                    {focus.eta != null ? (
                      <div className={`${prefix}-stat-cell`}>
                        <span>ETA</span>
                        <strong>{focus.eta}y</strong>
                      </div>
                    ) : null}
                    {(focus.current_league_id || focus.returning_to) && (
                      <div className={`${prefix}-stat-cell`}>
                        <span>League</span>
                        <strong>
                          {String(focus.current_league_id || focus.returning_to)
                            .replace(/^CHL_/, "")
                            .slice(0, 10)}
                        </strong>
                      </div>
                    )}
                    {env.grade ? (
                      <div className={`${prefix}-stat-cell`}>
                        <span>Env</span>
                        <strong>{humanizeLabel(env.grade)}</strong>
                      </div>
                    ) : null}
                    <div className={`${prefix}-stat-cell`}>
                      <span>Slide</span>
                      <strong>
                        {focus.legal_elc_terms?.slide_eligible ?? focus.elc_slide_eligible
                          ? "Yes"
                          : "No"}
                      </strong>
                    </div>
                  </div>
                </article>

                <div className={`${prefix}-reasons`}>
                  <div className={`${prefix}-reason-col is-sign`}>
                    <h4>Sign</h4>
                    <ul>
                      {(pros.length ? pros : ["Locks NHL rights"]).slice(0, 4).map((item) => (
                        <li key={`pro-${item}`}>✓ {item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className={`${prefix}-reason-col is-wait`}>
                    <h4>Wait</h4>
                    <ul>
                      {(cons.length ? cons : ["Uses a contract slot"]).slice(0, 4).map((item) => (
                        <li key={`con-${item}`}>! {item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              <div className={`${prefix}-nego-col`}>
                <div className={`${prefix}-contract-doc`}>
                  <div className={`${prefix}-contract-doc-head`}>
                    <div>
                      <h3>Entry Level Contract</h3>
                      <p>{isSigning ? selectedPackage?.title : "Path decision — no ELC"}</p>
                    </div>
                    {acceptance?.outlook_label ? (
                      <span className={`${prefix}-offer-tag`}>{acceptance.outlook_label}</span>
                    ) : null}
                  </div>
                  <div className={`${prefix}-contract-hero`}>
                    {isSigning ? (
                      <>
                        <strong>{offer?.term_years || selectedPackage?.termYears || 3} Years</strong>
                        <span>{offer?.aav_display || selectedPackage?.aavDisplay || "$950,000"}</span>
                      </>
                    ) : (
                      <>
                        <strong>Hold rights</strong>
                        <span>No slot burn</span>
                      </>
                    )}
                  </div>
                  {isSigning ? (
                    <div className={`${prefix}-contract-rows`}>
                      {assignmentOptions.length ? (
                        <div className={`${prefix}-contract-row`}>
                          <span>Assignment</span>
                          <select
                            className={`${prefix}-assignment-select`}
                            value={effectiveAssignmentPlan || ""}
                            onChange={(e) => setAssignmentPlanOverride(e.target.value)}
                            disabled={busy}
                          >
                            {assignmentOptions.map((opt) => (
                              <option key={opt.id} value={opt.id} disabled={!opt.enabled}>
                                {opt.label}
                                {!opt.enabled && opt.blocked_reason ? ` (${opt.blocked_reason})` : ""}
                              </option>
                            ))}
                          </select>
                        </div>
                      ) : null}
                      <div className={`${prefix}-contract-row`}>
                        <span>Signing bonus</span>
                        <strong>
                          {offer?.signing_bonus_display ||
                            selectedPackage?.signingBonusDisplay ||
                            "—"}
                        </strong>
                      </div>
                      <div className={`${prefix}-contract-row`}>
                        <span>Schedule A</span>
                        <strong>{offer?.schedule_a_display || selectedPackage?.scheduleA || "—"}</strong>
                      </div>
                      <div className={`${prefix}-contract-row`}>
                        <span>Schedule B</span>
                        <strong>{offer?.schedule_b_display || selectedPackage?.scheduleB || "None"}</strong>
                      </div>
                      <div className={`${prefix}-contract-row`}>
                        <span>Cap hit</span>
                        <strong>{offer?.aav_display || "$950K"}</strong>
                      </div>
                      <div className={`${prefix}-contract-row`}>
                        <span>Slide</span>
                        <strong>
                          {(offer?.slide_eligible ?? selectedPackage?.slide) ? "Eligible" : "No"}
                        </strong>
                      </div>
                      <div className={`${prefix}-contract-row`}>
                        <span>Slot</span>
                        <strong>Yes (+1)</strong>
                      </div>
                    </div>
                  ) : null}

                  {isSigning && acceptance?.acceptance_pct != null ? (
                    <div className={`${prefix}-accept-meter`} aria-label="Acceptance chance">
                      <div className={`${prefix}-accept-meter-head`}>
                        <span>Acceptance</span>
                        <strong>{acceptance.acceptance_pct}%</strong>
                      </div>
                      <div className={`${prefix}-meter`}>
                        <span style={{ width: `${Math.max(4, acceptance.acceptance_pct)}%` }} />
                      </div>
                      {agentWants.length ? (
                        <ul className={`${prefix}-agent-wants`}>
                          {agentWants.slice(0, 4).map((w) => (
                            <li key={w.id || w.label}>✓ {w.label}</li>
                          ))}
                        </ul>
                      ) : null}
                      {acceptance.main_concern ? (
                        <p className={`${prefix}-context`}>Concern: {acceptance.main_concern}</p>
                      ) : null}
                    </div>
                  ) : null}
                </div>

                {["sign", "development", "rights"].map((section) => {
                  const rows = sectionPackages(section);
                  if (!rows.length) return null;
                  const label =
                    section === "sign"
                      ? "Sign contract"
                      : section === "development"
                        ? "Development"
                        : "Rights";
                  return (
                    <React.Fragment key={section}>
                      <p className={`${prefix}-offer-label`}>{label}</p>
                      <div className={`${prefix}-offer-scroll`} style={{ flex: "0 0 auto", maxHeight: section === "sign" ? "9.5rem" : "6.5rem" }}>
                        {rows.map((pkg) => {
                          const selected =
                            selectedPackage && pkg.packageId === selectedPackage.packageId;
                          return (
                            <button
                              key={pkg.packageId}
                              type="button"
                              disabled={Boolean(pkg.disabled) || busy}
                              className={`${prefix}-offer-btn${selected ? " is-selected" : ""}${
                                pkg.recommended ? " is-recommended" : ""
                              }`}
                              title={pkg.disabled ? pkg.action?.blocked_reason || pkg.blurb : pkg.blurb}
                              onClick={() => {
                                if (pkg.disabled) return;
                                setSelectedPackageId(pkg.packageId);
                                setError("");
                                setFeedback("");
                              }}
                            >
                              <span>
                                <strong>{pkg.title}</strong>
                                <p>
                                  {[
                                    pkg.termYears != null ? `${pkg.termYears}y` : null,
                                    pkg.aavDisplay,
                                    pkg.scheduleA && pkg.scheduleA !== "None" ? "Sch A" : null,
                                    pkg.scheduleB && pkg.scheduleB !== "None" ? "Sch B" : null,
                                    pkg.blurb && pkg.kind === "path" ? pkg.blurb : null,
                                  ]
                                    .filter(Boolean)
                                    .slice(0, 3)
                                    .join(" · ")}
                                </p>
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </React.Fragment>
                  );
                })}

                {isSigning && signingResult ? (
                  <div className={`${prefix}-apply-bar`}>
                    <p className={`${prefix}-offer-label`}>Signing result</p>
                    <div className={`${prefix}-preview`}>
                      <span>
                        Starts <strong>{signingResult.contract_starts}</strong>
                      </span>
                      <span>
                        Cap <strong>{signingResult.cap_impact_display}</strong>
                      </span>
                      <span>
                        Slots <strong>{signingResult.contract_slots_after}</strong>
                      </span>
                      <span>
                        Assign <strong>{humanizeLabel(signingResult.assignment)}</strong>
                      </span>
                      <span>
                        Rights <strong>{signingResult.rights}</strong>
                      </span>
                    </div>
                  </div>
                ) : null}

                <div className={`${prefix}-apply-bar`}>
                  <button
                    type="button"
                    className={`${prefix}-decision-apply`}
                    disabled={busy || !selectedPackage || selectedPackage.disabled}
                    onClick={applyDecision}
                  >
                    {applyLabel}
                  </button>
                  {error ? <p className={`${prefix}-decision-feedback is-error`}>{error}</p> : null}
                  {feedback ? <p className={`${prefix}-decision-feedback is-ok`}>{feedback}</p> : null}
                </div>
              </div>

              <div className={`${prefix}-nego-col is-side`}>
                <p className={`${prefix}-offer-label`}>Rights timeline</p>
                {timeline.length ? (
                  <div className={`${prefix}-timeline`} aria-label="Rights timeline">
                    {timeline.map((step) => (
                      <div
                        key={step.key}
                        className={`${prefix}-tl-step${
                          step.tone === "future" ? " is-future" : ""
                        }${step.tone === "deadline" ? " is-deadline" : ""}`}
                      >
                        <div className={`${prefix}-tl-dot`} />
                        <span>{step.label}</span>
                        <strong>{step.value}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className={`${prefix}-empty`}>No timeline</p>
                )}
                <div className={`${prefix}-contract-doc`} style={{ marginTop: "0.2rem" }}>
                  <div className={`${prefix}-contract-doc-head`}>
                    <div>
                      <h3>Rights</h3>
                      <p>{humanizeLabel(focus.rights_status) || "Exclusive"}</p>
                    </div>
                  </div>
                  <div className={`${prefix}-contract-rows`}>
                    <div className={`${prefix}-contract-row`}>
                      <span>Type</span>
                      <strong>{humanizeLabel(focus.rights_type) || "—"}</strong>
                    </div>
                    <div className={`${prefix}-contract-row`}>
                      <span>Through</span>
                      <strong>{focus.rights_through ?? "—"}</strong>
                    </div>
                    <div className={`${prefix}-contract-row`}>
                      <span>Deadline</span>
                      <strong>{focus.rights_signing_deadline || "—"}</strong>
                    </div>
                    <div className={`${prefix}-contract-row`}>
                      <span>Term options</span>
                      <strong>
                        {(focus.legal_elc_terms?.legal_terms || []).join("/") || "—"}
                      </strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className={`${prefix}-workspace`}>
            <p className={`${prefix}-empty`}>No unsigned org prospects on file</p>
          </div>
        )
      }
      railContent={
        sortedProspects.length ? (
          sortedProspects.map((p, i) => {
            const id = String(p.player_id || i);
            const active = String(p.player_id || "") === String(selectedId);
            const meta = prospectPriorityMeta(p, { priorityIds, seasonYear });
            const rowOvr = p.overall != null ? p.overall : null;
            return (
              <button
                key={id}
                type="button"
                className={`${prefix}-rail-btn`}
                ref={(node) => {
                  if (p.player_id != null) railItemRefs.current[String(p.player_id)] = node;
                }}
                onClick={() => setSelectedId(String(p.player_id))}
                onKeyDown={(e) => onRailKeyDown(e, p.player_id)}
                aria-pressed={active}
              >
                <article className={`${prefix}-haul-card${active ? " is-active" : ""}`}>
                  <div className={`${prefix}-haul-rank`}>
                    {p.position || getPlayerPosition(p) || "—"}
                  </div>
                  <div className={`${prefix}-haul-body`}>
                    <strong>{p.name || getPlayerName(p)}</strong>
                    <span
                      className={`${prefix}-rail-priority${
                        meta.tone === "calm" ? " is-calm" : meta.tone === "soft" ? " is-soft" : ""
                      }`}
                    >
                      {meta.label}
                    </span>
                    <div className={`${prefix}-card-details`}>
                      <span>
                        {[
                          p.age != null ? `Age ${p.age}` : null,
                          rowOvr != null ? `OVR ${rowOvr}` : null,
                          p.potential != null
                            ? `POT ${p.potential}`
                            : p.potential_score != null
                              ? `POT ${Math.round(Number(p.potential_score))}`
                              : null,
                          p.eta != null ? `ETA ${p.eta}y` : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </div>
                  </div>
                  <div
                    className={`${prefix}-haul-grade-letter tone-${meta.rank >= 4 ? "c" : "n"}`}
                    aria-hidden
                  >
                    {meta.rank >= 4 ? "!" : "U"}
                  </div>
                </article>
              </button>
            );
          })
        ) : (
          <p className={`${prefix}-empty`}>No unsigned org prospects</p>
        )
      }
    />
    </>
  );
}
