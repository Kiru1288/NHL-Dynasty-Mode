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
import CapContractNegotiation from "../components/contracts/CapContractNegotiation";
import NegotiationMeetingPanel, { InterestFactors } from "../components/contracts/NegotiationMeetingPanel";
import { getContractOffice, qualifyRfa, releaseRfaRights, evaluateContractOffer, prospectRightsDecision, previewElcOffer, submitElcOffer, fileArbitration, settleArbitration, matchOfferSheet, declineOfferSheet, submitOfferSheet, advanceFreeAgencyDay, simFreeAgencyToSeasonStart, advanceContractNegotiationDay, signFreeAgent, getFreeAgentDetail, getFreeAgencyDesk, getSocialFeed } from "../services/franchiseService";
import { useGameUI } from "../game/GameUIContext";
import { projectNegotiationCap } from "../utils/contractNegotiation";
import { formatMoney, formatPick, getPlayerName, getPlayerOverall, getPlayerPosition, pickFranchiseData, safeArray } from "./shared/eventHelpers";
import { getOverallModifierBadge } from "../utils/playerOverall";
import { flagApiUrl, resolveCountryCode } from "../utils/countryFlags";

function computeOfferCapHitM(aav, years, signingBonus = 0) {
  const y = Math.max(1, Number(years) || 1);
  return Math.round(((Number(aav) || 0) * y + (Number(signingBonus) || 0)) / y * 1000) / 1000;
}

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
  String(franchiseState?.user_team_id || "");

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
  // Open on the players who need a decision, not the whole signed roster.
  const [filter, setFilter] = React.useState("expiring");
  const [sortKey, setSortKey] = React.useState("overall");
  const [sortDir, setSortDir] = React.useState("desc");
  const [selectedId, setSelectedId] = React.useState(null);
  const [panelMode, setPanelMode] = React.useState("detail"); // detail | negotiate
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [, setResponse] = React.useState(null);
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
    setResponse(null);
    setError("");
  }, [selectedId]);

  const resignSeasonYear = intOr(payload?.season_year, 0);

  const resignRowStatus = (row) => {
    if (!row) return "—";
    if (row.extension_signed) return "Extended";
    const phase = String(row.phase_status || row.negotiation_status || "").toLowerCase();
    if (phase === "accepted") return "Accepted";
    if (phase === "rejected") return "Rejected";
    if (phase === "countered") return "Countered";
    if (phase === "pending") return "Pending";
    if (phase === "released") return "Released";
    if (phase === "lapsed") return "Lapsed";
    if (row.contract_status === "rfa_rights") return "RFA Rights";
    if (!resignIsThisJuly(row, resignSeasonYear) && intOr(row.years_remaining, 0) > 0) {
      const clause = String(row.clause_label || "");
      if (clause && clause !== "None") return clause;
      return "Signed";
    }
    if (resignIsThisJuly(row, resignSeasonYear)) return "Expiring";
    return row.expiry_status || row.contract_status || "—";
  };

  const resignOffer = (row) => {
    const phase = String(row?.phase_status || row?.negotiation_status || "").toLowerCase();
    if (phase === "pending") return row?.pending_offer || row?.phase_last_offer || null;
    if (phase === "accepted") return row?.phase_terms || null;
    return null;
  };

  const resignMoney = (row) => {
    const offer = resignOffer(row);
    const phase = String(row?.phase_status || row?.negotiation_status || "").toLowerCase();
    if (offer && offer.aav_m != null && (phase === "pending" || (phase === "accepted" && resignIsThisJuly(row, resignSeasonYear)))) {
      return offer.aav_m;
    }
    return row?.aav_m ?? row?.cap_hit_m ?? row?.current_cap_hit;
  };

  const resignTermText = (row) => {
    const phase = String(row?.phase_status || row?.negotiation_status || "").toLowerCase();
    const offer = resignOffer(row);
    if (phase === "pending" && offer?.years) return `Offer ${offer.years}y`;
    if (phase === "accepted" && offer?.years && resignIsThisJuly(row, resignSeasonYear)) {
      return [offer.expiry_year, `${offer.years}y`].filter(Boolean).join(" · ");
    }
    return (
      [row?.expiry_year, row?.years_remaining != null ? `${row.years_remaining}y` : null]
        .filter(Boolean)
        .join(" · ") || "—"
    );
  };

  const filterCounts = React.useMemo(() => {
    const rows = contracts;
    const thisJuly = (r) => resignIsThisJuly(r, resignSeasonYear);
    return {
      all: rows.length,
      expiring: rows.filter((r) => !r.extension_signed && thisJuly(r)).length,
      ufa: rows.filter(
        (r) => thisJuly(r) && (String(r.expiry_status || r.expiry_type || "").toUpperCase() === "UFA" || r.own_ufa),
      ).length,
      rfa: rows.filter(
        (r) =>
          r.contract_status === "rfa_rights" ||
          (thisJuly(r) && String(r.expiry_status || r.expiry_type || "").toUpperCase() === "RFA"),
      ).length,
      signed: rows.filter((r) => !thisJuly(r) && intOr(r.years_remaining, 0) > 0 && r.contract_status !== "rfa_rights").length,
      extension: rows.filter((r) => r.extension_eligible === true).length,
      minors: rows.filter((r) => r.in_minors || String(r.role || "").toLowerCase().includes("minor")).length,
      unsigned: rows.filter((r) => r.contract_status === "rfa_rights" || r.qualifying_offer_eligible).length,
    };
  }, [contracts, resignSeasonYear]);

  const visibleRows = React.useMemo(() => {
    let rows = [...contracts];
    const phaseStatus = (r) => String(r.phase_status || r.negotiation_status || "").toLowerCase();
    const isPhaseAccepted = (r) => phaseStatus(r) === "accepted";
    const isPhaseReleased = (r) => phaseStatus(r) === "released" || phaseStatus(r) === "lapsed";
    const isPhaseRejected = (r) => phaseStatus(r) === "rejected";
    const wasExpiring = (r) => {
      if (isPhaseAccepted(r) || r.pending_extension || r.extension_signed) return false;
      if (isPhaseReleased(r) || isPhaseRejected(r)) return true;
      return resignIsThisJuly(r, resignSeasonYear);
    };

    if (filter === "expiring") {
      rows = rows.filter((r) => wasExpiring(r));
    } else if (filter === "ufa") {
      rows = rows.filter(
        (r) =>
          r.own_ufa === true ||
          (resignIsThisJuly(r, resignSeasonYear) &&
            String(r.expiry_status || r.expiry_type || "").toUpperCase() === "UFA") ||
          (isPhaseRejected(r) && String(r.expiry_status || "").toUpperCase() === "UFA")
      );
    } else if (filter === "rfa") {
      rows = rows.filter(
        (r) =>
          r.contract_status === "rfa_rights" ||
          (resignIsThisJuly(r, resignSeasonYear) &&
            String(r.expiry_status || r.expiry_type || "").toUpperCase() === "RFA") ||
          (isPhaseReleased(r) && String(r.expiry_status || r.expiry_type || "").toUpperCase() === "RFA")
      );
    } else if (filter === "signed") {
      rows = rows.filter(
        (r) =>
          isPhaseAccepted(r) ||
          (!resignIsThisJuly(r, resignSeasonYear) &&
            intOr(r.years_remaining, 0) > 0 &&
            r.contract_status !== "rfa_rights" &&
            r.contract_status !== "released")
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
  }, [contracts, filter, sortKey, sortDir, resignSeasonYear]);

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
  const resignBonusRules = payload?.signing_bonus || {};
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
          : `Day ${result?.own_fa_window?.day ?? "—"} / ${result?.own_fa_window?.days_total ?? 7} — no signatures yet`
      );
      setResponse(result);
    } catch (e) {
      setError(String(e?.message || "Could not advance day"));
    } finally {
      setBusy(false);
    }
  };

  const selectedOvr = selected?.overall ?? selected?.ovr ?? getPlayerOverall(selected);
  const selectedInterest = resignInterestLabel(selected);
  const selectedArchetype = resignArchetype(selected);
  const agentBits = resignAgentBits(selected, askAav, askYears);
  const decisionCount = summary.pendingDecisions ?? pending.length;
  const ufaCount = summary.ufaCount ?? filterCounts.ufa;
  const rfaCount = summary.rfaCount ?? filterCounts.rfa;
  const ownWindow = payload?.own_fa_window || {};
  const windowDay = Number(ownWindow.day || 0);
  const windowTotal = Number(ownWindow.days_total || 7);
  const windowOpen = Boolean(ownWindow.complete) || windowDay >= windowTotal;
  const continueDisabled = busy || !canContinue || !windowOpen;
  const continueTitle = continueDisabled
    ? (!windowOpen
        ? `Sim all ${windowTotal} re-sign days before opening free agency (${windowDay}/${windowTotal})`
        : safeArray(payload?.blocking_reasons)[0] ||
          (blocking[0]?.message || blocking[0]) ||
          "Resolve required RFA decisions before continuing")
    : undefined;
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
      hideRail
      railTitle="Queue"
      railHint={null}
      heroContent={
        <div className={`${prefix}-workspace`}>
          <div className={`${prefix}-head`}>
            <div className={`${prefix}-head__title`}>
              <h1 className={`${prefix}-page-title`}>Re-sign your players</h1>
              <p>
                <b>{decisionCount}</b> {decisionCount === 1 ? "decision" : "decisions"} · {ufaCount} UFA · {rfaCount} RFA
                {slotUsed != null && slots.limit != null ? ` · ${slotUsed}/${slots.limit} contracts` : ""}
              </p>
            </div>
            <div className={`${prefix}-head__cap`}>
              <strong>{formatMoney(capSpace)}</strong>
              <span>cap space</span>
            </div>
            <div className={`${prefix}-head__window`}>
              <span>
                Exclusive window · day {ownWindow.day ?? 0} of {ownWindow.days_total ?? 6}
                {ownWindow.days_remaining != null ? ` · ${ownWindow.days_remaining} left` : ""}
              </span>
              <button
                type="button"
                className={`${prefix}-sim-day-btn`}
                disabled={busy || Boolean(ownWindow.complete)}
                onClick={simNegotiationDay}
              >
                Sim a day
              </button>
            </div>
          </div>
          {dayFlash || warnings[0] ? (
            <p className={`${prefix}-head__flash`} role="status">{dayFlash || warnings[0]}</p>
          ) : null}
          <div className={`${prefix}-filter-row`} role="tablist" aria-label="Filter players">
            {[
              ["expiring", "Decisions"],
              ["ufa", "UFAs"],
              ["rfa", "RFAs"],
              ["extension", "Extension-eligible"],
              ["all", "Whole roster"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={filter === id}
                className={`${prefix}-filter-chip${filter === id ? " is-active" : ""}`}
                onClick={() => setFilter(id)}
              >
                {label}
                <em>{filterCounts[id] ?? ""}</em>
              </button>
            ))}
          </div>

          <div className={`${prefix}-table-layout${panelMode === "negotiate" && selected ? " is-negotiating" : ""}`}>
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
                        ...row,
                        id,
                        player_id: id,
                      });
                      const phase = String(row.phase_status || row.negotiation_status || "").toLowerCase();
                      const status = resignRowStatus(row);
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
                            primaryAction ? "is-actionable" : "is-settled",
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
                              <strong>
                                {row.name || getPlayerName(row)}
                                {row.roster_level && row.roster_level !== "NHL" ? ` · ${row.roster_level}` : ""}
                              </strong>
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
                          <td>{formatMoney(resignMoney(row))}</td>
                          <td>{resignTermText(row)}</td>
                          <td>
                            {primaryAction || statusTone ? (
                              <span className={`${prefix}-interest-pill tone-${resignInterestTone(interest)}`}>
                                {interest}
                              </span>
                            ) : null}
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
                            ) : null}
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
                          ...selected,
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
                      <h3>
                        {String(selected.name || getPlayerName(selected) || "")}
                        {selected.roster_level && selected.roster_level !== "NHL" ? ` · ${selected.roster_level}` : ""}
                      </h3>
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
                                            selected.qualifying_offer_aav_m
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
                      {selected.ineligible_reason ? (
                        <p className={`${prefix}-context`}>{selected.ineligible_reason}</p>
                      ) : null}
                      {error ? <p className={`${prefix}-warn`} role="alert">{error}</p> : null}
                    </div>
                  ) : (
                    <CapContractNegotiation
                      row={selected}
                      capSnapshot={cap}
                      nextYearProjection={payload?.nextYearProjection || payload?.next_year_projection || {}}
                      signingBonusElig={resignBonusRules}
                      seasonLabel={seasonLabel(franchiseState)}
                      contractTypes={safeArray(selected.legal_contract_types)}
                      backLabel="Back to contract details"
                      busy={busy}
                      onBusy={(v) => setBusy(Boolean(v))}
                      onBack={() => setPanelMode("detail")}
                      onResult={(result, meta) => {
                        if (!result || meta?.preview) return;
                        if (result.re_sign || result.contracts?.contracts) {
                          applyOfficeOrResign(result.office, result.re_sign || result.contracts);
                        } else if (result.office) {
                          applyOfficeOrResign(result.office, null);
                        }
                        setResponse(result);
                        const st = String(result.status || result.player_response?.status || "").toLowerCase();
                        if (["accepted", "countered", "rejected"].includes(st) || meta?.signed) {
                          const popup = buildDealOutcome(
                            { ...result, status: st || (meta?.signed ? "accepted" : result.status) },
                            { playerName: getPlayerName(selected), contextLabel: "Player re-sign" }
                          );
                          if (popup) setDealPopup(popup);
                        }
                        if (st === "accepted" || meta?.signed) {
                          setPanelMode("detail");
                          if (selected?.player_id) setSelectedId(String(selected.player_id));
                        }
                      }}
                    />
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
            const status = resignRowStatus(p);
            return (
              <button
                key={id}
                type="button"
                className={`${prefix}-rail-btn`}
                onClick={() => selectPlayer(id, { openNegotiate: true })}
              >
                <article className={`${prefix}-queue-card${active ? " is-active" : ""}`}>
                  <PlayerHeadshot player={ensurePlayerHeadshotFields({ ...p, id, player_id: id })} size="xs" />
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


/** This July's free-agent class. A year that still covers next season is a signed deal. */
function resignIsThisJuly(row, seasonYear) {
  if (!row) return false;
  if (row.extension_signed || row.pending_extension) return false;
  const phase = String(row.phase_status || row.negotiation_status || "").toLowerCase();
  if (phase === "accepted") return false;
  if (row.own_ufa === true || row.contract_status === "own_ufa") return true;
  if (row.contract_status === "rfa_rights") return true;
  const yrs = intOr(row.years_remaining, 0);
  const expiry = intOr(row.expiry_year, 0);
  const july = Boolean(row.pending_july1_expiry);
  const faYear = intOr(seasonYear, 0) + 1;
  if (yrs > 1 && !july) return false;
  if (expiry > faYear && !july) return false;
  if (july && yrs <= 1) return true;
  return yrs <= 1 && (expiry <= 0 || expiry <= faYear);
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

function dedupeWire(items) {
  const seen = new Set();
  const out = [];
  for (const n of safeArray(items)) {
    const key = String(n?.text || `${n?.team_id}|${n?.player_id}|${n?.aav_m}`);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(n);
  }
  return out;
}

function FaSigningRow({ prefix, row, tone }) {
  const abbrev = row.team_abbrev || "";
  return (
    <div className={`${prefix}-day-row${tone ? ` is-${tone}` : ""}`}>
      <TeamLogoBadge
        teamLogo={resolveFranchiseTeamLogo(
          { abbrev, team_abbrev: abbrev, name: row.team_name },
          row.team_name || abbrev
        )}
        teamName={row.team_name || abbrev}
        size={26}
        variant="circle"
      />
      <div className={`${prefix}-day-main`}>
        <strong>{row.name || row.player_id}</strong>
        <em>
          {[row.position, row.overall ? `${row.overall} OVR` : null].filter(Boolean).join(" · ")}
          {" → "}
          {row.team_name || abbrev || "—"}
        </em>
      </div>
      <span className={`${prefix}-day-terms`}>
        {row.aav_m != null ? formatMoney(row.aav_m) : "—"}
        <i>× {row.years || "?"}y</i>
      </span>
    </div>
  );
}

function FaSocialPost({ prefix, post }) {
  const name = post.author_name || "Puckr";
  const handle = post.handle || "";
  const color = post.author_color || null;
  const verified = Boolean(post.verified);
  const initials = String(name)
    .split(/\s+/)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  return (
    <article className={`${prefix}-social-post`}>
      <span className={`${prefix}-social-avatar`} style={color ? { background: color } : undefined}>
        {initials}
      </span>
      <div>
        <header>
          <strong>{name}</strong>
          {verified ? <i className={`${prefix}-social-check`}>✓</i> : null}
          <span>{handle}</span>
          {post.time ? <span>· {post.time}</span> : null}
        </header>
        <p>{post.text}</p>
      </div>
    </article>
  );
}

function rightsTag(row) {
  const s = String(row?.ufaOrRfa || row?.status || row?.expiry_status || "").toUpperCase();
  if (s.includes("RFA")) return "RFA";
  if (s.includes("UFA")) return "UFA";
  return null;
}

function RightsBadge({ prefix, row }) {
  const tag = rightsTag(row);
  if (!tag) return null;
  return (
    <span
      className={`${prefix}-rights-badge is-${tag.toLowerCase()}`}
      title={
        tag === "RFA"
          ? "Restricted free agent: his club holds his rights (offer sheet / qualifying offer)"
          : "Unrestricted free agent: free to sign anywhere"
      }
    >
      {tag}
    </span>
  );
}

function offerSheetCompensation(tiers, aavM) {
  const aav = Number(aavM) || 0;
  const sorted = safeArray(tiers)
    .filter((t) => t && t.tier !== "none")
    .sort((a, b) => Number(b.aav_floor_m || 0) - Number(a.aav_floor_m || 0));
  const hit = sorted.find((t) => aav >= Number(t.aav_floor_m || 0));
  return hit ? hit.label || safeArray(hit.rounds).join(", ") : "No compensation";
}

export function FreeAgencyEventMenu({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
  onReopenStage,
  standalone = false,
  ctaLabel: ctaLabelProp,
}) {
  const prefix = "fa";
  const { mergeFranchiseState, setFranchiseEventForceOpen } = useGameUI();
  const phase = String(franchiseState?.phase || franchiseState?.season_phase || "").toLowerCase();
  const inOffseasonWire =
    phase === "offseason" && String(franchiseState?.offseason_stage || "") === "free_agency";
  const ctaLabel =
    ctaLabelProp ||
    (standalone || !inOffseasonWire ? "Back to Hub" : "Continue to New Season");
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
  const [offerNtcModified, setOfferNtcModified] = React.useState(false);
  const [offerNmc, setOfferNmc] = React.useState(false);
  const [nhlRoster, setNhlRoster] = React.useState(false);
  const offerNtcMode = offerNmc ? "NONE" : offerNtc ? (offerNtcModified ? "MODIFIED" : "FULL") : "NONE";
  const [offerBonus, setOfferBonus] = React.useState("0");
  const [contractCategory, setContractCategory] = React.useState("nhl_one_way");
  const [faMeetingRev, setFaMeetingRev] = React.useState(0);
  const [railTab, setRailTab] = React.useState("today");
  const [dayReport, setDayReport] = React.useState(null);
  const [socialRev, setSocialRev] = React.useState(0);
  const [social, setSocial] = React.useState({ posts: [], loading: false, error: "" });

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

  React.useEffect(() => {
    if (railTab !== "social") return undefined;
    let cancelled = false;
    setSocial((prev) => ({ ...prev, loading: true, error: "" }));
    getSocialFeed(franchiseState?.session_id, { tab: "fa" })
      .then((feed) => {
        if (!cancelled) setSocial({ posts: safeArray(feed?.puckr), loading: false, error: "" });
      })
      .catch(() => {
        if (!cancelled) setSocial((prev) => ({ ...prev, loading: false, error: "Couldn't load the feed." }));
      });
    return () => {
      cancelled = true;
    };
  }, [railTab, socialRev, franchiseState?.session_id]);

  const bonus = market.signing_bonus || {};
  const bonusAllowed = Boolean(bonus.eligible);
  const recent = safeArray(market.recent_league_signings);
  const news = safeArray(market.market_news);
  const capSnap = market.cap_snapshot || {};
  const capSpace = Number(
    capSnap.usable_cap_space_m ?? market.cap_space_m ?? 0
  );
  const slots = market.contract_slots || {};
  const offerSheetInfo = market.offer_sheets || {};
  const sheetTargets = React.useMemo(
    () => safeArray(market?.offer_sheets?.targets).map((t) => ({ ...t, _offerSheet: true })),
    [market]
  );
  const selected =
    (filter === "offersheet" ? sheetTargets : rows).find(
      (p) => String(p.player_id || p.id) === String(selectedId)
    ) ||
    rows.find((p) => String(p.player_id || p.id) === String(selectedId)) ||
    null;
  const isOfferSheet = Boolean(selected?._offerSheet);
  const minSalaryM = Number(bonus.min_salary_m || 0.85);
  // The detail payload carries a projected ask; the board row carries the live
  // market ask the player negotiates from. Keep the board's ask fields.
  const deskPlayer = faDetail?.free_agent
    ? {
        ...selected,
        ...faDetail.free_agent,
        askingAav: selected?.askingAav,
        asking_aav: selected?.asking_aav,
        ask_aav_m: selected?.ask_aav_m,
        askingTerm: selected?.askingTerm,
        asking_term: selected?.asking_term,
        wants_clause: selected?.wants_clause,
        bonus_demand_label: selected?.bonus_demand_label,
        bonus_demand_pct: selected?.bonus_demand_pct,
      }
    : selected;

  React.useEffect(() => {
    if (!selectedId) {
      setFaDetail(null);
      setFaDetailLoading(false);
      return undefined;
    }
    let cancelled = false;
    setFaDetail(null);
    if (filter === "offersheet") {
      setFaDetailLoading(false);
      return undefined;
    }
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
  }, [selectedId, filter]);

  React.useEffect(() => {
    if (!selected) return;
    const ask =
      selected.askingAav ??
      selected.asking_aav_m ??
      selected.ask_aav_m ??
      selected.asking_price ??
      selected.want_aav_m;
    setOfferAav(ask != null ? String(Number(ask).toFixed(3)) : "1.000");
    const term = Number(selected.askingTerm || selected.asking_term || selected.want_years || 2);
    setOfferYears(String(term));
    const wants = String(selected.wants_clause || "None").toUpperCase();
    setOfferNmc(wants === "NMC");
    setOfferNtc(wants === "NTC" || wants === "M-NTC" || wants === "NMC");
    setOfferNtcModified(wants === "M-NTC");
    const bonusPct = Number(selected.bonus_demand_pct || 0);
    setOfferBonus(
      bonusPct > 0 && ask != null && bonusAllowed
        ? (Math.ceil(Number(ask) * term * bonusPct * 40) / 40).toFixed(3)
        : "0"
    );
    setNhlRoster(false);
    setError("");
    setFeedback("");
    setResponse(null);
  }, [selected?.player_id || selected?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const offerAavNum = Number(offerAav) || 0;
  const maxTermUfa = Math.max(1, Number(bonus.max_term_ufa || 6));
  const termOptionsUfa = Array.from({ length: maxTermUfa }, (_, i) => i + 1);
  const offerYearsNum = Math.min(maxTermUfa, Math.max(1, parseInt(offerYears, 10) || 1));
  const bonusCapM = bonusAllowed
    ? Math.floor(offerAavNum * offerYearsNum * Number(bonus.max_bonus_pct || 0) * 40) / 40
    : 0;
  const offerBonusNum = Math.min(bonusCapM, Math.max(0, Number(offerBonus) || 0));
  const offerCapHitNum = computeOfferCapHitM(offerAavNum, offerYearsNum, offerBonusNum);
  const capProj = projectNegotiationCap({
    capSnapshot: capSnap,
    playerRow: deskPlayer || {},
    offerCapHitM: offerCapHitNum,
    replaceCurrentHit: false,
  });
  const projectedSpace = capProj.projectedAfterM;
  const negoInterest = Number(response?.evaluation?.interest ?? response?.player_response?.interest ?? 0) || 0;
  const negoFeedback =
    response?.player_response?.feedback ||
    response?.evaluation?.reason ||
    response?.player_response?.clause_note ||
    null;

  React.useEffect(() => {
    if (!selected?.player_id && !selected?.id) return undefined;
    if (selected?._offerSheet) {
      setResponse(null);
      return undefined;
    }
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const result = await evaluateContractOffer({
          player_id: selected.player_id || selected.id,
          aav_m: offerAavNum,
          years: offerYearsNum,
          ntc: offerNtcMode === "FULL" || offerNtcMode === "MODIFIED",
          ntc_mode: offerNtcMode,
          m_ntc: offerNtcMode === "MODIFIED",
          nmc: offerNmc,
          signing_bonus_m: offerBonusNum,
          contract_category: contractCategory,
          two_way: contractCategory === "nhl_two_way",
          nhl_roster: nhlRoster,
          nhl_only: nhlRoster,
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
  }, [selectedId, offerAav, offerYears, offerNtcMode, offerNmc, offerBonus, contractCategory, nhlRoster, faMeetingRev]);

  const filtered = React.useMemo(() => {
    let list = filter === "offersheet" ? [...sheetTargets] : [...rows];
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
    } else if (filter === "ufa") {
      list = list.filter((p) => String(p.ufaOrRfa || p.status || "").toUpperCase() === "UFA");
    } else if (filter === "rfa") {
      list = list.filter((p) => String(p.ufaOrRfa || p.status || "").toUpperCase() === "RFA");
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
            ? Number(a.askingAav ?? a.asking_aav_m ?? a.ask_aav_m ?? a.want_aav_m ?? 0)
            : sortKey === "interest"
              ? Number(a.interest_to_user || 0)
              : Number(getPlayerOverall(a) || 0);
      const bv =
        sortKey === "age"
          ? Number(b.age || 0)
          : sortKey === "ask"
            ? Number(b.askingAav ?? b.asking_aav_m ?? b.ask_aav_m ?? b.want_aav_m ?? 0)
            : sortKey === "interest"
              ? Number(b.interest_to_user || 0)
              : Number(getPlayerOverall(b) || 0);
      if (av === bv) return String(getPlayerName(a)).localeCompare(String(getPlayerName(b)));
      return av < bv ? -1 * (sortKey === "age" || sortKey === "ask" ? 1 : dir) : 1 * (sortKey === "age" || sortKey === "ask" ? 1 : dir);
    });
    return list;
  }, [rows, sheetTargets, filter, search, sortKey, watch]);

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
      const dayEvents = result?.free_agency_market?.day_events || {};
      const userLost = safeArray(dayEvents.user_lost);
      const lostIds = new Set(userLost.map((x) => String(x.player_id)));
      const userRejected = safeArray(result?.user_resolve?.rejected).filter(
        (x) => !lostIds.has(String(x.player_id))
      );
      const cpuSigned = safeArray(dayEvents.recent_signings);
      setDayReport({
        day: result?.day || result?.free_agency_market?.fa_market_day,
        days: days,
        signings: cpuSigned,
        userSigned,
        userLost,
        userRejected,
      });
      setRailTab("today");
      setSocialRev((v) => v + 1);
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
      if (userLost.length) {
        const first = userLost[0];
        setDealPopup({
          tone: "deny",
          kicker: "Free agency",
          title: "Signed elsewhere",
          player: first.name || first.player_id,
          body: `${first.name || "He"} took ${first.signed_team_name || "another club"}'s offer instead of yours${
            first.your_aav_m != null ? ` (you offered ${formatMoney(first.your_aav_m)} × ${first.your_years}y)` : ""
          }.${userLost.length > 1 ? ` ${userLost.length - 1} more of your targets also signed elsewhere.` : ""}`,
          terms:
            first.aav_m != null ? `${first.signed_team_abbrev || ""} · ${formatMoney(first.aav_m)} × ${first.years}y` : null,
          cta: "Close",
        });
      } else if (userSigned.length) {
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

  const jumpToSeason = async () => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await simFreeAgencyToSeasonStart();
      if (result?.state) mergeFranchiseState(result.state);
      setFranchiseEventForceOpen(false);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : e?.message || "Could not open the season");
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
        ntc: offerNtcMode === "FULL" || offerNtcMode === "MODIFIED",
        ntc_mode: offerNtcMode,
        m_ntc: offerNtcMode === "MODIFIED",
        nmc: offerNmc,
        signing_bonus_m: offerBonusNum,
        contract_category: contractCategory,
        two_way: contractCategory === "nhl_two_way",
        nhl_roster: nhlRoster,
        nhl_only: nhlRoster,
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

  const submitSheet = async () => {
    if (!selected || busy) return;
    setBusy(true);
    setError("");
    setFeedback("");
    try {
      const pid = selected.player_id || selected.id;
      const result = await submitOfferSheet({
        player_id: pid,
        rights_team_id: selected.rights_team_id,
        aav_m: offerAavNum,
        years: offerYearsNum,
        signing_bonus_m: offerBonusNum,
      });
      if (result?.ok) {
        const comp =
          result?.offer_sheet?.compensation_label || offerSheetCompensation(offerSheetInfo.tiers, offerCapHitNum);
        setDealPopup({
          tone: "pending",
          kicker: "Offer sheet",
          title: "Offer sheet tendered",
          player: getPlayerName(selected),
          body: `${selected.rights_team_name || selected.rights_team_abbrev || "His club"} has ${
            result?.offer_sheet?.match_deadline_days ?? 7
          } days to match. If they decline, he signs with you and you owe: ${comp}.`,
          terms: `${formatMoney(offerAavNum)} × ${offerYearsNum}y`,
          cta: "Got it",
        });
        setMarket((prev) => ({
          ...prev,
          offer_sheets: {
            ...(prev.offer_sheets || {}),
            targets: safeArray(prev.offer_sheets?.targets).map((t) =>
              String(t.player_id || t.id) === String(pid)
                ? { ...t, offer_sheet_pending: true, offer_sheet_by_user: true }
                : t
            ),
          },
        }));
        setFeedback(`Offer sheet tendered to ${getPlayerName(selected)} — Sim Day for the match decision.`);
      } else {
        setError(result?.reason || "Offer sheet rejected");
      }
    } catch (e) {
      setError(e?.message || "Offer sheet failed");
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
      rootClassName={`${prefix}-root--market edraft-skin`}
      seasonLabel={seasonLabel(franchiseState)}
      title="Market Board"
      hideTitle
      hideEyebrow
      hideTicker
      ctaLabel={ctaLabel}
      onContinue={continueHandler}
      secondaryCtaLabel={
        inOffseasonWire && typeof onReopenStage === "function" ? "NHL Re-Sign" : ""
      }
      onSecondaryCta={
        inOffseasonWire && typeof onReopenStage === "function"
          ? () => onReopenStage("re_sign")
          : undefined
      }
      onBack={onBack}
      footerAlign="split"
      railTitle={null}
      railHint={null}
      heroContent={
        <div className={`${prefix}-workspace`}>
          <div className={`${prefix}-fa-bar`}>
            <div className={`${prefix}-fa-phase`}>
              <h2>{market.market_phase_label || "Open market"}</h2>
              <p>
                Day {market.fa_market_day ?? 0} · {market.available_count ?? rows.length} available
              </p>
            </div>
            <div className={`${prefix}-fa-cap`}>
              <strong>{formatMoney(capSpace)}</strong>
              <span>
                cap space · {slotUsed != null && slots.limit != null ? `${slotUsed}/${slots.limit}` : `${slots.open ?? "—"} open`} contracts
                {bonusAllowed ? " · bonuses allowed" : ""}
              </span>
            </div>
            <div className={`${prefix}-fa-sim`}>
              <button type="button" className={`${prefix}-sim-btn is-primary`} disabled={busy} onClick={() => advanceDay(1)}>
                {busy ? "Simming…" : "Sim a day"}
              </button>
              <button type="button" className={`${prefix}-sim-btn`} disabled={busy} onClick={() => advanceDay(7)}>
                Sim a week
              </button>
              <button type="button" className={`${prefix}-sim-btn`} disabled={busy} onClick={jumpToSeason}>
                Sim to season start
              </button>
            </div>
          </div>
          {market.cap_release?.text ? (
            <details className={`${prefix}-fa-notice`}>
              <summary>{String(market.cap_release.text).split(":")[0]}</summary>
              <p>{market.cap_release.text}</p>
            </details>
          ) : null}

          <div className={`${prefix}-fa-tools`}>
            <input
              className={`${prefix}-fa-search`}
              type="search"
              aria-label="Search free agents"
              placeholder="Search name, team, country"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <div className={`${prefix}-fa-filters`} role="tablist" aria-label="Filter free agents">
              {[
                ["all", "All"],
                ["F", "Forwards"],
                ["D", "Defense"],
                ["G", "Goalies"],
                null,
                ["elite", "Elite"],
                ["top6", "Top 6"],
                ["young", "Young"],
                ["vet", "Veterans"],
                ["cheap", "Cheap"],
                null,
                ["ufa", "UFAs"],
                ["offersheet", `Offer-sheet RFAs ${sheetTargets.length}`],
                ["hot", "Bidding wars"],
                ["watch", "Watchlist"],
              ].map((item, i) =>
                item ? (
                  <button
                    key={item[0]}
                    type="button"
                    role="tab"
                    aria-selected={filter === item[0]}
                    className={`${prefix}-fa-chip${filter === item[0] ? " is-active" : ""}`}
                    onClick={() => setFilter(item[0])}
                  >
                    {item[1]}
                  </button>
                ) : (
                  <i key={`sep-${i}`} className={`${prefix}-fa-sep`} aria-hidden="true" />
                )
              )}
            </div>
            <select
              className={`${prefix}-fa-sort`}
              aria-label="Sort free agents"
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
            >
              <option value="overall">Sort by OVR</option>
              <option value="ask">Sort by ask</option>
              <option value="age">Sort by age</option>
              <option value="interest">Sort by interest in you</option>
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
                  const prevAbbr = p.previous_team_abbrev || p.rights_team_abbrev || "";
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
                      <span className={`${prefix}-fa-prev`} title={prevName || undefined}>
                        <PlayerHeadshot player={ensurePlayerHeadshotFields({ ...p, id, player_id: id })} size="xs" />
                      </span>
                      <span className={`${prefix}-fa-row-body`}>
                        <strong>
                          {getPlayerName(p)} <RightsBadge prefix={prefix} row={p} />
                        </strong>
                        <em>
                          {[
                            getPlayerPosition(p),
                            p.age != null ? `Age ${p.age}` : null,
                            p._offerSheet ? `Rights: ${p.rights_team_abbrev || p.rights_team_name || "—"}` : prevAbbr || null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </em>
                        <span className={`${prefix}-fa-row-meta`}>
                          {p._offerSheet
                            ? p.offer_sheet_pending
                              ? p.offer_sheet_by_user
                                ? "Your offer sheet is pending"
                                : "Offer sheet pending from another club"
                              : `Qualified · QO ${formatMoney(p.qualifying_offer_aav_m)}`
                            : `${String(p.decision_state || "awaiting").replace(/_/g, " ")} · Interest ${String(interest)}`}
                        </span>
                      </span>
                      <span className={`${prefix}-fa-rating`}>
                        <span>
                          <b>{getPlayerOverall(p) ?? "—"}</b>
                          <i>OVR</i>
                          {getOverallModifierBadge(p) ? <em className="ovr-mod-badge">{getOverallModifierBadge(p)}</em> : null}
                        </span>
                        <span className="is-pot">
                          <b>{p.potential ?? p.pot ?? "—"}</b>
                          <i>POT</i>
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
                        {formatMoney(p.askingAav ?? p.asking_aav_m ?? p.ask_aav_m ?? p.want_aav_m)}
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
                <p className={`${prefix}-empty`}>
                  {filter === "offersheet"
                    ? sheetTargets.length
                      ? "Select an RFA to draft an offer sheet."
                      : "No unsigned RFAs right now. Clubs qualify their RFAs on Opening Day and settle them within the first week."
                    : "Select a free agent to open the signing desk."}
                </p>
              ) : (
                <div className="nd nd2 nd2--fa">
                  <header className="nd2-hero">
                    <span aria-hidden="true" />
                    <div className="nd2-face">
                      <PlayerHeadshot
                        player={ensurePlayerHeadshotFields({
                          ...deskPlayer,
                          id: deskPlayer.player_id || deskPlayer.id,
                          player_id: deskPlayer.player_id || deskPlayer.id,
                          name: getPlayerName(deskPlayer),
                          position: getPlayerPosition(deskPlayer),
                          age: deskPlayer.age,
                          nationality: deskPlayer.nationality,
                        })}
                        size="md"
                        showFlag={false}
                      />
                    </div>
                    <div className="nd2-who">
                      <h3>
                        {getPlayerName(deskPlayer)} <RightsBadge prefix={prefix} row={deskPlayer} />
                      </h3>
                      <p>
                        {[
                          getPlayerPosition(deskPlayer),
                          getPlayerOverall(deskPlayer) != null ? `${getPlayerOverall(deskPlayer)} OVR` : null,
                          deskPlayer.age != null ? `age ${deskPlayer.age}` : null,
                          deskPlayer.potential != null ? `potential ${deskPlayer.potential}` : null,
                          deskPlayer.role,
                        ].filter(Boolean).join(" · ")}
                      </p>
                      <p className="nd2-agent">
                        {isOfferSheet ? "Rights held by " : "Last with "}
                        {deskPlayer.rights_team_name || deskPlayer.previous_team || deskPlayer.current_team || deskPlayer.nhl_team || "no club"}
                        {deskPlayer.nationality ? ` · ${deskPlayer.nationality}` : ""}
                      </p>
                    </div>
                    <div className="nd2-gap" aria-label="Their ask against your offer">
                      {isOfferSheet ? (
                        <div>
                          <span>Compensation if unmatched</span>
                          <strong>{offerSheetCompensation(offerSheetInfo.tiers, offerCapHitNum)}</strong>
                          <em>market {formatMoney(selected.market_value_m)}</em>
                        </div>
                      ) : (
                        <div>
                          <span>They want</span>
                          <strong>{formatMoney(deskPlayer.askingAav ?? deskPlayer.ask_aav_m)}</strong>
                          <em>
                            {deskPlayer.askingTerm ?? deskPlayer.asking_term ?? "—"} yrs
                            {deskPlayer.wants_clause && deskPlayer.wants_clause !== "None" ? ` · ${deskPlayer.wants_clause}` : ""}
                          </em>
                        </div>
                      )}
                      <i aria-hidden="true">vs</i>
                      <div className="is-you">
                        <span>You offer</span>
                        <strong>{formatMoney(offerAavNum)}</strong>
                        <em>
                          {offerYearsNum} yrs
                          {offerNmc ? " · NMC" : offerNtc ? (offerNtcModified ? " · M-NTC" : " · NTC") : ""}
                        </em>
                      </div>
                    </div>
                  </header>

                  {isOfferSheet ? (
                    <section className="nd2-gauge" aria-label="Offer sheet">
                      <strong>{"⇄"}</strong>
                      <div className="nd2-gauge__body">
                        <span>
                          {selected.rights_team_name || selected.rights_team_abbrev || "His club"} can match. Clubs usually match offers near market value.
                        </span>
                        <small>Previous AAV {formatMoney(selected.previous_aav_m)}</small>
                        {selected.offer_sheet_pending ? (
                          <small className="nd2-warn">
                            {selected.offer_sheet_by_user
                              ? "Your offer sheet is already on the table."
                              : "Another club already has an offer sheet pending on this player."}
                          </small>
                        ) : null}
                      </div>
                    </section>
                  ) : (
                    <section
                      className={`nd2-gauge tone-${negoInterest >= 88 ? "instant" : negoInterest >= 62 ? "good" : negoInterest >= 40 ? "mid" : "bad"}${negoInterest >= 62 ? " will-accept" : ""}`}
                      aria-label="Deal interest"
                    >
                      <strong>{negoInterest ? Math.round(negoInterest) : "—"}</strong>
                      <div className="nd2-gauge__body">
                        <span>{negoInterest >= 62 ? "They'd take this" : "Not enough to sign yet"}</span>
                        <div className="nd2-gauge__track">
                          <i className="nd2-gauge__fill" style={{ width: `${Math.max(4, Math.min(100, negoInterest || 4))}%` }} />
                          <i className="nd2-gauge__mark" style={{ left: "62%" }} title="Accept line" />
                        </div>
                        <small>
                          {negoFeedback ||
                            [
                              selected.interest_to_user_label ? `${selected.interest_to_user_label} interest in your club` : null,
                              selected.sign_likelihood != null ? `${Math.round(Number(selected.sign_likelihood) * 100)}% likely to sign somewhere this window` : null,
                            ].filter(Boolean).join(" · ")}
                        </small>
                      </div>
                    </section>
                  )}

                  <div className="nd2-body">
                    <section className="nd2-offer" aria-label="Your offer">
                      {isOfferSheet ? null : (
                        <div className="nd2-field nd2-field--row">
                          <span>Contract</span>
                          <div className="nd-seg" role="group" aria-label="Contract type">
                            {[
                              ["nhl_one_way", "One-way"],
                              ["nhl_two_way", "Two-way"],
                              ["ahl", "AHL"],
                              ["pto", "PTO"],
                            ].map(([id, label]) => (
                              <button
                                key={id}
                                type="button"
                                className={contractCategory === id ? "is-active" : ""}
                                aria-pressed={contractCategory === id}
                                disabled={busy}
                                onClick={() => setContractCategory(id)}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="nd2-field">
                        <div className="nd2-field__head">
                          <span>Cap hit per year</span>
                          <strong>{formatMoney(offerAavNum)}</strong>
                        </div>
                        <div className="nd-range">
                          <input
                            type="range"
                            aria-label="Offer cap hit"
                            min={minSalaryM}
                            max={Math.min(Number(bonus.max_salary_m || 99), Math.max(12, offerAavNum * 1.35, Number(selected.ask_aav_m || selected.askingAav || 4) * 1.4))}
                            step="0.025"
                            value={Math.min(
                              Math.min(Number(bonus.max_salary_m || 99), Math.max(12, offerAavNum * 1.35, Number(selected.ask_aav_m || selected.askingAav || 4) * 1.4)),
                              Math.max(minSalaryM, offerAavNum || minSalaryM)
                            )}
                            disabled={busy}
                            onChange={(e) => setOfferAav(Number(e.target.value).toFixed(3))}
                          />
                        </div>
                      </div>
                      <div className="nd2-field nd2-field--row">
                        <span>Years</span>
                        <div className="nd-seg" role="group" aria-label="Contract years">
                          {termOptionsUfa.map((y) => (
                            <button
                              key={y}
                              type="button"
                              className={offerYearsNum === y ? "is-active" : ""}
                              aria-pressed={offerYearsNum === y}
                              disabled={busy}
                              onClick={() => setOfferYears(String(y))}
                            >
                              {y}
                            </button>
                          ))}
                        </div>
                      </div>
                      {isOfferSheet || offerAavNum > 1.15 ? null : (
                        <div className="nd2-field nd2-field--row">
                          <span>NHL roster</span>
                          <label className="nd2-note">
                            <input
                              type="checkbox"
                              checked={nhlRoster}
                              disabled={busy}
                              onChange={(e) => setNhlRoster(e.target.checked)}
                            />
                            {" "}Keep him up. Off, and a deal at $1.15M or less reports to the AHL. Waiver-exempt and two-way deals go straight down; anyone else has to clear waivers. The hit is $0 once he's down.
                          </label>
                        </div>
                      )}
                      {isOfferSheet ? null : (
                        <div className="nd2-field nd2-field--row">
                          <span>Protection</span>
                          <div className="nd-seg" role="group" aria-label="Contract clauses">
                            {[
                              ["none", "None", !offerNtc && !offerNmc],
                              ["ntc", "NTC", offerNtc && !offerNtcModified && !offerNmc],
                              ["mntc", "M-NTC", offerNtc && offerNtcModified && !offerNmc],
                              ["nmc", "NMC", offerNmc],
                            ].map(([id, label, on]) => (
                              <button
                                key={id}
                                type="button"
                                className={on ? "is-active" : ""}
                                aria-pressed={on}
                                disabled={busy}
                                onClick={() => {
                                  setOfferNmc(id === "nmc");
                                  setOfferNtc(id === "ntc" || id === "mntc" || id === "nmc");
                                  setOfferNtcModified(id === "mntc");
                                }}
                              >
                                {label}
                                {label !== "None" && deskPlayer.wants_clause === label ? <em>asked</em> : null}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="nd2-field">
                        <div className="nd2-field__head">
                          <span>Signing bonus</span>
                          <strong>{bonusAllowed ? formatMoney(offerBonusNum) : "Not available"}</strong>
                        </div>
                        {bonusAllowed ? (
                          <div className="nd-range">
                            <input
                              type="range"
                              aria-label="Signing bonus"
                              min={0}
                              max={bonusCapM}
                              step="0.025"
                              value={offerBonusNum}
                              disabled={busy}
                              onChange={(e) => setOfferBonus(Number(e.target.value).toFixed(3))}
                            />
                          </div>
                        ) : (
                          <p className="nd2-note">
                            {bonus.label || `Signing bonuses need club revenue of $${Math.round(Number(bonus.floor_m) || 155)}M or more.`}
                          </p>
                        )}
                      </div>
                    </section>

                    <section className="nd2-talk" aria-label="The market">
                      {competing.length ? (
                        <div className="nd2-market">
                          <span className="nd2-market__label">Offers on the table</span>
                          {competing.slice(0, 5).map((o) => (
                            <div key={`${o.team_id}-${o.aav_m}`} className={`nd2-market__row${o.is_user ? " is-you" : ""}`}>
                              <TeamLogoBadge
                                teamLogo={resolveFranchiseTeamLogo(
                                  { abbrev: o.team_abbrev, team_abbrev: o.team_abbrev, name: o.team_name },
                                  o.team_name || o.team_abbrev
                                )}
                                teamName={o.team_name || o.team_abbrev}
                                size={20}
                                variant="circle"
                              />
                              <strong>{o.is_user ? "You" : o.team_abbrev || o.team_name}</strong>
                              <em>{formatMoney(o.aav_m)} × {o.years}</em>
                            </div>
                          ))}
                        </div>
                      ) : interested.length ? (
                        <div className="nd2-market">
                          <span className="nd2-market__label">Clubs circling</span>
                          <div className="nd2-market__logos">
                            {interested.map((t) => (
                              <TeamLogoBadge
                                key={t.team_id || t.team_abbrev}
                                teamLogo={resolveFranchiseTeamLogo(
                                  { abbrev: t.team_abbrev, team_abbrev: t.team_abbrev, name: t.team_name },
                                  t.team_name || t.team_abbrev
                                )}
                                teamName={t.team_name || t.team_abbrev}
                                size={24}
                                variant="circle"
                              />
                            ))}
                          </div>
                        </div>
                      ) : (
                        <p className="nd2-note">No other club has made an offer yet.</p>
                      )}
                      <details className="nd2-more">
                        <summary>Why he'd sign here{isOfferSheet ? "" : ", and meeting options"}</summary>
                        <InterestFactors
                          factors={response?.evaluation?.interest_factors || selected.interest_factors}
                          fit={{
                            ...(response?.evaluation?.destination_fit || selected.destination_fit || {}),
                            fit_premium_pct: response?.evaluation?.fit_premium_pct,
                          }}
                        />
                        {isOfferSheet ? null : (
                          <NegotiationMeetingPanel
                            playerId={selected.player_id || selected.id}
                            onChanged={() => setFaMeetingRev((v) => v + 1)}
                          />
                        )}
                      </details>
                      <details className="nd2-more">
                        <summary>Scouting report</summary>
                        {faDetailLoading ? <p className="nd2-note">Loading Stats Central…</p> : null}
                        <dl className="nd2-stats">
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
                                <dt>{k}</dt>
                                <dd>{k === "WAR" ? fmtWar(v) : fmtStat(v, 0)}</dd>
                              </div>
                            ))}
                        </dl>
                        {percentiles.length ? (
                          <div className="nd2-pctiles">
                            <span className="nd2-market__label">Against {sc.pool_size || percentiles.length} peers</span>
                            {percentiles.map((row) => (
                              <div key={row.key || row.label} className="nd2-pctile">
                                <span>{row.label}</span>
                                <i><b style={{ width: `${Math.max(2, Number(row.percentile) || 0)}%` }} /></i>
                                <strong>{fmtPctile(row.percentile)}</strong>
                              </div>
                            ))}
                          </div>
                        ) : null}
                        {previousTeams.length ? (
                          <div className="nd2-history">
                            <span className="nd2-market__label">Previous teams</span>
                            {previousTeams.slice(0, 4).map((t, i) => {
                              const st = t.stats || {};
                              const line = st.is_goalie
                                ? `${st.gp ?? "—"} GP · ${st.save_pct != null ? Number(st.save_pct).toFixed(3) : "—"} SV%`
                                : `${st.gp ?? "—"} GP · ${st.pts ?? st.points ?? "—"} PTS${st.war != null ? ` · ${fmtWar(st.war)} WAR` : ""}`;
                              return (
                                <p key={`${t.team || "club"}-${i}`}>
                                  <strong>{t.team || t.league || "—"}</strong> {[t.label, t.league, t.season].filter(Boolean).join(" · ")} · {line}
                                </p>
                              );
                            })}
                          </div>
                        ) : null}
                      </details>
                    </section>
                  </div>

                  <footer className="nd2-foot">
                    <dl className="nd2-cap">
                      <div>
                        <dt>Cap hit</dt>
                        <dd>{formatMoney(offerCapHitNum)}</dd>
                      </div>
                      <div>
                        <dt>Space now</dt>
                        <dd>{formatMoney(capSpace)}</dd>
                      </div>
                      <div>
                        <dt>Space after</dt>
                        <dd className={projectedSpace != null && projectedSpace < 0 ? "is-bad" : ""}>
                          {projectedSpace != null ? formatMoney(projectedSpace) : "—"}
                        </dd>
                      </div>
                      <div className="nd2-cap__bar" title="Cap used after this offer">
                        <i
                          className={(capProj.capBarPct ?? 0) >= 100 ? "is-over" : (capProj.capBarPct ?? 0) >= 95 ? "is-tight" : ""}
                          style={{ width: `${Math.max(0, Math.min(100, capProj.capBarPct ?? 0))}%` }}
                        />
                      </div>
                    </dl>
                    <div className="nd2-actions">
                      <button
                        type="button"
                        className="nd-btn nd-btn--primary nd2-primary"
                        disabled={busy || (isOfferSheet && selected.offer_sheet_pending)}
                        onClick={isOfferSheet ? submitSheet : submitSign}
                      >
                        {busy ? "Working…" : isOfferSheet ? "Tender offer sheet" : "Submit offer"}
                      </button>
                    </div>
                  </footer>
                </div>
              )}
            </aside>
          </div>
        </div>
      }
      railContent={
        <div className={`${prefix}-rail-shell`}>
          <div className={`${prefix}-rail-tabs`} role="tablist">
            {[
              ["today", dayReport ? `Day ${dayReport.day ?? ""}` : "Signings"],
              ["wire", "Wire"],
              ["social", "Social"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={railTab === id}
                className={railTab === id ? "is-active" : ""}
                onClick={() => setRailTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {railTab === "today" ? (
            <div className={`${prefix}-rail-scroll`}>
              {dayReport ? (
                <>
                  {safeArray(dayReport.userLost).length ? (
                    <section className={`${prefix}-day-block is-lost`}>
                      <h4>Lost to another club</h4>
                      {safeArray(dayReport.userLost).map((x) => (
                        <div key={`lost-${x.player_id}`} className={`${prefix}-day-row is-lost`}>
                          <div className={`${prefix}-day-main`}>
                            <strong>{x.name}</strong>
                            <em>
                              Signed with {x.signed_team_name || x.signed_team_abbrev || "another club"}
                              {x.your_aav_m != null ? ` · your offer ${formatMoney(x.your_aav_m)} × ${x.your_years}y` : ""}
                            </em>
                          </div>
                          <span className={`${prefix}-day-terms`}>
                            {x.aav_m != null ? formatMoney(x.aav_m) : "—"}
                            <i>× {x.years || "?"}y</i>
                          </span>
                        </div>
                      ))}
                    </section>
                  ) : null}
                  {safeArray(dayReport.userSigned).length ? (
                    <section className={`${prefix}-day-block is-won`}>
                      <h4>Signed with you</h4>
                      {safeArray(dayReport.userSigned).map((x) => (
                        <FaSigningRow
                          key={`won-${x.player_id}`}
                          prefix={prefix}
                          tone="won"
                          row={{ ...x, team_name: "Your club", team_abbrev: franchiseState?.team?.abbreviation }}
                        />
                      ))}
                    </section>
                  ) : null}
                  {safeArray(dayReport.userRejected).length ? (
                    <section className={`${prefix}-day-block is-lost`}>
                      <h4>Turned you down</h4>
                      {safeArray(dayReport.userRejected).map((x) => (
                        <p key={`rej-${x.player_id}`} className={`${prefix}-day-note`}>
                          <strong>{x.name || x.player_id}</strong> — {x.feedback || x.reason || "declined"}
                        </p>
                      ))}
                    </section>
                  ) : null}
                  <section className={`${prefix}-day-block`}>
                    <h4>
                      {dayReport.days > 1 ? `Signings · last ${dayReport.days} days` : `Signings · day ${dayReport.day ?? ""}`}
                      <span>{safeArray(dayReport.signings).length}</span>
                    </h4>
                    {safeArray(dayReport.signings).length ? (
                      safeArray(dayReport.signings).map((x, i) => (
                        <FaSigningRow key={`sig-${x.player_id}-${i}`} prefix={prefix} row={x} />
                      ))
                    ) : (
                      <p className={`${prefix}-empty`}>Quiet day — nobody signed.</p>
                    )}
                  </section>
                </>
              ) : (
                <section className={`${prefix}-day-block`}>
                  <h4>
                    Latest signings <span>{recent.length}</span>
                  </h4>
                  {recent.length ? (
                    [...recent].reverse().map((x, i) => (
                      <FaSigningRow key={`recent-${x.player_id}-${i}`} prefix={prefix} row={x} />
                    ))
                  ) : (
                    <p className={`${prefix}-empty`}>Sim Day to see who signs where.</p>
                  )}
                </section>
              )}
            </div>
          ) : null}

          {railTab === "wire" ? (
            <div className={`${prefix}-rail-scroll`}>
              {dedupeWire(news.length ? news : recent).slice(-24).reverse().map((n, i) => (
                <p key={`${n.text || n.player_id || "wire"}-${i}`} className={`${prefix}-wire-item is-${n.kind || "market"}`}>
                  {n.text ||
                    [
                      n.team_name || n.team_abbrev || "A club",
                      n.name || n.player_id,
                      n.aav_m != null ? `${n.aav_m}M × ${n.years || "?"}y` : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                </p>
              ))}
              {!news.length && !recent.length ? (
                <p className={`${prefix}-empty`}>Submit offers or Sim Day — signings and pending decisions land here.</p>
              ) : null}
            </div>
          ) : null}

          {railTab === "social" ? (
            <div className={`${prefix}-rail-scroll`}>
              {social.loading && !social.posts.length ? <p className={`${prefix}-empty`}>Loading the feed…</p> : null}
              {social.error ? <p className={`${prefix}-warn`}>{social.error}</p> : null}
              {social.posts.map((post, i) => (
                <FaSocialPost key={post.id || `post-${i}`} prefix={prefix} post={post} />
              ))}
              {!social.loading && !social.error && !social.posts.length ? (
                <p className={`${prefix}-empty`}>No chatter yet — Sim Day and the league will react to signings.</p>
              ) : null}
            </div>
          ) : null}
        </div>
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
            Active roster {nhlCount ?? "—"} (needs {nhlMin}–{nhlMax})
          </span>
          {[
            ["Forwards", forwards, minF],
            ["Defence", defense, minD],
            ["Goalies", goalies, minG],
          ].map(([label, have, need]) => (
            <span key={label} className={`${prefix}-chip${have != null && have < need ? " is-blocked" : ""}`}>
              {label} {have ?? "—"} (needs {need})
            </span>
          ))}
          {irCount != null || ltirCount != null ? (
            <span className={`${prefix}-chip`}>
              IR {irCount ?? 0} · LTIR {ltirCount ?? 0}
            </span>
          ) : null}
          <span className={`${prefix}-chip`}>Space {formatMoney(raw.cap_space_m)}</span>
          <span className={`${prefix}-chip`}>
            Contracts {raw.contract_slots_used ?? "—"} of {raw.contract_slots_limit ?? "—"}
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
                    ? `Resolve in ${item.route === "free_agency" ? "Free Agency" : humanizeLabel(item.route)}`
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
  // Skip markers without a name; an untitled card says nothing.
  const markers = safeArray(raw.calendar_markers).filter((m) => m && (m.label || m.key)).slice(0, 6);

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
              title={m.label || String(m.key).replace(/_/g, " ")}
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

/** "OFFENSIVE D" -> "Offensive D": all-caps data reads as shouting next to sentence case. */
function sentenceLabel(text) {
  const t = String(text || "").trim();
  if (!t || t !== t.toUpperCase()) return t || null;
  return t.toLowerCase().replace(/\b([a-z])/g, (c) => c.toUpperCase()).replace(/\b(D|G|C|Lw|Rw)\b/g, (m) => m.toUpperCase());
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
  const ppgStat = stats.find(([label]) => label === "PPG");
  const otherStats = stats.filter(([label]) => label !== "PPG");


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
          <div className={`${prefix}-workspace dr`}>
            <section className="dr-verdict" ref={haulRef} aria-label="Draft class verdict">
              <div className={`dr-verdict__grade tone-${gradeTone(gradeLetter(haulGrade))}`}>
                <strong>{haulGrade || "—"}</strong>
                <span>Class grade</span>
              </div>
              <div className="dr-verdict__copy">
                <strong>{haulLabel || "Draft class review"}</strong>
                {haulExplain ? <p>{haulExplain}</p> : null}
                {classChips.length ? <p className="dr-verdict__mix">{classChips.slice(0, 5).join(" · ")}</p> : null}
              </div>
              {bestValueName || closestName ? (
                <dl className="dr-verdict__facts">
                  {bestValueName ? (
                    <div><dt>Best value</dt><dd>{bestValueName}</dd></div>
                  ) : null}
                  {closestName ? (
                    <div><dt>Closest to the NHL</dt><dd>{closestName}</dd></div>
                  ) : null}
                </dl>
              ) : null}
            </section>

            <div className="dr-main">
              <section className="dr-pick" ref={fitRef} aria-label="Selected pick">
                <div className="dr-pick__id">
                  <div className={`dr-pick__mark pos-${posGroup}`}>
                    <strong>{pos || "—"}</strong>
                    <span>{pick.round != null ? `Round ${pick.round}` : "Pick"}</span>
                  </div>
                  <div className="dr-pick__copy">
                    <p className="dr-pick__slot">
                      {[
                        pick.overall_pick != null ? `Pick ${pick.overall_pick}` : null,
                        pick.board_range ? (/^board/i.test(String(pick.board_range)) ? String(pick.board_range) : `Board ${pick.board_range}`) : null,
                        pick.selection_delta_label,
                      ].filter(Boolean).join(" · ")}
                    </p>
                    <h2>{pick.prospect_name || getPlayerName(pick)}</h2>
                    <p>
                      {[
                        sentenceLabel(humanizeLabel(pick.archetype || pick.player_type)),
                        pick.age != null ? `${pick.age}` : null,
                        pick.shoots ? `shoots ${/^r/i.test(String(pick.shoots)) ? "right" : /^l/i.test(String(pick.shoots)) ? "left" : String(pick.shoots).toLowerCase()}` : null,
                        [pick.height, pick.weight != null ? `${pick.weight} lb` : null].filter(Boolean).join(" "),
                      ].filter(Boolean).join(" · ")}
                    </p>
                    <p className="dr-pick__club">{[pick.club, pick.league, pick.nationality].filter(Boolean).join(" · ")}</p>
                  </div>
                </div>
                {pickGrade || pick.selection_grade_label || oneLineVerdict ? (
                  <div className={`dr-pick__verdict tone-${gradeTone(pickGrade)}`}>
                    {pickGrade ? <strong className="dr-pick__grade">{pickGrade}</strong> : null}
                    <div>
                      <span>{[pick.selection_grade_label, pick.selection_verdict].filter(Boolean).join(" · ") || "Pick verdict"}</span>
                      {oneLineVerdict && ![pick.selection_grade_label, pick.selection_verdict].includes(oneLineVerdict) ? <p>{oneLineVerdict}</p> : null}
                    </div>
                  </div>
                ) : null}
                <div className="dr-observed">
                  <p className="dr-label">
                    {production.mode === "scouting" ? "Scouting read" : "This season · observed"}
                  </p>
                  <div className="dr-observed__row">
                    {ppgStat ? (
                      <div className="dr-observed__hero">
                        <strong>{formatProductionStat(ppgStat[0], ppgStat[1])}</strong>
                        <span>points per game</span>
                      </div>
                    ) : production.headline ? (
                      <p className="dr-observed__headline">{production.headline}</p>
                    ) : null}
                    {otherStats.slice(0, 4).map(([label, value]) => (
                      <div key={label} className="dr-observed__stat">
                        <strong>{formatProductionStat(label, value)}</strong>
                        <span>{label === "GP" ? "games" : label === "G" ? "goals" : label === "A" ? "assists" : label === "PTS" ? "points" : label}</span>
                      </div>
                    ))}
                  </div>
                  {prodContext ? <p className="dr-note">{prodContext}</p> : null}
                </div>
              </section>

              <section className="dr-proj" ref={planRef} aria-label="Projection">
                <p className="dr-label">Projection · scouts' estimate</p>
                {plan.nhl_projection || fit.fit_label || pick.selection_verdict ? (
                  <p className="dr-proj__role">{plan.nhl_projection || fit.fit_label || pick.selection_verdict}</p>
                ) : null}
                {plan.eta_range ? <p className="dr-proj__eta">NHL in {plan.eta_range}</p> : null}
                {pathStages.length ? (
                  <ol className="dr-road" aria-label="Development path">
                    {pathStages.map((step) => (
                      <li key={step.key} className={`is-${step.status}`}>
                        <span className="dr-road__when">
                          {step.status === "next" ? "Now" : step.status === "projection" ? "Goal" : "Next"}
                        </span>
                        <strong>{step.label}</strong>
                        {step.detail ? <span className="dr-road__what">{step.detail}</span> : null}
                      </li>
                    ))}
                  </ol>
                ) : nextHero ? (
                  <p className="dr-proj__eta">Next: {nextHero}</p>
                ) : null}
                {[plan.recommended_role, plan.minutes_target, plan.special_teams_role].filter(Boolean).length ? (
                  <p className="dr-note">
                    {[plan.recommended_role, plan.minutes_target, plan.special_teams_role].filter(Boolean).join(" · ")}
                  </p>
                ) : null}
                <ul className="dr-reads" aria-label="How sure the projection is">
                  {[
                    ["Projection", plan.nhl_projection_confidence, false],
                    ["Scouting", pick.scouting_confidence_label, false],
                    ["Timeline", plan.eta_confidence, false],
                    ["Development risk", pick.risk_level, true],
                  ]
                    .filter(([, v]) => v)
                    .map(([label, value, inverse]) => {
                      const lvl = { high: 3, med: 2, low: 1 }[riskTone(value)] || 0;
                      const tone = inverse ? (lvl >= 3 ? "down" : lvl === 2 ? "mid" : "up") : (lvl >= 3 ? "up" : lvl === 2 ? "mid" : "down");
                      return (
                        <li key={label} className={`is-${tone}`}>
                          <span>{label}</span>
                          <i aria-hidden="true">{[1, 2, 3].map((n) => <b key={n} className={n <= lvl ? "is-on" : ""} />)}</i>
                          <em>{String(value)}</em>
                        </li>
                      );
                    })}
                </ul>
              </section>
            </div>

            {(pros.length || cons.length || pick.risk_reason || hasOrgPlan) ? (
              <section className="dr-case" ref={rightsRef} aria-label="Evidence and fit">
                <div className="dr-evidence">
                  {(pros.length ? pros : neutralNotes.slice(0, 2)).map((item) => (
                    <p key={`pro-${item.text}`} className="is-up"><b aria-hidden="true">+</b>{item.text}</p>
                  ))}
                  {(cons.length ? cons : pick.risk_reason ? [{ text: pick.risk_reason }] : []).map((item) => (
                    <p key={`con-${item.text}`} className="is-down"><b aria-hidden="true">{"−"}</b>{item.text}</p>
                  ))}
                </div>
                {hasOrgPlan ? (
                  <div className="dr-fit">
                    <p className="dr-label">Fit with your club</p>
                    <p className="dr-fit__line">
                      {[fit.need_filled ? `Fills ${String(fit.need_filled).toLowerCase()}` : null, [fit.fit_grade, fit.fit_label].filter(Boolean).join(" ")].filter(Boolean).join(" · ") || "Fit not graded"}
                    </p>
                    {hasDepth ? (
                      <p className="dr-fit__depth">
                        {[
                          fit.nhl_players_ahead != null ? `${fit.nhl_players_ahead} NHL` : null,
                          fit.ahl_players_ahead != null ? `${fit.ahl_players_ahead} AHL` : null,
                          fit.prospects_ahead != null ? `${fit.prospects_ahead} prospects` : null,
                        ].filter(Boolean).join(" · ")} ahead at {pos || "his position"}
                      </p>
                    ) : null}
                    {[fit.depth_status || fit.path_congestion, fit.environment_grade ? `${fit.environment_grade} environment` : null].filter(Boolean).length ? (
                      <p className="dr-note">{[fit.depth_status || fit.path_congestion, fit.environment_grade ? `${fit.environment_grade} environment` : null].filter(Boolean).join(" · ")}</p>
                    ) : null}
                    {rights.rights_status_label || rights.rights_deadline_label || rights.signing_recommendation ? (
                      <p className="dr-fit__rights">
                        {[rights.rights_status_label, rights.signing_recommendation, rights.rights_deadline_label].filter(Boolean).join(" · ")}
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </section>
            ) : (
              <div ref={rightsRef} />
            )}

            {advancedLines.length ? (
              <div className="dr-more">
                <button type="button" aria-expanded={showMore} onClick={() => setShowMore((v) => !v)}>
                  {showMore ? "Hide scouting detail" : "Scouting detail"}
                </button>
                {showMore ? (
                  <div className="dr-more__panel">
                    {[developReason, ...advancedLines]
                      .filter(Boolean)
                      .filter((line, i, arr) => arr.indexOf(line) === i)
                      .slice(0, 8)
                      .map((line) => <p key={line}>{line}</p>)}
                  </div>
                ) : null}
              </div>
            ) : null}
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

/* ── Rights Desk helpers (dossier-style layout) ── */
const PR_LEAGUE_ACRONYMS = /\b(ncaa|chl|ohl|whl|qmjhl|ushl|ahl|echl|shl|khl|nhl|vhl|mhl|bchl|ajhl|nll|u18|u20)\b/gi;
function prLeagueName(raw) {
  const s = String(raw || "").replace(/^CHL_/, "").replace(/_/g, " ").trim();
  if (!s) return "";
  return s
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(PR_LEAGUE_ACRONYMS, (m) => m.toUpperCase())
    .replace(/\bNCAA College\b/, "NCAA");
}
function prLeague(p) {
  const raw = p?.current_league || p?.returning_to || p?.league || p?.season_stats?.league || "";
  return prLeagueName(raw);
}
function prPositionName(pos) {
  const P = String(pos || "").toUpperCase();
  return { C: "Center", LW: "Left wing", RW: "Right wing", D: "Defense", LD: "Defense", RD: "Defense", G: "Goalie" }[P] || P;
}
function prDraftLine(p) {
  const bits = [];
  if (p?.draft_year) bits.push(`Drafted ${p.draft_year}`);
  if (p?.draft_round) bits.push(`Round ${p.draft_round}`);
  const pick = p?.draft_overall_pick ?? p?.draft_pick;
  if (pick) bits.push(`#${pick} overall`);
  const ys = Number(p?.years_since_drafted);
  if (Number.isFinite(ys) && ys >= 1) bits.push(`${ys} year${ys === 1 ? "" : "s"} in the system`);
  return bits.join(" · ") || "Unsigned prospect";
}
function prTierKey(pot) {
  const n = Number(pot);
  if (!Number.isFinite(n)) return "none";
  return n >= 88 ? "elite" : n >= 84 ? "high" : n >= 79 ? "mid" : "low";
}
function prTier(pot, pos) {
  const n = Number(pot);
  if (!Number.isFinite(n)) return null;
  const P = String(pos || "").toUpperCase();
  const grp = P.includes("G") ? "G" : P.startsWith("D") || P === "LD" || P === "RD" ? "D" : "F";
  const names = {
    F: ["First-line ceiling", "Top-six ceiling", "Middle-six ceiling", "Depth ceiling"],
    D: ["Top-pair ceiling", "Top-four ceiling", "Third-pair ceiling", "Depth ceiling"],
    G: ["Franchise starter", "Starter ceiling", "Backup ceiling", "Depth ceiling"],
  }[grp];
  return names[n >= 88 ? 0 : n >= 84 ? 1 : n >= 79 ? 2 : 3];
}
function prDeadlineYear(p) {
  const d = String(p?.rights_signing_deadline || p?.rights_through || p?.rights_expiry_year || "");
  const m = d.match(/(\d{4})/);
  return m ? Number(m[1]) : null;
}
function prDeadlineUrgent(p, seasonYear) {
  const y = prDeadlineYear(p);
  return y != null && seasonYear != null && y <= Number(seasonYear) + 1;
}
function prDate(raw) {
  const m = String(raw || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return String(raw || "—");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[Number(m[2]) - 1]} ${Number(m[3])}, ${m[1]}`;
}
const PR_PATH_BLURBS = {
  keep_unsigned: "Hold his rights. No contract, no slot used.",
  delay: "Come back to him before this offseason ends.",
  allow_expire: "Let his rights lapse. He becomes a free agent.",
};
function prPathBlurb(pkg) {
  return PR_PATH_BLURBS[pkg?.actionId] || pkg?.blurb || "";
}
function prMeterTone(pct) {
  const n = Number(pct);
  if (!Number.isFinite(n)) return "none";
  return n >= 75 ? "good" : n >= 50 ? "mid" : "bad";
}
function prMoneyM(display) {
  const m = String(display || "").replace(/,/g, "").match(/\$?\s*([\d.]+)\s*([KkMm])?/);
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isFinite(n)) return null;
  const unit = (m[2] || "").toUpperCase();
  return unit === "M" ? n : unit === "K" ? n / 1000 : n / 1e6;
}
function prMoneyLabel(m) {
  const n = Number(m) || 0;
  return n >= 1 ? `$${n.toFixed(2)}M` : `$${Math.round(n * 1000)}K`;
}
function PrFlag({ country }) {
  const iso = resolveCountryCode(country);
  const src = iso ? `https://flagcdn.com/w80/${iso.toLowerCase()}.png` : flagApiUrl(country, 32);
  if (!src) return null;
  return <img src={src} alt="" onError={(e) => { e.currentTarget.style.display = "none"; }} />;
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
  const warnings = safeArray(payload?.warning_reasons);
  const seasonYear = payload?.season_year || franchiseState?.season_year || franchiseState?.seasonYear;
  const slots = parseContractSlots(payload);

  const sortedProspects = React.useMemo(() => {
    return [...prospects].sort((a, b) => {
      const oa = Number(a?.overall ?? a?.ovr ?? 0);
      const ob = Number(b?.overall ?? b?.ovr ?? 0);
      if (ob !== oa) return ob - oa;
      const pa = Number(a?.potential ?? a?.potential_score ?? 0);
      const pb = Number(b?.potential ?? b?.potential_score ?? 0);
      if (pb !== pa) return pb - pa;
      const ea = Number(a?.rights_through ?? a?.rights_expiry_year ?? 9999);
      const eb = Number(b?.rights_through ?? b?.rights_expiry_year ?? 9999);
      return ea - eb;
    });
  }, [prospects]);
  const expiringSoon = sortedProspects.filter((p) => {
    const through = Number(p?.rights_through ?? p?.rights_expiry_year);
    return Number.isFinite(through) && seasonYear != null && through <= Number(seasonYear) + 1;
  });
  const [leaveArmed, setLeaveArmed] = React.useState(false);
  const handleLeave = () => {
    if (expiringSoon.length && !leaveArmed) {
      setLeaveArmed(true);
      setError(
        `Warning: ${expiringSoon.length} prospect${expiringSoon.length === 1 ? "" : "s"} have rights expiring this offseason. Continue again to leave.`
      );
      return;
    }
    onContinue?.();
  };

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
  const packages = React.useMemo(() => buildDecisionPackages(focus), [focus]);
  const selectedPackage =
    packages.find((p) => p.packageId === selectedPackageId) ||
    packages.find((p) => p.recommended && !p.disabled) ||
    packages.find((p) => !p.disabled) ||
    null;
  const isSigning = selectedPackage?.kind === "elc";
  const [assignmentPlanOverride, setAssignmentPlanOverride] = React.useState(null);
  // Negotiated ELC bonus money ($M). null = the package's template amounts.
  const [elcBonus, setElcBonus] = React.useState({ sb: null, sa: null });
  const elcLegal = focus?.legal_elc_terms || {};
  React.useEffect(() => {
    setElcBonus({ sb: null, sa: null });
  }, [selectedPackage?.packageId]);
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
    setElcBonus({ sb: null, sa: null });
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
          signing_bonus_total_m: elcBonus.sb ?? undefined,
          schedule_a_bonus_m: elcBonus.sa ?? undefined,
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
  }, [focus?.player_id, selectedPackage?.packageId, selectedPackage?.templateId, effectiveAssignmentPlan, elcBonus.sb, elcBonus.sa]);

  const offer = preview?.offer || null;
  const acceptance = preview?.acceptance || focus?.elc_acceptance_summary || null;
  const signingResult = preview?.signing_result || null;
  const slotPreview = preview?.slot_preview || null;
  const ovr = focus?.overall != null ? focus.overall : getPlayerOverall(focus);
  const pot =
    focus?.potential != null
      ? focus.potential
      : focus?.potential_score != null
        ? focus.potential_score
        : focus?.pot != null
          ? focus.pot
          : null;

  const sbMax = Number(elcLegal.max_signing_bonus_total_m || 0.285) * (Number(offer?.term_years || 3) / 3);
  const saMax = Number(elcLegal.max_schedule_a_m || 0.2125);

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
          signing_bonus_total_m: elcBonus.sb ?? undefined,
          schedule_a_bonus_m: elcBonus.sa ?? undefined,
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
        // The agent's counter names numbers: load them on the desk to accept or haggle.
        if (result?.counter_offer?.signing_bonus_total_m != null) {
          setElcBonus({
            sb: Number(result.counter_offer.signing_bonus_total_m),
            sa: Number(result.counter_offer.schedule_a_bonus_m ?? elcBonus.sa ?? 0),
          });
        }
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
      onContinue={handleLeave}
      onBack={onBack}
      railTitle="Unsigned prospects"
      railHint={prospects.length ? "↑↓ switch" : null}
      rootClassName="pr-desk"
      hideTicker
      heroContent={
        focus ? (
          <div className={`${prefix}-workspace pr pr2`}>
            {warnings[0] ? <p className="pr2-alert" role="status">{warnings[0]}</p> : null}

            {/* ── who he is: dossier paper card ── */}
            <header className="pr2-card">
              <div className="pr2-face">
                <PlayerHeadshot player={{ ...focus, id: focus.player_id }} size="lg" showFlag={false} />
              </div>
              <div className="pr2-who">
                <span className="pr2-tag">{prDraftLine(focus)}</span>
                <h3 className="pr2-name">
                  {focus.nationality || focus.country ? (
                    <span className="pr2-flag">
                      <PrFlag country={focus.nationality || focus.country} />
                    </span>
                  ) : null}
                  {focus.name || getPlayerName(focus)}
                </h3>
                <p className="pr2-line">
                  {[
                    prPositionName(focus.position || getPlayerPosition(focus)),
                    focus.age != null ? `Age ${focus.age}` : null,
                    prLeague(focus),
                  ].filter(Boolean).join(" · ")}
                </p>
                {prTier(pot, focus.position || getPlayerPosition(focus)) ? (
                  <span className="pr2-stamp">{prTier(pot, focus.position || getPlayerPosition(focus))}</span>
                ) : null}
              </div>
              <div className="pr2-pot" aria-label="Current and ceiling">
                <div>
                  <small>Now</small>
                  <strong>{ovr ?? "—"}</strong>
                </div>
                <i aria-hidden="true" />
                <div className="is-ceiling">
                  <small>Ceiling</small>
                  <strong>{pot != null ? Math.round(Number(pot)) : "—"}</strong>
                </div>
              </div>
              <div className={`pr2-rights${prDeadlineUrgent(focus, seasonYear) ? " is-urgent" : ""}`}>
                <div>
                  <small>Rights held through</small>
                  <strong>{focus.rights_through ?? "—"}</strong>
                </div>
                {focus.rights_signing_deadline ? (
                  <div>
                    <small>Sign by</small>
                    <strong>{prDate(focus.rights_signing_deadline)}</strong>
                    <em>{prDeadlineUrgent(focus, seasonYear) ? "or lose him" : "no rush yet"}</em>
                  </div>
                ) : null}
              </div>
            </header>

            <div className="pr2-body">
              {/* ── the choice ── */}
              <section className="pr2-choose" aria-label="Your decision">
                {sectionPackages("sign").length ? (
                  <>
                    <h4 className="pr2-h">Sign him to an entry-level deal</h4>
                    <div className="pr2-elcs" role="radiogroup" aria-label="Entry-level contracts">
                      {sectionPackages("sign").map((pkg) => {
                        const on = selectedPackage && pkg.packageId === selectedPackage.packageId;
                        return (
                          <button
                            key={pkg.packageId}
                            type="button"
                            role="radio"
                            aria-checked={Boolean(on)}
                            disabled={Boolean(pkg.disabled) || busy}
                            className={`pr2-elc${on ? " is-on" : ""}${pkg.recommended ? " is-rec" : ""}`}
                            title={pkg.disabled ? pkg.action?.blocked_reason || pkg.blurb : pkg.blurb}
                            onClick={() => {
                              if (pkg.disabled) return;
                              setSelectedPackageId(pkg.packageId);
                              setError("");
                              setFeedback("");
                            }}
                          >
                            {pkg.recommended ? <span className="pr2-elc__rec">Recommended</span> : null}
                            <strong className="pr2-elc__name">{pkg.title}</strong>
                            <span className="pr2-elc__bonus">
                              <small>Signing bonus</small>
                              <b>{pkg.signingBonusDisplay || "$0"}</b>
                            </span>
                            <span className="pr2-elc__terms">
                              {[pkg.termYears != null ? `${pkg.termYears} years` : null, pkg.aavDisplay ? `${pkg.aavDisplay} cap hit` : null]
                                .filter(Boolean)
                                .join(" · ")}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </>
                ) : null}

                {sectionPackages("development").length || sectionPackages("rights").length ? (
                  <>
                    <h4 className="pr2-h">Or don't sign him yet</h4>
                    <div className="pr2-paths" role="radiogroup" aria-label="Other decisions">
                      {[...sectionPackages("development"), ...sectionPackages("rights")].map((pkg) => {
                        const on = selectedPackage && pkg.packageId === selectedPackage.packageId;
                        return (
                          <button
                            key={pkg.packageId}
                            type="button"
                            role="radio"
                            aria-checked={Boolean(on)}
                            disabled={Boolean(pkg.disabled) || busy}
                            className={`pr2-path${on ? " is-on" : ""}${pkg.actionId === "allow_expire" ? " is-danger" : ""}`}
                            onClick={() => {
                              if (pkg.disabled) return;
                              setSelectedPackageId(pkg.packageId);
                              setError("");
                              setFeedback("");
                            }}
                          >
                            <strong>{pkg.title}</strong>
                            <small>{pkg.disabled ? pkg.action?.blocked_reason || pkg.blurb : prPathBlurb(pkg)}</small>
                          </button>
                        );
                      })}
                    </div>
                  </>
                ) : null}

                {focus.season_stats || focus.key_stats ? (
                  <p className="pr2-stats">
                    {focus.season_stats
                      ? `This season${focus.season_stats.league ? ` in the ${prLeagueName(focus.season_stats.league)}` : ""}: ${
                          focus.season_stats.save_pct != null && focus.position === "G"
                            ? `${focus.season_stats.gp} GP, ${focus.season_stats.wins ?? 0} W, ${Number(focus.season_stats.save_pct).toFixed(3)} save %`
                            : `${focus.season_stats.gp} GP · ${focus.season_stats.goals ?? 0} G · ${focus.season_stats.assists ?? 0} A · ${focus.season_stats.points ?? 0} P`
                        }`
                      : `Season line: ${focus.key_stats.gp ?? 0} GP · ${focus.key_stats.goals ?? 0} G · ${focus.key_stats.assists ?? 0} A · ${focus.key_stats.points ?? 0} P`}
                  </p>
                ) : null}
              </section>

              {/* ── the offer: meter sits on top of the controls that move it ── */}
              <aside className="pr2-offer" aria-label="Offer">
                {isSigning ? (
                  <>
                    <div className={`pr2-meter tone-${prMeterTone(acceptance?.acceptance_pct)}`} aria-label="Chance he signs">
                      <div className="pr2-meter__top">
                        <strong>{acceptance?.acceptance_pct != null ? `${Math.round(acceptance.acceptance_pct)}%` : "—"}</strong>
                        <span>
                          <b>Chance he signs</b>
                          <small>{acceptance?.outlook_label || "Bonus money and a clear path raise it"}</small>
                        </span>
                      </div>
                      <div className="pr2-meter__track">
                        <i style={{ width: `${Math.max(3, Math.min(100, Number(acceptance?.acceptance_pct) || 0))}%` }} />
                        <span style={{ left: "50%" }} aria-hidden="true" />
                        <span style={{ left: "75%" }} aria-hidden="true" />
                      </div>
                    </div>

                    <div className="pr2-field">
                      <div className="pr2-field__head">
                        <span>Signing bonus</span>
                        <strong>{offer?.signing_bonus_display || selectedPackage?.signingBonusDisplay || "$0"}</strong>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={sbMax}
                        step={0.005}
                        value={Math.min(sbMax, Number(elcBonus.sb ?? offer?.signing_bonus_total_m ?? prMoneyM(selectedPackage?.signingBonusDisplay) ?? 0))}
                        onChange={(e) => setElcBonus((b) => ({ ...b, sb: Number(e.target.value) }))}
                        disabled={busy}
                        aria-label="Signing bonus"
                      />
                      <div className="pr2-field__scale"><span>$0</span><span>{prMoneyLabel(sbMax)} max</span></div>
                    </div>
                    <div className="pr2-field">
                      <div className="pr2-field__head">
                        <span>Performance bonuses</span>
                        <strong>{offer?.schedule_a_display || selectedPackage?.scheduleA || "$0"}</strong>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={saMax}
                        step={0.005}
                        value={Math.min(saMax, Number(elcBonus.sa ?? offer?.schedule_a_bonus_m ?? prMoneyM(selectedPackage?.scheduleA) ?? 0))}
                        onChange={(e) => setElcBonus((b) => ({ ...b, sa: Number(e.target.value) }))}
                        disabled={busy}
                        aria-label="Performance bonuses"
                      />
                      <div className="pr2-field__scale"><span>$0</span><span>{prMoneyLabel(saMax)} max</span></div>
                    </div>
                    <p className="pr2-hint">
                      More bonus money raises his chance of signing. Performance bonuses only count against the cap if he earns them
                      {preview?.cap_preview?.bonus_exposure_m != null
                        ? ` (up to $${Math.round(Number(preview.cap_preview.bonus_exposure_m) * 1000)}K this season).`
                        : "."}
                    </p>

                    {assignmentOptions.length ? (
                      <label className="pr2-assign">
                        <span>Where he plays</span>
                        <select
                          value={effectiveAssignmentPlan || ""}
                          onChange={(e) => setAssignmentPlanOverride(e.target.value)}
                          disabled={busy}
                        >
                          {assignmentOptions.map((opt) => (
                            <option key={opt.id} value={opt.id} disabled={!opt.enabled}>
                              {prLeague({ current_league: opt.label }) || opt.label}
                              {!opt.enabled && opt.blocked_reason ? ` (${opt.blocked_reason})` : ""}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : null}
                  </>
                ) : (
                  <div className="pr2-decision">
                    <small>Your decision</small>
                    <strong>{selectedPackage?.title || "Pick an option"}</strong>
                    <p>{selectedPackage ? prPathBlurb(selectedPackage) : ""}</p>
                  </div>
                )}

                <dl className="pr2-sum">
                  <div>
                    <dt>Cap hit</dt>
                    <dd>{isSigning ? preview?.signing_result?.cap_impact_display || offer?.aav_display || "—" : "No change"}</dd>
                  </div>
                  <div>
                    <dt>Contracts</dt>
                    <dd>
                      {slotPreview
                        ? isSigning
                          ? `${slotPreview.contract_slots_used} → ${slotPreview.after_signing} / ${slotPreview.contract_slots_limit}`
                          : `${slotPreview.contract_slots_used} / ${slotPreview.contract_slots_limit}`
                        : slots.used != null
                          ? `${slots.used} / ${slots.limit}`
                          : payload?.contracts || "—"}
                    </dd>
                  </div>
                  {isSigning && signingResult ? (
                    <div>
                      <dt>Plays in</dt>
                      <dd>{prLeague({ current_league: humanizeLabel(signingResult.assignment) }) || humanizeLabel(signingResult.assignment)}</dd>
                    </div>
                  ) : null}
                </dl>

                {error ? <p className="pr2-error" role="alert">{error}</p> : null}
                {feedback ? <p className="pr2-ok">{feedback}</p> : null}
                <button
                  type="button"
                  className="nhlcal-advance-button pr2-primary"
                  disabled={busy || !selectedPackage || selectedPackage.disabled}
                  onClick={applyDecision}
                >
                  {applyLabel}
                </button>
              </aside>
            </div>
          </div>
        ) : (
          <div className={`${prefix}-workspace pr2`}>
            <p className="pr2-empty">No unsigned prospects on file. Continue to Re-Sign.</p>
          </div>
        )
      }
      railContent={
        sortedProspects.length ? (
          sortedProspects.map((p, i) => {
            const id = String(p.player_id || i);
            const active = String(p.player_id || "") === String(selectedId);
            const rowOvr = p.overall != null ? p.overall : null;
            const rowPos = p.position || getPlayerPosition(p) || "—";
            const urgent = prDeadlineUrgent(p, seasonYear);
            return (
              <button
                key={id}
                type="button"
                className={`pr2-row${active ? " is-active" : ""}${urgent ? " is-urgent" : ""}`}
                ref={(node) => {
                  if (p.player_id != null) railItemRefs.current[String(p.player_id)] = node;
                }}
                onClick={() => setSelectedId(String(p.player_id))}
                onKeyDown={(e) => onRailKeyDown(e, p.player_id)}
                aria-pressed={active}
              >
                <span className="pr2-row__pos">{rowPos}</span>
                <span className="pr2-row__who">
                  <strong>{p.name || getPlayerName(p)}</strong>
                  <small>
                    {[p.age != null ? `Age ${p.age}` : null, (p.rights_through || p.rights_expiry_year) ? `Rights ${p.rights_through || p.rights_expiry_year}` : null]
                      .filter(Boolean)
                      .join(" · ")}
                  </small>
                </span>
                <span className="pr2-row__nums">
                  <b>{rowOvr ?? "—"}</b>
                  <i>→</i>
                  <b className={`is-pot tier-${prTierKey(p.potential)}`}>{p.potential ?? "—"}</b>
                </span>
              </button>
            );
          })
        ) : (
          <p className="pr2-empty">No unsigned prospects</p>
        )
      }
    />
    </>
  );
}
