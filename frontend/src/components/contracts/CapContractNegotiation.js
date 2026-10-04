import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  advanceContractNegotiationDay,
  evaluateContractOffer,
  reSignContract,
} from "../../services/franchiseService";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import {
  computeOfferCapHitM,
  estimateOfferInterestM,
  interestMeterTone,
  projectNegotiationCap,
  resignInterestLabel,
} from "../../utils/contractNegotiation";
import {
  agentDifficultyIcon,
  agentDifficultyTone,
  resolveAgentDealDifficulty,
} from "../../utils/playerAgentDisplay";
import NegotiationMeetingPanel from "./NegotiationMeetingPanel";

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function formatMoneyM(v) {
  const n = safeNum(v, NaN);
  if (!Number.isFinite(n)) return "—";
  return `$${n.toFixed(1)}M`;
}

function safeText(v, fallback) {
  const s = v == null ? "" : String(v).trim();
  return s || fallback;
}

function defaultAsk(row) {
  const ext = row?.extension_estimate || {};
  const aav =
    safeNum(row?.player_ask_aav_m, NaN) ||
    safeNum(ext.likelyAav, NaN) ||
    safeNum(row?.aav_m, NaN) * 1.05 ||
    1.0;
  const years =
    safeNum(row?.requested_term, NaN) ||
    safeNum(ext.likelyTerm, NaN) ||
    safeNum(row?.extension_years_remaining, NaN) ||
    3;
  return { aav, years: Math.max(1, Math.round(years)) };
}

function readEvaluation(response) {
  if (!response) {
    return {
      interest: NaN,
      acceptCut: NaN,
      agentMood: "",
      feedback: "",
      projectedAfter: null,
      projectedNext: null,
      status: "",
    };
  }
  const ev = response?.evaluation || {};
  const pr = response?.player_response || {};
  return {
    interest: safeNum(ev.interest ?? pr.interest, NaN),
    acceptCut: safeNum(ev.accept_cut ?? pr.accept_cut, NaN),
    agentMood: pr.agent_mood || ev.agent_mood || "",
    feedback: pr.feedback || ev.reason || response?.reason || "",
    projectedAfter: ev.projected_cap_after_m ?? pr.projected_cap_after_m,
    projectedNext: ev.projected_cap_space_next_season_m,
    status: response?.status || pr.status || "",
    wantAavM: safeNum(ev.want_aav_m ?? pr.want_aav_m, NaN),
    wantYears: safeNum(ev.want_years ?? pr.want_years, NaN),
    stayInterest: safeNum(ev.stay_interest ?? pr.stay_interest, NaN),
    preferredClause: ev.preferred_clause || pr.preferred_clause || "",
  };
}

function normalizeNegoStatus(response) {
  const raw = response?.status || response?.player_response?.status || "";
  return String(raw || "").toLowerCase();
}

function extractCounter(response) {
  const pr = response?.player_response || {};
  const co = response?.evaluation?.counter_offer || {};
  const aav = pr.counter_cap_hit ?? co.aav_m ?? co.cap_hit_m;
  const years = pr.counter_term ?? co.years;
  if (aav == null || years == null) return null;
  const mode = String(pr.ntc_mode || co.ntc_mode || "").toUpperCase();
  return {
    aav_m: Number(aav),
    years: Number(years),
    ntc: Boolean(pr.counter_ntc ?? co.ntc),
    nmc: Boolean(pr.counter_nmc ?? co.nmc),
    ntc_mode: mode || (pr.counter_ntc || co.ntc ? "FULL" : "NONE"),
    signing_bonus_m: safeNum(pr.counter_signing_bonus_m ?? co.signing_bonus_m, 0),
  };
}

function agentStatusLabel(status) {
  const s = String(status || "").toLowerCase();
  if (s === "countered") return "Counter";
  if (s === "rejected") return "Declined";
  if (s === "pending") return "Considering";
  if (s === "accepted") return "Accepted";
  if (s === "evaluated") return "Read";
  return s ? s.replace(/_/g, " ") : "Agent";
}

export default function CapContractNegotiation({
  row,
  capSnapshot = {},
  nextYearProjection = {},
  signingBonusElig = {},
  seasonLabel = "",
  busy,
  onBusy,
  onResult,
  onBack,
}) {
  const ask = useMemo(() => defaultAsk(row), [row]);
  const [offerAav, setOfferAav] = useState(String(ask.aav));
  const [offerYears, setOfferYears] = useState(String(ask.years));
  const [offerNtcMode, setOfferNtcMode] = useState("NONE");
  const [offerNmc, setOfferNmc] = useState(false);
  const [offerBonus, setOfferBonus] = useState("0");
  const [response, setResponse] = useState(null);
  const [localError, setLocalError] = useState("");
  const [previewPending, setPreviewPending] = useState(false);
  const [previewOfferKey, setPreviewOfferKey] = useState("");
  const [responseSource, setResponseSource] = useState(null);
  const previewSeqRef = useRef(0);

  useEffect(() => {
    setOfferAav(String(ask.aav));
    setOfferYears(String(ask.years));
    setOfferNtcMode("NONE");
    setOfferNmc(false);
    setOfferBonus("0");
    setResponse(null);
    setPreviewOfferKey("");
    setResponseSource(null);
    setLocalError("");
  }, [row?.player_id, row?.id, ask.aav, ask.years]);

  const offerAavNum = safeNum(offerAav, ask.aav);
  const offerYearsNum = Math.min(
    Math.max(1, Number(signingBonusElig?.max_term_own || 7)),
    Math.max(1, safeNum(offerYears, ask.years))
  );
  const maxTerm = Math.max(1, Number(signingBonusElig?.max_term_own || 7));
  const termOptions = Array.from({ length: maxTerm }, (_, i) => i + 1);
  const bonusAllowed = Boolean(signingBonusElig?.eligible);
  const bonusMaxPct = bonusAllowed ? Number(signingBonusElig?.max_bonus_pct || 0) : 0;
  const bonusMaxM = Math.max(0, Math.floor(offerAavNum * Math.min(offerYearsNum, maxTerm) * bonusMaxPct * 40) / 40);
  // Never send more bonus than the rules allow (the slider clamps, the draft value must too).
  const offerBonusNum = Math.min(bonusMaxM, Math.max(0, safeNum(offerBonus, 0)));
  const offerCapHitNum = computeOfferCapHitM(offerAavNum, offerYearsNum, offerBonusNum);
  const clauseAsk =
    response?.evaluation?.preferred_clause ||
    row?.clause_ask ||
    (row?.clause_label && row.clause_label !== "None" ? row.clause_label : "");

  const capProj = useMemo(
    () =>
      projectNegotiationCap({
        capSnapshot,
        nextYearProjection,
        playerRow: row,
        offerCapHitM: offerCapHitNum,
      }),
    [capSnapshot, nextYearProjection, row, offerCapHitNum],
  );

  const offerKey = `${offerAavNum.toFixed(3)}|${offerYearsNum}|${offerNtcMode}|${offerNmc ? 1 : 0}|${offerBonusNum.toFixed(3)}`;
  const evalSnap = readEvaluation(response);
  const negoStatus = normalizeNegoStatus(response);
  const isSubmitTurn = responseSource === "submit" && negoStatus && negoStatus !== "evaluated";
  const previewFresh = Boolean(response && previewOfferKey === offerKey && !isSubmitTurn);
  const offerStale = Boolean(
    response && previewOfferKey && previewOfferKey !== offerKey && responseSource !== "submit",
  );
  const negotiationRound = safeNum(response?.negotiation?.current_round, 0);
  const counterTerms = useMemo(() => extractCounter(response), [response]);

  const wantAavM = Number.isFinite(evalSnap.wantAavM) ? evalSnap.wantAavM : ask.aav;
  const wantYears = Number.isFinite(evalSnap.wantYears) ? evalSnap.wantYears : ask.years;
  const preferredClause = evalSnap.preferredClause || clauseAsk || "None";
  const localInterest = useMemo(
    () =>
      estimateOfferInterestM({
        offerAavM: offerAavNum,
        wantAavM,
        offerYears: offerYearsNum,
        wantYears,
        stayInterest: evalSnap.stayInterest,
        signingBonusM: offerBonusNum,
        totalValueM: offerAavNum * offerYearsNum,
        preferredClause,
        ntcMode: offerNtcMode,
        nmc: offerNmc,
      }),
    [
      offerAavNum,
      wantAavM,
      offerYearsNum,
      wantYears,
      evalSnap.stayInterest,
      offerBonusNum,
      preferredClause,
      offerNtcMode,
      offerNmc,
    ],
  );
  const negoInterest =
    Number.isFinite(evalSnap.interest) && !offerStale && !previewPending
      ? evalSnap.interest
      : localInterest;
  const acceptCut = Number.isFinite(evalSnap.acceptCut) ? evalSnap.acceptCut : 62;

  const projectedAfter = capProj.projectedAfterM;
  const projectedNext = capProj.projectedNextSeasonM;

  const agent = row?.agent || row?.contract?.agent || {};
  const agentName = agent.name || "Player agent";
  const agentAgency = agent.agency || "";
  const agentStyle = agent.style_label || agent.style || "";
  const agentDiff = resolveAgentDealDifficulty(agent);
  const player = ensurePlayerHeadshotFields(row);

  const agentLine = isSubmitTurn
    ? evalSnap.feedback ||
      evalSnap.agentMood ||
      safeText(response?.reason, "The agent responded to your offer.")
    : previewFresh
      ? evalSnap.feedback || evalSnap.agentMood || ""
      : offerStale
        ? `${agentName}: terms changed — Talk to agent or submit again for an updated read.`
        : `${agentName} represents ${safeText(row?.name, "the player")}. Set AAV and years, then submit or talk to agent.`;

  const sliderMax = Math.min(
    Number(signingBonusElig?.max_salary_m || 99),
    Math.max(12, ask.aav * 1.45, offerAavNum, safeNum(row?.aav_m, 1) * 1.8)
  );
  const sliderMin = 0.775;

  const buildPayload = useCallback(
    () => ({
      player_id: row.player_id || row.id,
      aav_m: offerAavNum,
      years: offerYearsNum,
      context: "re_sign",
      ntc: offerNtcMode === "FULL" || offerNtcMode === "MODIFIED",
      ntc_mode: offerNtcMode,
      m_ntc: offerNtcMode === "MODIFIED",
      nmc: offerNmc,
      signing_bonus_m: offerBonusNum,
    }),
    [row, offerAavNum, offerYearsNum, offerNtcMode, offerNmc, offerBonusNum],
  );

  const runPreview = useCallback(async () => {
    if (!row?.player_id && !row?.id) return;
    if (previewPending) return;
    const seq = previewSeqRef.current + 1;
    previewSeqRef.current = seq;
    setPreviewPending(true);
    setLocalError("");
    try {
      const result = await evaluateContractOffer(buildPayload());
      if (previewSeqRef.current !== seq) return;
      setResponse(result);
      setResponseSource("preview");
      setPreviewOfferKey(offerKey);
      if (!result?.ok && result?.reason) setLocalError(result.reason);
    } catch (e) {
      if (previewSeqRef.current === seq) {
        setLocalError(String(e?.message || "Could not reach agent"));
      }
    } finally {
      if (previewSeqRef.current === seq) {
        setPreviewPending(false);
      }
    }
  }, [row, buildPayload, previewPending, offerKey]);

  useEffect(() => {
    if (!row?.player_id && !row?.id) return undefined;
    if (busy) return undefined;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      const seq = previewSeqRef.current + 1;
      previewSeqRef.current = seq;
      setPreviewPending(true);
      setLocalError("");
      try {
        const result = await evaluateContractOffer(buildPayload());
        if (cancelled || previewSeqRef.current !== seq) return;
        setResponse(result);
        setResponseSource("preview");
        setPreviewOfferKey(offerKey);
        if (!result?.ok && result?.reason) setLocalError(result.reason);
      } catch (e) {
        if (!cancelled && previewSeqRef.current === seq) {
          setLocalError(String(e?.message || "Could not reach agent"));
        }
      } finally {
        if (!cancelled && previewSeqRef.current === seq) {
          setPreviewPending(false);
        }
      }
    }, 320);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [row?.player_id, row?.id, offerKey, busy, buildPayload]);

  const applyCounterToSliders = useCallback((counter) => {
    if (!counter) return;
    setOfferAav(String(counter.aav_m));
    setOfferYears(String(counter.years));
    if (counter.nmc) {
      setOfferNmc(true);
      setOfferNtcMode("NONE");
    } else if (counter.ntc_mode === "MODIFIED") {
      setOfferNmc(false);
      setOfferNtcMode("MODIFIED");
    } else if (counter.ntc_mode === "FULL" || counter.ntc) {
      setOfferNmc(false);
      setOfferNtcMode("FULL");
    }
    if (counter.signing_bonus_m > 0) {
      setOfferBonus(String(counter.signing_bonus_m));
    }
    setResponseSource(null);
    setPreviewOfferKey("");
  }, []);

  const runSubmit = async () => {
    if (!row?.player_id && !row?.id) return;
    onBusy?.(true);
    setLocalError("");
    try {
      const result = await reSignContract(buildPayload());
      setResponse(result);
      setResponseSource("submit");
      setPreviewOfferKey(offerKey);
      onResult?.(result, { preview: false });
      const st = normalizeNegoStatus(result);
      if (!result?.ok && result?.reason && st !== "countered" && st !== "rejected") {
        setLocalError(result.reason);
      }
    } catch (e) {
      setLocalError(String(e?.message || "Offer failed"));
    } finally {
      onBusy?.(false);
    }
  };

  const signAtCounter = async () => {
    const counter = counterTerms || extractCounter(response);
    if (!counter) return;
    applyCounterToSliders(counter);
    onBusy?.(true);
    setLocalError("");
    try {
      const result = await reSignContract({
        ...buildPayload(),
        aav_m: counter.aav_m,
        years: counter.years,
        ntc: counter.ntc_mode === "FULL" || counter.ntc_mode === "MODIFIED",
        ntc_mode: counter.ntc_mode,
        m_ntc: counter.ntc_mode === "MODIFIED",
        nmc: counter.nmc,
        signing_bonus_m: counter.signing_bonus_m,
        force: true,
      });
      setResponse(result);
      setResponseSource("submit");
      onResult?.(result, { preview: false });
      if (!result?.ok && result?.reason) setLocalError(result.reason);
    } catch (e) {
      setLocalError(String(e?.message || "Could not accept counter"));
    } finally {
      onBusy?.(false);
    }
  };

  const runSimNegotiationDay = async () => {
    onBusy?.(true);
    setLocalError("");
    try {
      const result = await advanceContractNegotiationDay(1);
      const pid = String(row.player_id || row.id || "");
      const signed = Array.isArray(result?.signed)
        ? result.signed.find((s) => String(s.player_id) === pid)
        : null;
      if (signed) {
        onResult?.({ ok: true, status: "accepted", ...result }, { preview: false, signed: true });
        return;
      }
      const still = Array.isArray(result?.still_pending)
        ? result.still_pending.find((s) => String(s.player_id) === pid)
        : null;
      const held = still?.days_held ?? 0;
      const need = still?.resolve_days ?? safeNum(response?.player_response?.resolve_days, 2);
      setResponse((prev) => ({
        ...(prev || {}),
        ok: true,
        status: "pending",
        player_response: {
          ...(prev?.player_response || {}),
          status: "pending",
          feedback: held
            ? `Still reviewing — day ${held} of ${need} on the table`
            : "Offer is on the table — Sim Day to hear back",
          resolve_days: need,
        },
      }));
      setResponseSource("submit");
      onResult?.(result, { preview: false, simDay: true });
    } catch (e) {
      setLocalError(String(e?.message || "Could not advance negotiation day"));
    } finally {
      onBusy?.(false);
    }
  };

  const hasCounter = isSubmitTurn && negoStatus === "countered" && counterTerms;
  const isPending = isSubmitTurn && negoStatus === "pending";
  const isRejected = isSubmitTurn && negoStatus === "rejected";

  const interestDisplay = Number.isFinite(negoInterest) ? Math.round(negoInterest) : "—";
  const meterTone = Number.isFinite(negoInterest) ? interestMeterTone(negoInterest) : "mid";
  const meterWidth = Number.isFinite(negoInterest) ? negoInterest : 12;

  return (
    <div className="cap-nego-desk">
      <header className="cap-nego-desk__bar">
        <button type="button" className="cap-nego__back cap-edraft-action-btn" disabled={busy} onClick={onBack}>
          ← Dossier
        </button>
        <div className="cap-nego-desk__title">
          <span className="cap-office-kicker">Negotiation desk</span>
          <h3>{safeText(row?.name, "Player")}</h3>
          {seasonLabel ? <em>{seasonLabel}</em> : null}
        </div>
        <div className="cap-nego-desk__wells">
          <div className="cap-dossier-tile">
            <span className="cap-dossier-tile__label">Cap now</span>
            <strong className="cap-num-pop tone-green">
              {formatMoneyM(capSnapshot.usable_cap_space_m)}
            </strong>
          </div>
          <div className="cap-dossier-tile">
            <span className="cap-dossier-tile__label">After offer</span>
            <strong
              className={`cap-num-pop ${projectedAfter != null && projectedAfter < 0 ? "tone-danger" : "tone-cyan"}`}
            >
              {projectedAfter != null ? formatMoneyM(projectedAfter) : "—"}
            </strong>
          </div>
          <div className="cap-dossier-tile">
            <span className="cap-dossier-tile__label">Next season</span>
            <strong
              className={`cap-num-pop ${projectedNext != null && projectedNext < 0 ? "tone-danger" : "tone-gold"}`}
            >
              {projectedNext != null ? formatMoneyM(projectedNext) : "—"}
            </strong>
          </div>
        </div>
      </header>

      <div className="cap-nego-desk__grid">
        <aside className="cap-nego-desk__agent">
          <div className="cap-nego-desk__hero">
            <PlayerHeadshot
              player={player}
              size="xl"
              showFlag={false}
              preferPhoto
              className="cap-nego-desk__hero-shot"
            />
            <div className="cap-nego-desk__hero-meta">
              <strong>{safeText(row?.name)}</strong>
              <p>
                {safeText(row?.position)} · OVR {row?.overall ?? row?.ovr ?? "—"}
              </p>
            </div>
          </div>
          <div className="cap-nego__agent">
            <div className="cap-nego__agent-id">
              <span
                className={`cap-icon-well tone-${agentDifficultyTone(agentDiff.tier)} cap-nego__agent-badge`}
                title={agentDiff.label}
                aria-label={agentDiff.label}
              >
                {agentDifficultyIcon(agentDiff.tier)}
              </span>
              <div className="cap-nego__agent-meta">
                <strong>{agentName}</strong>
                {agentAgency ? <em>{agentAgency}</em> : null}
                {agentStyle ? <span className="cap-nego__agent-style">{agentStyle}</span> : null}
                <span className={`cap-nego__agent-tier tone-${agentDiff.tier}`}>{agentDiff.label}</span>
              </div>
            </div>
            <p className="cap-nego__agent-line">{agentLine}</p>
            {evalSnap.agentMood && evalSnap.agentMood !== agentLine ? (
              <p className="cap-nego__agent-mood">{evalSnap.agentMood}</p>
            ) : null}
            {isSubmitTurn ? (
              <div
                className={`cap-nego-thread tone-${negoStatus === "rejected" ? "reject" : negoStatus === "countered" ? "counter" : negoStatus === "pending" ? "pending" : "neutral"}`}
                role="status"
              >
                <div className="cap-nego-thread__head">
                  <span className="cap-nego-thread__badge">{agentStatusLabel(negoStatus)}</span>
                  {negotiationRound > 0 ? (
                    <em className="cap-nego-thread__round">Round {negotiationRound}</em>
                  ) : null}
                </div>
                <p className="cap-nego-thread__body">
                  {safeText(response?.player_response?.feedback, evalSnap.feedback) ||
                    safeText(response?.reason, "Waiting on the agent.")}
                </p>
                {hasCounter ? (
                  <p className="cap-nego-thread__terms">
                    Counter:{" "}
                    <strong className="cap-num-pop tone-gold">
                      {formatMoneyM(counterTerms.aav_m)} × {counterTerms.years}y
                    </strong>
                    {counterTerms.nmc ? " · NMC" : null}
                    {!counterTerms.nmc && counterTerms.ntc_mode === "FULL" ? " · NTC" : null}
                    {!counterTerms.nmc && counterTerms.ntc_mode === "MODIFIED" ? " · M-NTC" : null}
                  </p>
                ) : null}
                {isPending ? (
                  <p className="cap-nego-thread__hint">
                    Competitive offer — use Sim Day while they compare the market.
                  </p>
                ) : null}
                {isRejected ? (
                  <p className="cap-nego-thread__hint">
                    Revise money, term, or protection and submit again — or walk away from the desk.
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
          <div className="cap-nego-desk__ask">
            <span>Player ask</span>
            <strong className="cap-num-pop tone-gold">
              {formatMoneyM(ask.aav)} × {ask.years}y
            </strong>
          </div>
        </aside>

        <section className="cap-nego-desk__controls">
          <NegotiationMeetingPanel
            playerId={row?.player_id || row?.id}
            compact
            onChanged={() => {
              runPreview();
            }}
          />
          <div className="cap-nego-meter" aria-label="Deal interest">
            <div className="cap-nego-meter-head">
              <span>Deal interest</span>
              <strong className="cap-num-pop tone-cyan">{interestDisplay}</strong>
              {previewPending ? <em className="cap-nego__sync">Agent…</em> : null}
            </div>
            <div className="cap-nego-meter-track">
              <span
                className={`cap-nego-meter-fill tone-${meterTone}`}
                style={{
                  width: `${Math.max(4, Math.min(100, meterWidth))}%`,
                }}
              />
              <i className="cap-nego-meter-mark" style={{ left: `${acceptCut}%` }} title="Accept threshold" />
              <i className="cap-nego-meter-mark is-instant" style={{ left: "88%" }} title="Instant accept" />
            </div>
            <p className="cap-nego-meter-note">
              {previewFresh
                ? `Needs ≥ ${Math.round(acceptCut)} to accept · ${resignInterestLabel(row)} stay interest${
                    evalSnap.status ? ` · ${evalSnap.status}` : ""
                  }`
                : previewPending
                  ? "Updating interest for this offer…"
                  : `Live estimate · ${resignInterestLabel(row)} stay interest · agent confirms on submit`}
            </p>
          </div>

          <div className="cap-nego__slider">
            <div className="cap-nego__slider-head">
              <span>Your offer (AAV)</span>
              <strong className="cap-num-pop tone-gold">{formatMoneyM(offerAavNum)}</strong>
            </div>
            <input
              type="range"
              className="cap-nego__range"
              min={sliderMin}
              max={sliderMax}
              step="0.025"
              value={Math.min(sliderMax, Math.max(sliderMin, offerAavNum))}
              disabled={busy}
              onChange={(e) => setOfferAav(e.target.value)}
            />
          </div>

          <div className="cap-nego__clauses" role="group" aria-label="Contract clauses">
            <span>Protection</span>
            <div className="cap-nego__term-btns cap-nego__clause-btns">
              <button
                type="button"
                className={offerNtcMode === "FULL" ? "is-active" : ""}
                disabled={busy || offerNmc}
                onClick={() => setOfferNtcMode("FULL")}
              >
                NTC
                {clauseAsk === "NTC" ? <em className="cap-nego__clause-tag">asked</em> : null}
              </button>
              <button
                type="button"
                className={offerNtcMode === "MODIFIED" ? "is-active" : ""}
                disabled={busy || offerNmc}
                onClick={() => setOfferNtcMode("MODIFIED")}
              >
                M-NTC
                {clauseAsk === "M-NTC" ? <em className="cap-nego__clause-tag">asked</em> : null}
              </button>
              <button
                type="button"
                className={offerNtcMode === "NONE" && !offerNmc ? "is-active" : ""}
                disabled={busy || offerNmc}
                onClick={() => setOfferNtcMode("NONE")}
              >
                None
              </button>
              <button
                type="button"
                className={offerNmc ? "is-active" : ""}
                disabled={busy}
                onClick={() => {
                  setOfferNmc((v) => {
                    const next = !v;
                    if (next) setOfferNtcMode("NONE");
                    return next;
                  });
                }}
              >
                NMC
                {clauseAsk === "NMC" ? <em className="cap-nego__clause-tag">asked</em> : null}
              </button>
            </div>
          </div>

          <div className="cap-nego__slider">
            <div className="cap-nego__slider-head">
              <span>Signing bonus</span>
              <strong className="cap-num-pop tone-cyan">
                {bonusAllowed ? formatMoneyM(offerBonusNum) : "Locked"}
              </strong>
            </div>
            {bonusAllowed ? (
              <input
                type="range"
                className="cap-nego__range"
                min={0}
                max={bonusMaxM}
                step="0.025"
                value={offerBonusNum}
                disabled={busy}
                onChange={(e) => setOfferBonus(e.target.value)}
              />
            ) : null}
            {bonusAllowed ? (
              <p className="cap-nego-desk__cap-note">
                Max {formatMoneyM(bonusMaxM)} ({Math.round(bonusMaxPct * 100)}% of contract value)
                {signingBonusElig?.cash_note ? ` · ${signingBonusElig.cash_note}` : ""}
              </p>
            ) : (
              <p className="cap-nego-desk__cap-note">
                {signingBonusElig?.label ||
                  "Signing bonuses require NHL revenue ≥ $130M for your club."}
              </p>
            )}
          </div>

          <div className="cap-nego__term">
            <span>Years</span>
            <div className="cap-nego__term-btns">
              {termOptions.map((y) => (
                <button
                  key={y}
                  type="button"
                  className={`cap-edraft-action-btn${offerYearsNum === y ? " is-active" : ""}`}
                  disabled={busy}
                  onClick={() => setOfferYears(String(y))}
                >
                  {y}
                </button>
              ))}
            </div>
          </div>

          <p className="cap-nego-desk__cap-note">
            Cap hit {formatMoneyM(offerCapHitNum)}
            {offerBonusNum > 0 ? ` (incl. ${formatMoneyM(offerBonusNum)} bonus)` : ""}
            {capProj.capDeltaNowM != null ? ` · Δ this season ${formatMoneyM(capProj.capDeltaNowM)}` : ""}
          </p>

          {localError ? <p className="cap-nego__error">{localError}</p> : null}

          <div className="cap-nego-desk__actions">
            <button
              type="button"
              className="cap-action-btn cap-edraft-action-btn"
              disabled={busy || previewPending}
              onClick={runPreview}
            >
              {previewPending ? "Agent…" : "Talk to agent"}
            </button>
            <button
              type="button"
              className="cap-action-btn cap-edraft-action-btn cap-action-btn--sign"
              disabled={busy || previewPending}
              onClick={runSubmit}
            >
              {isSubmitTurn && (hasCounter || isRejected) ? "Submit revised offer" : "Submit offer"}
            </button>
            {hasCounter ? (
              <>
                <button
                  type="button"
                  className="cap-action-btn cap-edraft-action-btn"
                  disabled={busy}
                  onClick={() => applyCounterToSliders(counterTerms)}
                >
                  Load counter on desk
                </button>
                <button
                  type="button"
                  className="cap-action-btn cap-edraft-action-btn cap-action-btn--sign"
                  disabled={busy}
                  onClick={signAtCounter}
                >
                  Sign at counter ({formatMoneyM(counterTerms.aav_m)} × {counterTerms.years}y)
                </button>
              </>
            ) : null}
            {isPending ? (
              <button
                type="button"
                className="cap-action-btn cap-edraft-action-btn cap-action-btn--sign"
                disabled={busy}
                onClick={runSimNegotiationDay}
              >
                Sim negotiation day
              </button>
            ) : null}
          </div>
        </section>
      </div>
    </div>
  );
}
