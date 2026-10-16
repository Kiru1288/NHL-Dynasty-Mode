import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  advanceContractNegotiationDay,
  evaluateContractOffer,
  reSignContract,
  runNegotiationMeeting,
} from "../../services/franchiseService";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import {
  computeOfferCapHitM,
  estimateOfferInterestM,
  interestMeterTone,
  maxNegotiationOfferAavM,
  projectNegotiationCap,
  resignInterestLabel,
} from "../../utils/contractNegotiation";
import {
  agentDifficultyIcon,
  agentDifficultyTone,
  resolveAgentDealDifficulty,
} from "../../utils/playerAgentDisplay";
import NegotiationMeetingPanel, { InterestFactors } from "./NegotiationMeetingPanel";

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function formatMoneyM(v) {
  const n = safeNum(v, NaN);
  if (!Number.isFinite(n)) return "—";
  return `$${n.toFixed(2)}M`;
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
  contractTypes = [],
  backLabel = "Back to contract",
}) {
  // Optional contract categories (re-sign phase: one-way / two-way / ...).
  const legalTypes = (Array.isArray(contractTypes) ? contractTypes : []).filter((t) => t && t.enabled !== false);
  const ask = useMemo(() => defaultAsk(row), [row]);
  const [offerAav, setOfferAav] = useState(String(ask.aav));
  const [offerYears, setOfferYears] = useState(String(ask.years));
  const [offerNtcMode, setOfferNtcMode] = useState("NONE");
  const [offerNmc, setOfferNmc] = useState(false);
  const [offerBonus, setOfferBonus] = useState("0");
  const [offerType, setOfferType] = useState("");
  const [response, setResponse] = useState(null);
  const [localError, setLocalError] = useState("");
  const [previewPending, setPreviewPending] = useState(false);
  const [previewOfferKey, setPreviewOfferKey] = useState("");
  const [responseSource, setResponseSource] = useState(null);
  const [dialogue, setDialogue] = useState([]);
  const [hometownBusy, setHometownBusy] = useState(false);
  const previewSeqRef = useRef(0);

  useEffect(() => {
    setOfferAav(String(ask.aav));
    setOfferYears(String(ask.years));
    setOfferNtcMode("NONE");
    setOfferNmc(false);
    setOfferBonus("0");
    setOfferType("");
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

  const contractCategory = offerType || (legalTypes[0] ? legalTypes[0].id : "");
  const offerKey = `${offerAavNum.toFixed(3)}|${offerYearsNum}|${offerNtcMode}|${offerNmc ? 1 : 0}|${offerBonusNum.toFixed(3)}|${contractCategory}`;
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
  const inSeasonCap = Boolean(capProj.inSeasonCap || capSnapshot?.in_season_cap);

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

  const sliderMax = useMemo(
    () =>
      maxNegotiationOfferAavM({
        capSnapshot,
        nextYearProjection,
        playerRow: row,
        cbaMaxSalaryM: signingBonusElig?.max_salary_m,
      }),
    [capSnapshot, nextYearProjection, row, signingBonusElig?.max_salary_m],
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
      ...(contractCategory
        ? {
            contract_category: contractCategory,
            two_way: contractCategory === "nhl_two_way",
            contract_type: String(contractCategory).startsWith("nhl") ? undefined : contractCategory,
          }
        : {}),
    }),
    [row, offerAavNum, offerYearsNum, offerNtcMode, offerNmc, offerBonusNum, contractCategory],
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
      const spoken =
        result?.evaluation?.agent_dialogue ||
        result?.player_response?.agent_dialogue ||
        result?.player_response?.feedback ||
        "";
      if (spoken) {
        setDialogue((prev) => [...prev, String(spoken)].slice(-8));
      }
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

  const askHometownDiscount = useCallback(async () => {
    const playerId = row?.player_id || row?.id;
    if (!playerId || hometownBusy) return;
    setHometownBusy(true);
    setLocalError("");
    try {
      const res = await runNegotiationMeeting(playerId, "hometown", "");
      const line = res?.message || res?.reason || "He didn't answer.";
      setDialogue((prev) => [...prev, String(line)].slice(-8));
      if (!res?.ok && res?.reason) setLocalError(res.reason);
      if (res?.ok) {
        const preview = await evaluateContractOffer(buildPayload());
        setResponse(preview);
        setResponseSource("preview");
        setPreviewOfferKey(offerKey);
      }
    } catch (e) {
      setLocalError(String(e?.message || "Could not ask for a hometown discount"));
    } finally {
      setHometownBusy(false);
    }
  }, [row, hometownBusy, buildPayload, offerKey]);

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
  const willAccept = Number.isFinite(negoInterest) && negoInterest >= acceptCut;
  const askPct = sliderMax > sliderMin ? ((ask.aav - sliderMin) / (sliderMax - sliderMin)) * 100 : null;
  const offerVsAsk = offerAavNum - ask.aav;
  const ovr = row?.overall ?? row?.ovr ?? "—";

  // Two sides of one table: what they want (left) and what you put down (right).
  // One primary action at a time: sign the counter, sim the waiting day, or submit.
  const primary = hasCounter
    ? { label: `Sign at ${formatMoneyM(counterTerms.aav_m)} × ${counterTerms.years}`, run: signAtCounter }
    : isPending
      ? { label: "Sim a day for an answer", run: runSimNegotiationDay }
      : { label: isSubmitTurn && isRejected ? "Submit revised offer" : "Submit offer", run: runSubmit };

  return (
    <div className="nd">
      <header className="nd-top">
        <button type="button" className="nd-back" disabled={busy} onClick={onBack} aria-label={backLabel} title={backLabel}>
          ←
        </button>
        <PlayerHeadshot player={player} size="md" showFlag={false} preferPhoto className="nd-top__face" />
        <div className="nd-top__who">
          <h3>{safeText(row?.name, "Player")}</h3>
          <p>
            {safeText(row?.position, "—")} · {ovr} OVR{row?.age ? ` · ${row.age}` : ""}
            {seasonLabel ? ` · ${seasonLabel}` : ""}
          </p>
        </div>
        <div className={`nd-agent tone-${agentDifficultyTone(agentDiff.tier)}`} title={agentDiff.label}>
          <span className="nd-agent__icon" aria-hidden="true">{agentDifficultyIcon(agentDiff.tier)}</span>
          <span className="nd-agent__text">
            <strong>{agentName}</strong>
            <em>{[agentAgency, agentStyle, agentDiff.label].filter(Boolean).join(" · ")}</em>
          </span>
        </div>
      </header>

      <div className="nd-table">
        <section className="nd-side nd-side--them" aria-label="Their side">
          <div className="nd-ask">
            <span>They want</span>
            <strong>{formatMoneyM(wantAavM)} <small>× {wantYears} {wantYears === 1 ? "yr" : "yrs"}</small></strong>
            {preferredClause && preferredClause !== "None" ? <em>plus {preferredClause}</em> : null}
          </div>

          <div className="nd-thread" aria-live="polite">
            <p className="nd-bubble">{agentLine}</p>
            {dialogue.map((line, i) => (
              <p key={`${i}-${line.slice(0, 24)}`} className="nd-bubble">{line}</p>
            ))}
            {evalSnap.agentMood && evalSnap.agentMood !== agentLine && !dialogue.includes(evalSnap.agentMood) ? (
              <p className="nd-bubble is-mood">{evalSnap.agentMood}</p>
            ) : null}
            {isSubmitTurn ? (
              <div className={`nd-verdict is-${negoStatus || "neutral"}`} role="status">
                <span className="nd-verdict__badge">
                  {agentStatusLabel(negoStatus)}
                  {negotiationRound > 0 ? ` · round ${negotiationRound}` : ""}
                </span>
                {hasCounter ? (
                  <p>
                    Counter: <strong>{formatMoneyM(counterTerms.aav_m)} × {counterTerms.years} {counterTerms.years === 1 ? "yr" : "yrs"}</strong>
                    {counterTerms.nmc ? " · NMC" : null}
                    {!counterTerms.nmc && counterTerms.ntc_mode === "FULL" ? " · NTC" : null}
                    {!counterTerms.nmc && counterTerms.ntc_mode === "MODIFIED" ? " · M-NTC" : null}
                    <button type="button" className="nd-link" disabled={busy} onClick={() => applyCounterToSliders(counterTerms)}>
                      Put it on my side
                    </button>
                  </p>
                ) : null}
                {isPending ? <p>They're comparing the market. Sim a day to hear back.</p> : null}
                {isRejected ? <p>Change the money, term or protection and try again.</p> : null}
              </div>
            ) : null}
          </div>

          <InterestFactors
            title="Why they'd stay"
            factors={response?.evaluation?.interest_factors || row?.interest_factors}
            fit={{
              ...(response?.evaluation?.destination_fit || row?.destination_fit || {}),
              fit_premium_pct: response?.evaluation?.fit_premium_pct,
            }}
          />

          <NegotiationMeetingPanel
            playerId={row?.player_id || row?.id}
            compact
            onChanged={() => {
              runPreview();
            }}
          />
        </section>

        <section className="nd-side nd-side--you" aria-label="Your offer">
          <div className={`nd-meter tone-${meterTone}${willAccept ? " will-accept" : ""}`} aria-label="Deal interest">
            <div className="nd-meter__head">
              <strong>{interestDisplay}</strong>
              <span>
                {previewPending
                  ? "Checking with the agent…"
                  : willAccept
                    ? "They'd take this"
                    : `Needs ${Math.round(acceptCut)} to accept`}
              </span>
            </div>
            <div className="nd-meter__track">
              <i className="nd-meter__fill" style={{ width: `${Math.max(4, Math.min(100, meterWidth))}%` }} />
              <i className="nd-meter__mark" style={{ left: `${acceptCut}%` }} title="Accept line" />
              <i className="nd-meter__mark is-instant" style={{ left: "88%" }} title="Instant accept" />
            </div>
            <p className="nd-meter__note">
              {previewFresh || previewPending ? "Agent read" : "Live estimate"} · {resignInterestLabel(row)} interest in staying
            </p>
          </div>

          <div className="nd-field">
            <div className="nd-field__head">
              <span>Cap hit per year</span>
              <strong>{formatMoneyM(offerAavNum)}</strong>
              <em className={offerVsAsk >= 0 ? "is-up" : "is-down"}>
                {Math.abs(offerVsAsk) < 0.005 ? "= their ask" : `${offerVsAsk > 0 ? "+" : "−"}${formatMoneyM(Math.abs(offerVsAsk))} vs ask`}
              </em>
            </div>
            <div className="nd-range">
              <input
                type="range"
                min={sliderMin}
                max={sliderMax}
                step="0.025"
                value={Math.min(sliderMax, Math.max(sliderMin, offerAavNum))}
                disabled={busy}
                aria-label="Offer cap hit"
                onChange={(e) => setOfferAav(e.target.value)}
              />
              {askPct != null && askPct >= 0 && askPct <= 100 ? (
                <i className="nd-range__ask" style={{ left: `${askPct}%` }} title={`Their ask ${formatMoneyM(ask.aav)}`} />
              ) : null}
            </div>
            <div className="nd-range__scale">
              <span>{formatMoneyM(sliderMin)}</span>
              <span>{formatMoneyM(sliderMax)}</span>
            </div>
          </div>

          {legalTypes.length > 1 ? (
            <div className="nd-field">
              <div className="nd-field__head"><span>Contract type</span></div>
              <div className="nd-seg" role="group" aria-label="Contract type">
                {legalTypes.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    className={contractCategory === t.id ? "is-active" : ""}
                    aria-pressed={contractCategory === t.id}
                    disabled={busy}
                    onClick={() => setOfferType(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <div className="nd-field">
            <div className="nd-field__head"><span>Years</span></div>
            <div className="nd-seg" role="group" aria-label="Contract years">
              {termOptions.map((y) => (
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

          <div className="nd-field">
            <div className="nd-field__head"><span>Protection</span></div>
            <div className="nd-seg" role="group" aria-label="Contract clauses">
              {[
                ["NONE", "None"],
                ["FULL", "NTC"],
                ["MODIFIED", "M-NTC"],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  type="button"
                  className={offerNtcMode === mode && !offerNmc ? "is-active" : ""}
                  aria-pressed={offerNtcMode === mode && !offerNmc}
                  disabled={busy || offerNmc}
                  onClick={() => setOfferNtcMode(mode)}
                >
                  {label}
                  {label !== "None" && clauseAsk === label ? <em>asked</em> : null}
                </button>
              ))}
              <button
                type="button"
                className={offerNmc ? "is-active" : ""}
                aria-pressed={offerNmc}
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
                {clauseAsk === "NMC" ? <em>asked</em> : null}
              </button>
            </div>
          </div>

          <div className="nd-field">
            <div className="nd-field__head">
              <span>Signing bonus</span>
              <strong>{bonusAllowed ? formatMoneyM(offerBonusNum) : "Not available"}</strong>
            </div>
            {bonusAllowed ? (
              <>
                <div className="nd-range">
                  <input
                    type="range"
                    min={0}
                    max={bonusMaxM}
                    step="0.025"
                    value={offerBonusNum}
                    disabled={busy}
                    aria-label="Signing bonus"
                    onChange={(e) => setOfferBonus(e.target.value)}
                  />
                </div>
                <p className="nd-note">
                  Up to {formatMoneyM(bonusMaxM)} ({Math.round(bonusMaxPct * 100)}% of the deal)
                  {signingBonusElig?.cash_note ? ` · ${signingBonusElig.cash_note}` : ""}
                </p>
              </>
            ) : (
              <p className="nd-note">{signingBonusElig?.label || "Signing bonuses need club revenue of $155M or more."}</p>
            )}
          </div>

          <dl className="nd-cap">
            <div>
              <dt>Cap hit</dt>
              <dd>{formatMoneyM(offerCapHitNum)}{offerBonusNum > 0 ? <small> incl. bonus</small> : null}</dd>
            </div>
            <div>
              <dt>{capProj.dealStartsNextYear ? "Space this year" : "Space after signing"}</dt>
              <dd className={projectedAfter != null && projectedAfter < 0 ? "is-bad" : ""}>
                {projectedAfter != null ? formatMoneyM(projectedAfter) : "—"}
              </dd>
            </div>
            <div>
              <dt>{capProj.dealStartsNextYear ? "Space next year after" : inSeasonCap ? "Next year (info)" : "Next season"}</dt>
              <dd className={projectedNext != null && projectedNext < 0 ? "is-bad" : ""}>
                {projectedNext != null ? formatMoneyM(projectedNext) : "—"}
              </dd>
            </div>
            {capProj.capBarPct != null ? (
              <div className="nd-cap__bar" title="Cap used after this offer">
                <i
                  className={capProj.capBarPct >= 100 ? "is-over" : capProj.capBarPct >= 95 ? "is-tight" : ""}
                  style={{ width: `${Math.min(100, capProj.capBarPct)}%` }}
                />
              </div>
            ) : null}
            {capProj.dealStartsNextYear ? <p className="nd-note">Starts next season, so it doesn't touch this year's cap.</p> : null}
          </dl>

          {localError ? <p className="nd-error" role="alert">{localError}</p> : null}

          <div className="nd-actions">
            <button type="button" className="nd-btn nd-btn--primary" disabled={busy || previewPending} onClick={primary.run}>
              {primary.label}
            </button>
            <button type="button" className="nd-btn" disabled={busy || previewPending} onClick={runPreview}>
              {previewPending ? "Asking…" : "Talk to agent"}
            </button>
            <button type="button" className="nd-btn" disabled={busy || previewPending || hometownBusy} onClick={askHometownDiscount}>
              {hometownBusy ? "Asking…" : "Hometown discount?"}
            </button>
            {hasCounter || isPending ? (
              <button type="button" className="nd-btn" disabled={busy || previewPending} onClick={runSubmit}>
                Submit my offer instead
              </button>
            ) : null}
          </div>
        </section>
      </div>
    </div>
  );
}
