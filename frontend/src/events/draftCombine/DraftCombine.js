import React, { useCallback, useEffect, useMemo, useState } from "react";
import OffseasonTimeline from "../offseasonTimeline";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { getTeamLogoSrc } from "../../utils/teamLogos";
import { getDraftCombineState, submitCombineMeeting } from "../../services/franchiseService";
import { safeArray } from "../shared/eventHelpers";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import "../../styles/nhlcalShell.css";
import "./DraftCombine.css";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "meetings", label: "Meetings" },
  { id: "prospects", label: "Prospects" },
  { id: "results", label: "Results" },
  { id: "board", label: "Final Board" },
];

function Chip({ children, tone }) {
  if (children === null || children === undefined || children === "") return null;
  return <span className={`dcb-chip${tone ? ` tone-${tone}` : ""}`}>{children}</span>;
}

function valueOrDash(value) {
  if (value === null || value === undefined || value === "") return "—";
  return value;
}

function numberOrDash(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return Math.round(n);
}

function getName(p) {
  return p?.name || p?.prospect_name || p?.full_name || "—";
}

function getId(p) {
  return p?.prospect_id || p?.id || p?.key || `${getName(p)}-${p?.rank ?? "rank"}`;
}

function getRank(p) {
  return p?.rank ?? p?.public_rank ?? p?.final_rank ?? "—";
}

function getPosition(p) {
  return p?.position || p?.pos || "—";
}

function getLeague(p) {
  return p?.league || p?.draft_league || p?.team_league || p?.source_league || "";
}

function getCountry(p) {
  return p?.country || p?.nationality || p?.birth_country || "";
}

function getStockDelta(p) {
  const raw =
    p?.combine_stock_delta ??
    p?.stock_delta ??
    p?.rank_delta ??
    p?.draft_stock_delta ??
    0;

  const n = Number(raw);
  return Number.isFinite(n) ? n : 0;
}

function getCombineScore(p) {
  return (
    p?.combine_score ??
    p?.testing_score ??
    p?.athletic_score ??
    p?.physical_testing_score ??
    null
  );
}

function getInterviewScore(p) {
  return (
    p?.interview_score ??
    p?.character_score ??
    p?.meeting_score ??
    p?.personality_score ??
    null
  );
}

function getMedicalRisk(p) {
  if (p?.medical_risk_level) return p.medical_risk_level;
  if (p?.medical_flag) return "Flag";
  if (p?.medical_risk) return p.medical_risk;
  return "";
}

function getOldRank(p) {
  return (
    p?.pre_combine_rank ??
    p?.old_rank ??
    p?.previous_rank ??
    p?.rank_before_combine ??
    null
  );
}

function getNewRank(p) {
  return (
    p?.post_combine_rank ??
    p?.new_rank ??
    p?.final_rank ??
    p?.rank_after_combine ??
    p?.rank ??
    null
  );
}

function formatDelta(delta) {
  if (delta > 0) return `↑ ${delta}`;
  if (delta < 0) return `↓ ${Math.abs(delta)}`;
  return "—";
}

function getDeltaTone(delta) {
  if (delta > 0) return "safe";
  if (delta < 0) return "risk";
  return "story";
}

function getRiskTone(value) {
  const text = String(value || "").toLowerCase();

  if (
    text.includes("high") ||
    text.includes("major") ||
    text.includes("severe") ||
    text.includes("red") ||
    text.includes("flag")
  ) {
    return "risk";
  }

  if (
    text.includes("medium") ||
    text.includes("moderate") ||
    text.includes("watch") ||
    text.includes("concern")
  ) {
    return "story";
  }

  return "safe";
}

function scoreGrade(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return "—";
  if (n >= 92) return "A+";
  if (n >= 88) return "A";
  if (n >= 84) return "A-";
  if (n >= 80) return "B+";
  if (n >= 76) return "B";
  if (n >= 72) return "B-";
  if (n >= 68) return "C+";
  if (n >= 64) return "C";
  return "D";
}

function EmptyText({ children = "—" }) {
  return <p className="dcb-empty">{children}</p>;
}

function TabButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      className={`nhlcal-quick-link${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function FilterButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      className={`nhlcal-quick-link${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function Panel({ title, right, children, empty, style }) {
  return (
    <section className="dcb-panel-card" style={style}>
      <div className="dcb-panel-head">
        <h3>{title}</h3>
        {right ? <div>{right}</div> : null}
      </div>
      {empty ? <EmptyText>{empty}</EmptyText> : children}
    </section>
  );
}

function StatTile({ label, value, tone }) {
  const toneClass =
    tone === "risk" ? " is-risk" : tone === "safe" ? " is-safe" : "";
  return (
    <div className={`dcb-stat${toneClass}`}>
      <span>{label}</span>
      <strong>{valueOrDash(value)}</strong>
    </div>
  );
}

function ProspectAvatar({ p }) {
  const name = getName(p);
  const position = getPosition(p);

  return (
    <div className="dcb-avatar" title={name}>
      <PlayerHeadshot player={p} size="sm" style={{ "--size": "34px" }} />
      <span className="dcb-avatar-pos">{position}</span>
    </div>
  );
}

function ProspectCard({ p, selected, onClick, compact }) {
  const delta = getStockDelta(p);
  const combineScore = getCombineScore(p);
  const interviewScore = getInterviewScore(p);
  const medicalRisk = getMedicalRisk(p);
  const league = getLeague(p);
  const country = getCountry(p);

  return (
    <button
      type="button"
      onClick={onClick}
      className={`dcb-prospect-row${selected ? " is-selected" : ""}`}
    >
      <ProspectAvatar p={p} />

      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: "flex", gap: 7, alignItems: "center", flexWrap: "wrap", marginBottom: 4 }}>
          <strong>#{getRank(p)}</strong>
          <strong>{getName(p)}</strong>
          <Chip>{getPosition(p)}</Chip>
          {delta !== 0 ? <Chip tone={getDeltaTone(delta)}>{formatDelta(delta)}</Chip> : null}
          {medicalRisk ? <Chip tone={getRiskTone(medicalRisk)}>Medical</Chip> : null}
        </div>

        {!compact ? (
          <div className="dcb-prospect-meta">
            {league ? <span>{league}</span> : null}
            {country ? <span>{country}</span> : null}
            {combineScore != null ? <span>Test {numberOrDash(combineScore)}</span> : null}
            {interviewScore != null ? <span>Interview {scoreGrade(interviewScore)}</span> : null}
          </div>
        ) : null}
      </div>
    </button>
  );
}

function SmallResultRow({ p, type }) {
  const delta = getStockDelta(p);
  const oldRank = getOldRank(p);
  const newRank = getNewRank(p);
  const combineScore = getCombineScore(p);
  const interviewScore = getInterviewScore(p);
  const medicalRisk = getMedicalRisk(p);

  return (
    <div
      className="an-result-row"
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) auto",
        gap: 10,
        alignItems: "center",
        padding: "10px 0",
        borderBottom: "1px solid rgba(255,255,255,0.07)",
      }}
    >
      <div style={{ minWidth: 0 }}>
        <strong
          style={{
            display: "block",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          #{getRank(p)} {getName(p)}
        </strong>

        <span className="an-muted" style={{ fontSize: 12 }}>
          {getPosition(p)}
          {getLeague(p) ? ` · ${getLeague(p)}` : ""}
        </span>
      </div>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
        {type === "movement" ? (
          <>
            <Chip tone={getDeltaTone(delta)}>{formatDelta(delta)}</Chip>
            <Chip>{oldRank || newRank ? `#${valueOrDash(oldRank)} → #${valueOrDash(newRank)}` : null}</Chip>
          </>
        ) : null}

        {type === "testing" ? <Chip tone="story">{numberOrDash(combineScore)}</Chip> : null}
        {type === "interview" ? <Chip tone="safe">{scoreGrade(interviewScore)}</Chip> : null}
        {type === "medical" ? <Chip tone={getRiskTone(medicalRisk)}>{medicalRisk || "Flag"}</Chip> : null}
      </div>
    </div>
  );
}

function DetailPanel({ prospect }) {
  if (!prospect) {
    return (
      <Panel title="Prospect Detail">
        <EmptyText />
      </Panel>
    );
  }

  const delta = getStockDelta(prospect);
  const combineScore = getCombineScore(prospect);
  const interviewScore = getInterviewScore(prospect);
  const medicalRisk = getMedicalRisk(prospect);
  const oldRank = getOldRank(prospect);
  const newRank = getNewRank(prospect);

  return (
    <Panel title="Prospect Detail">
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        <ProspectAvatar p={prospect} />

        <div style={{ minWidth: 0 }}>
          <h3
            className="an-stage-line"
            style={{
              margin: 0,
              fontSize: 22,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            #{getRank(prospect)} {getName(prospect)}
          </h3>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>
            <Chip>{getPosition(prospect)}</Chip>
            <Chip>{getLeague(prospect)}</Chip>
            <Chip>{getCountry(prospect)}</Chip>
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: 8,
        }}
      >
        <StatTile label="Testing" value={numberOrDash(combineScore)} />
        <StatTile label="Interview" value={scoreGrade(interviewScore)} />
        <StatTile
          label="Stock"
          value={delta !== 0 ? formatDelta(delta) : "—"}
          tone={getDeltaTone(delta)}
        />
        <StatTile
          label="Medical"
          value={medicalRisk || "—"}
          tone={medicalRisk ? getRiskTone(medicalRisk) : undefined}
        />
      </div>

      {(oldRank || newRank) ? (
        <p className="an-muted" style={{ margin: "12px 0 0", fontSize: 13 }}>
          Rank: #{valueOrDash(oldRank)} → #{valueOrDash(newRank)}
        </p>
      ) : null}

      {(prospect?.scout_note ||
        prospect?.interview_summary ||
        prospect?.summary ||
        prospect?.report ||
        prospect?.medical_summary ||
        prospect?.medical_note) ? (
        <p className="an-muted" style={{ margin: "12px 0 0", fontSize: 13, lineHeight: 1.45 }}>
          {prospect?.scout_note ||
            prospect?.interview_summary ||
            prospect?.summary ||
            prospect?.report ||
            prospect?.medical_summary ||
            prospect?.medical_note}
        </p>
      ) : null}
    </Panel>
  );
}

function MeetingOption({ option, disabled, loading, selected, onSelect, onMeeting }) {
  const prospectId = option?.prospect_id || option?.id || option?.key;
  const alreadyMet = option?.already_met || option?.met || false;
  const risk = option?.risk_label || option?.scout_risk || option?.medical_risk_level || "";
  const scoutRead =
    option?.scout_read ||
    option?.team_fit_label ||
    option?.internal_read ||
    option?.summary ||
    "";

  return (
    <div className={`dcb-meeting-row${selected ? " is-selected" : ""}`}>
      <button
        type="button"
        onClick={onSelect}
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          minWidth: 0,
          border: 0,
          background: "transparent",
          color: "inherit",
          padding: 0,
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        <ProspectAvatar p={option} />
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", gap: 7, alignItems: "center", flexWrap: "wrap", marginBottom: 4 }}>
            <strong>#{valueOrDash(option?.rank)} {getName(option)}</strong>
            <Chip>{getPosition(option)}</Chip>
            {risk ? <Chip tone={getRiskTone(risk)}>{risk}</Chip> : null}
            {alreadyMet ? <Chip tone="safe">Met</Chip> : null}
          </div>
          {scoutRead ? (
            <p className="dcb-empty" style={{ margin: 0, maxWidth: 460 }}>
              {scoutRead}
            </p>
          ) : null}
        </div>
      </button>

      <div className="dcb-meeting-actions">
        {!alreadyMet ? (
          <>
            <button
              type="button"
              disabled={disabled || loading}
              onClick={() => onMeeting(prospectId, "interview")}
            >
              {loading ? "..." : "Interview"}
            </button>
            <button
              type="button"
              disabled={disabled || loading}
              onClick={() => onMeeting(prospectId, "dinner")}
            >
              Dinner
            </button>
          </>
        ) : (
          <Chip tone="safe">Logged</Chip>
        )}
      </div>
    </div>
  );
}

export default function DraftCombine({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const gameUI = useGameUI();
  const { setFranchiseState } = gameUI;
  const team = franchiseState?.team || {};

  const initial = franchiseState?.draft_combine || eventData?.draft_combine || {};

  const [combine, setCombine] = useState(initial);
  const [loading, setLoading] = useState(!initial?.completed);
  const [meetingLoading, setMeetingLoading] = useState("");
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedProspectId, setSelectedProspectId] = useState("");
  const [filter, setFilter] = useState("all");
  const [meetingReveal, setMeetingReveal] = useState(null);

  useEffect(() => {
    if (initial?.completed) return;

    let cancelled = false;

    (async () => {
      try {
        const res = await getDraftCombineState();

        if (cancelled) return;

        if (res?.state) setFranchiseState(res.state);
        if (res?.draft_combine) setCombine(res.draft_combine);
      } catch (e) {
        if (!cancelled) setError(e?.message || "Failed to load combine");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [initial?.completed, setFranchiseState]);

  const prospects = safeArray(combine.prospects);
  const topTesters = safeArray(combine.top_testers);
  const medical = safeArray(combine.medical_concerns);
  const bestInt = safeArray(combine.best_interviews);
  const worstInt = safeArray(combine.worst_interviews);
  const risers = safeArray(combine.late_risers);
  const fallers = safeArray(combine.late_fallers);
  const meetings = safeArray(combine.meeting_options);
  const finalRankings = safeArray(combine.final_rankings);

  const userMeetings = useMemo(() => {
    const imps = combine.user_team_impressions || {};
    return Object.keys(imps).length;
  }, [combine.user_team_impressions]);

  const meetingLimit =
    combine.meeting_limit ??
    combine.max_user_meetings ??
    combine.private_meeting_limit ??
    combine.total_meeting_slots ??
    null;

  const dinnerLimit =
    combine.private_dinner_limit ??
    combine.dinner_limit ??
    combine.total_dinner_slots ??
    null;

  const dinnersUsed = useMemo(() => {
    const impressions = combine.user_team_impressions || {};
    return Object.values(impressions).filter((item) => {
      const type = String(item?.meeting_type || item?.type || "").toLowerCase();
      return type.includes("dinner");
    }).length;
  }, [combine.user_team_impressions]);

  const meetingsRemaining =
    meetingLimit === null ? null : Math.max(0, Number(meetingLimit) - userMeetings);

  const dinnersRemaining =
    dinnerLimit === null ? null : Math.max(0, Number(dinnerLimit) - dinnersUsed);

  const allKnownProspects = useMemo(() => {
    const map = new Map();

    [
      ...prospects,
      ...topTesters,
      ...medical,
      ...bestInt,
      ...worstInt,
      ...risers,
      ...fallers,
      ...meetings,
      ...finalRankings,
    ].forEach((p) => {
      map.set(getId(p), p);
    });

    return Array.from(map.values());
  }, [
    prospects,
    topTesters,
    medical,
    bestInt,
    worstInt,
    risers,
    fallers,
    meetings,
    finalRankings,
  ]);

  useEffect(() => {
    if (!selectedProspectId && allKnownProspects.length) {
      setSelectedProspectId(getId(allKnownProspects[0]));
    }
  }, [allKnownProspects, selectedProspectId]);

  const selectedProspect = useMemo(() => {
    return allKnownProspects.find((p) => getId(p) === selectedProspectId) || null;
  }, [allKnownProspects, selectedProspectId]);

  const biggestRiser = useMemo(() => {
    return [...risers, ...allKnownProspects]
      .filter((p) => getStockDelta(p) > 0)
      .sort((a, b) => getStockDelta(b) - getStockDelta(a))[0] || null;
  }, [risers, allKnownProspects]);

  const biggestFaller = useMemo(() => {
    return [...fallers, ...allKnownProspects]
      .filter((p) => getStockDelta(p) < 0)
      .sort((a, b) => getStockDelta(a) - getStockDelta(b))[0] || null;
  }, [fallers, allKnownProspects]);

  const bestTester = useMemo(() => {
    return [...topTesters, ...allKnownProspects]
      .filter((p) => getCombineScore(p) != null)
      .sort((a, b) => Number(getCombineScore(b)) - Number(getCombineScore(a)))[0] || null;
  }, [topTesters, allKnownProspects]);

  const topMedicalFlag = useMemo(() => {
    return medical[0] || allKnownProspects.find((p) => getMedicalRisk(p)) || null;
  }, [medical, allKnownProspects]);

  const movementList = useMemo(() => {
    const map = new Map();

    [...risers, ...fallers, ...allKnownProspects]
      .filter((p) => getStockDelta(p) !== 0)
      .forEach((p) => map.set(getId(p), p));

    return Array.from(map.values()).sort(
      (a, b) => Math.abs(getStockDelta(b)) - Math.abs(getStockDelta(a))
    );
  }, [risers, fallers, allKnownProspects]);

  const filteredProspects = useMemo(() => {
    const source = prospects.length ? prospects : allKnownProspects;

    return source.filter((p) => {
      const pos = String(getPosition(p)).toLowerCase();
      const delta = getStockDelta(p);
      const medicalRisk = getMedicalRisk(p);

      if (filter === "all") return true;
      if (filter === "forwards") {
        return (
          pos.includes("c") ||
          pos.includes("lw") ||
          pos.includes("rw") ||
          pos.includes("f") ||
          pos.includes("wing")
        );
      }
      if (filter === "defense") return pos.includes("d") || pos.includes("def");
      if (filter === "goalies") return pos.includes("g") || pos.includes("goal");
      if (filter === "risers") return delta > 0;
      if (filter === "fallers") return delta < 0;
      if (filter === "medical") return !!medicalRisk;
      if (filter === "meetings") return meetings.some((m) => getId(m) === getId(p));

      return true;
    });
  }, [prospects, allKnownProspects, filter, meetings]);

  const runMeeting = useCallback(
    async (prospectId, meetingType) => {
      if (!prospectId) return;

      setMeetingLoading(`${prospectId}-${meetingType}`);
      setError("");
      setMeetingReveal(null);

      try {
        const res = await submitCombineMeeting({
          prospect_id: prospectId,
          meeting_type: meetingType,
        });

        if (res?.state) setFranchiseState(res.state);
        if (res?.draft_combine) setCombine(res.draft_combine);

        const prospect =
          meetings.find((m) => String(m.prospect_id || m.id || m.key) === String(prospectId)) ||
          allKnownProspects.find((p) => String(getId(p)) === String(prospectId));

        setMeetingReveal({
          prospect,
          meetingType,
          result:
            res?.meeting_result ||
            res?.result ||
            res?.impression ||
            res?.draft_combine?.last_meeting_result ||
            null,
        });

        if (prospect) setSelectedProspectId(getId(prospect));
      } catch (e) {
        setError(e?.message || "Meeting failed");
      } finally {
        setMeetingLoading("");
      }
    },
    [setFranchiseState, meetings, allKnownProspects]
  );

  const renderOverview = () => (
    <div className="dcb-scroll">
      <div className="dcb-stat-grid">
        <StatTile label="Invited" value={combine.invite_count || prospects.length || allKnownProspects.length} />
        <StatTile label="Medical" value={medical.length} tone={medical.length ? "risk" : undefined} />
        <StatTile label="Risers" value={risers.length} tone={risers.length ? "safe" : undefined} />
        <StatTile label="Fallers" value={fallers.length} tone={fallers.length ? "risk" : undefined} />
        <StatTile label="Meetings" value={meetingLimit === null ? userMeetings : `${userMeetings}/${meetingLimit}`} />
      </div>

      <div className="dcb-overview-grid">
        <Panel title="Biggest Riser" empty={!biggestRiser ? "—" : null}>
          <ProspectCard
            p={biggestRiser}
            selected={selectedProspectId === getId(biggestRiser)}
            onClick={() => setSelectedProspectId(getId(biggestRiser))}
          />
        </Panel>

        <Panel title="Biggest Faller" empty={!biggestFaller ? "—" : null}>
          <ProspectCard
            p={biggestFaller}
            selected={selectedProspectId === getId(biggestFaller)}
            onClick={() => setSelectedProspectId(getId(biggestFaller))}
          />
        </Panel>

        <Panel title="Best Tester" empty={!bestTester ? "—" : null}>
          <ProspectCard
            p={bestTester}
            selected={selectedProspectId === getId(bestTester)}
            onClick={() => setSelectedProspectId(getId(bestTester))}
          />
        </Panel>

        <Panel title="Medical Watch" empty={!topMedicalFlag ? "—" : null}>
          <ProspectCard
            p={topMedicalFlag}
            selected={selectedProspectId === getId(topMedicalFlag)}
            onClick={() => setSelectedProspectId(getId(topMedicalFlag))}
          />
        </Panel>
      </div>
    </div>
  );

  const renderMeetings = () => (
    <div className="dcb-meetings-grid">
      <Panel
        title="Team Meetings"
        empty={!meetings.length ? "—" : null}
        right={
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
            {meetingLimit !== null ? <Chip>{valueOrDash(meetingsRemaining)} left</Chip> : null}
            {dinnerLimit !== null ? <Chip>{valueOrDash(dinnersRemaining)} dinners</Chip> : null}
          </div>
        }
      >
        <div
          style={{
            display: "grid",
            gap: 8,
            maxHeight: 620,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          {meetings.map((m) => {
            const prospectId = m?.prospect_id || m?.id || m?.key;
            const loadingKeyInterview = `${prospectId}-interview`;
            const loadingKeyDinner = `${prospectId}-dinner`;

            return (
              <MeetingOption
                key={getId(m)}
                option={m}
                selected={selectedProspectId === getId(m)}
                onSelect={() => setSelectedProspectId(getId(m))}
                disabled={!!meetingLoading}
                loading={
                  meetingLoading === loadingKeyInterview ||
                  meetingLoading === loadingKeyDinner
                }
                onMeeting={runMeeting}
              />
            );
          })}
        </div>
      </Panel>

      <div style={{ display: "grid", gap: 12 }}>
        {meetingReveal ? (
          <Panel title="Meeting Result">
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <ProspectAvatar p={meetingReveal.prospect} />
              <div style={{ minWidth: 0 }}>
                <strong>{meetingReveal.prospect ? getName(meetingReveal.prospect) : "—"}</strong>
                <p className="dcb-empty" style={{ margin: "6px 0 0" }}>
                  {meetingReveal.result?.summary ||
                    meetingReveal.result?.note ||
                    meetingReveal.result?.message ||
                    valueOrDash(meetingReveal.meetingType)}
                </p>
              </div>
            </div>
          </Panel>
        ) : null}
      </div>
    </div>
  );

  const renderProspects = () => (
    <div className="dcb-scroll">
      <Panel
        title="Invite List"
        empty={!filteredProspects.length ? "—" : null}
        right={<span className="dcb-empty">{filteredProspects.length}</span>}
      >
        <div className="dcb-filter-row">
          <FilterButton active={filter === "all"} onClick={() => setFilter("all")}>All</FilterButton>
          <FilterButton active={filter === "forwards"} onClick={() => setFilter("forwards")}>Forwards</FilterButton>
          <FilterButton active={filter === "defense"} onClick={() => setFilter("defense")}>Defense</FilterButton>
          <FilterButton active={filter === "goalies"} onClick={() => setFilter("goalies")}>Goalies</FilterButton>
          <FilterButton active={filter === "risers"} onClick={() => setFilter("risers")}>Risers</FilterButton>
          <FilterButton active={filter === "fallers"} onClick={() => setFilter("fallers")}>Fallers</FilterButton>
          <FilterButton active={filter === "medical"} onClick={() => setFilter("medical")}>Medical</FilterButton>
          <FilterButton active={filter === "meetings"} onClick={() => setFilter("meetings")}>Meetings</FilterButton>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(255px, 1fr))",
            gap: 9,
            maxHeight: 620,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          {filteredProspects.map((p) => (
            <ProspectCard
              key={getId(p)}
              p={p}
              selected={selectedProspectId === getId(p)}
              onClick={() => setSelectedProspectId(getId(p))}
            />
          ))}
        </div>
      </Panel>
    </div>
  );

  const renderResults = () => (
    <div className="dcb-overview-grid dcb-scroll">
      <Panel title="Top Testers" empty={!topTesters.length ? "—" : null}>
        <div style={{ maxHeight: 360, overflowY: "auto", paddingRight: 4 }}>
          {topTesters.map((p) => (
            <SmallResultRow key={getId(p)} p={p} type="testing" />
          ))}
        </div>
      </Panel>

      <Panel title="Medical" empty={!medical.length ? "—" : null}>
        <div style={{ maxHeight: 360, overflowY: "auto", paddingRight: 4 }}>
          {medical.map((p) => (
            <SmallResultRow key={getId(p)} p={p} type="medical" />
          ))}
        </div>
      </Panel>

      <Panel title="Best Interviews" empty={!bestInt.length ? "—" : null}>
        <div style={{ maxHeight: 360, overflowY: "auto", paddingRight: 4 }}>
          {bestInt.map((p) => (
            <SmallResultRow key={getId(p)} p={p} type="interview" />
          ))}
        </div>
      </Panel>

      <Panel title="Interview Concerns" empty={!worstInt.length ? "—" : null}>
        <div style={{ maxHeight: 360, overflowY: "auto", paddingRight: 4 }}>
          {worstInt.map((p) => (
            <SmallResultRow key={getId(p)} p={p} type="interview" />
          ))}
        </div>
      </Panel>
    </div>
  );

  const renderBoard = () => (
    <div className="dcb-scroll" style={{ display: "grid", gap: 12 }}>
      <Panel title="Stock Movement" empty={!movementList.length ? "—" : null}>
        <div style={{ maxHeight: 320, overflowY: "auto", paddingRight: 4 }}>
          {movementList.map((p) => (
            <SmallResultRow key={getId(p)} p={p} type="movement" />
          ))}
        </div>
      </Panel>

      <Panel title="Final Rankings" empty={!finalRankings.length ? "—" : null}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(255px, 1fr))",
            gap: 9,
            maxHeight: 320,
            overflowY: "auto",
            paddingRight: 4,
          }}
        >
          {finalRankings.map((p) => (
            <ProspectCard
              key={getId(p)}
              p={p}
              compact
              selected={selectedProspectId === getId(p)}
              onClick={() => setSelectedProspectId(getId(p))}
            />
          ))}
        </div>
      </Panel>
    </div>
  );

  function renderActiveTab() {
    if (activeTab === "meetings") return renderMeetings();
    if (activeTab === "prospects") return renderProspects();
    if (activeTab === "results") return renderResults();
    if (activeTab === "board") return renderBoard();
    return renderOverview();
  }

  const navigate = (screen) => {
    if (typeof gameUI?.setScreen === "function") gameUI.setScreen(screen);
  };

  const handleBack = () => {
    if (onBack) {
      onBack();
      return;
    }
    navigate(SCREENS.CALENDAR);
  };

  const seasonLabel = () => {
    const y = franchiseState?.season_year || franchiseState?.seasonYear;
    return y ? `${y}–${Number(y) + 1}` : "Offseason";
  };

  const teamCity = () =>
    team?.city || team?.team_city || team?.market || team?.location || "Franchise";

  const teamName = () =>
    team?.name || team?.team_name || team?.full_name || team?.nickname || team?.abbr || "Team";

  const TeamLogo = () => {
    const src = getTeamLogoSrc(team);
    if (!src) {
      const label = String(team?.abbrev || team?.abbr || team?.name || "TM").slice(0, 3).toUpperCase();
      return (
        <span className="nhlcal-team-logo size-large" aria-hidden style={{ color: "var(--cyan)", fontWeight: 900, fontSize: 11 }}>
          {label}
        </span>
      );
    }
    return (
      <span className="nhlcal-team-logo size-large">
        <img src={src} alt="" loading="lazy" />
      </span>
    );
  };

  return (
    <div className="nhlcal-root dcb-root">
      <aside className="nhlcal-sidebar">
        <button type="button" className="nhlcal-brand-button" onClick={() => navigate(SCREENS.OFFICE)} title="Office">
          <span className="nhlcal-shield-icon">⌂</span>
        </button>
        <nav className="nhlcal-side-nav" aria-label="Franchise navigation">
          <button type="button" className="nhlcal-side-button" onClick={() => navigate(SCREENS.OFFICE)}>
            <span className="nhlcal-side-icon">▦</span>
            <span className="nhlcal-side-label">Office</span>
          </button>
          <button type="button" className="nhlcal-side-button" onClick={() => navigate(SCREENS.CALENDAR)}>
            <span className="nhlcal-side-icon">◫</span>
            <span className="nhlcal-side-label">Calendar</span>
          </button>
          <button type="button" className="nhlcal-side-button is-active">
            <span className="nhlcal-side-icon">◉</span>
            <span className="nhlcal-side-label">Combine</span>
          </button>
        </nav>
      </aside>

      <main className="nhlcal-main dcb-main">
        <header className="nhlcal-topbar">
          <section className="nhlcal-team-identity">
            <TeamLogo />
            <div>
              <p className="nhlcal-team-city">{teamCity()}</p>
              <h1>{teamName()}</h1>
            </div>
          </section>

          <section className="nhlcal-month-control" aria-label="Draft combine">
            <p>Offseason Event</p>
            <h2>Draft Combine</h2>
            <p className="nhlcal-subtitle">
              {loading
                ? "Loading combine data…"
                : `${combine.invite_count || prospects.length || allKnownProspects.length || 0} prospects`}
            </p>
          </section>

          <section className="nhlcal-action-cluster">
            <button type="button" className="nhlcal-quick-link" onClick={handleBack}>
              {onBack ? "Hub World" : "Calendar"}
            </button>
            <div className="nhlcal-date-chip">
              <span className="nhlcal-date-icon">◫</span>
              <div>
                <strong>{combine.draft_year ? `${combine.draft_year} Draft` : seasonLabel()}</strong>
                <span>Combine Week</span>
              </div>
            </div>
            {onContinue ? (
              <button
                type="button"
                className="nhlcal-advance-button"
                disabled={loading}
                onClick={onContinue}
              >
                Enter Entry Draft
              </button>
            ) : null}
          </section>
        </header>

        {!loading ? (
          <p className="dcb-summary-line">
            {userMeetings ? `${userMeetings} meetings` : "No meetings yet"}
            {medical.length ? ` · ${medical.length} medical` : ""}
            {movementList.length ? ` · ${movementList.length} moves` : ""}
          </p>
        ) : null}

        {error ? <p className="dcb-error">{error}</p> : null}

        <nav className="dcb-tab-bar" aria-label="Combine sections">
          {TABS.map((tab) => (
            <TabButton
              key={tab.id}
              active={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </TabButton>
          ))}
        </nav>

        <section className="nhlcal-content-grid dcb-grid">
          <section className="nhlcal-calendar-panel dcb-panel">
            {renderActiveTab()}
          </section>
          <aside className="nhlcal-right-rail dcb-detail">
            <DetailPanel prospect={selectedProspect} />
          </aside>
        </section>

        <div className="dcb-timeline-wrap">
          <OffseasonTimeline franchiseState={franchiseState} />
        </div>
      </main>
    </div>
  );
}