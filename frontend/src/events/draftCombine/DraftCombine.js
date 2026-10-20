import React, { useCallback, useEffect, useMemo, useState } from "react";
import OffseasonTimeline from "../offseasonTimeline";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { getTeamLogoSrc } from "../../utils/teamLogos";
import { getDraftCombineState, submitCombineMeeting } from "../../services/franchiseService";
import { safeArray } from "../shared/eventHelpers";

/* Draft Combine — one screen: movers, top performers, full results table, inline detail. */

const STANDOUT = 85;
const RED_FLAG = 15;

function fmtTest(cat, v) {
  if (v === null || v === undefined || v === "") return "—";
  if (cat?.id === "height" || cat?.id === "wingspan") {
    const n = Number(v);
    return `${Math.floor(n / 12)}'${(n % 12).toFixed(n % 1 ? 1 : 0)}"`;
  }
  const d = Number(cat?.decimals ?? 1);
  return Number(v).toFixed(d);
}

const getId = (p) => String(p?.key || p?.prospect_id || p?.id || p?.name || "");
const getName = (p) => p?.name || p?.prospect_name || p?.full_name || "—";

function rankChange(p) {
  if (p?.combine_rank_change != null) return Number(p.combine_rank_change) || 0;
  return Number(p?.combine_stock_delta) || 0;
}

function cellFor(p, c) {
  const t = p?.tests?.[c.id];
  if (t) return { v: t.value, pct: t.pct };
  const m = p?.measurements?.[c.id];
  return { v: m, pct: p?.measurements?.[`${c.id}_pct`] };
}

function pctTone(pct, kind) {
  if (pct == null || kind === "measurement") return "";
  if (pct >= STANDOUT) return "hi";
  if (pct <= RED_FLAG) return "lo";
  return "";
}

function moverReason(p) {
  const t = p?.combine_stock_test;
  const base = p?.combine_stock_reason || p?.stock_reason || "";
  if (t && /test/i.test(base)) return `${t.short || t.label} ${t.value}, ${
      t.pct >= 99 ? "best in class" : t.pct <= 1 ? "last in class" : t.pct >= 50 ? `top ${100 - t.pct}%` : `bottom ${t.pct}%`
    }`;
  // A faller who "met expectations" dropped because others tested past him; say that.
  if ((!base || /met expectations/i.test(base)) && rankChange(p) < 0) return "Tested fine, others moved past him";
  return base || "Met expectations";
}

function MoverRow({ p, selected, onSelect }) {
  const d = rankChange(p);
  const pre = p?.pre_combine_rank;
  const post = p?.post_combine_rank ?? p?.rank;
  return (
    <button type="button" className={`dcb-mover${selected ? " is-selected" : ""}`} onClick={onSelect}>
      <span className={`dcb-arrow ${d > 0 ? "up" : "down"}`}>{d > 0 ? "▲" : "▼"}{Math.abs(d)}</span>
      <span className="dcb-mover-main">
        <strong>{getName(p)}</strong>
        <em>{moverReason(p)}</em>
      </span>
      <span className="dcb-mover-rank">{pre ? `#${pre} → ` : ""}#{post ?? "—"}</span>
    </button>
  );
}

function DetailPanel({ p, catalog, meeting, meetingLoading, onMeeting }) {
  if (!p) return <p className="dcb-empty">Click a prospect to see their combine file.</p>;
  const d = rankChange(p);
  const med = p.medical || {};
  return (
    <div className="dcb-detail-body">
      <div className="dcb-detail-head">
        <h3>{getName(p)}</h3>
        <p>
          {[p.position, p.league_display || p.league, p.post_combine_rank || p.rank ? `#${p.post_combine_rank || p.rank}` : null]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      <dl className="dcb-kv">
        <div><dt>Athletic</dt><dd>{p.athletic_pct != null ? `${p.athletic_pct} pct` : "—"}{p.athletic_rank ? ` · #${p.athletic_rank}` : ""}</dd></div>
        <div><dt>Combine</dt><dd>{p.combine_label || "—"}</dd></div>
        <div><dt>Interview</dt><dd>{p.interview_score ?? "—"}</dd></div>
        <div><dt>Medical</dt><dd className={p.medical_risk_level === "High" ? "lo" : ""}>{p.medical_risk_level || "Low"}</dd></div>
        <div>
          <dt>Stock</dt>
          <dd className={d > 0 ? "hi" : d < 0 ? "lo" : ""}>
            {d ? `${d > 0 ? "▲" : "▼"}${Math.abs(d)}` : "Flat"}
            {p.pre_combine_rank ? ` (#${p.pre_combine_rank} → #${p.post_combine_rank})` : ""}
          </dd>
        </div>
      </dl>
      {d ? <p className="dcb-note">{moverReason(p)}</p> : null}
      {med.note && p.medical_risk_level !== "Low" ? <p className="dcb-note lo">{med.note}</p> : null}
      <table className="dcb-mini">
        <tbody>
          {catalog.map((c) => {
            const { v, pct } = cellFor(p, c);
            if (v == null) return null;
            return (
              <tr key={c.id}>
                <th>{c.label}</th>
                <td>{fmtTest(c, v)}{c.unit && c.id !== "height" && c.id !== "wingspan" ? ` ${c.unit}` : ""}</td>
                <td className={pctTone(pct, c.kind)}>{pct ?? ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {meeting ? (
        <div className="dcb-meet">
          {meeting.already_met ? (
            <span className="dcb-note">Meeting logged</span>
          ) : (
            <>
              <button type="button" disabled={!!meetingLoading} onClick={() => onMeeting(meeting.prospect_id, "interview")}>
                Interview
              </button>
              <button type="button" disabled={!!meetingLoading} onClick={() => onMeeting(meeting.prospect_id, "dinner")}>
                Dinner
              </button>
            </>
          )}
        </div>
      ) : null}
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
  const [meetingNote, setMeetingNote] = useState("");
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [sortKey, setSortKey] = useState("combine");
  const [desc, setDesc] = useState(true);

  useEffect(() => {
    if (initial?.completed) return undefined;
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
  const catalog = safeArray(combine.test_catalog);
  const fitness = useMemo(() => catalog.filter((c) => c.kind !== "measurement"), [catalog]);
  const meetings = safeArray(combine.meeting_options);
  const risers = safeArray(combine.late_risers);
  const fallers = safeArray(combine.late_fallers);

  const byId = useMemo(() => new Map(prospects.map((p) => [getId(p), p])), [prospects]);
  const selected = byId.get(selectedId) || null;
  const meeting = meetings.find((m) => String(m.prospect_id) === selectedId) || null;

  const leaders = useMemo(
    () =>
      fitness
        .map((c) => {
          let best = null;
          prospects.forEach((p) => {
            const t = p?.tests?.[c.id];
            if (t?.pct != null && (!best || t.pct > best.t.pct)) best = { p, t };
          });
          return best ? { c, ...best } : null;
        })
        .filter(Boolean),
    [fitness, prospects]
  );

  const sorted = useMemo(() => {
    const val = (p) => {
      if (sortKey === "combine") return Number(p.athletic_pct ?? -1);
      if (sortKey === "rank") return -Number(p.post_combine_rank || p.rank || 999);
      if (sortKey === "move") return rankChange(p);
      const c = catalog.find((x) => x.id === sortKey);
      const { v, pct } = cellFor(p, c || { id: sortKey });
      if (pct != null) return Number(pct);
      return v != null ? Number(v) : -1;
    };
    return [...prospects].sort((a, b) => (val(b) - val(a)) * (desc ? 1 : -1));
  }, [prospects, catalog, sortKey, desc]);

  const head = (id, label, title) => (
    <th
      key={id}
      title={title}
      onClick={() => (sortKey === id ? setDesc(!desc) : (setSortKey(id), setDesc(true)))}
      className={sortKey === id ? "is-sorted" : ""}
    >
      {label}
      {sortKey === id ? (desc ? " ↓" : " ↑") : ""}
    </th>
  );

  const runMeeting = useCallback(
    async (prospectId, meetingType) => {
      if (!prospectId) return;
      setMeetingLoading(`${prospectId}-${meetingType}`);
      setError("");
      setMeetingNote("");
      try {
        const res = await submitCombineMeeting({ prospect_id: prospectId, meeting_type: meetingType });
        if (res?.state) setFranchiseState(res.state);
        if (res?.draft_combine) setCombine(res.draft_combine);
        const r = res?.meeting_result || res?.result || res?.impression || res?.draft_combine?.last_meeting_result;
        setMeetingNote(r?.summary || r?.note || r?.message || `${meetingType} logged`);
      } catch (e) {
        setError(e?.response?.data?.detail || e?.message || "Meeting failed");
      } finally {
        setMeetingLoading("");
      }
    },
    [setFranchiseState]
  );

  const navigate = (screen) => {
    if (typeof gameUI?.setScreen === "function") gameUI.setScreen(screen);
  };
  const handleBack = () => (onBack ? onBack() : navigate(SCREENS.CALENDAR));
  const logo = getTeamLogoSrc(team);

  return (
    <div className="nhlcal-root dcb-root edraft-skin">
      <main className="nhlcal-main dcb-main">
        <header className="nhlcal-topbar">
          <section className="nhlcal-team-identity">
            {logo ? (
              <span className="nhlcal-team-logo size-large">
                <img src={logo} alt="" loading="lazy" />
              </span>
            ) : null}
            <div>
              <p className="nhlcal-team-city">{team?.city || team?.market || "Franchise"}</p>
              <h1>{team?.name || team?.team_name || team?.abbr || "Team"}</h1>
            </div>
          </section>
          <section className="nhlcal-month-control" aria-label="Draft combine">
            <p>{combine.draft_year ? `${combine.draft_year} NHL Draft` : "Offseason Event"}</p>
            <h2>Draft Combine</h2>
            <p className="nhlcal-subtitle">
              {loading ? "Loading combine data…" : `${combine.invite_count || prospects.length} invited · ${fitness.length} tests`}
            </p>
          </section>
          <section className="nhlcal-action-cluster">
            <button type="button" className="nhlcal-quick-link" onClick={handleBack}>
              {onBack ? "Hub World" : "Calendar"}
            </button>
            {onContinue ? (
              <button type="button" className="nhlcal-advance-button" disabled={loading} onClick={onContinue}>
                Enter Entry Draft
              </button>
            ) : null}
          </section>
        </header>

        {error ? <p className="dcb-error">{error}</p> : null}

        <section className="dcb-layout">
          <div className="dcb-col">
            <section className="dcb-movers">
              <div>
                <h3 className="dcb-h">Risers</h3>
                {risers.length ? (
                  risers.map((p) => (
                    <MoverRow key={getId(p)} p={p} selected={selectedId === getId(p)} onSelect={() => setSelectedId(getId(p))} />
                  ))
                ) : (
                  <p className="dcb-empty">No significant risers.</p>
                )}
              </div>
              <div>
                <h3 className="dcb-h">Fallers</h3>
                {fallers.length ? (
                  fallers.map((p) => (
                    <MoverRow key={getId(p)} p={p} selected={selectedId === getId(p)} onSelect={() => setSelectedId(getId(p))} />
                  ))
                ) : (
                  <p className="dcb-empty">No significant fallers.</p>
                )}
              </div>
            </section>

            {leaders.length ? (
              <section className="dcb-leaders" aria-label="Top performer per test">
                {leaders.map(({ c, p, t }) => (
                  <button key={c.id} type="button" className="dcb-leader" onClick={() => setSelectedId(getId(p))} title={c.label}>
                    <span>{c.short || c.label}</span>
                    <strong>{getName(p)}</strong>
                    <em>{fmtTest(c, t.value)} {c.unit || ""}</em>
                  </button>
                ))}
              </section>
            ) : null}

            <section className="dcb-table-wrap">
              {!prospects.length || !catalog.length ? (
                <p className="dcb-empty">{loading ? "Loading…" : "Testing results publish when the combine runs."}</p>
              ) : (
                <table className="dcb-test-table">
                  <thead>
                    <tr>
                      {head("rank", "Board", "Post-combine rank")}
                      <th className="who">Prospect</th>
                      {head("move", "Move", "Rank change at the combine")}
                      {head("combine", "Ath", "Athletic index percentile")}
                      {catalog.map((c) => head(c.id, c.short || c.label, `${c.label}${c.unit ? ` (${c.unit})` : ""}`))}
                      <th>Med</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((p) => {
                      const d = rankChange(p);
                      return (
                        <tr
                          key={getId(p)}
                          className={selectedId === getId(p) ? "is-selected" : ""}
                          onClick={() => setSelectedId(getId(p))}
                        >
                          <td>{p.post_combine_rank || p.rank || "—"}</td>
                          <td className="who">
                            <strong>{getName(p)}</strong>
                            <span>{[p.position, p.league_display || p.league].filter(Boolean).join(" · ")}</span>
                          </td>
                          <td className={d > 0 ? "hi" : d < 0 ? "lo" : ""}>{d ? `${d > 0 ? "▲" : "▼"}${Math.abs(d)}` : "—"}</td>
                          <td className={pctTone(p.athletic_pct, "fitness")}>{p.athletic_pct ?? "—"}</td>
                          {catalog.map((c) => {
                            const { v, pct } = cellFor(p, c);
                            return (
                              <td key={c.id} className={pctTone(pct, c.kind)}>
                                {fmtTest(c, v)}
                                {pct != null ? <span>{pct}</span> : null}
                              </td>
                            );
                          })}
                          <td className={p.medical_risk_level === "High" ? "lo" : ""}>{p.medical_risk_level || "Low"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
              <p className="dcb-legend">
                Small number = class percentile. <span className="hi">Standout</span> ≥ {STANDOUT}th · <span className="lo">Red flag</span> ≤ {RED_FLAG}th. Click a header to sort.
              </p>
            </section>
          </div>

          <aside className="dcb-detail">
            <DetailPanel
              p={selected}
              catalog={catalog}
              meeting={meeting}
              meetingLoading={meetingLoading}
              onMeeting={runMeeting}
            />
            {meetingNote ? <p className="dcb-note">{meetingNote}</p> : null}
          </aside>
        </section>

        <div className="dcb-timeline-wrap">
          <OffseasonTimeline franchiseState={franchiseState} />
        </div>
      </main>
    </div>
  );
}
