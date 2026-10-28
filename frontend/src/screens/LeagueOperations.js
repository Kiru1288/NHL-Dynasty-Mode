import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getGovernance, getLeagueOperations } from "../services/franchiseService";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

const EMPTY_OBJ = Object.freeze({});
const EMPTY_ARR = Object.freeze([]);

const ACTION_STATES = ["Stable", "Monitoring", "Negotiating", "Vote Required"];
const SORT_KEYS = ["revenue", "trend", "market", "risk", "profit"];
const WORKSPACES = [
  { id: "overview", label: "Overview" },
  { id: "cba", label: "CBA" },
  { id: "markets", label: "Markets" },
  { id: "risk", label: "Franchise Risk" },
  { id: "board", label: "Board & Values" },
];
const MARKET_FILTERS = [
  { id: "all", label: "All" },
  { id: "growing", label: "Growing" },
  { id: "losing", label: "Losing money" },
  { id: "reloc", label: "Relocation Risk" },
];
const CALM_MOODS = new Set(["calm", "quiet", "stable"]);



function BoardWorkspace({ userTeamId }) {
  const [gov, setGov] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    let alive = true;
    getGovernance()
      .then((d) => {
        if (alive) setGov(d?.governance || null);
      })
      .catch((e) => {
        if (alive) setErr(e?.message || "Board data unavailable");
      });
    return () => {
      alive = false;
    };
  }, []);
  if (!gov) {
    return <div className="lo-empty">{err || "Loading Board of Governors…"}</div>;
  }
  const impact = asObject(gov.user_impact);
  const rules = asArray(gov.rulebook);
  const hist = asArray(gov.history);
  const values = asArray(gov.franchise_values);
  const relocs = asArray(gov.relocations);
  const exp = asObject(gov.expansion);
  const pending = asArray(exp.pending);
  const joined = asArray(exp.joined);
  const valueChange = Number(impact.value_change_pct || 0);
  return (
    <div className="lo-board lo-enter">
      <div className="lo-board-col">
        <section className="lo-board-panel">
          <h3 className="lo-section-title">Rules in force ({rules.length})</h3>
          {rules.length ? (
            <div className="lo-table-wrap" style={{ height: "auto", maxHeight: 320 }}>
              <table className="lo-table">
                <thead>
                  <tr>
                    <th style={{ width: "10%" }}>Year</th>
                    <th style={{ width: "34%" }}>Rule</th>
                    <th style={{ width: "18%" }}>Area</th>
                    <th>Effect</th>
                  </tr>
                </thead>
                <tbody>
                  {rules.map((r) => (
                    <tr key={r.rule_id}>
                      <td>{r.season}</td>
                      <td title={r.summary}>{r.title}</td>
                      <td>{r.category_label}</td>
                      <td className="lo-board-effects">{asArray(r.effects).map((e) => `${e.label} ${e.text}`).join(" · ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="lo-board-note">No Board rules yet. The Board meets every offseason, after Retirements.</p>
          )}
        </section>
        <section className="lo-board-panel">
          <h3 className="lo-section-title">Vote record</h3>
          {hist.length ? (
            <div className="lo-table-wrap" style={{ height: "auto", maxHeight: 300 }}>
              <table className="lo-table">
                <thead>
                  <tr>
                    <th style={{ width: "10%" }}>Year</th>
                    <th style={{ width: "46%" }}>Proposal</th>
                    <th style={{ width: "18%" }}>Result</th>
                    <th>Your vote</th>
                  </tr>
                </thead>
                <tbody>
                  {hist.map((h) => (
                    <tr key={h.proposal_id}>
                      <td>{h.season}</td>
                      <td>{h.title}</td>
                      <td className={h.status === "passed" ? "lo-board-pass" : "lo-board-fail"}>
                        {h.status === "passed" ? "Passed" : "Failed"} {h.yes}–{h.no}
                      </td>
                      <td>{h.user_vote}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="lo-board-note">No votes yet.</p>
          )}
        </section>
      </div>
      <div className="lo-board-col">
        <section className="lo-board-panel">
          <h3 className="lo-section-title">How revenue reaches your club</h3>
          <div className="lo-board-kv">
            <span>Annual revenue</span>
            <strong>{impact.annual_revenue_m != null ? `$${Number(impact.annual_revenue_m).toFixed(1)}M` : "—"}</strong>
            <span>Franchise value</span>
            <strong>{impact.franchise_value_b != null ? `$${Number(impact.franchise_value_b).toFixed(2)}B` : "—"}</strong>
            <span>Value change (last offseason)</span>
            <strong className={valueChange > 0 ? "up" : valueChange < 0 ? "down" : ""}>
              {`${valueChange >= 0 ? "+" : ""}${valueChange.toFixed(1)}%`}
            </strong>
            <span>Scouting budget (revenue-funded)</span>
            <strong>{impact.scouting_budget ? `$${(Number(impact.scouting_budget) / 1e6).toFixed(2)}M` : "—"}</strong>
            <span>Signing-bonus room</span>
            <strong className={impact.bonus_eligible ? "up" : "down"}>
              {impact.bonus_eligible
                ? `Up to ${Math.round(Number(impact.bonus_max_pct || 0) * 100)}% of a deal`
                : `Locked: needs $${Number(impact.bonus_floor_m || 155).toFixed(0)}M revenue`}
            </strong>
            <span>Star revenue spike next season</span>
            <strong>{impact.star_spike_next_m ? `+$${Number(impact.star_spike_next_m).toFixed(1)}M` : "—"}</strong>
          </div>
          <p className="lo-board-note" style={{ marginTop: 8 }}>
            Some free agents only sign where they get a signing bonus, so clubs below the revenue floor can't land them.
          </p>
        </section>
        <section className="lo-board-panel">
          <h3 className="lo-section-title">Franchise values</h3>
          <div className="lo-table-wrap" style={{ height: "auto", maxHeight: 360 }}>
            <table className="lo-table">
              <thead>
                <tr>
                  <th style={{ width: "34%" }}>Club</th>
                  <th style={{ width: "18%" }}>Value</th>
                  <th style={{ width: "16%" }}>Change</th>
                  <th>Why</th>
                </tr>
              </thead>
              <tbody>
                {values.map((v) => (
                  <tr key={v.team_id} className={String(v.team_id) === String(userTeamId) ? "user" : ""}>
                    <td>
                      {v.abbr} · {v.name}
                    </td>
                    <td>${Number(v.value_b).toFixed(2)}B</td>
                    <td className={Number(v.change_pct) >= 0 ? "lo-board-pass" : "lo-board-fail"}>
                      {`${Number(v.change_pct) >= 0 ? "+" : ""}${Number(v.change_pct || 0).toFixed(1)}%`}
                    </td>
                    <td className="lo-board-effects">{asArray(v.drivers).join(", ") || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="lo-board-panel">
          <h3 className="lo-section-title">Relocation &amp; expansion</h3>
          <ul className="lo-board-list">
            {relocs.map((r) => (
              <li key={`${r.season}-${r.team_id}`}>
                <span>
                  {r.season}: {r.from_city} → {r.to_city} ({r.to_abbr})
                </span>
                <span>
                  {asArray(r.demands).length} trade request{asArray(r.demands).length === 1 ? "" : "s"}
                </span>
              </li>
            ))}
            {pending.map((p, i) => (
              <li key={`pending-${i}`}>
                <span>Approved: {asArray(p.cities).map((c) => c.city).join(" & ")}</span>
                <span>
                  Expansion draft: {p.draft_offseason || Number(p.start_season) - 1} Board meeting · joins{" "}
                  {p.start_season}-{String((Number(p.start_season) + 1) % 100).padStart(2, "0")}
                </span>
              </li>
            ))}
            {joined.map((j) => (
              <li key={`joined-${j.team_id}`}>
                <span>
                  {j.city} {j.name} ({j.abbr})
                </span>
                <span>Joined {j.season}</span>
              </li>
            ))}
            {!relocs.length && !pending.length && !joined.length ? (
              <li>
                <span>No moves or new clubs yet.</span>
              </li>
            ) : null}
          </ul>
        </section>
      </div>
    </div>
  );
}

function InlineStyles() {
  return null;
}

function asObject(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? v : EMPTY_OBJ;
}

function asArray(v) {
  return Array.isArray(v) ? v : EMPTY_ARR;
}

function num(v, fb = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fb;
}

function fmtMoneyM(v) {
  return `$${Math.round(num(v, 0))}M`;
}

function fmtCap(v) {
  // Two decimals, same as Contracts and the Trade Desk.
  return `$${num(v, 0).toFixed(2)}M`;
}

function riskClass(label) {
  const s = String(label || "").toLowerCase();
  if (s.includes("critical") || s.includes("high")) return "lo-risk-high";
  if (s.includes("med")) return "lo-risk-med";
  return "lo-risk-low";
}

function isThreatened(team) {
  const score = num(team.relocation_risk, 0);
  const label = String(team.relocation_risk_label || team.risk_label || "").toLowerCase();
  return score >= 0.55 || label.includes("critical") || label.includes("high");
}

function deriveDisplayStatus(team) {
  const profit = num(team.profit, 0);
  const risk = num(team.relocation_risk, 0);
  if (profit >= 18) return "Surge";
  if (profit >= 8) return "Profit";
  if (profit >= 2) return "Thin";
  if (profit >= -2) return "Flat";
  if (risk >= 0.55 || profit < -10) return "Risk";
  if (profit < -2) return "Loss";
  return "Flat";
}

function deriveRelocReason(row) {
  const pressure = String(row.pressure || row.market_pressure || "").trim();
  const reason = String(row.reason || "").trim();
  if (reason && reason !== "Market" && reason !== "Stable") return reason;
  if (pressure && pressure !== "Stable") {
    const map = {
      Gate: "Gate",
      Arena: "Arena",
      Revenue: "Revenue",
      "Small Market": "Small Market",
      Owner: "Owner",
      Attendance: "Fan Drop",
      Lease: "Lease",
    };
    return map[pressure] || pressure;
  }
  const tags = asArray(row.reason_tags);
  if (tags.includes("Arena Drag")) return "Arena";
  if (tags.includes("Weak Gate")) return "Gate";
  if (tags.includes("Fan Freeze")) return "Fan Drop";
  if (tags.includes("Small Loss")) return "Revenue";
  if (row.market_tier === "Small") return "Small Market";
  return reason || "Lease";
}

function deriveCapDrivers(data) {
  const existing = asArray(data.cap_drivers);
  if (existing.length) return existing;

  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const smallBe = num(data.small_market_break_even, 0.5);
  const losing = num(data.losing_teams_count, 0);
  const cbaP = num(asObject(data.cba).pressure, 0.3);
  const teams = asArray(data.teams);
  const avgFan = teams.reduce((s, t) => s + num(t.fan_sentiment, 55), 0) / Math.max(teams.length, 1);

  return [
    { label: "Escrow", sign: escrow >= 1 ? "+" : "-" },
    { label: "Revenue", sign: health >= 0.55 ? "+" : "-" },
    { label: "Small Teams", sign: smallBe >= 0.65 ? "+" : "-" },
    { label: "Loss Teams", sign: losing <= 10 ? "+" : "-" },
    { label: "CBA", sign: cbaP < 0.55 ? "+" : "-" },
    { label: "Fan Heat", sign: avgFan >= 52 ? "+" : "-" },
  ];
}

function deriveOwnerMood(data) {
  const m = asObject(data.owner_mood);
  if (m.owners && m.owners !== "—") return m;

  const capType = String(data.cap_change_type || "");
  const health = num(data.revenue_health, 0.5);
  const cbaP = num(asObject(data.cba).pressure, 0.3);
  const losing = num(data.losing_teams_count, 0);

  let owners = "Calm";
  if (cbaP >= 0.65) owners = "Cautious";
  else if (cbaP >= 0.45) owners = "Watchful";

  let players = "Stable";
  if (capType.includes("Drop") || capType.includes("Freeze")) players = "Angry";
  else if (capType.includes("Jump")) players = "Pushy";
  else if (cbaP >= 0.5) players = "Restless";

  let fans = "Stable";
  if (health >= 0.62) fans = "Strong";
  else if (health < 0.42) fans = "Cold";

  let media = "Quiet";
  if (losing >= 18 || cbaP >= 0.6) media = "Loud";
  else if (cbaP >= 0.4) media = "Watching";

  return { owners, players, fans, media };
}



function deriveTeamEvent(team) {
  if (isThreatened(team)) {
    return { kind: "risk", label: "Relocation" };
  }
  const stars = asArray(team.superstar_tags);
  if (stars.length) {
    return { kind: "star", label: "Star" };
  }
  const yoy = num(team.revenue_yoy_delta, 0);
  const dir = String(team.revenue_yoy_direction || "");
  if (
    Math.abs(yoy) >= 8
    || (dir === "up" && Math.abs(yoy) >= 5)
    || (dir === "down" && Math.abs(yoy) >= 5)
  ) {
    return { kind: "rev", label: (yoy >= 0 || dir === "up") ? "Spike" : "Drop" };
  }
  return null;
}

function normalizeTeams(rawTeams) {
  const sorted = [...asArray(rawTeams)].sort((a, b) => num(b.revenue, 0) - num(a.revenue, 0));
  const maxRev = Math.max(...sorted.map((t) => num(t.revenue, 0)), 1);

  return sorted.map((t, index) => {
    const rank = num(t.rank, 0) > 0 ? num(t.rank, 0) : index + 1;
    const riskScore = num(t.relocation_risk, 0);
    const riskLabel = t.relocation_risk_label || (riskScore >= 0.55 ? "High" : riskScore >= 0.35 ? "Med" : "Low");
    return {
      ...t,
      rank,
      display_status: t.revenue_status || deriveDisplayStatus(t),
      revenue_bar_pct: num(t.revenue_bar_pct, 0) > 0 ? num(t.revenue_bar_pct, 0) : (num(t.revenue, 0) / maxRev) * 100,
      relocation_risk: riskScore,
      relocation_risk_label: riskLabel,
      reloc_reason: deriveRelocReason(t),
      threatened: isThreatened({ ...t, relocation_risk: riskScore, relocation_risk_label: riskLabel }),
      event: deriveTeamEvent({ ...t, relocation_risk: riskScore, relocation_risk_label: riskLabel }),
    };
  });
}

function synthesizeCapSeries(data) {
  const existing =
    asArray(data.cap_forecast).length
      ? asArray(data.cap_forecast)
      : asArray(data.cap_history).length
        ? asArray(data.cap_history)
        : asArray(data.projected_seasons);

  if (existing.length >= 3) {
    return existing.slice(0, 6).map((row, i) => {
      const year = num(row.year ?? row.season ?? row.season_year, num(data.current_year, 2025) + i);
      const cap = num(row.cap ?? row.salary_cap ?? row.value ?? row.projected, num(data.salary_cap, 88));
      const low = num(row.low ?? row.cap_low, cap * 0.97);
      const high = num(row.high ?? row.cap_high, cap * 1.03);
      return { year, label: String(year), cap, low, high };
    });
  }

  const currentYear = num(data.current_year, 2025);
  const current = num(data.salary_cap ?? data.cap ?? data.current_cap, 88);
  const projected = num(data.projected_salary_cap, current + num(data.cap_change, 0));
  const change = projected - current;
  const type = String(data.cap_change_type || "");
  let growth = change;
  if (Math.abs(growth) < 0.05) {
    if (type.includes("Jump")) growth = current * 0.05;
    else if (type.includes("Rise")) growth = current * 0.025;
    else if (type.includes("Drop")) growth = -current * 0.015;
    else growth = current * 0.01;
  }

  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const uncertainty = Math.max(0.8, (Math.abs(1 - escrow) * 8 + Math.abs(0.55 - health) * 10 + 1.2));

  const series = [];
  for (let i = 0; i < 5; i++) {
    const year = currentYear + i;
    const decay = Math.pow(0.92, i);
    const cap = i === 0 ? current : current + growth * i * (0.85 + decay * 0.2);
    const band = uncertainty * (0.7 + i * 0.35);
    series.push({
      year,
      label: String(year),
      cap: Math.round(cap * 10) / 10,
      low: Math.round((cap - band) * 10) / 10,
      high: Math.round((cap + band) * 10) / 10,
    });
  }

  if (series.length > 1) {
    series[1].cap = Math.round(projected * 10) / 10;
    series[1].low = Math.round((projected - uncertainty) * 10) / 10;
    series[1].high = Math.round((projected + uncertainty) * 10) / 10;
  }

  return series;
}

function deriveConfidence(data, series) {
  const next = series[1] || series[0] || { low: 0, high: 0, cap: 0 };
  const escrow = num(data.escrow_progress, 1);
  const health = num(data.revenue_health, 0.5);
  const width = Math.max(0.5, next.high - next.low);
  const mid = next.cap || (next.low + next.high) / 2;
  const confidencePct = Math.round(Math.max(40, Math.min(92, 88 - Math.abs(1 - escrow) * 35 - Math.abs(0.55 - health) * 40)));
  const confidenceLevel = confidencePct >= 75 ? "High" : confidencePct >= 55 ? "Medium" : "Low";
  return {
    low: next.low,
    high: next.high,
    mid,
    width,
    label: `${fmtCap(next.low)} – ${fmtCap(next.high)}`,
    confidencePct,
    confidenceLevel,
    rangeLabel: `Projected Range ${fmtCap(next.low)}–${fmtCap(next.high)} · ${confidenceLevel} Confidence`,
  };
}

function assignActionState(row) {
  const status = String(row.status || row.state || "").toLowerCase();
  const likelihood = num(row.likelihood ?? row.support_pct, 0);
  const owners = num(row.owner_support, 0);
  const players = num(row.player_support, 0);
  const gap = Math.abs(owners - players);
  const voteReady = likelihood >= 72 && gap <= 12;

  if (status.includes("vote") || voteReady) return "Vote Required";
  if (status === "open" || status.includes("negotiat")) return "Negotiating";
  if (status === "likely" || status.includes("monitor")) return "Monitoring";
  if (status === "long") return "Stable";
  if (gap >= 18) return "Negotiating";
  if (likelihood >= 48) return "Monitoring";
  if (ACTION_STATES.includes(row.status)) return row.status;
  return "Stable";
}

function issueUrgency(issue) {
  const stateRank = { "Vote Required": 4, Negotiating: 3, Monitoring: 2, Stable: 1 };
  const impact = num(issue.likelihood, 0) * 0.55 + num(issue.support_pct, 0) * 0.25 + (stateRank[issue.action_state] || 1) * 18;
  return impact;
}

function clipWords(text, max = 15) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "";
  return words.slice(0, max).join(" ");
}

function issueConsequence(issue) {
  const note = String(issue.consequence || issue.impact || issue.summary || issue.detail || "").trim();
  if (note) return clipWords(note, 15);

  const status = String(issue.status || issue.action_state || "Open");
  const likelihood = num(issue.likelihood, 0);
  const owners = num(issue.owner_support, 0);
  const players = num(issue.player_support, 0);
  const name = String(issue.name || "Issue");

  if (owners || players) {
    const lean =
      owners > players + 6
        ? "Owners currently lead."
        : players > owners + 6
          ? "Players currently lead."
          : "Sides are nearly even.";
    return clipWords(`${name}: ${status} at ${likelihood}%. ${lean}`, 15);
  }

  if (status === "Vote Required") return clipWords(`${name} may reshape next-season costs.`, 15);
  if (status === "Negotiating" || status === "Open") return clipWords(`${name} talks can shift cap room.`, 15);
  if (status === "Monitoring" || status === "Likely") return clipWords(`${name} is ${likelihood}% likely this cycle.`, 15);
  return clipWords(`${name} sits on a longer runway.`, 15);
}

function issueDeadline(issue) {
  const explicit = issue.deadline || issue.when || issue.window;
  if (explicit) return String(explicit).slice(0, 16);
  const status = String(issue.status || "").toLowerCase();
  if (status === "likely") return "Near term";
  if (status === "open") return "Active window";
  if (status === "long") return "Long runway";
  if (issue.bargaining_deadline) return String(issue.bargaining_deadline).slice(0, 10);
  return "Ongoing";
}

function deriveActiveIssues(data) {
  const cba = asObject(data.cba);
  const bargainingDeadline = cba.bargaining_deadline || null;
  const fromNeg = asArray(cba.potential_changes).map((row) => ({
    ...row,
    id: `cba-${row.name}`,
    source: "CBA",
    kind: "issue",
    bargaining_deadline: bargainingDeadline,
  }));
  const fromRules = asArray(data.rule_changes).map((row) => ({
    ...row,
    id: `rule-${row.name}`,
    source: "Rule",
    kind: "issue",
    bargaining_deadline: bargainingDeadline,
  }));

  const merged = new Map();
  [...fromNeg, ...fromRules].forEach((row) => {
    const key = String(row.name || row.id);
    if (!merged.has(key)) merged.set(key, row);
  });

  return [...merged.values()]
    .map((row) => {
      const action_state = assignActionState(row);
      const owners = num(row.owner_support, 0);
      const players = num(row.player_support, 0);
      const support = num(row.support_pct ?? row.likelihood, Math.round((owners + players) / 2));
      const enriched = {
        ...row,
        action_state,
        support_pct: support,
        scale_label:
          owners || players
            ? `Owners ${owners}% · Players ${players}%`
            : `Support ${support}%`,
      };
      return {
        ...enriched,
        deadline: issueDeadline(enriched),
        consequence: issueConsequence(enriched),
      };
    })
    .sort((a, b) => issueUrgency(b) - issueUrgency(a))
    .slice(0, 5);
}

function deriveCbaTimeline(data) {
  const cba = asObject(data.cba);
  const year = num(data.current_year, 2025);
  const endYear = num(cba.end_year, year + num(cba.years_remaining, 3));
  const startYear = num(cba.start_year, endYear - 7);
  const deadline = String(cba.bargaining_deadline || `${endYear - 1}-06-30`).slice(0, 10);
  const yearsLeft = num(cba.years_remaining, Math.max(0, endYear - year));
  const pressure = String(cba.pressure_level || "Low");
  const items = [
    {
      id: "start",
      when: String(startYear),
      title: cba.current_agreement || `CBA ${startYear}–${endYear}`,
      detail: "Current agreement in force",
      tone: "",
    },
    {
      id: "now",
      when: `Season ${year}`,
      title: "Active term",
      detail: `${yearsLeft} year${yearsLeft === 1 ? "" : "s"} remaining · Pressure ${pressure}`,
      tone: yearsLeft <= 2 || pressure === "High" ? "warn" : "active",
    },
    {
      id: "talks",
      when: deadline,
      title: "Negotiation window",
      detail: yearsLeft <= 2 ? "Formal talks expected" : "Milestone: bargaining opens",
      tone: yearsLeft <= 2 ? "warn" : "",
    },
    {
      id: "expiry",
      when: String(endYear),
      title: "Agreement expiry",
      detail: `Ends after ${endYear - 1}–${endYear} season`,
      tone: yearsLeft <= 1 ? "warn" : "",
    },
  ];

  const relevantRules = asArray(cba.key_rules).filter((rule) => {
    const r = String(rule).toLowerCase();
    const pressureHot = num(cba.pressure, 0) >= 0.55;
    if (!pressureHot) return false;
    return r.includes("escrow") || r.includes("cap") || r.includes("share") || r.includes("tax");
  });

  if (relevantRules.length) {
    items.push({
      id: "rule-shift",
      when: "Rule shift",
      title: relevantRules[0],
      detail: "Elevated under current CBA pressure",
      tone: "active",
    });
  }

  return items;
}

function normalizeLeagueOps(raw) {
  const data = asObject(raw);
  if (!Object.keys(data).length) return EMPTY_OBJ;

  const teams = normalizeTeams(data.teams);
  const cap_forecast = synthesizeCapSeries(data);
  const confidence = deriveConfidence(data, cap_forecast);

  return {
    ...data,
    teams,
    salary_cap: num(data.salary_cap ?? data.cap?.salary_cap, 0),
    projected_salary_cap: num(data.projected_salary_cap ?? data.cap?.projected_salary_cap, 0),
    cap_change: num(data.cap_change ?? data.cap?.cap_change, 0),
    cap_change_type: data.cap_change_type || data.cap?.cap_change_type || "Flat Cap",
    cap_drivers: deriveCapDrivers({ ...data, teams }),
    owner_mood: deriveOwnerMood(data),
    league_status:
      data.league_health_label
      || asObject(data.relocation).league_stability
      || data.league_status
      || data.cap_change_type
      || "Stable",
    cap_forecast,
    cap_forecast_note:
      data.cap_forecast_note
      || "Projection sketch from current ceiling + next-year model (not full HRR accounting).",
    confidence,
    active_issues: deriveActiveIssues(data),
    cba_timeline: deriveCbaTimeline(data),
  };
}

function TeamLogo({ team, abbr, size = 22 }) {
  const url = resolveFranchiseTeamLogo(team, abbr);
  if (url) return <img src={url} alt="" style={{ width: size, height: size, objectFit: "contain" }} />;
  return <span className="abbr-fallback">{String(abbr || "?").slice(0, 3)}</span>;
}

function chartAxisLabel(points, i) {
  if (!points.length) return "";
  if (i === 0) return "Now";
  if (i === 1) return "Next";
  if (i === points.length - 1) return "Long";
  return String(points[i]?.year || "").slice(2);
}

function CapForecastChart({ series, note }) {
  const points = asArray(series);
  if (!points.length) return <div className="lo-empty">No forecast</div>;

  const padL = 52;
  const padR = 18;
  const padT = 18;
  const padB = 26;
  const w = 720;
  const h = 168;
  const fillMuted = "#93a8b8";
  const fillText = "#e9f7fb";
  const fillCyan = "#13d8e7";
  const fillGold = "#e9a83c";
  const caps = points.flatMap((p) => [p.low, p.high, p.cap]);
  const minY = Math.min(...caps) - 1;
  const maxY = Math.max(...caps) + 1;
  const span = Math.max(maxY - minY, 1);

  const xAt = (i) => padL + (i * (w - padL - padR)) / Math.max(points.length - 1, 1);
  const yAt = (v) => padT + ((maxY - v) / span) * (h - padT - padB);

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i)} ${yAt(p.cap)}`).join(" ");
  const band = [
    ...points.map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i)} ${yAt(p.high)}`),
    ...[...points].reverse().map((p, i) => {
      const idx = points.length - 1 - i;
      return `L ${xAt(idx)} ${yAt(p.low)}`;
    }),
    "Z",
  ].join(" ");

  const ticks = [minY + span * 0.2, minY + span * 0.5, minY + span * 0.8];

  return (
    <div className="lo-chart-wrap">
      <svg
        className="lo-chart"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label="Salary cap forecast"
        style={{ color: fillMuted }}
      >
        <path d={band} className="lo-chart-band" />
        <path d={line} className="lo-chart-line" fill="none" />
        {ticks.map((t) => (
          <g key={`tick-${t}`}>
            <line x1={padL} x2={w - padR} y1={yAt(t)} y2={yAt(t)} className="lo-chart-grid" />
            <text
              x={padL - 8}
              y={yAt(t) + 3}
              textAnchor="end"
              className="lo-chart-tick"
              fill={fillMuted}
              style={{ fill: fillMuted }}
            >
              {fmtCap(t)}
            </text>
          </g>
        ))}
        {points.map((p, i) => {
          const isNow = i === 0;
          const isNext = i === 1;
          const valueFill = isNow ? fillCyan : isNext ? fillGold : fillText;
          return (
            <g key={p.year || i}>
              <circle
                cx={xAt(i)}
                cy={yAt(p.cap)}
                r={isNow || isNext ? 4 : 3}
                className={`lo-chart-dot${isNow ? " is-now" : ""}`}
                fill={isNow ? fillCyan : fillGold}
              />
              <text
                x={xAt(i)}
                y={h - 8}
                textAnchor="middle"
                className="lo-chart-xlabel"
                fill={fillMuted}
                style={{ fill: fillMuted }}
              >
                {chartAxisLabel(points, i)}
              </text>
              {(isNow || isNext || i === points.length - 1) ? (
                <text
                  x={xAt(i)}
                  y={Math.max(12, yAt(p.cap) - 10)}
                  textAnchor="middle"
                  className={`lo-chart-value${isNow ? " is-now" : isNext ? " is-next" : ""}`}
                  fill={valueFill}
                  style={{ fill: valueFill }}
                >
                  {fmtCap(p.cap)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      {note ? <p className="lo-forecast-note">{note}</p> : null}
      <div className="lo-chart-legend" aria-hidden="true">
        <span className="now">Current</span>
        <span className="next">Next season</span>
        <span className="long">Long-term estimate</span>
      </div>
    </div>
  );
}

function stateClass(actionState) {
  if (actionState === "Vote Required") return "lo-state-vote";
  const s = String(actionState || "").toLowerCase();
  if (s === "monitoring") return "lo-state-monitoring";
  if (s === "negotiating") return "lo-state-negotiating";
  if (s === "stable") return "lo-state-stable";
  return "";
}

function unusualMoods(mood) {
  const m = asObject(mood);
  return [
    ["Owners", m.owners],
    ["Players", m.players],
    ["Fans", m.fans],
    ["Media", m.media],
  ].filter(([, value]) => {
    const s = String(value || "").toLowerCase();
    return s && !CALM_MOODS.has(s);
  });
}

function fanRatingLabel(row) {
  const r = asObject(row);
  const raw = r.fan_rating ?? r.fan_sentiment;
  if (raw === undefined || raw === null || raw === "") return "—";
  const rating = Math.round(num(raw, 0));
  const rank = num(r.fan_rank, 0);
  const of = num(r.fan_rank_of, 0);
  return rank > 0 ? `${rating} · #${rank}${of ? ` of ${of}` : ""}` : String(rating);
}

function YourImpactStrip({ user }) {
  const u = asObject(user);
  const risk = String(u.boycott_risk || u.relocation_risk_label || "Low");
  const riskLower = risk.toLowerCase();
  const riskTone = riskLower.includes("high") || riskLower.includes("critical") ? "warn" : riskLower.includes("med") ? "" : "ok";
  const influence =
    num(u.cap_contribution_m, 0) !== 0
      ? `${num(u.cap_contribution_m, 0) >= 0 ? "+" : ""}${num(u.cap_contribution_m, 0).toFixed(1)}M`
      : u.market_tier || u.market_health || "—";

  return (
    <div className="lo-impact-strip" title="Your Impact">
      <TeamLogo team={u} abbr={u.abbreviation || u.name} size={22} />
      <div className="lo-impact-pill">
        <span>Revenue</span>
        <b>{fmtMoneyM(u.revenue)}</b>
      </div>
      <div className="lo-impact-pill">
        <span>Influence</span>
        <b>{influence}</b>
      </div>
      <div className="lo-impact-pill" title="Fan rating (0-100) and league rank">
        <span>Fan Rating</span>
        <b>{fanRatingLabel(u)}</b>
      </div>
      <div className="lo-impact-pill">
        <span>Warning</span>
        <b className={riskTone}>{risk}</b>
      </div>
    </div>
  );
}

function TeamDossier({ team }) {
  const t = asObject(team);
  if (!Object.keys(t).length) {
    return (
      <div className="lo-inspector lo-dossier lo-empty-panel">
        <div className="lo-empty-panel-frame" aria-hidden="true" />
        <span className="lo-inspector-kicker">Market Intelligence</span>
        <h4>Select a franchise</h4>
        <p className="lo-inspector-note">
          Choose a club from the board to open revenue, market size, and relocation pressure.
        </p>
      </div>
    );
  }

  const yoy = num(t.revenue_yoy_delta, 0);
  const yoyDir = t.revenue_yoy_direction || "flat";
  const prior = Math.max(0, num(t.revenue, 0) - yoy);
  const attendance = num(t.attendance_rate, 0);
  const fan = num(t.fan_sentiment, 0);
  const profit = num(t.profit, 0);
  const tags = asArray(t.reason_tags);
  const ownership =
    t.owner_mood
    || t.ownership
    || (tags.find((tag) => /owner|boycott|patience/i.test(String(tag))) || null)
    || (String(t.market_pressure || "") === "Owner" ? "Owner pressure" : null)
    || (fan < 42 ? "Fan unrest" : fan >= 68 ? "Hot gate" : "Steady");
  const arena = t.reloc_reason || t.market_pressure || t.pressure || "Stable";

  return (
    <div className="lo-inspector lo-dossier lo-enter-soft" key={t.id || t.abbreviation}>
      <div className="lo-dossier-head">
        <TeamLogo team={t} abbr={t.abbreviation || t.name} size={28} />
        <div>
          <span className="lo-inspector-kicker">Market Dossier</span>
          <h4>
            {t.abbreviation || t.name}
            <em>{t.market_tier || "Market"}</em>
          </h4>
        </div>
      </div>
      <div className="lo-dossier-hero">
        <span>Revenue</span>
        <b>{fmtMoneyM(t.revenue)}</b>
        <em className={`lo-trend ${yoyDir === "up" || yoy > 0 ? "up" : yoyDir === "down" || yoy < 0 ? "down" : "flat"}`}>
          {yoyDir === "flat" && Math.abs(yoy) < 1
            ? "Flat YoY"
            : `${yoy >= 0 ? "+" : ""}${Math.round(yoy)}M YoY`}
        </em>
      </div>
      <div className="lo-inspector-rows">
        <div className="lo-inspector-row">
          <span>Prior revenue</span>
          <b className="gold">{fmtMoneyM(prior)}</b>
        </div>
        <div className="lo-inspector-row">
          <span>Profit</span>
          <b className={`lo-trend ${profit > 0 ? "up" : profit < 0 ? "down" : "flat"}`}>
            {profit >= 0 ? "+" : ""}{Math.round(profit)}M
          </b>
        </div>
        <div className="lo-inspector-row">
          <span>Attendance</span>
          <b>{attendance > 0 ? `${Math.round(attendance * 100)}%` : "—"}</b>
        </div>
        <div className="lo-inspector-row"><span>Fan rating</span><b>{fanRatingLabel(t)}</b></div>
        <div className="lo-inspector-row"><span>Arena pressure</span><b>{arena}</b></div>
        <div className="lo-inspector-row"><span>Ownership</span><b>{ownership}</b></div>
        <div className="lo-inspector-row">
          <span>Relocation risk</span>
          <b className={riskClass(t.relocation_risk_label)}>
            {t.relocation_risk_label} · {Math.round(num(t.relocation_risk, 0) * 100)}%
          </b>
        </div>
      </div>
      {tags.length ? (
        <p className="lo-inspector-note">{clipWords(tags.slice(0, 2).join(" · "), 15)}</p>
      ) : null}
      {t.event ? (
        <p className="lo-inspector-note">{clipWords(`${t.event.label} may shift local revenue outlook.`, 15)}</p>
      ) : null}
      {t.threatened ? (
        <p className="lo-inspector-note">Relocation pressure elevated. Watch gate and arena.</p>
      ) : null}
    </div>
  );
}

function IssueInspector({ issue }) {
  const row = asObject(issue);
  if (!Object.keys(row).length) {
    return (
      <div className="lo-inspector lo-empty-panel">
        <div className="lo-empty-panel-frame" aria-hidden="true" />
        <span className="lo-inspector-kicker">Policy Brief</span>
        <h4>Select a negotiation issue</h4>
        <p className="lo-inspector-note">
          Open an issue to review status, deadline, likelihood, and owner vs player positions.
        </p>
      </div>
    );
  }

  const owners = num(row.owner_support, 0);
  const players = num(row.player_support, 0);
  const hasPair = owners > 0 || players > 0;
  const gap = Math.abs(owners - players);
  const leader =
    !hasPair ? "—" : owners === players ? "Even" : owners > players ? "Owners" : "Players";

  return (
    <div className="lo-inspector lo-enter-soft" key={row.id || row.name}>
      <span className="lo-inspector-mark">Policy Brief · Display Only</span>
      <span className="lo-inspector-kicker">{row.source || "Issue"} · {row.action_state}</span>
      <h4>{row.name}</h4>
      <div className="lo-inspector-rows">
        <div className="lo-inspector-row"><span>Status</span><b>{row.status || row.action_state}</b></div>
        <div className="lo-inspector-row"><span>Deadline</span><b>{issueDeadline(row)}</b></div>
        <div className="lo-inspector-row"><span>Likelihood</span><b>{num(row.likelihood, 0)}%</b></div>
        <div className="lo-inspector-row"><span>Current lead</span><b>{leader}{hasPair ? ` · ${gap}pt gap` : ""}</b></div>
      </div>
      {hasPair ? (
        <div className="lo-support-bars">
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Owners</em><b>{owners}%</b></div>
            <div className="lo-support-track owners"><i style={{ width: `${Math.min(100, owners)}%` }} /></div>
          </div>
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Players</em><b>{players}%</b></div>
            <div className="lo-support-track"><i style={{ width: `${Math.min(100, players)}%` }} /></div>
          </div>
        </div>
      ) : (
        <div className="lo-support-bars">
          <div className="lo-support-row">
            <div className="lo-support-top"><em style={{ fontStyle: "normal" }}>Support</em><b>{num(row.support_pct, 0)}%</b></div>
            <div className="lo-support-track"><i style={{ width: `${Math.min(100, num(row.support_pct, 0))}%` }} /></div>
          </div>
        </div>
      )}
      <p className="lo-inspector-note">{issueConsequence(row)}</p>
    </div>
  );
}

function OverviewWorkspace({ data, onOpenIssue }) {
  const confidence = asObject(data.confidence);
  const capChange = num(data.cap_change, 0);
  const drivers = asArray(data.cap_drivers);
  const plusCount = drivers.filter((d) => d.sign === "+").length;
  const escrowPct = Math.min(100, Math.round(num(data.escrow_progress, 0) * 100));
  const revenueLabel =
    data.league_revenue_b != null
      ? `$${num(data.league_revenue_b, 0).toFixed(2)}B`
      : fmtMoneyM(data.league_revenue);
  const moods = unusualMoods(data.owner_mood);
  const topIssues = asArray(data.active_issues).slice(0, 3);
  const ownerMood = asObject(data.owner_mood);

  return (
    <div className="lo-ov lo-enter">
      {/* One hero: where the cap is going. Everything else is supporting fact. */}
      <section className="lo-hero" aria-label="Salary cap outlook">
        <div className="lo-hero__main">
          <span className="lo-hero__label">Cap next season</span>
          <strong>{fmtCap(data.projected_salary_cap)}</strong>
          <span className={`lo-hero__delta ${capChange < 0 ? "neg" : "pos"}`}>
            {capChange >= 0 ? "+" : "−"}{fmtCap(Math.abs(capChange))} · {data.cap_change_type || "Flat cap"}
          </span>
        </div>
        <dl className="lo-hero__facts">
          <div><dt>Cap now</dt><dd>{fmtCap(data.salary_cap)}</dd></div>
          <div><dt>League revenue</dt><dd>{revenueLabel}</dd></div>
          <div><dt>Escrow</dt><dd>{escrowPct}%</dd></div>
          <div><dt>Cap drivers lifting</dt><dd>{plusCount} of {Math.max(drivers.length, 1)}</dd></div>
          <div>
            <dt>Mood</dt>
            <dd>
              {ownerMood.players ? <>Players <b>{ownerMood.players}</b></> : null}
              {ownerMood.players && ownerMood.fans ? " · " : null}
              {ownerMood.fans ? <>Fans <b className="pos">{ownerMood.fans}</b></> : null}
              {!ownerMood.players && !ownerMood.fans && moods.length === 0 ? "Stable" : null}
            </dd>
          </div>
        </dl>
      </section>

      <div className="lo-ov-grid">
      <div className="lo-forecast">
        <div className="lo-forecast-head">
          <h2>Cap forecast</h2>
          <div className="lo-confidence-text">{confidence.rangeLabel || "—"}</div>
        </div>
        <CapForecastChart
          series={data.cap_forecast}
          note={data.cap_forecast_note || "Projection sketch — current + next-year model, not full HRR accounting."}
        />
      </div>

      <section className="lo-suspensions" aria-label="League suspension list">
        <h3 className="lo-priority-title">Suspension list</h3>
        {asArray(data.suspension_list).length ? (
          <table className="lo-susp-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th>Category</th>
                <th>Reason</th>
                <th>Status</th>
                <th>Games</th>
              </tr>
            </thead>
            <tbody>
              {asArray(data.suspension_list).map((row) => (
                <tr key={String(row.incident_id || row.player_id)}>
                  <td>{String(row.player_name || row.player_id || "—")}</td>
                  <td>{String(row.team_abbr || row.team_id || "—")}</td>
                  <td>{String(row.category || "—")}</td>
                  <td>{String(row.reason || "—")}</td>
                  <td>{String(row.status || "—")}</td>
                  <td>
                    {Number(row.games_remaining) > 0
                      ? `${Number(row.games_remaining)} left (${Number(row.games_initial) || "—"} total)`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="lo-muted">No active league or team discipline.</p>
        )}
      </section>

      <div className="lo-priority">
        <h3 className="lo-priority-title">Decisions on the table</h3>
        <div className="lo-priority-list">
          {topIssues.length ? (
            topIssues.map((issue) => {
              const id = issue.id || issue.name;
              const owners = num(issue.owner_support, 0);
              const players = num(issue.player_support, 0);
              const leader =
                owners === players
                  ? "Even"
                  : owners > players
                    ? "Owners lead"
                    : "Players lead";
              const consequence = issueConsequence(issue);
              const prefix = `${issue.name}: `;
              const detail = consequence.startsWith(prefix)
                ? consequence.slice(prefix.length)
                : consequence;
              return (
                <button
                  key={id}
                  type="button"
                  className="lo-priority-card"
                  onClick={() => onOpenIssue(issue)}
                >
                  <strong>{issue.name}</strong>
                  <div className="lo-priority-meta">
                    <span>{issueDeadline(issue)}</span>
                    <span>{leader}</span>
                  </div>
                  <p>{detail}</p>
                  <span className={`state ${stateClass(issue.action_state)}`}>{issue.action_state}</span>
                  {(owners > 0 || players > 0) ? (
                    <div className="lo-priority-balance" aria-hidden="true">
                      <i className="owners" style={{ width: `${Math.min(100, owners)}%` }} />
                      <i className="players" style={{ width: `${Math.min(100, players)}%` }} />
                    </div>
                  ) : null}
                </button>
              );
            })
          ) : (
            <div className="lo-empty">No priority decisions</div>
          )}
        </div>
      </div>
      </div>
    </div>
  );
}

function CbaWorkspace({ data, selectedIssueId, onSelectIssue }) {
  const issues = asArray(data.active_issues);
  const timeline = asArray(data.cba_timeline);
  const selected = issues.find((i) => (i.id || i.name) === selectedIssueId) || null;
  const cba = asObject(data.cba);
  const brief =
    cba.brief
    || "Intelligence display only — negotiations are pressure estimates and do not change in-sim rules.";

  return (
    <div className="lo-cba lo-enter">
      <div className="lo-cba-notice" role="note">
        <span className="lo-cba-notice-mark" aria-hidden="true" />
        <strong>Display only</strong>
        <span>{brief}</span>
      </div>

      <div className="lo-cba-left">
        <div className="lo-cba-panel lo-cba-timeline-panel">
          <div className="lo-cba-panel-head">
            <h3 className="lo-section-title">CBA Timeline</h3>
            <span className="lo-cba-pressure">
              Pressure {cba.pressure_level || "—"}
              {cba.years_remaining != null ? ` · ${cba.years_remaining}y left` : ""}
              {cba.lockout_risk_label ? ` · Lockout risk ${cba.lockout_risk_label}` : ""}
            </span>
          </div>
          <div className="lo-timeline-rail" role="list">
            {timeline.length ? (
              timeline.map((item, idx) => (
                <div
                  key={item.id}
                  role="listitem"
                  className={`lo-timeline-step ${item.tone || ""}`.trim()}
                >
                  <span className="lo-timeline-dot" aria-hidden="true" />
                  {idx < timeline.length - 1 ? <span className="lo-timeline-wire" aria-hidden="true" /> : null}
                  <span className="lo-timeline-when">{item.when}</span>
                  <b>{item.title}</b>
                  <em>{item.detail}</em>
                </div>
              ))
            ) : (
              <div className="lo-inspector-empty">No CBA milestones</div>
            )}
          </div>
        </div>

        <div className="lo-cba-panel">
          <h3 className="lo-section-title">Negotiations</h3>
          <div className="lo-issues-scroll">
            {issues.length ? (
              issues.map((issue) => {
                const id = issue.id || issue.name;
                const selectedCls = selectedIssueId === id ? " selected" : "";
                const owners = num(issue.owner_support, 0);
                const players = num(issue.player_support, 0);
                return (
                  <button
                    key={id}
                    type="button"
                    className={`lo-issue${selectedCls}`}
                    aria-selected={selectedIssueId === id}
                    onClick={() => onSelectIssue(issue)}
                  >
                    <span className="lo-issue-title">{issue.name}</span>
                    <span className="lo-issue-scale">{issue.scale_label || issueDeadline(issue)}</span>
                    <span className={`lo-issue-state ${stateClass(issue.action_state)}`}>{issue.action_state}</span>
                    {(owners > 0 || players > 0) ? (
                      <span className="lo-issue-balance" aria-label={`Owners ${owners}%, Players ${players}%`}>
                        <span className="owners" style={{ flexGrow: Math.max(owners, 1) }}>{owners}%</span>
                        <span className="players" style={{ flexGrow: Math.max(players, 1) }}>{players}%</span>
                      </span>
                    ) : null}
                  </button>
                );
              })
            ) : (
              <div className="lo-inspector-empty">No active issues</div>
            )}
          </div>
        </div>
      </div>

      <IssueInspector issue={selected} />
    </div>
  );
}

function filterTeams(teams, filterId) {
  const list = asArray(teams);
  if (filterId === "growing") {
    return list.filter((t) => {
      const yoy = num(t.revenue_yoy_delta, 0);
      const dir = t.revenue_yoy_direction || "";
      return dir === "up" || yoy > 0;
    });
  }
  if (filterId === "losing") {
    return list.filter((t) => num(t.profit, 0) < -2);
  }
  if (filterId === "reloc") {
    return list.filter((t) => num(t.relocation_risk, 0) >= 0.35 || t.threatened);
  }
  return list;
}

function MarketsWorkspace({
  teams,
  userTeamId,
  sortKey,
  sortDir,
  onSort,
  filterId,
  onFilter,
  selectedTeamId,
  onSelectTeam,
}) {
  const allTeams = asArray(teams);
  const filtered = filterTeams(teams, filterId);
  const selected = filtered.find((t) => t.id === selectedTeamId) || allTeams.find((t) => t.id === selectedTeamId) || null;
  const filterCounts = {
    all: allTeams.length,
    growing: filterTeams(allTeams, "growing").length,
    losing: filterTeams(allTeams, "losing").length,
    reloc: filterTeams(allTeams, "reloc").length,
  };

  const trendText = (t) => {
    const yoy = num(t.revenue_yoy_delta, 0);
    const dir = t.revenue_yoy_direction || "flat";
    if (dir === "flat" && Math.abs(yoy) < 1) return { text: "—", dir: "flat" };
    return {
      text: `${yoy >= 0 ? "+" : ""}${Math.round(yoy)}M`,
      dir: dir === "up" || yoy > 0 ? "up" : dir === "down" || yoy < 0 ? "down" : "flat",
    };
  };

  const sortLabel = (key, label) => (
    <button type="button" className={sortKey === key ? "sorted" : ""} onClick={() => onSort(key)}>
      {label}
      {sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
    </button>
  );

  return (
    <div className="lo-markets lo-enter">
      <div className="lo-markets-toolbar">
        <div className="lo-filters" role="tablist" aria-label="Market filters">
          {MARKET_FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              role="tab"
              aria-selected={filterId === f.id}
              className={filterId === f.id ? "active" : ""}
              onClick={() => onFilter(f.id)}
            >
              {f.label}
              <em>{filterCounts[f.id] ?? 0}</em>
            </button>
          ))}
        </div>
      </div>

      <div className="lo-markets-body">
        <div className="lo-table-wrap">
          <table className="lo-table">
            <thead>
              <tr>
                <th className="lo-col-team">Team</th>
                <th className="lo-col-rev">{sortLabel("revenue", "Revenue")}</th>
                <th className="lo-col-trend">{sortLabel("trend", "vs last yr")}</th>
                <th className="lo-col-profit">{sortLabel("profit", "Profit")}</th>
                <th className="lo-col-market">{sortLabel("market", "Market")}</th>
                <th className="lo-col-risk">{sortLabel("risk", "Risk")}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length ? (
                filtered.map((t) => {
                  const trend = trendText(t);
                  const isUser = String(t.id) === String(userTeamId);
                  const selectedCls = selectedTeamId === t.id;
                  return (
                    <tr
                      key={t.id}
                      tabIndex={0}
                      aria-selected={selectedCls}
                      className={[
                        t.threatened ? "threatened" : "",
                        selectedCls ? "selected" : "",
                        isUser ? "user" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      onClick={() => onSelectTeam(t)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          onSelectTeam(t);
                        }
                      }}
                    >
                      <td className="lo-col-team">
                        <div className="lo-team-cell">
                          <TeamLogo team={t} abbr={t.abbreviation} />
                          <strong>{t.abbreviation}</strong>
                          {t.event ? (
                            <span
                              className={`lo-event-mark ${t.event.kind === "risk" ? "risk" : t.event.kind === "rev" ? "rev" : ""}`}
                              title={t.event.label}
                            />
                          ) : null}
                        </div>
                      </td>
                      <td className="lo-col-rev">{fmtMoneyM(t.revenue)}</td>
                      <td className={`lo-col-trend lo-trend ${trend.dir}`}>{trend.text}</td>
                      <td className={`lo-col-profit lo-trend ${num(t.profit, 0) < 0 ? "down" : "up"}`}>
                        {num(t.profit, 0) < 0 ? "-" : "+"}{fmtMoneyM(Math.abs(num(t.profit, 0)))}
                      </td>
                      <td className="lo-col-market">{t.market_tier || "—"}</td>
                      <td className={`lo-col-risk ${riskClass(t.relocation_risk_label)}`}>
                        {t.relocation_risk_label}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6}>
                    <div className="lo-empty">No matching teams</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <TeamDossier team={selected} />
      </div>
    </div>
  );
}

function hasMeaningfulPressure(team) {
  const label = String(team.relocation_risk_label || "").toLowerCase();
  if (label.includes("low") && !team.threatened) return false;
  const reloc = num(team.relocation_risk, 0);
  const profit = num(team.profit, 0);
  const status = String(team.display_status || "");
  return reloc >= 0.35 || team.threatened || profit < -5 || status === "Loss" || status === "Risk";
}

function riskBucket(team) {
  const reloc = num(team.relocation_risk, 0);
  const label = String(team.relocation_risk_label || "").toLowerCase();
  if (team.threatened || reloc >= 0.55 || label.includes("high") || label.includes("critical")) {
    return "immediate";
  }
  if (reloc >= 0.42) return "monitor";
  return "stable";
}

function FranchiseRiskWorkspace({ teams, selectedTeamId, onSelectTeam, watchlist }) {
  let pressured = asArray(teams).filter(hasMeaningfulPressure);

  // Seed from backend relocation.watchlist when the hard filter would leave every column empty.
  if (!pressured.length) {
    const watchedIds = new Set(
      asArray(watchlist)
        .map((w) => String(w.id || w.abbreviation || ""))
        .filter(Boolean)
    );
    if (watchedIds.size) {
      pressured = asArray(teams).filter((t) => {
        const id = String(t.id || "");
        const abbr = String(t.abbreviation || "");
        return watchedIds.has(id) || watchedIds.has(abbr) || num(t.relocation_risk, 0) >= 0.35;
      });
    }
    if (!pressured.length) {
      pressured = asArray(teams)
        .slice()
        .sort((a, b) => num(b.relocation_risk, 0) - num(a.relocation_risk, 0))
        .slice(0, 5)
        .filter((t) => num(t.relocation_risk, 0) >= 0.28);
    }
  }

  const immediate = pressured.filter((t) => riskBucket(t) === "immediate");
  const monitor = pressured.filter((t) => riskBucket(t) === "monitor");
  const stable = pressured.filter((t) => riskBucket(t) === "stable");
  const selected =
    pressured.find((t) => t.id === selectedTeamId)
    || asArray(teams).find((t) => t.id === selectedTeamId)
    || null;

  const emptyLabel = "No elevated risk";

  const renderCards = (list) => (
    list.length ? (
      list.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`lo-risk-card${selectedTeamId === t.id ? " selected" : ""}`}
          onClick={() => onSelectTeam(t)}
        >
          <div className="lo-risk-card-top">
            <TeamLogo team={t} abbr={t.abbreviation} size={20} />
            <strong>{t.abbreviation}</strong>
          </div>
          <div className="lo-risk-card-meta">
            <span className={riskClass(t.relocation_risk_label)}>{t.relocation_risk_label}</span>
            <span>{t.reloc_reason || t.display_status}</span>
          </div>
          <div className="lo-risk-card-meta">
            <span style={{ color: "var(--lo-gold)" }}>{fmtMoneyM(t.revenue)}</span>
            <span>{Math.round(num(t.relocation_risk, 0) * 100)}%</span>
          </div>
        </button>
      ))
    ) : (
      <div className="lo-inspector-empty">{emptyLabel}</div>
    )
  );

  return (
    <div className={`lo-risk lo-enter${selected ? " has-detail" : ""}`}>
      <div className="lo-risk-col immediate">
        <h3>Immediate</h3>
        <div className="lo-risk-list">{renderCards(immediate)}</div>
      </div>
      <div className="lo-risk-col monitor">
        <h3>Monitor</h3>
        <div className="lo-risk-list">{renderCards(monitor)}</div>
      </div>
      <div className="lo-risk-col stable">
        <h3>Stable</h3>
        <div className="lo-risk-list">{renderCards(stable)}</div>
      </div>
      {selected ? <TeamDossier team={selected} /> : (
        <div className="lo-inspector lo-empty-panel">
          <div className="lo-empty-panel-frame" aria-hidden="true" />
          <span className="lo-inspector-kicker">Risk Detail</span>
          <h4>Select a club</h4>
          <p className="lo-inspector-note">
            Choose a franchise from the watch board to review revenue pressure and relocation risk.
          </p>
        </div>
      )}
    </div>
  );
}

function sortTeams(teams, sortKey, sortDir) {
  const list = [...asArray(teams)];
  const dir = sortDir === "asc" ? 1 : -1;
  list.sort((a, b) => {
    if (sortKey === "trend") {
      return (num(a.revenue_yoy_delta, 0) - num(b.revenue_yoy_delta, 0)) * dir;
    }
    if (sortKey === "profit") {
      return (num(a.profit, 0) - num(b.profit, 0)) * dir;
    }
    if (sortKey === "market") {
      const order = { Large: 3, Mid: 2, Small: 1 };
      return ((order[a.market_tier] || 0) - (order[b.market_tier] || 0)) * dir;
    }
    if (sortKey === "risk") {
      return (num(a.relocation_risk, 0) - num(b.relocation_risk, 0)) * dir;
    }
    return (num(a.revenue, 0) - num(b.revenue, 0)) * dir;
  });
  return list;
}

export default function LeagueOperations() {
  const { franchiseState, setScreen } = useGameUI();
  const [remote, setRemote] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [workspace, setWorkspace] = useState("overview");
  const [sortKey, setSortKey] = useState("revenue");
  const [sortDir, setSortDir] = useState("desc");
  const [marketFilter, setMarketFilter] = useState("all");
  const [selectedIssueId, setSelectedIssueId] = useState(null);
  const [selectedTeamId, setSelectedTeamId] = useState(null);

  const cached = useMemo(() => asObject(franchiseState?.league_operations), [franchiseState]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(false);

    getLeagueOperations()
      .then((res) => {
        if (!alive) return;
        setRemote(asObject(res?.league_operations));
        setLoading(false);
      })
      .catch(() => {
        if (!alive) return;
        setRemote(null);
        setError(true);
        setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, []);

  const data = useMemo(() => {
    const raw =
      (remote && Object.keys(remote).length)
        ? remote
        : (cached && Object.keys(cached).length)
          ? cached
          : EMPTY_OBJ;
    return normalizeLeagueOps(raw);
  }, [remote, cached]);

  const hasData = asArray(data.teams).length > 0;
  const userTeamId = franchiseState?.user_team_id || franchiseState?.team?.id;

  const sortedTeams = useMemo(
    () => sortTeams(data.teams, sortKey, sortDir),
    [data.teams, sortKey, sortDir]
  );

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") setScreen(SCREENS.HUB);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setScreen]);

  useEffect(() => {
    if (workspace !== "cba") return;
    const issues = asArray(data.active_issues);
    if (!issues.length) return;
    const stillValid = issues.some((i) => (i.id || i.name) === selectedIssueId);
    if (!stillValid) {
      const first = issues[0];
      setSelectedIssueId(first.id || first.name);
    }
  }, [workspace, data.active_issues, selectedIssueId]);

  const handleSort = (key) => {
    if (!SORT_KEYS.includes(key)) return;
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir(key === "market" ? "asc" : "desc");
    }
  };

  const leagueStatus = data.league_status || "Stable";

  return (
    <div className="game-screen hub-screen calendar-screen cal-fr lo-screen">
      <InlineStyles />

      <header className="lo-header">
        <div className="lo-header-left">
          <button type="button" className="lo-back" onClick={() => setScreen(SCREENS.HUB)} aria-label="Back to hub">
            ←
          </button>
          <div>
            <h1>League Operations</h1>
            <div className="lo-header-meta">
              <span className="season">{data.season || "—"}</span>
              <span aria-hidden="true">·</span>
              <span className="status">{leagueStatus}</span>
            </div>
          </div>
        </div>
        <div className="lo-header-right">
          <YourImpactStrip user={data.user_team} />
        </div>
      </header>

      <nav className="lo-tabs" aria-label="League Operations workspaces">
        {WORKSPACES.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={workspace === tab.id ? "active" : ""}
            aria-current={workspace === tab.id ? "page" : undefined}
            onClick={() => setWorkspace(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {!hasData && !loading ? (
        <div className="lo-empty">{error ? "No league data" : "No league data"}</div>
      ) : (
        <main className="lo-workspace" aria-live="polite">
          {workspace === "overview" ? (
            <OverviewWorkspace
              data={data}
              onOpenIssue={(issue) => {
                setSelectedIssueId(issue.id || issue.name);
                setWorkspace("cba");
              }}
            />
          ) : null}
          {workspace === "cba" ? (
            <CbaWorkspace
              data={data}
              selectedIssueId={selectedIssueId}
              onSelectIssue={(issue) => setSelectedIssueId(issue.id || issue.name)}
            />
          ) : null}
          {workspace === "markets" ? (
            <MarketsWorkspace
              teams={sortedTeams}
              userTeamId={userTeamId}
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
              filterId={marketFilter}
              onFilter={setMarketFilter}
              selectedTeamId={selectedTeamId}
              onSelectTeam={(t) => setSelectedTeamId(t.id)}
            />
          ) : null}
          {workspace === "board" ? <BoardWorkspace userTeamId={userTeamId} /> : null}
          {workspace === "risk" ? (
            <FranchiseRiskWorkspace
              teams={sortedTeams}
              watchlist={asArray(asObject(data.relocation).watchlist)}
              selectedTeamId={selectedTeamId}
              onSelectTeam={(t) => setSelectedTeamId(t.id)}
            />
          ) : null}
        </main>
      )}
    </div>
  );
}
