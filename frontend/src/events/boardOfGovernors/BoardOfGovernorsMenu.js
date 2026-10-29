import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGovernance, lobbyGovernance, setExpansionProtection, voteGovernance } from "../../services/franchiseService";
import { getTeamLogoSrc, toLogoUrl } from "../../utils/teamLogos";
import { pickFranchiseData } from "../shared/eventHelpers";

/*
 * Board of Governors — annual offseason meeting.
 * Visual language matches the Entry Draft floor (same tokens, type and buttons).
 */

const PREFIX = "bog";
const REVEAL_MS = 55;



function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function money(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  return `$${n.toFixed(1)}M`;
}

function TeamTile({ ballot, shown }) {
  const src = toLogoUrl(getTeamLogoSrc({ team_abbrev: ballot.abbr, abbr: ballot.abbr, name: ballot.name }));
  const cls = [
    `${PREFIX}-tile`,
    shown ? "shown" : "",
    shown ? ballot.vote : "",
    ballot.is_user ? "user" : "",
  ].filter(Boolean).join(" ");
  return (
    <div className={cls} title={`${ballot.name}: ${ballot.vote}${ballot.reason ? ` — ${ballot.reason}` : ""}`}>
      {src ? <img src={src} alt="" loading="lazy" /> : <span className="fb">{(ballot.abbr || "").slice(0, 3)}</span>}
      <span>{ballot.abbr}</span>
      {ballot.lobbied ? <span className="lob" aria-label="lobbied">●</span> : null}
    </div>
  );
}

function groupReasons(ballots) {
  // "Why governors voted this way": one line per (vote, reason) with the clubs behind it.
  const groups = new Map();
  ballots.filter((b) => !b.is_user && b.vote).forEach((b) => {
    const key = `${b.vote}|${b.reason || "No stated reason"}`;
    if (!groups.has(key)) groups.set(key, { vote: b.vote, reason: b.reason || "No stated reason", teams: [] });
    groups.get(key).teams.push(b.abbr);
  });
  return [...groups.values()].sort((a, b) => b.teams.length - a.teams.length);
}

function ProposalCard({ proposal, tokens, busy, onVote, onLobby, reveal, onNext, isLast }) {
  const [showAllWhy, setShowAllWhy] = useState(false);
  const decided = proposal.status !== "pending";
  const forecast = proposal.forecast || {};
  const needed = Number(proposal.votes_needed || 0);
  const ballots = asArray(proposal.ballots);
  const nTeams = ballots.length || Number(proposal._n_teams || 32);
  const expected = Number(forecast.expected_yes || 0);
  const shownCount = decided ? Math.min(reveal, ballots.length) : 0;
  const shown = ballots.slice(0, shownCount);
  const yes = shown.filter((b) => b.vote === "yes").length;
  const no = shown.filter((b) => b.vote === "no").length;
  const abst = shown.filter((b) => b.vote !== "yes" && b.vote !== "no").length;
  const revealDone = decided && shownCount >= ballots.length;
  const tally = proposal.tally || {};
  const lobbied = proposal.lobbied;
  const why = revealDone ? groupReasons(ballots) : [];
  const whyShown = showAllWhy ? why : why.slice(0, 4);
  const pct = (n) => `${Math.min(100, (n / Math.max(1, nTeams)) * 100)}%`;

  return (
    <section className="bog2-card" aria-live="polite">
      <header className="bog2-head">
        <p className="bog2-cat">
          {proposal.category_label} · {proposal.threshold_label} vote ({needed} of {nTeams})
        </p>
        <h2>{proposal.title}</h2>
        <p className="bog2-summary">{proposal.summary}</p>
        {lobbied ? (
          <p className="bog2-note">
            You lobbied {lobbied.side === "yes" ? "for" : "against"} it: {asArray(lobbied.teams).join(", ")}.
          </p>
        ) : null}
      </header>

      {asArray(proposal.effects).length ? (
        <dl className="bog2-impact">
          {asArray(proposal.effects).map((e) => (
            <div key={`${e.key}-${e.label}`} className={e.tone ? `is-${e.tone}` : ""}>
              <dt>{e.label}</dt>
              <dd>{e.text}{e.note ? <small>{e.note}</small> : null}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {proposal.pros_cons ? (
        <div className="bog2-args">
          <div>
            <h4>For</h4>
            {asArray(proposal.pros_cons.pros).length ? asArray(proposal.pros_cons.pros).map((t) => <p key={t}>{t}</p>) : <p className="muted">No clear upside for most clubs.</p>}
          </div>
          <div>
            <h4>Against</h4>
            {asArray(proposal.pros_cons.cons).length ? asArray(proposal.pros_cons.cons).map((t) => <p key={t}>{t}</p>) : <p className="muted">No obvious downside.</p>}
          </div>
          {asArray(proposal.pros_cons.mixed).length ? (
            <div>
              <h4>Trade-offs</h4>
              {asArray(proposal.pros_cons.mixed).map((t) => <p key={t}>{t}</p>)}
            </div>
          ) : null}
        </div>
      ) : null}

      <section className="bog2-chamber" aria-label="Vote">
        <div className="bog2-bar" aria-hidden="true">
          {decided ? (
            <>
              <i className="yes" style={{ width: pct(yes) }} />
              <i className="no" style={{ width: pct(no), left: `calc(100% - ${pct(no)})` }} />
            </>
          ) : (
            <i className="forecast" style={{ width: pct(expected) }} />
          )}
          <b style={{ left: pct(needed) }} title={`${needed} needed`} />
        </div>
        <p className="bog2-count">
          {decided ? (
            <>
              <strong className="yes">{yes} in favour</strong> · <strong className="no">{no} opposed</strong>
              {abst ? ` · ${abst} abstained` : ""} · {needed} needed · {nTeams} governors
            </>
          ) : (
            <>
              Board read: <strong>{forecast.label || "—"}</strong> · about {expected.toFixed(0)} in favour, {needed} needed
              {proposal.room_mood ? ` · ${proposal.room_mood}` : ""}
            </>
          )}
        </p>

        {decided ? (
          <div className="bog2-grid">
            {ballots.map((b, i) => (
              <TeamTile key={b.team_id || i} ballot={b} shown={i < shownCount} />
            ))}
          </div>
        ) : (
          <div className="bog2-actions">
            <button type="button" className="nhlcal-advance-button" disabled={busy} onClick={() => onVote("yes")}>
              Vote in favour
            </button>
            <button type="button" className="nhlcal-advance-button nhlcal-advance-button-secondary" disabled={busy} onClick={() => onVote("no")}>
              Vote against
            </button>
            <button type="button" className="bog2-link" disabled={busy} onClick={() => onVote("abstain")}>
              Abstain
            </button>
            <span className="bog2-spacer" />
            <button type="button" className="bog2-link" disabled={busy || tokens <= 0 || Boolean(lobbied)} onClick={() => onLobby("yes")}>
              Lobby for it
            </button>
            <button type="button" className="bog2-link" disabled={busy || tokens <= 0 || Boolean(lobbied)} onClick={() => onLobby("no")}>
              Lobby against
            </button>
            <p className="bog2-note">
              A lobbying call reaches the four most persuadable governors whose clubs look like yours ({tokens} left).
              Keys: A = in favour, D = against.
            </p>
          </div>
        )}
      </section>

      {revealDone ? (
        <>
          <footer className={`bog2-outcome ${proposal.status}`}>
            <div>
              <h3>{proposal.status === "passed" ? "Passed" : "Defeated"}</h3>
              <p>
                {tally.yes}–{tally.no}
                {tally.abstain ? `, ${tally.abstain} abstained` : ""} · {needed} needed · you voted {proposal.user_vote === "yes" ? "in favour" : proposal.user_vote === "no" ? "against" : "to abstain"}
              </p>
            </div>
            <button type="button" className="nhlcal-advance-button" onClick={onNext}>
              {isLast ? "Close the meeting" : "Next proposal →"}
            </button>
          </footer>
          {why.length ? (
            <section className="bog2-why">
              <h4>Why governors voted this way</h4>
              <ul>
                {whyShown.map((g) => (
                  <li key={`${g.vote}-${g.reason}`} className={g.vote}>
                    <span className="v">{g.vote === "yes" ? "For" : g.vote === "no" ? "Against" : "Abstained"}</span>
                    <span className="r">{g.reason}</span>
                    <span className="t">{g.teams.join(", ")}</span>
                  </li>
                ))}
              </ul>
              {why.length > 4 ? (
                <button type="button" className="bog2-link" onClick={() => setShowAllWhy((v) => !v)}>
                  {showAllWhy ? "Show fewer" : `Show all ${why.length} reasons`}
                </button>
              ) : null}
            </section>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

function tierClass(tier) {
  return `${PREFIX}-tier-${String(tier || "steady").toLowerCase()}`;
}

function FanTicker({ rows }) {
  const list = asArray(rows);
  if (!list.length) return null;
  const items = list.map((r) => {
    const d = Number(r.delta || 0);
    return (
      <span key={r.team_id} className={`${PREFIX}-fanticker-item ${r.is_user ? "is-user" : ""}`}>
        <span className="ab">{r.abbr}</span>
        <span className={`v ${tierClass(r.tier)}`}>{Math.round(Number(r.value || 0))}</span>
        <span className={`d ${d > 0.4 ? "up" : d < -0.4 ? "down" : "flat"}`}>
          {d > 0.4 ? "▲" : d < -0.4 ? "▼" : "•"}
          {Math.abs(d) >= 0.4 ? Math.abs(d).toFixed(1) : ""}
        </span>
        <span className="t">{r.tier}</span>
      </span>
    );
  });
  return (
    <div className={`${PREFIX}-fanticker`} aria-label="Fan confidence around the league">
      <div className={`${PREFIX}-fanticker-label`}>Fan confidence</div>
      <div className={`${PREFIX}-fanticker-viewport`}>
        {/* Two copies so the strip loops without a gap. */}
        <div className={`${PREFIX}-fanticker-track`}>
          {items}
          {items.map((el) => React.cloneElement(el, { key: `${el.key}-b`, "aria-hidden": true }))}
        </div>
      </div>
    </div>
  );
}

function FanPanel({ report }) {
  if (!report || report.value == null) return null;
  const v = Number(report.value || 0);
  const d = Number(report.delta || 0);
  const base = Number(report.baseline || 0);
  return (
    <>
      <div className={`${PREFIX}-panel-head`}>
        <h3>Your fans</h3>
      </div>
      <div className={`${PREFIX}-fan`}>
        <div className={`${PREFIX}-fan-head`}>
          <span className={`${PREFIX}-fan-num ${tierClass(report.tier)}`}>{Math.round(v)}</span>
          <span className={`${PREFIX}-fan-tier ${tierClass(report.tier)}`}>{report.tier}</span>
          <span className={`${PREFIX}-fan-delta`}>
            {Math.abs(d) >= 0.4 ? `${d > 0 ? "▲" : "▼"} ${Math.abs(d).toFixed(1)} this week` : "steady this week"}
          </span>
        </div>
        <div className={`${PREFIX}-fan-bar`} title={`Record supports about ${Math.round(base)}`}>
          <i style={{ width: `${Math.max(2, Math.min(100, v))}%` }} />
          {base ? <b style={{ left: `${Math.max(0, Math.min(99, base))}%` }} /> : null}
        </div>
        <div className={`${PREFIX}-fan-sub`}>What it's worth right now</div>
        <div className={`${PREFIX}-fan-list`}>
          {asArray(report.effects).map((e) => (
            <div key={e.label} className="row">
              <span>{e.label}</span>
              <strong className={e.tone}>{e.text}</strong>
            </div>
          ))}
        </div>
        <div className={`${PREFIX}-fan-sub`}>Why fans feel this way</div>
        <ul className={`${PREFIX}-fan-why`}>
          {asArray(report.reasons).map((r, i) => (
            <li key={i} className={r.tone}>
              {r.text}
            </li>
          ))}
        </ul>
        <div className={`${PREFIX}-fan-sub`}>How to raise it</div>
        <ul className={`${PREFIX}-fan-why`}>
          {asArray(report.how_to_raise).map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </div>
    </>
  );
}

function SidePanel({ gov }) {
  const impact = gov.user_impact || {};
  const mods = asArray(gov.modifiers);
  const hist = asArray(gov.history).slice(0, 8);
  const change = Number(impact.value_change_pct || 0);
  return (
    <aside className={`${PREFIX}-panel ${PREFIX}-side`}>
      <FanPanel report={gov.fan_report} />
      <div className={`${PREFIX}-panel-head`}>
        <h3>Your club</h3>
      </div>
      <div className={`${PREFIX}-kv`}>
        <span>Annual profit</span>
        <strong className={Number(impact.annual_profit_m) >= 0 ? "up" : "down"}>
          {impact.annual_profit_m != null ? money(impact.annual_profit_m) : "—"}
        </strong>
        <span>Annual revenue</span>
        <strong>{impact.annual_revenue_m != null ? money(impact.annual_revenue_m) : "—"}</strong>
        <span>Franchise value</span>
        <strong>{impact.franchise_value_b != null ? `$${Number(impact.franchise_value_b).toFixed(2)}B` : "—"}</strong>
        <span>Last change</span>
        <strong className={change > 0 ? "up" : change < 0 ? "down" : ""}>{`${change >= 0 ? "+" : ""}${change.toFixed(1)}%`}</strong>
        <span>Scouting budget</span>
        <strong>{impact.scouting_budget ? money(impact.scouting_budget) : "—"}</strong>
        <span>Signing bonuses</span>
        <strong className={impact.bonus_eligible ? "up" : "down"}>
          {impact.bonus_eligible ? `Up to ${Math.round(Number(impact.bonus_max_pct || 0) * 100)}%` : `Locked (< $${Number(impact.bonus_floor_m || 155).toFixed(0)}M)`}
        </strong>
        <span>Star spike next season</span>
        <strong>{impact.star_spike_next_m ? `+$${Number(impact.star_spike_next_m).toFixed(1)}M` : "—"}</strong>
      </div>
      <div className={`${PREFIX}-panel-head`}>
        <h3>Rules in force ({asArray(gov.rulebook).length})</h3>
      </div>
      <div className={`${PREFIX}-mods`}>
        {mods.length ? (
          mods.map((m) => (
            <div key={m.key} className={`${PREFIX}-mod`}>
              <span>{m.label}</span>
              <strong>{m.text}</strong>
            </div>
          ))
        ) : (
          <span className={`${PREFIX}-muted`}>No Board rules in force yet.</span>
        )}
      </div>
      {hist.length ? (
        <>
          <div className={`${PREFIX}-panel-head`}>
            <h3>Recent votes</h3>
          </div>
          <div className={`${PREFIX}-hist`}>
            {hist.map((h) => (
              <div key={h.proposal_id} className={`${PREFIX}-hrow`}>
                <span>
                  {h.season} · {h.title}
                </span>
                <em className={h.status}>
                  {h.status === "passed" ? "PASS" : "FAIL"} {h.yes}–{h.no}
                </em>
              </div>
            ))}
          </div>
        </>
      ) : null}
    </aside>
  );
}

const PROTECT_LABEL = { F: "Forwards", D: "Defense", G: "Goalies", S: "Skaters" };
const SCHEME_LIMITS = { "7-3-1": { F: 7, D: 3, G: 1 }, "8-1": { S: 8, G: 1 } };

function ExpansionProtectionPanel({ draft, onSaved }) {
  const players = asArray(draft?.players);
  const [scheme, setScheme] = useState(draft?.scheme || "7-3-1");
  const limits = SCHEME_LIMITS[scheme] || SCHEME_LIMITS["7-3-1"];
  const slotOf = (pos) => (pos === "G" || scheme === "7-3-1" ? pos : "S");
  const [picked, setPicked] = useState(() =>
    new Set(players.filter((p) => p.protected && !p.exempt).map((p) => p.player_id))
  );
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const counts = Object.fromEntries(Object.keys(limits).map((k) => [k, 0]));
  players.forEach((p) => {
    if (picked.has(p.player_id) && !p.exempt) counts[slotOf(p.pos)] += 1;
  });
  const exposure = { F: 0, D: 0, G: 0 };
  players.forEach((p) => {
    if (!picked.has(p.player_id) && p.exposable) exposure[p.pos] += 1;
  });
  const exposeNeed = draft?.exposure?.need || { F: 2, D: 1, G: 1 };
  const toggle = (p) => {
    if (p.exempt) return;
    setMsg("");
    const slot = slotOf(p.pos);
    if (picked.has(p.player_id)) {
      if (p.nmc) {
        setMsg("No-move clauses must be protected.");
        return;
      }
      setPicked((prev) => {
        const next = new Set(prev);
        next.delete(p.player_id);
        return next;
      });
    } else if (counts[slot] < Number(limits[slot] || 0)) {
      setPicked((prev) => new Set(prev).add(p.player_id));
    } else {
      setMsg(`You can protect ${limits[slot]} ${PROTECT_LABEL[slot].toLowerCase()}.`);
    }
  };
  const changeScheme = (next) => {
    setScheme(next);
    setMsg("");
    const lim = SCHEME_LIMITS[next];
    const used = {};
    const keep = new Set();
    players
      .filter((p) => picked.has(p.player_id) && !p.exempt)
      .sort((a, b) => Number(b.nmc) - Number(a.nmc) || b.ovr - a.ovr)
      .forEach((p) => {
        const s = p.pos === "G" || next === "7-3-1" ? p.pos : "S";
        if ((used[s] || 0) < lim[s]) {
          used[s] = (used[s] || 0) + 1;
          keep.add(p.player_id);
        }
      });
    setPicked(keep);
  };
  const save = async () => {
    setSaving(true);
    setMsg("");
    try {
      const data = await setExpansionProtection([...picked], scheme);
      if (data?.governance) onSaved?.(data.governance);
      setMsg("Protection list saved.");
    } catch (e) {
      setMsg(e?.response?.data?.detail || e?.message || "Could not save the list");
    } finally {
      setSaving(false);
    }
  };
  return (
    <section className={`${PREFIX}-panel`} style={{ padding: 14, display: "flex", flexDirection: "column", gap: 8 }}>
      <h3 style={{ margin: 0 }}>Expansion draft: {asArray(draft?.cities).join(" & ")}</h3>
      <p className={`${PREFIX}-note`} style={{ margin: 0 }}>
        {draft?.when}. Protect 7 forwards, 3 defensemen and 1 goalie, or any 8 skaters and 1 goalie. You lose{" "}
        {Number(draft?.n_new || 1) === 1 ? "one unprotected player" : `${draft.n_new} unprotected players`}. First- and
        second-year pros are exempt; no-move clauses must be protected. You must expose {exposeNeed.F} forwards and{" "}
        {exposeNeed.D} defenseman under contract with {"40+"} GP last season (or 70+ career), plus a signed goalie; if
        you don't, your weakest protected player at that position is exposed.
        {draft?.user_set ? "" : " Until you save, your highest-rated players are protected automatically."}
      </p>
      <div style={{ display: "flex", gap: 6 }}>
        {Object.keys(SCHEME_LIMITS).map((k) => (
          <button
            key={k}
            type="button"
            className="nhlcal-advance-button"
            style={{ opacity: scheme === k ? 1 : 0.55 }}
            onClick={() => changeScheme(k)}
          >
            {k === "7-3-1" ? "7F / 3D / 1G" : "8 skaters / 1G"}
          </button>
        ))}
      </div>
      <p className={`${PREFIX}-muted`} style={{ margin: 0 }}>
        {Object.keys(limits).map((k) => `${PROTECT_LABEL[k]} ${counts[k]}/${limits[k]}`).join(" · ")}
        {" — Exposed and eligible: "}
        {["F", "D", "G"].map((k) => `${k} ${exposure[k]}/${exposeNeed[k]}`).join(" · ")}
      </p>
      <div style={{ maxHeight: 280, overflow: "auto", display: "grid", gap: 2 }}>
        {players.map((p) => (
          <label key={p.player_id} style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, opacity: p.contract || p.exempt ? 1 : 0.6 }}>
            <input type="checkbox" checked={p.exempt || picked.has(p.player_id)} disabled={p.exempt} onChange={() => toggle(p)} />
            <span style={{ width: 18 }}>{p.pos}</span>
            <span style={{ flex: 1 }}>{p.name}</span>
            <span>{Math.round(p.ovr)} OVR · {p.age}</span>
            <span className={`${PREFIX}-muted`} style={{ width: 70, textAlign: "right" }}>
              {p.exempt ? "Exempt" : p.nmc ? "NMC" : !p.contract ? "Unsigned" : ""}
            </span>
          </label>
        ))}
      </div>
      <div className={`${PREFIX}-dock`}>
        <button type="button" className="nhlcal-advance-button" disabled={saving} onClick={save}>
          {saving ? "Saving…" : "Save protection list"}
        </button>
        {msg ? <span className={`${PREFIX}-muted`}>{msg}</span> : null}
      </div>
    </section>
  );
}

export default function BoardOfGovernorsMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const initial = pickFranchiseData(franchiseState, eventData, ["board_of_governors", "offseason.board_of_governors"]);
  const [gov, setGov] = useState(initial && typeof initial === "object" ? initial : null);
  const [stage, setStage] = useState("intro");
  const [active, setActive] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reveal, setReveal] = useState({});
  const timerRef = useRef(null);

  const proposals = useMemo(() => asArray(gov?.meeting?.proposals), [gov]);
  const tokens = Number(gov?.meeting?.lobby_tokens || 0);
  const allDone = proposals.length > 0 && proposals.every((p) => p.status !== "pending");

  useEffect(() => {
    let alive = true;
    getGovernance()
      .then((data) => {
        if (alive && data?.governance) setGov(data.governance);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => () => clearInterval(timerRef.current), []);

  // Land on the first undecided proposal.
  useEffect(() => {
    if (!proposals.length) return;
    const firstPending = proposals.findIndex((p) => p.status === "pending");
    if (firstPending >= 0 && proposals[active]?.status !== "pending" && reveal[proposals[active]?.id] == null) {
      setActive(firstPending);
    }
  }, [proposals]); // eslint-disable-line react-hooks/exhaustive-deps

  const startReveal = useCallback((pid, total) => {
    clearInterval(timerRef.current);
    setReveal((r) => ({ ...r, [pid]: 0 }));
    timerRef.current = setInterval(() => {
      setReveal((r) => {
        const n = (r[pid] || 0) + 1;
        if (n >= total) clearInterval(timerRef.current);
        return { ...r, [pid]: n };
      });
    }, REVEAL_MS);
  }, []);

  const current = proposals[active];

  const handleVote = async (vote) => {
    if (!current || busy) return;
    setBusy(true);
    setError("");
    try {
      const data = await voteGovernance({ proposal_id: current.id, vote });
      if (data?.governance) setGov(data.governance);
      const total = asArray(data?.proposal?.ballots).length;
      startReveal(current.id, total);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Vote failed");
    } finally {
      setBusy(false);
    }
  };

  const handleLobby = async (side) => {
    if (!current || busy) return;
    setBusy(true);
    setError("");
    try {
      const data = await lobbyGovernance({ proposal_id: current.id, side });
      if (data?.governance) setGov(data.governance);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Lobbying failed");
    } finally {
      setBusy(false);
    }
  };

  const handleNext = () => {
    const nextPending = proposals.findIndex((p, i) => i > active && p.status === "pending");
    const anyPending = proposals.findIndex((p) => p.status === "pending");
    if (nextPending >= 0) setActive(nextPending);
    else if (anyPending >= 0) setActive(anyPending);
    else setStage("adjourned");
  };

  useEffect(() => {
    if (stage !== "floor" || !current || current.status !== "pending" || busy) return undefined;
    const onKey = (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
      if (e.key === "a" || e.key === "A") handleVote("yes");
      if (e.key === "d" || e.key === "D") handleVote("no");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // re-bind with fresh closures

  const seasonLabel = gov?.season_label || "";
  const decidedCount = proposals.filter((p) => p.status !== "pending").length;
  const phaseLabel =
    stage === "intro"
      ? "Annual meeting"
      : allDone && stage === "adjourned"
        ? "Meeting adjourned"
        : `Proposal ${Math.min(active + 1, proposals.length)} of ${proposals.length} · ${decidedCount} decided`;

  return (
    <section className={`${PREFIX}-root`}>
      {null}

      <header className={`${PREFIX}-command`}>
        <div className={`${PREFIX}-command-left`}>
          <button type="button" className="nhlcal-quick-link" onClick={onBack}>
            ← Hub
          </button>
          <div className={`${PREFIX}-titles`}>
            <h1>Board of Governors</h1>
            <p>{[seasonLabel, gov?.meeting?.n_teams ? `${gov.meeting.n_teams} governors` : null].filter(Boolean).join(" · ")}</p>
          </div>
        </div>
        <p className={`${PREFIX}-phase`}>{phaseLabel}</p>
        <div className={`${PREFIX}-command-right`}>
          <span className={`${PREFIX}-pill ${tokens > 0 ? "is-on" : ""}`}>{tokens} lobby call{tokens === 1 ? "" : "s"}</span>
          <span className={`${PREFIX}-pill`}>{gov?.catalog_size || 200} rules on file</span>
        </div>
      </header>
      <FanTicker rows={gov?.fan_ticker} />

      {stage === "intro" ? (
        <main className={`${PREFIX}-intro`}>
          <div className={`${PREFIX}-seal`}>BOG</div>
          <h1>Board of Governors</h1>
          <p>
            {[seasonLabel, `${proposals.length || 5} proposals`, gov?.meeting?.n_teams ? `${gov.meeting.n_teams} votes` : null]
              .filter(Boolean)
              .join(" · ")}
          </p>
          <div className={`${PREFIX}-intro-lines`}>
            <span>On-ice and presentation rules need a majority. Bylaws need two-thirds. Relocation and expansion need three-quarters.</span>
            <span>Every governor votes their own club's interest: market size, profit, contender or rebuilder, cap room.</span>
            <span>You get two lobbying calls. Use them on the votes that matter to your club.</span>
            <span>Passed rules change on-ice play (scoring, penalties, overtime, the points system), revenue, the cap, contracts, trades, scouting, injuries and franchise values.</span>
            {gov?.on_ice?.points ? (
              <span>
                Current rules: {gov.on_ice.points} points · {gov.on_ice.ot_minutes}-minute OT ·{" "}
                {gov.on_ice.shootout ? "shootout" : "no shootout (ties stand)"}.
              </span>
            ) : null}
            {gov?.expansion_draft?.due ? (
              <span>
                Expansion draft this offseason ({asArray(gov.expansion_draft.cities).join(" & ")}): set your protection
                list before you leave the meeting.
              </span>
            ) : null}
            {asArray(gov?.expansion_draft?.upcoming).map((u, i) => (
              <span key={`up-${i}`}>
                Approved expansion ({asArray(u.cities).join(" & ")}): expansion draft at the {u.draft_offseason} Board
                meeting; clubs join for {u.start_season}-{String((Number(u.start_season) + 1) % 100).padStart(2, "0")}.
              </span>
            ))}
          </div>
          {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
          <button
            type="button"
            className="nhlcal-advance-button"
            disabled={!proposals.length}
            onClick={() => setStage(allDone ? "adjourned" : "floor")}
          >
            {allDone ? "Review the meeting" : "Call the meeting to order"}
          </button>
        </main>
      ) : null}

      {stage === "floor" && current ? (
        <div className={`${PREFIX}-floor`}>
          <nav className={`${PREFIX}-panel`} aria-label="Proposals">
            <div className={`${PREFIX}-panel-head`}>
              <h3>Agenda</h3>
              <span className={`${PREFIX}-muted`}>
                {decidedCount}/{proposals.length}
              </span>
            </div>
            <div className={`${PREFIX}-plist`}>
              {proposals.map((p, i) => (
                <button
                  key={p.id}
                  type="button"
                  className={`${PREFIX}-pitem ${i === active ? "is-active" : ""}`}
                  onClick={() => setActive(i)}
                >
                  <span className="n">{i + 1}</span>
                  <span>
                    <strong>{p.title}</strong>
                    <em>{p.category_label}</em>
                  </span>
                  <span className={`${PREFIX}-status ${p.status}`}>
                    {p.status === "pending" ? "Open" : p.status === "passed" ? `Pass ${p.tally?.yes}` : `Fail ${p.tally?.yes}`}
                  </span>
                </button>
              ))}
            </div>
            {allDone ? (
              <div style={{ padding: 12 }}>
                <button type="button" className="nhlcal-advance-button" onClick={() => setStage("adjourned")}>
                  Close the meeting
                </button>
              </div>
            ) : null}
          </nav>

          <div style={{ minHeight: 0, overflow: "auto", display: "flex", flexDirection: "column", gap: 10 }}>
            <ProposalCard
              proposal={current}
              tokens={tokens}
              busy={busy}
              onVote={handleVote}
              onLobby={handleLobby}
              reveal={reveal[current.id] != null ? reveal[current.id] : 999}
              onNext={handleNext}
              isLast={proposals.every((p, i) => i === active || p.status !== "pending")}
            />
            {error ? <p className={`${PREFIX}-error`}>{error}</p> : null}
          </div>

          <SidePanel gov={gov || {}} />
        </div>
      ) : null}

      {stage === "adjourned" ? (
        <div className={`${PREFIX}-floor`}>
          <div />
          <section className={`${PREFIX}-panel ${PREFIX}-adjourn`}>
            <h2>Meeting adjourned</h2>
            <ul>
              {proposals.map((p) => (
                <li key={p.id}>
                  <span>{p.title}</span>
                  <span className={`${PREFIX}-status ${p.status}`}>
                    {p.status === "passed" ? "Passed" : p.status === "failed" ? "Failed" : "Open"} {p.tally ? `${p.tally.yes}–${p.tally.no}` : ""}
                  </span>
                </li>
              ))}
            </ul>
            {Number(gov?.pending_cap_adjust_m || 0) !== 0 ? (
              <p className={`${PREFIX}-note`}>
                Next season's cap moves {Number(gov.pending_cap_adjust_m) > 0 ? "+" : ""}
                {Number(gov.pending_cap_adjust_m).toFixed(2)}M on top of the league projection.
              </p>
            ) : null}
            {gov?.expansion_draft?.due ? (
              <ExpansionProtectionPanel draft={gov.expansion_draft} onSaved={setGov} />
            ) : null}
            <div className={`${PREFIX}-dock`}>
              <button type="button" className="nhlcal-advance-button" onClick={onContinue}>
                {gov?.expansion_draft?.due ? "Hold the expansion draft & continue" : "Continue to Cap Report"}
              </button>
              <button type="button" className={`${PREFIX}-link`} onClick={() => setStage("floor")}>
                Back to the floor
              </button>
            </div>
          </section>
          <SidePanel gov={gov || {}} />
        </div>
      ) : null}
    </section>
  );
}
