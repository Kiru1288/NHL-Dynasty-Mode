import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGovernance, lobbyGovernance, voteGovernance } from "../../services/franchiseService";
import { getTeamLogoSrc, toLogoUrl } from "../../utils/teamLogos";
import { pickFranchiseData } from "../shared/eventHelpers";
import "../../styles/nhlcalShell.css";

/*
 * Board of Governors — annual offseason meeting.
 * Visual language matches the Entry Draft floor (same tokens, type and buttons).
 */

const PREFIX = "bog";
const REVEAL_MS = 55;

const SHEET = `
.bog-root{
  --bg:#04101a;--panel:rgba(9,25,38,.94);--panel-2:rgba(12,35,52,.94);--panel-3:rgba(15,46,66,.78);
  --line:rgba(156,218,236,.14);--line-soft:rgba(156,218,236,.08);--line-strong:rgba(73,231,240,.5);
  --text:#e9f7fb;--muted:#8096a8;--muted-2:#607789;
  --cyan:#13d8e7;--cyan-soft:rgba(19,216,231,.13);--gold:#e9a83c;--gold-soft:rgba(233,168,60,.14);
  --green:#52df94;--green-soft:rgba(82,223,148,.13);--red:#ff606d;--red-soft:rgba(255,96,109,.13);
  --depth-registered:inset 0 1px 0 rgba(255,255,255,.04);
  --ed-head:"Barlow Condensed", var(--font-broadcast-display), "Archivo Black", "Rajdhani", "Arial Narrow", sans-serif;
  --ed-mono:var(--font-mono-data, "IBM Plex Mono", Consolas, "Courier New", monospace);
  --ed-body:var(--font-ops-ui, Inter, ui-sans-serif, system-ui, "Segoe UI", sans-serif);
  position:relative;display:flex;flex-direction:column;min-height:100dvh;height:100%;max-height:100dvh;
  background:
    radial-gradient(circle at 24% 0%, rgba(19,216,231,.12), transparent 30%),
    radial-gradient(circle at 92% 18%, rgba(233,168,60,.08), transparent 26%),
    linear-gradient(180deg,#06131f 0%,#020a11 100%);
  color:var(--text);font-family:var(--ed-body);font-size:14px;letter-spacing:.01em;overflow:hidden;
}
.bog-root *{box-sizing:border-box;}
.bog-root h1,.bog-root h2,.bog-root h3,.bog-root h4{font-family:var(--ed-head);letter-spacing:.04em;margin:0;}
.bog-root button{font-family:inherit;}
.bog-tabular{font-family:var(--ed-mono);font-variant-numeric:tabular-nums;letter-spacing:.02em;}
.bog-muted{color:var(--muted);font-size:12.5px;}
.bog-root .nhlcal-quick-link{
  border:1px solid var(--line);border-radius:4px;background:rgba(12,31,47,.72);color:var(--text);
  padding:9px 14px;font-size:11px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;
  transition:border-color .2s ease,background .2s ease,transform .2s ease;
}
.bog-root .nhlcal-quick-link:hover{border-color:var(--line-strong);background:rgba(19,216,231,.12);color:var(--cyan);transform:translateY(-1px);}

/* command bar */
.bog-command{position:relative;z-index:3;display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:14px;padding:12px;
  background:radial-gradient(circle at 0% 0%, rgba(19,216,231,.08), transparent 42%), var(--panel);
  border-bottom:1px solid var(--line);box-shadow:var(--depth-registered);}
.bog-command-left{display:flex;align-items:center;gap:12px;min-width:0;}
.bog-titles h1{font-weight:900;font-size:20px;line-height:1;letter-spacing:.05em;text-transform:uppercase;}
.bog-titles p{margin:2px 0 0;font-size:11px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;font-weight:800;}
.bog-phase{margin:0;text-align:center;font-size:12px;font-weight:1000;letter-spacing:.14em;text-transform:uppercase;color:var(--gold);text-shadow:0 0 18px rgba(233,168,60,.28);}
.bog-command-right{display:flex;align-items:center;gap:8px;}
.bog-pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.bog-pill.is-on{color:var(--cyan);border-color:rgba(19,216,231,.4);background:var(--cyan-soft);}

/* intro */
.bog-intro{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:13px;padding:40px 24px;text-align:center;}
.bog-seal{width:112px;height:112px;display:flex;align-items:center;justify-content:center;font-family:var(--ed-head);font-size:29px;letter-spacing:.05em;color:var(--gold);
  border:1px solid rgba(233,168,60,.42);background:radial-gradient(circle at 50% 30%, rgba(233,168,60,.18), transparent 70%), rgba(6,21,34,.82);
  box-shadow:0 0 18px rgba(233,168,60,.22);clip-path:polygon(50% 0,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);}
.bog-intro h1{font-size:58px;font-weight:500;line-height:1;letter-spacing:.05em;text-transform:uppercase;}
.bog-intro p{margin:0;color:var(--muted);font-size:13.5px;letter-spacing:.09em;}
.bog-intro-lines{display:flex;flex-direction:column;gap:5px;max-width:640px;margin-top:6px;}
.bog-intro-lines span{font-size:12.5px;color:var(--muted);padding:6px 12px;background:rgba(6,21,34,.72);border-left:2px solid var(--gold);text-align:left;box-shadow:inset 0 1px 0 rgba(255,255,255,.04);}

/* floor */
.bog-floor{flex:1;min-height:0;display:grid;grid-template-columns:minmax(230px,280px) minmax(0,1fr) minmax(260px,320px);gap:12px;padding:12px;}
.bog-panel{min-height:0;overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel);box-shadow:var(--depth-registered);}
.bog-panel-head{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--line-soft);}
.bog-panel-head h3{font-size:14px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}

.bog-plist{display:flex;flex-direction:column;}
.bog-pitem{all:unset;cursor:pointer;display:grid;grid-template-columns:26px 1fr auto;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line-soft);}
.bog-pitem:hover{background:rgba(255,255,255,.03);}
.bog-pitem.is-active{background:var(--cyan-soft);box-shadow:inset 2px 0 0 var(--cyan);}
.bog-pitem .n{font-family:var(--ed-head);font-size:18px;color:var(--muted);text-align:center;}
.bog-pitem strong{display:block;font-size:12.5px;font-weight:800;line-height:1.2;}
.bog-pitem em{display:block;font-style:normal;font-size:10.5px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:2px;}
.bog-status{font-size:10px;font-weight:1000;letter-spacing:.1em;text-transform:uppercase;padding:3px 7px;border-radius:4px;border:1px solid var(--line);color:var(--muted);white-space:nowrap;}
.bog-status.passed{color:var(--green);border-color:rgba(82,223,148,.4);background:var(--green-soft);}
.bog-status.failed{color:var(--red);border-color:rgba(255,96,109,.4);background:var(--red-soft);}
.bog-status.pending{color:var(--gold);border-color:rgba(233,168,60,.35);}

/* proposal card */
.bog-card{display:flex;flex-direction:column;gap:14px;padding:18px;}
.bog-kicker{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
.bog-chip{font-size:10.5px;font-weight:1000;letter-spacing:.12em;text-transform:uppercase;padding:4px 8px;border-radius:4px;background:var(--gold-soft);color:var(--gold);border:1px solid rgba(233,168,60,.3);}
.bog-chip.cyan{background:var(--cyan-soft);color:var(--cyan);border-color:rgba(19,216,231,.3);}
.bog-card h2{font-size:40px;font-weight:600;line-height:1;text-transform:uppercase;letter-spacing:.04em;}
.bog-summary{margin:0;font-size:14px;color:#c9dce6;line-height:1.5;max-width:760px;}
.bog-effects{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;}
.bog-effect{border:1px solid var(--line);border-radius:6px;padding:9px 11px;background:rgba(6,21,34,.62);}
.bog-effect span{display:block;font-size:10.5px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;font-weight:800;}
.bog-effect strong{display:block;margin-top:3px;font-family:var(--ed-head);font-size:22px;letter-spacing:.03em;}
.bog-effect em{display:block;margin-top:3px;font-style:normal;font-size:11.5px;color:var(--muted);}
.bog-effect.is-pro{border-color:rgba(82,223,148,.45);}
.bog-effect.is-pro strong{color:#52df94;}
.bog-effect.is-con{border-color:rgba(255,96,109,.45);}
.bog-effect.is-con strong{color:#ff606d;}
.bog-effect.is-mixed strong{color:var(--gold);}
.bog-proscons{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin-top:10px;}
.bog-proscons > div{border:1px solid var(--line);border-radius:6px;padding:9px 11px;background:rgba(6,21,34,.62);}
.bog-proscons span{display:block;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;font-weight:900;margin-bottom:4px;}
.bog-proscons .pros span{color:#52df94;}
.bog-proscons .cons span{color:#ff606d;}
.bog-proscons .mixed span{color:var(--gold);}
.bog-proscons p{margin:3px 0;font-size:12.5px;line-height:1.35;}
.bog-proscons p.none{color:var(--muted);}
.bog-forecast{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;font-size:12px;color:var(--muted);}
.bog-meter{position:relative;height:8px;border-radius:4px;background:rgba(148,185,205,.12);overflow:visible;}
.bog-meter i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:linear-gradient(90deg,var(--cyan),rgba(19,216,231,.5));}
.bog-meter b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--gold);box-shadow:0 0 8px rgba(233,168,60,.5);}
.bog-dock{display:flex;flex-wrap:wrap;gap:10px;align-items:center;}
.bog-dock .nhlcal-advance-button{min-width:170px;height:48px;}
.bog-link{all:unset;cursor:pointer;font-size:11.5px;font-weight:800;color:var(--cyan);letter-spacing:.04em;}
.bog-link:disabled,.bog-link[aria-disabled="true"]{opacity:.45;cursor:not-allowed;}
.bog-error{color:var(--red);font-size:12.5px;margin:0;}
.bog-note{font-size:12px;color:var(--muted);margin:0;}

/* tally */
.bog-tally{display:flex;flex-direction:column;gap:12px;border-top:1px solid var(--line-soft);padding-top:14px;}
.bog-count{display:grid;grid-template-columns:1fr auto 1fr;align-items:end;gap:12px;}
.bog-count .side{display:flex;flex-direction:column;}
.bog-count .side.no{align-items:flex-end;}
.bog-count .num{font-family:var(--ed-head);font-size:56px;line-height:.9;}
.bog-count .yes .num{color:var(--green);}
.bog-count .no .num{color:var(--red);}
.bog-count .lbl{font-size:10.5px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}
.bog-count .need{text-align:center;font-size:11px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;font-weight:800;}
.bog-tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:6px;}
.bog-tile{position:relative;display:flex;align-items:center;gap:5px;padding:5px 6px;border-radius:5px;border:1px solid var(--line);background:rgba(6,21,34,.62);font-size:11px;font-weight:900;opacity:.25;transition:opacity .18s ease,border-color .18s ease,background .18s ease;}
.bog-tile.shown{opacity:1;}
.bog-tile.yes{border-color:rgba(82,223,148,.5);background:var(--green-soft);}
.bog-tile.no{border-color:rgba(255,96,109,.45);background:var(--red-soft);}
.bog-tile.abstain{border-color:var(--line);}
.bog-tile.user{box-shadow:0 0 0 1px var(--gold) inset;}
.bog-tile img{width:18px;height:18px;object-fit:contain;}
.bog-tile .fb{width:18px;height:18px;display:grid;place-items:center;font-size:8px;color:var(--muted);}
.bog-tile .lob{position:absolute;top:-5px;right:-4px;font-size:9px;color:var(--cyan);}
.bog-result{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-radius:6px;border:1px solid var(--line);}
.bog-result h3{font-size:30px;font-weight:700;text-transform:uppercase;}
.bog-result.passed{border-color:rgba(82,223,148,.45);background:var(--green-soft);}
.bog-result.passed h3{color:var(--green);}
.bog-result.failed{border-color:rgba(255,96,109,.45);background:var(--red-soft);}
.bog-result.failed h3{color:var(--red);}
.bog-reasons{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:4px 12px;font-size:11.5px;color:var(--muted);}
.bog-reasons b{color:var(--text);}

/* side */
.bog-side{display:flex;flex-direction:column;gap:0;}
.bog-kv{display:grid;grid-template-columns:1fr auto;gap:6px 10px;padding:12px;font-size:12px;}
.bog-kv span{color:var(--muted);}
.bog-kv strong{text-align:right;font-weight:800;}
.bog-kv strong.up{color:var(--green);} .bog-kv strong.down{color:var(--red);}
.bog-mods{display:flex;flex-direction:column;gap:4px;padding:10px 12px;}
.bog-mod{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;border-bottom:1px solid var(--line-soft);padding:4px 0;}
.bog-mod span{color:var(--muted);}
.bog-hist{display:flex;flex-direction:column;padding:6px 12px 12px;}
.bog-hrow{display:grid;grid-template-columns:1fr auto;gap:8px;font-size:11.5px;padding:5px 0;border-bottom:1px solid var(--line-soft);}
.bog-hrow em{font-style:normal;font-weight:900;font-size:10px;letter-spacing:.08em;}
.bog-hrow em.passed{color:var(--green);} .bog-hrow em.failed{color:var(--red);}

/* adjourned */
.bog-adjourn{display:flex;flex-direction:column;gap:12px;padding:18px;}
.bog-adjourn h2{font-size:40px;font-weight:600;text-transform:uppercase;}
.bog-adjourn ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:6px;}
.bog-adjourn li{display:flex;justify-content:space-between;gap:12px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;font-size:13px;}

@media (max-width: 1100px){
  .bog-floor{grid-template-columns:1fr;overflow:auto;}
  .bog-card h2{font-size:30px;}
}
@media (prefers-reduced-motion: reduce){ .bog-tile{transition:none;} }
`;

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

function ProposalCard({ proposal, tokens, busy, onVote, onLobby, reveal, onNext, isLast }) {
  const decided = proposal.status !== "pending";
  const forecast = proposal.forecast || {};
  const needed = Number(proposal.votes_needed || 0);
  const nTeams = asArray(proposal.ballots).length || 32;
  const expected = Number(forecast.expected_yes || 0);
  const ballots = asArray(proposal.ballots);
  const shownCount = decided ? Math.min(reveal, ballots.length) : 0;
  const shown = ballots.slice(0, shownCount);
  const yes = shown.filter((b) => b.vote === "yes").length;
  const no = shown.filter((b) => b.vote === "no").length;
  const revealDone = decided && shownCount >= ballots.length;
  const tally = proposal.tally || {};
  const lobbied = proposal.lobbied;

  return (
    <section className={`${PREFIX}-panel ${PREFIX}-card`} aria-live="polite">
      <div className={`${PREFIX}-kicker`}>
        <span className={`${PREFIX}-chip`}>{proposal.category_label}</span>
        <span className={`${PREFIX}-chip cyan`}>
          {proposal.threshold_label} · {needed} votes needed
        </span>
        {lobbied ? (
          <span className={`${PREFIX}-chip cyan`}>
            You lobbied {lobbied.side === "yes" ? "for" : "against"}: {asArray(lobbied.teams).join(", ")}
          </span>
        ) : null}
      </div>
      <h2>{proposal.title}</h2>
      <p className={`${PREFIX}-summary`}>{proposal.summary}</p>

      <div className={`${PREFIX}-effects`}>
        {asArray(proposal.effects).map((e) => (
          <div key={`${e.key}-${e.label}`} className={`${PREFIX}-effect ${e.tone ? `is-${e.tone}` : ""}`}>
            <span>{e.label}</span>
            <strong>{e.text}</strong>
            {e.note ? <em>{e.note}</em> : null}
          </div>
        ))}
      </div>

      {proposal.pros_cons ? (
        <div className={`${PREFIX}-proscons`}>
          <div className="pros">
            <span>Pros</span>
            {asArray(proposal.pros_cons.pros).length ? (
              asArray(proposal.pros_cons.pros).map((t) => <p key={t}>+ {t}</p>)
            ) : (
              <p className="none">No clear upside for most clubs</p>
            )}
          </div>
          <div className="cons">
            <span>Cons</span>
            {asArray(proposal.pros_cons.cons).length ? (
              asArray(proposal.pros_cons.cons).map((t) => <p key={t}>− {t}</p>)
            ) : (
              <p className="none">No obvious downside</p>
            )}
          </div>
          {asArray(proposal.pros_cons.mixed).length ? (
            <div className="mixed">
              <span>Trade-offs</span>
              {asArray(proposal.pros_cons.mixed).map((t) => <p key={t}>± {t}</p>)}
            </div>
          ) : null}
        </div>
      ) : null}

      {!decided ? (
        <>
          <div className={`${PREFIX}-forecast`}>
            <span>Board read</span>
            <div className={`${PREFIX}-meter`} aria-hidden="true">
              <i style={{ width: `${Math.min(100, (expected / Math.max(1, nTeams)) * 100)}%` }} />
              <b style={{ left: `${Math.min(100, (needed / Math.max(1, nTeams)) * 100)}%` }} />
            </div>
            <strong>
              {forecast.label || "—"} · ≈{expected.toFixed(0)} of {needed}
            </strong>
          </div>
          <div className={`${PREFIX}-dock`}>
            <button type="button" className="nhlcal-advance-button" disabled={busy} onClick={() => onVote("yes")}>
              Agree
            </button>
            <button
              type="button"
              className="nhlcal-advance-button nhlcal-advance-button-secondary"
              disabled={busy}
              onClick={() => onVote("no")}
            >
              Disagree
            </button>
            <button type="button" className={`${PREFIX}-link`} disabled={busy} onClick={() => onVote("abstain")}>
              Abstain
            </button>
            <span style={{ flex: 1 }} />
            <button
              type="button"
              className={`${PREFIX}-link`}
              disabled={busy || tokens <= 0 || Boolean(lobbied)}
              onClick={() => onLobby("yes")}
            >
              Lobby for it
            </button>
            <button
              type="button"
              className={`${PREFIX}-link`}
              disabled={busy || tokens <= 0 || Boolean(lobbied)}
              onClick={() => onLobby("no")}
            >
              Lobby against
            </button>
          </div>
          <p className={`${PREFIX}-note`}>
            Lobbying calls the four most persuadable governors whose clubs look like yours. {tokens} call
            {tokens === 1 ? "" : "s"} left this meeting.
          </p>
        </>
      ) : (
        <div className={`${PREFIX}-tally`}>
          <div className={`${PREFIX}-count`}>
            <div className="side yes">
              <span className="num bog-tabular">{yes}</span>
              <span className="lbl">Agree</span>
            </div>
            <div className="need">
              {needed} needed
              <br />
              {revealDone && tally.abstain ? `${tally.abstain} abstained` : ""}
            </div>
            <div className="side no">
              <span className="num bog-tabular">{no}</span>
              <span className="lbl">Disagree</span>
            </div>
          </div>
          <div className={`${PREFIX}-tiles`}>
            {ballots.map((b, i) => (
              <TeamTile key={b.team_id || i} ballot={b} shown={i < shownCount} />
            ))}
          </div>
          {revealDone ? (
            <>
              <div className={`${PREFIX}-result ${proposal.status}`}>
                <div>
                  <h3>{proposal.status === "passed" ? "Passed" : "Failed"}</h3>
                  <span className={`${PREFIX}-muted`}>
                    {tally.yes}–{tally.no} · you voted {proposal.user_vote}
                  </span>
                </div>
                <button type="button" className="nhlcal-advance-button" onClick={onNext}>
                  {isLast ? "Close the meeting" : "Next proposal"}
                </button>
              </div>
              <div className={`${PREFIX}-reasons`}>
                {ballots
                  .filter((b) => !b.is_user)
                  .slice(0, 12)
                  .map((b) => (
                    <span key={b.team_id}>
                      <b>{b.abbr}</b> {b.vote === "yes" ? "✓" : "✗"} {b.reason}
                    </span>
                  ))}
              </div>
            </>
          ) : null}
        </div>
      )}
    </section>
  );
}

function SidePanel({ gov }) {
  const impact = gov.user_impact || {};
  const mods = asArray(gov.modifiers);
  const hist = asArray(gov.history).slice(0, 8);
  const change = Number(impact.value_change_pct || 0);
  return (
    <aside className={`${PREFIX}-panel ${PREFIX}-side`}>
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
      <style>{SHEET}</style>

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
            <span>Passed rules change revenue, the cap, contracts, trades, scouting, injuries and franchise values.</span>
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
            <div className={`${PREFIX}-dock`}>
              <button type="button" className="nhlcal-advance-button" onClick={onContinue}>
                Continue to Cap Report
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
