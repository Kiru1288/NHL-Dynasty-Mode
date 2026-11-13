import React, { useState } from "react";
import { arr, s, num, Face, TeamMark, DeskHead, DeskMissing } from "./benchBuzzShared";

/* Rumour Ledger: every trade / contract rumour on the wire, checked against what
   actually happened in the save, plus each insider's hit rate. */

const VERDICT = {
  hit: { label: "Called it", cls: "is-good" },
  miss: { label: "Missed", cls: "is-bad" },
  settled: { label: "Settled", cls: "is-warn" },
  pending: { label: "Pending", cls: "" },
};

function LedgerRow({ it }) {
  const v = VERDICT[it.verdict] || VERDICT.pending;
  const pending = it.verdict === "pending";
  return (
    <li className={`bbe-ledger-row is-${s(it.verdict)}`}>
      <Face row={{ ...it, name: it.player_name, team_abbr: it.team_abbr }} px={60} />
      <div className="bbe-ledger-row__body">
        <b>{s(it.headline)}</b>
        <i>
          {s(it.reporter)}{it.outlet ? ` (${it.outlet})` : ""} · filed {s(it.filed_label)}
          {it.kind === "contract" ? " · contract" : " · trade"}
          {num(it.mentions) > 1 ? ` · repeated ${it.mentions}×` : ""}
        </i>
        {pending ? (
          <em>{num(it.age_days) != null ? `${it.age_days} day${it.age_days === 1 ? "" : "s"} old` : "Fresh"}{it.note ? ` · ${it.note}` : ""}</em>
        ) : (
          <em>{s(it.outcome)}{it.resolved_label ? ` · ${it.resolved_label}` : ""}</em>
        )}
      </div>
      <TeamMark teamId={it.team_id} abbr={it.team_abbr} />
      <span className={`ccx-stamp bbe-verdict ${v.cls}`}>{v.label}</span>
    </li>
  );
}

export default function RumourLedgerDesk({ features }) {
  const [tab, setTab] = useState("resolved");
  const data = features?.rumour_ledger;
  if (!data) return <DeskMissing kicker="Insider accountability" title="Rumour Ledger" />;
  if (data.error) {
    return (
      <section className="bbe-desk">
        <DeskHead kicker="Insider accountability" title="Rumour Ledger" />
        <div className="bbe-empty"><strong>The ledger didn't balance today.</strong><p>Advance a day and it re-checks the wire.</p></div>
      </section>
    );
  }
  const resolved = arr(data.resolved);
  const pending = arr(data.pending);
  const reporters = arr(data.reporters);
  const active = tab === "pending" || !resolved.length ? "pending" : "resolved";
  const list = active === "pending" ? pending : resolved;
  const rate = data.decided ? Math.round((data.hits / data.decided) * 100) : null;
  return (
    <section className="bbe-desk bbe-ledger-desk">
      <DeskHead
        kicker="Insider accountability"
        title="Rumour Ledger"
        lede="Every trade and contract rumour on the wire, checked against what really happened in your league. Who actually knows things, and who just fills airtime?"
      >
        <div className="ccx-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={active === "resolved"} className={`ccx-tab ${active === "resolved" ? "is-active" : ""}`} onClick={() => setTab("resolved")} disabled={!resolved.length}>
            Played out <span className="ccx-tab__count">{resolved.length}</span>
          </button>
          <button type="button" role="tab" aria-selected={active === "pending"} className={`ccx-tab ${active === "pending" ? "is-active" : ""}`} onClick={() => setTab("pending")}>
            Still open <span className="ccx-tab__count">{pending.length}</span>
          </button>
        </div>
      </DeskHead>

      <div className="bbe-ledger-strip">
        <span><b className="bbe-bignum">{data.tracked || 0}</b><small>rumours tracked</small></span>
        <span><b className="bbe-bignum">{data.hits || 0}</b><small>called it</small></span>
        <span><b className="bbe-bignum">{data.misses || 0}</b><small>missed</small></span>
        <span><b className="bbe-bignum">{rate != null ? `${rate}%` : "—"}</b><small>wire accuracy</small></span>
        <p>{s(data.note)}</p>
      </div>

      <div className="bbe-ledger-layout">
        <div className="bbe-sheet">
          <div className="bbe-sheet__top">
            <span className="ccx-kicker">{active === "pending" ? "Still open, newest first" : "The ledger, latest first"}</span>
          </div>
          <ol className="bbe-ledger">
            {list.map((it) => <LedgerRow key={it.key} it={it} />)}
            {!list.length ? (
              <li className="bbe-ledger-row is-empty">
                No trade or contract rumours are on file yet. When the insiders start talking, each report lands here and gets checked against the record.
              </li>
            ) : null}
          </ol>
        </div>
        <aside className="bbe-sheet bbe-board">
          <div className="bbe-sheet__top"><span className="ccx-kicker">Reporter leaderboard</span></div>
          <ol className="bbe-boardlist">
            {reporters.map((r, i) => (
              <li key={r.reporter}>
                <span className="bbe-boardlist__rank">{i + 1}</span>
                <span className="bbe-boardlist__who">
                  <b>{s(r.reporter)}</b>
                  <i>{r.outlet ? `${r.outlet} · ` : ""}{r.filed} filed · {r.pending} open{r.settled ? ` · ${r.settled} settled` : ""}</i>
                </span>
                <span className="bbe-boardlist__rate">
                  {r.hit_rate != null ? <><b>{Math.round(r.hit_rate * 100)}%</b><small>{r.hits}/{r.decided}</small></> : <small>no verdicts yet</small>}
                </span>
              </li>
            ))}
            {!reporters.length ? <li className="is-empty">No insiders on the board yet.</li> : null}
          </ol>
        </aside>
      </div>
    </section>
  );
}
