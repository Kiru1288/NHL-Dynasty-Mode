import React, { useState } from "react";
import { arr, s, num, TeamMark, DeskHead, DeskMissing } from "./benchBuzzShared";

/* Hot Seat: job-security heat for every head coach and GM, built from points pace
   vs projection, streak, org/owner pressure and fan mood. Your seats up top. */

const HEAT_KEY = (label) => s(label).toLowerCase().replace(/\s+/g, "-");

function Thermo({ heat }) {
  const h = Math.max(0, Math.min(100, num(heat) || 0));
  return (
    <div className="bbe-thermo" aria-label={`Heat ${Math.round(h)} of 100`}>
      <div className="bbe-thermo__tube"><i style={{ height: `${h}%` }} /></div>
      <div className="bbe-thermo__bulb" />
    </div>
  );
}

function MySeat({ row, heading }) {
  if (!row) return null;
  return (
    <article className={`bbe-seat is-${HEAT_KEY(row.label)}`}>
      <Thermo heat={row.heat} />
      <div className="bbe-seat__main">
        <div className="bbe-seat__top">
          <span className="ccx-kicker">{heading}</span>
          <span className={`ccx-stamp bbe-heat-stamp is-${HEAT_KEY(row.label)}`}>{s(row.label)}</span>
        </div>
        <h3 className="bbe-seat__name">
          <TeamMark teamId={row.team_id} abbr={row.team_abbr} name={row.team_name} />
          {s(row.name)}
        </h3>
        <div className="bbe-seat__nums">
          <span><b className="bbe-bignum">{Math.round(num(row.heat) || 0)}</b><small>heat / 100</small></span>
          <span><b>{s(row.record)}</b><small>record</small></span>
          <span><b>{row.expected_points ?? "—"}</b><small>projected pts</small></span>
          <span><b>#{row.rank}</b><small>hottest of 32</small></span>
        </div>
        <ul className="bbe-reasons">
          {arr(row.reasons).map((r) => (
            <li key={r.text} className={r.delta > 0 ? "is-up" : "is-down"}>
              <span>{r.delta > 0 ? "▲" : "▼"}</span>
              {r.text}
            </li>
          ))}
          {!arr(row.reasons).length ? <li className="is-flat">Nothing is moving the needle yet.</li> : null}
        </ul>
      </div>
    </article>
  );
}

function SeatTable({ rows, caption, showContract }) {
  return (
    <div className="bbe-sheet bbe-seat-table">
      <div className="bbe-sheet__top"><span className="ccx-kicker">{caption}</span></div>
      <ol className="bbe-seatlist">
        {rows.map((r) => (
          <li key={`${r.role}-${r.team_id}`} className={`${r.is_user_team ? "is-you" : ""} is-${HEAT_KEY(r.label)}`} title={arr(r.reasons).map((x) => `${x.delta > 0 ? "+" : ""}${x.delta} ${x.text}`).join("\n")}>
            <span className="bbe-seatlist__rank">{r.rank}</span>
            <TeamMark teamId={r.team_id} abbr={r.team_abbr} name={r.team_name} />
            <span className="bbe-seatlist__who">
              <b>{s(r.name)}</b>
              <i>{s(r.team_abbr)} · {s(r.record)}{r.streak ? ` · ${r.streak}` : ""}{showContract && r.contract_years != null ? ` · ${r.contract_years} yr left` : ""}</i>
            </span>
            <span className="bbe-seatlist__why">{s(arr(r.reasons)[0]?.text)}</span>
            <span className="bbe-heatbar" aria-hidden><i style={{ width: `${Math.max(3, Math.min(100, num(r.heat) || 0))}%` }} /></span>
            <span className="bbe-seatlist__heat">{Math.round(num(r.heat) || 0)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function HotSeatDesk({ features }) {
  const [showAll, setShowAll] = useState(false);
  const data = features?.hot_seat;
  if (!data) return <DeskMissing kicker="Job security" title="The Hot Seat" />;
  if (data.error) {
    return (
      <section className="bbe-desk">
        <DeskHead kicker="Job security" title="The Hot Seat" />
        <div className="bbe-empty"><strong>The hot seat meter didn't build.</strong><p>Advance a day and it rebuilds.</p></div>
      </section>
    );
  }
  const coaches = arr(data.coaches);
  const gms = arr(data.gms);
  const limit = showAll ? 32 : 16;
  return (
    <section className="bbe-desk bbe-seat-desk">
      <DeskHead
        kicker="Job security · league-wide"
        title="The Hot Seat"
        lede={`${s(data.note)}${data.early ? " It's early, so the record counts for less until clubs have about 20 games in." : ""}`}
      >
        <button type="button" className="ccx-btn ccx-btn--sm" onClick={() => setShowAll((v) => !v)}>{showAll ? "Hottest 16" : "All 32 clubs"}</button>
      </DeskHead>
      <div className="bbe-myseats">
        <MySeat row={data.user_gm} heading="Your seat · general manager" />
        <MySeat row={data.user_coach} heading="Your bench boss · head coach" />
      </div>
      <div className="bbe-seat-tables">
        <SeatTable rows={coaches.slice(0, limit)} caption="Head coaches · hottest first" showContract />
        <SeatTable rows={gms.slice(0, limit)} caption="General managers · hottest first" />
      </div>
    </section>
  );
}
