import React, { useMemo, useState } from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

/* Fallout: league discipline and off-ice consequences, split into "your club" and
   "around the league". Every effect shown here is one the game actually applies:
   cap penalties count against the season's cap, forfeited picks leave the draft,
   fines hit club finances, suspended players can't dress. Every field is optional;
   older saves without enrichment still render. */

const arr = (v) => (Array.isArray(v) ? v : []);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const s = (v) => (v == null ? "" : String(v));
const num = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};
const money = (v) => (num(v) > 0 ? `$${num(v).toFixed(2)}M` : "");
const label = (key) => s(key).replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
const seasonLabel = (y) => (num(y) ? `${num(y)}-${String((num(y) + 1) % 100).padStart(2, "0")}` : "");
const ordinal = (n) => {
  const v = num(n);
  if (v === 1) return "1st";
  if (v === 2) return "2nd";
  if (v === 3) return "3rd";
  return `${v}th`;
};

const EVENT_COPY = {
  UNDER_TABLE_PAYMENTS: "Under-the-table payments",
  CAP_CIRCUMVENTION: "Cap circumvention",
  ILLEGAL_TEAM_WORKOUTS: "Illegal team workouts",
};

const STATUS_COPY = {
  suspended: "Suspended",
  banned: "Banned",
  investigative_leave: "Investigative leave",
  personal_leave: "Personal leave",
  family_leave: "Family leave",
  deceased: "Deceased",
  unavailable: "Unavailable",
};

function teamName(row, teamsById) {
  if (row?.team_name) return s(row.team_name);
  const t = teamsById[s(row?.team_id)];
  return t ? s(t.name || t.abbr || row.team_id) : s(row?.team_id || "Unknown club");
}

function TeamMark({ row, teamsById, size = 40 }) {
  const name = teamName(row, teamsById);
  const abbr = s(row?.team_abbr || teamsById[s(row?.team_id)]?.abbr);
  const logo = resolveFranchiseTeamLogo({ team_id: row?.team_id, team_name: name, team_abbrev: abbr, abbrev: abbr }, name) || "";
  return (
    <span className="bbd-logo" style={{ "--bbd-logo": `${size}px` }} aria-hidden>
      {logo ? <img src={logo} alt="" /> : <b>{(abbr || name).slice(0, 3).toUpperCase()}</b>}
    </span>
  );
}

/** What the sanction actually did, in plain sentences. */
function sanctionEffects(x) {
  const applied = obj(x.applied);
  const out = [];
  const cap = obj(applied.cap);
  if (num(x.cap_penalty_m) > 0) {
    const ssn = cap.season_label || seasonLabel(cap.season || x.season);
    out.push({
      tone: "cap",
      text: `${money(x.cap_penalty_m)} cap penalty ${cap.amount_m ? "applied to" : "against"} the ${ssn} cap`,
    });
  }
  const pickResults = arr(applied.picks);
  if (pickResults.length) {
    for (const p of pickResults) {
      if (s(p.status) === "forfeited") {
        out.push({ tone: "pick", text: `${s(p.year || p.requested_year)} ${ordinal(p.round)}-round pick forfeited` });
      } else {
        out.push({ tone: "pick", text: `${s(p.requested_year)} ${ordinal(p.round)}-round pick owed (none left to take)` });
      }
    }
  } else {
    for (const p of arr(x.forfeited_picks)) {
      out.push({ tone: "pick", text: `${s(p.applied_year || p.draft_year)} ${ordinal(p.round)}-round pick forfeited` });
    }
  }
  if (num(x.fine_m) > 0) {
    out.push({ tone: "fine", text: `${money(x.fine_m)} fine charged to club finances` });
  }
  return out;
}

function Section({ tone = "", heading, count, children, wide = false }) {
  return (
    <section className={`bbd-sec${tone ? ` is-${tone}` : ""}${wide ? " is-wide" : ""}`}>
      <header className="bbd-sec__head">
        <h3 className="bbd-h">{heading}</h3>
        {count != null ? <span className="bbd-count">{count}</span> : null}
      </header>
      <div className="bbd-sec__body">{children}</div>
    </section>
  );
}

function Tally({ value, text, tone }) {
  return (
    <div className={`bbd-tally${tone ? ` is-${tone}` : ""}`}>
      <b>{value}</b>
      <small>{text}</small>
    </div>
  );
}

export default function FalloutBoard({ narrativeUniverse, stories = [], onOpenStory, franchiseState }) {
  const [scope, setScope] = useState("club");
  const userTid = s(franchiseState?.user_team_id ?? franchiseState?.team?.id ?? "");
  const teamsById = useMemo(() => {
    const out = {};
    for (const o of arr(franchiseState?.roster_browser?.organizations)) {
      out[s(o?.team_id)] = { name: o?.name, abbr: o?.abbr };
    }
    return out;
  }, [franchiseState?.roster_browser?.organizations]);

  const nu = obj(narrativeUniverse);
  const isMine = (row) => (row?.is_user != null ? Boolean(row.is_user) : !!userTid && s(row?.team_id) === userTid);
  const seasonNow = num(nu.fallout_season);
  const seasonNowLabel = s(nu.fallout_season_label) || seasonLabel(seasonNow);

  const sanctions = arr(nu.team_sanctions).filter((x) => x && x.active !== false).slice().reverse();
  const picks = arr(nu.forfeited_picks).filter(Boolean).slice().reverse();
  const capRows = arr(nu.cap_penalty_rows).length
    ? arr(nu.cap_penalty_rows)
    : Object.entries(obj(nu.cap_penalties))
        .map(([team_id, amt]) => ({ team_id, amount_m: typeof amt === "object" && amt ? num(amt.cap_penalty_m || amt.amount) : num(amt) }))
        .filter((r) => num(r.amount_m) > 0);
  const availability = Object.entries(obj(nu.player_availability))
    .map(([pid, row]) => ({ pid, ...obj(row) }))
    .filter((r) => r.active !== false && s(r.status) !== "available");
  const promises = obj(obj(nu.player_meetings).promises);
  const broken = arr(promises.broken).slice(-6).reverse();
  const budget = obj(nu.major_event_budget);

  const inScope = (rows, sc) => rows.filter((r) => (sc === "club" ? isMine(r) : !isMine(r)));
  const outIn = (sc) =>
    inScope(availability.filter((r) => r.team_id || r.team_name), sc).concat(
      sc === "club" ? availability.filter((r) => !r.team_id && !r.team_name) : []
    );

  const build = (sc) => {
    const sx = inScope(sanctions, sc);
    const px = inScope(picks, sc);
    const cx = inScope(capRows, sc);
    const ox = outIn(sc);
    const bx = sc === "club" ? broken : [];
    const activeCap = cx.filter((r) => !r.status || r.status === "active").reduce((a, r) => a + num(r.amount_m), 0);
    const fines = sx.filter((x) => !seasonNow || num(x.season) === seasonNow).reduce((a, x) => a + num(x.fine_m), 0);
    const pickCount = px.filter((p) => s(p.status) !== "no_pick_owned").length;
    return { sx, px, cx, ox, bx, activeCap, fines, pickCount, empty: !sx.length && !px.length && !cx.length && !ox.length && !bx.length };
  };
  const club = build("club");
  const league = build("league");
  const view = scope === "club" ? club : league;

  const clubCount = club.sx.length + club.ox.length + club.bx.length;
  const leagueCount = league.sx.length + league.ox.length;

  const linkedStory = (eventId) =>
    eventId ? stories.find((st) => s(st?.causeEventId) === s(eventId) || s(st?.raw?.cause_event_id) === s(eventId)) : null;

  const renderSanctions = (rows, sc) =>
    rows.map((x) => {
      const story = linkedStory(x.source_event_id);
      const effects = sanctionEffects(x);
      const expired = seasonNow && num(x.season) && num(x.season) < seasonNow;
      return (
        <article key={s(x.id)} className={`bbd-sanction${expired ? " is-old" : ""}`}>
          <TeamMark row={x} teamsById={teamsById} size={54} />
          <div className="bbd-sanction__main">
            <span className="bbd-label">{sc === "club" ? "League ruling" : teamName(x, teamsById)} · {seasonLabel(x.season) || "this season"}</span>
            <h4 className="bbd-h bbd-h--sm">{EVENT_COPY[s(x.event_type)] || label(x.event_type || "League sanction")}</h4>
            {effects.length ? (
              <ul className="bbd-effects">
                {effects.map((e, i) => (
                  <li key={i} className={`is-${e.tone}`}>{e.text}</li>
                ))}
              </ul>
            ) : (
              <p className="bbd-note">Ruling recorded. No cap, draft or cash penalty attached.</p>
            )}
            {story && onOpenStory ? (
              <button type="button" className="bbd-link" onClick={() => onOpenStory(story.id)}>Read the story</button>
            ) : null}
          </div>
          <span className={`bbd-stamp${expired ? " is-done" : ""}`}>{expired ? "Served" : "In effect"}</span>
        </article>
      );
    });

  const renderOut = (rows, sc) =>
    rows.map((r) => {
      const left = [num(r.games_remaining) > 0 ? `${num(r.games_remaining)} GP` : "", num(r.days_remaining) > 0 ? `${num(r.days_remaining)} days` : ""]
        .filter(Boolean)
        .join(" · ");
      const portrait = ensurePlayerHeadshotFields({ ...r, id: r.player_id || r.pid, name: r.player_name });
      return (
        <div key={r.pid} className="bbd-out">
          <span className="bbd-face">
            <PlayerHeadshot player={portrait} size="sm" preferPhoto showFlag={false} />
          </span>
          <div className="bbd-out__who">
            <b>{s(r.player_name) || "Unnamed player"}</b>
            <small>
              {STATUS_COPY[s(r.status)] || label(r.status || "unavailable")}
              {r.reason_public || r.reason ? ` · ${s(r.reason_public || r.reason).replace(/_/g, " ")}` : ""}
            </small>
            {sc === "league" ? (
              <span className="bbd-out__team">
                <TeamMark row={r} teamsById={teamsById} size={20} /> {teamName(r, teamsById)}
              </span>
            ) : null}
          </div>
          <span className="bbd-out__left">{left || "Out"}</span>
        </div>
      );
    });

  const renderPicks = (rows, sc) =>
    rows.map((p, i) => {
      const year = s(p.applied_year || p.draft_year);
      const lost = s(p.status) !== "no_pick_owned";
      return (
        <div key={s(p.applied_pick_id || `${p.team_id}-${p.draft_year}-${p.round}-${i}`)} className={`bbd-pick${lost ? "" : " is-owed"}`}>
          <TeamMark row={p} teamsById={teamsById} size={34} />
          <div>
            <b>{year} {ordinal(p.round)}-round pick</b>
            <small>{sc === "league" ? `${teamName(p, teamsById)} · ` : ""}{lost ? "Forfeited, skipped in the draft order" : "Owed, no pick left to strip"}</small>
          </div>
          <span className="bbd-xmark" aria-hidden>{lost ? "✕" : "…"}</span>
        </div>
      );
    });

  const renderCaps = (rows, sc) =>
    rows.map((r, i) => (
      <div key={s(r.sanction_id || `${r.team_id}-${i}`)} className={`bbd-capline${r.status && r.status !== "active" ? " is-old" : ""}`}>
        <TeamMark row={r} teamsById={teamsById} size={30} />
        <span>
          <b>{sc === "league" ? teamName(r, teamsById) : r.season_label ? `${r.season_label} cap` : "Cap penalty"}</b>
          <small>{r.status === "expired" ? "Expired" : r.status === "pending" ? "Starts next season" : "Counts against the cap now"}</small>
        </span>
        <em>{money(r.amount_m) || "—"}</em>
      </div>
    ));

  const sections = (v, sc) => (
    <>
      {v.sx.length ? (
        <Section tone="bad" heading={sc === "club" ? "League rulings" : "Sanctions"} count={v.sx.length} wide>
          <div className="bbd-sanctions">{renderSanctions(v.sx, sc)}</div>
        </Section>
      ) : null}
      {v.ox.length ? (
        <Section tone="bad" heading="Out of the lineup" count={v.ox.length}>
          <div className="bbd-outs">{renderOut(v.ox, sc)}</div>
        </Section>
      ) : null}
      {v.px.length ? (
        <Section tone="warn" heading="Draft table" count={v.pickCount}>
          <div className="bbd-picks">{renderPicks(v.px, sc)}</div>
        </Section>
      ) : null}
      {v.cx.length ? (
        <Section tone="warn" heading="Cap ledger" count={v.cx.length}>
          <div className="bbd-caps">{renderCaps(v.cx, sc)}</div>
        </Section>
      ) : null}
      {v.bx.length ? (
        <Section tone="warn" heading="Broken promises" count={v.bx.length}>
          <ul className="bbd-promises">
            {v.bx.map((p, i) => (
              <li key={s(p.id || i)}>
                <b>{s(p.player_name || "Player")}</b>
                <small>{s(p.description || label(p.type || "promise"))}</small>
              </li>
            ))}
          </ul>
          <p className="bbd-note">Trust and morale took the hit. A "repair relationship" meeting can win some of it back.</p>
        </Section>
      ) : null}
    </>
  );

  return (
    <div className="bbd-board">
      <header className="ccx-head bbd-top">
        <div className="bbd-top__copy">
          <span className="bbd-label">League office · discipline desk</span>
          <h2 className="ccx-title bbd-pagetitle">Fallout</h2>
          <p className="bbd-lede">
            The bill for what happened off the ice. Cap penalties count against this season's cap, forfeited picks are struck from the draft,
            fines come out of club finances and suspended players can't dress.
          </p>
          <div className="bbd-scope" role="tablist" aria-label="Fallout scope">
            {[["club", "Your club", clubCount], ["league", "Around the league", leagueCount]].map(([id, text, n]) => (
              <button key={id} type="button" role="tab" aria-selected={scope === id} className={scope === id ? "is-active" : ""} onClick={() => setScope(id)}>
                {text}
                {n ? <span className="bbd-count">{n}</span> : null}
              </button>
            ))}
          </div>
        </div>
        <div className="bbd-tallies">
          <Tally value={view.activeCap > 0 ? `$${view.activeCap.toFixed(2)}M` : "$0"} text={`Cap penalty${seasonNowLabel ? ` · ${seasonNowLabel}` : ""}`} tone={view.activeCap > 0 ? "bad" : ""} />
          <Tally value={view.pickCount} text="Picks forfeited" tone={view.pickCount ? "bad" : ""} />
          <Tally value={view.fines > 0 ? `$${view.fines.toFixed(2)}M` : "$0"} text="Fines this season" tone={view.fines > 0 ? "warn" : ""} />
          <Tally value={view.ox.length} text="Players out" tone={view.ox.length ? "warn" : ""} />
          {budget.target ? <Tally value={`${num(budget.generated)}/${num(budget.target)}`} text="Major incidents league-wide" /> : null}
        </div>
      </header>

      {view.empty ? (
        <div className="bbd-grid">
          <section className="bbd-clean is-wide">
            <span className="bbd-sticker">Clean sheet</span>
            <div>
              <h3 className="bbd-h">{scope === "club" ? "Nothing hanging over your club" : "The league office is quiet"}</h3>
              <p>
                {scope === "club"
                  ? "No fines, cap penalties, forfeited picks, suspensions or broken promises right now. Keep it that way."
                  : "No other club is serving a sanction at the moment."}
              </p>
            </div>
          </section>
          {scope === "club" && !league.empty ? (
            <>
              <p className="bbd-divider is-wide"><span>Meanwhile, around the league</span></p>
              {sections(league, "league")}
            </>
          ) : null}
        </div>
      ) : (
        <div className="bbd-grid">{sections(view, scope)}</div>
      )}
    </div>
  );
}
