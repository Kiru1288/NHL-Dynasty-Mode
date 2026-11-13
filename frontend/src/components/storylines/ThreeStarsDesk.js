import React, { useState } from "react";
import { arr, s, Face, TeamMark, DeskHead, DeskMissing } from "./benchBuzzShared";

/* Three Stars of the week: league + your club, from the box-score book over the
   last 7 sim days, plus last night's stars. Podium order 2-1-3. */

const STAR_WORD = ["First", "Second", "Third"];

function Stars({ n }) {
  return (
    <span className="bbe-stars" aria-label={`${STAR_WORD[n - 1] || ""} star`}>
      {"★".repeat(4 - n)}
    </span>
  );
}

function PodiumStep({ row }) {
  const rank = Number(row.rank) || 1;
  return (
    <article className={`bbe-step is-r${rank}`}>
      <div className="bbe-step__sticker">
        <Stars n={rank} />
        <b>{STAR_WORD[rank - 1]} star</b>
      </div>
      <div className="bbe-step__face">
        <Face row={row} px={rank === 1 ? 168 : 132} />
        <TeamMark teamId={row.team_id} abbr={row.team_abbr} name={row.team_name} className="bbe-step__logo" />
      </div>
      <h3 className="bbe-step__name">{s(row.name, "Player")}</h3>
      <p className="bbe-step__who">{s(row.team_abbr)} · {s(row.position)} · {row.gp} {row.is_goalie ? (row.gp === 1 ? "start" : "starts") : "GP"}</p>
      <dl className="bbe-statrow">
        {arr(row.stats).map((st) => (
          <div key={st.k}>
            <dt>{st.k}</dt>
            <dd>{s(st.v, "0")}</dd>
          </div>
        ))}
      </dl>
      <p className="bbe-step__blurb">{s(row.blurb)}</p>
      <div className="bbe-step__base" aria-hidden>{rank}</div>
    </article>
  );
}

function MiniStar({ row, extra }) {
  return (
    <li className="bbe-mini">
      <span className="bbe-mini__rank"><Stars n={Number(row.rank) || 1} /></span>
      <Face row={row} px={64} />
      <span className="bbe-mini__body">
        <b>{s(row.name, "Player")}</b>
        <i>{s(row.team_abbr)} · {s(row.line)}</i>
        {extra ? <em>{extra}</em> : null}
      </span>
      <TeamMark teamId={row.team_id} abbr={row.team_abbr} name={row.team_name} />
    </li>
  );
}

export default function ThreeStarsDesk({ features }) {
  const [view, setView] = useState("league");
  const data = features?.three_stars;
  if (!data) return <DeskMissing kicker="Weekly honours" title="Three Stars" />;
  if (data.error) {
    return (
      <section className="bbe-desk">
        <DeskHead kicker="Weekly honours" title="Three Stars" />
        <div className="bbe-empty"><strong>The stars desk hit a snag building this week.</strong><p>Advance a day and it rebuilds from the box scores.</p></div>
      </section>
    );
  }
  if (!data.available) {
    return (
      <section className="bbe-desk">
        <DeskHead kicker="Weekly honours" title="Three Stars" />
        <div className="bbe-empty"><strong>No stars yet.</strong><p>{s(data.note)}</p></div>
      </section>
    );
  }
  const rows = view === "club" ? arr(data.club) : arr(data.league);
  const other = view === "club" ? arr(data.league) : arr(data.club);
  const podium = [rows[1], rows[0], rows[2]].filter(Boolean);
  const night = data.last_night || {};
  return (
    <section className="bbe-desk bbe-stars-desk">
      <DeskHead
        kicker={`Weekly honours · ${s(data.window_label)}`}
        title="Three Stars of the week"
        lede={`Picked from every box score in the last ${data.window_days || 7} days (${data.games_in_window || 0} games): points, goals, game-winners, and for goalies, wins, save percentage and shutouts.`}
      >
        <div className="ccx-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={view === "league"} className={`ccx-tab ${view === "league" ? "is-active" : ""}`} onClick={() => setView("league")}>League</button>
          <button type="button" role="tab" aria-selected={view === "club"} className={`ccx-tab ${view === "club" ? "is-active" : ""}`} onClick={() => setView("club")}>{s(data.club_name, "Your club")}</button>
        </div>
      </DeskHead>

      <div className="bbe-stars-layout">
        <div className="bbe-podium">
          {podium.length ? podium.map((r) => <PodiumStep key={`${view}-${r.player_id}`} row={r} />) : (
            <div className="bbe-empty"><strong>Nobody qualified.</strong><p>{view === "club" ? "Your club hasn't played in the last week." : "No games in the last week."}</p></div>
          )}
        </div>

        <aside className="bbe-stars-rail">
          <div className="bbe-sheet">
            <div className="bbe-sheet__top">
              <span className="ccx-kicker">Last night · {s(night.label)}</span>
              <span className="ccx-stamp">{night.games || 0} {night.games === 1 ? "game" : "games"}</span>
            </div>
            <ol className="bbe-minilist">
              {arr(night.stars).map((r) => <MiniStar key={`n-${r.player_id}`} row={r} extra={r.game} />)}
              {!arr(night.stars).length ? <li className="bbe-mini is-empty">No box scores from last night.</li> : null}
            </ol>
          </div>
          <div className="bbe-sheet">
            <div className="bbe-sheet__top">
              <span className="ccx-kicker">{view === "club" ? "League three stars" : `${s(data.club_name, "Your club")} three stars`}</span>
            </div>
            <ol className="bbe-minilist">
              {other.map((r) => <MiniStar key={`o-${r.player_id}`} row={r} />)}
              {!other.length ? <li className="bbe-mini is-empty">Nobody from your club played this week.</li> : null}
            </ol>
          </div>
        </aside>
      </div>
    </section>
  );
}
