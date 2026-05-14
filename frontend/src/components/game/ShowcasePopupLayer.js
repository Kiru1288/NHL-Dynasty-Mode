import React, { useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";

function ShowcaseGameBody({ pop }) {
  const h = pop.home || {};
  const a = pop.away || {};
  const ot = pop.overtime ? " · OT" : "";
  return (
    <div className="showcase-popup__game">
      <div className="showcase-popup__scoreline">
        <span className="showcase-popup__mono">{h.abbr || "?"}</span>
        <span className="showcase-popup__score">
          {pop.home_goals}–{pop.away_goals}
          {ot}
        </span>
        <span className="showcase-popup__mono">{a.abbr || "?"}</span>
      </div>
      <div className="showcase-popup__sub">
        {h.name || ""} vs {a.name || ""}
      </div>
    </div>
  );
}

function WjcBody({ pop }) {
  const [openRr, setOpenRr] = useState(false);
  const standings = pop.standings || [];
  const medals = pop.medal_labels || {};
  const po = pop.playoffs || {};
  const prospects = pop.user_prospects || [];
  const complete = Boolean(pop.medals_final || pop.wjc_phase === "complete");
  const dayNum = pop.wjc_day;
  const dayTot = pop.wjc_days_total;
  const calIso = pop.calendar_iso;

  const rrGames = useMemo(() => (pop.round_robin_games || []).slice(), [pop]);

  return (
    <div className="showcase-popup__wjc">
      {dayNum && dayTot ? (
        <p className="showcase-popup__wjc-banner">
          U20 World Juniors — day {dayNum} of {dayTot}
          {calIso ? <span className="showcase-popup__wjc-iso"> · {calIso}</span> : null}
          {!complete ? <span className="showcase-popup__wjc-live"> · tournament in progress</span> : null}
        </p>
      ) : null}

      {complete ? (
        <div className="showcase-popup__medals">
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--gold">Gold</span>{" "}
            {medals.gold || "—"}
          </div>
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--silver">Silver</span>{" "}
            {medals.silver || "—"}
          </div>
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--bronze">Bronze</span>{" "}
            {medals.bronze || "—"}
          </div>
        </div>
      ) : (
        <p className="showcase-popup__muted">Medals are awarded after the gold medal game (Jan 5).</p>
      )}

      <h4 className="showcase-popup__h">Round robin — standings to date</h4>
      <div className="showcase-popup__table-wrap">
        <table className="showcase-popup__table">
          <thead>
            <tr>
              <th>#</th>
              <th>Country</th>
              <th>GP</th>
              <th>W</th>
              <th>L</th>
              <th>GF</th>
              <th>GA</th>
              <th>Pts</th>
            </tr>
          </thead>
          <tbody>
            {standings.map((row) => (
              <tr key={row.code}>
                <td>{row.place}</td>
                <td>
                  <strong>{row.code}</strong> {row.label}
                </td>
                <td>{row.gp}</td>
                <td>{row.w}</td>
                <td>{row.l}</td>
                <td>{row.gf}</td>
                <td>{row.ga}</td>
                <td>{row.pts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="showcase-popup__h">Knockout bracket</h4>
      {(po.quarterfinals || []).length ? (
        <ul className="showcase-popup__ko">
          {(po.quarterfinals || []).map((g, i) => (
            <li key={`qf-${i}`}>
              QF: {g.home_label || g.home} {g.home_goals}–{g.away_goals} {g.away_label || g.away} →{" "}
              <strong>{g.winner_label || g.winner}</strong>
            </li>
          ))}
          {(po.semifinals || []).map((g, i) => (
            <li key={`sf-${i}`}>
              SF: {g.home_label || g.home} {g.home_goals}–{g.away_goals} {g.away_label || g.away} →{" "}
              <strong>{g.winner_label || g.winner}</strong>
            </li>
          ))}
          {po.bronze ? (
            <li>
              Bronze: {po.bronze.home_label || po.bronze.home} {po.bronze.home_goals}–{po.bronze.away_goals}{" "}
              {po.bronze.away_label || po.bronze.away} → <strong>{po.bronze.winner_label || po.bronze.winner}</strong>
            </li>
          ) : null}
          {po.gold ? (
            <li>
              Final: {po.gold.home_label || po.gold.home} {po.gold.home_goals}–{po.gold.away_goals}{" "}
              {po.gold.away_label || po.gold.away} → <strong>{po.gold.winner_label || po.gold.winner}</strong>
            </li>
          ) : null}
        </ul>
      ) : (
        <p className="showcase-popup__muted">Knockout rounds unlock as the calendar moves through the tournament.</p>
      )}

      <button type="button" className="showcase-popup__toggle" onClick={() => setOpenRr((v) => !v)}>
        {openRr ? "Hide" : "Show"} all round-robin scores ({rrGames.length})
      </button>
      {openRr ? (
        <ul className="showcase-popup__rr">
          {rrGames.map((g, i) => (
            <li key={`rr-${i}`}>
              {g.home_label || g.home} {g.home_goals}–{g.away_goals} {g.away_label || g.away}
            </li>
          ))}
        </ul>
      ) : null}

      {prospects.length ? (
        <>
          <h4 className="showcase-popup__h">Your club — U20 tied to national teams</h4>
          <ul className="showcase-popup__prospects">
            {prospects.map((p) => (
              <li key={p.player_id || p.name}>
                <strong>{p.name}</strong> ({p.age}) — {p.roster ? `${p.roster} · ` : ""}
                {p.wjc_country_label || p.wjc_country}{" "}
                {p.made_wjc_team ? <span className="showcase-popup__tag showcase-popup__tag--yes">Roster</span> : null}
                {!p.made_wjc_team ? (
                  <span className="showcase-popup__tag showcase-popup__tag--no">Released</span>
                ) : null}
                <div className="showcase-popup__note">{p.note}</div>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="showcase-popup__muted">
          No U20 players on your AHL affiliate (and no NHL U20s loaned to a national team) for this recap.
        </p>
      )}
    </div>
  );
}

function InjuryBody({ pop }) {
  const tier = String(pop.tier || "").toLowerCase();
  const inj = pop.injury_type ? String(pop.injury_type) : "";
  return (
    <div className="showcase-popup__wjc">
      <p className="showcase-popup__wjc-banner">Medical Update</p>
      <h3 className="showcase-popup__h" style={{ marginTop: 0 }}>
        {pop.headline || "Player injured"}
      </h3>
      <p className="showcase-popup__muted" style={{ fontSize: 14, lineHeight: 1.45 }}>
        {pop.player_name || "A player"} from <strong>{pop.team_abbrev || "the club"}</strong> is expected to miss{" "}
        <strong>{pop.games != null ? pop.games : "multiple"}</strong> games
        {inj ? ` (${inj})` : ""}.
      </p>
      <div className="showcase-popup__columns" style={{ marginTop: 12 }}>
        <div>
          <div className="showcase-popup__muted" style={{ fontSize: 11, textTransform: "uppercase" }}>
            Team
          </div>
          <strong>{pop.team_abbrev || "—"}</strong>
        </div>
        <div>
          <div className="showcase-popup__muted" style={{ fontSize: 11, textTransform: "uppercase" }}>
            Severity
          </div>
          <strong style={{ textTransform: "capitalize" }}>{tier || "unknown"}</strong>
        </div>
        <div>
          <div className="showcase-popup__muted" style={{ fontSize: 11, textTransform: "uppercase" }}>
            Timeline
          </div>
          <strong>{pop.games != null ? `${pop.games} games` : "TBD"}</strong>
        </div>
      </div>
      {pop.game_day_injury ? (
        <p className="showcase-popup__muted" style={{ marginTop: 10, fontSize: 13 }}>
          Timing: this injury was applied after today&apos;s scheduled game on the calendar.
        </p>
      ) : null}
      {pop.requires_decision ? (
        <p className="showcase-popup__note" style={{ marginTop: 14 }}>
          Your hockey operations staff may need a response plan for this injury.
        </p>
      ) : null}
    </div>
  );
}

function AllStarBody({ pop }) {
  const ua = pop.user_allstars || [];
  return (
    <div className="showcase-popup__asg">
      <div className="showcase-popup__scoreline showcase-popup__scoreline--lg">
        <span>{pop.team_a_label}</span>
        <span className="showcase-popup__score">
          {pop.team_a_score}–{pop.team_b_score}
        </span>
        <span>{pop.team_b_label}</span>
      </div>
      {ua.length ? (
        <p className="showcase-popup__highlight">Your players selected: {ua.join(", ")}</p>
      ) : (
        <p className="showcase-popup__muted">No players from your NHL roster made this year&apos;s showcase.</p>
      )}
      <div className="showcase-popup__columns">
        <div>
          <h4 className="showcase-popup__h">{pop.team_a_label}</h4>
          <ul>
            {(pop.team_a || []).map((r) => (
              <li key={r.name} className={r.is_user ? "showcase-popup__you" : ""}>
                {r.name}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="showcase-popup__h">{pop.team_b_label}</h4>
          <ul>
            {(pop.team_b || []).map((r) => (
              <li key={r.name} className={r.is_user ? "showcase-popup__you" : ""}>
                {r.name}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export function ShowcasePopupLayer() {
  const { franchiseState, onDismissShowcasePopups } = useGameUI();
  const rawQueue = franchiseState?.pending_ui_popups || [];
  const hasPendingDecisions =
    Array.isArray(franchiseState?.pending_decisions) && franchiseState.pending_decisions.length > 0;
  // Show injury popups first when GM decisions are pending, but keep showcases/WJC/ASG in queue order after.
  const visiblePopups = hasPendingDecisions
    ? [...rawQueue.filter((p) => p && p.kind === "injury"), ...rawQueue.filter((p) => p && p.kind !== "injury")]
    : rawQueue;
  const first = visiblePopups[0];

  if (!first) return null;

  const kind = first.kind;

  return (
    <div className="showcase-popup">
      <div className="showcase-popup__backdrop" aria-hidden />
      <div className="showcase-popup__panel" role="dialog" aria-modal="true" aria-labelledby="showcase-popup-title">
        <header className="showcase-popup__head">
          <h2 id="showcase-popup-title" className="showcase-popup__title">
            {first.title || "League showcase"}
          </h2>
          {first.season_label ? <div className="showcase-popup__season">{first.season_label}</div> : null}
        </header>
        <div className="showcase-popup__body">
          {kind === "wjc_tournament" ? <WjcBody pop={first} /> : null}
          {kind === "showcase_game" ? <ShowcaseGameBody pop={first} /> : null}
          {kind === "allstar_game" ? <AllStarBody pop={first} /> : null}
          {kind === "injury" ? <InjuryBody pop={first} /> : null}
          {!["wjc_tournament", "showcase_game", "allstar_game", "injury"].includes(kind) ? (
            <pre className="showcase-popup__raw">{JSON.stringify(first, null, 2)}</pre>
          ) : null}
        </div>
        <footer className="showcase-popup__foot">
          {visiblePopups.length > 1 ? (
            <span className="showcase-popup__queue">+{visiblePopups.length - 1} more after this</span>
          ) : null}
          <button type="button" className="showcase-popup__btn" onClick={() => onDismissShowcasePopups([first.id])}>
            Continue
          </button>
        </footer>
      </div>
    </div>
  );
}
