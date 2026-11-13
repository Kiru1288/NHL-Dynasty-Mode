import React from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

/* Small shared bits for the four Bench Buzz feature desks (Worker E, prefix bbe-). */

export const arr = (v) => (Array.isArray(v) ? v : []);
export const s = (v, f = "") => (v == null || v === "" ? f : String(v));
export const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : null);

export function teamLogo(teamId, abbr, name) {
  return (
    resolveFranchiseTeamLogo(
      { team_id: teamId, abbrev: abbr, team_abbrev: abbr, abbreviation: abbr, team_name: name, name },
      name || abbr
    ) || ""
  );
}

export function TeamMark({ teamId, abbr, name, className = "" }) {
  const src = teamLogo(teamId, abbr, name);
  return (
    <span className={`bbe-logo ${className}`} title={name || abbr}>
      {src ? <img src={src} alt="" /> : <b>{s(abbr, "—").slice(0, 3)}</b>}
    </span>
  );
}

export function Face({ row, px = 96, className = "" }) {
  const pid = s(row?.player_id);
  const player = ensurePlayerHeadshotFields({
    ...(row || {}),
    id: pid,
    player_id: pid,
    name: s(row?.name || row?.player_name, "Player"),
    position: row?.position,
    team_abbrev: row?.team_abbr,
  });
  return (
    <span className={`bbe-face ${className}`} style={{ "--bbe-face": `${px}px` }}>
      <PlayerHeadshot player={player} size="xl" preferPhoto showFlag={false} style={{ "--size": `${px}px` }} />
    </span>
  );
}

export function DeskHead({ kicker, title, lede, children }) {
  return (
    <header className="ccx-head bbe-head">
      <div>
        <span className="ccx-kicker">{kicker}</span>
        <h2 className="ccx-title">{title}</h2>
        {lede ? <p className="bbe-lede">{lede}</p> : null}
      </div>
      {children ? <div className="bbe-head__side">{children}</div> : null}
    </header>
  );
}

export function DeskMissing({ kicker, title }) {
  return (
    <section className="bbe-desk">
      <DeskHead kicker={kicker} title={title} />
      <div className="bbe-empty">
        <strong>This desk needs a newer game server.</strong>
        <p>The save loaded fine, but the server running it doesn't build this page yet. Restart the backend and reopen the Bench Buzz.</p>
      </div>
    </section>
  );
}
