import React, { useEffect, useMemo, useState } from "react";
import PlayerHeadshot from "../../PlayerHeadshot";
import { getTeamLinesView } from "../../../services/franchiseService";
import "./TeamLinesViewer.css";

const arr = (v) => (Array.isArray(v) ? v : []);

function PlayerChip({ p }) {
  if (!p) return <div className="tlv-chip is-empty">Empty</div>;
  return (
    <div className={`tlv-chip${p.injured ? " is-injured" : ""}`}>
      <PlayerHeadshot player={{ ...p, id: p.id }} size="sm" showFlag={false} />
      <div className="tlv-chip__body">
        <strong title={p.name}>{p.name}</strong>
        <small>{[p.position, p.age ? `${p.age}y` : ""].filter(Boolean).join(" · ")}{p.injured ? " · injured" : ""}</small>
      </div>
      <b className="tlv-chip__ovr">{p.ovr}</b>
    </div>
  );
}

function Unit({ label, unit, slots }) {
  const players = arr(unit?.players);
  const capped = players.slice(0, slots);
  const padded = [...capped, ...Array(Math.max(0, slots - capped.length)).fill(null)];
  return (
    <div className="tlv-unit">
      <div className="tlv-unit__head">
        <span>{label}</span>
        {unit?.avg_ovr != null ? <em>avg {unit.avg_ovr}</em> : null}
      </div>
      <div className="tlv-unit__row">{padded.map((p, i) => <PlayerChip key={p?.id || `e${i}`} p={p} />)}</div>
    </div>
  );
}

/** Read-only scouting view of another club's lines (NHL or AHL). */
export default function TeamLinesViewer({ teamId, level: initialLevel = "nhl", teams: teamsProp, onChangeTeam, onBack }) {
  const [level, setLevel] = useState(initialLevel);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!teamId) return undefined;
    let active = true;
    setLoading(true);
    setError("");
    getTeamLinesView(teamId, level)
      .then((d) => { if (active) setData(d || null); })
      .catch((e) => { if (active) setError(e?.response?.data?.detail || "Couldn't load those lines."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [teamId, level]);

  const teams = useMemo(() => arr(data?.teams).length ? arr(data.teams) : arr(teamsProp), [data?.teams, teamsProp]);
  const isNhl = level === "nhl";

  return (
    <div className="tlv-root">
      <header className="tlv-head">
        <div className="tlv-head__title">
          <p className="tlv-kicker">Scouting view · read only</p>
          <h2>{data?.team_name || "Team"} {isNhl ? "lines" : `AHL lines${data?.affiliate_name ? ` · ${data.affiliate_name}` : ""}`}</h2>
          <div className="tlv-badges">
            {isNhl && data?.deployment_label ? <span className="tlv-badge">{data.deployment_label}</span> : null}
            {isNhl && data?.strength_rank ? <span className="tlv-badge">Team strength #{data.strength_rank} of {teams.length || 32}</span> : null}
          </div>
        </div>
        <div className="tlv-controls">
          <label>
            <span>Team</span>
            <select value={teamId} onChange={(e) => onChangeTeam?.(e.target.value)}>
              {teams.map((t) => (
                <option key={t.team_id} value={t.team_id}>{t.is_user ? `${t.name} (you)` : t.name}</option>
              ))}
            </select>
          </label>
          <div className="tlv-seg">
            {["nhl", "ahl"].map((lv) => (
              <button key={lv} type="button" className={level === lv ? "is-active" : ""} onClick={() => setLevel(lv)}>{lv.toUpperCase()}</button>
            ))}
          </div>
          <button type="button" className="tlv-back" onClick={onBack}>Back to my lines</button>
        </div>
      </header>

      {error ? <p className="tlv-error">{error}</p> : null}
      {loading && !data ? <p className="tlv-note">Loading lines…</p> : null}

      {data?.ok ? (
        <div className={`tlv-grid${loading ? " is-loading" : ""}`}>
          <section>
            <h3>Forwards</h3>
            {arr(data.forwards).map((u, i) => <Unit key={`f${i}`} label={`Line ${i + 1}`} unit={u} slots={3} />)}
          </section>
          <section>
            <h3>Defence</h3>
            {arr(data.defense).map((u, i) => <Unit key={`d${i}`} label={`Pair ${i + 1}`} unit={u} slots={2} />)}
            <h3>Goalies</h3>
            <div className="tlv-unit">
              <div className="tlv-unit__head"><span>Starter / Backup</span></div>
              <div className="tlv-unit__row">
                <PlayerChip p={data.goalies?.starter} />
                <PlayerChip p={data.goalies?.backup} />
              </div>
            </div>
          </section>
          {isNhl ? (
            <section>
              <h3>Special teams</h3>
              {arr(data.power_play).map((u, i) => <Unit key={`pp${i}`} label={`PP${i + 1}`} unit={u} slots={5} />)}
              {arr(data.penalty_kill).map((u, i) => <Unit key={`pk${i}`} label={`PK${i + 1}`} unit={u} slots={4} />)}
            </section>
          ) : null}
          <section>
            <h3>Extras / scratches</h3>
            {arr(data.extras).length ? (
              <div className="tlv-unit__row tlv-unit__row--wrap">{arr(data.extras).map((p) => <PlayerChip key={p.id} p={p} />)}</div>
            ) : <p className="tlv-note">Everyone is dressed.</p>}
          </section>
        </div>
      ) : null}
    </div>
  );
}
