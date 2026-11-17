import React from "react";
import { createPortal } from "react-dom";
import { getDisasterDraft, pickDisasterDraft } from "../../services/franchiseService";

/**
 * Emergency Rehabilitation Draft (backend: services/disaster_draft.py).
 * Shown over everything while your club owes picks after a team disaster.
 */
export default function DisasterDraftRoom({ onDone }) {
  const [data, setData] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [pos, setPos] = React.useState("ALL");
  const [error, setError] = React.useState("");

  const load = React.useCallback(() => {
    getDisasterDraft()
      .then(setData)
      .catch((e) => setError(String(e?.message || "Couldn't load the draft.")));
  }, []);
  React.useEffect(load, [load]);

  const pick = async (body) => {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await pickDisasterDraft(body);
      if (res?.ok === false && res?.reason) setError(res.reason);
      const next = res?.draft || res;
      setData(next);
      if (next && next.active === false) onDone?.();
    } catch (e) {
      setError(String(e?.message || "Pick failed."));
    } finally {
      setBusy(false);
    }
  };

  if (!data || data.active === false) return null;
  const pool = (data.pool || []).filter((p) => pos === "ALL" || (pos === "F" ? !["LD", "RD", "D", "G"].includes(p.position) : pos === "D" ? ["LD", "RD", "D"].includes(p.position) : p.position === "G"));

  const body = (
    <div className="ddraft" role="dialog" aria-modal="true" aria-label="Emergency draft">
      <header className="ddraft-head">
        <div>
          <span>NHL Emergency Rehabilitation Plan</span>
          <h1>Rebuilding the {data.team_name}</h1>
          <p>
            {data.incident ? `${data.incident[0].toUpperCase()}${data.incident.slice(1)}.` : ""} Lost: {(data.killed || []).join(", ")}
            {data.disabled?.length ? `; career-ending injuries: ${data.disabled.join(", ")}` : ""}.
          </p>
        </div>
        <div className="ddraft-count">
          <b>{data.picks_left}</b>
          <span>of {data.picks_total} picks left</span>
        </div>
      </header>
      <div className="ddraft-body">
        <aside className="ddraft-side">
          <p className="ddraft-rules">{data.rules}</p>
          {data.need_goalie ? <p className="ddraft-need">You lost a goalie. Take one.</p> : null}
          {data.bought?.length ? (
            <div>
              <span className="ddraft-lab">Bought from willing clubs</span>
              {data.bought.map((b) => (
                <p key={b.name}>{b.name} · {b.from}</p>
              ))}
            </div>
          ) : null}
          {data.picks?.length ? (
            <div>
              <span className="ddraft-lab">Your picks</span>
              {data.picks.map((p) => (
                <p key={p.player_id}>
                  <b>{p.name}</b> · {p.position} · {Math.round(p.ovr)} OVR · from {p.team}
                </p>
              ))}
            </div>
          ) : null}
          <button type="button" className="ddraft-auto" disabled={busy} onClick={() => pick({ auto: true })}>
            Let the staff finish the draft
          </button>
          {error ? <p className="ddraft-error">{error}</p> : null}
        </aside>
        <section className="ddraft-pool">
          <div className="ddraft-filters">
            {["ALL", "F", "D", "G"].map((k) => (
              <button key={k} type="button" className={pos === k ? "is-on" : ""} onClick={() => setPos(k)}>
                {k === "ALL" ? "Everyone" : k === "F" ? "Forwards" : k === "D" ? "Defence" : "Goalies"}
              </button>
            ))}
          </div>
          <table>
            <thead>
              <tr>
                <th>Player</th><th>Pos</th><th>Age</th><th>OVR</th><th>Cap hit</th><th>From</th><th />
              </tr>
            </thead>
            <tbody>
              {pool.map((p) => (
                <tr key={p.player_id}>
                  <td><b>{p.name}</b></td>
                  <td>{p.position}</td>
                  <td>{p.age || ""}</td>
                  <td>{Math.round(p.ovr)}</td>
                  <td>${Number(p.cap_hit_m || 0).toFixed(2)}M</td>
                  <td>{p.team}</td>
                  <td>
                    <button type="button" disabled={busy} onClick={() => pick({ player_id: p.player_id })}>
                      Select
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
  return typeof document !== "undefined" ? createPortal(body, document.body) : body;
}
