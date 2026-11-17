import React from "react";
import { createPortal } from "react-dom";
import { executeCapPlot, getCapPlotOptions } from "../../services/franchiseService";

/**
 * The cap-dump plot planning board (backend: services/cap_plot.py).
 * Six choices, a live read on success and suspicion, a two-step confirm, then the outcome.
 */
export default function CapPlotRoom({ playerId, onClose, onDone }) {
  const [data, setData] = React.useState(null);
  const [plan, setPlan] = React.useState({});
  const [armed, setArmed] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let off = false;
    getCapPlotOptions(playerId)
      .then((d) => !off && setData(d))
      .catch((e) => !off && setError(String(e?.message || "The burner phone isn't answering.")));
    return () => {
      off = true;
    };
  }, [playerId]);

  React.useEffect(() => {
    const onKey = (e) => e.key === "Escape" && !busy && onClose?.();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const stages = data?.stages || [];
  const complete = stages.length > 0 && stages.every((s) => plan[s.id]);
  let success = Number(data?.base_success || 0.62);
  let suspicion = Number(data?.base_suspicion || 0.24) + Number(data?.motive || 0);
  stages.forEach((s) => {
    const o = s.options.find((x) => x.id === plan[s.id]);
    if (o) {
      success += o.success;
      suspicion += o.suspicion;
    }
  });
  if (plan.accomplice === "fixer" && plan.tracks === "burner") suspicion -= 0.04;
  if (plan.accomplice === "equipment" && plan.tracks === "none") suspicion += 0.1;
  if (plan.cover === "rumours" && plan.alibi === "home") suspicion += 0.08;
  success = Math.max(0.15, Math.min(0.92, success));
  suspicion = Math.max(Number(data?.min_suspicion || 0.14), Math.min(0.95, suspicion));

  const go = async () => {
    if (!armed) {
      setArmed(true);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await executeCapPlot({ player_id: playerId, plan, confirm: true });
      if (res?.ok) {
        setResult(res);
        onDone?.(res);
      } else {
        setError(res?.reason || "It fell apart before it started.");
      }
    } catch (e) {
      setError(String(e?.message || "Something went wrong."));
    } finally {
      setBusy(false);
    }
  };

  const body = (
    <div className="plot" role="dialog" aria-modal="true" aria-label="The plan">
      <header className="plot-bar">
        <div>
          <span>Off the books · strictly confidential</span>
          <strong>{data?.player_name ? `The ${data.player_name} problem` : "The plan"}</strong>
        </div>
        <button type="button" className="plot-close" disabled={busy} onClick={onClose}>
          {result ? "Leave" : "Forget I said anything"} ✕
        </button>
      </header>
      {error ? <p className="plot-error">{error}</p> : null}
      {!data ? (
        <p className="plot-note">Opening the folder…</p>
      ) : data.ok === false ? (
        <p className="plot-note">{data.reason || "Not possible right now."}</p>
      ) : result ? (
        <div className="plot-result">
          <h2>{result.outcome === "done" ? "It's done." : "It didn't work."}</h2>
          <p>{result.message}</p>
          {result.cap_freed_m ? <p className="plot-big">${Number(result.cap_freed_m).toFixed(2)}M off your cap</p> : null}
          {result.investigation ? (
            <p className="plot-note">
              {result.investigation.detective} is on the case. Heat: {result.investigation.heat_pct}%. Keep your head down until the inquiry closes.
            </p>
          ) : null}
        </div>
      ) : (
        <div className="plot-board">
          <section className="plot-file">
            <span className="plot-lab">The target</span>
            <h2>{data.player_name}</h2>
            <p>
              ${Number(data.cap_hit_m).toFixed(2)}M a year · {data.years_left} more {data.years_left === 1 ? "year" : "years"}
            </p>
            <p className="plot-note">
              Motive on paper: {Math.round(Number(data.motive) * 100)}%. The more he's overpaid, the more obvious you look.
            </p>
            {data.blocked ? <p className="plot-blocked">{data.blocked}</p> : null}
            {data.investigation ? (
              <p className="plot-blocked">
                Open inquiry ({data.investigation.detective}) · heat {data.investigation.heat_pct}% · {data.investigation.progress_pct}% through
              </p>
            ) : null}
            <div className="plot-meters">
              <div>
                <span>Chance it works</span>
                <i><b className="is-go" style={{ width: `${Math.round(success * 100)}%` }} /></i>
                <strong>{Math.round(success * 100)}%</strong>
              </div>
              <div>
                <span>Chance you get caught</span>
                <i><b className="is-risk" style={{ width: `${Math.round(suspicion * 100)}%` }} /></i>
                <strong>{Math.round(suspicion * 100)}%</strong>
              </div>
            </div>
            <p className="plot-warn">If you're caught, the GM is arrested and this save is over. You'll have to start a new franchise.</p>
            <button type="button" className={`plot-go${armed ? " is-armed" : ""}`} disabled={!complete || busy || Boolean(data.blocked)} onClick={go}>
              {busy ? "…" : armed ? "Last chance. Go through with it." : complete ? "Go through with it" : "Finish the plan first"}
            </button>
            {armed ? (
              <button type="button" className="plot-abort" onClick={() => setArmed(false)}>
                Wait, no
              </button>
            ) : null}
          </section>
          <section className="plot-steps">
            {stages.map((s, i) => (
              <div key={s.id} className="plot-step">
                <span className="plot-lab">
                  {i + 1} · {s.title}
                </span>
                <div className="plot-opts">
                  {s.options.map((o) => (
                    <button
                      key={o.id}
                      type="button"
                      className={plan[s.id] === o.id ? "is-on" : ""}
                      onClick={() => {
                        setArmed(false);
                        setPlan((p) => ({ ...p, [s.id]: o.id }));
                      }}
                    >
                      <b>{o.label}</b>
                      <small>{o.blurb}</small>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </section>
        </div>
      )}
    </div>
  );
  return typeof document !== "undefined" ? createPortal(body, document.body) : body;
}

/** Shown everywhere once the GM is arrested: the save is over. */
export function CapPlotGameOver({ info, onNewGame }) {
  if (!info) return null;
  const body = (
    <div className="plot plot--over" role="alertdialog" aria-modal="true" aria-label="Game over">
      <div className="plot-over">
        <span className="plot-lab">Breaking news</span>
        <h1>{info.headline || "GM arrested"}</h1>
        <p>{info.summary}</p>
        <button type="button" className="plot-go" onClick={onNewGame}>
          Start a new franchise
        </button>
      </div>
    </div>
  );
  return typeof document !== "undefined" ? createPortal(body, document.body) : body;
}
