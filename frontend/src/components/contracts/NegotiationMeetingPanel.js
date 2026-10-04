import React from "react";
import { getNegotiationMeetings, runNegotiationMeeting } from "../../services/franchiseService";

/**
 * "Meet before you offer" — recruiting pitch (free agents), agent meeting (everyone),
 * hometown-discount ask (your own expiring players). Every result feeds the real
 * contract engine, so the Deal interest meter moves after each meeting.
 */
export default function NegotiationMeetingPanel({ playerId, onChanged, compact = false }) {
  const [data, setData] = React.useState(null);
  const [busy, setBusy] = React.useState("");
  const [flash, setFlash] = React.useState(null);
  const [open, setOpen] = React.useState(!compact);

  const load = React.useCallback(async () => {
    if (!playerId) return;
    try {
      const res = await getNegotiationMeetings(playerId);
      setData(res && res.ok ? res : null);
    } catch {
      setData(null);
    }
  }, [playerId]);

  React.useEffect(() => {
    setFlash(null);
    setData(null);
    load();
  }, [load]);

  const run = async (kind, option) => {
    if (busy) return;
    setBusy(`${kind}:${option}`);
    try {
      const res = await runNegotiationMeeting(playerId, kind, option);
      if (res?.ok) {
        setFlash({ tone: toneFor(res), text: res.message });
        if (res.options) setData(res.options);
        else await load();
        if (typeof onChanged === "function") onChanged(res);
      } else {
        setFlash({ tone: "warn", text: res?.reason || "That meeting didn't happen." });
      }
    } catch (err) {
      setFlash({ tone: "warn", text: err?.message || "That meeting didn't happen." });
    } finally {
      setBusy("");
    }
  };

  if (!data) return null;
  const agent = data.agent || {};
  const pitch = data.pitch;
  const agentMeet = data.agent_meeting || {};
  const hometown = data.hometown;
  const bonus = Number(data.interest_bonus) || 0;

  return (
    <section className="nmp">
      <style>{CSS}</style>
      <button type="button" className="nmp__toggle" onClick={() => setOpen((v) => !v)}>
        <span>Meet before you offer</span>
        <em>
          {bonus ? `${bonus > 0 ? "+" : ""}${bonus.toFixed(1)} interest from meetings` : "Pitch · agent · discounts"}
        </em>
        <i aria-hidden>{open ? "▾" : "▸"}</i>
      </button>
      {open ? (
        <div className="nmp__body">
          <div className="nmp__read">
            <div>
              <span className="nmp__label">What he cares about</span>
              <strong>{(data.priorities || []).join(" · ") || "—"}</strong>
            </div>
            {agent.name ? (
              <div>
                <span className="nmp__label">Agent</span>
                <strong>
                  {agent.name}
                  {agent.style_label ? <small> · {agent.style_label}</small> : null}
                </strong>
                {agent.trust != null ? (
                  <div className="nmp__trust" title="Agent's trust in you (carries over to all his clients)">
                    <i style={{ width: `${Math.max(3, Math.min(100, Number(agent.trust)))}%` }} />
                    <b>{agent.trust}</b>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          {flash ? <p className={`nmp__flash is-${flash.tone}`}>{flash.text}</p> : null}

          {pitch ? (
            <div className="nmp__block">
              <span className="nmp__label">Recruiting pitch · one shot</span>
              {pitch.used ? (
                <p className="nmp__result">
                  {pitch.result?.line}{" "}
                  <b className={Number(pitch.result?.interest_delta) >= 0 ? "is-pos" : "is-neg"}>
                    {fmtDelta(pitch.result?.interest_delta)} interest
                  </b>
                </p>
              ) : (
                <div className="nmp__grid">
                  {(pitch.options || []).map((o) => (
                    <button
                      key={o.id}
                      type="button"
                      className="nmp__opt"
                      disabled={Boolean(busy)}
                      onClick={() => run("pitch", o.id)}
                    >
                      <strong>{o.label}</strong>
                      <span>{o.fact}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : null}

          <div className="nmp__block">
            <span className="nmp__label">Meet his agent · once per window</span>
            {agentMeet.used ? (
              <p className="nmp__result">
                {agentMeet.result?.line}{" "}
                <b className={Number(agentMeet.result?.interest_delta) >= 0 ? "is-pos" : "is-neg"}>
                  {fmtDelta(agentMeet.result?.interest_delta)} interest
                </b>
              </p>
            ) : (
              <div className="nmp__grid">
                {(agentMeet.options || []).map((o) => (
                  <button
                    key={o.id}
                    type="button"
                    className="nmp__opt"
                    disabled={Boolean(busy)}
                    onClick={() => run("agent", o.id)}
                  >
                    <strong>{o.label}</strong>
                    <span>{o.line}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {hometown ? (
            <div className="nmp__block">
              <span className="nmp__label">Hometown discount</span>
              {hometown.active_discount_pct ? (
                <p className="nmp__result is-pos">
                  Agreed: about {Math.round(Number(hometown.active_discount_pct))}% under market on his next deal with you.
                </p>
              ) : hometown.used ? (
                <p className="nmp__result">
                  {hometown.result?.line}
                  {hometown.result?.chance != null ? ` (odds were ${hometown.result.chance}%)` : ""}
                </p>
              ) : hometown.eligible ? (
                <button
                  type="button"
                  className="nmp__opt nmp__opt--wide"
                  disabled={Boolean(busy)}
                  onClick={() => run("hometown", "")}
                >
                  <strong>Ask him to take less to stay</strong>
                  <span>
                    Loyalty, trust in you and goodwill from meetings raise the odds. A no costs a little
                    morale.
                  </span>
                </button>
              ) : (
                <p className="nmp__muted">{hometown.reason}</p>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function fmtDelta(v) {
  const n = Number(v) || 0;
  return `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
}

function toneFor(res) {
  if (res.kind === "hometown") return res.accepted ? "good" : "warn";
  const d = Number(res.interest_delta) || 0;
  if (d >= 2) return "good";
  if (d <= -1) return "warn";
  return "neutral";
}

const CSS = `
.nmp { border: 1px solid rgba(115,229,241,.22); border-radius: 10px; margin: 10px 0; background: rgba(19,216,231,.03); }
.nmp__toggle { width: 100%; display: flex; align-items: center; gap: 10px; padding: 9px 12px; background: transparent;
  border: 0; color: inherit; cursor: pointer; text-align: left; }
.nmp__toggle span { font-size: 11px; font-weight: 900; letter-spacing: .1em; text-transform: uppercase; }
.nmp__toggle em { font-style: normal; font-size: 11px; color: #8fb4c4; margin-left: auto; }
.nmp__toggle i { font-style: normal; color: #8fb4c4; }
.nmp__body { padding: 0 12px 12px; display: grid; gap: 10px; }
.nmp__read { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.nmp__read strong { display: block; font-size: 12.5px; }
.nmp__read small { font-weight: 600; color: #8fb4c4; }
.nmp__label { display: block; font-size: 9.5px; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; color: #8fb4c4; margin-bottom: 4px; }
.nmp__trust { position: relative; height: 6px; border-radius: 3px; background: rgba(255,255,255,.08); margin-top: 6px; }
.nmp__trust i { position: absolute; inset: 0 auto 0 0; border-radius: 3px; background: #13d8e7; }
.nmp__trust b { position: absolute; right: 0; top: -15px; font-size: 10px; color: #8fb4c4; }
.nmp__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px; }
.nmp__opt { text-align: left; padding: 8px 10px; border-radius: 8px; cursor: pointer; color: inherit;
  border: 1px solid rgba(115,229,241,.22); background: rgba(255,255,255,.02); display: grid; gap: 3px; }
.nmp__opt:hover:not(:disabled) { border-color: #13d8e7; background: rgba(19,216,231,.08); }
.nmp__opt:disabled { opacity: .5; cursor: default; }
.nmp__opt strong { font-size: 12px; }
.nmp__opt span { font-size: 10.5px; color: #8fb4c4; line-height: 1.35; }
.nmp__opt--wide { width: 100%; }
.nmp__result { margin: 0; font-size: 12px; line-height: 1.45; }
.nmp__result b.is-pos, .nmp__result.is-pos { color: #3ccf8e; }
.nmp__result b.is-neg { color: #ff7a7a; }
.nmp__muted { margin: 0; font-size: 11.5px; color: #8fb4c4; }
.nmp__flash { margin: 0; font-size: 12px; padding: 7px 10px; border-radius: 6px; background: rgba(255,255,255,.04); }
.nmp__flash.is-good { border-left: 3px solid #3ccf8e; }
.nmp__flash.is-warn { border-left: 3px solid #e9a83c; }
.nmp__flash.is-neutral { border-left: 3px solid #8fb4c4; }
@media (max-width: 640px) { .nmp__read { grid-template-columns: 1fr; } }
`;
