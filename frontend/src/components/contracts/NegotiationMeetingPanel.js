import React from "react";
import { getNegotiationMeetings, runNegotiationMeeting } from "../../services/franchiseService";

/**
 * "Meet before you offer" — recruiting pitch (free agents), agent meeting (everyone),
 * hometown-discount ask (your own expiring players). Every result feeds the real
 * contract engine, so the Deal interest meter moves after each meeting.
 */
/**
 * Why he does or doesn't want to sign with this club: city, market, Canadian
 * market, roster and role. Effects are points on the Deal interest meter.
 */
export function InterestFactors({ factors, fit, title = "Fit with your club" }) {
  const list = Array.isArray(factors) ? factors : Array.isArray(fit?.factors) ? fit.factors : [];
  const score = Number(fit?.score);
  const premium = Number(fit?.fit_premium_pct);
  const tone = (n) => (Number(n) >= 0 ? "is-good" : "is-bad");
  return (
    <div className="nif" aria-label="Signing factors">
      <div className="nif__head">
        <span>{title}</span>
        <span className={Number.isFinite(score) ? tone(score) : "is-muted"}>
          {fit?.label || "Neutral"}
          {Number.isFinite(score) && score !== 0 ? ` (${score > 0 ? "+" : ""}${score.toFixed(1)})` : ""}
        </span>
      </div>
      {list.length ? (
        list.map((f) => (
          <div key={`${f.key}-${f.label}`} className="nif__row">
            <span>{f.label}</span>
            <span className={tone(f.effect)}>
              {Number(f.effect) > 0 ? "+" : ""}
              {Number(f.effect).toFixed(1)}
            </span>
          </div>
        ))
      ) : (
        <p className="nif__empty">No strong feelings about the city, market or roster.</p>
      )}
      {Number.isFinite(premium) && premium > 0.5 ? (
        <p className="nif__premium">Wants about {Math.round(premium)}% over his ask to sign here.</p>
      ) : null}
    </div>
  );
}

export default function NegotiationMeetingPanel({ playerId, onChanged, compact = false }) {
  const [data, setData] = React.useState(null);
  const [busy, setBusy] = React.useState("");
  const [flash, setFlash] = React.useState(null);
  const [open, setOpen] = React.useState(true);

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
      {null}
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
              {hometown.active_discount_m || hometown.active_discount_pct ? (
                <p className="nmp__result is-pos">
                  {hometown.active_discount_m
                    ? `Agreed: about $${Number(hometown.active_discount_m).toFixed(2)}M a year under his number on his next deal with you.`
                    : `Agreed: about ${Math.round(Number(hometown.active_discount_pct))}% under market on his next deal with you.`}
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

          {data.goofy_demand ? (
            <div className="nmp__block">
              <span className="nmp__label">Unusual request</span>
              {data.goofy_demand.used ? (
                <p className="nmp__result">{data.goofy_demand.result?.line}</p>
              ) : (
                <>
                  <p className="nmp__result">{data.goofy_demand.text}</p>
                  <div className="nmp__grid">
                    <button
                      type="button"
                      className="nmp__opt"
                      disabled={Boolean(busy)}
                      onClick={() => run("goofy_demand", "accept")}
                    >
                      <strong>Grant it</strong>
                      <span>{data.goofy_demand.reward}</span>
                    </button>
                    <button
                      type="button"
                      className="nmp__opt"
                      disabled={Boolean(busy)}
                      onClick={() => run("goofy_demand", "refuse")}
                    >
                      <strong>Absolutely not</strong>
                      <span>He'll be a little less keen.</span>
                    </button>
                  </div>
                </>
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


