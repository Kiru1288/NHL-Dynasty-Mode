import React, { useCallback, useEffect, useState } from "react";
import { prospectTalk } from "../../services/franchiseService";

const PATH_LABEL = { ncaa: "College (NCAA)", junior: "Junior", europe: "Europe", other: "Prospect" };

function InterestMeter({ signing }) {
  if (!signing) return null;
  const pct = Math.max(0, Math.min(100, Number(signing.interest) || 0));
  return (
    <div className={`ptalk-meter is-${signing.tone || "warn"}`}>
      <div className="ptalk-meter__head">
        <span>Signing interest</span>
        <strong>
          {Math.round(pct)} · {signing.label}
        </strong>
      </div>
      <div className="ptalk-meter__bar">
        <i style={{ width: `${pct}%` }} />
      </div>
      {signing.warning ? <p className="ptalk-warning">⚠ {signing.warning}</p> : null}
      {Array.isArray(signing.factors) && signing.factors.length ? (
        <ul className="ptalk-factors">
          {signing.factors.map((f) => (
            <li key={f.label} className={Number(f.delta) >= 0 ? "is-up" : "is-down"}>
              {Number(f.delta) >= 0 ? "▲" : "▼"} {f.label}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** Check-in conversation with a rights-held prospect (junior / college / Europe). */
export default function ProspectTalkModal({ playerId, playerName, onClose, onChanged }) {
  const [data, setData] = useState(null);
  const [lines, setLines] = useState([]);
  const [topic, setTopic] = useState("");
  const [approaches, setApproaches] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const name = data?.player_name || playerName || "Prospect";

  const run = useCallback(
    async (payload) => {
      setBusy(true);
      setError("");
      try {
        const res = await prospectTalk({ player_id: playerId, ...payload });
        if (!res || res.ok === false) {
          setError(res?.reason || "He didn't pick up.");
          return null;
        }
        setData(res);
        return res;
      } catch (e) {
        setError(e?.message || "Call failed.");
        return null;
      } finally {
        setBusy(false);
      }
    },
    [playerId]
  );

  useEffect(() => {
    let alive = true;
    run({}).then((res) => {
      if (alive && res) setLines([{ who: "him", text: res.opener }]);
    });
    return () => {
      alive = false;
    };
  }, [run]);

  const askTopic = async (t) => {
    const res = await run({ topic: t.id });
    if (!res) return;
    setTopic(t.id);
    setApproaches(res.approaches || []);
    setLines((prev) => [...prev, { who: "gm", text: t.label }, { who: "him", text: res.answer }]);
  };

  const pickApproach = async (a) => {
    const res = await run({ topic, approach: a.id });
    if (!res) return;
    setApproaches([]);
    setTopic("");
    setLines((prev) => [
      ...prev,
      { who: "gm", text: a.label },
      { who: "him", text: res.reply },
      { who: "sys", text: res.message, tone: Number(res.delta) > 0 ? "up" : Number(res.delta) < 0 ? "down" : "" },
    ]);
    if (typeof onChanged === "function") onChanged(res.signing);
  };

  const line = data?.season_line || {};
  return (
    <div className="ptalk-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="ptalk-card" onClick={(e) => e.stopPropagation()}>
        <header className="ptalk-head">
          <div>
            <span className="ptalk-kicker">Prospect check-in · {PATH_LABEL[data?.path] || "Prospect"}</span>
            <h2>{name}</h2>
            {data ? (
              <span className="ptalk-sub">
                {data.position} · Age {data.age} · OVR {data.ovr}
                {Number(data.ovr_delta_season) ? ` (${data.ovr_delta_season > 0 ? "+" : ""}${data.ovr_delta_season} this season)` : ""}
                {line.gp ? ` · ${line.league || ""} ${line.gp} GP, ${line.pts} PTS` : ""}
                {data.rights_signing_deadline ? ` · Sign by ${data.rights_signing_deadline}` : ""}
              </span>
            ) : null}
          </div>
          <button type="button" className="ptalk-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <InterestMeter signing={data?.signing} />

        <div className="ptalk-log">
          {lines.map((l, i) => (
            <div key={i} className={`ptalk-bubble is-${l.who}${l.tone ? ` is-${l.tone}` : ""}`}>
              {l.who === "him" ? <b>{name.split(" ").slice(-1)[0]}:</b> : l.who === "gm" ? <b>You:</b> : null} {l.text}
            </div>
          ))}
          {busy ? <div className="ptalk-bubble is-sys">…</div> : null}
          {error ? <div className="ptalk-bubble is-sys is-down">{error}</div> : null}
        </div>

        <div className="ptalk-choices">
          {approaches.length
            ? approaches.map((a) => (
                <button key={a.id} type="button" className="ptalk-btn is-approach" disabled={busy} onClick={() => pickApproach(a)}>
                  <strong>{a.label}</strong>
                  <em>{a.detail}</em>
                </button>
              ))
            : (data?.topics || []).map((t) => (
                <button key={t.id} type="button" className="ptalk-btn" disabled={busy} onClick={() => askTopic(t)}>
                  <strong>{t.label}</strong>
                  {t.cooldown_days ? <em>Talked recently ({t.cooldown_days}d)</em> : null}
                </button>
              ))}
        </div>
      </div>
    </div>
  );
}
