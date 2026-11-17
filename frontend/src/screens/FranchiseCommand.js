import React from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { api, setFranchiseSessionId } from "../services/api";

/**
 * Franchise Command: the commissioner's office.
 * 50 sliders that reshape the save live (backend: services/franchise_command.py),
 * plus disk save / load and "start a new sim".
 */

const GROUP_ICON = { ice: "🏒", growth: "📈", chaos: "🎲", money: "💰", world: "🌎" };
const PIXEL_ICON = { ice: "ICE", growth: "DEV", chaos: "LIFE", money: "CAP", world: "MAP" };

function fmtValue(s, v) {
  const n = Number(v);
  if (s.fmt === "toggle") return n >= 0.5 ? "On" : "Off";
  if (s.fmt === "pct") {
    const p = Math.round(n * 1000) / 10;
    return `${p > 0 ? "+" : ""}${p}%`;
  }
  if (s.fmt === "money") return `${n > 0 ? "+" : n < 0 ? "−" : ""}$${Math.abs(n).toFixed(2)}M`;
  const clean = Math.abs(n) >= 10 || Number.isInteger(n) ? String(Math.round(n * 100) / 100) : n.toFixed(2);
  if (s.unit === "%" && Number(s.default) === 0) return `${n > 0 ? "+" : ""}${clean}%`;
  if (s.unit === " pts" && Number(s.default) === 0) return `${n > 0 ? "+" : ""}${clean} pts`;
  if (s.unit === "" && Number(s.default) === 0) return `${n > 0 ? "+" : ""}${clean}`;
  return `${clean}${s.unit || ""}`;
}

function isChanged(s, v) {
  return Math.abs(Number(v) - Number(s.default)) > 1e-9;
}

function SliderCard({ s, value, onChange }) {
  const changed = isChanged(s, value);
  const pct = ((Number(value) - s.min) / (s.max - s.min || 1)) * 100;
  const defPct = ((Number(s.default) - s.min) / (s.max - s.min || 1)) * 100;
  if (s.fmt === "toggle") {
    const on = Number(value) >= 0.5;
    return (
      <div className={`fcmd-card is-toggle${changed ? " is-changed" : ""}`}>
        <div className="fcmd-card__head">
          <strong>{s.label}</strong>
          <button type="button" className={`fcmd-switch${on ? " is-on" : ""}`} onClick={() => onChange(s.id, on ? 0 : 1)} aria-pressed={on}>
            <span>{on ? s.high || "On" : s.low || "Off"}</span>
            <i />
          </button>
        </div>
        <p>{s.desc}</p>
      </div>
    );
  }
  return (
    <div className={`fcmd-card${changed ? " is-changed" : ""}`}>
      <div className="fcmd-card__head">
        <strong>{s.label}</strong>
        <b className="fcmd-val">{fmtValue(s, value)}</b>
      </div>
      <div className="fcmd-track" style={{ "--lo": `${Math.min(pct, defPct)}%`, "--hi": `${Math.max(pct, defPct)}%`, "--def": `${defPct}%` }}>
        <input
          type="range"
          min={s.min}
          max={s.max}
          step={s.step}
          value={value}
          onChange={(e) => onChange(s.id, Number(e.target.value))}
          aria-label={s.label}
        />
        <span className="fcmd-track__def" aria-hidden />
      </div>
      <div className="fcmd-card__ends">
        <span>{s.low || fmtValue(s, s.min)}</span>
        {changed ? (
          <button type="button" className="fcmd-mini" onClick={() => onChange(s.id, s.default)}>Default</button>
        ) : (
          <em>Default</em>
        )}
        <span>{s.high || fmtValue(s, s.max)}</span>
      </div>
      <p>{s.desc}</p>
    </div>
  );
}

export default function FranchiseCommand() {
  const { setScreen, refreshFranchise } = useGameUI();
  const [data, setData] = React.useState(null);
  const [vals, setVals] = React.useState({});
  const [group, setGroup] = React.useState("ice");
  const [side, setSide] = React.useState("presets");
  const [err, setErr] = React.useState("");
  const [toast, setToast] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [saveName, setSaveName] = React.useState("");
  const [carry, setCarry] = React.useState(true);
  const [confirmNew, setConfirmNew] = React.useState(false);
  const pending = React.useRef({});
  const timer = React.useRef(null);
  const toastTimer = React.useRef(null);

  const flash = React.useCallback((msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 2200);
  }, []);

  const absorb = React.useCallback((d) => {
    if (!d || !d.sliders) return;
    setData(d);
    const v = {};
    d.sliders.forEach((s) => { v[s.id] = s.value; });
    setVals(v);
  }, []);

  React.useEffect(() => {
    let live = true;
    api.get("/api/franchise/command")
      .then(({ data: d }) => { if (live) absorb(d); })
      .catch((e) => { if (live) setErr(e?.response?.data?.detail || e.message || "Could not open Franchise Command."); });
    return () => { live = false; clearTimeout(timer.current); clearTimeout(toastTimer.current); };
  }, [absorb]);

  const post = React.useCallback(async (body, msg) => {
    setBusy(true);
    try {
      const { data: d } = await api.post("/api/franchise/command", body);
      absorb(d);
      if (msg) flash(msg);
      refreshFranchise?.();
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "Change failed.");
    } finally {
      setBusy(false);
    }
  }, [absorb, flash, refreshFranchise]);

  const onChange = React.useCallback((id, v) => {
    setVals((cur) => ({ ...cur, [id]: v }));
    pending.current[id] = v;
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const values = pending.current;
      pending.current = {};
      post({ values }, "Applied to your save");
    }, 280);
  }, [post]);

  const doSave = async () => {
    setBusy(true);
    try {
      const { data: d } = await api.post("/api/franchise/save", { name: saveName });
      setData((cur) => (cur ? { ...cur, saves: d.saves || cur.saves } : cur));
      flash(`Saved “${d?.save?.name || "franchise"}”`);
      setSaveName("");
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "Save failed.");
    } finally {
      setBusy(false);
    }
  };

  const doLoad = async (slot) => {
    setBusy(true);
    try {
      const { data: d } = await api.post("/api/franchise/load", { slot });
      if (d?.session_id) {
        setFranchiseSessionId(d.session_id);
        window.location.reload();
      }
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "Load failed.");
      setBusy(false);
    }
  };

  const doNewSim = async () => {
    setBusy(true);
    try {
      await api.post("/api/franchise/new-sim", { carry_sliders: carry });
      setScreen(SCREENS.SETUP);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message || "Could not start a new sim.");
      setBusy(false);
    }
  };

  const sliders = data?.sliders || [];
  const groups = data?.groups || [];
  const env = data?.environment;
  const rules = data?.rules || {};
  const activeGroup = groups.find((g) => g.id === group);
  const changedBy = (gid) => sliders.filter((s) => s.group === gid && isChanged(s, vals[s.id])).length;
  const changedAll = sliders.filter((s) => isChanged(s, vals[s.id])).length;

  return (
    <div className="fcmd">
      <aside className="fcmd-rail">
        <button type="button" className="fcmd-rail__home" onClick={() => setScreen(SCREENS.HUB)} title="Back to the office">
          <span>⌂</span>
          <small>Office</small>
        </button>
        <nav>
          {groups.map((g) => (
            <button key={g.id} type="button" className={`fcmd-rail__btn${group === g.id ? " is-on" : ""}`} onClick={() => setGroup(g.id)}>
              <span className="fcmd-rail__icon">{GROUP_ICON[g.id] || "•"}</span>
              <small>{g.label}</small>
              {changedBy(g.id) ? <b>{changedBy(g.id)}</b> : null}
            </button>
          ))}
        </nav>
      </aside>

      <main className="fcmd-main">
        <header className="fcmd-top">
          <div className="fcmd-top__title">
            <span>Commissioner's office · {data?.team || "Your club"}{data?.season ? ` · ${data.season}` : ""}</span>
            <h1>Franchise Command</h1>
          </div>
          <div className="fcmd-top__chips">
            <span className="fcmd-chip">{changedAll} of {sliders.length || 50} changed</span>
            <span className="fcmd-chip is-preset">{data?.preset || "NHL Standard"}</span>
          </div>
          <div className="fcmd-top__actions">
            <button type="button" className="fcmd-btn" disabled={busy} onClick={() => post({ randomize: true }, "Surprise league rolled")}>🎲 Surprise me</button>
            <button type="button" className="fcmd-btn is-ghost" disabled={busy || !changedAll} onClick={() => post({ reset: true }, "Back to NHL Standard")}>Reset all</button>
          </div>
        </header>

        {err ? <div className="fcmd-error" onClick={() => setErr("")}>{err} <small>(tap to dismiss)</small></div> : null}
        {!data && !err ? <div className="fcmd-loading">Opening the commissioner's office…</div> : null}

        {data ? (
          <>
            <section className="fcmd-hero">
              <div className="fcmd-hero__paper">
                <span className="fcmd-kicker">Scoring environment</span>
                <h2>{env?.label}</h2>
                <p>
                  {env?.era_adjusted ? (
                    <>A 70-point season here is judged like <b>{env.example.forward_equivalent}</b> for a forward and <b>{env.example.defense_equivalent}</b> for a defenceman in a normal league. Ratings follow the adjusted number, so a big-scoring league does not inflate overalls.</>
                  ) : (
                    <>Era adjustment is off: raw points move ratings as they are. A high-scoring league will push overalls up fast.</>
                  )}
                </p>
              </div>
              <div className="fcmd-hero__meter">
                <span className="fcmd-kicker">Goals vs normal</span>
                <strong>×{Number(env?.scoring_env || 1).toFixed(2)}</strong>
                <small>D-men ×{Number(env?.defense_env || 1).toFixed(2)}</small>
              </div>
              <div className="fcmd-hero__rules">
                <div><span>OT</span><b>{rules.ot_minutes ?? 5} min</b></div>
                <div><span>Ties</span><b>{rules.ties_allowed ? "Yes" : "No"}</b></div>
                <div><span>Points</span><b>{rules.pts_reg_win ?? 2}-{rules.pts_ot_win ?? 2}-{rules.pts_otl ?? 1}</b></div>
                <div><span>PP</span><b>×{Number(rules.pp_mult || 1).toFixed(2)}</b></div>
              </div>
            </section>

            <section className="fcmd-body">
              <div className="fcmd-board">
                <div className="fcmd-board__head">
                  <span className="fcmd-pixel">{PIXEL_ICON[group] || ""}</span>
                  <h3>{activeGroup?.label}</h3>
                  <p>{activeGroup?.blurb} Every change is live on the next sim day.</p>
                </div>
                <div className="fcmd-grid">
                  {sliders.filter((s) => s.group === group).map((s) => (
                    <SliderCard key={s.id} s={s} value={vals[s.id] ?? s.default} onChange={onChange} />
                  ))}
                </div>
              </div>

              <aside className="fcmd-side">
                <div className="fcmd-tabs">
                  {[["presets", "Presets"], ["save", "Save & load"], ["new", "New sim"], ["log", "Change log"]].map(([id, label]) => (
                    <button key={id} type="button" className={side === id ? "is-on" : ""} onClick={() => setSide(id)}>{label}</button>
                  ))}
                </div>
                <div className="fcmd-side__panel">
                  {side === "presets" ? (
                    <div className="fcmd-presets">
                      {(data.presets || []).map((p) => (
                        <button key={p.id} type="button" disabled={busy} className={data.preset === p.label ? "is-on" : ""} onClick={() => post({ preset: p.id }, `${p.label} loaded`)}>
                          <strong>{p.label}</strong>
                          <small>{p.blurb}</small>
                        </button>
                      ))}
                    </div>
                  ) : null}

                  {side === "save" ? (
                    <div className="fcmd-saves">
                      <label className="fcmd-field">
                        <span>Save name</span>
                        <input value={saveName} maxLength={60} placeholder={`${data.team || "Franchise"} ${data.season || ""}`.trim()} onChange={(e) => setSaveName(e.target.value)} />
                      </label>
                      <button type="button" className="fcmd-btn is-go" disabled={busy} onClick={doSave}>💾 Save franchise</button>
                      <p className="fcmd-note">Saves go to the backend's saves folder and survive restarts. Same name overwrites.</p>
                      <div className="fcmd-savelist">
                        {(data.saves || []).length === 0 ? <p className="fcmd-note">No saves yet.</p> : null}
                        {(data.saves || []).map((sv) => (
                          <div key={sv.slot} className="fcmd-save">
                            <div>
                              <strong>{sv.name}</strong>
                              <small>{[sv.team, sv.season, sv.date, sv.saved_at].filter(Boolean).join(" · ")}</small>
                            </div>
                            <button type="button" className="fcmd-btn is-small" disabled={busy} onClick={() => doLoad(sv.slot)}>Load</button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {side === "new" ? (
                    <div className="fcmd-new">
                      <p>Start a brand-new franchise from team select. Save this one first if you want to come back to it.</p>
                      <label className="fcmd-check">
                        <input type="checkbox" checked={carry} onChange={(e) => setCarry(e.target.checked)} />
                        <span>Carry these {changedAll} slider changes into the new sim</span>
                      </label>
                      {!confirmNew ? (
                        <button type="button" className="fcmd-btn is-warn" disabled={busy} onClick={() => setConfirmNew(true)}>🏁 Start a new sim</button>
                      ) : (
                        <div className="fcmd-confirm">
                          <strong>Leave this franchise?</strong>
                          <div>
                            <button type="button" className="fcmd-btn is-warn" disabled={busy} onClick={doNewSim}>Yes, new sim</button>
                            <button type="button" className="fcmd-btn is-ghost" onClick={() => setConfirmNew(false)}>Not yet</button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : null}

                  {side === "log" ? (
                    <ol className="fcmd-log">
                      {(data.history || []).length === 0 ? <li className="is-empty">No changes yet.</li> : null}
                      {(data.history || []).map((h, i) => (
                        <li key={i}>
                          <small>{h.date || h.season || ""}</small>
                          <span>{h.what}</span>
                        </li>
                      ))}
                    </ol>
                  ) : null}
                </div>
              </aside>
            </section>
          </>
        ) : null}
      </main>
      {toast ? <div className="fcmd-toast">✔ {toast}</div> : null}
    </div>
  );
}
