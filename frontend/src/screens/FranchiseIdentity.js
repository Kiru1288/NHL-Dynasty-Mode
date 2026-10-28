/**
 * Franchise Identity: the club's headquarters. Overview, branding studio (3D uniform +
 * layered crest), philosophy, captaincy and identity history.
 *
 * Rebrands are drafted locally: nothing reaches the save until "Confirm rebrand", which
 * sends the whole identity once (backend validates, then appends one version). Cancel
 * just drops the draft.
 */
import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getTeamLogoSrc } from "../utils/teamLogos";
import { CrestSVG, contrastRatio } from "../components/legacy/crest";
import CrestEditor from "../components/legacy/CrestEditor";
import { apiError, legacyApi, seasonLabel, setLegacyFocus } from "../components/legacy/legacyApi";

const UniformStudio3D = React.lazy(() => import("../components/legacy/UniformStudio3D"));

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "uniform", label: "Uniform Studio" },
  { id: "crest", label: "Crest Studio" },
  { id: "brand", label: "Name & Colours" },
  { id: "philosophy", label: "Philosophy" },
  { id: "captaincy", label: "Captaincy" },
  { id: "history", label: "Identity History" },
];
const UNIFORM_COLOR_FIELDS = [
  ["body", "Body"], ["sleeve", "Sleeves"], ["shoulder", "Shoulder yoke"], ["stripe_color", "Stripe"], ["stripe_accent", "Stripe accent"],
  ["collar_color", "Collar / trim"], ["number_color", "Numbers"], ["number_outline", "Number outline"], ["name_color", "Nameplate"],
];
const clone = (x) => JSON.parse(JSON.stringify(x));
const plural = (n, w) => `${n} ${w}${Number(n) === 1 ? "" : "s"}`;

function Swatch({ label, value, onChange }) {
  return (
    <label className="lh-field">
      <span>{label}</span>
      <span className="lh-color">
        <input type="color" value={value || "#000000"} onChange={(e) => onChange(e.target.value.toUpperCase())} />
        <code>{value}</code>
      </span>
    </label>
  );
}

function Fact({ label, value, sub, onOpen }) {
  if (value == null || value === "") return null;
  return (
    <div className={`lh-fact ${onOpen ? "is-link" : ""}`} onClick={onOpen} role={onOpen ? "button" : undefined} tabIndex={onOpen ? 0 : undefined}>
      <span className="lh-fact-label">{label}</span>
      <span className="lh-fact-value">{value}</span>
      {sub ? <span className="lh-fact-sub">{sub}</span> : null}
    </div>
  );
}

function crestFor(identity, fallbackLogo) {
  return identity?.use_custom_crest ? <CrestSVG crest={identity.crest} size={120} /> : fallbackLogo ? <img src={fallbackLogo} alt="" className="lh-logo-img" /> : <CrestSVG crest={identity?.crest} size={120} />;
}

export default function FranchiseIdentity() {
  const { setScreen, refreshFranchise } = useGameUI();
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [tab, setTab] = useState("overview");
  const [draft, setDraft] = useState(null);
  const [undo, setUndo] = useState([]);
  const [uniKind, setUniKind] = useState("home");
  const [view, setView] = useState("front");
  const [review, setReview] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [preview, setPreview] = useState({ name: "", number: 19, letter: "C" });
  const [presetName, setPresetName] = useState("");
  const reqRef = useRef(null);

  const load = useCallback(() => {
    setErr("");
    return legacyApi.identity().then(setData).catch((e) => setErr(apiError(e)));
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const saved = data?.identity;
  const editing = draft != null;
  const ident = draft || saved;
  const logo = useMemo(() => getTeamLogoSrc(data?.original_identity?.abbreviation || ""), [data]);

  useEffect(() => {
    const cap = data?.captains?.captain;
    if (cap && !preview.name) setPreview((p) => ({ ...p, name: String(cap.name || "").split(" ").slice(-1)[0] }));
  }, [data, preview.name]);

  const startEdit = () => {
    if (!saved) return;
    setDraft(clone(saved));
    setUndo([]);
    reqRef.current = `rb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  };
  const edit = (fn) => {
    if (!data?.is_user_team) return;
    if (!draft) reqRef.current = `rb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    const base = draft || clone(saved);
    const next = clone(base);
    fn(next);
    setUndo((u) => [...u.slice(-49), base]);
    setDraft(next);
  };
  const doUndo = () => {
    if (!undo.length) return;
    setDraft(undo[undo.length - 1]);
    setUndo(undo.slice(0, -1));
  };
  const discard = () => {
    setDraft(null);
    setUndo([]);
    setReview(false);
  };

  const uniform = ident?.uniforms?.[uniKind] || ident?.uniforms?.home;
  const warnings = useMemo(() => {
    if (!ident) return [];
    const w = [];
    if (contrastRatio(ident.colors.primary, ident.colors.secondary) < 1.6) w.push("Primary and secondary are too close to tell apart.");
    ["home", "away", "alternate"].forEach((k) => {
      const u = ident.uniforms?.[k];
      if (u && contrastRatio(u.number_color, u.body) < 2) w.push(`${k} numbers are hard to read against the jersey.`);
    });
    if (!/^[A-Z]{2,4}$/.test(ident.abbreviation || "")) w.push("Abbreviation must be 2-4 capital letters.");
    return w;
  }, [ident]);
  const changed = useMemo(() => {
    if (!draft || !saved) return [];
    return Object.keys(draft).filter((k) => JSON.stringify(draft[k]) !== JSON.stringify(saved[k]));
  }, [draft, saved]);

  const confirm = async () => {
    setSaving(true);
    setErr("");
    try {
      const out = await legacyApi.rebrand(draft, "Rebrand from Franchise Identity", reqRef.current);
      setNotice(out.unchanged ? "Nothing changed." : `Identity v${out.version} saved: ${out.changed?.join(", ")}.`);
      setDraft(null);
      setUndo([]);
      setReview(false);
      await load();
      refreshFranchise?.();
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setSaving(false);
    }
  };

  if (err && !data) {
    return (
      <div className="lh-screen">
        <div className="lh-empty">{err}<button type="button" className="dd-btn" onClick={load}>Retry</button></div>
      </div>
    );
  }
  if (!data || !ident) return <div className="lh-screen"><div className="lh-empty">Opening the front office…</div></div>;

  const s = data.summary || {};
  const cap = data.captains?.captain;
  const era = (data.eras || []).slice(-1)[0];
  const openPlayer = (pid) => {
    setLegacyFocus({ type: "player", player_id: pid });
    setScreen(SCREENS.LEGACY_HALL);
  };

  return (
    <div className="lh-screen" style={{ "--team-1": ident.colors.primary, "--team-2": ident.colors.secondary, "--team-3": ident.colors.accent }}>
      <header className="lh-rail">
        <button type="button" className="dd-btn dd-btn--secondary lh-back" onClick={() => (editing ? setReview(true) : setScreen(SCREENS.HUB))}>
          {editing ? "Review" : "Back"}
        </button>
        <div className="lh-rail-crest">{crestFor(ident, logo)}</div>
        <div className="lh-rail-title">
          <span className="lh-eyebrow">Franchise Identity{editing ? " · editing draft" : ""}</span>
          <h1 className="dd-title lh-h1">{ident.name}</h1>
          <span className="lh-rail-meta">
            {ident.abbreviation} · identity v{data.current_version}
            {era ? ` · ${era.label}` : ""}
          </span>
        </div>
        <nav className="lh-tabs" aria-label="Identity sections">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={`lh-tab ${tab === t.id ? "is-on" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>
          ))}
        </nav>
      </header>

      {notice ? <div className="lh-notice" onClick={() => setNotice("")}>{notice}</div> : null}
      {err ? <div className="lh-notice lh-notice--err">{err}</div> : null}

      {tab === "overview" && (
        <section className="lh-overview">
          <div className="lh-focal">
            <Suspense fallback={<div className="lh-studio-canvas" />}>
              <UniformStudio3D uniform={ident.uniforms.home} crest={ident.use_custom_crest ? ident.crest : null} crestFallbackSrc={ident.use_custom_crest ? null : logo} preview={preview} view="front" height={430} />
            </Suspense>
          </div>
          <div className="lh-sheet">
            <div className="lh-sheet-head">
              <span className="lh-brass">{s.championships?.length || 0} championships</span>
              <span>{s.playoff_appearances ?? 0} playoff seasons of {s.playoff_seasons_known ?? 0} tracked</span>
            </div>
            <Fact label="In the save since" value={data.first_season_in_save ? seasonLabel(data.first_season_in_save) : null} />
            <Fact label="Captain" value={cap?.name || "No captain named"} sub={cap ? `since ${seasonLabel(cap.start_season)}` : "Name one under Captaincy"} onOpen={cap ? () => openPlayer(cap.player_id) : () => setTab("captaincy")} />
            <Fact label="Philosophy" value={data.philosophy?.declared_label || "Not declared"} sub={data.philosophy?.conflict || (data.philosophy?.observed || []).map((o) => o.label).join(" · ") || null} onOpen={() => setTab("philosophy")} />
            <Fact label="Best season" value={s.best_season ? `${s.best_season.label}: ${s.best_season.pts} pts` : null} sub={s.best_season ? `${s.best_season.w}-${s.best_season.l}-${s.best_season.otl}` : null} />
            <Fact label="Most games for the club" value={s.most_games ? `${s.most_games.name} · ${s.most_games.gp} GP` : null} onOpen={s.most_games ? () => openPlayer(s.most_games.player_id) : null} />
            <Fact label="Top homegrown player" value={s.top_homegrown ? `${s.top_homegrown.name} · ${s.top_homegrown.pts} pts` : null} onOpen={s.top_homegrown ? () => openPlayer(s.top_homegrown.player_id) : null} />
            <Fact label="Longest-serving active" value={s.longest_serving_active ? `${s.longest_serving_active.name} · ${plural(s.longest_serving_active.seasons, "season")}` : null} onOpen={s.longest_serving_active ? () => openPlayer(s.longest_serving_active.player_id) : null} />
            <Fact label="Longest playoff streak" value={s.longest_playoff_streak != null ? plural(s.longest_playoff_streak, "season") : null} />
            <div className="lh-recent">
              <span className="lh-fact-label">Recent franchise history</span>
              {(s.recent_events || []).length === 0 ? <p className="lh-muted">{s.coverage_note}</p> : null}
              {(s.recent_events || []).slice(0, 5).map((e) => (
                <div key={e.id} className="lh-event"><b>{seasonLabel(e.season)}</b> {e.headline}</div>
              ))}
            </div>
            <div className="lh-sheet-foot">
              <span className="lh-muted">{s.coverage_note}</span>
              <button type="button" className="dd-btn" onClick={() => { setLegacyFocus({ type: "franchise" }); setScreen(SCREENS.LEGACY_HALL); }}>Legacy Hall</button>
            </div>
          </div>
        </section>
      )}

      {(tab === "uniform" || tab === "crest" || tab === "brand") && (
        <section className="lh-studio">
          <div className="lh-studio-main">
            {tab === "crest" ? (
              <CrestEditor
                crest={ident.crest}
                colors={ident.colors}
                options={data.options}
                initials={(ident.nickname || ident.abbreviation || "").split(" ").map((w) => w[0]).join("").slice(0, 3).toUpperCase()}
                onChange={(c) => edit((d) => { d.crest = c; d.use_custom_crest = true; })}
              />
            ) : (
              <>
                <div className="lh-view-bar">
                  {["home", "away", "alternate"].map((k) => (
                    <button key={k} type="button" className={`lh-chip ${uniKind === k ? "is-on" : ""}`} onClick={() => setUniKind(k)}>
                      {k}{k === "alternate" && !ident.uniforms.alternate ? " (none)" : ""}
                    </button>
                  ))}
                  <span className="lh-sep" />
                  {["front", "side", "back"].map((v) => (
                    <button key={v} type="button" className={`lh-chip ${view === v ? "is-on" : ""}`} onClick={() => setView(v)}>{v}</button>
                  ))}
                </div>
                {uniKind === "alternate" && !ident.uniforms.alternate ? (
                  <div className="lh-empty lh-empty--inline">
                    No alternate uniform yet.
                    <button type="button" className="dd-btn" onClick={() => edit((d) => { d.uniforms.alternate = { ...clone(d.uniforms.home), body: d.colors.secondary, sleeve: d.colors.secondary, number_color: d.colors.primary }; })}>Create from home</button>
                  </div>
                ) : (
                  <Suspense fallback={<div className="lh-studio-canvas" style={{ height: 470 }} />}>
                    <UniformStudio3D uniform={uniform} crest={ident.use_custom_crest ? ident.crest : null} crestFallbackSrc={ident.use_custom_crest ? null : logo} preview={preview} view={view} height={470} />
                  </Suspense>
                )}
                <div className="lh-preview-bar">
                  <label className="lh-field"><span>Preview name</span><input maxLength={14} value={preview.name} onChange={(e) => setPreview({ ...preview, name: e.target.value })} /></label>
                  <label className="lh-field"><span>#</span><input type="number" min={0} max={99} value={preview.number} onChange={(e) => setPreview({ ...preview, number: Math.max(0, Math.min(99, Number(e.target.value) || 0)) })} /></label>
                  <label className="lh-field"><span>Letter</span>
                    <select value={preview.letter} onChange={(e) => setPreview({ ...preview, letter: e.target.value })}><option value="C">C</option><option value="A">A</option><option value="">none</option></select>
                  </label>
                </div>
              </>
            )}
          </div>
          <aside className="lh-studio-side">
            {!editing ? (
              <div className="lh-empty lh-empty--inline">
                <p>Changes are drafted here and only saved when you confirm. Name, colours, crest and uniforms can change; the franchise, its players and its history stay the same.</p>
                <button type="button" className="dd-btn" onClick={startEdit} disabled={!data.is_user_team}>Start rebrand draft</button>
              </div>
            ) : (
              <>
                {tab === "brand" && (
                  <div className="lh-form">
                    {[["city", "City / market"], ["nickname", "Nickname"], ["name", "Full name"]].map(([k, label]) => (
                      <label key={k} className="lh-field"><span>{label}</span><input maxLength={32} value={ident[k] || ""} onChange={(e) => edit((d) => { d[k] = e.target.value; })} /></label>
                    ))}
                    <label className="lh-field"><span>Abbreviation</span><input maxLength={4} value={ident.abbreviation || ""} onChange={(e) => edit((d) => { d.abbreviation = e.target.value.toUpperCase().replace(/[^A-Z]/g, ""); })} /></label>
                    {["primary", "secondary", "accent", "neutral"].map((k) => (
                      <Swatch key={k} label={`${k} colour`} value={ident.colors[k]} onChange={(v) => edit((d) => { d.colors[k] = v; })} />
                    ))}
                    <label className="lh-field"><span>Ice logo</span>
                      <select value={ident.arena?.ice_logo || "crest"} onChange={(e) => edit((d) => { d.arena = { ...(d.arena || {}), ice_logo: e.target.value }; })}>
                        <option value="crest">Crest</option><option value="wordmark">Wordmark</option><option value="none">None</option>
                      </select>
                    </label>
                    <label className="lh-field"><span>Crest on jerseys</span>
                      <select value={ident.use_custom_crest ? "custom" : "current"} onChange={(e) => edit((d) => { d.use_custom_crest = e.target.value === "custom"; })}>
                        <option value="current">Current club logo</option><option value="custom">Custom crest</option>
                      </select>
                    </label>
                  </div>
                )}
                {tab === "uniform" && uniform && (
                  <div className="lh-form">
                    {UNIFORM_COLOR_FIELDS.map(([k, label]) => (
                      <Swatch key={k} label={label} value={uniform[k]} onChange={(v) => edit((d) => { d.uniforms[uniKind][k] = v; })} />
                    ))}
                    {[["stripe", "Stripes", data.options.stripes], ["collar", "Collar", data.options.collars], ["number_font", "Number font", data.options.number_fonts], ["nameplate", "Nameplate", data.options.nameplates]].map(([k, label, opts]) => (
                      <label key={k} className="lh-field"><span>{label}</span>
                        <select value={uniform[k]} onChange={(e) => edit((d) => { d.uniforms[uniKind][k] = e.target.value; })}>
                          {(opts || []).map((o) => <option key={o} value={o}>{o}</option>)}
                        </select>
                      </label>
                    ))}
                    <label className="lh-field lh-field--check"><input type="checkbox" checked={!!uniform.waist_stripe} onChange={(e) => edit((d) => { d.uniforms[uniKind].waist_stripe = e.target.checked; })} /><span>Waist stripes</span></label>
                    <label className="lh-field lh-field--check"><input type="checkbox" checked={!!uniform.captain_letter} onChange={(e) => edit((d) => { d.uniforms[uniKind].captain_letter = e.target.checked; })} /><span>Captain letters</span></label>
                    {[["crest_scale", "Crest size", 0.5, 1.5, 0.05], ["crest_y", "Crest height", -0.3, 0.3, 0.01], ["fabric_roughness", "Fabric sheen", 0.3, 1, 0.02]].map(([k, label, lo, hi, st]) => (
                      <label key={k} className="lh-field"><span>{label}</span>
                        <input type="range" min={lo} max={hi} step={st} value={uniform[k]} onChange={(e) => edit((d) => { d.uniforms[uniKind][k] = Number(e.target.value); })} />
                      </label>
                    ))}
                    {uniKind === "alternate" && ident.uniforms.alternate ? (
                      <button type="button" className="lh-chip lh-chip--warn" onClick={() => edit((d) => { d.uniforms.alternate = null; })}>Remove alternate</button>
                    ) : null}
                    <div className="lh-preset-row">
                      <input placeholder="Preset name" maxLength={28} value={presetName} onChange={(e) => setPresetName(e.target.value)} />
                      <button type="button" className="lh-chip" disabled={!presetName.trim()} onClick={async () => {
                        try {
                          await legacyApi.savePreset(presetName.trim(), uniform);
                          setNotice(`Preset “${presetName.trim()}” saved.`);
                          setPresetName("");
                          load();
                        } catch (e) { setErr(apiError(e)); }
                      }}>Save preset</button>
                    </div>
                    {(data.presets || []).length ? (
                      <div className="lh-row-actions">
                        {data.presets.map((p) => (
                          <button key={p.name} type="button" className="lh-chip" onClick={() => edit((d) => { d.uniforms[uniKind] = clone(p.uniform); })}>{p.name}</button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                )}
                {warnings.length ? <ul className="lh-warn">{warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}
                <div className="lh-draft-actions">
                  <button type="button" className="lh-chip" onClick={doUndo} disabled={!undo.length}>Undo</button>
                  <button type="button" className="lh-chip" onClick={() => { setUndo((u) => [...u, draft]); setDraft(clone(saved)); }}>Reset to saved</button>
                  <button type="button" className="lh-chip lh-chip--warn" onClick={discard}>Discard draft</button>
                  <button type="button" className="dd-btn" onClick={() => setReview(true)} disabled={!changed.length}>Review ({changed.length})</button>
                </div>
              </>
            )}
          </aside>
        </section>
      )}

      {tab === "philosophy" && <PhilosophyPanel data={data.philosophy} onSaved={load} onError={setErr} canEdit={data.is_user_team} />}
      {tab === "captaincy" && <CaptaincyPanel onSaved={load} onError={setErr} canEdit={data.is_user_team} openPlayer={openPlayer} />}
      {tab === "history" && (
        <section className="lh-history">
          {[...(data.versions || [])].reverse().map((v) => (
            <div key={v.version} className={`lh-version ${v.version === data.current_version ? "is-current" : ""}`}>
              <div className="lh-version-crest">{v.identity?.use_custom_crest ? <CrestSVG crest={v.identity.crest} size={86} /> : logo ? <img src={logo} alt="" className="lh-logo-img lh-logo-img--sm" /> : <CrestSVG crest={v.identity?.crest} size={86} />}</div>
              <div>
                <b>v{v.version} · {v.identity?.name}</b> <span className="lh-muted">({v.identity?.abbreviation})</span>
                <div className="lh-muted">{v.effective_season ? `From ${seasonLabel(v.effective_season)}` : "Original"} · {v.reason}</div>
                {v.changed?.length ? <div className="lh-muted">Changed: {v.changed.join(", ")}</div> : null}
              </div>
              <div className="lh-version-colors">{["primary", "secondary", "accent", "neutral"].map((k) => <span key={k} className="lh-swatch" style={{ background: v.identity?.colors?.[k] }} />)}</div>
            </div>
          ))}
          <div className="lh-sheet">
            <span className="lh-fact-label">Retired numbers</span>
            {Object.keys(data.retired_numbers || {}).length === 0 ? <p className="lh-muted">No numbers retired yet. Retire one from a player's dossier in Legacy Hall.</p> : null}
            {Object.entries(data.retired_numbers || {}).map(([n, r]) => (
              <div key={n} className="lh-event is-link" onClick={() => openPlayer(r.player_id)}><b>#{n}</b> {r.name} <span className="lh-muted">retired {seasonLabel(r.season)}</span></div>
            ))}
          </div>
        </section>
      )}

      {review && draft ? (
        <div className="lh-modal" role="dialog" aria-modal="true" aria-label="Review rebrand">
          <div className="lh-modal-card">
            <h2 className="dd-title">Review rebrand</h2>
            <div className="lh-compare">
              <div><span className="lh-fact-label">Current · v{data.current_version}</span>{crestFor(saved, logo)}<b>{saved.name}</b><span className="lh-muted">{saved.abbreviation}</span></div>
              <div className="lh-compare-arrow">→</div>
              <div><span className="lh-fact-label">Proposed</span>{crestFor(draft, logo)}<b>{draft.name}</b><span className="lh-muted">{draft.abbreviation}</span></div>
            </div>
            <p><b>Changing:</b> {changed.join(", ") || "nothing"}</p>
            <p className="lh-muted">Your franchise keeps its id, roster, contracts, standings and every past record. Past seasons keep the identity they were played under. This adds a rebrand entry to the franchise history; it has no effect on morale or ratings.</p>
            {warnings.length ? <ul className="lh-warn">{warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}
            <div className="lh-draft-actions">
              <button type="button" className="lh-chip" onClick={() => setReview(false)}>Keep editing</button>
              <button type="button" className="lh-chip lh-chip--warn" onClick={discard}>Discard</button>
              <button type="button" className="dd-btn" disabled={saving || !changed.length} onClick={confirm}>{saving ? "Saving…" : "Confirm rebrand"}</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function PhilosophyPanel({ data, onSaved, onError, canEdit }) {
  const [busy, setBusy] = useState(false);
  if (!data) return null;
  const pick = async (k) => {
    setBusy(true);
    try {
      await legacyApi.setPhilosophy(k);
      onSaved();
    } catch (e) {
      onError(apiError(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="lh-two">
      <div className="lh-sheet">
        <span className="lh-fact-label">Declared direction</span>
        <div className="lh-option-grid">
          {Object.entries(data.options || {}).map(([k, label]) => (
            <button key={k} type="button" disabled={busy || !canEdit} className={`lh-option ${data.declared === k ? "is-on" : ""}`} onClick={() => pick(k)}>{label}</button>
          ))}
        </div>
        <p className="lh-muted">{data.effect_note}</p>
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">What the record shows</span>
        {data.conflict ? <div className="lh-notice lh-notice--warn">{data.conflict}</div> : null}
        {(data.observed || []).length === 0 ? <p className="lh-muted">No clear pattern in the evidence yet.</p> : null}
        {(data.observed || []).map((o) => <div key={o.key} className="lh-event"><b>{o.label}</b> <span className="lh-muted">{o.evidence}</span></div>)}
        <table className="lh-table">
          <tbody>
            {Object.entries(data.evidence || {}).map(([k, v]) => (
              <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{v == null ? "unknown" : typeof v === "number" && v < 1 && v > 0 ? `${Math.round(v * 100)}%` : String(v)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CaptaincyPanel({ onSaved, onError, canEdit, openPlayer }) {
  const [d, setD] = useState(null);
  const [captain, setCaptain] = useState("");
  const [alts, setAlts] = useState([]);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => legacyApi.captaincy().then((x) => {
    setD(x);
    setCaptain(x.captain?.player_id || "");
    setAlts((x.alternates || []).map((a) => a.player_id));
  }).catch((e) => onError(apiError(e))), [onError]);
  useEffect(() => { load(); }, [load]);
  if (!d) return <div className="lh-empty">Loading leadership group…</div>;
  const save = async () => {
    setBusy(true);
    try {
      await legacyApi.appoint(captain || null, alts);
      await load();
      onSaved();
    } catch (e) {
      onError(apiError(e));
    } finally {
      setBusy(false);
    }
  };
  const evidenceKeys = Array.from(new Set((d.candidates || []).flatMap((c) => Object.keys(c.evidence || {}))));
  return (
    <section className="lh-two">
      <div className="lh-sheet">
        <span className="lh-fact-label">Leadership candidates</span>
        <table className="lh-table">
          <thead><tr><th>Player</th>{evidenceKeys.map((k) => <th key={k}>{k}</th>)}<th>Score</th><th>Role</th></tr></thead>
          <tbody>
            {(d.candidates || []).map((c) => (
              <tr key={c.player_id}>
                <td className="is-link" onClick={() => openPlayer(c.player_id)}>{c.name} <span className="lh-muted">{c.position}</span></td>
                {evidenceKeys.map((k) => <td key={k}>{c.evidence?.[k] ?? "–"}</td>)}
                <td><b>{c.score ?? "–"}</b></td>
                <td>
                  <select disabled={!canEdit} value={captain === c.player_id ? "C" : alts.includes(c.player_id) ? "A" : ""} onChange={(e) => {
                    const v = e.target.value;
                    setAlts((a) => a.filter((x) => x !== c.player_id).concat(v === "A" ? [c.player_id] : []).slice(0, 3));
                    if (v === "C") setCaptain(c.player_id);
                    else if (captain === c.player_id) setCaptain("");
                  }}>
                    <option value="">–</option><option value="C">C</option><option value="A">A</option>
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="lh-muted">{d.method}</p>
        <button type="button" className="dd-btn" disabled={busy || !canEdit} onClick={save}>Confirm leadership group</button>
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">Captains on record</span>
        {(d.history || []).length === 0 ? <p className="lh-muted">No captaincy recorded yet.</p> : null}
        {[...(d.history || [])].reverse().map((t) => (
          <div key={`${t.player_id}-${t.start_iso}-${t.start_season}`} className="lh-event is-link" onClick={() => openPlayer(t.player_id)}>
            <b>{t.name}</b> {seasonLabel(t.start_season)} – {t.end_season ? seasonLabel(t.end_season) : "present"}
            {t.end_reason ? <span className="lh-muted"> · {t.end_reason}</span> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
