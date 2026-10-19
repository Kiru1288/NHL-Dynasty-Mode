import React, { useState } from "react";
import { OutcomeDonut, PercentileBars, ProspectVisualStyles, SkillRadar } from "./ProspectVisuals";
import { formatRank, formatRating } from "./prospectDossierModel";

const TAB_KEY = "nhl.dossier.tab";

export function readDossierTab() {
  try {
    return window.localStorage.getItem(TAB_KEY) === "intel" ? "intel" : "file";
  } catch (err) {
    return "file";
  }
}

export function writeDossierTab(tab) {
  try {
    window.localStorage.setItem(TAB_KEY, tab === "intel" ? "intel" : "file");
  } catch (err) {
    /* private mode / blocked storage */
  }
}

export function useDossierTab() {
  const [tab, setTab] = useState(readDossierTab);
  const select = (next) => {
    const value = next === "intel" ? "intel" : "file";
    setTab(value);
    writeDossierTab(value);
  };
  return [tab, select];
}

function prettyTier(label) {
  const s = String(label || "").toLowerCase();
  if (!s) return "—";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function DossierHeading({ title, aside }) {
  return (
    <header className="dd-h">
      <h3>{title}</h3>
      {aside ? <span>{aside}</span> : null}
    </header>
  );
}

export function skillBarRows(skills) {
  return (skills || []).map((skill) => ({
    label: skill.label,
    value: skill.exact ? skill.value : (skill.low != null && skill.high != null ? (skill.low + skill.high) / 2 : null),
    low: skill.low,
    high: skill.high,
    text: formatRating(skill),
    locked: skill.locked,
    tier: skill.tier,
    percentile: skill.percentile,
    classAvg: skill.classAvg,
    soft: !skill.exact,
  }));
}

function radarAxes(skills) {
  return (skills || [])
    .filter((skill) => !skill.locked && skill.low != null && skill.high != null)
    .map((skill) => ({
      label: skill.label,
      short: skill.short,
      value: skill.exact ? skill.value : (skill.low + skill.high) / 2,
      classAvg: skill.classAvg,
    }));
}

function FileBar({ model }) {
  const label = prettyTier(model.file.certaintyLabel);
  return (
    <div className="dd-filebar">
      <i style={{ width: `${model.file.pct}%` }} />
      <span>Your file {model.display.file} · {label}</span>
    </div>
  );
}

function VerdictTile({ kicker, value, tone, revealed, children }) {
  return (
    <article className={`dd-tile dd-tile--${tone || "now"}${revealed ? " is-revealed" : ""}`}>
      <span className="dd-tile__kicker">{kicker}</span>
      <strong className="dd-tile__value">{value}</strong>
      {revealed ? <em className="dd-revealed">Revealed</em> : null}
      {children}
    </article>
  );
}

export function DossierVerdict({ model, portrait, compact = false }) {
  const id = model.identity;
  const frame = [id.height, id.weight].filter(Boolean).join(" · ");
  const consensusDiffers = model.rank.consensus != null && model.rank.consensus !== model.rank.board;
  return (
    <section className={`dd-verdict${compact ? " is-compact" : ""}`}>
      {compact ? null : (
        <div className="dd-id">
          <div className="dd-id__shot">{portrait}</div>
          <div className="dd-id__copy">
            <h2>{id.name}</h2>
            <p>{[id.pos, id.age != null ? `${id.age}` : null].filter(Boolean).join(" · ")}</p>
            <p>{[id.shoots, frame].filter(Boolean).join(" · ") || "—"}</p>
            <p>{[id.nation, id.club].filter(Boolean).join(" · ") || "—"}</p>
            <p>{[id.league, id.seasonLine].filter(Boolean).join(" · ") || "—"}</p>
            <p className="dd-id__tags">
              {id.archetype ? <span className="dd-role">{id.archetype}</span> : null}
              {id.overage ? <span className="dd-role dd-role--risk">Overager</span> : null}
            </p>
          </div>
        </div>
      )}
      <div className="dd-verdict__main">
        <div className="dd-tiles">
          <VerdictTile kicker="Now" value={model.display.ovr} tone="now" revealed={model.ovr.revealed}>
            <FileBar model={model} />
          </VerdictTile>
          <VerdictTile kicker="Ceiling" value={model.peak.hidden ? "—" : model.display.peak} tone="ceiling">
            {model.peak.hidden ? <span className="dd-filebar__label">Scout to reveal</span> : <FileBar model={model} />}
          </VerdictTile>
          <VerdictTile kicker="Risk" value={model.risk} tone={String(model.risk || "").toLowerCase()} />
          <VerdictTile kicker="Draft slot" value={model.display.rank} tone="slot">
            <span className="dd-filebar__label">
              {model.display.delta ? model.display.delta : null}
              {consensusDiffers ? ` League ${formatRank(model.rank.consensus)}` : null}
              {!model.display.delta && !consensusDiffers ? "Board" : null}
            </span>
          </VerdictTile>
        </div>
        <p className="dd-sentence">{model.sentence}</p>
        <p className="dd-fileline">
          <span className="is-you">Your file {model.display.file}</span>
          {model.file.nextUnlock ? <span> · {model.file.nextUnlock.unlocks} at {model.file.nextUnlock.atPct}%</span> : <span> · numbers are in</span>}
          {model.display.consensus ? <span className="is-league"> · League viewings {model.display.consensus}</span> : null}
        </p>
      </div>
    </section>
  );
}

function NoteList({ items, empty }) {
  if (!items?.length) return <p className="dd-empty">{empty}</p>;
  return (
    <ul className="dd-notes">
      {items.map((item) => (
        <li key={`${item.key}-${item.detail}`}>
          <b>{item.label}</b>
          <span> — {item.detail}</span>
        </li>
      ))}
    </ul>
  );
}

export function DossierSkillProfile({ model }) {
  const axes = radarAxes(model.skills);
  return (
    <div className="dd-skills">
      <section className="dd-card">
        <DossierHeading
          title={model.goalie ? "Goalie profile" : "Skill profile"}
          aside="Tier = league scale · % = vs. this draft class"
        />
        <PercentileBars rows={skillBarRows(model.skills)} headless />
      </section>
      {axes.length >= 3 ? (
        <section className="dd-card">
          <DossierHeading
            title="Shape"
            aside={<span className="dd-legend-inline"><i className="is-you" /> Him <i className="is-avg" /> Class avg</span>}
          />
          <SkillRadar axes={axes} headless color="#3ec6ff" />
        </section>
      ) : null}
      <section className="dd-card">
        <div className="dd-split-notes">
          <div>
            <DossierHeading title="Strengths" />
            <NoteList items={model.strengths} empty="Scout him to see what carries." />
          </div>
          <div>
            <DossierHeading title="Concerns" />
            <NoteList items={model.concerns} empty="No chapter sits at the bottom of this class." />
          </div>
        </div>
      </section>
    </div>
  );
}

function Fact({ label, value }) {
  return (
    <li>
      <span>{label}</span>
      <strong>{value == null || value === "" ? "—" : value}</strong>
    </li>
  );
}

export function DossierContext({ model }) {
  const role = model.role;
  return (
    <aside className="dd-context">
      {role.outcomes?.length ? (
        <section className="dd-card">
          <DossierHeading title="Career outcomes" />
          <OutcomeDonut segments={role.outcomes} headless />
        </section>
      ) : null}
      {model.character.visible ? (
        <section className="dd-card">
          <DossierHeading title="Character" />
          <ul className="dd-traits">
            {model.character.traits.map((trait) => (
              <li key={trait.key}>
                <span>{trait.label}</span>
                <b style={{ color: trait.tier.color }}>{trait.tier.label}</b>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <section className="dd-card">
        <DossierHeading title="Readiness" />
        <ul className="dd-facts">
          <Fact label="Readiness" value={role.readiness} />
          <Fact label="Projected role" value={role.projected} />
          <Fact label="ETA" value={role.eta} />
          <Fact label="NHL odds" value={model.display.nhlOdds} />
        </ul>
      </section>
    </aside>
  );
}

function Trajectory({ points }) {
  if (!points || points.length < 2) return null;
  const ranks = points.map((p) => p.rank);
  const minR = Math.min(...ranks);
  const maxR = Math.max(...ranks);
  const w = 320;
  const h = 96;
  const pad = 12;
  const span = Math.max(1, points.length - 1);
  const ySpan = Math.max(1, maxR - minR);
  const coords = points.map((pt, i) => {
    const x = pad + ((w - pad * 2) * i) / span;
    const y = pad + ((pt.rank - minR) / ySpan) * (h - pad * 2);
    return { x, y, rank: pt.rank };
  });
  const poly = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const first = points[0];
  const last = points[points.length - 1];
  const rising = last.rank < first.rank;
  return (
    <section className="dd-card">
      <DossierHeading title="Value trajectory" aside={rising ? "Rising" : last.rank > first.rank ? "Falling" : "Flat"} />
      <svg className="dd-traj" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Board rank over time">
        <polyline fill="none" stroke="var(--c-you, #3ec6ff)" strokeWidth="2.5" points={poly} />
        {coords.map((c, i) => (
          <circle key={`${c.rank}-${i}`} cx={c.x} cy={c.y} r={i === coords.length - 1 ? 3.5 : 2.2} fill="var(--c-gold, #f6c453)" />
        ))}
      </svg>
      <p className="dd-empty">{formatRank(first.rank)} to {formatRank(last.rank)}</p>
    </section>
  );
}

export function DossierIntel({ model }) {
  const reports = model.reports;
  const analytics = model.analytics;
  return (
    <div className="dd-intel">
      <section className="dd-card">
        <DossierHeading title="Scouting desk" aside={`Reports: ${reports.returned} returned`} />
        {reports.entries.length ? (
          <ul className="dd-reports">
            {reports.entries.map((row) => (
              <li key={row.key} className={row.locked ? "is-locked" : ""}>
                <b className="dd-grade">{row.locked ? "??" : (row.grade || "—")}</b>
                <div>
                  {row.locked ? <p>Report still out.</p> : <p>{row.quote ? `"${row.quote}"` : "—"}</p>}
                  <small>{[row.scout, row.meta].filter(Boolean).join(" · ")}</small>
                </div>
              </li>
            ))}
          </ul>
        ) : <p className="dd-empty">No reports returned yet.</p>}
      </section>
      <section className="dd-card">
        <DossierHeading
          title="Season & analytics"
          aside={analytics.sampleThin && analytics.gp ? `Thin sample (${analytics.gp} GP)` : null}
        />
        {analytics.cells.length ? (
          <div className="dd-stats">
            {analytics.cells.map((cell) => (
              <div key={cell.label}>
                <span>{cell.label}</span>
                <strong>{cell.value}</strong>
              </div>
            ))}
          </div>
        ) : <p className="dd-empty">No season line on file.</p>}
      </section>
      <Trajectory points={model.trajectory} />
      {model.wjc ? (
        <section className="dd-card">
          <DossierHeading title="World Juniors" aside={model.wjc.grade || null} />
          {model.wjc.headline ? <p className="dd-sentence">{model.wjc.headline}</p> : null}
          <div className="dd-stats">
            {(model.wjc.tiles || []).filter((tile) => tile.value && tile.value !== "—").map((tile) => (
              <div key={tile.label}>
                <span>{tile.label}</span>
                <strong>{tile.value}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

export function DossierTabs({ tab, onTab }) {
  return (
    <div className="dd-tabs" role="tablist" aria-label="Dossier sections">
      <button type="button" role="tab" aria-selected={tab === "file"} className={tab === "file" ? "is-active" : ""} onClick={() => onTab("file")}>
        Scout file
      </button>
      <button type="button" role="tab" aria-selected={tab === "intel"} className={tab === "intel" ? "is-active" : ""} onClick={() => onTab("intel")}>
        Intel desk
      </button>
    </div>
  );
}

export function DossierActions({ actions, assignPanel, statusNote }) {
  return (
    <footer className="dd-actions">
      {statusNote ? <p className="dd-empty">{statusNote}</p> : null}
      {assignPanel || null}
      <div className="dd-actions__row">
        {(actions || []).map((action) => (
          <button
            key={action.id}
            type="button"
            className={`dd-btn${action.primary ? " is-primary" : ""}${action.disabled ? " is-disabled" : ""}`}
            disabled={action.disabled}
            title={action.title || action.hint}
            onClick={action.onClick}
          >
            <strong>{action.label}</strong>
            <small>{action.hint}</small>
          </button>
        ))}
      </div>
    </footer>
  );
}

export function ProspectDossier({ model, portrait, tab, onTab, actions, assignPanel, statusNote }) {
  return (
    <div className="dd-root">
      <ProspectVisualStyles />
      <DossierLayoutStyles />
      <DossierVerdict model={model} portrait={portrait} />
      <DossierTabs tab={tab} onTab={onTab} />
      <div className="dd-scroll">
        {tab === "intel" ? (
          <DossierIntel model={model} />
        ) : (
          <div className="dd-columns">
            <DossierSkillProfile model={model} />
            <DossierContext model={model} />
          </div>
        )}
      </div>
      <DossierActions actions={actions} assignPanel={assignPanel} statusNote={statusNote} />
    </div>
  );
}

export function DossierLayoutStyles() {
  return <style>{DD_CSS}</style>;
}

const DD_CSS = `
.dd-root, .dc-profile-modal--prospect{
  --fs-verdict:34px; --fs-h:13px; --fs-body:14px; --fs-meta:11px;
  --sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:24px;
  --dd-radius:8px;
  --c-you:#3ec6ff; --c-league:#8ab4ff; --c-good:#52df94; --c-risk:#ff7a5c; --c-gold:#f6c453;
  --c-surface:rgba(10,28,62,.72); --c-text:#e8f4fb; --c-muted:#9eb4c6;
  color:var(--c-text);
  font-size:var(--fs-body);
}
.dd-root{display:flex;flex-direction:column;height:100%;min-height:0;min-width:0;}
.dc-dossier-shell > .dd-root{flex:1;min-height:0;height:auto;}
.dd-verdict{display:grid;grid-template-columns:minmax(220px,260px) minmax(0,1fr);gap:var(--sp-4);padding:var(--sp-3) var(--sp-4) var(--sp-2);align-items:start;}
.dd-verdict.is-compact{grid-template-columns:minmax(0,1fr);padding:var(--sp-2) 0;}
.dd-id{display:flex;gap:var(--sp-3);min-width:0;margin:0;transform:none;}
.dd-id__shot{position:relative;width:72px;height:72px;flex:none;overflow:hidden;}
.dd-id__shot .dd-flag{position:absolute;right:0;bottom:0;width:22px;height:16px;}
.dd-id__shot img,.dd-id__shot .player-headshot{max-width:72px;max-height:72px;}
.dd-id__copy{min-width:0;}
.dd-id__copy h2{margin:0;font-size:clamp(18px,2vw,26px);line-height:1.05;overflow-wrap:anywhere;font-weight:800;}
.dd-id__copy p{margin:2px 0 0;font-size:var(--fs-meta);color:var(--c-muted);overflow-wrap:anywhere;}
.dd-id__tags{display:flex;flex-wrap:wrap;gap:var(--sp-1);}
.dd-role{display:inline-block;padding:2px 8px;border:1px solid var(--c-gold);color:var(--c-gold);border-radius:99px;font-size:var(--fs-meta);letter-spacing:.06em;text-transform:uppercase;}
.dd-role--risk{border-color:var(--c-risk);color:var(--c-risk);}
.dd-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--sp-2);}
.dd-tile{background:var(--c-surface);border:1px solid rgba(158,180,198,.18);border-radius:var(--dd-radius);padding:var(--sp-2) var(--sp-3);min-width:0;}
.dd-tile__kicker{display:block;font-size:var(--fs-h);letter-spacing:.12em;text-transform:uppercase;color:var(--c-muted);font-weight:800;}
.dd-tile__value{display:block;font-size:var(--fs-verdict);line-height:1;font-weight:800;font-variant-numeric:tabular-nums;white-space:nowrap;margin:var(--sp-1) 0;}
.dd-tile--ceiling .dd-tile__value{color:var(--c-gold);}
.dd-tile--low .dd-tile__value{color:var(--c-good);}
.dd-tile--medium .dd-tile__value{color:var(--c-gold);}
.dd-tile--high .dd-tile__value{color:var(--c-risk);}
.dd-revealed{display:inline-block;margin-left:var(--sp-2);font-style:normal;font-size:var(--fs-meta);letter-spacing:.08em;text-transform:uppercase;color:var(--c-good);}
.dd-filebar{position:relative;height:6px;border-radius:99px;background:rgba(158,180,198,.18);overflow:hidden;margin-top:var(--sp-1);}
.dd-filebar i{display:block;height:100%;background:var(--c-you);width:0;}
.dd-filebar span,.dd-filebar__label{display:block;margin-top:var(--sp-1);font-size:var(--fs-meta);color:var(--c-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dd-sentence{margin:var(--sp-2) 0 0;font-size:var(--fs-body);line-height:1.35;max-width:72ch;}
.dd-fileline{margin:var(--sp-1) 0 0;font-size:var(--fs-meta);color:var(--c-muted);}
.dd-fileline .is-you{color:var(--c-you);}
.dd-fileline .is-league{color:var(--c-league);}
.dd-tabs{display:grid;grid-template-columns:1fr 1fr;height:40px;margin:0 var(--sp-4);border-bottom:1px solid rgba(158,180,198,.2);}
.dd-tabs button{appearance:none;background:transparent;border:0;border-bottom:3px solid transparent;color:var(--c-muted);font-size:var(--fs-h);letter-spacing:.12em;text-transform:uppercase;font-weight:800;cursor:pointer;}
.dd-tabs button.is-active{color:var(--c-text);border-bottom-color:var(--c-you);}
.dd-scroll{flex:1;min-height:0;overflow:auto;padding:var(--sp-3) var(--sp-4) var(--sp-4);align-content:start;}
.dd-columns{display:grid;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);gap:var(--sp-3);align-items:start;}
.dd-skills,.dd-context,.dd-intel{display:flex;flex-direction:column;gap:var(--sp-3);min-width:0;}
.dd-card{background:var(--c-surface);border:1px solid rgba(158,180,198,.16);border-radius:var(--dd-radius);padding:var(--sp-3);min-width:0;}
.dd-card .pv-card{background:transparent;border:0;box-shadow:none;padding:0;}
.dd-h{display:flex;justify-content:space-between;align-items:baseline;gap:var(--sp-3);margin:0 0 var(--sp-2);}
.dd-h h3{margin:0;font-size:var(--fs-h);letter-spacing:.12em;text-transform:uppercase;font-weight:800;color:var(--c-text);}
.dd-h span{font-size:var(--fs-meta);color:var(--c-muted);text-align:right;}
.dd-legend-inline{display:inline-flex;align-items:center;gap:6px;}
.dd-legend-inline i{width:14px;height:3px;display:inline-block;}
.dd-legend-inline i.is-you{background:var(--c-you);}
.dd-legend-inline i.is-avg{background:rgba(158,180,198,.7);}
.dd-split-notes{display:grid;grid-template-columns:1fr 1fr;gap:var(--sp-4);}
.dd-notes{list-style:none;margin:0;padding:0;display:grid;gap:var(--sp-2);}
.dd-notes li{font-size:var(--fs-body);line-height:1.35;}
.dd-notes b{font-weight:800;}
.dd-empty{margin:0;color:var(--c-muted);font-size:var(--fs-body);}
.dd-facts,.dd-traits{list-style:none;margin:0;padding:0;display:grid;gap:var(--sp-2);}
.dd-facts li,.dd-traits li{display:flex;justify-content:space-between;gap:var(--sp-3);align-items:baseline;font-size:var(--fs-body);}
.dd-facts span,.dd-traits span{color:var(--c-muted);font-size:var(--fs-meta);letter-spacing:.08em;text-transform:uppercase;}
.dd-facts strong,.dd-traits b{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:var(--fs-body);font-weight:800;}
.dd-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:var(--sp-2);}
.dd-stats div{min-width:0;}
.dd-stats span{display:block;font-size:var(--fs-meta);letter-spacing:.08em;text-transform:uppercase;color:var(--c-muted);}
.dd-stats strong{display:block;font-size:var(--fs-body);font-variant-numeric:tabular-nums;white-space:nowrap;}
.dd-reports{list-style:none;margin:0;padding:0;display:grid;gap:var(--sp-2);}
.dd-reports li{display:grid;grid-template-columns:36px minmax(0,1fr);gap:var(--sp-2);align-items:start;}
.dd-grade{display:grid;place-items:center;width:36px;height:28px;border:1px solid var(--c-you);color:var(--c-you);border-radius:6px;font-size:var(--fs-meta);}
.dd-reports p{margin:0;font-size:var(--fs-body);}
.dd-reports small{color:var(--c-muted);font-size:var(--fs-meta);}
.dd-traj{width:100%;height:96px;display:block;}
.dd-actions{flex:none;padding:var(--sp-2) var(--sp-4) var(--sp-3);border-top:1px solid rgba(158,180,198,.18);display:flex;flex-direction:column;gap:var(--sp-2);}
.dd-actions__row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--sp-2);}
.dd-btn{appearance:none;background:transparent;border:1px solid rgba(158,180,198,.35);color:var(--c-text);border-radius:var(--dd-radius);padding:8px 10px;text-align:left;cursor:pointer;min-width:0;}
.dd-btn strong,.dd-btn small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dd-btn strong{font-size:var(--fs-body);}
.dd-btn small{font-size:var(--fs-meta);color:var(--c-muted);font-weight:600;}
.dd-btn.is-primary{background:var(--c-you);border-color:var(--c-you);color:#062033;}
.dd-btn.is-primary small{color:#062033;}
.dd-btn.is-disabled{opacity:.55;cursor:not-allowed;}
.dd-assign{display:flex;flex-wrap:wrap;gap:var(--sp-1);}
.dd-assign button{appearance:none;background:transparent;border:1px solid rgba(158,180,198,.35);color:var(--c-text);border-radius:99px;padding:4px 10px;font-size:var(--fs-meta);cursor:pointer;}
.dd-assign button.is-active{border-color:var(--c-you);color:var(--c-you);}
@media (max-width:1100px){
  .dd-verdict,.dd-columns,.dd-split-notes,.dd-tiles,.dd-actions__row{grid-template-columns:1fr 1fr;}
  .dd-verdict{grid-template-columns:1fr;}
}
@media (max-width:720px){
  .dd-tiles,.dd-actions__row,.dd-columns,.dd-split-notes{grid-template-columns:1fr;}
}
`;
