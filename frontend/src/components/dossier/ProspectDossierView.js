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

export function DossierSkillProfile({ model, showShape = true }) {
  const axes = showShape ? radarAxes(model.skills) : [];
  return (
    <div className="dd-skills">
      <section className="dd-card">
        <DossierHeading
          title={model.goalie ? "Goalie profile" : "Skill profile"}
          aside={model.file.pct < 72 ? "Ranges until your file reaches 72%" : null}
        />
        <PercentileBars rows={skillBarRows(model.skills)} headless />
        <p className="dd-legend-note">Bar shows the rating. The tick is the class average. "Top 30%" is where he ranks in this draft class.</p>
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

/** Most likely career outcome from the backend distribution; null when it sent none. */
function likelyOutcome(model) {
  const segs = model.role.outcomes || [];
  if (!segs.length) return null;
  return segs.reduce((best, s) => (s.pct > best.pct ? s : best), segs[0]);
}

/** Draft hero: who he is, then only Now / Ceiling / Your file. */
export function DossierHero({ model, portrait, tag = "P1 file" }) {
  const id = model.identity;
  const frame = [id.height, id.weight].filter(Boolean).join(" ");
  const nowPublic = !model.ovr.revealed && model.file.pct < 72;
  return (
    <section className="dd-hero">
      {tag && tag !== "P1 file" ? <span className="dd-hero__tag">{tag}</span> : null}
      <div className="dd-hero__shot">{portrait}</div>
      <div className="dd-hero__id">
        <h2>{id.name}</h2>
        <p>{[id.pos, id.age != null ? String(id.age) : null, id.shoots, frame].filter(Boolean).join(" · ")}</p>
        <p className="is-muted">{[id.club, id.league].filter(Boolean).join(" · ") || "—"}</p>
      </div>
      <div className="dd-hero__eval">
        <div className="dd-hero__nums">
          <div>
            <span>Now</span>
            <strong>{model.display.ovr}</strong>
            {nowPublic ? <small>Public read</small> : null}
          </div>
          <div className="is-ceiling">
            <span>Ceiling</span>
            <strong>{model.peak.hidden ? "?" : model.display.peak}</strong>
          </div>
        </div>
        <div className="dd-hero__file" title="How much of this player your scouts have seen">
          <span>Scouting file {model.display.file}</span>
          <i><b style={{ width: `${model.file.pct}%` }} /></i>
        </div>
      </div>
    </section>
  );
}

/** Three plain statements: slot, likely outcome, club need. Each is omitted when the data is missing. */
export function DossierVerdictStrip({ model, eta, slotNote }) {
  const likely = likelyOutcome(model);
  const board = model.rank.consensus ?? model.rank.board;
  const slot = [board != null ? `${formatRank(board)} public` : null, model.rank.board != null && model.rank.board !== board ? `${formatRank(model.rank.board)} yours` : null]
    .filter(Boolean).join(" · ");
  const outcome = likely
    ? `${likely.label} (${Math.round(likely.pct)}%)`
    : (model.role.projected ? `${model.role.projected} if it hits` : null);
  const known = (v) => (v && !/unknown/i.test(String(v)) ? String(v) : null);
  const timeline = known(eta) || known(model.role.eta);
  const fit = model.orgFit;
  const cells = [
    slot ? { k: "Board", v: slot, sub: slotNote ? `${slotNote} at this pick` : null } : null,
    outcome ? { k: "Likely outcome", v: outcome, sub: timeline } : null,
    fit ? { k: "Club need", v: fit.label || "Noted", sub: fit.reasons[0] || null } : null,
  ].filter(Boolean);
  if (!cells.length) return null;
  return (
    <div className="dd-strip">
      {cells.map((c) => (
        <div key={c.k} className="dd-strip__cell">
          <span>{c.k}</span>
          <strong>{c.v}</strong>
          {c.sub ? <small>{c.sub}</small> : null}
        </div>
      ))}
    </div>
  );
}

const CS_LO = 20;
const CS_HI = 99;
const csPos = (v) => Math.max(0, Math.min(100, ((Number(v) - CS_LO) / (CS_HI - CS_LO)) * 100));

const TIER_ORDER = ["awful", "bad", "weak", "average", "good", "very-good", "great", "elite", "world-class"];
function tierTone(tier) {
  const i = TIER_ORDER.indexOf(tier?.key);
  if (i < 0) return "mid";
  if (i >= 6) return "up";
  if (i <= 2) return "down";
  return "mid";
}

function pctTone(p) {
  if (p == null) return "mid";
  if (p >= 67) return "up";
  if (p < 40) return "down";
  return "mid";
}

function sentenceCase(label) {
  const s = String(label || "");
  return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : "";
}

/** Now → ceiling: one rail, a fill for current ability and a bracket for the scouted ceiling band. */
function PotentialRail({ model }) {
  const ovr = model.ovr;
  const peak = model.peak;
  const nowLo = ovr.exact ? ovr.value : ovr.low;
  const nowHi = ovr.exact ? ovr.value : ovr.high;
  return (
    <div className="ws-pot">
      <div className="ws-pot__nums">
        <span className="ws-pot__now"><small>Now</small>{model.display.ovr}</span>
        <span className="ws-pot__arrow" aria-hidden="true" />
        <span className="ws-pot__ceil"><small>Ceiling</small>{peak.hidden ? "?" : model.display.peak}</span>
      </div>
      <div className="ws-rail" role="img" aria-label={`Now ${model.display.ovr}, ceiling ${peak.hidden ? "unknown" : model.display.peak}`}>
        {nowLo != null ? <i className="ws-rail__now" style={{ width: `${csPos(nowHi ?? nowLo)}%`, opacity: nowHi > nowLo ? 0.6 : 1 }} /> : null}
        {!peak.hidden && peak.low != null && peak.high != null ? (
          <i className="ws-rail__ceil" style={{ left: `${csPos(peak.low)}%`, width: `${Math.max(1.5, csPos(peak.high) - csPos(peak.low))}%` }} />
        ) : null}
      </div>
    </div>
  );
}

const CONF_FILL = {
  Confirmed: 3, Narrow: 3, Reliable: 3, Read: 3,
  Solid: 2, Moderate: 2, Usable: 2,
  Early: 1, Wide: 1, Thin: 1, Unread: 1,
};
const CONF_ROWS = [
  ["current", "Current"],
  ["development", "Development"],
  ["translation", "Stats"],
  ["character", "Character"],
];

/** Four separate certainty reads. None of them is a chance of NHL success. */
function ConfidenceMeter({ confidence, filePct }) {
  if (!confidence) {
    return <p className="ws-conf__fallback">Your file {filePct}%</p>;
  }
  return (
    <ul className="ws-conf" aria-label="Scouting confidence">
      {CONF_ROWS.map(([key, label]) => {
        const read = confidence[key] || {};
        const fill = CONF_FILL[read.level] ?? 0;
        return (
          <li key={key} title={read.why || undefined}>
            <span>{label}</span>
            <i aria-hidden="true">{[0, 1, 2].map((n) => <b key={n} className={n < fill ? "is-on" : ""} />)}</i>
            <em>{read.level || "—"}</em>
          </li>
        );
      })}
    </ul>
  );
}

function MatrixCell({ skill }) {
  const exact = skill.exact && skill.value != null;
  const lo = exact ? skill.value : skill.low;
  const hi = exact ? skill.value : skill.high;
  const tone = skill.locked ? "lock" : tierTone(skill.tier);
  const pct = skill.percentile == null ? null
    : skill.percentile >= 50 ? `Top ${Math.max(1, 100 - Math.round(skill.percentile))}%` : `Btm ${Math.max(1, Math.round(skill.percentile))}%`;
  return (
    <div className={`ws-cell is-${tone}`} title={skill.label}>
      <span className="ws-cell__lab">{skill.label}</span>
      <strong className="ws-cell__val">{skill.locked ? "?" : formatRating(skill)}</strong>
      <i className="ws-cell__bar">
        {skill.locked || lo == null ? null : (
          <b style={{ left: exact ? 0 : `${csPos(lo)}%`, width: exact ? `${csPos(lo)}%` : `${Math.max(3, csPos(hi) - csPos(lo))}%` }} />
        )}
        {skill.classAvg != null ? <em style={{ left: `${csPos(skill.classAvg)}%` }} /> : null}
      </i>
      <small>
        {skill.locked ? "Scout to reveal" : sentenceCase(skill.tier?.label)}
        {pct ? <span className={`is-${pctTone(skill.percentile)}`}> {pct}</span> : null}
      </small>
    </div>
  );
}

function FindingBlock({ finding, others, tone, title }) {
  if (!finding) return null;
  return (
    <div className={`ws-find is-${tone}`}>
      <small className="ws-find__kick">{title} {"·"} {finding.kind === "projection" ? "projection risk" : "current"}</small>
      <strong>{finding.title}</strong>
      <p>{finding.evidence}</p>
      <p className="ws-find__bench">{finding.benchmark}{finding.why ? ` — ${finding.why}` : ""}</p>
      {finding.confidence ? <small className="ws-find__conf">Confidence: {finding.confidence}</small> : null}
      {others?.length ? (
        <ul>
          {others.map((o) => <li key={o.key}>{o.title} <span>{o.evidence}</span></li>)}
        </ul>
      ) : null}
    </div>
  );
}

const OUTCOME_TONES = ["down", "slate", "mid", "blue", "up", "up"];

function OutcomeRail({ outcomes, mostLikely, ceilingOutcome, method }) {
  const list = (outcomes || []).filter((o) => o.pct >= 0);
  if (!list.some((o) => o.pct > 0)) return null;
  const total = list.reduce((a, o) => a + o.pct, 0) || 1;
  const topKey = mostLikely?.key || list.reduce((a, b) => (b.pct > a.pct ? b : a), list[0]).key;
  const toneAt = (i) => OUTCOME_TONES[Math.min(OUTCOME_TONES.length - 1, Math.round((i / Math.max(1, list.length - 1)) * (OUTCOME_TONES.length - 2)))];
  return (
    <div className="ws-out">
      <div className="ws-out__head">
        <span>Career outcomes</span>
        <span>{ceilingOutcome ? `Ceiling: ${ceilingOutcome.label}` : null}{method === "heuristic" ? " · heuristic model" : ""}</span>
      </div>
      <div className="ws-out__bar" role="img" aria-label={list.map((o) => `${o.label} ${Math.round(o.pct)}%`).join(", ")}>
        {list.map((o, i) => (
          o.pct > 0 ? <i key={o.key} className={`is-${toneAt(i)}${o.key === topKey ? " is-top" : ""}`} style={{ width: `${(o.pct / total) * 100}%` }} /> : null
        ))}
      </div>
      <ul className="ws-out__legend">
        {list.map((o, i) => (
          <li key={o.key} className={o.key === topKey ? "is-top" : ""}>
            <i className={`is-${toneAt(i)}`} />
            {o.label} <b>{Math.round(o.pct)}%</b>
          </li>
        ))}
      </ul>
    </div>
  );
}

function fitTone(label) {
  const s = String(label || "").toLowerCase();
  if (/elite|good|strong|great|ideal/.test(s)) return "up";
  if (/poor|weak|crowd|bad|conflict/.test(s)) return "down";
  return "mid";
}

const ESTIMATE_LABELS = {
  war: "WAR", offensive_war: "Off WAR", defensive_war: "Def WAR", xgf_pct: "xGF%", cf_pct: "CF%", shot_rate: "Shots/G", gsax: "GSAx",
};

/**
 * Draft scouting workspace. Left: who he is, his upside, how sure the read is, and
 * the draft verdict. Right: the evaluation, an attribute matrix, the defining
 * advantage against the defining concern, and the outcome split. Evidence sits
 * behind one disclosure. Every value comes from the dossier model or the caller.
 */
export function DossierCommandSheet({
  model,
  portrait,
  tag,
  slotNote,
  eta,
  playStyle,
  production,
  currentPick,
  footnote,
}) {
  const id = model.identity;
  const frame = [id.height, id.weight].filter(Boolean).join(" ");
  const board = model.rank.consensus ?? model.rank.board;
  const fit = model.orgFit;
  const known = (v) => (v && !/unknown/i.test(String(v)) ? String(v) : null);
  const when = known(eta) || known(model.role.eta);
  const intel = model.intel;
  const tr = intel?.translation;
  const findings = model.findings || { strengths: [], concerns: [] };
  const ranks = model.value;
  const rankLine = ranks
    ? [ranks.ceiling != null ? `Ceiling #${ranks.ceiling}` : null, ranks.ability != null ? `now #${ranks.ability}` : null].filter(Boolean).join(", ")
    : null;
  const vsPick = currentPick != null ? `You pick #${currentPick}` : null;
  const traits = model.character.visible ? model.character.traits : [];
  const unmodelled = model.character.unmodelled || [];
  const estimates = tr?.estimates ? Object.entries(tr.estimates).filter(([k]) => ESTIMATE_LABELS[k]) : [];
  return (
    <div className="cs-host">
      <div className="ws">
        <section className="ws-left">
          <div className="ws-id">
            {tag ? <span className="ws-id__tag">{tag}</span> : null}
            <div className="ws-id__row">
              <div className="ws-id__flag">{portrait}</div>
              <div className="ws-id__copy">
                <h2>{id.name}</h2>
                <p>{[id.pos, id.age != null ? String(id.age) : null, id.shoots, frame].filter(Boolean).join(" · ")}</p>
                <p className="is-soft">{[id.club, id.league].filter(Boolean).join(" · ") || "—"}</p>
              </div>
            </div>
            {playStyle || production ? (
              <div className="ws-id__style">
                {playStyle ? <span className="ws-stamp">{playStyle}</span> : null}
                {production ? <span className="ws-id__prod"><b>{production.line}</b>{production.context ? ` · ${production.context}` : ""}</span> : null}
              </div>
            ) : null}
            <PotentialRail model={model} />
            <ConfidenceMeter confidence={intel?.confidence} filePct={model.file.pct} />
          </div>

          <div className="ws-verdict">
            {board != null ? (
              <div className="ws-board">
                <small>Board</small>
                <strong>#{board}</strong>
                {slotNote ? <span>{slotNote}</span> : null}
              </div>
            ) : null}
            <dl className="ws-verdict__facts">
              {rankLine ? (
                <div>
                  <dt>In this class</dt>
                  <dd>{rankLine}<small>{[`of ${ranks.size}`, vsPick].filter(Boolean).join(" · ")}</small></dd>
                </div>
              ) : null}
              {fit ? (
                <div>
                  <dt>Club need</dt>
                  <dd className={`is-${fitTone(fit.label)}`}>{fit.label || "Noted"}{fit.reasons[0] ? <small>{fit.reasons[0]}</small> : null}</dd>
                </div>
              ) : null}
              {model.role.mostLikely ? (
                <div>
                  <dt>Most likely</dt>
                  <dd>{model.role.mostLikely.label}{when ? <small>ETA {when}</small> : null}</dd>
                </div>
              ) : null}
            </dl>
          </div>
        </section>

        <section className="ws-right">
          {model.summary ? <p className="ws-sum">{model.summary}</p> : null}
          <div className={`ws-matrix${model.goalie ? " is-goalie" : ""}`}>
            {model.skills.map((skill) => <MatrixCell key={skill.key} skill={skill} />)}
            <span className="ws-matrix__key"><em /> class average</span>
          </div>
          <div className="ws-av">
            <FindingBlock finding={findings.strengths[0]} others={findings.strengths.slice(1)} tone="up" title="Advantage" />
            <FindingBlock finding={findings.concerns[0]} others={findings.concerns.slice(1)} tone="down" title="Concern" />
            {!findings.strengths.length && !findings.concerns.length ? (
              <p className="ws-muted">Not enough evidence yet to name an advantage or a concern.</p>
            ) : null}
          </div>
          <OutcomeRail
            outcomes={model.role.outcomes}
            mostLikely={model.role.mostLikely}
            ceilingOutcome={model.role.ceilingOutcome}
            method={model.role.outcomeMethod}
          />
        </section>
      </div>

      <details className="ws-evidence">
        <summary>Evidence: stats, comparables, confidence, character</summary>
        <div className="ws-evidence__grid">
          {tr ? (
            <div>
              <h4>Stats in context</h4>
              <p>
                {tr.observed?.gp ? `${tr.observed.gp} GP` : "No games on record"}
                {tr.nhle != null ? ` · ${tr.nhle} NHLe pts/82` : ""}
                {tr.context_band ? ` · ${tr.context_band} for position` : ""}
              </p>
              {tr.nhle_factor != null ? <p className="ws-muted">League translation factor {tr.nhle_factor}. {tr.context_basis}.</p> : null}
              {estimates.length ? (
                <>
                  <p className="ws-est">
                    {estimates.map(([k, v]) => <span key={k}>{ESTIMATE_LABELS[k]} <b>{Number(v).toFixed(1)}</b></span>)}
                  </p>
                  <p className="ws-muted">{tr.estimates_label}: {tr.estimates_note}</p>
                </>
              ) : null}
            </div>
          ) : null}
          <div>
            <h4>Class comparables</h4>
            {model.comparables?.length ? (
              <ul className="ws-comps">
                {model.comparables.map((c) => (
                  <li key={c.id || c.name}>
                    <b>{c.name}</b>{c.rank ? ` #${c.rank}` : ""} <span className="ws-sim">{c.similarity}</span>
                    <small>Closest on {c.closest.toLowerCase()}{c.apart ? `; ${c.apart.toLowerCase()}` : ""}</small>
                  </li>
                ))}
              </ul>
            ) : <p className="ws-muted">Not enough scouted overlap in this class to compare.</p>}
            <p className="ws-muted">Profile match within this draft class (0 to 100). Not a career forecast.</p>
          </div>
          {intel?.confidence ? (
            <div>
              <h4>How sure we are</h4>
              <ul className="ws-why">
                {CONF_ROWS.map(([key, label]) => (
                  <li key={key}><b>{label}: {intel.confidence[key]?.level}</b> {intel.confidence[key]?.why}</li>
                ))}
              </ul>
            </div>
          ) : null}
          <div>
            <h4>Character</h4>
            {traits.length ? (
              <ul className="ws-traits">
                {traits.map((t) => {
                  const tone = tierTone(t.tier) === "down" ? "down" : (["good", "very-good", "great", "elite", "world-class"].includes(t.tier?.key) ? "up" : "mid");
                  return <li key={t.key} className={`is-${tone}`}><span>{t.label}</span><b>{sentenceCase(t.tier.label)}</b></li>;
                })}
              </ul>
            ) : <p className="ws-muted">{model.file.pct < 36 && !unmodelled.length ? "Opens once your file reaches 36%." : "No character traits on file."}</p>}
            {unmodelled.length ? <p className="ws-muted">Not modelled: {unmodelled.join(", ")}.</p> : null}
          </div>
          {footnote ? <p className="ws-muted ws-evidence__foot">{footnote}</p> : null}
        </div>
      </details>
    </div>
  );
}

/** Projection, not evidence: role and ETA are read off the scouted ceiling band. */
export function DossierProjection({ model }) {
  const role = model.role;
  return (
    <section className="dd-card">
      <DossierHeading title="Projection" aside={`Scouting estimate · ${prettyTier(model.file.certaintyLabel)}`} />
      <ul className="dd-facts">
        <Fact label="Readiness" value={role.readiness} />
        <Fact label="Projected role" value={role.projected} />
        <Fact label="ETA" value={role.eta} />
        <Fact label="NHL odds" value={model.display.nhlOdds} />
      </ul>
    </section>
  );
}

export function DossierOrgFit({ model }) {
  const fit = model.orgFit;
  if (!fit) return null;
  return (
    <section className="dd-card">
      <DossierHeading title="Club fit" aside={fit.label} />
      {fit.reasons.length ? (
        <ul className="dd-notes">
          {fit.reasons.map((line) => <li key={line}>{line}</li>)}
        </ul>
      ) : <p className="dd-empty">No positional need flagged.</p>}
    </section>
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
      <DossierProjection model={model} />
      <DossierOrgFit model={model} />
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
      {model.playoffs ? (
        <section className="dd-card">
          <DossierHeading
            title="Playoffs"
            aside={model.playoffs.sampleThin ? `Thin sample (${model.playoffs.gp} GP)` : "Kept apart from the season line"}
          />
          <div className="dd-stats">
            {model.playoffs.cells.map((cell) => (
              <div key={cell.label}>
                <span>{cell.label}</span>
                <strong>{cell.value}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}
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
.dd-hero{position:relative;display:grid;grid-template-columns:52px minmax(0,1fr) auto;gap:18px;align-items:center;margin:0 0 4px;padding:20px 4px 18px;border-bottom:1px solid rgba(248,213,138,.28);}
.dd-hero__tag{position:absolute;top:2px;left:4px;font-size:12px;color:var(--dd-gold-hi,#f8d58a);}
.dd-hero__shot{width:52px;height:52px;border-radius:6px;overflow:hidden;display:grid;place-items:center;background:rgba(255,255,255,.05);}
.dd-hero__shot img{max-width:100%;max-height:100%;}
.dd-hero__id{min-width:0;}
.dd-hero__id h2{margin:0;font-family:var(--dd-font-ui);font-size:clamp(22px,2.4vw,30px);font-weight:800;line-height:1.1;color:#fff;text-transform:none;letter-spacing:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.dd-hero__id p{margin:4px 0 0;font-size:14px;color:var(--dd-cream,#ffe7c4);overflow-wrap:anywhere;}
.dd-hero__id p.is-muted{font-size:13px;color:var(--c-muted);margin-top:2px;}
.dd-hero__eval{display:flex;flex-direction:column;align-items:flex-end;gap:8px;}
.dd-hero__nums{display:flex;gap:28px;text-align:right;}
.dd-hero__nums span{display:block;font-size:12px;color:var(--c-muted);}
.dd-hero__nums strong{display:block;font-family:var(--dd-font-ui);font-size:40px;font-weight:800;line-height:1;color:#fff;font-variant-numeric:tabular-nums;white-space:nowrap;}
.dd-hero__nums .is-ceiling strong{color:var(--dd-gold-hi,#f8d58a);}
.dd-hero__nums small{display:block;font-size:11px;color:var(--c-muted);margin-top:2px;}
.dd-hero__file{width:180px;font-size:12px;color:var(--c-muted);text-align:right;}
.dd-hero__file i{display:block;height:3px;margin-top:4px;background:rgba(158,180,198,.2);border-radius:2px;overflow:hidden;}
.dd-hero__file b{display:block;height:100%;background:var(--c-you);}
.dd-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin:0 0 20px;}
.dd-strip__cell{padding:14px 16px 14px 4px;min-width:0;}
.dd-strip__cell + .dd-strip__cell{padding-left:16px;border-left:1px solid rgba(158,180,198,.16);}
.dd-strip__cell span{display:block;font-size:12px;color:var(--c-muted);}
.dd-strip__cell strong{display:block;margin-top:2px;font-size:16px;font-weight:700;color:#fff;}
.dd-strip__cell small{display:block;margin-top:2px;font-size:13px;color:var(--dd-cream,#ffe7c4);opacity:1;overflow-wrap:anywhere;}
.dd-legend-note{margin:8px 0 0;font-size:12px;color:var(--c-muted);}
@media (max-width:720px){
  .dd-hero{grid-template-columns:44px minmax(0,1fr);}
  .dd-hero__eval{grid-column:1 / -1;align-items:stretch;}
  .dd-hero__nums{justify-content:flex-start;text-align:left;}
  .dd-hero__file{width:auto;text-align:left;}
  .dd-strip__cell + .dd-strip__cell{border-left:0;padding-left:4px;border-top:1px solid rgba(158,180,198,.16);}
}
.cs-host{container-type:inline-size;--ws-cyan:#3ee6ff;--ws-orange:#ff8a1f;--ws-lime:#9be564;--ws-blue:#5aa2ff;--ws-cream:var(--dd-cream,#ffe7c4);--ws-coral:#ff6b5c;--ws-slate:#5a6b93;--ws-ink:#071233;--ws-deep:#050c2c;--ws-soft:#a9bfdc;
  color:#eef4ff;font-family:var(--dd-font-ui);font-size:14px;line-height:1.35;}
.ws{display:grid;grid-template-columns:minmax(260px,0.82fr) minmax(0,1.18fr);gap:0;align-items:stretch;}
.ws small{display:block;font-size:11px;font-weight:700;color:var(--ws-soft);}
.ws-muted{margin:0;font-size:12px;color:var(--ws-soft);}
.ws-left{display:flex;flex-direction:column;background:var(--ws-cream);color:var(--ws-ink);clip-path:polygon(0 0,100% 0,calc(100% - 18px) 100%,0 100%);}
.ws-id{padding:16px 30px 14px 18px;display:flex;flex-direction:column;gap:10px;}
.ws-id__tag{font-size:11px;font-weight:900;color:#a8300f;letter-spacing:.04em;}
.ws-id__row{display:flex;gap:12px;align-items:center;min-width:0;}
.ws-id__flag{flex:none;width:44px;height:44px;display:grid;place-items:center;background:var(--ws-ink);clip-path:polygon(0 0,100% 0,100% 76%,76% 100%,0 100%);}
.ws-id__copy{min-width:0;}
.ws-id h2{margin:0;font-family:var(--dd-font-ui);font-size:clamp(24px,3.2cqi,32px);font-weight:900;line-height:1;color:var(--ws-ink);letter-spacing:-.01em;text-transform:none;overflow-wrap:anywhere;}
.ws-id p{margin:4px 0 0;font-size:13px;font-weight:700;color:#24315a;}
.ws-id p.is-soft{font-weight:600;color:#4b5a82;margin-top:1px;}
.ws-id__style{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;font-size:13px;color:#24315a;}
.ws-id__prod b{color:var(--ws-ink);font-weight:900;}
.ws-stamp{display:inline-block;padding:2px 8px;font-size:11px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:#fff;background:var(--ws-ink);transform:rotate(-2deg);}
.ws-pot__nums{display:flex;align-items:flex-end;gap:10px;}
.ws-pot small{color:#4b5a82;}
.ws-pot__now,.ws-pot__ceil{font-size:40px;font-weight:900;line-height:.95;font-variant-numeric:tabular-nums;color:var(--ws-ink);}
.ws-pot__ceil{color:#2f7d16;}
.ws-pot__arrow{flex:1;height:3px;margin-bottom:12px;background:var(--ws-ink);position:relative;}
.ws-pot__arrow::after{content:"";position:absolute;right:-2px;top:-5px;border-left:10px solid var(--ws-ink);border-top:6px solid transparent;border-bottom:6px solid transparent;}
.ws-rail{position:relative;height:12px;margin-top:6px;background:repeating-linear-gradient(90deg,rgba(7,18,51,.14) 0 calc(10% - 2px),transparent calc(10% - 2px) 10%);}
.ws-rail__now{position:absolute;top:0;bottom:0;left:0;background:var(--ws-ink);}
.ws-rail__ceil{position:absolute;top:-4px;bottom:-4px;border:3px solid #2f7d16;}
.ws-conf{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:4px 14px;}
.ws-conf li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 8px;align-items:center;font-size:11px;font-weight:800;color:#4b5a82;}
.ws-conf li i{display:flex;gap:2px;grid-row:2;}
.ws-conf li i b{width:16px;height:5px;background:rgba(7,18,51,.15);}
.ws-conf li i b.is-on{background:var(--ws-ink);}
.ws-conf li em{font-style:normal;color:var(--ws-ink);grid-row:2;}
.ws-conf__fallback{margin:0;font-size:12px;font-weight:800;color:#4b5a82;}
.ws-verdict{margin-top:auto;display:grid;grid-template-columns:auto minmax(0,1fr);background:var(--ws-orange);color:#2a1200;}
.ws-board{padding:12px 16px;background:#e2621a;clip-path:polygon(0 0,100% 0,calc(100% - 12px) 100%,0 100%);padding-right:24px;}
.ws-board small{color:#2a1200;}
.ws-board strong{display:block;font-family:var(--dd-font-display);font-size:38px;font-weight:400;line-height:1;color:#fff;-webkit-text-stroke:1px #2a1200;paint-order:stroke fill;}
.ws-board span{display:block;font-size:12px;font-weight:900;}
.ws-verdict__facts{margin:0;padding:10px 30px 10px 12px;display:grid;gap:6px;}
.ws .ws-verdict__facts dt{font-size:11px;font-weight:900;color:#4a1f02!important;opacity:1;text-transform:none;letter-spacing:0;}
.ws-verdict__facts dd{margin:0;font-size:15px;font-weight:900;color:#1a0b00;}
.ws-verdict__facts dd small{color:#5a2a05;font-weight:700;}
.ws-verdict__facts dd.is-up{color:#14460a;}
.ws-verdict__facts dd.is-down{color:#7a1206;}
.ws-right{padding:14px 4px 6px 22px;display:flex;flex-direction:column;gap:14px;min-width:0;}
.ws-sum{margin:0;font-size:16px;font-weight:800;line-height:1.35;color:#fff;}
.ws-matrix{position:relative;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:rgba(169,191,220,.16);padding-bottom:18px;}
.ws-matrix.is-goalie{grid-template-columns:repeat(3,minmax(0,1fr));}
.ws-matrix__key{position:absolute;right:0;bottom:0;font-size:11px;color:var(--ws-soft);display:inline-flex;align-items:center;gap:5px;}
.ws-matrix__key em{display:inline-block;width:2px;height:10px;background:#fff;}
.ws-cell{background:var(--ws-deep);padding:8px 10px;display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:baseline;gap:2px 6px;border-top:3px solid var(--ws-cream);}
.ws-cell.is-up{border-top-color:var(--ws-lime);}
.ws-cell.is-down{border-top-color:var(--ws-coral);}
.ws-cell.is-lock{border-top-color:var(--ws-slate);}
.ws-cell__lab{font-size:12px;font-weight:800;color:#d6e2f4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.ws-cell__val{font-size:22px;font-weight:900;font-variant-numeric:tabular-nums;color:#fff;white-space:nowrap;}
.ws-cell.is-up .ws-cell__val{color:var(--ws-lime);}
.ws-cell.is-down .ws-cell__val{color:var(--ws-coral);}
.ws-cell__bar{grid-column:1 / -1;position:relative;height:6px;background:rgba(169,191,220,.16);}
.ws-cell__bar b{position:absolute;top:0;bottom:0;background:var(--ws-cream);}
.ws-cell.is-up .ws-cell__bar b{background:var(--ws-lime);}
.ws-cell.is-down .ws-cell__bar b{background:var(--ws-coral);}
.ws-cell__bar em{position:absolute;top:-3px;bottom:-3px;width:2px;margin-left:-1px;background:#fff;}
.ws-cell small{grid-column:1 / -1;font-size:11px;font-weight:800;color:#d6e2f4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ws-cell small span.is-up{color:var(--ws-lime);}
.ws-cell small span.is-down{color:var(--ws-coral);}
.ws-cell small span.is-mid{color:var(--ws-blue);}
.ws-av{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.ws-find{padding:10px 12px;color:var(--ws-ink);}
.ws-find.is-up{background:var(--ws-lime);}
.ws-find.is-down{background:var(--ws-coral);color:#2a0702;}
.ws-find small{color:inherit;opacity:.8;}
.ws-find__kick{text-transform:uppercase;letter-spacing:.05em;font-weight:900;}
.ws-find strong{display:block;font-size:18px;font-weight:900;line-height:1.1;margin:2px 0;}
.ws-find p{margin:0;font-size:13px;font-weight:800;}
.ws-find p.ws-find__bench{font-weight:600;}
.ws-find__conf{margin-top:4px;}
.ws-find ul{list-style:none;margin:6px 0 0;padding:6px 0 0;border-top:1px solid rgba(7,18,51,.25);font-size:12px;font-weight:800;}
.ws-find ul span{font-weight:600;}
.ws-out__head{display:flex;justify-content:space-between;gap:8px;font-size:12px;font-weight:800;color:var(--ws-soft);margin-bottom:5px;}
.ws-out__head span:first-child{color:#fff;font-size:13px;}
.ws-out__bar{display:flex;height:16px;gap:2px;}
.ws-out__bar i{display:block;min-width:3px;}
.ws-out__bar i.is-top{outline:2px solid #fff;outline-offset:1px;}
.ws-out__legend{list-style:none;margin:6px 0 0;padding:0;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ws-soft);}
.ws-out__legend li{display:inline-flex;align-items:center;gap:5px;}
.ws-out__legend li b{color:#eef4ff;}
.ws-out__legend li.is-top{color:#fff;font-weight:900;}
.ws-out__legend li i{width:9px;height:9px;display:inline-block;}
.ws-out .is-down{background:var(--ws-coral);}
.ws-out .is-slate{background:var(--ws-slate);}
.ws-out .is-mid{background:var(--ws-cream);}
.ws-out .is-blue{background:var(--ws-blue);}
.ws-out .is-up{background:var(--ws-lime);}
.ws-evidence{margin-top:12px;border-top:2px solid var(--ws-cyan);}
.ws-evidence > summary{cursor:pointer;list-style:none;padding:8px 0;font-size:13px;font-weight:900;color:var(--ws-cyan);}
.ws-evidence > summary::-webkit-details-marker{display:none;}
.ws-evidence__grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 24px;padding-bottom:6px;}
.ws-evidence h4{margin:0 0 4px;font-size:13px;font-weight:900;color:#fff;text-transform:none;letter-spacing:0;}
.ws-evidence p{margin:0 0 4px;font-size:13px;}
.ws-evidence__foot{grid-column:1 / -1;}
.ws-est{display:flex;flex-wrap:wrap;gap:4px 14px;}
.ws-est b{color:var(--ws-blue);}
.ws-comps,.ws-why,.ws-traits{list-style:none;margin:0 0 4px;padding:0;display:grid;gap:4px;font-size:13px;}
.ws-comps small{font-weight:600;}
.ws-sim{display:inline-block;margin-left:6px;padding:0 5px;font-size:11px;font-weight:900;color:var(--ws-ink);background:var(--ws-cyan);}
.ws-why b{color:#fff;}
.ws-traits li{display:flex;justify-content:space-between;gap:8px;}
.ws-traits li.is-up b{color:var(--ws-lime);}
.ws-traits li.is-down b{color:var(--ws-coral);}
.ws-traits li.is-mid b{color:var(--ws-cream);}
.cs-more{margin-top:8px;}
.cs-more > summary{cursor:pointer;font-size:13px;font-weight:800;color:#3ee6ff;list-style:none;padding:6px 0;}
.cs-more > summary::-webkit-details-marker{display:none;}
.cs-more[open] > summary{margin-bottom:8px;}
@container (max-width:620px){
  .ws{grid-template-columns:minmax(0,1fr);}
  .ws-left{clip-path:none;}
  .ws-id{padding-right:18px;}
  .ws-verdict__facts{padding-right:14px;}
  .ws-right{padding:14px 0 6px;}
  .ws-evidence__grid{grid-template-columns:minmax(0,1fr);}
}
@container (max-width:460px){
  .ws-matrix{grid-template-columns:repeat(2,minmax(0,1fr));}
  .ws-av{grid-template-columns:1fr;}
  .ws-pot__now,.ws-pot__ceil{font-size:32px;}
}
@media (max-width:1100px){
  .dd-verdict,.dd-columns,.dd-split-notes,.dd-tiles,.dd-actions__row{grid-template-columns:1fr 1fr;}
  .dd-verdict{grid-template-columns:1fr;}
}
@media (max-width:720px){
  .dd-tiles,.dd-actions__row,.dd-columns,.dd-split-notes{grid-template-columns:1fr;}
}
`;
