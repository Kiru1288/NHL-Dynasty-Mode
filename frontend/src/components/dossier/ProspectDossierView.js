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
  return null;
}


