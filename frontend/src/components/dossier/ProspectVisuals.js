/**
 * Visual building blocks for draft-prospect dossiers (Entry Draft + Draft Class).
 *
 * The goal: a scout file you *see* instead of read.
 *  - OVR / Potential always come with how sure your staff is (certainty ring + a
 *    "could really be" band), so a thin file never looks like a finished one.
 *  - Skills show a dramatic tier (AWFUL → WORLD-CLASS) and where the kid ranks
 *    among this draft class (percentile bars + radar).
 *  - Class rank, NHL odds and career outcomes as dials / donuts.
 *
 * Pure SVG + CSS, no chart library.
 */

import React, { createContext, useContext } from "react";
import { createPortal } from "react-dom";

/* ------------------------------------------------------------------ */
/* Class pool context (for percentiles)                                 */
/* ------------------------------------------------------------------ */

/** Provide the whole draft class so dossiers can compute percentiles. */
export const ProspectClassContext = createContext([]);

export function useProspectClass() {
  const pool = useContext(ProspectClassContext);
  return Array.isArray(pool) ? pool : [];
}

/* ------------------------------------------------------------------ */
/* Scales                                                               */
/* ------------------------------------------------------------------ */

function n(v) {
  if (v == null || v === "") return null; // Number(null) is 0, which read as a real rating
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/** Dramatic skill tiers — the word should tell you how good, at a glance. */
export const SKILL_TIERS = [
  { min: 92, key: "world-class", label: "WORLD-CLASS", color: "#ff5fd2" },
  { min: 86, key: "elite", label: "ELITE", color: "#f6c453" },
  { min: 80, key: "great", label: "GREAT", color: "#52df94" },
  { min: 74, key: "very-good", label: "VERY GOOD", color: "#13d8e7" },
  { min: 66, key: "good", label: "GOOD", color: "#8ab4ff" },
  { min: 58, key: "average", label: "AVERAGE", color: "#9fb3c4" },
  { min: 50, key: "weak", label: "WEAK", color: "#ffb35c" },
  { min: 42, key: "bad", label: "BAD", color: "#ff7a5c" },
  { min: 0, key: "awful", label: "AWFUL", color: "#ff4d5e" },
];

export function skillTier(value) {
  const v = n(value);
  if (v == null || v <= 0) return { key: "unknown", label: "UNKNOWN", color: "#607789" };
  return SKILL_TIERS.find((t) => v >= t.min) || SKILL_TIERS[SKILL_TIERS.length - 1];
}

/** How sure the staff is about a number. */
export const CERTAINTY_TIERS = [
  { min: 90, key: "locked", label: "LOCKED IN", color: "#52df94" },
  { min: 75, key: "confident", label: "CONFIDENT", color: "#13d8e7" },
  { min: 55, key: "solid", label: "SOLID READ", color: "#8ab4ff" },
  { min: 35, key: "guess", label: "EDUCATED GUESS", color: "#f6c453" },
  { min: 15, key: "rumor", label: "RUMOR MILL", color: "#ffb35c" },
  { min: 0, key: "blind", label: "FLYING BLIND", color: "#ff606d" },
];

export function certaintyTier(pct) {
  const v = n(pct);
  if (v == null) return { key: "blind", label: "NO FILE", color: "#607789" };
  return CERTAINTY_TIERS.find((t) => v >= t.min) || CERTAINTY_TIERS[CERTAINTY_TIERS.length - 1];
}

/**
 * Certainty of the current-OVR and potential reads.
 * - scoutPct: your dedicated file (72% reveals the real OVR, 100% is a full file)
 * - publicConf: ambient league coverage (games watched by everyone)
 * Potential is a projection, so it trails OVR and never reaches 100% before he plays.
 */
export function prospectCertainty({ scoutPct, publicConf, revealed = false, abilityRevealed = false, ceilingHidden = false, rank = null } = {}) {
  const s = n(scoutPct) ?? 0;
  const pub = n(publicConf) ?? 0;
  const r = n(rank);
  let ovr;
  if (abilityRevealed) ovr = 100;
  else if (revealed || s >= 72) ovr = clamp(82 + (s - 72) * 0.64, 82, 100);
  else ovr = clamp(Math.max(s * 0.95, pub * 0.55) + (r != null && r <= 32 ? 8 : 0), 8, 70);
  let pot;
  if (ceilingHidden) pot = 5;
  else if (abilityRevealed) pot = 90;
  else pot = clamp(ovr * 0.8, 5, 85);
  return { ovr: Math.round(ovr), pot: Math.round(pot) };
}

/** How many OVR points either way the true value could plausibly sit. */
export function impliedSpread(certainty) {
  const c = n(certainty) ?? 0;
  return Math.max(0, Math.round(((100 - c) / 100) * 12));
}

/** Parse "64–68", "64-68", "66" or [64, 68] into {lo, hi, mid}. */
export function parseRange(label, lo = null, hi = null) {
  let a = n(lo);
  let b = n(hi);
  if ((a == null || b == null) && label != null) {
    const parts = String(label)
      .replace(/[^0-9.–-]/g, "")
      .split(/[–-]/)
      .map((x) => (String(x).trim() === "" ? null : n(String(x).trim())))
      .filter((x) => x != null);
    if (parts.length >= 2) [a, b] = [parts[0], parts[1]];
    else if (parts.length === 1) a = b = parts[0];
  }
  if (a == null && b == null) return null;
  if (a == null) a = b;
  if (b == null) b = a;
  if (a > b) [a, b] = [b, a];
  return { lo: a, hi: b, mid: (a + b) / 2 };
}

/** Percentile (0–100) of `value` within `values` (higher is better). */
export function percentileOf(values, value) {
  const v = n(value);
  const list = (values || []).map(n).filter((x) => x != null);
  if (v == null || list.length < 3) return null;
  const below = list.filter((x) => x < v).length;
  const equal = list.filter((x) => x === v).length;
  return Math.round(((below + equal * 0.5) / list.length) * 100);
}

export function percentileLabel(p) {
  if (p == null) return "—";
  const rounded = Math.round(p);
  if (rounded >= 50) return `Top ${Math.max(1, 100 - rounded)}%`;
  return `Bottom ${Math.max(1, rounded)}%`;
}

/* ------------------------------------------------------------------ */
/* Components                                                           */
/* ------------------------------------------------------------------ */

/** NHL-role ladder for a potential/ceiling number (calibrated to the NHL rating spread). */
export const PROJECTION_TIERS = [
  { min: 94, key: "franchise", label: "FRANCHISE", color: "#f6c453" },
  { min: 90, key: "elite", label: "FIRST LINE", color: "#52df94" },
  { min: 86, key: "great", label: "TOP SIX", color: "#13d8e7" },
  { min: 82, key: "very-good", label: "MIDDLE SIX", color: "#8ab4ff" },
  { min: 77, key: "good", label: "BOTTOM SIX", color: "#9fb3c4" },
  { min: 72, key: "average", label: "FRINGE NHL", color: "#ffb35c" },
  { min: 0, key: "weak", label: "LONG SHOT", color: "#ff7a5c" },
];

export function projectionTier(value) {
  const v = n(value);
  if (v == null || v <= 0) return { key: "unknown", label: "UNKNOWN", color: "#607789" };
  return PROJECTION_TIERS.find((t) => v >= t.min) || PROJECTION_TIERS[PROJECTION_TIERS.length - 1];
}

/** Class tier ladder colours (backend class_tier.key). */
const CLASS_TIER_COLORS = {
  transcendent: "#ffd24a",
  generational: "#ff6b2d",
  franchise: "#f6c453",
  blue_chip: "#52df94",
  first_line: "#13d8e7",
  first_round: "#8ab4ff",
  second_round: "#9fb3c4",
  mid_round: "#8096a8",
  late_flyer: "#a78b6a",
};

const PCT_LABELS = {
  scoring: "Scoring",
  nhle: "NHLe",
  size: "Size",
  skating: "Skating",
  shooting: "Shooting",
  hockey_iq: "Hockey IQ",
  defense: "Defense",
  physical: "Physical",
  save_pct: "Save %",
};

/**
 * Plain, common draft indicators: consensus rank, class tier, stat line, NHLe and
 * percentile ranks within the class (from backend class_indicators).
 */
export function ClassIndicators({ indicators }) {
  const ci = indicators && typeof indicators === "object" ? indicators : null;
  if (!ci) return null;
  const tier = ci.tier || {};
  const line = ci.stat_line || {};
  const pcts = ci.percentiles || {};
  const goalie = ci.position_group === "G";
  const gp = n(line.gp) ?? 0;
  const pm = n(line.plus_minus);
  const color = CLASS_TIER_COLORS[tier.key] || "#8ab4ff";
  const pctKeys = Object.keys(PCT_LABELS).filter((k) => pcts[k] != null);
  return (
    <section className="pv-class">
      <div className="pv-class__head">
        {tier.label ? <TierBadge tier={{ key: tier.key, label: tier.label, color }} /> : null}
        {ci.consensus_rank != null ? <strong>Consensus #{ci.consensus_rank}</strong> : null}
        {ci.class_percentile != null ? <span>{percentileLabel(ci.class_percentile)} of class</span> : null}
        {ci.nhle != null ? <span>NHLe {ci.nhle} pts / 82</span> : null}
      </div>
      <div className="pv-class__line">
        {line.league ? <span><b>LEAGUE</b>{line.league}</span> : null}
        <span><b>GP</b>{gp}</span>
        {goalie ? (
          <>
            <span><b>W</b>{line.wins ?? 0}</span>
            <span><b>SV%</b>{n(line.save_pct) ? Number(line.save_pct).toFixed(3) : "—"}</span>
            <span><b>GAA</b>{n(line.gaa) ? Number(line.gaa).toFixed(2) : "—"}</span>
          </>
        ) : (
          <>
            <span><b>G</b>{line.goals ?? 0}</span>
            <span><b>A</b>{line.assists ?? 0}</span>
            <span><b>P</b>{line.points ?? 0}</span>
            <span><b>+/-</b>{pm == null ? (gp > 0 ? 0 : "—") : pm > 0 ? `+${pm}` : pm}</span>
            <span><b>P/GP</b>{line.ppg != null ? Number(line.ppg).toFixed(2) : "—"}</span>
          </>
        )}
      </div>
      {pctKeys.length ? (
        <div className="pv-class__pcts">
          {pctKeys.map((k) => {
            const p = clamp(n(pcts[k]) ?? 0, 0, 100);
            const c = p >= 90 ? "#52df94" : p >= 65 ? "#13d8e7" : p >= 35 ? "#8ab4ff" : "#ffb35c";
            return (
              <div key={k} className="pv-class__pct" title={`${p}th percentile in this class`}>
                <span>{PCT_LABELS[k]}</span>
                <strong>{p}</strong>
                <i style={{ "--w": `${Math.max(2, p)}%`, "--c": c }} />
              </div>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

export function TierBadge({ value, tier: tierIn, small = false }) {
  const tier = tierIn || skillTier(value);
  return (
    <span
      className={`pv-tier pv-tier--${tier.key}${small ? " is-small" : ""}`}
      style={{ "--tier": tier.color }}
    >
      {tier.label}
    </span>
  );
}

/** Donut ring with a number in the middle. */
export function Ring({ pct, size = 64, stroke = 7, color = "#13d8e7", center, sub, title }) {
  const v = clamp(n(pct) ?? 0, 0, 100);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div className="pv-ring" style={{ width: size, height: size }} title={title}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(148,185,205,.14)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${(v / 100) * c} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ filter: `drop-shadow(0 0 4px ${color}66)`, transition: "stroke-dasharray .6s ease" }}
        />
      </svg>
      <div className="pv-ring__label">
        <strong>{center ?? `${Math.round(v)}%`}</strong>
        {sub ? <small>{sub}</small> : null}
      </div>
    </div>
  );
}

/**
 * OVR / Potential with certainty: the scouts' number (bright band), what it could
 * really be given how thin the file is (faint band), and a certainty ring.
 */
export function RatingCertaintyGauge({ title, valueText, range, certainty, tone = "cyan", hidden = false, hiddenText = "Ungraded" }) {
  const MIN = 40;
  const MAX = 99;
  const pos = (v) => `${((clamp(v, MIN, MAX) - MIN) / (MAX - MIN)) * 100}%`;
  const tier = certaintyTier(certainty);
  const color = tone === "gold" ? "#f6c453" : "#13d8e7";
  const r = range;
  const fogLo = r ? Math.round(Math.min(r.lo, 99)) : null;
  const fogHi = r ? Math.round(Math.min(r.hi, 99)) : null;
  return (
    <div className={`pv-gauge pv-gauge--${tone}`}>
      <div className="pv-gauge__head">
        <span>{title}</span>
        <strong style={{ color }}>{hidden ? hiddenText : valueText || "—"}</strong>
        {!hidden && r ? <TierBadge tier={tone === "gold" ? projectionTier(r.mid) : undefined} value={r.mid} small /> : null}
      </div>
      <div className="pv-gauge__body">
        <div className="pv-gauge__track" aria-hidden="true">
          {[50, 60, 70, 80, 90].map((t) => (
            <i key={t} className="pv-gauge__tick" style={{ left: pos(t) }}><b>{t}</b></i>
          ))}
          {!hidden && r ? (
            <>
              <span className="pv-gauge__fog" style={{ left: pos(fogLo), width: `calc(${pos(fogHi)} - ${pos(fogLo)})` }} />
              <span
                className="pv-gauge__band"
                style={{ left: pos(r.lo), width: `max(6px, calc(${pos(r.hi)} - ${pos(r.lo)}))`, background: color }}
              />
              <span className="pv-gauge__dot" style={{ left: pos(r.mid), background: color }} />
            </>
          ) : (
            <span className="pv-gauge__fog is-full" />
          )}
        </div>
        <Ring
          pct={hidden ? 0 : certainty}
          size={58}
          stroke={6}
          color={hidden ? "#607789" : tier.color}
          center={hidden ? "?" : `${Math.round(certainty ?? 0)}%`}
          title="How sure your staff is"
        />
      </div>
      <div className="pv-gauge__foot">
        <span className="pv-certainty" style={{ color: hidden ? "#607789" : tier.color }}>
          {hidden ? "NOT ON FILE" : tier.label}
        </span>
        <span>
          {hidden
            ? "Assign a scout to open this read."
            : r && Math.round(r.lo) !== Math.round(r.hi)
              ? `${Math.round(Math.min(r.lo, 99))}–${Math.round(Math.min(r.hi, 99))}`
              : "This is the real number."}
        </span>
      </div>
    </div>
  );
}

function ratingScale(v) {
  // Floor at 20 so weak ratings (a 38 character read) still draw a visible bar.
  return ((clamp(n(v) ?? 20, 20, 99) - 20) / 79) * 100;
}

const RADAR_SHORT = {
  offence: "OFF",
  defense: "DEF",
  defence: "DEF",
  transition: "TRN",
  mental: "MEN",
  physicality: "PHY",
  physical: "PHY",
  character: "CHR",
  glove: "GLV",
  blocker: "BLK",
  stick: "STK",
};

function radarShort(axis) {
  const key = String(axis?.label || "").toLowerCase();
  return RADAR_SHORT[key] || axis?.short || String(axis?.label || "").slice(0, 3).toUpperCase();
}

/** Skill rows: value, dramatic tier, and percentile vs this draft class. */
export function PercentileBars({ rows, title = "Skill profile", note, headless = false }) {
  const list = (rows || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <section className="pv-card">
      {headless ? null : (
        <header className="pv-card__head">
          <h4>{title}</h4>
          {note ? <span>{note}</span> : <span>Tier = league scale · % = vs. this draft class</span>}
        </header>
      )}
      <div className="pv-bars">
        {list.map((row) => {
          const p = row.percentile;
          const t = row.locked ? null : (row.tier || skillTier(row.value));
          const low = n(row.low);
          const high = n(row.high);
          const ranged = !row.locked && low != null && high != null && Math.round(low) !== Math.round(high);
          const valueText = row.locked
            ? "Scout to reveal"
            : (row.text || (n(row.value) != null ? String(Math.round(n(row.value))) : "—"));
          return (
            <div key={row.label} className={`pv-bar${row.locked ? " is-locked" : ""}${ranged || row.soft ? " is-soft" : ""}`}>
              <span className="pv-bar__label">{row.label}</span>
              <div className="pv-bar__track" title={p != null ? percentileLabel(p) : undefined}>
                {ranged ? (
                  <i
                    className="is-range"
                    style={{
                      left: `${ratingScale(Math.min(low, high))}%`,
                      width: `${Math.max(2, ratingScale(Math.max(low, high)) - ratingScale(Math.min(low, high)))}%`,
                      backgroundColor: t?.color,
                    }}
                  />
                ) : (
                  <i style={{ width: `${row.locked ? 0 : ratingScale(row.value)}%`, background: t?.color }} />
                )}
                {n(row.classAvg) != null ? <b style={{ left: `${ratingScale(row.classAvg)}%` }} /> : null}
              </div>
              <strong className="pv-bar__value">{valueText}</strong>
              {row.locked || !t ? <span className="pv-bar__pct">—</span> : <TierBadge tier={t} small />}
              <span className="pv-bar__pct">{row.locked ? "—" : percentileLabel(p)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/** Radar of up to 8 skills (0–100). */
export function SkillRadar({ axes, size = 260, color = "#3ec6ff", headless = false }) {
  const list = (axes || []).filter((a) => a && n(a.value) != null).slice(0, 8);
  if (list.length < 3) return null;
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 36;
  const pt = (i, v) => {
    const ang = (Math.PI * 2 * i) / list.length - Math.PI / 2;
    const rr = (clamp(v, 0, 100) / 100) * R;
    return [cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr];
  };
  const poly = list.map((a, i) => pt(i, n(a.value)).join(",")).join(" ");
  const avgPoly = list.map((a, i) => pt(i, n(a.classAvg) ?? 60).join(",")).join(" ");
  return (
    <section className="pv-card pv-radar">
      {headless ? null : (
        <header className="pv-card__head">
          <h4>Shape</h4>
          <span><i className="pv-key" style={{ background: color }} /> Him <i className="pv-key is-avg" /> Class avg</span>
        </header>
      )}
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ maxWidth: size }} role="img" aria-label="Skill radar">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <polygon
            key={f}
            points={list.map((_, i) => pt(i, f * 100).join(",")).join(" ")}
            fill="none"
            stroke="rgba(148,185,205,.14)"
          />
        ))}
        {list.map((a, i) => {
          const [x, y] = pt(i, 100);
          const [lx, ly] = pt(i, 118);
          return (
            <g key={a.label}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(148,185,205,.12)" />
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" className="pv-radar__label">
                {radarShort(a)}
              </text>
            </g>
          );
        })}
        <polygon points={avgPoly} fill="rgba(148,185,205,.06)" stroke="rgba(148,185,205,.45)" strokeDasharray="3 3" />
        <polygon points={poly} fill={`${color}33`} stroke={color} strokeWidth="2" style={{ filter: `drop-shadow(0 0 6px ${color}66)` }} />
        {list.map((a, i) => {
          const [x, y] = pt(i, n(a.value));
          return <circle key={`d-${a.label}`} cx={x} cy={y} r="3" fill={skillTier(a.value).color} />;
        })}
      </svg>
    </section>
  );
}

const OUTCOME_COLORS = ["#f6c453", "#52df94", "#13d8e7", "#8ab4ff", "#9fb3c4", "#ff7a5c", "#ff4d5e"];

/** Career outcome donut. segments: [{key,label,pct}] */
export function OutcomeDonut({ segments, title = "Career outcomes", size = 132, headless = false }) {
  const list = (segments || []).filter((s) => s && n(s.pct) > 0);
  if (!list.length) return null;
  const total = list.reduce((a, s) => a + n(s.pct), 0) || 1;
  const stroke = 16;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  let acc = 0;
  const top = list.reduce((a, b) => (n(b.pct) > n(a.pct) ? b : a), list[0]);
  return (
    <section className="pv-card pv-donut">
      {headless ? null : (
        <header className="pv-card__head">
          <h4>{title}</h4>
        </header>
      )}
      <div className="pv-donut__body">
        <div className="pv-ring" style={{ width: size, height: size }}>
          <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} role="img" aria-label="Career outcome distribution">
            {list.map((s, i) => {
              const frac = n(s.pct) / total;
              const dash = `${frac * c} ${c}`;
              const off = -acc * c;
              acc += frac;
              return (
                <circle
                  key={s.key || s.label}
                  cx={size / 2}
                  cy={size / 2}
                  r={r}
                  fill="none"
                  stroke={OUTCOME_COLORS[i % OUTCOME_COLORS.length]}
                  strokeWidth={stroke}
                  strokeDasharray={dash}
                  strokeDashoffset={off}
                  transform={`rotate(-90 ${size / 2} ${size / 2})`}
                >
                  <title>{`${s.label} ${Math.round(n(s.pct))}%`}</title>
                </circle>
              );
            })}
          </svg>
          <div className="pv-ring__label">
            <strong>{Math.round((n(top.pct) / total) * 100)}%</strong>
            <small>{String(top.label).toUpperCase()}</small>
          </div>
        </div>
        <ul className="pv-legend">
          {list.map((s, i) => (
            <li key={`l-${s.key || s.label}`}>
              <i style={{ background: OUTCOME_COLORS[i % OUTCOME_COLORS.length] }} />
              <span>{s.label}</span>
              <b>{Math.round((n(s.pct) / total) * 100)}%</b>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/** A row of headline dials: class rank, NHL odds, file depth, production. */
export function SnapshotDials({ items }) {
  const list = (items || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <div className="pv-dials">
      {list.map((it) => (
        <div key={it.label} className="pv-dial">
          <Ring pct={it.pct} size={74} stroke={7} color={it.color || "#13d8e7"} center={it.center} sub={it.sub} title={it.title} />
          <span>{it.label}</span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Styles                                                               */
/* ------------------------------------------------------------------ */

export function ProspectVisualStyles() {
  return <style>{PV_CSS}</style>;
}

const PV_CSS = `
.pv-tier{
  display:inline-block;padding:0;background:none;border:0;border-radius:0;
  font-family:var(--ed-head, inherit);font-size:11px;font-weight:800;letter-spacing:.08em;
  text-transform:uppercase;color:var(--tier);white-space:nowrap;line-height:1.1;
}
.pv-tier.is-small{font-size:11px;letter-spacing:.06em;}
.pv-tier--world-class,.pv-tier--franchise,.pv-tier--generational,.pv-tier--transcendent{text-shadow:0 0 8px color-mix(in srgb, var(--tier) 45%, transparent);}
.pv-class{display:grid;gap:10px;margin:10px 0;padding:10px 12px;border-top:1px solid rgba(128,150,168,.22);border-bottom:1px solid rgba(128,150,168,.22);}
.pv-class__head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;}
.pv-class__head strong{font-size:12px;letter-spacing:.08em;color:#e2edf6;}
.pv-class__head span{font-size:11px;color:#8096a8;}
.pv-class__line{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:#c9d6e2;font-variant-numeric:tabular-nums;}
.pv-class__line b{color:#8096a8;font-weight:600;margin-right:3px;font-size:11px;letter-spacing:.08em;}
.pv-class__pcts{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:6px 12px;}
.pv-class__pct{display:grid;grid-template-columns:1fr auto;gap:2px 6px;font-size:11px;color:#8096a8;letter-spacing:.06em;text-transform:uppercase;}
.pv-class__pct strong{color:#e2edf6;font-variant-numeric:tabular-nums;}
.pv-class__pct i{grid-column:1 / -1;height:3px;background:rgba(128,150,168,.18);position:relative;}
.pv-class__pct i::after{content:"";position:absolute;left:0;top:0;bottom:0;width:var(--w);background:var(--c);}
.pv-tier--awful,.pv-tier--bad{font-style:italic;}
.pv-tier--unknown{--tier:#607789;color:#8096a8;}
@keyframes pv-shimmer{0%,100%{filter:brightness(1)}50%{filter:brightness(1.35)}}

.pv-ring{position:relative;display:inline-grid;place-items:center;flex:none;}
.pv-ring svg{display:block;}
.pv-ring__label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;line-height:1;}
.pv-ring__label strong{font-family: var(--dd-font-ui, Nunito, sans-serif);font-size:19px;font-weight:800;color:#e9f7fb;}
.pv-ring__label small{margin-top:2px;font-size:11px;font-weight:800;letter-spacing:.06em;color:#8096a8;max-width:90%;}

.pv-gauge-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
.pv-gauge{border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:10px 12px;background:rgba(13, 37, 85, .62);box-shadow:inset 0 1px 0 rgba(255,255,255,.04);min-width:0;}
.pv-gauge__head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;}
.pv-gauge__head span{font-size:11px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8096a8;}
.pv-gauge__head strong{font-family: var(--dd-font-ui, Nunito, sans-serif);font-size:30px;line-height:1;font-weight:800;letter-spacing:.02em;}
.pv-gauge__body{display:flex;align-items:center;gap:12px;margin-top:6px;}
.pv-gauge__track{position:relative;flex:1;height:12px;border-radius:6px;background:rgba(148,185,205,.10);margin:10px 0 14px;}
.pv-gauge__tick{position:absolute;top:-3px;bottom:-3px;width:1px;background:rgba(148,185,205,.18);}
.pv-gauge__tick b{position:absolute;top:18px;left:-10px;width:22px;text-align:center;font-size:11px;font-weight:700;color:#607789;font-style:normal;}
.pv-gauge__fog{position:absolute;top:0;bottom:0;border-radius:6px;
  background:repeating-linear-gradient(135deg,rgba(246,196,83,.20) 0 4px,rgba(246,196,83,.06) 4px 8px);
  border:1px dashed rgba(246,196,83,.45);}
.pv-gauge__fog.is-full{left:0;right:0;background:repeating-linear-gradient(135deg,rgba(96,119,137,.25) 0 4px,transparent 4px 8px);border-color:rgba(96,119,137,.4);}
.pv-gauge__band{position:absolute;top:2px;bottom:2px;border-radius:5px;opacity:.9;box-shadow:0 0 10px rgba(46, 230, 214, .35);}
.pv-gauge--gold .pv-gauge__band{box-shadow:0 0 10px rgba(246,196,83,.4);}
.pv-gauge__dot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;border:2px solid #0c214d;}
.pv-gauge__foot{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:11.5px;color:#8096a8;}
.pv-certainty{font-weight:1000;letter-spacing:.14em;font-size:11px;}

.pv-card{border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:10px 12px;background:rgba(13, 37, 85, .62);box-shadow:inset 0 1px 0 rgba(255,255,255,.04);min-width:0;}
.pv-card__head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px;}
.pv-card__head h4{margin:0;font-family: var(--dd-font-ui, Nunito, sans-serif);font-size:15px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#e9f7fb;}
.pv-card__head span{font-size:11px;color:#8096a8;letter-spacing:.06em;text-transform:uppercase;font-weight:800;display:inline-flex;align-items:center;gap:4px;}
.pv-key{display:inline-block;width:10px;height:3px;border-radius:2px;}
.pv-key.is-avg{background:rgba(148,185,205,.6);}

.pv-bars{display:flex;flex-direction:column;gap:8px;}
.pv-bar{display:grid;grid-template-columns:110px minmax(0,1fr) minmax(64px,max-content) max-content max-content;gap:8px;align-items:center;}
.pv-bar__label{font-size:14px;font-weight:800;color:#c9dce6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pv-bar__track{position:relative;height:9px;border-radius:5px;background:rgba(148,185,205,.10);overflow:visible;}
.pv-bar__track i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;transition:width .6s ease;}
.pv-bar__track i.is-range{background-image:repeating-linear-gradient(135deg, rgba(255,255,255,.35) 0 3px, transparent 3px 6px);}
.pv-bar__track b{position:absolute;top:-3px;bottom:-3px;width:2px;margin-left:-1px;background:rgba(233,247,251,.7);}
.pv-bar__value{font-family: var(--dd-font-ui, Nunito, sans-serif);font-size:14px;font-weight:800;text-align:right;color:#e9f7fb;white-space:nowrap;font-variant-numeric:tabular-nums;min-width:64px;}
.pv-bar__pct{font-size:11px;font-weight:800;letter-spacing:.04em;color:#9eb4c6;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;}
.pv-bar.is-soft .pv-bar__value{color:#f6c453;}
.pv-bar.is-locked .pv-bar__label{color:#607789;}

.pv-radar svg{display:block;margin:0 auto;min-height:220px;}
.pv-radar__label{font-size:11px;font-weight:800;letter-spacing:.04em;fill:#c9dce6;text-transform:uppercase;}

.pv-donut__body{display:flex;align-items:center;gap:16px;flex-wrap:wrap;justify-content:flex-start;}
.pv-legend{list-style:none;margin:0;padding:0;display:grid;gap:6px;flex:0 1 auto;min-width:0;}
.pv-legend li{display:flex;align-items:center;justify-content:flex-start;gap:8px;font-size:14px;color:#c9dce6;}
.pv-legend i{width:10px;height:10px;border-radius:3px;flex:none;}
.pv-legend span{white-space:nowrap;}
.pv-legend b{font-family: var(--dd-font-ui, Nunito, sans-serif);font-size:14px;font-variant-numeric:tabular-nums;}

.pv-dials{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:8px;}
.pv-dial{display:flex;flex-direction:column;align-items:center;gap:4px;border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:8px 4px;background:rgba(13, 37, 85, .62);}
.pv-dial > span{font-size:11px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#8096a8;text-align:center;}

.pv-visual-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(200px,1fr);gap:10px;align-items:start;}
.pv-stack{display:flex;flex-direction:column;gap:10px;min-width:0;}

/* Board certainty chip */
.pv-cert-chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:1000;letter-spacing:.1em;white-space:nowrap;}
.pv-cert-chip i{display:inline-block;width:26px;height:5px;border-radius:3px;background:rgba(148,185,205,.15);position:relative;overflow:hidden;}
.pv-cert-chip i::after{content:"";position:absolute;left:0;top:0;bottom:0;width:var(--w);background:var(--c);}

@media (max-width: 760px){
  .pv-gauge-pair,.pv-visual-grid{grid-template-columns:1fr;}
  .pv-bar{grid-template-columns:92px minmax(0,1fr) max-content;}
  .pv-bar__pct{display:none;}
}
@media (prefers-reduced-motion: reduce){
  .pv-tier--generational{animation:none;}
}
`;

/** Compact certainty chip for board rows. */
export function CertaintyChip({ certainty }) {
  const t = certaintyTier(certainty);
  const short = { locked: "LOCKED", confident: "SURE", solid: "SOLID", guess: "GUESS", rumor: "RUMOR", blind: "BLIND" }[t.key] || t.label;
  return (
    <span className="pv-cert-chip" style={{ color: t.color }} title={`${t.label} — ${Math.round(n(certainty) ?? 0)}% certain`}>
      <i style={{ "--w": `${clamp(n(certainty) ?? 0, 0, 100)}%`, "--c": t.color }} />
      {short}
    </span>
  );
}


/* ------------------------------------------------------------------ */
/* Box scores + player game logs (Calendar + Player Dossier)            */
/* Styles: components/dossier/PlayerDossier.css (bxs-*), drawn in the  */
/* Entry Draft broadcast language (ops navy panels, gold numbers).      */
/* ------------------------------------------------------------------ */

const fmtClock = (sec) => {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const fmtPct = (v, digits = 1) => (v == null || !Number.isFinite(Number(v)) ? "—" : `${(Number(v) * 100).toFixed(digits)}%`);
const fmtSv = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(3).replace(/^0/, ""));
const fmtSigned = (v) => (Number(v) > 0 ? `+${v}` : String(v ?? 0));
const fmtNum = (v, digits = 0) => (v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(digits));

function BoxSkaterTable({ rows, onOpenPlayer }) {
  if (!rows || !rows.length) return <p className="bxs-empty">No skater lines stored.</p>;
  return (
    <div className="bxs-table-wrap">
      <table className="bxs-table">
        <thead>
          <tr>
            <th className="bxs-l">Skater</th>
            <th>G</th>
            <th>A</th>
            <th>P</th>
            <th>+/-</th>
            <th>PIM</th>
            <th>SOG</th>
            <th>HIT</th>
            <th>BLK</th>
            <th title="Faceoffs won-lost">FO</th>
            <th title="Takeaways">TK</th>
            <th title="Giveaways">GV</th>
            <th>CF%</th>
            <th>TOI</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.player_id} className={r.pts > 0 ? "is-scored" : ""}>
              <td className="bxs-l">
                {onOpenPlayer ? (
                  <button type="button" className="bxs-link" onClick={() => onOpenPlayer(r.player_id)}>
                    {r.name}
                  </button>
                ) : (
                  r.name
                )}
                <em>{r.pos}</em>
              </td>
              <td>{r.g}</td>
              <td>{r.a}</td>
              <td className="bxs-strong">{r.pts}</td>
              <td className={r.pm > 0 ? "bxs-good" : r.pm < 0 ? "bxs-bad" : ""}>{fmtSigned(r.pm)}</td>
              <td>{r.pim}</td>
              <td>{r.sog}</td>
              <td>{r.hit}</td>
              <td>{r.blk}</td>
              <td title={r.fo_pct != null ? fmtPct(r.fo_pct, 0) : undefined}>{r.fow + r.fol ? `${r.fow}-${r.fol}` : "—"}</td>
              <td>{r.tk ?? 0}</td>
              <td>{r.gv ?? 0}</td>
              <td>{fmtPct(r.cf_pct, 0)}</td>
              <td>{fmtClock(r.toi_sec)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BoxGoalieTable({ rows }) {
  if (!rows || !rows.length) return null;
  return (
    <div className="bxs-table-wrap">
      <table className="bxs-table">
        <thead>
          <tr>
            <th className="bxs-l">Goalie</th>
            <th>DEC</th>
            <th>SA</th>
            <th>SV</th>
            <th>GA</th>
            <th>SV%</th>
            <th>TOI</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((g) => (
            <tr key={g.player_id}>
              <td className="bxs-l">
                {g.name}
                {g.shutout ? <em className="bxs-tag">SO</em> : null}
              </td>
              <td>{g.decision || "—"}</td>
              <td>{g.sa}</td>
              <td>{g.sv}</td>
              <td>{g.ga}</td>
              <td className="bxs-strong">{fmtSv(g.sv_pct)}</td>
              <td>{fmtClock(g.toi_sec)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Full box score sheet. Pass `box` directly or a `gameId` to fetch. */
export function BoxScoreView({ box }) {
  const [side, setSide] = React.useState("away");
  if (!box) return null;
  const { home, away } = box;
  const labels = box.period_labels || ["1st", "2nd", "3rd"];
  const teamName = (tid) => (String(tid) === String(home.id) ? home.abbr || home.name : away.abbr || away.name);
  const teamStats = [
    ["Shots", away.sog, home.sog],
    ["Shot attempts", away.shot_attempts, home.shot_attempts],
    ["xG", fmtNum(away.xg, 2), fmtNum(home.xg, 2)],
    ["Power play", `${away.pp_goals}/${away.ppo}`, `${home.pp_goals}/${home.ppo}`],
    ["PIM", away.pim, home.pim],
    ["Hits", away.hits, home.hits],
    ["Blocks", away.blocks, home.blocks],
    [
      "Faceoffs",
      away.faceoffs ? `${away.faceoffs.won} (${fmtPct(away.faceoffs.pct, 0)})` : "—",
      home.faceoffs ? `${home.faceoffs.won} (${fmtPct(home.faceoffs.pct, 0)})` : "—",
    ],
    ["Takeaways", away.takeaways ?? "—", home.takeaways ?? "—"],
    ["Giveaways", away.giveaways ?? "—", home.giveaways ?? "—"],
  ];
  const byPeriod = {};
  (box.goals || []).forEach((g) => {
    const key = g.period ? (g.period >= 4 ? "OT" : labels[g.period - 1] || `P${g.period}`) : "Scoring";
    (byPeriod[key] = byPeriod[key] || []).push(g);
  });
  const sel = side === "home" ? home : away;
  return (
    <div className="bxs-sheet">
      <section className="bxs-hero">
        {[away, home].map((t, i) => (
          <div key={t.id} className={`bxs-hero-team ${t.goals > (i ? away.goals : home.goals) ? "is-winner" : ""}`}>
            <span className="bxs-hero-abbr">{t.abbr || t.id}</span>
            <span className="bxs-hero-name">{t.name}</span>
            <strong className="bxs-hero-score">{t.goals}</strong>
          </div>
        ))}
        <div className="bxs-hero-meta">
          <span>{box.shootout ? "Final / SO" : box.overtime ? "Final / OT" : "Final"}</span>
          <span>{box.date}</span>
          {box.scope && box.scope !== "regular_season" ? <span>{String(box.scope).replace(/_/g, " ")}</span> : null}
        </div>
      </section>

      <section className="bxs-grid-2">
        <div className="bxs-card">
          <h4>Scoring by period</h4>
          <table className="bxs-table bxs-compact">
            <thead>
              <tr>
                <th className="bxs-l">Team</th>
                {labels.map((l) => (
                  <th key={l}>{l}</th>
                ))}
                <th>T</th>
              </tr>
            </thead>
            <tbody>
              {[away, home].map((t) => (
                <tr key={t.id}>
                  <td className="bxs-l">{t.abbr || t.name}</td>
                  {labels.map((l, i) => (
                    <td key={l}>{t.periods?.[i] ?? 0}</td>
                  ))}
                  <td className="bxs-strong">{t.goals}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4>Shots by period</h4>
          <table className="bxs-table bxs-compact">
            <tbody>
              {[away, home].map((t) => (
                <tr key={t.id}>
                  <td className="bxs-l">{t.abbr || t.name}</td>
                  {labels.map((l, i) => (
                    <td key={l}>{t.shots?.[i] ?? 0}</td>
                  ))}
                  <td className="bxs-strong">{t.sog}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bxs-card">
          <h4>Team stats</h4>
          <div className="bxs-teamstats">
            <div className="bxs-teamstats-head">
              <span>{away.abbr || away.name}</span>
              <span />
              <span>{home.abbr || home.name}</span>
            </div>
            {teamStats.map(([label, a, h]) => (
              <div key={label} className="bxs-teamstats-row">
                <b>{a}</b>
                <span>{label}</span>
                <b>{h}</b>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="bxs-card">
        <h4>Scoring summary</h4>
        {!box.event_detail ? (
          <p className="bxs-note">Simmed in bulk: scorers are recorded, goal times are not.</p>
        ) : null}
        {Object.keys(byPeriod).length === 0 ? <p className="bxs-empty">No goals.</p> : null}
        {Object.entries(byPeriod).map(([period, goals]) => (
          <div key={period} className="bxs-period">
            <div className="bxs-period-label">{period}</div>
            {goals.map((g, i) => (
              <div key={i} className="bxs-goal">
                <span className="bxs-goal-time">{g.clock || "—"}</span>
                <span className="bxs-goal-team">{teamName(g.team_id)}</span>
                <span className="bxs-goal-who">
                  <strong>{g.scorer?.name}</strong>
                  {g.assists?.length ? <small> ({g.assists.map((a) => a.name).join(", ")})</small> : <small> (unassisted)</small>}
                </span>
                <span className="bxs-goal-tags">
                  {g.strength && g.strength !== "EV" ? <em className="bxs-tag">{g.strength}</em> : null}
                  {g.empty_net ? <em className="bxs-tag">EN</em> : null}
                  {g.overtime ? <em className="bxs-tag">OT</em> : null}
                  {g.gwg ? <em className="bxs-tag is-gold">GWG</em> : null}
                  {g.tag ? (
                    <em className="bxs-tag is-gold" title="Rink oddity">
                      {{
                        goalie_goal: "GOALIE GOAL",
                        penalty_shot: "PENALTY SHOT",
                        fluke_ref_skate: "OFF REF'S SKATE",
                        fluke_broken_stick: "BROKEN STICK",
                        fluke_boards: "BOARDS BOUNCE",
                      }[g.tag] || "FLUKE"}
                    </em>
                  ) : null}
                </span>
                {g.score ? <span className="bxs-goal-score">{g.score}</span> : null}
              </div>
            ))}
          </div>
        ))}
      </section>

      {box.fun_events?.length ? (
        <section className="bxs-card">
          <h4>Rink oddities</h4>
          {box.fun_events.map((ev, i) => (
            <p key={`${ev.kind}-${i}`} className="bxs-note">
              <strong>{ev.headline}</strong> {ev.text}
            </p>
          ))}
        </section>
      ) : null}

      {box.stars?.length ? (
        <section className="bxs-card">
          <h4>Three stars</h4>
          <div className="bxs-stars">
            {box.stars.map((s) => (
              <div key={s.player_id} className="bxs-star">
                <span className="bxs-star-rank">{"★".repeat(s.rank)}</span>
                <strong>{s.name}</strong>
                <small>
                  {teamName(s.team_id)} · {s.line}
                </small>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="bxs-card">
        <div className="bxs-tabs">
          {[
            ["away", away],
            ["home", home],
          ].map(([k, t]) => (
            <button key={k} type="button" className={side === k ? "is-active" : ""} onClick={() => setSide(k)}>
              {t.name}
            </button>
          ))}
        </div>
        <BoxSkaterTable rows={box.skaters?.[side]} />
        <BoxGoalieTable rows={box.goalies?.[side]} />
        <p className="bxs-note">{sel.name}: {sel.hits} hits, {sel.blocks} blocks.</p>
      </section>
    </div>
  );
}

/** Modal wrapper — fetches `/api/franchise/box-score/{gameId}`. */
export function BoxScoreModal({ gameId, onClose, fetchBox }) {
  const [box, setBox] = React.useState(null);
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    let live = true;
    setBox(null);
    setError("");
    if (!gameId) return undefined;
    Promise.resolve(fetchBox(gameId))
      .then((b) => live && setBox(b))
      .catch((e) => live && setError(e?.response?.status === 404 ? "No box score stored for this game." : "Could not load the box score."));
    return () => {
      live = false;
    };
  }, [gameId, fetchBox]);
  React.useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose?.();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  if (!gameId) return null;
  const modal = (
    <div className="bxs-backdrop" role="presentation" onClick={onClose}>
      <div className="bxs-modal" role="dialog" aria-modal="true" aria-label="Box score" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <p>Box score</p>
            <h2>{box ? `${box.away.name} @ ${box.home.name}` : "Loading…"}</h2>
          </div>
          <button type="button" className="bxs-close" onClick={onClose} aria-label="Close box score">
            ×
          </button>
        </header>
        <div className="bxs-modal-body">
          {error ? <p className="bxs-empty">{error}</p> : box ? <BoxScoreView box={box} /> : <p className="bxs-empty">Loading…</p>}
        </div>
      </div>
    </div>
  );
  return typeof document !== "undefined" && document.body ? createPortal(modal, document.body) : modal;
}

function PerfTile({ label, value, tone }) {
  return (
    <div className={`bxs-tile ${tone ? `is-${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/**
 * Performance tab body: season totals (counting + analytics), game log, WJC.
 * `log` is the `/game-log` payload; `onOpenGame(gameId)` opens a box score.
 */
export function PlayerPerformanceLog({ log, onOpenGame, scope: scopeProp }) {
  const totalsByScope = log?.totals || {};
  const scopes = Object.keys(totalsByScope);
  const [scope, setScope] = React.useState(scopeProp || "regular_season");
  const activeScope = totalsByScope[scope] ? scope : scopes[0];
  const t = totalsByScope[activeScope] || null;
  const games = (log?.games || []).filter((g) => !activeScope || (g.scope || "regular_season") === activeScope);
  const L = log?.season_ledger || {};
  const goalie = Boolean(log?.is_goalie);
  const xgf = Number(L.xgf || 0);
  const xga = Number(L.xga || 0);
  const wjc = log?.wjc || [];
  return (
    <div className="bxs-perf">
      {scopes.length > 1 ? (
        <div className="bxs-tabs">
          {scopes.map((s) => (
            <button key={s} type="button" className={s === activeScope ? "is-active" : ""} onClick={() => setScope(s)}>
              {s.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      ) : null}
      {t ? (
        goalie ? (
          <div className="bxs-tiles">
            <PerfTile label="GP / GS" value={`${t.gp} / ${t.gs}`} />
            <PerfTile label="W-L-OTL" value={`${t.w}-${t.l}-${t.otl}`} />
            <PerfTile label="SV%" value={fmtSv(t.sv_pct)} tone="gold" />
            <PerfTile label="GAA" value={fmtNum(t.gaa, 2)} />
            <PerfTile label="SA / SV" value={`${t.sa} / ${t.sv}`} />
            <PerfTile label="SO" value={t.so} />
            <PerfTile label="QS / QS%" value={`${t.qs} / ${fmtPct(t.qs_pct, 0)}`} />
            <PerfTile label="TOI" value={fmtClock(t.toi_sec)} />
            <PerfTile label="xGA" value={L.goalie_xga != null ? fmtNum(L.goalie_xga, 1) : "—"} />
            <PerfTile
              label="GSAx"
              value={L.goalie_xga != null && L.ga != null ? fmtNum(Number(L.goalie_xga) - Number(L.ga), 1) : "—"}
            />
            <PerfTile label="Stars" value={t.stars} />
          </div>
        ) : (
          <>
            <h4 className="bxs-h">Counting</h4>
            <div className="bxs-tiles">
              <PerfTile label="GP" value={t.gp} />
              <PerfTile label="G" value={t.g} tone="gold" />
              <PerfTile label="A" value={t.a} />
              <PerfTile label="PTS" value={t.pts} tone="gold" />
              <PerfTile label="+/-" value={fmtSigned(t.pm)} tone={t.pm > 0 ? "good" : t.pm < 0 ? "bad" : ""} />
              <PerfTile label="PIM" value={t.pim} />
              <PerfTile label="PPG / PPP" value={`${t.ppg} / ${t.ppp}`} />
              <PerfTile label="SHG / SHP" value={`${t.shg} / ${t.shp}`} />
              <PerfTile label="EVG / EVP" value={`${t.evg} / ${t.evp}`} />
              <PerfTile label="GWG" value={t.gwg} />
              <PerfTile label="SOG" value={t.sog} />
              <PerfTile label="SH%" value={fmtPct(t.sh_pct)} />
              <PerfTile label="Hits" value={t.hit} />
              <PerfTile label="Blocks" value={t.blk} />
              <PerfTile label="FO%" value={t.fo_pct != null ? `${fmtPct(t.fo_pct)} (${t.fow}-${t.fol})` : "—"} />
              <PerfTile label="TK / GV" value={`${t.tk ?? 0} / ${t.gv ?? 0}`} />
              <PerfTile label="Multi-pt games" value={t.multi_point_games} />
              <PerfTile label="Best pt streak" value={t.longest_point_streak} />
            </div>
            <h4 className="bxs-h">Analytics &amp; usage</h4>
            <div className="bxs-tiles">
              <PerfTile label="TOI/GP" value={fmtClock(t.toi_per_gp)} />
              <PerfTile label="EV TOI/GP" value={fmtClock(t.ev_toi_per_gp)} />
              <PerfTile label="PP TOI/GP" value={fmtClock(t.pp_toi_per_gp)} />
              <PerfTile label="SH TOI/GP" value={fmtClock(t.pk_toi_per_gp)} />
              <PerfTile label="CF%" value={fmtPct(t.cf_pct)} tone={t.cf_pct >= 0.52 ? "good" : t.cf_pct && t.cf_pct < 0.48 ? "bad" : ""} />
              <PerfTile label="ixG" value={fmtNum(t.ixg, 1)} />
              <PerfTile label="G - ixG" value={fmtNum(t.g - t.ixg, 1)} />
              <PerfTile label="xGF%" value={xgf + xga > 0 ? fmtPct(xgf / (xgf + xga)) : "—"} />
              <PerfTile label="On-ice xGF / xGA" value={xgf + xga > 0 ? `${fmtNum(xgf, 1)} / ${fmtNum(xga, 1)}` : "—"} />
              <PerfTile label="P/60" value={fmtNum(t.p60, 2)} />
              <PerfTile label="A1 / A2" value={L.primary_assists != null ? `${L.primary_assists} / ${L.secondary_assists ?? 0}` : "—"} />
              <PerfTile label="Stars" value={t.stars} />
            </div>
          </>
        )
      ) : (
        <p className="bxs-empty">No stored games this season yet. Lines appear as games are played.</p>
      )}

      {games.length ? (
        <>
          <h4 className="bxs-h">Game log · {log.season}</h4>
          <div className="bxs-table-wrap bxs-log">
            <table className="bxs-table">
              <thead>
                {goalie ? (
                  <tr>
                    <th className="bxs-l">Date</th>
                    <th className="bxs-l">Opp</th>
                    <th className="bxs-l">Result</th>
                    <th>DEC</th>
                    <th>SA</th>
                    <th>SV</th>
                    <th>GA</th>
                    <th>SV%</th>
                    <th>TOI</th>
                  </tr>
                ) : (
                  <tr>
                    <th className="bxs-l">Date</th>
                    <th className="bxs-l">Opp</th>
                    <th className="bxs-l">Result</th>
                    <th>G</th>
                    <th>A</th>
                    <th>P</th>
                    <th>+/-</th>
                    <th>PIM</th>
                    <th>SOG</th>
                    <th>PPP</th>
                    <th>HIT</th>
                    <th>BLK</th>
                    <th title="Faceoffs won-lost">FO</th>
                    <th title="Takeaways / giveaways">TK/GV</th>
                    <th>CF%</th>
                    <th>ixG</th>
                    <th>TOI</th>
                  </tr>
                )}
              </thead>
              <tbody>
                {[...games].reverse().map((g) => (
                  <tr
                    key={g.game_id}
                    className={`${onOpenGame ? "is-clickable" : ""} ${g.pts >= 2 || g.shutout ? "is-scored" : ""}`}
                    onClick={onOpenGame ? () => onOpenGame(g.game_id) : undefined}
                    title={onOpenGame ? "Open box score" : undefined}
                  >
                    <td className="bxs-l">{String(g.date || "").slice(5)}</td>
                    <td className="bxs-l">
                      {g.home ? "vs" : "@"} {g.opp_name}
                    </td>
                    <td className={`bxs-l ${g.team_result === "W" ? "bxs-good" : "bxs-bad"}`}>{g.result}</td>
                    {goalie ? (
                      <>
                        <td>{g.decision}</td>
                        <td>{g.sa}</td>
                        <td>{g.sv}</td>
                        <td>{g.ga}</td>
                        <td className="bxs-strong">{fmtSv(g.sv_pct)}</td>
                        <td>{fmtClock(g.toi_sec)}</td>
                      </>
                    ) : (
                      <>
                        <td>
                          {g.g}
                          {g.gwg ? <em className="bxs-tag is-gold">GW</em> : null}
                        </td>
                        <td>{g.a}</td>
                        <td className="bxs-strong">{g.pts}</td>
                        <td className={g.pm > 0 ? "bxs-good" : g.pm < 0 ? "bxs-bad" : ""}>{fmtSigned(g.pm)}</td>
                        <td>{g.pim}</td>
                        <td>{g.sog}</td>
                        <td>{g.ppg + g.ppa}</td>
                        <td>{g.hit}</td>
                        <td>{g.blk}</td>
                        <td>{g.fow + g.fol ? `${g.fow}-${g.fol}` : "—"}</td>
                        <td>{`${g.tk ?? 0}/${g.gv ?? 0}`}</td>
                        <td>{fmtPct(g.cf_pct, 0)}</td>
                        <td>{fmtNum(g.ixg, 2)}</td>
                        <td>{fmtClock(g.toi_sec)}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      {wjc.length ? (
        <>
          <h4 className="bxs-h">World Junior Championship</h4>
          <div className="bxs-table-wrap">
            <table className="bxs-table">
              <thead>
                <tr>
                  <th className="bxs-l">Year</th>
                  <th className="bxs-l">Team</th>
                  <th className="bxs-l">Result</th>
                  <th>GP</th>
                  {goalie ? (
                    <>
                      <th>W-L</th>
                      <th>SV%</th>
                      <th>SO</th>
                    </>
                  ) : (
                    <>
                      <th>G</th>
                      <th>A</th>
                      <th>P</th>
                      <th>+/-</th>
                    </>
                  )}
                  <th className="bxs-l">Grade</th>
                </tr>
              </thead>
              <tbody>
                {wjc.map((w) => (
                  <tr key={w.year}>
                    <td className="bxs-l">{w.year}</td>
                    <td className="bxs-l">{w.team || "—"}</td>
                    <td className="bxs-l">{w.result || "—"}</td>
                    <td>{w.gp}</td>
                    {goalie ? (
                      <>
                        <td>{`${w.w ?? 0}-${w.l ?? 0}`}</td>
                        <td>{w.sv_pct != null ? fmtSv(Number(w.sv_pct) > 1 ? Number(w.sv_pct) / 100 : w.sv_pct) : "—"}</td>
                        <td>{w.shutouts ?? 0}</td>
                      </>
                    ) : (
                      <>
                        <td>{w.goals}</td>
                        <td>{w.assists}</td>
                        <td className="bxs-strong">{w.points}</td>
                        <td>{fmtSigned(w.plus_minus ?? 0)}</td>
                      </>
                    )}
                    <td className="bxs-l">{w.performance_grade ? String(w.performance_grade).replace(/_/g, " ") : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {wjc.filter((w) => w.summary).map((w) => (
            <p key={`s${w.year}`} className="bxs-note">
              {w.year}: {w.summary}
            </p>
          ))}
        </>
      ) : null}
    </div>
  );
}
