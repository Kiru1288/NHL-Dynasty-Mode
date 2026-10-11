/**
 * Visual building blocks for draft-prospect dossiers (Entry Draft + Draft Class).
 *
 * The goal: a scout file you *see* instead of read.
 *  - OVR / Potential always come with how sure your staff is (certainty ring + a
 *    "could really be" band), so a thin file never looks like a finished one.
 *  - Skills show a dramatic tier (AWFUL → GENERATIONAL) and where the kid ranks
 *    among this draft class (percentile bars + radar).
 *  - Class rank, NHL odds and career outcomes as dials / donuts.
 *
 * Pure SVG + CSS, no chart library.
 */

import React, { createContext, useContext } from "react";

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
  { min: 92, key: "generational", label: "GENERATIONAL", color: "#ff5fd2" },
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
  if (p >= 99) return "TOP 1%";
  if (p >= 50) return `TOP ${Math.max(1, 100 - p)}%`;
  return `BOTTOM ${Math.max(1, p)}%`;
}

/* ------------------------------------------------------------------ */
/* Components                                                           */
/* ------------------------------------------------------------------ */

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
  const spread = impliedSpread(certainty);
  const color = tone === "gold" ? "#f6c453" : "#13d8e7";
  const r = range;
  const fogLo = r ? Math.min(r.lo, Math.round(r.mid) - spread) : null;
  const fogHi = r ? Math.max(r.hi, Math.round(r.mid) + spread) : null;
  return (
    <div className={`pv-gauge pv-gauge--${tone}`}>
      <div className="pv-gauge__head">
        <span>{title}</span>
        <strong style={{ color }}>{hidden ? hiddenText : valueText || "—"}</strong>
        {!hidden && r ? <TierBadge value={r.mid} small /> : null}
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
            : spread > 0 && r
              ? `Could really be ${fogLo}–${fogHi}`
              : "This is the real number."}
        </span>
      </div>
    </div>
  );
}

/** Skill rows: value, dramatic tier, and percentile vs this draft class. */
export function PercentileBars({ rows, title = "Skill profile", note }) {
  const list = (rows || []).filter(Boolean);
  if (!list.length) return null;
  return (
    <section className="pv-card">
      <header className="pv-card__head">
        <h4>{title}</h4>
        {note ? <span>{note}</span> : <span>vs. this draft class</span>}
      </header>
      <div className="pv-bars">
        {list.map((row) => {
          const p = row.percentile;
          const t = row.locked ? null : skillTier(row.value);
          return (
            <div key={row.label} className={`pv-bar${row.locked ? " is-locked" : ""}${row.soft ? " is-soft" : ""}`}>
              <span className="pv-bar__label">{row.label}</span>
              <div className="pv-bar__track" title={p != null ? `${p}th percentile in the class` : undefined}>
                <i style={{ width: `${row.locked ? 0 : clamp(p ?? n(row.value) ?? 0, 2, 100)}%`, background: t?.color }} />
                <b style={{ left: "50%" }} />
              </div>
              <strong className="pv-bar__value">{row.locked ? "?" : row.text ?? Math.round(n(row.value) ?? 0)}</strong>
              {row.locked ? <span className="pv-tier pv-tier--unknown is-small">SCOUT IT</span> : <TierBadge tier={t} small />}
              <span className="pv-bar__pct">{row.locked ? "" : percentileLabel(p)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/** Radar of up to 8 skills (0–100). */
export function SkillRadar({ axes, size = 210, color = "#13d8e7" }) {
  const list = (axes || []).filter((a) => a && n(a.value) != null).slice(0, 8);
  if (list.length < 3) return null;
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 30;
  const pt = (i, v) => {
    const ang = (Math.PI * 2 * i) / list.length - Math.PI / 2;
    const rr = (clamp(v, 0, 100) / 100) * R;
    return [cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr];
  };
  const poly = list.map((a, i) => pt(i, n(a.value)).join(",")).join(" ");
  const avgPoly = list.map((a, i) => pt(i, n(a.classAvg) ?? 60).join(",")).join(" ");
  return (
    <section className="pv-card pv-radar">
      <header className="pv-card__head">
        <h4>Shape</h4>
        <span><i className="pv-key" style={{ background: color }} /> him · <i className="pv-key is-avg" /> class avg</span>
      </header>
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
          const [lx, ly] = pt(i, 122);
          return (
            <g key={a.label}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(148,185,205,.12)" />
              <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" className="pv-radar__label">
                {a.short || a.label}
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
export function OutcomeDonut({ segments, title = "Career outcomes", size = 132 }) {
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
      <header className="pv-card__head">
        <h4>{title}</h4>
      </header>
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
  display:inline-flex;align-items:center;justify-content:center;
  padding:3px 8px;border-radius:4px;font-size:10.5px;font-weight:1000;letter-spacing:.12em;
  color:var(--tier);background:color-mix(in srgb, var(--tier) 14%, transparent);
  border:1px solid color-mix(in srgb, var(--tier) 45%, transparent);
  white-space:nowrap;line-height:1.1;
}
.pv-tier.is-small{padding:2px 6px;font-size:9.5px;letter-spacing:.1em;}
.pv-tier--elite,.pv-tier--generational{box-shadow:0 0 10px color-mix(in srgb, var(--tier) 40%, transparent);}
.pv-tier--generational{animation:pv-shimmer 2.4s ease-in-out infinite;}
.pv-tier--awful,.pv-tier--bad{font-style:italic;}
.pv-tier--unknown{--tier:#607789;color:#8096a8;}
@keyframes pv-shimmer{0%,100%{filter:brightness(1)}50%{filter:brightness(1.35)}}

.pv-ring{position:relative;display:inline-grid;place-items:center;flex:none;}
.pv-ring svg{display:block;}
.pv-ring__label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;line-height:1;}
.pv-ring__label strong{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:19px;font-weight:800;color:#e9f7fb;}
.pv-ring__label small{margin-top:2px;font-size:8.5px;font-weight:900;letter-spacing:.1em;color:#8096a8;max-width:90%;}

.pv-gauge-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
.pv-gauge{border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:10px 12px;background:rgba(6,21,34,.62);box-shadow:inset 0 1px 0 rgba(255,255,255,.04);min-width:0;}
.pv-gauge__head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;}
.pv-gauge__head span{font-size:10.5px;font-weight:900;letter-spacing:.14em;text-transform:uppercase;color:#8096a8;}
.pv-gauge__head strong{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:30px;line-height:1;font-weight:800;letter-spacing:.02em;}
.pv-gauge__body{display:flex;align-items:center;gap:12px;margin-top:6px;}
.pv-gauge__track{position:relative;flex:1;height:12px;border-radius:6px;background:rgba(148,185,205,.10);margin:10px 0 14px;}
.pv-gauge__tick{position:absolute;top:-3px;bottom:-3px;width:1px;background:rgba(148,185,205,.18);}
.pv-gauge__tick b{position:absolute;top:18px;left:-8px;width:16px;text-align:center;font-size:9px;font-weight:700;color:#607789;font-style:normal;}
.pv-gauge__fog{position:absolute;top:0;bottom:0;border-radius:6px;
  background:repeating-linear-gradient(135deg,rgba(246,196,83,.20) 0 4px,rgba(246,196,83,.06) 4px 8px);
  border:1px dashed rgba(246,196,83,.45);}
.pv-gauge__fog.is-full{left:0;right:0;background:repeating-linear-gradient(135deg,rgba(96,119,137,.25) 0 4px,transparent 4px 8px);border-color:rgba(96,119,137,.4);}
.pv-gauge__band{position:absolute;top:2px;bottom:2px;border-radius:5px;opacity:.9;box-shadow:0 0 10px rgba(19,216,231,.35);}
.pv-gauge--gold .pv-gauge__band{box-shadow:0 0 10px rgba(246,196,83,.4);}
.pv-gauge__dot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;border:2px solid #04101a;}
.pv-gauge__foot{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:11.5px;color:#8096a8;}
.pv-certainty{font-weight:1000;letter-spacing:.14em;font-size:11px;}

.pv-card{border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:10px 12px;background:rgba(6,21,34,.62);box-shadow:inset 0 1px 0 rgba(255,255,255,.04);min-width:0;}
.pv-card__head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px;}
.pv-card__head h4{margin:0;font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:15px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#e9f7fb;}
.pv-card__head span{font-size:10.5px;color:#8096a8;letter-spacing:.06em;text-transform:uppercase;font-weight:800;display:inline-flex;align-items:center;gap:4px;}
.pv-key{display:inline-block;width:10px;height:3px;border-radius:2px;}
.pv-key.is-avg{background:rgba(148,185,205,.6);}

.pv-bars{display:flex;flex-direction:column;gap:6px;}
.pv-bar{display:grid;grid-template-columns:minmax(76px,.9fr) minmax(80px,2fr) 30px 92px 66px;gap:8px;align-items:center;}
.pv-bar__label{font-size:12px;font-weight:800;color:#c9dce6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pv-bar__track{position:relative;height:9px;border-radius:5px;background:rgba(148,185,205,.10);overflow:visible;}
.pv-bar__track i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;transition:width .6s ease;box-shadow:0 0 8px rgba(19,216,231,.25);}
.pv-bar__track b{position:absolute;top:-3px;bottom:-3px;width:1px;background:rgba(233,247,251,.35);}
.pv-bar__value{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:18px;font-weight:800;text-align:right;color:#e9f7fb;}
.pv-bar__pct{font-size:10px;font-weight:900;letter-spacing:.06em;color:#8096a8;text-align:right;white-space:nowrap;}
.pv-bar.is-soft .pv-bar__value{color:#f6c453;}
.pv-bar.is-locked .pv-bar__label{color:#607789;}

.pv-radar svg{display:block;margin:0 auto;}
.pv-radar__label{font-size:9.5px;font-weight:900;letter-spacing:.06em;fill:#8096a8;text-transform:uppercase;}

.pv-donut__body{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.pv-legend{list-style:none;margin:0;padding:0;display:grid;gap:4px;flex:1;min-width:140px;}
.pv-legend li{display:grid;grid-template-columns:10px 1fr auto;gap:6px;align-items:center;font-size:11.5px;color:#c9dce6;}
.pv-legend i{width:10px;height:10px;border-radius:3px;}
.pv-legend b{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-size:15px;}

.pv-dials{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:8px;}
.pv-dial{display:flex;flex-direction:column;align-items:center;gap:4px;border:1px solid rgba(156,218,236,.14);border-radius:8px;padding:8px 4px;background:rgba(6,21,34,.62);}
.pv-dial > span{font-size:10px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:#8096a8;text-align:center;}

.pv-visual-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(200px,1fr);gap:10px;align-items:start;}
.pv-stack{display:flex;flex-direction:column;gap:10px;min-width:0;}

/* Board certainty chip */
.pv-cert-chip{display:inline-flex;align-items:center;gap:5px;font-size:9.5px;font-weight:1000;letter-spacing:.1em;white-space:nowrap;}
.pv-cert-chip i{display:inline-block;width:26px;height:5px;border-radius:3px;background:rgba(148,185,205,.15);position:relative;overflow:hidden;}
.pv-cert-chip i::after{content:"";position:absolute;left:0;top:0;bottom:0;width:var(--w);background:var(--c);}

@media (max-width: 760px){
  .pv-gauge-pair,.pv-visual-grid{grid-template-columns:1fr;}
  .pv-bar{grid-template-columns:minmax(70px,1fr) minmax(60px,1.4fr) 28px 80px;}
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
