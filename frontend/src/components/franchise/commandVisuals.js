import React from "react";

export function ProjectionRange({ low, mid, high, confidence = 100, className = "" }) {
  const lo = Number.isFinite(Number(low)) ? Math.round(Number(low)) : null;
  const hi = Number.isFinite(Number(high)) ? Math.round(Number(high)) : null;
  const md = Number.isFinite(Number(mid))
    ? Math.round(Number(mid))
    : lo != null && hi != null
      ? Math.round((lo + hi) / 2)
      : lo ?? hi;
  const conf = Math.max(0, Math.min(100, Number(confidence) || 0));
  if (md == null) return <span className={`fc-proj is-empty ${className}`.trim()}>—</span>;
  return (
    <span
      className={`fc-proj ${conf < 40 ? "is-fog-heavy" : conf < 70 ? "is-fog-mid" : ""} ${className}`.trim()}
      style={{ "--fc-conf": `${conf}%` }}
      title={`Scouting confidence ${conf}%`}
    >
      {lo != null && lo !== md ? <em>{lo}</em> : null}
      {lo != null && lo !== md ? <span className="fc-proj__arrow">←</span> : null}
      <strong>{md}</strong>
      {hi != null && hi !== md ? <span className="fc-proj__arrow">→</span> : null}
      {hi != null && hi !== md ? <em>{hi}</em> : null}
    </span>
  );
}

export function StockTrajectory({ delta = 0, direction = "", className = "" }) {
  const n = Number(delta) || 0;
  const dir = String(direction || "").toUpperCase();
  const rise = n > 0 || dir === "UP" || dir === "RISE";
  const fall = n < 0 || dir === "DOWN" || dir === "FALL";
  const cls = rise ? "is-up" : fall ? "is-down" : "is-flat";
  const label = n > 0 ? `+${n}` : n < 0 ? String(n) : dir === "NEW" ? "NEW" : "—";
  return (
    <span className={`fc-stock ${cls} ${className}`.trim()} title={`Draft stock ${label}`}>
      <i className="fc-stock__shaft" aria-hidden="true" />
      <b>{label}</b>
    </span>
  );
}

export function RankChange({ rank, previous, delta }) {
  const current = Number(rank);
  if (!Number.isFinite(current)) return null;
  const d = Number(delta);
  let prev = Number(previous);
  if (!Number.isFinite(prev) && Number.isFinite(d) && d !== 0) {
    prev = current + d;
  }
  if (!Number.isFinite(prev) || prev === current) {
    return <span className="fc-rank">#{current}</span>;
  }
  const improved = prev > current;
  return (
    <span className={`fc-rank-delta ${improved ? "is-up" : "is-down"}`}>
      <s>#{prev}</s>
      <span aria-hidden="true">→</span>
      <strong>#{current}</strong>
    </span>
  );
}

export function CategoryPills({ player, min = 80 }) {
  const tags = [];
  if (Number(player?.skating) >= min) tags.push("SKATING");
  if (Number(player?.shooting) >= min) tags.push("SHOT");
  if (Number(player?.hockeyIQ) >= min) tags.push("IQ");
  if (Number(player?.passing) >= min) tags.push("VISION");
  if (Number(player?.defense) >= min) tags.push("DEFENSE");
  if (Number(player?.physical) >= min) tags.push("PHYSICAL");
  if (!tags.length) return null;
  return (
    <span className="fc-cats">
      {tags.slice(0, 3).map((tag) => (
        <em key={tag}>{tag}</em>
      ))}
    </span>
  );
}

export function PerformanceStrip({ player }) {
  const gp = Number(player?.gp);
  if (!Number.isFinite(gp) || gp <= 0) return null;
  const pos = String(player?.position || "").toUpperCase();
  const stock = Number(player?.draftStock?.deltaRank ?? player?.stock) || 0;
  const stockLabel = stock > 0 ? `+${stock} stock` : stock < 0 ? `${stock} stock` : "stock even";
  if (pos === "G") {
    return (
      <p className="fc-perf">
        {gp} GP · {player?.wins ?? 0} W · {player?.savePct || "—"} SV% · {stockLabel}
      </p>
    );
  }
  return (
    <p className="fc-perf">
      {gp} GP · {player?.goals ?? 0} G · {player?.points ?? 0} PTS · {stockLabel}
    </p>
  );
}

export function ContractStrip({ aav, years, className = "" }) {
  const money = Number(aav);
  const term = Number(years);
  if (!Number.isFinite(money) || money <= 0) return <span className={`fc-deal ${className}`.trim()}>—</span>;
  const label = money >= 10 ? `$${money.toFixed(1)}M` : `$${money.toFixed(2)}M`;
  return (
    <span className={`fc-deal ${className}`.trim()}>
      {label}
      {Number.isFinite(term) && term > 0 ? <i> × {term}</i> : null}
    </span>
  );
}

export function LeagueBadge({ league }) {
  const code = String(league || "").trim();
  if (!code || code === "—") return <span className="fc-league">—</span>;
  const short = code.replace(/[^A-Za-z0-9]/g, "").slice(0, 5).toUpperCase() || code.slice(0, 4).toUpperCase();
  return (
    <span className="fc-league" title={code}>
      <i aria-hidden="true">{short.slice(0, 3)}</i>
      <b>{code}</b>
    </span>
  );
}

export function StatusSeal({ label, tone = "neutral" }) {
  if (!label) return null;
  return <span className={`fc-seal is-${tone}`}>{label}</span>;
}
