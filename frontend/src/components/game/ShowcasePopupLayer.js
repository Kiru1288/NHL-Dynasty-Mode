import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { isFranchiseCinematicPopup } from "../../events/franchiseEventKinds";
import { getTeamLogoSrc, resolveFranchiseTeamLogo, toLogoUrl } from "../../utils/teamLogos";
import { claimWaiverPlayer } from "../../services/franchiseService";
import PlayerHeadshot from "../PlayerHeadshot";

function resolveAlertTheme(pop) {
  const theme = pop.theme || pop.presentation_type || "";
  if (theme === "obituary" || pop.kind === "player_death") return "obituary";
  if (theme === "danger" || pop.legal_severity === "major" || pop.kind === "legal_trouble") {
    return "danger";
  }
  if (theme === "warning" || pop.kind === "injury") return "warning";
  if (theme === "positive") return "positive";
  if (theme === "info") return "info";
  return "neutral";
}

function isTradePopup(pop) {
  if (!pop) return false;
  if (String(pop.type || "").toLowerCase() === "trade") return true;
  if (String(pop.cause_type || "").toLowerCase().includes("trade")) return true;
  if (String(pop.event_type || "").toUpperCase().includes("TRADE")) return true;
  const teams = Array.isArray(pop.teams) ? pop.teams : [];
  return teams.some((t) => t && Array.isArray(t.acquired_assets) && t.acquired_assets.length > 0);
}

const IMPORTANT_POPUP_KINDS = new Set(["injury", "legal_trouble", "player_death", "waiver", "player_meeting", "fa_decision"]);

/** Only what the GM must know or act on interrupts the screen: injuries, suspensions,
 *  trade demands / stability / morale warnings, bargains, his own trades, and things that
 *  need an answer. Hat tricks, hot streaks, skids and goofy news stay in the feed. */
function isImportantPopup(pop, userTeamId) {
  if (!pop) return false;
  const kind = String(pop.kind || "").toLowerCase();
  if (IMPORTANT_POPUP_KINDS.has(kind) || kind === "wjc_loan_permission") return true;
  // Own-club GM alerts (suspension / return, playoff clinch, trade deadline).
  if (pop.gm_alert) return true;
  const pt = String(pop.presentation_type || "").toLowerCase();
  if (pt.startsWith("trade_") || pt.includes("stability") || pt.includes("morale")) return true;
  const actions = Array.isArray(pop.actions) ? pop.actions : [];
  if (actions.some((a) => a && String(a.id || "") === "waiver_claim")) return true;
  const text = `${pop.headline || ""} ${pop.title || ""} ${pop.event_type || ""} ${pop.cause_type || ""}`.toLowerCase();
  if (/suspen|banned|injur|trade demand|wants out|requested a trade|asked for a trade/.test(text)) return true;
  if (isTradePopup(pop)) {
    const uid = String(userTeamId || "");
    if (!uid) return true;
    const teams = Array.isArray(pop.teams) ? pop.teams : [];
    return String(pop.team_id || "") === uid || teams.some((t) => t && String(t.team_id || t.id || "") === uid);
  }
  return false;
}

function StatCard({ label, value, sub }) {
  return (
    <div className="media-alert__fact">
      <span className="media-alert__fact-label">{label}</span>
      <span className="media-alert__fact-value">{value}</span>
      {sub ? <span className="media-alert__fact-sub">{sub}</span> : null}
    </div>
  );
}

/** Free-form label/value lines. Rows with nothing to say are dropped, never shown as "—". */
function FactList({ facts }) {
  const rows = (facts || []).filter(
    (f) => f && f.value != null && String(f.value).trim() !== "" && f.value !== "—"
  );
  if (!rows.length) return null;
  return (
    <div className="media-alert__facts">
      {rows.map((f) => (
        <StatCard key={f.label} label={f.label} value={f.value} sub={f.sub} />
      ))}
    </div>
  );
}

function OvrImpactBlock({ before, after, delta, reason }) {
  if (before == null && after == null) return null;
  const d = Number(delta ?? (after != null && before != null ? after - before : 0));
  if (!d && before === after) return null;
  return (
    <div className={`media-alert__ovr ${d < 0 ? "is-neg" : d > 0 ? "is-pos" : ""}`}>
      <div className="media-alert__ovr-label">Rating Impact</div>
      <div className="media-alert__ovr-row">
        <span className="media-alert__ovr-before">{before ?? "—"}</span>
        <span className="media-alert__ovr-arrow" aria-hidden>
          {d < 0 ? "↓" : d > 0 ? "↑" : "→"}
        </span>
        <span className="media-alert__ovr-after">{after ?? "—"}</span>
      </div>
      {d ? (
        <div className="media-alert__ovr-delta">
          {d > 0 ? "+" : ""}
          {d} OVR
        </div>
      ) : null}
      {reason ? <p className="media-alert__ovr-reason">{reason}</p> : null}
    </div>
  );
}

function formatCapHit(m) {
  if (m == null || !Number.isFinite(Number(m))) return null;
  const n = Number(m);
  return `$${n.toFixed(n >= 10 ? 1 : 2)}M`;
}

function formatStat(v, digits = 0) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return digits > 0 ? n.toFixed(digits) : String(Math.round(n));
}

function teamAbbr(team) {
  return String(team?.abbreviation || team?.team_id || "?").toUpperCase();
}

function TeamMark({ team }) {
  const abbr = teamAbbr(team);
  const src = toLogoUrl(
    getTeamLogoSrc({
      abbrev: abbr,
      team_abbrev: abbr,
      name: team?.display_name,
      team_name: team?.display_name,
    })
  );
  return src ? <img className="trade-wire__logo" src={src} alt="" /> : null;
}

function formatSvPct(v) {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return null;
  const t = n.toFixed(3);
  return t.startsWith("0") ? t.slice(1) : t;
}

function tradeStatItems(stats) {
  const has = (v) => v != null && v !== "" && Number.isFinite(Number(v));
  const items = [];
  const add = (label, v, digits = 0) => {
    if (has(v)) items.push([label, formatStat(v, digits)]);
  };
  add("GP", stats.gp);
  if (stats.kind === "goalie") {
    add("W", stats.w);
    add("L", stats.l);
    const sv = formatSvPct(stats.sv_pct);
    if (sv) items.push(["SV%", sv]);
    add("GAA", stats.gaa, 2);
    add("SO", stats.so);
  } else {
    add("G", stats.g);
    add("A", stats.a);
    add("PTS", stats.pts);
    add("xGF%", stats.xgf_pct, 1);
    add("WAR", stats.war, 1);
  }
  return items;
}

/** One plain line of stats. Says so when there are none instead of a row of dashes. */
function TradeStatLine({ stats, fallbackLabel }) {
  const gp = Number(stats?.gp);
  const known = stats && typeof stats === "object" && Object.keys(stats).length > 0;
  const hasLine = known && (stats.source ? stats.source !== "none" : gp > 0);
  if (!hasLine) {
    const note = stats?.note || (known ? "No games played this season" : "Season stats not available");
    return <div className="trade-wire__stats-none">{note}</div>;
  }
  const label = stats.label || (fallbackLabel ? `${fallbackLabel} stats` : "");
  const suffix = stats.source === "nhl_prior" ? " · last season" : "";
  return (
    <div className="trade-wire__stats">
      {label ? (
        <span className="trade-wire__stats-label">
          {label}
          {suffix}
        </span>
      ) : null}
      {tradeStatItems(stats).map(([k, v]) => (
        <span key={k} className="trade-wire__stat-item">
          <strong>{v}</strong> {k}
        </span>
      ))}
    </div>
  );
}

function TradePlayerCard({ asset, seasonLabel }) {
  const role = asset?.role_line || [asset?.position, asset?.archetype].filter(Boolean).join(" | ");
  const cap = formatCapHit(asset?.cap_hit_m);
  const years = asset?.years_left;
  const age = asset?.age;
  const metaBits = [];
  if (age != null) metaBits.push(`Age ${age}`);
  if (cap) metaBits.push(`${cap} cap hit`);
  if (years != null) metaBits.push(`${years} year${years === 1 ? "" : "s"} left`);
  if (asset?.retained_salary) metaBits.push(`${asset.retained_salary}% retained`);

  return (
    <div className="trade-wire__player">
      <div className="trade-wire__player-line">
        <span className="trade-wire__player-name">{asset?.display_name || "Player"}</span>
        {asset?.ovr != null ? <span className="trade-wire__ovr-inline">{asset.ovr} OVR</span> : null}
      </div>
      {role ? <div className="trade-wire__player-role">{role}</div> : null}
      {metaBits.length ? <div className="trade-wire__contract">{metaBits.join(" · ")}</div> : null}
      <TradeStatLine stats={asset?.season_stats} fallbackLabel={seasonLabel} />
    </div>
  );
}

function TradePickChip({ asset }) {
  return (
    <div className="trade-wire__pick">
      <span className="trade-wire__pick-label">{asset?.display_name || "Draft Pick"}</span>
      {asset?.trade_value != null ? (
        <span className="trade-wire__pick-val"> · Trade Value {formatStat(asset.trade_value, 1)}</span>
      ) : null}
    </div>
  );
}

function TradeSideColumn({ team, seasonLabel }) {
  const assets = Array.isArray(team?.acquired_assets) ? team.acquired_assets : [];
  const players = assets.filter((a) => String(a?.asset_type || "").toLowerCase() === "player");
  const picks = assets.filter((a) => {
    const t = String(a?.asset_type || "").toLowerCase();
    return t === "draft_pick" || t === "pick";
  });
  const other = assets.filter((a) => {
    const t = String(a?.asset_type || "").toLowerCase();
    return t !== "player" && t !== "draft_pick" && t !== "pick";
  });
  const abbr = teamAbbr(team);

  return (
    <div className="trade-wire__side">
      <div className="trade-wire__side-head">
        <TeamMark team={team} />
        <div className="trade-wire__side-title">
          <span className="trade-wire__side-name">{abbr}</span>
          <span className="trade-wire__side-receives"> receives</span>
        </div>
      </div>
      <div className="trade-wire__side-assets">
        {players.map((p, i) => (
          <TradePlayerCard key={`${p.player_id || p.display_name}-${i}`} asset={p} seasonLabel={seasonLabel} />
        ))}
        {picks.length ? (
          <div className="trade-wire__picks">
            <div className="trade-wire__picks-label">Draft picks</div>
            {picks.map((pk, i) => (
              <TradePickChip key={`${pk.pick_id || pk.display_name}-${i}`} asset={pk} />
            ))}
          </div>
        ) : null}
        {other.map((a, i) => (
          <div key={`other-${i}`} className="trade-wire__pick">
            <span className="trade-wire__pick-label">{a.display_name || "Asset"}</span>
          </div>
        ))}
        {!players.length && !picks.length && !other.length ? (
          <div className="trade-wire__empty">No assets listed</div>
        ) : null}
      </div>
    </div>
  );
}

function TradeValueBar({ leftTeam, rightTeam, leftValue, rightValue }) {
  const left = Number(leftValue);
  const right = Number(rightValue);
  const hasLeft = Number.isFinite(left);
  const hasRight = Number.isFinite(right);
  if (!hasLeft && !hasRight) return null;
  const l = hasLeft ? Math.max(0, left) : 0;
  const r = hasRight ? Math.max(0, right) : 0;
  const total = l + r || 1;
  const leftPct = Math.round((l / total) * 100);

  return (
    <section className="trade-wire__value">
      <div className="trade-wire__section-label">Trade value</div>
      <div className="trade-wire__value-row">
        <div className="trade-wire__value-end">
          <span className="trade-wire__abbr">{teamAbbr(leftTeam)}</span>
          <strong>{hasLeft ? formatStat(left, 1) : "—"}</strong>
        </div>
        <div className="trade-wire__value-track" aria-hidden>
          <div className="trade-wire__value-fill is-left" style={{ width: `${leftPct}%` }} />
          <div className="trade-wire__value-fill is-right" style={{ width: `${100 - leftPct}%` }} />
        </div>
        <div className="trade-wire__value-end is-right">
          <strong>{hasRight ? formatStat(right, 1) : "—"}</strong>
          <span className="trade-wire__abbr">{teamAbbr(rightTeam)}</span>
        </div>
      </div>
    </section>
  );
}

function rewriteTradeSummary(raw, left, right) {
  const leftAbbr = String(left?.abbreviation || left?.team_id || "").toUpperCase();
  const rightAbbr = String(right?.abbreviation || right?.team_id || "").toUpperCase();
  const leftId = String(left?.team_id || "");
  const rightId = String(right?.team_id || "");
  let text = String(raw || "").trim();
  if (!text) {
    return `${leftAbbr || "Team"} completes a league trade with ${rightAbbr || "partner"}.`;
  }
  // Replace bare numeric club ids with abbreviations when present.
  if (leftId && leftAbbr && leftId !== leftAbbr) {
    text = text.replace(new RegExp(`\\b${leftId}\\b`, "g"), leftAbbr);
  }
  if (rightId && rightAbbr && rightId !== rightAbbr) {
    text = text.replace(new RegExp(`\\b${rightId}\\b`, "g"), rightAbbr);
  }
  return text;
}

function TradeWireBody({ pop, onDismiss, onDismissAllTrades, onAction, queuedTradeCount = 0 }) {
  const teams = (Array.isArray(pop.teams) ? pop.teams : []).filter(
    (t) => t && Array.isArray(t.acquired_assets)
  );
  const left = teams[0] || { abbreviation: pop.team_abbrev || pop.team_id, acquired_assets: [] };
  const right = teams[1] || { abbreviation: pop.from_team_id, acquired_assets: [] };
  const tv = pop.trade_value || {};
  const leftValue = tv.left_value ?? left.trade_value;
  const rightValue = tv.right_value ?? right.trade_value;
  const summary = rewriteTradeSummary(
    pop.summary || pop.details || "",
    left,
    right
  );
  const reasoning = pop.reason_text || pop.cause || pop.story_report || pop.effect_summary || "";
  const tradeType = pop.trade_type_label || String(pop.trade_category || "League Trade").replace(/_/g, " ");
  const seasonLabel = pop.season_label || "";
  const isRumor = /rumou?r/i.test(
    `${pop.trade_category || ""} ${pop.cause_type || ""} ${pop.knowledge_type || ""} ${pop.category || ""}`
  );

  return (
    <div className="trade-wire">
      <div className="trade-wire__source-row">
        <div className="trade-wire__source-copy">
          <p className="trade-wire__source">
            {pop.source_label || "League Trade Wire"}
            {pop.calendar_iso ? <span className="trade-wire__date"> · {pop.calendar_iso}</span> : null}
          </p>
        </div>
        <button type="button" className="trade-wire__close" onClick={onDismiss} aria-label="Close">
          ×
        </button>
      </div>

      <h3 className="trade-wire__title">{isRumor ? "Trade rumor" : "Trade completed"}</h3>
      <p className="trade-wire__summary">{summary}</p>

      <div className="trade-wire__exchange">
        <TradeSideColumn team={left} seasonLabel={seasonLabel} />
        <TradeSideColumn team={right} seasonLabel={seasonLabel} />
      </div>

      <TradeValueBar leftTeam={left} rightTeam={right} leftValue={leftValue} rightValue={rightValue} />

      <section className="trade-wire__reason">
        <div className="trade-wire__reason-meta">
          <span className="trade-wire__section-label">Trade type</span>
          <span className="trade-wire__type-text">{tradeType}</span>
        </div>
        <div className="trade-wire__reason-body">
          <div className="trade-wire__section-label">Reasoning</div>
          <p>{reasoning || "Roster management trade."}</p>
        </div>
      </section>

      <div className="trade-wire__foot">
        <button type="button" className="trade-wire__btn" onClick={() => onAction({ id: "tradehub" })}>
          View Trade
        </button>
        <button type="button" className="trade-wire__btn" onClick={() => onAction({ id: "storylines" })}>
          Open Storylines
        </button>
        {queuedTradeCount > 1 && typeof onDismissAllTrades === "function" ? (
          <button
            type="button"
            className="trade-wire__btn trade-wire__btn--clear"
            onClick={onDismissAllTrades}
            title="Clear every queued trade alert"
          >
            Clear all trades ({queuedTradeCount})
          </button>
        ) : null}
        <button type="button" className="trade-wire__btn is-primary" onClick={onDismiss}>
          Continue →
        </button>
      </div>
    </div>
  );
}

function popTeam(pop) {
  const c = pop.player_card || {};
  const abbr = String(c.team_abbr || pop.team_abbrev || pop.team_abbr || "").toUpperCase();
  const name = c.team_name || pop.team_name || "";
  const logo =
    resolveFranchiseTeamLogo({ team_id: c.team_id || pop.team_id, team_name: name, name, team_abbrev: abbr, abbrev: abbr }, name) ||
    (abbr ? toLogoUrl(getTeamLogoSrc(abbr)) : "") ||
    "";
  return { abbr, name, logo };
}

const pctTone = (p) => (p >= 80 ? "elite" : p >= 60 ? "good" : p >= 40 ? "avg" : p >= 20 ? "low" : "poor");

/** Hero: photo, who he is, his club's logo and the three numbers that matter (OVR, POT, trade value). */
function AlertIdentity({ pop }) {
  const c = pop.player_card || {};
  const ovr = Number(pop.player_overall || pop.overall || 0);
  const pot = Number(pop.player_potential || pop.potential || 0);
  const name = pop.player_name;
  const team = popTeam(pop);
  if (!name && !team.name && !team.abbr) return null;
  const pos = c.position || pop.player_position || pop.position || "";
  const age = c.age ?? pop.player_age;
  const meta = [pos, age ? `Age ${age}` : null, c.shoots ? `Shoots ${c.shoots}` : null, c.archetype || null].filter(Boolean);
  const tv = c.trade_value;
  return (
    <div className="pa-hero">
      {name ? (
        <div className="pa-hero__face">
          <PlayerHeadshot
            player={{ ...pop, name, id: pop.player_id, position: pos, overall: ovr || undefined, potential: pot || undefined }}
            size="lg"
            preferPhoto
            showFlag={false}
          />
        </div>
      ) : null}
      <div className="pa-hero__who">
        {name ? <b className="pa-hero__name">{name}</b> : null}
        {meta.length ? <span className="pa-hero__meta">{meta.join(" · ")}</span> : null}
        {team.name || team.abbr ? (
          <span className="pa-hero__team">
            <span>{team.name || team.abbr}</span>
          </span>
        ) : null}
      </div>
      {team.logo ? (
        <div className="pa-hero__logo" title={team.name || team.abbr}>
          <img src={team.logo} alt={team.name || team.abbr || ""} />
        </div>
      ) : null}
      <div className="pa-hero__nums">
        {ovr > 0 ? (
          <div className="pa-num is-ovr">
            <b>{Math.round(ovr)}</b>
            <small>OVR</small>
          </div>
        ) : null}
        {pot > 0 ? (
          <div className="pa-num is-pot">
            <b>{Math.round(pot)}</b>
            <small>POT</small>
          </div>
        ) : null}
        {tv != null ? (
          <div className="pa-num is-tv" title={c.value_tier ? `Trade value tier: ${c.value_tier}` : "Trade value"}>
            <b>{Number(tv).toFixed(1)}</b>
            <small>Trade value</small>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* 8x8 pixel icons (Double Dash style), drawn as crisp SVG squares. Letters pick the colour. */
const PX_COLORS = { k: "#071233", y: "#ffd23a", o: "#ff8a1f", w: "#ffffff", b: "#9fd8ff", g: "#9be564", r: "#ff4f8b", c: "#ffe7c4", n: "#8a5a20" };
const PX_ICONS = {
  claim: ["..kkkk..", ".kcyyck.", ".kcccck.", ".kckkck.", ".kcccck.", ".kckkck.", ".kcccck.", ".kkkkkk."],
  clock: ["..kkkk..", ".kwwwwk.", "kwwkwwwk", "kwwkwwwk", "kwwkkkwk", "kwwwwwwk", ".kwwwwk.", "..kkkk.."],
  cash: ["...kk...", ".kkggkk.", "kggkkggk", "kggkgggk", ".kggkgk.", "kgggkggk", ".kkggkk.", "...kk..."],
  depth: ["kkkkkkkk", "kbbbbbbk", "kkkkkkkk", "kyyyyy.k", "kkkkkkkk", "kbbb...k", "kkkkkkkk", "........"],
  stick: [".....kk.", ".....kn.", "....kn..", "....kn..", "...kn...", "..kn....", "kkkn....", "kkkk...."],
  chart: ["......k.", ".....kr.", "...k.kr.", "..ko.kr.", ".kyokr..", "kyyokr..", "kkkkkkkk", "........"],
  star: ["...kk...", "...yy...", "kkyyyykk", ".kyyyyk.", "..kyyk..", ".kyykyk.", ".kyk.kyk", "........"],
  muscle: ["..kk....", ".kook...", ".kook...", "kooookk.", "koooooo.", "kooooook", ".kooooo.", "..kkkk.."],
  puck: ["........", "..kkkk..", ".kwwwwk.", "kkkkkkkk", "kkkkkkkk", ".kkkkkk.", "........", "........"],
};
function PixelIcon({ name, size = 22 }) {
  const rows = PX_ICONS[name] || PX_ICONS.puck;
  const cells = [];
  rows.forEach((row, y) =>
    String(row)
      .slice(0, 8)
      .split("")
      .forEach((ch, x) => {
        if (PX_COLORS[ch]) cells.push(<rect key={`${x}-${y}`} x={x} y={y} width="1.02" height="1.02" fill={PX_COLORS[ch]} />);
      })
  );
  return (
    <svg className="pa-px" width={size} height={size} viewBox="0 0 8 8" shapeRendering="crispEdges" aria-hidden="true">
      {cells}
    </svg>
  );
}

function seasonLine(st) {
  if (!st || !(Number(st.gp) > 0) || st.source === "none") return [];
  const goalie = st.kind === "goalie";
  const sv = st.sv_pct != null ? (Number(st.sv_pct) > 1 ? Number(st.sv_pct) / 100 : Number(st.sv_pct)).toFixed(3).replace(/^0/, "") : null;
  const rows = goalie
    ? [["GP", st.gp], ["W", st.w], ["L", st.l], ["SV%", sv], ["GAA", st.gaa != null ? Number(st.gaa).toFixed(2) : null], ["SO", st.so]]
    : [["GP", st.gp], ["G", st.g], ["A", st.a], ["PTS", st.pts], ["P/GP", Number(st.gp) > 0 && st.pts != null ? (Number(st.pts) / Number(st.gp)).toFixed(2) : null], ["+/-", st.plus_minus != null ? (Number(st.plus_minus) > 0 ? `+${st.plus_minus}` : st.plus_minus) : null], ["xGF%", st.xgf_pct != null ? Number(st.xgf_pct).toFixed(1) : null], ["WAR", st.war != null ? Number(st.war).toFixed(1) : null]];
  return rows.filter(([, v]) => v != null && v !== "");
}

function Sec({ icon, title, sub, tone = "", children }) {
  return (
    <section className={`pa-sec ${tone ? `pa-sec--${tone}` : ""}`}>
      <h4>
        <PixelIcon name={icon} />
        <span>{title}</span>
        {sub ? <em>{sub}</em> : null}
      </h4>
      {children}
    </section>
  );
}

/** This season (and last), league percentiles at his position, skills, contract and value. Only what the save knows. */
function PlayerCardSections({ card }) {
  if (!card || typeof card !== "object") return null;
  const st = card.season || {};
  const cur = seasonLine(st);
  const last = seasonLine(card.last_season ? { ...card.last_season, source: "prior" } : null);
  const pcts = Array.isArray(card.percentiles) ? card.percentiles : [];
  const pool = card.percentile_pool || {};
  const money = [
    card.cap_hit_m != null ? ["Cap hit", `$${Number(card.cap_hit_m).toFixed(2)}M`] : null,
    card.years_left != null ? ["Term left", `${card.years_left} ${card.years_left === 1 ? "year" : "years"}`] : null,
    card.trade_value != null ? ["Trade value", Number(card.trade_value).toFixed(1)] : null,
    card.value_tier ? ["Value tier", card.value_tier] : null,
  ].filter(Boolean);
  const skills = Array.isArray(card.skills) ? card.skills : [];
  const wv = card.waiver || null;
  const fit = card.depth_fit || null;
  const career = card.career && Number(card.career.gp) > 0 ? card.career : null;
  const ranked = [...skills].sort((x, y) => y.value - x.value);
  const read = { best: ranked.slice(0, 2), worst: ranked.length > 3 ? ranked.slice(-2).reverse() : [] };
  if (!cur.length && !last.length && !pcts.length && !money.length && !skills.length && !wv) return null;
  const statBlock = (rows) => (
    <div className="pa-line">
      {rows.map(([k, v]) => (
        <div key={k}>
          <b>{v}</b>
          <small>{k}</small>
        </div>
      ))}
    </div>
  );
  return (
    <div className="pa-card">
      <div className="pa-col">
        {wv || fit ? (
          <Sec icon="claim" title={wv ? "Waiver claim" : "Fit on your roster"} tone="deal">
            <div className="pa-money">
              {wv?.claim_priority ? (
                <div className={wv.claim_priority <= 8 ? "is-good" : wv.claim_priority >= 24 ? "is-bad" : ""}>
                  <small>Your claim priority</small>
                  <b>
                    #{wv.claim_priority}
                    <em> of {wv.claim_teams}</em>
                  </b>
                </div>
              ) : null}
              {wv ? (
                <div>
                  <small>Claims close</small>
                  <b>{wv.days_left <= 0 ? "Today" : wv.days_left === 1 ? "In 1 day" : `In ${wv.days_left} days`}</b>
                </div>
              ) : null}
              {wv?.user_cap_space_m != null ? (
                <div className={wv.fits_cap ? "is-good" : "is-bad"}>
                  <small>Your cap space</small>
                  <b>
                    ${Number(wv.user_cap_space_m).toFixed(2)}M<em>{wv.fits_cap ? " · fits" : " · won't fit"}</em>
                  </b>
                </div>
              ) : null}
              {fit ? (
                <div className={fit.rank <= Math.ceil(fit.of / 3) ? "is-good" : ""}>
                  <small>On your depth chart</small>
                  <b>
                    #{fit.rank}
                    <em> of {fit.of} {fit.group}</em>
                  </b>
                </div>
              ) : null}
            </div>
          </Sec>
        ) : null}
        {cur.length ? (
          <Sec icon="stick" title={st.source === "nhl_prior" ? `Last season · ${st.label || ""}` : `This season · ${st.label || ""}`}>
            {statBlock(cur)}
          </Sec>
        ) : (
          <Sec icon="stick" title="This season">
            <p className="pa-note">{st.note || "No games played yet this season."}</p>
          </Sec>
        )}
        {last.length ? (
          <Sec icon="clock" title={`Last season · ${card.last_season.label || ""}`}>
            {statBlock(last)}
          </Sec>
        ) : null}
        {career ? (
          <Sec icon="star" title="NHL career" sub={`${career.seasons} ${career.seasons === 1 ? "season" : "seasons"}`}>
            {statBlock(
              (career.w != null
                ? [["GP", career.gp], ["W", career.w], ["SO", career.so]]
                : [["GP", career.gp], ["G", career.g], ["A", career.a], ["PTS", career.pts], ["P/GP", career.gp ? (career.pts / career.gp).toFixed(2) : null]]
              ).filter(([, v]) => v != null)
            )}
          </Sec>
        ) : null}
        {read.best.length ? (
          <Sec icon="puck" title="Scouting read">
            <div className="pa-read">
              <div className="is-up">
                <small>Strengths</small>
                {read.best.map((k) => (
                  <span key={k.key}>▲ {String(k.key).replace(/_/g, " ")} <b>{k.value}</b></span>
                ))}
              </div>
              <div className="is-down">
                <small>Watch out for</small>
                {read.worst.map((k) => (
                  <span key={k.key}>▼ {String(k.key).replace(/_/g, " ")} <b>{k.value}</b></span>
                ))}
              </div>
            </div>
          </Sec>
        ) : null}
      </div>
      <div className="pa-col">
        <Sec
          icon="chart"
          title="League analytics"
          sub={pcts.length ? `percentile vs ${pool.n || ""} ${pool.group || "players"}${pool.season ? ` · ${pool.season}` : ""}${pool.min_gp ? ` · ${pool.min_gp}+ GP` : ""}` : ""}
        >
          {pcts.length ? (
            <ul className="pa-pcts">
              {pcts.map((p) => (
                <li key={p.key} className={`is-${pctTone(p.pct)}`}>
                  <span className="pa-pcts__label">{p.label}</span>
                  <span className="pa-pcts__val">{p.value}</span>
                  <span className="pa-pcts__bar" aria-hidden="true">
                    <i style={{ width: `${Math.max(3, p.pct)}%` }} />
                  </span>
                  <b className="pa-pcts__pct">
                    {p.pct}
                    <small>th</small>
                  </b>
                </li>
              ))}
            </ul>
          ) : (
            <p className="pa-note">Percentiles appear once enough of the league has played a few games.</p>
          )}
        </Sec>
        {money.length ? (
          <Sec icon="cash" title="Contract & value">
            <div className="pa-money">
              {money.map(([k, v]) => (
                <div key={k}>
                  <small>{k}</small>
                  <b>{v}</b>
                </div>
              ))}
            </div>
          </Sec>
        ) : null}
        {skills.length ? (
          <Sec icon="muscle" title="Skill profile">
            <ul className="pa-skills">
              {skills.map((k) => (
                <li key={k.key} className={`is-${k.value >= 82 ? "elite" : k.value >= 75 ? "good" : k.value >= 68 ? "avg" : "low"}`}>
                  <span>{String(k.key).replace(/_/g, " ")}</span>
                  <i aria-hidden="true">
                    <em style={{ width: `${Math.max(4, Math.min(100, ((k.value - 40) / 55) * 100))}%` }} />
                  </i>
                  <b>{k.value}</b>
                </li>
              ))}
            </ul>
          </Sec>
        ) : null}
      </div>
    </div>
  );
}

function MediaAlertShell({ pop, children, onDismiss, onAction, actions = [], queueCount = 0 }) {
  const theme = resolveAlertTheme(pop);
  const source = pop.source_label || pop.title || "League Update";
  const kindLabel = String(pop.kind || pop.type || "")
    .replace(/_/g, " ")
    .toUpperCase();
  const showKind = kindLabel && kindLabel.toLowerCase() !== String(source).toLowerCase();

  return (
    <div className={`media-alert media-alert--${theme} media-alert--v2 media-alert--free`}>
      <div className="media-alert__dateline">
        <span className="media-alert__source">{source}</span>
        {showKind ? <span className="media-alert__dateline-part">{kindLabel}</span> : null}
        {pop.calendar_iso ? <span className="media-alert__dateline-part">{pop.calendar_iso}</span> : null}
        <button type="button" className="media-alert__close" onClick={onDismiss} aria-label="Dismiss alert">
          ×
        </button>
      </div>

      <div className="media-alert__hero-text">
        <h3 className="media-alert__headline" id="showcase-popup-title">
          {pop.headline || pop.title || "Update"}
        </h3>
      </div>

      <AlertIdentity pop={pop} />

      <div className="media-alert__body-scroll">
        <PlayerCardSections card={pop.player_card} />
        {false && pop.player_id && (!pop.player_card || Array.isArray(pop.player_card._errors)) ? (
          <p className="pa-diag">
            {!pop.player_card
              ? "Player card not sent by the server (restart the API with start_api.ps1)."
              : `Player card partly failed: ${pop.player_card._errors.join(" | ")}`}
          </p>
        ) : null}
        {children}
      </div>

      {actions.length ? (
        <div className="media-alert__actions">
          {actions.map((act) => (
            <button
              key={act.id}
              type="button"
              className={`media-alert__action ${act.primary ? "is-primary" : ""}`}
              onClick={() => onAction(act)}
            >
              {act.label}
            </button>
          ))}
        </div>
      ) : null}

      <div className="media-alert__foot">
        <button type="button" className="media-alert__continue is-primary" onClick={onDismiss}>
          Continue →
        </button>
      </div>
    </div>
  );
}

function ObituaryBody({ pop, onDismiss }) {
  const age = pop.player_age != null && pop.player_age !== "" ? `age ${pop.player_age}` : "";
  const meta = [pop.player_position, age, pop.player_country, pop.team_name].filter(Boolean);
  return (
    <div className="obituary">
      <p className="obituary__kicker">League announcement</p>
      <h3 className="obituary__name" id="showcase-popup-title">
        {pop.player_name || "A player"}
      </h3>
      {meta.length ? <p className="obituary__meta">{meta.join("   ·   ")}</p> : null}
      <div className="obituary__rule" aria-hidden />
      <p className="obituary__lead">{pop.headline || "Has died"}</p>
      {pop.summary ? <p className="obituary__body">{pop.summary}</p> : null}
      {pop.cause ? <p className="obituary__cause">{pop.cause}</p> : null}
      <p className="obituary__note">
        He is permanently removed from competition. His contract comes off the books and the roster spot is open.
      </p>
      <button type="button" className="obituary__ack" onClick={onDismiss}>
        Acknowledge
      </button>
    </div>
  );
}

function StorylineBody({ pop, onDismiss, onDismissAllTrades, onAction, queuedTradeCount = 0, queueCount = 0 }) {
  if (isTradePopup(pop) && !pop.trade_demand) {
    return (
      <TradeWireBody
        pop={pop}
        onDismiss={onDismiss}
        onDismissAllTrades={onDismissAllTrades}
        onAction={onAction}
        queuedTradeCount={queuedTradeCount}
      />
    );
  }

  const demand = pop.trade_demand || null;
  const storyText =
    pop.body ||
    (pop.summary && pop.summary !== pop.headline ? pop.summary : "") ||
    pop.story_report ||
    pop.headline ||
    pop.description ||
    pop.storyline_text ||
    "";
  const impactText = pop.franchise_impact || pop.effect_summary || "";
  const hasGames = Number(pop.games_remaining) > 0;
  const sourceLabel =
    pop.source_label ||
    (demand ? "Player Trade Demand" : null) ||
    pop.title ||
    "Team Report";

  const stats = [
    { label: "Player", value: pop.player_name ? null : pop.culprit_player_name || null },
    {
      label: "Status",
      value: demand
        ? demand.disruptor
          ? "Locker-room disruptor"
          : "Trade demand"
        : pop.kind === "waiver"
          ? pop.waiver_result?.status_label || "On waivers"
          : hasGames
          ? "Away from team"
          : pop.legal_severity === "major"
            ? "Under review"
            : "Active",
    },
  ];
  if (pop.waiver_result && Array.isArray(pop.waiver_result.facts)) {
    for (const f of pop.waiver_result.facts) {
      if (f && f.label) stats.push({ label: f.label, value: f.value, sub: f.sub || "" });
    }
  }
  if (demand) {
    stats.push({
      label: "Trade value",
      value: `${demand.value_before ?? "—"} → ${demand.value_after ?? "—"}`,
      sub: demand.value_delta != null ? `Δ ${demand.value_delta}` : "",
    });
    if (demand.remaining_seconds != null) {
      const sec = Math.max(0, Math.floor(Number(demand.remaining_seconds) || 0));
      stats.push({
        label: "Crisis timer",
        value: `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`,
        sub: demand.formal_crisis ? "Real-time deadline" : "",
      });
    }
    if (demand.agent?.name) {
      stats.push({ label: "Agent", value: demand.agent.name, sub: demand.agent.style_label || "" });
    }
    stats.push({
      label: "Willing destinations",
      value: demand.destination_label || `${(demand.preferred_destinations || []).length || 0} teams`,
    });
  }
  if (pop.cause || pop.cause_type) {
    stats.push({
      label: "Cause",
      value: pop.cause ? String(pop.cause).slice(0, 72) : String(pop.cause_type || "").replace(/_/g, " "),
    });
  }
  if (hasGames) {
    stats.push({
      label: "Expected return",
      value: pop.return_estimate || `${pop.games_remaining} games`,
      sub: pop.return_date || "",
    });
  }

  // Only show the impact section when there is something real to say (never a filler line).
  const hasImpactContent = Boolean(
    impactText ||
      pop.overall_delta != null ||
      pop.overall_before != null ||
      pop.base_overall != null ||
      hasGames ||
      (Array.isArray(pop.impact_lines) && pop.impact_lines.length)
  );

  let actions;
  if (Array.isArray(pop.actions)) {
    actions = pop.actions.filter((a) => a && a.id);
  } else if (pop.kind === "breaking_news" || pop.kind === "waiver") {
    actions = [];
  } else {
    actions = [
      { id: "storylines", label: "Open Storylines", primary: !demand },
      { id: demand ? "tradehub" : "roster", label: demand ? "Open Trade Hub" : "View Player", primary: Boolean(demand) },
    ];
  }

  return (
    <MediaAlertShell pop={{ ...pop, source_label: sourceLabel, headline: pop.headline || sourceLabel }} onDismiss={onDismiss} onAction={onAction} actions={actions} queueCount={queueCount}>
      <FactList facts={stats} />

      {demand?.dossier_label ? (
        <p className="media-alert__callout">{demand.dossier_label}</p>
      ) : null}

      {pop.cause ? (
        <section className="media-alert__section">
          <h4 className="media-alert__section-title">Trigger / Cause</h4>
          <p className="media-alert__story">{pop.cause}</p>
        </section>
      ) : null}

      {Array.isArray(pop.trigger_reasons) && pop.trigger_reasons.length ? (
        <section className="media-alert__section">
          <h4 className="media-alert__section-title">Why this story fired</h4>
          <ul className="media-alert__impact-list">
            {pop.trigger_reasons.map((row, idx) => (
              <li key={`${row.code || "trigger"}-${idx}`}>
                <strong>{String(row.label || row.code || "Signal")}</strong>
                {row.value != null ? `: ${String(row.value)}` : ""}
              </li>
            ))}
          </ul>
        </section>
      ) : pop.trigger_reason ? (
        <section className="media-alert__section">
          <h4 className="media-alert__section-title">Why this story fired</h4>
          <p className="media-alert__story">{String(pop.trigger_reason)}</p>
        </section>
      ) : null}

      {pop.kind === "waiver" && pop.player_card && (pop.player_card.age || pop.player_card.waiver) ? null : (
        <section className="media-alert__section">
          <h4 className="media-alert__section-title">{demand ? "Demand" : "Story Report"}</h4>
          <p className="media-alert__story">{storyText}</p>
        </section>
      )}

      {demand ? (
        <section className="media-alert__section media-alert__section--impact">
          <h4 className="media-alert__section-title">Trade Value Impact</h4>
          <OvrImpactBlock
            before={demand.value_before}
            after={demand.value_after}
            delta={demand.value_delta}
            reason={
              demand.disruptor
                ? "Disruptor penalty — value torpedoed while forcing a move"
                : "Demand discount — clubs leverage his desire to leave"
            }
          />
        </section>
      ) : hasImpactContent ? (
        <section className="media-alert__section media-alert__section--impact">
          <h4 className="media-alert__section-title">Franchise Impact</h4>
          <OvrImpactBlock
            before={pop.overall_before ?? pop.base_overall}
            after={pop.overall_after ?? pop.effective_overall}
            delta={pop.overall_delta}
            reason={
              pop.impact_reason ||
              pop.cause ||
              (hasGames
                ? "Player temporarily unavailable — investigation / conduct penalty"
                : "Temporary performance modifier from verified franchise trigger")
            }
          />
          {impactText && !pop.overall_delta ? <p className="media-alert__impact-line">{impactText}</p> : null}
          {Array.isArray(pop.impact_lines) && pop.impact_lines.length ? (
            <ul className="media-alert__impact-list">
              {pop.impact_lines.map((line, idx) => (
                <li key={idx}>{String(line)}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}

      {pop.requires_decision ? (
        <p className="media-alert__callout">GM response may be required — check Storylines → Decisions.</p>
      ) : null}
    </MediaAlertShell>
  );
}

function FaDecisionBody({ pop, onDismiss, onAction, queueCount = 0 }) {
  const accepted = pop.decision === "accepted";
  const decisionLabel = accepted
    ? "Signed"
    : pop.decision === "signed_elsewhere"
      ? "Signed elsewhere"
      : "Declined";
  const actions = accepted
    ? [
        { id: "lines", label: "Edit Lines" },
        { id: "roster", label: "View Roster", primary: true },
      ]
    : [{ id: "freeagency", label: "Free Agents", primary: true }];
  return (
    <MediaAlertShell pop={pop} onDismiss={onDismiss} onAction={onAction} actions={actions} queueCount={queueCount}>
      <FactList
        facts={[
          { label: "Decision", value: decisionLabel },
          { label: "Contract", value: accepted ? pop.terms || null : null },
        ]}
      />
      <section className="media-alert__section">
        <h4 className="media-alert__section-title">Player Response</h4>
        <p className="media-alert__story">{pop.body}</p>
      </section>
    </MediaAlertShell>
  );
}

function InjuryBody({ pop, onDismiss, onAction, queueCount = 0 }) {
  const tier = String(pop.tier || "").toLowerCase();
  const inj = pop.injury_type ? String(pop.injury_type) : "";

  return (
    <MediaAlertShell
      pop={{
        ...pop,
        source_label: "Medical Desk Report",
        theme: "warning",
      }}
      onDismiss={onDismiss}
      onAction={onAction}
      queueCount={queueCount}
      actions={[
        { id: "roster", label: "View Player", primary: true },
        { id: "storylines", label: "Open Storylines" },
      ]}
    >
      <FactList
        facts={[
          { label: "Severity", value: tier || null },
          { label: "Timeline", value: pop.games != null ? `${pop.games} games` : null, sub: inj || "" },
        ]}
      />

      <section className="media-alert__section">
        <h4 className="media-alert__section-title">Medical Report</h4>
        <p className="media-alert__story">
          {pop.headline || `${pop.player_name || "Player"} injured`} — expected to miss{" "}
          <strong>{pop.games != null ? pop.games : "multiple"}</strong> games.
        </p>
      </section>

      <section className="media-alert__section media-alert__section--impact">
        <h4 className="media-alert__section-title">Franchise Impact</h4>
        <p className="media-alert__impact-line">Unavailable for listed games · depth chart stress</p>
      </section>
    </MediaAlertShell>
  );
}

function ShowcaseGameBody({ pop }) {
  const h = pop.home || {};
  const a = pop.away || {};
  const ot = pop.overtime ? " · OT" : "";
  return (
    <div className="showcase-popup__game">
      <div className="showcase-popup__scoreline">
        <span className="showcase-popup__mono">{h.abbr || "?"}</span>
        <span className="showcase-popup__score">
          {pop.home_goals}–{pop.away_goals}
          {ot}
        </span>
        <span className="showcase-popup__mono">{a.abbr || "?"}</span>
      </div>
      <div className="showcase-popup__sub">
        {h.name || ""} vs {a.name || ""}
      </div>
    </div>
  );
}

function LeagueNoticeBody({ pop }) {
  const kindLabel = String(pop?.kind || pop?.type || "league notice")
    .replace(/_/g, " ")
    .trim();
  const message =
    pop?.message ||
    pop?.body ||
    pop?.detail ||
    pop?.summary ||
    pop?.text ||
    "This league update is ready. Continue to return to the hub.";
  return (
    <div className="showcase-popup__notice">
      <p className="showcase-popup__notice-kicker">{kindLabel}</p>
      <p className="showcase-popup__notice-copy">{String(message)}</p>
    </div>
  );
}

function WjcBody({ pop }) {
  const [openRr, setOpenRr] = useState(false);
  const standings = pop.standings || [];
  const medals = pop.medal_labels || {};
  const complete = Boolean(pop.medals_final || pop.wjc_phase === "complete");
  const dayNum = pop.wjc_day;
  const dayTot = pop.wjc_days_total;
  const calIso = pop.calendar_iso;
  const rrGames = useMemo(() => (pop.round_robin_games || []).slice(), [pop]);

  return (
    <div className="showcase-popup__wjc">
      {dayNum && dayTot ? (
        <p className="showcase-popup__wjc-banner">
          U20 World Juniors — day {dayNum} of {dayTot}
          {calIso ? <span className="showcase-popup__wjc-iso"> · {calIso}</span> : null}
          {!complete ? <span className="showcase-popup__wjc-live"> · tournament in progress</span> : null}
        </p>
      ) : null}
      {complete ? (
        <div className="showcase-popup__medals">
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--gold">Gold</span> {medals.gold || "—"}
          </div>
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--silver">Silver</span> {medals.silver || "—"}
          </div>
          <div>
            <span className="showcase-popup__medal showcase-popup__medal--bronze">Bronze</span> {medals.bronze || "—"}
          </div>
        </div>
      ) : (
        <p className="showcase-popup__muted">Medals are awarded after the gold medal game (Jan 5).</p>
      )}
      {Array.isArray(pop.user_highlights) && pop.user_highlights.length ? (
        <div className="showcase-popup__wjc-yours">
          <h4 className="showcase-popup__h">{complete ? "Your prospects — tournament totals" : "Your prospects at the tournament"}</h4>
          <ul>
            {pop.user_highlights.map((h) => (
              <li key={h.player_id || h.name}>
                <b>{h.name}</b> <span>{h.country || ""}{h.position ? ` · ${h.position}` : ""}</span>
                {h.gp ? <em>{h.gp} GP · {h.g}G {h.a}A · {h.pts} PTS</em> : <em>Dressing for his country</em>}
              </li>
            ))}
          </ul>
          {!complete ? <p className="showcase-popup__muted">No more daily pop-ups — follow along on the World Juniors desk. You'll get one recap at the final.</p> : null}
        </div>
      ) : null}
      <h4 className="showcase-popup__h">Round robin — standings to date</h4>
      <div className="showcase-popup__table-wrap">
        <table className="showcase-popup__table">
          <thead>
            <tr>
              <th>#</th>
              <th>Country</th>
              <th>GP</th>
              <th>W</th>
              <th>L</th>
              <th>GF</th>
              <th>GA</th>
              <th>Pts</th>
            </tr>
          </thead>
          <tbody>
            {standings.map((row) => (
              <tr key={row.code}>
                <td>{row.place}</td>
                <td>
                  <strong>{row.code}</strong> {row.label}
                </td>
                <td>{row.gp}</td>
                <td>{row.w}</td>
                <td>{row.l}</td>
                <td>{row.gf}</td>
                <td>{row.ga}</td>
                <td>{row.pts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button type="button" className="showcase-popup__toggle" onClick={() => setOpenRr((v) => !v)}>
        {openRr ? "Hide" : "Show"} all round-robin scores ({rrGames.length})
      </button>
      {openRr ? (
        <ul className="showcase-popup__rr">
          {rrGames.map((g, i) => (
            <li key={`rr-${i}`}>
              {g.home_label || g.home} {g.home_goals}–{g.away_goals} {g.away_label || g.away}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function AllStarBody({ pop }) {
  const ua = pop.user_allstars || [];
  return (
    <div className="showcase-popup__asg">
      <div className="showcase-popup__scoreline showcase-popup__scoreline--lg">
        <span>{pop.team_a_label}</span>
        <span className="showcase-popup__score">
          {pop.team_a_score}–{pop.team_b_score}
        </span>
        <span>{pop.team_b_label}</span>
      </div>
      {ua.length ? (
        <p className="showcase-popup__highlight">Your players selected: {ua.join(", ")}</p>
      ) : (
        <p className="showcase-popup__muted">No players from your NHL roster made this year&apos;s showcase.</p>
      )}
    </div>
  );
}

function ShowcasePopupStyles() {
  return (
    null
  );
}

function PermissionDecisionCard({ decision, onResolve }) {
  const options = Array.isArray(decision?.options) ? decision.options : [];
  return (
    <div className="showcase-popup showcase-popup--v2" role="dialog" aria-label="Permission needed">
      <div className="showcase-popup__sheet">
        <p className="showcase-popup__kicker">Permission needed</p>
        <h2 className="showcase-popup__title">{decision?.title || "Decision required"}</h2>
        <p className="showcase-popup__summary">{decision?.message || decision?.description || "Choose before the day can advance."}</p>
        <div className="media-alert__actions">
          {options.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={`media-alert__action ${opt.recommended ? "is-primary" : ""}`}
              onClick={() => onResolve?.(decision.id, opt.id)}
            >
              {opt.label || opt.id}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function ShowcasePopupLayer() {
  // Hold league pop-ups (waivers, injuries…) while a live game is on screen; they show after the game.
  const [liveOn, setLiveOn] = React.useState(() => typeof document !== "undefined" && !!document.querySelector(".st-root"));
  React.useEffect(() => {
    const on = (e) => setLiveOn(!!e?.detail);
    window.addEventListener("nhl-live-sim", on);
    return () => window.removeEventListener("nhl-live-sim", on);
  }, []);
  const { franchiseState, onDismissShowcasePopups, setScreen, resumeInterruptedSim, cancelSimResume, setStatsCentralTab, onResolveDecision, refreshFranchise } = useGameUI();
  const userTeamId = franchiseState?.user_team_id || franchiseState?.team?.team_id || "";
  const rawQueue = (franchiseState?.pending_ui_popups || []).filter(
    (p) => p && !isFranchiseCinematicPopup(p) && isImportantPopup(p, userTeamId)
  );
  const hasPendingDecisions =
    Array.isArray(franchiseState?.pending_decisions) && franchiseState.pending_decisions.length > 0;
  const visiblePopups = hasPendingDecisions
    ? [
        ...rawQueue.filter((p) => p && p.kind === "injury"),
        ...rawQueue.filter((p) => p && (p.kind === "legal_trouble" || p.kind === "storyline")),
        ...rawQueue.filter(
          (p) => p && p.kind !== "injury" && p.kind !== "legal_trouble" && p.kind !== "storyline"
        ),
      ]
    : rawQueue;
  const first = visiblePopups[0];
  const [flash, setFlash] = useState("");
  // The alert built right after a sim can carry an incomplete player card (the waiver wire is still being
  // written while that response is built). Ask for fresh state once per alert; the next build has it all.
  const refreshedRef = React.useRef(new Set());
  useEffect(() => {
    const card = first?.player_card;
    const pid = first?.player_id;
    const key = String(first?.id || pid || "");
    if (!pid || !key || refreshedRef.current.has(key)) return;
    if (card && !Array.isArray(card._errors) && (card.age || card.waiver)) return;
    refreshedRef.current.add(key);
    const t = window.setTimeout(() => {
      try {
        refreshFranchise?.();
      } catch (e) {
        /* ignore */
      }
    }, 150);
    return () => window.clearTimeout(t);
  }, [first, refreshFranchise]);

  useEffect(() => {
    if (!flash) return undefined;
    const t = window.setTimeout(() => setFlash(""), 4500);
    return () => window.clearTimeout(t);
  }, [flash]);

  const queueLen = visiblePopups.length;
  const dismiss = useCallback(() => {
    if (!first) return;
    const pid = String(first.id || first.popup_id || "").trim();
    if (pid) onDismissShowcasePopups([pid]);
    else onDismissShowcasePopups([`__drop_first__:${String(first.kind || "popup")}`]);
    // Continue on the last alert of an interrupted multi-day sim resumes the remaining days.
    if (queueLen <= 1) resumeInterruptedSim?.();
  }, [first, onDismissShowcasePopups, queueLen, resumeInterruptedSim]);

  useEffect(() => {
    if (!first || liveOn) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") dismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [first, dismiss, liveOn]);

  const pendingDecisions = (Array.isArray(franchiseState?.pending_decisions) ? franchiseState.pending_decisions : [])
    .filter((d) => d && Array.isArray(d.options) && d.options.length);
  if (liveOn) return null;

  if (pendingDecisions.length) {
    return <PermissionDecisionCard decision={pendingDecisions[0]} onResolve={onResolveDecision} />;
  }

  if (!first) {
    return flash ? (
      <>
        <ShowcasePopupStyles />
        <div className="alert-flash" role="status">{flash}</div>
      </>
    ) : null;
  }

  const kind = first.kind;
  const dismissAll = () => {
    const ids = visiblePopups.map((p) => String(p.id || p.popup_id || "").trim()).filter(Boolean);
    if (ids.length) {
      onDismissShowcasePopups(ids);
      resumeInterruptedSim?.();
    } else dismiss();
  };
  const theme = resolveAlertTheme(first);

  const tradeQueueIds = visiblePopups.filter(isTradePopup).map((p) => p.id).filter(Boolean);
  const dismissAllTrades = () => {
    if (tradeQueueIds.length) onDismissShowcasePopups(tradeQueueIds);
  };

  const handleAction = (act) => {
    // Going to another screen to act on the alert ends the paused sim; a waiver claim
    // is answered in place, so the sim may still resume.
    if (act.id !== "waiver_claim") cancelSimResume?.();
    dismiss();
    if (act.id === "waiver_claim") {
      const pid = act.player_id || first.player_id;
      claimWaiverPlayer(pid, true)
        .then((r) =>
          setFlash(
            r?.ok
              ? `Claim placed on ${first.player_name || "player"} — resolves when the 24h window closes.`
              : r?.reason || "Claim failed"
          )
        )
        .catch((e) => setFlash(e?.message || "Claim failed"));
      return;
    }
    if (act.id === "waivers") setScreen?.(SCREENS.ROSTER);
    else if (act.id === "storylines") setScreen?.(SCREENS.STORYLINES);
    else if (act.id === "roster") setScreen?.(SCREENS.ROSTER);
    else if (act.id === "calendar") setScreen?.(SCREENS.CALENDAR);
    else if (act.id === "tradehub") setScreen?.(SCREENS.TRADE);
    else if (act.id === "lines") setScreen?.(SCREENS.EDIT_LINES);
    else if (act.id === "freeagency") setScreen?.(SCREENS.FREE_AGENCY);
    else if (act.id === "standings") {
      setStatsCentralTab?.("standings");
      setScreen?.(SCREENS.STATS);
    }
  };

  const isObituary = kind === "player_death";
  const isMediaAlert =
    isObituary ||
    kind === "storyline" ||
    kind === "legal_trouble" ||
    kind === "injury" ||
    kind === "player_meeting" ||
    kind === "breaking_news" ||
    kind === "waiver" ||
    kind === "fa_decision";
  const isTradeAlert = isMediaAlert && isTradePopup(first);

  return (
    <>
      <ShowcasePopupStyles />
      <div className={`showcase-popup showcase-popup--v2 ${isObituary ? "showcase-popup--obituary" : ""}`}>
      <div
        className="showcase-popup__backdrop showcase-popup__backdrop--v2"
        aria-hidden
        onClick={isObituary ? undefined : dismiss}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") dismiss();
        }}
        role="button"
        tabIndex={-1}
      />
      <div
        className={`showcase-popup__panel showcase-popup__panel--v2 ${isMediaAlert ? "showcase-popup__panel--media" : ""} ${isTradeAlert ? "showcase-popup__panel--trade-wire" : ""} ${isObituary ? "showcase-popup__panel--obituary" : ""} ${first?.player_card && (first.player_card.age || first.player_card.waiver || first.player_card.season) ? "showcase-popup__panel--card" : ""} showcase-popup__panel--${theme}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="showcase-popup-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        {visiblePopups.length > 1 || flash ? (
          <div className="alert-queue-bar">
            {visiblePopups.length > 1 ? (
              <span className="alert-queue-bar__count">
                Alert <strong>1</strong> of {visiblePopups.length}
              </span>
            ) : null}
            {flash ? <span className="alert-queue-bar__flash">{flash}</span> : null}
            <span className="alert-queue-bar__spacer" />
            {tradeQueueIds.length > 1 ? (
              <button type="button" className="alert-queue-bar__btn" onClick={dismissAllTrades}>
                Clear trades ({tradeQueueIds.length})
              </button>
            ) : null}
            {visiblePopups.length > 1 ? (
              <button type="button" className="alert-queue-bar__btn" onClick={dismissAll}>
                Dismiss all
              </button>
            ) : null}
          </div>
        ) : null}
        {!isMediaAlert ? (
          <header className="showcase-popup__head">
            <h2 id="showcase-popup-title" className="showcase-popup__title">
              {first.title || "League showcase"}
            </h2>
            {first.season_label ? <div className="showcase-popup__season">{first.season_label}</div> : null}
          </header>
        ) : null}
        <div className="showcase-popup__body">
          {kind === "wjc_tournament" ? <WjcBody pop={first} /> : null}
          {kind === "showcase_game" ? <ShowcaseGameBody pop={first} /> : null}
          {kind === "allstar_game" ? <AllStarBody pop={first} /> : null}
          {kind === "player_death" ? <ObituaryBody pop={first} onDismiss={dismiss} /> : null}
          {kind === "injury" ? (
            <InjuryBody pop={first} onDismiss={dismiss} onAction={handleAction} queueCount={visiblePopups.length} />
          ) : null}
          {kind === "storyline" || kind === "legal_trouble" || kind === "player_meeting" || kind === "breaking_news" || kind === "waiver" ? (
            <StorylineBody
              pop={first}
              onDismiss={dismiss}
              onDismissAllTrades={dismissAllTrades}
              onAction={handleAction}
              queuedTradeCount={tradeQueueIds.length}
              queueCount={visiblePopups.length}
            />
          ) : null}
          {kind === "fa_decision" ? (
            <FaDecisionBody pop={first} onDismiss={dismiss} onAction={handleAction} queueCount={visiblePopups.length} />
          ) : null}
          {!["wjc_tournament", "showcase_game", "allstar_game", "injury", "storyline", "legal_trouble", "player_meeting", "breaking_news", "waiver", "fa_decision", "player_death"].includes(
            kind
          ) ? (
            <LeagueNoticeBody pop={first} />
          ) : null}
        </div>
        {!isMediaAlert ? (
          <footer className="showcase-popup__foot">
            <span />
            <button type="button" className="showcase-popup__btn" onClick={dismiss}>
              Continue
            </button>
          </footer>
        ) : null}
      </div>
    </div>
    </>
  );
}
