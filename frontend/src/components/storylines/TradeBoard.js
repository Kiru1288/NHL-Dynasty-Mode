/**
 * Bench Buzz trade board (deal sheet tickets + the board inside an opened trade story).
 * Values are the trade engine's market numbers stored on the story at execution time.
 * The engine scale is steep (a depth regular ~3-10, a first-round pick ~40-60, a star 100+),
 * so every asset also gets a plain-words tier and its share of the side's return.
 */
import React from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

const arr = (v) => (Array.isArray(v) ? v : []);
const s = (v, f = "") => (v == null || v === "" ? f : String(v));
const num = (v) => (v == null || v === "" || !Number.isFinite(Number(v)) ? null : Number(v));

export const isPickAsset = (a) => s(a?.asset_type).toLowerCase().includes("pick");

export function pickRound(a) {
  const r = num(a?.round);
  if (r) return r;
  const m = s(a?.display_name).match(/(\d)(?:st|nd|rd|th)[-\s]round|round\s*(\d)/i);
  return m ? Number(m[1] || m[2]) : null;
}

function pickYear(a) {
  const y = num(a?.year);
  if (y) return y;
  const m = s(a?.display_name).match(/\b(20\d\d)\b/);
  return m ? Number(m[1]) : null;
}

/** Plain-words tier derived from the stored market value (and pick round). */
export function assetTier(a) {
  if (isPickAsset(a)) {
    const r = pickRound(a);
    if (r === 1) return { label: "First-rounder", tone: "gold" };
    if (r === 2) return { label: "Second-rounder", tone: "sky" };
    if (r && r <= 4) return { label: "Mid-round pick", tone: "lime" };
    return { label: "Late-round flyer", tone: "plain" };
  }
  const tv = num(a?.trade_value);
  const cap = num(a?.cap_hit_m) || 0;
  if (tv == null) return null;
  if (tv >= 120) return { label: "Franchise piece", tone: "gold" };
  if (tv >= 55) return { label: "Top-line piece", tone: "gold" };
  if (tv >= 25) return { label: "Core piece", tone: "sky" };
  if (tv >= 10) return { label: "Lineup regular", tone: "lime" };
  if (tv >= 3) return { label: "Depth piece", tone: "plain" };
  return { label: cap >= 1.5 ? "Salary in the deal" : "Throw-in", tone: "coral" };
}

export function tradeStatLine(stats) {
  const st = stats && typeof stats === "object" ? stats : null;
  if (!st || !Number(st.gp)) return "";
  const label = s(st.label);
  if (st.w != null || st.sv_pct != null) {
    const sv = Number(st.sv_pct);
    const svTxt = Number.isFinite(sv) && sv > 0 ? ` · ${(sv > 1 ? sv / 100 : sv).toFixed(3).replace(/^0/, "")} SV%` : "";
    return `${label ? `${label}: ` : ""}${st.gp} GP · ${st.w ?? 0}-${st.l ?? 0}${svTxt}`;
  }
  return `${label ? `${label}: ` : ""}${st.gp} GP · ${st.g ?? 0}G ${st.a ?? 0}A · ${st.pts ?? 0} PTS`;
}

export function tradeVerdict(lv, rv, leftAbbr, rightAbbr) {
  const total = Math.max(lv + rv, 1);
  const delta = Math.abs(lv - rv);
  const gapPct = delta / Math.max(lv, rv, 1);
  const winAbbr = lv > rv ? leftAbbr || "Left" : rightAbbr || "Right";
  const loseAbbr = lv > rv ? rightAbbr || "Right" : leftAbbr || "Left";
  if (delta <= Math.max(4, total * 0.08)) return { text: "Balanced deal", tone: "even" };
  if (gapPct < 0.2) return { text: `Edge ${winAbbr}`, tone: "edge" };
  if (gapPct < 0.35) return { text: `${winAbbr} wins it`, tone: "win" };
  return { text: `${winAbbr} fleeces ${loseAbbr}`, tone: "fleece" };
}

const VALUE_HELP =
  "Market value from the trade engine at the time of the deal. The scale is steep: a depth regular is worth about 3-10, a first-round pick roughly 40-60, a star 100+.";

function AssetRow({ asset, sideTotal, maxValue, solo, compact }) {
  const name = s(asset?.display_name || asset?.player_name || asset?.name, "Asset");
  const pick = isPickAsset(asset);
  const ovr = num(asset?.ovr) != null ? Math.round(Number(asset.ovr)) : null;
  const pot = num(asset?.pot) != null ? Math.round(Number(asset.pot)) : null;
  const tv = num(asset?.trade_value);
  const cap = num(asset?.cap_hit_m);
  const pos = s(asset?.position);
  const tier = assetTier(asset);
  const stat = pick ? "" : tradeStatLine(asset?.season_stats);
  const r0 = pick ? pickRound(asset) : null;
  const share = tv != null && sideTotal > 0 && tv > 0 ? Math.round((tv / sideTotal) * 100) : null;
  const meta = pick
    ? [/\d{4}/.test(name) || !pickYear(asset) ? "" : `${pickYear(asset)} draft`].filter(Boolean)
    : [
        pos,
        asset?.age ? `Age ${asset.age}` : "",
        cap != null && cap > 0 ? `$${cap.toFixed(2)}M${asset?.years_left ? ` × ${asset.years_left}y` : ""}` : "",
        s(asset?.archetype),
      ].filter(Boolean);
  const r = r0;
  return (
    <div className={`bbb-asset${pick ? " is-pick" : ""}${solo ? " is-solo" : ""}${compact ? " is-compact" : ""}`}>
      <span className="bbb-asset__face">
        {pick ? (
          <span className="bbb-asset__pk">
            <b>{r ? `R${r}` : "PK"}</b>
            <small>{pickYear(asset) || "pick"}</small>
          </span>
        ) : (
          <PlayerHeadshot
            player={ensurePlayerHeadshotFields({ ...asset, id: asset?.player_id, name, position: pos })}
            size="lg"
            preferPhoto
            showFlag={false}
            number={null}
          />
        )}
      </span>
      <span className="bbb-asset__body">
        <b className="bbb-asset__name" title={name}>{name}</b>
        {meta.length ? <span className="bbb-asset__meta">{meta.join(" · ")}</span> : null}
        {stat ? <span className="bbb-asset__stats">{stat}</span> : null}
        {tier || tv != null ? (
          <span className="bbb-asset__value" title={VALUE_HELP}>
            {tier ? <em className={`bbb-asset__tier is-${tier.tone}`}>{tier.label}</em> : null}
            {tv != null ? (
              <i className="bbb-asset__bar" aria-hidden>
                <i style={{ width: `${Math.max(4, Math.min(100, (Math.max(0, tv) / Math.max(maxValue, 1)) * 100))}%` }} />
              </i>
            ) : null}
            {tv != null ? <small>Value {tv.toFixed(1)}{share != null && share < 100 ? ` · ${share}% of return` : ""}</small> : null}
          </span>
        ) : null}
      </span>
      {!pick && ovr != null ? (
        <span className="bbb-asset__rate">
          <span><b>{ovr}</b><small>OVR</small></span>
          {pot != null && pot > ovr ? <span className="is-pot"><b>{pot}</b><small>POT</small></span> : null}
        </span>
      ) : null}
    </div>
  );
}

function Side({ team, abbr, value, otherCount, maxValue, compact, align }) {
  const assets = arr(team?.acquired_assets);
  const name = s(team?.display_name || abbr, abbr || "Club");
  const logo = resolveFranchiseTeamLogo({ team_abbrev: abbr, abbrev: abbr, abbreviation: abbr, team_name: name }, name) || "";
  const sideTotal = assets.reduce((n, a) => n + Math.max(0, num(a?.trade_value) || 0), 0);
  const solo = assets.length === 1 && otherCount > 1;
  return (
    <div className={`bbb-tb__side is-${align}`}>
      <header className="bbb-tb__club">
        <span className="bbb-tb__logo">{logo ? <img src={logo} alt="" /> : <b>{abbr}</b>}</span>
        <span className="bbb-tb__clubname">
          <b>{name}</b>
          <small>Receives · {assets.length} {assets.length === 1 ? "asset" : "assets"}</small>
        </span>
        {value > 0 ? (
          <span className="bbb-tb__sum" title={VALUE_HELP}>
            <b>{value.toFixed(1)}</b>
            <small>value in</small>
          </span>
        ) : null}
      </header>
      <div className="bbb-tb__assets">
        {assets.length ? (
          assets.map((a, i) => (
            <AssetRow key={a?.player_id || a?.pick_id || a?.id || i} asset={a} sideTotal={sideTotal} maxValue={maxValue} solo={solo} compact={compact} />
          ))
        ) : (
          <p className="bbb-tb__none">Assets undisclosed</p>
        )}
      </div>
    </div>
  );
}

/**
 * board: { left, right, leftValue, rightValue, reason, category } (from extractTradeBoard).
 * variant: "sheet" (deal sheet ticket) | "case" (opened story) | "compact" (story card).
 */
export default function TradeBoard({ board, variant = "sheet", showCategory = false, showReason = true }) {
  if (!board) return null;
  const compact = variant === "compact";
  const leftAbbr = s(board.left?.abbreviation || board.left?.team_abbrev).slice(0, 4).toUpperCase();
  const rightAbbr = s(board.right?.abbreviation || board.right?.team_abbrev).slice(0, 4).toUpperCase();
  const la = arr(board.left?.acquired_assets);
  const ra = arr(board.right?.acquired_assets);
  const lv = Math.max(0, Number(board.leftValue) || 0);
  const rv = Math.max(0, Number(board.rightValue) || 0);
  const maxValue = Math.max(1, ...[...la, ...ra].map((a) => Math.max(0, num(a?.trade_value) || 0)));
  const hasValue = lv > 0 || rv > 0;
  const verdict = hasValue ? tradeVerdict(lv, rv, leftAbbr, rightAbbr) : null;
  const total = Math.max(lv + rv, 1);
  const leftPct = Math.round((lv / total) * 100);
  return (
    <section className={`bbb-tb is-${variant}`} aria-label="Trade summary">
      {showCategory && board.category ? <span className="bbb-tb__cat">{board.category}</span> : null}
      <div className="bbb-tb__grid">
        <Side team={board.left} abbr={leftAbbr} value={lv} otherCount={ra.length} maxValue={maxValue} compact={compact} align="left" />
        <div className="bbb-tb__mid">
          <span className="bbb-tb__swap" aria-hidden>⇄</span>
          {verdict ? (
            <>
              <b className={`bbb-tb__verdict is-${verdict.tone}`}>{verdict.text}</b>
              <small>{leftPct} / {100 - leftPct} value split</small>
            </>
          ) : (
            <small>Values not filed</small>
          )}
        </div>
        <Side team={board.right} abbr={rightAbbr} value={rv} otherCount={la.length} maxValue={maxValue} compact={compact} align="right" />
      </div>
      {hasValue ? (
        <div className="bbb-tb__meter" title={VALUE_HELP}>
          <span>{leftAbbr || "A"} {lv.toFixed(1)}</span>
          <i className="bbb-tb__track" aria-hidden>
            <i className="is-l" style={{ width: `${leftPct}%` }} />
            <i className="is-r" style={{ width: `${100 - leftPct}%` }} />
          </i>
          <span>{rightAbbr || "B"} {rv.toFixed(1)}</span>
        </div>
      ) : null}
      {showReason && board.reason && !compact ? <p className="bbb-tb__reason">{board.reason}</p> : null}
    </section>
  );
}

/** Rail card: the most valuable pieces moved across the listed deals. */
export function TradeValueLeaders({ stories, extract, onOpen, limit = 6 }) {
  const rows = [];
  for (const st of arr(stories)) {
    const b = extract(st);
    if (!b) continue;
    for (const [team, from] of [
      [b.left, b.right],
      [b.right, b.left],
    ]) {
      for (const a of arr(team?.acquired_assets)) {
        const tv = num(a?.trade_value);
        if (tv == null || tv <= 0) continue;
        rows.push({ a, tv, to: s(team?.abbreviation).toUpperCase(), from: s(from?.abbreviation).toUpperCase(), sid: st.id });
      }
    }
  }
  rows.sort((x, y) => y.tv - x.tv);
  const top = rows.slice(0, limit);
  if (!top.length) return null;
  return (
    <section className="sl-dtrail__card bbb-vl">
      <h4>Priciest pieces moved</h4>
      <ol className="bbb-vl__list">
        {top.map((r, i) => {
          const pick = isPickAsset(r.a);
          const name = s(r.a.display_name || r.a.name, "Asset");
          const tier = assetTier(r.a);
          return (
            <li key={`${r.sid}-${i}`}>
              <button type="button" onClick={() => onOpen(r.sid)} title={VALUE_HELP}>
                <span className="bbb-vl__face">
                  {pick ? (
                    <b>{pickRound(r.a) ? `R${pickRound(r.a)}` : "PK"}</b>
                  ) : (
                    <PlayerHeadshot player={ensurePlayerHeadshotFields({ ...r.a, id: r.a.player_id, name })} size="sm" preferPhoto showFlag={false} number={null} />
                  )}
                </span>
                <span className="bbb-vl__who">
                  <b>{name}</b>
                  <small>{r.from && r.to ? `${r.from} → ${r.to}` : ""}{tier ? ` · ${tier.label}` : ""}</small>
                </span>
                <span className="bbb-vl__val">{r.tv.toFixed(1)}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
