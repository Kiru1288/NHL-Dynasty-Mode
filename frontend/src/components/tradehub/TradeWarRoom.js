/**
 * Trade War Room panels — Trade Hub / Trade Finder data surfaces.
 *
 *   TradeValueHero       big Trade Value number, tier, league percentile, demand
 *   AssetIntelPanel      interested clubs (backend /trade/asset-intel) + comparable deals
 *   PackageList          "Assets You Send / Receive" list with + Add Asset (drag-drop kept)
 *   TradeAnalysisBoard   You Send / You Receive / Difference / Recommendation / Fans / Risk
 *   GmNegotiationPanel   partner GM reaction, interest, reasons, counter-offers, relationship
 *   MarketIntelPanel     recent league deals, market demand, most active clubs
 *   MarketableAssets     your most marketable assets (Trade Finder idle state)
 *
 * Presentation only: every number comes from the Trade Hub meta, the evaluator
 * response, trade history, or the asset-intel / counter endpoints.
 */
import React, { useEffect, useMemo, useState } from "react";
import { getTradeAssetIntel, getTradeCounter } from "../../services/franchiseService";

const arr = (v) => (Array.isArray(v) ? v : []);
const num = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
const clampN = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt1 = (v) => (Number.isFinite(Number(v)) ? (Math.round(Number(v) * 10) / 10).toFixed(1) : "—");
const money = (m) => {
  const n = Number(m);
  if (!Number.isFinite(n)) return "—";
  const sign = n < 0 ? "−" : "";
  return `${sign}$${Math.abs(n).toFixed(2)}M`;
};

/* ------------------------------------------------------------------ */
/* Trade Value language                                                */
/* ------------------------------------------------------------------ */

export function tradeValueTier(value, type = "player") {
  const v = Number(value);
  if (!Number.isFinite(v)) return "Unrated";
  if (type === "pick") {
    if (v >= 60) return "Premium First-Rounder";
    if (v >= 38) return "Valuable Pick";
    if (v >= 18) return "Mid-Round Pick";
    return "Late-Round Flyer";
  }
  if (type === "prospect") {
    if (v >= 60) return "Blue-Chip Prospect";
    if (v >= 30) return "Strong Prospect";
    return "Development Project";
  }
  if (v >= 150) return "Generational Franchise Player";
  if (v >= 120) return "Elite Franchise Player";
  if (v >= 90) return "Franchise Cornerstone";
  if (v >= 60) return "Top-Line Difference Maker";
  if (v >= 38) return "Core Contributor";
  if (v >= 18) return "Depth Piece";
  if (v >= 0) return "Replacement Level";
  return "Negative-Value Contract";
}

function tierTone(value) {
  const v = num(value, 0);
  if (v >= 120) return "elite";
  if (v >= 60) return "high";
  if (v >= 18) return "mid";
  if (v >= 0) return "low";
  return "neg";
}

/** All Trade Values in the league for this asset kind (players + affiliates, or picks). */
function leagueValues(meta, type) {
  if (!meta) return [];
  const out = [];
  if (type === "pick") {
    Object.values(meta.picks || {}).forEach((list) =>
      arr(list).forEach((p) => {
        const v = Number(p?.tradeValue ?? p?.value_hint);
        if (Number.isFinite(v)) out.push(v);
      }),
    );
    return out;
  }
  [meta.players, meta.affiliates].forEach((group) =>
    Object.values(group || {}).forEach((list) =>
      arr(list).forEach((p) => {
        const v = Number(p?.tradeValue);
        if (Number.isFinite(v)) out.push(v);
      }),
    ),
  );
  return out;
}

export function leaguePercentileLabel(value, meta, type = "player") {
  const v = Number(value);
  const all = leagueValues(meta, type === "prospect" ? "player" : type);
  if (!Number.isFinite(v) || all.length < 10) return "";
  const above = all.filter((x) => x > v).length;
  const top = Math.max(1, Math.ceil(((above + 1) / all.length) * 100));
  const scope = type === "pick" ? "of tradeable picks" : "league-wide";
  if (top <= 50) return `Top ${top}% ${scope}`;
  return `Bottom ${Math.max(1, 101 - top)}% ${scope}`;
}

function assetKind(asset) {
  if (!asset) return "player";
  if (asset.type === "pick") return "pick";
  if (asset.type === "prospect") return "prospect";
  return "player";
}

function assetTitle(asset) {
  if (!asset) return "";
  if (asset.type === "pick") {
    const r = Number(asset.round);
    const suf = r === 1 ? "1st" : r === 2 ? "2nd" : r === 3 ? "3rd" : `${r}th`;
    return `${asset.year || ""} ${suf} round pick`.trim();
  }
  return asset.name || "Player";
}

function assetValue(asset) {
  const v = Number(asset?.tradeValue ?? asset?.value_hint ?? asset?.value);
  return Number.isFinite(v) ? v : null;
}

/* ------------------------------------------------------------------ */
/* Asset intel (backend)                                               */
/* ------------------------------------------------------------------ */

const INTEL_CACHE = new Map();
const INTEL_TTL_MS = 45000;

export function useAssetIntel(asset, teamId) {
  const type = asset?.type === "pick" ? "pick" : asset ? "player" : "";
  const id = asset ? String(asset.type === "pick" ? asset.pick_id || asset.id : asset.id || "") : "";
  const tid = String(teamId || asset?.teamId || asset?.team || "");
  const key = type && id && tid ? `${type}:${id}:${tid}` : "";
  const [state, setState] = useState(() => {
    const hit = key ? INTEL_CACHE.get(key) : null;
    return hit && Date.now() - hit.at < INTEL_TTL_MS ? { status: "done", data: hit.data } : { status: key ? "loading" : "idle", data: null };
  });
  useEffect(() => {
    if (!key) {
      setState({ status: "idle", data: null });
      return undefined;
    }
    const hit = INTEL_CACHE.get(key);
    if (hit && Date.now() - hit.at < INTEL_TTL_MS) {
      setState({ status: "done", data: hit.data });
      return undefined;
    }
    let alive = true;
    setState({ status: "loading", data: null });
    getTradeAssetIntel({ assetType: type, assetId: id, teamId: tid, limit: 8 })
      .then((data) => {
        INTEL_CACHE.set(key, { at: Date.now(), data });
        if (alive) setState({ status: "done", data });
      })
      .catch((e) => {
        if (alive) setState({ status: "error", data: null, error: e?.response?.data?.detail || "No market read" });
      });
    return () => {
      alive = false;
    };
  }, [key, type, id, tid]);
  return state;
}

function demandTone(demand) {
  const d = String(demand || "").toLowerCase();
  if (d.includes("hot")) return "hot";
  if (d === "high") return "high";
  if (d === "moderate") return "mid";
  return "cold";
}

/* ------------------------------------------------------------------ */
/* TradeValueHero                                                      */
/* ------------------------------------------------------------------ */

export function TradeValueHero({ asset, meta, intel = null, compact = false }) {
  if (!asset) return null;
  const kind = assetKind(asset);
  const value = assetValue(asset);
  const tier = tradeValueTier(value, kind);
  const pct = leaguePercentileLabel(value, meta, kind);
  const data = intel?.data || null;
  const loading = intel?.status === "loading";
  return (
    <div className={`twr-tv-hero tone-${tierTone(value)}${compact ? " is-compact" : ""}`}>
      <div className="twr-tv-hero-num">
        <span className="twr-kicker">Trade Value</span>
        <strong>{value != null ? fmt1(value) : "—"}</strong>
      </div>
      <div className="twr-tv-hero-body">
        <b className="twr-tv-tier">{tier}</b>
        {pct ? <span className="twr-tv-pct">{pct}</span> : null}
        <div className="twr-tv-chips">
          <span className="twr-chip">
            {loading ? "Polling GMs…" : data ? `${data.interested_count} teams interested` : "Interest unknown"}
          </span>
          {data ? <span className={`twr-chip twr-demand tone-${demandTone(data.demand)}`}>Demand: {data.demand}</span> : null}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Comparable deals from trade history                                 */
/* ------------------------------------------------------------------ */

function buildPlayerIndex(meta) {
  const idx = new Map();
  [meta?.players, meta?.affiliates, meta?.prospects].forEach((group) =>
    Object.values(group || {}).forEach((list) =>
      arr(list).forEach((p) => {
        if (p?.id != null) idx.set(String(p.id), p);
      }),
    ),
  );
  return idx;
}

function posGroup(pos) {
  const p = String(pos || "").toUpperCase();
  if (p.startsWith("G")) return "G";
  if (p.includes("D")) return "D";
  return "F";
}

function teamById(meta, id) {
  return arr(meta?.teams).find((t) => String(t.id) === String(id)) || null;
}

export function comparableDeals(asset, meta, tradeHistory, limit = 4) {
  const hist = arr(tradeHistory).slice().reverse();
  if (!hist.length) return { rows: [], isComparable: false };
  const idx = buildPlayerIndex(meta);
  const v = assetValue(asset);
  const kind = assetKind(asset);
  const scored = [];
  hist.forEach((t) => {
    let best = null;
    arr(t.moved_players).forEach((m) => {
      const p = idx.get(String(m.asset_id || m.player_id || ""));
      const pv = Number(p?.tradeValue);
      let score = 0;
      if (kind !== "pick" && p && posGroup(p.pos) === posGroup(asset?.pos)) score += 1;
      if (v != null && Number.isFinite(pv) && v > 0) {
        const ratio = pv / v;
        if (ratio >= 0.65 && ratio <= 1.5) score += 2;
      }
      if (!best || score > best.score) best = { score, name: m.player_name || p?.name, value: Number.isFinite(pv) ? pv : null, pos: p?.pos };
    });
    if (kind === "pick" && arr(t.moved_picks).length) best = { score: 2, name: `${arr(t.moved_picks).length} pick(s)`, value: null };
    scored.push({ t, best });
  });
  const matches = scored.filter((r) => r.best && r.best.score >= 2);
  const rows = (matches.length ? matches : scored).slice(0, limit).map(({ t, best }) => ({
    id: t.trade_id,
    headline: t.headline || "Trade completed",
    when: t.calendar_iso || (t.season_year ? `${t.season_year}` : ""),
    teams: arr(t.participating_teams).map((id) => teamById(meta, id)).filter(Boolean),
    keyName: best?.name || "",
    keyValue: best?.value,
    keyPos: best?.pos || "",
    userInvolved: Boolean(t.user_involved),
    gap: t.fairness_gap,
  }));
  return { rows, isComparable: matches.length > 0 };
}

function DealRow({ row, renderLogo }) {
  return (
    <li className={`twr-deal${row.userInvolved ? " is-user" : ""}`}>
      <div className="twr-deal-logos">
        {row.teams.slice(0, 2).map((t, i) => (
          <React.Fragment key={t.id}>
            {i > 0 ? <i aria-hidden="true">⇄</i> : null}
            {renderLogo ? renderLogo(t, 22) : <b>{t.abbr}</b>}
          </React.Fragment>
        ))}
      </div>
      <div className="twr-deal-main">
        <strong title={row.headline}>{row.headline}</strong>
        <span>
          {row.keyName ? `${row.keyName}${row.keyPos ? ` (${row.keyPos})` : ""}` : "Package deal"}
          {row.keyValue != null ? ` · Trade Value ${fmt1(row.keyValue)}` : ""}
          {row.when ? ` · ${row.when}` : ""}
        </span>
      </div>
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* AssetIntelPanel                                                     */
/* ------------------------------------------------------------------ */

export function AssetIntelPanel({
  asset,
  teamId,
  meta,
  tradeHistory = [],
  renderLogo,
  onOpenTalks,
  title = "Asset Intel",
  emptyText = "Click any player or pick to see who is calling about him.",
}) {
  const intel = useAssetIntel(asset, teamId);
  const comps = useMemo(() => comparableDeals(asset, meta, tradeHistory, 4), [asset, meta, tradeHistory]);
  if (!asset) {
    return (
      <section className="twr-panel twr-intel is-empty">
        <header className="twr-panel-head">
          <span className="twr-kicker">{title}</span>
        </header>
        <p className="twr-empty">{emptyText}</p>
      </section>
    );
  }
  const data = intel.data;
  const bidders = arr(data?.bidders).slice(0, 6);
  return (
    <section className="twr-panel twr-intel">
      <header className="twr-panel-head">
        <span className="twr-kicker">{title}</span>
        <strong title={assetTitle(asset)}>{assetTitle(asset)}</strong>
      </header>
      <TradeValueHero asset={asset} meta={meta} intel={intel} compact />
      {data ? (
        <dl className="twr-intel-stats">
          <div><dt>Market price</dt><dd>{fmt1(data.market_value)}</dd></div>
          <div><dt>Top bid</dt><dd>{fmt1(data.top_bid_value)}</dd></div>
          <div><dt>Clubs checked</dt><dd>{data.teams_checked}</dd></div>
        </dl>
      ) : null}
      <div className="twr-sub">
        <span className="twr-sub-title">Active bidders</span>
        {intel.status === "loading" ? <p className="twr-empty">Calling around the league…</p> : null}
        {intel.status === "error" ? <p className="twr-empty">{String(intel.error || "No market read for this asset.")}</p> : null}
        {data && !bidders.length ? <p className="twr-empty">No club is calling about this asset.</p> : null}
        {bidders.length ? (
          <ul className="twr-bidders">
            {bidders.map((b) => {
              const team = teamById(meta, b.team_id);
              return (
                <li key={b.team_id}>
                  <button
                    type="button"
                    className={`twr-bidder${b.cap_ok ? "" : " is-capped"}`}
                    onClick={onOpenTalks ? () => onOpenTalks(b.team_id) : undefined}
                    disabled={!onOpenTalks}
                    title={onOpenTalks ? `Open talks with ${b.team_name}` : b.team_name}
                  >
                    {team && renderLogo ? renderLogo(team, 24) : null}
                    <span className="twr-bidder-name">
                      <b>{team?.abbr || b.team_name}</b>
                      <em>{arr(b.reasons)[0] || b.window_label}</em>
                    </span>
                    <span className="twr-bidder-meter" aria-label={`Interest ${b.interest}%`}>
                      <i style={{ width: `${clampN(num(b.interest), 2, 100)}%` }} />
                    </span>
                    <strong className="twr-num">{b.interest}%</strong>
                  </button>
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>
      <div className="twr-sub">
        <span className="twr-sub-title">{comps.isComparable ? "Comparable deals" : "Recent league deals"}</span>
        {comps.rows.length ? (
          <ul className="twr-deals">
            {comps.rows.map((r) => <DealRow key={r.id} row={r} renderLogo={renderLogo} />)}
          </ul>
        ) : (
          <p className="twr-empty">No trades on record yet this season.</p>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* PackageList — replaces the five big drop slots                      */
/* ------------------------------------------------------------------ */

export function PackageList({
  title,
  side,
  team,
  slots,
  maxSlots = 5,
  totalValue = null,
  renderLogo,
  renderAsset,
  onAdd,
  onDropAsset,
  addLabel = "+ Add Asset",
  emptyHint = "",
}) {
  const [over, setOver] = useState(false);
  const filled = arr(slots)
    .map((asset, index) => ({ asset, index }))
    .filter((r) => r.asset);
  const full = filled.length >= maxSlots;
  return (
    <section
      className={`twr-pkg side-${side}${over ? " is-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!over) setOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setOver(false);
      }}
      onDrop={(e) => {
        setOver(false);
        onDropAsset?.(e);
      }}
    >
      <header className="twr-pkg-head">
        {team && renderLogo ? renderLogo(team, 34) : null}
        <div className="twr-pkg-title">
          <strong>{title}</strong>
          <span>
            {filled.length} of {maxSlots} assets
            {totalValue != null && filled.length ? ` · Trade Value ${fmt1(totalValue)}` : ""}
          </span>
        </div>
      </header>
      {filled.length ? (
        <ol className="twr-pkg-list">
          {filled.map(({ asset, index }) => (
            <li key={`${side}-${index}-${asset.type}-${asset.id}`} className="twr-pkg-item">
              {renderAsset(asset, index)}
            </li>
          ))}
        </ol>
      ) : (
        <p className="twr-pkg-empty">{emptyHint || "Nothing here yet."}</p>
      )}
      <button type="button" className="twr-add-btn" onClick={onAdd} disabled={full || !onAdd}>
        {full ? "Package full" : addLabel}
      </button>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* TradeAnalysisBoard                                                  */
/* ------------------------------------------------------------------ */

function riskRead({ userOutgoing, partnerOutgoing, userAfterCap }) {
  const flags = [];
  const out = arr(userOutgoing).filter(Boolean);
  const inc = arr(partnerOutgoing).filter(Boolean);
  if (Number.isFinite(userAfterCap) && userAfterCap < 1.5) flags.push(userAfterCap < 0 ? "Over the cap" : "Cap tight after deal");
  if (out.some((p) => p.type !== "pick" && num(p.age, 99) <= 24 && num(p.ovr) >= 78)) flags.push("Moving young core");
  if (out.some((p) => p.type === "pick" && num(p.round) === 1)) flags.push("First-round pick out");
  if (inc.some((p) => p.type === "player" && num(p.age) >= 32)) flags.push("Aging return");
  if (inc.some((p) => p.type === "player" && num(p.capHit) >= 6 && num(p.contractYears ?? p.years_remaining ?? p.term, 0) >= 4)) flags.push("Long-term money");
  if (inc.some((p) => p.is_injured)) flags.push("Injured return");
  const level = flags.length >= 3 ? "High" : flags.length >= 1 ? "Medium" : "Low";
  return { level, flags };
}

export function recommendationFor(userGive, partnerGive, bothSides) {
  if (!bothSides) return { label: "—", tone: "neutral" };
  const diff = num(partnerGive) - num(userGive);
  const scale = Math.max(num(userGive), num(partnerGive), 1);
  const rel = diff / scale;
  if (diff >= 3 && rel >= 0.08) return { label: "Favorable", tone: "good" };
  if (diff <= -3 && rel <= -0.08) return { label: "Unfavorable", tone: "bad" };
  return { label: "Even", tone: "even" };
}

export function TradeAnalysisBoard({
  userGive,
  partnerGive,
  userOutgoing,
  partnerOutgoing,
  partnerAbbr,
  fanData,
  userAfterCap = null,
  evaluation = null,
  hasProposed = false,
  onReviewClick,
}) {
  const bothSides = arr(userOutgoing).some(Boolean) && arr(partnerOutgoing).some(Boolean);
  const anyAssets = arr(userOutgoing).some(Boolean) || arr(partnerOutgoing).some(Boolean);
  const diff = num(partnerGive) - num(userGive);
  const rec = recommendationFor(userGive, partnerGive, bothSides);
  const risk = riskRead({ userOutgoing, partnerOutgoing, userAfterCap });
  const fd = fanData || {};
  const heat = Number.isFinite(Number(fd.heat)) ? Number(fd.heat) : null;
  const fanLabel = anyAssets ? fd.heatLabel || fd.category || (heat != null ? (heat >= 70 ? "Furious" : heat >= 45 ? "Uneasy" : "Calm") : "—") : "—";
  const fanTone = heat == null ? "neutral" : heat >= 70 ? "bad" : heat >= 45 ? "warn" : "good";
  const peak = Math.max(num(userGive), num(partnerGive), 1);
  const verdict = hasProposed && evaluation
    ? evaluation.accepted && evaluation.can_execute
      ? "GM says yes"
      : evaluation.can_execute === false
        ? "Rules block it"
        : "GM says no"
    : "";
  return (
    <section className="twr-panel twr-analysis">
      <header className="twr-panel-head">
        <span className="twr-kicker">Trade Analysis</span>
        {verdict ? <strong className={`twr-verdict tone-${evaluation?.accepted && evaluation?.can_execute ? "good" : "bad"}`}>{verdict}</strong> : null}
        {hasProposed && bothSides && onReviewClick ? (
          <button type="button" className="twr-mini-btn" onClick={onReviewClick}>Full review</button>
        ) : null}
      </header>
      <div className="twr-analysis-grid">
        <div className="twr-stat">
          <span>You Send</span>
          <strong>{anyAssets ? fmt1(userGive) : "—"}</strong>
          <i className="twr-bar side-left"><b style={{ width: `${clampN((num(userGive) / peak) * 100, 0, 100)}%` }} /></i>
        </div>
        <div className="twr-stat">
          <span>You Receive</span>
          <strong>{anyAssets ? fmt1(partnerGive) : "—"}</strong>
          <i className="twr-bar side-right"><b style={{ width: `${clampN((num(partnerGive) / peak) * 100, 0, 100)}%` }} /></i>
        </div>
        <div className={`twr-stat tone-${!bothSides ? "neutral" : diff >= 0 ? "good" : "bad"}`}>
          <span>Trade Value Difference</span>
          <strong>{bothSides ? `${diff >= 0 ? "+" : "−"}${fmt1(Math.abs(diff))}` : "—"}</strong>
          <em>{bothSides ? (diff >= 0 ? `Trade Value Added for you` : `Trade Value Removed from you`) : "Add assets to both sides"}</em>
        </div>
        <div className={`twr-stat tone-${rec.tone}`}>
          <span>Recommendation</span>
          <strong>{rec.label}</strong>
          <em>{bothSides ? (rec.tone === "good" ? `You win on value vs ${partnerAbbr || "them"}` : rec.tone === "bad" ? "You are overpaying" : "Close to fair") : "—"}</em>
        </div>
        <div className={`twr-stat tone-${anyAssets ? fanTone : "neutral"}`}>
          <span>Fan Reaction</span>
          <strong>{fanLabel}</strong>
          <em>{anyAssets && arr(fd.factors).length ? arr(fd.factors).slice(0, 2).join(" · ") : anyAssets ? "Fans are watching" : "—"}</em>
        </div>
        <div className={`twr-stat tone-${!anyAssets ? "neutral" : risk.level === "High" ? "bad" : risk.level === "Medium" ? "warn" : "good"}`}>
          <span>Risk</span>
          <strong>{anyAssets ? risk.level : "—"}</strong>
          <em>{anyAssets ? (risk.flags.slice(0, 2).join(" · ") || "No red flags") : "—"}</em>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* GmNegotiationPanel                                                  */
/* ------------------------------------------------------------------ */

const GM_LINES = {
  "Prospect Hoarder": {
    yes: "Kids and picks — that's how you build a contender. Deal.",
    close: "We're close. Throw in a pick or a prospect and I'll sign today.",
    no: "I'm not giving up on our future for that. Come back with youth.",
    illegal: "I like the idea, but the league won't approve it as built.",
  },
  "Aggressive Buyer": {
    yes: "That's the piece we needed for a run. Let's do it.",
    close: "I want this done before someone else calls. Sweeten it a little.",
    no: "We're trying to win now — that package doesn't move the needle.",
    illegal: "Love the hockey side, but the cap math kills it.",
  },
  "Cap Dumping": {
    yes: "Cap relief and a fair return. You've got a deal.",
    close: "I need a bit more for the trouble — or take some salary off my hands.",
    no: "That doesn't help my books or my rebuild. Pass.",
    illegal: "My hands are tied on the cap. Fix the money and call me back.",
  },
  "Win Now": {
    yes: "Good hockey trade. We both get better.",
    close: "I'm interested. Find me a little more and we'll shake on it.",
    no: "My fans would run me out of town for that one.",
    illegal: "The rules won't let me do that one.",
  },
  "Future Assets": {
    yes: "Futures for now — works for us.",
    close: "Add a pick and we're there.",
    no: "Not enough future value in there for me.",
    illegal: "Can't register that one with the league.",
  },
  default: {
    yes: "That works for us. Let's get it done.",
    close: "We're in the neighbourhood. A little more and it's done.",
    no: "I'll pass. The value isn't there for my club.",
    illegal: "Can't make that one work under the rules.",
  },
};

function gmMood(evaluation, interest) {
  if (!evaluation) return "idle";
  if (evaluation.accepted && evaluation.can_execute) return "yes";
  if (evaluation.can_execute === false) return "illegal";
  if (interest != null && interest >= 0.4) return "close";
  return "no";
}

function relationshipRead(evaluation, mood) {
  const pat = evaluation?.gm_patience;
  if (mood === "yes") return { tone: "good", text: "Relationship improves — he'll take your calls first next time." };
  if (pat && Number.isFinite(Number(pat.rejections_today))) {
    const left = Math.max(0, num(pat.limit) - num(pat.rejections_today));
    return {
      tone: left <= 1 ? "bad" : "warn",
      text: left > 0
        ? `Patience wearing thin: ${pat.rejections_today} rejected offer(s) today, ${left} more before he hangs up.`
        : "He has stopped taking your calls today.",
    };
  }
  if (mood === "no") return { tone: "warn", text: "Lowball offers cost goodwill. Close the gap before proposing again." };
  if (mood === "illegal") return { tone: "neutral", text: "No relationship hit — the deal never reached his desk." };
  return { tone: "neutral", text: "" };
}

function cleanReason(r) {
  return String(r || "").replace(/^[A-Z]{2,4}:\s*/, "").trim();
}

function SuggestionRow({ s, onApply, renderName }) {
  const adds = arr(s.add);
  const drops = arr(s.drop);
  const label = adds.length
    ? `Add ${adds.map(renderName).join(" + ")}`
    : drops.length
      ? `Drop ${drops.map(renderName).join(" + ")} from your ask`
      : "Adjust package";
  return (
    <li className="twr-counter">
      <div>
        <strong>{label}</strong>
        <span>
          {s.interest != null ? `GM interest ${Math.round(num(s.interest) * 100)}% · accepted as-is` : "Accepted as-is"}
          {adds.length && s.value_added ? ` · Trade Value Added ${fmt1(s.value_added)}` : ""}
        </span>
      </div>
      {onApply ? (
        <button type="button" className="twr-mini-btn is-go" onClick={() => onApply(s)}>Apply</button>
      ) : null}
    </li>
  );
}

function suggestionName(a) {
  if (!a) return "?";
  if (a.type === "pick") {
    const r = Number(a.round);
    const suf = r === 1 ? "1st" : r === 2 ? "2nd" : r === 3 ? "3rd" : `${r}th`;
    return `${a.year || ""} ${suf}`.trim();
  }
  return a.name || "player";
}

export function GmNegotiationPanel({
  evaluation,
  hasProposed,
  partnerTeam,
  partnerId,
  assetsPayload,
  onApplySuggestion,
  renderLogo,
}) {
  const [counter, setCounter] = useState({ status: "idle", data: null });
  const payloadKey = useMemo(() => JSON.stringify(assetsPayload || {}), [assetsPayload]);
  useEffect(() => {
    setCounter({ status: "idle", data: null });
  }, [payloadKey]);

  if (!partnerTeam) return null;
  const interestRaw = evaluation?.interest_level?.[partnerId];
  const interest = hasProposed && interestRaw != null ? clampN(num(interestRaw), 0, 1) : null;
  const mood = hasProposed ? gmMood(evaluation, interest) : "idle";
  const lines = GM_LINES[partnerTeam.gmPersonality] || GM_LINES.default;
  const quote = mood === "idle"
    ? `${partnerTeam.gmPersonality || "Pragmatic Dealer"}. Build a package and propose — I'll tell you straight.`
    : lines[mood];
  const reasons = hasProposed ? arr(evaluation?.rejection_reasons).map(cleanReason).filter(Boolean).slice(0, 3) : [];
  const engineCounters = hasProposed ? arr(evaluation?.suggested_counteroffers).slice(0, 2) : [];
  const rel = relationshipRead(evaluation, mood);
  const canAsk = hasProposed && mood !== "yes" && mood !== "illegal" && assetsPayload;

  const ask = async () => {
    setCounter({ status: "loading", data: null });
    try {
      const data = await getTradeCounter({ assets_by_team: assetsPayload });
      setCounter({ status: "done", data });
    } catch (e) {
      setCounter({ status: "error", data: null, error: e?.response?.data?.detail || "He didn't pick up." });
    }
  };
  const sugg = arr(counter.data?.suggestions);

  return (
    <section className={`twr-panel twr-gm mood-${mood}`}>
      <header className="twr-panel-head">
        {renderLogo ? renderLogo(partnerTeam, 30) : null}
        <div className="twr-gm-who">
          <span className="twr-kicker">{partnerTeam.abbr || "Partner"} GM</span>
          <strong>{partnerTeam.gmPersonality || "Pragmatic Dealer"}</strong>
        </div>
        {interest != null ? (
          <div className="twr-gm-interest" title="How close their GM is to saying yes">
            <span>Interest</span>
            <strong className="twr-num">{Math.round(interest * 100)}%</strong>
            <i className="twr-bar"><b style={{ width: `${Math.round(interest * 100)}%` }} /></i>
          </div>
        ) : null}
      </header>
      <blockquote className="twr-gm-quote">“{quote}”</blockquote>
      {reasons.length ? (
        <div className="twr-sub">
          <span className="twr-sub-title">{mood === "illegal" ? "Why it can't be registered" : "Why he said no"}</span>
          <ul className="twr-reasons">
            {reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
        </div>
      ) : null}
      {engineCounters.length ? (
        <div className="twr-sub">
          <span className="twr-sub-title">His counter</span>
          <ul className="twr-counters">
            {engineCounters.map((c, i) => (
              <li key={`${c.label}-${i}`} className="twr-counter">
                <div>
                  <strong>{c.label}</strong>
                  {c.explanation ? <span>{c.explanation}</span> : null}
                </div>
                {c.add_asset && onApplySuggestion ? (
                  <button type="button" className="twr-mini-btn is-go" onClick={() => onApplySuggestion({ add: [c.add_asset], drop: [] })}>Add</button>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {canAsk ? (
        <div className="twr-sub">
          <span className="twr-sub-title">What would it take?</span>
          {counter.status === "idle" ? (
            <button type="button" className="twr-ask-btn" onClick={ask}>Ask their GM what works</button>
          ) : null}
          {counter.status === "loading" ? <p className="twr-empty">He's running the numbers…</p> : null}
          {counter.status === "error" ? <p className="twr-empty">{String(counter.error)}</p> : null}
          {counter.status === "done" && counter.data?.illegal ? (
            <p className="twr-empty">Fix the rule problem first: {cleanReason(arr(counter.data.reasons)[0])}</p>
          ) : null}
          {counter.status === "done" && !counter.data?.illegal && !sugg.length ? (
            <p className="twr-empty">
              Nothing from your organization closes a {fmt1(counter.data?.value_gap)} Trade Value gap on its own. Rework the ask.
            </p>
          ) : null}
          {sugg.length ? (
            <ul className="twr-counters">
              {sugg.map((s, i) => (
                <SuggestionRow key={i} s={s} onApply={onApplySuggestion} renderName={suggestionName} />
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
      {rel.text ? <p className={`twr-rel tone-${rel.tone}`}>{rel.text}</p> : null}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* MarketIntelPanel                                                    */
/* ------------------------------------------------------------------ */

function sideOf(team) {
  const label = String(team?.tradeDirectionLabel || team?.direction || "").toUpperCase();
  if (/SELL|REBUILD|RETOOL|TANK|DECLIN/.test(label)) return "seller";
  if (/CONTEND|BUYER|WIN/.test(label)) return "buyer";
  return "neutral";
}

export function MarketIntelPanel({ meta, tradeHistory = [], tradeMarket = null, renderLogo, onOpenTalks }) {
  const teams = arr(meta?.teams).filter((t) => String(t.id) !== String(meta?.userTeamId));
  const buyers = teams.filter((t) => sideOf(t) === "buyer").length;
  const sellers = teams.filter((t) => sideOf(t) === "seller").length;
  const hist = arr(tradeHistory);
  const activity = {};
  hist.forEach((t) => arr(t.participating_teams).forEach((id) => { activity[String(id)] = (activity[String(id)] || 0) + 1; }));
  const active = Object.entries(activity)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([id, n]) => ({ team: teamById(meta, id), n }))
    .filter((r) => r.team);
  const capLeaders = teams
    .filter((t) => Number.isFinite(Number(t.capSpace)))
    .sort((a, b) => num(b.capSpace) - num(a.capSpace))
    .slice(0, 4);
  const recent = comparableDeals(null, meta, hist, 6).rows;
  const ratio = buyers + sellers > 0 ? buyers / (buyers + sellers) : 0.5;
  const demand = ratio >= 0.62 ? "Seller's market" : ratio <= 0.38 ? "Buyer's market" : "Balanced market";
  return (
    <aside className="twr-market-intel">
      <section className="twr-panel">
        <header className="twr-panel-head">
          <span className="twr-kicker">Market Demand</span>
          <strong>{demand}</strong>
        </header>
        <div className="twr-demand-split" aria-label={`${buyers} buyers, ${sellers} sellers`}>
          <i className="is-buy" style={{ flexGrow: Math.max(1, buyers) }}><b>{buyers}</b> buyers</i>
          <i className="is-sell" style={{ flexGrow: Math.max(1, sellers) }}><b>{sellers}</b> sellers</i>
        </div>
        <p className="twr-note">
          {ratio >= 0.62
            ? "More clubs are buying than selling — your veterans fetch a premium."
            : ratio <= 0.38
              ? "Sellers outnumber buyers — targets come cheaper right now."
              : "Prices sit near market. Fit decides close deals."}
          {tradeMarket?.market_temperature ? ` Market is ${String(tradeMarket.market_temperature).toLowerCase()}.` : ""}
        </p>
      </section>
      <section className="twr-panel">
        <header className="twr-panel-head">
          <span className="twr-kicker">Trade History</span>
          <strong>{hist.length} deal{hist.length === 1 ? "" : "s"}</strong>
        </header>
        {recent.length ? (
          <ul className="twr-deals">{recent.map((r) => <DealRow key={r.id} row={r} renderLogo={renderLogo} />)}</ul>
        ) : (
          <p className="twr-empty">No trades yet this season. Be the first mover.</p>
        )}
      </section>
      <section className="twr-panel">
        <header className="twr-panel-head">
          <span className="twr-kicker">Who's Dealing</span>
        </header>
        <div className="twr-two-col">
          <div>
            <span className="twr-sub-title">Most active</span>
            {active.length ? (
              <ul className="twr-mini-list">
                {active.map(({ team, n }) => (
                  <li key={team.id}>
                    <button type="button" onClick={onOpenTalks ? () => onOpenTalks(team.id) : undefined} disabled={!onOpenTalks}>
                      {renderLogo ? renderLogo(team, 20) : null}<b>{team.abbr}</b><em>{n} deal{n === 1 ? "" : "s"}</em>
                    </button>
                  </li>
                ))}
              </ul>
            ) : <p className="twr-empty">Quiet so far.</p>}
          </div>
          <div>
            <span className="twr-sub-title">Most cap room</span>
            <ul className="twr-mini-list">
              {capLeaders.map((team) => (
                <li key={team.id}>
                  <button type="button" onClick={onOpenTalks ? () => onOpenTalks(team.id) : undefined} disabled={!onOpenTalks}>
                    {renderLogo ? renderLogo(team, 20) : null}<b>{team.abbr}</b><em>{money(team.capSpace)}</em>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/* MarketableAssets — Trade Finder idle state                          */
/* ------------------------------------------------------------------ */

export function MarketableAssets({ meta, teamId, onPick, limit = 6 }) {
  const tid = String(teamId || meta?.userTeamId || "");
  const rows = [...arr(meta?.players?.[tid]), ...arr(meta?.affiliates?.[tid])]
    .filter((p) => p && Number.isFinite(Number(p.tradeValue)))
    .sort((a, b) => num(b.tradeValue) - num(a.tradeValue))
    .slice(0, limit);
  if (!rows.length) return null;
  return (
    <section className="twr-panel twr-marketable">
      <header className="twr-panel-head">
        <span className="twr-kicker">Your most marketable assets</span>
      </header>
      <ul className="twr-marketable-list">
        {rows.map((p) => (
          <li key={p.id}>
            <button type="button" onClick={onPick ? () => onPick(p) : undefined} disabled={!onPick}>
              <span className={`twr-tv-num tone-${tierTone(p.tradeValue)}`}>{fmt1(p.tradeValue)}</span>
              <span className="twr-marketable-main">
                <b>{p.name}</b>
                <em>{p.pos} · {tradeValueTier(p.tradeValue, assetKind(p))}</em>
              </span>
              <span className="twr-marketable-pct">{leaguePercentileLabel(p.tradeValue, meta, assetKind(p)).replace(" league-wide", "")}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
