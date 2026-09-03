import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { isFranchiseCinematicPopup } from "../../events/franchiseEventKinds";
import { getTeamLogoSrc, toLogoUrl } from "../../utils/teamLogos";

function playerInitials(name) {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function resolveAlertTheme(pop) {
  const theme = pop.theme || pop.presentation_type || "";
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

function StatCard({ label, value, sub }) {
  return (
    <div className="media-alert__stat">
      <span className="media-alert__stat-label">{label}</span>
      <strong className="media-alert__stat-value">{value}</strong>
      {sub ? <small className="media-alert__stat-sub">{sub}</small> : null}
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

function TeamMark({ team }) {
  const abbr = String(team?.abbreviation || team?.team_id || "?").toUpperCase();
  const src = toLogoUrl(
    getTeamLogoSrc({
      abbrev: abbr,
      team_abbrev: abbr,
      name: team?.display_name,
      team_name: team?.display_name,
    })
  );
  if (src) {
    return <img className="trade-wire__logo" src={src} alt="" />;
  }
  return <span className="trade-wire__logo-fallback">{abbr.slice(0, 3)}</span>;
}

function TradePlayerCard({ asset, seasonLabel }) {
  const stats = asset?.season_stats || {};
  const role = asset?.role_line || [asset?.position, asset?.archetype].filter(Boolean).join(" | ");
  const cap = formatCapHit(asset?.cap_hit_m);
  const years = asset?.years_left;
  const age = asset?.age;
  const metaBits = [];
  if (age != null) metaBits.push(`AGE ${age}`);
  if (cap) metaBits.push(`${cap} CAP HIT`);
  if (years != null) metaBits.push(`${years} YEAR${years === 1 ? "" : "S"} LEFT`);
  if (asset?.retained_salary) metaBits.push(`${asset.retained_salary}% RET`);

  return (
    <div className="trade-wire__player-card">
      <div className="trade-wire__player-top">
        <div className="trade-wire__player-id">
          <div className="trade-wire__player-name">{asset?.display_name || "Player"}</div>
          {role ? <div className="trade-wire__player-role">{role}</div> : null}
        </div>
        {asset?.ovr != null ? (
          <div className="trade-wire__ovr">
            <span>OVR</span>
            <strong>{asset.ovr}</strong>
          </div>
        ) : null}
      </div>
      {metaBits.length ? <div className="trade-wire__contract">{metaBits.join(" | ")}</div> : null}
      <div className="trade-wire__stats-head">{seasonLabel ? `${seasonLabel} STATS` : "SEASON STATS"}</div>
      <div className="trade-wire__stats-row">
        {[
          ["GP", formatStat(stats.gp)],
          ["G", formatStat(stats.g)],
          ["A", formatStat(stats.a)],
          ["PTS", formatStat(stats.pts)],
          ["xGF%", stats.xgf_pct != null ? formatStat(stats.xgf_pct, 1) : "—"],
          ["WAR", stats.war != null ? formatStat(stats.war, 1) : "—"],
        ].map(([label, value]) => (
          <div key={label} className="trade-wire__stat">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function TradePickChip({ asset }) {
  return (
    <div className="trade-wire__pick-chip">
      <span className="trade-wire__pick-label">{asset?.display_name || "Draft Pick"}</span>
      {asset?.trade_value != null ? (
        <span className="trade-wire__pick-val">TV {formatStat(asset.trade_value, 1)}</span>
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
  const abbr = String(team?.abbreviation || team?.team_id || "TEAM").toUpperCase();

  return (
    <div className="trade-wire__side">
      <div className="trade-wire__side-head">
        <TeamMark team={team} />
        <div>
          <div className="trade-wire__side-name">{abbr}</div>
          <div className="trade-wire__side-receives">RECEIVES</div>
        </div>
      </div>
      <div className="trade-wire__side-assets">
        {players.length
          ? players.map((p, i) => (
              <TradePlayerCard key={`${p.player_id || p.display_name}-${i}`} asset={p} seasonLabel={seasonLabel} />
            ))
          : null}
        {picks.length ? (
          <div className="trade-wire__picks">
            <div className="trade-wire__picks-label">DRAFT PICKS</div>
            {picks.map((pk, i) => (
              <TradePickChip key={`${pk.pick_id || pk.display_name}-${i}`} asset={pk} />
            ))}
          </div>
        ) : null}
        {other.map((a, i) => (
          <div key={`other-${i}`} className="trade-wire__pick-chip">
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
      <div className="trade-wire__section-label">
        <span aria-hidden>▮</span> TRADE VALUE
      </div>
      <div className="trade-wire__value-row">
        <div className="trade-wire__value-end">
          <TeamMark team={leftTeam} />
          <strong>{hasLeft ? formatStat(left, 1) : "—"}</strong>
        </div>
        <div className="trade-wire__value-track" aria-hidden>
          <div className="trade-wire__value-fill is-left" style={{ width: `${leftPct}%` }} />
          <div className="trade-wire__value-fill is-right" style={{ width: `${100 - leftPct}%` }} />
        </div>
        <div className="trade-wire__value-end is-right">
          <strong>{hasRight ? formatStat(right, 1) : "—"}</strong>
          <TeamMark team={rightTeam} />
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

  return (
    <div className="trade-wire">
      <div className="trade-wire__source-row">
        <span className="trade-wire__source-icon" aria-hidden>
          ⇄
        </span>
        <div className="trade-wire__source-copy">
          <p className="trade-wire__source">{pop.source_label || "League Trade Wire"}</p>
          {pop.calendar_iso ? <span className="trade-wire__date">{pop.calendar_iso}</span> : null}
        </div>
        <button type="button" className="trade-wire__close" onClick={onDismiss} aria-label="Close">
          ×
        </button>
      </div>

      <h3 className="trade-wire__title">TRADE COMPLETED</h3>
      <p className="trade-wire__summary">{summary}</p>

      <div className="trade-wire__exchange">
        <TradeSideColumn team={left} seasonLabel={seasonLabel} />
        <div className="trade-wire__swap" aria-hidden>
          ⇄
        </div>
        <TradeSideColumn team={right} seasonLabel={seasonLabel} />
      </div>

      <TradeValueBar leftTeam={left} rightTeam={right} leftValue={leftValue} rightValue={rightValue} />

      <section className="trade-wire__reason">
        <div className="trade-wire__reason-meta">
          <span className="trade-wire__section-label">TRADE TYPE</span>
          <span className="trade-wire__type-pill">{tradeType}</span>
        </div>
        <div className="trade-wire__reason-body">
          <div className="trade-wire__section-label">REASONING</div>
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

function MediaAlertShell({ pop, children, onDismiss, onAction, actions = [], queueCount = 0 }) {
  const theme = resolveAlertTheme(pop);
  const source = pop.source_label || pop.title || "League Update";
  const icon = pop.icon || "◉";
  const kindLabel = String(pop.kind || pop.type || "alert")
    .replace(/_/g, " ")
    .toUpperCase();

  return (
    <div className={`media-alert media-alert--${theme} media-alert--v2`}>
      <div className="media-alert__topbar">
        <span className="media-alert__kind-pill">{kindLabel}</span>
        {queueCount > 1 ? (
          <span className="media-alert__queue-pill">{queueCount - 1} more queued</span>
        ) : null}
        <button type="button" className="media-alert__close" onClick={onDismiss} aria-label="Dismiss alert">
          ×
        </button>
      </div>

      <div className="media-alert__source-row">
        <span className="media-alert__icon" aria-hidden>
          {icon}
        </span>
        <div>
          <p className="media-alert__source">{source}</p>
          {pop.calendar_iso ? <span className="media-alert__date">{pop.calendar_iso}</span> : null}
        </div>
      </div>

      <div className="media-alert__hero">
        <div className="media-alert__avatar">{playerInitials(pop.player_name)}</div>
        <div className="media-alert__hero-text">
          <h3 className="media-alert__headline" id="showcase-popup-title">
            {pop.headline || pop.title || "Update"}
          </h3>
          <p className="media-alert__player-line">
            <strong>{pop.player_name || "—"}</strong>
            {pop.team_abbrev || pop.team_abbr ? (
              <span className="media-alert__team-badge">{pop.team_abbrev || pop.team_abbr}</span>
            ) : null}
          </p>
        </div>
      </div>

      <div className="media-alert__body-scroll">{children}</div>

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
    pop.headline ||
    pop.story_report ||
    pop.summary ||
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
    { label: "Source", value: sourceLabel },
    { label: "Player", value: pop.player_name || pop.culprit_player_name || "—" },
    { label: "Team", value: pop.team_abbrev || pop.team_abbr || "—" },
    {
      label: "Status",
      value: demand
        ? demand.disruptor
          ? "Locker-room disruptor"
          : "Trade demand"
        : hasGames
          ? "Away from team"
          : pop.legal_severity === "major"
            ? "Under review"
            : "Active",
    },
  ];
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

  const actions = [
    { id: "storylines", label: "Open Storylines", primary: !demand },
    { id: demand ? "tradehub" : "roster", label: demand ? "Open Trade Hub" : "View Player", primary: Boolean(demand) },
  ];
  if (pop.is_user_team) {
    actions.unshift({ id: "calendar", label: "View Calendar" });
  }

  return (
    <MediaAlertShell pop={{ ...pop, source_label: sourceLabel, headline: pop.headline || sourceLabel }} onDismiss={onDismiss} onAction={onAction} actions={actions} queueCount={queueCount}>
      <div className="media-alert__stat-grid">
        {stats.map((s) => (
          <StatCard key={s.label} label={s.label} value={s.value} sub={s.sub} />
        ))}
      </div>

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

      <section className="media-alert__section">
        <h4 className="media-alert__section-title">{demand ? "Demand" : "Story Report"}</h4>
        <p className="media-alert__story">{storyText}</p>
      </section>

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
      ) : (
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
          {!impactText && pop.overall_delta == null && !(Array.isArray(pop.impact_lines) && pop.impact_lines.length) ? (
            <p className="media-alert__impact-muted">No direct rating change reported.</p>
          ) : null}
        </section>
      )}

      {pop.requires_decision ? (
        <p className="media-alert__callout">GM response may be required — check Storylines → Decisions.</p>
      ) : null}
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
        icon: "+",
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
      <div className="media-alert__stat-grid">
        <StatCard label="Player" value={pop.player_name || "—"} />
        <StatCard label="Team" value={pop.team_abbrev || "—"} />
        <StatCard label="Severity" value={tier || "unknown"} />
        <StatCard label="Timeline" value={pop.games != null ? `${pop.games} games` : "TBD"} sub={inj || ""} />
      </div>

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
  const po = pop.playoffs || {};
  const prospects = pop.user_prospects || [];
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
    <style>{`
      .showcase-popup {
        position: fixed;
        inset: 0;
        z-index: 14000;
        display: grid;
        place-items: stretch;
        pointer-events: auto;
      }
      .showcase-popup__backdrop {
        position: absolute;
        inset: 0;
        background:
          linear-gradient(90deg, rgba(2, 10, 17, 0.94) 0%, rgba(2, 10, 17, 0.82) 38%, rgba(2, 10, 17, 0.55) 100%),
          repeating-linear-gradient(0deg, rgba(19, 216, 231, 0.03) 0px, rgba(19, 216, 231, 0.03) 1px, transparent 1px, transparent 3px);
        backdrop-filter: blur(2px);
      }
      .showcase-popup__panel {
        position: relative;
        margin: 0;
        width: min(720px, calc(100vw - 24px));
        max-height: calc(100dvh - 24px);
        align-self: center;
        justify-self: end;
        margin-right: max(12px, env(safe-area-inset-right));
        border-radius: var(--radius-hud, 4px);
        border: 1px solid var(--ops-grid-2, rgba(115, 229, 241, 0.25));
        border-left: 4px solid var(--ops-cyan, #13d8e7);
        background: var(--ops-panel, rgba(9, 25, 38, 0.98));
        box-shadow: var(--depth-overlay, 0 24px 70px rgba(0, 0, 0, 0.42));
        overflow: hidden;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) auto;
      }
      .showcase-popup__panel--media,
      .showcase-popup__panel--trade-wire {
        width: min(560px, calc(100vw - 24px));
      }
      .showcase-popup__head {
        padding: 10px 14px 8px;
        border-bottom: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        background: rgba(0, 0, 0, 0.22);
      }
      .showcase-popup__title {
        margin: 0;
        font-size: var(--type-ops-heading-size, 0.95rem);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ops-text, #e9f7fb);
      }
      .showcase-popup__season {
        margin-top: 4px;
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ops-text-secondary, #8096a8);
      }
      .showcase-popup__body {
        min-height: 0;
        overflow: auto;
        padding: 12px 14px;
      }
      .showcase-popup__notice {
        display: grid;
        gap: 8px;
      }
      .showcase-popup__notice-kicker {
        margin: 0;
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ops-cyan, #13d8e7);
      }
      .showcase-popup__notice-copy {
        margin: 0;
        color: var(--ops-text, #e9f7fb);
        font-size: 0.9rem;
        line-height: 1.45;
      }
      .showcase-popup__foot {
        padding: 10px 14px;
        border-top: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        background: rgba(0, 0, 0, 0.18);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }
      .showcase-popup__queue,
      .media-alert__queue {
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ops-gold, #e9a83c);
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }
      .media-alert__clear-trades,
      .trade-wire__btn--clear {
        min-height: 30px;
        padding: 0 10px;
        border-radius: var(--radius-control, 6px);
        border: 1px solid rgba(233, 168, 60, 0.45);
        background: rgba(233, 168, 60, 0.12);
        color: var(--ops-gold, #e9a83c);
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
      }
      .media-alert__clear-trades:hover,
      .trade-wire__btn--clear:hover {
        border-color: var(--ops-gold, #e9a83c);
        background: rgba(233, 168, 60, 0.22);
      }
      .showcase-popup__btn,
      .media-alert__continue,
      .media-alert__dismiss {
        min-height: 34px;
        padding: 0 14px;
        border-radius: var(--radius-control, 6px);
        border: 1px solid var(--ops-grid-2, rgba(115, 229, 241, 0.25));
        background: rgba(255, 255, 255, 0.04);
        color: var(--ops-text, #e9f7fb);
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        cursor: pointer;
      }
      .showcase-popup__btn:hover,
      .media-alert__continue:hover {
        border-color: var(--ops-cyan, #13d8e7);
        background: var(--ops-cyan-soft, rgba(19, 216, 231, 0.13));
      }
      .media-alert {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .media-alert__source-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
      }
      .media-alert__icon {
        width: 28px;
        height: 28px;
        flex: 0 0 28px;
        display: grid;
        place-items: center;
        border: 1px solid var(--ops-grid-2, rgba(115, 229, 241, 0.25));
        border-radius: var(--radius-ops, 2px);
        background: rgba(0, 0, 0, 0.22);
        font-size: 12px;
        font-weight: 900;
        color: var(--ops-cyan, #13d8e7);
      }
      .media-alert__source {
        margin: 0;
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ops-cyan, #13d8e7);
      }
      .media-alert__date {
        display: block;
        margin-top: 2px;
        font-size: var(--type-table-meta-size, 0.72rem);
        color: var(--ops-text-secondary, #8096a8);
      }
      .media-alert__hero {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 10px;
        align-items: center;
      }
      .media-alert__avatar {
        width: 44px;
        height: 44px;
        display: grid;
        place-items: center;
        border: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        border-radius: var(--radius-ops, 2px);
        background: rgba(0, 0, 0, 0.22);
        font-weight: 900;
        letter-spacing: 0.04em;
        color: var(--ops-text-secondary, #8096a8);
      }
      .media-alert__headline {
        margin: 0;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.2;
      }
      .media-alert__player-line {
        margin: 4px 0 0;
        font-size: var(--type-body-size, 0.875rem);
        color: var(--ops-text-secondary, #8096a8);
      }
      .media-alert__team-badge {
        margin-left: 6px;
        padding: 2px 6px;
        border: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        border-radius: var(--radius-ops, 2px);
        font-size: var(--type-table-meta-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .media-alert__stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
      }
      .media-alert__stat {
        padding: 8px;
        border: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        border-radius: var(--radius-ops, 2px);
        background: rgba(0, 0, 0, 0.14);
      }
      .media-alert__stat-label {
        display: block;
        font-size: var(--type-phase-label-size, 0.68rem);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ops-text-secondary, #8096a8);
      }
      .media-alert__stat-value {
        display: block;
        margin-top: 4px;
        font-size: var(--type-body-size, 0.875rem);
        font-weight: 800;
      }
      .media-alert__section {
        padding: 8px 0;
        border-top: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
      }
      .media-alert__section-title {
        margin: 0 0 6px;
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ops-text-secondary, #8096a8);
      }
      .media-alert__story,
      .media-alert__impact-line {
        margin: 0;
        font-size: var(--type-body-size, 0.875rem);
        line-height: 1.45;
        color: var(--ops-text, #e9f7fb);
      }
      .media-alert__impact-list {
        margin: 8px 0 0;
        padding-left: 18px;
        color: var(--ops-text-secondary, #b8c8d4);
        font-size: 0.82rem;
        line-height: 1.4;
      }
      .media-alert__actions {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .media-alert__action {
        min-height: 32px;
        padding: 0 12px;
        border-radius: var(--radius-control, 6px);
        border: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
        background: rgba(255, 255, 255, 0.03);
        color: var(--ops-text, #e9f7fb);
        font-size: var(--type-dept-label-size, 0.72rem);
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
      }
      .media-alert__action.is-primary {
        border-color: var(--ops-cyan, #13d8e7);
        background: var(--ops-cyan, #13d8e7);
        color: var(--ops-navy, #04101a);
      }
      .media-alert__foot {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        padding-top: 8px;
        border-top: 1px solid var(--ops-grid, rgba(156, 218, 236, 0.14));
      }
      .trade-wire__title {
        margin: 0 0 8px;
        font-size: var(--type-ops-heading-size, 0.95rem);
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }
    `}</style>
  );
}

export function ShowcasePopupLayer() {
  const { franchiseState, onDismissShowcasePopups, setScreen } = useGameUI();
  const rawQueue = (franchiseState?.pending_ui_popups || []).filter(
    (p) => p && !isFranchiseCinematicPopup(p)
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

  const dismiss = useCallback(() => {
    if (!first) return;
    const pid = String(first.id || first.popup_id || "").trim();
    if (pid) {
      onDismissShowcasePopups([pid]);
      return;
    }
    onDismissShowcasePopups([`__drop_first__:${String(first.kind || "popup")}`]);
  }, [first, onDismissShowcasePopups]);

  useEffect(() => {
    if (!first) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") dismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [first, dismiss]);

  if (!first) return null;

  const kind = first.kind;
  const theme = resolveAlertTheme(first);

  const tradeQueueIds = visiblePopups.filter(isTradePopup).map((p) => p.id).filter(Boolean);
  const dismissAllTrades = () => {
    if (tradeQueueIds.length) onDismissShowcasePopups(tradeQueueIds);
  };

  const handleAction = (act) => {
    dismiss();
    if (act.id === "storylines") setScreen?.(SCREENS.STORYLINES);
    else if (act.id === "roster") setScreen?.(SCREENS.ROSTER);
    else if (act.id === "calendar") setScreen?.(SCREENS.CALENDAR);
    else if (act.id === "tradehub") setScreen?.(SCREENS.TRADE);
    else if (act.id === "lines") setScreen?.(SCREENS.EDIT_LINES);
  };

  const isMediaAlert =
    kind === "storyline" ||
    kind === "legal_trouble" ||
    kind === "injury" ||
    kind === "player_meeting" ||
    kind === "breaking_news";
  const isTradeAlert = isMediaAlert && isTradePopup(first);

  return (
    <>
      <ShowcasePopupStyles />
      <div className="showcase-popup showcase-popup--v2">
      <div
        className="showcase-popup__backdrop showcase-popup__backdrop--v2"
        aria-hidden
        onClick={dismiss}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") dismiss();
        }}
        role="button"
        tabIndex={-1}
      />
      <div
        className={`showcase-popup__panel showcase-popup__panel--v2 ${isMediaAlert ? "showcase-popup__panel--media" : ""} ${isTradeAlert ? "showcase-popup__panel--trade-wire" : ""} showcase-popup__panel--${theme}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="showcase-popup-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
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
          {kind === "injury" ? (
            <InjuryBody pop={first} onDismiss={dismiss} onAction={handleAction} queueCount={visiblePopups.length} />
          ) : null}
          {kind === "storyline" || kind === "legal_trouble" || kind === "player_meeting" || kind === "breaking_news" ? (
            <StorylineBody
              pop={first}
              onDismiss={dismiss}
              onDismissAllTrades={dismissAllTrades}
              onAction={handleAction}
              queuedTradeCount={tradeQueueIds.length}
              queueCount={visiblePopups.length}
            />
          ) : null}
          {!["wjc_tournament", "showcase_game", "allstar_game", "injury", "storyline", "legal_trouble", "player_meeting", "breaking_news"].includes(
            kind
          ) ? (
            <LeagueNoticeBody pop={first} />
          ) : null}
        </div>
        {!isMediaAlert ? (
          <footer className="showcase-popup__foot">
            {visiblePopups.length > 1 ? (
              <span className="showcase-popup__queue">+{visiblePopups.length - 1} more after this</span>
            ) : null}
            <button type="button" className="showcase-popup__btn" onClick={dismiss}>
              Continue
            </button>
          </footer>
        ) : null}
        {isMediaAlert && visiblePopups.length > 1 ? (
          <div className="media-alert__queue">
            +{visiblePopups.length - 1} more alerts queued
            {tradeQueueIds.length > 1 ? (
              <button type="button" className="media-alert__clear-trades" onClick={dismissAllTrades}>
                Clear all trades
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
    </>
  );
}
