import React, { useMemo, useState } from "react";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

/*
 * Kart-pixel prospect dossier. Reads the dossier model (fog rules live there) plus
 * the backend war_room block (live scouting file, staff, draft range, ranks).
 * Public reads (tier words, class percentiles) unlock once the public file is
 * credible; exact numbers only once YOUR file reaches the reveal line.
 */

const TABS = [
  ["report", "Scouting report"],
  ["production", "Production"],
  ["outlook", "Draft outlook"],
  ["desk", "Scouting desk"],
];

const TIER_COLORS = {
  "world-class": "#ffcf1f",
  elite: "#b28aff",
  great: "#5fa8ff",
  "very good": "#3fd0c9",
  good: "#5fe08a",
  average: "#d9cfae",
  weak: "#ff9d5c",
  bad: "#ff6a5c",
  awful: "#ff5d5d",
};

function n(v) {
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function tierColor(tier) {
  const k = String(tier || "").toLowerCase();
  return TIER_COLORS[k] || "#d9cfae";
}

function money(v) {
  const x = n(v);
  if (x == null) return "—";
  if (Math.abs(x) >= 1e6) return `$${(x / 1e6).toFixed(2)}M`;
  if (Math.abs(x) >= 1e3) return `$${Math.round(x / 1e3)}K`;
  return `$${Math.round(x)}`;
}

function ordinal(k) {
  const v = Number(k);
  if (!Number.isFinite(v)) return "—";
  const s = ["th", "st", "nd", "rd"];
  const m = v % 100;
  return `${v}${s[(m - 20) % 10] || s[m] || s[0]}`;
}

function FileMeter({ yours, pub, revealAt = 72, exactAt = 92 }) {
  const y = Math.max(0, Math.min(100, n(yours) || 0));
  const p = Math.max(0, Math.min(100, n(pub) || 0));
  return (
    <div className="kdd-file" title={`Your file ${Math.round(y)}% · public file ${Math.round(p)}%`}>
      <div className="kdd-file-track">
        <i className="kdd-file-pub" style={{ width: `${p}%` }} />
        <i className="kdd-file-you" style={{ width: `${y}%` }} />
        <b className="kdd-file-mark" style={{ left: `${revealAt}%` }}><span>OVR</span></b>
        <b className="kdd-file-mark" style={{ left: `${exactAt}%` }}><span>CEIL</span></b>
      </div>
      <div className="kdd-file-legend">
        <span className="is-you">Your file <b>{Math.round(y)}%</b></span>
        <span className="is-pub">Public <b>{Math.round(p)}%</b></span>
        <span className="is-next">
          {y < revealAt ? `${Math.round(revealAt - y)}% to exact OVR` : y < exactAt ? `${Math.round(exactAt - y)}% to exact ceiling` : "Full file"}
        </span>
      </div>
    </div>
  );
}

function BigTile({ kicker, value, sub, tone = "", wide = false }) {
  return (
    <div className={`kdd-tile${tone ? ` tone-${tone}` : ""}${wide ? " is-wide" : ""}`}>
      <em>{kicker}</em>
      <b>{value}</b>
      {sub ? <small>{sub}</small> : null}
    </div>
  );
}

function RangeBar({ low, high, exact, value, tone }) {
  const lo = n(low);
  const hi = n(high);
  const v = n(value);
  if (exact && v != null) {
    return (
      <span className="kdd-range">
        <i className="kdd-range-fill" style={{ width: `${v}%`, background: tone }} />
      </span>
    );
  }
  if (lo == null || hi == null) {
    return (
      <span className="kdd-range is-locked">
        <i className="kdd-range-fog" style={{ left: "35%", width: "40%" }} />
      </span>
    );
  }
  return (
    <span className="kdd-range">
      <i className="kdd-range-fill" style={{ width: `${lo}%`, background: tone, opacity: 0.55 }} />
      <i className="kdd-range-fog" style={{ left: `${lo}%`, width: `${Math.max(2, hi - lo)}%` }} />
    </span>
  );
}

function SectionCard({ title, aside, tone = "", children, className = "" }) {
  return (
    <section className={`kdd-card${tone ? ` tone-${tone}` : ""}${className ? ` ${className}` : ""}`}>
      <header className="kdd-card-head">
        <strong>{title}</strong>
        {aside ? <span>{aside}</span> : null}
      </header>
      {children}
    </section>
  );
}

function ReportTab({ model, profile, filePct, pubPct }) {
  const exactOk = filePct >= 72;
  const publicOk = Math.max(filePct, pubPct) >= 30;
  const tools = Array.isArray(profile?.tools) ? profile.tools : [];
  const zones = profile?.zone_map?.zones || {};
  const pctiles = profile?.classIndicators?.percentiles || {};
  const cmp = profile?.player_comparison || null;
  const tags = Array.isArray(profile?.tags) ? profile.tags : [];
  const skills = Array.isArray(model?.skills) ? model.skills : [];
  const char = model?.character || { traits: [], visible: false };
  const headline = profile?.character_read?.headline;
  const strengths = Array.isArray(model?.strengths) ? model.strengths : [];
  const concerns = Array.isArray(model?.concerns) ? model.concerns : [];
  const summary = model?.summary || model?.sentence || "";
  return (
    <div className="kdd-grid kdd-grid--report">
      <SectionCard title="Head scout's read" aside={model?.identity?.archetype || ""} tone="blue" className="is-span2">
        <p className="kdd-bubble">“{typeof summary === "string" && summary ? summary : model?.sentence || "We need eyes on him before we commit to a read."}”</p>
        {tags.length || cmp ? (
          <div className="kdd-chips">
            {tags.map((t) => <span key={t} className="kdd-chip is-tag">{t}</span>)}
            {cmp?.archetype ? <span className="kdd-chip is-cmp"><em>Plays like</em>{cmp.archetype}{cmp.provisional ? " (early read)" : ""}</span> : null}
            {profile?.play_style ? <span className="kdd-chip"><em>Style</em>{profile.play_style}</span> : null}
          </div>
        ) : null}
      </SectionCard>

      <SectionCard
        title="Tool kit"
        aside={exactOk ? "Your grades" : publicOk ? "Public read · tiers only" : "Not enough intel"}
        tone="yellow"
      >
        {publicOk && tools.length ? (
          <div className="kdd-tools">
            {tools.map((t) => (
              <div key={t.label} className="kdd-tool" style={{ "--tool": tierColor(t.tier) }}>
                <span className="kdd-tool-name">{t.label}</span>
                <span className="kdd-tool-tier">{t.tier || "—"}</span>
                <span className="kdd-tool-num">{exactOk && n(t.grade) != null ? Math.round(t.grade) : "?"}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="kdd-empty">Send a scout. The public book on him is too thin to grade his tools.</p>
        )}
      </SectionCard>

      <SectionCard title="Where he wins" aside="Rink zones" tone="green">
        {publicOk && Object.keys(zones).length ? (
          <div className="kdd-rink">
            {["defensive", "transition", "offensive"].map((z) => {
              const row = zones[z] || {};
              const hero = profile?.zone_map?.hero_zone === z;
              return (
                <div key={z} className={`kdd-zone${hero ? " is-hero" : ""}`} style={{ "--zone": tierColor(row.tier) }}>
                  <em>{z}</em>
                  <b>{row.tier || "—"}</b>
                  {exactOk && n(row.value) != null ? <small>{Math.round(row.value)}</small> : null}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="kdd-empty">Zone read unlocks once the public file reaches 30%.</p>
        )}
      </SectionCard>

      <SectionCard title="Chapter grades" aside={exactOk ? "Exact" : "Ranges narrow as your file grows"} tone="purple">
        <div className="kdd-skills">
          {skills.map((s) => {
            const tone = tierColor(s.tier?.label);
            const txt = s.exact && s.value != null ? String(s.value) : s.low != null && s.high != null ? `${s.low}–${s.high}` : "?";
            return (
              <div key={s.key} className="kdd-skill">
                <span className="kdd-skill-name">{s.label}</span>
                <RangeBar low={s.low} high={s.high} exact={s.exact} value={s.value} tone={tone} />
                <span className="kdd-skill-num">{txt}</span>
              </div>
            );
          })}
        </div>
      </SectionCard>

      <SectionCard title="Class percentiles" aside={publicOk ? "vs. the whole draft class" : "Locked"} tone="cyan">
        {publicOk && Object.keys(pctiles).length ? (
          <div className="kdd-pctiles">
            {Object.entries(pctiles).map(([k, v]) => (
              <div key={k} className="kdd-pctile">
                <span>{k.replace(/_/g, " ")}</span>
                <i><s style={{ width: `${Math.max(3, Math.min(100, Number(v) || 0))}%` }} /></i>
                <b>{Math.round(Number(v) || 0)}</b>
              </div>
            ))}
          </div>
        ) : (
          <p className="kdd-empty">Class comparisons need a credible file.</p>
        )}
      </SectionCard>

      <SectionCard title="Strengths & concerns" tone="green">
        <ul className="kdd-list">
          {strengths.slice(0, 5).map((s, i) => <li key={`s${i}`} className="is-good"><b>+</b>{s.label}{s.detail ? <em> — {s.detail}</em> : null}</li>)}
          {concerns.slice(0, 4).map((s, i) => <li key={`c${i}`} className="is-bad"><b>−</b>{s.label}{s.detail ? <em> — {s.detail}</em> : null}</li>)}
          {!strengths.length && !concerns.length ? <li className="kdd-empty">Nothing confirmed yet. Get a scout on him.</li> : null}
        </ul>
      </SectionCard>

      <SectionCard title="Character" aside={headline || ""} tone="red">
        {char.visible && char.traits.length ? (
          <div className="kdd-traits">
            {char.traits.map((t) => (
              <div key={t.key} className="kdd-trait" style={{ "--trait": tierColor(t.tier?.label) }}>
                <span>{t.label}</span>
                <b>{t.tier?.label || "—"}</b>
              </div>
            ))}
          </div>
        ) : (
          <p className="kdd-empty">Character reads need 36% of your own file, or a red flag from the league.</p>
        )}
      </SectionCard>
    </div>
  );
}

function ProductionTab({ model, profile, war }) {
  const cells = Array.isArray(model?.analytics?.cells) ? model.analytics.cells : [];
  const proj = war?.projected_season || {};
  const ranks = war?.ranks || {};
  const ctx = war?.context || {};
  const hist = Array.isArray(profile?.scouting_history) ? profile.scouting_history : [];
  const nhle = profile?.classIndicators?.nhle;
  return (
    <div className="kdd-grid kdd-grid--prod">
      <SectionCard title="Season so far" aside={model?.analytics?.sampleThin ? `Thin sample · ${model?.analytics?.gp || 0} GP` : `${model?.analytics?.gp || 0} GP`} tone="yellow" className="is-span2">
        <div className="kdd-statrow">
          {cells.length ? cells.map((c) => (
            <div key={c.label} className="kdd-stat">
              <em>{c.label}</em>
              <b>{c.value}</b>
            </div>
          )) : <p className="kdd-empty">No games logged yet.</p>}
        </div>
      </SectionCard>

      <SectionCard title="On pace for" aside="Full-season projection" tone="green">
        <div className="kdd-statrow is-big">
          <div className="kdd-stat"><em>GP</em><b>{proj.gp ?? "—"}</b></div>
          <div className="kdd-stat"><em>G</em><b>{proj.goals ?? "—"}</b></div>
          <div className="kdd-stat"><em>A</em><b>{proj.assists ?? "—"}</b></div>
          <div className="kdd-stat is-hot"><em>PTS</em><b>{proj.points ?? "—"}</b></div>
          {nhle != null ? <div className="kdd-stat"><em>NHLe</em><b>{Number(nhle).toFixed(1)}</b></div> : null}
        </div>
      </SectionCard>

      <SectionCard title="How he ranks" tone="blue">
        <div className="kdd-ranks">
          {ranks.league_points ? <div className="kdd-rankcell"><b>{ordinal(ranks.league_points.rank)}</b><span>in {ranks.league} points<br /><small>of {ranks.league_points.of} draft-eligible</small></span></div> : null}
          {ranks.league_ppg ? <div className="kdd-rankcell"><b>{ordinal(ranks.league_ppg.rank)}</b><span>in {ranks.league} PPG<br /><small>min. 3 GP</small></span></div> : null}
          {ranks.position_board ? <div className="kdd-rankcell"><b>{ordinal(ranks.position_board.rank)}</b><span>{ranks.position_board.group === "G" ? "goalie" : ranks.position_board.group === "D" ? "defenseman" : "forward"} on the board<br /><small>of {ranks.position_board.of}</small></span></div> : null}
          {ranks.class_war ? <div className="kdd-rankcell"><b>{ordinal(ranks.class_war.rank)}</b><span>in class WAR</span></div> : null}
          {ranks.class_nhle ? <div className="kdd-rankcell"><b>{ordinal(ranks.class_nhle.rank)}</b><span>in class NHLe</span></div> : null}
          {!ranks.league_points && !ranks.position_board ? <p className="kdd-empty">Rankings appear once games are logged.</p> : null}
        </div>
      </SectionCard>

      <SectionCard title="League context" tone="purple">
        <div className="kdd-kv">
          {[
            ["League", profile?.league_display || profile?.league],
            ["Competition", ctx.league_difficulty],
            ["Scoring", ctx.scoring_environment],
            ["Production", ctx.production_context],
            ["Translation risk", ctx.translation_risk],
            ["Hometown", ctx.hometown],
          ].filter(([, v]) => v).map(([k, v]) => (
            <div key={k}><em>{k}</em><b>{String(v)}</b></div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Report wire" aside={`${hist.length} filed`} tone="cyan">
        <ul className="kdd-wire">
          {hist.length ? hist.map((h, i) => (
            <li key={i} className={`tone-${h.tone || "blue"}`}>
              <span className="kdd-wire-grade">{h.grade_label || "—"}</span>
              <span className="kdd-wire-body"><b>“{h.quote}”</b><em>{h.scout}{h.meta ? ` · ${h.meta}` : ""}</em></span>
            </li>
          )) : <li className="kdd-empty">No reports yet.</li>}
        </ul>
      </SectionCard>
    </div>
  );
}

function RankPath({ path }) {
  const pts = (path || []).filter((p) => n(p.rank) != null);
  if (pts.length < 2) {
    return <p className="kdd-empty">Not enough board history yet. Check back after a few weekly updates.</p>;
  }
  const max = Math.max(...pts.map((p) => p.rank), 10);
  const W = 100;
  const H = 40;
  const xy = pts.map((p, i) => [8 + (i * (W - 16)) / (pts.length - 1), 6 + ((p.rank - 1) / Math.max(1, max - 1)) * (H - 12)]);
  return (
    <div className="kdd-path">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <polyline points={xy.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="#15112b" strokeWidth="2.2" />
        {xy.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.6" fill="#ffcf1f" stroke="#15112b" strokeWidth="1.2" />)}
      </svg>
      <div className="kdd-path-labels">
        {pts.map((p) => <span key={p.label}><b>#{p.rank}</b>{p.label}</span>)}
      </div>
    </div>
  );
}

function OutlookTab({ model, profile, war }) {
  const dr = war?.draft_range || {};
  const outcomes = Array.isArray(model?.role?.outcomes) ? model.role.outcomes : [];
  const total = outcomes.reduce((a, o) => a + (Number(o.pct) || 0), 0) || 1;
  const palette = ["#ffcf1f", "#b28aff", "#5fa8ff", "#5fe08a", "#ff9d5c", "#ff6a5c"];
  const gem = profile?.gem;
  return (
    <div className="kdd-grid kdd-grid--outlook">
      <SectionCard
        title="Where he goes"
        aside={dr.low ? `Projected picks ${dr.low}–${dr.high}` : "—"}
        tone="yellow"
        className="is-span2"
      >
        <div className="kdd-range-teams">
          {(dr.teams || []).map((t) => {
            const logo = resolveFranchiseTeamLogo({ team_id: t.team_id, id: t.team_id, abbr: t.abbr, name: t.name }, t.name);
            return (
              <div key={t.pick} className={`kdd-pickslot${t.is_user ? " is-user" : ""}`}>
                <em>#{t.pick}</em>
                {logo ? <img src={logo} alt="" /> : <span className="kdd-pick-abbr">{t.abbr}</span>}
                <b>{t.abbr}</b>
              </div>
            );
          })}
        </div>
        <p className={`kdd-callout${dr.user_in_range ? " is-good" : ""}`}>
          {dr.user_in_range
            ? `Your projected pick (#${dr.user_pick}) lands right in his range. He could be yours without a trade.`
            : dr.user_pick && dr.low
              ? dr.user_pick > dr.high
                ? `Your pick projects #${dr.user_pick}. He should be gone by then; you'd need to trade up about ${dr.user_pick - dr.high} spots.`
                : `Your pick projects #${dr.user_pick}, ahead of his range. You could trade down and still get him.`
              : "Draft order firms up as the standings settle."}
        </p>
      </SectionCard>

      <SectionCard title="Board history" tone="blue">
        <RankPath path={war?.rank_path} />
      </SectionCard>

      <SectionCard title="Career outcomes" aside={model?.role?.mostLikely ? `Most likely: ${model.role.mostLikely}` : ""} tone="purple">
        {outcomes.length ? (
          <>
            <div className="kdd-stack">
              {outcomes.map((o, i) => (
                <i key={o.key} style={{ width: `${((Number(o.pct) || 0) / total) * 100}%`, background: palette[i % palette.length] }} title={`${o.label} ${Math.round(o.pct)}%`} />
              ))}
            </div>
            <div className="kdd-stack-legend">
              {outcomes.map((o, i) => (
                <span key={o.key}><i style={{ background: palette[i % palette.length] }} />{o.label} <b>{Math.round(o.pct)}%</b></span>
              ))}
            </div>
          </>
        ) : <p className="kdd-empty">Outcome model needs more file.</p>}
      </SectionCard>

      <SectionCard title="Projection" tone="green">
        <div className="kdd-kv">
          <div><em>Projected role</em><b>{model?.role?.projected || "—"}</b></div>
          <div><em>Readiness</em><b>{model?.role?.readiness || "—"}</b></div>
          <div><em>ETA</em><b>{model?.role?.eta || "—"}</b></div>
          <div><em>NHL odds</em><b>{model?.display?.nhlOdds || "—"}</b></div>
          <div><em>Risk</em><b>{model?.risk || "—"}</b></div>
          {gem?.label ? <div><em>Value read</em><b>{gem.label}{gem.reason ? ` · ${gem.reason}` : ""}</b></div> : null}
        </div>
      </SectionCard>
    </div>
  );
}

function DeskTab({ war, busy, onAssign, onAction, meta }) {
  const file = war?.file || {};
  const scouts = Array.isArray(war?.scouts) ? war.scouts : [];
  const budget = war?.budget || {};
  const prices = budget.prices || {};
  return (
    <div className="kdd-grid kdd-grid--desk">
      <SectionCard title="Your scouting staff" aside="Assign one to keep the file growing daily" tone="blue" className="is-span2">
        <div className="kdd-scouts">
          {scouts.map((s) => {
            const on = file.assigned_scout_id === s.id;
            return (
              <div key={s.id} className={`kdd-scout${on ? " is-on" : ""}${s.recommended ? " is-rec" : ""}`}>
                <div className="kdd-scout-face" aria-hidden="true">{s.name.split(" ").map((w) => w[0]).join("")}</div>
                <div className="kdd-scout-main">
                  <b>{s.name}</b>
                  <em>{s.region} · {s.tier}</em>
                  <span>{s.load} on file · ~{s.days_to_72}d to exact OVR</span>
                </div>
                {s.recommended && !on ? <span className="kdd-rec">Best fit</span> : null}
                <button type="button" className={`kdd-btn${on ? " is-on" : ""}`} disabled={busy} onClick={() => onAssign(on ? "" : s.id)}>
                  {on ? "Recall" : "Assign"}
                </button>
              </div>
            );
          })}
        </div>
      </SectionCard>

      <SectionCard title="File status" tone="yellow">
        <div className="kdd-kv">
          <div><em>Assigned</em><b>{file.assigned_scout_name || "Nobody"}</b></div>
          <div><em>Days on him</em><b>{file.days_assigned != null ? file.days_assigned : "—"}</b></div>
          <div><em>Shortlist</em><b>{meta?.watchlist || file.watchlist ? "Yes" : "No"}</b></div>
          <div><em>Target</em><b>{meta?.target || file.target ? "Yes" : "No"}</b></div>
          <div><em>Film study</em><b>{file.film_study_done ? "Done" : "Not yet"}</b></div>
          <div><em>Medical</em><b>{file.medical_status || "—"}</b></div>
        </div>
      </SectionCard>

      <SectionCard title="Budget & trips" aside={`${money(budget.left)} left of ${money(budget.total)}`} tone="green">
        <div className="kdd-trips">
          <button type="button" className="kdd-trip" disabled={busy} onClick={() => onAction("player_focus")}>
            <b>Scout visit</b><span>+file now · {money(prices.player_focus)}</span>
          </button>
          <button type="button" className="kdd-trip" disabled={busy || file.film_study_done} onClick={() => onAction("film_study")}>
            <b>Film study</b><span>{file.film_study_done ? "Done" : `Needs a scout or shortlist · ${money(prices.film_study)}`}</span>
          </button>
        </div>
        <i className="kdd-budgetbar"><s style={{ width: `${budget.total ? Math.min(100, (budget.used / budget.total) * 100) : 0}%` }} /></i>
      </SectionCard>

      <SectionCard title="Scout notes" aside={file.red_flags?.length ? `${file.red_flags.length} red flag(s)` : ""} tone="red">
        <ul className="kdd-list">
          {(file.red_flags || []).map((r, i) => <li key={`r${i}`} className="is-bad"><b>!</b>{r}</li>)}
          {(file.notes || []).slice().reverse().map((t, i) => <li key={`n${i}`}><b>•</b>{t}</li>)}
          {!file.notes?.length && !file.red_flags?.length ? <li className="kdd-empty">No notes yet.</li> : null}
        </ul>
      </SectionCard>
    </div>
  );
}

function posGroupOf(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "G") return "G";
  if (p === "D" || p === "LD" || p === "RD") return "D";
  return "F";
}

/** Client-side war room: the same read-outs computed from board data, used when the
 * backend block is missing (older API) and merged under it when present. */
function localWarRoom(player, allProspects, filePct, pubPct, staff, budgetFallback) {
  const p = player || {};
  const all = Array.isArray(allProspects) ? allProspects : [];
  const rankOf = (x) => Number(x?.boardRank || x?.rank || 0) || null;
  const rank = rankOf(p);
  const league = p.leagueDisplay || p.league || "";
  const group = posGroupOf(p.position);
  const rankIn = (pool, key, minGp = 0) => {
    const rows = pool.filter((x) => Number(x?.[key]) > 0 && (!minGp || Number(x?.gp) >= minGp));
    rows.sort((a, b) => Number(b[key]) - Number(a[key]));
    const i = rows.findIndex((x) => x.id === p.id);
    return i >= 0 ? { rank: i + 1, of: rows.length } : null;
  };
  const sameLeague = all.filter((x) => (x.leagueDisplay || x.league || "") === league);
  const samePos = all.filter((x) => posGroupOf(x.position) === group).sort((a, b) => (rankOf(a) || 9999) - (rankOf(b) || 9999));
  const posIdx = samePos.findIndex((x) => x.id === p.id);
  const prev = Number(p?.draftStock?.previousRank || p?.previousRank || 0) || null;
  return {
    file: { yours: filePct, public: pubPct, reveal_at: 72, exact_at: 92, watchlist: Boolean(p.watchlist), target: Boolean(p.target) },
    scouts: staff || [],
    budget: budgetFallback || {},
    draft_range: rank ? { rank, low: Math.max(1, rank - 3), high: rank + 5, teams: [] } : {},
    ranks: {
      league,
      league_points: group !== "G" ? rankIn(sameLeague, "points") : null,
      league_ppg: group !== "G" ? rankIn(sameLeague, "ppg", 3) : null,
      position_board: posIdx >= 0 ? { rank: posIdx + 1, of: samePos.length, group } : null,
    },
    rank_path: [
      { label: "Last week", rank: prev },
      { label: "Now", rank },
    ],
    projected_season: {
      gp: p.projectedGp != null ? Math.round(p.projectedGp) : null,
      goals: p.projectedGoals != null ? Math.round(p.projectedGoals) : null,
      assists: p.projectedAssists != null ? Math.round(p.projectedAssists) : null,
      points: p.projectedPoints != null ? Math.round(p.projectedPoints) : null,
    },
    context: {
      production_context: p.productionContext,
      league_difficulty: p.leagueDifficulty,
      scoring_environment: p.scoringEnvironment,
      translation_risk: p.translationRisk,
      hometown: p.hometown || p.birthCity,
    },
  };
}

function mergeWar(local, remote) {
  if (!remote) return local;
  const out = { ...local };
  Object.entries(remote).forEach(([k, v]) => {
    if (v && typeof v === "object" && !Array.isArray(v) && local[k] && typeof local[k] === "object" && !Array.isArray(local[k])) {
      const merged = { ...local[k] };
      Object.entries(v).forEach(([kk, vv]) => { if (vv != null && !(Array.isArray(vv) && !vv.length && Array.isArray(merged[kk]) && merged[kk].length)) merged[kk] = vv; });
      out[k] = merged;
    } else if (v != null && !(Array.isArray(v) && !v.length && Array.isArray(local[k]) && local[k].length)) {
      out[k] = v;
    }
  });
  return out;
}

export default function KartDossier({
  model,
  profile,
  portrait,
  meta,
  busy = false,
  note = "",
  onClose,
  onToggleWatchlist,
  onToggleTarget,
  onToggleDND,
  onAssign,
  onAction,
  enriching = false,
  player = null,
  allProspects = [],
  staff = [],
  budgetFallback = null,
}) {
  const [tab, setTab] = useState("report");
  const remoteWar = profile?.war_room || null;
  const filePct = n(remoteWar?.file?.yours) ?? n(model?.file?.pct) ?? 0;
  const pubPct = n(remoteWar?.file?.public) ?? n(model?.file?.consensusPct) ?? 0;
  const war = useMemo(
    () => mergeWar(localWarRoom(player, allProspects, filePct, pubPct, staff, budgetFallback), remoteWar),
    [player, allProspects, filePct, pubPct, staff, budgetFallback, remoteWar],
  );
  const id = model?.identity || {};
  const rank = model?.rank || {};
  const peakTier = useMemo(() => {
    const hi = n(model?.peak?.high ?? model?.peak?.value);
    if (hi == null || model?.peak?.hidden) return "fog";
    if (hi >= 94) return "gold";
    if (hi >= 90) return "purple";
    if (hi >= 86) return "blue";
    return "green";
  }, [model]);
  const delta = n(rank?.delta);
  const watch = Boolean(meta?.watchlist || war?.file?.watchlist);
  const target = Boolean(meta?.target || war?.file?.target);
  const dnd = Boolean(meta?.doNotDraft || war?.file?.do_not_draft);
  const recId = war?.file?.recommended_scout_id
    || (war?.scouts || []).find((sc) => sc.recommended)?.id
    || (war?.scouts || []).slice().sort((x, y) => (Number(y.quality) || 0) - (Number(x.quality) || 0))[0]?.id
    || "auto";

  return (
    <div className="kdd-shell">
      <header className="kdd-hero">
        <div className="kdd-portrait">{portrait}</div>
        <div className="kdd-ident">
          <span className="kdd-kicker">{id.pos} · Draft {profile?.draft_year || ""} prospect</span>
          <h2>{id.name}</h2>
          <div className="kdd-chips">
            <span className="kdd-chip is-pos">{id.pos}</span>
            {id.age != null ? <span className="kdd-chip"><em>Age</em>{id.age}</span> : null}
            {id.shoots ? <span className="kdd-chip"><em>Shoots</em>{id.shoots}</span> : null}
            {id.height ? <span className="kdd-chip"><em>Size</em>{id.height}{id.weight ? ` · ${id.weight}` : ""}</span> : null}
            {id.nation ? <span className="kdd-chip"><em>Nation</em>{id.nation}</span> : null}
            {id.overage ? <span className="kdd-chip is-warn">Overager</span> : null}
          </div>
          <span className="kdd-club">{[id.club, id.league].filter(Boolean).join(" · ")}{id.seasonLine ? ` · ${id.seasonLine}` : ""}</span>
        </div>
        <div className="kdd-tiles">
          <BigTile
            kicker="Board rank"
            value={`#${rank.board || profile?.rank || "—"}`}
            sub={delta ? `${delta > 0 ? "▲" : "▼"} ${Math.abs(delta)} this week` : `Public #${profile?.publicRank || profile?.consensusRank || "—"}`}
            tone="yellow"
          />
          <BigTile kicker="Now" value={model?.display?.ovr || "?"} sub={filePct >= 72 ? "Exact" : "Range"} tone="blue" />
          <BigTile kicker="Ceiling" value={model?.peak?.hidden ? "?" : model?.display?.peak || "?"} sub={model?.peak?.hidden ? "Unreported" : filePct >= 92 ? "Exact" : "Range"} tone={peakTier} />
          <BigTile kicker="NHL odds" value={model?.display?.nhlOdds || "—"} sub={`Risk ${model?.risk || "—"}`} tone={model?.risk === "High" ? "red" : model?.risk === "Low" ? "green" : "purple"} />
        </div>
        <button type="button" className="kdd-close" onClick={onClose} aria-label="Close dossier">×</button>
        <FileMeter yours={filePct} pub={pubPct} revealAt={war?.file?.reveal_at || 72} exactAt={war?.file?.exact_at || 92} />
      </header>

      <nav className="kdd-tabs" role="tablist">
        {TABS.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={tab === k} className={`kdd-tab${tab === k ? " is-on" : ""}`} onClick={() => setTab(k)}>
            {label}
          </button>
        ))}
        {enriching ? <span className="kdd-sync">Syncing file…</span> : null}
      </nav>

      <div className="kdd-body">
        {tab === "report" ? <ReportTab model={model} profile={profile} filePct={filePct} pubPct={pubPct} /> : null}
        {tab === "production" ? <ProductionTab model={model} profile={profile} war={war} /> : null}
        {tab === "outlook" ? <OutlookTab model={model} profile={profile} war={war} /> : null}
        {tab === "desk" ? <DeskTab war={war} busy={busy} onAssign={onAssign} onAction={onAction} meta={meta} /> : null}
      </div>

      <footer className="kdd-dock">
        <button type="button" className={`kdd-dock-btn${watch ? " is-on" : ""}`} onClick={onToggleWatchlist}>
          <b>{watch ? "★ Shortlisted" : "☆ Shortlist"}</b><span>{watch ? "On your board" : "Pin to your board"}</span>
        </button>
        <button type="button" className={`kdd-dock-btn${target ? " is-on" : ""}`} onClick={onToggleTarget}>
          <b>{target ? "Target set" : "Mark target"}</b><span>Priority on draft day</span>
        </button>
        <button type="button" className={`kdd-dock-btn is-red${dnd ? " is-on" : ""}`} onClick={onToggleDND}>
          <b>{dnd ? "Do not draft" : "Flag DND"}</b><span>Hide from auto-picks</span>
        </button>
        <span className="kdd-dock-note" aria-live="polite">{note}</span>
        <button type="button" className="kdd-dock-btn" disabled={busy} onClick={() => onAction("player_focus")}>
          <b>{busy ? "Scouting…" : "Scout visit"}</b><span>{money(war?.budget?.prices?.player_focus)}</span>
        </button>
        <button type="button" className="kdd-dock-btn is-go" disabled={busy} onClick={() => onAssign(war?.file?.assigned_scout_id ? war.file.assigned_scout_id : recId)}>
          <b>{war?.file?.assigned_scout_name ? `On file: ${war.file.assigned_scout_name.split(" ")[0]}` : "Assign best scout"}</b>
          <span>{war?.file?.assigned_scout_name ? "File grows daily" : "Daily file growth"}</span>
        </button>
      </footer>
    </div>
  );
}
