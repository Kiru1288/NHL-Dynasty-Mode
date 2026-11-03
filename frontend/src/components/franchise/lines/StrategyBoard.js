import React, { useMemo, useState } from "react";

/* Strategy Board deck — sits above the line editor.
   Five tabs, FHM-style: Identity · Tactics · Deployment · Performance · Adjustments.
   Everything is read from data the screen already has (lines, roster rows with the
   backend chemistry profile, the chemistry report, the coach archetype and record).
   Controls call the editor's own handlers (onDeploy / onShareSet / onCoach), so the
   lineup save path is unchanged. */

const arr = (v) => (Array.isArray(v) ? v : []);
const num = (v, d = null) => (Number.isFinite(Number(v)) ? Number(v) : d);
const cap = (v) => (v ? String(v).charAt(0).toUpperCase() + String(v).slice(1) : "");
const avg = (xs) => { const ok = xs.filter((x) => Number.isFinite(x)); return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null; };
const FWD_SHARE_DEFAULT = [33, 28, 22, 17];
const DEF_SHARE_DEFAULT = [45, 33, 22];

/* Mirrors SimEngine systems/chemistry.py _COACH_SYSTEM_FIT (favoured, penalised). */
const COACH_SYSTEMS = {
  balanced: { label: "Balanced", favored: ["two_way", "playmaker", "sniper", "shutdown", "puck_mover"], penalized: [], pitch: "Plays whoever earns it. No style is punished." },
  defense_first: { label: "Defence first", favored: ["shutdown", "defensive_defenseman", "grinder", "two_way"], penalized: ["offensive_defenseman"], pitch: "Structure, low risk, a conservative kill. Rewards checkers and stay-at-home D." },
  offensive: { label: "Attack", favored: ["sniper", "playmaker", "offensive_defenseman", "power_forward"], penalized: ["grinder"], pitch: "High risk, D join the rush. Rewards skill; grinders lose minutes." },
  young_fast: { label: "Development", favored: ["sniper", "playmaker", "puck_mover", "two_way"], penalized: ["grinder"], pitch: "Trusts youth, pushes pace and growth." },
  players_coach: { label: "Players' coach", favored: ["two_way", "playmaker", "sniper", "shutdown", "puck_mover"], penalized: [], pitch: "Merit-based minutes and a warm room. Morale carries the system." },
};
function coachSystemKey(raw) {
  const k = String(raw || "").toLowerCase().replace(/[\s-]+/g, "_");
  if (!k) return "balanced";
  if (/develop|teacher|young/.test(k)) return "young_fast";
  if (/defen|structure|shutdown|trap/.test(k)) return "defense_first";
  if (/aggress|attack|offen|run/.test(k)) return "offensive";
  if (/player|culture|leader/.test(k)) return "players_coach";
  return "balanced";
}
/* Editor role keys -> backend playstyle keys used by the coach table. */
function playstyleKey(p) {
  const raw = String(p?.chemistry_profile?.playstyle || p?.playstyle || p?.archetype || p?.play_role_label || "").toLowerCase().replace(/[\s-]+/g, "_");
  if (!raw) return "";
  if (raw.includes("offensive_d") || raw === "od") return "offensive_defenseman";
  if (raw.includes("defensive_d") || raw === "dd" || raw.includes("shutdown")) return /^(ld|rd|d)$/i.test(p?.position || "") ? "defensive_defenseman" : "shutdown";
  for (const k of ["playmaker", "sniper", "power_forward", "grinder", "enforcer", "puck_mover", "two_way"]) if (raw.includes(k)) return k;
  if (raw.includes("power")) return "power_forward";
  return raw;
}
const STYLE_LABEL = {
  playmaker: "Playmaker", sniper: "Sniper", power_forward: "Power forward", grinder: "Grinder", enforcer: "Enforcer",
  two_way: "Two-way", shutdown: "Shutdown", defensive_defenseman: "Defensive D", offensive_defenseman: "Offensive D", puck_mover: "Puck mover",
};
const SKILL_STYLES = new Set(["playmaker", "sniper", "offensive_defenseman", "puck_mover"]);
const GRIT_STYLES = new Set(["grinder", "enforcer", "power_forward", "shutdown", "defensive_defenseman"]);

function shareOf(line, i, group) {
  const n = Number(line?.toi_share);
  return Number.isFinite(n) ? n : ((group === "defense" ? DEF_SHARE_DEFAULT : FWD_SHARE_DEFAULT)[i] || 10);
}
function zoneOf(line, i) { return line?.zone || (i < 2 ? "offensive" : "defensive"); }
function usageOf(line, i) { return line?.usage || (i < 2 ? "attack" : "checking"); }
function pairRoleOf(line, i) { return line?.pair_role || (i === 0 ? "shutdown" : i === 1 ? "neutral" : "sheltered"); }

function unitRole(group, line, i) {
  if (group === "defense") {
    return {
      shutdown: ["Shutdown pair", "Draws the other team's top line. Needs reads, reach and buy-in more than points."],
      neutral: ["Second pair", "Fair matchups. Expected to break even and move the puck cleanly."],
      offensive: ["Offensive pair", "Gets offensive-zone draws and activates. Wants a puck-mover and a shot from the point."],
      sheltered: ["Sheltered pair", "Easy minutes against depth lines. Good place to grow a young D or hide a weak one."],
    }[pairRoleOf(line, i)] || ["Pair", ""];
  }
  const z = zoneOf(line, i); const u = usageOf(line, i);
  if (z === "offensive" && u === "attack") return [i === 0 ? "Top line" : "Scoring line", "Offensive-zone starts and licence to attack. Lives on finish and playmaking."];
  if (z === "defensive" && u === "checking") return [i >= 3 ? "Energy line" : "Checking line", "Defensive-zone draws, chip-and-chase, hard matchups. Needs grit and buy-in."];
  if (z === "defensive") return ["Transition line", "Starts in its own end but pushes the pace up ice. Needs skaters who can exit and carry."];
  return ["Two-way line", "Offensive-zone starts but plays it safe. A good home for responsible veterans."];
}

/* Why a player fits (or doesn't) his slot and the coach: same terms as the backend formula. */
function playerFit(p, group, line, i, system) {
  const prof = p?.chemistry_profile || {};
  const style = playstyleKey(p);
  const reasons = [];
  let score = num(prof.coach_system_fit);
  const buy = num(prof.defensive_buy_in), coachab = num(prof.coachability), adapt = num(prof.adaptability);
  if (score == null) {
    const base = 0.4 * (buy ?? 50) + 0.32 * (coachab ?? 50) + 0.28 * (adapt ?? 50);
    score = base + (system.favored.includes(style) ? 10 : 0) - (system.penalized.includes(style) ? 7 : 0);
  }
  if (system.favored.includes(style)) reasons.push({ tone: "good", text: `${STYLE_LABEL[style] || cap(style)} suits a ${system.label.toLowerCase()} coach` });
  if (system.penalized.includes(style)) reasons.push({ tone: "bad", text: `${STYLE_LABEL[style] || cap(style)} clashes with a ${system.label.toLowerCase()} coach` });
  if (buy != null && buy >= 68) reasons.push({ tone: "good", text: `buys in defensively (${buy})` });
  if (buy != null && buy <= 38) reasons.push({ tone: "bad", text: `cheats on defence (${buy} buy-in)` });
  if (coachab != null && coachab >= 70) reasons.push({ tone: "good", text: "very coachable" });
  if (coachab != null && coachab <= 35) reasons.push({ tone: "bad", text: "hard to coach" });
  // Deployment fit: does his style match what this unit is asked to do?
  let roleFit = 0;
  if (group === "forwards") {
    const attack = usageOf(line, i) === "attack";
    if (attack && SKILL_STYLES.has(style)) { roleFit = 1; reasons.push({ tone: "good", text: "skill on an attacking line" }); }
    if (attack && (style === "grinder" || style === "enforcer")) { roleFit = -1; reasons.push({ tone: "bad", text: "grinder asked to create offence" }); }
    if (!attack && GRIT_STYLES.has(style)) { roleFit = 1; reasons.push({ tone: "good", text: "built for a checking role" }); }
    if (!attack && style === "sniper") { roleFit = -1; reasons.push({ tone: "bad", text: "sniper buried in a checking role" }); }
  } else if (group === "defense") {
    const r = pairRoleOf(line, i);
    if (r === "shutdown" && (style === "defensive_defenseman" || style === "shutdown")) { roleFit = 1; reasons.push({ tone: "good", text: "true shutdown D" }); }
    if (r === "shutdown" && style === "offensive_defenseman") { roleFit = -1; reasons.push({ tone: "bad", text: "offensive D on the shutdown pair" }); }
    if (r === "offensive" && (style === "offensive_defenseman" || style === "puck_mover")) { roleFit = 1; reasons.push({ tone: "good", text: "moves the puck from the point" }); }
  }
  const total = Math.max(0, Math.min(100, Math.round(score + roleFit * 6)));
  return { score: total, style, reasons: reasons.slice(0, 3) };
}
function fitTone(score) { return score >= 70 ? "good" : score >= 52 ? "mid" : "bad"; }

function findStandingRow(franchiseState) {
  const tid = String(franchiseState?.user_team_id ?? franchiseState?.team?.id ?? "");
  const pools = [franchiseState?.standings, franchiseState?.standings?.rows, franchiseState?.league_standings, franchiseState?.standings_rows];
  for (const pool of pools) {
    const rows = Array.isArray(pool) ? pool : pool && typeof pool === "object" ? Object.values(pool).flat() : [];
    const hit = rows.find((r) => r && String(r.team_id ?? r.id ?? "") === tid);
    if (hit) return hit;
  }
  return null;
}

function Meter({ left, right, value, note }) {
  const v = Math.max(0, Math.min(100, Math.round(value)));
  return <div className="sbd-meter">
    <div className="sbd-meter__ends"><span>{left}</span><span>{right}</span></div>
    <div className="sbd-meter__track"><i style={{ left: `${v}%` }} /></div>
    {note ? <p className="sbd-meter__note">{note}</p> : null}
  </div>;
}
function Choice({ value, options, onPick }) {
  return <div className="ccx-choice" role="radiogroup">
    {options.map(([id, label, hint]) => (
      <button key={id} type="button" role="radio" aria-checked={value === id} className={`ccx-choice__opt ${value === id ? "is-on" : ""}`} onClick={() => onPick(id)} title={hint || label}>
        <b>{label}</b>{hint ? <small>{hint}</small> : null}
      </button>
    ))}
  </div>;
}

const TABS = [["identity", "Identity"], ["tactics", "Tactics"], ["deploy", "Deployment"], ["perf", "Performance"], ["adjust", "Adjustments"]];

export default function StrategyBoard({ franchiseState, lineState, playerMap, chemistryByUnit, chemReport, onDeploy, onShareSet, onCoach, shareTotals }) {
  const [tab, setTab] = useState(() => { try { return window.localStorage.getItem("sbd.tab") || "identity"; } catch (e) { return "identity"; } });
  const [open, setOpen] = useState(() => { try { return window.localStorage.getItem("sbd.open") !== "0"; } catch (e) { return true; } });
  const pickTab = (id) => { setTab(id); setOpen(true); try { window.localStorage.setItem("sbd.tab", id); window.localStorage.setItem("sbd.open", "1"); } catch (e) { /* storage off */ } };
  const toggle = () => setOpen((v) => { try { window.localStorage.setItem("sbd.open", v ? "0" : "1"); } catch (e) { /* storage off */ } return !v; });

  const coachName = franchiseState?.team?.coach || franchiseState?.head_coach_name || "Your coach";
  const archetype = franchiseState?.team?.coach_archetype || franchiseState?.coach_archetype || "balanced";
  const sysKey = coachSystemKey(archetype);
  const system = COACH_SYSTEMS[sysKey];
  const fwd = arr(lineState?.forwards);
  const def = arr(lineState?.defense);

  const units = useMemo(() => {
    const rows = [];
    for (const [group, list] of [["forwards", fwd], ["defense", def]]) {
      list.forEach((line, i) => {
        const players = Object.entries(line?.slots || {}).map(([slot, id]) => ({ slot, p: playerMap?.[String(id || "")] })).filter((x) => x.p);
        const share = shareOf(line, i, group);
        const chem = chemistryByUnit?.[`${group}:${line.id}`] || null;
        const fits = players.map(({ slot, p }) => ({ slot, p, fit: playerFit(p, group, line, i, system) }));
        const [role, roleWhy] = unitRole(group, line, i);
        rows.push({ group, line, i, share, chem, fits, role, roleWhy, minutes: Math.round((share / 100) * 60), fitAvg: avg(fits.map((f) => f.fit.score)) });
      });
    }
    return rows;
  }, [fwd, def, playerMap, chemistryByUnit, system]);

  /* Identity: TOI-weighted style mix, risk, age and depth. */
  const identity = useMemo(() => {
    let w = 0, skill = 0, grit = 0, attackW = 0, ageSum = 0, ageW = 0;
    const styleMinutes = {};
    for (const u of units) {
      const wt = u.share / (u.group === "defense" ? 100 : 100);
      if (u.group === "forwards" && usageOf(u.line, u.i) === "attack") attackW += u.share;
      if (u.group === "defense" && ["offensive", "neutral"].includes(pairRoleOf(u.line, u.i))) attackW += u.share * (pairRoleOf(u.line, u.i) === "offensive" ? 1 : 0.5);
      for (const f of u.fits) {
        w += wt;
        if (SKILL_STYLES.has(f.fit.style)) skill += wt;
        if (GRIT_STYLES.has(f.fit.style)) grit += wt;
        styleMinutes[f.fit.style] = (styleMinutes[f.fit.style] || 0) + wt;
        const age = num(f.p.age);
        if (age) { ageSum += age * wt; ageW += wt; }
      }
    }
    const skillPct = w ? (skill / (skill + grit || 1)) * 100 : 50;
    const riskPct = attackW / 2; // forwards attack share (0-100) + D activation (0-100) averaged
    const topShare = fwd.length ? shareOf(fwd[0], 0, "forwards") + (fwd[1] ? shareOf(fwd[1], 1, "forwards") : 0) : 61;
    const depthPct = Math.max(0, Math.min(100, ((topShare - 50) / 25) * 100));
    const avgAge = ageW ? ageSum / ageW : null;
    const topStyle = Object.entries(styleMinutes).filter(([k]) => k).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
    let label = "Balanced";
    let blurb = "No single trait dominates. You can win any type of game but rarely overwhelm anyone.";
    if (riskPct >= 62 && skillPct >= 58) { label = "Run and gun"; blurb = "Skill-heavy lineup with licence to attack. Expect goals at both ends."; }
    else if (riskPct <= 40 && skillPct <= 45) { label = "Heavy and structured"; blurb = "A checking identity: you grind teams down, win walls and live on low-event games."; }
    else if (riskPct >= 58) { label = "Pressure attack"; blurb = "You push the play even without elite skill. Forecheck and volume shooting."; }
    else if (riskPct <= 42) { label = "Defence first"; blurb = "Safe starts and shutdown matchups. Goals will be hard to come by for both sides."; }
    else if (skillPct >= 62) { label = "Puck possession"; blurb = "Skill and patience: control the puck, cycle and wait for the opening."; }
    if (depthPct >= 75) blurb += " The top six carry most of the minutes, so their health is everything.";
    else if (depthPct <= 20) blurb += " Minutes are spread evenly; depth is the edge.";
    return { label, blurb, skillPct, riskPct, depthPct, avgAge, topStyle };
  }, [units, fwd]);

  /* Coaching synergy: how well the roster's minutes fit the coach's system. */
  const synergy = useMemo(() => {
    const all = units.flatMap((u) => u.fits.map((f) => ({ ...f, share: u.share })));
    const weighted = all.length ? all.reduce((a, f) => a + f.fit.score * f.share, 0) / all.reduce((a, f) => a + f.share, 0) : null;
    const favoured = all.filter((f) => system.favored.includes(f.fit.style)).length;
    const clashes = all.filter((f) => system.penalized.includes(f.fit.style));
    let risk = "neutral";
    if (sysKey === "defense_first" && identity.riskPct >= 60) risk = "clash";
    if (sysKey === "offensive" && identity.riskPct <= 40) risk = "clash";
    if ((sysKey === "defense_first" && identity.riskPct <= 45) || (sysKey === "offensive" && identity.riskPct >= 55)) risk = "match";
    return { score: weighted == null ? null : Math.round(weighted), favoured, clashes, total: all.length, risk };
  }, [units, system, sysKey, identity.riskPct]);

  const record = franchiseState?.team?.record || {};
  const standing = findStandingRow(franchiseState);
  const gf = num(standing?.gf ?? standing?.goals_for), ga = num(standing?.ga ?? standing?.goals_against);
  const gp = num(record.gp ?? standing?.gp, 0);
  const room = chemReport?.room || {};

  /* Adjustments: concrete suggestions with a one-click fix where we can make one. */
  const suggestions = useMemo(() => {
    const out = [];
    for (const u of units) {
      for (const f of u.fits) {
        if (f.fit.score < 45) out.push({ key: `fit-${f.p.id}`, tone: "bad", text: `${f.p.name} (${STYLE_LABEL[f.fit.style] || "role?"}) is a poor fit on ${u.line.name || `${u.group === "defense" ? "Pair" : "Line"} ${u.i + 1}`}: ${f.fit.reasons.filter((r) => r.tone === "bad").map((r) => r.text).join(", ") || "low system fit"}.` });
      }
      if (u.group === "forwards") {
        const skillN = u.fits.filter((f) => SKILL_STYLES.has(f.fit.style)).length;
        const gritN = u.fits.filter((f) => GRIT_STYLES.has(f.fit.style)).length;
        if (usageOf(u.line, u.i) === "checking" && skillN >= 2) out.push({ key: `use-${u.line.id}`, tone: "warn", text: `${u.line.name || `Line ${u.i + 1}`} has ${skillN} skill players on a checking assignment.`, fix: "Switch to attack", run: () => onDeploy?.("forwards", u.line.id, { usage: "attack" }) });
        if (usageOf(u.line, u.i) === "attack" && gritN >= 2 && u.i >= 2) out.push({ key: `use-${u.line.id}`, tone: "warn", text: `${u.line.name || `Line ${u.i + 1}`} is built to check but is told to attack.`, fix: "Switch to checking", run: () => onDeploy?.("forwards", u.line.id, { usage: "checking", zone: "defensive" }) });
      }
      if (u.chem && num(u.chem.score) != null && num(u.chem.score) < 45 && u.share >= 25) out.push({ key: `chem-${u.line.id}`, tone: "warn", text: `${u.line.name || "A top unit"} plays ${Math.round(u.share)}% of the night with weak chemistry (${u.chem.score}).` });
    }
    if (synergy.risk === "clash") out.push({ key: "sys", tone: "bad", text: `Your deployment (${Math.round(identity.riskPct)}% attacking) fights a ${system.label.toLowerCase()} coach. Expect lower buy-in.` });
    if (shareTotals && (Math.round(shareTotals.forwards) !== 100 || Math.round(shareTotals.defense) !== 100)) out.push({ key: "toi", tone: "bad", text: `Ice time must add to 100% (forwards ${Math.round(shareTotals.forwards)}%, defence ${Math.round(shareTotals.defense)}%).` });
    if (num(room.tension) != null && room.tension >= 55) out.push({ key: "room", tone: "warn", text: `Room tension is ${Math.round(room.tension)}. Locking in stable lines helps more than tinkering.` });
    return out.slice(0, 8);
  }, [units, synergy.risk, identity.riskPct, system.label, shareTotals, room.tension, onDeploy]);

  const summary = `${identity.label} · ${system.label} coach${synergy.score != null ? ` · ${synergy.score} system fit` : ""}`;

  return (
    <section className={`sbd ccx-panel ${open ? "is-open" : "is-closed"}`} aria-label="Strategy board">
      <header className="sbd-head">
        <button type="button" className="sbd-toggle" onClick={toggle} aria-expanded={open}>
          <span className="ccx-kicker">Strategy board</span>
          <strong className="sbd-head__summary">{summary}</strong>
          <span className="sbd-caret" aria-hidden>{open ? "▾" : "▸"}</span>
        </button>
        <nav className="ccx-tabs sbd-tabs" role="tablist" aria-label="Strategy sections">
          {TABS.map(([id, label]) => (
            <button key={id} type="button" role="tab" aria-selected={open && tab === id} className={`ccx-tab ${open && tab === id ? "is-active" : ""}`} onClick={() => pickTab(id)}>
              {label}{id === "adjust" && suggestions.length ? <span className="ccx-tab__count">{suggestions.length}</span> : null}
            </button>
          ))}
        </nav>
      </header>

      {open ? <div className="sbd-body">
        {tab === "identity" ? (
          <div className="sbd-grid sbd-grid--identity">
            <article className="sbd-card sbd-card--hero">
              <span className="ccx-kicker">Tactical identity</span>
              <h3 className="sbd-identity">{identity.label}</h3>
              <p>{identity.blurb}</p>
              <p className="sbd-dim">Read from who plays and how much: every skater weighted by his unit's ice time.{identity.topStyle ? ` Most minutes go to ${(STYLE_LABEL[identity.topStyle] || cap(identity.topStyle)).toLowerCase()}s.` : ""}{identity.avgAge ? ` Minutes-weighted age ${identity.avgAge.toFixed(1)}.` : ""}</p>
            </article>
            <article className="sbd-card">
              <span className="ccx-kicker">Style dials</span>
              <Meter left="Grit" right="Skill" value={identity.skillPct} note="Share of weighted minutes going to skill vs heavy players." />
              <Meter left="Safe" right="Aggressive" value={identity.riskPct} note="From zone starts, line style and how active your D pairs are." />
              <Meter left="Spread" right="Top-heavy" value={identity.depthPct} note="How much of the night the top two lines take." />
            </article>
            <article className="sbd-card">
              <span className="ccx-kicker">Coaching synergy</span>
              <h4 className="sbd-coach">{coachName} <em className="ccx-chip">{system.label}</em></h4>
              <p>{system.pitch}</p>
              <div className="sbd-synergy">
                <div className={`sbd-big is-${synergy.score == null ? "mid" : fitTone(synergy.score)}`}>{synergy.score ?? "—"}<small>system fit</small></div>
                <ul>
                  <li><b>{synergy.favoured}</b> of {synergy.total} dressed skaters play a style he favours ({system.favored.slice(0, 4).map((k) => (STYLE_LABEL[k] || k).toLowerCase()).join(", ")}).</li>
                  {synergy.clashes.length ? <li className="is-bad"><b>{synergy.clashes.length}</b> clash: {synergy.clashes.slice(0, 3).map((f) => f.p.name).join(", ")}.</li> : <li>No dressed player plays a style he punishes.</li>}
                  <li className={synergy.risk === "clash" ? "is-bad" : synergy.risk === "match" ? "is-good" : ""}>{synergy.risk === "clash" ? "Your deployment pulls against his system." : synergy.risk === "match" ? "Your deployment matches his system." : "Deployment is neutral for his system."}</li>
                </ul>
              </div>
            </article>
          </div>
        ) : null}

        {tab === "tactics" ? (
          <div className="sbd-grid sbd-grid--tactics">
            <article className="sbd-card sbd-span">
              <span className="ccx-kicker">Bench management</span>
              <h4>How much the coach overrides your plan in-game</h4>
              <Choice value={lineState?.coach_adjustments || "light"} onPick={(id) => onCoach?.(id)} options={[
                ["off", "Hands off", "Follows your ice-time shares exactly"],
                ["light", "Light touch", "Small tweaks for fatigue and penalties"],
                ["full", "Full control", "Shortens the bench when trailing, rides a hot line"],
              ]} />
            </article>
            {units.filter((u) => u.group === "forwards").map((u) => (
              <article key={u.line.id} className="sbd-card">
                <span className="ccx-kicker">{u.line.name || `Line ${u.i + 1}`} · {u.role}</span>
                <div className="sbd-ctl"><span>Zone starts</span><Choice value={zoneOf(u.line, u.i)} onPick={(zone) => onDeploy?.("forwards", u.line.id, { zone })} options={[["offensive", "Offensive", "More O-zone faceoffs: offence up, risk up"], ["defensive", "Defensive", "Own-end draws: protects leads, needs centres who win draws"]]} /></div>
                <div className="sbd-ctl"><span>Line style</span><Choice value={usageOf(u.line, u.i)} onPick={(usage) => onDeploy?.("forwards", u.line.id, { usage })} options={[["attack", "Attack", "Pinch, shoot, create"], ["checking", "Checking", "Chip, cycle, deny"]]} /></div>
                <p className="sbd-dim">{u.roleWhy}</p>
              </article>
            ))}
            {units.filter((u) => u.group === "defense").map((u) => (
              <article key={u.line.id} className="sbd-card">
                <span className="ccx-kicker">{u.line.name || `Pair ${u.i + 1}`} · {u.role}</span>
                <div className="sbd-ctl"><span>Matchup</span><Choice value={pairRoleOf(u.line, u.i)} onPick={(pair_role) => onDeploy?.("defense", u.line.id, { pair_role })} options={[["shutdown", "Shutdown", "Faces top lines"], ["neutral", "Neutral", "Fair minutes"], ["offensive", "Offensive", "Activates, O-zone draws"], ["sheltered", "Sheltered", "Easy minutes"]]} /></div>
                <p className="sbd-dim">{u.roleWhy}</p>
              </article>
            ))}
          </div>
        ) : null}

        {tab === "deploy" ? (
          <div className="sbd-deploy">
            {[["forwards", "Forward lines", "Of 60 minutes at even strength each line skates about"], ["defense", "Defence pairs", "Pairs split the night; the top pair usually plays 22–25 minutes"]].map(([group, title, note]) => {
              const rows = units.filter((u) => u.group === group);
              const total = shareTotals ? Math.round(shareTotals[group]) : Math.round(rows.reduce((a, u) => a + u.share, 0));
              return (
                <article key={group} className="sbd-card">
                  <div className="sbd-card__top"><span className="ccx-kicker">{title}</span><span className={`ccx-stamp ${total === 100 ? "is-good" : "is-bad"}`}>{total}%</span></div>
                  <div className="sbd-stack" aria-hidden>
                    {rows.map((u) => <i key={u.line.id} style={{ flexGrow: Math.max(1, u.share) }} className={`is-${u.i}`}>{Math.round(u.share)}%</i>)}
                  </div>
                  <p className="sbd-dim">{note}.</p>
                  <table className="ccx-table sbd-toi">
                    <thead><tr><th>Unit</th><th>Role</th><th>Share</th><th>~Min</th><th>Fit</th><th /></tr></thead>
                    <tbody>
                      {rows.map((u) => (
                        <tr key={u.line.id}>
                          <td><b>{u.line.name || `${group === "defense" ? "Pair" : "Line"} ${u.i + 1}`}</b><small>{u.fits.map((f) => f.p.name.split(" ").slice(-1)[0]).join(" · ") || "Empty"}</small></td>
                          <td title={u.roleWhy}>{u.role}</td>
                          <td className="is-num">{Math.round(u.share)}%</td>
                          <td className="is-num">{group === "defense" ? Math.round((u.share / 100) * 60 * 1.0) : u.minutes}</td>
                          <td className={`is-num is-${u.fitAvg == null ? "mid" : fitTone(u.fitAvg)}`}>{u.fitAvg == null ? "—" : Math.round(u.fitAvg)}</td>
                          <td className="sbd-steps">
                            <button type="button" className="ccx-btn ccx-btn--sm" onClick={() => onShareSet?.(group, u.line.id, Math.max(0, Math.round(u.share) - 1))} aria-label="Less ice time">−</button>
                            <button type="button" className="ccx-btn ccx-btn--sm" onClick={() => onShareSet?.(group, u.line.id, Math.min(100, Math.round(u.share) + 1))} aria-label="More ice time">+</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </article>
              );
            })}
            <article className="sbd-card sbd-span">
              <span className="ccx-kicker">Tactical fit by player</span>
              <div className="sbd-fitlist">
                {units.map((u) => (
                  <div key={`${u.group}-${u.line.id}`} className="sbd-fitunit">
                    <div className="sbd-fitunit__head"><b>{u.line.name || `${u.group === "defense" ? "Pair" : "Line"} ${u.i + 1}`}</b><span>{u.role}</span></div>
                    {u.fits.map((f) => (
                      <div key={f.p.id} className="sbd-fitrow">
                        <span className={`sbd-fitbadge is-${fitTone(f.fit.score)}`}>{f.fit.score}</span>
                        <span className="sbd-fitrow__who"><b>{f.slot} {f.p.name}</b><small>{STYLE_LABEL[f.fit.style] || cap(f.fit.style) || "Role unknown"}</small></span>
                        <span className="sbd-fitrow__why">{f.fit.reasons.length ? f.fit.reasons.map((r, k) => <em key={k} className={`is-${r.tone}`}>{r.text}</em>) : <em>neutral fit</em>}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </article>
          </div>
        ) : null}

        {tab === "perf" ? (
          <div className="sbd-grid">
            <article className="sbd-card">
              <span className="ccx-kicker">Results</span>
              <div className="sbd-kpis">
                <div><small>Record</small><b>{gp ? `${record.w ?? 0}-${record.l ?? 0}-${record.otl ?? 0}` : "—"}</b></div>
                <div><small>Points</small><b>{record.pts ?? "—"}</b></div>
                <div><small>GF / GA</small><b>{gf != null && ga != null ? `${gf} / ${ga}` : "—"}</b></div>
                <div><small>Goal diff / GP</small><b>{gf != null && ga != null && gp ? `${((gf - ga) / gp > 0 ? "+" : "")}${((gf - ga) / gp).toFixed(2)}` : "—"}</b></div>
              </div>
              <p className="sbd-dim">{gf != null && ga != null && gp ? (gf / gp >= 3.3 && identity.riskPct < 45 ? "You score like an attacking team while deploying safely: there may be more to unlock." : ga / gp >= 3.3 && identity.riskPct >= 55 ? "You are leaking goals with an aggressive setup. Consider defensive starts for the bottom six." : "Results are in line with the identity you've built.") : "Play a few games to see how the plan holds up."}</p>
            </article>
            <article className="sbd-card">
              <span className="ccx-kicker">Room</span>
              <div className="sbd-kpis">
                <div><small>Room chemistry</small><b>{num(room.overall) != null ? Math.round(room.overall) : "—"}</b></div>
                <div><small>Tension</small><b>{num(room.tension) != null ? Math.round(room.tension) : "—"}</b></div>
              </div>
              <p className="sbd-dim">{chemReport?.narrative || "The chemistry report loads with the lineup."}</p>
            </article>
            <article className="sbd-card sbd-span">
              <span className="ccx-kicker">Unit report</span>
              <table className="ccx-table">
                <thead><tr><th>Unit</th><th>Role</th><th>Chem</th><th>Coach fit</th><th>Usage happy</th><th>Morale</th><th>Read</th></tr></thead>
                <tbody>
                  {units.map((u) => {
                    const c = u.chem || {};
                    const read = c.concerns?.[0] || c.factors?.[0] || (num(c.score) >= 70 ? "Clicking." : num(c.score) < 45 ? "Not clicking yet." : "Finding its feet.");
                    return (
                      <tr key={`${u.group}-${u.line.id}`}>
                        <td><b>{u.line.name || `${u.group === "defense" ? "Pair" : "Line"} ${u.i + 1}`}</b></td>
                        <td>{u.role}</td>
                        <td className={`is-num is-${num(c.score) == null ? "mid" : fitTone(c.score)}`}>{num(c.score) ?? "—"}{c.projected ? "*" : ""}</td>
                        <td className="is-num">{num(c.coachFit) ?? (u.fitAvg != null ? Math.round(u.fitAvg) : "—")}</td>
                        <td className="is-num">{num(c.usageSatisfaction) ?? "—"}</td>
                        <td className="is-num">{num(c.morale) ?? "—"}</td>
                        <td className="sbd-read">{typeof read === "string" ? read : read?.text || read?.label || ""}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="sbd-dim">* projected from unsaved changes. Chemistry, coach fit and usage come from the same model the sim uses.</p>
            </article>
          </div>
        ) : null}

        {tab === "adjust" ? (
          <div className="sbd-adjust">
            {suggestions.length ? suggestions.map((sg) => (
              <div key={sg.key} className={`sbd-tip is-${sg.tone}`}>
                <span className="sbd-tip__mark" aria-hidden>{sg.tone === "bad" ? "!" : "?"}</span>
                <p>{sg.text}</p>
                {sg.fix ? <button type="button" className="ccx-btn ccx-btn--primary ccx-btn--sm" onClick={sg.run}>{sg.fix}</button> : null}
              </div>
            )) : (
              <div className="sbd-tip is-good"><span className="sbd-tip__mark" aria-hidden>✓</span><p>No red flags. Every unit has a job that suits it, and the coach is on board.</p></div>
            )}
          </div>
        ) : null}
      </div> : null}
    </section>
  );
}
