import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

/*
===========================================================
STATS CENTRAL — FULL SYSTEM
===========================================================

This is the central analytics hub of franchise mode.

Includes:
- Tabs system
- Overview dashboard
- Players table
- Goalies table
- Team stats
- League leaders
- Advanced analytics
- Comparison tool
- Impact scoring system
- 100 stat formula reference

===========================================================
*/

// --------------------------------------------------------
// UTIL FUNCTIONS
// --------------------------------------------------------

function fmtScore(g) {
  const ot = g.overtime ? " OT" : "";
  return `${g.home_goals}-${g.away_goals}${ot}`;
}

function safe(n) {
  return Number(n || 0);
}

function pickStat(...values) {
  for (let i = 0; i < values.length; i += 1) {
    const n = Number(values[i]);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function perGame(val, gp) {
  if (!gp) return 0;
  return val / gp;
}

function per60(val, toi) {
  if (!toi) return 0;
  return (val / toi) * 60;
}

function pct(a, b) {
  if (!b) return 0;
  return a / b;
}

// --------------------------------------------------------
// IMPACT SCORE SYSTEM (FROM PROMPT)
// --------------------------------------------------------

function calculateImpact(player) {
  const gp = pickStat(player.gp, player.games_played, player.games, 1) || 1;
  const pos = player.position || player.pos || "F";
  const fallbackToiPerGame = pos === "G" ? 56 : pos === "D" ? 22 : 17;
  const toi = pickStat(player.toi, player.toi_total, player.total_toi, player.time_on_ice, gp * fallbackToiPerGame) || 0;
  const g = Math.round(pickStat(player.g, player.goals, 0) || 0);
  const a = Math.round(pickStat(player.a, player.assists, 0) || 0);
  const pts = g + a;
  const cf = pickStat(player.cf, player.corsi_for, player.shot_attempts_for, player.sog ? player.sog * 2 : null, 1) || 1;
  const ca = pickStat(player.ca, player.corsi_against, player.shot_attempts_against, cf * 0.95, 1) || 1;
  const xgf = pickStat(player.xgf, player.expected_goals_for, g * 0.7 + a * 0.3, 1) || 1;
  const xga = pickStat(player.xga, player.expected_goals_against, ca * 0.018, 1) || 1;
  const gfOn = pickStat(player.gf_on, player.on_ice_gf, player.gf, g + a * 0.5, 1) || 1;
  const gaOn = pickStat(player.ga_on, player.on_ice_ga, xga * 0.9, 1) || 1;

  const p60 = per60(pts, toi);
  const cfPct = pct(cf, cf + ca) * 100;
  const xgfPct = pct(xgf, xgf + xga) * 100;
  const gfPct = pct(gfOn, gfOn + gaOn) * 100;
  const usage = clamp((toi / Math.max(1, gp)), 10, pos === "G" ? 60 : 28);
  const impact = 0.34 * p60 + 0.22 * cfPct + 0.22 * xgfPct + 0.12 * usage + 0.1 * gfPct;
  return impact.toFixed(2);
}

function normalizeStatsCentral(sc, teamInfo, franchiseState) {
  const rawLeague = sc.league_leaders || [];
  const rawMine = sc.user_team_skaters || [];
  const sourcePlayers = rawLeague.length ? rawLeague : rawMine;
  const teamId = teamInfo?.abbrev || teamInfo?.id || teamInfo?.name || rawMine[0]?.team_id || "USR";
  const calendar = Array.isArray(sc.calendar) ? sc.calendar : [];
  const recent = sc.recent_games || [];

  const playersBase = sourcePlayers.map((p, idx) => {
    const gp = pickStat(p.gp, p.games_played, p.games, 0) || 0;
    const pos = p.position || p.pos || "F";
    const toiPerGame = pos === "G" ? 56 : pos === "D" ? 22 : 17;
    const g = Math.round(pickStat(p.g, p.goals, 0) || 0);
    const a = Math.round(pickStat(p.a, p.assists, 0) || 0);
    const pts = g + a;
    const pimRaw = Math.round(pickStat(p.pim, p.penalty_minutes, p.pims, p.penalties, 0) || 0);
    const toi = pickStat(p.toi, p.toi_total, p.total_toi, p.time_on_ice, gp * toiPerGame) || 0;
    const sog = Math.round(pickStat(p.sog, p.shots_on_goal, p.shots, 0) || 0);
    const hit = Math.round(pickStat(p.hit, p.hits, 0) || 0);
    const blk = Math.round(pickStat(p.blk, p.blocks, 0) || 0);
    const cf = pickStat(p.cf, p.corsi_for, p.shot_attempts_for, sog * 2.1, 1) || 1;
    const ca = pickStat(p.ca, p.corsi_against, p.shot_attempts_against, cf * 0.97, 1) || 1;
    const xgf = pickStat(p.xgf, p.expected_goals_for, g * 0.72 + a * 0.24, 0.1) || 0.1;
    const xga = pickStat(p.xga, p.expected_goals_against, ca * 0.018, 0.1) || 0.1;
    const gfOn = pickStat(p.gf_on, p.on_ice_gf, p.gf, g + a * 0.5, 1) || 1;
    const gaOn = pickStat(p.ga_on, p.on_ice_ga, xga * 0.9, 1) || 1;

    return {
      ...p,
      player_id: p.player_id || p.id || `${p.name || "p"}-${idx}`,
      team_id: p.team_id || p.team || teamId,
      position: pos,
      gp,
      g,
      a,
      pts,
      sog,
      hit,
      blk,
      pim: Math.round(clamp(pimRaw, 0, 80)),
      toi,
      cf,
      ca,
      xgf,
      xga,
      cf_pct: pct(cf, cf + ca),
      xgf_pct: pct(xgf, xgf + xga),
      gf_pct: pct(gfOn, gfOn + gaOn),
      ppg: Math.round(pickStat(p.ppg, p.power_play_goals, 0) || 0),
      ppa: Math.round(pickStat(p.ppa, p.power_play_assists, 0) || 0),
      sha: Math.round(pickStat(p.sha, p.short_handed_assists, 0) || 0),
      fow: Math.round(pickStat(p.fow, p.faceoff_wins, 0) || 0),
      fol: Math.round(pickStat(p.fol, p.faceoff_losses, 0) || 0),
      age: pickStat(p.age, 25) || 25,
      rookie: Boolean(p.rookie || p.is_rookie || (pickStat(p.age, 99) || 99) <= 21),
      captain: Boolean(p.captain || p.is_captain || p.role === "C"),
    };
  });

  const currentGoalsAvg = playersBase.reduce((s, p) => s + p.g, 0) / Math.max(1, playersBase.length);
  const currentPimAvg = playersBase.reduce((s, p) => s + p.pim, 0) / Math.max(1, playersBase.length);
  const gfScale = currentGoalsAvg > 0 ? 3.8 / currentGoalsAvg : 1;
  const pimTarget = clamp(currentPimAvg || 50, 40, 60);
  const pimScale = currentPimAvg > 0 ? pimTarget / currentPimAvg : 1;

  const players = playersBase.map((p) => {
    const g = Math.round(p.g || 0);
    const a = Math.round(p.a || 0);
    const pts = g + a;
    const pim = Math.round(clamp(p.pim || 0, 0, 80));
    const impactScore = calculateImpact({ ...p, g, a, pts, pim });
    return {
      ...p,
      g,
      a,
      pts,
      sog: Math.round(p.sog || 0),
      hit: Math.round(p.hit || 0),
      blk: Math.round(p.blk || 0),
      pim,
      impactScore,
    };
  });

  const myPlayers = (rawMine.length ? rawMine : players.filter((p) => p.team_id === teamId)).map((p, idx) => {
    const g = Math.round(pickStat(p.g, p.goals, 0) || 0);
    const a = Math.round(pickStat(p.a, p.assists, 0) || 0);
    const pts = g + a;
    const gp = Math.round(pickStat(p.gp, p.games_played, p.games, 0) || 0);
    const pos = p.position || p.pos || "F";
    const toi = pickStat(p.toi, p.toi_total, p.total_toi, p.time_on_ice, gp * (pos === "G" ? 56 : pos === "D" ? 22 : 17)) || 0;
    const out = {
      ...p,
      player_id: p.player_id || p.id || `${p.name || "u"}-${idx}`,
      team_id: p.team_id || p.team || teamId,
      position: pos,
      gp,
      g,
      a,
      pts,
      sog: Math.round(pickStat(p.sog, p.shots_on_goal, p.shots, 0) || 0),
      hit: Math.round(pickStat(p.hit, p.hits, 0) || 0),
      blk: Math.round(pickStat(p.blk, p.blocks, 0) || 0),
      pim: Math.round(clamp(pickStat(p.pim, p.penalty_minutes, p.pims, 0) || 0, 0, 80)),
      toi,
      cf: pickStat(p.cf, p.corsi_for, p.sog ? p.sog * 2.1 : 1, 1) || 1,
      ca: pickStat(p.ca, p.corsi_against, 1, 1) || 1,
      xgf: pickStat(p.xgf, p.expected_goals_for, g * 0.72 + a * 0.24, 0.1) || 0.1,
      xga: pickStat(p.xga, p.expected_goals_against, 0.1) || 0.1,
      ppg: Math.round(pickStat(p.ppg, p.power_play_goals, 0) || 0),
      ppa: Math.round(pickStat(p.ppa, p.power_play_assists, 0) || 0),
      sha: Math.round(pickStat(p.sha, p.short_handed_assists, 0) || 0),
    };
    return { ...out, impactScore: calculateImpact(out) };
  });
  const goalies = players
    .filter((p) => p.position === "G")
    .map((g) => {
      const gp = Math.round(pickStat(g.gp, 0) || 0);
      const toi = pickStat(g.toi, gp * 56) || gp * 56;
      const shotsAgainst = pickStat(g.sa, g.shots_against, gp * 30, 1) || 1;
      const goalsAgainst = pickStat(g.ga, g.goals_against, gp * 2.9, 1) || 1;
      const saves = pickStat(g.saves, shotsAgainst - goalsAgainst, 0) || 0;
      const sv_pct = pickStat(g.sv_pct, g.sv, pct(saves, shotsAgainst), 0) || 0;
      const gaa = pickStat(g.gaa, (goalsAgainst * 60) / Math.max(1, toi), 0) || 0;
      return {
        ...g,
        gp,
        wins: Math.round(pickStat(g.wins, g.w, gp * 0.5, 0) || 0),
        losses: Math.round(pickStat(g.losses, g.l, gp * 0.4, 0) || 0),
        so: Math.round(pickStat(g.so, g.shutouts, 0) || 0),
        sv_pct,
        gaa,
      };
    });

  const allGamesRaw = [...calendar.flatMap((d) => d.games || []), ...recent];
  const seenGames = new Set();
  const allGames = allGamesRaw.filter((g, i) => {
    const k = String(g.game_id || `${g.day || "d"}-${g.home_id || g.home_team || g.home}-${g.away_id || g.away_team || g.away}-${g.home_goals}-${g.away_goals}-${i}`);
    if (seenGames.has(k)) return false;
    seenGames.add(k);
    return true;
  });
  let gaFromLogs = 0;
  let gfFromLogs = 0;
  allGames.forEach((g) => {
    const home = g.home_team || g.home_id || g.home;
    const away = g.away_team || g.away_id || g.away;
    if (home !== teamId && away !== teamId) return;
    const isHome = home === teamId;
    gfFromLogs += pickStat(isHome ? g.home_goals : g.away_goals, 0) || 0;
    gaFromLogs += pickStat(isHome ? g.away_goals : g.home_goals, 0) || 0;
  });
  const gf = gfFromLogs || pickStat(sc.team_team_stats?.gf, sc.team_team_stats?.goals_for, 0) || myPlayers.reduce((s, p) => s + p.g, 0);
  const ga = gaFromLogs || pickStat(sc.team_team_stats?.ga, sc.team_team_stats?.goals_against, 0) || 0;
  const sfFallback = pickStat(sc.team_team_stats?.sf, gf * 9.1, 1) || 1;
  const saFallback = pickStat(sc.team_team_stats?.sa, ga * 9.4, 1) || 1;

  const leagueTeamsRaw = sc.league_team_stats || sc.league_teams || [];
  const leagueTeamsFromPayload = leagueTeamsRaw.map((t, i) => ({
    id: t.team_id || t.abbrev || t.name || `T${i}`,
    gf: pickStat(t.gf, t.goals_for, 0) || 0,
    ga: pickStat(t.ga, t.goals_against, 0) || 0,
    pp_pct: pickStat(t.pp_pct, t.ppPct, 0) || 0,
    pk_pct: pickStat(t.pk_pct, t.pkPct, 0) || 0,
    sh_pct: pickStat(t.sh_pct, t.shPct, 0) || 0,
    sv_pct: pickStat(t.sv_pct, t.svPct, 0) || 0,
    pdo: pickStat(t.pdo, 0) || 0,
    points: pickStat(t.points, t.pts, 0) || 0,
    wins: pickStat(t.wins, t.w, 0) || 0,
    losses: pickStat(t.losses, t.l, 0) || 0,
    otl: pickStat(t.otl, 0) || 0,
    division_rank: pickStat(t.division_rank, t.div_rank, 0) || 0,
    conference_rank: pickStat(t.conference_rank, t.conf_rank, 0) || 0,
    division: String(t.division || ""),
    conference: String(t.conference || ""),
  }));
  const aggTeamsMap = {};
  allGames.forEach((g) => {
    const hid = String(g.home_id || g.home_team || g.home || "");
    const aid = String(g.away_id || g.away_team || g.away || "");
    if (!hid || !aid) return;
    if (!aggTeamsMap[hid]) aggTeamsMap[hid] = { id: hid, gf: 0, ga: 0, sf: 0, sa: 0, ppg: 0, ppo: 0, ppga: 0, oppPpo: 0, wins: 0, losses: 0, otl: 0, points: 0 };
    if (!aggTeamsMap[aid]) aggTeamsMap[aid] = { id: aid, gf: 0, ga: 0, sf: 0, sa: 0, ppg: 0, ppo: 0, ppga: 0, oppPpo: 0, wins: 0, losses: 0, otl: 0, points: 0 };
    const hg = pickStat(g.home_goals, 0) || 0;
    const ag = pickStat(g.away_goals, 0) || 0;
    const hs = pickStat(g.home_shots, 0) || 0;
    const as = pickStat(g.away_shots, 0) || 0;
    aggTeamsMap[hid].gf += hg; aggTeamsMap[hid].ga += ag; aggTeamsMap[hid].sf += hs; aggTeamsMap[hid].sa += as;
    aggTeamsMap[aid].gf += ag; aggTeamsMap[aid].ga += hg; aggTeamsMap[aid].sf += as; aggTeamsMap[aid].sa += hs;
    const ot = Boolean(g.overtime);
    if (hg > ag) { aggTeamsMap[hid].wins += 1; aggTeamsMap[hid].points += 2; if (ot) { aggTeamsMap[aid].otl += 1; aggTeamsMap[aid].points += 1; } else aggTeamsMap[aid].losses += 1; }
    else if (ag > hg) { aggTeamsMap[aid].wins += 1; aggTeamsMap[aid].points += 2; if (ot) { aggTeamsMap[hid].otl += 1; aggTeamsMap[hid].points += 1; } else aggTeamsMap[hid].losses += 1; }
    (g.scoring_events || []).forEach((ev) => {
      const tid = String(ev.for_team_id || "");
      if (!aggTeamsMap[tid]) return;
      const opp = tid === hid ? aid : hid;
      if (ev.strength === "PP") { aggTeamsMap[tid].ppg += 1; aggTeamsMap[tid].ppo += 1; aggTeamsMap[opp].ppga += 1; aggTeamsMap[opp].oppPpo += 1; }
    });
    aggTeamsMap[hid].ppo += Math.max(0, Math.floor((pickStat(g.away_pim, 0) || 0) / 2));
    aggTeamsMap[aid].ppo += Math.max(0, Math.floor((pickStat(g.home_pim, 0) || 0) / 2));
    aggTeamsMap[hid].oppPpo += Math.max(0, Math.floor((pickStat(g.home_pim, 0) || 0) / 2));
    aggTeamsMap[aid].oppPpo += Math.max(0, Math.floor((pickStat(g.away_pim, 0) || 0) / 2));
  });
  const leagueTeamsFromGames = Object.values(aggTeamsMap).map((t) => {
    const pp = t.ppo > 0 ? pct(t.ppg, t.ppo) : 0;
    const pk = t.oppPpo > 0 ? 1 - pct(t.ppga, t.oppPpo) : 0;
    const sh = t.sf > 0 ? pct(t.gf, t.sf) : 0;
    const sv = t.sa > 0 ? pct(t.sa - t.ga, t.sa) : 0;
    return {
      id: t.id,
      gf: t.gf,
      ga: t.ga,
      sf: t.sf,
      sa: t.sa,
      ppo: t.ppo,
      ppg: t.ppg,
      opp_ppo: t.oppPpo,
      ppga: t.ppga,
      pp_pct: pp,
      pk_pct: pk,
      sh_pct: sh,
      sv_pct: sv,
      pdo: sh + sv,
      points: t.points,
      wins: t.wins,
      losses: t.losses,
      otl: t.otl,
      division_rank: 0,
      conference_rank: 0,
      division: "",
      conference: "",
    };
  });

  const leagueTeams = leagueTeamsFromPayload.length >= 4 ? leagueTeamsFromPayload : leagueTeamsFromGames;
  const mergedTeams = leagueTeams.some((t) => t.id === teamId)
    ? leagueTeams.map((t) => (t.id === teamId ? { ...t, gf, ga } : t))
    : [...leagueTeams, { id: teamId, gf, ga, sf: sfFallback, sa: saFallback, ppo: 0, ppg: 0, opp_ppo: 0, ppga: 0, pp_pct: 0, pk_pct: 0, sh_pct: 0, sv_pct: 0, pdo: 0, points: 0, wins: 0, losses: 0, otl: 0, division_rank: 0, conference_rank: 0, division: "", conference: "" }];

  const fromTeamAgg = mergedTeams.find((t) => String(t.id) === String(teamId)) || {};
  const ppo = pickStat(fromTeamAgg.ppo, sc.team_team_stats?.ppo, sc.team_team_stats?.power_play_opportunities, 0) || 0;
  const ppg = pickStat(fromTeamAgg.ppg, sc.team_team_stats?.ppg, sc.team_team_stats?.power_play_goals, 0) || 0;
  const oppPpo = pickStat(fromTeamAgg.opp_ppo, sc.team_team_stats?.opp_ppo, sc.team_team_stats?.opp_pp_opportunities, 0) || 0;
  const ppga = pickStat(fromTeamAgg.ppga, sc.team_team_stats?.ppga, sc.team_team_stats?.pp_goals_against, 0) || 0;
  const pp_pct = ppo > 0 ? pct(ppg, ppo) : pickStat(fromTeamAgg.pp_pct, 0) || 0;
  const pk_pct = oppPpo > 0 ? 1 - pct(ppga, oppPpo) : pickStat(fromTeamAgg.pk_pct, 0) || 0;
  const sf = pickStat(fromTeamAgg.sf, sfFallback, 1) || 1;
  const sa = pickStat(fromTeamAgg.sa, saFallback, 1) || 1;
  const sh_pct = sf > 0 ? pct(gf, sf) : (pickStat(fromTeamAgg.sh_pct, 0) || 0);
  const sv_pct = sa > 0 ? pct(sa - ga, sa) : (pickStat(fromTeamAgg.sv_pct, 0) || 0);
  const pdo = sh_pct + sv_pct;
  const rankFor = (metric, asc = false) => {
    const sorted = [...mergedTeams].sort((a, b) => (asc ? a[metric] - b[metric] : b[metric] - a[metric]));
    return sorted.findIndex((t) => t.id === teamId) + 1;
  };

  const standingsRows = franchiseState?.standings || [];
  const fromStanding = standingsRows.find((r) => String(r.team_id || r.id || "") === String(teamId));
  const fromTeamPayload = sc.team_record || {};
  const wins = pickStat(fromTeamPayload.wins, fromStanding?.w, fromTeamAgg.wins);
  const losses = pickStat(fromTeamPayload.losses, fromStanding?.l, fromTeamAgg.losses);
  const otl = pickStat(fromTeamPayload.otl, fromStanding?.otl, fromTeamAgg.otl);
  const points = pickStat(fromTeamPayload.points, fromStanding?.pts, (wins || 0) * 2 + (otl || 0), fromTeamAgg.points);
  const divRank = pickStat(fromTeamPayload.division_rank, fromTeamAgg.division_rank, fromStanding?.division_rank, fromStanding?.div_rank, 0) || 0;
  const confRank = pickStat(fromTeamPayload.conference_rank, fromTeamAgg.conference_rank, fromStanding?.conference_rank, fromStanding?.conf_rank, 0) || 0;

  return {
    teamId,
    players,
    myPlayers,
    goalies,
    calendar,
    recent,
    leaders: players,
    team: {
      gf,
      ga,
      ppg: Math.round(ppg || 0),
      ppga: Math.round(ppga || 0),
      gf_scale: gfScale,
      pim_scale: pimScale,
      pp_pct,
      pk_pct,
      sh_pct,
      sv_pct,
      pdo,
      gf_rank: rankFor("gf"),
      ga_rank: rankFor("ga", true),
      pp_rank: rankFor("pp_pct"),
      pk_rank: rankFor("pk_pct"),
      pdo_rank: rankFor("pdo"),
      leagueTeams: mergedTeams,
    },
    teamRecord: {
      wins: Math.round(wins || 0),
      losses: Math.round(losses || 0),
      otl: Math.round(otl || 0),
      points: Math.round(points || 0),
      division_rank: Math.round(divRank || rankFor("points")),
      conference_rank: Math.round(confRank || rankFor("points")),
    },
    hasFullCalendar: !Boolean(sc.calendar_partial),
  };
}

// --------------------------------------------------------
// TABS
// --------------------------------------------------------

const TABS = [
  "overview",
  "players",
  "goalies",
  "team",
  "leaders",
  "advanced",
  "special",
  "logs",
  "trends",
  "compare",
  "impact",
  "awards",
  "formulas",
];

/* Scoped layout + missing Stats Central rules (this screen referenced classes
   that had no stylesheet entries, so tabs defaulted to inline flow and the
   overview stacked in a narrow column). */
const STATS_CENTRAL_CSS = `
.stats-central-screen {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.stats-central-screen .stats-toolbar {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 6px 12px 4px;
  box-sizing: border-box;
  flex-shrink: 0;
}

.stats-central-screen .stats-toolbar .game-btn {
  flex-shrink: 0;
  padding: 8px 14px;
  font-size: 0.65rem;
}

.stats-central-screen .stats-toolbar input {
  flex: 1;
  min-width: 0;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: 0.72rem;
  padding: 8px 12px;
  border-radius: 6px;
  border: 2px solid #5a657e;
  background: rgba(8, 12, 22, 0.75);
  color: #eef0f5;
}

.stats-central-screen .stats-toolbar input::placeholder {
  color: rgba(200, 205, 216, 0.55);
}

.stats-central-screen .stats-tabs {
  display: grid;
  width: 100%;
  box-sizing: border-box;
  padding: 2px 10px 6px;
  gap: 4px;
  grid-template-columns: repeat(13, minmax(0, 1fr));
}

.stats-central-screen .stats-tab-btn {
  box-sizing: border-box;
  min-width: 0;
  margin: 0;
  font-family: "Chakra Petch", "Arial Black", sans-serif;
  font-size: clamp(0.52rem, 0.55vw + 0.45rem, 0.68rem);
  letter-spacing: 0.06em;
  line-height: 1.15;
  padding: 7px 3px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: pointer;
  border-radius: 6px;
  border: 2px solid #5a657e;
  color: #c8cdd8;
  background: linear-gradient(180deg, #2a3555 0%, rgba(26, 36, 61, 0.9) 100%);
  transition: border-color 0.15s ease, filter 0.15s ease;
}

.stats-central-screen .stats-tab-btn:hover {
  filter: brightness(1.08);
  border-color: rgba(224, 112, 32, 0.55);
}

.stats-central-screen .stats-tab-btn--active {
  border-color: #ffd4a8;
  color: #1a0a02;
  background: linear-gradient(180deg, #e07020 0%, #a85518 100%);
}

.stats-central-screen .stats-content {
  flex: 1;
  min-height: 0;
  min-width: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0 10px 6px;
  box-sizing: border-box;
}

.stats-central-screen .stats-overview {
  flex: 1;
  min-height: 0;
  min-width: 0;
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr) minmax(0, 1fr) minmax(0, 1.1fr);
  grid-template-rows: auto minmax(0, 1fr);
  gap: 8px 10px;
  align-items: stretch;
  box-sizing: border-box;
}

.stats-central-screen .stats-overview > .stats-overview-grid {
  grid-column: 1 / -1;
}

.stats-central-screen .stats-overview > .stats-section:nth-child(2) {
  grid-column: 1;
  grid-row: 2;
  min-height: 0;
}

.stats-central-screen .stats-overview > .stats-section:nth-child(3) {
  grid-column: 2;
  grid-row: 2;
  min-height: 0;
}

.stats-central-screen .stats-overview > .stats-section:nth-child(4) {
  grid-column: 3;
  grid-row: 2;
  min-height: 0;
}

.stats-central-screen .stats-overview > .stats-section:nth-child(5) {
  grid-column: 4;
  grid-row: 2;
  min-height: 0;
}

.stats-central-screen .stats-overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
  gap: 6px;
  width: 100%;
  box-sizing: border-box;
}

.stats-central-screen .stats-card {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 6px 8px;
  background: rgba(6, 10, 18, 0.72);
  min-width: 0;
}

.stats-central-screen .stats-card__label {
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(200, 205, 216, 0.65);
  margin-bottom: 2px;
}

.stats-central-screen .stats-card__value {
  font-family: "Chakra Petch", "Arial Black", sans-serif;
  font-size: 0.82rem;
  color: #e07020;
  line-height: 1.2;
  word-break: break-word;
}

.stats-central-screen .stats-card__sub {
  font-size: 0.62rem;
  color: rgba(200, 205, 216, 0.8);
  margin-top: 2px;
  line-height: 1.25;
}

.stats-central-screen .stats-section {
  display: flex;
  flex-direction: column;
  min-height: 0;
  min-width: 0;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 6px 8px;
  background: rgba(8, 12, 22, 0.45);
}

.stats-central-screen .stats-section__header {
  flex-shrink: 0;
}

.stats-central-screen .stats-section__header h2 {
  margin: 0 0 2px;
  font-family: "Chakra Petch", "Arial Black", sans-serif;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: #e07020;
}

.stats-central-screen .stats-section__header p {
  margin: 0 0 6px;
  font-size: 0.62rem;
  line-height: 1.25;
  color: rgba(200, 205, 216, 0.75);
}

.stats-central-screen .stats-section > h3 {
  margin: 0 0 6px;
  font-family: "Chakra Petch", "Arial Black", sans-serif;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  color: #9d5fd4;
}

.stats-central-screen .stats-leader-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px;
  flex: 1;
  min-height: 0;
  overflow: auto;
  align-content: start;
}

.stats-central-screen .stats-mini-card {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  padding: 5px 6px;
  background: rgba(6, 10, 18, 0.65);
  min-width: 0;
}

.stats-central-screen .stats-mini-card__title {
  font-size: 0.55rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.4);
}

.stats-central-screen .stats-mini-card__name {
  font-size: 0.68rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stats-central-screen .stats-mini-card__stat {
  font-size: 0.62rem;
  color: #e07020;
  font-weight: 700;
}

.stats-central-screen .stats-cal-list {
  flex: 1;
  min-height: 0;
  max-height: none;
  overflow-y: auto;
  overflow-x: hidden;
  gap: 4px;
  padding-right: 2px;
}

.stats-central-screen .stats-cal-pill {
  padding: 5px 8px;
  font-size: 0.65rem;
}

.stats-central-screen .stats-score-list {
  flex: 1;
  min-height: 0;
  max-height: none;
  overflow-y: auto;
  gap: 5px;
}

.stats-central-screen .stats-score-card {
  padding: 6px 8px;
}

.stats-central-screen .stats-score-card__line {
  font-size: 0.68rem;
}

.stats-central-screen .stats-tab {
  flex: 1;
  min-height: 0;
  min-width: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow-x: hidden;
  overflow-y: auto;
  box-sizing: border-box;
}

.stats-central-screen .stats-tab > .stats-score-list {
  flex: 1;
  min-height: 0;
}

.stats-central-screen .stats-tab > .stats-table-wrap {
  flex: 1;
  min-height: 0;
}

.stats-central-screen .stats-tab > .formula-sections {
  flex: 1;
  min-height: 0;
}

.stats-central-screen .stats-table-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.stats-central-screen .stats-table-body {
  flex: 1;
  min-height: 0;
  max-height: none !important;
  overflow-y: auto;
  overflow-x: auto;
}

.stats-central-screen .stats-table-header--ultra,
.stats-central-screen .stats-table-row--ultra {
  grid-template-columns: minmax(56px, 1.1fr) repeat(6, minmax(40px, 0.75fr)) minmax(52px, 0.85fr);
  gap: 4px;
  font-size: 0.6rem;
}

.stats-central-screen .stats-table-header--wide,
.stats-central-screen .stats-table-row--wide {
  grid-template-columns: 2fr 1fr repeat(8, 1fr);
  width: 100%;
  gap: 4px;
}

.stats-central-screen .stats-dual-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  flex: 1;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
}

.stats-central-screen .stats-dual-grid > .stats-section {
  min-height: 0;
}

.stats-central-screen .stats-notes {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  font-size: 0.62rem;
  line-height: 1.3;
}

.stats-central-screen .stats-note {
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(6, 10, 18, 0.55);
}

.stats-central-screen .stats-compare-selectors {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  flex-shrink: 0;
}

.stats-central-screen .stats-compare-selectors select {
  width: 100%;
  min-width: 0;
  font-size: 0.68rem;
  padding: 6px 8px;
  border-radius: 6px;
  border: 2px solid #5a657e;
  background: rgba(8, 12, 22, 0.85);
  color: #eef0f5;
}

.stats-central-screen .stats-compare-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  flex-shrink: 0;
}

.stats-central-screen .stats-compare-bref {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(88px, 0.55fr) minmax(0, 1fr);
  gap: 4px;
}

.stats-central-screen .stats-compare-better {
  color: #59d185;
  font-weight: 700;
}

.stats-central-screen .stats-compare-worse {
  color: #d16262;
}

.stats-central-screen .stats-compare-card {
  border-radius: 8px;
  border: 2px solid #5a657e;
  padding: 8px 10px;
  background: linear-gradient(180deg, rgba(42, 53, 85, 0.55) 0%, rgba(26, 36, 61, 0.75) 100%);
  font-size: 0.68rem;
}

.stats-central-screen .stats-compare-card__name {
  font-weight: 700;
  font-size: 0.75rem;
}

.stats-central-screen .stats-identity {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.stats-central-screen .identity-bar {
  width: 100%;
}

.stats-central-screen .formula-sections {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  align-content: start;
}

.stats-central-screen .formula-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 5px;
}

.stats-central-screen .formula-card {
  padding: 5px 7px;
  font-size: 0.58rem;
}

.stats-central-screen .formula-card__name {
  font-size: 0.62rem;
}

.stats-central-screen .formula-card__formula {
  font-size: 0.55rem;
  line-height: 1.25;
}

@media (max-width: 1320px) {
  .stats-central-screen .stats-tabs {
    grid-template-columns: repeat(7, minmax(0, 1fr));
  }
}

@media (max-width: 1100px) {
  .stats-central-screen .stats-overview {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto repeat(2, minmax(0, 1fr));
  }
  .stats-central-screen .stats-overview > .stats-section:nth-child(2) {
    grid-column: 1;
    grid-row: 2;
  }
  .stats-central-screen .stats-overview > .stats-section:nth-child(3) {
    grid-column: 2;
    grid-row: 2;
  }
  .stats-central-screen .stats-overview > .stats-section:nth-child(4) {
    grid-column: 1;
    grid-row: 3;
  }
  .stats-central-screen .stats-overview > .stats-section:nth-child(5) {
    grid-column: 2;
    grid-row: 3;
  }
  .stats-central-screen .stats-dual-grid {
    grid-template-columns: 1fr;
  }
  .stats-central-screen .formula-sections {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .stats-central-screen .stats-tabs {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }
  .stats-central-screen .stats-overview {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .stats-central-screen .stats-overview > .stats-section:nth-child(2),
  .stats-central-screen .stats-overview > .stats-section:nth-child(3),
  .stats-central-screen .stats-overview > .stats-section:nth-child(4),
  .stats-central-screen .stats-overview > .stats-section:nth-child(5) {
    flex: 0 0 auto;
    max-height: 32vh;
  }
}
`;

// --------------------------------------------------------
// MAIN COMPONENT
// --------------------------------------------------------

export function StatsCentralScreen() {
  const { franchiseState, setScreen } = useGameUI();

  const sc = franchiseState?.stats_central || {};
  const normalized = useMemo(
    () => normalizeStatsCentral(sc, franchiseState?.team, franchiseState),
    [sc, franchiseState?.team, franchiseState]
  );
  const calendar = normalized.calendar;
  const recent = normalized.recent;
  const leaders = normalized.leaders;
  const mine = normalized.myPlayers;
  const goalies = normalized.goalies;

  const [selectedDay, setSelectedDay] = useState(null);
  const [tab, setTab] = useState("overview");
  const [search, setSearch] = useState("");

  // --------------------------------------------------------
  // ESC BACK
  // --------------------------------------------------------

  useEffect(() => {
    function onKey(e) {
      if (e.target.matches("input")) return;
      if (e.key === "Escape") setScreen(SCREENS.HUB);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setScreen]);

  const gamesForDay = useMemo(() => {
    if (selectedDay == null) return recent;
    const block = calendar.find((c) => Number(c.day) === Number(selectedDay));
    return block?.games || [];
  }, [calendar, recent, selectedDay]);

  const onBack = useCallback(() => {
    setScreen(SCREENS.HUB);
  }, [setScreen]);

  // --------------------------------------------------------
  // FILTERED PLAYERS
  // --------------------------------------------------------

  const filteredPlayers = useMemo(() => {
    return mine.filter((p) =>
      p.name?.toLowerCase().includes(search.toLowerCase())
    );
  }, [mine, search]);

  const topScorer = useMemo(
    () => [...mine].sort((a, b) => (safe(b.pts) - safe(a.pts)) || (safe(b.g) - safe(a.g)))[0] || null,
    [mine]
  );
  const topGoalie = useMemo(
    () => {
      const minGp = Math.max(5, Math.floor((safe(normalized.teamRecord?.wins) + safe(normalized.teamRecord?.losses) + safe(normalized.teamRecord?.otl)) * 0.15));
      const pool = goalies.filter((g) => safe(g.gp) >= minGp);
      const src = pool.length ? pool : goalies;
      return [...src].sort((a, b) => (safe(b.sv_pct) - safe(a.sv_pct)) || (safe(a.gaa) - safe(b.gaa)))[0] || null;
    },
    [goalies, normalized.teamRecord]
  );
  const bestImpact = useMemo(
    () => [...filteredPlayers].sort((a, b) => Number(calculateImpact(b)) - Number(calculateImpact(a)))[0] || null,
    [filteredPlayers]
  );

  // --------------------------------------------------------
  // RENDER
  // --------------------------------------------------------

  return (
    <div className="game-screen stats-central-screen">
      <style dangerouslySetInnerHTML={{ __html: STATS_CENTRAL_CSS }} />
      <GameHeader
        teamName={franchiseState?.team?.name || "—"}
        sectionTitle="STATS CENTRAL"
      />

      {/* TOOLBAR */}
      <div className="stats-toolbar">
        <button type="button" className="game-btn" onClick={onBack}>
          ← HUB
        </button>

        <input
          type="search"
          placeholder="Search player..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* TABS — equal-width grid so every tab stays visible (no clustered inline row). */}
      <div className="stats-tabs" role="tablist" aria-label="Stats Central sections">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            className={
              tab === t ? "stats-tab-btn stats-tab-btn--active" : "stats-tab-btn"
            }
            onClick={() => setTab(t)}
          >
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {/* CONTENT */}
      <div className="stats-content">
        {tab === "overview" && (
          <div className="stats-overview">
            <div className="stats-overview-grid">
              <StatCard
                label="Record"
                value={`${safe(normalized.teamRecord?.wins)}-${safe(normalized.teamRecord?.losses)}-${safe(normalized.teamRecord?.otl)}`}
                sub={`Points: ${safe(normalized.teamRecord?.points)}`}
              />
              <StatCard
                label="Division Rank"
                value={safe(normalized.teamRecord?.division_rank) || "—"}
                sub={`Conference: ${safe(normalized.teamRecord?.conference_rank) || "—"}`}
              />
              <StatCard
                label="Goals For"
                value={Math.round(normalized.team.gf)}
                sub={`Rank: ${normalized.team.gf_rank || "—"}`}
              />
              <StatCard
                label="Goals Against"
                value={Math.round(normalized.team.ga)}
                sub={`Rank: ${normalized.team.ga_rank || "—"}`}
              />
              <StatCard
                label="Power Play"
                value={`${(normalized.team.pp_pct * 100).toFixed(1)}%`}
                sub={`Rank: ${normalized.team.pp_rank || "—"}`}
              />
              <StatCard
                label="Penalty Kill"
                value={`${(normalized.team.pk_pct * 100).toFixed(1)}%`}
                sub={`Rank: ${normalized.team.pk_rank || "—"}`}
              />
              <StatCard
                label="Team Save %"
                value={`${(normalized.team.sv_pct * 100).toFixed(1)}%`}
                sub="All goalies combined"
              />
              <StatCard
                label="Team Shooting %"
                value={`${(normalized.team.sh_pct * 100).toFixed(1)}%`}
                sub="All skaters combined"
              />
              <StatCard
                label="PDO"
                value={normalized.team.pdo.toFixed(3)}
                sub="Luck / finishing proxy"
              />
              <StatCard
                label="Top Scorer"
                value={topScorer?.name || "—"}
                sub={`${Math.round(pickStat(topScorer?.pts, 0) || 0)} PTS`}
              />
              <StatCard
                label="Top Goalie"
                value={topGoalie?.name || "—"}
                sub={`${((pickStat(topGoalie?.sv_pct, 0) || 0) * 100).toFixed(1)} SV%`}
              />
              <StatCard
                label="Best Impact"
                value={bestImpact?.name || "—"}
                sub={`Score: ${bestImpact ? calculateImpact(bestImpact) : "0.00"}`}
              />
            </div>

            {normalized.hasFullCalendar && (
            <div className="stats-section">
              <div className="stats-section__header">
                <h2>Calendar</h2>
                <p>Pick a league day to load every final that night.</p>
              </div>

              <div className="stats-cal-list">
                <button
                  type="button"
                  className={`stats-cal-pill ${selectedDay === null ? "is-active" : ""}`}
                  onClick={() => setSelectedDay(null)}
                >
                  Latest (all)
                </button>

                {calendar.map((c) => (
                  <button
                    key={c.day}
                    type="button"
                    className={`stats-cal-pill ${selectedDay === Number(c.day) ? "is-active" : ""}`}
                    onClick={() => setSelectedDay(Number(c.day))}
                  >
                    Day {c.day}
                    <span className="stats-cal-pill__n">{(c.games || []).length}</span>
                  </button>
                ))}
              </div>
            </div>)}

            <div className="stats-section">
              <div className="stats-section__header">
                <h2>Scores</h2>
                <p>
                  {selectedDay == null
                    ? "Most recent league results."
                    : `Finals from calendar day ${selectedDay}.`}
                </p>
              </div>

              <div className="stats-score-list">
                {gamesForDay.length === 0 ? (
                  <div className="stats-empty">
                    Sim games appear here after you advance the calendar.
                  </div>
                ) : (
                  gamesForDay.map((g, i) => (
                    <article
                      key={`${g.day}-${g.home_id}-${g.away_id}-${i}`}
                      className="stats-score-card"
                    >
                      <div className="stats-score-card__line">
                        <span className="stats-score-card__team">{g.home_name}</span>
                        <span className="stats-score-card__goals">{g.home_goals}</span>
                        <span className="stats-score-card__dash">—</span>
                        <span className="stats-score-card__goals">{g.away_goals}</span>
                        <span className="stats-score-card__team stats-score-card__team--away">
                          {g.away_name}
                        </span>
                      </div>

                      <div className="stats-score-card__meta">
                        Day {g.day} · {fmtScore(g)}
                      </div>

                      {(g.home_scoring?.length > 0 || g.away_scoring?.length > 0) && (
                        <div className="stats-score-card__scorers">
                          <div>
                            <span className="stats-score-card__lbl">Home</span>
                            <span>{(g.home_scoring || []).join(" · ") || "—"}</span>
                          </div>
                          <div>
                            <span className="stats-score-card__lbl">Away</span>
                            <span>{(g.away_scoring || []).join(" · ") || "—"}</span>
                          </div>
                        </div>
                      )}
                    </article>
                  ))
                )}
              </div>
            </div>

            <div className="stats-section">
              <div className="stats-section__header">
                <h2>Quick Leaders</h2>
                <p>Fast executive snapshot of your room.</p>
              </div>

              <div className="stats-leader-grid">
                <LeaderMiniCard
                  title="Goals"
                  player={topBy(filteredPlayers, (p) => safe(p.g))}
                  stat={`${Math.round(pickStat(topBy(filteredPlayers, (p) => safe(p.g))?.g, 0) || 0)} G`}
                />
                <LeaderMiniCard
                  title="Assists"
                  player={topBy(filteredPlayers, (p) => safe(p.a))}
                  stat={`${Math.round(pickStat(topBy(filteredPlayers, (p) => safe(p.a))?.a, 0) || 0)} A`}
                />
                <LeaderMiniCard
                  title="Points"
                  player={topBy(filteredPlayers, (p) => safe(p.pts))}
                  stat={`${Math.round(pickStat(topBy(filteredPlayers, (p) => safe(p.pts))?.pts, 0) || 0)} P`}
                />
                <LeaderMiniCard
                  title="Shots"
                  player={topBy(filteredPlayers, (p) => safe(p.sog))}
                  stat={`${safe(topBy(filteredPlayers, (p) => safe(p.sog))?.sog)} SOG`}
                />
                <LeaderMiniCard
                  title="Hits"
                  player={topBy(filteredPlayers, (p) => safe(p.hit))}
                  stat={`${safe(topBy(filteredPlayers, (p) => safe(p.hit))?.hit)} HIT`}
                />
                <LeaderMiniCard
                  title="Blocks"
                  player={topBy(filteredPlayers, (p) => safe(p.blk))}
                  stat={`${Math.round(pickStat(topBy(filteredPlayers, (p) => safe(p.blk))?.blk, 0) || 0)} BLK`}
                />
                <LeaderMiniCard
                  title="TOI"
                  player={topBy(filteredPlayers, (p) => safe(p.toi))}
                  stat={`${safe(topBy(filteredPlayers, (p) => safe(p.toi))?.toi).toFixed(1)} TOI`}
                />
                <LeaderMiniCard
                  title="Impact"
                  player={bestImpactPlayer(filteredPlayers)}
                  stat={`${bestImpactPlayer(filteredPlayers)?.impactScore || "0.00"} IMP`}
                />
                <LeaderMiniCard
                  title="P/GP"
                  player={topBy(filteredPlayers, (p) => perGame(safe(p.pts), Math.max(1, safe(p.gp))))}
                  stat={`${(perGame(safe(topBy(filteredPlayers, (p) => perGame(safe(p.pts), Math.max(1, safe(p.gp))))?.pts), Math.max(1, safe(topBy(filteredPlayers, (p) => perGame(safe(p.pts), Math.max(1, safe(p.gp))))?.gp))) || 0).toFixed(2)} P/GP`}
                />
                <LeaderMiniCard
                  title="CF%"
                  player={topBy(filteredPlayers, (p) => pct(safe(p.cf), safe(p.cf) + safe(p.ca)))}
                  stat={`${((pct(safe(topBy(filteredPlayers, (p) => pct(safe(p.cf), safe(p.cf) + safe(p.ca)))?.cf), safe(topBy(filteredPlayers, (p) => pct(safe(p.cf), safe(p.cf) + safe(p.ca)))?.cf) + safe(topBy(filteredPlayers, (p) => pct(safe(p.cf), safe(p.cf) + safe(p.ca)))?.ca)) || 0) * 100).toFixed(1)} CF%`}
                />
                <LeaderMiniCard
                  title="xGF%"
                  player={topBy(filteredPlayers, (p) => pct(safe(p.xgf), safe(p.xgf) + safe(p.xga)))}
                  stat={`${((pct(safe(topBy(filteredPlayers, (p) => pct(safe(p.xgf), safe(p.xgf) + safe(p.xga)))?.xgf), safe(topBy(filteredPlayers, (p) => pct(safe(p.xgf), safe(p.xgf) + safe(p.xga)))?.xgf) + safe(topBy(filteredPlayers, (p) => pct(safe(p.xgf), safe(p.xgf) + safe(p.xga)))?.xga)) || 0) * 100).toFixed(1)} xGF%`}
                />
                <LeaderMiniCard
                  title="PP Pts"
                  player={topBy(filteredPlayers, (p) => safe(p.ppg) + safe(p.ppa))}
                  stat={`${safe(topBy(filteredPlayers, (p) => safe(p.ppg) + safe(p.ppa))?.ppg) + safe(topBy(filteredPlayers, (p) => safe(p.ppg) + safe(p.ppa))?.ppa)} PPP`}
                />
              </div>
            </div>

            <div className="stats-section">
              <div className="stats-section__header">
                <h2>League Leaders</h2>
                <p>Points leaders around the league.</p>
              </div>

              <div className="stats-table-wrap">
                <div className="stats-table-header">
                  <span>#</span>
                  <span>Player</span>
                  <span>Tm</span>
                  <span>Pos</span>
                  <span>GP</span>
                  <span>G</span>
                  <span>A</span>
                  <span>P</span>
                </div>

                <div className="stats-table-body">
                  {leaders.length === 0 ? (
                    <div className="stats-empty stats-empty--sm">No skater stats yet.</div>
                  ) : (
                    leaders.slice(0, 40).map((r, idx) => (
                      <div key={r.player_id || idx} className="stats-table-row">
                        <span>{idx + 1}</span>
                        <span className="stats-table-name" title={r.name}>
                          {r.name}
                        </span>
                        <span className="stats-table-abbr" title={r.team_id}>
                          {String(r.team_id || "").slice(0, 3)}
                        </span>
                        <span>{r.position}</span>
                        <span>{r.gp}</span>
                        <span>{Math.round(pickStat(r.g, 0) || 0)}</span>
                        <span>{Math.round(pickStat(r.a, 0) || 0)}</span>
                        <span className="stats-table-pts">{Math.round(pickStat(r.pts, 0) || 0)}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "players" && (
          <PlayersTab players={filteredPlayers} />
        )}

        {tab === "goalies" && (
          <GoaliesTab goalies={goalies} />
        )}

        {tab === "team" && (
          <TeamTab sc={sc} players={filteredPlayers} goalies={goalies} normalized={normalized} />
        )}

        {tab === "leaders" && (
          <LeadersTab leaders={leaders} players={filteredPlayers} goalies={goalies} />
        )}

        {tab === "advanced" && (
          <AdvancedTab players={filteredPlayers} />
        )}

        {tab === "special" && (
          <SpecialTeamsTab players={filteredPlayers} sc={sc} normalized={normalized} />
        )}

        {tab === "logs" && (
          <GameLogsTab recent={recent} calendar={calendar} />
        )}

        {tab === "trends" && (
          <TrendsTab players={filteredPlayers} goalies={goalies} sc={sc} recent={recent} leaders={leaders} />
        )}

        {tab === "compare" && (
          <CompareTab players={leaders} goalies={goalies} />
        )}

        {tab === "impact" && (
          <ImpactTab players={filteredPlayers} />
        )}

        {tab === "awards" && (
          <AwardsWatchTab players={leaders} goalies={goalies} leaders={leaders} normalized={normalized} />
        )}

        {tab === "formulas" && (
          <FormulasTab />
        )}
      </div>

      <GameFooter />
    </div>
  );
}

/* =========================================================
   SMALL COMPONENTS
========================================================= */

function StatCard({ label, value, sub }) {
  return (
    <div className="stats-card">
      <div className="stats-card__label">{label}</div>
      <div className="stats-card__value">{value}</div>
      <div className="stats-card__sub">{sub}</div>
    </div>
  );
}

function LeaderMiniCard({ title, player, stat }) {
  return (
    <div className="stats-mini-card">
      <div className="stats-mini-card__title">{title}</div>
      <div className="stats-mini-card__name">{player?.name || "—"}</div>
      <div className="stats-mini-card__stat">{stat || "—"}</div>
    </div>
  );
}

function topBy(arr, fn) {
  if (!arr?.length) return null;
  return [...arr].sort((a, b) => fn(b) - fn(a))[0];
}

function bestImpactPlayer(players) {
  if (!players?.length) return null;

  const enriched = players.map((p) => ({
    ...p,
    impactScore: calculateImpact({
      ...p,
      toi: safe(p.toi || p.toi_total || p.total_toi || 0),
      cf: safe(p.cf || p.corsi_for || p.shot_attempts_for || 0),
      ca: safe(p.ca || p.corsi_against || p.shot_attempts_against || 1),
    }),
  }));

  return enriched.sort((a, b) => Number(b.impactScore) - Number(a.impactScore))[0];
}/* =========================================================
   PLAYERS TAB — FULL TABLE + SORTING + IMPACT
========================================================= */

function PlayersTab({ players }) {
  const [sortKey, setSortKey] = useState("pts");
  const [desc, setDesc] = useState(true);

  const sorted = useMemo(() => {
    const arr = [...players].map((p) => {
      const gp = pickStat(p.gp, 0) || 0;
      const pos = p.position || "F";
      const minPerGame = pos === "G" ? 56 : pos === "D" ? 22 : 17;
      const toi = pickStat(p.toi, p.toi_total, gp * minPerGame, 0) || 0;
      const pim = clamp(pickStat(p.pim, p.penalty_minutes, 0) || 0, 0, 80);
      return { ...p, toi, pim, impactScore: calculateImpact({ ...p, toi, pim }) };
    });

    return arr.sort((a, b) => {
      const A = pickStat(a[sortKey], 0) || 0;
      const B = pickStat(b[sortKey], 0) || 0;
      return desc ? B - A : A - B;
    });
  }, [players, sortKey, desc]);

  const toggleSort = (key) => {
    if (key === sortKey) setDesc(!desc);
    else {
      setSortKey(key);
      setDesc(true);
    }
  };

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Players</h2>
        <p>Full roster analytics — sortable + impact driven.</p>
      </div>

      <div className="stats-table-wrap">
        <div className="stats-table-header stats-table-header--wide">
          <span onClick={() => toggleSort("name")}>Player</span>
          <span>Pos</span>
          <span onClick={() => toggleSort("gp")}>GP</span>
          <span onClick={() => toggleSort("g")}>G</span>
          <span onClick={() => toggleSort("a")}>A</span>
          <span onClick={() => toggleSort("pts")}>P</span>
          <span onClick={() => toggleSort("sog")}>SOG</span>
          <span onClick={() => toggleSort("pim")}>PIM</span>
          <span onClick={() => toggleSort("toi")}>TOI</span>
          <span onClick={() => toggleSort("cf_pct")}>CF%</span>
          <span onClick={() => toggleSort("impactScore")}>Impact</span>
        </div>

        <div className="stats-table-body">
          {sorted.map((p) => (
            <div key={p.player_id} className="stats-table-row stats-table-row--wide">
              <span className="stats-table-name">{p.name}</span>
              <span>{p.position}</span>
              <span>{p.gp}</span>
              <span>{Math.round(pickStat(p.g, 0) || 0)}</span>
              <span>{Math.round(pickStat(p.a, 0) || 0)}</span>
              <span className="stats-table-pts">{Math.round(pickStat(p.pts, 0) || 0)}</span>
              <span>{Math.round(pickStat(p.sog, 0) || 0)}</span>
              <span>{Math.round(pickStat(p.pim, 0) || 0)}</span>
              <span>{safe(p.toi).toFixed(1)}</span>
              <span>{((pickStat(p.cf_pct, pct(p.cf, (p.cf || 0) + (p.ca || 0)), 0) || 0) * 100).toFixed(1)}</span>
              <span className="stats-impact">{p.impactScore}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   GOALIES TAB — SEPARATE SYSTEM
========================================================= */

function GoaliesTab({ goalies }) {
  const sorted = useMemo(() => {
    return [...goalies].sort((a, b) => (pickStat(b.sv_pct, 0) || 0) - (pickStat(a.sv_pct, 0) || 0));
  }, [goalies]);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Goalies</h2>
        <p>Dedicated goalie analytics — separate from skaters.</p>
      </div>

      <div className="stats-table-wrap">
        <div className="stats-table-header stats-table-header--wide">
          <span>Player</span>
          <span>GP</span>
          <span>W</span>
          <span>L</span>
          <span>SV%</span>
          <span>GAA</span>
          <span>SO</span>
        </div>

        <div className="stats-table-body">
          {sorted.map((g) => (
            <div key={g.player_id} className="stats-table-row stats-table-row--wide">
              <span>{g.name}</span>
              <span>{g.gp}</span>
              <span>{Math.round(pickStat(g.wins, 0) || 0)}</span>
              <span>{Math.round(pickStat(g.losses, 0) || 0)}</span>
              <span>{(safe(g.sv_pct) * 100).toFixed(1)}</span>
              <span>{safe(g.gaa).toFixed(2)}</span>
              <span>{Math.round(pickStat(g.so, 0) || 0)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   TEAM TAB — MACRO ANALYTICS
========================================================= */

function TeamTab({ players, normalized }) {
  const team = normalized.team;
  const teams = team.leagueTeams || [];
  const metrics = [
    { key: "gf", label: "Goals For", betterHigh: true, value: team.gf, explanation: "Offensive conversion and scoring depth." },
    { key: "ga", label: "Goals Against", betterHigh: false, value: team.ga, explanation: "Team defense and goaltending suppression." },
    { key: "pp_pct", label: "Power Play %", betterHigh: true, value: team.pp_pct, explanation: "Special teams finishing efficiency." },
    { key: "pk_pct", label: "Penalty Kill %", betterHigh: true, value: team.pk_pct, explanation: "Defensive structure while shorthanded." },
    { key: "pdo", label: "PDO", betterHigh: true, value: team.pdo, explanation: "Combined team shooting and save rates." },
  ];
  const ranked = metrics.map((m) => {
    const sorted = [...teams].sort((a, b) => (m.betterHigh ? b[m.key] - a[m.key] : a[m.key] - b[m.key]));
    const rank = sorted.findIndex((x) => x.id === normalized.teamId) + 1;
    return { ...m, rank };
  });
  const strengths = [...ranked].sort((a, b) => a.rank - b.rank).slice(0, 3);
  const weaknesses = [...ranked].sort((a, b) => b.rank - a.rank).slice(0, 3);
  const drivers = [...players]
    .map((p) => ({ ...p, strengthDrive: (p.pts || 0) + (p.cf_pct || 0) * 30 + (p.blk || 0) * 0.2, weaknessRisk: (p.pim || 0) + (1 - (p.cf_pct || 0)) * 50 }))
    .sort((a, b) => b.strengthDrive - a.strengthDrive);
  const weakContrib = [...players]
    .map((p) => ({ ...p, weaknessRisk: (p.pim || 0) + (1 - (p.cf_pct || 0)) * 50 }))
    .sort((a, b) => b.weaknessRisk - a.weaknessRisk);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Team Analytics</h2>
        <p>Macro-level performance + identity.</p>
      </div>

      <div className="stats-dual-grid">
        <div className="stats-section">
          <h3>Strengths (Top 3 vs League)</h3>
          <div className="stats-notes">
            {strengths.map((s) => (
              <div key={s.key} className="stats-note">
                <strong>{s.label}</strong> · Rank {s.rank}/{teams.length}
                <div>{s.explanation}</div>
              </div>
            ))}
          </div>
          <h3>Player Drivers</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header"><span>Player</span><span>Drive Score</span></div>
            <div className="stats-table-body">
              {drivers.slice(0, 5).map((p) => (
                <div key={`drv-${p.player_id}`} className="stats-table-row"><span>{p.name}</span><span>{p.strengthDrive.toFixed(2)}</span></div>
              ))}
            </div>
          </div>
        </div>
        <div className="stats-section">
          <h3>Weaknesses (Bottom 3 vs League)</h3>
          <div className="stats-notes">
            {weaknesses.map((w) => (
              <div key={w.key} className="stats-note">
                <strong>{w.label}</strong> · Rank {w.rank}/{teams.length}
                <div>{w.explanation}</div>
              </div>
            ))}
          </div>
          <h3>Weakness Contributors</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header"><span>Player</span><span>Risk Score</span></div>
            <div className="stats-table-body">
              {weakContrib.slice(0, 5).map((p) => (
                <div key={`wrk-${p.player_id}`} className="stats-table-row"><span>{p.name}</span><span>{p.weaknessRisk.toFixed(2)}</span></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


/* =========================================================
   LEAGUE LEADERS TAB
========================================================= */

function LeadersTab({ leaders, players }) {
  const [sortKey, setSortKey] = useState("pts");
  const [desc, setDesc] = useState(true);
  const sortedLeague = useMemo(() => {
    return [...leaders].sort((a, b) => {
      const A = pickStat(a[sortKey], a.cf_pct, a.xgf_pct, 0) || 0;
      const B = pickStat(b[sortKey], b.cf_pct, b.xgf_pct, 0) || 0;
      return desc ? B - A : A - B;
    });
  }, [leaders, sortKey, desc]);
  const toggle = (k) => {
    if (k === sortKey) setDesc((d) => !d);
    else {
      setSortKey(k);
      setDesc(true);
    }
  };
  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>League Leaders</h2>
        <p>Top players across the league.</p>
      </div>

      <div className="stats-dual-grid">
      <div className="stats-table-wrap" style={{ flex: "2 1 0%" }}>
        <div className="stats-table-header">
          <span onClick={() => toggle("name")}>Player</span>
          <span onClick={() => toggle("team_id")}>Team</span>
          <span onClick={() => toggle("gp")}>GP</span>
          <span onClick={() => toggle("g")}>G</span>
          <span onClick={() => toggle("a")}>A</span>
          <span onClick={() => toggle("pts")}>P</span>
          <span onClick={() => toggle("cf_pct")}>CF%</span>
          <span onClick={() => toggle("xgf_pct")}>xGF%</span>
        </div>

        <div className="stats-table-body">
          {sortedLeague.slice(0, 50).map((p, i) => (
            <div key={i} className="stats-table-row">
              <span>{p.name}</span>
              <span>{p.team_id}</span>
              <span>{p.gp}</span>
              <span>{Math.round(pickStat(p.g, 0) || 0)}</span>
              <span>{Math.round(pickStat(p.a, 0) || 0)}</span>
              <span className="stats-table-pts">{Math.round(pickStat(p.pts, 0) || 0)}</span>
              <span>{((pickStat(p.cf_pct, pct(p.cf, (p.cf || 0) + (p.ca || 0)), 0) || 0) * 100).toFixed(1)}</span>
              <span>{((pickStat(p.xgf_pct, pct(p.xgf, (p.xgf || 0) + (p.xga || 0)), 0) || 0) * 100).toFixed(1)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="stats-section" style={{ flex: "1 1 0%" }}>
        <h3>Top Impact Players (Your Team)</h3>

        <div className="stats-table-wrap">
          <div className="stats-table-header">
            <span>Player</span>
            <span>Impact</span>
          </div>

          <div className="stats-table-body">
            {[...players]
              .map((p) => ({
                ...p,
                impact: calculateImpact(p),
              }))
              .sort((a, b) => b.impact - a.impact)
              .slice(0, 10)
              .map((p) => (
                <div key={p.player_id} className="stats-table-row">
                  <span>{p.name}</span>
                  <span>{p.impact}</span>
                </div>
              ))}
          </div>
        </div>
      </div></div>
    </div>
  );
}/* =========================================================
   ADVANCED TAB — POSSESSION / xG / RATE STATS
========================================================= */

function AdvancedTab({ players }) {
  const enriched = useMemo(() => {
    return players.map((p) => {
      const gp = safe(p.gp);
      const toi = safe(p.toi || p.toi_total || 0);
      const g = Math.round(pickStat(p.g, 0) || 0);
      const a = Math.round(pickStat(p.a, 0) || 0);
      const pts = g + a;
      const sog = Math.round(pickStat(p.sog, 0) || 0);
      const hit = Math.round(pickStat(p.hit, 0) || 0);
      const blk = Math.round(pickStat(p.blk, 0) || 0);
      const pim = Math.round(clamp(pickStat(p.pim, 0) || 0, 0, 80));
      const fow = safe(p.fow);
      const fol = safe(p.fol);
      const cf = pickStat(p.cf, p.corsi_for, sog * 2.1, 1) || 1;
      const ca = pickStat(p.ca, p.corsi_against, cf * 0.97, 1) || 1;
      const ff = pickStat(p.ff, p.fenwick_for, cf, 1) || 1;
      const fa = pickStat(p.fa, p.fenwick_against, ca, 1) || 1;
      const xgf = pickStat(p.xgf, p.expected_goals_for, g * 0.72 + a * 0.24, 0.1) || 0.1;
      const xga = pickStat(p.xga, p.expected_goals_against, ca * 0.018, 0.1) || 0.1;

      return {
        ...p,
        gpg: perGame(g, gp).toFixed(2),
        apg: perGame(a, gp).toFixed(2),
        ppg: perGame(pts, gp).toFixed(2),
        sogpg: perGame(sog, gp).toFixed(2),
        hitpg: perGame(hit, gp).toFixed(2),
        blkpg: perGame(blk, gp).toFixed(2),
        ppg60: per60(pts, toi).toFixed(2),
        g60: per60(g, toi).toFixed(2),
        a60: per60(a, toi).toFixed(2),
        sog60: per60(sog, toi).toFixed(2),
        hit60: per60(hit, toi).toFixed(2),
        blk60: per60(blk, toi).toFixed(2),
        pim60: per60(pim, toi).toFixed(2),
        foPct: ((pct(fow, fow + fol) || 0) * 100).toFixed(1),
        cfPct: ((pct(cf, cf + ca) || 0) * 100).toFixed(1),
        ffPct: ((pct(ff, ff + fa) || 0) * 100).toFixed(1),
        xgfPct: ((pct(xgf, xgf + xga) || 0) * 100).toFixed(1),
        shPct: ((pct(g, sog) || 0) * 100).toFixed(1),
        finishing: (g - xgf).toFixed(2),
      };
    });
  }, [players]);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Advanced Analytics</h2>
        <p>Rate stats, possession, expected goals, efficiency, and finishing.</p>
      </div>

      <div className="stats-table-wrap">
        <div className="stats-table-header stats-table-header--ultra">
          <span>Player</span>
          <span>P/GP</span>
          <span>G/60</span>
          <span>A/60</span>
          <span>P/60</span>
          <span>SOG/60</span>
          <span>HIT/60</span>
          <span>BLK/60</span>
          <span>FO%</span>
          <span>CF%</span>
          <span>FF%</span>
          <span>xGF%</span>
          <span>SH%</span>
          <span>Finish</span>
        </div>

        <div className="stats-table-body">
          {enriched.map((p) => (
            <div key={p.player_id} className="stats-table-row stats-table-row--ultra">
              <span className="stats-table-name">{p.name}</span>
              <span>{p.ppg}</span>
              <span>{p.g60}</span>
              <span>{p.a60}</span>
              <span>{p.ppg60}</span>
              <span>{p.sog60}</span>
              <span>{p.hit60}</span>
              <span>{p.blk60}</span>
              <span>{p.foPct}</span>
              <span>{p.cfPct}</span>
              <span>{p.ffPct}</span>
              <span>{p.xgfPct}</span>
              <span>{p.shPct}</span>
              <span>{p.finishing}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   SPECIAL TEAMS TAB
========================================================= */

function SpecialTeamsTab({ players, normalized }) {
  const ppPlayers = [...players].sort((a, b) => safe(b.ppg) + safe(b.ppa) - (safe(a.ppg) + safe(a.ppa))).slice(0, 10);
  const pkPlayers = [...players].sort((a, b) => safe(b.sha) + safe(b.blk) - (safe(a.sha) + safe(a.blk))).slice(0, 10);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Special Teams</h2>
        <p>Power play and penalty kill impact.</p>
      </div>

      <div className="stats-overview-grid">
        <StatCard label="PP%" value={`${((normalized.team.pp_pct || 0) * 100).toFixed(1)}%`} sub="Team power play" />
        <StatCard label="PK%" value={`${((normalized.team.pk_pct || 0) * 100).toFixed(1)}%`} sub="Team penalty kill" />
        <StatCard label="PP Goals" value={pickStat(normalized.team.ppg, 0) || 0} sub="Total" />
        <StatCard label="PK Goals Against" value={pickStat(normalized.team.ppga, 0) || 0} sub="Allowed while shorthanded" />
      </div>

      <div className="stats-dual-grid">
        <div className="stats-section">
          <h3>Power Play Leaders</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>Player</span>
              <span>PPG</span>
              <span>PPA</span>
              <span>PP Pts</span>
            </div>
            <div className="stats-table-body">
              {ppPlayers.map((p) => (
                <div key={p.player_id} className="stats-table-row">
                  <span>{p.name}</span>
              <span>{Math.round(pickStat(p.ppg, 0) || 0)}</span>
              <span>{Math.round(pickStat(p.ppa, 0) || 0)}</span>
              <span className="stats-table-pts">{Math.round((pickStat(p.ppg, 0) || 0) + (pickStat(p.ppa, 0) || 0))}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3>Penalty Kill Core</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>Player</span>
              <span>SHA</span>
              <span>BLK</span>
              <span>PK Value</span>
            </div>
            <div className="stats-table-body">
              {pkPlayers.map((p) => (
                <div key={p.player_id} className="stats-table-row">
                  <span>{p.name}</span>
                  <span>{Math.round(pickStat(p.sha, 0) || 0)}</span>
                  <span>{Math.round(pickStat(p.blk, 0) || 0)}</span>
                  <span>{(safe(p.sha) * 2 + safe(p.blk) * 0.25).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   GAME LOGS TAB
========================================================= */

function GameLogsTab({ recent, calendar }) {
  const allGames = useMemo(() => {
    const calGames = calendar.flatMap((c) => c.games || []);
    return [...recent, ...calGames];
  }, [recent, calendar]);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Game Logs</h2>
        <p>Recent league results and historical day blocks.</p>
      </div>

      <div className="stats-cal-list">
        {calendar.map((d) => (
          <div key={`cal-${d.day}`} className="stats-cal-pill">
            Day {d.day} <span className="stats-cal-pill__n">{(d.games || []).length}</span>
          </div>
        ))}
      </div>
      <div className="stats-score-list">
        {allGames.length === 0 ? (
          <div className="stats-empty">No game logs available yet.</div>
        ) : (
          allGames.map((g, i) => (
            <article key={`${g.day}-${g.home_id}-${g.away_id}-${i}`} className="stats-score-card">
              <div className="stats-score-card__line">
                <span className="stats-score-card__team">{g.home_name}</span>
                <span className="stats-score-card__goals">{g.home_goals}</span>
                <span className="stats-score-card__dash">—</span>
                <span className="stats-score-card__goals">{g.away_goals}</span>
                <span className="stats-score-card__team stats-score-card__team--away">{g.away_name}</span>
              </div>
              <div className="stats-score-card__meta">
                Day {g.day} · {fmtScore(g)}
              </div>
              <div className="stats-score-card__meta">
                Shots {(pickStat(g.home_shots, 0) || 0)}-{(pickStat(g.away_shots, 0) || 0)} · Hits {(pickStat(g.home_hits, 0) || 0)}-{(pickStat(g.away_hits, 0) || 0)}
              </div>
              <div className="stats-score-card__meta">
                xG {(pickStat(g.home_xg, 0) || 0).toFixed(2)}-{(pickStat(g.away_xg, 0) || 0).toFixed(2)} · Poss {(pickStat(g.home_possession, 50) || 50).toFixed(1)}%-{(pickStat(g.away_possession, 50) || 50).toFixed(1)}%
              </div>
            </article>
          ))
        )}
      </div>
    </div>
  );
}

/* =========================================================
   TRENDS TAB
========================================================= */

function TrendsTab({ players, goalies, sc, recent, leaders }) {
  const hottestSkater = topBy(players, (p) => safe(p.pts));
  const volumeShooter = topBy(players, (p) => safe(p.sog));
  const bestFinisher = topBy(players, (p) => {
    const g = safe(p.g);
    const sog = safe(p.sog);
    return pct(g, sog);
  });
  const bestGoalie = topBy(goalies, (g) => safe(g.sv_pct));

  const last10 = recent.slice(-10);
  const prev10 = recent.slice(-20, -10);
  const avg = (arr, fn) => (arr.length ? arr.reduce((s, x) => s + fn(x), 0) / arr.length : 0);
  const teamShootingDelta = avg(last10, (g) => pct(pickStat(g.home_goals, 0) || 0, Math.max(1, pickStat(g.home_shots, 1) || 1))) - avg(prev10, (g) => pct(pickStat(g.home_goals, 0) || 0, Math.max(1, pickStat(g.home_shots, 1) || 1)));
  const teamPaceDelta = avg(last10, (g) => pickStat(g.home_goals, 0) || 0) - avg(prev10, (g) => pickStat(g.home_goals, 0) || 0);
  const leaguePaceDelta = avg(last10, (g) => ((pickStat(g.home_goals, 0) || 0) + (pickStat(g.away_goals, 0) || 0)) / 2) - avg(prev10, (g) => ((pickStat(g.home_goals, 0) || 0) + (pickStat(g.away_goals, 0) || 0)) / 2);
  const leagueShootDelta = avg(leaders.slice(0, 50), (p) => pct(pickStat(p.g, 0) || 0, Math.max(1, pickStat(p.sog, 1) || 1))) - avg(leaders.slice(50, 100), (p) => pct(pickStat(p.g, 0) || 0, Math.max(1, pickStat(p.sog, 1) || 1)));
  const teamGF = players.reduce((s, p) => s + (pickStat(p.g, 0) || 0), 0);
  const teamGA = recent.reduce((s, g) => s + (pickStat(g.away_goals, g.home_goals, 0) || 0), 0) || 1;
  const teamSF = players.reduce((s, p) => s + (pickStat(p.sog, 0) || 0), 0) || 1;
  const teamSA = recent.reduce((s, g) => s + (pickStat(g.away_shots, g.home_shots, 0) || 0), 0) || (teamGA * 9.2);
  const teamPDO = pct(teamGF, teamSF) + pct(teamSA - teamGA, teamSA);
  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Trends</h2>
        <p>Quick trend-style insights and hot/cold snapshot cards.</p>
      </div>

      <div className="stats-dual-grid">
        <div className="stats-section">
          <h3>Team Trends</h3>
          <div className="stats-notes">
            <div className="stats-note"><strong>{teamShootingDelta >= 0 ? "Positive" : "Negative"}</strong> Shooting % delta (10g): {(teamShootingDelta * 100).toFixed(2)}%</div>
            <div className="stats-note"><strong>{teamPaceDelta >= 0 ? "Positive" : "Negative"}</strong> Scoring pace delta (10g): {teamPaceDelta.toFixed(2)}</div>
          </div>
        </div>
        <div className="stats-section">
          <h3>League Trends</h3>
          <div className="stats-notes">
            <div className="stats-note"><strong>{leaguePaceDelta >= 0 ? "Positive" : "Negative"}</strong> League scoring pace delta: {leaguePaceDelta.toFixed(2)}</div>
            <div className="stats-note"><strong>{leagueShootDelta >= 0 ? "Positive" : "Negative"}</strong> League shooting delta: {(leagueShootDelta * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>
      <div className="stats-overview-grid">
        <StatCard
          label="Hottest Skater"
          value={hottestSkater?.name || "—"}
          sub={`${safe(hottestSkater?.pts)} points`}
        />
        <StatCard
          label="Volume Shooter"
          value={volumeShooter?.name || "—"}
          sub={`${safe(volumeShooter?.sog)} shots`}
        />
        <StatCard
          label="Best Finisher"
          value={bestFinisher?.name || "—"}
          sub={`${((pct(safe(bestFinisher?.g), safe(bestFinisher?.sog)) || 0) * 100).toFixed(1)} SH%`}
        />
        <StatCard
          label="Best Goalie"
          value={bestGoalie?.name || "—"}
          sub={`${((safe(bestGoalie?.sv_pct) || 0) * 100).toFixed(1)} SV%`}
        />
        <StatCard
          label="PDO"
          value={teamPDO.toFixed(3)}
          sub="Team finishing / save run"
        />
        <StatCard
          label="GF Trend"
          value={Math.round(teamGF)}
          sub="Current season total"
        />
      </div>

      <div className="stats-section">
        <h3>Trend Notes</h3>
        <div className="stats-notes">
          <div className="stats-note">
            <strong>Hot:</strong> Players with strong points, shot generation, and impact tend to rise here.
          </div>
          <div className="stats-note">
            <strong>Cold:</strong> Low output with high usage can be a red flag for demotion or line adjustment.
          </div>
          <div className="stats-note">
            <strong>Regression Risk:</strong> Very high shooting percentage with weak chance quality is worth monitoring.
          </div>
          <div className="stats-note">
            <strong>Buy Low:</strong> Strong CF% / xGF% with weak box score can indicate hidden value.
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   COMPARE TAB
========================================================= */

function CompareTab({ players }) {
  const [leftId, setLeftId] = useState(players[0]?.player_id || "");
  const [rightId, setRightId] = useState(players[1]?.player_id || "");

  const left = players.find((p) => String(p.player_id) === String(leftId)) || players[0];
  const right = players.find((p) => String(p.player_id) === String(rightId)) || players[1];

  const leftImpact = left ? calculateImpact(left) : "0.00";
  const rightImpact = right ? calculateImpact(right) : "0.00";

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Comparison</h2>
        <p>Compare two players side by side for roster decisions.</p>
      </div>

      <div className="stats-compare-selectors">
        <select value={leftId} onChange={(e) => setLeftId(e.target.value)}>
          {players.map((p) => (
            <option key={p.player_id} value={p.player_id}>
              {p.name}
            </option>
          ))}
        </select>

        <select value={rightId} onChange={(e) => setRightId(e.target.value)}>
          {players.map((p) => (
            <option key={p.player_id} value={p.player_id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="stats-compare-grid">
        <CompareCard player={left} impact={leftImpact} side="left" />
        <CompareCard player={right} impact={rightImpact} side="right" />
      </div>

      <div className="stats-table-wrap">
        <div className="stats-table-header stats-compare-bref">
          <span>{left?.name || "Player A"}</span>
          <span>STAT</span>
          <span>{right?.name || "Player B"}</span>
        </div>

        <div className="stats-table-body">
          <CompareRow label="GP" left={pickStat(left?.gp, 0) || 0} right={pickStat(right?.gp, 0) || 0} />
          <CompareRow label="G" left={pickStat(left?.g, 0) || 0} right={pickStat(right?.g, 0) || 0} />
          <CompareRow label="A" left={pickStat(left?.a, 0) || 0} right={pickStat(right?.a, 0) || 0} />
          <CompareRow label="P" left={pickStat(left?.pts, 0) || 0} right={pickStat(right?.pts, 0) || 0} />
          <CompareRow label="CF%" left={(pickStat(left?.cf_pct, pct(left?.cf || 0, (left?.cf || 0) + (left?.ca || 0)), 0) || 0) * 100} right={(pickStat(right?.cf_pct, pct(right?.cf || 0, (right?.cf || 0) + (right?.ca || 0)), 0) || 0) * 100} />
          <CompareRow label="xGF%" left={(pickStat(left?.xgf_pct, pct(left?.xgf || 0, (left?.xgf || 0) + (left?.xga || 0)), 0) || 0) * 100} right={(pickStat(right?.xgf_pct, pct(right?.xgf || 0, (right?.xgf || 0) + (right?.xga || 0)), 0) || 0) * 100} />
          <CompareRow label="PIM" left={pickStat(left?.pim, 0) || 0} right={pickStat(right?.pim, 0) || 0} lowerIsBetter />
          <CompareRow label="TOI" left={pickStat(left?.toi, 0) || 0} right={pickStat(right?.toi, 0) || 0} />
          <CompareRow label="Impact" left={Number(leftImpact)} right={Number(rightImpact)} />
        </div>
      </div>
    </div>
  );
}

function CompareCard({ player, impact, side }) {
  return (
    <div className={`stats-compare-card stats-compare-card--${side}`}>
      <div className="stats-compare-card__name">{player?.name || "—"}</div>
      <div className="stats-compare-card__meta">
        {player?.position || "—"} · GP {safe(player?.gp)}
      </div>
      <div className="stats-compare-card__impact">Impact {impact}</div>
    </div>
  );
}

function CompareRow({ label, left, right, lowerIsBetter = false }) {
  let edge = "Even";
  if (left !== right) {
    if (lowerIsBetter) edge = left < right ? "Left" : "Right";
    else edge = left > right ? "Left" : "Right";
  }

  const isCountStat = ["GP", "G", "A", "P", "PIM", "TOI"].includes(label);
  const leftText =
    typeof left === "number"
      ? (isCountStat ? String(Math.round(left)) : (left.toFixed?.(2) ?? left))
      : left;
  const rightText =
    typeof right === "number"
      ? (isCountStat ? String(Math.round(right)) : (right.toFixed?.(2) ?? right))
      : right;

  return (
    <div className="stats-table-row stats-compare-bref">
      <span className={edge === "Left" ? "stats-compare-better" : edge === "Right" ? "stats-compare-worse" : ""}>{leftText}</span>
      <span>{label}</span>
      <span className={edge === "Right" ? "stats-compare-better" : edge === "Left" ? "stats-compare-worse" : ""}>{rightText}</span>
    </div>
  );
}

/* =========================================================
   IMPACT TAB
========================================================= */

function ImpactTab({ players }) {
  const ranked = useMemo(() => {
    return [...players]
      .map((p) => {
        const toi = safe(p.toi || p.toi_total || 0);
        const cf = safe(p.cf || 0);
        const ca = safe(p.ca || 1);
        const rawG = Math.round(pickStat(p.g, 0) || 0);
        const rawA = Math.round(pickStat(p.a, 0) || 0);
        const rawPts = rawG + rawA;
        const xgf = pickStat(p.xgf, p.expected_goals_for, rawG * 0.72 + rawA * 0.24, 0.1) || 0.1;
        const xga = pickStat(p.xga, p.expected_goals_against, safe(p.ca) * 0.018, 0.1) || 0.1;
        const gfOn = pickStat(p.gf_on, p.on_ice_gf, rawG + rawA * 0.5, 1) || 1;
        const gaOn = pickStat(p.ga_on, p.on_ice_ga, xga * 0.9, 1) || 1;

        const scoringImpact = per60(rawPts, toi);
        const playmakingImpact = per60(rawA, toi);
        const possessionImpact = pct(cf, cf + ca) * 100;
        const chanceImpact = pct(xgf, xgf + xga) * 100;
        const usageImpact = toi;
        const gfPctImpact = pct(gfOn, gfOn + gaOn) * 100;
        const netImpact =
          0.34 * scoringImpact +
          0.22 * possessionImpact +
          0.22 * chanceImpact +
          0.12 * (usageImpact / 20) +
          0.10 * gfPctImpact;

        return {
          ...p,
          scoringImpact: scoringImpact.toFixed(2),
          playmakingImpact: playmakingImpact.toFixed(2),
          possessionImpact: possessionImpact.toFixed(2),
          chanceImpact: chanceImpact.toFixed(2),
          usageImpact: usageImpact.toFixed(2),
          netImpact: netImpact.toFixed(2),
          badge: getImpactBadge({
            pts: rawPts,
            sog: Math.round(pickStat(p.sog, 0) || 0),
            blk: Math.round(pickStat(p.blk, 0) || 0),
            hit: Math.round(pickStat(p.hit, 0) || 0),
            pim: Math.round(pickStat(p.pim, 0) || 0),
            netImpact,
          }),
        };
      })
      .sort((a, b) => Number(b.netImpact) - Number(a.netImpact));
  }, [players]);

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Impact Rankings</h2>
        <p>Most impactful, least impactful, role-breaking, and hidden value indicators.</p>
      </div>

      <div className="stats-table-wrap">
        <div className="stats-table-header stats-table-header--ultra">
          <span>Player</span>
          <span>Scoring</span>
          <span>Playmaking</span>
          <span>Possession</span>
          <span>Chance</span>
          <span>Usage</span>
          <span>Net Impact</span>
          <span>Badge</span>
        </div>

        <div className="stats-table-body">
          {ranked.map((p) => (
            <div key={p.player_id} className="stats-table-row stats-table-row--ultra">
              <span className="stats-table-name">{p.name}</span>
              <span>{p.scoringImpact}</span>
              <span>{p.playmakingImpact}</span>
              <span>{p.possessionImpact}</span>
              <span>{p.chanceImpact}</span>
              <span>{p.usageImpact}</span>
              <span className="stats-impact">{p.netImpact}</span>
              <span>{p.badge}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function getImpactBadge({ pts, sog, blk, hit, pim, netImpact }) {
  if (netImpact >= 12) return "Elite Driver";
  if (pts >= 50) return "Top Producer";
  if (sog >= 200) return "Volume Shooter";
  if (blk >= 100) return "Shot Suppressor";
  if (hit >= 150) return "Physical Force";
  if (pim >= 100) return "Discipline Risk";
  if (netImpact <= 4) return "Replacement Level";
  return "Core Contributor";
}/* =========================================================
   AWARDS WATCH TAB
========================================================= */

function AwardsWatchTab({ players, goalies, leaders, normalized }) {
  const artRoss = [...leaders].sort((a, b) => safe(b.pts) - safe(a.pts)).slice(0, 10);
  const rocket = [...leaders].sort((a, b) => safe(b.g) - safe(a.g)).slice(0, 10);
  const teamMvp = [...players]
    .map((p) => ({ ...p, impact: Number(calculateImpact(p)) }))
    .sort((a, b) => b.impact - a.impact)
    .slice(0, 5);
  const vezina = [...goalies].sort((a, b) => ((safe(b.sv_pct) - safe(a.sv_pct)) + (safe(a.gaa) - safe(b.gaa)))).slice(0, 5);
  const teams = normalized.team.leagueTeams || [];
  const presidents = [...teams].sort((a, b) => safe(b.gf) - safe(a.gf))[0];
  const jennings = [...teams].sort((a, b) => safe(a.ga) - safe(b.ga))[0];
  const hart = [...players].sort((a, b) => Number(calculateImpact(b)) - Number(calculateImpact(a)))[0];
  const norris = [...players]
    .filter((p) => p.position === "D")
    .sort(
      (a, b) =>
        (safe(b.cf_pct) * 100 + perGame(safe(b.toi), safe(b.gp)) + perGame(safe(b.pts), safe(b.gp))) -
        (safe(a.cf_pct) * 100 + perGame(safe(a.toi), safe(a.gp)) + perGame(safe(a.pts), safe(a.gp)))
    )[0];
  const selke = [...players]
    .filter((p) => p.position !== "G")
    .sort(
      (a, b) =>
        (safe(b.cf_pct) * 100 + safe(b.blk) + safe(b.sha) * 2) -
        (safe(a.cf_pct) * 100 + safe(a.blk) + safe(a.sha) * 2)
    )[0];
  const calder = [...players].filter((p) => p.rookie).sort((a, b) => (safe(b.pts) + Number(calculateImpact(b))) - (safe(a.pts) + Number(calculateImpact(a))))[0];
  const byng = [...players].sort((a, b) => (safe(b.pts) / Math.max(1, safe(b.pim) + 1)) - (safe(a.pts) / Math.max(1, safe(a.pim) + 1)))[0];
  const masterton = [...players].sort((a, b) => safe(b.age) - safe(a.age))[0];
  const leadership = [...players].sort((a, b) => ((safe(b.toi) + (b.captain ? 120 : 0)) - (safe(a.toi) + (a.captain ? 120 : 0))))[0];

  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>Awards Watch</h2>
        <p>League races and team-level internal award tracking.</p>
      </div>

      <div className="stats-dual-grid">
        <div className="stats-section">
          <h3>Art Ross Watch</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>#</span>
              <span>Player</span>
              <span>Team</span>
              <span>P</span>
            </div>
            <div className="stats-table-body">
              {artRoss.map((p, i) => (
                <div key={`${p.player_id || p.name}-art`} className="stats-table-row">
                  <span>{i + 1}</span>
                  <span>{p.name}</span>
                  <span>{p.team_id}</span>
                  <span className="stats-table-pts">{Math.round(pickStat(p.pts, 0) || 0)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3>Rocket Watch</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>#</span>
              <span>Player</span>
              <span>Team</span>
              <span>G</span>
            </div>
            <div className="stats-table-body">
              {rocket.map((p, i) => (
                <div key={`${p.player_id || p.name}-rocket`} className="stats-table-row">
                  <span>{i + 1}</span>
                  <span>{p.name}</span>
                  <span>{p.team_id}</span>
                  <span>{Math.round(pickStat(p.g, 0) || 0)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="stats-dual-grid">
        <div className="stats-section">
          <h3>Team MVP Candidates</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>Player</span>
              <span>Pos</span>
              <span>Pts</span>
              <span>Impact</span>
            </div>
            <div className="stats-table-body">
              {teamMvp.map((p) => (
                <div key={`${p.player_id}-mvp`} className="stats-table-row">
                  <span>{p.name}</span>
                  <span>{p.position}</span>
                  <span>{Math.round(pickStat(p.pts, 0) || 0)}</span>
                  <span className="stats-impact">{p.impact.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3>Vezina Track</h3>
          <div className="stats-table-wrap">
            <div className="stats-table-header">
              <span>Goalie</span>
              <span>GP</span>
              <span>SV%</span>
              <span>GAA</span>
            </div>
            <div className="stats-table-body">
              {vezina.map((g) => (
                <div key={`${g.player_id}-vez`} className="stats-table-row">
                  <span>{g.name}</span>
                  <span>{safe(g.gp)}</span>
                  <span>{((safe(g.sv_pct) || 0) * 100).toFixed(1)}</span>
                  <span>{safe(g.gaa).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="stats-section">
        <h3>Full NHL Awards (Formula Driven)</h3>
        <div className="stats-table-wrap">
          <div className="stats-table-header"><span>Award</span><span>Leader</span><span>Formula</span></div>
          <div className="stats-table-body">
            {[
              ["Stanley Cup", presidents?.id || "N/A", "Best playoff proxy by top team profile"],
              ["Presidents Trophy", presidents?.id || "N/A", "Best regular season profile"],
              ["Art Ross", artRoss[0]?.name || "N/A", "Most points"],
              ["Rocket Richard", rocket[0]?.name || "N/A", "Most goals"],
              ["Hart", hart?.name || "N/A", "Highest impact score"],
              ["Norris", norris?.name || "N/A", "Best D: CF% + TOI + PTS"],
              ["Vezina", vezina[0]?.name || "N/A", "Best goalie: SV% + GAA"],
              ["Jennings", jennings?.id || "N/A", "Lowest team GA"],
              ["Selke", selke?.name || "N/A", "Best defensive forward: CF% + BLK + PK"],
              ["Calder", calder?.name || "N/A", "Best rookie"],
              ["Lady Byng", byng?.name || "N/A", "Low PIM + high production"],
              ["Jack Adams", presidents?.id || "N/A", "Best team improvement proxy"],
              ["GM Award", presidents?.id || "N/A", "Roster improvement score"],
              ["Masterton", masterton?.name || "N/A", "Oldest player"],
              ["Leadership", leadership?.name || "N/A", "Highest TOI + captain weight"],
            ].map(([award, who, formula]) => (
              <div key={award} className="stats-table-row"><span>{award}</span><span>{who}</span><span>{formula}</span></div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   FORMULAS TAB — 100 HOCKEY STATS + FORMULAS
========================================================= */

const FORMULA_SECTIONS = [
  {
    title: "Basic Counting Stats",
    color: "green",
    items: [
      ["1. Goals (G)", "Total goals scored", "G = total goals scored"],
      ["2. Assists (A)", "Total assists", "A = total assists"],
      ["3. Points (P)", "Goals + Assists", "P = G + A"],
      ["4. Games Played (GP)", "Total games played", "GP = total games played"],
      ["5. Shots on Goal (SOG)", "Total shots on net", "SOG = total shots on goal"],
      ["6. Hits (HIT)", "Total hits delivered", "HIT = total body checks delivered"],
      ["7. Blocked Shots (BLK)", "Total shots blocked", "BLK = total opponent shot attempts blocked"],
      ["8. Penalty Minutes (PIM)", "Penalty minutes taken", "PIM = sum of all penalty minutes taken"],
      ["9. Power Play Goals (PPG)", "Goals scored on power play", "PPG = total goals scored on power play"],
      ["10. Power Play Assists (PPA)", "Assists on power play goals", "PPA = assists on power play goals"],
      ["11. Short-Handed Goals (SHG)", "Goals scored while shorthanded", "SHG = goals scored while short-handed"],
      ["12. Short-Handed Assists (SHA)", "Assists while shorthanded", "SHA = assists while short-handed"],
      ["13. Game Winning Goals (GWG)", "Goals that became the winning margin", "GWG = goals that gave final winning lead"],
      ["14. Overtime Goals (OTG)", "Goals in overtime", "OTG = goals scored in overtime"],
      ["15. Faceoff Wins (FOW)", "Total faceoffs won", "FOW = total faceoffs won"],
      ["16. Faceoff Losses (FOL)", "Total faceoffs lost", "FOL = total faceoffs lost"],
      ["17. Takeaways (TAK)", "Credited puck takeaways", "TAK = total credited puck takeaways"],
      ["18. Giveaways (GIV)", "Credited puck giveaways", "GIV = total credited puck giveaways"],
      ["19. Plus/Minus (+/-)", "Even-strength goals for on ice minus against", "+/- = on-ice EV goals for − on-ice EV goals against"],
      ["20. Time on Ice (TOI)", "Total minutes played", "TOI = total minutes played"],
    ],
  },
  {
    title: "Rate Stats",
    color: "blue",
    items: [
      ["21. Goals Per Game", "Goals each game", "G/GP = G / GP"],
      ["22. Assists Per Game", "Assists each game", "A/GP = A / GP"],
      ["23. Points Per Game", "Points each game", "P/GP = P / GP"],
      ["24. Shots Per Game", "Shots each game", "SOG/GP = SOG / GP"],
      ["25. Hits Per Game", "Hits each game", "HIT/GP = HIT / GP"],
      ["26. Blocks Per Game", "Blocks each game", "BLK/GP = BLK / GP"],
      ["27. PIM Per Game", "Penalty minutes each game", "PIM/GP = PIM / GP"],
      ["28. Faceoff Win %", "Win rate on draws", "FO% = FOW / (FOW + FOL)"],
      ["29. Shooting %", "Goals per shot", "SH% = G / SOG"],
      ["30. TOI Per Game", "Minutes played each game", "TOI/GP = TOI / GP"],
      ["31. Goals Per 60", "Goals scaled by ice time", "G/60 = (G / TOI) × 60"],
      ["32. Assists Per 60", "Assists scaled by ice time", "A/60 = (A / TOI) × 60"],
      ["33. Points Per 60", "Points scaled by ice time", "P/60 = (P / TOI) × 60"],
      ["34. Shots Per 60", "Shots scaled by ice time", "SOG/60 = (SOG / TOI) × 60"],
      ["35. Hits Per 60", "Hits scaled by ice time", "HIT/60 = (HIT / TOI) × 60"],
      ["36. Blocks Per 60", "Blocks scaled by ice time", "BLK/60 = (BLK / TOI) × 60"],
      ["37. Takeaways Per 60", "Takeaways scaled by ice time", "TAK/60 = (TAK / TOI) × 60"],
      ["38. Giveaways Per 60", "Giveaways scaled by ice time", "GIV/60 = (GIV / TOI) × 60"],
      ["39. PIM Per 60", "Penalty minutes scaled by ice time", "PIM/60 = (PIM / TOI) × 60"],
      ["40. Even Strength Points Per 60", "EV scoring scaled by EV TOI", "ESP/60 = (ESP / TOI_even) × 60"],
    ],
  },
  {
    title: "Team Stats",
    color: "yellow",
    items: [
      ["41. Goals For (GF)", "Total goals scored by team", "GF = total goals scored by team"],
      ["42. Goals Against (GA)", "Total goals allowed", "GA = total goals allowed"],
      ["43. Goal Differential (GD)", "GF minus GA", "GD = GF − GA"],
      ["44. Win %", "Wins divided by games", "Win% = Wins / GP"],
      ["45. Points %", "Share of possible standings points won", "Points% = Points Earned / Max Possible Points"],
      ["46. Shots For (SF)", "Total shots taken", "SF = total shots taken"],
      ["47. Shots Against (SA)", "Total shots allowed", "SA = total shots allowed"],
      ["48. Shot Differential", "Shots for minus shots against", "Shot Diff = SF − SA"],
      ["49. Power Play %", "PP conversion rate", "PP% = PPG / PPO"],
      ["50. Penalty Kill %", "PK prevention rate", "PK% = 1 − (PPGA / Opp PPO)"],
      ["51. Team Save %", "All goalie saves divided by shots against", "Team SV% = (SA − GA) / SA"],
      ["52. PDO", "Shooting + save percentage", "PDO = Team SH% + Team SV%"],
      ["53. Team Faceoff %", "Draw win rate", "FO% = Team FOW / (Team FOW + Team FOL)"],
      ["54. Team Goals Per Game", "Goals each game", "GF/GP = GF / GP"],
      ["55. Team Goals Against Per Game", "Goals allowed each game", "GA/GP = GA / GP"],
    ],
  },
  {
    title: "Advanced Possession",
    color: "red",
    items: [
      ["56. Corsi For (CF)", "All shot attempts for", "CF = shots on goal + missed shots + blocked shot attempts"],
      ["57. Corsi Against (CA)", "All shot attempts against", "CA = opponent shot attempts against"],
      ["58. Corsi %", "Share of shot attempts", "CF% = CF / (CF + CA)"],
      ["59. Fenwick For (FF)", "Unblocked shot attempts for", "FF = shots on goal + missed shots"],
      ["60. Fenwick Against (FA)", "Unblocked shot attempts against", "FA = opponent unblocked shot attempts"],
      ["61. Fenwick %", "Share of unblocked attempts", "FF% = FF / (FF + FA)"],
      ["62. Relative Corsi", "On-ice Corsi compared with team off-ice", "Rel CF% = on-ice CF% − off-ice team CF%"],
      ["63. Shot Attempts Differential", "Net shot attempt margin", "CF Diff = CF − CA"],
      ["64. Shots For %", "Share of shots on goal", "SF% = SF / (SF + SA)"],
      ["65. Zone Start %", "O-zone starts share", "ZS% = O-zone starts / (O-zone starts + D-zone starts)"],
      ["66. Offensive Zone Start Ratio", "O-zone starts per total shifts", "OZS Ratio = O-zone starts / total shifts"],
      ["67. Defensive Zone Start Ratio", "D-zone starts per total shifts", "DZS Ratio = D-zone starts / total shifts"],
      ["68. Neutral Zone Start %", "Neutral zone start share", "NZS% = NZ starts / total starts"],
      ["69. Possession Time %", "Puck possession share", "Possession Time% = team puck possession time / total game time"],
      ["70. Entries With Control %", "Controlled entry rate", "Controlled Entry% = controlled entries / total zone entries"],
    ],
  },
  {
    title: "Expected Goals & Shot Quality",
    color: "purple",
    items: [
      ["71. Expected Goals (xG)", "Probability-weighted shot value", "xG = sum of shot probabilities based on location, angle, shot type, pre-shot movement, rush/rebound status"],
      ["72. Expected Goals For (xGF)", "Expected goals created", "xGF = sum of xG for team/player on offense"],
      ["73. Expected Goals Against (xGA)", "Expected goals allowed", "xGA = sum of xG against on defense"],
      ["74. xG Differential", "xGF minus xGA", "xG Diff = xGF − xGA"],
      ["75. xG %", "Share of expected goals", "xG% = xGF / (xGF + xGA)"],
      ["76. Goals Above Expected", "Actual goals minus expected goals", "GAx = Actual Goals − xG"],
      ["77. Shots from Slot %", "Share of shots from slot", "Slot Shot% = slot shots / total shots"],
      ["78. High Danger Chances For (HDCF)", "Count of high-danger attempts", "HDCF = count of high-danger shot attempts"],
      ["79. High Danger Chance %", "Share of high-danger chances", "HDCF% = HDCF / (HDCF + HDCA)"],
      ["80. Medium Danger Chance %", "Share of medium-danger chances", "MDCF% = MDCF / (MDCF + MDCA)"],
      ["81. Low Danger Chance %", "Share of low-danger chances", "LDCF% = LDCF / (LDCF + LDCA)"],
      ["82. Rebound Shots %", "Share of rebound shots", "Rebound Shot% = rebound shots / total shots"],
      ["83. Rush Chances %", "Share of rush shots", "Rush Chance% = rush shots / total shots"],
      ["84. Finishing", "Goals minus xG", "Finishing = Goals − xG"],
      ["85. Expected Shooting %", "Expected goals per shot", "xSH% = xG / SOG"],
    ],
  },
  {
    title: "Goalie Stats",
    color: "brown",
    items: [
      ["86. Save %", "Saves divided by shots against", "SV% = Saves / Shots Against"],
      ["87. Goals Against Average", "Goals allowed per 60", "GAA = (GA × 60) / TOI"],
      ["88. Shutouts", "Games with zero goals allowed", "SO = number of games with 0 goals allowed"],
      ["89. Goals Saved Above Expected", "Expected goals against minus actual goals against", "GSAx = xGA − GA"],
      ["90. High Danger Save %", "High-danger saves divided by HD shots against", "HDSV% = HD Saves / HD Shots Against"],
      ["91. Medium Danger Save %", "Medium-danger saves divided by MD shots against", "MDSV% = MD Saves / MD Shots Against"],
      ["92. Low Danger Save %", "Low-danger saves divided by LD shots against", "LDSV% = LD Saves / LD Shots Against"],
      ["93. Rebound Control %", "Saves without rebound divided by total saves", "Rebound Control% = no-rebound saves / total saves"],
      ["94. Save % vs Rush Chances", "Rush saves divided by rush shots faced", "Rush SV% = rush saves / rush shots faced"],
      ["95. Quality Start %", "Quality starts divided by starts", "QS% = Quality Starts / Starts"],
    ],
  },
  {
    title: "Hybrid / Impact Metrics",
    color: "black",
    items: [
      ["96. Wins Above Replacement (WAR)", "Wins contributed above replacement", "WAR = total wins contributed above replacement-level player"],
      ["97. Goals Above Replacement (GAR)", "Goal value above replacement", "GAR = offensive GAR + defensive GAR + special teams GAR"],
      ["98. Points Above Replacement (PAR)", "Points above replacement baseline", "PAR = player points − replacement-level points over same usage"],
      ["99. On-Ice Goals For %", "Share of on-ice goals for", "GF% = GF_on / (GF_on + GA_on)"],
      ["100. Impact Score", "Custom franchise composite metric", "Impact Score = (Offense Weight × normalized G/60) + (Defense Weight × normalized CF%) + (Usage Weight × normalized TOI share) + (Chance Weight × normalized xGF%) + (Results Weight × normalized GF%)"],
    ],
  },
];

function FormulasTab() {
  return (
    <div className="stats-tab">
      <div className="stats-section__header">
        <h2>100 Hockey Stats + Formulas</h2>
        <p>Built directly from your design brief. This is the internal stat library for Stats Central.</p>
      </div>

      <div className="formula-sections">
        {FORMULA_SECTIONS.map((section) => (
          <div key={section.title} className={`formula-section formula-section--${section.color}`}>
            <div className="formula-section__title">{section.title}</div>
            <div className="formula-list">
              {section.items.map(([name, desc, formula]) => (
                <div key={name} className="formula-card">
                  <div className="formula-card__name">{name}</div>
                  <div className="formula-card__desc">{desc}</div>
                  <div className="formula-card__formula">{formula}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   FILE END HELPERS / DEFAULT EXPORT OPTIONAL
========================================================= */

// If your project expects default export instead, uncomment this:
// export default StatsCentralScreen;