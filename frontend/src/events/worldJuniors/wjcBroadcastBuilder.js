import { WJC_PHRASE_BANK, phraseApplies } from "./wjcPhraseBank";

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function pick(arr, fallback = null) {
  const list = asArray(arr).filter(Boolean);
  if (!list.length) return fallback;
  return list[Math.floor(Math.random() * list.length)];
}

function fmtRecord(w, l) {
  return `${w ?? 0}-${l ?? 0}`;
}

function gameCode(g, side) {
  return String(g?.[side] || g?.[`${side}_label`] || "?").slice(0, 3).toUpperCase();
}

function formatScoreLine(g) {
  const home = gameCode(g, "home");
  const away = gameCode(g, "away");
  const hg = g?.home_goals;
  const ag = g?.away_goals;
  if (hg != null && ag != null) return `${home} ${hg}, ${away} ${ag}`;
  return `${home} vs ${away}`;
}

function standingByCode(standings) {
  const map = {};
  asArray(standings).forEach((row) => {
    map[String(row.code)] = row;
  });
  return map;
}

function labelForCode(code, standings, countries) {
  const st = standingByCode(standings)[code];
  if (st?.label) return st.label;
  const c = asArray(countries).find((x) => String(x.code) === String(code));
  return c?.label || code;
}

function recordForCode(code, standings) {
  const st = standingByCode(standings)[code];
  if (!st) return "0-0";
  return fmtRecord(st.w, st.l);
}

function interpolate(template, vars) {
  return String(template || "").replace(/\{(\w+)\}/g, (_, key) => {
    const val = vars[key];
    return val != null && val !== "" ? String(val) : "—";
  });
}

function buildTemplateVars(payload) {
  const standings = asArray(payload?.standings);
  const games = asArray(payload?.all_games).length
    ? asArray(payload?.all_games)
    : asArray(payload?.round_robin_games);
  const playerStats = asArray(payload?.player_stats);
  const prospects = asArray(payload?.tournament_prospects);
  const countries = asArray(payload?.countries);
  const stMap = standingByCode(standings);
  const leader = standings[0] || {};
  const featuredGame = pick(games);
  const standout = pick(playerStats.filter((p) => int(p.pts) > 0)) || playerStats[0] || {};
  const ptsLeader = playerStats[0] || {};
  const goalLeader =
    [...playerStats].sort((a, b) => int(b.g) - int(a.g))[0] || ptsLeader;
  const userProspect =
    pick(asArray(payload?.user_prospects).filter((p) => p.made_wjc_team)) ||
    pick(asArray(payload?.user_prospects));

  let winnerCode = "";
  let loserCode = "";
  if (featuredGame) {
    const hg = int(featuredGame.home_goals);
    const ag = int(featuredGame.away_goals);
    winnerCode = hg > ag ? gameCode(featuredGame, "home") : gameCode(featuredGame, "away");
    loserCode = hg > ag ? gameCode(featuredGame, "away") : gameCode(featuredGame, "home");
  }

  const risers = [...prospects].sort((a, b) => int(b.stock_delta) - int(a.stock_delta));
  const fallers = [...prospects].sort((a, b) => int(a.stock_delta) - int(b.stock_delta));
  const riser = risers[0] || {};
  const faller = fallers[0] || {};

  const surprise = pick(
    standings.filter((s) => int(s.place) >= 4 && int(s.w) >= 2)
  ) || pick(standings.slice(3));

  const userStat = userProspect
    ? playerStats.find((p) => String(p.player_id) === String(userProspect.player_id))
  : null;

  const medalLabels = payload?.medal_labels || {};
  const po = payload?.playoffs || {};
  const qf = asArray(po.quarterfinals);
  const sf = asArray(po.semifinals);
  const bronze = po.bronze;
  const gold = po.gold;

  const todayGames = getTodayGames(payload);

  return {
    day_label: payload?.wjc_day
      ? `Day ${payload.wjc_day} of ${payload.wjc_days_total || 11}`
      : "Tournament standby",
    top_story: leader.code
      ? `${leader.label || leader.code} leads at ${leader.pts} points`
      : "Group stage action continues",
    wjc_day: payload?.wjc_day || "—",
    games_today_count: todayGames.length,
    day_highlight: todayGames.length
      ? todayGames.map(formatScoreLine).join(" · ")
      : "Awaiting next game batch",
    score_line: featuredGame ? formatScoreLine(featuredGame) : "No final yet",
    home_code: featuredGame ? gameCode(featuredGame, "home") : "—",
    away_code: featuredGame ? gameCode(featuredGame, "away") : "—",
    home_goals: featuredGame?.home_goals ?? "—",
    away_goals: featuredGame?.away_goals ?? "—",
    winner_label: winnerCode ? labelForCode(winnerCode, standings, countries) : "—",
    loser_label: loserCode ? labelForCode(loserCode, standings, countries) : "—",
    winner_code: winnerCode || "—",
    loser_code: loserCode || "—",
    winner_record: winnerCode ? recordForCode(winnerCode, standings) : "—",
    loser_record: loserCode ? recordForCode(loserCode, standings) : "—",
    upset_line: featuredGame
      ? `${labelForCode(winnerCode, standings, countries)} beat ${labelForCode(loserCode, standings, countries)}`
      : "A major result just hit the board",
    underdog_code: winnerCode || "—",
    underdog_label: winnerCode ? labelForCode(winnerCode, standings, countries) : "—",
    underdog_record: winnerCode ? recordForCode(winnerCode, standings) : "—",
    favorite_code: loserCode || "—",
    favorite_label: loserCode ? labelForCode(loserCode, standings, countries) : "—",
    favorite_record: loserCode ? recordForCode(loserCode, standings) : "—",
    leader_code: leader.code || "—",
    leader_label: leader.label || leader.code || "—",
    leader_pts: leader.pts ?? "—",
    leader_w: leader.w ?? 0,
    leader_l: leader.l ?? 0,
    leader_diff: int(leader.gf) - int(leader.ga),
    standout_name: standout.name || "—",
    standout_country: standout.wjc_country || "—",
    standout_pts: standout.pts ?? 0,
    standout_g: standout.g ?? 0,
    standout_a: standout.a ?? 0,
    standout_gp: standout.gp ?? 0,
    standout_pm: standout.plus_minus ?? 0,
    standout_sog: standout.sog ?? 0,
    stock_before: findProspectStock(prospects, standout, "before"),
    stock_after: findProspectStock(prospects, standout, "after"),
    pts_leader_name: ptsLeader.name || "—",
    pts_leader_pts: ptsLeader.pts ?? 0,
    pts_leader_g: ptsLeader.g ?? 0,
    pts_leader_a: ptsLeader.a ?? 0,
    pts_leader_country: ptsLeader.wjc_country || "—",
    goal_leader_name: goalLeader.name || "—",
    goal_leader_g: goalLeader.g ?? 0,
    goal_leader_gp: goalLeader.gp ?? 0,
    goal_leader_country: goalLeader.wjc_country || "—",
    user_name: userProspect?.name || "your prospect",
    user_country_label: userProspect?.wjc_country_label || userProspect?.wjc_country || "—",
    user_pts: userStat?.pts ?? 0,
    user_gp: userStat?.gp ?? 0,
    user_stock_note: userProspect
      ? deriveUserStockNote(userProspect, prospects)
      : "No active prospect in the field",
    riser_name: riser.name || "—",
    riser_country: riser.wjc_country || "—",
    riser_before: riser.stock_rank_before ?? "—",
    riser_after: riser.stock_rank_after ?? "—",
    riser_pts: riser.tournament_pts ?? 0,
    faller_name: faller.name || "—",
    faller_country: faller.wjc_country || "—",
    faller_before: faller.stock_rank_before ?? "—",
    faller_after: faller.stock_rank_after ?? "—",
    team_code: leader.code || "—",
    team_gf: leader.gf ?? 0,
    team_ga: leader.ga ?? 0,
    team_gp: leader.gp ?? 0,
    team_w: leader.w ?? 0,
    team_l: leader.l ?? 0,
    playoff_note: qf.length ? `${qf.length} quarterfinals decided` : "Group stage wrapping up",
    gold_label: medalLabels.gold || "—",
    silver_label: medalLabels.silver || "—",
    bronze_label: medalLabels.bronze || "—",
    surprise_label: surprise?.label || surprise?.code || "—",
    surprise_w: surprise?.w ?? 0,
    surprise_l: surprise?.l ?? 0,
    analytics_player: standout.name || "—",
    analytics_pts: standout.pts ?? 0,
    analytics_sog: standout.sog ?? 0,
    analytics_gp: standout.gp ?? 0,
    analytics_pm: standout.plus_minus ?? 0,
    analytics_trend: int(standout.pts) >= 4 ? "up sharply" : int(standout.pts) >= 2 ? "up" : "flat",
    _featuredGame: featuredGame,
    _qf: qf,
    _sf: sf,
    _bronze: bronze,
    _gold: gold,
    _standout: standout,
    _ptsLeader: ptsLeader,
    _prospects: prospects,
    _playerStats: playerStats,
    _stMap: stMap,
  };
}

function int(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function findProspectStock(prospects, player, which) {
  const pid = String(player?.player_id || "");
  const row = prospects.find((p) => String(p.player_id) === pid);
  if (!row) return "—";
  if (which === "before") return row.stock_rank_before ?? "—";
  return row.stock_rank_after ?? row.stock_rank_before ?? "—";
}

function deriveUserStockNote(userProspect, prospects) {
  const pid = String(userProspect.player_id || "");
  const row = prospects.find((p) => String(p.player_id) === pid);
  if (row?.stock_delta != null) {
    const d = int(row.stock_delta);
    if (d > 0) return `up ${d} spots on the board`;
    if (d < 0) return `down ${Math.abs(d)} spots`;
    return "holding steady";
  }
  return userProspect.made_wjc_team ? "on the national roster" : "cut from camp";
}

function getTodayGames(payload) {
  if (asArray(payload?.games_today).length) return asArray(payload.games_today);
  if (!payload?.hasData) return [];
  const games = asArray(payload.all_games).length
    ? asArray(payload.all_games)
    : asArray(payload.round_robin_games);
  const day = payload.wjc_day || 1;
  return games.filter((g) => Number(g.game_day) === day);
}

const SPEAKER_NAMES = {
  host_1: "Marcus Cole",
  host_2: "Jordan Hayes",
  host_3: "Dr. Elena Park",
};

export function buildWjcBroadcastLines(payload) {
  const vars = buildTemplateVars(payload);
  const pool = WJC_PHRASE_BANK.filter((p) => phraseApplies(p, vars, payload));
  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  const picked = shuffled.slice(0, Math.min(28, shuffled.length));

  const lines = picked.map((tpl, index) => {
    const text = interpolate(tpl.text, vars).replace(/\s+/g, " ").trim();
    return {
      id: `${tpl.tag || "line"}-${tpl.speakerId}-${index}`,
      speakerId: tpl.speakerId,
      speakerName: SPEAKER_NAMES[tpl.speakerId] || "Jordan Hayes",
      emotion: tpl.emotion,
      text,
      durationMs: tpl.speakerId === "host_1" ? 6200 : tpl.speakerId === "host_3" ? 6800 : 5800,
      meta: { tag: tpl.tag },
    };
  }).filter((line) => line.text && line.text.length > 12);

  if (!lines.length) {
    lines.push({
      id: "fallback",
      speakerId: "host_2",
      speakerName: "Jordan Hayes",
      emotion: "neutral",
      text: "World Juniors desk standing by. Sim the next tournament day for live scores, prospect tallies, and draft stock movement.",
      durationMs: 6000,
    });
  }

  return lines;
}

export function buildWjcShowcaseCards(payload) {
  const playerStats = asArray(payload?.player_stats);
  const prospects = asArray(payload?.tournament_prospects);
  const standings = asArray(payload?.standings);
  const games = asArray(payload?.all_games).length
    ? asArray(payload?.all_games)
    : asArray(payload?.round_robin_games);

  const cards = [];

  playerStats.slice(0, 12).forEach((p) => {
    const pr = prospects.find((x) => String(x.player_id) === String(p.player_id));
    cards.push({
      type: "player",
      player_id: p.player_id,
      name: p.name,
      wjc_country: p.wjc_country,
      g: p.g,
      a: p.a,
      pts: p.pts,
      gp: p.gp,
      plus_minus: p.plus_minus,
      sog: p.sog,
      stock_before: pr?.stock_rank_before,
      stock_after: pr?.stock_rank_after,
      stock_delta: pr?.stock_delta,
      is_user_prospect: p.is_user_prospect || pr?.is_user_prospect,
    });
  });

  standings.slice(0, 9).forEach((row) => {
    cards.push({
      type: "nation",
      code: row.code,
      label: row.label,
      w: row.w,
      l: row.l,
      pts: row.pts,
      gf: row.gf,
      ga: row.ga,
    });
  });

  games.slice(-6).forEach((g, i) => {
    cards.push({
      type: "game",
      id: `game-${i}`,
      home: g.home,
      away: g.away,
      home_goals: g.home_goals,
      away_goals: g.away_goals,
      round: g.round || "Final",
    });
  });

  return cards;
}

export function buildWjcDraftStockRows(payload, franchiseState) {
  const backend = asArray(payload?.tournament_prospects);
  if (backend.length) {
    const rows = backend.map((p) => ({
      player_id: p.player_id,
      draft_prospect_id: p.draft_prospect_id || p.player_id,
      prospect_classification: p.prospect_classification || (p.is_user_prospect ? "drafted_user" : "draft_eligible"),
      name: p.name,
      wjc_country: p.wjc_country,
      wjc_country_label: p.wjc_country_label || p.wjc_country,
      age: p.age,
      position: p.position,
      stock_before: p.stock_rank_before,
      stock_after: p.stock_rank_after,
      stock_delta: p.stock_delta ?? int(p.stock_rank_before) - int(p.stock_rank_after),
      tournament_pts: p.tournament_pts ?? 0,
      tournament_g: p.tournament_g ?? 0,
      tournament_gp: p.tournament_gp ?? 0,
      junior_league: p.junior_league || "",
      junior_team: p.junior_team || "",
      junior_gp: p.junior_gp ?? 0,
      junior_g: p.junior_g ?? 0,
      junior_a: p.junior_a ?? 0,
      junior_pts: p.junior_pts ?? 0,
      scouting_confidence: p.scouting_confidence,
      owner_team_abbr: p.owner_team_abbr,
      is_user_prospect: p.is_user_prospect,
      is_npc: p.is_npc,
    }));

    const draftEligible = rows.filter((r) => r.prospect_classification === "draft_eligible");
    draftEligible.sort((a, b) => Math.abs(int(b.stock_delta)) - Math.abs(int(a.stock_delta)));
    const notable = draftEligible.filter((r) => int(r.stock_before) <= 40);
    const merged = [...draftEligible.slice(0, 16)];
    notable.forEach((r) => {
      if (!merged.some((m) => m.player_id === r.player_id)) merged.push(r);
    });
    return merged.slice(0, 24);
  }

  return asArray(payload?.user_prospects).map((p) => ({
    player_id: p.player_id,
    prospect_classification: "drafted_user",
    name: p.name,
    wjc_country: p.wjc_country,
    wjc_country_label: p.wjc_country_label,
    age: p.age,
    stock_before: null,
    stock_after: null,
    stock_delta: null,
    owner_team_abbr: franchiseState?.team?.abbreviation || franchiseState?.team?.abbr || "YOU",
    is_user_prospect: true,
  }));
}

export function buildWjcStatLeaders(payload) {
  const stats = asArray(payload?.player_stats);
  const standings = asArray(payload?.standings);

  const skaters = stats.filter((p) => String(p.position || "F").toUpperCase() !== "G");
  const byPts = [...skaters].sort((a, b) => int(b.pts) - int(a.pts)).slice(0, 10);
  const byGoals = [...skaters].sort((a, b) => int(b.g) - int(a.g)).slice(0, 8);
  const byPm = [...skaters].sort((a, b) => int(b.plus_minus) - int(a.plus_minus)).slice(0, 8);

  const teamLeaders = [...standings]
    .sort((a, b) => int(b.pts) - int(a.pts))
    .slice(0, 9)
    .map((row) => ({
      code: row.code,
      label: row.label,
      gp: row.gp,
      w: row.w,
      l: row.l,
      gf: row.gf,
      ga: row.ga,
      pts: row.pts,
    }));

  return { byPts, byGoals, byPm, teamLeaders };
}

export { formatScoreLine, gameCode, getTodayGames };
