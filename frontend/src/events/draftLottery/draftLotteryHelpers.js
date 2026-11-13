import { getTeamLogoSrc, getTeamAbbreviation } from "../../utils/teamLogos";
import { getTeamName, safeArray } from "../shared/eventHelpers";

function viaLabel(pick) {
  if (!pick?.is_traded) return "";
  const viaId = pick.via_team_id || pick.original_owner_team_id;
  const viaName = pick.via_team_name || pick.original_owner_team_name;
  if (!viaId && !viaName) return "";
  return getTeamAbbreviation({ team_id: viaId, name: viaName, abbreviation: viaName }) || viaName || "";
}

export function normalizeLotteryPicks(raw) {
  const picks = safeArray(raw?.picks || raw?.order || raw?.final_order || raw);
  return picks
    .map((p, i) => {
      const pick = Number(p.pick ?? i + 1);
      const teamName = p.team_name || getTeamName(p);
      const viaId = p.via_team_id || p.original_owner_team_id || null;
      const viaName = p.via_team_name || p.original_owner_team_name || null;
      const isTraded = Boolean(p.is_traded) || Boolean(viaId && String(viaId) !== String(p.team_id));
      return {
        pick,
        team_id: p.team_id,
        team_name: teamName,
        lottery_team_id: p.lottery_team_id || p.original_owner_team_id || p.team_id,
        lottery_team_name: p.lottery_team_name || p.original_owner_team_name || teamName,
        original_owner_team_id: p.original_owner_team_id || p.lottery_team_id || p.team_id,
        original_owner_team_name: p.original_owner_team_name || p.lottery_team_name || teamName,
        original_rank: p.original_rank ?? pick,
        movement: p.movement ?? 0,
        odds: p.odds,
        is_traded: isTraded,
        via_team_id: isTraded ? viaId : null,
        via_team_name: isTraded ? viaName : null,
        via_abbr: isTraded ? viaLabel({ ...p, is_traded: true, via_team_id: viaId, via_team_name: viaName }) : "",
        logoSrc: getTeamLogoSrc({
          team_id: p.team_id,
          name: teamName,
          team_name: teamName,
        }),
        viaLogoSrc:
          isTraded && viaId
            ? getTeamLogoSrc({
                team_id: viaId,
                name: viaName,
                team_name: viaName,
              })
            : null,
      };
    })
    .sort((a, b) => a.pick - b.pick);
}

export function buildRevealSequence(picks) {
  return [...picks].sort((a, b) => b.pick - a.pick);
}

export function formatMovement(movement) {
  const n = Number(movement) || 0;
  if (n > 0) return { label: `↑ ${n} spot${n === 1 ? "" : "s"}`, tone: "up" };
  if (n < 0) return { label: `↓ ${Math.abs(n)} spot${Math.abs(n) === 1 ? "" : "s"}`, tone: "down" };
  return { label: "Holds position", tone: "hold" };
}

export function revealPaceMs(pickNumber) {
  const n = Number(pickNumber) || 1;
  if (n <= 3) return 1800;
  if (n <= 8) return 1500;
  return 1200;
}

/** Reveal pacing per speed: how long the envelope stays sealed, then how long the result holds. */
export const REVEAL_SPEEDS = {
  dramatic: { label: "Dramatic", sealed: 2600, hold: 3000, drumroll: 2600 },
  normal: { label: "Normal", sealed: 1400, hold: 1800, drumroll: 1400 },
  fast: { label: "Fast", sealed: 450, hold: 750, drumroll: 0 },
};

export function revealTiming(speedKey, pickNumber) {
  const sp = REVEAL_SPEEDS[speedKey] || REVEAL_SPEEDS.dramatic;
  const n = Number(pickNumber) || 99;
  // The last three envelopes get a longer drumroll; #1 the longest.
  const extra = n === 1 ? sp.drumroll * 1.4 : n <= 3 ? sp.drumroll : 0;
  return { sealed: Math.round(sp.sealed + extra), hold: Math.round(sp.hold + (n <= 3 ? sp.hold * 0.4 : 0)) };
}

/** NHL odds table (worst to best) used when the payload lacks per-team odds. */
export const DEFAULT_LOTTERY_ODDS = [18.5, 13.5, 11.5, 9.5, 8.5, 7.5, 6.5, 6.0, 5.0, 3.5, 3.0, 2.5, 2.0, 1.5, 0.5, 0.5];
export const LOTTERY_MAX_JUMP = 10;

function sameId(a, b) {
  return a != null && b != null && String(a) !== "" && String(a).toLowerCase() === String(b).toLowerCase();
}

/**
 * The 16 lottery clubs in pre-lottery order (worst first), joined with the backend
 * presentation snapshot when present. Works with the older payload (picks only).
 */
export function normalizeLotteryField(raw, picks, userTeamId) {
  const byRank = new Map();
  for (const p of picks) {
    byRank.set(Number(p.original_rank) || p.pick, p);
  }
  const fieldRaw = safeArray(raw?.field);
  const pre = fieldRaw.length ? fieldRaw : safeArray(raw?.pre_lottery_order);
  const ranks = pre.length
    ? pre.map((r, i) => Number(r.lottery_rank) || i + 1)
    : [...byRank.keys()].sort((a, b) => a - b);
  return ranks
    .map((rank, i) => {
      const src = pre[i] || {};
      const pick = byRank.get(rank) || picks.find((p) => sameId(p.lottery_team_id, src.team_id)) || {};
      const teamId = src.team_id || pick.lottery_team_id || pick.team_id;
      const teamName = src.team_name || pick.lottery_team_name || pick.team_name || "Team";
      const record = src.record || (src.points != null ? { pts: src.points, w: src.wins } : null);
      const odds = Number(src.odds ?? pick.odds);
      return {
        rank,
        team_id: teamId,
        team_name: teamName,
        abbrev: src.abbrev || getTeamAbbreviation({ team_id: teamId, name: teamName }) || "",
        logoSrc: getTeamLogoSrc({ team_id: teamId, name: teamName, team_name: teamName }),
        record,
        points_back: src.points_back,
        team_state: src.team_state || "",
        team_state_label: src.team_state_label || "",
        tank_pressure: src.tank_pressure,
        gm_name: src.gm_name || "",
        fan_mood: src.fan_mood || "",
        fan_confidence: src.fan_confidence,
        fan_trust_in_gm: src.fan_trust_in_gm,
        odds: Number.isFinite(odds) ? odds : DEFAULT_LOTTERY_ODDS[rank - 1] ?? 0,
        final_pick: pick.pick ?? src.final_pick ?? null,
        movement: pick.movement ?? src.movement ?? 0,
        pickRow: pick,
        is_traded: Boolean(pick.is_traded),
        owner_team_name: pick.team_name,
        is_user:
          Boolean(src.is_user) ||
          sameId(teamId, userTeamId) ||
          sameId(pick.team_id, userTeamId),
      };
    })
    .sort((a, b) => a.rank - b.rank);
}

/**
 * Every possible lottery result (two weighted draws, max jump 10), mirroring the
 * engine's run_draft_lottery. Each outcome: { p, order: [lottery rank at slot 1..n] }.
 */
export function enumerateLotteryOutcomes(oddsByRank) {
  const n = oddsByRank.length;
  if (n < 2) return [];
  const w = oddsByRank.map((o) => Math.max(0, Number(o) || 0));
  const total = w.reduce((a, b) => a + b, 0);
  if (total <= 0) return [];
  const outcomes = [];
  for (let a = 0; a < n; a += 1) {
    if (!w[a]) continue;
    const pa = w[a] / total;
    const rest = total - w[a];
    for (let b = 0; b < n; b += 1) {
      if (b === a || !w[b] || rest <= 0) continue;
      const p = pa * (w[b] / rest);
      const assigned = {};
      [a, b].forEach((idx, k) => {
        const pickNo = k + 1;
        const rank = idx + 1;
        assigned[rank - LOTTERY_MAX_JUMP <= pickNo ? pickNo : rank - LOTTERY_MAX_JUMP] = rank;
      });
      const used = new Set(Object.values(assigned));
      const remaining = [];
      for (let r = 1; r <= n; r += 1) if (!used.has(r)) remaining.push(r);
      const order = [];
      for (let slot = 1; slot <= n; slot += 1) {
        order.push(assigned[slot] != null ? assigned[slot] : remaining.shift());
      }
      outcomes.push({ p, order });
    }
  }
  return outcomes;
}

/**
 * Probability table rank -> slot, optionally conditioned on slots already revealed
 * ({ slot: rank }). Returns { probs: Map(rank -> number[] by slot index), total }.
 */
export function slotProbabilities(outcomes, n, revealed = {}) {
  const probs = new Map();
  for (let r = 1; r <= n; r += 1) probs.set(r, new Array(n).fill(0));
  const fixed = Object.entries(revealed).map(([s, r]) => [Number(s), Number(r)]);
  let total = 0;
  for (const o of outcomes) {
    if (fixed.some(([s, r]) => o.order[s - 1] !== r)) continue;
    total += o.p;
    o.order.forEach((rank, i) => {
      const row = probs.get(rank);
      if (row) row[i] += o.p;
    });
  }
  if (total > 0) {
    for (const row of probs.values()) for (let i = 0; i < row.length; i += 1) row[i] /= total;
  }
  return { probs, total };
}

export function pct(v, digits = 1) {
  const n = Number(v) * 100;
  if (!Number.isFinite(n) || n <= 0) return "0%";
  if (n >= 99.95) return "100%";
  if (n < 0.1) return "<0.1%";
  return `${n.toFixed(n >= 10 ? 0 : digits)}%`;
}

export function slotRange(row) {
  if (!row) return { best: null, worst: null };
  let best = null;
  let worst = null;
  row.forEach((p, i) => {
    if (p > 1e-9) {
      if (best == null) best = i + 1;
      worst = i + 1;
    }
  });
  return { best, worst };
}

export function normalizeProspects(raw) {
  return safeArray(raw?.top_prospects)
    .filter((p) => p && p.name)
    .map((p, i) => ({ ...p, id: p.id || p.key || `pros-${i}`, rank: Number(p.rank) || i + 1 }))
    .sort((a, b) => a.rank - b.rank);
}

export function formatRecord(rec) {
  if (!rec) return "";
  if (rec.w != null && rec.l != null) return `${rec.w}-${rec.l}-${rec.otl ?? 0}`;
  return "";
}

export function prospectLine(p) {
  if (!p) return "";
  const bits = [];
  if (p.age) bits.push(`${p.age}`);
  const club = p.team || p.team_name;
  if (club) bits.push(club);
  const lg = p.league_display || p.league;
  if (lg) bits.push(lg);
  return bits.join(" · ");
}

export function prospectProduction(p) {
  if (!p) return "";
  if (String(p.position || "").toUpperCase() === "G") {
    if (p.save_pct != null) return `${Number(p.save_pct).toFixed(3).replace(/^0/, "")} sv%${p.gaa != null ? ` · ${Number(p.gaa).toFixed(2)} GAA` : ""}`;
    return "";
  }
  if (p.gp) return `${p.points ?? 0} pts in ${p.gp} GP`;
  return "";
}

/** One plain sentence on what the lottery did to this club. */
export function lotteryMeaning(team, pickNumber) {
  const mv = Number(team?.movement) || 0;
  const odds = Number(team?.odds) || 0;
  const won = team?.pickRow?.won_pick != null || (pickNumber <= 2 && mv > 0);
  if (won && mv > 0) return `Wins draw ${pickNumber} from ${odds}% odds and jumps ${mv} spot${mv === 1 ? "" : "s"}.`;
  if (won) return `Wins draw ${pickNumber} and keeps the spot it earned.`;
  if (pickNumber <= 2 && mv === 0) return "Keeps its spot after a draw winner was capped by the 10-spot rule.";
  if (mv > 0) return `Draw winner capped by the 10-spot rule, still climbs ${mv}.`;
  if (mv < 0) return `Leapfrogged by a draw winner, slides ${Math.abs(mv)} spot${mv === -1 ? "" : "s"}.`;
  if (pickNumber <= 3) return "Holds a top-three pick.";
  return "Picks exactly where the standings put them.";
}

export function formatPickOwnershipLabel(pick) {
  const team = pick.team_name || "Team";
  const pickNum = pick.pick;
  if (pick.is_traded && (pick.via_team_name || pick.via_abbr || pick.original_owner_team_name)) {
    const via = pick.via_team_name || pick.original_owner_team_name || pick.via_abbr;
    return `${team} — Pick #${pickNum} (via ${via})`;
  }
  return `${team} — Pick #${pickNum}`;
}

export function pickOrdinal(n) {
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  const mod100 = num % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${num}th`;
  switch (num % 10) {
    case 1:
      return `${num}st`;
    case 2:
      return `${num}nd`;
    case 3:
      return `${num}rd`;
    default:
      return `${num}th`;
  }
}
