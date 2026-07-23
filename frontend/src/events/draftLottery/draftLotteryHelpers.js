import { getTeamLogoSrc } from "../../utils/teamLogos";
import { getTeamName, safeArray } from "../shared/eventHelpers";

export function normalizeLotteryPicks(raw) {
  const picks = safeArray(raw?.picks || raw?.order || raw);
  return picks
    .map((p, i) => {
      const pick = Number(p.pick ?? i + 1);
      const teamName = p.team_name || getTeamName(p);
      return {
        pick,
        team_id: p.team_id,
        team_name: teamName,
        original_rank: p.original_rank ?? pick,
        movement: p.movement ?? 0,
        odds: p.odds,
        logoSrc: getTeamLogoSrc({
          team_id: p.team_id,
          name: teamName,
          team_name: teamName,
        }),
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
  if (n <= 3) return 3200;
  if (n <= 8) return 2600;
  return 2000;
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
