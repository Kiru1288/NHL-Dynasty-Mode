/**
 * Map API roster row → dense table + card fields (PS2-style columns).
 * Backend sends: name, position, ovr, morale
 */

import { ensurePlayerHeadshotFields } from "../utils/playerHeadshots";

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h << 5) - h + s.charCodeAt(i);
  return Math.abs(h);
}

export function heightFromCm(heightCm) {
  const cm = Number(heightCm);
  if (!Number.isFinite(cm) || cm <= 0) return "";
  const totalIn = Math.round(cm / 2.54);
  const ft = Math.floor(totalIn / 12);
  const inch = totalIn % 12;
  return `${ft}'${inch}"`;
}

export function enrichRosterPlayer(p, index) {
  const name = p.name || "?";
  const h = hashStr(name);
  const num = 1 + (h % 98);
  const cpt = index === 0 ? "C" : "—";
  const asg = p.assignment || {};
  const status =
    asg.level === "ufa"
      ? asg.overseas
        ? "INT"
        : "UFA"
      : asg.level === "junior"
        ? "DEV"
        : asg.level === "ahl"
          ? "AHL"
          : asg.level === "echl"
            ? "ECH"
            : p.morale >= 0.55
              ? "ACT"
              : "RST";
  const off = Math.round((p.ovr || 70) * 0.42 + (h % 12));
  const age = p.age != null && p.age > 0 ? p.age : 18 + (h % 18);
  const natFromApi = (p.nationality || "").slice(0, 3).toUpperCase();
  const nat = natFromApi || ["CAN", "USA", "SWE", "FIN", "RUS"][(h >> 2) % 5];
  let hgt = p.height_display || p.height || "";
  if (!hgt && p.height_cm > 0) {
    hgt = heightFromCm(p.height_cm);
  }
  let wgt = "";
  if (p.weight > 0) {
    wgt = `${Math.round(Number(p.weight))} lb`;
  } else if (p.weight_kg > 0) {
    wgt = `${Math.round(Number(p.weight_kg) * 2.20462)} lb`;
  }
  return ensurePlayerHeadshotFields({
    ...p,
    num,
    cpt,
    status,
    off,
    age,
    hgt,
    wgt,
    nat,
    nationality: p.nationality || nat,
    shot: h % 2 === 0 ? "L" : "R",
  });
}
