/**
 * Map API roster row → dense table + card fields (PS2-style columns).
 * Backend sends: name, position, ovr, morale
 */

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h << 5) - h + s.charCodeAt(i);
  return Math.abs(h);
}

export function enrichRosterPlayer(p, index) {
  const name = p.name || "?";
  const h = hashStr(name);
  const num = 1 + (h % 98);
  const cpt = index === 0 ? "C" : "—";
  const status = p.morale >= 0.55 ? "ACT" : "RST";
  const off = Math.round((p.ovr || 70) * 0.42 + (h % 12));
  return {
    ...p,
    num,
    cpt,
    status,
    off,
    age: 18 + (h % 18),
    hgt: `${5 + ((h >> 3) % 5)}'${10 + ((h >> 5) % 9)}"`,
    wgt: `${185 + (h % 40)}`,
    nat: ["CAN", "USA", "SWE", "FIN", "RUS"][(h >> 2) % 5],
    shot: h % 2 === 0 ? "L" : "R",
  };
}
