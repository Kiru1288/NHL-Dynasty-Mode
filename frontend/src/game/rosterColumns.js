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
  let hgt = p.height_display || "";
  if (!hgt && p.height_cm > 0) {
    const totalIn = Math.round(p.height_cm / 2.54);
    const ft = Math.floor(totalIn / 12);
    const inch = totalIn % 12;
    hgt = `${ft}'${inch}"`;
  }
  if (!hgt) {
    const totalIn = 66 + (h % 13);
    const ft = Math.floor(totalIn / 12);
    const inch = totalIn % 12;
    hgt = `${ft}'${inch}"`;
  }
  return {
    ...p,
    num,
    cpt,
    status,
    off,
    age,
    hgt,
    wgt: `${185 + (h % 40)}`,
    nat,
    shot: h % 2 === 0 ? "L" : "R",
  };
}
