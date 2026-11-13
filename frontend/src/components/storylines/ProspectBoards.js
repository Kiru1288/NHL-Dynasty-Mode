/**
 * Bench Buzz prospect desks: "Top Prospects" (league-wide potential board) and
 * "Prospect Pools" (org pipelines ranked by their top-N prospects).
 * Data comes pre-built from StorylinesScreen (roster_browser organizations + development leagues).
 */
import React, { useMemo } from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

const arr = (v) => (Array.isArray(v) ? v : []);
const s = (v, f = "") => (v == null || v === "" ? f : String(v));

function logoFor(abbr, name) {
  return resolveFranchiseTeamLogo({ team_abbrev: abbr, abbrev: abbr, abbreviation: abbr, team_name: name }, name || abbr) || "";
}

function Face({ p, size = "lg" }) {
  const raw = p?.raw && typeof p.raw === "object" ? p.raw : {};
  const player = ensurePlayerHeadshotFields({
    ...raw,
    id: p?.player_id || raw.player_id || raw.id,
    name: p?.name || raw.name,
    position: p?.position || raw.position,
    age: p?.age || raw.age,
  });
  return <PlayerHeadshot player={player} size={size} preferPhoto showFlag={false} number={null} />;
}

function act(onOpen, raw) {
  return {
    role: "button",
    tabIndex: 0,
    onClick: () => raw && onOpen?.(raw),
    onKeyDown: (e) => {
      if ((e.key === "Enter" || e.key === " ") && raw) {
        e.preventDefault();
        onOpen?.(raw);
      }
    },
  };
}

function Tally({ title, rows, max }) {
  if (!rows.length) return null;
  const top = Math.max(1, max || Math.max(...rows.map((r) => r[1])));
  return (
    <section className="bbb-rail__box">
      <h4>{title}</h4>
      <ol className="bbb-tally">
        {rows.map(([label, n]) => (
          <li key={label}>
            <span>{label}</span>
            <i aria-hidden><i style={{ width: `${Math.round((n / top) * 100)}%` }} /></i>
            <b>{n}</b>
          </li>
        ))}
      </ol>
    </section>
  );
}

function countBy(list, fn) {
  const m = new Map();
  for (const x of list) {
    const k = fn(x);
    if (!k) continue;
    m.set(k, (m.get(k) || 0) + 1);
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
}

const posGroup = (pos) => {
  const p = s(pos).toUpperCase();
  if (p.startsWith("G")) return "Goalies";
  if (p === "D" || p === "LD" || p === "RD") return "Defence";
  if (p === "C") return "Centres";
  if (p) return "Wingers";
  return "";
};

/* ------------------------------------------------------------------ */

export function TopProspectsBoard({ rows, userRows, userTeamId, onOpen, empty }) {
  const list = arr(rows);
  const mine = arr(userRows);
  const uid = s(userTeamId).toUpperCase();
  const byPos = useMemo(() => countBy(list, (p) => posGroup(p.position)), [list]);
  const byLeague = useMemo(() => countBy(list, (p) => s(p.league)).slice(0, 8), [list]);
  const byOrg = useMemo(() => countBy(list, (p) => s(p.team_name || p.team_abbr || p.team_id)).slice(0, 6), [list]);

  return (
    <section className="bbb-desk" aria-label="Top prospects">
      <header className="ccx-head bbb-head">
        <div>
          <span className="ccx-kicker">Prospect desk</span>
          <h2 className="ccx-title">Top Prospects</h2>
          <p className="bbb-lede">The league's {list.length || 75} best prospects by development ceiling, wherever they're playing. Under 25 NHL games to qualify. Click anyone for their file.</p>
        </div>
      </header>
      {!list.length ? (
        empty
      ) : (
        <div className="bbb-split">
          <ol className="bbb-pgrid">
            {list.map((p, idx) => {
              const abbr = p.team_abbr || p.team_id;
              const logo = logoFor(abbr, p.team_name);
              const isUser = s(p.team_id).toUpperCase() === uid;
              const ovr = p.ovr ? Math.round(p.ovr) : null;
              const pot = Number(p.potential) || 0;
              return (
                <li
                  key={p.player_id || `${p.team_id}:${p.name}`}
                  className={`bbb-prow${isUser ? " is-user" : ""}${idx < 3 ? " is-podium" : ""}`}
                  title={`Open ${p.name}'s file`}
                  {...act(onOpen, p.raw)}
                >
                  <span className="bbb-prow__rank">{p.leagueRank}</span>
                  <span className="bbb-prow__face"><Face p={p} /></span>
                  <span className="bbb-prow__who">
                    <b>{p.name}</b>
                    <small>{[p.position, p.age != null ? `Age ${p.age}` : "", p.league].filter(Boolean).join(" · ")}</small>
                    <span className="bbb-prow__club">
                      {logo ? <img src={logo} alt="" /> : null}
                      <em>{p.team_name || abbr}</em>
                      {isUser ? <i>Your org</i> : null}
                    </span>
                  </span>
                  <span className="bbb-prow__nums">
                    {ovr ? <span className="bbb-num"><b>{ovr}</b><small>OVR</small></span> : null}
                    <span className="bbb-num is-pot"><b>{pot}</b><small>POT</small></span>
                  </span>
                  {ovr ? (
                    <span className="bbb-prow__gap" aria-hidden title={`${Math.max(0, pot - ovr)} points of growth left`}>
                      <i style={{ width: `${Math.min(100, (ovr / Math.max(pot, 1)) * 100)}%` }} />
                    </span>
                  ) : null}
                </li>
              );
            })}
          </ol>
          <aside className="bbb-rail">
            <section className="bbb-rail__box is-gold">
              <h4>Your org on the board</h4>
              <p className="bbb-rail__big">{mine.length}</p>
              <p className="bbb-rail__note">
                {mine[0] ? `Best: ${mine[0].name} (#${mine[0].leagueRank} · ${mine[0].potential} POT)` : `No prospects in the top ${list.length} yet`}
              </p>
              {mine.length ? (
                <ol className="bbb-mini">
                  {mine.slice(0, 8).map((p) => (
                    <li key={p.player_id || p.name} {...act(onOpen, p.raw)}>
                      <span className="bbb-mini__face"><Face p={p} size="sm" /></span>
                      <span className="bbb-mini__who"><b>{p.name}</b><small>#{p.leagueRank} · {p.position} · {p.league || "—"}</small></span>
                      <em>{p.potential}</em>
                    </li>
                  ))}
                </ol>
              ) : null}
            </section>
            <Tally title="Most prospects on the board" rows={byOrg} />
            <Tally title="By position" rows={byPos} />
            <Tally title="Where they're playing" rows={byLeague} />
          </aside>
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */

export function ProspectPoolsBoard({ rows, userRow, topN = 5, onOpen, empty }) {
  const list = arr(rows);
  const ranked = list.filter((r) => r.count > 0);
  const maxCeil = Math.max(1, ...ranked.map((r) => Number(r.avgPotential) || 0));
  const minCeil = Math.min(...ranked.map((r) => Number(r.avgPotential) || maxCeil), maxCeil);
  const deepest = [...ranked].sort((a, b) => b.count - a.count)[0];
  const bestHeadliner = [...ranked].filter((r) => r.topProspect).sort((a, b) => (a.topProspect.potentialRank || 999) - (b.topProspect.potentialRank || 999))[0];

  return (
    <section className="bbb-desk" aria-label="Prospect pools">
      <header className="ccx-head bbb-head">
        <div>
          <span className="ccx-kicker">Prospect desk</span>
          <h2 className="ccx-title">Prospect Pools</h2>
          <p className="bbb-lede">Every organization's pipeline, ranked by where its top {topN} prospects sit on the league potential board (lower average rank = deeper pool).</p>
        </div>
      </header>
      {!ranked.length ? (
        empty
      ) : (
        <div className="bbb-split">
          <ol className="bbb-pools">
            {list.map((row) => {
              const isUser = userRow && row.team_id === userRow.team_id;
              const logo = logoFor(row.team_id, row.name);
              const ceil = Number(row.avgPotential) || 0;
              const span = Math.max(1, maxCeil - minCeil);
              return (
                <li key={row.team_id} className={`bbb-pool${isUser ? " is-user" : ""}${row.count ? "" : " is-empty"}`}>
                  <span className="bbb-pool__rank">{row.poolRank}</span>
                  <span className="bbb-pool__logo">{logo ? <img src={logo} alt="" /> : <b>{row.team_id}</b>}</span>
                  <span className="bbb-pool__who">
                    <b>{row.name}{isUser ? <i>Your org</i> : null}</b>
                    <small>
                      {row.count} ranked {row.count === 1 ? "prospect" : "prospects"} · avg top-{topN} rank {row.avgRank != null ? row.avgRank.toFixed(1) : "—"}
                    </small>
                    <span className="bbb-pool__ceil" title={`Average ceiling ${ceil ? ceil.toFixed(1) : "—"} POT`}>
                      <i style={{ width: `${row.count ? Math.max(6, ((ceil - minCeil) / span) * 100) : 0}%` }} />
                      <em>{ceil ? `${ceil.toFixed(1)} avg POT` : "—"}</em>
                    </span>
                  </span>
                  <span className="bbb-pool__faces">
                    {arr(row.prospects).slice(0, 3).map((p) => (
                      <span key={p.player_id || p.name} className="bbb-pool__face" title={`${p.name} · #${p.potentialRank} · ${p.potential} POT`} {...act(onOpen, p.raw)}>
                        <Face p={p} size="sm" />
                        <small>{p.potential}</small>
                      </span>
                    ))}
                  </span>
                </li>
              );
            })}
          </ol>
          <aside className="bbb-rail">
            <section className="bbb-rail__box is-gold">
              <h4>Your pool rank</h4>
              <p className="bbb-rail__big">{userRow ? `#${userRow.poolRank}` : "—"}</p>
              <p className="bbb-rail__note">
                {userRow?.avgRank != null
                  ? `Avg top-${topN} rank ${userRow.avgRank.toFixed(1)} across ${userRow.count} ranked prospects`
                  : "Pipeline not ranked yet"}
              </p>
              {arr(userRow?.prospects).length ? (
                <ol className="bbb-mini">
                  {userRow.prospects.slice(0, 8).map((p) => (
                    <li key={p.player_id || p.name} {...act(onOpen, p.raw)}>
                      <span className="bbb-mini__face"><Face p={p} size="sm" /></span>
                      <span className="bbb-mini__who"><b>{p.name}</b><small>#{p.potentialRank} league{p.age ? ` · age ${p.age}` : ""}</small></span>
                      <em>{p.potential}</em>
                    </li>
                  ))}
                </ol>
              ) : null}
            </section>
            <section className="bbb-rail__box">
              <h4>Pipeline notes</h4>
              <ul className="bbb-facts">
                {ranked[0] ? <li><span>Deepest pool</span><b>{ranked[0].name}</b></li> : null}
                {deepest ? <li><span>Most ranked prospects</span><b>{deepest.name} ({deepest.count})</b></li> : null}
                {bestHeadliner?.topProspect ? (
                  <li><span>Top prospect</span><b>{bestHeadliner.topProspect.name} · {bestHeadliner.name}</b></li>
                ) : null}
                {ranked.length > 1 ? <li><span>Thinnest ranked pool</span><b>{ranked[ranked.length - 1].name}</b></li> : null}
              </ul>
            </section>
          </aside>
        </div>
      )}
    </section>
  );
}
