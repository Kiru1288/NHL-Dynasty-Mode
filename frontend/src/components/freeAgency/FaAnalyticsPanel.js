import React from "react";

/** D7: last season line, advanced analytics and NHL career for a free agent under evaluation. */
function cell(v, dp = 0) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return dp ? n.toFixed(dp) : String(Math.round(n));
}

function signed(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n > 0 ? `+${n}` : String(n);
}

export default function FaAnalyticsPanel({ analytics, loading }) {
  if (loading && !analytics) return <div className="fa-an"><p className="fa-an__empty">Loading stats & analytics…</p></div>;
  if (!analytics) return null;
  const goalie = Boolean(analytics.is_goalie);
  const last = analytics.last_season || {};
  const adv = Array.isArray(analytics.advanced) ? analytics.advanced : [];
  const reads = Array.isArray(analytics.reads) ? analytics.reads : [];
  const career = analytics.career || {};
  const tot = career.totals || {};
  const recent = Array.isArray(career.recent) ? career.recent : [];
  const hasLast = Number(last.gp) > 0;
  const lastCells = goalie
    ? [["GP", cell(last.gp)], ["W", cell(last.w)], ["SV%", last.sv_pct != null ? Number(last.sv_pct).toFixed(3) : "—"], ["GAA", cell(last.gaa, 2)], ["SO", cell(last.so)]]
    : [["GP", cell(last.gp)], ["G", cell(last.g)], ["A", cell(last.a)], ["P", cell(last.pts)], ["+/-", signed(last.plus_minus)], ["SOG", cell(last.shots)], ["TOI", last.toi || "—"]];
  return (
    <section className="fa-an" aria-label="Stats and analytics">
      <div className="fa-an__head">
        <h4>Stats &amp; Analytics</h4>
        {last.season ? <span className="fa-an__tag">{last.season}</span> : null}
      </div>
      {hasLast ? (
        <div className="fa-an__line">
          {lastCells.map(([k, v]) => (
            <div key={k} className="fa-an__cell">
              <span>{k}</span>
              <strong>{v}</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="fa-an__empty">No NHL games logged last season.</p>
      )}
      {adv.length ? (
        <>
          <div className="fa-an__sub">Advanced</div>
          <div className="fa-an__adv">
            {adv.map((m) => (
              <div key={m.key} className="fa-an__chip" title={m.key}>
                <span>{m.label}</span>
                <strong>{m.value}</strong>
              </div>
            ))}
          </div>
        </>
      ) : null}
      {reads.length ? (
        <ul className="fa-an__reads">
          {reads.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      ) : null}
      {Number(tot.gp) > 0 ? (
        <>
          <div className="fa-an__sub">
            NHL career · {tot.seasons || recent.length} season{Number(tot.seasons) === 1 ? "" : "s"}
          </div>
          <div className="fa-an__career-tot">
            {goalie
              ? `${tot.gp} GP · ${tot.wins ?? 0} W · ${tot.shutouts ?? 0} SO`
              : `${tot.gp} GP · ${tot.g ?? 0} G · ${tot.a ?? 0} A · ${tot.pts ?? 0} P${tot.ppg != null ? ` · ${Number(tot.ppg).toFixed(2)} P/GP` : ""}`}
          </div>
          <table className="fa-an__table">
            <thead>
              <tr>
                <th>Season</th>
                <th>Team</th>
                <th>GP</th>
                {goalie ? <><th>W</th><th>SV%</th><th>GAA</th></> : <><th>G</th><th>A</th><th>P</th><th>+/-</th></>}
              </tr>
            </thead>
            <tbody>
              {recent.map((r, i) => (
                <tr key={`${r.season}-${r.team}-${i}`}>
                  <td>{r.season || "—"}</td>
                  <td>{r.team || "—"}</td>
                  <td>{cell(r.gp)}</td>
                  {goalie ? (
                    <><td>{cell(r.wins)}</td><td>{r.sv_pct != null ? Number(r.sv_pct).toFixed(3) : "—"}</td><td>{cell(r.gaa, 2)}</td></>
                  ) : (
                    <><td>{cell(r.g)}</td><td>{cell(r.a)}</td><td>{cell(r.pts)}</td><td>{signed(r.plus_minus)}</td></>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}
    </section>
  );
}
