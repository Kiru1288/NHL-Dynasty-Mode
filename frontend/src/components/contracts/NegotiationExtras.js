import React from "react";
import { getNegotiationMeetings, getPlayerDossierRow, runNegotiationMeeting } from "../../services/franchiseService";

/* ------------------------------------------------------------------ tactics board */

const GROUP_ORDER = ["Locker room", "Off the ice", "Promises", "Hardball", "Wildcard"];
const GROUP_BLURB = {
  "Locker room": "Your players make the call",
  "Off the ice": "Sell the club, the city, the plan",
  Promises: "Give him something he wants",
  Hardball: "Move the price, risk the mood",
  Wildcard: "Who knows",
};

/**
 * Thirty negotiation moves (backend: services/negotiation_tactics.py). Each one moves the real
 * interest meter and/or his ask. You get a few moves per negotiation; each tactic once.
 */
export function NegotiationTactics({ playerId, onTranscript, onChanged, disabled = false }) {
  const [data, setData] = React.useState(null);
  const [group, setGroup] = React.useState("Locker room");
  const [busy, setBusy] = React.useState("");
  const [error, setError] = React.useState("");

  const load = React.useCallback(async () => {
    if (!playerId) return;
    try {
      const res = await getNegotiationMeetings(playerId);
      setData(res && res.ok ? res.tactics || null : null);
    } catch {
      setData(null);
    }
  }, [playerId]);

  React.useEffect(() => {
    setData(null);
    setError("");
    load();
  }, [load]);

  const run = async (opt) => {
    if (busy || disabled) return;
    setBusy(opt.id);
    setError("");
    try {
      const res = await runNegotiationMeeting(playerId, "tactic", opt.id);
      if (res?.ok) {
        if (Array.isArray(res.transcript)) onTranscript?.(res.transcript, res);
        if (res.options?.tactics) setData(res.options.tactics);
        else await load();
        onChanged?.(res);
      } else {
        setError(res?.reason || "That didn't happen.");
      }
    } catch (e) {
      setError(String(e?.message || "Couldn't reach his camp."));
    } finally {
      setBusy("");
    }
  };

  if (!data) return null;
  const options = Array.isArray(data.options) ? data.options : [];
  const groups = GROUP_ORDER.filter((g) => options.some((o) => o.group === g));
  const shown = options.filter((o) => o.group === group);
  const pips = Array.from({ length: Number(data.max) || 3 }, (_, i) => i < (Number(data.used) || 0));

  return (
    <section className="ntac" aria-label="Negotiation tactics">
      <div className="ntac-head">
        <span>Tactics</span>
        <em>
          {data.left} of {data.max} moves left
          <i className="ntac-pips" aria-hidden="true">
            {pips.map((on, i) => (
              <b key={i} className={on ? "is-used" : ""} />
            ))}
          </i>
        </em>
      </div>
      <div className="ntac-tabs" role="tablist">
        {groups.map((g) => (
          <button key={g} type="button" role="tab" aria-selected={g === group} className={g === group ? "is-on" : ""} onClick={() => setGroup(g)}>
            {g}
          </button>
        ))}
      </div>
      <p className="ntac-blurb">{GROUP_BLURB[group]}</p>
      <div className="ntac-grid">
        {shown.map((o) => {
          const res = o.result;
          const state = res ? (res.ok ? "is-landed" : "is-missed") : o.available ? "" : "is-locked";
          return (
            <button
              key={o.id}
              type="button"
              className={`ntac-move ${state} read-${String(o.read || "").toLowerCase().replace(/\s+/g, "-")}`}
              disabled={!o.available || disabled || Boolean(busy)}
              onClick={() => run(o)}
              title={o.why_not || o.line}
            >
              <strong>{o.label}</strong>
              <span className="ntac-line">{o.helper ? `${o.helper}: ${o.line.toLowerCase()}` : o.line}</span>
              {o.fact ? <small>{o.fact}</small> : null}
              <em className="ntac-read">
                {busy === o.id
                  ? "Working…"
                  : res
                    ? `${res.ok ? "Landed" : "Missed"} ${Number(res.interest_delta) > 0 ? "+" : ""}${Number(res.interest_delta || 0).toFixed(1)}${res.ask_shift_m ? ` · ask ${res.ask_shift_m > 0 ? "+" : "−"}$${Math.abs(res.ask_shift_m).toFixed(2)}M` : ""}`
                    : o.available
                      ? o.read
                      : o.why_not || "Locked"}
              </em>
            </button>
          );
        })}
      </div>
      {error ? <p className="ntac-error">{error}</p> : null}
    </section>
  );
}

/* ------------------------------------------------------------------ scouting card */

const STAT_LABELS = [
  ["gp", "GP"],
  ["g", "G"],
  ["goals", "G"],
  ["a", "A"],
  ["assists", "A"],
  ["pts", "PTS"],
  ["points", "PTS"],
  ["plus_minus", "+/-"],
  ["pim", "PIM"],
  ["toi_per_game", "TOI"],
  ["w", "W"],
  ["wins", "W"],
  ["gaa", "GAA"],
  ["sv_pct", "SV%"],
  ["save_pct", "SV%"],
  ["so", "SO"],
  ["war", "WAR"],
];

function statLine(stats) {
  const s = stats && typeof stats === "object" ? stats : {};
  const seen = new Set();
  const out = [];
  for (const [k, label] of STAT_LABELS) {
    const v = s[k];
    if (v == null || v === "" || seen.has(label)) continue;
    seen.add(label);
    let txt = v;
    if (label === "SV%") txt = Number(v).toFixed(3).replace(/^0/, "");
    else if (label === "GAA" || label === "WAR") txt = Number(v).toFixed(2);
    else if (label === "TOI" && typeof v === "number") txt = `${Math.floor(v)}:${String(Math.round((v % 1) * 60)).padStart(2, "0")}`;
    else if (label === "+/-" && Number(v) > 0) txt = `+${v}`;
    out.push({ label, value: String(txt) });
  }
  return out;
}

function tone(v) {
  const n = Number(v);
  if (n >= 85) return "is-elite";
  if (n >= 75) return "is-good";
  if (n >= 62) return "is-ok";
  return "is-low";
}

const SKIP_GROUPS = new Set(["Personality", "Special Traits", "Development"]);

/** Stats, recent seasons and attribute ratings for the player across the table. */
export function NegotiationScoutCard({ playerId, fallback = {} }) {
  const [row, setRow] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [openGroup, setOpenGroup] = React.useState(null); // null = first group open

  React.useEffect(() => {
    let cancelled = false;
    if (!playerId) return undefined;
    setLoading(true);
    setRow(null);
    getPlayerDossierRow(playerId)
      .then((r) => {
        if (!cancelled) setRow(r && r.found !== false ? r : null);
      })
      .catch(() => {
        if (!cancelled) setRow(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [playerId]);

  const p = { ...(fallback || {}), ...(row || {}) };
  const ovr = p.ovr ?? p.overall ?? p.effective_ovr;
  const pot = p.potential ?? p.potential_score;
  const season = statLine(p.season_stats);
  const career = statLine(p.career_totals);
  const seasons = (Array.isArray(p.career_seasons) ? p.career_seasons : []).slice(-3).reverse();
  const groups = (Array.isArray(p.rating_groups) ? p.rating_groups : []).filter((g) => !SKIP_GROUPS.has(g.title) && Array.isArray(g.rows) && g.rows.length);
  const avg = (g) => Math.round(g.rows.reduce((s, r) => s + Number(r.v || 0), 0) / Math.max(1, g.rows.length));
  const bio = [
    p.position,
    p.shoots || p.handedness ? `shoots ${String(p.shoots || p.handedness).charAt(0).toUpperCase()}` : null,
    p.height_display || null,
    p.weight ? `${p.weight} lb` : null,
    p.nationality || p.birth_country || null,
    p.archetype || p.play_role_label || null,
  ].filter(Boolean);

  return (
    <section className="nscout" aria-label="Scouting report">
      <div className="nscout-top">
        <div className={`nscout-ovr ${tone(ovr)}`}>
          <b>{ovr != null ? Math.round(Number(ovr)) : "—"}</b>
          <span>OVR</span>
        </div>
        {pot != null ? (
          <div className="nscout-ovr is-pot">
            <b>{Math.round(Number(pot))}</b>
            <span>POT</span>
          </div>
        ) : null}
        <p className="nscout-bio">{bio.join(" · ") || (loading ? "Loading his file…" : "")}</p>
      </div>

      {season.length ? (
        <div className="nscout-block">
          <span className="nscout-lab">This season{p.season_stats?.league ? ` · ${p.season_stats.league}` : ""}</span>
          <div className="nscout-stats">
            {season.map((s) => (
              <div key={s.label}>
                <b>{s.value}</b>
                <span>{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {seasons.length ? (
        <div className="nscout-block">
          <span className="nscout-lab">Recent seasons</span>
          <table className="nscout-table">
            <tbody>
              {seasons.map((s, i) => {
                const line = statLine(s).filter((x) => ["GP", "G", "A", "PTS", "W", "SV%", "GAA"].includes(x.label));
                return (
                  <tr key={`${s.season || i}-${s.team_name || s.league || ""}`}>
                    <th>
                      {s.season_label || s.season || "—"}
                      <small>{s.team_name || s.team || s.league || ""}</small>
                    </th>
                    {line.map((x) => (
                      <td key={x.label}>
                        <b>{x.value}</b>
                        <span>{x.label}</span>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      {career.length ? (
        <p className="nscout-career">
          <span className="nscout-lab">Career</span> {career.filter((c) => ["GP", "G", "A", "PTS", "W", "SV%"].includes(c.label)).map((c) => `${c.value} ${c.label}`).join(" · ")}
        </p>
      ) : null}

      {groups.length ? (
        <div className="nscout-block">
          <span className="nscout-lab">Attribute ratings</span>
          <div className="nscout-groups">
            {groups.map((g, gi) => {
              const a = avg(g);
              const open = openGroup === null ? gi === 0 : openGroup === g.title;
              return (
                <div key={g.title} className={`nscout-group${open ? " is-open" : ""}`}>
                  <button type="button" className="nscout-group-head" onClick={() => setOpenGroup(open ? "" : g.title)} aria-expanded={open}>
                    <span>{g.title}</span>
                    <i>
                      <b className={tone(a)} style={{ width: `${Math.max(4, Math.min(100, ((a - 40) / 59) * 100))}%` }} />
                    </i>
                    <strong className={tone(a)}>{a}</strong>
                  </button>
                  {open ? (
                    <div className="nscout-rows">
                      {g.rows.map((r) => (
                        <div key={r.id || r.label} className="nscout-row">
                          <span>{r.label}</span>
                          <i>
                            <b className={tone(r.v)} style={{ width: `${Math.max(4, Math.min(100, ((Number(r.v) - 40) / 59) * 100))}%` }} />
                          </i>
                          <strong className={tone(r.v)}>{r.v}</strong>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
          <p className="nscout-hint">Tap a group to see every rating.</p>
        </div>
      ) : loading ? (
        <p className="nscout-hint">Loading ratings…</p>
      ) : null}
    </section>
  );
}
