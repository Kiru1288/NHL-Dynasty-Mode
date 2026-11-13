/**
 * "Hot or Not" rankings for the Storylines gossip page.
 * Data: GET /api/franchise/storylines/rankings (services/player_rankings.py). Every row
 * carries the number that put it there and a one-line "why", so the cover line is earned.
 */
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../../services/api";
import PlayerHeadshot from "../PlayerHeadshot";
import TeamLogoBadge from "../ui/TeamLogoBadge";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";

const asPlayer = (r) => ensurePlayerHeadshotFields({ ...r, id: r.player_id, player_id: r.player_id, team_abbrev: r.team, overall: r.ovr });
const logoFor = (r) => resolveFranchiseTeamLogo({ team_id: r.team_id, abbrev: r.team, abbreviation: r.team, name: r.team });

const COVER_LINES = {
  best_player: (r) => `${r.name} is the best player alive. Discuss.`,
  worst_player: (r) => `Somebody keeps dressing ${r.name}.`,
  most_improved: (r) => `${r.name}'s glow-up: ${r.display}.`,
  biggest_decline: (r) => `What happened to ${r.name}?`,
  best_value: (r) => `${r.name}: the steal of the cap era.`,
  worst_contract: (r) => `${r.name}'s deal is a crime scene.`,
  best_finisher: (r) => `${r.name} can't miss.`,
  unluckiest: (r) => `${r.name} has angered the hockey gods.`,
  best_two_way: (r) => `${r.name} wins every shift.`,
  best_leader: (r) => `The room follows ${r.name}.`,
  biggest_hothead: (r) => `${r.name} needs a nap and a hug.`,
  most_underrated: (r) => `Nobody's talking about ${r.name}. We are.`,
  best_rookie: (r) => `${r.name} is already here.`,
  best_prospect: (r) => `${r.name}'s ceiling has no roof.`,
  tallest: (r) => `${r.name} ducks under door frames.`,
  heaviest: (r) => `${r.name} tips the scales.`,
  shortest: (r) => `${r.name}: small frame, big noise.`,
  lightest: (r) => `${r.name} skates like a feather.`,
  most_useful: (r) => `When ${r.name} is out there, they win.`,
  least_useful: (r) => `${r.name}'s ice time is a mystery.`,
  most_important: (r) => `${r.team} would sink without ${r.name}.`,
  fastest: (r) => `Blink and you miss ${r.name}.`,
  hardest_shot: (r) => `Goalies flinch at ${r.name}'s slapper.`,
  most_ice_time: (r) => `${r.name} never leaves the ice.`,
  most_hits: (r) => `${r.name} runs everybody.`,
  most_blocks: (r) => `${r.name} eats pucks for breakfast.`,
  most_pim: (r) => `${r.name} has a reserved seat in the box.`,
  highest_paid: (r) => `${r.name} cashes the biggest cheque.`,
  oldest: (r) => `${r.name}, ${r.age}, refuses to retire.`,
  sniper_pct: (r) => `${r.name} only shoots to score.`,
};

const TONES = ["hot", "pink", "lime", "sky", "tang", "grape"];

function errText(e) {
  return e?.response?.data?.detail || e?.message || "Could not load rankings";
}

export default function PlayerRankings({ userTeamId, onOpenPlayer }) {
  const [scope, setScope] = useState("league");
  const [kind, setKind] = useState("qualitative");
  const [key, setKey] = useState("best_player");
  const [state, setState] = useState({ data: null, err: "", loading: true });

  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, err: "" }));
    api
      .get("/api/franchise/storylines/rankings", { params: scope === "team" && userTeamId != null ? { team_id: userTeamId } : undefined, timeout: 30000 })
      .then(({ data }) => alive && setState({ data, err: "", loading: false }))
      .catch((e) => alive && setState({ data: null, err: String(errText(e)), loading: false }));
    return () => {
      alive = false;
    };
  }, [scope, userTeamId]);

  const boards = useMemo(() => state.data?.[kind] || [], [state.data, kind]);
  const board = boards.find((b) => b.key === key) || boards[0];
  useEffect(() => {
    if (boards.length && !boards.some((b) => b.key === key)) setKey(boards[0].key);
  }, [boards, key]);

  const top = board?.rows?.[0];
  const rest = board?.rows?.slice(1) || [];
  const half = Math.ceil(rest.length / 2);
  const cols = rest.length > 6 ? [rest.slice(0, half), rest.slice(half)] : [rest];
  const coverLine = top ? (COVER_LINES[board.key] || ((r) => r.name))(top) : "";
  const showOvrCorner = top && !/OVR$/.test(String(top.display || ""));

  return (
    <section className="slr bbb-hon" aria-label="Player rankings">
      <header className="ccx-head bbb-head bbb-hon__head">
        <div>
          <span className="ccx-kicker">The Bench Buzz rankings</span>
          <h2 className="ccx-title">Hot or Not</h2>
          <p className="bbb-lede">
            {board ? `${board.label}: ${board.method}` : "Who's hot, who's not, and the numbers that put them there."}
            {state.data?.stats_source ? ` Stats: ${state.data.stats_source}.` : ""}
          </p>
        </div>
        <div className="slr-switches">
          <div className="slr-seg" role="group" aria-label="Ranking type">
            {[["qualitative", "The Verdicts"], ["quantitative", "By the Numbers"]].map(([k, l]) => (
              <button key={k} type="button" className={kind === k ? "is-on" : ""} aria-pressed={kind === k} onClick={() => setKind(k)}>{l}</button>
            ))}
          </div>
          <div className="slr-seg" role="group" aria-label="Scope">
            {[["league", "Whole league"], ["team", "Your club"]].map(([k, l]) => (
              <button key={k} type="button" className={scope === k ? "is-on" : ""} aria-pressed={scope === k} onClick={() => setScope(k)}>{l}</button>
            ))}
          </div>
        </div>
      </header>

      {state.err ? <p className="slr-empty">{state.err}</p> : null}

      <nav className="slr-cats" aria-label="Ranking categories">
        {boards.map((b, i) => (
          <button
            key={b.key}
            type="button"
            className={`slr-cat slr-tone-${TONES[i % TONES.length]}${board?.key === b.key ? " is-on" : ""}`}
            aria-pressed={board?.key === b.key}
            onClick={() => setKey(b.key)}
          >
            {b.label}
          </button>
        ))}
      </nav>

      {state.loading && !state.data ? <p className="slr-empty">Counting the votes…</p> : null}

      {board ? (
        <div className="bbb-hon__board">
          {top ? (
            <article className="bbb-cover" onClick={() => onOpenPlayer?.(top.player_id)} role="button" tabIndex={0}
              onKeyDown={(e) => (e.key === "Enter" ? onOpenPlayer?.(top.player_id) : null)}>
              <span className="bbb-cover__sticker">#1</span>
              <div className="bbb-cover__photo">
                <PlayerHeadshot player={asPlayer(top)} size="lg" preferPhoto showFlag={false} number={null} />
                <span className="bbb-cover__logo"><TeamLogoBadge teamLogo={logoFor(top)} teamName={top.team} size={58} variant="badge" /></span>
                {showOvrCorner ? <span className="bbb-cover__ovr"><small>OVR</small>{top.ovr}</span> : null}
              </div>
              <div className="bbb-cover__copy">
                <span className="bbb-cover__cat">{board.label}</span>
                <h3>{coverLine}</h3>
                <p className="bbb-cover__num">{top.display}<Trend d={top.ovr_delta} /></p>
                <p className="bbb-cover__who">{top.name} · {top.position} · {top.team} · age {top.age}</p>
                <p className="bbb-cover__why">{top.why}{top.line && top.line !== top.why ? ` · ${top.line}` : ""}</p>
              </div>
            </article>
          ) : (
            <p className="slr-empty">{board.empty_reason || "Nobody qualifies yet."}</p>
          )}
          {rest.length ? (
            <div className={`bbb-hon__cols is-${cols.length}`}>
              {cols.map((col, ci) => (
                <ol key={ci} className="bbb-hon__list">
                  {col.map((r) => (
                    <li key={r.player_id}>
                      <button type="button" className="bbb-row" onClick={() => onOpenPlayer?.(r.player_id)}>
                        <span className="bbb-row__rank">{r.rank}</span>
                        <span className="bbb-row__face">
                          <PlayerHeadshot player={asPlayer(r)} size="md" preferPhoto showFlag={false} number={null} />
                          <span className="bbb-row__logo"><TeamLogoBadge teamLogo={logoFor(r)} teamName={r.team} size={30} variant="badge" /></span>
                        </span>
                        <span className="bbb-row__who">
                          <b>{r.name}</b>
                          <small>{r.position} · {r.team} · age {r.age}{/OVR$/.test(String(r.display || "")) ? "" : ` · ${r.ovr} OVR`}</small>
                          <em>{r.why}{r.line && r.line !== r.why ? ` · ${r.line}` : ""}</em>
                        </span>
                        <span className="bbb-row__val">{r.display}<Trend d={r.ovr_delta} /></span>
                      </button>
                    </li>
                  ))}
                </ol>
              ))}
            </div>
          ) : null}
          <p className="slr-method bbb-hon__method"><b>How it's ranked:</b> {board.method} Source: {board.source}.</p>
        </div>
      ) : null}
    </section>
  );
}

function Trend({ d }) {
  const n = Number(d);
  if (!Number.isFinite(n) || Math.abs(n) < 0.5) return null;
  return (
    <span className={`bbb-trend ${n > 0 ? "is-up" : "is-down"}`} title={`${n > 0 ? "+" : ""}${n.toFixed(1)} OVR since opening night`}>
      {n > 0 ? "▲" : "▼"}{Math.abs(n).toFixed(1)}
    </span>
  );
}
