/**
 * "Hot or Not" rankings for the Storylines gossip page.
 * Data: GET /api/franchise/storylines/rankings (services/player_rankings.py). Every row
 * carries the number that put it there and a one-line "why", so the cover line is earned.
 */
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../../services/api";
import PlayerHeadshot from "../PlayerHeadshot";

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

  return (
    <section className="slr" aria-label="Player rankings">
      <header className="slr-head">
        <div>
          <span className="slr-kicker">The Bench Buzz rankings</span>
          <h2 className="slr-title">Hot or Not</h2>
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
        <div className="slr-board">
          {top ? (
            <article className="slr-cover" onClick={() => onOpenPlayer?.(top.player_id)} role="button" tabIndex={0}
              onKeyDown={(e) => (e.key === "Enter" ? onOpenPlayer?.(top.player_id) : null)}>
              <span className="slr-sticker">#1</span>
              <div className="slr-cover__face">
                <PlayerHeadshot player={{ id: top.player_id, player_id: top.player_id, name: top.name, team_abbrev: top.team }} size={120} />
              </div>
              <div className="slr-cover__copy">
                <span className="slr-cover__cat">{board.label}</span>
                <h3>{(COVER_LINES[board.key] || ((r) => r.name))(top)}</h3>
                <p className="slr-cover__num">{top.display}</p>
                <p className="slr-cover__why">{top.name} · {top.position} · {top.team} — {top.why}</p>
              </div>
            </article>
          ) : (
            <p className="slr-empty">{board.empty_reason || "Nobody qualifies yet."}</p>
          )}
          {rest.length ? (
            <ol className="slr-list" start={2}>
              {rest.map((r) => (
                <li key={r.player_id}>
                  <button type="button" onClick={() => onOpenPlayer?.(r.player_id)}>
                    <span className="slr-rank">{r.rank}</span>
                    <span className="slr-who"><b>{r.name}</b><small>{r.position} · {r.team} · age {r.age}</small></span>
                    <span className="slr-why">{r.why}</span>
                    <span className="slr-val">{r.display}</span>
                  </button>
                </li>
              ))}
            </ol>
          ) : null}
          <p className="slr-method"><b>How it's ranked:</b> {board.method} Source: {board.source}.</p>
        </div>
      ) : null}
    </section>
  );
}
