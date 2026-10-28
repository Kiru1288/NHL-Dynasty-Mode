import React, { useEffect, useRef, useState } from "react";
import { useFranchiseState, useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { askPuckGpt } from "../services/franchiseService";
import { PlayerProfileModal, normalizeRosterBrowserPlayer } from "./RosterScreen";
import { Pucky } from "../components/game/PuckBuddy";

const STARTERS = [
  "Why are we losing?",
  "Who's my best D?",
  "Who fits my 2nd pair?",
  "Best trade targets",
  "Who carried us?",
  "Top prospects",
  "Most goals this season",
  "Worst contract on my team",
];

const RECENT_KEY = "puckgpt_recent_v1";

function loadRecent() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(RECENT_KEY) || "[]");
    return Array.isArray(raw) ? raw.slice(0, 8) : [];
  } catch {
    return [];
  }
}

function saveRecent(list) {
  try {
    window.localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
  } catch {
    /* private window or blocked storage: recent list just isn't kept */
  }
}

// Same walk the Storylines desk uses: find a player row already loaded in state.
function findPlayerInState(state, pid) {
  const want = String(pid || "");
  if (!want || !state) return null;
  const seen = new Set();
  const queue = [[state, 0]];
  while (queue.length) {
    const [node, depth] = queue.shift();
    if (!node || typeof node !== "object" || seen.has(node)) continue;
    seen.add(node);
    if (!Array.isArray(node) && String(node.player_id ?? node.id ?? "") === want && (node.name || node.full_name)) return node;
    if (depth >= 5) continue;
    const kids = Array.isArray(node) ? node : Object.values(node);
    for (const k of kids) if (k && typeof k === "object") queue.push([k, depth + 1]);
  }
  return null;
}

function ReadAs({ interp }) {
  if (!interp) return null;
  const pct = Math.round((interp.confidence || 0) * 100);
  return (
    <div className={`puckgpt-readas is-${interp.band || "high"}`} title={`Intent: ${interp.intent}`}>
      <span>Read as</span> “{interp.rewritten}” <b>{pct}%</b>
    </div>
  );
}

function ResultTable({ table, onOpenPlayer }) {
  if (!table || !Array.isArray(table.rows) || !table.rows.length) return null;
  const ids = Array.isArray(table.ids) ? table.ids : [];
  return (
    <div className="puckgpt-table-wrap">
      <table className="puckgpt-table">
        <thead>
          <tr>
            {table.cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => {
            const pid = ids[i];
            return (
              <tr key={`${pid || "r"}-${i}`} className={pid ? "is-link" : ""} onClick={pid ? () => onOpenPlayer(pid, row[0]) : undefined}>
                {row.map((cell, j) => (
                  <td key={j}>{cell === null || cell === undefined || cell === "" ? "—" : String(cell)}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Turn({ turn, onAsk, onOpenPlayer }) {
  const res = turn.result;
  return (
    <article className="puckgpt-turn">
      <p className="puckgpt-q">{turn.query}</p>
      {turn.error ? <p className="puckgpt-error" role="alert">{turn.error}</p> : null}
      {!res && !turn.error ? <p className="puckgpt-thinking"><Pucky size="sm" talking bobbing />{turn.first ? "Indexing this save…" : "Reading the save…"}</p> : null}
      {res ? (
        <div className="puckgpt-reply">
          <ReadAs interp={res.interpretation} />
          {res.answer ? <h2 className="puckgpt-answer">{res.answer}</h2> : null}
          {Array.isArray(res.analysis) && res.analysis.length ? (
            <ul className="puckgpt-analysis">
              {res.analysis.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          ) : null}
          {Array.isArray(res.clarify) && res.clarify.length ? (
            <div className="puckgpt-chips puckgpt-chips--clarify">
              {res.clarify.map((c) => (
                <button key={c.query} type="button" className="puckgpt-chip" onClick={() => onAsk(c.query)}>
                  {c.label}
                  <small>{c.query}</small>
                </button>
              ))}
            </div>
          ) : null}
          {Array.isArray(res.cards) && res.cards.length ? (
            <div className="puckgpt-cards">
              {res.cards.map((c) => (
                <button key={c.id} type="button" className="puckgpt-card" onClick={() => onOpenPlayer(c.id, c.label)}>
                  <b>{c.label}</b>
                  <span>{c.sub}</span>
                </button>
              ))}
            </div>
          ) : null}
          <ResultTable table={res.table} onOpenPlayer={onOpenPlayer} />
          {Array.isArray(res.followups) && res.followups.length ? (
            <div className="puckgpt-chips">
              {res.followups.map((f) => (
                <button key={f} type="button" className="puckgpt-chip" onClick={() => onAsk(f)}>
                  {f}
                </button>
              ))}
            </div>
          ) : null}
          {typeof res.elapsed_ms === "number" ? <p className="puckgpt-meta">{res.elapsed_ms} ms</p> : null}
        </div>
      ) : null}
    </article>
  );
}

export default function PuckGptScreen() {
  const { setScreen } = useGameUI();
  const franchiseState = useFranchiseState();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState([]);
  const [recent, setRecent] = useState(loadRecent);
  const [dossier, setDossier] = useState(null);
  const [dossierTab, setDossierTab] = useState("overview");
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns]);

  useEffect(() => {
    inputRef.current?.focus?.();
  }, []);

  const ask = async (text) => {
    const q = String(text ?? query).trim();
    if (!q || busy) return;
    setQuery("");
    setBusy(true);
    const id = `${Date.now()}-${Math.random()}`;
    setTurns((prev) => [...prev, { id, query: q, first: prev.length === 0 }]);
    const nextRecent = [q, ...recent.filter((r) => r !== q)].slice(0, 8);
    setRecent(nextRecent);
    saveRecent(nextRecent);
    try {
      const result = await askPuckGpt(q);
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, result } : t)));
    } catch (err) {
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: err?.message || "PuckGPT could not read this save." } : t)));
    } finally {
      setBusy(false);
      inputRef.current?.focus?.();
    }
  };

  const openPlayer = (pid, label) => {
    const row = findPlayerInState(franchiseState, pid);
    if (row) {
      setDossier(row);
      setDossierTab("overview");
    } else if (label) {
      ask(`tell me about ${label}`);
    }
  };

  const empty = turns.length === 0;

  return (
    <div className="game-screen puckgpt-screen" data-register="shell">
      <header className="puckgpt-bar">
        <div>
          <div className="puckgpt-mark"><Pucky size="md" title="Pucky" />PUCKGPT</div>
          <p>Ask anything about this save. Typos, slang, shorthand and other languages are fine.</p>
        </div>
        <div className="puckgpt-bar__actions">
          {!empty ? (
            <button type="button" className="puckgpt-back" onClick={() => setTurns([])}>
              Clear
            </button>
          ) : null}
          <button type="button" className="puckgpt-back" onClick={() => setScreen(SCREENS.HUB)}>
            Back to hub
          </button>
        </div>
      </header>

      <section className="puckgpt-thread" aria-live="polite">
        {empty ? (
          <div className="puckgpt-empty">
            <div className="puckgpt-hello"><Pucky bobbing title="Pucky" /><p>Hey boss! What do you wanna know?</p></div>
            <h2>What do you want to know?</h2>
            <div className="puckgpt-chips">
              {STARTERS.map((item) => (
                <button key={item} type="button" className="puckgpt-chip" onClick={() => ask(item)}>
                  {item}
                </button>
              ))}
            </div>
            {recent.length ? (
              <>
                <h3>Recent</h3>
                <div className="puckgpt-chips">
                  {recent.map((item) => (
                    <button key={item} type="button" className="puckgpt-chip puckgpt-chip--ghost" onClick={() => ask(item)}>
                      {item}
                    </button>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        ) : (
          turns.map((t) => <Turn key={t.id} turn={t} onAsk={ask} onOpenPlayer={openPlayer} />)
        )}
        <div ref={endRef} />
      </section>

      <form
        className="puckgpt-search"
        onSubmit={(e) => {
          e.preventDefault();
          ask(query);
        }}
      >
        <input
          ref={inputRef}
          className="puckgpt-query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="whos my bets dman · why are we losing · compare mcdavid and matthews"
          aria-label="Ask PuckGPT"
          maxLength={300}
        />
        <button type="submit" className="puckgpt-go" disabled={busy || !query.trim()}>
          {busy ? "Reading" : "Ask"}
        </button>
      </form>

      {dossier ? (
        <PlayerProfileModal
          player={normalizeRosterBrowserPlayer(dossier, franchiseState, 0)}
          players={[normalizeRosterBrowserPlayer(dossier, franchiseState, 0)]}
          playerIndex={0}
          onSelectPlayer={() => {}}
          activeTab={dossierTab}
          setActiveTab={setDossierTab}
          storylines={[]}
          franchiseState={franchiseState}
          onClose={() => setDossier(null)}
        />
      ) : null}
    </div>
  );
}
