import React, { useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { askPuckGpt } from "../services/franchiseService";

const STARTERS = [
  "Who played on Feb 15 2029?",
  "Which players said the worst things on social media?",
  "Which players said the best things on social media?",
  "Best fit for my second-pair LD?",
];

export default function PuckGptScreen() {
  const { setScreen } = useGameUI();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const ask = async (text) => {
    const q = String(text || query || "").trim();
    if (!q || busy) return;
    setQuery(q);
    setBusy(true);
    setError("");
    try {
      setResult(await askPuckGpt(q));
    } catch (err) {
      setError(err?.message || "PuckGPT could not read this save.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="game-screen puckgpt-screen" data-register="shell">
      <header className="puckgpt-bar">
        <div>
          <div className="puckgpt-mark">PUCKGPT</div>
          <p>Search the save. A date, a social post, or a spot on your roster.</p>
        </div>
        <button type="button" className="puckgpt-back" onClick={() => setScreen(SCREENS.HUB)}>
          Back to hub
        </button>
      </header>
      <form
        className="puckgpt-search"
        onSubmit={(e) => {
          e.preventDefault();
          ask(query);
        }}
      >
        <input
          className="puckgpt-query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Who played on Feb 15 2029?"
          aria-label="Ask PuckGPT"
        />
        <button type="submit" className="puckgpt-go" disabled={busy}>
          {busy ? "Searching" : "Search"}
        </button>
      </form>
      <div className="puckgpt-chips">
        {STARTERS.map((item) => (
          <button key={item} type="button" className="puckgpt-chip" onClick={() => ask(item)}>
            {item}
          </button>
        ))}
      </div>
      {error ? <p className="puckgpt-error" role="alert">{error}</p> : null}
      {result?.answer ? <h2 className="puckgpt-answer">{result.answer}</h2> : null}
      <ul className="puckgpt-hits">
        {(result?.hits || []).map((hit, i) => (
          <li key={`${hit.kind || "note"}-${hit.date || ""}-${i}`} className="puckgpt-hit">
            {hit.date ? <em>{hit.date}</em> : null}
            {hit.summary || hit.kind}
          </li>
        ))}
      </ul>
    </div>
  );
}
