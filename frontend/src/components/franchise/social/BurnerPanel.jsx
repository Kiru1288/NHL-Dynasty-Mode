import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getBurnerState, postBurnerMessage, previewBurnerPost } from "../../../services/franchiseService";

const RISKY_MID = new Set([
  "coach", "bench", "deal", "management", "soft", "joke", "gm",
]);
const RISKY_HIGH = new Set([
  "trade", "traded", "shop", "shopping", "dump", "fire", "fired", "quit", "resign",
  "owner", "ownership", "cheap", "lazy", "selfish", "washed", "overpaid", "embarrassing",
  "tank", "tanking", "choke", "choked", "clown", "disgrace", "garbage",
]);

function tokenize(text) {
  return String(text || "").split(/(\s+)/);
}

function highlightHtml(text) {
  return tokenize(text)
    .map((chunk) => {
      if (!chunk.trim()) return chunk;
      const bare = chunk.toLowerCase().replace(/[^a-z']/g, "");
      if (RISKY_HIGH.has(bare)) {
        return `<mark class="burner-hl burner-hl--danger">${chunk}</mark>`;
      }
      if (RISKY_MID.has(bare)) {
        return `<mark class="burner-hl burner-hl--warn">${chunk}</mark>`;
      }
      return chunk;
    })
    .join("");
}

function riskBand(risk) {
  if (risk < 35) return "low";
  if (risk < 60) return "mid";
  return "high";
}

function outcomeCopy(band, marketLabel, caught = false) {
  const m = marketLabel || "this market";
  if (caught) {
    if (band === "high") return `Major exposure in ${m}. Owner patience and fan trust take a hit.`;
    if (band === "mid") return `Desk links the post to the front office. Minor scandal cycle.`;
    return `Traceable pattern noted. Small media bump, lingering suspicion.`;
  }
  if (band === "high") return `Bold post lands. Fan pulse shifts if you stay anonymous.`;
  if (band === "mid") return `Room noise settles slightly. Suspicion still accumulates.`;
  return `Minimal splash. Low reward, low trace risk.`;
}

function RiskGauge({ risk }) {
  const r = Math.max(0, Math.min(100, Number(risk) || 0));
  const angle = -90 + (r / 100) * 180;
  const cx = 60;
  const cy = 58;
  const rad = (angle * Math.PI) / 180;
  const nx = cx + 42 * Math.cos(rad);
  const ny = cy + 42 * Math.sin(rad);
  return (
    <svg className="burner-gauge" viewBox="0 0 120 70" aria-hidden>
      <path d="M 18 58 A 42 42 0 0 1 102 58" fill="none" stroke="rgba(156,218,236,.2)" strokeWidth="8" />
      <path d="M 18 58 A 42 42 0 0 1 102 58" fill="none" stroke="var(--gold)" strokeWidth="8" strokeDasharray={`${(r / 100) * 132} 132`} />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--text)" strokeWidth="2.5" />
      <circle cx={cx} cy={cy} r="4" fill="var(--text)" />
      <text x={cx} y={66} textAnchor="middle" className="burner-gauge__label">{r}% risk</text>
    </svg>
  );
}

export default function BurnerPanel({ sessionId, marketProfiles, defaultMarketKey, onPosted }) {
  const [state, setState] = useState(null);
  const [text, setText] = useState("");
  const [marketKey, setMarketKey] = useState(defaultMarketKey || "default");
  const [previewRisk, setPreviewRisk] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const backdropRef = useRef(null);
  const textareaRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const data = await getBurnerState(sessionId);
      setState(data);
      if (data?.default_market_key) setMarketKey(data.default_market_key);
    } catch (e) {
      setError("Burner desk unavailable.");
    }
  }, [sessionId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!text.trim()) {
      setPreviewRisk(0);
      return undefined;
    }
    const t = setTimeout(async () => {
      try {
        const res = await previewBurnerPost(text, marketKey, sessionId);
        setPreviewRisk(Number(res?.risk) || 0);
      } catch {
        setPreviewRisk(0);
      }
    }, 900);
    return () => clearTimeout(t);
  }, [text, marketKey, sessionId]);

  const syncScroll = () => {
    if (backdropRef.current && textareaRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop;
      backdropRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  };

  const markets = useMemo(() => {
    const prof = marketProfiles && typeof marketProfiles === "object" ? marketProfiles : {};
    return Object.entries(prof).map(([key, val]) => ({
      key,
      label: val?.label || key,
    }));
  }, [marketProfiles]);

  const band = riskBand(previewRisk);
  const marketLabel = markets.find((m) => m.key === marketKey)?.label || marketKey;

  const handlePost = async () => {
    if (!text.trim()) {
      setError("Write something before you post it.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await postBurnerMessage(text, marketKey, sessionId);
      setText("");
      setState((prev) => ({
        ...(prev || {}),
        ...(res?.handle ? { handle: res.handle } : {}),
        posts: [...(prev?.posts || []), res].slice(-20),
        suspicion_score: res?.caught
          ? 100
          : (prev?.suspicion_score || 0) + (Number(res?.risk) || 0) * 0.12,
      }));
      if (onPosted) onPosted(res);
      await load();
    } catch (e) {
      setError("Post failed. Server rejected the request.");
    } finally {
      setBusy(false);
    }
  };

  const investigation = state?.investigation || {};
  const posts = Array.isArray(state?.posts) ? state.posts.slice(-5).reverse() : [];

  return (
    <div className="burner-panel">
      <header className="burner-panel__head">
        <div>
          <p className="sl-kicker">Burner account</p>
          <h3>{state?.handle || "Anonymous desk"}</h3>
        </div>
        <div className="burner-panel__suspicion">
          <span>Suspicion</span>
          <strong>{Math.round(Number(state?.suspicion_score) || 0)}</strong>
        </div>
      </header>

      {investigation?.reporter_id ? (
        <div className="burner-investigation">
          <span>{investigation.reporter_name || "Investigative desk"} tracking patterns</span>
          <strong>{Math.round(Number(investigation.progress) || 0)}%</strong>
        </div>
      ) : null}

      <label className="burner-field">
        <span>Market lens</span>
        <select value={marketKey} onChange={(e) => setMarketKey(e.target.value)}>
          {markets.map((m) => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
      </label>

      <div className="burner-composer">
        <div
          ref={backdropRef}
          className="burner-composer__backdrop"
          aria-hidden
          dangerouslySetInnerHTML={{ __html: highlightHtml(text || " ") }}
        />
        <textarea
          ref={textareaRef}
          className="burner-composer__input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onScroll={syncScroll}
          placeholder="Draft a post the room cannot trace back to you."
          rows={5}
        />
      </div>

      <div className="burner-risk-row">
        <RiskGauge risk={previewRisk} />
        <div className="burner-outcomes">
          <div className="burner-outcome burner-outcome--ok">
            <span>If it lands</span>
            <p>{outcomeCopy(band, marketLabel, false)}</p>
          </div>
          <div className="burner-outcome burner-outcome--bad">
            <span>If you are made</span>
            <p>{outcomeCopy(band, marketLabel, true)}</p>
          </div>
        </div>
      </div>

      {error ? <p className="burner-error">{error}</p> : null}

      <button type="button" className="sl-primary-btn" disabled={busy || state?.exposed} onClick={handlePost}>
        {busy ? "Posting…" : "Post from burner"}
      </button>

      <div className="burner-history">
        <h4>Recent posts</h4>
        {posts.length ? (
          posts.map((row, idx) => (
            <div key={`${row.day}-${idx}`} className={`burner-history__row ${row.caught ? "burner-history__row--caught" : ""}`}>
              <div>
                <strong>{row.caught ? "Exposed" : "Clean"}</strong>
                <span> · risk {row.risk}</span>
              </div>
              <p>{row.text}</p>
              {row.outcome ? <em>{row.outcome}</em> : null}
            </div>
          ))
        ) : (
          <p className="sl-decision-empty">No burner history yet.</p>
        )}
      </div>
    </div>
  );
}
