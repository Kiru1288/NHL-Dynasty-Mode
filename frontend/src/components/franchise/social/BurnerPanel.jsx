import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getBurnerState, postBurnerMessage, previewBurnerPost } from "../../../services/franchiseService";
import "./BurnerPanel.css";

// Highlighting uses the server's own weights (burner state payload), so the preview
// marks exactly the words the risk score counts (bug F10).

function tokenize(text) {
  return String(text || "").split(/(\s+)/);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function highlightHtml(text, weights = {}, names = new Set()) {
  return tokenize(text)
    .map((raw) => {
      const chunk = escapeHtml(raw);
      if (!raw.trim()) return chunk;
      const bare = raw.toLowerCase().replace(/[^a-z']/g, "");
      const w = Number(weights[bare]) || 0;
      if (w >= 16) return `<mark class="burner-hl burner-hl--danger">${chunk}</mark>`;
      if (w > 0) return `<mark class="burner-hl burner-hl--warn">${chunk}</mark>`;
      if (names.has(bare)) return `<mark class="burner-hl burner-hl--warn" title="Named in a live storyline">${chunk}</mark>`;
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
    </svg>
  );
}

export default function BurnerPanel({ sessionId, marketProfiles, defaultMarketKey, onPosted }) {
  const [state, setState] = useState(null);
  const [text, setText] = useState("");
  // Your club's market always applies; it isn't a dial you can turn (bug U5).
  const [marketKey, setMarketKey] = useState(defaultMarketKey || "default");
  const [previewRisk, setPreviewRisk] = useState(0);
  const [preview, setPreview] = useState(null);
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
      setPreview(null);
      return undefined;
    }
    const t = setTimeout(async () => {
      try {
        const res = await previewBurnerPost(text, marketKey, sessionId);
        setPreviewRisk(Number(res?.risk) || 0);
        setPreview(res || null);
      } catch {
        setPreviewRisk(0);
        setPreview(null);
      }
    }, 450);
    return () => clearTimeout(t);
  }, [text, marketKey, sessionId]);

  const syncScroll = () => {
    if (backdropRef.current && textareaRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop;
      backdropRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  };

  const weights = useMemo(() => (state?.risky_words && typeof state.risky_words === "object" ? state.risky_words : {}), [state?.risky_words]);
  const nameSet = useMemo(() => new Set((state?.storyline_names || []).map((n) => String(n).toLowerCase())), [state?.storyline_names]);

  const band = riskBand(previewRisk);
  const marketLabel = state?.default_market_label || (marketProfiles && marketProfiles[marketKey]?.label) || marketKey;
  const canPost = state ? state.can_post !== false : true;
  const hypeCount = Number(state?.recent_hype_posts) || 0;

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
      if (onPosted) onPosted(res);
      // Reload the real numbers instead of guessing the suspicion bump (bug F10).
      await load();
    } catch (e) {
      // Show the server's reason (bug F11), e.g. a burned account's cooldown.
      setError(e?.response?.data?.detail || "Post failed. Server rejected the request.");
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

      <div className="burner-field">
        <span>Market</span>
        <strong>{marketLabel}</strong>
      </div>
      {state?.exposed ? (
        <p className="burner-error">
          This account was traced to you{state?.days_until_new_account ? `. A new one can be opened in ${state.days_until_new_account} days.` : ". You can open a new one now."}
        </p>
      ) : null}
      {hypeCount >= 2 ? (
        <p className="burner-note">
          {hypeCount} hype posts this week. An account that only cheers for the club starts to look like team PR, and each one does less.
        </p>
      ) : null}

      <div className="burner-composer">
        <div
          ref={backdropRef}
          className="burner-composer__backdrop"
          aria-hidden
          dangerouslySetInnerHTML={{ __html: highlightHtml(text || " ", weights, nameSet) }}
        />
        <textarea
          ref={textareaRef}
          className="burner-composer__input"
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, 280))}
          onScroll={syncScroll}
          placeholder="Draft a post the room cannot trace back to you."
          rows={5}
        />
      </div>

      <div className="burner-risk-row">
        <div className={`burner-gauge-wrap is-${band}`}>
          <RiskGauge risk={previewRisk} />
          <div className="burner-gauge-read">
            <strong>{previewRisk}</strong>
            <span>trace risk</span>
            {preview?.catch_pct != null ? <em>{preview.catch_pct}% chance you're made on this post</em> : null}
          </div>
        </div>
        <div className="burner-outcomes">
          {preview?.tone ? (
            <span className={`burner-tone is-${preview.tone}`}>
              {String(preview.tone).replace(/_/g, " ")}
              {Array.isArray(preview.targets) && preview.targets.length ? ` · aimed at ${preview.targets.join(", ")}` : ""}
            </span>
          ) : null}
          <div className="burner-outcome burner-outcome--ok">
            <span>If it lands</span>
            <p>{preview?.if_lands || outcomeCopy(band, marketLabel, false)}</p>
          </div>
          <div className="burner-outcome burner-outcome--bad">
            <span>If you are made</span>
            <p>{preview?.if_caught || outcomeCopy(band, marketLabel, true)}</p>
          </div>
        </div>
      </div>

      {error ? <p className="burner-error">{error}</p> : null}

      <div className="burner-count">{text.length}/280</div>
      <button type="button" className="burner-post-btn" disabled={busy || !canPost || !text.trim()} onClick={handlePost}>
        {busy ? "Posting…" : state?.exposed && canPost ? "Open a new burner and post" : "Post from burner"}
      </button>

      <div className="burner-history">
        <h4>Recent posts</h4>
        {posts.length ? (
          posts.map((row, idx) => (
            <div key={`${row.day}-${idx}`} className={`burner-history__row ${row.caught ? "burner-history__row--caught" : ""}`}>
              <div>
                <strong>{row.caught ? "Exposed" : "Clean"}</strong>
                <span> · risk {row.risk}{row.tone ? ` · ${String(row.tone).replace(/_/g, " ")}` : ""}</span>
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
