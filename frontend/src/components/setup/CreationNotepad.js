import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { api, getFranchiseSessionId } from "../../services/api";
import { PixelPencil, PixelZamboni } from "./pixelSprites";

/*
  Setup loading notepad. A pixel spiral notepad that hand-writes the league's
  creation facts line by line while POST /api/franchise/start runs.

  - Starts with generic lines + warmup steps.
  - Polls GET /api/franchise/creation-report (live capture during /start, then the
    saved session) and writes real league facts as soon as they exist.
  - Fun facts are written on the bottom strip; a pixel Zamboni erases each one.
*/

const POLL_MS = 1500;
const FIRST_POLL_MS = 900;
const CHAR_MS = 24;
const FAST_CHAR_MS = 11;
const LINE_PAUSE_MS = 240;
const HEAD_PAUSE_MS = 420;
const FLIP_MS = 560;
const FACT_CHAR_MS = 32;
const FACT_HOLD_MS = 3600;
const ZAMBONI_MS = 2600;
const FACT_GAP_MS = 450;

let lineSeq = 0;
function mk(kind, text) {
  lineSeq += 1;
  return { id: `cnp${lineSeq}`, kind, text: String(text == null ? "" : text) };
}

function fmt1(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(1) : "--";
}

function signed(v) {
  const n = Number(v) || 0;
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}`;
}

function playerTag(p) {
  return `${p.name} (${p.team} ${p.pos})`;
}

function buildCapLines(rep) {
  const caps = Array.isArray(rep?.cap_leaders) ? rep.cap_leaders : [];
  if (!caps.length) return [];
  return [
    mk("head", "Cap kings"),
    ...caps.map((p) => mk("item", `${p.name} (${p.team}) $${Number(p.cap).toFixed(2)}M`)),
  ];
}

function buildFactLines(rep) {
  const out = [mk("gap", ""), mk("head", "League founded")];
  if (rep.created_label) out.push(mk("item", `Opening day: ${rep.created_label}`));
  if (rep.season_label) out.push(mk("item", `The ${rep.season_label} season is on the books`));
  if (rep.team_count) out.push(mk("item", `${rep.team_count} clubs, ${rep.player_count} NHL players`));

  const best = rep.best_teams || [];
  if (best.length) {
    out.push(mk("head", "Teams to beat"));
    best.forEach((t, i) => out.push(mk("item", `${i + 1}. ${t.name} - ${fmt1(t.ovr)} OVR`)));
  }
  const worst = rep.worst_teams || [];
  if (worst.length) {
    out.push(mk("head", "Basement watch"));
    worst.forEach((t) => out.push(mk("item", `${t.rank}. ${t.name} - ${fmt1(t.ovr)} OVR`)));
  }
  const stars = rep.best_players || [];
  if (stars.length) {
    out.push(mk("head", "Best players"));
    stars.forEach((p) => out.push(mk("item", `${playerTag(p)} - ${p.ovr}`)));
  }
  const weak = rep.worst_players || [];
  if (weak.length) {
    out.push(mk("head", "Weakest links"));
    weak.forEach((p) => out.push(mk("item", `${playerTag(p)} - ${p.ovr}`)));
  }
  const kids = rep.top_prospects || [];
  if (kids.length) {
    out.push(mk("head", "Top prospects"));
    kids.forEach((p) =>
      out.push(mk("item", `${p.name} (${p.team}, ${p.age}) ${p.ovr} OVR / ${p.pot} POT`))
    );
  }
  out.push(...buildCapLines(rep));
  const stories = rep.storylines || [];
  if (stories.length) {
    out.push(mk("head", "Storylines"));
    stories.forEach((s) => out.push(mk("item", s)));
  }
  return out;
}

function buildDetailLines(rep) {
  const out = [mk("gap", ""), mk("title", "Scout's notes")];
  const divs = rep.divisions || [];
  if (divs.length) {
    out.push(mk("head", "Division notes"));
    divs.forEach((d) => {
      out.push(mk("sub", `${d.name} (${fmt1(d.avg)} avg)`));
      out.push(mk("item", `Favourite: ${d.favorite?.name} (${fmt1(d.favorite?.ovr)})`));
      if (d.dark_horse) out.push(mk("item", `Dark horse: ${d.dark_horse.name}`));
      out.push(mk("item", `Basement: ${d.basement?.name} | gap ${fmt1(d.gap)}`));
    });
  }
  const mine = rep.user_team;
  const rivals = rep.rivals || [];
  if (rivals.length) {
    out.push(mk("head", mine ? `Rivals of the ${mine.abbr}` : "Rivals"));
    rivals.forEach((r) =>
      out.push(
        mk(
          "item",
          `${r.name} ${signed(r.delta)} OVR vs you${r.star ? ` - watch ${r.star}` : ""}`
        )
      )
    );
  }
  if (mine) {
    out.push(mk("head", `Your outlook: ${mine.name}`));
    out.push(mk("item", `Ranked ${mine.rank} of ${mine.of} | #${mine.division_rank} in the ${mine.division}`));
    out.push(mk("item", `Team OVR ${fmt1(mine.ovr)} | avg age ${mine.avg_age}`));
    if (mine.star) out.push(mk("item", `Franchise player: ${mine.star.name} (${mine.star.pos}, ${mine.star.ovr})`));
    if (mine.goalie) out.push(mk("item", `Starting goalie: ${mine.goalie.name} (${mine.goalie.ovr})`));
    if (mine.top_prospect) {
      out.push(
        mk("item", `Top prospect: ${mine.top_prospect.name}, ${mine.top_prospect.age} (${mine.top_prospect.pot} POT)`)
      );
    }
    if (mine.verdict) out.push(mk("verdict", mine.verdict));
  }
  out.push(mk("gap", ""));
  out.push(mk("note", "End of notes. Puck drops soon!"));
  return out;
}

function wrapLine(line, maxChars) {
  const text = line.text;
  if (!maxChars || text.length <= maxChars) return [line];
  const out = [];
  let cur = "";
  text.split(" ").forEach((word) => {
    let w = word;
    while (w.length > maxChars) {
      if (cur) {
        out.push(cur);
        cur = "";
      }
      out.push(w.slice(0, maxChars));
      w = w.slice(maxChars);
    }
    const next = cur ? `${cur} ${w}` : w;
    if (next.length > maxChars && cur) {
      out.push(cur);
      cur = w;
    } else {
      cur = next;
    }
  });
  if (cur) out.push(cur);
  return out.map((t, i) => ({ ...line, id: `${line.id}-${i}`, text: t, cont: i > 0 }));
}

function PencilAnchor({ idle = false }) {
  return (
    <span className={`cnp-pencil-anchor${idle ? " is-idle" : ""}`} aria-hidden="true">
      <PixelPencil className="cnp-pencil" />
    </span>
  );
}

function NoteLine({ line, children }) {
  return (
    <div className={`cnp-line is-${line.kind}${line.cont ? " is-cont" : ""}`}>
      <span className="cnp-line__ink">{line.text}</span>
      {children}
    </div>
  );
}

function FunFactStrip({ facts, paused }) {
  const [idx, setIdx] = useState(0);
  const [n, setN] = useState(0);
  const [phase, setPhase] = useState("write");
  const list = Array.isArray(facts) && facts.length ? facts : ["Hockey is better with a fresh sheet of ice."];
  const text = list[idx % list.length];

  useEffect(() => {
    if (paused) return undefined;
    let t;
    if (phase === "write") {
      if (n < text.length) t = window.setTimeout(() => setN((v) => v + 1), FACT_CHAR_MS);
      else t = window.setTimeout(() => setPhase("hold"), 0);
    } else if (phase === "hold") {
      t = window.setTimeout(() => setPhase("erase"), FACT_HOLD_MS);
    } else if (phase === "erase") {
      t = window.setTimeout(() => setPhase("gap"), ZAMBONI_MS);
    } else {
      t = window.setTimeout(() => {
        setIdx((v) => v + 1);
        setN(0);
        setPhase("write");
      }, FACT_GAP_MS);
    }
    return () => window.clearTimeout(t);
  }, [phase, n, text, paused]);

  const zamStyle = { "--cnp-zam-ms": `${ZAMBONI_MS}ms` };
  return (
    <div className="cnp-facts">
      <span className="cnp-facts__tag">Fun fact #{(idx % list.length) + 1}</span>
      <div className="cnp-facts__lane">
        <div
          key={`f${idx}`}
          className={`cnp-facts__text${phase === "erase" ? " is-erasing" : ""}${phase === "gap" ? " is-gone" : ""}`}
          style={zamStyle}
        >
          {text.slice(0, n)}
          {phase === "write" && !paused ? <PencilAnchor /> : null}
        </div>
        {phase === "erase" ? (
          <div key={`z${idx}`} className="cnp-zamboni" style={zamStyle} aria-hidden="true">
            <i className="cnp-zamboni__trail" />
            <PixelZamboni className="cnp-zamboni__sprite" />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function CreationNotepad({
  teamName,
  crestSrc,
  gmName,
  playerUniverse,
  injuriesEnabled,
  warmup,
  warmupLabels,
  facts,
  failed,
  error,
  loading,
  onRetry,
  onBack,
}) {
  const [page, setPage] = useState({ no: 1, lines: [], flipping: false });
  const [cur, setCur] = useState(null);
  const [wake, setWake] = useState(0);
  const [grid, setGrid] = useState({ rows: 14, cols: 1 });

  const stepQ = useRef([]);
  const factQ = useRef([]);
  const metricsRef = useRef({ capacity: 14, maxChars: 40 });
  const bodyRef = useRef(null);
  const probeRef = useRef(null);
  const progressRef = useRef({ seen: new Set(), facts: false, cap: false, warm: new Set() });
  const doneRef = useRef(false);
  const initialSidRef = useRef(getFranchiseSessionId());
  const introRef = useRef(false);
  const wasFailedRef = useRef(Boolean(failed));

  const enqueue = useCallback((lines, priority) => {
    if (!lines.length) return;
    (priority ? stepQ : factQ).current.push(...lines);
    setWake((w) => w + 1);
  }, []);

  // Opening lines (guarded so StrictMode's double effect does not write them twice).
  useEffect(() => {
    if (introRef.current) return;
    introRef.current = true;
    const universe = playerUniverse === "real_nhl" ? "Real NHL players" : "Generated players";
    enqueue(
      [
        mk("note", `Club: ${teamName || "NHL"}`),
        mk("note", `GM: ${String(gmName || "").trim() || "General Manager"}`),
        mk("note", `${universe} | injuries ${injuriesEnabled ? "on" : "off"}`),
        mk("gap", ""),
        mk("head", "Setting up the league"),
        mk("step", "Sharpening pencils..."),
        mk("step", "Flooding the rink..."),
      ],
      false
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Retry after a failure: start a fresh page and listen again.
  useEffect(() => {
    if (wasFailedRef.current && !failed) {
      stepQ.current = [];
      factQ.current = [];
      progressRef.current = { seen: new Set(), facts: false, cap: false, warm: new Set() };
      doneRef.current = false;
      setCur(null);
      setPage((p) => ({ no: p.no + 1, lines: [], flipping: false }));
      enqueue([mk("head", "Take two"), mk("step", "Re-sharpening pencils...")], false);
    }
    wasFailedRef.current = Boolean(failed);
  }, [failed, enqueue]);

  // Hub warmup stages (executive office, crests, records).
  useEffect(() => {
    if (!warmup || typeof warmup !== "object") return;
    const fresh = [];
    Object.entries(warmupLabels || {}).forEach(([key, label]) => {
      if (warmup[key] === "ready" && !progressRef.current.warm.has(key)) {
        progressRef.current.warm.add(key);
        fresh.push(mk("step", `${label} ready`));
      }
    });
    enqueue(fresh, true);
  }, [warmup, warmupLabels, enqueue]);

  const handleReport = useCallback(
    (data, initialSid) => {
      if (!data || typeof data !== "object") return;
      if (data.source === "session" && initialSid && getFranchiseSessionId() === initialSid) {
        return; // the previous save, not the league being built
      }
      const st = progressRef.current;
      const steps = Array.isArray(data.steps) ? data.steps : [];
      const fresh = steps.filter((s) => s && s.key && !st.seen.has(s.key));
      fresh.forEach((s) => st.seen.add(s.key));
      enqueue(fresh.map((s) => mk("step", s.line)), true);
      const rep = data.ready && data.report && typeof data.report === "object" ? data.report : null;
      if (rep && !st.facts) {
        st.facts = true;
        st.cap = Boolean(rep.cap_leaders && rep.cap_leaders.length);
        enqueue([...buildFactLines(rep), ...buildDetailLines(rep)], false);
      } else if (rep && !st.cap && rep.cap_leaders && rep.cap_leaders.length) {
        st.cap = true;
        enqueue([mk("gap", ""), ...buildCapLines(rep)], false);
      }
      if (data.source === "session" && data.ready) doneRef.current = true;
    },
    [enqueue]
  );

  // Light polling of the creation report while the league is being built.
  useEffect(() => {
    if (failed) return undefined;
    let alive = true;
    let timer = null;
    const initialSid = initialSidRef.current;
    const tick = async () => {
      try {
        const { data } = await api.get("/api/franchise/creation-report", { timeout: 5000 });
        if (alive) handleReport(data, initialSid);
      } catch (_err) {
        // The report is decoration; a failed poll just tries again.
      }
      if (alive && !doneRef.current) timer = window.setTimeout(tick, POLL_MS);
    };
    timer = window.setTimeout(tick, FIRST_POLL_MS);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [failed, handleReport]);

  // Page metrics: rows/columns that fit and how many characters a line holds.
  useLayoutEffect(() => {
    const body = bodyRef.current;
    if (!body) return undefined;
    const measure = () => {
      const probe = probeRef.current;
      if (!probe || !bodyRef.current) return;
      const lineH = probe.querySelector(".cnp-probe__h")?.getBoundingClientRect().height || 30;
      const charW = (probe.querySelector(".cnp-probe__w")?.getBoundingClientRect().width || 200) / 20;
      const cs = window.getComputedStyle(bodyRef.current);
      const padL = parseFloat(cs.paddingLeft) || 0;
      const padR = parseFloat(cs.paddingRight) || 0;
      const innerW = bodyRef.current.clientWidth - padL - padR;
      const cols = innerW >= 820 ? 2 : 1;
      const gap = cols > 1 ? parseFloat(cs.columnGap) || 40 : 0;
      const rows = Math.max(4, Math.floor(bodyRef.current.clientHeight / Math.max(12, lineH)));
      const colW = (innerW - gap * (cols - 1)) / cols;
      const maxChars = Math.max(16, Math.floor(colW / Math.max(4, charW)) - 4);
      metricsRef.current = { capacity: rows * cols, maxChars };
      setGrid((g) => (g.rows === rows && g.cols === cols ? g : { rows, cols }));
    };
    measure();
    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(measure);
      ro.observe(body);
    }
    window.addEventListener("resize", measure);
    try {
      document.fonts?.ready?.then(measure).catch(() => {});
    } catch (_err) {
      // fonts API unavailable
    }
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  // The writer: one character at a time, one line at a time, flip when the page is full.
  useEffect(() => {
    if (failed || page.flipping) return undefined;
    let t = null;
    if (cur) {
      const len = cur.line.text.length;
      if (cur.n < len) {
        const backlog = stepQ.current.length + factQ.current.length;
        const delay = backlog > 10 ? FAST_CHAR_MS : CHAR_MS;
        const step = backlog > 30 ? 3 : backlog > 10 ? 2 : 1;
        t = window.setTimeout(
          () => setCur((c) => (c ? { ...c, n: Math.min(c.line.text.length, c.n + step) } : c)),
          delay
        );
      } else {
        const pause = cur.line.kind === "head" || cur.line.kind === "title" ? HEAD_PAUSE_MS : LINE_PAUSE_MS;
        t = window.setTimeout(() => {
          const done = cur.line;
          setCur(null);
          setPage((p) => ({ ...p, lines: [...p.lines, done] }));
        }, cur.line.kind === "gap" ? 60 : pause);
      }
    } else {
      // Dequeue inside a timer so an effect re-run (StrictMode) cannot drop a line.
      t = window.setTimeout(() => {
        const q = stepQ.current.length ? stepQ.current : factQ.current;
        const next = q.shift();
        if (!next) return;
        if (next.kind === "gap" && page.lines.length === 0) {
          setWake((w) => w + 1);
          return;
        }
        const pieces = wrapLine(next, metricsRef.current.maxChars);
        const first = pieces[0];
        q.unshift(...pieces.slice(1));
        if (page.lines.length >= metricsRef.current.capacity) {
          q.unshift(first);
          setPage((p) => ({ ...p, flipping: true }));
          return;
        }
        setCur({ line: first, n: 0 });
      }, 0);
    }
    return () => window.clearTimeout(t);
  }, [cur, page, wake, failed]);

  // Finish a page flip.
  useEffect(() => {
    if (!page.flipping) return undefined;
    const t = window.setTimeout(() => setPage((p) => ({ no: p.no + 1, lines: [], flipping: false })), FLIP_MS);
    return () => window.clearTimeout(t);
  }, [page.flipping]);

  const lastLine = page.lines[page.lines.length - 1];
  const idle = !cur && !failed && !page.flipping;

  return (
    <div className="cnp-stage" role="status" aria-busy={!failed}>
      <div className="cnp-pad">
        <div className="cnp-spiral" aria-hidden="true">
          {Array.from({ length: 20 }).map((_, i) => (
            <i key={i} />
          ))}
        </div>
        <div className="cnp-paper">
          <header className="cnp-head">
            <div className="cnp-head__title">
              <b>League log</b>
              <span>{teamName || "NHL"}</span>
            </div>
            <div className="cnp-head__meta">
              <span className={`cnp-stamp${failed ? " is-halt" : ""}`}>{failed ? "Halted" : "Loading"}</span>
              <span className="cnp-pageno">Pg {page.no}</span>
            </div>
            {crestSrc ? (
              <span className="cnp-crest" aria-hidden="true">
                <img src={crestSrc} alt="" />
              </span>
            ) : null}
          </header>

          <div
            ref={bodyRef}
            className={`cnp-body${grid.cols > 1 ? " is-two-col" : ""}${page.flipping ? " is-flipping" : ""}`}
            style={{
              gridTemplateRows: `repeat(${grid.rows}, var(--cnp-line))`,
              gridTemplateColumns: `repeat(${grid.cols}, minmax(0, 1fr))`,
            }}
            aria-hidden="true"
          >
            {page.lines.map((line) => (
              <NoteLine key={line.id} line={line}>
                {idle && line === lastLine ? <PencilAnchor idle /> : null}
              </NoteLine>
            ))}
            {cur ? (
              <div className={`cnp-line is-${cur.line.kind}${cur.line.cont ? " is-cont" : ""} is-writing`}>
                <span className="cnp-line__ink">{cur.line.text.slice(0, cur.n)}</span>
                <PencilAnchor />
              </div>
            ) : null}
          </div>

          <FunFactStrip facts={facts} paused={Boolean(failed)} />

          <div className="cnp-probe" ref={probeRef} aria-hidden="true">
            <div className="cnp-probe__h" />
            <span className="cnp-probe__w">abcdefghijABCDEFGHIJ</span>
          </div>
        </div>

        {failed ? (
          <div className="cnp-error" role="alert">
            <div className="cnp-error__card">
              <b className="cnp-error__title">Whistle!</b>
              <p className="cnp-error__text">{error}</p>
              <div className="cnp-error__actions setup-loading-actions">
                {typeof onRetry === "function" ? (
                  <button
                    type="button"
                    className="setup-loading-retry cnp-btn is-go"
                    onClick={onRetry}
                    disabled={loading && !failed}
                  >
                    Retry
                  </button>
                ) : null}
                {typeof onBack === "function" ? (
                  <button type="button" className="setup-loading-back cnp-btn" onClick={onBack}>
                    Back to setup
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        ) : null}
      </div>
      <p className="setup-sr-status" aria-live="polite">
        {failed ? error : lastLine && lastLine.kind !== "gap" ? lastLine.text : "Building your league."}
      </p>
    </div>
  );
}
