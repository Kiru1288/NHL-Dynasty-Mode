import React, { useCallback, useEffect, useRef, useState } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { wjcFlagUrl } from "../../utils/countryFlags";
import { WJC_HOSTS } from "./wjcBroadcastScripts";

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function safeText(value, fallback = "—") {
  if (value == null || value === "") return fallback;
  const text = String(value).trim();
  return text || fallback;
}

function abbrCode(value) {
  return safeText(value, "?").slice(0, 3).toUpperCase();
}

function gameIsComplete(game) {
  return game?.home_goals != null && game?.away_goals != null;
}

function gameStatusLabel(game) {
  if (gameIsComplete(game)) {
    if (game?.shootout || game?.decided_by === "so") return "Final / SO";
    if (game?.overtime || game?.decided_by === "ot") return "Final / OT";
    return "Final";
  }
  if (game?.is_live || game?.status === "live") return "Live";
  if (!game?.home || !game?.away) return "TBD";
  return "Scheduled";
}

export function wjcPlayerHeadshot(row) {
  const code = row?.wjc_country || "";
  return ensurePlayerHeadshotFields({
    id: row?.player_id,
    name: row?.name,
    position: row?.position,
    nationality_code: code,
    nationality: row?.wjc_country_label || code,
    country: row?.wjc_country_label || code,
    birth_country: row?.nationality || row?.wjc_country_label,
  });
}

export function FlagImg({ code, size = 40, className = "" }) {
  const src = wjcFlagUrl(code, size, "flat");
  if (!src) {
    return <span className="wjc-flag-fallback">{String(code || "?").slice(0, 3)}</span>;
  }
  return (
    <img
      className={className}
      src={src}
      alt=""
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={(e) => {
        e.currentTarget.style.display = "none";
      }}
    />
  );
}

export const HOST_TTS = {
  host_1: { rate: 1.08, pitch: 1.12 },
  host_2: { rate: 0.96, pitch: 1.0 },
  host_3: { rate: 0.9, pitch: 0.92 },
};

export function useWjcSpeech(voiceOn) {
  const spokenIdRef = useRef("");

  const speak = useCallback(
    (line) => {
      if (!voiceOn || !line?.text || typeof window === "undefined" || !window.speechSynthesis) return;
      if (spokenIdRef.current === line.id) return;
      spokenIdRef.current = line.id;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(line.text);
      const tune = HOST_TTS[line.speakerId] || HOST_TTS.host_2;
      utter.rate = tune.rate;
      utter.pitch = tune.pitch;
      window.speechSynthesis.speak(utter);
    },
    [voiceOn]
  );

  const cancel = useCallback(() => {
    spokenIdRef.current = "";
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const resetSpoken = useCallback(() => {
    spokenIdRef.current = "";
  }, []);

  return { speak, cancel, resetSpoken };
}

export function NationFlagsBar({ standings, countries }) {
  const stByCode = {};
  asArray(standings).forEach((row) => {
    stByCode[row.code] = row;
  });
  const labelBy = {};
  asArray(countries).forEach((c) => {
    labelBy[c.code] = c.label;
  });

  const nationList = asArray(countries).length
    ? asArray(countries)
    : Object.keys(stByCode).map((code) => ({ code, label: labelBy[code] || code }));

  const rows = nationList.map((c) => {
    const code = c.code;
    const st = stByCode[code];
    return (
      st || {
        code,
        label: labelBy[code] || c.label || code,
        w: 0,
        l: 0,
        pts: 0,
      }
    );
  });

  const leaderPts = Math.max(0, ...rows.map((row) => Number(row.pts) || 0));
  const hasStandings = rows.some((row) => (Number(row.gp) || 0) > 0 || (Number(row.pts) || 0) > 0);

  return (
    <div className="wjc-nation-bar" aria-label="Tournament nations">
      {rows.map((row) => {
        const isLeader =
          hasStandings && leaderPts > 0 && Number(row.pts) === leaderPts;
        const fullName = safeText(row.label || labelBy[row.code] || row.code);
        const record = `${row.w ?? 0}-${row.l ?? 0}`;
        return (
          <div
            key={row.code}
            className={`wjc-nation-chip${isLeader ? " is-leader" : ""}`}
            title={`${fullName} U20 · ${record}${hasStandings ? ` · ${row.pts ?? 0} PTS` : ""}`}
          >
            <FlagImg code={row.code} size={40} className="wjc-nation-chip__flag" />
            <span className="wjc-nation-chip__code">{abbrCode(row.code)}</span>
            <span className="wjc-nation-chip__record">{record}</span>
            {hasStandings ? (
              <span className="wjc-nation-chip__pts">{row.pts ?? 0} PTS</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

export function DeskControls({
  audioRef,
  isMuted,
  setIsMuted,
  voiceOn,
  setVoiceOn,
  onLeave,
  onSimDay,
  simBusy,
  onOpenDraftBoard,
}) {
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return undefined;
    const sync = () => setIsPlaying(!audio.paused);
    audio.addEventListener("play", sync);
    audio.addEventListener("pause", sync);
    return () => {
      audio.removeEventListener("play", sync);
      audio.removeEventListener("pause", sync);
    };
  }, [audioRef]);

  const toggleMusic = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio) return;
    try {
      if (audio.paused) await audio.play();
      else audio.pause();
    } catch (error) {
      console.warn("World Juniors music could not start:", error);
    }
  }, [audioRef]);

  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
    if (audioRef.current) audioRef.current.muted = !isMuted;
  }, [audioRef, isMuted, setIsMuted]);

  return (
    <div className="wjc-desk-controls" role="group" aria-label="Broadcast desk controls">
      {typeof onSimDay === "function" ? (
        <button type="button" className="wjc-sim-day" onClick={onSimDay} disabled={simBusy}>
          {simBusy ? "Simulating…" : "Sim Day"}
        </button>
      ) : null}
      <button type="button" className="wjc-desk-btn" onClick={toggleMusic} aria-pressed={isPlaying}>
        {isPlaying ? "Music On" : "Music Off"}
      </button>
      <button type="button" className="wjc-desk-btn" onClick={toggleMute} aria-pressed={isMuted}>
        {isMuted ? "Sound Off" : "Sound On"}
      </button>
      <button type="button" className="wjc-desk-btn" onClick={() => setVoiceOn(!voiceOn)} aria-pressed={voiceOn}>
        {voiceOn ? "Voices On" : "Voices Off"}
      </button>
      {typeof onOpenDraftBoard === "function" ? (
        <button type="button" className="wjc-desk-btn wjc-draft-class-btn" onClick={() => onOpenDraftBoard()}>
          Draft Class
        </button>
      ) : null}
      {typeof onLeave === "function" ? (
        <button type="button" className="wjc-desk-btn wjc-leave-desk" onClick={onLeave}>
          Leave Desk
        </button>
      ) : null}
    </div>
  );
}

export function DraftStockSidebar({ rows, onSelectPlayer }) {
  const draftRows = asArray(rows).filter(
    (row) =>
      row.prospect_classification !== "drafted_user" &&
      row.prospect_classification !== "tournament_npc"
  );
  const displayRows = draftRows.length
    ? draftRows
    : asArray(rows).filter((r) => r.prospect_classification === "drafted_user");

  const sorted = [...displayRows].sort(
    (a, b) =>
      Math.abs(Number(b.stock_delta) || 0) -
      Math.abs(Number(a.stock_delta) || 0)
  );

  return (
    <aside className="wjc-side-panel wjc-side-panel--stock" aria-label="Draft stock watch">
      <header className="wjc-side-panel__head">
        <span>Draft Risers</span>
        <em>WJC Board Movement</em>
      </header>
      {sorted.length === 0 ? (
        <p className="wjc-empty">No prospect data</p>
      ) : (
        <ul className="wjc-stock-sidebar-list wjc-scroll-panel">
          {sorted.slice(0, 20).map((row) => {
            const isDrafted = row.prospect_classification === "drafted_user";
            const delta = Number(row.stock_delta);
            const hasDelta = Number.isFinite(delta);
            const player = wjcPlayerHeadshot(row);
            const pts =
              row.tournament_pts ?? row.pts ?? null;
            const moveClass = !hasDelta
              ? ""
              : delta > 0
                ? " is-up"
                : delta < 0
                  ? " is-down"
                  : " is-flat";
            const moveText = !hasDelta
              ? "—"
              : delta > 0
                ? `▲ ${delta}`
                : delta < 0
                  ? `▼ ${Math.abs(delta)}`
                  : "— 0";

            return (
              <li key={row.player_id || row.name}>
                <button
                  type="button"
                  className="wjc-stock-sidebar-card"
                  onClick={() => onSelectPlayer?.(row)}
                  title={safeText(row.name)}
                >
                  <span className="wjc-stock-sidebar-card__shot">
                    <PlayerHeadshot
                      player={player}
                      size="sm"
                      variant="card"
                      flag={row.wjc_country}
                    />
                  </span>
                  <div className="wjc-stock-sidebar-card__body">
                    <strong>{safeText(row.name, "Unknown")}</strong>
                    <span className="wjc-stock-sidebar-card__meta">
                      <FlagImg code={row.wjc_country} size={16} />
                      <span>{abbrCode(row.wjc_country)}</span>
                      <span>{safeText(row.position, "—")}</span>
                      {row.age != null ? <span>Age {row.age}</span> : null}
                    </span>
                    {isDrafted ? (
                      <span className="wjc-stock-sidebar-card__owned">
                        Org · {safeText(row.owner_team_abbr, "YOU")}
                      </span>
                    ) : (
                      <div className="wjc-stock-sidebar-card__metrics">
                        <span title="Draft rank before / after WJC">
                          #{row.stock_before ?? "—"}
                          <i aria-hidden="true">→</i>
                          #{row.stock_after ?? "—"}
                        </span>
                        <span
                          className={`wjc-stock-sidebar-card__delta${moveClass}`}
                          title="Board spots gained or lost"
                        >
                          {moveText}
                        </span>
                        {pts != null ? (
                          <span className="wjc-stock-sidebar-card__pts">
                            {pts} PTS
                          </span>
                        ) : null}
                      </div>
                    )}
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}

export function StatLeadersSidebar({ leaders }) {
  const blocks = [
    { id: "pts", title: "Points", rows: asArray(leaders?.byPts), metric: "pts", label: "PTS" },
    { id: "g", title: "Goals", rows: asArray(leaders?.byGoals), metric: "g", label: "G" },
    { id: "pm", title: "Plus/Minus", rows: asArray(leaders?.byPm), metric: "plus_minus", label: "+/−" },
    { id: "teams", title: "Teams", rows: asArray(leaders?.teamLeaders), metric: "pts", label: "PTS", teams: true },
  ];
  const [activeId, setActiveId] = useState("pts");
  const active = blocks.find((b) => b.id === activeId) || blocks[0];

  return (
    <aside className="wjc-side-panel wjc-side-panel--leaders" aria-label="Tournament leaders">
      <header className="wjc-side-panel__head">
        <span>Tournament Leaders</span>
        <em>{active.title}</em>
      </header>
      <div className="wjc-leader-tabs" role="tablist" aria-label="Leader categories">
        {blocks.map((block) => (
          <button
            key={block.id}
            type="button"
            role="tab"
            aria-selected={activeId === block.id}
            className={activeId === block.id ? "is-active" : ""}
            onClick={() => setActiveId(block.id)}
          >
            {block.title}
          </button>
        ))}
      </div>
      <div className="wjc-scroll-panel">
        {active.teams ? (
          <table className="wjc-standings-table">
            <thead>
              <tr>
                <th>#</th>
                <th />
                <th>Team</th>
                <th>W</th>
                <th>L</th>
                <th>PTS</th>
              </tr>
            </thead>
            <tbody>
              {active.rows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="wjc-empty-inline">
                    No standings yet
                  </td>
                </tr>
              ) : (
                active.rows.slice(0, 10).map((row, i) => (
                  <tr key={row.code || i}>
                    <td>{i + 1}</td>
                    <td>
                      <FlagImg code={row.code} size={22} className="wjc-standings-table__flag" />
                    </td>
                    <td title={row.label || row.code}>{abbrCode(row.code)}</td>
                    <td>{row.w ?? 0}</td>
                    <td>{row.l ?? 0}</td>
                    <td>{row.pts ?? 0}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        ) : (
          <table className="wjc-mini-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Player</th>
                <th>Cty</th>
                <th>{active.label}</th>
              </tr>
            </thead>
            <tbody>
              {active.rows.length === 0 ? (
                <tr>
                  <td colSpan={4} className="wjc-empty-inline">
                    No stats yet
                  </td>
                </tr>
              ) : (
                active.rows.slice(0, 10).map((row, i) => (
                  <tr key={row.player_id || `${active.id}-${i}`}>
                    <td>{i + 1}</td>
                    <td title={safeText(row.name)}>{safeText(row.name, "—")}</td>
                    <td>{abbrCode(row.wjc_country)}</td>
                    <td>{row[active.metric] ?? 0}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        )}
      </div>
    </aside>
  );
}

export function ShowcaseOverlay({ card, onSelectPlayer }) {
  if (!card) return null;

  if (card.type === "player") {
    const player = wjcPlayerHeadshot(card);
    return (
      <button
        type="button"
        className="wjc-showcase wjc-showcase--player"
        aria-live="polite"
        onClick={() => onSelectPlayer?.(card)}
      >
        <PlayerHeadshot player={player} size="lg" variant="hero" flag={card.wjc_country} />
        <div className="wjc-showcase__stats">
          <strong>{card.name}</strong>
          <div className="wjc-showcase__nation">
            <FlagImg code={card.wjc_country} size={24} />
            <span>{card.wjc_country}</span>
          </div>
          <div className="wjc-showcase__grid">
            <span>{card.g}G</span>
            <span>{card.a}A</span>
            <span>{card.pts} PTS</span>
            <span>{card.gp} GP</span>
            <span>±{card.plus_minus}</span>
            <span>{card.sog} SOG</span>
          </div>
          {card.stock_before != null ? (
            <div className="wjc-showcase__stock">
              Stock {card.stock_before} → {card.stock_after}
            </div>
          ) : null}
        </div>
      </button>
    );
  }

  if (card.type === "nation") {
    return (
      <div className="wjc-showcase wjc-showcase--nation" aria-live="polite">
        <FlagImg code={card.code} size={56} className="wjc-showcase__flag" />
        <strong>{card.label || card.code}</strong>
        <span>
          {card.w}-{card.l} · {card.pts} PTS
        </span>
        <span>
          {card.gf} GF · {card.ga} GA
        </span>
      </div>
    );
  }

  if (card.type === "game") {
    const home = String(card.home || "?").slice(0, 3).toUpperCase();
    const away = String(card.away || "?").slice(0, 3).toUpperCase();
    return (
      <div className="wjc-showcase wjc-showcase--game" aria-live="polite">
        <em>{card.round}</em>
        <strong>
          {home} {card.home_goals} — {away} {card.away_goals}
        </strong>
      </div>
    );
  }

  return null;
}

export function BroadcastSubtitle({ line, activeSpeakerId }) {
  if (!line?.text) return null;
  const host = WJC_HOSTS[activeSpeakerId] || WJC_HOSTS.host_2;
  return (
    <div className={`wjc-subtitle wjc-subtitle--${activeSpeakerId}`} aria-live="polite">
      <span className="wjc-subtitle__name">{host.name}</span>
      <p>{line.text}</p>
    </div>
  );
}

export function ProspectDetailModal({ prospect, tournamentStats, franchiseState, onClose, onOpenDraftBoard }) {
  if (!prospect) return null;
  const player = wjcPlayerHeadshot(prospect);
  const wjc = tournamentStats || {};
  const prospectId = prospect.draft_prospect_id || prospect.player_id;
  const profiles = franchiseState?.draft_class_hud?.prospect_profiles_by_id || {};
  const profile = profiles[prospectId] || null;
  const isDraftEligible = prospect.prospect_classification === "draft_eligible";
  const isDraftedUser = prospect.prospect_classification === "drafted_user";
  const draftRank = prospect.stock_rank_before ?? profile?.rank ?? profile?.current_rank;
  const stockDelta = Number(prospect.stock_delta) || 0;
  const scoutPct = prospect.scouting_confidence ?? profile?.scouting_confidence;

  return (
    <div className="wjc-game-modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="wjc-game-modal wjc-prospect-modal"
        role="dialog"
        aria-label="Prospect profile"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <span>{prospect.wjc_country_label || prospect.wjc_country} U20</span>
          <button type="button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="wjc-prospect-modal__hero">
          <PlayerHeadshot player={player} size="lg" variant="hero" flag={prospect.wjc_country} />
          <div>
            <h3>{prospect.name}</h3>
            <p>
              {prospect.position || profile?.position || "F"} · Age {prospect.age ?? profile?.age ?? "—"}
            </p>
            {isDraftEligible && draftRank != null ? (
              <p className="wjc-prospect-modal__draft-rank">
                DRAFT RANK #{draftRank}
                {stockDelta !== 0 ? (
                  <span className={stockDelta > 0 ? "is-up" : "is-down"}>
                    {" "}
                    {stockDelta > 0 ? "UP" : "DOWN"} {Math.abs(stockDelta)} AT WJC
                  </span>
                ) : null}
              </p>
            ) : null}
            {isDraftedUser ? (
              <p className="wjc-prospect-modal__drafted">DRAFTED PROSPECT · {prospect.owner_team_abbr || "YOUR ORG"}</p>
            ) : null}
            {scoutPct != null ? <p>SCOUTED {Math.round(Number(scoutPct))}%</p> : null}
            <div className="wjc-prospect-modal__nation">
              <FlagImg code={prospect.wjc_country} size={28} />
              <span>{prospect.wjc_country_label}</span>
            </div>
          </div>
        </div>
        <div className="wjc-prospect-modal__grid">
          <article>
            <h4>Junior League</h4>
            <p>{prospect.junior_league || profile?.league || "—"}</p>
            <p>{prospect.junior_team || profile?.team || "—"}</p>
            <ul>
              <li>
                {prospect.junior_gp ?? profile?.gp ?? 0} GP · {prospect.junior_g ?? profile?.goals ?? 0}G ·{" "}
                {prospect.junior_a ?? profile?.assists ?? 0}A · {prospect.junior_pts ?? profile?.points ?? 0} PTS
              </li>
            </ul>
          </article>
          <article>
            <h4>World Juniors</h4>
            <ul>
              <li>
                {wjc.gp ?? prospect.tournament_gp ?? 0} GP · {wjc.g ?? prospect.tournament_g ?? 0}G · {wjc.a ?? 0}A ·{" "}
                {wjc.pts ?? prospect.tournament_pts ?? 0} PTS
              </li>
              <li>Plus-minus {wjc.plus_minus ?? 0}</li>
            </ul>
          </article>
          {isDraftEligible ? (
            <article>
              <h4>Draft Stock</h4>
              <p>
                #{prospect.stock_before ?? "—"} → #{prospect.stock_after ?? "—"}
              </p>
              <p className={stockDelta >= 0 ? "is-up" : "is-down"}>
                {stockDelta > 0 ? "▲" : stockDelta < 0 ? "▼" : "—"} {Math.abs(stockDelta)} board spots
              </p>
            </article>
          ) : null}
          {profile?.strengths?.length ? (
            <article>
              <h4>Strengths</h4>
              <p>{profile.strengths.slice(0, 4).join(" · ")}</p>
            </article>
          ) : null}
          {profile?.weaknesses?.length ? (
            <article>
              <h4>Weaknesses</h4>
              <p>{profile.weaknesses.slice(0, 3).join(" · ")}</p>
            </article>
          ) : null}
          {profile?.comparable ? (
            <article>
              <h4>Comparable</h4>
              <p>{profile.comparable}</p>
            </article>
          ) : null}
        </div>
        {isDraftEligible && typeof onOpenDraftBoard === "function" ? (
          <footer className="wjc-prospect-modal__actions">
            <button
              type="button"
              className="wjc-desk-btn"
              onClick={() => onOpenDraftBoard(prospectId)}
            >
              View On Draft Board
            </button>
          </footer>
        ) : null}
      </div>
    </div>
  );
}

export function GameResultModal({ game, onClose, formatScoreLine, gameCode }) {
  if (!game) return null;
  const box = game.box_score || {};
  const homeLines = asArray(box.home);
  const awayLines = asArray(box.away);

  return (
    <div className="wjc-game-modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="wjc-game-modal"
        role="dialog"
        aria-label="Game result"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <span>
            Day {game.game_day || "—"} · {game.round || "FINAL"}
          </span>
          <button type="button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="wjc-game-modal__score">
          <div>
            <FlagImg code={game.home} size={28} />
            <strong>{game.home_label || game.home}</strong>
            <b>{game.home_goals}</b>
          </div>
          <div>
            <FlagImg code={game.away} size={28} />
            <strong>{game.away_label || game.away}</strong>
            <b>{game.away_goals}</b>
          </div>
        </div>
        {homeLines.length || awayLines.length ? (
          <div className="wjc-game-modal__box">
            <div>
              <h4>{gameCode(game, "home")} Skaters</h4>
              <ul>
                {homeLines.map((r) => (
                  <li key={r.player_id}>
                    {r.name} — {r.g}G {r.a}A ({r.pts} PTS)
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4>{gameCode(game, "away")} Skaters</h4>
              <ul>
                {awayLines.map((r) => (
                  <li key={r.player_id}>
                    {r.name} — {r.g}G {r.a}A ({r.pts} PTS)
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p className="wjc-empty">Box score not available for this game.</p>
        )}
        <p className="wjc-game-modal__line">{formatScoreLine(game)}</p>
      </div>
    </div>
  );
}

export function GamesBrowser({ games, onSelectGame, formatScoreLine }) {
  const list = asArray(games);
  const byDay = {};
  list.forEach((g) => {
    const d = Number(g.game_day) || 0;
    if (!byDay[d]) byDay[d] = [];
    byDay[d].push(g);
  });
  const dayKeys = Object.keys(byDay)
    .map(Number)
    .sort((a, b) => a - b);

  const completedCount = list.filter(gameIsComplete).length;

  return (
    <section className="wjc-games-browser" aria-label="Tournament games">
      {list.length === 0 ? (
        <p className="wjc-empty">No games scheduled. Use Sim Day to advance.</p>
      ) : (
        <div className="wjc-games-browser__scroll">
          <p className="wjc-games-browser__summary">
            <span>{completedCount} completed</span>
            <span aria-hidden="true">·</span>
            <span>{list.length} total</span>
          </p>
          {dayKeys.map((day) => (
            <div key={day} className="wjc-games-day-group">
              <h4>Day {day || "—"}</h4>
              <div className="wjc-games-browser__list">
                {byDay[day].map((g, i) => {
                  const complete = gameIsComplete(g);
                  const homeCode = abbrCode(g.home || g.home_label);
                  const awayCode = abbrCode(g.away || g.away_label);
                  const hg = g.home_goals;
                  const ag = g.away_goals;
                  const homeWins = complete && Number(hg) > Number(ag);
                  const awayWins = complete && Number(ag) > Number(hg);
                  const status = gameStatusLabel(g);
                  const roundLabel = safeText(g.round, "Game");

                  return (
                    <button
                      key={`${g.home}-${g.away}-${day}-${i}`}
                      type="button"
                      className={`wjc-fixture-row${complete ? " is-final" : " is-upcoming"}`}
                      onClick={() => onSelectGame?.(g)}
                      aria-label={`${roundLabel}: ${awayCode} versus ${homeCode}, ${status}`}
                    >
                      <span className="wjc-fixture-row__stage">
                        <em>{roundLabel}</em>
                      </span>
                      <span className="wjc-fixture-row__matchup">
                        <span className={`wjc-fixture-team${awayWins ? " is-winner" : ""}`}>
                          <FlagImg code={g.away} size={28} />
                          <strong>{awayCode}</strong>
                          <b>{complete ? ag : "—"}</b>
                        </span>
                        <span className="wjc-fixture-row__sep" aria-hidden="true">
                          {complete ? "—" : "vs"}
                        </span>
                        <span className={`wjc-fixture-team${homeWins ? " is-winner" : ""}`}>
                          <b>{complete ? hg : "—"}</b>
                          <strong>{homeCode}</strong>
                          <FlagImg code={g.home} size={28} />
                        </span>
                      </span>
                      <span className="wjc-fixture-row__status">{status}</span>
                      <span className="wjc-sr-only">{formatScoreLine?.(g)}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
