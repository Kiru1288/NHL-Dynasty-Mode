import React, { useMemo } from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { collectLockerPulse } from "../../utils/lockerRoomPulse";

/* Room & Life: locker-room culture, off-ice story beats, recent room triggers and
   who each player is. Everything comes from the save (narrative_universe hallmark
   panels, human dossiers, storyline events); nothing here is invented. */

const arr = (v) => (Array.isArray(v) ? v : []);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const s = (v, f = "") => (v == null || v === "" ? f : String(v));
const label = (k) => s(k).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const LOWER_IS_BETTER = new Set(["tension", "friction", "toxicity", "drama"]);
const CULTURE_HINT = {
  unity: "Pulling the same way",
  tension: "Friction in the room",
  confidence: "Belief in the group",
  accountability: "Owning mistakes",
  work_ethic: "Practice habits",
  leadership: "Voices that steer it",
  belonging: "Feeling at home",
};

function cultureTone(key, v) {
  const n = Number(v) || 0;
  const good = LOWER_IS_BETTER.has(key) ? n <= 30 : n >= 65;
  const bad = LOWER_IS_BETTER.has(key) ? n >= 60 : n < 40;
  return good ? "good" : bad ? "bad" : "mid";
}

function useRosterPortraits(franchiseState) {
  return useMemo(() => {
    const byId = new Map();
    const byName = new Map();
    const add = (p) => {
      if (!p || typeof p !== "object") return;
      const id = s(p.id || p.player_id);
      const name = s(p.name || p.player_name).toLowerCase();
      if (id && !byId.has(id)) byId.set(id, p);
      if (name && !byName.has(name)) byName.set(name, p);
    };
    arr(franchiseState?.roster).forEach(add);
    arr(franchiseState?.roster_browser?.organizations).forEach((org) => arr(org?.players).forEach(add));
    return { byId, byName };
  }, [franchiseState]);
}

function Face({ portrait, id, name, size = 72 }) {
  const player = ensurePlayerHeadshotFields({ ...(portrait || {}), id: id || portrait?.id, player_id: id || portrait?.player_id, name });
  return (
    <div className="bbc-face" style={{ "--bbc-face": `${size}px` }}>
      <PlayerHeadshot player={player} size="lg" preferPhoto />
    </div>
  );
}

function SectionHead({ title, count, note }) {
  return (
    <div className="bbc-sechead">
      <h3 className="bbc-h bbc-h--sm">{title}</h3>
      {count != null ? <span className="bbc-count">{count}</span> : null}
      {note ? <span className="bbc-sechead__note">{note}</span> : null}
    </div>
  );
}

export default function RoomAndLifeDesk({ narrativeUniverse, franchiseState, stories = [], clubLabel = "", emptyState = null }) {
  const portraits = useRosterPortraits(franchiseState);
  const hallmark = obj(narrativeUniverse?.hallmark_panels);
  const culture = obj(hallmark.room_pulse || narrativeUniverse?.locker_room?.culture);
  const risks = arr(hallmark.character_risks);
  const leaders = arr(hallmark.unheralded_leaders);
  const pulse = collectLockerPulse(franchiseState, { limit: 12 });
  const uid = s(franchiseState?.user_team_id);

  const portraitFor = (id, name) =>
    (id && portraits.byId.get(s(id))) || (name && portraits.byName.get(s(name).toLowerCase())) || null;

  const roomWire = stories.filter((st) => {
    if (st.categoryKey === "personal_life" || st.categoryKey === "locker_room") return true;
    const tokens = s(st.category || st.type).toLowerCase().split(/[^a-z0-9]+/);
    if (tokens.includes("press")) return false;
    return tokens.includes("locker") || tokens.includes("personal") || tokens.includes("life");
  });
  const beats = (pulse.lifeStories.length ? pulse.lifeStories : roomWire.slice(0, 12)).map((row) => ({
    id: s(row.id || row.headline),
    name: s(row.playerName),
    playerId: s(row.playerId || row.raw?.player_id),
    headline: s(row.headline),
    summary: s(row.summary),
    tag: row.categoryKey === "locker_room" ? "Room" : "Off ice",
  }));
  const triggers = arr(narrativeUniverse?.recent_universe_events)
    .filter((ev) => !uid || s(ev?.team_id) === uid)
    .slice(-8)
    .reverse();

  const cultureRows = Object.entries(culture).filter(([, v]) => v != null && v !== "" && Number.isFinite(Number(v)));
  const pulseParts = [culture.unity, culture.confidence, culture.belonging].filter((v) => v != null && v !== "");
  const pulseScore = pulseParts.length
    ? Math.round(pulseParts.reduce((sum, v) => sum + Number(v), 0) / pulseParts.length)
    : null;
  const pulseWord = pulseScore == null ? "" : pulseScore >= 70 ? "Tight room" : pulseScore >= 55 ? "Steady" : pulseScore >= 40 ? "Uneasy" : "Fractured";

  if (!cultureRows.length && !risks.length && !leaders.length && !pulse.people.length && !beats.length && !triggers.length) {
    return emptyState;
  }

  const hasSide = triggers.length || risks.length || leaders.length;

  return (
    <div className="bbc-room">
      <header className="bbc-room-head">
        <div className="bbc-room-head__title">
          <span className="bbc-label">Culture · {clubLabel || "Your club"}</span>
          <h2 className="bbc-h">Room &amp; Life</h2>
          <p>Locker-room pulse, character profiles and off-ice beats tied to your roster.</p>
        </div>
        {pulseScore != null ? (
          <div className={`bbc-gauge is-${cultureTone("unity", pulseScore)}`} style={{ "--pulse": `${pulseScore}%` }}>
            <strong>{pulseScore}</strong>
            <span>Room pulse</span>
            <em>{pulseWord}</em>
          </div>
        ) : null}
        {cultureRows.length ? (
          <div className="bbc-culture">
            {cultureRows.map(([key, val]) => {
              const n = Math.round(Number(val) || 0);
              const tone = cultureTone(key, n);
              return (
                <div key={key} className={`bbc-culture__item is-${tone}`}>
                  <span className="bbc-culture__name">{label(key)}</span>
                  <strong>{n}</strong>
                  <i className="bbc-culture__bar" aria-hidden>
                    <b style={{ width: `${Math.max(3, Math.min(100, n))}%` }} />
                  </i>
                  {CULTURE_HINT[key] ? <em>{CULTURE_HINT[key]}</em> : null}
                </div>
              );
            })}
          </div>
        ) : null}
      </header>

      <div className={`bbc-room-main${hasSide ? "" : " is-solo"}`}>
        <section className="bbc-room-beats">
          <SectionHead title="Story beats" count={beats.length} note="Off the ice and around the room" />
          {beats.length ? (
            <div className="bbc-beats">
              {beats.map((b) => {
                const p = portraitFor(b.playerId, b.name);
                return (
                  <article key={b.id} className="bbc-paper bbc-beat">
                    {b.name ? <Face portrait={p} id={b.playerId || p?.id} name={b.name} size={76} /> : null}
                    <div className="bbc-beat__body">
                      <span className="bbc-stamp">{b.tag}</span>
                      {b.name ? <span className="bbc-beat__who">{b.name}</span> : null}
                      <strong>{b.headline}</strong>
                      {b.summary && b.summary !== b.headline ? <p>{b.summary}</p> : null}
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <p className="bbc-quiet">No off-ice or locker-room stories on the wire yet. They show up as the days go by.</p>
          )}
        </section>

        {hasSide ? (
          <aside className="bbc-room-side">
            {triggers.length ? (
              <section>
                <SectionHead title="Recent triggers" count={triggers.length} />
                <ul className="bbc-triggers">
                  {triggers.map((ev, i) => {
                    const name = s(ev.player_name);
                    const p = portraitFor(ev.player_id, name);
                    return (
                      <li key={s(ev.id || `${ev.kind}-${i}`)} className="bbc-paper bbc-trigger">
                        {name ? <Face portrait={p} id={s(ev.player_id) || p?.id} name={name} size={52} /> : null}
                        <div>
                          <span className="bbc-stamp is-small">{label(ev.kind || ev.type || "event")}</span>
                          <strong>{name || s(ev.headline, "Roster event")}</strong>
                          <p>{s(ev.summary || ev.headline || ev.description, "A character or room trigger fired.")}</p>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ) : null}
            {leaders.length ? (
              <section>
                <SectionHead title="Unheralded leaders" count={leaders.length} />
                <ul className="bbc-mini">
                  {leaders.map((row) => (
                    <li key={s(row.player_id)} className="bbc-paper bbc-mini__row is-good">
                      <Face portrait={portraitFor(row.player_id, row.player_name)} id={s(row.player_id)} name={s(row.player_name)} size={52} />
                      <div>
                        <strong>{s(row.player_name, "Player")}</strong>
                        <span>Room value {Math.round(Number(row.room_value) || 0)}</span>
                        <div className="bbc-traits">
                          {arr(row.niche_abilities).map((n) => (
                            <em key={s(n.id)}>{s(n.label || n.id)}</em>
                          ))}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
            {risks.length ? (
              <section>
                <SectionHead title="Character risks" count={risks.length} />
                <ul className="bbc-mini">
                  {risks.map((row) => (
                    <li key={s(row.player_id)} className="bbc-paper bbc-mini__row is-bad">
                      <Face portrait={portraitFor(row.player_id, row.player_name)} id={s(row.player_id)} name={s(row.player_name)} size={52} />
                      <div>
                        <strong>{s(row.player_name, "Player")}</strong>
                        <span>Disruption risk {Math.round(Number(row.disruption_risk) || 0)}</span>
                        <div className="bbc-traits">
                          {arr(row.niche_abilities).map((n) => (
                            <em key={s(n.id)}>{s(n.label || n.id)}</em>
                          ))}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </aside>
        ) : null}
      </div>

      {pulse.people.length ? (
        <section className="bbc-room-people">
          <SectionHead title="Who they are" count={pulse.people.length} note="Character, life away from the rink, and mood" />
          <div className="bbc-people">
            {pulse.people.map((row) => (
              <article key={row.playerId || row.name} className="bbc-paper bbc-person">
                <Face portrait={row.portrait || portraitFor(row.playerId, row.name)} id={row.playerId} name={row.name} size={96} />
                <div className="bbc-person__body">
                  <strong>{row.name}</strong>
                  <span className="bbc-person__meta">
                    {row.position ? `${row.position} · ` : ""}
                    {row.character || "Profile"}
                  </span>
                  {row.line ? <p>{row.line}</p> : null}
                  {row.chips.length ? (
                    <div className="bbc-traits">
                      {row.chips.map((c) => (
                        <em key={c}>{c}</em>
                      ))}
                    </div>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
