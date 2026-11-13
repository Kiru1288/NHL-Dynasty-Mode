import React from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";

/* Press Room: media availabilities about your club. Each question shows who is asking
   (reporter, outlet, beat, how they get on with the player), the question as a speech
   bubble, and the answers side by side with a readable effect preview. */

const arr = (v) => (Array.isArray(v) ? v : []);
const s = (v, f = "") => (v == null || v === "" ? f : String(v));

// Effects where a smaller number is good news for the GM.
const LOWER_IS_BETTER = /media pressure|media stress|story heat|trade (rumou?r|heat)|tension|stress|friction|grievance/i;

function parseEffects(preview) {
  return s(preview)
    .split(/\s*[·|,]\s*/)
    .map((part) => {
      const m = part.trim().match(/^(.*?)\s*([+\-\u2212]\s*\d+(?:\.\d+)?)$/);
      if (!m) return part.trim() ? { label: part.trim(), delta: null, tone: "flat" } : null;
      const delta = Number(m[2].replace(/\s/g, "").replace("\u2212", "-"));
      const label = m[1].trim();
      const up = delta > 0;
      const good = LOWER_IS_BETTER.test(label) ? !up : up;
      return { label, delta, tone: delta === 0 ? "flat" : good ? "good" : "bad" };
    })
    .filter(Boolean);
}

const TONE_WORD = {
  diplomatic: "Diplomatic",
  neutral: "Measured",
  firm: "Firm",
  cold: "Cold",
  supportive: "Supportive",
  defiant: "Defiant",
  honest: "Candid",
  humorous: "Light",
};

function initials(name) {
  return s(name, "?")
    .split(/\s+/)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function ReporterId({ q }) {
  const info = q.reporter_info || {};
  const mood = s(q.reporter_mood);
  const rel = s(info.relationship) || (mood === "hostile" ? "Has an edge" : mood === "friendly" ? "Friendly" : "");
  const relTone = /edge|wary/i.test(rel) ? "bad" : /friend/i.test(rel) ? "good" : "flat";
  return (
    <div className="bbc-press-who">
      <span className="bbc-press-mono" aria-hidden>{initials(q.reporter_name)}</span>
      <div className="bbc-press-whotext">
        <strong>{s(q.reporter_name, "Reporter")}</strong>
        {q.outlet ? <span className="bbc-press-outlet">{s(q.outlet)}</span> : null}
        {info.role ? <span className="bbc-press-beat">{s(info.role)}</span> : null}
        <div className="bbc-press-whostats">
          {info.credibility != null ? <em>Cred {Math.round(Number(info.credibility))}</em> : null}
          {rel ? <em className={`is-${relTone}`}>{rel}</em> : null}
        </div>
      </div>
    </div>
  );
}

function AnswerOption({ resp, disabled, busy, onClick }) {
  const tone = s(resp.tone).toLowerCase();
  const effects = parseEffects(resp.effect_preview || resp.effect_summary);
  return (
    <button type="button" className={`bbc-answer is-${tone || "neutral"}`} disabled={disabled} onClick={onClick}>
      <span className="bbc-answer__tone">{TONE_WORD[tone] || s(resp.tone, "Answer").replace(/_/g, " ")}</span>
      <strong className="bbc-answer__label">{s(resp.label)}</strong>
      {resp.description || resp.detail ? <span className="bbc-answer__desc">{s(resp.description || resp.detail)}</span> : null}
      {effects.length ? (
        <span className="bbc-answer__fx">
          {effects.map((fx, i) => (
            <i key={i} className={`is-${fx.tone}`}>
              {fx.label}
              {fx.delta != null ? <b>{fx.delta > 0 ? `+${fx.delta}` : fx.delta}</b> : null}
            </i>
          ))}
        </span>
      ) : null}
      {busy ? <em className="bbc-answer__busy">Answering…</em> : null}
    </button>
  );
}

function recordLine(ctx) {
  const c = ctx || {};
  const gp = Number(c.games_played);
  const rec = s(c.record);
  if (!rec || (Number.isFinite(gp) && gp <= 0) || rec === "0-0-0") return "No games played yet";
  const rank = c.league_rank != null && Number(c.league_rank) > 0 ? ` · #${Number(c.league_rank)} in the league` : "";
  return `Record ${rec}${rank}`;
}

export default function PressRoomDesk({
  queue = [],
  recent = [],
  calendarDay,
  busyChoice,
  outcome = null,
  emptyState = null,
  windowDays = 4,
  onRespond,
}) {
  const today = Number(calendarDay);
  const open = arr(queue);
  return (
    <div className="bbc-press">
      <header className="bbc-press-head">
        <div className="bbc-press-head__title">
          <span className="bbc-label">Media room · on the record</span>
          <h2 className="bbc-h">Press Room</h2>
          <p>
            When a story about your club heats up, reporters book a media availability. Every answer changes
            things for real, and you can see each effect before you pick.
          </p>
        </div>
        <div className="bbc-press-head__rules">
          <div className="bbc-press-levers">
            {["Fan confidence", "Player morale & trust", "Story heat", "Trade rumours", "Reporter relationships"].map((l) => (
              <span key={l}>{l}</span>
            ))}
          </div>
          <p>
            Availabilities close after {windowDays} days, two a week at most. Skipping one reads as a cold "no
            comment"; simming past one sends a short team statement at half the cost. Dodge a question and the
            reporter may follow up. Reporters remember what you said, and the player you talk about hears it.
          </p>
        </div>
        <div className="bbc-press-head__count">
          <strong>{open.length}</strong>
          <span>{open.length === 1 ? "Availability open" : "Availabilities open"}</span>
        </div>
      </header>

      {outcome}

      {open.length
        ? open.map((press) => {
            const daysLeft =
              press.expires_day != null && Number.isFinite(today) ? Math.max(0, Number(press.expires_day) - today) : null;
            const qs = arr(press.questions);
            const answered = new Set(arr(press.answered_questions).map((id) => s(id)));
            const coverage = new Map(arr(press.coverage).map((c) => [s(c.question_id), c]));
            const pid = s(press.player_id);
            return (
              <article key={s(press.id)} className="bbc-presser">
                <div className="bbc-presser__top">
                  {pid ? (
                    <div className="bbc-face bbc-presser__face" style={{ "--bbc-face": "112px" }}>
                      <PlayerHeadshot
                        player={ensurePlayerHeadshotFields({
                          ...(press.portrait || {}),
                          id: pid,
                          player_id: pid,
                          name: s(press.player_name),
                        })}
                        size="lg"
                        preferPhoto
                      />
                    </div>
                  ) : null}
                  <div className="bbc-presser__headline">
                    <span className="bbc-label">Media availability{press.player_name ? ` · about ${s(press.player_name)}` : ""}</span>
                    <h3 className="bbc-h bbc-h--md">{s(press.headline, "Media availability scheduled")}</h3>
                    {press.summary ? <p>{s(press.summary)}</p> : null}
                    <div className="bbc-presser__why">
                      <span className="bbc-presser__whylabel">Why the media is here</span>
                      <span className="bbc-presser__trigger is-record">{recordLine(press.context)}</span>
                      {arr(press.context_triggers).map((t) => (
                        <span key={s(t.code || t.label)} className="bbc-presser__trigger">
                          {s(t.label)}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="bbc-presser__status">
                    <div className="bbc-presser__stat">
                      <strong>
                        {answered.size}/{qs.length}
                      </strong>
                      <span>answered</span>
                    </div>
                    {daysLeft != null ? (
                      <div className={`bbc-presser__stat${daysLeft <= 1 ? " is-urgent" : ""}`}>
                        <strong>{daysLeft === 0 ? "Today" : daysLeft}</strong>
                        <span>{daysLeft === 0 ? "closes" : daysLeft === 1 ? "day left" : "days left"}</span>
                      </div>
                    ) : null}
                  </div>
                </div>

                <div className="bbc-presser__qs">
                  {qs.map((q, qi) => {
                    const done = answered.has(s(q.id));
                    const cov = coverage.get(s(q.id));
                    return (
                      <section key={s(q.id)} className={`bbc-q${done ? " is-done" : ""}`}>
                        <ReporterId q={q} />
                        <div className="bbc-q__main">
                          <div className="bbc-q__bubble">
                            <span className="bbc-q__num">{q.is_followup ? "Follow-up" : `Q${qi + 1}`}</span>
                            <p>{s(q.question)}</p>
                          </div>
                          {done ? (
                            <div className="bbc-q__said">
                              {cov?.quote ? (
                                <blockquote>“{s(cov.quote)}”</blockquote>
                              ) : (
                                <p>
                                  You answered: <b>{s(cov?.response_label, "on the record")}</b>
                                </p>
                              )}
                              {cov?.headline ? <p className="bbc-q__ran">Ran as: <em>{s(cov.headline)}</em></p> : null}
                              {cov?.reaction ? <p className="bbc-q__react">{s(cov.reaction)}</p> : null}
                            </div>
                          ) : (
                            <div className="bbc-q__answers">
                              {arr(q.responses).map((resp) => {
                                const sid = s(press.storyline_id || press.id);
                                const choiceId = `${s(q.id)}:${s(resp.id)}`;
                                return (
                                  <AnswerOption
                                    key={s(resp.id)}
                                    resp={resp}
                                    disabled={Boolean(busyChoice)}
                                    busy={busyChoice === `${sid}:${choiceId}`}
                                    onClick={() => onRespond?.(press, s(q.id), s(resp.id))}
                                  />
                                );
                              })}
                            </div>
                          )}
                        </div>
                      </section>
                    );
                  })}
                </div>
              </article>
            );
          })
        : emptyState}

      {arr(recent).length ? (
        <section className="bbc-press-recent">
          <h3 className="bbc-h bbc-h--sm">Recent coverage</h3>
          <div className="bbc-press-recent__grid">
            {arr(recent).map((press) => (
              <div key={s(press.id)} className={`bbc-clip is-${s(press.status)}`}>
                <span className="bbc-clip__stamp">{s(press.status) === "expired" ? "Skipped" : "Held"}</span>
                <strong>{s(press.headline, "Media availability")}</strong>
                {arr(press.coverage).map((c, i) => (
                  <p key={i}>
                    {c.quote ? <>“{s(c.quote)}” </> : c.response_label ? <b>{s(c.response_label)}. </b> : null}
                    {c.headline ? <em>{s(c.headline)}</em> : null}
                    {c.reaction ? <span className="bbc-clip__react"> {s(c.reaction)}</span> : null}
                  </p>
                ))}
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
