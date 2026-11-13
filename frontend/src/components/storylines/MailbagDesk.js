import React, { useEffect, useState } from "react";
import { arr, s, num, Face, DeskHead, DeskMissing } from "./benchBuzzShared";
import { answerBenchBuzzMailbag } from "../../services/franchiseService";

/* Weekly fan mailbag: letters built from the club's real situation. Each can be
   answered once; replies move fan confidence / media pressure like Press Room. */

function Letter({ letter, busy, onAnswer, idx }) {
  const done = letter.answered;
  return (
    <article className={`bbe-letter ${done ? "is-answered" : ""}`} style={{ "--bbe-tilt": `${idx % 2 ? 0.5 : -0.5}deg` }}>
      <header className="bbe-letter__top">
        <span className="bbe-letter__stamp" aria-hidden>✉</span>
        <div className="bbe-letter__from">
          <b>{s(letter.author)}</b>
          <i>{s(letter.town)}</i>
        </div>
        {letter.player_id ? <Face row={{ ...letter, name: letter.player_name }} px={72} /> : null}
      </header>
      <h3 className="bbe-letter__subject">{s(letter.subject)}</h3>
      <p className="bbe-letter__body">“{s(letter.body)}”</p>
      {arr(letter.facts).filter(Boolean).length ? (
        <ul className="bbe-facts" aria-label="From the file">
          {arr(letter.facts).filter(Boolean).map((f) => <li key={f}>{f}</li>)}
        </ul>
      ) : null}
      {done ? (
        <div className="bbe-sent">
          <span className="ccx-stamp is-good">Sent · {s(done.label)}</span>
          <p>“{s(done.text)}”</p>
          <small>
            {s(done.effect_preview)}
            {num(done.fan_confidence_before) != null && num(done.fan_confidence_after) != null
              ? ` · Fans ${Math.round(done.fan_confidence_before)} → ${Math.round(done.fan_confidence_after)}`
              : ""}
          </small>
        </div>
      ) : (
        <div className="bbe-replies">
          {arr(letter.replies).map((r) => (
            <button key={r.id} type="button" className={`bbe-reply is-${s(r.tone)}`} disabled={busy} onClick={() => onAnswer(letter.id, r.id)}>
              <b>{s(r.label)}</b>
              <span>“{s(r.text)}”</span>
              <small>{s(r.effect_preview)}</small>
            </button>
          ))}
        </div>
      )}
    </article>
  );
}

export default function MailbagDesk({ features, onRefresh }) {
  const [local, setLocal] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const incoming = features?.mailbag;
  useEffect(() => {
    setLocal(null);
  }, [incoming?.week_key, incoming?.answered_count]);
  const data = local || incoming;
  if (!data) return <DeskMissing kicker="Reader letters" title="The Mailbag" />;
  if (data.error) {
    return (
      <section className="bbe-desk">
        <DeskHead kicker="Reader letters" title="The Mailbag" />
        <div className="bbe-empty"><strong>The mailbag didn't sort this week.</strong><p>Advance a day and the letters get re-sorted.</p></div>
      </section>
    );
  }
  const letters = arr(data.letters);
  const conf = num(data.fan_confidence);
  const prev = num(data.fan_confidence_last_week);
  const delta = conf != null && prev != null ? Math.round((conf - prev) * 10) / 10 : null;
  const answered = letters.filter((l) => l.answered).length;

  async function handleAnswer(letterId, replyId) {
    setBusy(true);
    setErr("");
    try {
      const res = await answerBenchBuzzMailbag(letterId, replyId);
      if (res?.bench_buzz_features?.mailbag) setLocal(res.bench_buzz_features.mailbag);
      onRefresh?.();
    } catch (e) {
      setErr(s(e?.response?.data?.detail || e?.message, "The reply didn't go through."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="bbe-desk bbe-mail-desk">
      <DeskHead
        kicker={`Reader letters · ${s(data.week_label)}`}
        title="The Mailbag"
        lede={`Fans of the ${s(data.team_name, "club")} write in about what they're seeing. Every letter is built from this week's record, stats and cap sheet. Pick one reply per letter.`}
      />
      {err ? <p className="bbe-error" role="alert">{err}</p> : null}
      <div className="bbe-mail-layout">
        <div className="bbe-letters">
          {letters.map((l, i) => <Letter key={l.id} letter={l} idx={i} busy={busy} onAnswer={handleAnswer} />)}
          {!letters.length ? <div className="bbe-empty"><strong>Empty mailbag.</strong><p>Letters arrive once there's something to write about.</p></div> : null}
        </div>
        <aside className="bbe-mail-rail">
          <div className="bbe-meter">
            <span className="ccx-kicker">Fan confidence</span>
            <strong className="bbe-bignum">{conf != null ? Math.round(conf) : "—"}</strong>
            <div className="bbe-meter__bar" aria-hidden><i style={{ width: `${Math.max(0, Math.min(100, conf || 0))}%` }} /></div>
            <p>
              {delta == null ? "No weekly reading yet." : delta === 0 ? "Unchanged since last week." : `${delta > 0 ? "Up" : "Down"} ${Math.abs(delta)} since last week.`}
              {num(data.fan_trust_in_gm) != null ? ` Trust in the GM: ${Math.round(data.fan_trust_in_gm)}.` : ""}
            </p>
            <div className="bbe-tally">
              <span><b>{answered}</b>answered</span>
              <span><b>{letters.length - answered}</b>waiting</span>
            </div>
          </div>
          <div className="bbe-sheet">
            <div className="bbe-sheet__top"><span className="ccx-kicker">Past replies</span></div>
            <ul className="bbe-history">
              {arr(data.history).map((h) => (
                <li key={h.letter_id}>
                  <b>{s(h.subject)}</b>
                  <i>{s(h.author)}, {s(h.town)} · you chose “{s(h.label)}”</i>
                  <small>{s(h.effect_preview)}</small>
                </li>
              ))}
              {!arr(data.history).length ? <li className="is-empty">Your earlier replies will be filed here.</li> : null}
            </ul>
          </div>
          <p className="bbe-footnote">{s(data.effect_note)}</p>
        </aside>
      </div>
    </section>
  );
}
