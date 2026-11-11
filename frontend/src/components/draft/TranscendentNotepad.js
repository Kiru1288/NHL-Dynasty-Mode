import React, { useEffect, useState } from "react";
import { PixelPencil } from "../setup/pixelSprites";

/*
  Transcendent reveal: a large, translucent sealed-file notepad hand-writes
  suspense about the prospect over the live board, then the name lands in red
  ink and the stamp comes down. Calls onDone once the file is written and held.
*/

const CHAR_MS = 28;
const LINE_PAUSE_MS = 380;
const HOLD_MS = 2400;

export default function TranscendentNotepad({ name, headline, onDone }) {
  const script = [
    { kind: "head", text: "SEALED FILE - CLASS OF 2027" },
    { kind: "body", text: "Scouts have stopped saying the word prospect." },
    { kind: "body", text: "Three front offices quietly changed their tank plans." },
    { kind: "body", text: "The public board ranks him nowhere. The public is wrong." },
    { kind: "body", text: "Skates like the ice owes him money." },
    { kind: "body", text: "Odds of this talent: 1 in 10,000." },
    { kind: "body", text: "Breaking the seal..." },
    { kind: "reveal", text: `Subject: ${name}` },
    { kind: "reveal", text: "Now 99. Ceiling 99. No character flags." },
    { kind: "reveal", text: headline || "The lottery is the season." },
  ];
  const [idx, setIdx] = useState(0);
  const [n, setN] = useState(0);
  const done = idx >= script.length;

  useEffect(() => {
    if (done) return undefined;
    const line = script[idx];
    const full = line.text.length;
    const delay = n < full ? CHAR_MS : LINE_PAUSE_MS;
    const t = window.setTimeout(() => {
      if (n < full) setN(n + 1);
      else { setIdx(idx + 1); setN(0); }
    }, delay);
    return () => window.clearTimeout(t);
  }, [idx, n, done]);

  useEffect(() => {
    if (!done) return undefined;
    const t = window.setTimeout(() => { if (onDone) onDone(); }, HOLD_MS);
    return () => window.clearTimeout(t);
  }, [done]);

  return (
    <div className="tn-wrap" role="dialog" aria-label="Transcendent scouting file">
      <div className="tn-pad">
        <div className="tn-spiral" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
        <div className="tn-head">
          <b>TRANSCENDENT FILE</b>
          <span>1 in 10,000</span>
          <em className="tn-sealed">Sealed</em>
        </div>
        <div className="tn-body">
          {script.slice(0, Math.min(idx + 1, script.length)).map((line, i) => {
            const isCur = i === idx && !done;
            const text = i < idx ? line.text : line.text.slice(0, n);
            return (
              <p key={i} className={`tn-line tn-${line.kind}`}>
                {text}{isCur ? "_" : ""}
              </p>
            );
          })}
        </div>
        <PixelPencil className="tn-pencil" />
        {done ? <div className="tn-stamp">TRANSCENDENT</div> : null}
      </div>
    </div>
  );
}
