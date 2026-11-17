import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

/**
 * Standalone negotiation pop-up used by the Re-sign desk and Free Agency.
 * It takes over the whole screen (portal on <body>) so the offer, the conversation
 * and the player's file each get a full column instead of being squeezed into a side panel.
 */
export default function NegotiationRoom({ open = true, kicker = "Negotiation", title = "Negotiation room", onClose, children }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  if (!open || typeof document === "undefined") return null;
  return createPortal(
    <div className="nroom" role="dialog" aria-modal="true" aria-label={title}>
      <header className="nroom-bar">
        <div className="nroom-bar__title">
          <span>{kicker}</span>
          <strong>{title}</strong>
        </div>
        <button type="button" className="nroom-close" onClick={onClose} aria-label="Leave the room">
          Leave the room <b aria-hidden="true">✕</b>
        </button>
      </header>
      <div className="nroom-body">{children}</div>
    </div>,
    document.body
  );
}

/** Chat-style transcript: agent on the left, you on the right, GM notes as pixel captions. */
export function NegotiationTranscript({ lines = [], pending = false, agentName = "Agent" }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length, pending]);
  const initials = String(agentName || "A")
    .split(/\s+/)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  return (
    <div className="nroom-chat" ref={ref} aria-live="polite">
      {lines.map((l, i) => (
        <div key={`${i}-${l.who}`} className={`nroom-line is-${l.who}`}>
          {l.who === "agent" || l.who === "player" ? (
            <span className="nroom-line__face" aria-hidden="true">
              {l.who === "agent" ? initials : "#"}
            </span>
          ) : null}
          <div className="nroom-line__bubble">
            {l.who !== "note" && l.name ? <span className="nroom-line__who">{l.name}</span> : null}
            <p>{l.text}</p>
          </div>
        </div>
      ))}
      {pending ? (
        <div className="nroom-line is-agent is-typing">
          <span className="nroom-line__face" aria-hidden="true">{initials}</span>
          <div className="nroom-line__bubble">
            <p>
              <i />
              <i />
              <i />
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function money(v) {
  const n = Number(v);
  return Number.isFinite(n) ? `$${n.toFixed(2)}M` : "—";
}

function pick(list, seed) {
  if (!list.length) return "";
  return list[Math.abs(seed) % list.length];
}

function hash(text) {
  let h = 0;
  for (const ch of String(text || "")) h = (h * 31 + ch.charCodeAt(0)) | 0;
  return h;
}

function firstName(name) {
  const s = String(name || "").trim();
  return s.split(/\s+/)[0] || "He";
}

/** How you put the offer across the table, in words. */
export function offerLine({ aav, years, clause = "", bonus = 0, kind = "" }) {
  const bits = [`${money(aav)} a year for ${years} ${Number(years) === 1 ? "year" : "years"}`];
  if (clause && clause !== "None") bits.push(`with a ${clause}`);
  if (Number(bonus) > 0) bits.push(`and ${money(bonus)} up front`);
  if (kind) bits.push(`(${kind})`);
  return `Here's where we are: ${bits.join(" ")}.`;
}

/** The agent's reply to a submitted offer or a "talk to agent" read. */
export function replyLine(result, { counter = null, fallback = "" } = {}) {
  const pr = result?.player_response || {};
  const ev = result?.evaluation || {};
  const spoken = ev.agent_dialogue || pr.agent_dialogue || pr.feedback || ev.reason || result?.reason || "";
  const status = String(result?.status || pr.status || "").toLowerCase();
  if (status === "accepted") return spoken || "You've got a deal. Send the papers over.";
  if (status === "countered" && counter) {
    const clause = counter.nmc ? " with an NMC" : counter.ntc_mode === "MODIFIED" ? " with a modified NTC" : counter.ntc_mode === "FULL" || counter.ntc ? " with an NTC" : "";
    return `${spoken ? `${spoken} ` : ""}We'd sign at ${money(counter.aav_m)} for ${counter.years} ${counter.years === 1 ? "year" : "years"}${clause}.`;
  }
  if (status === "pending") return spoken || "Let me take it to him. Give us a day.";
  if (status === "rejected") return spoken || "That's not going to get it done.";
  return spoken || fallback;
}

/**
 * Opening conversation for a meeting, built from what the club knows going in.
 * kind: "re_sign" | "ufa" | "offer_sheet"
 */
export function openingLines({
  kind = "re_sign",
  playerName = "",
  agentName = "",
  agentStyle = "",
  difficulty = "medium",
  askAav = null,
  askYears = null,
  clause = "",
  interest = "",
  age = null,
  ovr = null,
  competing = 0,
  circling = 0,
  signLikelihood = null,
  rightsTeam = "",
  acceptCut = null,
}) {
  const seed = hash(playerName);
  const first = firstName(playerName);
  const agent = agentName || "His agent";
  const lines = [];
  const lvl = String(interest || "").toLowerCase();

  if (kind === "offer_sheet") {
    lines.push({ who: "note", text: `${playerName} is a restricted free agent. ${rightsTeam || "His club"} can match anything you tender.` });
    lines.push({
      who: "agent",
      name: agent,
      text: pick(
        [
          `If you're serious, put it in writing. ${rightsTeam || "They"} will have a week to match.`,
          `An offer sheet gets everyone's attention. Make it big enough that matching hurts.`,
          `${first} wants to play. A real number forces ${rightsTeam || "his club"}'s hand.`,
        ],
        seed
      ),
    });
    return lines;
  }

  // Greeting coloured by how keen he is.
  const warm = /high|very|strong|elite/.test(lvl);
  const cold = /low|weak|cold/.test(lvl);
  if (kind === "re_sign") {
    lines.push({
      who: "agent",
      name: agent,
      text: warm
        ? pick([`${first} loves it here. Let's not make this complicated.`, `He wants to stay, and his family's settled. Let's find the number.`, `Good news first: ${first} wants to be part of what you're building.`], seed)
        : cold
          ? pick([`I'll be honest with you: ${first} is curious what's out there on July 1.`, `He's open to staying, but he's not giving you a hometown rate.`, `We've had a lot of calls. Convince him this is still the right room.`], seed)
          : pick([`${first} is open to staying. It comes down to term and money.`, `He likes the group. Show him you value him and we'll get there.`, `We're here to listen. He'd prefer to keep this off the open market.`], seed),
    });
  } else {
    lines.push({
      who: "agent",
      name: agent,
      text: competing > 0
        ? `${first} already has ${competing} ${competing === 1 ? "offer" : "offers"} on the table, so let's be quick about this.`
        : circling > 0
          ? `Our phone's been busy: ${circling} ${circling === 1 ? "club has" : "clubs have"} called about ${first}.`
          : pick([`Thanks for the call. ${first} is weighing where he fits best.`, `${first} wants a real role and a team that can win. Tell me about both.`], seed),
    });
  }

  if (askAav != null) {
    const clauseTxt = clause && clause !== "None" ? `, and the ${clause} isn't optional` : "";
    lines.push({
      who: "agent",
      name: agent,
      text: `We're looking at ${money(askAav)} a year${askYears ? ` over ${askYears} ${Number(askYears) === 1 ? "year" : "years"}` : ""}${clauseTxt}.`,
    });
  }

  if (difficulty === "hard") {
    lines.push({ who: "agent", name: agent, text: pick(["I don't negotiate against myself. Bring a real offer.", "Lowball me and we walk.", "Every dollar under the ask costs you."], seed + 1) });
  } else if (difficulty === "easy") {
    lines.push({ who: "agent", name: agent, text: pick(["We're flexible on the structure if the total is right.", "Get close and we'll meet you in the middle.", "He'd rather sign than drag this out."], seed + 1) });
  }

  const notes = [];
  if (ovr != null) notes.push(`${ovr} OVR`);
  if (age != null) notes.push(`age ${age}`);
  if (agentStyle) notes.push(`agent: ${String(agentStyle).toLowerCase()}`);
  if (signLikelihood != null) notes.push(`${Math.round(Number(signLikelihood) * 100)}% likely to sign somewhere this window`);
  if (acceptCut != null) notes.push(`interest needs to reach ${Math.round(acceptCut)} to sign`);
  if (notes.length) lines.push({ who: "note", text: `GM notebook: ${notes.join(" · ")}.` });
  return lines;
}
