import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGovernance, lobbyGovernance, setExpansionProtection, voteGovernance } from "../../services/franchiseService";
import { getTeamLogoSrc, toLogoUrl } from "../../utils/teamLogos";
import { pickFranchiseData } from "../shared/eventHelpers";

/*
 * Board of Governors — annual offseason meeting.
 * Three stages: intro (agenda preview), floor (one proposal at a time), adjourned (recap).
 * Every number on screen comes from /api/franchise/governance (proposal.analysis is the
 * per-proposal read: effects on the user's club, who wins / loses, projected vote by club).
 * Older payloads without `analysis` fall back to proposal.effects / pros_cons / forecast.
 */

const P = "gov3";
const REVEAL_MS = 55;

const LEAN_LABEL = {
  for: "For",
  lean_for: "Leaning for",
  undecided: "Undecided",
  lean_against: "Leaning against",
  against: "Against",
};
const LEAN_ORDER = ["for", "lean_for", "undecided", "lean_against", "against"];
const SCALE_COPY = {
  minor: "Minor change: small numbers for every club",
  moderate: "Moderate change: some clubs will feel it",
  major: "Major change: real money or points move between clubs",
};
const FAN_TIER_ROWS = [
  ["Electric", "85+"],
  ["Loyal", "70–84"],
  ["Steady", "55–69"],
  ["Restless", "40–54"],
  ["Hostile", "under 40"],
];

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function num(v, d = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function money(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  return `$${n.toFixed(1)}M`;
}

function signedM(v) {
  const n = num(v);
  return `${n >= 0 ? "+" : "−"}$${Math.abs(n).toFixed(1)}M`;
}

function listText(arr) {
  const a = asArray(arr).filter(Boolean);
  if (a.length <= 1) return a.join("");
  return `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`;
}

function lowerFirst(s) {
  const t = String(s || "");
  return t ? t[0].toLowerCase() + t.slice(1) : t;
}

function logoFor(abbr, name) {
  return toLogoUrl(getTeamLogoSrc({ team_abbrev: abbr, abbr, name }));
}

function Logo({ abbr, name, size = 28, className = "" }) {
  const [bad, setBad] = useState(false);
  const src = bad ? "" : logoFor(abbr, name);
  return (
    <span className={`${P}-logo ${className}`} style={{ width: size, height: size }} title={name || abbr}>
      {src ? <img src={src} alt="" loading="lazy" onError={() => setBad(true)} /> : <span>{String(abbr || "").slice(0, 3)}</span>}
    </span>
  );
}

function tierTone(tier) {
  return `is-${String(tier || "steady").toLowerCase()}`;
}

function votesNeededFor(kind, n) {
  if (!n) return 0;
  if (kind === "majority") return Math.floor(n / 2) + 1;
  if (kind === "two_thirds") return Math.ceil((2 / 3) * n - 1e-9);
  return Math.ceil(0.75 * n - 1e-9);
}

function verdictText(club) {
  if (!club) return "";
  const who = club.abbr || "your club";
  if (club.verdict === "good") return `Good for ${who}`;
  if (club.verdict === "bad") return `Bad for ${who}`;
  return `Little direct effect on ${who}`;
}

function popularityText(pop) {
  if (pop == null) return "";
  if (pop >= 0.08) return "Popular with owners going in";
  if (pop <= -0.05) return "Owners are wary of it going in";
  return "No strong league-wide lean going in";
}

/* Case for / against, built from the rule's real effects and who each one helps. */
function buildCase(proposal) {
  const a = proposal.analysis || {};
  const pros = [];
  const cons = [];
  const trade = [];
  const effects = asArray(a.effects).length ? asArray(a.effects) : asArray(proposal.effects);
  const notes = new Set();
  effects.forEach((e) => {
    const head = `${e.label} ${e.text}`;
    const helps = asArray(e.helps);
    const hurts = asArray(e.hurts);
    if (e.note) notes.add(e.note);
    if (e.tone === "pro") {
      pros.push({ head, body: `${e.note || ""}${helps.length ? `. Matters most to ${lowerFirst(listText(helps))}` : ". Every club gains the same"}` });
      if (hurts.length) cons.push({ head, body: `${listText(hurts)} come out behind on it` });
    } else if (e.tone === "con") {
      cons.push({ head, body: `${e.note || ""}${hurts.length ? `. Hits ${lowerFirst(listText(hurts))} hardest` : ". Every club pays it"}` });
      if (helps.length) pros.push({ head, body: `${listText(helps)} come out ahead on it` });
    } else if (helps.length || hurts.length) {
      if (helps.length) pros.push({ head, body: `${e.note || ""}. Helps ${lowerFirst(listText(helps))}` });
      if (hurts.length) cons.push({ head, body: `Works against ${lowerFirst(listText(hurts))}` });
    } else {
      trade.push({ head, body: e.note || "" });
    }
  });
  const pc = proposal.pros_cons || {};
  asArray(pc.pros).forEach((t) => { if (!notes.has(t)) pros.push({ head: t, body: "" }); });
  asArray(pc.cons).forEach((t) => { if (!notes.has(t)) cons.push({ head: t, body: "" }); });
  asArray(pc.mixed).forEach((t) => { if (!notes.has(t)) trade.push({ head: t, body: "" }); });
  if (!cons.length) {
    cons.push({
      head: "No club type loses on paper",
      body: "Governors who vote no are owners who don't think it's worth changing the rulebook for.",
    });
  }
  if (!pros.length) {
    pros.push({ head: "No club type gains on paper", body: "Support comes from owners who like the idea itself." });
  }
  return { pros, cons, trade };
}

/* ---------------------------------------------------------------- ticker */

function FanTicker({ rows }) {
  const list = asArray(rows);
  if (!list.length) return null;
  const items = list.map((r) => {
    const d = num(r.delta);
    const dir = d > 0.4 ? "up" : d < -0.4 ? "down" : "flat";
    return (
      <span key={r.team_id} className={`${P}-tick ${r.is_user ? "is-user" : ""}`}>
        <Logo abbr={r.abbr} size={22} />
        <span className={`${P}-tick-ab`}>{r.abbr}</span>
        <span className={`${P}-tick-v ${tierTone(r.tier)}`}>{Math.round(num(r.value))}</span>
        <span className={`${P}-tick-d ${dir}`}>
          {dir === "up" ? "▲" : dir === "down" ? "▼" : "·"}
          {dir !== "flat" ? Math.abs(d).toFixed(1) : ""}
        </span>
        <span className={`${P}-tick-t`}>{r.tier}</span>
      </span>
    );
  });
  return (
    <div className={`${P}-ticker`} aria-label="Fan confidence around the league">
      <div className={`${P}-ticker-label`}>
        Fan confidence <small>0–100 · weekly move</small>
      </div>
      <div className={`${P}-ticker-view`}>
        <div className={`${P}-ticker-track`}>
          {items}
          {items.map((el) => React.cloneElement(el, { key: `${el.key}-b`, "aria-hidden": true }))}
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- vote meter */

function VoteMeter({ proposal, reveal, userAbbr, userName }) {
  const decided = proposal.status !== "pending";
  const a = proposal.analysis || {};
  const proj = a.projection;
  const needed = num(proposal.votes_needed);
  let slots = [];
  if (decided) {
    const ballots = asArray(proposal.ballots);
    const shownIds = new Set(ballots.slice(0, Math.min(reveal, ballots.length)).map((b) => b.team_id));
    const order = { yes: 0, abstain: 1, no: 2 };
    slots = [...ballots]
      .sort((x, y) => (order[x.vote] ?? 1) - (order[y.vote] ?? 1) || Number(Boolean(y.is_user)) - Number(Boolean(x.is_user)))
      .map((b) => ({ abbr: b.abbr, name: b.name, state: b.vote, user: b.is_user, lobbied: b.lobbied, shown: shownIds.has(b.team_id), reason: b.reason }));
  } else if (proj && asArray(proj.teams).length) {
    const teams = asArray(proj.teams);
    const cut = teams.findIndex((t) => t.p < 0.5);
    const at = cut < 0 ? teams.length : cut;
    slots = teams.map((t) => ({ abbr: t.abbr, name: t.name, state: t.lean, lobbied: t.lobbied, shown: true, reason: t.reason, p: t.p }));
    slots.splice(at, 0, { abbr: userAbbr, name: userName || "Your club", state: "user", user: true, shown: true, reason: "Your vote" });
  }
  const n = slots.length || num(proposal.analysis?.projection?.n_voters, 32);
  const pct = (k) => `${(k / Math.max(1, n)) * 100}%`;
  return (
    <div className={`${P}-meter`}>
      {slots.length ? (
        <div className={`${P}-meter-row`} style={{ gridTemplateColumns: `repeat(${slots.length}, minmax(0, 1fr))` }}>
          {slots.map((s, i) => (
            <span
              key={`${s.abbr}-${i}`}
              className={`${P}-seat is-${s.state} ${s.user ? "is-user" : ""} ${s.shown ? "is-shown" : ""}`}
              title={`${s.name || s.abbr}: ${s.user && !decided ? "your vote" : LEAN_LABEL[s.state] || s.state}${s.reason ? ` — ${s.reason}` : ""}${s.p != null ? ` (${Math.round(s.p * 100)}% yes)` : ""}`}
            >
              <Logo abbr={s.abbr} name={s.name} size={30} />
              <i />
              {s.lobbied ? <b className={`${P}-seat-lob`} aria-label="lobbied" /> : null}
            </span>
          ))}
          <span className={`${P}-meter-line`} style={{ left: pct(needed) }}>
            <em>{needed} to pass</em>
          </span>
        </div>
      ) : (
        <div className={`${P}-meter-bar`}>
          <i style={{ width: pct(num(proposal.forecast?.expected_yes)) }} />
          <span className={`${P}-meter-line`} style={{ left: pct(needed) }}>
            <em>{needed} to pass</em>
          </span>
        </div>
      )}
      <div className={`${P}-meter-key`}>
        {decided ? (
          <>
            <span className="k-yes">In favour</span>
            <span className="k-no">Opposed</span>
            <span className="k-abs">Abstained</span>
          </>
        ) : (
          <>
            <span className="k-for">For</span>
            <span className="k-lean_for">Leaning for</span>
            <span className="k-undecided">Undecided</span>
            <span className="k-lean_against">Leaning against</span>
            <span className="k-against">Against</span>
            <span className="k-user">Your seat</span>
          </>
        )}
        <span className="k-lob">Lobbied by you</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- proposal */

function GroupBars({ groups }) {
  const g = asArray(groups);
  if (!g.length) return <p className={`${P}-muted`}>Every kind of club is affected the same way.</p>;
  const max = Math.max(...g.map((x) => Math.abs(num(x.score))), 0.01);
  return (
    <div className={`${P}-gbars`}>
      {g.map((x) => {
        const w = (Math.abs(num(x.score)) / max) * 100;
        return (
          <div key={x.group} className={`${P}-gbar ${x.tone}`}>
            <span className={`${P}-gbar-name`}>{x.label}</span>
            <span className={`${P}-gbar-track`}>
              <span className="neg">{x.score < 0 ? <i style={{ width: `${w}%` }} /> : null}</span>
              <span className="pos">{x.score > 0 ? <i style={{ width: `${w}%` }} /> : null}</span>
            </span>
            <span className={`${P}-gbar-why`}>
              {x.score > 0 ? "Gains" : "Loses"}
              {x.driver ? ` · ${lowerFirst(x.driver)}` : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ProposalView({ proposal, tokens, busy, onVote, onLobby, reveal, onNext, isLast, userAbbr, userName, lobbyMult, lobbyReach }) {
  const [showAllWhy, setShowAllWhy] = useState(false);
  const a = proposal.analysis || {};
  const decided = proposal.status !== "pending";
  const forecast = proposal.forecast || {};
  const proj = a.projection;
  const needed = num(proposal.votes_needed);
  const ballots = asArray(proposal.ballots);
  const nTeams = ballots.length || num(proj?.n_voters, 32);
  const shownCount = decided ? Math.min(reveal, ballots.length) : 0;
  const shown = ballots.slice(0, shownCount);
  const yes = shown.filter((b) => b.vote === "yes").length;
  const no = shown.filter((b) => b.vote === "no").length;
  const abst = shown.filter((b) => b.vote !== "yes" && b.vote !== "no").length;
  const revealDone = decided && shownCount >= ballots.length;
  const tally = proposal.tally || {};
  const lobbied = proposal.lobbied;
  const effects = asArray(a.effects).length ? asArray(a.effects) : asArray(proposal.effects);
  const kase = buildCase(proposal);
  const club = a.club;
  const why = useMemo(() => {
    if (!revealDone) return [];
    const groups = new Map();
    ballots.filter((b) => !b.is_user && b.vote).forEach((b) => {
      const key = `${b.vote}|${b.reason || "No stated reason"}`;
      if (!groups.has(key)) groups.set(key, { vote: b.vote, reason: b.reason || "No stated reason", teams: [] });
      groups.get(key).teams.push(b);
    });
    return [...groups.values()].sort((x, y) => y.teams.length - x.teams.length);
  }, [revealDone, ballots]);
  const whyShown = showAllWhy ? why : why.slice(0, 6);
  const userVoteTxt = proposal.user_vote === "yes" ? "in favour" : proposal.user_vote === "no" ? "against" : "to abstain";
  const userWon =
    (proposal.status === "passed" && proposal.user_vote === "yes") || (proposal.status === "failed" && proposal.user_vote === "no");

  return (
    <article className={`${P}-prop`} aria-live="polite">
      <header className={`${P}-prop-head`}>
        <div className={`${P}-prop-cat`}>
          {proposal.category_label} · {proposal.threshold_label} vote, {needed} of {nTeams} needed
        </div>
        <div className={`${P}-prop-name`} role="heading" aria-level={2}>
          {proposal.title}
        </div>
        <p className={`${P}-prop-sum`}>{proposal.summary}</p>
        <p className={`${P}-prop-meta`}>
          {[a.scale ? SCALE_COPY[a.scale] : null, popularityText(a.popularity)].filter(Boolean).join(" · ")}
        </p>
        {lobbied ? (
          <p className={`${P}-prop-lob`}>
            You lobbied {lobbied.side === "yes" ? "for" : "against"} it: {asArray(lobbied.teams).join(", ")}.
          </p>
        ) : null}
      </header>

      <div className={`${P}-prop-cols`}>
      <div className={`${P}-prop-info`}>
      <section className={`${P}-sec`}>
        <div className={`${P}-sec-head`}>What it changes</div>
        {effects.length ? (
          <div className={`${P}-fx`}>
            {effects.map((e) => (
              <div key={`${e.key}-${e.label}`} className={`${P}-fx-row is-${e.tone || "mixed"}`}>
                <div className={`${P}-fx-what`}>
                  <span className={`${P}-fx-label`}>{e.label}</span>
                  <span className={`${P}-fx-note`}>{e.note}</span>
                </div>
                <div className={`${P}-fx-val`}>{e.text}</div>
                <div className={`${P}-fx-club`}>
                  {e.club ? (
                    <>
                      <span className={`${P}-fx-for`}>For {club?.abbr || "you"}</span>
                      {e.club}
                    </>
                  ) : (
                    <span className={`${P}-muted`}>{asArray(e.helps).length ? `Helps ${listText(e.helps)}` : ""}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className={`${P}-muted`}>No measurable effect on file.</p>
        )}
      </section>

      <section className={`${P}-case`}>
        <div>
          <div className={`${P}-sec-head is-pro`}>The case for</div>
          {kase.pros.map((x, i) => (
            <div key={`p${i}`} className={`${P}-arg`}>
              <strong>{x.head}</strong>
              {x.body ? <span>{x.body.replace(/^\.\s*/, "")}</span> : null}
            </div>
          ))}
        </div>
        <div>
          <div className={`${P}-sec-head is-con`}>The case against</div>
          {kase.cons.map((x, i) => (
            <div key={`c${i}`} className={`${P}-arg`}>
              <strong>{x.head}</strong>
              {x.body ? <span>{x.body.replace(/^\.\s*/, "")}</span> : null}
            </div>
          ))}
        </div>
        {kase.trade.length ? (
          <div className={`${P}-case-wide`}>
            <div className={`${P}-sec-head`}>Trade-offs</div>
            {kase.trade.map((x, i) => (
              <div key={`t${i}`} className={`${P}-arg`}>
                <strong>{x.head}</strong>
                {x.body ? <span>{x.body}</span> : null}
              </div>
            ))}
          </div>
        ) : null}
      </section>

      {a.groups || club ? (
        <section className={`${P}-split`}>
          <div>
            <div className={`${P}-sec-head`}>Who wins, who loses</div>
            <GroupBars groups={a.groups} />
          </div>
          {club ? (
            <div className={`${P}-verdict is-${club.verdict}`}>
              <div className={`${P}-sec-head`}>Your club</div>
              <div className={`${P}-verdict-big`}>{verdictText(club)}</div>
              <p className={`${P}-verdict-traits`}>{listText(club.traits)}</p>
              {asArray(club.because).map((t) => (
                <p key={t} className={`${P}-verdict-why`}>{t}</p>
              ))}
              {club.net_m != null ? (
                <p className={`${P}-verdict-net`}>
                  Measurable money: <strong>{signedM(club.net_m)}</strong> a year
                </p>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}

      {why.length ? (
        <section className={`${P}-sec`}>
          <div className={`${P}-sec-head`}>Why governors voted this way</div>
          <div className={`${P}-why`}>
            {whyShown.map((g) => (
              <div key={`${g.vote}-${g.reason}`} className={`${P}-why-row is-${g.vote}`}>
                <span className={`${P}-why-v`}>{g.vote === "yes" ? "For" : g.vote === "no" ? "Against" : "Abstained"} · {g.teams.length}</span>
                <span className={`${P}-why-r`}>{g.reason}</span>
                <span className={`${P}-why-t`}>
                  {g.teams.map((b) => (
                    <Logo key={b.team_id || b.abbr} abbr={b.abbr} name={b.name} size={22} />
                  ))}
                </span>
              </div>
            ))}
          </div>
          {why.length > whyShown.length ? (
            <p className={`${P}-muted`}>Plus {why.length - whyShown.length} smaller groups; the room column lists every governor.</p>
          ) : null}
        </section>
      ) : null}
      </div>
      <div className={`${P}-prop-side`}>
      <section className={`${P}-floorvote`} aria-label="Vote">
        <div className={`${P}-sec-head`}>{decided ? "The vote" : "Projected vote"}</div>
        <VoteMeter proposal={proposal} reveal={reveal} userAbbr={userAbbr} userName={userName} />
        <p className={`${P}-count`}>
          {decided ? (
            <>
              <strong className="yes">{yes} in favour</strong>
              <strong className="no">{no} opposed</strong>
              {abst ? <span>{abst} abstained</span> : null}
              <span>{needed} needed of {nTeams}</span>
            </>
          ) : proj ? (
            <>
              <strong>{forecast.label || "—"}</strong>
              <span>
                About {num(proj.expected_yes).toFixed(1)} ± {num(proj.sd).toFixed(1)} governors in favour before your vote, {needed} needed
              </span>
              <span>
                Pass chance <b className="yes">{Math.round(num(proj.p_pass_if_yes) * 100)}%</b> if you vote for,{" "}
                <b className="no">{Math.round(num(proj.p_pass_if_no) * 100)}%</b> if you vote against
              </span>
            </>
          ) : (
            <>
              <strong>{forecast.label || "—"}</strong>
              <span>About {num(forecast.expected_yes).toFixed(0)} in favour, {needed} needed</span>
              {proposal.room_mood ? <span>{proposal.room_mood}</span> : null}
            </>
          )}
        </p>

        {!decided ? (
          <div className={`${P}-actions`}>
            <div className={`${P}-actions-main`}>
              <button type="button" className={`${P}-go`} disabled={busy} onClick={() => onVote("yes")}>
                Vote in favour
              </button>
              <button type="button" className={`${P}-go is-alt`} disabled={busy} onClick={() => onVote("no")}>
                Vote against
              </button>
              <button type="button" className={`${P}-quiet`} disabled={busy} onClick={() => onVote("abstain")}>
                Abstain
              </button>
              <span className={`${P}-keys`}>Keys: A in favour · D against</span>
            </div>
            <div className={`${P}-actions-lobby`}>
              <span className={`${P}-lobby-copy`}>
                {lobbied
                  ? "You've already worked the phones on this one."
                  : tokens > 0
                    ? `Lobbying calls the ${lobbyReach} most persuadable governors whose clubs look like yours and shifts each one's chance of voting your way by about ${Math.round(18 * (lobbyMult || 1))} points. ${tokens} call${tokens === 1 ? "" : "s"} left this meeting.`
                    : "No lobbying calls left this meeting."}
              </span>
              <button type="button" className={`${P}-quiet`} disabled={busy || tokens <= 0 || Boolean(lobbied)} onClick={() => onLobby("yes")}>
                Lobby for it
              </button>
              <button type="button" className={`${P}-quiet`} disabled={busy || tokens <= 0 || Boolean(lobbied)} onClick={() => onLobby("no")}>
                Lobby against
              </button>
            </div>
          </div>
        ) : null}

        {revealDone ? (
          <div className={`${P}-outcome is-${proposal.status}`}>
            <div>
              <div className={`${P}-outcome-big`}>{proposal.status === "passed" ? "Passed" : "Defeated"}</div>
              <p>
                {tally.yes}–{tally.no}
                {tally.abstain ? `, ${tally.abstain} abstained` : ""} · {needed} needed · you voted {userVoteTxt}
                {proposal.user_vote !== "abstain" ? (userWon ? ", on the winning side" : ", on the losing side") : ""}
              </p>
              {asArray(a.result_blocs).length ? (
                <p className={`${P}-blocline`}>
                  {asArray(a.result_blocs).map((b) => (
                    <span key={b.group}>
                      {b.label} <b>{b.yes}–{b.no}</b>
                    </span>
                  ))}
                </p>
              ) : null}
            </div>
            <button type="button" className={`${P}-go`} onClick={onNext}>
              {isLast ? "Close the meeting" : "Next proposal →"}
            </button>
          </div>
        ) : null}
      </section>

      </div>
      </div>
    </article>
  );
}

/* ---------------------------------------------------------------- side columns */

function RoomPanel({ proposal }) {
  const a = proposal?.analysis || {};
  const decided = proposal?.status !== "pending";
  if (decided) {
    const ballots = asArray(proposal.ballots).filter((b) => !b.is_user);
    const blocs = asArray(a.result_blocs);
    return (
      <aside className={`${P}-room`}>
        <div className={`${P}-col-head`}>How the room voted</div>
        {blocs.length ? (
          <div className={`${P}-blocs`}>
            {blocs.map((b) => (
              <div key={b.group} className={`${P}-bloc`}>
                <span>{b.label}</span>
                <span className={`${P}-bloc-bar`}>
                  <i style={{ width: `${(b.yes / Math.max(1, b.n)) * 100}%` }} />
                </span>
                <b>{b.yes}–{b.no}</b>
              </div>
            ))}
          </div>
        ) : null}
        {["yes", "no"].map((v) => {
          const rows = ballots.filter((b) => b.vote === v);
          if (!rows.length) return null;
          return (
            <div key={v} className={`${P}-roomgrp`}>
              <div className={`${P}-roomgrp-head is-${v}`}>
                {v === "yes" ? "In favour" : "Opposed"} <span>{rows.length}</span>
              </div>
              {rows.map((b) => (
                <div key={b.team_id} className={`${P}-gov`}>
                  <Logo abbr={b.abbr} name={b.name} size={28} />
                  <span className={`${P}-gov-ab`}>{b.abbr}</span>
                  <span className={`${P}-gov-why`}>{b.reason}{b.lobbied ? " · lobbied" : ""}</span>
                </div>
              ))}
            </div>
          );
        })}
      </aside>
    );
  }
  const proj = a.projection;
  if (!proj) {
    return (
      <aside className={`${P}-room`}>
        <div className={`${P}-col-head`}>The room</div>
        <p className={`${P}-muted`}>
          {proposal?.room_mood || "Opinions vary club by club."} Restart the game server to see the projected vote by club.
        </p>
      </aside>
    );
  }
  const teams = asArray(proj.teams);
  return (
    <aside className={`${P}-room`}>
      <div className={`${P}-col-head`}>The room</div>
      <p className={`${P}-room-sum`}>
        {LEAN_ORDER.filter((k) => proj.counts?.[k]).map((k) => `${proj.counts[k]} ${LEAN_LABEL[k].toLowerCase()}`).join(" · ")}
      </p>
      {asArray(proj.blocs).length ? (
        <div className={`${P}-blocs`}>
          {asArray(proj.blocs).map((b) => (
            <div key={b.group} className={`${P}-bloc`}>
              <span>{b.label} <small>{b.n}</small></span>
              <span className={`${P}-bloc-bar`}>
                <i style={{ width: `${num(b.share) * 100}%` }} />
              </span>
              <b>{Math.round(num(b.share) * 100)}%</b>
            </div>
          ))}
          <p className={`${P}-muted`}>Share of each bloc expected to vote in favour.</p>
        </div>
      ) : null}
      {LEAN_ORDER.map((k) => {
        const rows = teams.filter((t) => t.lean === k);
        if (!rows.length) return null;
        return (
          <div key={k} className={`${P}-roomgrp`}>
            <div className={`${P}-roomgrp-head is-${k}`}>
              {LEAN_LABEL[k]} <span>{rows.length}</span>
            </div>
            {rows.map((t) => (
              <div key={t.team_id} className={`${P}-gov`} title={listText(t.traits)}>
                <Logo abbr={t.abbr} name={t.name} size={28} />
                <span className={`${P}-gov-ab`}>{t.abbr}</span>
                <span className={`${P}-gov-why`}>
                  {t.reason}
                  {t.lobbied ? " · lobbied" : ""}
                </span>
                <span className={`${P}-gov-p`}>{Math.round(num(t.p) * 100)}%</span>
              </div>
            ))}
          </div>
        );
      })}
    </aside>
  );
}

function ClubSnapshot({ gov, club }) {
  const impact = gov.user_impact || {};
  const fan = gov.fan_report || {};
  const change = num(impact.value_change_pct);
  return (
    <div className={`${P}-club`}>
      <div className={`${P}-col-head`}>{club?.name || "Your club"}</div>
      {club?.traits?.length ? <p className={`${P}-club-traits`}>{listText(club.traits)}</p> : null}
      <dl className={`${P}-kv`}>
        <dt>Revenue</dt>
        <dd>{impact.annual_revenue_m != null ? money(impact.annual_revenue_m) : "—"}</dd>
        <dt>Profit</dt>
        <dd className={num(impact.annual_profit_m) >= 0 ? "up" : "down"}>{impact.annual_profit_m != null ? money(impact.annual_profit_m) : "—"}</dd>
        <dt>Franchise value</dt>
        <dd>
          {impact.franchise_value_b != null ? `$${num(impact.franchise_value_b).toFixed(2)}B` : "—"}
          <small className={change >= 0 ? "up" : "down"}> {`${change >= 0 ? "+" : ""}${change.toFixed(1)}%`}</small>
        </dd>
        <dt>Fan confidence</dt>
        <dd className={tierTone(fan.tier)}>
          {fan.value != null ? Math.round(num(fan.value)) : "—"} <small>{fan.tier}</small>
        </dd>
        <dt>Scouting budget</dt>
        <dd>{impact.scouting_budget ? money(impact.scouting_budget) : "—"}</dd>
        <dt>Signing bonuses</dt>
        <dd className={impact.bonus_eligible ? "up" : "down"}>
          {impact.bonus_eligible ? `Up to ${Math.round(num(impact.bonus_max_pct) * 100)}%` : `Locked (under $${num(impact.bonus_floor_m, 155).toFixed(0)}M)`}
        </dd>
      </dl>
    </div>
  );
}

function RulesInForce({ gov, compact = false }) {
  const book = asArray(gov.rulebook);
  const mods = asArray(gov.modifiers).filter((m) => Math.abs(num(m.value)) > 1e-6 || ["ot_minutes", "ties_allowed", "pts_reg_win", "pts_ot_win", "pts_otl"].includes(m.key));
  return (
    <div className={`${P}-rules`}>
      <div className={`${P}-col-head`}>
        Rules in force <span>{book.length}</span>
      </div>
      {book.length ? (
        <>
          {book.slice(0, compact ? 6 : 40).map((r) => (
            <div key={r.rule_id} className={`${P}-rule`}>
              <span className={`${P}-rule-name`}>
                {r.title} <small>{r.season}</small>
              </span>
              <span className={`${P}-rule-fx`}>{asArray(r.effects).map((e) => `${e.label} ${e.text}`).join(" · ")}</span>
            </div>
          ))}
          {compact && book.length > 6 ? <p className={`${P}-muted`}>And {book.length - 6} more.</p> : null}
          {mods.length ? (
            <>
              <div className={`${P}-sub`}>Combined effect of every rule</div>
              {mods.map((m) => (
                <div key={m.key} className={`${P}-rule-tot`}>
                  <span>{m.label}</span>
                  <b className={m.tone === "pro" ? "up" : m.tone === "con" ? "down" : ""}>{m.text}</b>
                </div>
              ))}
            </>
          ) : (
            <p className={`${P}-muted`}>The rules in force cancel each other out.</p>
          )}
        </>
      ) : (
        <p className={`${P}-muted`}>No Board rules in force yet. The league plays under the standard NHL rulebook.</p>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- expansion */

const PROTECT_LABEL = { F: "Forwards", D: "Defense", G: "Goalies", S: "Skaters" };
const SCHEME_LIMITS = { "7-3-1": { F: 7, D: 3, G: 1 }, "8-1": { S: 8, G: 1 } };

function ExpansionProtectionPanel({ draft, onSaved }) {
  const players = asArray(draft?.players);
  const [scheme, setScheme] = useState(draft?.scheme || "7-3-1");
  const limits = SCHEME_LIMITS[scheme] || SCHEME_LIMITS["7-3-1"];
  const slotOf = (pos) => (pos === "G" || scheme === "7-3-1" ? pos : "S");
  const [picked, setPicked] = useState(() => new Set(players.filter((p) => p.protected && !p.exempt).map((p) => p.player_id)));
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const counts = Object.fromEntries(Object.keys(limits).map((k) => [k, 0]));
  players.forEach((p) => {
    if (picked.has(p.player_id) && !p.exempt) counts[slotOf(p.pos)] += 1;
  });
  const exposure = { F: 0, D: 0, G: 0 };
  players.forEach((p) => {
    if (!picked.has(p.player_id) && p.exposable) exposure[p.pos] += 1;
  });
  const exposeNeed = draft?.exposure?.need || { F: 2, D: 1, G: 1 };
  const toggle = (p) => {
    if (p.exempt) return;
    setMsg("");
    const slot = slotOf(p.pos);
    if (picked.has(p.player_id)) {
      if (p.nmc) {
        setMsg("No-move clauses must be protected.");
        return;
      }
      setPicked((prev) => {
        const next = new Set(prev);
        next.delete(p.player_id);
        return next;
      });
    } else if (counts[slot] < num(limits[slot])) {
      setPicked((prev) => new Set(prev).add(p.player_id));
    } else {
      setMsg(`You can protect ${limits[slot]} ${PROTECT_LABEL[slot].toLowerCase()}.`);
    }
  };
  const changeScheme = (next) => {
    setScheme(next);
    setMsg("");
    const lim = SCHEME_LIMITS[next];
    const used = {};
    const keep = new Set();
    players
      .filter((p) => picked.has(p.player_id) && !p.exempt)
      .sort((a, b) => Number(b.nmc) - Number(a.nmc) || b.ovr - a.ovr)
      .forEach((p) => {
        const s = p.pos === "G" || next === "7-3-1" ? p.pos : "S";
        if ((used[s] || 0) < lim[s]) {
          used[s] = (used[s] || 0) + 1;
          keep.add(p.player_id);
        }
      });
    setPicked(keep);
  };
  const save = async () => {
    setSaving(true);
    setMsg("");
    try {
      const data = await setExpansionProtection([...picked], scheme);
      if (data?.governance) onSaved?.(data.governance);
      setMsg("Protection list saved.");
    } catch (e) {
      setMsg(e?.response?.data?.detail || e?.message || "Could not save the list");
    } finally {
      setSaving(false);
    }
  };
  return (
    <section className={`${P}-xp`}>
      <div className={`${P}-sec-head`}>Expansion draft: {asArray(draft?.cities).join(" & ")}</div>
      <p className={`${P}-xp-copy`}>
        {draft?.when}. Protect 7 forwards, 3 defensemen and 1 goalie, or any 8 skaters and 1 goalie. You lose{" "}
        {num(draft?.n_new, 1) === 1 ? "one unprotected player" : `${draft.n_new} unprotected players`}. First- and second-year pros are
        exempt; no-move clauses must be protected. You must expose {exposeNeed.F} forwards and {exposeNeed.D} defenseman under contract
        with 40+ GP last season (or 70+ career), plus a signed goalie; if you don't, your weakest protected player at that position is
        exposed.
        {draft?.user_set ? "" : " Until you save, your highest-rated players are protected automatically."}
      </p>
      <div className={`${P}-xp-schemes`}>
        {Object.keys(SCHEME_LIMITS).map((k) => (
          <button key={k} type="button" className={`${P}-quiet ${scheme === k ? "is-on" : ""}`} onClick={() => changeScheme(k)}>
            {k === "7-3-1" ? "7F / 3D / 1G" : "8 skaters / 1G"}
          </button>
        ))}
        <span className={`${P}-muted`}>
          {Object.keys(limits).map((k) => `${PROTECT_LABEL[k]} ${counts[k]}/${limits[k]}`).join(" · ")}
          {" — exposed and eligible: "}
          {["F", "D", "G"].map((k) => `${k} ${exposure[k]}/${exposeNeed[k]}`).join(" · ")}
        </span>
      </div>
      <div className={`${P}-xp-list`}>
        {players.map((p) => (
          <label key={p.player_id} className={`${P}-xp-row ${p.contract || p.exempt ? "" : "is-dim"}`}>
            <input type="checkbox" checked={p.exempt || picked.has(p.player_id)} disabled={p.exempt} onChange={() => toggle(p)} />
            <span className={`${P}-xp-pos`}>{p.pos}</span>
            <span className={`${P}-xp-name`}>{p.name}</span>
            <span>
              {Math.round(p.ovr)} OVR · {p.age}
            </span>
            <span className={`${P}-muted`}>{p.exempt ? "Exempt" : p.nmc ? "NMC" : !p.contract ? "Unsigned" : ""}</span>
          </label>
        ))}
      </div>
      <div className={`${P}-xp-dock`}>
        <button type="button" className={`${P}-go is-alt`} disabled={saving} onClick={save}>
          {saving ? "Saving…" : "Save protection list"}
        </button>
        {msg ? <span className={`${P}-muted`}>{msg}</span> : null}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- intro */

function Intro({ gov, proposals, allDone, error, onStart }) {
  const n = num(gov?.meeting?.n_teams, 32);
  const fan = gov?.fan_report || {};
  const lobbyFx = asArray(fan.effects).find((e) => e.label === "Board lobbying");
  const club = proposals.find((p) => p.analysis?.club)?.analysis?.club;
  const neverHeard = gov?.rules_never_proposed;
  return (
    <div className={`${P}-intro`}>
      <main className={`${P}-intro-main`}>
        <div className={`${P}-intro-top`}>
          <div>
            <div className={`${P}-intro-name`} role="heading" aria-level={1}>
              Tonight's agenda
            </div>
            <p className={`${P}-intro-sub`}>
              {gov?.season_label} annual meeting · {proposals.length || 5} proposals · {n} governors, one vote each
              {gov?.catalog_size ? ` · picked by the competition committee from a ${gov.catalog_size}-rule docket${neverHeard != null ? ` (${neverHeard} never voted on)` : ""}` : ""}
            </p>
          </div>
          <button type="button" className={`${P}-go is-big`} disabled={!proposals.length} onClick={onStart}>
            {allDone ? "Review the meeting" : "Call the meeting to order"}
          </button>
        </div>
        {error ? <p className={`${P}-error`}>{error}</p> : null}
        <ol className={`${P}-agenda-preview`}>
          {proposals.map((p, i) => {
            const a = p.analysis || {};
            const eff = asArray(a.effects).length ? asArray(a.effects) : asArray(p.effects);
            const proj = a.projection;
            return (
              <li key={p.id} className={`${P}-ap`}>
                <span className={`${P}-ap-n`}>{i + 1}</span>
                <div className={`${P}-ap-body`}>
                  <div className={`${P}-ap-cat`}>
                    {p.category_label} · {p.threshold_label}, {p.votes_needed} of {n} needed
                    {a.scale ? ` · ${a.scale} change` : ""}
                  </div>
                  <div className={`${P}-ap-name`}>{p.title}</div>
                  <p className={`${P}-ap-sum`}>{p.summary}</p>
                  <p className={`${P}-ap-fx`}>
                    {eff.map((e) => (
                      <span key={`${e.key}-${e.label}`} className={`is-${e.tone || "mixed"}`}>
                        {e.label} <b>{e.text}</b>
                      </span>
                    ))}
                  </p>
                </div>
                <div className={`${P}-ap-side`}>
                  {a.club ? (
                    <>
                      <div className={`${P}-ap-verdict is-${a.club.verdict}`}>{verdictText(a.club)}</div>
                      <p className={`${P}-ap-why`}>
                        {asArray(a.club.because)[0] || (a.club.net_m != null ? `${signedM(a.club.net_m)} a year` : "No club-specific effect")}
                      </p>
                    </>
                  ) : null}
                  {p.status !== "pending" ? (
                    <div className={`${P}-ap-read is-${p.status}`}>
                      {p.status === "passed" ? "Passed" : "Defeated"} {p.tally?.yes}–{p.tally?.no}
                    </div>
                  ) : (
                    <div className={`${P}-ap-read`}>
                      {p.forecast?.label || "—"}
                      <small>
                        {proj ? `≈${num(proj.expected_yes).toFixed(0)} for, ${p.votes_needed} needed` : p.forecast ? `≈${num(p.forecast.expected_yes).toFixed(0)} for, ${p.votes_needed} needed` : ""}
                      </small>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </main>

      <aside className={`${P}-intro-side`}>
        <section>
          <div className={`${P}-col-head`}>How the room decides</div>
          <dl className={`${P}-kv`}>
            <dt>Majority</dt>
            <dd>{votesNeededFor("majority", n)} votes <small>on-ice, player, schedule, media, uniforms</small></dd>
            <dt>Two-thirds</dt>
            <dd>{votesNeededFor("two_thirds", n)} votes <small>cap, contracts, revenue, trades, draft</small></dd>
            <dt>Three-quarters</dt>
            <dd>{votesNeededFor("three_quarters", n)} votes <small>relocation, expansion, league structure</small></dd>
          </dl>
          <p className={`${P}-prose`}>
            Each governor votes their own club's interest: market size, profit, contender or rebuilder, cap room. Revenue sharing splits
            big and small markets; playoff money splits contenders and rebuilders; cap and floor moves split rich and money-losing clubs.
          </p>
        </section>
        <section>
          <div className={`${P}-col-head`}>Lobbying</div>
          <p className={`${P}-prose`}>
            You get {num(gov?.meeting?.lobby_tokens, 2)} call{num(gov?.meeting?.lobby_tokens, 2) === 1 ? "" : "s"} left this meeting, one per
            proposal. A call reaches the four most persuadable governors whose clubs look like yours and moves each of them toward your side.
            Your fans set how hard you can push{lobbyFx ? `: right now ${lobbyFx.text}` : ""}. An electric fan base (80+) reaches a fifth governor.
          </p>
        </section>
        <section>
          <div className={`${P}-col-head`}>The fan confidence ticker</div>
          <p className={`${P}-prose`}>
            The strip at the top is every club's fan confidence (0–100) with this week's move. It is real money: gate and merch revenue,
            home-ice edge, free-agent interest and the weight of your lobbying calls.
          </p>
          <div className={`${P}-tiers`}>
            {FAN_TIER_ROWS.map(([t, r]) => (
              <span key={t} className={`${tierTone(t)} ${fan.tier === t ? "is-you" : ""}`}>
                <b>{t}</b> {r}
              </span>
            ))}
          </div>
          {fan.value != null ? (
            <>
              <div className={`${P}-sub`}>
                Your fans: <b className={tierTone(fan.tier)}>{Math.round(num(fan.value))} {fan.tier}</b>
                {Math.abs(num(fan.delta)) >= 0.4 ? ` · ${num(fan.delta) > 0 ? "▲" : "▼"} ${Math.abs(num(fan.delta)).toFixed(1)} this week` : ""}
              </div>
              <dl className={`${P}-kv is-tight`}>
                {asArray(fan.effects).map((e) => (
                  <React.Fragment key={e.label}>
                    <dt>{e.label}</dt>
                    <dd className={e.tone === "pro" ? "up" : e.tone === "con" ? "down" : ""}>{e.text}</dd>
                  </React.Fragment>
                ))}
              </dl>
              {asArray(fan.reasons).map((r, i) => (
                <p key={i} className={`${P}-prose is-${r.tone}`}>{r.text}</p>
              ))}
            </>
          ) : null}
        </section>
        <section>
          <div className={`${P}-col-head`}>Standing rules</div>
          {gov?.on_ice?.points ? (
            <p className={`${P}-prose`}>
              {gov.on_ice.points} points · {gov.on_ice.ot_minutes}-minute overtime · {gov.on_ice.shootout ? "shootout decides ties" : "no shootout (ties stand)"}
            </p>
          ) : null}
          {gov?.expansion_draft?.due ? (
            <p className={`${P}-prose`}>
              Expansion draft this offseason ({asArray(gov.expansion_draft.cities).join(" & ")}): set your protection list before you leave the meeting.
            </p>
          ) : null}
          {asArray(gov?.expansion_draft?.upcoming).map((u, i) => (
            <p key={`up-${i}`} className={`${P}-prose`}>
              Approved expansion ({asArray(u.cities).join(" & ")}): expansion draft at the {u.draft_offseason} meeting; clubs join for{" "}
              {u.start_season}-{String((num(u.start_season) + 1) % 100).padStart(2, "0")}.
            </p>
          ))}
          <RulesInForce gov={gov || {}} compact />
        </section>
        {club ? <ClubSnapshot gov={gov || {}} club={club} /> : null}
      </aside>
    </div>
  );
}

/* ---------------------------------------------------------------- adjourned */

function Adjourned({ gov, proposals, onContinue, onFloor, setGov }) {
  const passed = proposals.filter((p) => p.status === "passed");
  const failed = proposals.filter((p) => p.status === "failed");
  const voted = proposals.filter((p) => p.user_vote === "yes" || p.user_vote === "no");
  const won = voted.filter(
    (p) => (p.status === "passed" && p.user_vote === "yes") || (p.status === "failed" && p.user_vote === "no")
  ).length;
  const netNext = passed.reduce((s, p) => s + (p.analysis?.club?.net_m != null ? num(p.analysis.club.net_m) : 0), 0);
  const hasNet = passed.some((p) => p.analysis?.club?.net_m != null);
  const club = proposals.find((p) => p.analysis?.club)?.analysis?.club;
  return (
    <div className={`${P}-adj`}>
      <main className={`${P}-adj-main`}>
        <div className={`${P}-adj-top`}>
          <div>
            <div className={`${P}-intro-name`} role="heading" aria-level={1}>
              Meeting adjourned
            </div>
            <p className={`${P}-intro-sub`}>
              {passed.length} passed · {failed.length} defeated
              {voted.length ? ` · you were on the winning side ${won} of ${voted.length} times` : ""}
              {hasNet ? ` · measurable money for ${club?.abbr || "your club"} next season ${signedM(netNext)} a year` : ""}
            </p>
          </div>
          <div className={`${P}-adj-dock`}>
            <button type="button" className={`${P}-go is-big`} onClick={onContinue}>
              {gov?.expansion_draft?.due ? "Hold the expansion draft & continue" : "Continue to Cap Report"}
            </button>
            <button type="button" className={`${P}-quiet`} onClick={onFloor}>
              Back to the floor
            </button>
          </div>
        </div>

        <div className={`${P}-recap`}>
          {proposals.map((p) => {
            const a = p.analysis || {};
            const t = p.tally || {};
            const total = num(t.yes) + num(t.no) + num(t.abstain);
            const uv = p.user_vote === "yes" ? "for" : p.user_vote === "no" ? "against" : p.user_vote ? "abstained" : "—";
            const sideWon =
              (p.status === "passed" && p.user_vote === "yes") || (p.status === "failed" && p.user_vote === "no");
            const eff = asArray(a.effects).length ? asArray(a.effects) : asArray(p.effects);
            return (
              <article key={p.id} className={`${P}-rc is-${p.status}`}>
                <div className={`${P}-rc-res`}>
                  <div className={`${P}-rc-word`}>{p.status === "passed" ? "Passed" : p.status === "failed" ? "Defeated" : "Open"}</div>
                  <div className={`${P}-rc-split`}>
                    {t.yes ?? "–"}<span>–</span>{t.no ?? "–"}
                  </div>
                  {total ? (
                    <span className={`${P}-rc-bar`}>
                      <i className="y" style={{ width: `${(num(t.yes) / total) * 100}%` }} />
                      <b style={{ left: `${(num(p.votes_needed) / total) * 100}%` }} />
                    </span>
                  ) : null}
                  <small>{p.votes_needed} needed</small>
                </div>
                <div className={`${P}-rc-body`}>
                  <div className={`${P}-ap-cat`}>{p.category_label}</div>
                  <div className={`${P}-rc-name`}>{p.title}</div>
                  <p className={`${P}-rc-you`}>
                    {club?.abbr || "You"} voted <b>{uv}</b>
                    {p.user_vote && p.user_vote !== "abstain" ? (sideWon ? ", on the winning side" : ", on the losing side") : ""}
                  </p>
                  {asArray(a.result_blocs).length ? (
                    <p className={`${P}-blocline`}>
                      {asArray(a.result_blocs).map((b) => (
                        <span key={b.group}>
                          {b.label} <b>{b.yes}–{b.no}</b>
                        </span>
                      ))}
                    </p>
                  ) : null}
                </div>
                <div className={`${P}-rc-next`}>
                  <div className={`${P}-sub`}>{p.status === "passed" ? "What changes next season" : "What you keep"}</div>
                  {p.status === "passed" ? (
                    <>
                      {eff.map((e) => (
                        <p key={`${e.key}-${e.label}`} className={`${P}-rc-fx is-${e.tone || "mixed"}`}>
                          <b>{e.label} {e.text}</b>
                          {e.club ? <span>{e.club}</span> : e.note ? <span>{e.note}</span> : null}
                        </p>
                      ))}
                      {a.club ? <p className={`${P}-rc-verdict is-${a.club.verdict}`}>{verdictText(a.club)}</p> : null}
                    </>
                  ) : (
                    <>
                      <p className={`${P}-rc-fx`}>
                        <span>Nothing changes: the current rule stands.</span>
                      </p>
                      {a.club && a.club.verdict !== "neutral" ? (
                        <p className={`${P}-rc-verdict is-${a.club.verdict === "good" ? "bad" : "good"}`}>
                          {a.club.verdict === "good" ? `${club?.abbr || "You"} missed out on a rule that helped you` : `${club?.abbr || "You"} avoided a rule that would have hurt you`}
                        </p>
                      ) : null}
                    </>
                  )}
                </div>
              </article>
            );
          })}
        </div>

        {num(gov?.pending_cap_adjust_m) !== 0 ? (
          <p className={`${P}-prose is-note`}>
            Next season's cap moves {num(gov.pending_cap_adjust_m) > 0 ? "+" : "−"}${Math.abs(num(gov.pending_cap_adjust_m)).toFixed(2)}M on top of the league projection.
          </p>
        ) : null}
        {gov?.expansion_draft?.due ? <ExpansionProtectionPanel draft={gov.expansion_draft} onSaved={setGov} /> : null}
      </main>
      <aside className={`${P}-adj-side`}>
        <RulesInForce gov={gov || {}} />
        <ClubSnapshot gov={gov || {}} club={club} />
        {asArray(gov?.history).length ? (
          <div className={`${P}-rules`}>
            <div className={`${P}-col-head`}>Past votes</div>
            {asArray(gov.history).slice(0, 10).map((h) => (
              <div key={h.proposal_id} className={`${P}-rule-tot`}>
                <span>
                  {h.season} · {h.title}
                </span>
                <b className={h.status === "passed" ? "up" : "down"}>
                  {h.status === "passed" ? "Passed" : "Failed"} {h.yes}–{h.no}
                </b>
              </div>
            ))}
          </div>
        ) : null}
      </aside>
    </div>
  );
}

/* ---------------------------------------------------------------- main */

export default function BoardOfGovernorsMenu({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const initial = pickFranchiseData(franchiseState, eventData, ["board_of_governors", "offseason.board_of_governors"]);
  const [gov, setGov] = useState(initial && typeof initial === "object" ? initial : null);
  const [stage, setStage] = useState("intro");
  const [active, setActive] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reveal, setReveal] = useState({});
  const timerRef = useRef(null);

  const proposals = useMemo(() => asArray(gov?.meeting?.proposals), [gov]);
  const tokens = num(gov?.meeting?.lobby_tokens);
  const allDone = proposals.length > 0 && proposals.every((p) => p.status !== "pending");
  const userRow = asArray(gov?.fan_ticker).find((r) => r.is_user);
  const club = proposals.find((p) => p.analysis?.club)?.analysis?.club;
  const userAbbr = club?.abbr || userRow?.abbr || "";
  const userName = club?.name || "";
  const lobbyMult = useMemo(() => {
    const v = num(gov?.fan_report?.value, 55);
    return 0.8 + 0.4 * Math.min(1, Math.max(0, (v - 40) / 50));
  }, [gov]);

  useEffect(() => {
    let alive = true;
    getGovernance()
      .then((data) => {
        if (alive && data?.governance) setGov(data.governance);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => () => clearInterval(timerRef.current), []);

  // Land on the first undecided proposal.
  useEffect(() => {
    if (!proposals.length) return;
    const firstPending = proposals.findIndex((p) => p.status === "pending");
    if (firstPending >= 0 && proposals[active]?.status !== "pending" && reveal[proposals[active]?.id] == null) {
      setActive(firstPending);
    }
  }, [proposals]); // eslint-disable-line react-hooks/exhaustive-deps

  const startReveal = useCallback((pid, total) => {
    clearInterval(timerRef.current);
    setReveal((r) => ({ ...r, [pid]: 0 }));
    timerRef.current = setInterval(() => {
      setReveal((r) => {
        const n = (r[pid] || 0) + 1;
        if (n >= total) clearInterval(timerRef.current);
        return { ...r, [pid]: n };
      });
    }, REVEAL_MS);
  }, []);

  const current = proposals[active];

  const handleVote = async (vote) => {
    if (!current || busy) return;
    setBusy(true);
    setError("");
    try {
      const data = await voteGovernance({ proposal_id: current.id, vote });
      if (data?.governance) setGov(data.governance);
      const total = asArray(data?.proposal?.ballots).length;
      startReveal(current.id, total);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Vote failed");
    } finally {
      setBusy(false);
    }
  };

  const handleLobby = async (side) => {
    if (!current || busy) return;
    setBusy(true);
    setError("");
    try {
      const data = await lobbyGovernance({ proposal_id: current.id, side });
      if (data?.governance) setGov(data.governance);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "Lobbying failed");
    } finally {
      setBusy(false);
    }
  };

  const handleNext = () => {
    const nextPending = proposals.findIndex((p, i) => i > active && p.status === "pending");
    const anyPending = proposals.findIndex((p) => p.status === "pending");
    if (nextPending >= 0) setActive(nextPending);
    else if (anyPending >= 0) setActive(anyPending);
    else setStage("adjourned");
  };

  useEffect(() => {
    if (stage !== "floor" || !current || current.status !== "pending" || busy) return undefined;
    const onKey = (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
      if (e.key === "a" || e.key === "A") handleVote("yes");
      if (e.key === "d" || e.key === "D") handleVote("no");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // re-bind with fresh closures

  const seasonLabel = gov?.season_label || "";
  const decidedCount = proposals.filter((p) => p.status !== "pending").length;
  const phaseLabel =
    stage === "intro"
      ? "Annual meeting"
      : allDone && stage === "adjourned"
        ? "Meeting adjourned"
        : `Proposal ${Math.min(active + 1, proposals.length)} of ${proposals.length} · ${decidedCount} decided`;
  const bookCount = asArray(gov?.rulebook).length;

  return (
    <section className={`${P}-root`}>
      <header className={`${P}-top`}>
        <div className={`${P}-top-left`}>
          <button type="button" className={`${P}-quiet ${P}-hub`} onClick={onBack}>
            ← Hub
          </button>
          <div>
            <div className={`${P}-brand`} role="heading" aria-level={1}>
              Board of Governors
            </div>
            <div className={`${P}-brand-sub`}>
              {[seasonLabel, gov?.meeting?.n_teams ? `${gov.meeting.n_teams} governors` : null].filter(Boolean).join(" · ")}
            </div>
          </div>
        </div>
        <div className={`${P}-phase`}>{phaseLabel}</div>
        <div className={`${P}-top-right`}>
          <span className={tokens > 0 ? "is-on" : ""}>
            <b>{tokens}</b> lobbying call{tokens === 1 ? "" : "s"} left
          </span>
          <span>
            <b>{bookCount}</b> Board rule{bookCount === 1 ? "" : "s"} in force
          </span>
        </div>
      </header>
      <FanTicker rows={gov?.fan_ticker} />

      {stage === "intro" ? (
        <Intro gov={gov || {}} proposals={proposals} allDone={allDone} error={error} onStart={() => setStage(allDone ? "adjourned" : "floor")} />
      ) : null}

      {stage === "floor" && current ? (
        <div className={`${P}-floor`}>
          <nav className={`${P}-left`} aria-label="Proposals">
            <div className={`${P}-col-head`}>
              Agenda <span>{decidedCount} of {proposals.length} decided</span>
            </div>
            <div className={`${P}-agenda`}>
              {proposals.map((p, i) => (
                <button
                  key={p.id}
                  type="button"
                  className={`${P}-ag ${i === active ? "is-active" : ""} is-${p.status}`}
                  onClick={() => setActive(i)}
                >
                  <span className={`${P}-ag-n`}>{i + 1}</span>
                  <span className={`${P}-ag-body`}>
                    <span className={`${P}-ag-name`}>{p.title}</span>
                    <span className={`${P}-ag-cat`}>{p.category_label}</span>
                  </span>
                  <span className={`${P}-ag-st`}>
                    {p.status === "pending" ? p.forecast?.label || "Open" : `${p.status === "passed" ? "Passed" : "Failed"} ${p.tally?.yes}–${p.tally?.no}`}
                  </span>
                </button>
              ))}
            </div>
            {allDone ? (
              <button type="button" className={`${P}-go`} onClick={() => setStage("adjourned")}>
                Close the meeting
              </button>
            ) : null}
            <ClubSnapshot gov={gov || {}} club={club} />
            <RulesInForce gov={gov || {}} compact />
          </nav>

          <div className={`${P}-center`}>
            <ProposalView
              proposal={current}
              tokens={tokens}
              busy={busy}
              onVote={handleVote}
              onLobby={handleLobby}
              reveal={reveal[current.id] != null ? reveal[current.id] : 999}
              onNext={handleNext}
              isLast={proposals.every((p, i) => i === active || p.status !== "pending")}
              userAbbr={userAbbr}
              userName={userName}
              lobbyMult={lobbyMult}
              lobbyReach={num(gov?.fan_report?.value, 55) >= 80 ? 5 : 4}
            />
            {error ? <p className={`${P}-error`}>{error}</p> : null}
          </div>

          <RoomPanel proposal={current} />
        </div>
      ) : null}

      {stage === "adjourned" ? (
        <Adjourned gov={gov || {}} proposals={proposals} onContinue={onContinue} onFloor={() => setStage("floor")} setGov={setGov} />
      ) : null}
    </section>
  );
}
