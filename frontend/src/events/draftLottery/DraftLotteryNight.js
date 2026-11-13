import React, { useCallback, useEffect, useMemo, useState } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { pickFranchiseData } from "../shared/eventHelpers";
import {
  REVEAL_SPEEDS,
  buildRevealSequence,
  enumerateLotteryOutcomes,
  formatRecord,
  lotteryMeaning,
  normalizeLotteryField,
  normalizeLotteryPicks,
  normalizeProspects,
  pct,
  pickOrdinal,
  prospectLine,
  prospectProduction,
  revealTiming,
  slotProbabilities,
  slotRange,
} from "./draftLotteryHelpers";

const SPEED_KEY = "lot2-reveal-speed";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "";
}

function draftYearOf(franchiseState, raw) {
  const metaYear = Number(raw?.lottery_meta?.draft_year);
  if (Number.isFinite(metaYear) && metaYear > 0) return metaYear;
  const y = Number(franchiseState?.season_year || franchiseState?.seasonYear);
  return Number.isFinite(y) && y > 0 ? y + 1 : null;
}

function readSpeed() {
  try {
    const v = window.localStorage.getItem(SPEED_KEY);
    return REVEAL_SPEEDS[v] ? v : "dramatic";
  } catch {
    return "dramatic";
  }
}

function Logo({ src, label, size = 48, className = "" }) {
  const style = { width: size, height: size };
  if (src) {
    return (
      <span className={`lot2-logo ${className}`} style={style}>
        <img src={src} alt="" loading="lazy" />
      </span>
    );
  }
  return (
    <span className={`lot2-logo lot2-logo--text ${className}`} style={style}>
      {(label || "?").slice(0, 3).toUpperCase()}
    </span>
  );
}

function Move({ movement, big = false }) {
  const n = Number(movement) || 0;
  if (n > 0) return <span className={`lot2-move is-up${big ? " is-big" : ""}`}>▲ {n}</span>;
  if (n < 0) return <span className={`lot2-move is-down${big ? " is-big" : ""}`}>▼ {Math.abs(n)}</span>;
  return <span className={`lot2-move is-hold${big ? " is-big" : ""}`}>=</span>;
}

function Prospect({ p, size = 64, label }) {
  if (!p) return null;
  return (
    <div className="lot2-prospect">
      <PlayerHeadshot
        player={{ ...p, id: p.id, name: p.name, position: p.position, age: p.age }}
        size="md"
        preferPhoto
        showFlag={false}
        style={{ "--size": `${size}px` }}
      />
      <div className="lot2-prospect__text">
        {label ? <div className="lot2-prospect__label">{label}</div> : null}
        <div className="lot2-prospect__name">
          {p.name} <span className="lot2-prospect__pos">{p.position}</span>
        </div>
        <div className="lot2-prospect__meta">{prospectLine(p)}</div>
        <div className="lot2-prospect__meta">
          Ranked #{p.rank}
          {prospectProduction(p) ? ` · ${prospectProduction(p)}` : ""}
          {p.potential_score != null ? ` · ceiling ${Math.round(Number(p.potential_score))}` : ""}
        </div>
      </div>
    </div>
  );
}

function SpeedToggle({ speed, onChange }) {
  return (
    <div className="lot2-speed" role="radiogroup" aria-label="Reveal speed">
      <span className="lot2-speed__label">Reveal speed</span>
      {Object.entries(REVEAL_SPEEDS).map(([key, sp]) => (
        <button
          key={key}
          type="button"
          role="radio"
          aria-checked={speed === key}
          className={`lot2-speed__opt${speed === key ? " is-on" : ""}`}
          onClick={() => onChange(key)}
        >
          {sp.label}
        </button>
      ))}
    </div>
  );
}

/* ───────────────────────── Intro: the field ───────────────────────── */

function FieldTable({ field, baseProbs }) {
  // Columns with no data on this save (older payloads) are dropped so club names get the room.
  const hasRec = field.some((t) => formatRecord(t.record));
  const hasPts = field.some((t) => t.record?.pts != null);
  const hasState = field.some((t) => t.team_state_label);
  const hasFans = field.some((t) => t.fan_mood || t.fan_confidence != null);
  const cols = [
    "34px",
    "minmax(240px, 1.7fr)",
    hasRec ? "84px" : null,
    hasPts ? "40px" : null,
    hasState ? "minmax(96px, 1fr)" : null,
    hasFans ? "minmax(74px, 0.8fr)" : null,
    "minmax(200px, 2.2fr)",
    "minmax(64px, 0.5fr)",
    "minmax(64px, 0.5fr)",
    "minmax(84px, 0.6fr)",
  ].filter(Boolean).join(" ");
  const rowStyle = { gridTemplateColumns: cols };
  return (
    <div className="lot2-field" role="table" aria-label="Lottery field">
      <div className="lot2-field__row lot2-field__row--head" role="row" style={rowStyle}>
        <span role="columnheader">Slot</span>
        <span role="columnheader">Club</span>
        {hasRec ? <span role="columnheader">Record</span> : null}
        {hasPts ? <span role="columnheader">Pts</span> : null}
        {hasState ? <span role="columnheader">State</span> : null}
        {hasFans ? <span role="columnheader">Fans</span> : null}
        <span role="columnheader">Odds at #1</span>
        <span role="columnheader">Top 2</span>
        <span role="columnheader">Top 3</span>
        <span role="columnheader">Range</span>
      </div>
      {field.map((t) => {
        const row = baseProbs.get(t.rank) || [];
        const p1 = row[0] || 0;
        const top2 = (row[0] || 0) + (row[1] || 0);
        const top3 = top2 + (row[2] || 0);
        const { best, worst } = slotRange(row);
        return (
          <div key={t.rank} className={`lot2-field__row${t.is_user ? " is-user" : ""}`} role="row" style={rowStyle}>
            <span className="lot2-field__slot" role="cell">
              {t.rank}
            </span>
            <span className="lot2-field__club" role="cell">
              <Logo src={t.logoSrc} label={t.abbrev || t.team_name} size={44} />
              <span>
                <span className="lot2-field__name">{t.team_name}</span>
                <span className="lot2-field__sub">
                  {t.is_user ? "Your club" : t.gm_name ? `GM ${t.gm_name.replace(/^GM\s+/i, "")}` : ""}
                  {t.is_traded && t.owner_team_name ? ` · pick owned by ${t.owner_team_name}` : ""}
                </span>
              </span>
            </span>
            {hasRec ? (
              <span className="lot2-field__num" role="cell">
                {formatRecord(t.record) || "—"}
              </span>
            ) : null}
            {hasPts ? (
              <span className="lot2-field__num" role="cell">
                {t.record?.pts ?? "—"}
              </span>
            ) : null}
            {hasState ? (
              <span className={`lot2-field__state is-${t.team_state || "none"}`} role="cell">
                {t.team_state_label || "—"}
                {Number(t.points_back) > 0 ? <small>{t.points_back} pts out</small> : null}
              </span>
            ) : null}
            {hasFans ? (
              <span className="lot2-field__fans" role="cell">
                {t.fan_mood || "—"}
                {t.fan_confidence != null ? <small>{t.fan_confidence}/100</small> : null}
              </span>
            ) : null}
            <span className="lot2-field__odds" role="cell">
              <span className="lot2-meter" aria-hidden="true">
                <span style={{ width: `${Math.min(100, p1 * 100 * 4.5)}%` }} />
              </span>
              <b>{pct(p1)}</b>
            </span>
            <span className="lot2-field__num" role="cell">
              {pct(top2)}
            </span>
            <span className="lot2-field__num" role="cell">
              {pct(top3)}
            </span>
            <span className="lot2-field__num" role="cell">
              {best && worst ? (best === worst ? `#${best}` : `#${best}–#${worst}`) : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Intro({ field, baseProbs, prospects, draftYear, userRow, speed, onSpeed, onWatch, onViewAll, hasPicks }) {
  const userProbs = userRow ? baseProbs.get(userRow.rank) || [] : [];
  const userRange = slotRange(userProbs);
  const favourite = field[0];
  const longest = [...field].reverse().find((t) => (baseProbs.get(t.rank) || [])[0] > 0);
  return (
    <main className={`lot2-intro${prospects.length ? "" : " no-prize"}`}>
      <section className="lot2-hero">
        <div className="lot2-hero__year">{draftYear ? `${draftYear} NHL Draft` : "NHL Draft"}</div>
        <div className="lot2-hero__head" role="heading" aria-level={1}>
          Lottery Night
        </div>
        <p className="lot2-hero__lede">
          {field.length} clubs missed the playoffs. Two balls are drawn, one for the first pick and one for the
          second. Any club can win a draw, but nobody climbs more than 10 spots, so a long shot that wins lands
          ten places higher instead.
        </p>
        <div className="lot2-draws" aria-hidden="true">
          <span className="lot2-ball">1</span>
          <span className="lot2-ball">2</span>
          <span className="lot2-draws__text">
            Favourite {favourite?.team_name} {pct((baseProbs.get(favourite?.rank) || [])[0] || 0)}
            {longest && longest !== favourite ? ` · long shot ${longest.team_name} ${pct((baseProbs.get(longest.rank) || [])[0])}` : ""}
          </span>
        </div>

        <dl className="lot2-glance">
          <div>
            <dt>Clubs in the draw</dt>
            <dd>{field.length}</dd>
          </div>
          <div>
            <dt>Balls drawn</dt>
            <dd>2</dd>
          </div>
          <div>
            <dt>Biggest possible jump</dt>
            <dd>10</dd>
          </div>
          <div>
            <dt>Possible draws</dt>
            <dd>{field.length * Math.max(0, field.length - 1)}</dd>
          </div>
        </dl>

        {userRow ? (
          <div className="lot2-fate">
            <div className="lot2-fate__head">Your fate</div>
            <div className="lot2-fate__line">
              <Logo src={userRow.logoSrc} label={userRow.abbrev} size={52} />
              <span>
                {userRow.team_name} enter in slot {userRow.rank} with <b>{pct(userProbs[0] || 0)}</b> to win #1,{" "}
                <b>{pct((userProbs[0] || 0) + (userProbs[1] || 0))}</b> for a top-two pick, and can land anywhere from #
                {userRange.best} to #{userRange.worst}.
              </span>
            </div>
          </div>
        ) : (
          <div className="lot2-fate is-out">
            <div className="lot2-fate__head">Your club</div>
            <div className="lot2-fate__line">
              <span>You made the playoffs or traded your pick, so tonight decides who you face on the draft floor.</span>
            </div>
          </div>
        )}

        {hasPicks ? (
          <div className="lot2-intro__acts">
            <button type="button" className="lot2-go" onClick={onWatch}>
              Watch the reveal
            </button>
            <button type="button" className="lot2-link" onClick={onViewAll}>
              Jump to final order
            </button>
          </div>
        ) : (
          <p className="lot2-empty">Lottery data unavailable.</p>
        )}
        <SpeedToggle speed={speed} onChange={onSpeed} />
      </section>

      <section className="lot2-intro__field">
        <div className="lot2-sect">
          <span className="lot2-sect__head">The field</span>
          <span className="lot2-sect__note">Worst record first. Odds come from all possible draws.</span>
        </div>
        <FieldTable field={field} baseProbs={baseProbs} />
      </section>

      {prospects.length ? (
      <aside className="lot2-intro__prize">
        <div className="lot2-sect">
          <span className="lot2-sect__head">The prize</span>
          <span className="lot2-sect__note">Top of the class</span>
        </div>
        {prospects.length ? (
          <ol className="lot2-prize">
            {prospects.slice(0, 6).map((p) => (
              <li key={p.id}>
                <span className="lot2-prize__rank">{p.rank}</span>
                <Prospect p={p} size={p.rank === 1 ? 84 : 60} />
              </li>
            ))}
          </ol>
        ) : (
          null
        )}
      </aside>
      ) : null}
    </main>
  );
}

/* ───────────────────────── Reveal ───────────────────────── */

function drumrollLine(pickNo, remaining) {
  if (pickNo === 1) return "For the first overall pick";
  if (pickNo === 2) return "Two envelopes left";
  if (pickNo === 3) return "Down to the final three";
  return `${remaining} envelopes left`;
}

function Reveal({
  sequence,
  index,
  phase,
  field,
  fieldByRank,
  outcomes,
  prospects,
  speed,
  onSpeed,
  onSkip,
}) {
  const n = field.length;
  const current = sequence[index];
  const slot = current?.pick;
  const opened = phase === "open";

  const revealedMap = useMemo(() => {
    const m = {};
    sequence.slice(0, index + (opened ? 1 : 0)).forEach((p) => {
      m[p.pick] = Number(p.original_rank) || p.pick;
    });
    return m;
  }, [sequence, index, opened]);

  const { probs } = useMemo(() => slotProbabilities(outcomes, n, revealedMap), [outcomes, n, revealedMap]);
  const revealedRanks = new Set(Object.values(revealedMap));
  const team = current ? fieldByRank.get(Number(current.original_rank) || current.pick) : null;
  const isUser = Boolean(team?.is_user);

  const candidates = field
    .filter((t) => !revealedRanks.has(t.rank) || (opened && team && t.rank === team.rank))
    .map((t) => ({ t, p: (probs.get(t.rank) || [])[slot - 1] || 0 }))
    .filter((c) => c.p > 1e-6)
    .sort((a, b) => b.p - a.p);

  const alive = (s) =>
    field
      .filter((t) => !revealedRanks.has(t.rank))
      .map((t) => ({ t, p: (probs.get(t.rank) || [])[s - 1] || 0 }))
      .filter((c) => c.p > 1e-6)
      .sort((a, b) => b.p - a.p);
  const alive1 = alive(1);
  const alive2 = alive(2);

  const userRow = field.find((t) => t.is_user);
  const userPending = userRow && !revealedRanks.has(userRow.rank);
  const userProbs = userRow ? probs.get(userRow.rank) || [] : [];
  const userRange = slotRange(userProbs);
  const prospectAtSlot = prospects[slot - 1];
  const remaining = sequence.length - index;

  return (
    <main className={`lot2-reveal${isUser && opened ? " is-user-moment" : ""}`}>
      <section className={`lot2-stage${prospectAtSlot ? " has-prospect" : ""}`}>
        <div className="lot2-stage__top">
          <span className="lot2-stage__count">
            Envelope {index + 1} of {sequence.length}
          </span>
          <SpeedToggle speed={speed} onChange={onSpeed} />
        </div>

        <div className="lot2-stage__drum">{!opened ? drumrollLine(slot, remaining) : opened && isUser ? "Your fate is sealed" : ""}</div>

        <div className={`lot2-env${opened ? " is-open" : ""}${isUser && opened ? " is-user" : ""}`} key={slot}>
          <div className="lot2-env__inner">
            <div className="lot2-env__face lot2-env__front">
              <div className="lot2-env__ord">The {pickOrdinal(slot)} pick</div>
              <div className="lot2-env__num">#{slot}</div>
              <div className="lot2-env__seal" aria-hidden="true" />
              <div className="lot2-env__hint">
                {candidates.length === 1
                  ? "Only one club can be in here"
                  : `${candidates.length} clubs could be in this envelope`}
              </div>
            </div>
            <div className="lot2-env__face lot2-env__back">
              {team ? (
                <>
                  <div className="lot2-env__ord">With the {pickOrdinal(slot)} pick</div>
                  <Logo src={current.logoSrc || team.logoSrc} label={team.abbrev} size={168} className="lot2-env__logo" />
                  <div className="lot2-env__team">{current.team_name || team.team_name}</div>
                  {current.is_traded && current.via_team_name ? (
                    <div className="lot2-env__via">via {current.via_team_name}</div>
                  ) : null}
                  <div className="lot2-env__facts">
                    <Move movement={current.movement} big />
                    <span>
                      {Number(current.movement) > 0
                        ? `Jumps from slot ${current.original_rank}`
                        : Number(current.movement) < 0
                          ? `Falls from slot ${current.original_rank}`
                          : `Holds slot ${current.original_rank}`}
                    </span>
                    <span>{team.odds}% draw odds</span>
                    {formatRecord(team.record) ? <span>{formatRecord(team.record)}</span> : null}
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>

        {!opened && candidates.length > 1 ? (
          <div className="lot2-cands">
            {candidates.slice(0, 6).map(({ t, p }) => (
              <span key={t.rank} className={`lot2-cand${t.is_user ? " is-user" : ""}`}>
                <Logo src={t.logoSrc} label={t.abbrev} size={36} />
                <b>{pct(p)}</b>
              </span>
            ))}
          </div>
        ) : null}

        {prospectAtSlot ? (
          <div className="lot2-stage__prospect">
            <Prospect
              p={prospectAtSlot}
              size={92}
              label={`Best available on the board at #${slot}`}
            />
          </div>
        ) : null}
      </section>

      <aside className="lot2-side">
        {userRow ? (
          <div className={`lot2-fate lot2-fate--live${userPending ? "" : " is-done"}`}>
            <div className="lot2-fate__head">Your fate</div>
            <div className="lot2-fate__line">
              <Logo src={userRow.logoSrc} label={userRow.abbrev} size={44} />
              <span>
                {userPending ? (
                  <>
                    Still sealed. <b>{pct(userProbs[0] || 0)}</b> for #1, <b>{pct(userProbs[1] || 0)}</b> for #2, now
                    landing #{userRange.best}
                    {userRange.worst !== userRange.best ? `–#${userRange.worst}` : ""}.
                  </>
                ) : (
                  <>
                    {userRow.team_name} pick <b>#{Object.keys(revealedMap).find((s) => revealedMap[s] === userRow.rank)}</b>.
                  </>
                )}
              </span>
            </div>
          </div>
        ) : null}

        <div className="lot2-alive">
          <div className="lot2-sect">
            <span className="lot2-sect__head">Alive for #1</span>
            <span className="lot2-sect__note">{alive1.length} clubs</span>
          </div>
          <div className="lot2-alive__list">
            {alive1.slice(0, 8).map(({ t, p }) => (
              <span key={t.rank} className={`lot2-alive__item${t.is_user ? " is-user" : ""}`}>
                <Logo src={t.logoSrc} label={t.abbrev} size={30} />
                <span>{t.abbrev || t.team_name}</span>
                <b>{pct(p)}</b>
              </span>
            ))}
            {alive1.length > 8 ? <span className="lot2-muted">+{alive1.length - 8} more</span> : null}
          </div>
          <div className="lot2-sect">
            <span className="lot2-sect__head">Alive for #2</span>
            <span className="lot2-sect__note">{alive2.length} clubs</span>
          </div>
          <div className="lot2-alive__list">
            {alive2.slice(0, 8).map(({ t, p }) => (
              <span key={t.rank} className={`lot2-alive__item${t.is_user ? " is-user" : ""}`}>
                <Logo src={t.logoSrc} label={t.abbrev} size={30} />
                <span>{t.abbrev || t.team_name}</span>
                <b>{pct(p)}</b>
              </span>
            ))}
          </div>
        </div>

        <div className="lot2-sect">
          <span className="lot2-sect__head">The order so far</span>
          <span className="lot2-sect__note">Revealed from #{sequence[0]?.pick} up</span>
        </div>
        <ol className="lot2-ladder">
          {[...sequence].reverse().map((p) => {
            const shown = revealedMap[p.pick] != null;
            const t = shown ? fieldByRank.get(revealedMap[p.pick]) : null;
            const active = p.pick === slot;
            return (
              <li
                key={p.pick}
                className={[
                  "lot2-ladder__row",
                  shown ? "is-shown" : "is-sealed",
                  active ? "is-active" : "",
                  t?.is_user ? "is-user" : "",
                  shown && Number(p.movement) > 0 ? "is-jump" : "",
                  shown && Number(p.movement) < 0 ? "is-fall" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
              >
                <span className="lot2-ladder__num">{p.pick}</span>
                {shown ? (
                  <>
                    <Logo src={p.logoSrc || t?.logoSrc} label={t?.abbrev} size={30} />
                    <span className="lot2-ladder__name">{p.team_name}</span>
                    <Move movement={p.movement} />
                  </>
                ) : (
                  <span className="lot2-ladder__name lot2-muted">{active ? "Opening…" : "Sealed"}</span>
                )}
              </li>
            );
          })}
        </ol>
        <button type="button" className="lot2-link" onClick={onSkip}>
          Skip to the final order
        </button>
      </aside>
    </main>
  );
}

/* ───────────────────────── Final order ───────────────────────── */

function Results({ picks, fieldByRank, prospects, userRow }) {
  const first = picks[0];
  const firstTeam = first ? fieldByRank.get(Number(first.original_rank) || 1) : null;
  const biggestFall = [...picks].sort((a, b) => Number(a.movement) - Number(b.movement))[0];
  const userPick = userRow ? picks.find((p) => Number(p.original_rank) === userRow.rank) : null;
  const headline =
    first && Number(first.movement) > 0
      ? `${first.team_name} jump ${first.movement} spots to win the first pick with ${firstTeam?.odds ?? first.odds}% odds.`
      : first
        ? `${first.team_name} hold on to the first overall pick.`
        : "";

  return (
    <main className={`lot2-results${prospects.length ? "" : " no-pros"}`}>
      <header className="lot2-results__top">
        <div>
          <div className="lot2-hero__year">Final lottery order</div>
          <div className="lot2-results__head" role="heading" aria-level={1}>
            Draft order set
          </div>
        </div>
        <div className="lot2-results__story">
          <p>{headline}</p>
          {biggestFall && Number(biggestFall.movement) < 0 ? (
            <p>
              {biggestFall.team_name} are the big loser, sliding {Math.abs(biggestFall.movement)} from slot{" "}
              {biggestFall.original_rank} to #{biggestFall.pick}.
            </p>
          ) : null}
          {userPick ? (
            <p className="is-user">
              Your club picks #{userPick.pick}
              {Number(userPick.movement) > 0
                ? `, up ${userPick.movement} from slot ${userPick.original_rank}.`
                : Number(userPick.movement) < 0
                  ? `, down ${Math.abs(userPick.movement)} from slot ${userPick.original_rank}.`
                  : ", right where you finished."}
              {prospects[userPick.pick - 1] ? ` Best on the board there: ${prospects[userPick.pick - 1].name}.` : ""}
            </p>
          ) : null}
        </div>
      </header>

      <div className="lot2-order">
        {picks.map((p) => {
          const t = fieldByRank.get(Number(p.original_rank) || p.pick) || {};
          const isUser = Boolean(t.is_user);
          const pros = prospects[p.pick - 1];
          return (
            <article
              key={`${p.pick}-${p.team_id}`}
              className={[
                "lot2-order__row",
                isUser ? "is-user" : "",
                Number(p.movement) > 0 ? "is-jump" : "",
                Number(p.movement) < 0 ? "is-fall" : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              <span className="lot2-order__num">{p.pick}</span>
              <Logo src={p.logoSrc} label={t.abbrev || p.team_name} size={60} />
              <div className="lot2-order__team">
                <div className="lot2-order__name">
                  {p.team_name}
                  {isUser ? <span className="lot2-order__you">Your club</span> : null}
                </div>
                <div className="lot2-order__meta">
                  {p.is_traded && p.via_team_name ? `via ${p.via_team_name} · ` : ""}
                  {formatRecord(t.record) ? `${formatRecord(t.record)} · ` : ""}
                  slot {p.original_rank} · {t.odds ?? p.odds ?? 0}% draw odds
                </div>
                <div className="lot2-order__means">{lotteryMeaning({ ...t, movement: p.movement, pickRow: p }, p.pick)}</div>
              </div>
              <Move movement={p.movement} big />
              <div className="lot2-order__pros">
                {pros ? (
                  <>
                    <PlayerHeadshot
                      player={{ ...pros, id: pros.id, name: pros.name, position: pros.position }}
                      size="sm"
                      preferPhoto
                      showFlag={false}
                      style={{ "--size": "48px" }}
                    />
                    <span>
                      <span className="lot2-order__pname">{pros.name}</span>
                      <span className="lot2-order__pmeta">
                        {pros.position} · ranked #{pros.rank}
                      </span>
                    </span>
                  </>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>

      {prospects.length ? (
        <section className="lot2-class">
          <div className="lot2-sect">
            <span className="lot2-sect__head">Top of the class</span>
            <span className="lot2-sect__note">Public rankings going into the combine</span>
          </div>
          <div className="lot2-class__strip">
            {prospects.slice(0, 8).map((p) => (
              <Prospect key={p.id} p={p} size={72} />
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}

/* ───────────────────────── Root ───────────────────────── */

export default function DraftLotteryNight({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const raw = pickFranchiseData(franchiseState, eventData, ["draft_lottery", "offseason.draft_lottery"]);

  const picks = useMemo(() => normalizeLotteryPicks(raw), [raw]);
  const revealSequence = useMemo(() => buildRevealSequence(picks), [picks]);
  const userTeamId = franchiseState?.user_team_id || franchiseState?.team?.id || raw?.lottery_meta?.user_team_id || "";
  const field = useMemo(() => normalizeLotteryField(raw, picks, userTeamId), [raw, picks, userTeamId]);
  const fieldByRank = useMemo(() => new Map(field.map((t) => [t.rank, t])), [field]);
  const outcomes = useMemo(() => enumerateLotteryOutcomes(field.map((t) => t.odds)), [field]);
  const baseProbs = useMemo(() => slotProbabilities(outcomes, field.length).probs, [outcomes, field.length]);
  const prospects = useMemo(() => normalizeProspects(raw), [raw]);
  const userRow = field.find((t) => t.is_user) || null;
  const draftYear = draftYearOf(franchiseState, raw);

  const [stage, setStage] = useState("intro");
  const [revealIndex, setRevealIndex] = useState(0);
  const [phase, setPhase] = useState("sealed");
  const [speed, setSpeedState] = useState(readSpeed);

  const setSpeed = useCallback((key) => {
    setSpeedState(key);
    try {
      window.localStorage.setItem(SPEED_KEY, key);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const skipToResults = useCallback(() => {
    setStage("results");
    setRevealIndex(revealSequence.length);
  }, [revealSequence.length]);

  const startReveal = useCallback(() => {
    setRevealIndex(0);
    setPhase("sealed");
    setStage("reveal");
  }, []);

  const currentReveal = revealSequence[revealIndex] || null;

  useEffect(() => {
    if (stage !== "reveal" || !currentReveal) return undefined;
    const { sealed, hold } = revealTiming(speed, currentReveal.pick);
    const timer = window.setTimeout(
      () => {
        if (phase === "sealed") {
          setPhase("open");
          return;
        }
        if (revealIndex >= revealSequence.length - 1) {
          setStage("results");
          return;
        }
        setPhase("sealed");
        setRevealIndex((i) => i + 1);
      },
      phase === "sealed" ? sealed : hold
    );
    return () => window.clearTimeout(timer);
  }, [stage, phase, speed, revealIndex, currentReveal, revealSequence.length]);

  return (
    <section className={`lot2-root lot2-root--${stage}`}>
      <header className="lot2-bar">
        <button type="button" className="lot2-link lot2-bar__back" onClick={onBack}>
          ← Hub World
        </button>
        <div className="lot2-bar__mid">{draftYear ? `${draftYear} NHL Draft Lottery` : "NHL Draft Lottery"}</div>
        <div className="lot2-bar__season">{seasonLabel(franchiseState)}</div>
      </header>

      {stage === "intro" && (
        <Intro
          field={field}
          baseProbs={baseProbs}
          prospects={prospects}
          draftYear={draftYear}
          userRow={userRow}
          speed={speed}
          onSpeed={setSpeed}
          onWatch={startReveal}
          onViewAll={skipToResults}
          hasPicks={picks.length > 0}
        />
      )}

      {stage === "reveal" && picks.length && currentReveal ? (
        <Reveal
          sequence={revealSequence}
          index={revealIndex}
          phase={phase}
          field={field}
          fieldByRank={fieldByRank}
          outcomes={outcomes}
          prospects={prospects}
          speed={speed}
          onSpeed={setSpeed}
          onSkip={skipToResults}
        />
      ) : null}

      {stage === "results" && picks.length ? (
        <Results picks={picks} fieldByRank={fieldByRank} prospects={prospects} userRow={userRow} />
      ) : null}

      <footer className="lot2-foot">
        <span className="lot2-foot__note">
          {stage === "results"
            ? "Order locked. The combine is next."
            : stage === "reveal"
              ? "Reveal runs from the last lottery pick up to #1."
              : `${field.length} clubs · 2 draws · max jump 10 spots`}
        </span>
        <button
          type="button"
          className={stage === "results" ? "lot2-go" : "lot2-link"}
          onClick={() => {
            if (stage !== "results") {
              skipToResults();
              return;
            }
            onContinue?.();
          }}
          disabled={stage === "results" && typeof onContinue !== "function"}
        >
          {stage === "results" ? "Enter combine" : "Skip to final order"}
        </button>
      </footer>
    </section>
  );
}
