import React, { useState } from "react";
import PlayerHeadshot from "../../PlayerHeadshot";
import "./PlayerSocial.css";

const arr = (v) => (Array.isArray(v) ? v : []);
const s = (v) => (v == null ? "" : String(v));

export function formatFollowers(n) {
  const v = Number(n) || 0;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(v >= 10_000_000 ? 0 : 1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(v >= 100_000 ? 0 : 1)}K`;
  return String(v);
}

function initials(name) {
  return s(name)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase() || "?";
}

const PERSONA_TONE = {
  "Loose cannon": "hot",
  "Hype man": "hype",
  "PR-trained": "cool",
  "Family guy": "warm",
  "Meme lord": "hype",
  "Cryptic poster": "dim",
  "Lifestyle account": "warm",
  Lurker: "dim",
  "All business": "cool",
};

const INCIDENT_LABEL = {
  rant: "Late-night rant",
  rant_coach: "Shot at the coach",
  rant_teammates: "Called out teammates",
  rant_fans: "Went after the fans",
  unfollow: "Unfollowed the team",
  like: "Liked a post about the coach",
  guarantee_fail: "Guarantee blew up",
  guarantee_win: "Guarantee delivered",
  burner_unmasked: "Burner unmasked",
  cryptic: "Cryptic post",
};

export function Avatar({ avatar, playerId, name, color, size = "sm" }) {
  if (avatar && (avatar.headshot_id || avatar.avatar_seed || avatar.nhl_headshot_url || avatar.nhl_player_id || avatar.nhl_id)) {
    return (
      <span className="psx-avatar">
        <PlayerHeadshot player={{ ...avatar, id: playerId || avatar.player_id, name }} size={size} showFlag={false} />
      </span>
    );
  }
  return (
    <span className="sl-post__avatar" aria-hidden style={color ? { background: color } : undefined}>
      {initials(name)}
    </span>
  );
}

function QuoteCard({ quote }) {
  if (!quote) return null;
  return (
    <div className={`psx-quote${quote.deleted ? " is-deleted" : ""}`}>
      {quote.liked_by ? (
        <div className="psx-quote__liked">
          <Avatar avatar={quote.liked_by_avatar} name={quote.liked_by} size="xs" />
          <span>♥ Liked by <b>{quote.liked_by}</b></span>
        </div>
      ) : null}
      <div className="psx-quote__head">
        {quote.author_avatar ? <Avatar avatar={quote.author_avatar} name={quote.name} size="xs" /> : null}
        <strong>{quote.name}</strong>
        <span>{quote.handle}</span>
        {quote.time ? <em>· {quote.time}</em> : null}
        {quote.deleted ? <span className="psx-chip psx-chip--del">deleted</span> : null}
      </div>
      <p>{quote.text}</p>
    </div>
  );
}

function EvidenceCard({ card }) {
  if (!card || card.type !== "bio_diff") return null;
  return (
    <div className="psx-bio">
      <Avatar avatar={card.author_avatar} name={card.handle} size="xs" />
      <div>
        <small>{card.handle} · bio</small>
        <p><s>{card.before}</s></p>
        <p className="psx-bio__after">{card.after}</p>
      </div>
    </div>
  );
}

function Replies({ replies }) {
  const [open, setOpen] = useState(false);
  const rows = arr(replies);
  if (!rows.length) return null;
  const shown = open ? rows : rows.slice(0, 2);
  return (
    <div className="psx-replies" onClick={(e) => e.stopPropagation()} role="presentation">
      {shown.map((r, i) => (
        <div key={`${r.handle}-${i}`} className={`psx-reply${r.player_id ? " is-player" : ""}`}>
          <Avatar avatar={r.author_avatar} playerId={r.player_id} name={r.author_name || r.handle} size="xs" />
          <div>
            <div className="psx-reply__head">
              <strong>{r.author_name || r.handle}</strong>
              {r.verified ? <span className="sl-post__verified">✓</span> : null}
              <span>{r.handle}</span>
            </div>
            <p>{r.text}</p>
            <small>♥ {Number(r.likes || 0).toLocaleString()}</small>
          </div>
        </div>
      ))}
      {rows.length > 2 ? (
        <button type="button" className="psx-linkbtn" onClick={() => setOpen((v) => !v)}>
          {open ? "Hide replies" : `Show ${rows.length - 2} more ${rows.length - 2 === 1 ? "reply" : "replies"}`}
        </button>
      ) : null}
    </div>
  );
}

function Attachments({ attach }) {
  if (!attach) return null;
  if (attach.type === "score") {
    return (
      <div className="sl-attach sl-attach--score">
        <span>{attach.status}</span>
        <b>{attach.away?.abbr} {attach.away?.score}</b>
        <b>{attach.home?.abbr} {attach.home?.score}</b>
        <small>SOG {attach.away?.shots ?? "—"}-{attach.home?.shots ?? "—"} · xG {Number(attach.away?.xg || 0).toFixed(1)}-{Number(attach.home?.xg || 0).toFixed(1)}</small>
        {arr(attach.stars).length ? <small>{arr(attach.stars).map((st, si) => `${si + 1}★ ${st.name} (${st.line})`).join(" · ")}</small> : null}
      </div>
    );
  }
  if (attach.type === "statline" || attach.type === "contract") {
    return (
      <div className="sl-attach sl-attach--player">
        <PlayerHeadshot player={{ ...attach, id: attach.player_id, position: attach.pos }} size="sm" />
        <div>
          <b>{attach.name}</b>
          <small>{[attach.pos, attach.abbr, attach.age ? `${attach.age}y` : ""].filter(Boolean).join(" · ")}</small>
          <small>
            {attach.type === "contract"
              ? `${attach.years}y × $${Number(attach.aav || 0).toFixed(2)}M`
              : `${attach.label ? `${attach.label}: ` : ""}${attach.line}`}
          </small>
        </div>
      </div>
    );
  }
  if (attach.type === "standings") {
    return (
      <div className="sl-attach sl-attach--standings">
        {arr(attach.rows).map((r) => (
          <small key={r.team_id} className={r.focus ? "is-focus" : ""}>
            {r.rank}. {r.abbr} {r.pts} pts ({r.gp} GP) · {Math.round(Number(r.odds || 0) * 100)}%
          </small>
        ))}
      </div>
    );
  }
  return null;
}

/** One Puckr post. Clickable only when there's something to open (story or player). */
export function SocialPostCard({ post, index = 0, onOpen, onAuthor }) {
  const [reveal, setReveal] = useState(false);
  const clickable = Boolean(post.storyId || (post.authorPlayerId && onAuthor));
  const handleClick = () => {
    if (post.storyId) onOpen?.(post);
    else if (post.authorPlayerId) onAuthor?.(post.authorPlayerId);
  };
  const isPlayer = post.authorType === "player";
  const hidden = post.deleted && !reveal;
  const classes = [
    "sl-post",
    "psx-post",
    clickable ? "is-clickable" : "",
    isPlayer ? "is-player" : "",
    post.kind === "player_burner" ? "is-burner" : "",
    post.kind === "player_rant" ? "is-rant" : "",
    post.deleted ? "is-deleted" : "",
  ].filter(Boolean).join(" ");
  return (
    <article
      className={classes}
      style={{ animationDelay: `${Math.min(index, 10) * 24}ms` }}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? handleClick : undefined}
      onKeyDown={clickable ? (e) => { if (e.key === "Enter") handleClick(); } : undefined}
    >
      <div className="sl-post__head">
        <Avatar avatar={post.authorAvatar} playerId={post.authorPlayerId} name={post.name} color={post.color} />
        <strong>{post.name}</strong>
        {post.verified ? <span className="sl-post__verified">✓</span> : null}
        <span>{post.handle}</span>
        {post.personaLabel ? (
          <span className={`psx-chip psx-chip--${PERSONA_TONE[post.personaLabel] || "dim"}`}>{post.personaLabel}</span>
        ) : post.badge && post.badge !== "fan" ? (
          <span className="sl-post__badge">{post.badge}</span>
        ) : null}
        <em>{post.age}{post.time ? ` · ${post.time}` : ""}</em>
      </div>
      {isPlayer && post.followers ? <div className="psx-sub">{formatFollowers(post.followers)} followers</div> : null}
      {post.kind === "player_burner" ? (
        <div className="psx-sub psx-sub--burner">
          Anonymous account{post.burnerSuspicion ? ` · sleuths ${post.burnerSuspicion}% sure who runs it` : ""}
        </div>
      ) : null}
      {hidden ? (
        <div className="psx-deleted">
          <span>This post was deleted. {post.deletedNote ? `(${post.deletedNote.toLowerCase()})` : ""}</span>
          <button type="button" className="psx-linkbtn" onClick={(e) => { e.stopPropagation(); setReveal(true); }}>
            See the screenshot
          </button>
        </div>
      ) : (
        <p className={post.deleted ? "psx-ghost" : ""}>{post.text}</p>
      )}
      <QuoteCard quote={post.quote} />
      <EvidenceCard card={post.evidenceCard} />
      <Attachments attach={post.attach} />
      {post.beef ? <div className="psx-beef">🥊 Beef with {post.beef.with_name}</div> : null}
      <div className="sl-post__meta">
        {post.related && post.related !== post.text && !post.text.includes(post.related) ? <span className="sl-post__related">{post.related}</span> : null}
        {post.cred && post.cred !== "social" ? <span>{post.cred}</span> : null}
        {post.ratioed ? <span className="psx-chip psx-chip--hot">ratioed</span> : null}
        {post.likes != null ? (
          <>
            <span>{Number(post.replies || 0).toLocaleString()} replies</span>
            <span>{Number(post.reposts || 0).toLocaleString()} reposts</span>
            <span>{Number(post.likes || 0).toLocaleString()} likes</span>
          </>
        ) : null}
      </div>
      <Replies replies={post.threadReplies} />
    </article>
  );
}

function Meter({ value, tone = "gold" }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <span className={`psx-meter psx-meter--${tone}`}>
      <i style={{ width: `${v}%` }} />
    </span>
  );
}

/** Right rail for Puckr: your players' accounts, burner watch and the incident log. */
export function PlayerSocialRail({ summary, onMeet, onAuthor, activeAuthor }) {
  const [tab, setTab] = useState("room");
  const accounts = arr(summary?.accounts);
  const watch = arr(summary?.burner_watch);
  const incidents = arr(summary?.incidents);
  const flagged = accounts.filter((a) => a.incident && !a.incident.handled && a.incident.kind !== "guarantee_win");
  return (
    <div className="sl-panel psx-rail">
      <div className="psx-tabs">
        {[
          ["room", `Your room${flagged.length ? ` · ${flagged.length}` : ""}`],
          ["burners", `Burner watch${watch.length ? ` · ${watch.length}` : ""}`],
          ["log", "Incidents"],
        ].map(([id, label]) => (
          <button key={id} type="button" className={tab === id ? "is-active" : ""} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {tab === "room" ? (
        <div className="psx-list">
          {accounts.length ? accounts.map((a) => {
            const inc = a.incident && !a.incident.handled && a.incident.kind !== "guarantee_win" ? a.incident : null;
            return (
              <div key={a.player_id} className={`psx-acct${inc ? " has-incident" : ""}${activeAuthor === a.player_id ? " is-active" : ""}`}>
                <button type="button" className="psx-acct__main" onClick={() => onAuthor?.(a.player_id)} title="Show his posts">
                  <Avatar avatar={a.avatar} playerId={a.player_id} name={a.name} size="xs" />
                  <div>
                    <strong>{a.name}</strong>
                    <small>{a.handle} · {formatFollowers(a.followers)}</small>
                  </div>
                  <span className={`psx-chip psx-chip--${PERSONA_TONE[a.persona] || "dim"}`}>{a.persona}</span>
                </button>
                <div className="psx-acct__bars">
                  <label>Fans <Meter value={a.fan_sentiment} tone={a.fan_sentiment < 40 ? "red" : "green"} /></label>
                  <label>Media heat <Meter value={a.media_stress} tone={a.media_stress > 65 ? "red" : "gold"} /></label>
                </div>
                {a.posting_ban ? <div className="psx-ban">📵 Posting ban in effect</div> : null}
                {inc ? (
                  <div className="psx-incident">
                    <span>⚠ {INCIDENT_LABEL[inc.kind] || "Social incident"}</span>
                    {inc.text ? <q>{s(inc.text).slice(0, 110)}</q> : null}
                    <button type="button" className="psx-meet" onClick={() => onMeet?.(a.player_id)}>Call him in</button>
                  </div>
                ) : null}
              </div>
            );
          }) : <p className="sl-decision-empty">No accounts yet. Sim a day.</p>}
        </div>
      ) : null}

      {tab === "burners" ? (
        <div className="psx-list">
          <p className="psx-note">Anonymous accounts that know too much. Fan sleuths are building cases; sometimes they have the wrong guy.</p>
          {watch.length ? watch.map((b) => (
            <div key={b.handle} className={`psx-burner${b.unmasked ? " is-unmasked" : ""}${b.is_user_team ? " is-mine" : ""}`}>
              <div className="psx-burner__head">
                <strong>{b.handle}</strong>
                {b.is_user_team ? <span className="psx-chip psx-chip--hot">your team</span> : null}
                <em>{b.posts} posts</em>
              </div>
              {b.unmasked ? (
                <div className="psx-burner__verdict">UNMASKED: <b>{b.owner_name}</b></div>
              ) : (
                <>
                  <label className="psx-burner__meter">Case strength <Meter value={b.suspicion} tone={b.suspicion >= 70 ? "red" : "gold"} /><b>{b.suspicion}%</b></label>
                  {b.accused_name ? (
                    <div className="psx-burner__suspect">
                      <Avatar avatar={b.accused_avatar} playerId={b.accused_id} name={b.accused_name} size="xs" />
                      <span>Sleuths suspect <b>{b.accused_name}</b></span>
                    </div>
                  ) : <small className="psx-note">No suspect yet.</small>}
                </>
              )}
              {arr(b.evidence).length ? (
                <ul>{arr(b.evidence).map((e) => <li key={e}>{e}</li>)}</ul>
              ) : null}
            </div>
          )) : <p className="sl-decision-empty">No burner cases open. Yet.</p>}
        </div>
      ) : null}

      {tab === "log" ? (
        <div className="psx-list">
          {incidents.length ? incidents.map((i, idx) => (
            <div key={`${i.player_id}-${i.kind}-${idx}`} className={`psx-log${i.is_user_team ? " is-mine" : ""}`}>
              <Avatar avatar={i.avatar} playerId={i.player_id} name={i.player_name} size="xs" />
              <div>
                <strong>{i.player_name}</strong>
                <small>{INCIDENT_LABEL[i.kind] || i.kind} · {s(i.iso).slice(5)}</small>
                <span className="psx-sev">{"●".repeat(Math.max(1, Number(i.severity) || 1))}</span>
              </div>
            </div>
          )) : <p className="sl-decision-empty">Quiet night online.</p>}
        </div>
      ) : null}
    </div>
  );
}
