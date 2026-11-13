import React, { useMemo, useState } from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

/*
  Bench Buzz newsroom pieces (Worker A, `bba-` namespace): story faces, effect stat lines,
  prose cleanup for old saves, story dedupe, and the Insiders + Archive desks.
  Data rules: read what the save gives us; never invent numbers.
*/

const s = (v, fb = "") => (v == null ? fb : String(v));
const arr = (v) => (Array.isArray(v) ? v : []);

/* ------------------------------------------------------------------ */
/* prose cleanup + structured effects                                   */
/* ------------------------------------------------------------------ */

// "(home stability 37/100)", "(sleep quality 49/100)", "(stress 72)" — engine readings in prose.
const READING_PAREN_RE = /\s*\((?:[a-z][a-z' -]{1,40}\s)?\d{1,3}(?:\.\d+)?(?:\/100)?\)/gi;
const EFFECT_TAIL_RE = /\s*Effect:\s*([^]*?)(?:\s+for about\s+(\d+)\s+days?)?\.?\s*$/i;

/** Effects parsed from legacy "Effect: morale +4.8, focus -3.2 for about 9 days." prose. */
export function parseLegacyEffects(text) {
  const m = s(text).match(EFFECT_TAIL_RE);
  if (!m) return { lines: [], days: 0 };
  const lines = m[1]
    .split(/,\s*/)
    .map((part) => {
      const mm = part.trim().match(/^(.*?)\s*([+-]\d+(?:\.\d+)?)$/);
      if (!mm) return null;
      return { label: mm[1].trim(), value: Number(mm[2]) };
    })
    .filter((row) => row && row.label && Number.isFinite(row.value));
  return { lines, days: Number(m[2]) || 0 };
}

/** Story copy as newspaper prose: drop engine readings and the "Effect:" receipt. */
export function cleanStoryProse(text) {
  let out = s(text);
  if (!out) return "";
  out = out.replace(EFFECT_TAIL_RE, "");
  out = out.replace(READING_PAREN_RE, "");
  // "...at the rink: personal stress is up to 72/100 for a player who..." → "...at the rink."
  out = out.replace(/:\s*[^.:]*\d{1,3}\/100[^.]*\./g, ".");
  const kept = out.replace(/[^.]*\d{1,3}\/100[^.]*\.\s*/g, "");
  out = kept.trim() ? kept : out.replace(/\s*\d{1,3}\/100/g, "");
  out = out.replace(/\s{2,}/g, " ").replace(/\s+([,.])/g, "$1").trim();
  if (out && !/[.!?"”]$/.test(out)) out += ".";
  return out;
}

/** Structured effect lines: backend `effect_lines` first, legacy prose second. */
export function storyEffectLines(raw = {}) {
  const lines = arr(raw.effect_lines)
    .map((row) => ({ label: s(row?.label || row?.key).replace(/_/g, " "), value: Number(row?.value) }))
    .filter((row) => row.label && Number.isFinite(row.value) && row.value !== 0);
  if (lines.length) return { lines, days: Number(raw.effect_days) || 0 };
  const legacy = parseLegacyEffects(raw.summary || raw.short_summary || raw.description || "");
  return legacy;
}

export function BuzzEffects({ lines = [], days = 0, compact = false }) {
  if (!lines.length) return null;
  return (
    <div className={`bba-fx${compact ? " is-compact" : ""}`} aria-label="Effect on the player">
      {lines.slice(0, compact ? 3 : 5).map((row) => (
        <span key={row.label} className={`bba-fx__item ${row.value > 0 ? "is-up" : "is-down"}`}>
          <b>{row.value > 0 ? "+" : ""}{Number(row.value).toFixed(Math.abs(row.value) % 1 ? 1 : 0)}</b>
          {row.label}
        </span>
      ))}
      {days ? <span className="bba-fx__days">{days} day{days === 1 ? "" : "s"}</span> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* dedupe                                                               */
/* ------------------------------------------------------------------ */

function dayNumber(iso) {
  const m = s(iso).match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return null;
  return Math.floor(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])) / 86400000);
}

function dedupeScore(story) {
  const raw = story?.raw || {};
  let n = 0;
  if (s(raw.source) !== "universe_backfill") n += 2;
  if (story?.reporterName) n += 1;
  if (story?.requiresAction) n += 4;
  return n;
}

/**
 * Same headline about the same player/team inside two weeks is one story, not two
 * (old saves still carry reprints from the wire bridge). Keeps the richer copy.
 */
export function dedupeStories(list, windowDays = 14) {
  const byKey = new Map();
  const keep = [];
  arr(list).forEach((story) => {
    const head = s(story?.headline).trim().toLowerCase();
    if (!head) {
      keep.push(story);
      return;
    }
    const who = s(story.playerId || story.playerName || story.teamId || "");
    const key = `${head}|${who}`;
    const prevIdx = byKey.get(key);
    if (prevIdx != null) {
      const prev = keep[prevIdx];
      const a = dayNumber(prev?.date);
      const b = dayNumber(story?.date);
      if (a == null || b == null || Math.abs(a - b) < windowDays) {
        if (dedupeScore(story) > dedupeScore(prev)) keep[prevIdx] = story;
        return;
      }
    }
    byKey.set(key, keep.length);
    keep.push(story);
  });
  return keep;
}

/* ------------------------------------------------------------------ */
/* faces                                                                */
/* ------------------------------------------------------------------ */

function initials(name) {
  const parts = s(name).trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export function teamLogoFor({ teamId, teamName, teamAbbrev } = {}) {
  const name = /^\d+$/.test(s(teamName)) ? "" : s(teamName);
  const abbr = s(teamAbbrev);
  if (!name && !abbr) return "";
  return (
    resolveFranchiseTeamLogo(
      { team_id: teamId, team_name: name, name, team_abbrev: abbr, abbrev: abbr },
      name || abbr
    ) || ""
  );
}

/**
 * Story portrait that fills its frame: real NHL photo when the row carries NHL ids,
 * generated face otherwise, team crest when there is no player.
 */
export function BuzzFace({ name, playerId, position, raw = {}, teamId, teamName, teamAbbrev, size = 84, showLogo = true, className = "" }) {
  const logo = teamLogoFor({ teamId, teamName, teamAbbrev: teamAbbrev || raw.team_abbrev });
  const player = useMemo(
    () =>
      name
        ? {
            ...raw,
            name,
            id: playerId || raw.player_id,
            player_id: playerId || raw.player_id,
            position: position || raw.player_position,
            team_abbrev: teamAbbrev || raw.team_abbrev,
            team_name: teamName,
          }
        : null,
    [name, playerId, position, raw, teamAbbrev, teamName]
  );
  return (
    <span className={`bba-face ${className}`} style={{ width: size, height: size }}>
      {player ? (
        <PlayerHeadshot
          player={player}
          size="lg"
          preferPhoto
          showFlag={false}
          className="bba-face__shot"
          style={{ "--size": `${size}px` }}
        />
      ) : logo ? (
        <img className="bba-face__crest" src={logo} alt="" />
      ) : (
        <span className="bba-face__init">{initials(teamName || "League")}</span>
      )}
      {player && logo && showLogo ? <img className="bba-face__logo" src={logo} alt="" /> : null}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* shared bits                                                          */
/* ------------------------------------------------------------------ */

export function BuzzPageHead({ kicker, title, sub, children }) {
  return (
    <header className="ccx-head bba-head">
      <div>
        {kicker ? <span className="ccx-kicker">{kicker}</span> : null}
        <h2 className="ccx-title">{title}</h2>
        {sub ? <p className="bba-head__sub">{sub}</p> : null}
      </div>
      {children ? <div className="bba-head__side">{children}</div> : null}
    </header>
  );
}

function prettyDay(iso, opts = { month: "short", day: "numeric" }) {
  const m = s(iso).match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return "";
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString(undefined, opts);
}

const CATEGORY_WORDS = {
  locker_room: "Locker room",
  personal_life: "Off ice",
  performance: "Performance",
  goalie: "Crease",
  business: "Business",
  contract: "Contract",
  league_discipline: "Discipline",
  team: "Team",
  media: "Media",
  injury: "Injury",
  trade: "Trade wire",
  rumor: "Trade wire",
  draft: "Draft",
  league: "League",
  storyline: "League",
};

export function categoryWord(cat) {
  const key = s(cat).toLowerCase();
  if (!key) return "League";
  if (CATEGORY_WORDS[key]) return CATEGORY_WORDS[key];
  if (/life|family|home|expense|pregnan|sleep|community|charity/.test(key)) return "Off ice";
  if (/trade|rumou?r/.test(key)) return "Trade wire";
  if (/injur/.test(key)) return "Injury";
  return key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function heatWord(h) {
  const v = Number(h) || 0;
  if (v >= 85) return "Boiling";
  if (v >= 65) return "Hot";
  if (v >= 40) return "Warm";
  if (v > 0) return "Simmering";
  return "";
}

export function BuzzHeat({ value }) {
  const v = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  if (!v) return null;
  const tier = v >= 85 ? "boil" : v >= 65 ? "hot" : v >= 40 ? "warm" : "cool";
  return (
    <span className={`bba-heat bba-heat--${tier}`} title={`Heat ${v}/100`}>
      <span className="bba-heat__bar"><i style={{ width: `${v}%` }} /></span>
      <em>{heatWord(v)}</em>
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Insiders desk                                                        */
/* ------------------------------------------------------------------ */

const LEVELS = [
  { id: "all", label: "All" },
  { id: "confirmed", label: "Confirmed" },
  { id: "reported", label: "Reported" },
  { id: "rumour", label: "Rumours" },
  { id: "private", label: "Off the record" },
];

function levelOf(item) {
  const lvl = s(item?.public_knowledge_level).toLowerCase();
  const kind = s(item?.knowledge_type).toLowerCase();
  if (lvl.includes("confirm") || kind === "fact") return "confirmed";
  if (/rumou?r|develop/.test(lvl) || kind === "claim") return "rumour";
  if (lvl.includes("private") || lvl.includes("internal")) return "private";
  return "reported";
}

const LEVEL_COPY = {
  confirmed: { word: "Confirmed", pips: 4 },
  reported: { word: "Reported", pips: 3 },
  rumour: { word: "Rumour", pips: 2 },
  private: { word: "Off the record", pips: 1 },
};

function Credibility({ item }) {
  const lvl = levelOf(item);
  const meta = LEVEL_COPY[lvl];
  return (
    <span className={`bba-cred bba-cred--${lvl}`} title={`Sourcing: ${meta.word}`}>
      <span className="bba-cred__pips" aria-hidden>
        {[1, 2, 3, 4].map((n) => (
          <i key={n} className={n <= meta.pips ? "is-on" : ""} />
        ))}
      </span>
      {meta.word}
    </span>
  );
}

function storyForItem(item, stories) {
  const sid = s(item?.storyline_id || item?.world_event_id);
  return (
    arr(stories).find((st) => sid && (s(st.storylineId) === sid || s(st.id) === sid)) ||
    arr(stories).find((st) => item?.headline && st.headline === item.headline) ||
    null
  );
}

function Scoop({ item, story, onOpen, lead = false }) {
  const raw = { ...(story?.raw || {}), ...item };
  const fx = storyEffectLines({ ...(story?.raw || {}), summary: item.summary || story?.raw?.summary });
  const summary = cleanStoryProse(item.summary || story?.summary || "");
  const reporter = s(item.reporter_name || item.source_label || "Insider");
  const outlet = s(item.outlet_name);
  return (
    <button type="button" className={`bba-scoop${lead ? " is-lead" : ""}`} onClick={() => onOpen(item)}>
      <BuzzFace
        name={s(item.player_name || story?.playerName)}
        playerId={s(item.player_id || story?.playerId)}
        raw={raw}
        teamId={s(item.team_id || story?.teamId)}
        teamName={s(story?.teamName || item.team_name)}
        size={lead ? 150 : 92}
      />
      <span className="bba-scoop__main">
        <span className="bba-scoop__top">
          <span className="bba-scoop__outlet">{outlet || "Wire"}</span>
          <span className="bba-scoop__cat">{categoryWord(item.category || story?.category)}</span>
          <Credibility item={item} />
          {item.gm_knows_more ? <span className="bba-scoop__more">You know more</span> : null}
        </span>
        <strong className="bba-scoop__head">{s(item.headline || story?.headline || "Desk note")}</strong>
        {summary ? <span className="bba-scoop__text">{summary}</span> : null}
        <BuzzEffects lines={fx.lines} days={fx.days} compact={!lead} />
        <span className="bba-scoop__by">
          <span className="bba-scoop__mono" aria-hidden>{initials(reporter)}</span>
          <span>
            <b>{reporter}</b>
            {outlet ? ` · ${outlet}` : ""}
          </span>
          {item.calendar_iso ? <time>{prettyDay(item.calendar_iso)}</time> : null}
          <BuzzHeat value={item.heat} />
        </span>
      </span>
    </button>
  );
}

export function InsidersDesk({ items = [], stories = [], beatWriters = [], dossiers = [], onOpen, renderDossier }) {
  const [level, setLevel] = useState("all");
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(40);

  const ordered = useMemo(() => {
    const seen = new Set();
    const out = [];
    arr(items)
      .slice()
      .reverse()
      .forEach((item) => {
        const key = `${s(item?.headline).toLowerCase()}|${s(item?.player_id || item?.team_id)}`;
        if (!s(item?.headline) || seen.has(key)) return;
        seen.add(key);
        out.push(item);
      });
    return out;
  }, [items]);

  const counts = useMemo(() => {
    const c = { all: ordered.length, confirmed: 0, reported: 0, rumour: 0, private: 0 };
    ordered.forEach((it) => {
      c[levelOf(it)] += 1;
    });
    return c;
  }, [ordered]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return ordered.filter((it) => {
      if (level !== "all" && levelOf(it) !== level) return false;
      if (!needle) return true;
      return [it.headline, it.summary, it.player_name, it.reporter_name, it.outlet_name]
        .map((x) => s(x).toLowerCase())
        .some((x) => x.includes(needle));
    });
  }, [ordered, level, q]);

  const filedBy = useMemo(() => {
    const m = new Map();
    ordered.forEach((it) => {
      const k = s(it.reporter_name || it.source_label);
      if (k) m.set(k, (m.get(k) || 0) + 1);
    });
    return m;
  }, [ordered]);

  const [lead, ...rest] = filtered;

  return (
    <section className="bba-desk">
      <BuzzPageHead
        kicker="Insider desk"
        title="Insiders"
        sub={`${counts.all} notes on file · ${counts.confirmed} confirmed · ${counts.reported} reported · ${counts.rumour} rumours`}
      />
      <div className="bba-desk__grid">
        <div className="bba-desk__main">
          <div className="bba-bar">
            <div className="bba-bar__opts" role="tablist" aria-label="Sourcing level">
              {LEVELS.map((l) => (
                <button
                  key={l.id}
                  type="button"
                  className={`bba-opt${level === l.id ? " is-on" : ""}`}
                  onClick={() => setLevel(l.id)}
                >
                  {l.label}
                  <b>{counts[l.id] ?? 0}</b>
                </button>
              ))}
            </div>
            <input
              type="search"
              className="bba-search"
              placeholder="Search player, reporter, outlet…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          {!filtered.length ? (
            <div className="bba-none">
              <strong>{ordered.length ? "Nothing matches" : "No private layers yet"}</strong>
              <span>
                {ordered.length
                  ? "Try another sourcing level or clear the search."
                  : "Rumours, claims and confirmed facts land here as reporters work their sources."}
              </span>
            </div>
          ) : (
            <>
              {lead ? <Scoop item={lead} story={storyForItem(lead, stories)} onOpen={onOpen} lead /> : null}
              <div className="bba-scoops">
                {rest.slice(0, limit).map((item, idx) => (
                  <Scoop
                    key={`${s(item.storyline_id || item.world_event_id || item.headline)}-${idx}`}
                    item={item}
                    story={storyForItem(item, stories)}
                    onOpen={onOpen}
                  />
                ))}
              </div>
              {rest.length > limit ? (
                <button type="button" className="bba-more" onClick={() => setLimit((n) => n + 40)}>
                  Show {Math.min(40, rest.length - limit)} more of {rest.length - limit}
                </button>
              ) : null}
            </>
          )}
        </div>
        <aside className="bba-rail">
          {arr(beatWriters).length ? (
            <div className="bba-box">
              <h3>Beat desks</h3>
              <ul className="bba-writers">
                {arr(beatWriters)
                  .slice(0, 12)
                  .map((w) => (
                    <li key={s(w.id || w.name)}>
                      <span className="bba-scoop__mono" aria-hidden>{initials(w.name)}</span>
                      <span className="bba-writers__who">
                        <b>{s(w.name)}</b>
                        <em>{[s(w.outlet), s(w.specialty || w.role).replace(/_/g, " ")].filter(Boolean).join(" · ")}</em>
                      </span>
                      <span className="bba-writers__n" title="Notes filed">{filedBy.get(s(w.name)) || 0}</span>
                    </li>
                  ))}
              </ul>
            </div>
          ) : null}
          <div className="bba-box">
            <h3>Player dossiers</h3>
            {arr(dossiers).length && renderDossier ? (
              <div className="bba-dossiers">{arr(dossiers).slice(0, 8).map((d) => renderDossier(d))}</div>
            ) : (
              <p className="bba-muted">Dossiers publish after the next calendar tick.</p>
            )}
          </div>
        </aside>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Archive desk                                                         */
/* ------------------------------------------------------------------ */

function monthKey(iso) {
  const m = s(iso).match(/^(\d{4})-(\d{2})/);
  return m ? `${m[1]}-${m[2]}` : "undated";
}

function monthLabel(key) {
  if (key === "undated") return "Undated";
  const [y, m] = key.split("-").map(Number);
  const d = new Date(y, m - 1, 1);
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

const PAGE = 60;

export function ArchiveDesk({ eras = [], archive = [], onOpen }) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("all");
  const [eraId, setEraId] = useState("all");
  const [limit, setLimit] = useState(PAGE);

  const eraList = useMemo(() => arr(eras).slice().reverse(), [eras]);

  const all = useMemo(() => {
    const seen = new Set();
    const topIds = new Set();
    arr(eras).forEach((era) =>
      arr(era.top_stories).forEach((st) => topIds.add(s(st.storyline_id || st.headline)))
    );
    const out = [];
    const push = (st) => {
      if (!st || !s(st.headline)) return;
      const key = s(st.storyline_id) || `${s(st.headline).toLowerCase()}|${s(st.calendar_iso).slice(0, 10)}`;
      const hkey = `${s(st.headline).toLowerCase()}|${s(st.player_id || st.team_id)}`;
      if (seen.has(key) || seen.has(hkey)) return;
      seen.add(key);
      seen.add(hkey);
      out.push({ ...st, _top: topIds.has(s(st.storyline_id || st.headline)) });
    };
    arr(archive).forEach(push);
    arr(eras).forEach((era) => arr(era.top_stories).forEach(push));
    out.sort((a, b) => s(b.calendar_iso).localeCompare(s(a.calendar_iso)));
    return out;
  }, [eras, archive]);

  const cats = useMemo(() => {
    const m = new Map();
    all.forEach((st) => {
      const w = categoryWord(st.category);
      m.set(w, (m.get(w) || 0) + 1);
    });
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [all]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return all.filter((st) => {
      if (eraId !== "all" && s(st.season) !== eraId) return false;
      if (cat !== "all" && categoryWord(st.category) !== cat) return false;
      if (!needle) return true;
      return [st.headline, st.summary, st.player_name, st.reporter_name]
        .map((x) => s(x).toLowerCase())
        .some((x) => x.includes(needle));
    });
  }, [all, q, cat, eraId]);

  const groups = useMemo(() => {
    const out = [];
    const idx = new Map();
    filtered.slice(0, limit).forEach((st) => {
      const k = monthKey(st.calendar_iso);
      if (!idx.has(k)) {
        idx.set(k, out.length);
        out.push({ key: k, label: monthLabel(k), rows: [] });
      }
      out[idx.get(k)].rows.push(st);
    });
    return out;
  }, [filtered, limit]);

  const totalOnRecord = arr(eras).reduce((n, e) => n + (Number(e.story_count) || 0), 0);
  const maxCat = cats.length ? cats[0][1] : 1;

  if (!all.length) {
    return (
      <section className="bba-desk">
        <BuzzPageHead kicker="The morgue" title="Archive" sub="Completed beats are filed here — themes, top stories and the moments that defined each season." />
        <div className="bba-none">
          <strong>No sealed eras yet</strong>
          <span>Stories move here once they cool off the front page.</span>
        </div>
      </section>
    );
  }

  return (
    <section className="bba-desk">
      <BuzzPageHead
        kicker="The morgue"
        title="Archive"
        sub={`${all.length} beats on file${totalOnRecord > all.length ? ` (${totalOnRecord} filed this era; older beats roll off the clip file)` : ""}`}
      />
      <div className="bba-desk__grid">
        <div className="bba-desk__main">
          <div className="bba-bar">
            <div className="bba-bar__opts">
              <button type="button" className={`bba-opt${cat === "all" ? " is-on" : ""}`} onClick={() => { setCat("all"); setLimit(PAGE); }}>
                All<b>{all.length}</b>
              </button>
              {cats.slice(0, 9).map(([w, n]) => (
                <button
                  key={w}
                  type="button"
                  className={`bba-opt${cat === w ? " is-on" : ""}`}
                  onClick={() => { setCat(w); setLimit(PAGE); }}
                >
                  {w}
                  <b>{n}</b>
                </button>
              ))}
            </div>
            {eraList.length > 1 ? (
              <select className="bba-search bba-search--sel" value={eraId} onChange={(e) => { setEraId(e.target.value); setLimit(PAGE); }} aria-label="Era">
                <option value="all">All eras</option>
                {eraList.map((era) => (
                  <option key={s(era.season)} value={s(era.season)}>{s(era.label || `Season ${era.season}`)}</option>
                ))}
              </select>
            ) : null}
            <input
              type="search"
              className="bba-search"
              placeholder="Search headline, player, reporter…"
              value={q}
              onChange={(e) => { setQ(e.target.value); setLimit(PAGE); }}
            />
          </div>

          {!filtered.length ? (
            <div className="bba-none">
              <strong>Nothing matches</strong>
              <span>Try another category or clear the search.</span>
            </div>
          ) : (
            groups.map((g) => (
              <div key={g.key} className="bba-month">
                <div className="bba-month__head">
                  <h3>{g.label}</h3>
                  <span>{g.rows.length} {g.rows.length === 1 ? "beat" : "beats"}</span>
                </div>
                <div className="bba-files">
                  {g.rows.map((st, i) => (
                    <button
                      key={`${s(st.storyline_id || st.headline)}-${i}`}
                      type="button"
                      className={`bba-file${st._top ? " is-top" : ""}`}
                      onClick={() => onOpen(st)}
                    >
                      <span className="bba-file__date">
                        <b>{prettyDay(st.calendar_iso, { day: "numeric" }) || "—"}</b>
                        <em>{prettyDay(st.calendar_iso, { month: "short" })}</em>
                      </span>
                      <span className="bba-file__main">
                        <span className="bba-file__top">
                          <span className="bba-scoop__cat">{categoryWord(st.category)}</span>
                          {st._top ? <span className="bba-file__star">Top story</span> : null}
                          <BuzzHeat value={st.heat} />
                        </span>
                        <strong>{s(st.headline)}</strong>
                        {st.summary ? <span className="bba-file__text">{cleanStoryProse(st.summary)}</span> : null}
                        <span className="bba-file__by">
                          {[s(st.player_name), s(st.reporter_name) ? `by ${st.reporter_name}` : ""].filter(Boolean).join(" · ")}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ))
          )}
          {filtered.length > limit ? (
            <button type="button" className="bba-more" onClick={() => setLimit((n) => n + PAGE)}>
              Show {Math.min(PAGE, filtered.length - limit)} more · {filtered.length - limit} left
            </button>
          ) : null}
        </div>

        <aside className="bba-rail">
          {eraList.map((era) => (
            <div key={s(era.season)} className="bba-box">
              <h3>{s(era.label || `Season ${era.season}`)}</h3>
              <div className="bba-era__nums">
                <span><b>{Number(era.story_count || 0)}</b>beats filed</span>
                <span><b>{arr(era.top_stories).length}</b>top stories</span>
              </div>
              {arr(era.themes).length ? (
                <div className="bba-era__themes">
                  {era.themes.map((t) => (
                    <span key={t}>{t}</span>
                  ))}
                </div>
              ) : null}
              {arr(era.top_stories).length ? (
                <ol className="bba-era__tops">
                  {arr(era.top_stories)
                    .slice(0, 6)
                    .map((st, i) => (
                      <li key={`${s(st.storyline_id || st.headline)}-${i}`}>
                        <button type="button" onClick={() => onOpen(st)}>
                          <strong>{s(st.headline)}</strong>
                          <em>{[categoryWord(st.category), prettyDay(st.calendar_iso)].filter(Boolean).join(" · ")}</em>
                        </button>
                      </li>
                    ))}
                </ol>
              ) : null}
            </div>
          ))}
          {cats.length ? (
            <div className="bba-box">
              <h3>By beat</h3>
              <ul className="bba-cats">
                {cats.map(([w, n]) => (
                  <li key={w}>
                    <button type="button" className={cat === w ? "is-on" : ""} onClick={() => { setCat(cat === w ? "all" : w); setLimit(PAGE); }}>
                      <span>{w}</span>
                      <span className="bba-cats__bar"><i style={{ width: `${Math.round((n / maxCat) * 100)}%` }} /></span>
                      <b>{n}</b>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </aside>
      </div>
    </section>
  );
}
