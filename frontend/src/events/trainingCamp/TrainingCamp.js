import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

/*
 * Training camp opening: the bridge from free agency into the new season.
 * Act 1 (intro): the old season's calendar page tears away and a Zamboni resurfaces the ice,
 * revealing the new season. Act 2 (board): report day — locker room, the coach's whiteboard of
 * job battles, new faces, and the road to opening night. Act 3: blow the whistle to start camp.
 * Data: next_season.training_camp from the backend; older servers fall back to the roster.
 */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const SLOTS = { F: 12, D: 6, G: 2 };
const GROUP_LABEL = { F: "Forwards", D: "Defence", G: "Goalies" };

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function ovr99(v) {
  const n = num(v);
  if (n == null) return null;
  return Math.round(n <= 1.5 ? n * 99 : n);
}

function groupOf(pos) {
  const p = String(pos || "").toUpperCase();
  if (p.startsWith("G")) return "G";
  if (p.startsWith("D") || p === "LD" || p === "RD") return "D";
  return "F";
}

function fmtDate(iso, long = false) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ""));
  if (!m) return "";
  const mon = MONTHS[Number(m[2]) - 1] || "";
  return long ? `${mon} ${Number(m[3])}, ${m[1]}` : `${mon} ${Number(m[3])}`;
}

function daysBetween(a, b) {
  const da = Date.parse(String(a || "").slice(0, 10));
  const db = Date.parse(String(b || "").slice(0, 10));
  if (!Number.isFinite(da) || !Number.isFinite(db)) return null;
  return Math.round((db - da) / 86400000);
}

function money(v) {
  const n = num(v);
  if (n == null) return "";
  return `$${n.toFixed(Math.abs(n) >= 10 ? 1 : 2)}M`;
}

function nickname(teamName) {
  const name = String(teamName || "").trim();
  const multi = ["Maple Leafs", "Red Wings", "Blue Jackets", "Golden Knights"];
  const hit = multi.find((n) => name.endsWith(n));
  if (hit) return hit;
  const parts = name.split(/\s+/);
  return parts.length > 1 ? parts[parts.length - 1] : name;
}

function useFitCount(rowPx) {
  const ref = useRef(null);
  const [n, setN] = useState(6);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const measure = () => setN(Math.max(1, Math.floor(el.clientHeight / rowPx)));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [rowPx]);
  return [ref, n];
}

/** Older servers: no camp block yet. Build what we honestly can from the roster. */
function campFromRoster(franchiseState, ns) {
  const roster = Array.isArray(franchiseState?.roster) ? franchiseState.roster : [];
  if (!roster.length) return null;
  const rows = roster.map((p) => ({
    ...p,
    player_id: String(p.player_id || p.id || ""),
    name: p.name || p.full_name || "",
    position: p.position,
    group: groupOf(p.position),
    age: num(p.age),
    ovr: ovr99(p.ovr ?? p.overall) || 0,
    level: "NHL",
    entry_level: Boolean(p.contract?.is_entry_level),
  }));
  const groups = [];
  const battles = [];
  Object.entries(SLOTS).forEach(([g, slots]) => {
    const list = rows.filter((r) => r.group === g).sort((a, b) => b.ovr - a.ovr);
    if (!list.length) return;
    const cut = list[Math.min(slots, list.length) - 1].ovr;
    const lockN = Math.max(0, slots - (g === "G" ? 1 : 2));
    const contenders = list.slice(lockN).filter((r) => r.ovr >= cut - 6).slice(0, g === "G" ? 3 : 4);
    groups.push({ group: g, label: GROUP_LABEL[g], slots, in_camp: list.length, locks: list.slice(0, lockN), cut_ovr: cut, long_shots: [] });
    if (contenders.length > slots - lockN) {
      battles.push({
        group: g,
        title: g === "G" ? "The backup job" : `Last ${slots - lockN} ${g === "F" ? "forward" : "defence"} spots`,
        open_spots: slots - lockN,
        candidates: contenders.map((r) => ({ ...r, gap: Math.round((r.ovr - cut) * 10) / 10 })),
        closeness: contenders[0].ovr - contenders[contenders.length - 1].ovr,
      });
    }
  });
  battles.sort((a, b) => a.closeness - b.closeness);
  const ages = rows.map((r) => r.age).filter(Boolean);
  return {
    fallback: true,
    season_year: ns?.season_year,
    season_label: ns?.season_label,
    first_opponent: ns?.first_opponent,
    opening_night: ns?.opening_night,
    camp_opens: ns?.preseason_start,
    schedule_games: ns?.schedule_games,
    beats: [
      ns?.preseason_start ? { key: "camp", label: "Camp opens", iso: ns.preseason_start } : null,
      ns?.opening_night ? { key: "opening", label: "Opening night", iso: ns.opening_night } : null,
    ].filter(Boolean),
    in_camp: rows.length,
    nhl_count: rows.length,
    roster_limit: 23,
    avg_age: ages.length ? Math.round((ages.reduce((a, b) => a + b, 0) / ages.length) * 10) / 10 : null,
    avg_ovr: rows.length ? Math.round((rows.reduce((a, r) => a + r.ovr, 0) / rows.length) * 10) / 10 : null,
    new_faces: [],
    rookies: rows.filter((r) => (r.age || 99) <= 21 || r.entry_level).sort((a, b) => b.ovr - a.ovr).slice(0, 8),
    veterans: rows.filter((r) => (r.age || 0) >= 33).sort((a, b) => b.ovr - a.ovr).slice(0, 5),
    summer_risers: [],
    depth: { groups, battles },
    cap: {},
    last_season: {},
    expectation: null,
  };
}

function Face({ row, size = 64 }) {
  const player = useMemo(
    () => ensurePlayerHeadshotFields({ ...row, id: row?.player_id || row?.id, full_name: row?.name }),
    [row]
  );
  return (
    <span className="tcmp-face" style={{ width: size, height: size }}>
      <PlayerHeadshot player={player} size="sm" preferPhoto showFlag={false} style={{ "--size": `${size}px` }} />
    </span>
  );
}

function Intro({ fromLabel, toLabel, teamName, logo, onDone }) {
  useEffect(() => {
    const t = setTimeout(onDone, 3600);
    return () => clearTimeout(t);
  }, [onDone]);
  return (
    <div className="tcmp-intro" onClick={onDone} role="presentation">
      <div className="tcmp-cal" aria-hidden="true">
        <div className="tcmp-cal-rings">
          <i /><i /><i /><i />
        </div>
        <div className="tcmp-cal-sheet is-new">
          <span className="tcmp-cal-sup">Season</span>
          <strong>{toLabel}</strong>
          <span className="tcmp-cal-sub">Camp opens</span>
        </div>
        <div className="tcmp-cal-sheet is-old">
          <span className="tcmp-cal-sup">Season</span>
          <strong>{fromLabel || "Offseason"}</strong>
          <span className="tcmp-cal-sub">Signed, sealed, done</span>
        </div>
      </div>
      <div className="tcmp-intro-words">
        {logo ? <img className="tcmp-intro-logo" src={logo} alt="" draggable={false} /> : null}
        <div className="tcmp-intro-big">Camp opens</div>
        <div className="tcmp-intro-team">{teamName}</div>
      </div>
      <div className="tcmp-rink" aria-hidden="true">
        <div className="tcmp-rink-fresh" />
        <div className="tcmp-zam">
          <div className="tcmp-zam-body">
            <span className="tcmp-zam-cab" />
            <span className="tcmp-zam-driver" />
            <span className="tcmp-zam-word">ZAMBONI</span>
          </div>
          <span className="tcmp-zam-wheel is-a" />
          <span className="tcmp-zam-wheel is-b" />
          <span className="tcmp-zam-mist" />
        </div>
      </div>
      <div className="tcmp-intro-skip">Click to skip</div>
    </div>
  );
}

function Battle({ battle }) {
  const cands = battle.candidates || [];
  const open = num(battle.open_spots) || 1;
  return (
    <div className={`tcmp-battle is-${String(battle.group || "F").toLowerCase()}`}>
      <div className="tcmp-battle-hd">
        <span className="tcmp-marker">{battle.title}</span>
        <span className="tcmp-battle-odds">
          {cands.length} for {open}
          {num(battle.closeness) != null ? ` · ${battle.closeness <= 1 ? "dead even" : `${battle.closeness} OVR spread`}` : ""}
        </span>
      </div>
      <div className="tcmp-mags">
        {cands.map((c, i) => (
          <React.Fragment key={c.player_id || c.name}>
            {i === open ? <span className="tcmp-cutline" aria-label="Cut line"><b>cut line</b></span> : null}
            <div className={`tcmp-mag${i < open ? " is-in" : ""}${c.level === "AHL" ? " is-ahl" : ""}`} style={{ "--tilt": `${((i * 37) % 7) - 3}deg` }}>
              <span className="tcmp-pin" aria-hidden="true" />
              <Face row={c} size={58} />
              <div className="tcmp-mag-body">
                <span className="tcmp-mag-name">{c.name}</span>
                <span className="tcmp-mag-meta">
                  {c.position}
                  {c.age ? ` · ${c.age}` : ""}
                  {c.level === "AHL" ? " · from AHL" : ""}
                  {c.arrival ? ` · ${c.arrival === "Free agency" ? "signed" : c.arrival.toLowerCase()}` : ""}
                </span>
              </div>
              <span className="tcmp-mag-ovr">{Math.round(c.ovr)}</span>
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function DepthTally({ groups }) {
  return (
    <div className="tcmp-tally">
      {groups.map((g) => (
        <div key={g.group} className="tcmp-tally-row">
          <span className="tcmp-tally-name">{g.label}</span>
          <span className="tcmp-tally-dots" aria-label={`${g.locks.length} of ${g.slots} jobs locked`}>
            {Array.from({ length: g.slots }).map((_, i) => (
              <i key={i} className={i < g.locks.length ? "is-lock" : "is-open"} />
            ))}
          </span>
          <span className="tcmp-tally-num">
            {g.locks.length}/{g.slots} locked · {g.in_camp} in camp
          </span>
        </div>
      ))}
    </div>
  );
}

function FaceList({ rows, mode }) {
  const [ref, perPage] = useFitCount(70);
  const [page, setPage] = useState(0);
  useEffect(() => setPage(0), [mode]);
  const pages = Math.max(1, Math.ceil(rows.length / perPage));
  const safe = Math.min(page, pages - 1);
  const slice = rows.slice(safe * perPage, safe * perPage + perPage);
  return (
    <>
      <div className="tcmp-list" ref={ref}>
        {slice.map((r) => (
          <div key={r.player_id || r.name} className="tcmp-row">
            <Face row={r} size={52} />
            <div className="tcmp-row-body">
              <span className="tcmp-row-name">{r.name}</span>
              <span className="tcmp-row-meta">
                {r.position}
                {r.age ? ` · ${r.age}` : ""}
                {num(r.cap_hit) ? ` · ${money(r.cap_hit)}` : ""}
                {num(r.years_left) ? ` × ${r.years_left}y` : ""}
                {r.level === "AHL" ? " · AHL" : ""}
              </span>
            </div>
            {mode === "new" && r.arrival ? (
              <span className={`tcmp-stamp is-${String(r.arrival).toLowerCase().replace(/\s+/g, "-")}`}>{r.arrival}</span>
            ) : mode === "risers" && num(r.summer_delta) ? (
              <span className="tcmp-stamp is-up">+{r.summer_delta}</span>
            ) : null}
            <span className="tcmp-row-ovr">{Math.round(r.ovr)}</span>
          </div>
        ))}
        {!slice.length ? (
          <p className="tcmp-quiet">
            {mode === "new" ? "Same group as last spring. No new arrivals this summer." : "Nobody in this group yet."}
          </p>
        ) : null}
      </div>
      {pages > 1 ? (
        <div className="tcmp-pager">
          <button type="button" disabled={safe <= 0} onClick={() => setPage(safe - 1)} aria-label="Previous page">‹</button>
          <span>{safe + 1} / {pages}</span>
          <button type="button" disabled={safe >= pages - 1} onClick={() => setPage(safe + 1)} aria-label="Next page">›</button>
        </div>
      ) : null}
    </>
  );
}

function Road({ beats, opponent, opponentLogo, scheduleGames }) {
  if (!beats.length) return <div className="tcmp-road is-empty">Calendar loads when the season is generated.</div>;
  const first = beats[0]?.iso;
  return (
    <div className="tcmp-road">
      <div className="tcmp-road-line" aria-hidden="true" />
      <span className="tcmp-road-puck" aria-hidden="true" />
      {beats.map((b, i) => {
        const d = i === 0 ? null : daysBetween(first, b.iso);
        return (
          <div key={b.key} className={`tcmp-stop is-${b.key}${i === 0 ? " is-now" : ""}`}>
            <span className="tcmp-stop-dot" aria-hidden="true" />
            <span className="tcmp-stop-date">{fmtDate(b.iso)}</span>
            <span className="tcmp-stop-name">{b.label}</span>
            {b.key === "opening" && opponent ? (
              <span className="tcmp-stop-vs">
                {opponentLogo ? <img src={opponentLogo} alt="" draggable={false} /> : null}
                vs {nickname(opponent)}
              </span>
            ) : d != null ? (
              <span className="tcmp-stop-in">Day {d + 1}</span>
            ) : (
              <span className="tcmp-stop-in">{scheduleGames ? `${scheduleGames} games ahead` : "Day 1"}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function TrainingCamp({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const ns = eventData?.next_season || franchiseState?.next_season || {};
  const camp = useMemo(() => {
    const c = ns?.training_camp;
    if (c && typeof c === "object" && Object.keys(c).length) return c;
    return campFromRoster(franchiseState, ns) || {};
  }, [ns, franchiseState]);

  const reduced =
    typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [act, setAct] = useState(reduced ? "board" : "intro");
  const [listMode, setListMode] = useState("new");
  const [whistle, setWhistle] = useState(false);
  const endIntro = useCallback(() => setAct("board"), []);

  const teamName = camp.team_name || franchiseState?.user_team?.team_name || franchiseState?.team_name || "Your club";
  const abbr = camp.team_abbr || franchiseState?.user_team?.abbreviation || "";
  const logo = resolveFranchiseTeamLogo({ team_abbrev: abbr || undefined, name: teamName, team_name: teamName, full_name: teamName }, teamName);
  const oppLogo = camp.first_opponent
    ? resolveFranchiseTeamLogo({ name: camp.first_opponent, team_name: camp.first_opponent, full_name: camp.first_opponent }, camp.first_opponent)
    : "";
  const seasonLabel = camp.season_label || ns.season_label || (ns.season_year ? `${ns.season_year}–${Number(ns.season_year) + 1}` : "New season");
  const last = camp.last_season || {};
  const exp = camp.expectation;
  const depth = camp.depth || { groups: [], battles: [] };
  const battles = (depth.battles || []).slice(0, 3);
  const groups = depth.groups || [];
  const beats = Array.isArray(camp.beats) ? camp.beats : [];
  const untilOpening = daysBetween(camp.camp_opens || ns.preseason_start, camp.opening_night || ns.opening_night);
  const newFaces = camp.new_faces || [];
  const risers = camp.summer_risers || [];
  const lists = [
    { key: "new", label: `New faces${camp.new_faces_total ? ` (${camp.new_faces_total})` : ""}`, rows: newFaces },
    { key: "rookies", label: `Rookie watch${camp.rookies_total ? ` (${camp.rookies_total})` : ""}`, rows: camp.rookies || [] },
    risers.length ? { key: "risers", label: "Summer risers", rows: risers } : { key: "vets", label: "Veterans", rows: camp.veterans || [] },
  ];
  const activeList = lists.find((l) => l.key === listMode) || lists[0];
  const overLimit = num(camp.nhl_count) != null && camp.nhl_count > (camp.roster_limit || 23);
  const record = last.w != null ? `${last.w}-${last.l}-${last.otl}` : null;
  const goalDiff = last.gf != null && last.ga != null ? last.gf - last.ga : null;

  const blow = () => {
    if (whistle) return;
    setWhistle(true);
    setTimeout(() => onContinue && onContinue(), reduced ? 50 : 1300);
  };

  return (
    <div className={`tcmp-root is-${act}${whistle ? " is-whistle" : ""}`}>
      {act === "intro" ? (
        <Intro fromLabel={last.season_label} toLabel={seasonLabel} teamName={teamName} logo={logo} onDone={endIntro} />
      ) : null}

      <header className="tcmp-top">
        {onBack ? (
          <button type="button" className="tcmp-back" onClick={onBack}>‹ Hub</button>
        ) : null}
        <div className="tcmp-top-id">
          {logo ? <img className="tcmp-top-logo" src={logo} alt="" draggable={false} /> : null}
          <div>
            <div className="tcmp-sup">Training camp · Report day · {seasonLabel}</div>
            <div className="tcmp-head">{nickname(teamName)} camp opens</div>
          </div>
        </div>
        <div className="tcmp-facts">
          <div className="tcmp-fact">
            <span>Camp opens</span>
            <strong>{fmtDate(camp.camp_opens || ns.preseason_start) || "Soon"}</strong>
          </div>
          <div className="tcmp-fact">
            <span>In camp</span>
            <strong>{camp.in_camp ?? "—"}</strong>
          </div>
          <div className={`tcmp-fact${overLimit ? " is-warn" : ""}`}>
            <span>NHL roster</span>
            <strong>
              {camp.nhl_count ?? "—"}/{camp.roster_limit || 23}
            </strong>
          </div>
          {camp.cap?.space != null ? (
            <div className={`tcmp-fact${camp.cap.space < 0 ? " is-warn" : ""}`}>
              <span>Cap space</span>
              <strong>{money(camp.cap.space)}</strong>
            </div>
          ) : null}
          {untilOpening != null ? (
            <div className="tcmp-fact is-count">
              <span>To opening night</span>
              <strong>{untilOpening} days</strong>
            </div>
          ) : null}
        </div>
      </header>

      <main className="tcmp-main">
        <section className="tcmp-room" aria-label="Locker room">
          <div className="tcmp-sec-hd">Locker room</div>
          {exp ? (
            <div className={`tcmp-sign is-${exp.tone || "neutral"}`}>
              <span className="tcmp-sign-sup">This year's mandate</span>
              <strong>{exp.title}</strong>
              <p>{exp.line}</p>
            </div>
          ) : (
            <div className="tcmp-sign is-neutral">
              <span className="tcmp-sign-sup">This year's mandate</span>
              <strong>Clean sheet</strong>
              <p>Last season's standings load with the camp report.</p>
            </div>
          )}
          {record ? (
            <div className="tcmp-score" aria-label="Last season">
              <div className="tcmp-score-hd">Last season{last.season_label ? ` · ${last.season_label}` : ""}</div>
              <div className="tcmp-score-grid">
                <div><b>{record}</b><span>W-L-OT</span></div>
                <div><b>{last.pts}</b><span>Points</span></div>
                <div><b>{last.league_rank ? `#${last.league_rank}` : "—"}</b><span>League</span></div>
                <div><b>{goalDiff != null ? `${goalDiff > 0 ? "+" : ""}${goalDiff}` : "—"}</b><span>Goal diff</span></div>
              </div>
              {last.champion ? <div className="tcmp-banner">Banner night on opening night</div> : null}
            </div>
          ) : null}
          <div className="tcmp-room-stats">
            {camp.avg_ovr != null ? (
              <div><span>Team OVR</span><b>{camp.avg_ovr}</b></div>
            ) : null}
            {camp.avg_age != null ? (
              <div><span>Avg age</span><b>{camp.avg_age}</b></div>
            ) : null}
            {camp.cap?.payroll != null ? (
              <div><span>Payroll</span><b>{money(camp.cap.payroll)}</b></div>
            ) : null}
          </div>
          {camp.captain ? (
            <div className="tcmp-capt">
              <Face row={camp.captain} size={72} />
              <div>
                <span className="tcmp-capt-c">C</span>
                <strong>{camp.captain.name}</strong>
                <span>Leads the room into camp</span>
              </div>
            </div>
          ) : (camp.veterans || [])[0] ? (
            <div className="tcmp-capt">
              <Face row={camp.veterans[0]} size={72} />
              <div>
                <span className="tcmp-capt-c is-vet">Vet</span>
                <strong>{camp.veterans[0].name}</strong>
                <span>Elder statesman · age {camp.veterans[0].age}</span>
              </div>
            </div>
          ) : null}
        </section>

        <section className="tcmp-board" aria-label="Job battles">
          <div className="tcmp-board-frame">
            <div className="tcmp-board-hd">
              <span className="tcmp-marker is-big">Job battles</span>
              <span className="tcmp-board-note">Closest races near the cut line · OVR</span>
            </div>
            <div className={`tcmp-battles n-${battles.length}`}>
              {battles.map((b) => (
                <Battle key={`${b.group}-${b.title}`} battle={b} />
              ))}
              {!battles.length ? (
                <div className="tcmp-quiet is-board">Every job is spoken for. The lineup is set before the first drill.</div>
              ) : null}
            </div>
            <DepthTally groups={groups} />
            <div className="tcmp-tray" aria-hidden="true">
              <i className="is-red" /><i className="is-blue" /><i className="is-black" /><span className="tcmp-eraser" />
            </div>
          </div>
        </section>

        <section className="tcmp-faces" aria-label="Camp roster">
          <div className="tcmp-tabs" role="tablist">
            {lists.map((l) => (
              <button
                key={l.key}
                type="button"
                role="tab"
                aria-selected={activeList.key === l.key}
                className={activeList.key === l.key ? "is-on" : ""}
                onClick={() => setListMode(l.key)}
              >
                {l.label}
              </button>
            ))}
          </div>
          <FaceList rows={activeList.rows} mode={activeList.key} />
          {camp.fallback ? <p className="tcmp-foot-note">Restart the server for the full camp report.</p> : null}
        </section>
      </main>

      <footer className="tcmp-bottom">
        <div className="tcmp-road-wrap">
          <div className="tcmp-sec-hd">Road to the playoffs</div>
          <Road beats={beats} opponent={camp.first_opponent} opponentLogo={oppLogo} scheduleGames={camp.schedule_games} />
        </div>
        <button type="button" className="tcmp-go" onClick={blow} disabled={whistle}>
          <span className="tcmp-whistle" aria-hidden="true">
            <i className="tcmp-whistle-body" />
            <i className="tcmp-whistle-cord" />
          </span>
          <span className="tcmp-go-words">
            <strong>Blow the whistle</strong>
            <span>Start camp · enter preseason</span>
          </span>
        </button>
      </footer>

      {whistle ? (
        <div className="tcmp-tweet" aria-hidden="true">
          <span className="tcmp-tweet-word">TWEEET!</span>
          <i className="tcmp-drop is-1" />
          <i className="tcmp-drop is-2" />
          <i className="tcmp-drop is-3" />
          <i className="tcmp-drop is-4" />
          <i className="tcmp-drop is-5" />
        </div>
      ) : null}
    </div>
  );
}
