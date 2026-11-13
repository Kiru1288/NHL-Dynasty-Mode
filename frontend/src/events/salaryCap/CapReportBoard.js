import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import { pickFranchiseData } from "../shared/eventHelpers";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import TeamLogoBadge from "../../components/ui/TeamLogoBadge";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";
import {
  formatCapChange,
  formatCapMoney,
  formatCapPercent,
  normalizeMoneyToMillions,
  pickCapReport,
  seasonCapLabel,
  statusTone,
} from "./capReportHelpers";

/* ------------------------------------------------------------------ helpers */

const POS_GROUPS = [
  { key: "F", label: "Forwards" },
  { key: "D", label: "Defence" },
  { key: "G", label: "Goalies" },
];

function asArrayLen(value) {
  return Array.isArray(value) ? value.length : 0;
}

function num(value) {
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function m(value) {
  return normalizeMoneyToMillions(value);
}

function money(value) {
  return formatCapMoney(value);
}

/** Contract-level precision: $0.85M, $10.13M. */
function money2(value) {
  const v = m(value);
  if (v == null) return "—";
  const sign = v < 0 ? "-" : "";
  return `${sign}$${Math.abs(v).toFixed(2)}M`;
}

function pct(value, digits = 1) {
  const v = num(value);
  return v == null ? "—" : `${v.toFixed(digits)}%`;
}

function ordinal(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  const s = ["th", "st", "nd", "rd"];
  const mod = v % 100;
  return `${v}${s[(mod - 20) % 10] || s[mod] || s[0]}`;
}

function seasonLabel(franchiseState) {
  const year = franchiseState?.season_year || franchiseState?.seasonYear;
  return year ? `${year}–${Number(year) + 1}` : "";
}

function shortSeason(label) {
  const raw = String(label || "");
  const parts = raw.split(/[-–]/);
  if (parts.length < 2) return raw;
  return `${parts[0]}–${String(parts[1]).slice(-2)}`;
}

function posGroup(row) {
  if (row?.pos_group) return row.pos_group;
  const p = String(row?.position || "").toUpperCase();
  if (p.startsWith("G")) return "G";
  if (p.startsWith("D") || p === "LD" || p === "RD") return "D";
  return "F";
}

function expiryText(row) {
  const rights = String(row?.rights || "").toUpperCase();
  if (row?.pending_july1 || Number(row?.years_remaining) <= 0) {
    return { when: "This summer", rights };
  }
  const yr = num(row?.expiry_year);
  return { when: yr ? `Summer ${yr}` : `${row?.years_remaining} yrs`, rights };
}

function contractNote(row) {
  const notes = [];
  if (row?.extension_signed && row?.extension_aav) {
    notes.push(`Extended ${money2(row.extension_aav)} × ${row.extension_years || "?"}`);
  }
  if (row?.entry_level) notes.push("Entry-level");
  if (row?.two_way) notes.push("Two-way");
  if (row?.value_label === "Bargain") notes.push("Bargain");
  if (row?.value_label === "Bad") notes.push("Overpaid");
  return notes.join(" · ");
}

function headshotPlayer(row) {
  return { ...row, id: row?.player_id || row?.id, name: row?.name, position: row?.position };
}

function logoFor(name, abbrev) {
  return resolveFranchiseTeamLogo({ name, team_name: name, abbrev }, name);
}

/* --------------------------------------------------------------- components */

function Figure({ label, value, sub, tone = "" }) {
  return (
    <div className={`capr2-figure${tone ? ` is-${tone}` : ""}`}>
      <span className="capr2-figure-label">{label}</span>
      <strong className="capr2-figure-value">{value}</strong>
      {sub ? <span className="capr2-figure-sub">{sub}</span> : null}
    </div>
  );
}

function SectionHead({ title, note }) {
  return (
    <div className="capr2-sechead">
      <div className="capr2-sechead-name" role="heading" aria-level={2}>{title}</div>
      {note ? <p className="capr2-sechead-note">{note}</p> : null}
    </div>
  );
}

function CapMeter({ cap, floor, payroll, byPosition }) {
  const capV = m(cap) || 0;
  const payV = m(payroll) || 0;
  const scale = Math.max(capV, payV, 1);
  const segs = POS_GROUPS.map((g) => {
    const row = (byPosition || []).find((r) => r.group === g.key) || {};
    return { key: g.key, label: g.label, value: m(row.total) || 0, count: row.count || 0 };
  }).filter((s) => s.value > 0);
  const posSum = segs.reduce((a, s) => a + s.value, 0);
  const other = Math.max(0, payV - posSum);
  if (!segs.length && payV > 0) {
    segs.push({ key: "P", label: "Payroll", value: payV, count: 0 });
  } else if (other > 0.01) {
    segs.push({ key: "O", label: "Other charges", value: other, count: 0 });
  }
  const floorV = m(floor);
  const floorLeft = floorV != null ? Math.min(100, (floorV / scale) * 100) : null;
  const usedPct = capV > 0 ? (payV / capV) * 100 : 0;
  return (
    <div className="capr2-meter">
      <div className="capr2-meter-top">
        <span>
          <b>{money(payV)}</b> committed · {pct(usedPct, 0)} of the cap
        </span>
        <span>
          Ceiling <b>{money(capV)}</b>
        </span>
      </div>
      <div className="capr2-meter-track" role="img" aria-label={`${Math.round(usedPct)}% of the salary cap used`}>
        {segs.map((s) => (
          <span
            key={s.key}
            className={`capr2-meter-seg is-${s.key}`}
            style={{ width: `${(s.value / scale) * 100}%` }}
            title={`${s.label}: ${money2(s.value)}${s.count ? ` (${s.count} players)` : ""}`}
          />
        ))}
        {floorLeft != null ? (
          <i className="capr2-meter-floor" style={{ left: `${floorLeft}%` }} title={`Cap floor ${money(floorV)}`} />
        ) : null}
      </div>
      <div className="capr2-meter-legend">
        {segs.map((s) => (
          <span key={s.key} className="capr2-legend-item">
            <i className={`capr2-swatch is-${s.key}`} />
            {s.label} <b>{money(s.value)}</b>
          </span>
        ))}
        {floorV != null ? (
          <span className="capr2-legend-item">
            <i className="capr2-swatch is-floor" />
            Floor <b>{money(floorV)}</b>
          </span>
        ) : null}
        <span className="capr2-legend-item">
          <i className="capr2-swatch is-space" />
          Open <b>{money(Math.max(0, capV - payV))}</b>
        </span>
      </div>
    </div>
  );
}

const SORTS = {
  cap: (a, b) => (num(b.cap_hit) || 0) - (num(a.cap_hit) || 0),
  ovr: (a, b) => (num(b.ovr) || 0) - (num(a.ovr) || 0),
  age: (a, b) => (num(a.age) || 0) - (num(b.age) || 0),
  yrs: (a, b) => (num(b.years_remaining) || 0) - (num(a.years_remaining) || 0),
};

/** How many fixed-height rows fit in a box (minus a header), re-measured on resize. The report never scrolls. */
function useFitCount(rowPx, headPx = 0) {
  const ref = useRef(null);
  const [n, setN] = useState(10);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const measure = () => setN(Math.max(2, Math.floor((el.clientHeight - headPx) / rowPx)));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [rowPx, headPx]);
  return [ref, n];
}

function Pager({ page, pages, onPage, label }) {
  if (pages <= 1) return label ? <span className="capr2-pager-note">{label}</span> : null;
  return (
    <div className="capr2-pager" role="group" aria-label="Pages">
      {label ? <span className="capr2-pager-note">{label}</span> : null}
      <button type="button" className="capr2-pager-step" disabled={page <= 0} onClick={() => onPage(page - 1)} aria-label="Previous page">
        ‹
      </button>
      <span>
        Page <b>{page + 1}</b> of {pages}
      </span>
      <button type="button" className="capr2-pager-step" disabled={page >= pages - 1} onClick={() => onPage(page + 1)} aria-label="Next page">
        ›
      </button>
    </div>
  );
}

function ContractTable({ contracts, cap }) {
  const [level, setLevel] = useState("NHL");
  const [sortKey, setSortKey] = useState("cap");
  const counts = useMemo(() => {
    const nhl = contracts.filter((r) => r.level === "NHL").length;
    return { NHL: nhl, MINORS: contracts.length - nhl, ALL: contracts.length };
  }, [contracts]);
  const rows = contracts.filter((r) =>
    level === "ALL" ? true : level === "NHL" ? r.level === "NHL" : r.level !== "NHL"
  );
  const capV = m(cap) || 0;
  const maxHit = Math.max(0.01, ...contracts.map((r) => num(r.cap_hit) || 0));
  const items = [];
  POS_GROUPS.forEach((g) => {
    const list = rows.filter((r) => posGroup(r) === g.key).sort(SORTS[sortKey]);
    if (!list.length) return;
    const total = list.reduce((a, r) => a + (num(r.cap_hit) || 0), 0);
    items.push({ kind: "g", g, list, total });
    list.forEach((r) => items.push({ kind: "r", g, r }));
  });
  const [wrapRef, perPage] = useFitCount(50, 44);
  const [pageRaw, setPage] = useState(0);
  // Pages are built greedily; a page that starts mid-group repeats that group's header.
  const pages = [];
  let cur = [];
  let curGroup = null;
  items.forEach((it) => {
    if (it.kind === "g") curGroup = it;
    if (cur.length >= perPage || (it.kind === "g" && cur.length >= perPage - 1)) {
      pages.push(cur);
      cur = it.kind === "r" && curGroup ? [{ ...curGroup, cont: true }] : [];
    }
    cur.push(it);
  });
  if (cur.length) pages.push(cur);
  const pageCount = Math.max(1, pages.length);
  const page = Math.min(pageRaw, pageCount - 1);
  const pageItems = pages[page] || [];

  const sortHead = (key, label, cls = "") => (
    <th className={cls} aria-sort={sortKey === key ? "descending" : "none"}>
      <span
        role="button"
        tabIndex={0}
        className={`capr2-sort${sortKey === key ? " is-on" : ""}`}
        onClick={() => {
          setSortKey(key);
          setPage(0);
        }}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSortKey(key)}
      >
        {label}
      </span>
    </th>
  );

  return (
    <section className="capr2-block capr2-contracts">
      <div className="capr2-contracts-head">
        <SectionHead
          title="Contracts"
          note={`Every deal in the organisation, grouped by position. ${counts.NHL} NHL · ${counts.MINORS} AHL/ECHL.`}
        />
        <div className="capr2-switch" role="tablist" aria-label="Roster level">
          {[
            ["NHL", `NHL (${counts.NHL})`],
            ["MINORS", `AHL & ECHL (${counts.MINORS})`],
            ["ALL", `All (${counts.ALL})`],
          ].map(([key, label]) => (
            <span
              key={key}
              role="tab"
              tabIndex={0}
              aria-selected={level === key}
              className={`capr2-switch-opt${level === key ? " is-on" : ""}`}
              onClick={() => {
                setLevel(key);
                setPage(0);
              }}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (setLevel(key), setPage(0))}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
      <Pager page={page} pages={pageCount} onPage={setPage} label={`${rows.length} contracts`} />
      <div className="capr2-table-wrap" ref={wrapRef}>
        <table className="capr2-table">
          <thead>
            <tr>
              <th className="is-player">Player</th>
              {sortHead("age", "Age", "is-num")}
              {sortHead("ovr", "OVR", "is-num")}
              {sortHead("cap", "Cap hit", "is-num")}
              <th className="is-share">Share of cap</th>
              {sortHead("yrs", "Years left", "is-num")}
              <th>Expires</th>
              <th>Clause</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {pageItems.map((it) => {
              if (it.kind === "g") {
                return (
                  <tr key={`g-${it.g.key}-${it.cont ? "c" : ""}`} className="capr2-group-row">
                    <td colSpan={9}>
                      <i className={`capr2-swatch is-${it.g.key}`} />
                      <b>
                        {it.g.label}
                        {it.cont ? " (continued)" : ""}
                      </b>
                      <span>
                        {it.list.length} contracts · {money(it.total)}
                        {capV > 0 ? ` · ${pct((it.total / capV) * 100)} of the cap` : ""}
                      </span>
                    </td>
                  </tr>
                );
              }
              const { r, g } = it;
              const exp = expiryText(r);
              const hit = num(r.cap_hit) || 0;
              return (
                <tr key={r.player_id || r.name} className={r.pending_july1 ? "is-expiring" : ""}>
                  <td className="is-player">
                    <div className="capr2-who">
                      <span className="capr2-face capr2-face--sm">
                        <PlayerHeadshot player={headshotPlayer(r)} size="sm" preferPhoto showFlag={false} />
                      </span>
                      <span className="capr2-who-text">
                        <b>{r.name}</b>
                        <span>
                          {r.position}
                          {r.level !== "NHL" ? ` · ${r.level}` : ""}
                        </span>
                      </span>
                    </div>
                  </td>
                  <td className="is-num">{r.age || "—"}</td>
                  <td className="is-num">{num(r.ovr) != null ? Math.round(r.ovr) : "—"}</td>
                  <td className="is-num is-money">{hit > 0 ? money2(hit) : <span className="capr2-dim">Exempt</span>}</td>
                  <td className="is-share">
                    <span className="capr2-share">
                      <span className="capr2-share-bar">
                        <i className={`is-${g.key}`} style={{ width: `${(hit / maxHit) * 100}%` }} />
                      </span>
                      <span>{r.cap_pct != null ? pct(r.cap_pct) : "—"}</span>
                    </span>
                  </td>
                  <td className="is-num">{r.pending_july1 ? 0 : r.years_remaining}</td>
                  <td>
                    {exp.when}
                    {exp.rights ? <span className={`capr2-rights is-${exp.rights.toLowerCase()}`}> {exp.rights}</span> : null}
                  </td>
                  <td>{r.clause || <span className="capr2-dim">—</span>}</td>
                  <td className="capr2-note">{contractNote(r) || <span className="capr2-dim">—</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!rows.length ? <p className="capr2-empty">No contracts at this level.</p> : null}
      </div>
    </section>
  );
}

function PositionSplit({ byPosition, cap, note }) {
  const capV = m(cap) || 0;
  const total = (byPosition || []).reduce((a, r) => a + (m(r.total) || 0), 0);
  if (!total) return null;
  return (
    <section className="capr2-block">
      <SectionHead title="Cap by position" note={note || "NHL contracts on next season's books."} />
      <div className="capr2-posbars">
        {POS_GROUPS.map((g) => {
          const row = (byPosition || []).find((r) => r.group === g.key) || {};
          const v = m(row.total) || 0;
          return (
            <div key={g.key} className="capr2-posrow" title={`${g.label}: ${money2(v)}`}>
              <div className="capr2-posrow-top">
                <b>{g.label}</b>
                <span className="capr2-posrow-val">{money(v)}</span>
              </div>
              <div className="capr2-hbar">
                <i className={`is-${g.key}`} style={{ width: `${(v / total) * 100}%` }} />
              </div>
              <div className="capr2-posrow-sub">
                {pct((v / total) * 100, 0)} of payroll
                {capV > 0 ? ` · ${pct((v / capV) * 100)} of cap` : ""} · {row.count || 0} players · avg {money2(row.avg)}
                {row.top ? ` · top: ${row.top}` : ""}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TopHits({ contracts, cap }) {
  const top = [...(contracts || [])].sort((a, b) => (num(b.cap_hit) || 0) - (num(a.cap_hit) || 0)).slice(0, 6);
  if (!top.length) return null;
  const capV = m(cap) || 0;
  return (
    <section className="capr2-block">
      <SectionHead title="Biggest cap hits" note="Where the money goes." />
      <ol className="capr2-tophits">
        {top.map((r) => {
          const hit = num(r.cap_hit) || 0;
          return (
            <li key={r.player_id || r.name}>
              <span className="capr2-face capr2-face--sm">
                <PlayerHeadshot player={headshotPlayer(r)} size="sm" preferPhoto showFlag={false} />
              </span>
              <span className="capr2-who-text">
                <b>{r.name}</b>
                <span>
                  {r.position} · {r.years_remaining != null ? `${r.years_remaining} yr${Number(r.years_remaining) === 1 ? "" : "s"} left` : ""}
                  {r.clause ? ` · ${r.clause}` : ""}
                </span>
              </span>
              <span className="capr2-tophits-money">
                <b>{money2(hit)}</b>
                <span>{capV > 0 ? pct((hit / capV) * 100) : ""}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Commitments({ commitments, note }) {
  if (!commitments?.length) return null;
  const scale = Math.max(...commitments.map((c) => m(c.cap) || 0), 1);
  return (
    <section className="capr2-block">
      <SectionHead title="Future commitments" note={note || "Signed NHL money by season against the cap (later caps are league projections)."} />
      <div className="capr2-commit">
        {commitments.map((c) => {
          const cv = m(c.committed) || 0;
          const capV = m(c.cap) || 0;
          return (
            <div key={c.season} className="capr2-commit-row" title={`${shortSeason(c.season)}: ${money2(cv)} committed, ${money2(c.space)} open`}>
              <span className="capr2-commit-season">{shortSeason(c.season)}</span>
              <div className="capr2-commit-track" style={{ width: `${(capV / scale) * 100}%` }}>
                <i style={{ width: `${capV > 0 ? Math.min(100, (cv / capV) * 100) : 0}%` }} />
              </div>
              <span className="capr2-commit-val">
                <b>{money(cv)}</b> · {c.players} players · {money(c.space)} open
                {c.projected_cap ? " · cap est." : ""}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CapHistory({ history, floor }) {
  if (!history?.length) return null;
  const max = Math.max(...history.map((h) => m(h.upper) || 0), 1);
  return (
    <section className="capr2-block">
      <SectionHead title="Cap history" note={`Upper limit by season, floor marked inside each bar${floor != null ? ` (this year ${money(floor)})` : ""}.`} />
      <div className="capr2-hist">
        {history.map((h) => {
          const up = m(h.upper) || 0;
          const lo = m(h.lower) || 0;
          return (
            <div
              key={h.season}
              className={`capr2-hist-col${h.is_current ? " is-current" : ""}${h.projected ? " is-projected" : ""}`}
              title={`${shortSeason(h.season)}: cap ${money(up)}, floor ${money(lo)}${h.projected ? " (projected)" : ""}`}
            >
              <span className="capr2-hist-val">{money(up)}</span>
              <div className="capr2-hist-bar-wrap">
                <div className="capr2-hist-bar" style={{ height: `${(up / max) * 100}%` }}>
                  <i className="capr2-hist-floor" style={{ bottom: `${up > 0 ? (lo / up) * 100 : 0}%` }} />
                </div>
              </div>
              <span className="capr2-hist-season">{shortSeason(h.season)}</span>
              <span className={`capr2-hist-chg${num(h.change_pct) < 0 ? " is-down" : ""}`}>
                {h.change_pct != null ? formatCapPercent(h.change_pct) : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Obligations({ obligations, deadCap, retained, bonus }) {
  const o = obligations || {};
  const rows = [
    ["Buried in the minors", o.buried],
    ["Buyouts", o.buyouts],
    ["Retained salary", o.retained ?? retained],
    ["Bonus overage carried in", o.bonus_overage ?? bonus],
    ["Performance-bonus reserve", o.bonus_reserve],
    ["Other dead cap", o.other_dead],
    ["LTIR cap hit", o.ltir_cap_hit],
  ];
  const nonZero = rows.filter(([, v]) => (m(v) || 0) > 0.0005);
  const items = Array.isArray(o.items) ? o.items : [];
  return (
    <section className="capr2-block">
      <SectionHead
        title="Dead cap & other charges"
        note={
          o.retained_slots_max != null
            ? `Retention slots used: ${o.retained_slots_used || 0} of ${o.retained_slots_max}.${(m(o.ltir_pool) || 0) > 0 ? ` LTIR pool ${money2(o.ltir_pool)}.` : ""}`
            : ""
        }
      />
      {nonZero.length ? (
        <dl className="capr2-deflist">
          {nonZero.map(([label, v]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{money2(v)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="capr2-plain">
          Clean books: no buried contracts, buyouts, retained salary or bonus carryover
          {(m(deadCap) || 0) > 0 ? ` (dead cap ${money2(deadCap)})` : ""}. Every dollar of payroll is on an active player.
        </p>
      )}
      {items.length ? (
        <ul className="capr2-items">
          {items.map((it, i) => (
            <li key={`${it.kind}-${it.name}-${i}`}>
              <span>
                <b>{it.name}</b> · {it.kind}
                {it.detail ? ` · ${it.detail}` : ""}
                {it.seasons_remaining != null ? ` · ${it.seasons_remaining} seasons left` : ""}
              </span>
              <span className="capr2-items-amt">{money2(it.amount)}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function Decisions({ expiringNow, expiringNext, capSpace }) {
  const top = (expiringNow || []).slice(0, 6);
  const bill = top.reduce((a, r) => a + (m(r.est_aav) || 0), 0);
  const left = (m(capSpace) || 0) - bill;
  return (
    <section className="capr2-block capr2-decisions">
      <SectionHead
        title="Key decisions"
        note={
          top.length
            ? `Top re-sign candidates by rating. Keeping all ${top.length} at the estimated market ask costs about ${money(bill)} and leaves ${money(left)} of space.`
            : "No pending free agents from your organisation this summer."
        }
      />
      {top.length ? (
        <div className="capr2-resign">
          {top.map((r) => (
            <div key={r.player_id || r.name} className="capr2-resign-item">
              <span className="capr2-face capr2-face--lg">
                <PlayerHeadshot player={headshotPlayer(r)} size="lg" preferPhoto showFlag={false} />
              </span>
              <div className="capr2-resign-body">
                <b className="capr2-resign-name">{r.name}</b>
                <span className="capr2-resign-meta">
                  {r.position} · {r.age || "?"} yrs · {num(r.ovr) != null ? `${Math.round(r.ovr)} OVR` : ""}
                  {num(r.potential) ? ` · ${Math.round(r.potential)} POT` : ""}
                </span>
                <span className="capr2-resign-meta">
                  <span className={`capr2-rights is-${String(r.rights || "").toLowerCase()}`}>{r.rights || "FA"}</span>
                  {(m(r.cap_hit) || 0) > 0 ? ` · was ${money2(r.cap_hit)}` : ""}
                </span>
                <span className="capr2-resign-ask">
                  {m(r.est_aav) != null ? (
                    <>
                      Est. ask <b>{money2(r.est_aav)}</b>
                      {r.est_term ? ` × ${r.est_term} yrs` : ""}
                    </>
                  ) : (
                    "No estimate yet"
                  )}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {(expiringNow || []).length > top.length ? (
        <p className="capr2-plain capr2-more">
          Also expiring:{" "}
          {(expiringNow || [])
            .slice(top.length, top.length + 12)
            .map((r) => `${r.name} (${r.position}, ${r.rights || "FA"})`)
            .join(", ")}
          {(expiringNow || []).length > top.length + 12 ? ` and ${(expiringNow || []).length - top.length - 12} more` : ""}.
        </p>
      ) : null}
      {expiringNext?.length ? (
        <>
          <div className="capr2-subhead" role="heading" aria-level={3}>Expiring after next season</div>
          <ul className="capr2-nextlist">
            {expiringNext.map((r) => (
              <li key={r.player_id || r.name}>
                <span className="capr2-face capr2-face--xs">
                  <PlayerHeadshot player={headshotPlayer(r)} size="sm" preferPhoto showFlag={false} />
                </span>
                <span className="capr2-nextlist-name">
                  <b>{r.name}</b> {r.position} · {r.age}
                </span>
                <span className={`capr2-rights is-${String(r.rights || "").toLowerCase()}`}>{r.rights}</span>
                <span className="capr2-nextlist-amt">{money2(r.cap_hit)}</span>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}

function LeagueTable({ rows, cap, rank, teamCount }) {
  if (!rows?.length) return null;
  const capV = m(cap) || 0;
  const maxSpace = Math.max(...rows.map((r) => Math.abs(m(r.cap_space) || 0)), 1);
  return (
    <section className="capr2-block capr2-league">
      <SectionHead
        title="League cap space"
        note={rank ? `You rank ${ordinal(rank)} of ${teamCount || rows.length} in open cap space.` : "All clubs, most space first."}
      />
      <table className="capr2-ltable">
        <thead>
          <tr>
            <th className="is-num">#</th>
            <th>Team</th>
            <th className="is-num">Payroll</th>
            <th className="is-num">Space</th>
            <th className="is-bar">Room under {money(capV)}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const space = m(r.cap_space) || 0;
            return (
              <tr key={r.team_id} className={r.is_user ? "is-user" : ""}>
                <td className="is-num">{r.rank}</td>
                <td>
                  <span className="capr2-lteam">
                    <TeamLogoBadge teamLogo={logoFor(r.team_name, r.abbrev)} teamName={r.team_name} size={30} variant="badge" />
                    <span>{r.team_name}</span>
                  </span>
                </td>
                <td className="is-num">{money(r.payroll)}</td>
                <td className={`is-num is-money${space < 0 ? " is-neg" : ""}`}>{money(space)}</td>
                <td className="is-bar">
                  <span className="capr2-hbar capr2-hbar--thin">
                    <i className={space < 0 ? "is-neg" : r.is_user ? "is-user" : "is-team"} style={{ width: `${(Math.abs(space) / maxSpace) * 100}%` }} />
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

/* --------------------------------------------------------------------- main */

export default function CapReportBoard({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const raw = pickFranchiseData(franchiseState, eventData, ["salary_cap", "cap", "offseason.salary_cap"]) || {};

  const report = useMemo(() => pickCapReport({ salary_cap: raw }, franchiseState), [franchiseState, raw]);
  const capPayload = raw?.cap_report || raw?.capReport || {};
  const d = capPayload?.details || {};
  const serverContracts = Array.isArray(d.contracts) ? d.contracts : [];
  const [tabRaw, setTab] = useState("overview");
  // Older saves/servers: build the contract table, position split and commitments from the roster's own contracts.
  const rosterRows = Array.isArray(franchiseState?.roster) ? franchiseState.roster : [];
  const capForPct = m(pickCapReport({ salary_cap: raw }, franchiseState)?.currentCap) || 0;
  const fallback = useMemo(() => {
    if (serverContracts.length || !rosterRows.length) return null;
    const rows = rosterRows
      .filter((p) => p && p.contract)
      .map((p) => {
        const c = p.contract || {};
        const hit = num(c.cap_hit ?? c.aav ?? c.salary) || 0;
        const yrs = num(c.years_remaining ?? c.years ?? c.term);
        return {
          ...p,
          player_id: p.id || p.player_id,
          name: p.name,
          position: p.position,
          age: p.age,
          ovr: num(p.ovr),
          cap_hit: hit,
          cap_pct: capForPct > 0 ? (hit / capForPct) * 100 : null,
          years_remaining: yrs,
          expiry_year: num(c.expiry_year ?? c.expiry),
          rights: c.rights_status || c.rights || "",
          clause: c.clause_active ? c.clause_display || c.clause || "" : "",
          entry_level: Boolean(c.is_entry_level),
          two_way: Boolean(c.two_way),
          level: "NHL",
        };
      })
      .filter((r) => r.cap_hit > 0);
    if (!rows.length) return null;
    const byPosition = POS_GROUPS.map((g) => {
      const list = rows.filter((r) => posGroup(r) === g.key);
      const total = list.reduce((a, r) => a + r.cap_hit, 0);
      const top = [...list].sort((a, b) => b.cap_hit - a.cap_hit)[0];
      return { group: g.key, total, count: list.length, avg: list.length ? total / list.length : 0, top: top ? `${top.name} ${money2(top.cap_hit)}` : "" };
    });
    const firstYear = parseInt(String(seasonLabel(franchiseState) || "").slice(0, 4), 10);
    const commitments = [0, 1, 2, 3, 4].map((k) => {
      const committed = rows.filter((r) => (r.years_remaining || 0) > k).reduce((a, r) => a + r.cap_hit, 0);
      const y = Number.isFinite(firstYear) ? firstYear + 1 + k : null;
      return { season: y ? `${y}-${y + 1}` : `Year ${k + 1}`, committed, cap: capForPct, space: capForPct - committed };
    });
    return { rows, byPosition, commitments };
  }, [serverContracts.length, rosterRows, capForPct, franchiseState]);
  const contracts = serverContracts.length ? serverContracts : fallback ? fallback.rows : [];

  const currentCap = m(report.currentCap) || 0;
  const payroll = m(report.userTeam.payroll) || 0;
  const capSpace = m(report.userTeam.capSpace);
  const floor = m(d.floor ?? capPayload?.user_team?.floor);
  const capSeason = seasonCapLabel(franchiseState, report.season);
  const currentSeason = seasonLabel(franchiseState);
  const teamName = report.userTeam.teamName || "Your club";
  const abbrev = capPayload?.user_team?.abbrev || "";
  const teamLogo = logoFor(teamName, abbrev);
  const tone = statusTone(report.userTeam.capStatus);
  const rank = d.cap_space_rank;
  const teamCount = d.league_team_count || (d.league_table || []).length;
  const change = m(report.capChange);
  const reason = String(report.movementReason || report.notes?.[0] || "").trim();

  // What the cap means for this club -- every line is derived from payload numbers.
  const meaning = [];
  const minSalary = m(d.league_min_salary);
  if (change != null && Math.abs(change) > 0.0001) {
    meaning.push(
      change > 0
        ? `The ceiling rose ${formatCapChange(change)} (${formatCapPercent(report.capChangePercent)})${
            minSalary ? `, the price of ${Math.floor(change / minSalary)} league-minimum deals at ${money2(minSalary)}` : ""
          }.`
        : `The ceiling fell ${formatCapChange(change)} (${formatCapPercent(report.capChangePercent)}); every club loses room.`
    );
  }
  const rosterCount = num(d.roster_count);
  const rosterMax = num(d.roster_max) || 23;
  if (capSpace != null && rosterCount != null) {
    const open = Math.max(0, rosterMax - rosterCount);
    meaning.push(
      open > 0
        ? `${rosterCount} of ${rosterMax} roster spots are filled. ${money(capSpace)} across ${open} open spot${open === 1 ? "" : "s"} is ${money2(capSpace / open)} per signing.`
        : `The roster is full at ${rosterCount}/${rosterMax}; new money means moving a contract out.`
    );
  }
  if (rank && d.league_avg_cap_space != null && capSpace != null) {
    const diff = capSpace - m(d.league_avg_cap_space);
    meaning.push(
      `${ordinal(rank)} most cap space of ${teamCount}, ${money(Math.abs(diff))} ${diff >= 0 ? "above" : "below"} the league average of ${money(d.league_avg_cap_space)}.`
    );
  }
  if (floor != null && payroll) {
    const gap = payroll - floor;
    meaning.push(
      gap >= 0
        ? `Payroll sits ${money(gap)} above the ${money(floor)} floor.`
        : `Payroll is ${money(-gap)} under the ${money(floor)} floor; you must add salary before opening night.`
    );
  }
  const nextCommit = (d.commitments || [])[1];
  if (nextCommit) {
    meaning.push(
      `${money(nextCommit.committed)} is already committed for ${shortSeason(nextCommit.season)} (${nextCommit.players} players) against a projected ${money(nextCommit.cap)} cap.`
    );
  }
  if (d.contract_count != null) {
    meaning.push(`${d.contract_count} of ${d.contract_max || 50} standard player contracts used.`);
  }

  const spaceSub =
    capSpace == null
      ? ""
      : capSpace < 0
        ? "Over the ceiling, must shed salary"
        : capSpace <= 1
          ? "Almost no room to add"
          : capSpace <= 4
            ? "Room for depth only"
            : capSpace >= 10
              ? "Room for an impact signing"
              : "Usable room";

  const tabs = [
    ["overview", "Overview"],
    contracts.length ? ["contracts", `Contracts · ${contracts.length}`] : null,
    d.expiring_now || d.expiring_next ? ["decisions", "Key decisions"] : null,
    asArrayLen(d.league_table) ? ["league", "League cap space"] : null,
  ].filter(Boolean);
  const tab = tabs.some(([k]) => k === tabRaw) ? tabRaw : "overview";

  return (
    <section className="capr2-root">
      <header className="capr2-top">
        <span
          role="button"
          tabIndex={0}
          className="capr2-back"
          onClick={onBack}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onBack?.()}
        >
          ← Back
        </span>
        <div className="capr2-top-mid">
          <div className="capr2-top-name" role="heading" aria-level={1}>Cap report</div>
          <span>Offseason · {capSeason || "next season"} league year</span>
        </div>
        <div className="capr2-top-season">
          <span>Season</span>
          <b>{currentSeason || "Current"}</b>
        </div>
      </header>

      <main className="capr2-main">
        <section className="capr2-hero">
          <div className="capr2-club">
            <TeamLogoBadge teamLogo={teamLogo} teamName={teamName} size={156} variant="badge" />
            <div className="capr2-club-text">
              <div className="capr2-club-name" role="heading" aria-level={2}>{teamName}</div>
              <p className={`capr2-club-status is-${tone}`}>
                {report.userTeam.capStatus || "Cap status"}
                {rank ? ` · ${ordinal(rank)} in cap space` : ""}
              </p>
              {reason ? <p className="capr2-club-reason">{reason}</p> : null}
            </div>
          </div>
          <div className="capr2-figures">
            <Figure label="Cap space" value={money(capSpace)} sub={spaceSub} tone={capSpace != null && capSpace < 0 ? "danger" : "gold"} />
            <Figure
              label={`${capSeason || "League"} salary cap`}
              value={money(currentCap)}
              sub={`${formatCapChange(change)} (${formatCapPercent(report.capChangePercent)}) from ${money(report.previousCap)}`}
            />
            <Figure label="Payroll" value={money(payroll)} sub={currentCap > 0 ? `${pct((payroll / currentCap) * 100, 0)} of the cap used` : ""} />
            <Figure label="Cap floor" value={floor != null ? money(floor) : "—"} sub={floor != null && payroll ? `${formatCapChange(payroll - floor)} vs your payroll` : ""} />
          </div>
          <CapMeter cap={currentCap} floor={floor} payroll={payroll} byPosition={d.by_position} />
        </section>

        <nav className="capr2-tabs" role="tablist" aria-label="Cap report sections">
          {tabs.map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              className={`capr2-tab${tab === key ? " is-on" : ""}`}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className={`capr2-panel is-${tab}`}>
          {tab === "overview" ? (
            <div className={`capr2-ov${contracts.length ? "" : " is-thin"}`}>
              <div className="capr2-ov-col">
                {meaning.length ? (
                  <section className="capr2-means">
                    <div className="capr2-means-head" role="heading" aria-level={2}>What the new cap means for you</div>
                    <ul>
                      {meaning.map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                  </section>
                ) : null}
                {!serverContracts.length && !contracts.length ? (
                  <p className="capr2-plain">
                    The full contract breakdown is not in this save's cap report yet. It is built when the salary-cap stage runs on the
                    updated server.
                  </p>
                ) : null}
                <PositionSplit byPosition={d.by_position || fallback?.byPosition} cap={currentCap} note={d.by_position ? "" : "Every contract on your current roster, today. The payroll above was taken when the cap stage ran."} />
              </div>
              {d.commitments || fallback ? (
                <div className="capr2-ov-col">
                  <Commitments commitments={d.commitments || fallback?.commitments} note={d.commitments ? "" : "Current roster contracts by season, against this year's cap."} />
                </div>
              ) : null}
              <div className="capr2-ov-col">
                <CapHistory history={d.cap_history} floor={floor} />
                <Obligations
                  obligations={d.obligations}
                  deadCap={report.userTeam.deadCap}
                  retained={report.userTeam.retainedSalary}
                  bonus={report.userTeam.bonusOverages}
                />
                {!d.cap_history ? <TopHits contracts={contracts} cap={currentCap} /> : null}
              </div>
            </div>
          ) : null}
          {tab === "contracts" ? <ContractTable contracts={contracts} cap={currentCap} /> : null}
          {tab === "decisions" ? <Decisions expiringNow={d.expiring_now} expiringNext={d.expiring_next} capSpace={capSpace} /> : null}
          {tab === "league" ? <LeagueTable rows={d.league_table} cap={currentCap} rank={rank} teamCount={teamCount} /> : null}
        </div>
      </main>

      <footer className="capr2-foot">
        <div className="capr2-foot-next">
          <span>Next stage</span>
          <b>Player development</b>
        </div>
        <button type="button" className="capr2-go" onClick={onContinue}>
          View Development <span aria-hidden="true">→</span>
        </button>
      </footer>
    </section>
  );
}
