/**
 * Team identity UI — deliberately quiet.
 *
 *   <TeamIdentityTag teamId abbr name />   small header line under a team name
 *   <TeamIdentityCard identity />          detail card (hover/focus on the tag)
 *   useTeamIdentities() / useTeamIdentity({ teamId, abbr, name })
 *
 * All data comes from the backend team identity engine; nothing is computed here.
 */
import React, { useEffect, useState } from "react";
import "./teamIdentity.css";
import {
  findTeamIdentity,
  loadTeamIdentities,
  peekTeamIdentities,
  subscribeTeamIdentities,
} from "./teamIdentityStore";

export function useTeamIdentities() {
  const [byKey, setByKey] = useState(() => peekTeamIdentities());
  useEffect(() => {
    let alive = true;
    const off = subscribeTeamIdentities((m) => alive && setByKey(m));
    loadTeamIdentities()
      .then((m) => alive && setByKey(m))
      .catch(() => {});
    return () => {
      alive = false;
      off();
    };
  }, []);
  return byKey;
}

export function useTeamIdentity(ref = {}) {
  const byKey = useTeamIdentities();
  return findTeamIdentity(byKey, ref);
}

const PROFILE_ROWS = [
  ["pace", "Pace"],
  ["shooting", "Shooting"],
  ["playmaking", "Playmaking"],
  ["puck", "Puck skill"],
  ["physical", "Physical"],
  ["defense", "Defence"],
  ["goaltending", "Goaltending"],
];

function styleKey(identity) {
  return String(identity?.primary?.key || "balanced").replace(/[^a-z_]/g, "");
}

export function TeamIdentityCard({ identity }) {
  if (!identity) return null;
  const pct = identity.profile_pct || {};
  return (
    <div className={`tid-card tid-style-${styleKey(identity)}`} role="tooltip">
      <div className="tid-card-head">
        <strong>{identity.primary?.label || "Balanced"}</strong>
        {identity.secondary?.label ? <span>+ {identity.secondary.label}</span> : null}
      </div>
      {identity.primary?.blurb ? <p className="tid-card-blurb">{identity.primary.blurb}</p> : null}
      <div className="tid-bars">
        {PROFILE_ROWS.map(([key, label]) => {
          const v = Math.max(0, Math.min(100, Number(pct[key] ?? 50)));
          return (
            <div className="tid-bar-row" key={key}>
              <span>{label}</span>
              <i>
                <b style={{ width: `${v}%` }} />
              </i>
              <em>{v}</em>
            </div>
          );
        })}
      </div>
      {identity.star?.name ? (
        <p className="tid-card-line">
          <span>Built around</span> {identity.star.name}
          {identity.star.style ? ` · ${identity.star.style}` : ""}
        </p>
      ) : null}
      {Array.isArray(identity.target_labels) && identity.target_labels.length ? (
        <p className="tid-card-line">
          <span>Hunting</span> {identity.target_labels.join(", ")}
        </p>
      ) : null}
      {Array.isArray(identity.traits) && identity.traits.length ? (
        <p className="tid-card-line">
          <span>Traits</span> {identity.traits.join(" · ")}
        </p>
      ) : null}
      <p className="tid-card-foot">League percentile vs. the other 31 clubs</p>
    </div>
  );
}

export function TeamIdentityTag({ teamId, abbr, name, compact = false, className = "" }) {
  const identity = useTeamIdentity({ teamId, abbr, name });
  const [open, setOpen] = useState(false);
  if (!identity) return null;
  const text = compact ? identity.primary?.label || "Balanced" : identity.label || identity.primary?.label;
  return (
    <span
      className={`tid-tag tid-style-${styleKey(identity)} ${compact ? "is-compact" : ""} ${className}`}
      tabIndex={compact ? undefined : 0}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onClick={(e) => e.stopPropagation()}
      aria-label={`Team identity: ${identity.label}`}
    >
      <i aria-hidden="true" className="tid-dot" />
      {text}
      {open ? <TeamIdentityCard identity={identity} /> : null}
    </span>
  );
}

export default TeamIdentityTag;
