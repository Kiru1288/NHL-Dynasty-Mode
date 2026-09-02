"""Start a real-NHL franchise, advance days, dump OVR / character / life / storylines."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NHL_FRANCHISE_DEBUG", "0")

from services.franchise_sim import start_franchise  # noqa: E402
from app.sim_engine.franchise.storyline_conduct import (  # noqa: E402
    get_base_ovr_display,
    get_effective_ovr_display,
)
from app.sim_engine.franchise.storyline_engine import (  # noqa: E402
    _u_mental_ovr,
    _u_personality,
    _u_sync_player_entities,
    _u_tier_label,
    build_human_dossier_payload,
    migrate_session_storyline_state,
)


def _name(p) -> str:
    ident = getattr(p, "identity", None)
    return str(getattr(ident, "name", None) or getattr(p, "name", "") or "?")


def _pos(p) -> str:
    ident = getattr(p, "identity", None)
    pos = getattr(ident, "position", None) if ident else None
    if pos is None:
        pos = getattr(p, "position", None)
    return str(getattr(pos, "value", pos) or "?")


def main() -> None:
    print("Starting real NHL franchise (Vancouver Canucks, seed 42)...", flush=True)
    session = start_franchise(
        team_query="Vancouver Canucks",
        head_coach_name="Probe",
        coach_archetype="balanced",
        seed=42,
        player_universe="real_nhl",
    )
    migrate_session_storyline_state(session)
    _u_sync_player_entities(session)
    team = session.team_by_id[str(session.user_team_id)]
    roster = [p for p in (getattr(team, "roster", None) or []) if not getattr(p, "retired", False)]
    print(f"User team roster: {len(roster)}  universe players: {len(getattr(session, 'universe_players', {}) or {})}", flush=True)

    print("\n=== BEFORE SIM — Vancouver roster (OVR / character / mental / life) ===", flush=True)
    dump_roster(session, roster, limit=18)

    print("\n=== League personality spread (sample of 80 NHL players) ===", flush=True)
    dump_league_spread(session)

    print("\nAdvancing 21 calendar days (light games) then running story/life ticks...", flush=True)
    from services.franchise_sim import advance_franchise_bulk

    bulk = advance_franchise_bulk(session, mode="day", count=21, auto_resolve_decisions=True)
    print(
        f"bulk status={bulk.get('status')} steps_completed={bulk.get('steps_completed')} "
        f"stopped={bulk.get('stopped_reason')} cursor={bulk.get('calendar_index')} iso={bulk.get('iso')}",
        flush=True,
    )
    from app.sim_engine.franchise.storyline_engine import franchise_cause_storyline_daily_pass

    session._universe_last_daily_tick = -1
    cal = list(getattr(session, "nhl_calendar", None) or [])
    end = int(getattr(session, "calendar_cursor", 0) or 0)
    start = max(0, end - int(bulk.get("steps_completed") or 0))
    ticks = 0
    for idx in range(start, end):
        day_meta = cal[idx] if idx < len(cal) else {}
        franchise_cause_storyline_daily_pass(session, idx, day_meta, rng=session.sim.rng)
        ticks += 1
    print(f"storyline/life daily ticks={ticks} games={len(getattr(session, 'game_results', []) or [])}", flush=True)
    _u_sync_player_entities(session)

    print("\n=== AFTER SIM — Vancouver roster ===", flush=True)
    dump_roster(session, roster, limit=18)

    print("\n=== Off-ice / family snapshots ===", flush=True)
    dump_life(session, roster)

    print("\n=== Storylines (user team + personal life) ===", flush=True)
    dump_storylines(session)

    print("\n=== Universe event log (off-ice) ===", flush=True)
    dump_universe_events(session)


def dump_roster(session, roster, limit: int = 18) -> None:
    entities = getattr(session, "universe_players", None) or {}
    rows = []
    for p in roster:
        pid = str(getattr(p, "id", "") or "")
        ent = entities.get(pid) or {}
        pers = ent.get("personality") or _u_personality(p, pid)
        mental = _u_mental_ovr(ent, p)
        traits = getattr(p, "traits", None)
        vol = float(getattr(traits, "volatility", 0.5) or 0.5) if traits is not None else 0.5
        rows.append(
            {
                "name": _name(p),
                "pos": _pos(p),
                "base": get_base_ovr_display(p),
                "eff": get_effective_ovr_display(p),
                "char": round(float(pers.get("character", 50)), 1),
                "char_t": _u_tier_label(float(pers.get("character", 50))),
                "mental": round(float(mental), 1),
                "vol": round(vol * 100),
                "ego": round(float(pers.get("ego", 50))),
                "lead": round(float(pers.get("leadership", 50))),
                "fam": round(float(pers.get("family_orientation", 50))),
                "tags": ", ".join((ent.get("personality_tags") or [])[:2]) or "—",
            }
        )
    rows.sort(key=lambda r: -float(r["base"] or 0))
    print(
        f"{'Player':<24} {'Pos':<4} {'OVR':>4} {'Eff':>4} {'Char':>5} {'C-tier':<14} {'Ment':>5} {'Vol':>3} {'Ego':>3} {'Ld':>3} {'Fam':>3}  Tags"
    )
    for r in rows[:limit]:
        print(
            f"{r['name'][:24]:<24} {r['pos']:<4} {r['base']:>4} {r['eff']:>4} {r['char']:>5} {r['char_t']:<14} "
            f"{r['mental']:>5} {r['vol']:>3} {r['ego']:>3} {r['lead']:>3} {r['fam']:>3}  {r['tags']}"
        )
    chars = [r["char"] for r in rows]
    vols = [r["vol"] for r in rows]
    print(
        f"n={len(rows)} character min/avg/max={min(chars):.0f}/{sum(chars)/len(chars):.0f}/{max(chars):.0f}  "
        f"volatility min/avg/max={min(vols)}/{sum(vols)/len(vols):.0f}/{max(vols)}"
    )


def dump_league_spread(session) -> None:
    entities = getattr(session, "universe_players", None) or {}
    tiers = Counter()
    tags = Counter()
    rels = Counter()
    n = 0
    for ent in list(entities.values())[:400]:
        if not ent.get("active_roster"):
            continue
        n += 1
        pers = ent.get("personality") or {}
        tiers[_u_tier_label(float(pers.get("character", 50)))] += 1
        for t in ent.get("personality_tags") or []:
            tags[t] += 1
        rels[str((ent.get("life") or {}).get("relationship_status") or "unknown")] += 1
        if n >= 80:
            break
    print("character tiers:", dict(tiers))
    print("top tags:", tags.most_common(12))
    print("relationship mix:", dict(rels))


def dump_life(session, roster) -> None:
    entities = getattr(session, "universe_players", None) or {}
    league_hist = 0
    for ent in entities.values():
        league_hist += len((ent.get("life") or {}).get("minor_event_history") or [])
    print(f"league-wide recorded minor life events: {league_hist}")
    shown = 0
    for p in sorted(roster, key=lambda x: -get_base_ovr_display(x))[:12]:
        pid = str(getattr(p, "id", "") or "")
        ent = entities.get(pid) or {}
        dossier = build_human_dossier_payload(session, ent, p, include_private=True)
        life = dossier.get("life") or {}
        state = dossier.get("current_state") or {}
        mw = dossier.get("mental_wellbeing") or {}
        raw_life = ent.get("life") or {}
        hist = list(raw_life.get("minor_event_history") or [])
        print(
            f"- {_name(p)} | {life.get('summary')} | home={life.get('home_stability_tier')} "
            f"city={life.get('city_attachment_tier')} reloc={life.get('relocation_tier')} | "
            f"morale={state.get('morale_tier')} conf={state.get('confidence_tier')} "
            f"mental={mw.get('tier')} ({mw.get('state')}) | off-ice events={len(hist)}"
        )
        if hist:
            print(f"    last events: {[h.get('event_type') for h in hist[-3:]]}")
        shown += 1


def dump_storylines(session) -> None:
    uid = str(session.user_team_id)
    events = list(getattr(session, "storyline_events", None) or [])
    user = [s for s in events if str(s.get("team_id") or s.get("team") or "") in ("", uid) or True]
    personal = [
        s
        for s in events
        if str(s.get("category") or s.get("cause_type") or "").lower()
        in ("personal_life", "positive_life_event", "minor_life_event")
        or "LIFE" in str(s.get("cause_type") or "").upper()
        or str(s.get("category") or "") == "personal_life"
    ]
    print(f"total storyline_events={len(events)} personal/life={len(personal)}")
    for s in events[-18:]:
        print(
            f"  [{s.get('cause_type') or s.get('category') or '?'}] "
            f"{s.get('headline') or s.get('title') or '(no headline)'}"
        )
        sm = str(s.get("summary") or s.get("description") or "").strip()
        if sm:
            print(f"      {sm[:160]}")


def dump_universe_events(session) -> None:
    log = list(getattr(session, "universe_event_log", None) or [])
    print(f"universe_event_log={len(log)}")
    for row in log[-16:]:
        print(f"  {row.get('kind') or row.get('id')}: {row.get('headline') or row.get('summary') or row}")


if __name__ == "__main__":
    main()
