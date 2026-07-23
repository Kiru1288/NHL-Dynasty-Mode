"""Season-end aging and progression."""

from __future__ import annotations

from app.sim_engine.franchise._shared import *  # noqa: F401,F403

class _FranchiseLifecycleLogger:
    """Satisfies run_sim career pass: no console spam, optional capture later."""

    log_level = "normal"

    def emit(self, *_args: Any, **_kwargs: Any) -> None:
        return
def _strip_retired_from_nhl_rosters(teams: List[Any]) -> int:
    removed = 0
    for team in teams:
        roster = list(getattr(team, "roster", None) or [])
        kept = [p for p in roster if not getattr(p, "retired", False)]
        removed += len(roster) - len(kept)
        team.roster = kept
    return removed
def _franchise_nhl_age_and_phase_tick(session: FranchiseSession, teams: List[Any]) -> None:
    """One calendar year: Player.advance_year + career phase (mirrors universe roster pass)."""
    from app.sim_engine import engine as eng_mod
    from app.sim_engine.engine import assign_career_phase_from_age

    team_instability = max(0.28, min(0.62, 1.05 - float(session.chaos_index)))
    for team in teams:
        roster = getattr(team, "roster", None) or []
        dev_quality = float(getattr(team, "development_quality", 0.5))
        dev_mod = dev_quality - 0.5
        for player in roster:
            if getattr(player, "retired", False):
                continue
            advance_fn = getattr(player, "advance_year", None)
            if not callable(advance_fn):
                ident = getattr(player, "identity", None)
                if ident is not None and hasattr(ident, "age"):
                    try:
                        ident.age = int(getattr(ident, "age", 0)) + 1
                    except (TypeError, ValueError):
                        pass
            else:
                try:
                    ident = getattr(player, "identity", None)
                    age = int(getattr(ident, "age", getattr(player, "age", 25)) if ident else getattr(player, "age", 25))
                    age_damp = max(0.35, min(1.0, 1.0 - max(0.0, (age - 26)) / 10.0))
                    morale = float(getattr(getattr(player, "psych", None), "morale", 0.5) or 0.5)
                    inj = float(getattr(getattr(player, "health", None), "injury_risk_baseline", 0.1) or 0.1)
                    try:
                        sys_dev = float(eng_mod.team_system_development_modifier(team))
                    except Exception:
                        sys_dev = 0.0
                    advance_fn(
                        season_morale=morale,
                        season_injury_risk=inj,
                        team_instability=team_instability,
                        development_modifier=dev_mod * age_damp + sys_dev,
                    )
                except Exception:
                    pass
            try:
                assign_career_phase_from_age(player)
            except Exception:
                pass
def _run_franchise_season_end_progression(session: FranchiseSession) -> Dict[str, Any]:
    """
    After the regular-season calendar: NHL roster aging + the same progression stack as the
    universe runner (development pass ΓåÆ major career events ΓåÆ soft anti-inflation guard).
    """
    out: Dict[str, Any] = {"aged": True, "lifecycle": None, "retired_removed": 0}
    sim = session.sim
    league = getattr(sim, "league", None)
    teams = list(getattr(league, "teams", None) or [])
    if not teams or league is None:
        return out

    rng = sim.rng
    sy = int(session.season_calendar_year)

    try:
        setattr(
            league,
            "_tuning_context",
            {
                "chaos_index": float(session.chaos_index),
                "parity_index": 0.52,
                "league_health": 0.58,
                "active_era": "modern",
            },
        )
    except Exception:
        pass

    _franchise_nhl_age_and_phase_tick(session, teams)

    if getattr(rs, "_run_player_progression_pass", None):
        try:
            rs._run_player_progression_pass(teams, rng, None)
        except Exception:
            pass

    if getattr(rs, "_run_career_lifecycle_pass", None):
        try:
            out["lifecycle"] = rs._run_career_lifecycle_pass(
                teams,
                rng,
                _FranchiseLifecycleLogger(),
                league=league,
                state=None,
                season_year=sy,
            )
        except Exception:
            out["lifecycle"] = {"skipped": True}

    try:
        from app.sim_engine.engine import apply_league_ovr_soft_regression_if_needed

        apply_league_ovr_soft_regression_if_needed(teams, rng, avg_trigger=74.5)
    except Exception:
        pass

    out["retired_removed"] = int(_strip_retired_from_nhl_rosters(teams))
    # Season calendar year advances only in generate_next_season (authoritative transition).
    out["season_calendar_year_unchanged"] = sy
    return out
