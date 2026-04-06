from __future__ import annotations

"""
NHL DYNASTY MODE — FULL UNIVERSE ENGINE
=======================================

This engine simulates the NHL as a living macro ecosystem.

It uses ONLY systems already implemented:
- League
- Team
- Coach
- Prospect
- Draft + Scouting
- League context system
- Team state system

No external systems.
No imaginary salary DB.
No missing modules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import math

from app.sim_engine.entities.league import League
from app.sim_engine.entities.team import Team
from app.sim_engine.entities.coach import Coach
from app.sim_engine.entities.prospect import Prospect

# ============================================================
# Economy Systems (Waivers)
# ============================================================
from app.sim_engine.economy.waiver_ai import (
    WaiverEngine,
    WaiverConfig,
    update_priority_after_claim,
)


# ============================================================
# Helpers
# ============================================================

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# ============================================================
# Macro Event Models
# ============================================================

@dataclass
class LeagueHeadline:
    year: int
    title: str
    severity: float


@dataclass
class DynastyRecord:
    team_id: str
    cups: int
    active: bool


# ============================================================
# Universe Engine
# ============================================================

class UniverseEngine:

    def __init__(self, seed: int | None = None, debug: bool = False):

        self.seed = seed if seed else random.randrange(1, 10**18)
        self.rng = random.Random(self.seed)
        self.year = 0
        self.debug = debug

        self.league = League(seed=self.seed)

        self.headlines: List[LeagueHeadline] = []
        self.cup_history: Dict[int, str] = {}

        self.dynasties: Dict[str, DynastyRecord] = {}

        self.rivalries: Dict[str, float] = {}
        self.relocation_risk: Dict[str, float] = {}
        self.gm_hot_seat: Dict[str, float] = {}

        # ---------------------------------------
        # Waiver System
        # ---------------------------------------
        self.waiver_engine = WaiverEngine(
            config=WaiverConfig(early_season_cutoff_day=30)
        )
        self.waiver_priority: List[str] = []

    # ========================================================
    # Main Season
    # ========================================================

    def simulate_season(self):

        self.year += 1

        # Advance league macro system
        result = self.league.advance_season(
            team_snapshots=[t.snapshot() for t in self.league.teams],
            team_count=len(self.league.teams),
        )

        league_context = result.get("league_context", {})
        economics = league_context.get("economics", {})
        parity = league_context.get("parity", {})
        era = league_context.get("era", {})

        salary_cap = economics.get("salary_cap", 88_000_000)
        parity_index = parity.get("parity_index", 0.5)
        active_era = (era.get("state") or {}).get("active_era", "modern")

        standings = self._simulate_regular_season(parity_index)

        # ---------------------------------------
        # Build waiver priority (worst → best)
        # NHL-like: worst teams get first dibs.
        # NOTE: This is macro, using current season standings.
        # ---------------------------------------
        standings_payload = []
        for team_id, win_pct, pts in standings:
            team = self.league.get_team(team_id)
            standings_payload.append({
                "team_id": team_id,
                "points": int(pts),
                "point_pct": float(win_pct),
                "goal_diff": int(getattr(team, "goal_diff", 0)),
            })

        standings_payload.sort(key=lambda x: x["points"])  # worst first
        self.waiver_priority = [t["team_id"] for t in standings_payload]

        cup_winner = self._simulate_playoffs(standings)
        self.cup_history[self.year] = cup_winner

        self._detect_dynasty(cup_winner)

        self._simulate_contention_cycles(standings)
        self._simulate_tank_cycles(standings)
        self._simulate_relocation_risk(standings)
        self._simulate_gm_pressure(standings)
        self._simulate_coaching_carousel(standings)
        self._simulate_trade_deadline(standings)

        # ---------------------------------------
        # Waiver wire (after deadline for drama)
        # ---------------------------------------
        self._simulate_waiver_wire(standings)

        self._simulate_offer_sheet_climate(parity_index)
        self._simulate_fan_unrest(standings)
        self._simulate_conference_imbalance(standings)
        self._simulate_rivalries(standings)
        self._simulate_attendance(economics)
        self._simulate_expansion_pressure(economics)
        self._simulate_rule_shifts(active_era)

        self.headlines.append(
            LeagueHeadline(self.year, f"{cup_winner} win the Stanley Cup", 0.9)
        )

        if self.debug:
            self._debug_dump()

    # ========================================================
    # Regular Season
    # ========================================================

    def _simulate_regular_season(self, parity_index: float):

        standings = []

        for team in self.league.teams:

            expected = float(team._expected_win_pct())

            chaos = self.rng.uniform(-0.08, 0.08) * (1.0 - parity_index)
            win_pct = clamp(expected + chaos, 0.25, 0.75)

            team.update_team_state(win_pct=win_pct)

            points = int(win_pct * 164)

            # Optional: if your Team tracks points/point_pct, store for other systems
            try:
                team.points = points
                team.point_pct = win_pct
            except Exception:
                pass

            standings.append((team.id, win_pct, points))

        standings.sort(key=lambda x: x[2], reverse=True)

        return standings

    # ========================================================
    # Playoffs
    # ========================================================

    def _simulate_playoffs(self, standings):

        playoff_teams = standings[:16]

        weighted = []

        for team_id, win_pct, pts in playoff_teams:
            team = self.league.get_team(team_id)

            strength = (
                win_pct * 0.5
                + team.state.stability * 0.2
                + team.state.competitive_score * 0.3
            )

            strength *= self.rng.uniform(0.85, 1.15)

            weighted.append((team_id, strength))

        weighted.sort(key=lambda x: x[1], reverse=True)

        return weighted[0][0]

    # ========================================================
    # Waiver Wire Simulation
    # ========================================================

    def _simulate_waiver_wire(self, standings):
        """
        Macro-level waiver simulation:
        - Generates waiver candidates from struggling teams (bottom chunk)
        - Runs the waiver claim order (worst teams claim first)
        - Updates rolling priority after successful claims
        - Emits headlines for storytelling

        IMPORTANT:
        - This does NOT move real Player entities or modify roster/cap tables.
        - It’s league “transaction noise” + narrative, consistent with the UniverseEngine scope.
        """
        if not self.waiver_priority:
            return

        # Create a few waiver candidates from the worst teams
        waiver_candidates = []

        bottom = standings[-6:] if len(standings) >= 6 else standings
        for team_id, win_pct, pts in bottom:
            # Only if they’re legitimately bad
            if win_pct >= 0.40:
                continue

            # One candidate per team, sometimes two if they’re a dumpster fire
            count = 1 + (1 if win_pct < 0.33 and self.rng.random() < 0.5 else 0)

            for _ in range(count):
                waiver_candidates.append({
                    "from_team_id": team_id,
                    "position": self.rng.choice(["W", "C", "D", "G"]),
                    "age": self.rng.randint(20, 34),
                    "cap_hit": float(self.rng.randint(800_000, 3_250_000)),
                    "contract_years_left": self.rng.randint(1, 4),
                    "overall_projection": clamp(self.rng.uniform(0.33, 0.68)),
                })

        if not waiver_candidates:
            return

        # Build team dicts for WaiverEngine evaluation
        team_dicts = []
        for t in self.league.teams:
            # You can later wire real roster needs/cap space if Team exposes it.
            team_dicts.append({
                "team_id": str(t.id),
                "points": int(getattr(t, "points", 0)),
                "point_pct": float(getattr(t, "point_pct", 0.5)),
                "goal_diff": int(getattr(t, "goal_diff", 0)),
                "cap_space": float(getattr(t, "cap_space", 5_000_000)),
                "competitive_window": str(getattr(t, "status", "bubble")),
                "roster_needs": list(getattr(t, "roster_needs", [])) if getattr(t, "roster_needs", None) else [],
                # Optional deterministic bias knob:
                "waiver_bias": float(getattr(t, "waiver_bias", 0.0)) if hasattr(t, "waiver_bias") else 0.0,
            })

        # Process each waiver player
        for p in waiver_candidates:
            player_payload = {
                "position": p["position"],
                "age": p["age"],
                "cap_hit": p["cap_hit"],
                "contract_years_left": p["contract_years_left"],
                "overall_projection": p["overall_projection"],
            }

            winner = self.waiver_engine.process_player(
                player=player_payload,
                teams=team_dicts,
                priority_order=self.waiver_priority,
            )

            if winner:
                # Rolling priority: winner goes to the back
                self.waiver_priority = update_priority_after_claim(self.waiver_priority, winner)

                # Headline (keep it light severity)
                from_tid = p["from_team_id"]
                self.headlines.append(
                    LeagueHeadline(
                        self.year,
                        f"{winner} claim a {p['position']} off waivers from {from_tid}",
                        0.35,
                    )
                )
            else:
                # Optional: only add headlines for “interesting” clears
                if p["overall_projection"] >= 0.62:
                    self.headlines.append(
                        LeagueHeadline(
                            self.year,
                            f"A notable waiver player clears unclaimed league-wide",
                            0.30,
                        )
                    )

    # ========================================================
    # Dynasty Detection
    # ========================================================

    def _detect_dynasty(self, winner):

        record = self.dynasties.get(winner)

        if not record:
            self.dynasties[winner] = DynastyRecord(winner, 1, True)
        else:
            record.cups += 1
            if record.cups >= 3:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{winner} are building a dynasty", 0.95)
                )

    # ========================================================
    # Contention Windows
    # ========================================================

    def _simulate_contention_cycles(self, standings):

        for team_id, win_pct, _ in standings[:8]:
            team = self.league.get_team(team_id)

            if team.state.competitive_score > 0.65:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} entering strong contention window", 0.6)
                )

    # ========================================================
    # Tank Cycles
    # ========================================================

    def _simulate_tank_cycles(self, standings):

        for team_id, win_pct, _ in standings[-5:]:

            if win_pct < 0.38:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} entering rebuild phase", 0.5)
                )

    # ========================================================
    # Relocation Risk
    # ========================================================

    def _simulate_relocation_risk(self, standings):

        for team_id, win_pct, _ in standings:

            team = self.league.get_team(team_id)

            risk = (
                (1 - team.state.stability) * 0.4
                + (1 - team.state.financial_health) * 0.4
            )

            self.relocation_risk[team_id] = clamp(risk)

            if risk > 0.8:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} facing relocation rumors", 0.8)
                )

    # ========================================================
    # GM Pressure
    # ========================================================

    def _simulate_gm_pressure(self, standings):

        for team_id, win_pct, _ in standings:

            pressure = (1 - win_pct) * 0.6

            self.gm_hot_seat[team_id] = clamp(pressure)

            if pressure > 0.75:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} GM on the hot seat", 0.7)
                )

    # ========================================================
    # Coaching Carousel
    # ========================================================

    def _simulate_coaching_carousel(self, standings):

        for team_id, win_pct, _ in standings:

            team = self.league.get_team(team_id)

            if win_pct < 0.40 and self.rng.random() < 0.4:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} fire head coach", 0.6)
                )

    # ========================================================
    # Trade Deadline
    # ========================================================

    def _simulate_trade_deadline(self, standings):

        intensity = self.rng.uniform(0.2, 0.9)

        if intensity > 0.7:
            self.headlines.append(
                LeagueHeadline(self.year, "Explosive trade deadline shakes league", 0.7)
            )

    # ========================================================
    # Offer Sheet Climate
    # ========================================================

    def _simulate_offer_sheet_climate(self, parity_index):

        if parity_index < 0.4:
            self.headlines.append(
                LeagueHeadline(self.year, "Aggressive offer sheet season emerges", 0.6)
            )

    # ========================================================
    # Fan Unrest
    # ========================================================

    def _simulate_fan_unrest(self, standings):

        for team_id, win_pct, _ in standings[-4:]:

            if win_pct < 0.35:
                self.headlines.append(
                    LeagueHeadline(self.year, f"{team_id} fans demanding changes", 0.5)
                )

    # ========================================================
    # Conference Imbalance
    # ========================================================

    def _simulate_conference_imbalance(self, standings):

        top_half = standings[:len(standings)//2]
        bottom_half = standings[len(standings)//2:]

        if not top_half or not bottom_half:
            return

        avg_top = sum(x[1] for x in top_half) / len(top_half)
        avg_bottom = sum(x[1] for x in bottom_half) / len(bottom_half)

        if abs(avg_top - avg_bottom) > 0.1:
            self.headlines.append(
                LeagueHeadline(self.year, "Conference imbalance emerging", 0.5)
            )

    # ========================================================
    # Rivalries
    # ========================================================

    def _simulate_rivalries(self, standings):

        if not standings:
            return

        if self.rng.random() < 0.4 and len(standings) >= 2:
            t1 = standings[0][0]
            t2 = standings[1][0]

            key = f"{t1}-{t2}"
            self.rivalries[key] = self.rivalries.get(key, 0) + 0.2

            self.headlines.append(
                LeagueHeadline(self.year, f"Rivalry intensifying between {t1} and {t2}", 0.6)
            )

    # ========================================================
    # Attendance / Revenue
    # ========================================================

    def _simulate_attendance(self, economics):

        # NOTE: economics might not include health_score; keep a safe fallback.
        health = economics.get("health_score", economics.get("league_health", 0.6))

        if health < 0.4:
            self.headlines.append(
                LeagueHeadline(self.year, "League revenue downturn concerns owners", 0.6)
            )

    # ========================================================
    # Expansion Pressure
    # ========================================================

    def _simulate_expansion_pressure(self, economics):

        if economics.get("salary_cap", 88_000_000) > 100_000_000:
            self.headlines.append(
                LeagueHeadline(self.year, "Expansion talks heating up", 0.5)
            )

    # ========================================================
    # Rule / Era Shifts
    # ========================================================

    def _simulate_rule_shifts(self, era):

        if era == "dead_puck":
            self.headlines.append(
                LeagueHeadline(self.year, "League considering scoring rule changes", 0.5)
            )

    # ========================================================
    # Debug
    # ========================================================

    def _debug_dump(self):

        print("\n==============================")
        print(f"   UNIVERSE YEAR {self.year}")
        print("==============================")

        print("\nStanley Cup Winner:")
        print(self.cup_history[self.year])

        print("\nTop Headlines:")
        for h in self.headlines[-12:]:
            if h.year == self.year:
                print("-", h.title)
