"""
NHL DYNASTY MODE — League_Stats.py
---------------------------------
League-wide statistical reality generator.

RESPONSIBILITY:
- Define the statistical environment of the league
- Generate distributions for skater & goalie metrics
- Provide percentile ladders ("how good is good?")
- Sample plausible season outcomes by role/tier

THIS FILE DOES NOT:
- Reference player objects
- Assign stats to players
- Modify ratings or OVR
- Simulate games

Downstream usage:
- Player_Stats.py samples from these distributions
- Team_Stats.py aggregates player outputs
"""

from __future__ import annotations

import math
import json
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List


# ============================================================
# Helpers
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def percentile(sorted_vals: List[float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def zscore(value: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return (value - mu) / sigma


# ============================================================
# Data Models
# ============================================================

@dataclass
class LeagueEnvironment:
    season: int
    era: str
    teams: int
    games_per_team: int
    salary_cap: int

    gf_per_gp: float
    pace_5v5_shots_per60: float
    finishing_rate: float

    pp_pct: float
    pk_pct: float

    sv_pct: float

    variance_scale: float
    outlier_frequency: float
    superstar_tail: float


@dataclass
class Distribution:
    name: str
    domain: str
    mean: float
    std: float
    min_val: float
    max_val: float

    skew_strength: float = 0.0
    fat_tail_prob: float = 0.0
    fat_tail_boost: float = 0.0
    replacement_compress: float = 0.0

    percentiles: Dict[int, float] = field(default_factory=dict)

    def sample(self, rng: random.Random, tier_mult: float = 1.0, var_mult: float = 1.0) -> float:
        if self.skew_strength <= 0.001:
            raw = rng.gauss(self.mean * tier_mult, self.std * var_mult)
        else:
            mu = math.log(max(1e-6, self.mean * tier_mult))
            sigma = clamp(self.skew_strength * 0.55 * var_mult, 0.05, 1.10)
            raw = rng.lognormvariate(mu, sigma)
            raw = lerp(raw, self.mean * tier_mult, 0.35)

        if self.replacement_compress > 0.0 and raw < self.mean * 0.8:
            raw = lerp(raw, self.mean * 0.6, self.replacement_compress)

        if self.fat_tail_prob > 0.0 and rng.random() < self.fat_tail_prob:
            raw += abs(rng.gauss(0.0, 1.0)) * (self.std * (1.0 + self.fat_tail_boost))

        return clamp(raw, self.min_val, self.max_val)


# ============================================================
# LeagueStats Engine
# ============================================================

class LeagueStats:
    """
    Generates league-wide statistical distributions and percentiles.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        season: int = 2026,
        era: str = "modern_offense",
        teams: int = 32,
        games_per_team: int = 82,
        salary_cap: int = 92_000_000,
    ) -> None:
        self.seed = seed if seed is not None else random.randrange(1, 10**18)
        self.rng = random.Random(self.seed)

        self.season = season
        self.era = era
        self.teams = teams
        self.games_per_team = games_per_team
        self.salary_cap = salary_cap

        self.env: Optional[LeagueEnvironment] = None
        self.distributions: Dict[str, Dict[str, Distribution]] = {"skater": {}, "goalie": {}}
        self.percentiles: Dict[str, Dict[str, Dict[int, float]]] = {"skater": {}, "goalie": {}}

        self.tier_multipliers = {
            "elite": 1.35,
            "top_line": 1.18,
            "middle_six": 1.00,
            "bottom_six": 0.78,
            "depth": 0.62,
        }

    # --------------------------------------------------------

    def generate(self) -> None:
        self.env = self._generate_environment()
        self._generate_distributions()
        self._compute_all_percentiles()

    # --------------------------------------------------------

    def sample_metric(
        self,
        domain: str,
        metric: str,
        tier: str = "middle_six",
        rng: Optional[random.Random] = None,
        var_mult: float = 1.0,
    ) -> float:
        if self.env is None:
            raise RuntimeError("LeagueStats not generated yet.")

        r = rng if rng is not None else self.rng
        tier_mult = self.tier_multipliers.get(tier, 1.0)
        return self.distributions[domain][metric].sample(r, tier_mult, var_mult)

    def value_to_percentile(self, domain: str, metric: str, value: float) -> int:
        ladder = self.percentiles.get(domain, {}).get(metric)
        if not ladder:
            return 50

        best = 0
        for p in range(0, 101):
            if ladder[p] <= value:
                best = p
            else:
                break
        return best

    # ========================================================
    # Internal Generation
    # ========================================================

    def _generate_environment(self) -> LeagueEnvironment:
        ERA = {
            "modern_offense": (3.15, 56.0, 0.22, 0.904, 1.00),
            "goalie_dominance": (2.80, 52.0, 0.19, 0.912, 0.95),
            "dead_puck": (2.65, 50.0, 0.18, 0.910, 0.85),
            "chaos": (3.45, 59.0, 0.24, 0.898, 1.20),
        }

        gf_gp, pace, pp, sv, var = ERA.get(self.era, ERA["modern_offense"])

        return LeagueEnvironment(
            season=self.season,
            era=self.era,
            teams=self.teams,
            games_per_team=self.games_per_team,
            salary_cap=self.salary_cap,
            gf_per_gp=gf_gp,
            pace_5v5_shots_per60=pace,
            finishing_rate=clamp(0.095 + (gf_gp - 3.0) * 0.012, 0.075, 0.125),
            pp_pct=pp,
            pk_pct=1.0 - pp,
            sv_pct=sv,
            variance_scale=var,
            outlier_frequency=var,
            superstar_tail=var,
        )

    def _generate_distributions(self) -> None:
        env = self.env
        var = env.variance_scale

        self.distributions["skater"]["goals"] = Distribution(
            "goals", "skater",
            mean=13.5 * env.gf_per_gp / 3.0,
            std=9.0 * var,
            min_val=0.0, max_val=70.0,
            skew_strength=0.55,
            fat_tail_prob=0.02 * var,
            fat_tail_boost=0.9,
            replacement_compress=0.35,
        )

        self.distributions["skater"]["assists"] = Distribution(
            "assists", "skater",
            mean=20.0 * env.gf_per_gp / 3.0,
            std=12.0 * var,
            min_val=0.0, max_val=85.0,
            skew_strength=0.40,
            fat_tail_prob=0.018 * var,
            fat_tail_boost=0.7,
            replacement_compress=0.30,
        )

        self.distributions["skater"]["points"] = Distribution(
            "points", "skater",
            mean=33.5 * env.gf_per_gp / 3.0,
            std=18.0 * var,
            min_val=0.0, max_val=140.0,
            skew_strength=0.50,
            fat_tail_prob=0.022 * var,
            fat_tail_boost=1.0,
            replacement_compress=0.25,
        )

        self.distributions["skater"]["war"] = Distribution(
            "war", "skater",
            mean=0.8,
            std=1.25 * var,
            min_val=-3.5, max_val=7.0,
            skew_strength=0.15,
            fat_tail_prob=0.02 * var,
            fat_tail_boost=1.0,
            replacement_compress=0.10,
        )

        self.distributions["skater"]["xgf_pct"] = Distribution(
            "xgf_pct", "skater",
            mean=0.500,
            std=0.035 * var,
            min_val=0.40, max_val=0.62,
        )

    def _compute_all_percentiles(self, sample_n: int = 50000) -> None:
        for domain, metrics in self.distributions.items():
            for metric, dist in metrics.items():
                vals = []
                for _ in range(sample_n):
                    vals.append(dist.sample(self.rng))
                vals.sort()
                ladder = {p: percentile(vals, p) for p in range(0, 101)}
                dist.percentiles = ladder
                self.percentiles[domain][metric] = ladder
