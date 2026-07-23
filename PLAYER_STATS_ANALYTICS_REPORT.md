# Player Stats & Analytics — Full Systems Report

**NHL Franchise Mode** · Code audit · July 2026  
**Scope:** How NHL stats are simulated, allocated, aggregated, analyzed, and how they connect to player growth, talent tiers, lines, and chemistry.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: Team Score Simulation](#2-phase-1-team-score-simulation)
3. [Phase 2: Player Stat Allocation](#3-phase-2-player-stat-allocation)
4. [Who Gets Stats vs Who Does Not](#4-who-gets-stats-vs-who-does-not)
5. [Goals, Assists & Points Distribution](#5-goals-assists--points-distribution)
6. [Lines, Usage & Chemistry Effects](#6-lines-usage--chemistry-effects)
7. [Advanced Analytics Deep Dive](#7-advanced-analytics-deep-dive)
8. [PDO & Shooting % Sustainability](#8-pdo--shooting--sustainability)
9. [CF%, xGF%, Possession & Finishing](#9-cf-xgf-possession--finishing)
10. [Goalie Stats & GSAx](#10-goalie-stats--gsax)
11. [Cumulative vs Averaged Season Stats](#11-cumulative-vs-averaged-season-stats)
12. [Superstars & Generational Talents](#12-superstars--generational-talents)
13. [Skilled Forward vs Defensive Defenseman](#13-skilled-forward-vs-defensive-defenseman)
14. [Stats → Player Progression & Growth](#14-stats--player-progression--growth)
15. [Prospect League Stat Simulation](#15-prospect-league-stat-simulation)
16. [World Systems: Morale, Fatigue, Momentum](#16-world-systems-morale-fatigue-momentum)
17. [Stats Central Pipeline](#17-stats-central-pipeline)
18. [Known Limitations & Interpretation Guide](#18-known-limitations--interpretation-guide)
19. [Key File Reference](#19-key-file-reference)

---

## 1. Architecture Overview

The sim uses a **two-phase, outcome-first** model:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: _simulate_game()                                      │
│  Inputs: team strength, offense skill, chemistry, fatigue, etc. │
│  Output: (home_goals, away_goals, overtime)                   │
│  Does NOT assign individual player stats                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: accumulate_unified_game_stats()                       │
│  Allocates SOG, G, A, TOI, hits, blocks, CF, xGF, goalie stats  │
│  to match the Phase 1 score exactly                             │
│  Mutates session.player_season_stats ledger                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 (display / analysis): player_analytics.py + UI         │
│  Derives PDO, CF%, finishing, WAR, GSAx from ledger totals        │
│  Stats Central normalizes and ranks on the frontend             │
└─────────────────────────────────────────────────────────────────┘
```

**Critical design principle:** The final score is sacred. Individual stats are **retroactively allocated** to reconcile with it. xG and Corsi do not drive goals — goals drive xG scaling.

**Source of truth:** `session.player_season_stats` — a per-player cumulative ledger (`player_id → row`). Every counting stat in the NHL UI traces back to game-by-game ledger mutations.

**Primary files:**
| Layer | File |
|-------|------|
| Score sim + stat allocation | `SimEngine/app/sim_engine/engine.py` |
| Possession / xG allocation | `SimEngine/app/sim_engine/gameplay/game_analytics_ledger.py` |
| Derived analytics | `SimEngine/app/sim_engine/generation/player_analytics.py` |
| Franchise wrapper | `backend/services/franchise_sim.py` |
| Progression | `SimEngine/app/sim_engine/progression/` |
| Stats Central UI | `frontend/src/screens/StatsCentralScreen.js` |

---

## 2. Phase 1: Team Score Simulation

**Function:** `SimEngine._simulate_game()` — `engine.py` ~11461

### Inputs

| Factor | Effect on scoring |
|--------|-------------------|
| `strength_map` × `home/away_strength_scale` | Base team strength differential (`diff`) |
| `_team_offense_skill` | Shot volume and shooting percentage |
| `_team_superstar_offense_impact` | Nonlinear star-driven scoring environment |
| `_chemistry_game_modifier` | ±6% on expected goals (`mu`) |
| `_team_goalie_suppression` / `_team_defense_suppression` | Reduces opponent `mu` |
| `_roster_injury_depth_penalty` | Applied as strength scale when roster is thin |
| `noise_scale` | Gaussian variance multiplier on final goal draw |
| World modules (momentum, morale, fatigue) | Combined into per-team strength scale in franchise mode |

### Core formulas

**Expected shots on goal per team:**
```
SOG = 30.5 + 4.2×(off_skill − 0.5) ± 3.0×diff + 4.8×star_impact + noise
clamped to [27.0, 41.5]
```

**Team shooting percentage (abstract, not yet per-player):**
```
SH% = clamp(0.092 + 0.010×(off_skill − 0.5) + 0.022×star_impact, 0.085, 0.128)
```

**Power play danger bonus:**
```
PP_share = clamp(0.225 + 0.06×pp_danger, 0.19, 0.28)
```

**Expected goals (mu) before noise:**
```
mu += SOG × SH% × (1 + 0.28×PP_share) × chemistry_modifier
mu += 0.62 × star_impact
mu += 0.34 + 0.58 × diff
mu −= opponent_goalie_suppression + opponent_defense_suppression
```

**Final goal draw:**
```
goals = max(0, round(Gaussian(mu, 1.06 × noise_scale × narrative_sigma)))
```

**Overtime:** If tied, 52% home / 48% away coin flip for one extra goal.

**Tuning targets (documented in code):**
- Combined GPG: ~6.4–7.1
- Per-team GPG: ~3.2–3.55
- SOG/team: 27–38
- League SH%: ~9.2–9.7%

---

## 3. Phase 2: Player Stat Allocation

**Function:** `SimEngine.accumulate_unified_game_stats()` — `engine.py` ~10768  
**Franchise entry:** `_accumulate_franchise_game_stats()` — `franchise_sim.py` ~2770

### Step-by-step per game

#### 3.1 Dress the roster

All **healthy, non-goalie** roster players dress. There is no 18-man dressed / bench split.

```python
home_sk = [p for p in self._gm_skaters(home) if not self._injury_sidelined(p)]
```

Every dressed skater receives `gp = 1` for that game.

#### 3.2 Team shot target

```
team_SOG_target = 30.5 + 4.2×(off_skill − 0.5) + 2.05×actual_goals + noise
clamped [27, 38]
adjusted down by opponent PK suppression factor
```

#### 3.3 Per-player TOI assignment

Lines built from OVR sort (`_gm_forward_lines`): 4 forward lines + separate D pool.

| Role | Line | TOI (minutes, random) |
|------|------|------------------------|
| Defense | — | 17–24 |
| Forward | Line 1 | 18–22 |
| Forward | Line 2 | 15–18 |
| Forward | Line 3 | 12–15 |
| Forward | Line 4 | 8–12 |

TOI scaled by `_gm_role_usage_mult` (line rank, depth role, age gate).

#### 3.4 Shot distribution

Each player's shot weight:
```
weight = offense_weight × PP_line_bias × (TOI_seconds / 3600)^0.48
shots = _gm_distribute_integer_shares(weights, team_SOG_target)
```

**PP line bias:** L1 forwards 1.24×, L2 1.12×, L3 1.05×, L4 1.0×; defense 1.04–1.14×.

#### 3.5 Peripheral counting stats (random, per dressed player)

| Stat | Distribution |
|------|-------------|
| PIM | 0 (56%), 2 (28%), 4 (12%), 6 (4%) |
| Hits (F) | 0–3 weighted |
| Hits (D) | 0–4 weighted |
| Blocks (F) | 0–2 weighted |
| Blocks (D) | 0–4 weighted |

#### 3.6 Goal & assist allocation

For each actual team goal scored in Phase 1:

1. **Strength roll:** PP (~21.5–31%), SH (~6.2%), else even strength
2. **Scorer:** `_gm_pick_goal_scorer` (weighted by line + offense)
3. **Assists:** `_gm_pick_assists` — always ≥1 if 2+ skaters; 58% chance of 2nd assist
4. PP goals: 14% chance to trim to 1 assist only

#### 3.7 Possession analytics (full mode only)

`allocate_team_possession_analytics()` in `game_analytics_ledger.py` assigns CF, CA, FF, FA, xGF, xGA, iXG, on-ice GF/GA per player.

Skipped in `light_mode` (bulk franchise sim).

#### 3.8 Goalie stats

One goalie per team, selected weighted by OVR.

```
expected_sv% = clamp(0.898 + 0.030 × (goalie_skill − 0.5), 0.888, 0.926)
shots_against = blend(skill_based_SA, volume_based_SA, opponent_SOG)
saves = SA − goals_against
record: W / L / OTL / SO
```

---

## 4. Who Gets Stats vs Who Does Not

| Player state | Gets GP & counting stats? | Notes |
|--------------|--------------------------|-------|
| Healthy skater on active roster | **Yes** | Full TOI, SOG, peripheral stats |
| Injured / sidelined | **No** | Excluded from `_gm_skaters` pool |
| Goalie | **One per team** | Weighted starter selection |
| Scratch list (`team.scratches`) | **Not used** | Scratches not enforced in stat pipeline |
| Tank-scratched (`_tank_scratched_ids`) | **Partial** | Removed from goal-scoring line pools only; still get counting stats |
| Retired | **No** | Filtered from active roster |

**There is no "healthy scratch" mechanic.** If a player is on the roster and healthy, they accumulate stats every game they are available. This is why depth players rack up GP at the same rate as stars — the sim models a full active roster, not a 12F/6D game-night dress.

**Players who don't dress (injured):** They receive zero ledger updates. Their season line is unchanged for that game. Morale may still shift from team results via `update_after_team_result` for roster players, but stat accumulation is a hard zero.

---

## 5. Goals, Assists & Points Distribution

### The offense weight formula (central allocator)

**Function:** `_gm_offense_weight()` — `engine.py` ~10482

```
base = (0.40×off_ratings + 0.34×passing + 0.26×IQ) / 99
weight = max(0.04, OVR^1.14 × base^1.08 × usage_mult × pos_mult × star_mult)
```

| Modifier | Value |
|----------|-------|
| Defensemen `pos_mult` | 0.86 |
| Age ≤19 `usage_mult` | ×0.78 |
| Age 20 | ×0.88 |
| Age 21 | ×0.95 |
| `star_mult` | From scoring tier / line rank (`_gm_scoring_involvement_mult`) |

This weight drives: shot share, goal probability, assist probability, xG rate, CF share.

### Goal scorer selection

**Function:** `_gm_pick_goal_scorer()` — `engine.py` ~10690

**Line-level goal share:** `[37%, 28%, 20%, 9%]` for lines 1–4.

**Defense goal share:** 12.5% EV, 6.8% PP, 18.5% SH.

**Within-line selection:** Weighted by `_gm_offense_weight`, with **repeat-goal damping**:
```
weight /= (1 + 0.45 × goals_already_scored_this_game)
```
Prevents one player from hoarding all goals in a single game.

### Assist selection

**Function:** `_gm_pick_assists()` — `engine.py` ~10729

- Pool: all skaters except scorer
- 1st assist weight: `offense_weight^1.08`
- 2nd assist: 58% chance, weight: `offense_weight`
- Maximum 2 assists per goal

### Integer share distribution

**Function:** `_gm_distribute_integer_shares()` — `engine.py` ~10566

Proportional allocation with largest-remainder rounding. Used for SOG, Corsi events, on-ice goals against, xGA units. Ensures team totals always sum exactly to the target integer.

### Points

```
points = goals + assists  (accumulated in ledger per game)
```

There is no separate "points allocation" — points emerge from G+A assignment.

---

## 6. Lines, Usage & Chemistry Effects

### In-game line building (per game)

**Function:** `_gm_forward_lines()` — `engine.py` ~10507

1. Sort forwards by OVR (with tank-pressure age tweaks)
2. Split into 4 even chunks → lines 1–4
3. Defensemen in separate pool (not mixed into forward lines)
4. Set per-game attributes: `_gm_game_line_idx`, `_gm_game_line_rank`, `_gm_game_fwd_rank`

### Role usage multiplier

**Function:** `_gm_role_usage_mult()` — `engine.py` ~10450

```
line_usage_map = (2.06, 1.38, 0.96, 0.72)  # L1–L4
within_line_rank_dampening on L1/L2 (1.0, 0.72, 0.58, 0.48)
also reads depth_role strings: "top", "line1", etc.
```

Higher usage → more TOI → more shots → more goal chances.

### Season-long line chemistry (preseason)

**Functions:** `calculate_line_chemistry`, `apply_line_chemistry_effects`, `run_line_chemistry_pass` — `engine.py` ~4131–4471

Chemistry (0–1 scale) modifies player **ratings** before games:
```
offense_mult  = 1.0 + (chemistry − 0.5) × 0.62
passing_mult  = 1.0 + (chemistry − 0.5) × 0.48
IQ_mult       = 1.0 + (chemistry − 0.5) × 0.36
```

This indirectly affects `_gm_offense_weight` → shots, goals, xG. Ratings are restored after the pass.

### In-game team chemistry modifier

**Function:** `_chemistry_game_modifier()` — `engine.py` ~11432

Uses team chemistry cache (overall, tension, buy_in, confidence):
```
modifier = clamp(1.0 + weighted_sum_of_chemistry_factors, 0.94, 1.06)
applied to mu in _simulate_game
```

### World chemistry (franchise mode)

**File:** `SimEngine/app/sim_engine/world/chemistry.py`

| Effect | Formula |
|--------|---------|
| Team strength | `1.0 + 0.04 × (chemistry − 0.5)` → ±2% |
| Post-game update | Win +0.012 (+0.008 blowout); loss −0.010 |
| Chaos dampening | High chemistry reduces score variance |

### How lines affect stats — summary

| Mechanism | What it changes |
|-----------|----------------|
| Line assignment (L1–L4) | TOI, shot weight, goal line share |
| Line rank within line | Usage multiplier |
| Line chemistry (seasonal) | Ratings → offense weight |
| Team chemistry (in-game) | Team expected goals |
| PP line bias | Shot weight multiplier |
| Tank scratch | Removed from goal pools only |

**Chemistry does not add a per-goal bonus in the ledger.** It flows through ratings and team scoring environment.

---

## 7. Advanced Analytics Deep Dive

Analytics are **computed after games** from ledger totals, not simulated event-by-event.

### Metric definitions (canonical)

From `StatsCentralScreen.js` glossary and `player_analytics.py`:

| Metric | Formula | Scale |
|--------|---------|-------|
| **SH%** | Goals ÷ SOG | 0–1 decimal |
| **SV%** | Saves ÷ SA | 0–1 decimal |
| **PDO** | SH% + SV% | ~1.00 neutral (team); ×100 for player on-ice |
| **CF%** | CF ÷ (CF + CA) | 0–1, 0.50 = even |
| **FF%** | FF ÷ (FF + FA) | 0–1 |
| **xGF%** | Average of per-game on-ice xGF% | 0–1, game-averaged |
| **iXG** | Sum of shot-quality-weighted SOG | Volume stat |
| **Finishing** | Goals − iXG | Positive = outscored expected |
| **SH% above expected** | SH% − (iXG ÷ SOG) | Per-shot luck signal |
| **GSAx** | xGA − GA | Positive = saved more than expected |
| **GSAA** | SA × (1 − league_avg_sv%) − GA | vs 0.905 league average |
| **WAR** | (offensive_GAR + defensive_GAR) ÷ 6.0 | Wins above replacement |

### Shot quality / iXG rate

**Function:** `_shot_quality_xg()` — `game_analytics_ledger.py` ~71

```
xg_rate = clamp(0.052 + 0.038 × offense_weight, 0.028, 0.165)
iXG = SOG × xg_rate
```

Better offensive players get higher per-shot xG rates → more iXG per shot → higher expected scoring.

### On-ice vs individual stats

| Stat | Individual | On-ice |
|------|-----------|--------|
| Goals | `g` | `gf_on` (integer share of team goals while on ice) |
| xG | `ixg` | `xgf` / `xga` |
| Corsi | player's `cf`/`ca` | derived from TOI-weighted team events |

On-ice GF/GA allocated by: `offense_weight × TOI × SOG` for GF; `TOI × 1.15 (D bonus)` for GA.

---

## 8. PDO & Shooting % Sustainability

### What PDO measures in this game

**Team PDO:**
```
PDO = (team_goals / team_SOG) + (team_saves / team_SA)
```
~1.000 = league average combined conversion luck.

**Player on-ice PDO:**
```
PDO = (on_ice_SH% + on_ice_SV%) × 100
```
100 = neutral.

### Is high PDO skill or luck?

**Mostly luck within a season**, because:

1. Goals are decided in `_simulate_game` **before** shots are allocated
2. SOG and saves are then distributed to match the predetermined score
3. PDO captures conversion efficiency relative to volume — a **post-hoc** measure

**Skill component exists at the team level:** Better offenses get more goals in Phase 1 via `off_skill`, `star_impact`, chemistry. But a team (or player) running PDO 1.05+ is not automatically "better" — they may be converting at an unsustainable rate.

### Sustainability mechanics

| Mechanism | Exists? | Details |
|-----------|---------|---------|
| Mid-season PDO regression | **No** | No forced decay toward 1.00 |
| SH% mean reversion per game | **No** | Hot shooters stay hot until noise shifts |
| Repeat-goal damping | **Yes, same game only** | Prevents 5-goal games from one player |
| Analytics regression penalty | **Yes** | PDO > 102 reduces `analytics_rating` / impact score |
| Stats Central UI warning | **Yes** | Team PDO ≥ 1.03 flagged "Regression watch" |
| Gaussian goal noise | **Implicit** | Game-to-game variance creates natural cooling |
| Prospect SH% clamps | **Yes** | 3.5–19.5% by position in prospect leagues |

**Analytics regression penalty** (`player_analytics.py` ~1180):
```
penalty = max(0, PDO − 102) × 1.5
        + max(0, finishing_per_60 − 1.0) × 2.0
        + max(0, goals_above_expected − 10) × 0.35
```

This affects **analytical ratings and award watch scores**, not actual in-game performance. A player with PDO 105 will still score at the same rate until the underlying sim inputs change.

### Is high shooting % sustainable?

**Short answer: No, not reliably.**

- Individual SH% = `G / SOG` where G is allocated after score is fixed
- A forward with 18% SH% over 20 games is likely benefiting from:
  - High `offense_weight` (skill — will persist)
  - Favorable goal allocation variance (luck — will regress in analytics view)
- `shooting_pct_above_expected = SH% − (iXG/SOG)` is the better sustainability signal
- Stats Central **Regression List** sorts players by `|G − iXG|` (finishing surplus)

**Practical guidance:**
- SH% > 15% over < 15 GP: treat as small sample
- SH% > 13% with high finishing (G − iXG > 8): flagged as unsustainable in analytics
- True elite finishers have high `offense_weight` AND sustained iXG — not just high SH%

---

## 9. CF%, xGF%, Possession & Finishing

### How CF is built (not tracked live)

CF is **synthesized** after the game, not counted from play-by-play:

```
team_CF = team_SOG × corsi_multiplier (~1.34–1.48) + noise
player_CF = integer_share(team_CF, weighted by offense × TOI × shots)
player_CA = integer_share(opponent_CF, weighted by TOI)
```

**CF% = CF / (CF + CA)**

### Fenwick

```
FF = SOG + missed_shots  (missed = extra CF − blocks)
FA ≈ 0.72 × CA + noise
FF% = FF / (FF + FA)
```

### xGF% — critical nuance

**Season xGF% is the average of per-game xGF%, NOT cumulative:**

```python
# season_xgf_pct_from_row (game_analytics_ledger.py)
if xgf_pct_gp > 0:
    return xgf_pct_sum / xgf_pct_gp   # game-average
else:
    return xgf / (xgf + xga)          # legacy fallback
```

Each game stores one xGF% snapshot per player. Season xGF% = mean of those snapshots.

**Team xGF is scaled to actual results:**
```
target_team_xGF = goals × 0.92 + team_iXG × 0.55 + noise(−0.08, +0.12)
```

This means xGF% correlates with winning by construction — it's a "process" narrative layered on results, not an independent prediction.

### Finishing

```
finishing = goals − iXG   (if iXG > 0)
```

| Finishing | Interpretation |
|-----------|---------------|
| +8 to +15 | Likely running hot; expect cooling |
| +3 to +7 | Above expected, monitor |
| −3 to +3 | Normal |
| −8 or below | Unlucky or usage mismatch |

### CF% skill vs luck

| Component | Driver |
|-----------|--------|
| TOI share | Skill (usage, line, coach) |
| Offense weight in CF allocation | Skill (ratings, OVR) |
| Corsi multiplier noise | Luck |
| Missed shot / block split | Luck |
| xG scaling noise | Luck |

**A player with CF% 54% is likely a genuine possession driver** (high TOI, good ratings). A player with CF% 52% and xGF% 56% but GF% 48% is "process good, results lagging" — the sim's intended narrative.

### Team CF% caveat

Backend `league_team_stats.cf_pct` is sometimes computed as **SOG share** (`SF / (SF + SA)`), not true Corsi. For accurate team CF%, aggregate player `cf`/`ca` sums or use Stats Central's `deriveTeamAnalyticsFromPlayerLedger`.

---

## 10. Goalie Stats & GSAx

### Per-game goalie ledger

| Stat | How computed |
|------|-------------|
| GP | 1 per start |
| GA | Actual goals against (from Phase 1) |
| SA | Blend of skill-based and volume-based |
| Saves | SA − GA |
| SV% | Saves / SA (derived at season export) |
| GAA | GA / GP |
| W/L/OTL/SO | From game result |
| xGA | **Not written in franchise game ledger** |

### Expected save percentage

```
expected_sv% = clamp(0.898 + 0.030 × (goalie_skill − 0.5), 0.888, 0.926)
goalie_skill = avg(GOALIE_KEYS ratings) / 99
```

Better goalies face **more shots** (skill-based SA formula) but stop a higher percentage.

### GSAx formula

```
GSAx = xGA − GA
GSAA = SA × (1 − 0.905) − GA
```

### GSAx in franchise mode — important limitation

**Goalie xGA is not populated during `accumulate_unified_game_stats`.** The xGA allocation in `allocate_team_possession_analytics` applies to **skaters only**.

**Result:** In live franchise play, `gsax` is typically **0** unless xGA is injected elsewhere. The frontend fallback `gsax = xga − ga` with `xga = 0` produces `−GA`, which is misleading.

**GSAA** (vs league average) still works because it only needs SA and GA.

**Prospect leagues** use a simpler proxy:
```
gsax = (save_pct − 0.905) × shots_against
```

### Goalie impact on team defense (skill path)

Goalie quality affects **opponent scoring** in Phase 1 via `_team_goalie_suppression` — this is the real skill impact. GSAx as displayed is often empty; SV% and GAA are reliable.

---

## 11. Cumulative vs Averaged Season Stats

### Cumulative (summed across all games)

| Category | Stats |
|----------|-------|
| Skater counting | GP, G, A, PTS, SOG, PIM, HIT, BLK, TOI_SEC |
| Special teams | PPG, PPA, SHG, SHA |
| Goalie counting | GP, W, L, OTL, SO, GA, SA, SV |
| Analytics volume | CF, CA, FF, FA, xGF, xGA, iXG, xA, GF_on, GA_on |
| xGF% tracking | xgf_pct_sum (sum of game xGF% values), xgf_pct_gp (game count) |

### Derived rates (computed at export, not summed)

| Stat | Formula |
|------|---------|
| SV% | saves / shots_against |
| GAA | ga / gp |
| TOI (display) | (toi_sec / gp) / 60 minutes |
| PPG (points per game) | points / gp |
| SH% | g / sog |
| CF% | cf / (cf + ca) |
| xGF% | xgf_pct_sum / xgf_pct_gp (**average**, not ratio of sums) |
| WAR (skater) | clamp((pts/gp) × 0.22, −2, 6.5) in basic export |
| performance_score | Skater: 28 + 62×min(1.35, (pts/gp)/1.15) |

### Season reset

`player_season_stats = {}` at offseason transition. Each year starts fresh. Career history stored separately on player objects.

---

## 12. Superstars & Generational Talents

### Talent tier hierarchy (bootstrap)

**File:** `league_hierarchy_bootstrap.py`

| Tier | Probability / assignment | Effects |
|------|-------------------------|---------|
| **Transcendent** | ~0.01% per draft class | `dev_potential` 99, `pipeline_tier="transcendent"`, mythic hype, tank target |
| **Franchise** | Rare slot | Elite dev, top OVR band, high ceiling |
| **Elite** | Top draft slots | +0.10 offensive talent, elite dev_type |
| **Top / Round 1** | Standard high picks | Above-average scouting stock |
| **Pool / Hidden upside** | Later rounds | Steal/bust variance |

### How stars separate in-game

#### 1. Superstar offense impact (team level)

**Function:** `_team_superstar_offense_impact()` — `engine.py` ~10390

```
over = max(0, OVR − 84)
player_impact = (over / 10)^1.78 × (0.72 + 0.28 × min(1.0, usage/2.0))
```

Nonlinear: a 95 OVR player bends team scoring far more than an 87. Applied to SOG, SH%, and mu.

#### 2. Offense weight exponents

```
weight = OVR^1.14 × base^1.08 × ...
```

High-OVR players dominate shot and goal shares exponentially, not linearly.

#### 3. Scoring involvement multiplier

Elite tiers get higher `star_mult` in `_gm_scoring_involvement_mult`. Line-1 rank 1 forwards get maximum usage.

#### 4. Repeat-goal damping (limits single-game dominance)

Same player less likely to score their 3rd+ goal in one game — but over a season, stars still lead because weight is so much higher.

#### 5. Young player age gate

Ages ≤21 get reduced `usage_mult` (0.78–0.95) — prevents teenagers from winning scoring titles on projection alone. **Truly elite ratings still rise**, but need time.

#### 6. Progression separation

| Factor | Elite / Transcendent | Average |
|--------|---------------------|---------|
| `dev_type` multiplier | ×1.10–1.38 | ×1.0 |
| `pipeline_tier` offensive bonus | +0.10 to +0.16 | baseline |
| Breakout phase probability | Higher | Lower |
| Prospect PPG scale | ×1.20–1.45 (transcendent) | ×0.90–1.14 |

### Generational talent (transcendent) special rules

- Forced #1 draft rank when in class (`transcendent_tank_behavior.py`)
- Tank pressure narrative on weak teams
- `is_transcendent` flag: +0.14 offensive talent, +0.12 analytics bonus
- Potential display capped at 99 in franchise UI

---

## 13. Skilled Forward vs Defensive Defenseman

### Stat allocation differences

| Aspect | Skilled Forward | Defensive D-Man |
|--------|----------------|-----------------|
| `pos_mult` in offense weight | 1.0 | 0.86 |
| Goal line share | Higher (forward pools) | 12.5% EV / 6.8% PP |
| PP shot bias | Up to 1.24× (L1) | 1.04–1.14× |
| TOI | 18–22 min (L1) | 17–24 min |
| Block/hit distributions | Lower | Higher |
| iXG rate | Higher (offense weight) | Lower |
| On-ice GA weight | Standard TOI | TOI × 1.15 bonus |

### Progression attribute routing

**Function:** `distribute_growth_by_player_type()` — `development.py`

| Style | Growth emphasis |
|-------|----------------|
| Sniper | shooting ×1.30 |
| Playmaker | passing ×1.30, off_aware ×1.15 |
| Defensive D | defense ×1.25, physical ×1.10, shooting ×0.70 |
| Offensive D | passing ×1.20, defense ×0.85 |
| Grinder | physical ×1.20, defense ×1.10, shooting ×0.80 |

### Career lifecycle weights

**Function:** `_career_attribute_weights_for_player()` — `engine.py`

- Shutdown D: def 1.2, block 1.0, shot 0.15
- Sniper: shot 1.25, def 0.15
- Playmaker: pass 1.25, def 0.20

### NHL readiness gates

- Forwards: readiness penalties below age 22
- Defense: penalties below age 20 (−6) and 22 (−2)
- Goalies: mature later (threshold 24)

### Prospect league differences

| Aspect | Defensive D | Skilled Forward |
|--------|-------------|-----------------|
| PPG ceiling | 0.62 | 2.62+ (CHL) |
| Goal share of points | 28–42% | 48–62% (sniper) |
| Plus-minus estimate | +0.14 GP bonus | −0.10 if low defense |
| Analytics weight | `_defensive_analytics_score` heavy | offense-driven |

### Points matter differently

- **Skilled forward:** High PPG directly boosts `performance_growth_modifier` (+0.65 OVR if >70 pts). Points signal drives breakout eligibility.
- **Defensive D:** Lower point totals are "normal." Growth routes to defense/physical ratings. CF%, blocks, and defensive analytics matter more for scouting stock. A 25-point defensive D with strong CF% may progress better than a 35-point offensive D with poor defense.

---

## 14. Stats → Player Progression & Growth

### Two growth systems

| System | When | Driven by |
|--------|------|-----------|
| `run_player_progression` | Offseason | Potential gap, age, dev_type, role, morale |
| `performance_growth_modifier` | After season | **Points totals** |
| `run_career_lifecycle_for_player` | Offseason | Breakout/bust events, trend, potential |

### Main growth formula

**Function:** `apply_player_development()` — `development.py` ~1321

```
growth_base = min(0.15, max(0, (potential − ovr) × 0.5)) + 0.02
growth_base ×= (0.7 + 0.6 × morale)
growth_base ×= ice_time_modifier     # GP, role, TOI quality
growth_base ×= (0.6 + 0.8 × dev_rate)
growth_base ×= age_multiplier        # 1.4 (<21), 1.22 (<24), 0.68 else
growth_base ×= career_arc_phase
growth_base ×= 0.52                  # global dampener
```

**Goals and assists are NOT in this formula directly.**

### How points DO matter

**Function:** `performance_growth_modifier()` — `engine.py` ~2872

| Season points | OVR nudge |
|---------------|-----------|
| > 70 | **+0.65** |
| < 20 | **−0.85** |
| Prime phase | Dampened to ×0.55 |

Points source: `player.season_stats[year].points`

### Indirect stat → growth paths

| Path | Mechanism |
|------|-----------|
| GP / ice time | `_ice_time_modifier`: GP < 30 → ×0.6; GP ≥ 70 → full |
| Morale | Wins/losses after each game → growth_base multiplier |
| Career momentum | `recent_performance_score` can trigger SPIKE phase |
| Contract evaluation | `pts/gp` capped 0–1.4 for trade/contract signals |
| Role assignment | OVR-based (`role_changes.py`), not stat-based |

### Dev type effects on growth

| dev_type | Growth multiplier |
|----------|------------------|
| bust | ×0.30–0.58 |
| elite / steal | ×1.10–1.38 |
| slow | ×0.72 |
| late_bloomer (age ≥22) | ×1.12 |

### Development phases (archetype-driven)

| Phase | Multiplier |
|-------|-----------|
| STALL | ×0.03–0.19 |
| SPIKE | ×1.38–2.05 |
| REGRESSION | ×−1.22 to −0.32 |
| NORMAL | ×1.0 |

### Summary: how much do points matter?

| Player type | Points importance |
|-------------|------------------|
| Elite forward | **High** — direct OVR nudge at 70+ pts, morale loop, star usage |
| Middle-6 forward | **Moderate** — ice time modifier, morale, contract value |
| Defensive D | **Low for growth, moderate for career** — growth routes to defense regardless; low points won't penalize as harshly if role is correct |
| Goalie | **Low** — SV%, GAA, GSAA matter more for perception; progression uses goalie-specific curves |
| Prospect | **High for stock, not for NHL OVR** — prospect PPG drives draft stock delta |

---

## 15. Prospect League Stat Simulation

**File:** `prospect_league_scoring.py`

Separate system from NHL game ledger. Simulates junior/college/overseas stat lines.

### Flow

```
initialize_prospect_season → projected GP, PPG, stat line
advance_prospect_stats_to_date → delta GP only (calendar-driven)
  → _simulate_skater_games or _simulate_goalie_games per delta
  → stock/analytics recompute
```

### Skater game simulation

Per game in delta:
```
game_pts = round(target_ppg × uniform(1−vol, 1+vol) × streak + gaussian(0, 0.35))
```

Boom-bust: 6% chance of 2.0–3.4× PPG burst. Character concerns: 4% zero-point game.

### League difficulty profiles

Each development league has `scoring_multiplier`, `difficulty`, `average_ppg_target`, `elite_ppg_target`, `environment_label`. CHL scores higher than NCAA; European leagues adjust translation.

### Draft stock

Computed from:
- `_production_vs_projection` (actual PPG vs projected)
- `_analytics_process_score` (talent + recent form)
- Weekly vs season stock fields

**Display stock is weekly; season signal stays internal for ranking.**

---

## 16. World Systems: Morale, Fatigue, Momentum

Applied in franchise mode before `_simulate_game` in `_simulate_franchise_slot`.

### Combined team strength scale

```
team_scale = momentum × chemistry × fatigue × morale  (each ~0.93–1.07)
```

| Module | File | Game effect |
|--------|------|-------------|
| Momentum | `world/momentum.py` | ±5% from W/L streaks; decays over days |
| Chemistry | `world/chemistry.py` | ±2% team strength; chaos dampening |
| Fatigue | `world/fatigue.py` | B2B +5.5 load; up to ~3.5% penalty |
| Morale | `world/morale.py` | ±1.5% team strength from avg morale |

### Post-game updates (every dressed player)

- Morale: win +1.1 + 0.35×goal_diff; loss −1.0 − 0.3×goal_diff
- Chemistry: win +0.012; loss −0.010
- Momentum: updated from result
- Fatigue: ticked for active skaters

**These affect tomorrow's game, not today's stat line.** Today's stats are already allocated.

---

## 17. Stats Central Pipeline

### Backend payload

**Function:** `_build_stats_central_payload()` — `franchise_sim.py` ~2899

```
session.player_season_stats → normalize → split skaters/goalies
→ leaders (top 45 by PTS)
→ league_team_stats (from game boxes)
→ integrity checks (player goals = box score goals)
```

**Default payload is raw ledger rows** — PDO, CF%, GSAx computed on frontend.

Richer path exists: `build_stats_central_player_payload()` in `player_analytics.py` (not wired into default franchise state).

### Frontend normalization

**File:** `StatsCentralScreen.js`

1. Merge `stats_central`, `analytics`, `player_analytics` from franchise state
2. `normalizeSkater` / `normalizeGoalie` / `normalizeTeam`
3. Derive CF%, xGF%, PDO, finishing, WAR client-side
4. `deriveTeamAnalyticsFromPlayerLedger` fills team gaps from player sums
5. Leaderboards: PTS, G, A, iXG, GSAx, SV%, analytics_rating
6. Regression watch: team PDO bands, finishing list

### Glossary (in-app)

The Stats Central screen includes a built-in formula glossary covering PDO, CF%, xGF%, finishing, GSAx, WAR, and 50+ other metrics.

---

## 18. Known Limitations & Interpretation Guide

| Topic | Reality in this sim |
|-------|---------------------|
| xGF drives goals | **No** — goals drive xGF scaling |
| PDO sustainability | **Not enforced in-game** — analytics/UI flag only |
| High SH% will regress | **Eventually via noise**, not explicit regression |
| GSAx for NHL goalies | **Often 0** — xGA not in goalie ledger |
| Team CF% in backend | **Sometimes SOG%**, not true Corsi |
| Bench players | **Don't exist** — all healthy roster skaters dress |
| Chemistry per goal | **No** — chemistry modifies ratings and team mu |
| Points → direct growth | **Only via performance_growth_modifier** (70+/20− thresholds) |
| light_mode bulk sim | Skips xGF allocation and full game boxes |

### How to read a player's season

1. **Check GP** — small samples distort everything
2. **Compare SH% to iXG/SOG** — finishing surplus = luck signal
3. **Check xGF% vs GF%** — process vs results gap
4. **Check PDO** — above 102 = regression risk in analytics
5. **For D-men** — weight CF%, blocks, defensive analytics over points
6. **For goalies** — trust SV%, GAA, GSAA; be skeptical of GSAx unless xGA populated

---

## 19. Key File Reference

| Topic | Path |
|-------|------|
| Game score simulation | `SimEngine/app/sim_engine/engine.py` → `_simulate_game` |
| Stat allocation | `SimEngine/app/sim_engine/engine.py` → `accumulate_unified_game_stats` |
| Goal/assist picker | `SimEngine/app/sim_engine/engine.py` → `_gm_pick_goal_scorer`, `_gm_pick_assists` |
| Offense weight | `SimEngine/app/sim_engine/engine.py` → `_gm_offense_weight` |
| xG / Corsi allocation | `SimEngine/app/sim_engine/gameplay/game_analytics_ledger.py` |
| PDO / WAR / finishing | `SimEngine/app/sim_engine/generation/player_analytics.py` |
| Franchise game loop | `backend/services/franchise_sim.py` → `_simulate_franchise_slot`, `_accumulate_franchise_game_stats` |
| Stats Central payload | `backend/services/franchise_sim.py` → `_build_stats_central_payload` |
| Stats Central UI | `frontend/src/screens/StatsCentralScreen.js` |
| Player progression | `SimEngine/app/sim_engine/progression/development.py` |
| Prospect stat sim | `SimEngine/app/sim_engine/generation/prospect_league_scoring.py` |
| Talent bootstrap | `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` |
| Line chemistry | `SimEngine/app/sim_engine/engine.py` → `calculate_line_chemistry` |
| World modifiers | `SimEngine/app/sim_engine/world/{momentum,morale,fatigue,chemistry}.py` |
| Season stat normalization | `SimEngine/app/sim_engine/franchise/serialization.py` |
| Long-run stability | `SimEngine/app/sim_engine/tuning/normalization.py` |

---

*Report generated from live codebase audit. Formulas reference implementation as of July 2026.*
