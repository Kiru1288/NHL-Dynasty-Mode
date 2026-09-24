# Player Development & Growth System Report

How players actually grow, plateau, and decline in **NHL Franchise Mode** today: the age curve, the potential model, how young / old / high-ceiling / low-ceiling players diverge, every outside source that touches growth, and what a GM can actually influence.

**Live source of truth:** `SimEngine/app/sim_engine/progression/` (`development.py` is the engine — 2,523 lines), orchestrated by `progression/__init__.py::run_player_progression`.

---

## 1. Architecture overview

```
Season loop
  _snapshot_season_start_ovrs()            franchise_sim.py:12304   freeze display OVR
  ...games played...
  _nhl_in_season_development_tick()        franchise_sim.py:12333   periodic pulses
    -> _dev_stamp_season_production()      real PPG/SV% -> production_score
    -> apply_in_season_development_pulse() draws from the 32% in-season pool

Offseason (development stage)
  prime_development_environment_for_rosters(teams, rng)   <- team/league context
  apply_narrative_mechanics_to_rosters(...)               <- storyline multipliers
  for each player: run_player_progression()
      1. apply_player_development()    growth (the 58% season-end pool)
      2. update_player_potential()     breakout / bust / stagnate drift
      3. apply_regression()            injury + morale wear
      4. update_player_role()          role label from OVR
      5. should_player_retire()
  career_aging_decline_try_v3()        engine.py:3008   the real age cliff
  run_unsigned_prospect_development_pass()               junior/NCAA/Euro path
  _apply_wjc_development_to_prospects()  franchise_sim.py:12756   tournament bumps
```

| Layer | Job | Key file |
|---|---|---|
| Growth engine | Budget → attribute deltas | `progression/development.py` |
| Potential model | Expected vs maximum ceiling, drift | `progression/potential.py` |
| Wear regression | Injury/morale decay past peak | `progression/regression.py` |
| Age cliff | Probabilistic OVR loss, league-capped | `engine.py::_career_aging_decline_try_v3` |
| Team/league context | Env multipliers per player | `development.py::prime_development_environment_for_rosters` |
| Storylines | Narrative growth/decline multipliers | `narrative/player_journeys.py` |
| Outside-org path | Unsigned prospects in CHL/NCAA/Euro | `backend/services/unsigned_prospect_development.py` |
| Tournament | WJC skill + potential bumps | `franchise_sim.py::_apply_wjc_development_to_prospects` |

**Scales:** everything internal is 0–1. `display = round(x * 99)`. A growth "budget" of `0.045` means a target of **+4.5 displayed OVR**.

**Idempotence:** every player carries a `development_ledger` keyed by season (`potential.py::ensure_development_ledger`). Growth, aging, and potential drift each fire **once per season**; re-running a stage is a no-op. Prior seasons roll into `development_history`.

---

## 2. The core growth equation

`calculate_season_growth_budget()` (development.py:1625) is where almost everything lands.

### 2a. The base: runway to your ceiling

```python
gap_exp  = expected_ceiling - current_ovr          # 0-1 scale
approach = min(0.072, gap_exp * 0.42 + 0.024)      # cap 0.072 = ~+7.1 OVR
if gap_exp <= 0.012:                               # already at ceiling
    approach = min(0.028, max(0.010, gap_max * 0.10 + 0.008))
```

**This is the single most important fact in the system: growth is driven by the gap, not by age.** Age only modulates how fast you close it. A player at his ceiling gets ~+1 OVR of polish no matter how young he is.

### 2b. The multiplier stack

Everything below multiplies into one `mod`:

| Factor | Range | Source |
|---|---|---|
| Morale | `0.78 + 0.48·morale` → **0.78–1.26** | `psych.morale` |
| Ice time / opportunity | **0.55–1.22** (clamped) | `_ice_time_modifier()` |
| Development rate | `0.70 + 0.70·rate` → **0.81–1.37** | profile `development_rate` (0.15–0.95) |
| Team environment | **0.72–1.28** | `_dev_env_growth_mult` |
| Narrative | **0.75–1.25** | `_narrative_prog_growth_mult` |
| Breakout momentum | `1 + 0.18·momentum` → **1.00–1.18** | `_dev_breakout_momentum` |
| **Age band** | **0.34–1.38** | see §3 |
| Big-runway bonus | 1.04–1.16 | young + large gap |
| Role | 0.88–1.14 | `elite/top_line` vs `depth/scratch` |
| Production vs expectation | **0.72–1.42** | real PPG / SV% |
| Injury days | ×0.78 (≥20d), ×0.62 (≥40d) | `injury_days` |
| Healthy scratches | ×0.78 (≥10), ×0.55 (≥20) | `healthy_scratches` |

Then phase, noise, and hard caps:

```
budget = approach * mod * noise(vol)
STALL      -> x 0.10-0.52
SPIKE      -> x 1.55-2.20  (+ breakout push into maximum_ceiling)
REGRESSION -> negative, -0.008 to -0.035
caps: hard_cap 0.085 (0.110 on SPIKE), gap_cap = gap_exp*0.78 + 0.022
floors: +0.045 (age <=20, gap >=0.10), +0.036 (age <=23, gap >=0.07), +0.030 (gap >=0.04)
decline_state: x0.70 (early_decline) / x0.50 (late_decline)
```

### 2c. Pool split

```python
_IN_SEASON_POOL_SHARE  = 0.32   # mid-season pulses
_SEASON_END_POOL_SHARE = 0.58   # offseason development stage
```

A player realizes ~**90%** of his computed budget across a full season. Mid-season pulses do not claw back the season-end pool; they draw from a separate capped bucket tracked on `_in_season_growth_spent_01`.

> **Code/comment drift:** the comment at development.py:2280 says the season-end pool is "65–75%" but the constant is `0.58`. The docstring on `apply_in_season_development_pulse` says "~30%" against `0.32`. Worth reconciling if the design doc is the reference.

### 2d. From budget to attributes

`allocate_growth_to_attributes()` → `distribute_growth_by_player_type()` weights rating keys by playstyle (sniper → shooting 1.30; playmaker → passing 1.30; goalie → rebound 1.25 / positioning 1.20 / skating 0.50), then **concentrates into the top 4–8 keys** rather than spraying evenly. Per-key deltas are clamped to `[0.8, 9.0]` on growth and `[-5.0, -0.5]` on regression.

Because attribute growth dilutes through the OVR formula, `ensure_displayed_ovr_delta()` runs a corrective loop so the visible OVR actually lands near the target.

---

## 3. The age curve (as actually implemented)

### Growth-side age multiplier — `calculate_season_growth_budget`

| Age | Skater × | Goalie × |
|---|---|---|
| <20 | **1.38** | 1.22 |
| 20–21 | **1.26** | 1.18 |
| 22–23 | 1.14 | 1.14 |
| 24–26 | 1.04 | 1.04 |
| 27–28 | 0.88 | 0.96 |
| 29–31 | **0.62** | 0.62 |
| 32+ | **0.34** | 0.34 |

### The hard eligibility gate — `apply_player_development`

```python
growth_eligible = age <= 29
  else: goalies 24-31 who are late_bloomer or win a 28% roll
  else: ages 29-32 who win a 22% roll
```

**Anyone 33+ is hard-locked out of growth.** They get `apply_prime_refinement` only (a +0.035–0.05 nudge on consistency/poise/awareness keys for ages 25–31) and are marked `outside_window` in the ledger.

### Decline side — two independent systems

**1. `career_aging_decline_try_v3` (engine.py:3008) — the live age cliff.**

| Age | Base | After ×0.55 (and ×0.4 if ≤29) | Magnitude |
|---|---|---|---|
| ≤24 | 0.00 | — | none |
| 25–29 | 0.12 | **2.6%** | 0.4–1.2 typical |
| 30–33 | 0.28 | **15.4%** | capped −1.8 |
| 34–36 | 0.42 | **23.1%** | capped −3.2 |
| 37+ | 0.58 | **31.9%** | capped −3.2 |

OVR ≥90 halves the chance and cuts magnitude ×0.7. The league caps total aging events at **18% of rostered players per season** (`prime_league_season_aging_v3`), so decline is rationed globally, not rolled independently.

**2. `apply_regression` (regression.py:180) — wear, not age.** Only fires past `expected_peak_age`. Injury history (≤0.022) + wear-and-tear + morale penalties (+0.012 under 0.4 morale), ×0.65 for elite veterans, spread evenly across *all* ratings.

> **Dead module:** `progression/aging_curves.py::get_age_modifier` defines a full position-specific curve (peak 26 F / 27 D / 29 G) and **is never called** — only re-exported in `__init__.py`. The live curve is the table above. Deleting it or wiring it in would remove a misleading source of truth.

---

## 4. Potential: two ceilings, and both move

`resolve_development_profile()` establishes per player:

- **`expected_ceiling`** — the projected potential. Mirrored to `player.potential` and `ratings.dev_potential`. This is what the growth gap is measured against.
- **`maximum_ceiling`** — the true hard cap. Mirrored to `ratings.dev_ceiling`. Only breakouts move it.
- `development_rate` (0.15–0.95), `volatility` (0.05–0.95), `bust_risk`, `breakout_chance`, `decline_state`.

Resolution priority: existing profile → explicit `true_potential`/`potential` fields → inferred from tools + production + age + archetype (headroom clamped to `0.015–0.18`, i.e. **at most ~18 OVR of inferred runway**) → safe default.

### Potential drift — `apply_potential_drift` (potential.py:96)

Once per season, ±0.01–0.02 (±1–2 OVR):

| Outcome | Gate | Base p |
|---|---|---|
| **breakout** | age <24, morale ≥0.6, potential <0.90 | 7.8% |
| **bust** | age <23, morale <0.4, OVR <0.70 | 10.0% |
| **bust_pressure** | age <24, `_bust_pressure` >0.5 | 6% + 14%·excess |
| **stagnate** | 23–27, morale <0.45 | 6% |

Archetype shifts those: `HIGH_VARIANCE` +3.5% breakout / +2.8% bust; `ELITE_CEILING_VOLATILE` +2% / +3.5%; `SAFE_LOW_CEILING` −2.2% / +1.2%; `LATE_BLOOMER` +4% breakout in the 22–26 window.

Asymmetry is deliberate: **a bust cuts `expected_ceiling` first** and only nicks `maximum` (×0.35); a breakout raises expected and pushes maximum by ×0.5. You lose projection faster than you lose true talent.

### Earning a higher ceiling — `reevaluate_ceilings_from_performance` (development.py:1961)

This is the "potential is an evaluation, not a destination" system. Age ≤29 only.

```python
expected_prod = clamp(0.28 + (ovr100 - 70) * 0.012, 0.30, 0.92)
overperf      = production_score - expected_prod
```

Overperformance accrues into `_dev_breakout_momentum` across seasons (+0.18–0.40 for a big year, −0.14 for a bad one), and then:

| Evidence | Projected pot | Maximum |
|---|---|---|
| overperf ≥0.22, momentum ≥0.45, age ≤24 | **+2.5 to +4.5** | +0.5 to +2.0 |
| overperf ≥0.14 or momentum ≥0.55 | +1.5 to +3.0 | 0 to +1.2 |
| overperf ≥0.08 | +0.8 to +1.8 | 0 |

A 68-OVR kid who posts 0.95 PPG raises his own ceiling for years. This is the mechanism by which low-drafted players become stars.

---

## 5. How the four player types actually behave

Worked traces at default morale (0.6), average production, healthy, neutral environment.

### 5a. Young + high potential — 19yo, 70 OVR, 88 potential

```
gap_exp  = 0.182
approach = min(0.072, 0.182*0.42 + 0.024 = 0.100) -> 0.072   (capped)
mod      = 1.068 (morale) * 0.92 (ice) * 1.05 (rate) * 1.38 (age<20)
           * 1.16 (gap >= 0.10 bonus) * 1.03 (prod) = ~1.70
budget   = 0.072 * 1.70 * 1.07 (NORMAL) = 0.131
caps     -> hard_cap 0.085  <- binds
realized = 0.085 * 0.90 = +7.6 displayed OVR
```

**Result: +7 to +8 per season, and SPIKE years reach the 0.110 cap (~+9.8).** Three seasons closes almost the whole runway. The floors (`+0.045` minimum at age ≤20 with a big gap) guarantee even a bad-luck year returns ~+4. These are the franchise kids, and the system is deliberately generous with them.

### 5b. Young + low potential — 20yo, 62 OVR, 66 potential

```
gap_exp  = 0.040
approach = 0.040*0.42 + 0.024 = 0.0408
role     = "prospect" (age <=22 and OVR <0.70 force it) -> ice base 0.85
mod      = ~1.14 -> budget = ~0.046
caps     gap_cap = 0.0532 (not binding)
realized = +4.1 displayed OVR
```

Next season the gap is ~0.005 and the whole formula collapses to the near-ceiling branch (`approach ≤ 0.028`, then the `gap_now ≤ 0.004` clamp to `0.012–0.022`): **~+1 OVR/year, forever.** Low-potential players sprint to their ceiling in 1–2 seasons and then flatline. Their only path upward is `reevaluate_ceilings_from_performance` — they must *outproduce their rating* to get a bigger ceiling.

The system also penalizes them structurally: being forced to `role = "prospect"` drops the ice-time base to 0.85 vs 1.20 for top-line, a permanent −29% on growth.

### 5c. Prime — 26yo, 82 OVR, 83 potential

```
gap_exp <= 0.012 -> approach = 0.011
age 24-26 -> x1.04
budget = ~0.012, then the "gap_now <= 0.02 and NORMAL" floor lifts it to 0.018
realized = +1.0 to +1.6 displayed
plus apply_prime_refinement: +0.035-0.05 on consistency/poise/awareness keys
```

**Prime is maintenance, not growth.** The upside case is a prime player with genuine unrealized headroom (`gap_exp ≥ 0.08` at age ≤26 gets a 1.04 bonus) or one who triggers a ceiling re-evaluation.

### 5d. Old — 33yo, 85 OVR

```
growth_eligible = False   (age > 32, no goalie exemption, no 29-32 roll)
-> ledger source_path = "apply_player_development:outside_window", zero growth
decline: v3 rolls 15.4%/season, magnitude capped at -1.8 (age <34)
         apply_regression adds injury/morale wear on top
         retirement: age >=35 only (8% at 35-36, 25% at 37+, 55% at 39+, 85% at 41+)
```

**34–36:** 23.1% decline chance, cap −3.2, plus a 0.5–1.0% rare-spike path for −3.5 to −5.0 in `regression.py`. **37+:** 31.9%.

Protections: OVR ≥90 halves chance and magnitude; `decline_cooldown` of 1–2 seasons after any decline; `_aging_decline_chain_active` damps a consecutive decline ×0.6; elite veterans (age ≥30, OVR ≥0.82, top role) take ×0.65 wear regression; cup wins subtract up to 15 points of retirement chance.

**Net:** a healthy 85-OVR 33-year-old loses ~0.2 OVR/season in expectation. A worn, low-morale one on a bad team loses meaningfully more. Aging in this build is gentle and heavily capped — by design, given the 18% league-wide event ceiling.

### 5e. Goalies

Treated as a separate species throughout: growth multipliers are flatter (1.22 vs 1.38 at age <20, but 0.96 vs 0.88 at 27–28), career stage thresholds are 1–2 years later, the growth window extends to 31 via the 28% roll, inferred headroom decays more slowly, and decline probability is cut ×0.78 before 34. Attribute growth funnels into rebound/glove/blocker/positioning with skating weighted at 0.50.

---

## 6. Development archetypes and windows

Two orthogonal randomizers sit on top of the ceiling math.

### Archetype — *what shape the career takes*

Assigned at generation or lazily by `_lazy_assign_dev_archetype()` weighted by potential tier. Drives the annual phase roll (`_dev_archetype_phase_roll`):

| Archetype | STALL | SPIKE | REGRESSION |
|---|---|---|---|
| `FAST_RISER` | 8.5% (14.5% at 21+) | 9.5% (5.5% at 21+) | 2.8% |
| `LATE_BLOOMER` | 15.5% (10.5% at 20–24) | 5.0% (13.5% at 20–24) | 3.8% |
| `HIGH_VARIANCE` | 11.5% | 15.5% | 7.5% |
| `SAFE_LOW_CEILING` | 9.8% | 4.8% | 2.4% |
| `ELITE_CEILING_VOLATILE` | 10.5% | 12.5% | 8.5% |
| `STALLED_DEVELOPER` | **21.5%** | 3.5% | 4.5% |

A `boom_bust` curve hint adds +4.5% spike / +3.5% regression; `slow` adds +4.5% stall.

### Development window — *when the growth lands*

`_assign_development_window()` gives a multiplier by age band:

| Window | ≤21 | 22–24 | 25–27 | 28+ |
|---|---|---|---|---|
| `early_developer` | 1.18 | 1.00 | 0.84 | 0.84 |
| `normal_developer` | 0.90 (≤19) / 1.08 | 1.08 | 0.92 | 0.92 |
| `late_bloomer` | 0.82 | 1.16 | 1.16 | 0.94 |
| `long_project` | 0.88 | 1.07 | 1.07 | 0.92 (29+) |
| `flash_prospect` | **1.22** | 0.96 | 0.78 | 0.78 |
| `raw_talent` | **0.74** | 1.13 | 1.13 | 0.94 |

### Phase overrides

`apply_player_development` will override a rolled phase when the evidence contradicts it:

- Age ≤26, gap ≥0.04, production ≥0.78 → STALL is cancelled (upgraded to SPIKE at 55% if production ≥0.88)
- Age ≤24, gap ≥0.08, production ≥0.85 → REGRESSION is cancelled 65% of the time
- `career_momentum` ≥70 → 22% chance to force SPIKE; ≤−70 → 24% to force STALL
- `_nhl_adjustment_years_remaining` >0 (rookies: 2 years if ≤21, 1 if ≤23) → 40% chance of a forced STALL/SPIKE/REGRESSION — the rookie-wall mechanic

---

## 7. Outside sources — everything external that touches growth

### 7a. Team environment — `prime_development_environment_for_rosters`

Computed per season, written to `_dev_env_growth_mult` (final clamp 0.70–1.42, re-clamped to 0.72–1.28 on consumption):

| Input | Effect |
|---|---|
| **Team window = rebuild** | growth ×1.075, variance ×1.055 |
| **Team window = contender** | growth ×1.0, variance ×1.06 (deliberately *not* a penalty — see comment at development.py:1141) |
| **Prospect pipeline score** | `0.76 + 0.48·pscore` → **×0.76 to ×1.24** — the biggest single org lever |
| **Superstar teammates** | +3% for C/D, +5% for wingers, scaled by count of 90+ OVR in the top 5 |
| **Young on a contender** | ×1.06 (age ≤23), ×1.02 (age ≤26) |
| **Opportunity proxy** | `0.92 + 0.10·(ice_mod − 0.85)` |
| **Org profile label** | `opportunity_driven` / `win_now_congested` / `development_strong` / `balanced` |

### 7b. League-wide ecosystem correction

The same function measures league talent against targets of **2.0 players ≥85 OVR, 11.0 ≥80, 20.0 ≥75 per team** and builds a `scarcity_index` (−0.35 to +0.35):

- `league_growth_mult = 1.0 + 0.24·scarcity` → **0.82–1.08**
- Positional targets (4.1 C, 7.4 D, 2.4 G per team) drive `pos_opportunity_mult` → **0.90–1.12**

A talent-starved league grows everyone faster; an overloaded one damps growth. The comment explicitly calls this out as anti-inflation protection for decade-long saves.

### 7c. Narrative storylines — `narrative/player_journeys.py`

Active narrative events compound into per-player multipliers, clamped at the end:

| Event | prog_growth | decline_p |
|---|---|---|
| `CONFIDENCE_COLLAPSE` | −14% | +14% |
| `BURNOUT_WEAR` | −12% | +12% |
| `MEDIA_STRAIN` | −8% | +8% |
| `RESURGENCE_DRIVE` | +11% | −10% |
| `MOMENTUM_HOT` | +9% | −12% |
| `LEADERSHIP_SURGE` | +6% | −6% |

Final clamps: growth **0.78–1.18**, regression rate 0.74–1.30, decline probability 0.72–1.32.

### 7d. World Juniors — `_apply_wjc_development_to_prospects`

The only system that writes attributes **outside** the normal budget and ledger. Score = `stock_delta·0.35` + production (goals ×1.8, assists ×1.2, +/− ×0.4; goalies: W ×2.5, L ×−1.5, SO ×3.0, SV% ≥.930 +4.0) + medal (gold +6.0, silver +3.0, bronze +1.5).

| Grade | Score | Skill bump | Potential bump |
|---|---|---|---|
| breakout | ≥8.0 | up to **+3.0** on 11 keys | up to **+2.5** |
| positive | ≥3.0 | up to +1.8 | up to +1.2 |
| neutral | −3 to 3 | +0.12 if GP ≥4 | 0 |
| quiet | <0 | down to −0.8 | down to −1.0 |
| setback | ≤−3.0 | down to −1.5 | down to **−1.8** |

Also feeds `session.wjc_draft_score_boosts` into draft rankings (`draft_ranking_logic.py:1193`) and writes a `source_path: "world_juniors"` entry into `development_history`.

### 7e. Outside-the-org development — `unsigned_prospect_development.py`

Drafted-but-unsigned players run the *same* core engine with contextual modifiers:

| League | Quality |
|---|---|
| Europe (SHL / Liiga / DEL / KHL) | **0.80** |
| NCAA | 0.78 |
| CHL (OHL / WHL / QMJHL) | 0.72 |
| USHL | 0.62 |
| other | 0.65 |

Applied as `league_quality_mod = 0.88 + 0.2·lq` (**0.99–1.04**), plus `org_mod` from the NHL team's development plan (0.86 + 0.22·plan), `coach_mod` (0.90 + 0.16·coaching_quality), and `prod_mod` from junior PPG. Context flags `overmatched` / `underchallenged` / `bench_or_scratch` are computed and passed in. A `request_league_transfer` gives the league-quality mod a ×1.05 bump (capped 1.08).

Randomness here is **deterministic** — hashed from `(player_id, season_year)` rather than the sim RNG, so the same prospect develops identically on replay.

### 7f. Morale, injuries, production

- **Morale** → `0.78 + 0.48·morale` on the budget, *plus* gates the breakout roll (needs ≥0.6) and the bust roll (needs <0.4). Written by chemistry (`systems/chemistry.py`), contract events, trades, injuries, and ELC ledger outcomes.
- **Injury days** → ×0.78 at 20+, ×0.62 at 40+, and separately feed `_aging_decline_wear` (0.42 weight on injury history — the heaviest term in the wear model).
- **Production** → `_dev_stamp_season_production` converts real season stats to `production_score` (skaters: `0.32 + PPG·0.55`, floored to 0.82/0.90/0.95 at 0.85/0.95/1.10 PPG; goalies: `0.35 + (SV% − .880)·8.5`). This feeds the budget multiplier, the phase overrides, and ceiling re-evaluation.

---

## 8. What can actually be influenced

### Direct GM levers (real, measurable)

| Lever | Mechanism | Magnitude |
|---|---|---|
| **Call up / send down** | Changes `games_played` → `_ice_time_modifier` gp bands | **gp ≥70 ×1.0, ≥50 ×0.9, ≥30 ×0.75, else ×0.6** — a −40% swing |
| **Roster construction** | `prospect_pipeline_score` drives env mult | **×0.76 to ×1.24** — the biggest controllable factor |
| **Team window (rebuild vs contend)** | `_infer_team_dev_window` from team strategy fields | ×1.075 growth on rebuild |
| **Acquiring 90+ OVR players** | `superstar_factor` teammate boost | +3% (C/D) to +5% (wingers) |
| **League placement for unsigned prospects** | `request_league_transfer` → league quality | ×1.05 (small) |
| **Playing time volume** | Healthy scratches counter | ×0.78 at 10, ×0.55 at 20 |
| **Managing morale** | Trades, contracts, chemistry | ×0.78–1.26, plus breakout/bust gates |
| **WJC roster participation** | Tournament impact pass | ±3.0 skill, ±2.5 potential in one shot |

### Indirect (emerges from winning / losing)

- Production → budget multiplier, phase overrides, and long-term ceiling re-evaluation. **Playing a kid in a role where he produces is the highest-leverage thing a GM does**, because it compounds through `_dev_breakout_momentum`.
- Injury management → both budget penalty and permanent `_aging_decline_wear` accumulation.
- League scarcity → other teams' rosters change *your* players' growth rate.

### Not influenceable

- Archetype and development window — assigned once, never re-rolled
- `maximum_ceiling` except through breakout events
- Phase rolls (RNG, though overridable by production evidence)
- The 33+ growth lockout
- The league-wide 18% aging-event budget

---

## 9. Issues

Severity-ordered. Items 1–5 were **reproduced by running the engine**, not inferred from reading it; the repro scripts are described inline.

### 🔴 1. Potential does not cap anything — careers converge on 92–99 regardless of ceiling

12-season careers via `run_player_progression`, 40 seeds each, production held at 0.55:

| Start | Final OVR (median) | Range | Final potential | Overshoot vs starting potential |
|---|---|---|---|---|
| 18yo, 60 OVR / **70 pot** | **96** | 91–99 | 99 | **+26** |
| 18yo, 65 OVR / **80 pot** | **99** | 95–99 | 99 | +19 |
| 19yo, 70 OVR / **88 pot** | **99** | 98–99 | 99 | +11 |
| 21yo, 72 OVR / **75 pot** | **94** | 88–96 | 96 | +18 |
| 23yo, 78 OVR / **79 pot** | **92** | 90–96 | 95 | +13 |

A 70-potential player finishes at 96. **The entire potential system — draft scouting, dev archetypes, bust/breakout drift, ceiling tiers — is cosmetic**, because every path converges on the same elite outcome. Causes are items 2 and 3.

### 🔴 2. `expected_prod` and `production_score` are on incompatible scales

Two formulas that must agree, and don't:

```
production_score = 0.32 + PPG * 0.55                              franchise_offseason.py:2717
expected_prod    = clamp(0.28 + (OVR - 70) * 0.012, 0.30, 0.92)   development.py:1986
overperf         = production_score - expected_prod               development.py:1988
```

`reevaluate_ceilings_from_performance` raises potential at `overperf ≥ 0.08` and grants the top tier (**+2.5 to +4.5 potential**) at `≥ 0.22`. But the break-even PPG required to merely *meet* expectations is:

| OVR | expected_prod | Break-even PPG | Realistic PPG | Result every season |
|---|---|---|---|---|
| 60 | 0.300 | **−0.036** | 0.15 | solid (+0.8–1.8) |
| 70 | 0.300 | **−0.036** | 0.30 | strong (+1.5–3) |
| 80 | 0.400 | 0.145 | 0.55 | **BREAKOUT (+2.5–4.5)** |
| 90 | 0.520 | 0.364 | 0.90 | **BREAKOUT (+2.5–4.5)** |
| 99 | 0.628 | 0.560 | 1.30 | **BREAKOUT (+2.5–4.5)** |

Anyone under ~78 OVR **mathematically cannot** meet expectations — the required PPG is negative. Every player in the league raises his own ceiling every year, and everyone 80+ maxes the tier. The §4 claim that "a 68-OVR kid who posts 0.95 PPG raises his own ceiling" is true of *literally every player*, which makes it meaningless.

**Fix:** `expected_prod` needs to be expressed on the `production_score` scale. `0.32 + 0.55 · expected_PPG(OVR)` with a realistic `expected_PPG` curve (≈0.15 at 60 → ≈1.20 at 99) would make the comparison honest.

### 🔴 3. The ceiling ratchet — potential chases OVR with no brake

Independent of item 2. Repro: player at 24 with **gap = 0.0000** (OVR exactly at potential) and production pinned to 0.20, below the `expected_prod` floor so no ceiling re-evaluation can fire:

```
age   24   25   26   27   28   29   30   31   32   33
OVR   82   83   83   83   83   83   84   85   85   85     (+5 with zero runway)
pot   82   83   83   83   83   83   84   85   85   85     (lockstep, forever)
gap  0.0000 every single season
```

Two pieces of code form the loop:

```python
# development.py:2277 — near-ceiling floor: budget never reaches zero
if gap_now <= 0.004 and dev_phase not in ("REGRESSION", "SPIKE"):
    budget = min(max(budget, 0.012), 0.022)
elif gap_now <= 0.02 and dev_phase == "NORMAL" and budget > 0:
    budget = max(budget, 0.018)

# development.py:1419 — potential is then raised to match the new OVR
if expected < current_ovr:
    expected = current_ovr
    maximum = max(maximum, expected)
```

Growth pushes OVR past potential → `_finalize` lifts potential to match → a new gap exists → the floor fires again. The floors were presumably added so capped players don't feel frozen, but combined with the ratchet they are an unbounded escalator. Ages 30–31 still grow here via the 22% roll, so the ratchet runs until the age-33 lockout.

**Fix:** either drop the floor to zero at `gap ≈ 0` (accept that capped players stop), or make the `expected < current_ovr` reconciliation clamp OVR down rather than ratchet potential up.

### 🟠 4. The anti-inflation backstop is exempted backwards

`apply_league_ovr_soft_regression_if_needed` (engine.py:2926) is the only global brake, and it **skips every player who grew this season**:

```python
if oa > ob + 0.004:
    continue                      # engine.py:2965 — skip anyone who developed
# and separately skips young players with breakout momentum
```

So it claws back 85+ players who were *stable* and exempts exactly the inflating population it exists to catch. It also only triggers above a league mean of 78.0 — by which point the damage is distributional, not average. This is a symptom-level patch for items 2 and 3, and the exemptions neutralize it.

### 🟠 5. `apply_regression` is a no-op for any healthy player

```python
injury_penalty = min(0.022, 0.004 * len(injury_hist)) + 0.008 * wear_and_tear
decline = float(injury_penalty)
...
if decline <= 0.0:
    return
```

With no injury history and no wear, `decline == 0.0` and the function returns before touching ratings. Morale penalties are added *after* the injury term but a clean 0.55-morale player adds nothing. Verified: a 36-year-old run through 5 seasons of `run_player_progression` loses **0 OVR**.

This means step 3 of the documented pipeline (§1) does nothing for the typical player. All real decline comes from the separate career-lifecycle pass (item 6).

### 🟠 6. Aging runs in a different pass, behind a bare `except: pass`

`run_player_progression` never ages anyone. The only path that does is `_run_career_lifecycle_pass` → `run_career_lifecycle_for_player` → `resolve_authoritative_major_progression_event` → `_career_aging_decline_try_v3`, invoked from `_run_franchise_season_end_progression` (franchise_sim.py:15127):

```python
if getattr(rs, "_run_career_lifecycle_pass", None):
    try:
        out["lifecycle"] = rs._run_career_lifecycle_pass(...)
    except Exception:
        out["lifecycle"] = {"skipped": True}
```

If that pass raises for any reason, **the entire league silently stops aging** and the only trace is `{"skipped": True}` in a payload nobody asserts on. The same file wraps the development pass, production stamping, and the soft-regression guard in identical bare handlers. `regression_check` (engine.py:3098) — the wrapper around the aging roll — has **zero callers** and is dead.

### 🟡 7. The in-season pool mostly evaporates

`apply_in_season_development_pulse` discards any pulse under `0.0020` (development.py:2484). Across 23 ticks/season (every 8 calendar days), a 20yo 70/88 prospect realizes **+1 OVR total** with `_in_season_growth_spent_01 = 0.027` against a nominal pool of 32%. Veterans and at-ceiling players return 0 on every tick. The documented 32%/58% split is really closer to ~5%/58% in practice.

### 🟡 8. Inert inputs and display-only outputs

1. **`progression/aging_curves.py` is dead code.** `get_age_modifier` has no callers anywhere in the repo. It defines a *different* curve (peaks 26/27/29, growth through 27) than the live table in `calculate_season_growth_budget`. Anyone tuning aging will read the wrong file.

2. **`toi_quality`, `role_stability`, `pp_usage`, `pk_usage` are never written.** `_ice_time_modifier` (development.py:1304) reads all four and `_update_career_momentum` reads three, but no code in `SimEngine/` or `backend/` ever sets them. They silently resolve to the 0.5 / 0.0 defaults, which means **roughly half of the intended ice-time model is inert** — only the `role` string and `games_played` bands actually do anything.

3. **`role` is derived from OVR, not from deployment.** All three writers (`role_changes.py:96`, `engine.py:5098` league percentile, `probability_tables.py:200`) compute role from rating or league rank. Line assignments never reach the development engine. So "give the kid top-line minutes" is not currently expressible — only "call him up so he plays 70 games."

4. **`development_fit_score` / `development_fit_label` are display-only.** Computed every season (development.py:379), surfaced in the Organizational Development Review UI, never read back into any growth calculation. "Being Rushed" and "Needs Bigger Role" are narration, not mechanics.

5. **Pool-share comments disagree with the constants** (§2c): 0.58 vs "65–75%", 0.32 vs "~30%".

6. **`healthy_scratches` as a per-player counter is never incremented.** `engine.py:12278` builds a *list* of scratched players for lineup purposes but never stamps a season count onto the player, so the ×0.55 / ×0.78 scratch penalties never fire.

### Suggested order of attack

1. Fix `expected_prod` scaling (item 2) — one formula, unblocks everything downstream.
2. Break the ratchet (item 3) — one conditional.
3. Re-measure. Items 4, 5 and 7 are likely compensating for 2 and 3; tune them only after the two root causes are gone.
4. Replace the bare `except: pass` handlers in `_run_franchise_season_end_progression` with logged failures (item 6) so a silent league-wide aging outage can't recur.
5. Wire or delete the inert inputs (item 8).

A regression test asserting *"a 70-potential player does not finish a 12-season career above ~75 OVR"* would have caught items 1–3 and would keep them caught.

---

## 10. Quick reference

```
Growth per season (displayed OVR, typical NORMAL year)
  18-20, big runway .............. +6 to +8   (SPIKE: up to +9.8)
  21-23, moderate runway ......... +4 to +6
  24-26, some runway ............. +2 to +4
  24-26, at ceiling .............. +1 to +1.6
  27-29 .......................... +1 to +2
  30-32 .......................... rare, roll-gated
  33+ ............................ 0 (hard lockout)

Decline per season (expected value)
  <=24 ........................... 0
  25-29 .......................... ~2.6% x ~0.8 = -0.02
  30-33 .......................... ~15%  x ~1.0 = -0.15
  34-36 .......................... ~23%  x ~1.4 = -0.32
  37+ ............................ ~32%  x ~1.6 = -0.51

Potential movement per season
  drift ......................... +/-1 to +/-2 (breakout / bust / stagnate)
  performance re-evaluation ..... +0.8 to +4.5 (age <=29, overproduction only)
  WJC ........................... -1.8 to +2.5 (draft-eligible prospects)
```
