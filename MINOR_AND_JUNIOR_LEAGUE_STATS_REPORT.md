# Minor-League & Junior Stats Report (AHL / ECHL / CHL / USHL / NCAA / Europe)

Where the stats for the "fake" leagues come from, how they are generated, who receives them, and what they actually change in the game.

**Live source of truth:** `SimEngine/app/sim_engine/generation/prospect_league_scoring.py` (2,733 lines: the whole stat model), with rosters created in `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` and calendar hooks in `backend/services/franchise_sim.py`.

**Method:** everything below was read from code. Section 5 adds *measured* output from a throwaway script (500 generated players per league, seed 1234, season run to 2026-04-15). The script lived in the session scratchpad and is not committed. Real-world comparisons are my general hockey knowledge, not anything in the repo, and are marked "approx."

---

> ## STATUS: fixes applied (see §9)
> This report was written first, as a diagnosis. The problems in §7 and the extra findings below have since been fixed in code. Sections 1-8 describe the model **as it was**; §9 lists what changed, what the model now produces, and what is still open. Where §9 corrects an earlier claim, it says so.

---

## 0. TL;DR (original diagnosis)

1. **None of these leagues are simulated.** There are no AHL/ECHL/CHL games, schedules, standings, or box scores. Every stat line is drawn from a per-player statistical model that runs off the calendar.
2. **One model serves them all.** Junior, NCAA, Europe, AHL and ECHL all go through `prospect_league_scoring.py`, differing only by a league profile (a dict of multipliers and PPG bands).
3. **Stats are a function of ratings, not the other way round.** A player's PPG target is computed from their attributes + OVR + age + style + league. Results are noise around that target.
4. **The stats matter for the draft, and almost nowhere else.** They feed the prospect board (score, consensus potential, stock, scouting confidence, reason codes), the emergent-identity pass, and the AHL roster row display. They do **not** feed player growth, contracts, valuation, or call-up logic (verified, §6).
5. **Problems found** are ranked in §7 (four significant, several minor). The biggest: season goals/assists are re-rolled on every update, so a player's goal total goes *down* in about 6% of monthly updates (measured).

---

## 1. Architecture

```
FRANCHISE START
  bootstrap_full_league_hierarchy()               league_hierarchy_bootstrap.py:424
    team.ahl_roster / team.echl_roster            spawned Players (or real-NHL overflow)
    league.development_leagues[]                  CHL/USHL/NCAA/Europe clubs -> Players
    initialize_prospect_season(p, code)           builds PROJECTED full-season line (junior only, here)

EVERY CALENDAR DAY (throttled)
  _sync_prospect_stats_to_calendar()              franchise_sim.py:11566-11680
    -> advance_all_development_league_stats()     prospect_league_scoring.py:1785
        -> advance_prospect_stats_to_date()       :1564   sim only the GP delta since last update
            -> _simulate_skater_games / _simulate_goalie_games
            -> attach_prospect_production_context (translation risk, adjusted score)
            -> weekly + season "stock" fields

ON DEMAND (API / UI)
  prospect_stats_for_api()                        :2591   flatten actual + projected + analytics
  _ahl_light_season_stats()                       franchise_sim.py:7111  AHL roster rows only

SEASON ROLLOVER
  _roll_development_league_draft_class()            franchise_offseason.py:6152
    age +1, cull undrafted >20, refill to 18 skaters + 2 G, initialize_prospect_season(force=True)
```

| Layer | Job | File |
|---|---|---|
| League profiles | difficulty, multipliers, PPG bands, GP ranges | `prospect_league_scoring.py:19-188` |
| Alias map | `CHL_OHL`, `EU_J_SHL`, `"ECHL PRO"` etc. → profile key | `:190-234`, `normalize_prospect_league_key` |
| Season projection | full-season target line (`_prospect_projected_stats`) | `initialize_prospect_season` `:1400` |
| Calendar pacing | how many GP should exist by date X | `expected_games_for_date` `:591` |
| Game sampling | Poisson-ish points per game, streaks, injuries | `_simulate_skater_games` `:729` |
| Analytics | WAR, xGF%, shots, +/-, QoC (derived, not simulated) | `derive_prospect_analytics` `:2405` |
| Stock | weekly heat and season signal for draft boards | `_compute_weekly_stock_fields` `:1216` |

All state lives as private attributes on the `Player` object: `_prospect_projected_stats`, `_prospect_season_stats`, `_prospect_expected_ppg`, `_prospect_season_year`, `_prospect_last_stat_update_iso`, `_prospect_week_baseline`.

---

## 2. Where the players come from

Stats attach to **generated `Player` objects**, so how those players are made matters. All created in `bootstrap_full_league_hierarchy` via `_spawn_player` (`league_hierarchy_bootstrap.py:246`).

| Pool | Roster shape (F/D/G) | Spawn OVR band (0–1) | Spawn age | Code |
|---|---|---|---|---|
| AHL (per NHL org) | 14 / 8 / 2 | F/D 0.42–0.62, G 0.48–0.68 | 20–28 | `:466, 531` |
| ECHL (per NHL org) | 11 / 5 / 2 | F/D 0.36–0.55, G 0.42–0.60 | 21–30 | `:467, 586` |
| OHL / WHL / QMJHL | 12 / 7 / 3 | 0.32–0.52 | 16–20 (weighted 12/28/36/16/8%) | `:704-706`, `_CHL_AGE_TABLE :118` |
| USHL | 12 / 6 / 3 | 0.30–0.48 | 16–19 | `:707` |
| NCAA | 13 / 7 / 3 | 0.34–0.55 | 18–24 | `:708` |
| Europe junior (10 leagues) | 5 / 3 / 1 | 0.30–0.50 | 16–20 | `:715-728` |

Notes:
- **Real-NHL leagues:** if a real roster import is active, AHL/ECHL rosters keep real NHL overflow players (`preserved_ahl` / `preserved_echl`) and can pull real AHL/prospect entries from the dynasty ratings registry (`:495-581`). Those players get the same synthetic stat treatment.
- **Junior skaters** get a "youth baseline" rating shape (`spawn_youth_baseline_profile`), then a 6% bust / 4.2% steal roll (`pipeline_bust`, `pipeline_steal`, `:376-387`). Draft-class star power is added afterwards by `_shape_draft_class_pipeline` (`:993`).
- **Refill each year:** the rollover tops every junior club back to ~18 skaters + 2 goalies with 17–18 year-olds (`franchise_offseason.py:6253-6320`).

---

## 3. How the stats are generated

### 3a. League profiles

Each profile carries: `difficulty`, `scoring_multiplier`, three PPG bands (`average` / `star` / `elite`), `overager_bonus`, `defensive_translation_penalty`, and a `gp_range`. Selected rows:

| League | difficulty | scoring_mult | avg PPG band | star band | elite band | GP range |
|---|---|---|---|---|---|---|
| QMJHL | 0.58 | 1.38 | 0.68–1.08 | 1.28–1.80 | 1.70–2.25 | 58–68 |
| OHL | 0.62 | 1.37 | 0.65–1.05 | 1.25–1.75 | 1.65–2.20 | 58–68 |
| WHL | 0.66 | 1.30 | 0.60–0.98 | 1.18–1.68 | 1.55–2.10 | 58–68 |
| USHL | 0.72 | 1.15 | 0.45–0.78 | 0.95–1.35 | 1.20–1.65 | 52–62 |
| NCAA | 0.82 | 0.90 | 0.35–0.62 | 0.75–1.10 | 0.95–1.35 | 32–42 |
| **ECHL** | 0.72 | 0.98 | 0.32–0.58 | 0.62–0.95 | 0.85–1.25 | 55–68 |
| **AHL** | 0.86 | 0.85 | 0.28–0.52 | 0.58–0.88 | 0.75–1.15 | 58–72 |
| SHL | 0.92 | 0.70 | 0.18–0.38 | 0.42–0.72 | 0.55–0.95 | 40–52 |
| EU junior | 0.84 | 0.82 | 0.24–0.45 | 0.50–0.82 | 0.70–1.10 | 40–52 |

A second, separate multiplier table (`_league_junior_stat_multiplier`, `:697`) is applied on top: QMJHL 1.24, OHL 1.22, WHL 1.18, USHL 1.08, NCAA 0.94, ECHL 0.92, AHL 0.88, SHL 0.72. See §7 (issue 3).

Unknown league codes fall back to `JUNIOR` (`:315`).

### 3b. The PPG target (skaters), `calculate_prospect_ppg_scale` (`:1948`)

1. **Offensive composite** from four latent skills (`_prospect_event_skills :1875`), each a ratings average blended 86–90% with current OVR:
   `offensive = 0.18·volume + 0.24·finishing + 0.32·playmaking + 0.26·process`.
   Inputs are *current ratings and OVR only*. Hidden potential is not used here.
2. **Tier pick** → base PPG drawn uniformly from the league band:
   - `offensive ≥ 0.78` → elite band; `≥ 0.66` → star; `≥ 0.50` → between avg-high and star-low; else average band.
3. `× U(0.94, 1.10)` noise.
4. `× league_mult × (1 + (0.62 − difficulty) × 0.06)` (the second, table-based league multiplier).
5. **Position/style:** defensemen × `defensive_translation_penalty`, then offensive-D ×1.06–1.18, defensive-D ×0.42–0.58, other D ×0.72–0.88; grinders ×0.52–0.72; scorers ×`0.92 + offensive·0.22`.
6. **Age:** ≥20 ×`overager_bonus`×U(1.00,1.05); 19 ×U(0.96,1.04); ≤17 ×U(0.92,0.98).
7. **Risk flags:** boom/bust ×U(0.88,1.12); character concerns ×U(0.86,1.04).
8. **Role multiplier** (`_prospect_role_multiplier :667`, range 0.55–1.12): skill, age, +8% for CHL-family, −14% for Euro pro leagues, +8% PP1, +6% top line. (PP1/top-line inputs are never set on these players; see §7 issue 6.)
9. **Clamp** to `[floor, ceiling]`: floor 0.32 F / 0.14–0.18 D; ceiling = `elite_hi × 1.02`, raised to 1.85/2.10/2.40 in the CHL family by offensive tier; D ceilings shrink by style.

The result is stored as `_prospect_expected_ppg`. It is re-derived on each update and **only ever ratchets up** (replaced when `fresh > old × 1.04`, `:1691`), so a player whose ratings fall keeps his old pace.

### 3c. Projection vs. actual

- **Projected line** (`initialize_prospect_season :1400`): GP drawn from the league `gp_range` (age ≤17 ×0.82–0.92, ≥20 ×0.96–1.02, hard range 28–72), points = `round(GP × ppg)` capped by a role max (CHL F 2.68, offensive D 1.62, defensive D 0.62), then a G/A split. Marked `stat_source = "season_projection"`. The UI shows this before GP > 0 (`stats_mode = "projected"`).
- **Actual line** starts at zero and grows only through `advance_prospect_stats_to_date` (`:1564`).

### 3d. Calendar pacing

`expected_games_for_date` uses a month→fraction-of-season curve (`_BASE_MONTH_FRAC :237`: Sep 5%, Oct 16%, Nov 28%, Dec 42%, Jan 56%, Feb 72%, Mar 86%, Apr 97%), shifted per league (NCAA −0.08, USHL −0.02, SHL +0.06, AHL/ECHL/CHL 0). Only the GP *delta* since the last update is simulated. Pacing is linear interpolation within a month.

### 3e. Per-game sampling, `_simulate_skater_games` (`:729`)

- Points per game drawn from a Poisson-style sampler (`_sample_prospect_game_points :638`) with mean = target PPG, plus:
  - **Streak reversion:** if the last 3 games ran ≥1.35× target, the next expectation is trimmed (×0.88–1.02); if ≤0.55×, boosted (×0.98–1.12).
  - **Hot games** (elite offense ≥0.80: 3.5%/game; boom-bust: 1.2%) multiply the mean ×1.14–1.38.
  - **Character-concern players** get a scoreless game 4% of the time.
  - **Overdispersion** scales with `_volatility_factor` (0.10–0.42).
- **Bulk shortcut:** when >16 games are owed (offseason catch-up, bulk sim), it samples 12 chunks instead of per-game, preserving the expected total.
- **Injuries** (`_prospect_injury_games :1521`): 1.4%/update base chance, misses 1–8 GP, flagged `prospect_injured`.
- **PIM:** 0–4 per game for grinders/power forwards, 0–2 at 35% for others.
- **Goals/assists** are *not* accumulated. They are split from total points after each update (`_split_goals_assists :2046`) using a style-specific goal share (sniper 48–62%, playmaker 22–38%, offensive D 18–32%…), with a 14% chance for scorer styles to swing to a 15–70% share. See §7 issue 1.

### 3f. Goalies

- **Projection** (`generate_goalie_prospect_line :2077`): `save_pct = 0.870 + off_talent × 0.045 + U(−0.025, 0.020)`, clamped 0.845–0.945; `gaa = 3.35 − off_talent × 1.05 − (1 − difficulty) × 0.35 ± 0.35`, clamped 1.85–3.80; wins from 42–58% of GP.
- **In-season** (`_simulate_goalie_games :846`): each chunk perturbs SV% and GAA around the projection by `± volatility`, samples wins from the projected win rate, and rolls a shutout when a chunk hits SV ≥ .940 and GAA ≤ 2.10. Season SV%/GAA are the average of the last 12 samples.
- SV% and GAA are drawn **independently** of each other. There are no shots-against, goals-against, or team context.

### 3g. Derived analytics, `derive_prospect_analytics` (`:2405`)

WAR, offensive/defensive WAR, xGF%, CF%, FF%, shot rate, shooting %, +/-, primary points, defensive impact, QoC/QoT and goalie GSAx/quality starts are **formulas over the stat line + talent scores**, gated at GP ≥ 5. They are not simulated events. Shots are estimated from a rate model (`_estimate_prospect_shots`), +/- from a talent/production formula, and per-prospect jitter is a hash of the player's name.

---

## 4. Allocation: who gets a line, and where it shows up

| Group | Stat line generated? | Advance path | Shown anywhere? |
|---|---|---|---|
| Junior / NCAA / Euro clubs (age ≤ 20) | Yes, at bootstrap | Daily bulk sync (`franchise_sim.py:11620-11629`) | Draft board, prospect pages, org prospect rows |
| Junior 21+ (NCAA up to 24) | Lazily, on API/board access | `ensure_prospect_season_stats(calendar_iso)` | Draft board (NCAA max age 24) |
| **AHL** roster, age ≤ 23 | Yes, daily sync as `"AHL"` | Daily bulk sync (`:11631-11642`) | AHL roster rows (`_ahl_light_season_stats`) |
| **AHL** roster, age 24+ | Lazily, when a roster row is built | `prospect_stats_for_api(p, "AHL")` | AHL roster rows |
| **ECHL** roster, age ≤ 23 | Yes, daily sync as `"ECHL"` | Daily bulk sync (`:11643-11653`) | **No display path found** (see §7 issue 5) |
| **ECHL** roster, age 24+ | Never | n/a | No |
| Anyone who plays real NHL games | Real stats in `session.player_season_stats` | Game sim | NHL rows; AHL rows preserve the NHL line as a career split (`franchise_sim.py:7025-7033`) |

Details:
- **AHL display** (`franchise_sim.py:7111-7198`): calls `prospect_stats_for_api(p, "AHL")`, and flattens to GP, PIM, G, A, PTS, +/-, WAR (skaters) or W/L/OTL/SV%/GAA/SO (goalies). Tagged `is_ahl_synthetic: True`. The docstring states the reason: "AHL games are not simulated player-by-player."
- **ECHL rows** only get `league = "ECHL"` set; the stat line is taken from NHL stats if any exist, otherwise blank (`:7045-7049`).
- **Draft board:** for every draft-age prospect the board builds a row from `ensure_prospect_season_stats` (cheap), then only the composed top ~320 get full `prospect_stats_for_api` analytics (`franchise_sim.py:8182-8336`).
- **Rollover:** `advance_prospect_stats_to_date` detects a new `season_year` and re-initializes; `_roll_development_league_draft_class` does it explicitly with `force=True` for the development leagues. **Last season's line is discarded.** Nothing archives junior/AHL/ECHL seasons to a player's history (grep of `_prospect_season_stats` finds no archiver; the AHL "career seasons" merge is a display-only fold, `franchise_sim.py:7318`). The one consumer of the old line is `progress_season_body_and_identity`, which runs *before* the reset (§6).
- **Randomness:** the bulk path passes the sim RNG; the lazy path builds a `Random` from the player's `rng_seed`.

---

## 5. Measured output

500 players per league, generated by the real bootstrap `_spawn_player` bands, run to 2026-04-15 (end of season). Forwards and defense filtered to GP ≥ 20.

| League | Fwd PPG mean | Fwd p50 | Fwd p90 | Fwd max | Fwd max pts | D PPG mean | Goalie SV% (mean, range) | Goalie GP mean |
|---|---|---|---|---|---|---|---|---|
| OHL | 1.12 | 1.08 | 1.41 | 1.87 | 120 | 0.78 | .888 (.863–.911) | 57.8 |
| QMJHL | 1.20 | 1.17 | 1.57 | 2.49 | 147 | 0.85 | .888 (.863–.910) | 58.5 |
| USHL | 0.61 | 0.61 | 0.80 | 1.00 | 55 | 0.47 | .888 (.855–.909) | 51.6 |
| NCAA | 0.51 | 0.48 | 0.74 | 1.10 | 37 | 0.35 | .890 (.867–.913) | 33.1 |
| SHL junior | 0.33 | 0.33 | 0.44 | 0.60 | 25 | 0.16 | .887 (.867–.918) | 44.0 |
| **AHL** | 0.51 | 0.52 | 0.66 | 0.83 | 56 | 0.38 | .898 (.872–.926) | 63.1 |
| **ECHL** | 0.56 | 0.57 | 0.72 | 0.92 | 58 | 0.44 | .894 (.871–.918) | 59.2 |

Reading it against approximate real-world levels (my knowledge, not the repo's):
- **CHL is heavily inflated across the whole roster, not just the top.** A *median* OHL forward scores ~1.08 PPG (~63 pts). Approx. real median CHL forward is closer to 0.5–0.7 PPG. Defensemen at 0.78 PPG are roughly double real. The top end (120–147 pts) is plausible; the middle is far too high. Cause: the base band draw starts at `avg_lo × 0.92` and is multiplied by ~1.2–1.5 of stacked league factors (§7 issue 3).
- **AHL/ECHL are flat and truncated at the top.** Best AHL forward in 500 players is 56 pts / 0.83 PPG; approx. real AHL leaders reach ~1.0+ PPG (75–90 pts). There is little separation between good and bad AHL players.
- **ECHL out-scores AHL** (0.56 vs 0.51 median). It comes from ECHL's `scoring_multiplier` 0.98 vs AHL 0.85 and its own PPG bands being defined relative to a lower-talent pool, so a given rating set scores *more* in ECHL. Plausible for real ECHL, but note that PPG is never normalized against the level of competition.
- **Goalie workload is unrealistic.** Mean goalie GP is 58 in CHL and 63 in AHL. With 3 (CHL) or 2 (AHL) goalies per club, each of them plays about a full season, so a team's goalies together play 2–3× the games the team plays.
- **Goals/assists instability (measured):** 300 OHL forwards updated monthly Oct→Apr (2,100 updates): **goals decreased in 128 updates (6.1%)**, assists in 78 (3.7%); worst single drop was 11 goals.

---

## 6. What the stats affect (and what they don't)

### They DO affect

| Consumer | How | Where |
|---|---|---|
| **Draft rank score** | `score = ovr·0.48 + pot·0.36 + prod_adj·5.5 + def_bonus + …` where `prod_adj` is role-adjusted PPG | `draft_ranking_logic.py:1280-1302`, `compute_role_adjusted_production :1069` |
| **Consensus (public) potential** | `production_ceiling = ovr + min(16, prod·9.5 + ppg·4)`; blended 58/42 with tools; +2.5 for prod ≥1.35 | `draft_ranking_logic.py:1199-1277` |
| **Reason codes** | `elite_junior_production`, `production_concern`, `offensive_role_underproduction`, `low_scoring_toolsy_defenseman` | `:1148` |
| **Weekly / season stock** ("Rising", "Crashing", "Volatile", "Scout Split") | production vs projection + analytics, sample-weighted, volatility-amplified | `prospect_league_scoring.py:1216-1397` |
| **Scouting confidence** | rises with GP: `38 + 1.15·GP` up to 92 | `franchise_sim.py:8229-8236` |
| **Translation risk / context labels** | age, character, league, PPG → Low/Medium/High; "Junior inflated", "Overager scoring" | `:2162-2200` |
| **Leaderboard nudge** | if the top CHL forward is below the elite floor, bump him into the elite band | `normalize_league_leader_board :2696` |
| **Emergent identity** | at rollover, junior players' last-season G/A/PPG re-infer playstyle and archetype (`refresh_player_identity`, falls back to `_prospect_season_stats` when GP = 0) | `prospect_identity.py:59-79, 298`, `franchise_offseason.py:6224` |
| **AHL roster rows** | the entire visible AHL stat line | `franchise_sim.py:7111` |

Concrete magnitude of the league inflation on the draft market (computed from the formula at equal OVR): a median OHL forward (1.08 PPG, adjusted ≈ 0.87) gets ≈ +12.6 production lift on the consensus ceiling; a median NCAA forward (0.48 PPG, adjusted ≈ 0.36) gets ≈ +5.3. After the 0.58 blend that is roughly **+4 points of public potential** from league environment alone, so CHL kids are systematically read as higher-ceiling than NCAA/Euro kids of the same rating.

### They DO NOT affect (verified)

- **Player growth.** `_dev_stamp_season_production` (`franchise_offseason.py:2695`) builds the growth "production score" from `session.player_season_stats`, which has no entry for AHL/ECHL/junior players, so it returns early at `gp <= 0` (`:2699`). `develop_unsigned_prospect` reads `player.ppg` / `player.production_score` / `ice_time_quality` (`unsigned_prospect_development.py:83-85`); none of those are ever set on these Players, so it runs on defaults (`production 0.5` → `prod_mod ≈ 0.98`, `ice 0.55`). The fake-league PPG never reaches `calculate_season_growth_budget`.
- **Contracts, asking prices, trade value, ELC/re-sign decisions.** No consumer of `_prospect_season_stats` outside the draft/identity/AHL-display paths above.
- **Call-up / send-down logic and "AHL breakout" text.** The report line "AHL breakout has improved NHL readiness" (`development.py:809`) is driven by the RNG `dev_phase == "SPIKE"`, not by the AHL stat line.
- **Standings, team records, playoffs, awards** for these leagues. None exist.

*(Caveat: I verified the `Player`-object path. The separate `entities/prospect.py` `Prospect` class has its own `context.ice_time_quality`, which I did not trace to any stat consumer.)*

### Development happens independently

Growth for these players comes from other systems: `_depth_pool_progression_tick` (every 5 calendar days, a random 72 from the pool incl. AHL/ECHL/junior/FA, `franchise_sim.py:11683`), `tick_extra_league_development` (a tiny per-day random +1 attribute bump: 0.11% junior, 0.065% AHL/ECHL, `league_hierarchy_bootstrap.py:1350`), `run_unsigned_prospect_development_pass` for drafted-unsigned kids, and the offseason `_run_user_org_depth_progression`. So a player's stat line and his development are two separate outputs of the same ratings.

---

## 7. Findings (ranked)

### 1. Goals and assists are re-rolled every update: goal totals go down. *(bug, measured)*
`_recalc_skater_line_from_totals` (`:619`) calls `_split_goals_assists` with fresh RNG after every advance, so G/A are re-derived from *cumulative points* each time instead of accumulating. A player with 20 G / 25 A in November can show 16 G / 35 A in December. Measured: goals fell in 6.1% of monthly updates, up to −11. The 14% "boom-bust split" (`_is_boom_bust_style :2073`) can swing goal share 15–70% between two reads. Also affects identity inference at rollover, which reads goals-vs-assists (`prospect_identity.py:223-226`).
**Fix direction:** split the *delta* points each update, add to running G/A; or store a per-player fixed goal share drawn once at `initialize_prospect_season`.

### 2. No team or league conservation.
Every player is sampled independently. Goalies each play ~a full schedule (58–63 GP mean), all 14 AHL forwards play ~62 GP, and team goals for/against don't exist, so there are no standings, no consistent SV%↔GAA, and no way to check that a league's goals add up. Any future AHL standings, leaders page, or "team X's goalie tandem" would need team-level allocation (GP splits, ice time, PP units) first.

### 3. League scaling is applied twice, and the junior baseline is too hot.
The profile's PPG bands are already league-specific, then `_league_junior_stat_multiplier` (1.24 QMJHL … 0.88 AHL) and `(1 + (0.62 − difficulty) × 0.06)` are multiplied on top. Combined with a band floor at `avg_lo × 0.92`, the whole CHL roster inflates (§5). AHL/ECHL are the opposite: compressed with no real top end. Suggest choosing *one* scaling mechanism and re-fitting bands to target medians (e.g., CHL median F ≈ 0.6–0.7, AHL top ≈ 1.0).

### 4. Hidden truth leaks into analytics and stock.
`calculate_prospect_ppg_scale` correctly uses only ratings + OVR. But `_offensive_talent_score` (`:452`) blends `draft_value_range`/`potential` (0.42 weight), `dev_potential`, `pipeline_tier`, `is_transcendent`, `pipeline_steal`, and `dev_type`. That score drives WAR, xGF%, `expected_ppg` "overproduction" (`:2543`, commented "no true-potential leak"), goalie SV%/GAA (`:2079`), and `_analytics_process_score`, which adds `+0.12` for `pipeline_steal` and `−0.18` for `pipeline_bust` straight into public stock (`:1009-1012`). Visible analytics and stock movement therefore correlate with hidden potential and the bust/gem flags. If "public info should not reveal truth" is the intent (as stated in `compute_consensus_potential_evaluation`), these paths break it.

### 5. ECHL stats are computed but never shown; ECHL 24+ get nothing.
The daily sync builds `"ECHL"` lines for age ≤ 23, and the career-merge helper even has an ECHL branch (`franchise_sim.py:7346`), but the roster-row code only calls `_ahl_light_season_stats` for AHL. ECHL rows show a blank line unless the player has NHL games. This is CPU spent on data nobody reads, and an inconsistent player-card experience between AHL and ECHL.

### 6. Dormant inputs.
`pp1_usage`, `pp_role`, `line_role` (`_prospect_role_multiplier :690-693`) and `ice_time_quality`/`line_role_score` (`unsigned_prospect_development.py:83`) are read but never assigned on `Player` objects, so role/usage has no effect on lines or development. Ice time isn't modeled at all.

### 7. Identity feedback loop.
Playstyle → goal share (`_split_goals_assists`) → season G/A → `infer_playstyle_from_identity` (goals ≥ 1.2×assists ⇒ "sniper"). A player's style is largely echoed back to himself. Combined with issue 1 this can also make styles flip on noise.

### 8. Smaller items
- **Pace ratchet:** `_prospect_expected_ppg` only increases (`:1691`); a declining player keeps his old pace within the season.
- **Stale difficulty table in draft logic:** `_ppg_to_production_score` (`draft_ranking_logic.py:1052`) hard-codes 0.62 / 0.82 / 0.88 and knows nothing of USHL, AHL, or ECHL (they fall to the 0.62 CHL default). The scoring module's own `difficulty` values (e.g., USHL 0.72) are ignored there.
- **Age filter mismatch:** bulk sync includes juniors ≤ 20 only, so NCAA 21–24 only advance lazily; AHL/ECHL only ≤ 23.
- **One-time retune hack:** `_maybe_retune_underproduced_prospect_line` (`:1464`) jumps under-scoring lines to 96% of target, capped at +0.95 P/GP, which is a catch-up for an old model version and leaves a visible discontinuity in mid-season lines.
- **Goalie stat gates:** goalie stock and analytics need GP ≥ 5 but projected lines display at GP 0 as "projected", so goalie stock is `Holding` until early season.

---

## 8. Suggested next steps

1. **Fix issue 1 first.** It's a small, contained change and it's user-visible today (goals going backwards).
2. Decide the intent for stats → development. Either keep the fake stats as display/draft-only (then delete the misleading `ppg`/`production_score` reads in `develop_unsigned_prospect`) or stamp `production_score` from `_prospect_season_stats` so AHL/junior performance nudges growth.
3. Re-fit league medians (issue 3) with a distribution test like the one in §5, and add it to `backend/tests/test_prospect_league_scoring.py` (currently asserts only that top CHL PPG ≥ 1.65 and CHL > NCAA > SHL by max).
4. Pick the ECHL policy: surface it like AHL, or stop computing it.
5. If public/hidden separation matters, give analytics and stock a "public talent" score built from ratings/OVR only (as PPG already is).
6. Only if AHL/ECHL standings or team pages are planned: introduce team-level GP/goal allocation so lines sum to something.

---

## Appendix: file map

| Concern | File:line |
|---|---|
| League profiles + aliases | `SimEngine/app/sim_engine/generation/prospect_league_scoring.py:19-234` |
| PPG target | `…/prospect_league_scoring.py:1948` |
| Season init / calendar advance | `…/prospect_league_scoring.py:1400, 1564, 1785` |
| Per-game sampling / goalies | `…/prospect_league_scoring.py:729, 846` |
| G/A split | `…/prospect_league_scoring.py:2046` (called from `:619`) |
| Analytics | `…/prospect_league_scoring.py:2405` |
| API flatten | `…/prospect_league_scoring.py:2591` |
| Roster creation | `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py:424, 246` |
| Daily sync | `backend/services/franchise_sim.py:11447, 11600-11680` (+ `franchise/advance.py:970`) |
| AHL display | `backend/services/franchise_sim.py:7111, 7013-7058` |
| Draft board consumption | `backend/services/franchise_sim.py:8017-8336`, `backend/services/draft_ranking_logic.py:1052-1302` |
| Season rollover | `backend/services/franchise_offseason.py:~6150-6362` |
| Growth stamp (NHL only, in effect) | `backend/services/franchise_offseason.py:2695` |
| Unsigned prospect dev | `backend/services/unsigned_prospect_development.py:33` |
| Identity from stats | `SimEngine/app/sim_engine/generation/prospect_identity.py:59, 187, 298` |


---

## 9. Fix status

### 9a. Corrections to this report's own findings
Measuring and fixing turned up three things the diagnosis above got wrong or missed:

1. **"Stats are a function of ratings" was only half true.** PPG was chosen by three absolute composite thresholds (0.66 / 0.78 for star / elite). I measured the composite of every generated player: **no one reached 0.66**, and 90-100% of every league fell into the single "average" band, where PPG was a uniform random draw. Within a league, scoring barely tracked ability at all. (§3b describes the code correctly; my TL;DR item 3 overstated its effect.)
2. **Update cadence changed the results** (found after the report was first written): the scoring target was redrawn on every update and kept if 4%+ higher, and injuries rolled once per update. Same seed, daily updates vs. one update: forward PPG 1.30 vs 1.12, injuries seen 55% vs 3%. My original §5 numbers used one update per player, so **the live game was even hotter than §5 shows**.
3. **Injuries never actually cost games.** A missed game was left in the "games owed" pool and simulated at the next update, and the injury counter was set to the full length *and* consumed again, so injuries only delayed games. (Separately, AHL/ECHL talent is compressed by the pool OVR floors in `_spawn_player`: AHL composite spans 0.585-0.613. That is spawn design and is **not** changed here.)

### 9b. What changed (all in `prospect_league_scoring.py` unless noted)

| # | Problem | Fix |
|---|---|---|
| 1 | Goals/assists re-rolled each update; goals fell in 6.1% of updates | G/A now **accumulate**: only newly scored points are split, using a per-player season goal share drawn once. `_recalc_skater_line_from_totals` no longer re-splits. Measured: 0 decreases in 2,100 updates. |
| 2 | Result depended on how often the calendar advanced | One block sampler for daily and bulk advances (binomial hot/scoreless/tail games, mean-preserving overdispersion scaled by 1/sqrt(n), real Poisson for large totals). PPG target is **deterministic per (player, season, league)** from a stable seeded stream: no redraw, no ratchet. Injury hazard is **per game**. Measured mean forward PPG at update intervals of 200 / 30 / 7 / 1 days: 0.807 / 0.800 / 0.807 / 0.814 (was 1.12 / 1.19 / 1.26 / 1.30). |
| 3 | Injuries only delayed games | New `gp_missed` counter; owed games = expected - played - missed. Injury length carried over correctly (no double count). |
| 4 | Stats belonged to the player, not the stint; call-ups and demotions back-filled or double-counted games | New **stints** (`begin/archive/end_prospect_stint`): one league + team + continuous stretch. A new league, a return after a call-up, or a missed sync opens a new stint with **no back-fill** (`start_frac`). Bulk sync reads AHL/ECHL membership from the **live roster lists** every run instead of a cached row list (which went stale on the first call-up). |
| 5 | AHL/ECHL vets (24+) got nothing; ECHL never displayed | Sync covers every AHL/ECHL player. `_ahl_light_season_stats` now takes a league code, so ECHL roster rows get a line (`franchise_sim.py`). Career-merge rows carry `is_synthetic`. |
| 6 | No history; every season/stint discarded | Closed stints and rolled-over seasons are filed to `player.prospect_stat_history` (cap 24), exposed as `stat_history` from `prospect_stats_for_api` and `prospect_stat_history` on roster rows. |
| 7 | League inflation; stars/elite tiers unreachable; double league scaling | Continuous PPG curve through the league's own average/star/elite bands, keyed on **talent relative to the league's typical roster** (`_LEAGUE_TALENT_CENTER`). Removed the second multiplier table and the role-multiplier league bonus. Added a per-season **usage** draw (deployment) since talent alone cannot separate a top-line AHL scorer from a fourth-liner. Defense factors retuned. Removed the one-shot `_maybe_retune_underproduced_prospect_line` catch-up hack. |
| 8 | Hidden potential leaked into analytics/stock/goalie lines | `_offensive_talent_score` / `_defensive_talent_score` use current OVR + ratings + style only. Removed the `pipeline_steal/bust` nudges from `_analytics_process_score`. QoC's rank signal now uses the public draft rank or current ability, not potential (found by a new test). `_draft_mid` is now marked draft-only. |
| 9 | Goalie SV% independent of GAA, "season" SV% = mean of last 12 draws, every goalie plays a full season | Goalies are simulated with **shots against and goals against**; SV% and GAA are derived from the totals, so they agree with each other and with the season. Quality comes from goalie ratings (`g_*`), not skater tools. GP is a **share** of the schedule (starter vs. backup). W/L/OTL sum to GP. |
| 10 | `_prospect_event_skills` OVR blend silently dead | `Player.ovr` is a method; `float(method)` failed and pinned it to 0.50. Now uses `_player_ovr_0_1`. |
| 11 | Projection and live target used different random draws for bootstrap-spawned players | Draw year is pinned at season init (`_prospect_ppg_seed_year`). |
| 12 | NCAA players deleted at 21 though eligible to 24 | Rollover cull and stat-sync age cap use `development_league_stat_max_age` (NCAA 24, others 20) (`franchise_offseason.py`, `franchise_sim.py`). |
| 13 | Draft market read raw league scoring as evidence; hard-coded difficulty table | Consensus-potential lift is now `neutral (8) + (lift earned - lift expected for this ovr/role/league)`, so a player matching expectation gets the same lift in any league (`draft_ranking_logic.py`). Difficulty comes from the scoring profiles. NCAA expectation retuned 0.56 -> 0.66 to match the re-fit output. |

### 9c. What the model produces now
500 generated players per league (same generation as §5), season run to 2026-04-15:

| League | Fwd PPG mean | p50 | p90 | best fwd (pts) | D PPG mean | Goalie SV% mean | Goalie GAA | Goalie GP mean |
|---|---|---|---|---|---|---|---|---|
| OHL | 0.82 | 0.76 | 1.30 | 2.02 (124) | 0.48 | .900 | 2.98 | 25.6 |
| QMJHL | 0.83 | 0.79 | 1.24 | 2.05 (133) | 0.47 | .899 | 3.05 | 26.8 |
| USHL | 0.56 | 0.52 | 0.89 | 1.35 (65) | 0.35 | .902 | 2.87 | 22.6 |
| NCAA | 0.54 | 0.52 | 0.85 | 1.38 (46) | 0.29 | .912 | 2.43 | 15.4 |
| SHL junior | 0.31 | 0.30 | 0.48 | 0.78 (35) | 0.19 | .913 | 2.30 | 19.7 |
| **AHL** | 0.41 | 0.40 | 0.64 | 1.16 (81) | 0.24 | .907 | 2.69 | 28.5 |
| **ECHL** | 0.43 | 0.41 | 0.66 | 1.08 (52) | 0.27 | .908 | 2.75 | 22.8 |

Compared with §5: the OHL median forward fell from 1.08 to 0.76 PPG, defensemen from 0.78 to 0.48, goalies play a share of the schedule instead of a full one, and the AHL now has a real top end (81 pts vs 56).

### 9d. Tests
`backend/tests/test_prospect_league_scoring.py`: 7 -> 21 tests. New regressions cover determinism/no ratchet, cadence independence, monotonic G/A, injuries costing games, stint archive and no back-fill (direct and through the bulk sync with live rosters), season archive, any-age AHL/ECHL lines, goalie consistency, hidden-potential invariance (skater analytics **and** goalie lines), talent -> scoring within a league, NCAA age cap, and projection/target draw pinning. One existing boundary assertion (`sep["gp"] <= 6`) was relaxed to `<= 8`: it sat exactly on the rounding edge (Sep 15 is ~10% of a 58-68 game season) and only passed before because the RNG stream happened to be fixed.

### 9e. Still open (not changed)
- **Public-vs-true ratings leak (issue 2).** PPG is computed from exact ratings while the draft board shows an OVR *band*; a player who reverse-engineers the formula could narrow the band. Needs product intent before changing (fuzzing stats vs. accepting it).
- **No team/league conservation.** Players are still sampled independently, so no AHL/ECHL standings, leaders page or team goalie tandems are possible yet. Goalie GP is now a *share* on average, but a team's goalies are not coordinated to sum to the schedule.
- **Draft score still counts production once more** (`prod_adj * 5.5`), on top of the now-expectation-relative consensus lift. The bias between leagues is removed; the double count of *over*-production remains.
- **AHL/ECHL talent compression** from the pool OVR floors (spawn design).
- **Frontend** does not yet render `prospect_stat_history` or the `is_synthetic` flag; the data is in the API payloads.
- **Existing saves:** legacy lines are adopted into a stint starting at season start. Anything already in `_prospect_season_stats` keeps its old G/A until the next season rollover.
- Four `test_draft_prospect_profile.py` tests fail on a clean `HEAD` as well and are unrelated.
