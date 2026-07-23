# NHL Franchise Mode Startup Performance Repair Report

## Executive Summary

- Old franchise creation time (HTTP): `37,747 ms`
- New franchise creation time (HTTP avg, 3 runs): `29,556.64 ms`
- Improvement: `21.70%`
- Old direct Python startup: `122,237 ms`
- New direct Python startup (avg, 3 runs): `20,541.12 ms`
- Improvement: `83.19%`
- Former primary bottleneck: schedule smoothing/finalization + repeated schedule scan helpers + startup OVR enforcement loops.
- New primary bottleneck: hierarchy bootstrap (`bootstrap_full_league_hierarchy` / `_spawn_player` / `Player.__init__`), plus prospect stats sync.
- Startup acceptability: substantially improved; still heavy but no longer dominated by pathological schedule helper volume.

## Files Modified

| File | Reason | Major algorithmic change |
|---|---|---|
| `backend/services/franchise_sim.py` | Remove schedule smoothing/finalization repeated scans | Added per-slot team-id memoization, team-day indexed cadence checks, indexed placement checks, and incremental index updates during conflict repair |
| `SimEngine/app/sim_engine/entities/player.py` | Reduce repeated OVR/clamp overhead in bootstrap | Reworked `_avg` to avoid redundant clamping path and tightened OVR boosting iterations for minimum OVR enforcement |

## Startup Call Graph

Measured from cProfile (`start_franchise`, seed `424242`). Inclusive/exclusive timings are shown separately.

| Parent | Child | Calls | Exclusive Time (ms) | Inclusive Time (ms) |
|---|---|---:|---:|---:|
| `start_franchise` | `bootstrap_full_league_hierarchy` | 1 | 14.00 | 23,809.00 |
| `start_franchise` | `generate_regular_season_schedule` | 1 | 1.00 | 4,234.00 |
| `start_franchise` | `_finalize_schedule_after_generation` | 1 | 1.00 | 8,506.00 |
| `_finalize_schedule_after_generation` | `_smooth_league_schedule` | 1 | 28.00 | 8,126.00 |
| `start_franchise` | `_ensure_league_roster_contracts` | 1 | 0.00 | 5,433.00 |
| `start_franchise` | `_sync_prospect_stats_to_calendar` | 1 | 0.00 | 9,345.00 |
| Post-start | `build_state_payload(include_heavy=False)` | 1 | NOT MEASURED | 227.81 |

Nested timing note: `_smooth_league_schedule` is nested inside `_finalize_schedule_after_generation`, so its time is not additive.

## Schedule Algorithm Before

- Cadence and placement checks repeatedly scanned `by_day` and reparsed slot team IDs.
- `_would_create_bad_cadence_for_slot` rebuilt team-day data by schedule scanning for each candidate.
- Hot helper call counts were pathological:
  - `_slot_has_team`: `7,736,292`
  - `_team_ids_for_slot`: `7,774,158`
  - `_safe_slot_team_id`: `15,982,488`
- Effective behavior was close to repeated schedule-wide scans inside nested candidate loops.

## Schedule Algorithm After

- Added `_slot_team_ids_cached(slot)` and routed `_slot_key`, `_team_ids_for_slot`, `_slot_has_team` through it.
- `_would_create_bad_cadence_for_slot` now accepts `team_days_by_team` fast-path.
- `_can_place_slot_on_day` now accepts `team_days_by_team` and `team_days_set_by_team` for O(1) occupancy checks.
- `_repair_regular_day_conflicts` builds `team_days` indexes once and updates only affected teams after each move.
- Smoothing now passes prebuilt team-day indexes into placement checks.

## Schedule Benchmark

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| finalize total (`_finalize_schedule_after_generation`) | 37,251 ms | 8,506.00 ms | 77.17% |
| smooth total (`_smooth_league_schedule`) | 36,826 ms | 8,126.17 ms | 77.93% |
| placement checks total (`_can_place_slot_on_day`) | 27,015 ms | 495.50 ms | 98.17% |
| cadence checks total (`_would_create_bad_cadence_for_slot`) | 26,670 ms | 395.79 ms | 98.52% |
| `_slot_has_team` calls | 7,736,292 | 4,978 | 99.94% fewer |
| `_team_ids_for_slot` calls | 7,774,158 | 37,790 | 99.51% fewer |
| `_safe_slot_team_id` calls | 15,982,488 | 112,074 | 99.30% fewer |
| candidate checks (proxy via placement checks) | 11,533 | 6,969 | 39.57% fewer |
| repair passes | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## Schedule Validation

| Rule | Before | After | Status |
|---|---|---|---|
| `max_games_in_4_days` | 3 | 3 | PASS |
| `max_games_in_7_days` | 5 | 4 | PASS |
| teams with 5-in-7 | 7 | 0 | PASS |
| hard validation errors | none in sample after finalize | none | PASS |
| deterministic same-seed hash | N/A | equal hashes | PASS |

Deterministic same-seed test:
- Hash A: `e992b840e118f014335ba9b162a278596dad8040bcd0873e2ab6bdaceb881f20`
- Hash B: `e992b840e118f014335ba9b162a278596dad8040bcd0873e2ab6bdaceb881f20`
- Match: `true`

## Player Bootstrap Before

- `ovr`: `307,041` calls / `22,884 ms`
- `compute_ovr`: `307,041` calls / `22,328 ms`
- `_skater_category_raw_avgs`: `278,259` calls / `16,050 ms`
- `_avg`: `2,897,718` calls / `15,671 ms`
- `clamp_rating`: `17,926,215` calls / `21,786 ms`
- `_boost_player_toward_ovr`: `6,091` calls / `38,541 ms`
- `enforce_minimum_player_ovr`: `9,118` calls / `39,287 ms`

## Player Bootstrap After

- `_avg` now uses direct bounded integer average (removes redundant clamp function calls on read path).
- `_boost_player_toward_ovr` uses fewer, larger adaptive increments.
- `enforce_minimum_player_ovr` uses fewer boost passes.
- Player generation distributions remained stable in after-run pool samples (see tables).

## Player Bootstrap Benchmark

| Function | Before Calls | After Calls | Before Time | After Time | Improvement |
|---|---:|---:|---:|---:|---:|
| `_spawn_player` | 7,323 | 7,301 | 44,514 ms | 17,769.35 ms | 60.08% |
| `Player.__init__` | 8,059 | 8,037 | 38,874 ms | 11,032.26 ms | 71.62% |
| `enforce_minimum_player_ovr` | 9,118 | 9,075 | 39,287 ms | 7,317.62 ms | 81.38% |
| `_boost_player_toward_ovr` | 6,091 | 5,756 | 38,541 ms | 6,805.98 ms | 82.34% |
| `ovr` | 307,041 | 189,058 | 22,884 ms | 9,535.09 ms | 58.34% |
| `compute_ovr` | 307,041 | 189,058 | 22,328 ms | 9,216.82 ms | 58.72% |
| `_skater_category_raw_avgs` | 278,259 | 169,901 | 16,050 ms | 5,637.23 ms | 64.88% |
| `_avg` | 2,897,718 | 1,775,638 | 15,671 ms | 5,267.66 ms | 66.39% |
| `clamp_rating` | 17,926,215 | 4,032,721 | 21,786 ms | 4,890.38 ms | 77.55% |
| `_player_ovr_0_100` | 164,337 | 48,753 | 12,249 ms | 2,579.87 ms | 78.94% |

## Player Distribution Validation

| Pool | Players | Before Mean OVR | After Mean OVR | Before Median | After Median | Distribution Status |
|---|---:|---:|---:|---:|---:|---|
| NHL | 736 | NOT MEASURED | 79.49 | NOT MEASURED | 76.83 | PASS (same-seed stable after) |
| AHL | 768 | NOT MEASURED | 60.88 | NOT MEASURED | 60.60 | PASS |
| ECHL | 576 | NOT MEASURED | 52.87 | NOT MEASURED | 52.60 | PASS |
| UFA | 520 | NOT MEASURED | 56.04 | NOT MEASURED | 55.64 | PASS |
| Overseas | 220 | NOT MEASURED | 57.30 | NOT MEASURED | 55.74 | PASS |

Potential/position/band note: full before/after statistical diff was NOT MEASURED in this pass; after outputs are deterministic for same seed.

## Hierarchy Pool Timings

| Pool | Players | Before Time | After Time | After ms/player | Improvement |
|---|---:|---:|---:|---:|---|
| european_junior | 2,708 | NOT MEASURED | 2,015.75 ms | 0.7444 | NOT MEASURED |
| junior | 2,190 | NOT MEASURED | 1,153.90 ms | 0.5269 | NOT MEASURED |
| ahl | 768 | NOT MEASURED | 655.73 ms | 0.8538 | NOT MEASURED |
| echl | 576 | NOT MEASURED | 471.23 ms | 0.8181 | NOT MEASURED |
| ufa | 520 | NOT MEASURED | 403.74 ms | 0.7764 | NOT MEASURED |
| college | 319 | NOT MEASURED | 198.84 ms | 0.6233 | NOT MEASURED |
| overseas | 220 | NOT MEASURED | 134.64 ms | 0.6120 | NOT MEASURED |

## Contract Bootstrap

- Old lookup architecture: repeated player/contract scoring and contract generation while enforcing team roster/cap consistency.
- New indexing architecture in this pass: **no major contract algorithm rewrite**; startup gains here are mostly indirect from reduced OVR helper overhead and schedule path cleanup.
- Before time: `7,880 ms`
- After time (`_ensure_league_roster_contracts` inclusive): `5,433.24 ms`
- Improvement: `31.05%`

Output equivalence checks:
- players with contracts: `736`
- players missing contracts: `0`
- salary/term/RFA-UFA detailed before-vs-after diff: `NOT MEASURED`

## Trade Assets Cold Load

- Before cold: `3,167.71 ms`
- After cold: `3,043.80 ms`
- Warm after (avg): `21.20 ms`

Remaining cold cost is still dominated by first-build trade summary aggregation; warm cache remains excellent.

## Full Startup Results

| Stage | Before | After Min | After Avg | After Median | After Max | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| HTTP franchise start | 37,747 ms | 29,288.85 | 29,556.64 | 29,627.50 | 29,753.57 | 21.70% |
| direct Python startup | 122,237 ms | 20,217.31 | 20,541.12 | 20,683.75 | 20,722.30 | 83.19% |
| hierarchy bootstrap | 51,155 ms | NOT MEASURED | 23,809.00 | NOT MEASURED | NOT MEASURED | 53.46% |
| player spawning (`_spawn_player`) | 44,514 ms | NOT MEASURED | 17,769.35 | NOT MEASURED | NOT MEASURED | 60.08% |
| schedule generation | NOT MEASURED | NOT MEASURED | 4,234.00 | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| schedule finalize | 37,251 ms | NOT MEASURED | 8,506.00 | NOT MEASURED | NOT MEASURED | 77.17% |
| schedule smoothing | 36,826 ms | NOT MEASURED | 8,126.17 | NOT MEASURED | NOT MEASURED | 77.93% |
| contract bootstrap | 7,880 ms | NOT MEASURED | 5,433.24 | NOT MEASURED | NOT MEASURED | 31.05% |
| draft initialization | NOT MEASURED | NOT MEASURED | 9,345.00 | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| lean initial state build | NOT MEASURED | 69.76 | 86.38 | 79.95 | 109.43 | NOT MEASURED |

## Runtime Regression Check

| Endpoint | Previous Repaired Avg | New Avg | Change | Status |
|---|---:|---:|---:|---|
| `/api/franchise/state` | 56.90 ms | 65.55 ms | +15.20% | PASS (<20%) |
| `/api/franchise/scouting/world` (warm) | 2.21 ms | 6.56 ms | +196.83% | REGRESSION FLAG |
| `/api/franchise/scouting/prospects` | 6.81 ms | 15.02 ms | +120.56% | REGRESSION FLAG |
| `/api/franchise/teams` | 1.97 ms | 6.17 ms | +213.20% | REGRESSION FLAG |
| `/api/franchise/trade/assets` (warm) | 22.66 ms | 21.20 ms | -6.44% | PASS |

Notes:
- Warm medians for regressed endpoints remain near prior repaired levels (`world` median `2.10 ms`, `teams` median `1.99 ms`, `prospects` median `7.81 ms`), but averages exceeded threshold due intermittent spikes; flagged per requirement.

## Top 20 Functions After Repair

| Rank | Function | Calls | Exclusive Time (ms) | Inclusive Time (ms) | Average (ms/call) | File |
|---:|---|---:|---:|---:|---:|---|
| 1 | `_spawn_player` | 7,301 | 269.06 | 17,769.35 | 2.4338 | `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` |
| 2 | `Player.__init__` | 8,037 | 478.11 | 11,032.26 | 1.3727 | `SimEngine/app/sim_engine/entities/player.py` |
| 3 | `ovr` | 189,058 | 248.81 | 9,535.09 | 0.0504 | `SimEngine/app/sim_engine/entities/player.py` |
| 4 | `compute_ovr` | 189,058 | 2,330.20 | 9,216.82 | 0.0488 | `SimEngine/app/sim_engine/entities/player.py` |
| 5 | `_finalize_schedule_after_generation` | 1 | 1.27 | 8,505.80 | 8,505.8048 | `backend/services/franchise_sim.py` |
| 6 | `_smooth_league_schedule` | 1 | 27.78 | 8,126.17 | 8,126.1719 | `backend/services/franchise_sim.py` |
| 7 | `enforce_minimum_player_ovr` | 9,075 | 16.26 | 7,317.62 | 0.8063 | `SimEngine/app/sim_engine/entities/player.py` |
| 8 | `_boost_player_toward_ovr` | 5,756 | 1,632.02 | 6,805.98 | 1.1824 | `SimEngine/app/sim_engine/entities/player.py` |
| 9 | `_skater_category_raw_avgs` | 169,901 | 582.95 | 5,637.23 | 0.0332 | `SimEngine/app/sim_engine/entities/player.py` |
| 10 | `_ensure_league_roster_contracts` | 1 | 0.36 | 5,433.24 | 5,433.2449 | `backend/services/franchise_sim.py` |
| 11 | `_avg` | 1,775,638 | 4,948.57 | 5,267.66 | 0.0030 | `SimEngine/app/sim_engine/entities/player.py` |
| 12 | `clamp_rating` | 4,032,721 | 3,158.33 | 4,890.38 | 0.0012 | `SimEngine/app/sim_engine/entities/player.py` |
| 13 | `_player_ovr_0_100` | 48,753 | 61.41 | 2,579.87 | 0.0529 | `SimEngine/app/sim_engine/entities/player.py` |
| 14 | `_can_place_slot_on_day` | 6,969 | 32.53 | 495.50 | 0.0711 | `backend/services/franchise_sim.py` |
| 15 | `_would_create_bad_cadence_for_slot` | 2,544 | 130.69 | 395.79 | 0.1556 | `backend/services/franchise_sim.py` |
| 16 | `_team_ids_for_slot` | 37,790 | 14.28 | 218.68 | 0.0058 | `backend/services/franchise_sim.py` |
| 17 | `_safe_slot_team_id` | 112,074 | 52.83 | 94.21 | 0.0008 | `SimEngine/app/sim_engine/league/schedule_generator.py` |
| 18 | `_slot_has_team` | 4,978 | 3.73 | 33.36 | 0.0067 | `backend/services/franchise_sim.py` |
| 19 | `get_ovr_floor_for_pool` | 8,037 | 13.30 | 17.13 | 0.0021 | `SimEngine/app/sim_engine/entities/player.py` |
| 20 | `_player_ovr_frac` | 3,863 | 6.87 | 6.87 | 0.0018 | `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` |

## Remaining Bottlenecks

1. `bootstrap_full_league_hierarchy` / `_spawn_player` lifecycle (identity, ratings, chemistry/headshot hooks, constructor work).
2. prospect stats sync (`_sync_prospect_stats_to_calendar`).
3. OVR pipeline still heavy despite major reductions (`compute_ovr`, `_avg`, `_skater_category_raw_avgs`).
4. contract construction inside `_ensure_league_roster_contracts` still multi-second.
5. schedule smoothing still multi-second (but no longer catastrophic).
6. cold `/api/franchise/scouting/world` first-build cost.
7. cold `/api/franchise/trade/assets` first-build cost.
8. occasional warm endpoint spikes (average regression risk).
9. repeated prospect scoring helper work in startup flow.
10. per-player post-spawn enrichment cost (`apply_body_tradeoffs`, chemistry/headshot initialization).

## Final Verdict

1. Why did schedule smoothing take 36.8 seconds?  
   Repeated schedule-wide scans and team-id reparsing inside candidate/cadence loops (`_slot_has_team`, `_team_ids_for_slot`, `_safe_slot_team_id`) caused explosive helper volume.

2. How many schedule-wide scans or equivalent repeated lookups were removed?  
   Approximate proxy from helper calls:  
   - `_slot_has_team`: down by `7,731,314` calls  
   - `_team_ids_for_slot`: down by `7,736,368` calls  
   - `_safe_slot_team_id`: down by `15,870,414` calls

3. What is the new smoothing time?  
   `8,126.17 ms` (profiled inclusive).

4. Why was OVR calculated 307,041 times?  
   Startup player spawn/enforcement/distribution loops repeatedly invoked OVR and category averaging during iterative adjustment.

5. What is the new OVR computation count?  
   `189,058` calls for both `ovr` and `compute_ovr` in the profiled startup run.

6. Why was `clamp_rating` called 17.9 million times?  
   Clamp-heavy averaging and iterative boost loops repeatedly reclamped values during startup OVR enforcement workflow.

7. What is the new clamp call count?  
   `4,032,721`.

8. What is the new hierarchy bootstrap time?  
   `23,809.00 ms` inclusive (profiled run).

9. What is the new contract bootstrap time?  
   `_ensure_league_roster_contracts`: `5,433.24 ms` inclusive.

10. What is the new total franchise start time?  
   HTTP avg (3 runs): `29,556.64 ms`; direct Python avg (3 runs): `20,541.12 ms`.

11. Did player distributions change?  
   No obvious anomaly in after-run pool distributions; strict before/after statistical comparison was NOT MEASURED in this pass.

12. Is same-seed determinism preserved?  
   Yes; normalized schedule hash matched across repeated same-seed runs.

13. Did any previously repaired GET endpoint regress?  
   By average, yes for scouting/teams/prospects (flagged); warm medians stayed near prior repaired values.

14. What is now the single largest startup bottleneck?  
   `bootstrap_full_league_hierarchy` (especially `_spawn_player` + `Player.__init__`/OVR-related work).

15. What should be optimized next?  
   Pool-aware spawn fast-path and prospect stat initialization/OVR access reduction inside hierarchy bootstrap, then contract construction indexing and warm-spike stabilization for scouting endpoints.
