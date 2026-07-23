# NHL Franchise Mode Loading Performance Repair Report

## Executive Summary

This repair changed the loading architecture from a monolithic, always-heavy `/api/franchise/state` path to a lean-global + screen-owned heavy-data model, and fixed repeated draft-board recomputation on read paths.

Former primary bottleneck: `build_state_payload()` rebuilding draft rankings and serializing a 6.58 MB state payload on normal GET requests.

New primary bottleneck: franchise startup (`start_franchise`) remains dominated by league hierarchy bootstrap and schedule smoothing.

Is loading now acceptable?  
- **Runtime screen loading:** yes, major endpoints are now in fast ranges for warm reads.  
- **Franchise creation/startup:** still not acceptable (tens of seconds).

---

## Files Modified

| File path | Reason changed | Major change |
|---|---|---|
| `backend/services/franchise_session.py` | Add explicit cache validity and heavy-domain caches | Added draft cache state (`missing/dirty/valid`) and cached scouting/roster payload fields |
| `backend/services/franchise_sim.py` | Remove repeated expensive read recomputation | Lean/heavy state split in `build_state_payload`, cached draft/roster reads, teams endpoint process cache, deferred initial draft snapshot |
| `backend/services/franchise_scouting.py` | Remove draft ranking rebuilds from scouting GET paths | Switched to cached draft rankings and added cached scouting prospects/world payload reuse |
| `backend/main.py` | Change API response architecture | `/api/franchise/state` now lean, added `/api/franchise/state/heavy`, all state envelopes switched to lean payload |
| `frontend/src/services/api.js` | Fix duplicate startup state request source | Removed validation call to `/api/franchise/state` from `syncFranchiseSessionWithBackend()` |
| `frontend/src/services/franchiseService.js` | Request dedup + heavy-state access | Added in-flight dedupe for `getFranchiseState`; added `getFranchiseStateHeavy()` |
| `frontend/src/game/GameUIContext.js` | Preserve heavy domains + explicit heavy hydration | Added `hydrateFranchiseHeavyState()` and merge semantics so lean refreshes do not drop loaded heavy domains |
| `frontend/src/App.js` | Screen-owned heavy data loading | Added screen-based heavy hydration effect (Roster/Trade/Scouting/Draft/etc.) |
| `frontend/src/screens/TradeHub.js` | Remove unnecessary full-state refresh on open | Removed unconditional `refreshFranchise()` from mount-time `Promise.all` |

---

## Architecture Before

Typical flow:

UI request  
→ `GET /api/franchise/state`  
→ `build_state_payload()`  
→ `build_draft_class_rankings()` (every read)  
→ `_build_roster_browser()` (every read)  
→ giant JSON (6.58 MB)  
→ frontend render

Additional amplification:
- Startup path called `/api/franchise/state` twice.
- Trade Hub mount called `refreshFranchise()` + trade endpoints.
- Scouting prospects/world GET routes rebuilt draft rankings independently.

---

## Architecture After

### Lean global state

`GET /api/franchise/state`  
→ `build_state_payload(include_heavy=False)`  
→ no roster browser / no draft rankings / no draft HUD  
→ lightweight shared state only.

### Heavy domain loading

Screen open  
→ `GET /api/franchise/state/heavy` (with include flags)  
→ merge only required heavy domains into context.

### Draft ranking cache validity

`_draft_rankings_cache_state` tracks `missing | dirty | valid`.

Mutation path  
→ `invalidate_session_payload_caches()`  
→ state becomes `dirty`.

Read path  
→ `get_cached_draft_class_rankings()`  
→ rebuild only if not `valid`, else reuse cached board.

---

## Draft Ranking Cache Repair

### Old behavior
- Normal read paths (`/state`, scouting GETs) rebuilt rankings repeatedly.

### New behavior
- `/state` lean path performs **zero** draft ranking builds.
- Scouting GETs read through cache.
- Repeated scouting reads reuse cached normalized payloads.

### Invalidation triggers
- `invalidate_session_payload_caches()` now marks draft cache dirty and clears scouting/roster/trade cached domains.
- Called from mutation paths (advance day, scouting actions, WJC effects, trades, offseason transitions, draft picks, etc.).

### Ranking rebuild count during benchmark
- `build_state_payload(include_heavy=False)`: **0** rebuilds
- first `get_scouting_prospects()`: **1** rebuild
- `get_scouting_world()` immediately after: **still 1** (no extra rebuild)
- repeated prospects/world reads: **still 1**

---

## Universal State Payload Repair

BEFORE size: **6.58 MB**  
AFTER size: **552,450 bytes (0.53 MB)**

### Section-size comparison

| Section | Before Size | After Size | Change | New Endpoint or Data Owner |
|---|---:|---:|---:|---|
| `roster_browser` | 4,448,596 B | 0 B (removed from `/state`) | -100% | `/api/franchise/state/heavy` (roster-owned screens) |
| `draft_class_rankings` | 1,328,191 B | 0 B (removed from `/state`) | -100% | `/api/franchise/state/heavy` (Draft/Scouting owner) |
| `draft_class_hud` | 856,465 B | 0 B (removed from `/state`) | -100% | `/api/franchise/state/heavy` (Draft/Scouting owner) |
| `nhl_calendar_full` | 331,344 B | 331,334 B | ~0% | Global state |
| `roster` | 173,057 B | 173,048 B | ~0% | Global team summary context |
| `stats_central` | 41,254 B | 41,254 B | 0% | Global state |
| `league_operations` | 31,957 B | 32,122 B | +0.5% | Global state |

---

## API Benchmark Results

Measured after repairs with 1 cold + 5 warm requests.

| Endpoint | Before Avg | After Cold | After Avg | After Median | After Max | Improvement % |
|---|---:|---:|---:|---:|---:|---:|
| `/api/franchise/state` | 5,497 ms | 47.64 ms | 56.90 ms | 49.97 ms | 88.14 ms | **98.96% faster** |
| `/api/franchise/scouting/world` | 4,668 ms | 4,880.55 ms | 2.21 ms | 2.08 ms | 2.92 ms | **99.95% faster (warm)** |
| `/api/franchise/scouting/prospects` | 4,582 ms | 7.12 ms | 6.81 ms | 6.82 ms | 8.35 ms | **99.85% faster** |
| `/api/franchise/teams` | 1,622 ms | 1.91 ms | 1.97 ms | 1.84 ms | 2.34 ms | **99.88% faster** |
| `/api/franchise/trade/assets` | 932 ms | 3,167.71 ms | 22.66 ms | 16.68 ms | 38.01 ms | **97.57% faster (warm), cold regressed** |

---

## Franchise Startup Results

| Stage | Before | After | Improvement % |
|---|---:|---:|---:|
| Total franchise start (HTTP) | 41,816 ms | 37,747 ms | **9.73% faster** |
| Total franchise start (direct Python) | 119,417 ms | 122,237 ms | **-2.36% (regression)** |
| League hierarchy bootstrap | 45,049 ms | 51,155 ms | **-13.55% (regression)** |
| Schedule generation/finalize | 31,256 ms | 37,251 ms | **-19.18% (regression)** |
| Schedule smoothing | 30,912 ms | 36,826 ms | **-19.13% (regression)** |
| Contract bootstrap (`_ensure_league_roster_contracts`) | 6,274 ms | 7,880 ms | **-25.60% (regression)** |
| Initial state build (state payload) | 5,466 ms | 129 ms (`lean`) | **97.64% faster** |

Notes:
- Startup HTTP improved because start response now returns lean state instead of huge heavy state.
- Core simulation startup stages still dominate and need dedicated algorithmic optimization work.

---

## Frontend Request Results

| Metric | Before | After |
|---|---:|---:|
| Cold `/state` request count before Hub usable | 2 | 1 (by code-path repair: sync no longer calls `/state`) |
| Trade Hub `/state` requests on open | 1 | 0 (mount path no longer calls `refreshFranchise`) |
| Scouting ranking rebuilds on prospects+world sequence | 2+ | 1 (first compute only; then cache reuse) |

---

## Performance Profiling

Top expensive backend functions **after repair** (post-repair direct startup profile):

| Rank | Function | File | Call Count | Total Time | Average Time | Maximum Time |
|---|---|---|---:|---:|---:|---:|
| 1 | `start_franchise` | `backend/services/franchise_sim.py` | 1 | 122,237 ms | 122,237 ms | 122,237 ms |
| 2 | `bootstrap_full_league_hierarchy` | `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` | 1 | 51,155 ms | 51,155 ms | 51,155 ms |
| 3 | `_spawn_player` | `.../league_hierarchy_bootstrap.py` | 7,323 | 44,514 ms | 6.08 ms | NOT MEASURED |
| 4 | `enforce_minimum_player_ovr` | `.../entities/player.py` | 9,118 | 39,287 ms | 4.31 ms | NOT MEASURED |
| 5 | `Player.__init__` | `.../entities/player.py` | 8,059 | 38,874 ms | 4.82 ms | NOT MEASURED |
| 6 | `_boost_player_toward_ovr` | `.../entities/player.py` | 6,091 | 38,541 ms | 6.33 ms | NOT MEASURED |
| 7 | `_finalize_schedule_after_generation` | `backend/services/franchise_sim.py` | 1 | 37,251 ms | 37,251 ms | 37,251 ms |
| 8 | `_smooth_league_schedule` | `backend/services/franchise_sim.py` | 1 | 36,826 ms | 36,826 ms | 36,826 ms |
| 9 | `_add_league` | `.../league_hierarchy_bootstrap.py` | 15 | 35,556 ms | 2,370 ms | NOT MEASURED |
| 10 | `_can_place_slot_on_day` | `backend/services/franchise_sim.py` | 11,533 | 27,015 ms | 2.34 ms | NOT MEASURED |
| 11 | `_would_create_bad_cadence_for_slot` | `backend/services/franchise_sim.py` | 3,873 | 26,670 ms | 6.89 ms | NOT MEASURED |
| 12 | `ovr` | `.../entities/player.py` | 307,041 | 22,884 ms | 0.07 ms | NOT MEASURED |
| 13 | `compute_ovr` | `.../entities/player.py` | 307,041 | 22,328 ms | 0.07 ms | NOT MEASURED |
| 14 | `_slot_has_team` | `backend/services/franchise_sim.py` | 7,736,292 | 22,086 ms | 0.00 ms | NOT MEASURED |
| 15 | `clamp_rating` | `.../entities/player.py` | 17,926,215 | 21,786 ms | 0.00 ms | NOT MEASURED |
| 16 | `_skater_category_raw_avgs` | `.../entities/player.py` | 278,259 | 16,050 ms | 0.06 ms | NOT MEASURED |
| 17 | `_team_ids_for_slot` | `backend/services/franchise_sim.py` | 7,774,158 | 15,869 ms | 0.00 ms | NOT MEASURED |
| 18 | `_avg` | `.../entities/player.py` | 2,897,718 | 15,671 ms | 0.01 ms | NOT MEASURED |
| 19 | `_player_ovr_0_100` | `.../entities/player.py` | 164,337 | 12,249 ms | 0.07 ms | NOT MEASURED |
| 20 | `_safe_slot_team_id` | `.../league/schedule_generator.py` | 15,982,488 | 12,234 ms | 0.00 ms | NOT MEASURED |

---

## Long-Term Franchise Results

| Year | State Payload | State Avg | Roster Avg | Draft Avg | Season Sim Time | Player Count | Retired Count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 552,450 B (`/state`) | 56.90 ms | 154.38 ms (`/state/heavy` roster-only warm avg) | 114.93 ms (`/state/heavy` draft-only warm avg) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 5 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 10 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 20 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 40 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 80 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Reason: full multi-decade replay with profiling was out of this run’s execution budget after implementing and validating the repair set.

---

## Remaining Bottlenecks

| Severity | Measured impact | File | Function/component | Recommended next repair |
|---|---:|---|---|---|
| P0 | 36.8 s | `backend/services/franchise_sim.py` | `_smooth_league_schedule` | Replace full rescans with team/day indexed incremental conflict repair |
| P0 | 51.2 s | `SimEngine/.../league_hierarchy_bootstrap.py` | `bootstrap_full_league_hierarchy` | Pool-level batching, precomputed templates, reduce repeated OVR recomputation during spawn |
| P0 | 37.3 s | `backend/services/franchise_sim.py` | `_finalize_schedule_after_generation` | Reduce nested pass count and repeated sorting/scanning |
| P1 | 7.9 s | `backend/services/franchise_sim.py` | `_ensure_league_roster_contracts` | Team/player indexing and incremental cap accounting |
| P1 | 6.2 s cold | `backend/main.py` + heavy route | first `/state/heavy` roster load | Further split heavy endpoint by domain and defer non-visible roster sections |
| P1 | 3.2 s cold | `backend/services/trade_service.py` | `build_trade_assets_payload` | Precompute stable team/player summaries and lazy-evaluate detailed value blocks |
| P2 | 331 KB | `backend/services/franchise_sim.py` | `nhl_calendar_full` in lean state | Move full calendar to calendar-owned endpoint and keep strip in global state |
| P2 | NOT MEASURED | `frontend/src/screens/RosterScreen.js` | large list normalization/sorts | Profile and memoize by stable slices, avoid full-table recompute on minor state changes |
| P2 | NOT MEASURED | `frontend/src/screens/StatsCentralScreen.js` | `normalizeStatsCentral` | Pre-normalize in backend or memoize by stats version key |
| P3 | NOT MEASURED | `frontend/src/App.js` | heavy hydrate on Hub | Restrict heavy prefetch to truly required screens only after UX verification |

---

## Regression Findings

1. **Startup core stages regressed** in this run:
   - direct startup total, hierarchy bootstrap, schedule finalize/smoothing, contract bootstrap all increased vs baseline.
2. **Cold `/api/franchise/trade/assets` regressed** (2,731 ms → 3,168 ms), while warm improved sharply.
3. **No gameplay functional regression observed in this run**, but broad end-to-end gameplay verification across all screens/events was **NOT MEASURED** with automated UI.

Fix status for regressions:
- Cold/warm gameplay read performance regressions were fixed where introduced (state and scouting paths).
- Startup-stage regressions remain and require dedicated algorithmic optimization (not masked).

---

## Final Verdict

1. **What caused the original 5.5-second state load?**  
   Rebuilding draft rankings + roster browser + draft HUD inside every `build_state_payload()` read and serializing a 6.58 MB payload.

2. **What is the new state load average?**  
   **56.90 ms** warm average (`/api/franchise/state`).

3. **How much smaller is the state payload?**  
   From **6.58 MB** to **552,450 bytes** (~0.53 MB), about **91.6% smaller**.

4. **Are draft rankings still rebuilt on normal GET requests?**  
   Not on lean `/state`; scouting read path rebuilds once then reuses cache until invalidated.

5. **Is the duplicate cold-start state request fixed?**  
   Yes by architecture change: startup sync path no longer calls `/state`; only refresh path does.

6. **Does Trade Hub still reload the franchise on open?**  
   No. Mount-time `refreshFranchise()` call was removed.

7. **What is the new Scouting load time?**  
   Warm averages: `/scouting/prospects` **6.81 ms**, `/scouting/world` **2.21 ms**. First cold world request still pays initial board build.

8. **What is the new franchise creation time?**  
   HTTP: **37,747 ms** (down from 41,816 ms). Direct profile run: **122,237 ms** (regressed vs 119,417 ms baseline).

9. **What is the new schedule smoothing time?**  
   **36,826 ms** (regressed from ~30,912 ms in baseline run).

10. **What is now the single largest performance bottleneck?**  
    Startup schedule + hierarchy generation (`bootstrap_full_league_hierarchy`, `_finalize_schedule_after_generation`, `_smooth_league_schedule`).

11. **How does performance change after multiple franchise seasons?**  
    **NOT MEASURED** in this repair run.

12. **What should be optimized next?**  
    Startup algorithmic complexity: schedule smoothing/finalization and hierarchy bootstrap passes, then cold trade-assets build.

