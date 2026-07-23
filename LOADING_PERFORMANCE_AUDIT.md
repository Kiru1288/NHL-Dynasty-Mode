# NHL Franchise Mode Loading Performance Audit

**Audit date:** 2026-07-08  
**Environment:** Windows 10, Python 3.14, React dev server (`npm start`), FastAPI/uvicorn on `127.0.0.1:8000`  
**Method:** Temporary profiling scripts (removed after audit), live HTTP benchmarks against running servers, direct Python timing/cProfile, static code trace. No application code was modified.

---

## Executive Summary

NHL Franchise Mode is slow to load because the backend serializes a **6.6 MB franchise-state JSON document on every `GET /api/franchise/state`**, and that serialization takes **~5.5 seconds**. The dominant cost inside that response is **`build_draft_class_rankings()`** (~4.5 s estimated from profiling proportions) followed by **`_build_roster_browser()`** (~0.9 s), which together also account for **~79% of payload size** (`roster_browser` 4.25 MB + `draft_class_rankings` 1.27 MB).

Creating a new franchise (`POST /api/franchise/start`) takes **42–119 seconds** depending on measurement path, driven by **league hierarchy bootstrap** (~7,249 procedural players), **schedule smoothing** (~31 s), and **contract bootstrap**.

The frontend amplifies backend cost by:
1. Fetching `/api/franchise/state` **twice on cold start** (session sync + `refreshFranchise`).
2. Opening screens that **re-trigger the same expensive backend work** (Scouting, Draft Class, Trade Hub) even when equivalent data already exists in `franchiseState`.

**There is no SQLite database.** Persistence is an in-memory Python dict (`franchise_store.py`). Database performance is not a factor.

**Single largest bottleneck:** `build_state_payload()` → `build_draft_class_rankings()` inside `GET /api/franchise/state`.

**Primary problem category:** **Backend payload construction + payload size**, compounded by **frontend duplicate requests**.

---

## Performance Score

| Area | Score ( /10 ) | Notes |
|------|---------------|-------|
| Startup Speed | **2** | Franchise creation 42–119 s; teams endpoint 1.6 s |
| API Performance | **3** | Core state endpoint 5.5 s CRITICAL |
| Frontend Rendering | **NOT MEASURED** | Code review shows heavy `useMemo` chains; no React Profiler run |
| Database Performance | **N/A** | No SQLite; in-memory sessions only |
| Payload Efficiency | **2** | 6.6 MB monolithic state; 61% in `roster_browser` alone |
| Long-Term Save Performance | **NOT MEASURED** | Years 5–80 not completed within audit time budget |

---

## Critical Findings

### 1. Monolithic franchise state rebuild on every GET

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `backend/services/franchise_sim.py` |
| **Function** | `build_state_payload()` (line 11663) |
| **Measured time** | 5,466–5,719 ms (5-run avg 5,523 ms); HTTP avg **5,497 ms** |
| **Root cause** | Full read-model rebuilt synchronously on each request; calls `build_draft_class_rankings()` directly (not session cache) at line 11790 |
| **Triggers** | `GET /api/franchise/state`, `POST /api/franchise/start` response, advance/decision responses that embed state |
| **Over seasons** | `build_draft_class_rankings` cost is dominated by prospect count (~4,039 entries at Year 1), not season history; `season_history` cleared per season but `league.players` grows — long-term impact **NOT MEASURED** |

### 2. Draft class rankings are the most expensive builder

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `backend/services/franchise_sim.py`, `backend/services/draft_ranking_logic.py` |
| **Function** | `build_draft_class_rankings()` (line 4453) |
| **Measured time** | 14,813 ms isolated cProfile run; ~83% of `build_state_payload` profiled time |
| **Root cause** | 4,039 prospects × `enrich_prospect_row_from_player`, `derive_prospect_analytics`, `prospect_stats_for_api`, hard-ranking floor passes |
| **Triggers** | Every `build_state_payload`, `_draft_entries()` in scouting, `snapshot_draft_rank_prev` at franchise start |
| **Over seasons** | NOT MEASURED |

### 3. Roster browser dominates payload size

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `backend/services/franchise_sim.py` |
| **Function** | `_build_roster_browser()` (line 3990) |
| **Measured time** | 3,043 ms isolated |
| **Payload** | **4,448,596 bytes** (61.4% of 7.24 MB state) |
| **Root cause** | Serializes ~7,014 player rows across 32 orgs (NHL/AHL/ECHL/prospects/dev leagues) with full rating groups for user org |
| **Triggers** | Every `build_state_payload` |
| **Over seasons** | NOT MEASURED |

### 4. Franchise creation is extremely slow

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `backend/services/franchise_sim.py`, `SimEngine/app/sim_engine/league_hierarchy_bootstrap.py` |
| **Function** | `start_franchise()` (line 1733) |
| **Measured time** | **119,417 ms** direct; **41,816 ms** via `POST /api/franchise/start` HTTP |
| **Root cause** | `bootstrap_full_league_hierarchy` 45 s (7,249 `_spawn_player` calls), `_smooth_league_schedule` 31 s, contract bootstrap 6 s, prospect stat sync 8 s |
| **Triggers** | Once per new franchise |
| **Over seasons** | N/A (one-time) |

### 5. Double franchise-state fetch on app bootstrap

| Field | Value |
|-------|-------|
| **Severity** | MAJOR |
| **File** | `frontend/src/services/api.js`, `frontend/src/game/GameUIContext.js` |
| **Function** | `syncFranchiseSessionWithBackend()` then `refreshFranchise()` |
| **Measured time** | Up to **~11 s** of duplicate state serialization on resume (2 × 5.5 s) |
| **Root cause** | `syncFranchiseSessionWithBackend` calls `GET /api/franchise/state` (line 71); `GameUIProvider` then calls `refreshFranchise()` (line 133) |
| **Triggers** | Every browser reload with saved session |
| **Over seasons** | Same per-request backend cost each reload |

### 6. Scouting endpoints repeat draft-ranking work

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **File** | `backend/services/franchise_scouting.py` |
| **Function** | `get_scouting_prospects()` → `_draft_entries()` → `build_draft_class_rankings()` |
| **Measured time** | `/api/franchise/scouting/prospects` **4,582 ms**; `/api/franchise/scouting/world` **4,668 ms** (world calls prospects internally) |
| **Root cause** | Scouting does not use `get_cached_draft_class_rankings()`; rebuilds full draft board |
| **Triggers** | Scouting screen mount (4 parallel GETs), Draft Class mount |
| **Over seasons** | NOT MEASURED |

### 7. Trade Hub forces redundant state refresh

| Field | Value |
|-------|-------|
| **Severity** | MAJOR |
| **File** | `frontend/src/screens/TradeHub.js` |
| **Function** | Mount `useEffect` `Promise.all([refreshFranchise(), ...])` (line 5837) |
| **Measured time** | +5,497 ms state + trade assets (931 ms avg, 2,731 ms cold) |
| **Root cause** | Unconditional `refreshFranchise()` on every Trade Hub open |
| **Triggers** | Navigating to Trade Hub |
| **Over seasons** | NOT MEASURED |

### 8. Teams list bootstraps full SimEngine

| Field | Value |
|-------|-------|
| **Severity** | MAJOR |
| **File** | `backend/services/franchise_sim.py` |
| **Function** | `list_teams_summary()` (line 8554) |
| **Measured time** | **1,622 ms** HTTP avg |
| **Root cause** | Constructs `SimEngine(seed=1)` and throws it away on every call |
| **Triggers** | Setup screen `loadTeams()` |
| **Over seasons** | N/A |

### 9. Schedule smoothing was removed from state GET (good) but remains in startup

| Field | Value |
|-------|-------|
| **Severity** | MAJOR (startup only) |
| **File** | `backend/services/franchise_sim.py` |
| **Function** | `_smooth_league_schedule()` (line 1259), `_finalize_schedule_after_generation()` (line 832) |
| **Measured time** | 30,912 ms in `start_franchise` profile |
| **Root cause** | Full-league schedule optimization at franchise creation |
| **Triggers** | `start_franchise` only — **not** on `GET /api/franchise/state` (explicitly disabled per comment at line 11665) |
| **Over seasons** | `generate_next_season` re-runs schedule finalization each year — per-season cost NOT MEASURED |

### 10. React StrictMode doubles effects in development

| Field | Value |
|-------|-------|
| **Severity** | MODERATE (dev only) |
| **File** | `frontend/src/index.js` |
| **Function** | `<StrictMode>` wrapper |
| **Measured time** | NOT MEASURED in browser |
| **Root cause** | Mount effects run twice in development |
| **Triggers** | All `useEffect` API calls in dev |
| **Over seasons** | N/A |

---

## Phase 1 — Startup Loading Test

| Measurement | Result |
|-------------|--------|
| Backend `import main` (cold) | **741 ms** |
| uvicorn process already running | Active before audit |
| Database initialization | **N/A** — no database |
| `start_franchise()` (direct Python) | **119,417 ms** (119.4 s) |
| `POST /api/franchise/start` (HTTP) | **41,816 ms** (41.8 s) |
| First `build_state_payload()` after start | **5,466 ms** |
| Frontend dev server (`npm start`) | Already running; **NOT MEASURED** for cold webpack compile |
| Initial HTML (`GET localhost:3000`) | **2.6 ms**, 1,711 bytes (shell only) |
| Time to first usable screen | **NOT MEASURED** in browser (requires Playwright/React Profiler) |

### Startup stage breakdown (`start_franchise` cProfile)

| Stage | Cumulative time | Frequency |
|-------|-----------------|-----------|
| `bootstrap_full_league_hierarchy` | 45.0 s | Once per franchise |
| `_spawn_player` (7,249 calls) | 39.1 s | Once per franchise |
| `_finalize_schedule_after_generation` / `_smooth_league_schedule` | 31.3 s | Once per franchise start; also in `generate_next_season` |
| `snapshot_draft_rank_prev` / `build_draft_class_rankings` | 14.8 s | At start + every state GET |
| `_sync_prospect_stats_to_calendar` | 8.4 s | Once per franchise |
| `_ensure_league_roster_contracts` | 6.3 s | Once per franchise |
| `SimEngine.initialize_universe` / `_populate_initial_rosters` | 7.2 s | Once per franchise |

### Task frequency summary

| Task | Once | Every startup | Every request | Unnecessarily repeated |
|------|------|---------------|---------------|------------------------|
| League hierarchy bootstrap | ✓ (per franchise) | | | No |
| Roster generation (NHL initial) | ✓ | | | No |
| Free-agent / AHL / ECHL generation | ✓ | | | No |
| Draft class ranking build | | | ✓ (state GET) | **Yes** — also in scouting endpoints |
| Historical season loading | | | | N/A — no disk save |
| Save deserialization | | | | N/A — in-memory only |
| Standings construction | | | ✓ (embedded in state) | Rebuilt each state GET |
| Player index / roster browser | | | ✓ | Rebuilt each state GET |
| Team data hydration | | | ✓ | Full `roster_browser` each GET |
| Schedule smoothing | ✓ (start) | | **No** on state GET | Removed from state GET (fixed prior bug) |

---

## Phase 2 — API Endpoint Performance

Measured via live HTTP against running uvicorn (3 runs each, warm session). Ratings: FAST <100 ms · ACCEPTABLE 100–300 ms · SLOW 300–1000 ms · CRITICAL >1000 ms.

| Endpoint | Method | Calling Screen | Avg (ms) | Worst (ms) | Payload (bytes) | Backend Function | DB Queries | Rating |
|----------|--------|----------------|----------|------------|-----------------|------------------|------------|--------|
| `/api/health` | GET | Bootstrap | 10.98 | 28.81 | 581 | `health()` | None | FAST |
| `/api/franchise/teams` | GET | Setup | 1,680.16 | 1,694.47 | 1,390 | `list_teams_summary()` | None | CRITICAL |
| `/api/franchise/state` | GET | Hub, Calendar, Roster, Stats, Awards, WJC, etc. | **5,497.45** | **5,557.10** | **6,580,881** | `build_state_payload()` | None | CRITICAL |
| `/api/franchise/contract-office` | GET | Cap Ledger / Contracts | 354.92 | 356.40 | 48,365 | `get_contract_office()` | None | SLOW |
| `/api/franchise/chemistry` | GET | Chemistry, Edit Lines, PP, PK | 22.29 | 37.69 | 4,636 | `get_franchise_chemistry_report()` | None | FAST |
| `/api/franchise/league-operations` | GET | League Operations | 27.13 | 39.36 | 29,517 | `build_league_operations_payload()` | None | FAST |
| `/api/franchise/trade/assets` | GET | Trade Hub, Draft Class | 931.80 | 2,730.95 | 1,808,695 | `build_trade_assets_payload()` | None | CRITICAL (cold) / SLOW (warm ~30 ms) |
| `/api/franchise/trade/market` | GET | Trade Hub | 7.60 | 14.88 | 1,542 | trade market builder | None | FAST |
| `/api/franchise/trade/history?limit=20` | GET | Trade Hub | 5.95 | 14.20 | 14 | trade history | None | FAST |
| `/api/franchise/scouting/state` | GET | Scouting | 13.85 | 26.56 | 2,177 | `get_scouting_state()` | None | FAST |
| `/api/franchise/scouting/world` | GET | Scouting | **4,667.84** | **4,836.30** | 3,863 | `get_scouting_world()` → `get_scouting_prospects()` | None | CRITICAL |
| `/api/franchise/scouting/prospects` | GET | Scouting, Draft Class | **4,582.17** | **4,645.05** | 620,020 | `get_scouting_prospects()` → `build_draft_class_rankings()` | None | CRITICAL |
| `/api/franchise/scouting/assignments` | GET | Scouting | 8.41 | 20.86 | 45 | `get_scouting_assignments()` | None | FAST |
| `/api/franchise/lines` | GET | Edit Lines | 5.44 | 12.93 | 22 | lines getter | None | FAST |
| `/api/franchise/entry-draft/state` | GET | Entry Draft | NOT MEASURED | | | offseason-gated | | |
| `/api/franchise/draft-combine/state` | GET | Draft Combine | NOT MEASURED | | | offseason-gated | | |

### Sequential fetch patterns documented (not fixed)

| Screen | Pattern |
|--------|---------|
| **App bootstrap** | `GET /api/health` → `GET /api/franchise/state` (sync) → `GET /api/franchise/state` (refresh) |
| **Trade Hub** | `refreshFranchise()` (state) ∥ trade assets ∥ market ∥ history — state refresh is redundant if context is fresh |
| **Scouting** | 4 parallel GETs; `world` and `prospects` both trigger full draft ranking rebuild |
| **Draft Class** | Reads `draft_class_rankings` from context **then** `GET scouting/prospects` + `GET trade/assets` on mount |
| **Cap Ledger** | `GET contract-office` on mount (state not required) |
| **Chemistry family** | Independent `GET chemistry` per screen visit (no shared cache) |

---

## Phase 3 — Franchise Save Payload Analysis

> **Note:** There is no disk save file. Measurements are of `build_state_payload()` JSON (in-memory session).

| Metric | Value |
|--------|-------|
| Total JSON size (uncompressed) | **7,242,010 bytes** (7.24 MB) |
| HTTP transferred size (`/api/franchise/state`) | **6,580,881 bytes** (6.58 MB) |
| Compression | JSON over HTTP; no gzip measured — sizes are raw body |
| Top-level fields | **59** |
| Players in `league.players` | **7,985** |
| NHL roster players (32 teams) | **736** |
| Free agents | **520** |
| Retired (Year 1) | **0** |
| `season_history` entries (Year 1) | **0** |
| `game_results` (Year 1) | **0** |

### 20 Largest JSON Sections (Year 1)

| Rank | Section | Size |
|------|---------|------|
| 1 | `roster_browser` | 4,448,596 bytes (4.25 MB) |
| 2 | `draft_class_rankings` | 1,328,191 bytes (1.27 MB) |
| 3 | `draft_class_hud` | 856,465 bytes (0.82 MB) |
| 4 | `nhl_calendar_full` | 331,344 bytes (0.32 MB) |
| 5 | `roster` | 173,057 bytes (0.17 MB) |
| 6 | `stats_central` | 41,254 bytes |
| 7 | `league_operations` | 31,957 bytes |
| 8 | `season_anchor_events` | 9,343 bytes |
| 9 | `nhl_calendar_strip` | 7,940 bytes |
| 10 | `schedule_upcoming` | 4,066 bytes |
| 11 | `standings` | 2,947 bytes |
| 12 | `team` | 1,690 bytes |
| 13 | `schedule_diagnostics` | 1,491 bytes |
| 14 | `notifications` | 768 bytes |
| 15 | `timeline` | 335 bytes |
| 16 | `nhl_today` | 330 bytes |
| 17 | `franchise_pulse` | 249 bytes |
| 18 | `flags` | 225 bytes |
| 19 | `gm_world` | 225 bytes |
| 20 | `dev_league_generation` | 151 bytes |

Embedded data confirmed in state: draft rankings, draft HUD, WJC bundle field (null at Year 1), transaction/trade data via `trade_assets` cache, statistics via `stats_central`, full calendar, league operations.

### Duplicate Player Data (verified in code + payload walk)

| Duplication | Evidence |
|-------------|----------|
| User NHL roster in `roster` **and** `roster_browser.organizations[user].nhl` | Player `PLAYER_2340248` appears in both `roster[0]` and `roster_browser.organizations[7].nhl[0]` |
| Draft prospects in `draft_class_rankings` **and** recomputed in `GET scouting/prospects` | `_draft_entries()` calls `build_draft_class_rankings()` again (`franchise_scouting.py:401-404`) |
| Draft HUD profiles overlap draft rankings | `draft_class_hud` (0.82 MB) references same prospect IDs |
| Full rating groups for user org players serialized multiple times across org pools | `rating_groups` attribute keys appear across `roster` + `roster_browser` depth ladders |
| Unique player IDs indexed in payload walk | 7,018 |
| Players appearing in multiple top-level sections | 127 (excluding rating-attribute false positives) |

---

## Phase 4 — Frontend Loading Analysis

**React render timings: NOT MEASURED** (no React Profiler or Playwright run in this audit). Analysis is from static code review.

### Screen ranking (estimated from API dependency + code complexity)

| Rank | Screen | Est. API time | Main bottleneck | Notes |
|------|--------|---------------|-----------------|-------|
| 1 | **Scouting** | ~4,600 ms+ (parallel) | `scouting/prospects` + `scouting/world` | 4 mount GETs; two rebuild draft rankings |
| 2 | **Trade Hub** | ~6,400 ms+ | `refreshFranchise` + cold `trade/assets` | 13,018-line component |
| 3 | **Draft Class** | ~5,500 ms+ | Context state + `scouting/prospects` + `trade/assets` | Board data already in `franchiseState.draft_class_rankings` |
| 4 | **App bootstrap / Hub** | ~11,000 ms (2× state) | Double `GET /api/franchise/state` | Hub also mounts WebGL `FirstPersonOfficeHub` |
| 5 | **Cap Ledger** | ~355 ms | `contract-office` | Dedicated endpoint |
| 6 | **League Operations** | ~27 ms | `league-operations` | Fast |
| 7 | **Roster** | 0 ms mount API | React: `buildFranchiseStatsLookup`, `normalizeLivePlayer`, `filteredPlayers` sort | 8,858 lines; ~25 `useMemo` hooks on `franchiseState` |
| 8 | **Stats Central** | 0 ms mount API | `normalizeStatsCentral()` full league normalize | 8,802 lines |
| 9 | **Calendar** | 0 ms mount API | Deep `useMemo` chain on schedule/standings | 10,410 lines |
| 10 | **Awards / WJC / Retirements / Cap Report** | 0 ms | Context only (event overlays) | |

### Confirmed frontend behaviors

| Screen | Mount API requests | Duplicate state? | Expensive render paths |
|--------|-------------------|--------------------|------------------------|
| Hub | 0 (uses context) | Via bootstrap | `FirstPersonOfficeHub` WebGL `useFrame` loop |
| Calendar | 0 | No | `monthGrid`, `gamesByDate` useMemos |
| Roster | 0 | No | `buildFranchiseStatsLookup` (line 642), `rawPlayers`/`filteredPlayers` (lines 4312–4514), `PlayerProfileModal` → `PS1PlayerPortrait` |
| Cap Ledger | 1 (`contract-office`) | No | — |
| Trade Hub | 4 (state+3 trade) | **Yes** `refreshFranchise` | Package evaluation useMemos |
| Draft Class | 2 (prospects, trade/assets) | Partial — rankings in state | Prospect filtering/sorting |
| Scouting | 4 parallel | No state GET | Large prospect maps |
| Stats Central | 0 | No | `normalizeStatsCentral` (line 1383) |
| Chemistry / Lines / PP / PK | 1 each (`chemistry`) | No | Per-visit fetch, no shared cache |

### Repeated calculations on render (code paths)

- `RosterScreen.buildFranchiseStatsLookup()` — rebuilds Map from all `stats_central` buckets when `franchiseState` changes
- `RosterScreen.normalizeLivePlayer()` — called per player in `players` useMemo
- `RosterScreen.filteredPlayers` — full filter + sort of entire pool (pagination is display-only, 16 rows/page)
- `StatsCentralScreen.normalizeStatsCentral()` — full league skater/goalie normalization per state change

---

## Phase 5 — Backend Function Profiling

Top 25 expensive functions (from `start_franchise` + `build_state_payload` cProfile runs):

| Rank | Function | File | Call count | Total (ms) | Avg (ms) | Max (ms) | Called from |
|------|----------|------|------------|------------|----------|----------|-------------|
| 1 | `start_franchise` | franchise_sim.py:1733 | 1 | 119,417 | 119,417 | 119,417 | `POST /api/franchise/start` |
| 2 | `bootstrap_full_league_hierarchy` | league_hierarchy_bootstrap.py:196 | 1 | 45,049 | 45,049 | 45,049 | `start_franchise` |
| 3 | `_spawn_player` | league_hierarchy_bootstrap.py:74 | 7,249 | 39,144 | 5.4 | — | hierarchy bootstrap |
| 4 | `build_state_payload` | franchise_sim.py:11663 | 5+ | 5,523 avg | 5,523 | 5,719 | `GET /api/franchise/state` |
| 5 | `build_draft_class_rankings` | franchise_sim.py:4453 | 1+ per state | 14,813 | 14,813 | 14,813 | `build_state_payload`, scouting |
| 6 | `_smooth_league_schedule` | franchise_sim.py:1259 | 1 | 30,912 | 30,912 | 30,912 | `start_franchise` |
| 7 | `enforce_minimum_player_ovr` | player.py:744 | 9,042 | 34,643 | 3.8 | — | player spawn |
| 8 | `Player.__init__` | player.py:1271 | 7,985 | 34,490 | 4.3 | — | spawn/bootstrap |
| 9 | `_boost_player_toward_ovr` | player.py:671 | 6,078 | 33,983 | 5.6 | — | spawn |
| 10 | `_finalize_schedule_after_generation` | franchise_sim.py:832 | 1 | 31,256 | 31,256 | 31,256 | `start_franchise` |
| 11 | `_can_place_slot_on_day` | franchise_sim.py:511 | 10,558 | 22,437 | 2.1 | — | schedule smooth |
| 12 | `compute_ovr` | player.py:540 | 331,662+ | 20,687+ | 0.06 | — | serialization/rankings |
| 13 | `snapshot_draft_rank_prev` | franchise_sim.py:4869 | 1 | 14,779 | 14,779 | 14,779 | `start_franchise` |
| 14 | `enrich_prospect_row_from_player` | draft_ranking_logic.py:321 | 4,039 | 4,427 | 1.1 | — | draft rankings |
| 15 | `_build_roster_browser` | franchise_sim.py:3990 | 1+ per state | 3,043 | 3,043 | 3,043 | `build_state_payload` |
| 16 | `_serialize_player_row` | franchise_sim.py:3843 | 7,014+ | 3,019 | 0.43 | — | roster browser |
| 17 | `build_trade_assets_payload` | trade_service.py | 1+ | 2,803 | 2,803 | 2,731 | trade endpoints |
| 18 | `get_scouting_prospects` | franchise_scouting.py:407 | 1+ | 4,612 | 4,612 | 4,645 | scouting API |
| 19 | `_sync_prospect_stats_to_calendar` | franchise_sim.py:5730 | 1 | 8,380 | 8,380 | 8,380 | `start_franchise` |
| 20 | `_ensure_league_roster_contracts` | franchise_sim.py:2311 | 1 | 6,274 | 6,274 | 6,274 | `start_franchise` |
| 21 | `prospect_stats_for_api` | prospect_league_scoring.py:1848 | 4,039 | 6,243 | 1.5 | — | draft rankings |
| 22 | `derive_prospect_analytics` | prospect_league_scoring.py:1684 | 4,039 | 5,917 | 1.5 | — | draft rankings |
| 23 | `build_contract_for_player` | contract_economy.py:745 | 736 | 5,650 | 7.7 | — | contract bootstrap |
| 24 | `list_teams_summary` | franchise_sim.py:8554 | 1+ | 1,622 | 1,622 | 1,640 | `GET /api/franchise/teams` |
| 25 | `get_contract_office` | franchise_sim.py:12565 | 1+ | 345 | 345 | 356 | Cap Ledger |

**Recalculates unchanged data:** `build_draft_class_rankings` and `_build_roster_browser` run on every `GET /api/franchise/state` despite session-level caches existing for draft rankings (`get_cached_draft_class_rankings` at line 12056) and trade assets (`get_cached_trade_assets_payload` at line 12429) — **`build_state_payload` does not use these caches** (calls `build_draft_class_rankings` directly at line 11790).

---

## Phase 6 — SQLite Database Testing

| Finding | Result |
|---------|--------|
| Database file | **None** |
| ORM / sqlite3 usage | **None in codebase** |
| Persistence | `franchise_store._SESSIONS` in-memory dict |
| Table row counts | N/A |
| Slow queries | N/A |
| Missing indexes | N/A |
| N+1 patterns | N/A |
| Screen open triggers DB writes | **No database** |

**UI screens triggering saves/simulation on open (in-memory writes):**

| Screen | Writes on open? |
|--------|-----------------|
| All GET screens | **No** — read-only handlers |
| `POST /api/franchise/advance` | Simulates — not on screen open |
| Scouting POST actions | Mutate `scouting_state` on user action only |

---

## Phase 7 — Long-Term Franchise Degradation

| Checkpoint | Year 1 | Year 5 | Year 10 | Year 20 | Year 40 | Year 80 |
|------------|--------|--------|---------|---------|---------|---------|
| Player count (`league.players`) | 7,985 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Retired players | 0 | NOT MEASURED | | | | |
| Database size | N/A | N/A | | | | |
| Save payload size | 7.24 MB | NOT MEASURED | | | | |
| `build_state_payload` time | 5,474 ms | NOT MEASURED | | | | |
| Hub load time | NOT MEASURED | | | | | |
| Roster load time | NOT MEASURED | | | | | |
| Draft Class load time | NOT MEASURED | | | | | |
| Player Profile load time | NOT MEASURED | | | | | |
| Simulation day speed | NOT MEASURED | | | | | |
| Simulation season speed | NOT MEASURED | | | | | |

**Partial simulation:** Automated season advancement ran **4 seasons** in **632,601 ms** (~10.5 min) before audit budget ended. Checkpoints at Years 5, 10, 20 were not captured because offseason/draft blocking stages prevented reliable fast-forward within time limits.

**Architectural expectations (code-based, not measured):**
- `generate_next_season` appends to `season_history` and retains `league.players` / `retired_players_archive`
- Per-season `game_results` and `player_season_stats` are **cleared** each year (`franchise_offseason.py:965-966`)
- Payload size at Year 1 is already dominated by static prospect pools, not game history

**Degradation curve:** NOT MEASURED for Years 5–80.

---

## Phase 8 — Frontend Request Waterfalls

### App cold start (saved session)

```
0 ms     — GameUIProvider mount
~10 ms   — GET /api/health
~5,510 ms — GET /api/franchise/state (syncFranchiseSessionWithBackend)
~11,020 ms — GET /api/franchise/state (refreshFranchise)  ← DUPLICATE
~11,020 ms — Hub renders (franchiseState set)
```

### App new franchise

```
0 ms     — User confirms Setup
~41,816 ms — POST /api/franchise/start (includes full state in response)
~41,816 ms — Hub renders from response state
```

### Trade Hub

```
0 ms     — TradeHub mount
0 ms     — Promise.all starts (parallel)
  ├─ ~5,497 ms — refreshFranchise → GET /api/franchise/state  ← REDUNDANT
  ├─ ~2,731 ms — GET /api/franchise/trade/assets (cold)
  ├─ ~8 ms     — GET /api/franchise/trade/market
  └─ ~6 ms     — GET /api/franchise/trade/history
~5,497 ms — First content if state is bottleneck (parallel)
```

### Scouting

```
0 ms     — Scouting mount
0 ms     — Promise.all (4 requests)
  ├─ ~14 ms    — GET /api/franchise/scouting/state
  ├─ ~4,668 ms — GET /api/franchise/scouting/world  → rebuilds draft rankings
  ├─ ~4,582 ms — GET /api/franchise/scouting/prospects → rebuilds draft rankings
  └─ ~8 ms     — GET /api/franchise/scouting/assignments
~4,668 ms — Fully loaded (parallel, prospects+world overlap)
```

### Draft Class

```
0 ms     — DraftClass mount (draft_class_rankings already in franchiseState)
0 ms     — GET /api/franchise/scouting/prospects (parallel)
0 ms     — GET /api/franchise/trade/assets (parallel)
~4,582 ms — Scouting overlay merge complete
```

### Cap Ledger

```
0 ms     — CapLedger mount
~355 ms  — GET /api/franchise/contract-office
~355 ms  — Tab content interactive
```

---

## Phase 9 — Loading Experience Audit

| Issue | Screen | Evidence |
|-------|--------|----------|
| Backend slow, UI waits on giant state | Hub, all context screens | 5.5 s blocking fetch before `franchiseState` exists |
| Backend fast, potential React slowness | Roster, Stats Central | 0 API on mount but 8k+ line components with heavy useMemos — **render time NOT MEASURED** |
| Duplicate fetch inflates wait | App resume | Two sequential 5.5 s state calls |
| Technically loaded but overlay persists | Trade Hub | `assetsLoading` until all 4 parallel calls finish |
| Blank screen risk | Setup → Start | 42 s with only `loading` flag on begin franchise |
| No granular progress | Franchise start | User sees loading spinner for 42–119 s with no stage text |
| Layout shift risk | Roster headshots | `PlayerHeadshot` / portrait assets load per row |
| WebGL mount on Hub | Hub | `FirstPersonOfficeHub` Three.js canvas loads even when user may navigate away |
| Information dump on state arrival | All context screens | 6.6 MB parsed into React state at once |

---

## Duplicate Work Summary

| Work | Occurrences | Locations |
|------|-------------|-----------|
| `build_draft_class_rankings` | Every state GET + scouting prospects + scouting world + draft class API | `franchise_sim.py:11790`, `franchise_scouting.py:403` |
| `GET /api/franchise/state` | 2× on bootstrap; again on Trade Hub open | `api.js:71`, `GameUIContext.js:133`, `TradeHub.js:5838` |
| `build_trade_assets_payload` | State does not include, but Trade Hub + Draft Class both fetch | Trade Hub, Draft Class |
| `GET /api/franchise/chemistry` | Each visit to Chemistry, Edit Lines, PP, PK | 4 screens |
| Full `roster_browser` serialization | Every state GET | `build_state_payload` |
| `list_teams_summary` SimEngine boot | Every Setup teams load | Discards engine after 32 team names |

---

## Priority Repair Order

### P0 — Critical

1. Cache or incrementally build `draft_class_rankings` inside `build_state_payload` (use existing `get_cached_draft_class_rankings`)
2. Stop calling `build_draft_class_rankings` from scouting endpoints when cache is valid
3. Remove duplicate `GET /api/franchise/state` on bootstrap (merge sync + refresh)
4. Split `GET /api/franchise/state` into lean core state vs. on-demand heavy sections (`roster_browser`, `draft_class_rankings`, `draft_class_hud`)

### P1 — Major

5. Reduce `roster_browser` payload — lazy-load org depth, strip rating groups for non-user teams
6. Remove unconditional `refreshFranchise()` on Trade Hub mount
7. Fix `list_teams_summary` to avoid full `SimEngine` construction
8. Cache `build_trade_assets_payload` across Draft Class + Trade Hub (already session-cached; share via context)
9. Parallelize or defer `bootstrap_full_league_hierarchy` / schedule smoothing at franchise start with progress reporting

### P2 — Moderate

10. Share chemistry endpoint data across Chemistry/Lines/PP/PK screens
11. Frontend: memoize `buildFranchiseStatsLookup` inputs more granularly; avoid full-list sort in Roster when only page displayed
12. Defer WebGL Hub office until user interacts
13. Add gzip compression middleware for JSON responses

### P3 — Minor

14. React StrictMode double-fetch awareness in dev
15. Headshot lazy loading / placeholder sizing in Roster
16. Persist franchise sessions to disk (future) with separate performance model

---

## Final Verdict

1. **Why is the application loading slowly?**  
   Because every load of franchise state rebuilds and transfers a **6.6 MB JSON document in ~5.5 seconds**, and the frontend often requests it **multiple times** while additional screens trigger the same expensive draft-ranking and trade-asset builds.

2. **What is the biggest bottleneck?**  
   **`build_draft_class_rankings()` inside `build_state_payload()`**, invoked on `GET /api/franchise/state`.

3. **Which screen is the slowest?**  
   **Scouting** (~4.6 s API, two ranking rebuilds) and **Trade Hub** (~5.5 s redundant state + trade assets) by API time; **Hub cold start** (~11 s) by user-visible bootstrap. React render ranking: **NOT MEASURED**.

4. **Which API endpoint is the slowest?**  
   **`GET /api/franchise/state`** — **5,497 ms average**, 6.58 MB.

5. **Which backend function consumes the most time?**  
   **`start_franchise()`** at franchise creation (**119 s**); during normal play **`build_draft_class_rankings()`** dominates state serialization.

6. **How large is the franchise payload?**  
   **6.58 MB** over HTTP (6,580,881 bytes); **7.24 MB** uncompressed in Python serialization.

7. **Does SQLite contribute significantly?**  
   **No.** The application does not use SQLite.

8. **Does React contribute significantly?**  
   **NOT MEASURED.** Code structure shows expensive normalization on large arrays (Roster, Stats Central) that will add client-side cost after the 5.5 s API wait, but no profiler data was collected.

9. **How much worse does loading become after 20, 40, and 80 years?**  
   **NOT MEASURED.** Season simulation did not reach those checkpoints within the audit time budget.

10. **What should be fixed first?**  
    **Cache `build_draft_class_rankings` in `build_state_payload` and eliminate duplicate full-state fetches on bootstrap** — highest impact for lowest risk.

---

*Temporary audit artifacts (`_audit_temp/`) should be deleted after report delivery. No application source files were modified.*
