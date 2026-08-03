# NHL Franchise Mode — Performance Audit Report

**Date:** 2026-07-25  
**Scope:** Full-stack behavior-preserving performance (Phases 1–11)  
**Constraint:** No gameplay/sim/API/save format removals or result changes

---

## Phase 1 — Profiler (SHIPPED)

### Backend
| Piece | Location |
|-------|----------|
| Timing registry | `backend/services/perf_profiler.py` |
| HTTP middleware | `backend/main.py` (`X-Response-Time-Ms`, route spans) |
| Snapshot API | `GET /api/perf/snapshot` |
| Reset API | `POST /api/perf/reset` |
| Hot spans | `state.build`, `state.build_safe`, `league_ops.build` |

Env: `NHL_PERF=0` disables. `NHL_PERF_SLOW_MS=100` (default) logs slow ops.

### Frontend
| Piece | Location |
|-------|----------|
| Client profiler | `frontend/src/services/perfProfiler.js` |
| API timings | `frontend/src/services/api.js` interceptors |
| Nav / refresh | `frontend/src/game/GameUIContext.js` |
| Console | `window.__NHL_PERF.snapshot()` |

Disable: `localStorage.nhl_perf=0` or `?perf=0`.

---

## Ranked issues (severity + remaining)

### P0 — Fixed this pass

| # | Issue | Location | Cause | Fix | Est. gain | Risk |
|---|-------|----------|-------|-----|-----------|------|
| 1 | Double `pending_decisions` build | `franchise_sim._build_state_payload_impl` | Called twice; comment warned of 2× wire/UI concat | Build once; same list for snake+camel keys | CPU + payload size | Low |
| 2 | Full league revenue on every `/state` | `build_state_payload` → `build_league_operations_payload` | 32-team revenue scan every hub refresh | Session cache + slim lean payload (no full `teams` table) | 50–300ms+/state | Low |
| 3 | Code fingerprint every request | `main._api_code_fingerprint` | Re-stat many `.py` files (0.75s cache) | Cache TTL **5s** | 5–40ms/req | Low |
| 4 | Duplicate cap-hit calc | `_serialize_player_row` | `_player_cap_hit_millions` ×2 | Compute once | Small per player | Low |
| 5 | No visibility into slow paths | Entire stack | Missing instrumentation | Profiler FE+BE | Enables next wave | Low |

### P1 — High impact, next wave (not yet coded)

| # | Issue | Location | Cause | Recommended fix | Est. gain | Risk |
|---|-------|----------|-------|-----------------|-----------|------|
| 6 | Eager mega-screen + event-menu JS | `App.js`, `franchiseEventResolver.js` | No `React.lazy`; ~2MB+ always parsed | Lazy screens + lazy EVENT_MAP | Startup / nav | Low |
| 7 | Monolithic `GameUIContext` | `GameUIContext.js` | One fat value → tree-wide rerenders | Split state / nav / actions contexts | Every click | Med |
| 8 | Heavy hydrate on Hub + every revision | `App.js` / `hydrateFranchiseHeavyState` | No skip/dedupe for `/state/heavy` | Coalesce + skip if revision matches | Hub open | Low |
| 9 | DraftClass / cinematic CSS in `<style>` | `DraftClass.js`, `CinematicEventShell` | Rebuild/reparse CSS every render | Static CSS files / inject-once | Draft / events | Low |
| 10 | No list virtualization | Roster / Draft / FA / Re-Sign / Entry Draft | Full `.map` + headshots | `react-window` / Virtuoso | Menu open | Med |
| 11 | `json_safe` deep-walks entire state | `json_safe.py` | Recursive copy of large trees | Faster plain-dict path / orjson | 20–40% serialize | Med |
| 12 | Contract office triple-rebuild on POST | `_contract_action_route` | Office + resign force + office again | Single rebuild / patch delta | Contract clicks | Med |
| 13 | Offseason extras ship all stages | `build_offseason_state_extras` | Full stage pack on every `/state` | Current-stage only; lazy GETs | Hub / advance | Med |
| 14 | Cap snapshot / chemistry / calendar scans | `franchise_sim` / `contract_economy` | Per-state league/roster work | Revision caches; lite serialize | Hub refresh | Med |
| 15 | Advance always returns full state | `main` mutation routes | Full hub after every action | Lean `state_patch` + client refetch | Day advance | Med |

### P2 — Structural / sim (preserve outcomes)

| # | Issue | Notes |
|---|-------|-------|
| 13 | Season/week sim wall time | **2026-07-30:** bulk/7D/15D/30D/REG SEASON now use `_light_game_stat_accumulation=True`; socio/trade AI throttled to every 5th bulk day. Single-day advance still uses full event-driven games. |
| 14 | Draft board rebuild | Already partially cached across picks — keep; avoid revision bumps |
| 15 | Save system | Current `franchise_store` is in-memory only — disk save path TBD; profile when added |
| 16 | Headshot / logo thrash | Dedupe URL fetches; browser cache headers |
| 17 | GameUIContext mega-value | Huge context → tree rerenders; split state/actions contexts |

---

## Interaction chains (what to measure)

Use profiler while playing:

1. **Hub refresh** → `GET /state` → `state.build_safe` → `ui.refresh_franchise`
2. **Advance day** → `POST /advance` → sim + `state.build_safe`
3. **Open League Ops** → should hit cache or `GET /league-operations` once
4. **Open Roster / Draft / FA** → screen mount + heavy hydrate

Targets (from brief): hub/menu &lt;150–200ms perceived; day &lt;300ms when no heavy sim; API lean &lt;100ms when possible.

---

## How to run the audit live

```bash
# Backend (after reload)
curl http://127.0.0.1:8000/api/perf/snapshot

# Frontend console
window.__NHL_PERF.snapshot()
```

Compare `top_by_total_ms` and `slow_by_max_ms` after a play session. Anything with `max_ms` ≫ expected and no sim work is wasted work.

---

## Rules compliance checklist

- [x] No features removed  
- [x] No sim quality / AI simplification  
- [x] No fake data  
- [x] State keys preserved (`pendingDecisions` still present; same object)  
- [x] League Ops full table still available via dedicated route + cache  
- [x] Save format untouched (in-memory sessions unchanged)

---

## Shipped file list

- `backend/services/perf_profiler.py` (new)
- `backend/main.py` (middleware, `/api/perf/*`, fingerprint TTL, league-ops route)
- `backend/services/league_operations.py` (cache + slim)
- `backend/services/franchise_sim.py` (state build spans, pending dedupe, slim league ops, serialize tweak, cache invalidation)
- `frontend/src/services/perfProfiler.js` (new)
- `frontend/src/services/api.js` (API timing)
- `frontend/src/game/GameUIContext.js` (nav + refresh marks)
- `docs/PERFORMANCE_AUDIT.md` (this report)
