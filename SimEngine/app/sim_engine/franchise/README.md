# Franchise mode (`app.sim_engine.franchise`)

Interactive franchise orchestration lives here. **Game rules** belong in sibling packages (`entities/`, `economy/`, `progression/`, `league/`, `trades/`, `ai/`, etc.). This package wires them into a playable season with API-friendly payloads.

## Layout

| Module | Role |
|--------|------|
| `engine.py` | Thin public facade — re-exports submodule API |
| `engine_core.py` | Session bootstrap, `start_franchise`, core wiring |
| `common.py` | Shared helpers, storyline recording, display utilities |
| `schedule.py` | Schedule mapping, slate helpers, calendar smoothing |
| `serialization.py` | Player/team rows, draft board, game detail, team list |
| `advance.py` | Day/game/season advance, postseason entry |
| `contracts.py` | Cap snapshots and contract-office helpers |
| `state.py` | `build_state_payload`, caches, trade hub, popups |
| `decisions.py` | Pending decisions, storyline choices, retirement decisions |
| `events.py` | Calendar/showcase events, WJC hooks |
| `progression.py` | In-season progression and development hooks |
| `api_bridge.py` | Chemistry report, contract office, draft pick, trade cache |
| `offseason.py` | Offseason stage machine (awards → retirements → draft → FA…) |
| `retirement.py` | Final Skate retirement pass (`ai/retirement_engine.py`) |
| `scouting.py` | GM scouting assignments and prospect views |
| `trade_service.py` | Franchise trade API bridge to `trades/*` |
| `session.py` | `FranchiseSession` mutable run state |
| `calendar.py` | NHL season calendar (days, phases, events) |
| `paths.py` | SimEngine `sys.path` bootstrap |
| `engine_monolith.py` | Pre-split backup (reference only) |

## Backend (`backend/`) — **this is what the live API runs**

The playable franchise API loads **`backend/services/franchise_sim.py`** (full engine) via `backend/main.py`.
It does **not** import this `app.sim_engine.franchise` package for HTTP routes.

| Live (edit these) | Legacy split (reference only — edits here do not change the game) |
|-------------------|---------------------------------------------------------------------|
| `backend/services/franchise_sim.py` | `engine_monolith.py`, `serialization.py`, `api_bridge.py` here |
| `backend/services/franchise_offseason.py` | `offseason.py` here |
| `backend/services/franchise_entry_draft.py` | Do not duplicate Entry Draft execution here |
| `backend/services/franchise_scouting.py` | `scouting.py` here (live combine + CPU scouting) |
| `backend/services/franchise_retirement.py` | `retirement.py` here |
| `backend/main.py` | — |

After starting the API, open `http://127.0.0.1:8000/api/health` and confirm `code.franchise_sim` points at `backend/services/franchise_sim.py` and `code.features` are all `true`.

## Existing SimEngine rules (use these, don’t duplicate)

| Domain | Package |
|--------|---------|
| Players, teams, contracts | `entities/` |
| Cap, trades, waivers, roster AI | `economy/`, `trades/` |
| Aging, dev, regression, retirement odds | `progression/`, `ai/retirement_engine.py` |
| Schedule, playoffs, awards | `league/` |
| Draft lottery / sim | `draft/` |
| Chemistry, morale, injuries | `systems/`, `world/` |
