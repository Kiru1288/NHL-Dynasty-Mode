# CURSOR AGENT PROMPT — LIVE SIM ("SIMTRACKER"), REAL-TIME SPATIAL ICE TRACKING, COACH CONTROLS, CROWD/MOMENTUM, AND ORIGINAL AI BROADCAST COMMENTARY + TTS

**Project:** NHL Franchise Mode (repo folder `NHL-Dynasty-Mode-15`)
**Audience:** a coding agent (Cursor) that has NOT seen this codebase before and needs exact, verifiable instructions.
**Written from:** a direct audit of the repository on 2026-11-09. Every claim below about existing code was checked against the source. Line numbers are approximate and drift as files change, so **always re-locate code by function name with grep before editing** and never trust a line number alone.

---

## HOW TO READ AND USE THIS PROMPT

1. Read the whole document before writing any code. Sections depend on each other.
2. Work in the **stage order in §17**. Every stage has acceptance checks; do not start the next stage until the current stage passes.
3. You may create exactly **two new source files**:
   - `backend/services/player_tracking_sim.py` (the "Player Tracking Sim": live game engine, tracking model, coach controller, crowd/momentum, alerts, commentary, TTS client/worker, and its FastAPI router)
   - `frontend/src/screens/SimTracker.js` (the "SimTracker" screen: rink, bench, scoreboard, feed, audio)
   Any other new file (data JSON, audio assets, model weights, tests) is a **proposal that needs Kiru's approval first**. List it in your stage report under "Proposed new files" and explain why it is needed. §16 has the pre-identified ones.
4. Edits to existing files are allowed **only** where §15 lists them. Each is a small, surgical hook. No refactors, no reformatting, no "while I'm here" cleanups.
5. When something in this prompt conflicts with what you see in the code, **the code wins**. Stop, report the discrepancy, and propose the smallest adjustment. Do not guess.
6. Every stage report must state: files touched, the exact functions touched, what changed, why, how it was verified, and anything uncertain.

---

## §0. NON-NEGOTIABLE RULES

### 0.1 Simulation authority
- **The SimEngine is the sole authority on game outcomes.** For a live game, the authority is the live event loop in `player_tracking_sim.py`, built from the SimEngine's own primitives (§5). The tracking model, crowd, momentum meter, alerts, commentary and TTS are **observers**. They never change scores, stats, players, clocks, winners or saved state.
- Coach inputs from the user are legitimate sim inputs. They act only through the documented injection points in §6, which mirror inputs the engine already accepts (deployment, line usage, goalie, tactics).
- The finished live game must enter the franchise pipeline through the existing event-game cache contract (§4.4), so every downstream system (box score, season ledger, standings, legacy hall, awards, morale, chemistry, injuries, goofy moments, storylines) runs **unchanged**.

### 0.2 Determinism and RNG discipline
- Never add, remove or reorder `rng` calls inside existing engine functions. Every `rng` draw shifts all later draws and silently changes results for the rest of the season.
- Example: `_run_event_driven_game_impl` builds each goal's display clock with `rng.randint(0, 19)` / `rng.choice([0, 15, 30, 45])` and advances the period with `rng.random() < 0.12`. Those values are **not** derived from the real game clock (§1.4.7). **Do not "fix" this in `engine.py`**, because removing the draws changes outcomes. The live engine derives its clock from its own `game_sec_elapsed`.
- Each live game gets its own `random.Random` seeded deterministically (§4.2). Never draw from `session.sim.rng` on a background thread.

### 0.3 Performance and threading
- No live-game work may block the FastAPI event loop or a running calendar advance. Live stepping runs in a bounded worker (§4.6). TTS runs in a **separate process** (§12.7).
- No unbounded queues, no unbounded caches, no per-request model loading.

### 0.4 Compatibility
- Preserve every existing API route, response shape, save field and frontend behavior.
- Fast sim (all current advance paths) must behave exactly as today when Live Sim is not used. Verify with the determinism check in §16.2.
- Nothing that cannot be pickled (threads, locks, sockets, model objects, lambdas) may be stored on the `FranchiseSession`. `franchise_session.py` already special-cases `*_lock` attributes for deepcopy (around line 224), so do not add new ones. Live runtime state lives in a module-level registry in `player_tracking_sim.py`, keyed by `session_id`.

### 0.5 Honesty of data (no fabrication)
- The engine records **no X/Y coordinates, no passes, no zone entries, no faceoff events, no hits as events, no penalty infraction types, no shot locations beyond chance type, and no in-game injuries** (§1.4). Anything spatial you draw is a **model**. It must be labelled "Tracking model" / "est." in the UI and must never be stated as fact by commentary.
- Commentary may only assert facts that exist in the event record: shooter, chance type, outcome, assists, strength, goalie, empty net, screened flag, score, clock, penalties (by duration only), coaching actions, and derived meters that are explicitly labelled.

### 0.6 Cost and licensing
- No paid AI APIs, paid TTS, per-request services or hosted inference you do not control.
- Verify the **weights** license, not just the code license, for any model (§12.2).
- No celebrity or real-broadcaster voice cloning, ever, including Gary Thorne. No NHL broadcast audio as training or reference material.
- Do not bundle copyrighted audio. Note: `frontend/src/soundtrack/` currently contains a FIFA World Cup OST track and a Super Mario 64 track, imported in `App.js` (setup theme) and `DraftClass.js`. **Flag this in your first report as a release blocker.** Do not touch it unless asked.

### 0.7 Workflow hazards specific to this repo
- `backend/services/franchise_store.get_session()` **expires every session when the backend code revision changes** (`_LIVE_CODE_REVISION` / `code_revision`). Every backend edit while the API is running invalidates open franchises. Expect "Unknown or expired franchise session" during development. Restart via `start_api.ps1` and re-create a test franchise. This is not a bug in your code.
- The API runs with auto-reload **off** (`start_api.ps1`, opt-in `-Reload`). Backend changes go live only after a restart.
- The dev frontend is CRA (`react-scripts 5`), on React 19, at `http://localhost:3000`, proxied to the API on port 8000 via `src/setupProxy.js`.
- Windows host. Use `pathlib`. Never hardcode `/` paths or shell out to Unix-only tools from Python.

---

## §1. VERIFIED ARCHITECTURE AUDIT (WHAT EXISTS TODAY)

### 1.1 Repository layout (relevant parts)
```
NHL-Dynasty-Mode-15/
  backend/
    main.py                         FastAPI app (~2,730 lines). Routes, mostly inline; mounts routers at the bottom.
    start_api.ps1                   Starts uvicorn on :8000 from backend/ using venv; reload is opt-in.
    requirements.txt                fastapi 0.135, starlette 1.0, uvicorn 0.43, pydantic 2.12 (nothing else).
    services/
      franchise_sim.py              Franchise orchestration (~29k lines): day loop, game slots, state payload.
      franchise_store.py            In-memory session dict: get_session / save_session / code-revision expiry.
      franchise_session.py          FranchiseSession object (pickle-safe; *_lock special-casing).
      sim_guard.py                  exclusive_sim(session_id): one sim per session, non-blocking lock -> 409.
      legacy_hall.py                rivalries(session, tid) (history-based rivalry score 0-100).
      playoff_preview.py            RIVALRIES dict: named rivalries by abbreviation pair.
      prospect_rights_talk_api.py   Router pattern example (APIRouter + _session(sid) helper).
      creation_report.py            Router pattern example.
  SimEngine/app/sim_engine/
    engine.py                       The simulation (~21k lines). All real game logic is here.
    gameplay/game_analytics_ledger.py  Chance types, xG, block / on-goal multipliers, goal probability.
    gameplay/*.py                   Mostly EMPTY placeholder modules (0 lines): game_sim, shot_model, shift_model,
                                    penalties, possession_model, scoring, special_teams, stat_accumulator,
                                    goalie_model, physicality. faceoffs.py has code but is not used by the event loop.
    world/{momentum,fatigue,morale,chemistry,injuries}.py  Season-level world modifiers (not in-game).
    league/game_rules.py            team_fan_confidence(team) etc.
  frontend/src/
    App.js                          Screen switch (lazy screens inside GameCanvas); setup soundtrack hook.
    game/constants.js               SCREENS enum; GAME_W = 1280, GAME_H = 720 (fixed logical canvas, scaled to fit).
    game/GameUIContext.js           Global UI state; onAdvanceFranchise; breaking-news toast (zIndex 11000);
                                    playGlobalBreakingSting (WebAudio oscillator precedent).
    screens/CalendarScreen.js       Calendar (~7k lines). Hero "Next game" card + Advance Day / +15 Days / Sim to Date.
    components/boxscore/BoxScoreArcade.js  TEAM_COLORS for all 32 clubs + exported teamPalette(team).
    components/PlayerHeadshot.js    Headshot component (use it for player tokens' portraits in the bench panel).
    components/broadcast/BroadcastTicker.jsx  Bottom ticker precedent.
    events/worldJuniors/WorldJuniorsBroadcastPanels.js  Audio element + mute/play precedent.
    services/franchiseService.js    axios API wrappers (api.get/post with X-Franchise-Session header injected by api.js).
    services/api.js                 SESSION_STORAGE_KEY = "nhl_franchise_session_id"; header injection.
    styles/game.css                 ~84k lines. Calendar arcade tokens (ccx, cp, tx, pxi, kd). See §13.
```

### 1.2 How a franchise game day runs today
1. `POST /api/franchise/advance` (`main.py: post_franchise_advance`) takes `sim_guard.exclusive_sim(session_id)`. If the lock is busy it returns **409**. It calls `_post_franchise_advance_impl`, which may first return a lineup gate (`franchise_sim.user_lineup_advance_gate`).
2. Day advance reaches `franchise_sim._simulate_slots_for_day` and, for each scheduled game, `franchise_sim._simulate_franchise_slot(session, slot)`.
3. Inside `_simulate_franchise_slot`, before scoring:
   - strength scales are built from `world_momentum.team_strength_modifier`, `world_chemistry.team_strength_modifier`, `world_fatigue.team_fatigue_strength_factor`, `world_morale.team_morale_strength_factor`, `sim._roster_injury_depth_penalty`, and `storyline_coverage.apply_matchup_to_scales`
   - noise comes from `session.chaos_index` and `sim._identity_runner_strength_noise_factors`
   - `world_fatigue.tick_roster_fatigue_for_game(...)` runs for both teams
   - `_attach_franchise_saved_lineups(session, home, away, home_id=..., away_id=..., user_tid=...)` attaches the user's saved lines
   - `use_light = not (is_user_game and bool(getattr(session, "event_sim_user_games", False)))`. **User games use the LIGHT path by default.** The event-driven path only runs if `session.event_sim_user_games` is truthy.
4. `score_res = _franchise_resolve_game_score(session, sim, r, home, away, hid, aid, d, ..., light_mode=use_light, home_strength_scale=..., away_strength_scale=..., noise_scale=..., is_user_game=...)`:
   - seed = `zlib.crc32(f"{session_id}|{season_calendar_year}|{hid}|{aid}|{day}") & 0xFFFFFFFF`
   - picks starters with `sim._gm_pick_game_starters(rng, home, away, calendar_day=..., home_b2b=..., away_b2b=..., is_playoff=...)` and locks them on `sim._locked_home_goalie` / `sim._locked_away_goalie`
   - calls `sim._simulate_game(rng, home, away, smap, **kw)` (attempt 1 uses the **passed rng**, which is `sim.rng`). On failure it retries with `random.Random(seed ^ 0x9E3779B9)` in light mode, then falls back to `sim._simulate_game_strength_ex`.
   - returns a `ScoreResult` (NamedTuple in engine.py: `home, away, overtime, ot_periods, ot_sec, mu, en_home, en_away, pulled_tie, shootout=False, shootout_winner_home=None, tie=False, source="light", home_goalie_id="", away_goalie_id=""`)
5. Post-score in `_simulate_franchise_slot`: `world_momentum.update_momentum_after_game`, `world_chemistry.update_after_game`, `chemistry.apply_post_game_familiarity`, `world_morale.update_after_team_result` per player, injuries (`world_injuries.maybe_injure_roster_subset`, which rolls **after** the game and is surfaced via `_franchise_log_injury_and_ui`), then stats.
6. Stats: `_accumulate_franchise_game_stats(...)` sets `light_stats` and calls `sim.accumulate_unified_game_stats(rng, home, away, hid, aid, hg, ag, ot, ledger, build_game_payload=..., light_mode=..., ...)`.
   - With `light_mode=True` it runs `_accumulate_light_strength_game_stats`.
   - Otherwise it **looks for a cached event game**: `cache_key = (str(hid), str(aid), int(hg), int(ag), bool(ot), False)` in `self._pending_event_games` (dict) or `self._pending_event_game` (single). On a hit it uses `cache["result"]` and merges `cache["scratch"]` into the season ledger with `self._merge_game_ledger(ledger, scratch)`. On a miss it runs a fresh `_run_event_driven_game`.
7. `_simulate_game` (engine.py, about line 19301) in event mode: computes tactics (`_tactic_game_goal_mults(..., event_path=True)`), runs `_run_event_driven_game(rng, home, away, hid, aid, scratch, ...)`, then stores `entry = {"_key": (hid, aid, hg, ag, ot, False), "result": result, "scratch": scratch}` in **both** `self._pending_event_game` and `self._pending_event_games[entry["_key"]]` (cleared when it grows past 64 entries). With `return_meta=True` it returns `self._score_result_from_event(result, home, away, hid, aid, is_playoff=...)`.

**This cache contract is how a live game plugs in (§4.4).** Produce a `result` dict with the same keys as `_run_event_driven_game` plus a `scratch` ledger, insert them into `_pending_event_games` under the same key, and hand the slot a `ScoreResult` built by `_score_result_from_event`. The rest of the pipeline then runs unchanged.

### 1.3 The event-driven game (`engine.py: _run_event_driven_game_impl`, about lines 15156–16253)
Measured on this repo: **about 123 ms median, 496 ms worst** per full game on a VM CPU, about 115 shot attempts and about 7 goals per game.

Setup, in order:
- `home_dressed, home_gl, home_scratches, _ = self._gm_build_dressed_lineup(home, rng)` (same for away)
- `home_units = self._gm_build_game_units(home_dressed, home)`, which returns a dict with keys `lines` (4 forward lines), `pairs` (3 D pairs), `pp1`, `pp2`, `pk1`, `pk2`, `fw`, `defs`, `deployment_style` (from `_gm_deployment_style(team)`: `"balanced" | "rolling_four" | "top_heavy"`), `pp1_share` (from `team._user_pp_strategy`), `pk1_share` (from `team._user_pk_strategy`). `_shift` is added at runtime.
- `home_toi = self._gm_allocate_conserved_toi(rng, home_dressed, home)` (pre-allocated per-player TOI seconds; **not** derived from events)
- starters via `_use_starter(...)` → `_gm_select_starting_goalie(...)` unless given or locked
- `self._gm_prime_game_player_caches(...)`, `team._gm_cached_hubs`
- chemistry: `chem_h = self._chemistry_game_modifier(home, situation="even")`
- penalties are **pre-generated** with `self._gm_generate_penalties(rng, dressed, tid, ledger, intensity=noise_scale)`, which returns `(pim, ppo_for_opponent, pp_seconds_for_opponent, intervals)`. `intervals` is a list of `(start_sec, duration_sec)`. The offender gets PIM immediately via `_gm_ledger_add(..., pim=mins)`. Durations come from `rng.choices([2, 4, 5], weights=[0.78, 0.16, 0.06])`. **No infraction type is recorded.**
- the attempt environment comes from `_gm_team_attempt_environment` / `_gm_regulation_attempt_split` (`h_attempts`, `a_attempts`); `ev_mean_delay = (3600*0.91 / attempts) * 1.12`

Main loop (inner function `_simulate_segment(h_attempts, a_attempts, track_goals, sudden_death, chronological, ev_mean_delay)`):
- `game_sec_elapsed` advances by sampled delays; the loop breaks at `_REGULATION_GAME_SECONDS = 3600`.
- manpower comes from `_manpower_at(t)` and `_resolve_manpower_state(t)`, returning `"ev" | "home_pp" | "away_pp" | "four_four"`, from the pre-generated penalty intervals.
- PP possession: in a PP state, `r < 0.10` gives a shorthanded attempt for the killers, `r < 0.64` gives `cycle_only` (an 8–12 s possession tick with no attempt), and otherwise a PP attempt.
- side selection: in EV, home attacks with probability `home_share = attempts_home / (attempts_home + attempts_away)`.
- on-ice units: `atk_unit, _ = self._gm_on_ice_unit(atk_units, strength, rng)` and `def_unit, _ = self._gm_on_ice_unit(def_units, "PK" if PP else ("PP" if SH else "EV"), rng)`. **The unit is sampled per event, not taken from a shift clock.** In EV, `_gm_on_ice_unit` keeps a pseudo-shift in `units["_shift"] = {"li", "pi", "left"}` for 0–2 extra events, and draws line/pair indices from `_EV_LINE_SHARES[style]` / `_EV_PAIR_SHARES[style]`. It stamps `_gm_shift_line_idx` / `_gm_shift_pair_idx` on players.
- shooter: `driver = self._gm_pick_sequence_driver(rng, atk_unit, atk_team, last_driver=..., strength=...)`, then `shooter = self._gm_pick_shooter_from_unit(rng, atk_unit, "", strength, atk_team, driver, ledger=ledger, team_id=tid)`
- chance: `chance = pick_chance_type_for_shooter(rng, strength, self._gm_shot_profile(shooter), quality_bias=unit_bias, team_mix=_game_chance_mix(atk_team, def_team, strength))`, using one of the `CHANCE_TYPE_RAW_XG` keys: `LOW_DANGER_PERIMETER .035, POINT_SHOT .045, RUSH_MEDIUM .095, SLOT .125, HIGH_DANGER_SLOT .185, NET_FRONT .210, REBOUND .195, ONE_TIMER .155, PP_SLOT .165, PP_ONE_TIMER .195, SH_RUSH .110`
- xG: `raw_xg = raw_xg_for_chance(chance, rng)`, adjusted by `_team_offense_skill`, `_team_defense_suppression`, game form, chemistry, `_gm_chance_context` (screens, tags), and `_gm_goalie_save_adjustment`
- outcome: block probability (`chance_block_mult`, `_gm_block_weight`), on-goal probability (`chance_on_goal_mult`), goal via `resolve_goal_probability`. The outcome string is `"BLOCKED" | "MISSED" | "SOG" | "GOAL"`. `rec_xg = unblocked_attempt_xg(...)` (0 when blocked).
- crediting: `credit_shot_attempt_event(...)`; on-ice cf/ca/ff/fa/xgf/xga via `_gm_ledger_add` for both units; goalie shots-against/GA recorded in ledger rows (`goalie_shots_against`, `goalie_ga`, `goalie_toi_sec`, `goalie_xga`)
- assists: `self._gm_pick_assists_from_unit(...)`; `credit_assist_xa`
- each goal appends a dict to `scoring_events`: `for_team_id, period, clock, scorer, scorer_id, assists, assist_ids, assist_count, strength, goalie_in_net, defending_goalie_id, empty_net, chance_type, screened, unit_chemistry, context_tags`
- PP rebound continuation via `_pp_rebound_next`
- `sudden_death` returns as soon as `reg_home != reg_away`

Overtime and shootout (after regulation):
- **Regular season:** one sudden-death segment with `ot_attempts = max(2, round(randint(4, 8) * _board_ot_attempt_scale(rules)))` per side; `ot_len = 60 * rules.get("ot_minutes", 5)`. If still tied: `shootout = True; shootout_home_win = rng.random() < 0.52`. **The shootout is one weighted coin flip with no shooters.**
- **Playoffs:** repeated 20-minute sudden-death periods with `randint(14, 22)` attempts per side, up to 12 periods.
- OT TOI is added to every dressed player (`ot_toi_bonus`).

Post-loop:
- every dressed player: `_gm_ledger_add(ledger, p, tid, gp=1, toi_sec=toi, ...)` with EV/PP/PK TOI splits from `_gm_estimate_special_teams_toi_splits(toi, p, units)`
- goalie rows via `_goalie_rows(...)`: `gp, ga, shots_against, saves, toi_sec`; the starter also gets `gs/starts`, `w`/`l`/`otl`, `so`. Then `self._gm_update_goalie_recent_performance(team, g0, saves, faced_sa)`
- result keys (exact): `away_blocked_attempts_for, away_dressed, away_ff, away_goalie, away_goalies, away_goals, away_missed_shots, away_pim, away_pp_goals, away_pp_seconds, away_ppo, away_shot_attempts, away_sog, away_xgf, display_away_goals, display_home_goals, hockey_away_goals, hockey_home_goals, home_*` (same set), `integrity_issues, ot_periods, ot_seconds, overtime, overtime_away_goals, overtime_home_goals, player_away_goals, player_home_goals, regulation_away_goals, regulation_home_goals, scoring_events, shootout, shootout_home_win`
- `validate_game_integrity` (from `game_analytics_ledger`) populates `integrity_issues`

### 1.4 What the engine does NOT have (do not pretend otherwise)
1. **No spatial data.** No player or puck coordinates, no velocities, no zones beyond what chance type implies.
2. **No continuous shift clock.** Units are sampled per attempt; TOI is pre-allocated, not accrued.
3. **No passes, zone entries, faceoffs, hits, giveaways or takeaways as events.** Hits and blocks land in ledgers as counts; the faceoffs module isn't wired into the event loop.
4. **No penalty infraction names.** Only duration (2 / 4 / 5 minutes) and offender.
5. **No in-game injuries.** Injuries roll after the game.
6. **No shootout rounds.** One coin flip.
7. **Goal clock and period are cosmetic random draws**, not from `game_sec_elapsed`.
8. **No in-game momentum model.** `world/momentum.py` is a season-level team modifier (`get_team_momentum`, `team_strength_modifier`, `update_momentum_after_game`).
9. **No in-game fatigue accumulation.** `world/fatigue.py` is per-player season fatigue (`get_fatigue`, `performance_factor`, `add_game_load`, `tick_roster_fatigue_for_game`). The engine's `_gm_unit_fatigue(unit, share, natural_cap)` computes an OVR penalty for usage above a natural cap, using stamina from `skg_stamina` / `stamina` / `endurance`. The light sim's ice script (`_gm_build_ice_script`: six 10-minute segments that move shares with score, a hot line, fatigue and penalties) is the closest thing to in-game fatigue.
10. **No WebSockets, SSE or streaming** anywhere in the backend. `BackgroundTasks` is used only in `/start`.
11. **No audio settings** (volume / mute) anywhere globally. Audio precedents: `App.js useSetupSoundtrack` (HTMLAudio), `WorldJuniorsBroadcastPanels` (audio element with mute/play), `GameUIContext.playGlobalBreakingSting` (WebAudio oscillator).
12. **No LLM / text generation** in the game. (PuckGPT, `services/puckgpt/*`, is a rule-based Q&A engine, not an LLM.)

### 1.5 Systems you will read (observer inputs)
- **Fan meter:** `session.fan_profiles[team_id]["fan_confidence"]` (0–100, neutral about 55; `franchise_sim._ensure_team_fan_profile`), mirrored to `team._fan_confidence` and read by `app.sim_engine.league.game_rules.team_fan_confidence(team)` (default `FAN_NEUTRAL`). League rows also carry `fan_sentiment` (`league_operations.py`). Labels come from `_fan_category_from_score` / `_fan_heat_label_from_heat` (Calm / Uneasy / Backlash / Furious).
- **Rivalry:**
  - (a) named rivalries in `services/playoff_preview.RIVALRIES` (frozenset of abbreviations → name, e.g. `{"OTT", "TOR"}: "Battle of Ontario"`, `{"EDM", "CGY"}: "Battle of Alberta"`; 15 pairs)
  - (b) division rivals: `home.division == away.division` (`franchise_sim._fan_is_division_rival`, and in `_goofy_on_ice_moments`)
  - (c) history score: `legacy_hall.rivalries(session, tid)` → `{"rivalries": [{"teams": [a, b], "score": 0-100, ...}]}`; needs a playoff series or 12+ one-goal games
  - (d) `league._user_rival_ids` (trade-market rivals, `trades/trade_market_adaptive.py`)
- **Hot / cold:** `franchise_sim` goofy step 10 maintains ON FIRE / ICE COLD streaks; players carry `streak_badge` (`"on_fire" | "ice_cold"`) in serialized rows (around line 8589) and `pl._goofy_fire_mult`. Frontend precedent: `RosterScreen.StreakBadge`.
- **Goofy moments:** `franchise_sim._goofy_on_ice_moments(session, box, compact, home, away, is_playoff=...)` rolls fights (`_GOOFY_P_FIGHT` 0.20 per game, +0.12 for division rivals), brawls, mascot bits, penalty shots, EBUG and flukes **after** the game, with an RNG seeded by game id. Master switch: `session.goofy_events_enabled`.
- **Team colors:** `components/boxscore/BoxScoreArcade.js` exports `teamPalette(team)` with `TEAM_COLORS[abbr] = [primary, secondary]` for all 32 clubs, plus hash fallbacks.
- **Today's game:** state payload `nhl_today` (`franchise_sim._nhl_today_payload`) includes `user_game_today: bool`, `calendar_index`, opponent info. Slots for a day are in `session.by_day[calendar_cursor]` (objects with `home_id`, `away_id`, `day`, `is_playoff`).

### 1.6 API and router conventions
- Routers live in service modules: `router = APIRouter()`, a `_session(sid)` helper that raises 400 when the header is missing and 404 when the session is unknown, return values wrapped with `services.json_safe.json_safe(...)`, and `save_session(s)` after mutations. They are mounted at the bottom of `main.py` with `app.include_router(...)` (see `creation_report` and `prospect_rights_talk_api`).
- Sync route handlers run on Starlette's threadpool.
- Header name: `X-Franchise-Session` (FastAPI param `x_franchise_session: Optional[str] = Header(default=None)`).

### 1.7 Frontend conventions
- `SCREENS` enum in `game/constants.js`. `App.js` lazy-loads each screen and renders `{screen === SCREENS.X && <X />}` inside `<GameCanvas>`.
- Navigation: `useGameUI().setScreen(SCREENS.X)`. Franchise state: `useFranchiseState()`.
- API wrappers: add to `services/franchiseService.js` (axios `api` instance with the session header).
- Fixed logical canvas **1280×720** (`GAME_W` / `GAME_H`), scaled to the window by `useFluidUiScale`. Design at 1280×720 and verify at 1366×768 and 1600×900.
- Root `<html>` has class `cartoon-ui` (`index.js`), which enables the arcade skin (`html.cartoon-ui ...` rules in game.css).
- Already installed: `framer-motion`, `gsap`, `three`, `@react-three/fiber`, `zustand`, `lucide-react`. **Do not add npm dependencies.** The rink is SVG plus DOM; no WebGL is needed.

---

## §2. PRODUCT SPEC: WHAT THE PLAYER EXPERIENCES

### 2.1 Entry point: "Live Sim" vs "Fast Sim"
- When the franchise is on a day where `nhl_today.user_game_today === true` and phase is `regular` (playoffs are Stage 9, see §17), the Calendar hero card (`CalendarScreen.js`, the `nhlcal-hero-actions` block next to **Advance Day**) shows a two-choice prompt:
  - **LIVE SIM** (yellow primary, pixel whistle icon `--pxi-whistle`): opens the SimTracker screen for today's game.
  - **FAST SIM** (cream secondary): the existing Advance Day behavior, unchanged.
- Advance Day, +15 Days, Sim to Date and every other bulk path keep fast-simming the user's games exactly as today. Live Sim is opt-in per game.
- If a live game is already in progress for this session (registry hit), the hero shows **RESUME LIVE GAME** instead.

### 2.2 What happens on Live Sim
1. The frontend calls `POST /api/franchise/live/start`. The backend:
   - runs every **pre-game** step `_simulate_franchise_slot` would run for the user slot (strength scales, fatigue tick, lineups, goalie starters) **without** scoring the game
   - builds a `LiveGame` with a deterministic seed
   - returns the initial snapshot
2. SimTracker opens. The game plays out in pseudo real time on an animated rink. The user can pause, change speed, and make coaching decisions.
3. The game ends (regulation, OT or shootout). The user sees a post-game recap card and presses **FINISH NIGHT**.
4. `POST /api/franchise/live/commit` finishes the calendar day. It hands the live result into the existing pipeline (§4.4), runs the remaining league games for the day, and returns the normal advance payload. The frontend goes back to the Calendar, and the existing box score is available from the day cell.
5. If the user leaves mid-game (Back button, closing the screen, navigating away), the game stays paused in the registry. **No calendar movement happens until commit.** The user can resume. Calendar advance buttons are disabled while a live game is open, and the backend enforces this (§4.7).

### 2.3 Screen layout at 1280×720 (logical px; scale is automatic)
```
┌──────────────────────────────────────────────────────────────────────────────┐ 0
│ TOP BAR (56px): ‹ Calendar | AWAY logo  2  SCOREBUG  3  logo HOME | P2 12:41  │
│  PP 1:22 chip | SOG 18–22 | speed ▶ 1x 2x 4x 8x ⏸ | 🔊 crowd/commentary  │
├───────────────┬──────────────────────────────────────────────┬───────────────┤ 56
│ BENCH (232px) │ RINK STAGE (816×400) + overlays              │ LIVE FEED     │
│ My team only  │  - SVG rink (200×85 ft scaled)               │ (232px)       │
│  L1 L2 L3 L4  │  - 12 player tokens + puck                   │ Alerts/plays  │
│  D1 D2 D3     │  - shot trails, goal burst, chant bubbles    │ newest on top │
│  G (starter)  │  - crowd meter (arc above rink)              │               │
│  per player:  │  - momentum tug bar (below rink)             │ Coach actions │
│  face, #, pos,│                                               │ log (folded)  │
│  energy pips, │                                               │               │
│  hot/cold,    │                                               │               │
│  TOI tonight  │                                               │               │
├───────────────┴──────────────────────────────────────────────┴───────────────┤ 456
│ COACH DECK (180px): [Line call] [Matchups] [Tactics] [Special teams]          │
│ [Goalie] [Timeout] | Tracking tabs: Heatmap · Speed · Distance · Spacing      │
│ CAPTION STRIP (commentary text, always on; TTS optional)                      │
└──────────────────────────────────────────────────────────────────────────────┘ 720
```
- The rink stage is the hero. Everything else frames it.
- At 1280 wide, the side columns are 232px each and the rink column is 816px. The rink SVG keeps a 200:85 aspect inside a 784×333 box, centred with padding for the crowd arc and momentum bar.

### 2.4 Interactive rink
- Hover a token → tooltip card (name, #, position, OVR, energy %, tonight's G-A-SOG-TOI). Hover state comes from the snapshot, not from extra requests.
- Click a token on the user's team → selects that player in the Bench panel and opens quick actions: **Bench next shift / Double-shift / Send to PP1 / Move up a line** (§6).
- Click empty ice → toggles the heatmap of the selected player (or the team).
- Click the puck → highlights the last event's chain (driver → shooter → outcome) as a dotted path. This is labelled "Tracking model".
- Scroll / pinch zoom is **not** required. Keep the rink fixed.

### 2.5 Live alerts the user should see (examples, not literal copy)
- "Brady Tkachuk is gassed: 78% energy, 2 double shifts. Consider sitting him."
- "Stützle is rolling: 2 G, 5 SOG, 1.4 xG tonight."
- "Your PK has allowed 2 PPG. PK1 is running hot."
- "Ullmark: 3 GA on 2.1 xGA. Below expectation."
- "Momentum swing: Toronto 6 straight attempts."
- "Crowd is ROARING: Battle of Ontario, 3–2 home lead."
- Penalties: "Chabot, 2-minute minor. Ottawa shorthanded 1:58." (duration only, §0.5)
- Every alert carries a severity (`info | good | warn | bad`) and an optional suggested action button that maps to a coach control.

### 2.6 Crowd and atmosphere
- The home crowd's volume follows fan confidence, score state, momentum, rivalry and playoff context (§8).
- Rivalry games get chants ("GO SENS GO!", "LET'S GO OTTAWA!" style; generic, no copyrighted chant lyrics). Use the team's city or nickname from team data.
- A goal causes a roar (louder for the home team, muted groans for the away team), plus a horn for home goals (synthesized, §11.7).
- A low fan meter means a quieter building, sarcastic cheers on routine saves, and boos when trailing late.
- All crowd audio is **procedurally synthesized with WebAudio**: no sample files, no licensing risk, and it fits the goofy look.

### 2.7 "Act like a sim"
- This is not an action game. The user never controls skaters. They control the bench and the tactics, then watch outcomes resolve. Speed 1x plays a 60-minute game in about 12 real minutes. Speed 8x plays it in about 90 seconds. **Sim to Intermission** and **Sim to End** jump instantly (the engine steps without pacing).

---

## §3. BACKEND MODULE: `backend/services/player_tracking_sim.py`

One module, organised into clearly separated sections with banner comments. Required section order:

```
# ═══ 0. Imports, constants, feature flags
# ═══ 1. Data contracts (TypedDict / dataclasses) + JSON shapes
# ═══ 2. Registry (module-level, thread-safe, bounded)
# ═══ 3. Pre-game builder (mirrors _simulate_franchise_slot pre-score steps)
# ═══ 4. LiveEventLoop (resumable port of _run_event_driven_game_impl)
# ═══ 5. CoachController (user) + CpuCoach (opponent)
# ═══ 6. TrackingModel (derived spatial layer)
# ═══ 7. MomentumModel, CrowdModel
# ═══ 8. AlertEngine
# ═══ 9. Commentary: context processor, priority/throttle, template writer, pronunciation
# ═══ 10. TTS client (HTTP to worker process) + audio cache index
# ═══ 11. Commit bridge (into the franchise pipeline)
# ═══ 12. FastAPI router
# ═══ 13. TTS worker process entry point  (python -m services.player_tracking_sim --tts-worker)
```

### 3.1 Feature flags (module constants; overridable by env vars)
```
LIVE_SIM_ENABLED = True
LIVE_TRACKING_ENABLED = True
LIVE_COMMENTARY_ENABLED = True       # text captions
LIVE_TTS_ENABLED = False             # off until a TTS worker is configured (§12)
LIVE_TTS_WORKER_URL = "http://127.0.0.1:8765"
LIVE_MAX_GAMES = 8                   # registry cap (one per session in practice)
LIVE_EVENT_BUFFER = 600              # per-game ring buffer of emitted events
LIVE_IDLE_EXPIRE_SEC = 3 * 3600      # stale live games are dropped (never auto-committed)
CROWD_HOME_LIFT_ENABLED = False      # OUTSTANDING DECISION D3 (§18): crowd/momentum may NOT affect sim unless approved
```

### 3.2 Data contracts (send exactly these shapes; add fields only additively)
**LiveSnapshot** (`GET /live/state` and the `/live/start` response):
```json
{
  "live_id": "str",
  "session_id": "str",
  "game_key": {"home_id": "str", "away_id": "str", "day": 0, "is_playoff": false},
  "status": "pregame|live|intermission|overtime|shootout|final|committed|aborted",
  "seq": 0,
  "clock": {"period": 1, "period_label": "1ST|2ND|3RD|OT|2OT|SO",
            "sec_elapsed_game": 0.0, "sec_left_period": 1200.0, "display": "20:00"},
  "score": {"home": 0, "away": 0},
  "shots": {"home": 0, "away": 0},
  "xg": {"home": 0.0, "away": 0.0},
  "manpower": {"state": "ev|home_pp|away_pp|four_four", "home_skaters": 5, "away_skaters": 5,
               "pp_sec_left": 0.0, "empty_net": {"home": false, "away": false}},
  "teams": {"home": TeamCard, "away": TeamCard},
  "on_ice": {"home": [player_id x5|6], "away": [...], "home_goalie": "id|null", "away_goalie": "id|null"},
  "bench": {"user_side": "home|away", "players": [BenchPlayer]},
  "momentum": {"value": 0.0, "label": "str", "window": 12},
  "crowd": {"level": 0.0, "mood": "roar|loud|buzz|quiet|restless|booing",
            "chant": null, "rivalry": {"tier": 0, "name": null}},
  "tracking": TrackingFrameBatch,
  "alerts": [Alert],
  "commentary": [CaptionLine],
  "coach": {"timeout_used": false, "goalie_pulled": false, "tactics": Tactics, "pending": [CoachOrder]},
  "speed": 1,
  "flags": {"tts": false, "tracking": true}
}
```
**TeamCard:** `{id, abbr, name, city, colors: [primary, secondary], logo_token, fan_confidence, fan_label, is_user}`

**BenchPlayer:** `{player_id, name, number, position, slot: "L1C|L1LW|...|D1L|G", ovr, energy: 0-1, energy_label, toi_sec, shifts, g, a, sog, xg, plus_minus, hot: "on_fire|rolling|null", cold: "ice_cold|null", benched: bool, injured_out: bool, penalty_sec_left: 0}`

**LiveEvent** (internal ring buffer; `GET /live/events?since=seq`):
```json
{"seq": 1, "t": 312.4, "period": 1, "clock": "14:47", "type": "attempt|goal|penalty|penalty_end|pp_cycle|line_change|goalie_change|goalie_pull|timeout|period_end|ot_start|shootout_attempt|final|coach_order|alert",
 "team_id": "str", "strength": "EV|PP|SH",
 "shooter_id": "str|null", "driver_id": "str|null", "assist_ids": [], "goalie_id": "str|null",
 "chance_type": "SLOT|...|null", "outcome": "BLOCKED|MISSED|SOG|GOAL|null", "xg": 0.0,
 "screened": false, "empty_net": false, "rebound": false,
 "on_ice": {"home": [...], "away": [...]},
 "score_after": {"home": 0, "away": 0},
 "meta": {}}
```
**TrackingFrameBatch:** see §7.6.
**Alert:** `{id, seq, t, severity, kind, text, player_id|null, action: {kind, label, payload}|null, ttl_sec}`
**CaptionLine:** `{id, seq, t, priority: 0-3, profile, text, tts_text, audio_url|null, expires_at_t}`

### 3.3 Registry (§0.4)
- `_LIVE: Dict[str, LiveGame]` keyed by `session_id` (one live game per session), guarded by `threading.Lock()`. Store a weak reference to nothing; keep only plain data plus references to engine objects already owned by the session.
- Each `LiveGame` has its own `threading.Lock` for stepping. Route handlers acquire it with a short timeout (0.5 s) and return 409 `{"busy": true}` on failure. They never wait indefinitely.
- Eviction: `LIVE_MAX_GAMES` (LRU by last access) plus `LIVE_IDLE_EXPIRE_SEC`. An evicted live game is **dropped, never committed**. The calendar day is untouched, so the user can fast-sim or start again.
- On `get_session()` returning `None` (expired), drop the registry entry.

### 3.4 Pre-game builder: `build_live_game(session) -> LiveGame`
Replicate exactly the pre-score part of `franchise_sim._simulate_franchise_slot` for the user's slot today. **Do not copy-paste blindly.** Call the same helpers, in the same order:
1. Find the slot in `session.by_day[session.calendar_cursor]` where the user team is home or away. If there is none: 400 `{"reason": "no_user_game_today"}`.
2. Run the same guards the advance path runs: `user_lineup_advance_gate(session, auto_fill=False)` must return `None`, otherwise return its block payload so the UI can route to Edit Lines. Respect `goalie_shortage`.
3. Compute `h_scale`, `a_scale` and `noise_scale` exactly as the slot does (world momentum, chemistry, fatigue, morale, injury depth penalty, `apply_matchup_to_scales`, chaos noise, identity noise). **Read only.** These calls must not mutate state. If one does mutate (verify `apply_matchup_to_scales`), call it at commit instead, not at start, and use a read-only estimate for the live run. Document the finding.
4. **Do NOT call `world_fatigue.tick_roster_fatigue_for_game` at start.** That is a state mutation the slot performs. It must happen exactly once, at commit, through the normal slot code path (§4.4).
5. Pick starters: `sim._gm_pick_game_starters(rng_pre, home, away, ...)` with `rng_pre = random.Random(seed ^ 0x51A7E)`. The user can override the starter in pregame (§6.2).
6. `_attach_franchise_saved_lineups(session, home, away, home_id=hid, away_id=aid, user_tid=user_tid)`, then `_gm_build_dressed_lineup` and `_gm_build_game_units` for both teams (§1.3) using the live rng.
7. Seed: `seed = zlib.crc32(f"{session_id}|{season_calendar_year}|{hid}|{aid}|{day}|live").` Same inputs as the fast path plus a `|live` salt, so the live game is deterministic for the same coaching inputs but is not expected to reproduce the fast-sim score.
8. Penalties: pre-generate with `sim._gm_generate_penalties(rng, dressed, tid, scratch, intensity=noise_scale)` for both sides, exactly as the engine does, into the live **scratch** ledger (never the season ledger).
9. Attempt environment: call the same helpers the engine uses to get `h_attempts` / `a_attempts` / `ev_mean_delay` (find them around `_gm_team_attempt_environment` and `_gm_regulation_attempt_split`, and how `_run_event_driven_game_impl` calls them). Apply `_tactic_game_goal_mults(home, away, home_b2b, away_b2b, event_path=True)` to the strength scales exactly as `_simulate_game` does before calling the event game.

### 3.5 Caching inside a live game
- Call `sim._gm_prime_game_player_caches(home, away, home_dressed, away_dressed)` once at start.
- The engine uses `self._gm_begin_game_scope()` / `self._gm_end_game_scope(owned)` around a game. A live game spans many requests, so **wrap each step batch in begin/end scope** and re-prime caches if the scope reset clears them. Read both functions and document exactly what they clear.
- `sim._gm_current_ledger` is set by the engine for hot-hand lookups. Set it to the **season** ledger (`session.player_season_stats` / the sim's ledger, as the slot does with `sim._gm_season_ledger_ref = session.player_season_stats`) at the start of each step batch, and restore the previous value at the end of the batch.

---

## §4. THE LIVE EVENT LOOP (RESUMABLE, AUTHORITATIVE)

### 4.1 Why a port, not a wrapper
`_run_event_driven_game_impl` runs a whole game in one call, and its loop state lives in local variables and closures. Live play needs to pause between attempts, accept coaching changes, and resume. **Do not refactor `engine.py` to make it resumable.** It is 1,100 lines of carefully calibrated code, and the fast path must stay byte-identical.

Instead, `LiveEventLoop` is a **faithful, resumable port** of the main loop body that calls the same engine methods and the same `game_analytics_ledger` functions, so the formulas are not re-implemented. Read `_run_event_driven_game_impl` in full before writing it. Port its logic decision for decision; drop only the audit-only instrumentation (`_audit_pp`, `_pp_own` timelines, `_audit_*` helpers, `_shot_tally` debug counters).

### 4.2 Loop state (all on the LiveEventLoop instance, pickle-free is fine since it is not saved)
`rng` (seeded, §3.4), `game_sec_elapsed`, `period` (derived: `1 + int(game_sec_elapsed // 1200)` in regulation), `reg_home`, `reg_away`, `ot_home`, `ot_away`, penalty intervals for both sides (plus live additions §6), `home_units` / `away_units`, dressed lists, goalie in net per side (or `None` when pulled), `last_driver_by_team`, `_pp_rebound_next`, `scoring_events`, team event stats (`cf, ff, sog, goals, xgf, missed, blocked_for` per side as in the engine), the **scratch ledger** (dict, same schema the engine's `ledger` uses, written only via `sim._gm_ledger_add` / `_gm_ledger_ensure`), per-player **live shift accounting** (§4.5), `event_seq`, and an emitted-event ring buffer.

### 4.3 One step = one engine iteration
`step()` executes exactly one iteration of the ported loop:
1. Resolve the manpower state at `game_sec_elapsed` from the penalty intervals. This includes coach-induced changes: goalie pulls add an attacker, and live penalties are appended in the order they occur. **No new penalties are generated beyond the pre-generated intervals in regulation.** Only the pre-generated ones exist. See §6.6 for why.
2. Branch exactly as the engine does: an EV side draw using `home_share`, the PP branch (`<0.10` SH chance / `<0.64` cycle / PP attempt), and the four-on-four coin flip.
3. **Unit selection is the coaching injection point (§6.1).** In place of the bare `self._gm_on_ice_unit(units, strength, rng)` call, call `LiveUnitSelector.select(side, strength)`. It must:
   - (a) consume the **same rng draws** as `_gm_on_ice_unit` when no coaching override is active, so an untouched live game stays statistically identical to the engine (enforced by §16.1)
   - (b) apply user overrides when present, and CPU coach logic for the opponent
   - (c) update `units["_shift"]` the same way the engine does
4. Driver → shooter → chance → xG → outcome → crediting → assists → scoring event, **calling the same engine methods in the same order with the same arguments**, including `_game_chance_mix` caching and the form/chemistry/context adjustments. If a method is a closure inside the engine function (e.g. `_game_chance_mix`, `_record_team_attempt`), port it as a private method on LiveEventLoop with identical math, and keep a comment pointing at the engine line it mirrors.
5. Advance `game_sec_elapsed` by the same delay distribution the engine uses (EV: based on `ev_mean_delay`; PP cycles: `uniform(8, 12)`).
6. Build the goal's display clock from `game_sec_elapsed` (`period = ...; mm:ss left in period`). In `scoring_events[*]`, write `period` and `clock` **from the real clock**. Also write `"clock_source": "live"` so downstream code can tell. Check that `franchise_sim._periods_from_scoring_events` / `stamp_game_period_lines` accept real clocks. They parse `period`, so verify and report.
7. Emit `LiveEvent`s (attempt / goal / penalty / pp_cycle) into the ring buffer with `on_ice` filled from the selected units.
8. Update live shift accounting, momentum, crowd and alerts (observers; they read the event and never feed back unless an approved flag allows it, §8.4).

Regulation ends when `game_sec_elapsed >= 3600`. Period boundaries at 1200 and 2400 s emit `period_end`, and the loop status becomes `intermission` until the client resumes (or auto-resumes after 3 real seconds at any speed > 1x).

### 4.4 Overtime, shootout and the commit bridge
- **Regular-season OT:** port the engine's block exactly: attempts per side `max(2, round(rng.randint(4, 8) * _board_ot_attempt_scale(self._board_game_rules())))`, `ot_len` from rules, sudden death. Units: use `sim._gm_ot_unit(rng, units, sk, is_playoff=...)` (3-on-3) for OT unit selection. Real clock: spread the attempts over `ot_len` with the same delay logic scaled to `ot_len / attempts`.
- **Shootout (decision D4, §18).** The engine's shootout is `rng.random() < 0.52`. Options:
  - **(A, default)** keep the coin flip as the authority. Draw it at the same point in the sequence. Then **present** it as a shootout with shooter picks that are **consistent with the decided winner**, picking shooters and goalies by attributes and labelling the result "decided".
  - **(B, needs approval)** a round-by-round model (best of 3, then sudden death) using each shooter's finishing and the goalie's save skill, calibrated so home win rate ≈ 52%.

  Implement A. Implement B only if Kiru approves D4.
- **Playoff OT:** Stage 9. Port the multi-period loop with 20-minute periods.

**Building the result:** after the final event, replicate the engine's post-loop block **exactly**:
- `ot_toi_bonus` distribution
- the per-player `_gm_ledger_add(... gp=1, toi_sec=..., ...)` with `_gm_estimate_special_teams_toi_splits`
- `_goalie_rows` (port it; mind two goalies per side when a goalie change happened, §6.4)
- `player_goals_*`, `regulation_only_*`, `display_*`, sog sums from the scratch ledger
- every result key listed in §1.3
- `validate_game_integrity(...)` → `integrity_issues`

TOI: the engine pre-allocates TOI with `_gm_allocate_conserved_toi`. In live mode, **TOI comes from live shift accounting** (§4.5), so benching actually reduces a player's TOI. Re-normalize so team skater TOI sums to the same conserved total the engine would produce (call `_gm_allocate_conserved_toi` once at start to get the target total, then scale the live per-player seconds to it). Goalie TOI: 3600 plus OT for the goalie in net, split when changed.

**Commit (in this order, inside `exclusive_sim(session_id)`):**
1. Build `result` and `scratch` as above.
2. `entry = {"_key": (hid, aid, hg, ag, ot, False), "result": result, "scratch": scratch}`; set `sim._pending_event_game = entry`; set `sim._pending_event_games[entry["_key"]] = entry`, honouring the 64-entry clear rule.
3. `score_res = sim._score_result_from_event(result, home, away, hid, aid, is_playoff=False)`.
4. Hand `score_res` to the day loop **for this one slot** through the minimal hook in §15.1. `_franchise_resolve_game_score` checks a one-shot override, `session._live_score_override = {(hid, aid, day): score_res}`, which is popped on use, and returns it without simulating. Everything else in `_simulate_franchise_slot` runs as normal: fatigue tick, momentum, chemistry, morale, injuries, stats via the cache hit, the goofy roll, storylines.
5. Run the normal day advance for the current day only, `advance_franchise_day(session)` (the same function the Advance Day button reaches; confirm the exact one via `_post_franchise_advance_impl` for `mode == "day", count == 1`). The user slot picks up the override, and every other slot sims normally.
6. Verify after advance: the user game's stored box score matches `hg` / `ag`; `processed_game_ids` contains its stable id; the season ledger grew by exactly the scratch totals for the user players. Mark the live game `committed`, drop it from the registry, `save_session(s)`, and return the advance payload `{"step": ..., "state": ...}` exactly like `/api/franchise/advance`.
7. If anything fails before step 5 completes: clear the override and the cache entry, leave the calendar untouched, keep the live game in `final` status, and return 500 with a reason. The user can retry commit or fast-sim the day.

### 4.5 Live shift accounting (needed for fatigue, TOI and the bench UI)
The engine has no shift clock, so the live loop keeps one without changing outcome math:
- When the selected EV forward line or D pair changes between steps, record a **line change** at `game_sec_elapsed`.
- Each player accrues on-ice seconds for the interval between consecutive events while they were in the last selected unit (EV, PP or PK unit as applicable).
- `shifts` = number of contiguous on-ice runs; `double_shift` = the same forward in two consecutive different-line runs (from §6 orders).
- **Energy (0–1) is an observer meter:** start at `1 - world_fatigue.get_fatigue(p)` × 0.4 (pre-game fatigue matters), drain per on-ice second at `drain = 0.0011 * (1.35 - 0.6 * stamina01)` (stamina from the same keys `_gm_unit_fatigue` uses), and recover per bench second at `0.0016`, clamped to [0.15, 1].

  **Sim effect:** energy affects outcomes **only** through the same mechanism the engine already uses for overuse. When the user's deployment pushes a unit's live usage share above its natural cap, apply the `_gm_unit_fatigue(unit, share, natural_cap)` penalty to that unit's offence/suppression weights for its next attempts, exactly as the light sim's ice script does. Find how `_gm_build_ice_script` / `_gm_coach_adjust_profile` apply it and mirror that, rather than inventing a new formula. Without coaching overrides, usage shares match the engine defaults and the penalty is 0, preserving parity.

### 4.6 Pacing and the worker
- The backend does not run a real-time clock. The **client drives time**: `POST /live/advance {"until_t": float}` asks the backend to step until `game_sec_elapsed >= until_t` (or until a stop condition: goal, penalty start, period end, final, or a pending coach order that needs confirmation). It returns all new events and the latest snapshot.
- Each call steps synchronously under the game's lock. Even worst case, a full game takes under 0.5 s, so a 60-second chunk costs a few ms. No background thread is needed for stepping. Starlette's threadpool already isolates the request.
- Speed levels map to how fast the client advances `until_t`: 1x = 5 game seconds per real second, 2x = 10, 4x = 20, 8x = 40. **Sim to Intermission** sets `until_t` to the next period boundary; **Sim to End** sets it to 3600 + OT, with stop conditions disabled except the final whistle.
- **Back-pressure:** the client never has more than one `/advance` request in flight. It requests about 1.5 s of game time ahead of playback so there's no stutter (§11.4).

### 4.7 Guarding the calendar
- While a live game exists for a session (any status except `committed` / `aborted`), `POST /api/franchise/advance` must refuse with 409 `{"reason": "live_game_open", "live_id": ...}`. Implement this as a 3-line check at the top of `_post_franchise_advance_impl` that calls `player_tracking_sim.has_open_live_game(session_id)` (§15.2).
- `POST /live/abort` drops the live game without touching the calendar (confirm in the UI: "Abandon the live game? The night will be fast-simmed when you advance.").

### 4.8 Parity requirement (blocking for Stage 2)
With no coaching inputs, the live loop must match the engine's distributions. Over 300 seeded games for the same team pairs, comparing `LiveEventLoop` run to the end vs `_run_event_driven_game`:
- mean goals per game within ±0.15
- SOG per team within ±1.2
- PP opportunities within ±0.2
- PP% within ±2.5 points
- save % within ±.004
- blocked share within ±2 points
- shooting % by chance type within ±1.5 points
- OT rate within ±2 points
- the integrity validator reports no issues

This is a targeted script (§16.1). It does not run seasons.

---

## §5. ENGINE PRIMITIVES THE LIVE LOOP MUST REUSE (DO NOT RE-IMPLEMENT)
Call these on `session.sim` (the engine instance). Read each before use.

| Purpose | Engine call |
|---|---|
| dressed lineup | `_gm_build_dressed_lineup(team, rng)` |
| units | `_gm_build_game_units(dressed, team)` |
| conserved TOI target | `_gm_allocate_conserved_toi(rng, dressed, team)` |
| starters | `_gm_pick_game_starters(...)`, `_gm_select_starting_goalie(...)` |
| penalties | `_gm_generate_penalties(rng, dressed, tid, ledger, intensity=...)` |
| EV/PP/PK unit draw | `_gm_on_ice_unit(units, strength, rng)` (wrapped, §4.3) |
| OT unit | `_gm_ot_unit(rng, units, sk, is_playoff=...)` |
| sequence driver | `_gm_pick_sequence_driver(rng, unit, team, last_driver=..., strength=...)` |
| shooter | `_gm_pick_shooter_from_unit(rng, unit, "", strength, team, driver, ledger=..., team_id=...)` |
| shot profile / quality | `_gm_shot_profile(p)`, `_gm_shot_quality_weight(p)` |
| team offence / suppression | `_team_offense_skill(team)`, `_team_defense_suppression(team)` |
| chance context | `_gm_chance_context(...)` |
| goalie adjustment | `_gm_goalie_save_adjustment(g, chance_type)` |
| block weight | `_gm_block_weight(p)` |
| assists | `_gm_pick_assists_from_unit(...)` |
| ledger | `_gm_ledger_add(ledger, p, tid, **kw)`, `_gm_ledger_ensure(ledger, p, tid)` |
| chemistry | `_chemistry_game_modifier(team, situation="even")` |
| ST TOI splits | `_gm_estimate_special_teams_toi_splits(toi, p, units)` |
| goalie recent form | `_gm_update_goalie_recent_performance(team, g, saves, sa)` |
| usage fatigue | `_gm_unit_fatigue(unit, share, natural_cap)` |
| coach adjust profile | `_gm_coach_adjust_profile(team, segment_index)` (CPU coach, §6.8) |
| tactics | `_tactic_game_goal_mults(home, away, home_b2b=..., away_b2b=..., event_path=True)` |
| board rules | `_board_game_rules()`, `_board_ot_attempt_scale(...)` |
| result → ScoreResult | `_score_result_from_event(result, home, away, hid, aid, is_playoff=...)` |

From `app.sim_engine.gameplay.game_analytics_ledger`: `CHANCE_TYPE_RAW_XG`, `chance_block_mult`, `chance_on_goal_mult`, `credit_assist_xa`, `credit_shot_attempt_event`, `pick_chance_type_for_shooter`, `raw_xg_for_chance`, `resolve_goal_probability`, `unblocked_attempt_xg`, `validate_game_integrity`. From `app.sim_engine.economy.coaching_ai`: `team_chance_mix`.

---

## §6. COACH CONTROLS (USER) AND CPU COACH (OPPONENT)

Every control becomes a `CoachOrder` `{id, kind, payload, issued_t, effective_from_t, status: "queued|active|expired|rejected", reason}`. The backend validates every order and rejects illegal ones with a human reason, e.g. "Can't pull the goalie while shorthanded". Orders take effect at the **next attempt boundary**. In hockey terms, line orders apply "next whistle", which is the next attempt event.

### 6.1 Line calls (EV)
- `roll_lines` (default): no override. The selector uses the engine draw exactly.
- `shorten_bench {lines: 3}`: forward line index 3 is excluded. Its share is redistributed proportionally to lines 0–2 before the weighted draw. Same for pairs with `{pairs: 2}`. This changes rng **outcomes**, not rng **consumption**. Keep one `rng.choices` call with modified weights so the draw count stays identical.
- `next_line {line: 0..3}` (one-shot): force the next EV forward unit to that line, then go back to the selector. One-shot orders also set `units["_shift"]` so the engine's 0–2 event persistence applies.
- `bench_player {player_id, until: "shift|period|game"}`: remove the player from his line. The replacement is the best available extra or the next-line player at the same position (C → C, then any F; D → D). Emit a `line_change` event and log it. A benched player accrues no TOI.
- `double_shift {player_id}`: the player also takes the slot of the same position on the next lower line for the next 3 EV units of that line.
- `line_match {opp_line: 0..3, my_line: 0..3, mode: "hard|soft"}`: when the opponent's sampled EV line is `opp_line`, the user's line becomes `my_line` with probability 0.85 (hard) or 0.55 (soft). Draw that with an **extra** rng from a separate stream, `self.rng_coach` seeded `seed ^ 0xC0AC`, so the main stream isn't disturbed. Opponent home teams (last change) get counter-matching (§6.8), and the UI explains this ("Away team: matchups are harder, the home team has last change").
- Usage share and fatigue: any order that raises a unit's live usage share above `natural_cap` triggers the `_gm_unit_fatigue` penalty (§4.5). The UI warns before confirming.

### 6.2 Goalie
- Pregame: choose the starter (any healthy dressed goalie). The engine respects a given goalie through the `home_goalie` / `away_goalie` parameters; the live loop uses the user's choice.
- `pull_goalie` (6-on-5): allowed when trailing, in the 3rd or OT, when the user's team is not shorthanded. Effects:
  - the defending goalie of the user's side becomes `None` when they defend (the engine's empty-net branch, `def_goalie is None`, sets `empty_net = True` and uses its xG/goal math; port that branch)
  - the user's attacking unit gets a 6th skater (best available forward by offensive composite) when they attack
  - attack share: increase the user's EV side probability by multiplying `home_share` for that side by 1.18, renormalized, **only while pulled**
  - auto-return after a goal for either side or when `game_sec_elapsed` reaches the end of regulation
- `goalie_change {goalie_id}`: mid-game swap.
  - Ledger: the outgoing goalie keeps the GA/SA so far; the incoming goalie gets `gp=1` and accrues from now.
  - Decision rules for W/L/OTL: the goalie in net at the time of the winning/tying goal per NHL convention (simplification: the goalie in net when the final deciding goal is scored gets the decision; document it).
  - `_goalie_rows` must emit two rows for that side; `gs`/`starts` go to the original starter only.

### 6.3 Timeout (once per game per team)
- `timeout`: recover energy on the current EV forward line and pair by +0.18 (observer energy meter) and reset momentum toward neutral by 35% (observer). Sim effect: if CROWD_HOME_LIFT is off (default), the only sim effect is through energy → usage-fatigue as defined in §4.5 (which is zero unless the user overused lines). Do not add a magic "timeout bonus" to scoring.

### 6.4 Tactics (maps onto existing team tactic settings)
- The game already has tactic plumbing used by `_tactic_game_goal_mults(..., event_path=True)` (the Strategy Board on the Lines screen). Read where team tactics live (`team.tactics` / strategy board fields; grep `_tactic_game_goal_mults` and the Strategy Board backend) and expose the **same fields** as in-game sliders:
  - Forecheck / aggression (low / balanced / high)
  - Defensive posture
  - Shot selection
- An in-game change re-computes the tactic multipliers for **subsequent** attempts by re-running `_tactic_game_goal_mults` with a temporary override of the user team's tactic fields, then restoring them. **Never persist in-game tactic changes to the team's saved Strategy Board** unless the user presses "Save as default" (this calls the existing lines/strategy save endpoint, not a new one).
- If the tactic system cannot be applied mid-game without restructuring, mark tactics read-only in Stage 3 and report back. Do not invent multipliers.

### 6.5 Special teams
- `pp_unit {unit: 1|2|auto}` and `pk_unit {unit: 1|2|auto}`: force or bias the PP/PK unit draw (replace `pp1_share` / `pk1_share` for the draw with 1.0 / 0.0 / the default).
- `pp_swap_player {out_id, in_id}`: edit the live `units["pp1"]` list. Validate position counts (PP allows 4F/1D or 3F/2D).

### 6.6 Penalties: why no new penalties appear
The engine pre-generates penalty intervals before the game (§1.3). The live loop uses exactly those. Coaching does not create penalties (no "goon" button). If Kiru later wants discipline-driven live penalties, that is a separate approved change. Penalty labels show **duration only**: "2-minute minor", "double minor", "5-minute major". **Never invent "tripping" or "hooking".** See decision D5 (§18).

### 6.7 Validation rules (reject with a reason)
- No orders after `final`.
- Pull goalie only when trailing, in P3/OT, and not shorthanded.
- Bench: never below 4F + 2D available for EV; never the last healthy goalie.
- One timeout per game.
- Goalie change only to a healthy dressed goalie.
- Line match requires valid line indices.
- Rate limit: max 6 orders per game minute (anti-spam).

### 6.8 CPU coach (opponent), deterministic, using engine logic
- Line deployment: the engine selector as-is (deployment style from `_gm_deployment_style`).
- Goalie pull: trailing by 1–2 in P3 with ≤ 2:00 left (≤ 3:00 when down by 2) and not shorthanded. Uses `rng_coach`.
- Last change: when the CPU team is home and the user hard-matches, CPU counter-matches its top line against the user's weakest line with probability 0.5.
- Timeout: after an icing-equivalent (not tracked), so instead use "after conceding 2 goals within 3:00", once.
- Use `_gm_coach_adjust_profile(team, segment_index)` (`segment_index = int(game_sec_elapsed // 600)`) to decide how aggressively the CPU deviates from its plan, as the light sim does.

---

## §7. TRACKING MODEL (DERIVED "REAL-TIME SPATIAL ICE TRACKING")

**Label everything in this section "Tracking model" in the UI.** The engine has no positions (§1.4.1). This layer produces believable, consistent motion and analytics **derived from authoritative events, player attributes and on-ice units**. It never changes outcomes, and it uses its own RNG stream: `rng_track = random.Random(seed ^ 0x7EAC)`.

### 7.1 Coordinate system
- NHL rink: 200 ft × 85 ft. Origin at centre ice. `x ∈ [-100, 100]` (goal lines at ±89, blue lines at ±25), `y ∈ [-42.5, 42.5]`. Corner radius 28 ft.
- Home attacks **+x** in periods 1 and 3 and **−x** in period 2 (teams switch ends). OT: same direction as P3.
- Nets at x = ±89. The crease is a 6-ft-radius semicircle. Faceoff dots: centre (0, 0); neutral-zone dots (±20, ±22); end-zone dots (±69, ±22).
- Units sent to the client are feet. The client scales to SVG.

### 7.2 Formation templates (per situation, attacking direction normalized to +x)
Anchor positions for 5 skaters by role (C, LW, RW, LD, RD) and situation. Each is a base point plus a jitter radius. Examples (implement all of these):

| Situation | C | LW | RW | LD | RD |
|---|---|---|---|---|---|
| OZ cycle | (72, 0) | (80, 18) | (78, -16) | (58, 18) | (58, -18) |
| OZ net-front / slot attack | (80, 4) | (70, 20) | (72, -12) | (55, 14) | (55, -16) |
| DZ coverage (defending −x) | (-70, 0) | (-60, 22) | (-60, -22) | (-82, 10) | (-82, -10) |
| Neutral zone 1-2-2 (with puck) | (5, 0) | (0, 20) | (0, -20) | (-25, 10) | (-25, -10) |
| Rush (3-on-2) | (30, 0) | (28, 18) | (28, -18) | (-5, 12) | (-5, -12) |
| PP umbrella 1-3-1 | (75, 0) bumper | (68, 22) half-wall | (68, -22) flank | (60, 0) point | (86, 5) net front |
| PK box | (-72, 10) | (-72, -10) | — | (-82, 8) | (-82, -8) |

PK box uses 4 skaters. 3-on-3 OT: triangle templates. 6-on-5 (empty net): add an extra skater at (84, -6).

### 7.3 Event anchors (chance type → shot origin), fixed table
| chance_type | shot origin region (attacking +x) |
|---|---|
| POINT_SHOT | x 58–64, y −20…20 |
| LOW_DANGER_PERIMETER | x 62–86, \|y\| 22–36, or x 40–58 |
| RUSH_MEDIUM | x 55–70, y −15…15, arriving from the neutral zone |
| SH_RUSH | same as rush, by the shorthanded team |
| SLOT | x 68–78, y −10…10 |
| HIGH_DANGER_SLOT | x 74–82, y −7…7 |
| NET_FRONT | x 82–87, y −5…5 |
| REBOUND | x 82–86, y −8…8, ≤ 2 s after a previous SOG |
| ONE_TIMER | x 66–76, \|y\| 8–20, pass from the opposite flank |
| PP_SLOT | x 72–80, y −6…6 |
| PP_ONE_TIMER | x 64–72, \|y\| 14–22 (the "office") |

The shooter token is driven to the origin. Linemates are pulled toward the situation template. Defenders are placed between the shooter and the net (goal-side), with the nearest defender at 6–12 ft. The puck travels shooter → net for SOG/GOAL, shooter → blocker for BLOCKED (the blocker is the nearest defender, cosmetic), and shooter → wide or boards for MISSED. **Commentary never names the blocker.** Only the event's shooter, assisters and goalie are real.

### 7.4 Timeline generation
Generate keyframes for the interval between consecutive events (`t0 = prev event`, `t1 = this event`):
1. Possession phase: the attacking unit moves from the previous context (the neutral zone after a change, or the DZ after a turnover/save) into the OZ template over the first 40% of the interval, then cycles with small waypoint loops.
2. Build-up: the driver (the event's `driver_id`) carries the puck for the last 25%. The puck follows the driver with a 1.5 ft lead.
3. The shot at `t1`.
4. After the event: save → puck to the goalie and a faceoff dot (cosmetic); goal → celebration cluster at the scorer and a reset to centre at the next event; blocked/missed → puck loose and possession re-drawn by the next event's side.
5. Line changes: departing players skate to the bench door (`x = ±20 · sign, y = -42.5` home bench / `y = +42.5` away bench, i.e. benches on opposite sides) and new players enter from it.

Keyframes: per player `[t, x, y]` at most every 0.5 game-seconds, compressed (§7.6). Speed is capped by an attribute-derived top speed (§7.5). If a path would exceed it, extend the path smoothly instead of teleporting.

### 7.5 Skating analytics (estimated, clearly labelled)
- Top speed (mph) = 17 + 9 × `speed01`, where `speed01` = normalized `skg_speed` / `speed` / `acceleration` ratings (read with `sim._gm_rating_lookup(p, "skg_speed", "speed", default=70)/99`).
- Accel ∝ `skg_acceleration`.
- Distance per shift / game: sum of keyframe segment lengths (ft → miles).
- Avg speed while on ice; max burst; time in OZ/NZ/DZ (from positions).
- Heatmap: 20 × 9 grid over the rink, occupancy seconds per player and per team.
- Line spacing / formation integrity: the mean pairwise distance of the 5 skaters vs the template's (score 0–100).
- Coverage: the nearest-defender distance to the shooter at shot time (model).
- Passing lanes / shooting opportunities: when the puck carrier has a line to the net not intersecting a defender disc (r = 3 ft), mark an "open lane" (visual only).
- Zone-entry pattern: classify each possession's first OZ arrival as carry vs dump. This is derived from the chance type: RUSH_* = carry; possession-start PP cycle = carry; otherwise 70/30 on `rng_track`. **Always mark it "model".**

These appear in the Tracking tabs (§11.6) with an "est." badge. They are never written to the season ledger or used by awards or contracts.

### 7.6 Wire format: `TrackingFrameBatch`
```json
{"t0": 312.0, "t1": 330.5, "dir_home": 1,
 "players": {"PID": {"team": "home", "kf": [[312.0, -10.2, 4.1], [312.5, -8.9, 4.6], ...]}},
 "puck": [[312.0, 0, 0], ...],
 "markers": [{"t": 330.5, "kind": "shot", "x": 74.2, "y": -3.1, "outcome": "SOG", "event_seq": 41}],
 "model": true}
```
Round coordinates to 0.1 ft. Drop keyframes when a player hasn't moved more than 0.5 ft (delta compression). Target under 25 KB per minute of game time.

---

## §8. MOMENTUM AND CROWD (OBSERVERS)

### 8.1 Momentum meter (−1 … +1, positive = home)
Computed after each event over a rolling window of the last 12 attempts:
```
m = 0.45 * (home_att - away_att)/max(1, home_att + away_att)
  + 0.35 * (home_xg - away_xg)/max(0.3, home_xg + away_xg)
  + goal_bumps (±0.35 decaying with half-life 90 game-seconds)
  + pp_bump (±0.10 while on PP)
```
Clamp to [−1, 1] and smooth with an EMA (α = 0.35). Labels: "Even", "Tilting {team}", "{team} pressing", "{team} in full control". A **momentum swing** alert fires when the sign flips and |Δ| > 0.5 within 60 game-seconds.

### 8.2 Crowd level (0 … 1) for the HOME building
```
base   = 0.35 + 0.35 * (fan_confidence/100)            # fan meter
rival  = {0: 0, 1: 0.06, 2: 0.12, 3: 0.20}[tier]       # tier: 1 division, 2 history score ≥ 40, 3 named rivalry
stakes = 0.10 if playoff else 0.04 if late P3 within 1 goal else 0
state  = +0.12 if home leads, -0.08 if trailing by 2+, +0.05 in OT
mom    = 0.15 * max(0, momentum)                        # only home momentum lifts the home crowd
spike  = goal(home): +0.45 decaying (half-life 6 real-time s); big save(home goalie, xg ≥ 0.15): +0.15
level  = clamp(base + rival + stakes + state + mom + spike, 0, 1)
```
Mood:
- `roar` > 0.85
- `loud` > 0.65
- `buzz` > 0.45
- `quiet` > 0.3
- `restless`: trailing late with level < 0.45
- `booing`: home trailing by 3+, or fan_confidence < 30 and trailing

Away goal: the home crowd drops (−0.25) and plays a "groan". A smaller away-fan pocket cheer layer scales with away fan_confidence × 0.3.

### 8.3 Chants (text plus synthesized rhythm)
- Triggered when level > 0.6 and no chant in the last 120 game-seconds; more often in rivalry tiers ≥ 2.
- Templates use team data only:
  - "LET'S GO {NICK}!" (clap-clap clap-clap-clap rhythm)
  - "GO {NICK} GO!"
  - "{CITY}! {CITY}!"
  - in rivalry games: "BEAT {OPP_NICK}!"
  - in a home rout: "WE WANT {N+1}!"
  - after a home save: "{GOALIE_LASTNAME}! {GOALIE_LASTNAME}!"
- A chant shows as a tilted cream sticker bubble above the stands (§13.4) for 4 s, plus a synthesized clap pattern (§11.7).
- No real-world copyrighted chants or songs.

### 8.4 Can crowd or momentum affect the sim? (Decision D3)
Default **NO**. `CROWD_HOME_LIFT_ENABLED = False`. If Kiru approves, the only allowed effect is multiplying the home team's EV attack share by `1 + 0.015 * (level - 0.5)` (≤ ±0.75%). The engine already includes home-ice advantage in its environment, so this would double-count. That is why it's off by default.

---

## §9. ALERT ENGINE (LIVE BENCH INTELLIGENCE)
Run after each event; dedupe by `(kind, player_id)` with cooldowns.

| kind | trigger | severity | suggested action |
|---|---|---|---|
| `fatigue` | energy < 0.55 (warn) / < 0.40 (bad) while on a line in the next rotation | warn/bad | Bench next shift / Shorten bench off |
| `rolling` | tonight ≥ 2 points, or ≥ 4 SOG, or ixG ≥ 0.9; or `streak_badge == "on_fire"` and ≥ 1 point | good | Double-shift / Move to PP1 |
| `cold` | 0 SOG in ≥ 10 min TOI and on_ice xGA ≥ 0.8 | warn | Move down a line |
| `goalie_struggle` | GA − xGA ≥ 1.5 with ≥ 12 SA, or 3 GA on ≤ 8 SA | bad | Goalie change |
| `goalie_hot` | saves ≥ 15 and GA − xGA ≤ −1.0 | good | — |
| `pp_struggle` | 0 for 3+ PP, or ≥ 1 SHGA | warn | Switch PP unit |
| `pk_struggle` | ≥ 2 PPGA | warn | Switch PK unit |
| `momentum_swing` | §8.1 | info | Timeout (if unused) |
| `penalty` | penalty start | info | — |
| `pp_start` / `pp_end` | manpower changes | info | — |
| `crowd` | mood enters roar/booing | info | — |
| `milestone` | a player reaches a round career number **tonight**, read from the season ledger plus live scratch (e.g. 100th goal) | good | — |
| `pull_suggest` | CPU-style rule for the user's side (trailing by 1–2, ≤ 2:00/3:00 left) | warn | Pull goalie |

Text must only use real stats from the scratch ledger or season ledger. Cooldowns: fatigue per player 180 game-s; rolling per player 300 game-s; others 120 game-s.

---

## §10. COMMENTARY ENGINE (TEXT FIRST, VOICE SECOND)
Commentary text is generated **in the backend file** from authoritative events. TTS (§12) only voices text that already exists. The commentator is an **original** persona; working name "Rinkside Reggie" (rename later). **Never imitate, name, or clone Gary Thorne or any real broadcaster.** Never use catchphrases associated with a real broadcaster.

### 10.1 Context processor (`CommentaryContext`)
Before each line is generated, build a read-only context from the live state:
`{period, clock_str, strength_label ("5-on-5", "power play", "4-on-4", "empty net"), score_home, score_away, score_state ("tied", "home up 1", ...), last_goal_t, time_since_last_goal, shots_period, momentum_label (§8.1), crowd_mood (§8.2), rivalry_tier, is_playoff, series_state (if playoff; read from playoff bracket state, never invent), shooter {name, pos, goals_tonight, season_goals_before_tonight, streak_badge}, goalie {name, saves_tonight, sv_tonight}, assisters[], chance_type, xg}`.
`season_goals_before_tonight` is read from the season ledger **as of the start of this game** (never including tonight's goals twice).

### 10.2 Delivery profiles (from the master TTS prompt; keep these exact names)
| Profile | Use | Text rules | Voice mapping (§12.4) |
|---|---|---|---|
| **P0 Routine** | faceoffs, routine saves, line changes, dump-ins | ≤ 9 words, even rhythm | calm, normal rate, low expressiveness |
| **P1 Build-up** | dangerous chances, sustained pressure, PP start, rush | ≤ 14 words, rising energy | slightly faster, medium expressiveness |
| **P2 Big moment** | big saves (xg ≥ 0.20), posts (none, as the engine has no posts, so never say "post"), goalie pulled, penalty shot (none), late tie attempts | ≤ 14 words, short punchy clauses | high expressiveness, faster, louder |
| **P3 Goal / climax** | goals, OT winner, shootout clincher, playoff series clincher | call (≤ 6 words) + 1 follow-up line (≤ 16 words) | max expressiveness for the call only, then drop to P1 for the follow-up |

### 10.3 Throttling and priority
- Each line has `priority` (P3 > P2 > P1 > P0) and `expires_at_t` (game time).
- At most one P0 line per 25 game-seconds. Drop P0 when any P1+ line is queued.
- When a P3 is generated, queued lower-priority lines are dropped and the audio manager interrupts any playing line (§11.7).
- Silence is good. Target 2.5–4 lines per game minute at 1x. Speeds ≥ 4x: P0 never voiced (text only); at 8x only P3 is voiced.
- Captions are always shown, even when audio is muted.

### 10.4 Template grammar (deterministic variation)
- Templates live in a dict in the backend file: `COMMENTARY_TEMPLATES[event_kind][profile] -> list[str]`, with at least **8 variants per (kind, profile)** for frequent kinds (shot_sog, save, blocked, missed, faceoff_won, line_change) and at least **12 variants for goals**.
- Choose a variant with `rng_comm = random.Random(seed ^ 0xC033)` and a no-repeat window of the last 6 variants per kind.
- Placeholders: `{shooter_last}`, `{shooter_full}`, `{goalie_last}`, `{team_city}`, `{team_nick}`, `{opp_nick}`, `{assist1_last}`, `{assist2_last}`, `{period_ord}`, `{clock}`, `{score_line}`, `{goals_tonight_ord}` (e.g. "his second"), `{season_goal_num}` (e.g. "his 23rd of the season").
- Chance-type flavour (templates must match the type; examples of intent, write your own wording):
  - `ONE_TIMER`, `PP_ONE_TIMER`: "one-timer", "hammered it from the left circle"
  - `REBOUND`: "rebound", "second chance"
  - `NET_FRONT`: "jammed home", "right on the doorstep"
  - `RUSH_MEDIUM`, `SH_RUSH`: "off the rush", "odd-man break"; SH_RUSH also "shorthanded"
  - `POINT_SHOT`: "from the point", "through traffic"
  - `HIGH_DANGER_SLOT`, `SLOT`, `PP_SLOT`: "from the slot", "in tight"
  - `LOW_DANGER_PERIMETER`: "from a sharp angle", "from the outside"
- Goal follow-up lines may reference: the score after the goal, assists, the scorer's goals tonight (hat-trick call only when it's truly the 3rd tonight), the season goal number, PP/SH/EN status, game-winning in OT, and the time since the last goal (< 60 s: "two goals in under a minute").

### 10.5 Accuracy rules (blocking)
1. Every name in a line must be a player in this game's event or lineup data.
2. Every number in a line must come from the scratch ledger, the season ledger, or the clock.
3. Never describe things the engine does not model: posts/crossbars, fights, penalty types, specific deflections, a specific player who blocked a shot, hits by name, injuries during the game (unless the engine reports one; it doesn't today).
4. "Rolling"/"hot" lines need the §9 thresholds.
5. Run `CommentaryLinter.check(line, context)` before emitting: regex for digits, every digit run must match a context value; every capitalized name token must match a roster name or team string. Failing lines are dropped and logged (`logger.warning("commentary_lint", ...)`), never emitted.

### 10.6 Pronunciation engine
- Every CaptionLine has `text` (for captions) and `speak` (for TTS) fields. `speak` is built by `PronunciationEngine.apply(text, names_in_line)`.
- Sources, in priority order:
  1. `session.commentary_pronunciations` (a dict `{player_id: "respelling"}`; created lazily; user-editable later; **never written by the sim**)
  2. `PRONUNCIATION_OVERRIDES` dict in the backend file (seed entries for hard names in the shipped rosters; Cursor must scan the rosters for names with diacritics or non-English clusters (Czech/Slovak/Finnish/Swedish/Russian/German) and add respellings)
  3. Rule fallback: strip diacritics for the TTS phonemizer (Č → "Ch", Š → "Sh", Ž → "Zh", Ř → "Rzh", ä → "a", ö → "er", å → "o", ü → "oo", ø → "er"); Finnish "j" → "y"; Czech "ch" kept; "-ov/-ev" endings kept as-is.
- Respelling style: hyphenated syllables with the stress in CAPS, e.g. "pahs-TER-nahk". Keep the spelling `text` correct; only `speak` changes.
- Numbers in `speak`: expand to words (`2-1` → "two to one", `3:41` → "three forty-one remaining" as appropriate, `23rd` → "twenty-third").
- A proposed (approval needed) data file `backend/data/commentary_pronunciations.json` is listed in §16.7.

---

## §11. FRONTEND: `frontend/src/screens/SimTracker.js`

### 11.1 File structure (one file; internal components)
```
SimTracker (default export)            // screen root; props: { sessionId, gameId, onExit }
 ├─ useLiveSession()                   // state machine + polling + commands
 ├─ usePlaybackClock()                 // client playback time, buffering, speed
 ├─ useAudioManager()                  // WebAudio crowd/horn/chants + TTS queue
 ├─ <TopScoreBar/>                     // 56px
 ├─ <BenchPanel side="user"/>          // 232px left
 ├─ <RinkStage/>                       // 816px centre: <RinkSvg/>, <Tokens/>, <PuckLayer/>, <Markers/>, <ChantBubbles/>, <HoverCard/>
 ├─ <FeedPanel/>                       // 232px right: Play-by-play | Alerts | Tracking tabs
 ├─ <CoachDeck/>                       // 180px bottom: lines, goalie, tactics, special teams, timeout, speed
 ├─ <IntermissionCard/> / <FinalCard/> // overlays
 └─ <CommitBar/>                       // "Commit result" / "Abort" at final
```
Export nothing else. Constants such as rink geometry and token radii live at the top of the file.

### 11.2 State machine (`useReducer`)
`idle → starting → pregame → live ↔ paused → intermission → live → final → committing → committed → exit`. Error edges: any state → `error` (shows a retry/abort card). Transitions are only driven by backend responses or explicit user actions. The **commit** button exists only in `final`. Leaving the screen in `live`/`paused` shows a confirm: "Leave the game? Progress stays saved for this session until you abort or commit." (The backend keeps the live game in the registry; §3 / §4.7.)

### 11.3 Data flow
- On mount: `liveStart(sessionId, gameId)` → snapshot. If the backend says a live game already exists for that slot, `liveState()` resumes it.
- Pacing (client-driven, §4.6): every 400 ms of real time, when `buffer_ahead < 6 game-seconds` and not paused, call `liveAdvance({until_t: playhead + 12 * speedMult})`. The response carries new events, frames and alerts since `since_seq`. Never issue overlapping advance calls (keep an in-flight ref).
- Playback: `playhead` advances at `speed * 5` game-seconds per real second (1x = 5 game s/s; options 1x/2x/4x/8x; §4.6). Show events when `playhead >= event.t`. Keep a 1.5 s render buffer so animation never outruns data. If the buffer empties, show a small "buffering" pixel spinner (`cp-spin`) and freeze tokens.
- Coach orders: `liveCoach(order)` → returns order status. Show a toast: accepted ("Next whistle: Line 2"), or rejected with the backend reason.
- Polling fallback: when the advance call fails twice, back off 1 s, 2 s, 4 s (max 8 s) and show "Reconnecting…". After a backend 404 "Unknown or expired franchise session" (sessions expire on a code reload, §0.7), show "Session expired — your franchise reloaded" and an Exit button.

### 11.4 Rendering the rink (`RinkSvg`)
- SVG `viewBox="-100 -42.5 200 85"` (feet), `preserveAspectRatio="xMidYMid meet"`, inside an 816 × 390 box (aspect 200:85 ≈ 2.35). No canvas library.
- Layers: boards (rounded rect r = 28, stroke `--ccx-ink` 1.2 + 0.6 cream inner line), ice fill `var(--tx-ice-dark)` applied to the container div behind the SVG, red centre line (dashed `stroke-dasharray="2 1.2"`, a nod to the double-dash feel), blue lines at ±25 (`--ccx-sky` darkened, width 1), goal lines at ±89, creases (`--ccx-sky` at 35% fill), faceoff circles r = 15 and dots, home/away team colours as tinted end-zone overlays at 8% opacity (`teamPalette`), centre logo = the home team abbreviation in Luckiest Guy as SVG text with an ink stroke.
- Tokens: circle r = 2.2 ft; fill = team primary; 0.5 ft ink stroke; jersey number (or position initials if numbers are missing) in Silkscreen 2.2 ft; the puck = r 0.9 black with a cream glint. Goalies are r 2.6 with a square-ish pixel border.
- Interpolation: rAF loop; position = linear interpolation between keyframes at `playhead`; ease only on line-change entry/exit. **No React state updates per frame.** Write `transform` attributes via refs (`el.setAttribute("transform", ...)`).
- Interactivity:
  - hover a token → `HoverCard` (name, pos, tonight G-A-PTS, SOG, TOI, energy bar, ovr) using `ccx-panel` styling
  - click a token → pin the card plus quick actions ("Bench next shift", "Double-shift", "Move to PP1")
  - click the empty ice → toggle heatmap overlay (team / selected player)
  - drag-select (shift + drag) → measure spacing between selected tokens (shows feet)
  - right-click is not used
  - shot markers: circles (SOG), × (missed), squares (blocked), stars (goals), sized by xG; hover a marker shows chance type plus xG
- `Markers` persist per period with a "clear" toggle.

### 11.5 Bench panel (left)
- Header: user team chip plus "BENCH" kicker.
- Four forward lines plus three pairs plus goalies, each row: number, last name, position, energy bar (8 pixel blocks; lime > 0.7, yellow 0.55–0.7, orange 0.4–0.55, coral < 0.4), tonight points, and a status stamp (`ON ICE`, `NEXT`, `BENCHED`, `HOT`, `TIRED`). The on-ice unit row glows with a 2 px yellow inset.
- Clicking a row opens the same action menu as a token click.

### 11.6 Feed panel (right) tabs
1. **Play-by-play**: newest on top; goals are a full-width yellow card with a `cp-slam` entry; penalties are an orange stamp; P0 lines in muted cream.
2. **Alerts**: §9 alerts with severity stamps and one-tap suggested action buttons (`ccx-btn--sm`).
3. **Tracking** (badge "MODEL"): sub-tabs Speed (top 5 max burst + avg), Distance (per player tonight), Zone time (OZ/NZ/DZ bars per team), Spacing (formation integrity score per line), Lanes (count of open lanes per period), Entries (carry vs dump %). Each value shows an "est." suffix.
4. **Box**: live box score from the scratch ledger (`ccx-table`).

### 11.7 Audio manager (`useAudioManager`)
- One `AudioContext`, created on the first user gesture (the LIVE SIM click counts). Master gain → three buses: `commentaryBus`, `crowdBus`, `sfxBus`. Settings: master volume, mute commentary, mute crowd (persist in `localStorage` key `simtracker.audio.v1` inside try/catch).
- Crowd bed, **procedural, no audio files**: pink noise (AudioWorklet is not required; use a `ScriptProcessor`-free approach: prefill a 4 s noise `AudioBuffer`, loop it) → bandpass (350–1800 Hz) → gain = `0.08 + 0.55 * crowd.level`. Add a second layer (higher band 1.5–4 kHz) for roar, gain `max(0, level - 0.65) * 1.2`. Smooth gain changes with `setTargetAtTime(…, 0.4)`.
- Mood colour:
  - booing: low-pass 500 Hz plus a slow LFO (0.6 Hz) on the gain
  - groan (away goal): a 1.2 s downward filter sweep
- Goal horn: a synthesized horn (3 detuned sawtooth oscillators at 110/165/220 Hz → low-pass 1.2 kHz → envelope 0.05 s attack, 2.4 s sustain, 0.6 s release). Home goals only, plus a crowd spike. Away goals: crowd groan plus an away-pocket cheer (thin noise burst).
- Chants: synthesized claps (short noise bursts, 12 ms, high-pass 1 kHz) in the chant rhythm (§8.3), with the crowd bed ducked −3 dB for the chant length. Text bubble from §8.3.
- Commentary: an `<audio>`-free path. Fetch TTS clips (§12) as ArrayBuffer → `decodeAudioData` → play on `commentaryBus`. Duck the crowd −6 dB while commentary plays (−2 dB for P3 calls so the roar stays big).
  - Queue: priority queue by (priority desc, t asc).
  - A new P3 interrupts the current clip (100 ms fade).
  - Drop clips older than 6 game-seconds at their play time.
  - Cache decoded buffers in a Map keyed by the `audio_key` (§12.6), max 120 entries (LRU).
- If TTS is unavailable (worker down, 503), captions only, no error toasts after the first one.
- Reduced motion (`matchMedia('(prefers-reduced-motion: reduce)')`): no shakes/`cp-slam`/`cp-wobble`; tokens still move (they are information) but the goal celebration is a static banner.

### 11.8 Cleanup
On unmount and between games: cancel the rAF, abort in-flight fetches (`AbortController`), stop all oscillators/sources, `audioCtx.close()`, clear intervals. No global listeners left behind (check with React StrictMode double-mount in dev).

### 11.9 Performance budget
- ≤ 22 skater/goalie tokens plus the puck.
- 60 fps target at 1280×720 on a mid laptop; never exceed 4 ms per frame in JS.
- Feed list virtualized after 150 entries (simple windowing, no new deps).
- Frame batches are merged into a ring buffer covering playhead −30 s … +30 s. Drop older data.

---

## §12. TEXT-TO-SPEECH SUBSYSTEM

### 12.1 Hard constraints (from the master prompt)
- Free / open-source only. **No paid APIs** (no ElevenLabs, no Azure, no Google Cloud TTS, no OpenAI TTS).
- Weights **and** code must allow commercial use. Verify each model's licence file in the downloaded repo and weights card before integrating; record the licence text path in the doc.
- **No cloning of Gary Thorne or any real broadcaster, and no NHL broadcast audio as reference or training data.** Any cloning reference must be a consenting, paid/contracted voice actor recorded for this project (§12.9), or a stock voice shipped with a permissive model.
- No file creation or deletion outside the two target files without Kiru's approval (proposed files are in §16.7).

### 12.2 Model comparison (verified Nov 2026; Cursor must re-verify before integrating)
| Model | Code licence | Weights licence | Cloning | Emotion/energy control | CPU real-time | Verdict |
|---|---|---|---|---|---|---|
| MeloTTS | MIT | MIT | no (base voices) | speed only | yes | CPU tier base |
| OpenVoice V2 | MIT | MIT (since Apr 2024) | tone-colour conversion on top of a base TTS (MeloTTS) | inherits base; limited | yes-ish (extra conversion pass) | optional: give MeloTTS output the voice actor's timbre |
| Kokoro-82M | Apache-2.0 | Apache-2.0 | no (fixed voices) | speed only | yes, very fast | CPU tier alternative; note the phonemizer dependency espeak-ng is GPLv3, so it must be called as a separate program/process, never statically linked or vendored into a closed distribution |
| Chatterbox (Resemble AI) | MIT | MIT | zero-shot from a short reference | `exaggeration` and `cfg_weight` controls | slow on CPU; GPU preferred | **GPU tier: recommended for P2/P3 energy**. Outputs carry an imperceptible PerTh watermark; keep it on |
| XTTS-v2 (Coqui) | MPL-2.0 | **CPML (non-commercial)** | yes | some | no | **EXCLUDED** |
| F5-TTS | MIT | **CC-BY-NC** | yes | some | no | **EXCLUDED** |

### 12.3 Recommendation and decision D1
- **Tier A (GPU available):** Chatterbox, cloned from the contracted voice actor's reference clip (§12.9). Map delivery profiles to `exaggeration`/`cfg_weight` (§12.4). Prerender and cache everything that can be prerendered (§12.5).
- **Tier B (CPU only, default for most players):** Kokoro-82M with a chosen stock voice, or MeloTTS (+ OpenVoice V2 tone conversion toward the voice actor). Energy comes from rate, text shape (short punchy clauses), and post-processing (§12.4).
- Tier choice is runtime: the worker reports `capabilities` (§12.7) and the backend picks. **D1 (Kiru decides): Tier A model, Tier B model, and whether a GPU worker exists at all.** Until decided, build against an abstract `TTSBackend` interface with `kokoro`, `melo`, `chatterbox` adapters, and implement **one** adapter first (Kokoro, CPU) so the pipeline works end to end.

### 12.4 Delivery profile → real controls
| Profile | Chatterbox | Kokoro / Melo | Post-processing (FFmpeg filter chain) |
|---|---|---|---|
| P0 | exaggeration 0.35, cfg 0.5 | speed 1.0 | `loudnorm=I=-20` |
| P1 | exaggeration 0.55, cfg 0.45 | speed 1.06 | `loudnorm=I=-18`, `acompressor` light |
| P2 | exaggeration 0.75, cfg 0.35 | speed 1.12 | `loudnorm=I=-16`, `acompressor=ratio=3`, +1.5 dB presence EQ at 3 kHz |
| P3 call | exaggeration 0.95, cfg 0.30 | speed 1.18; text uppercased emphasis words are split into separate short clips and concatenated with 40 ms gaps | `loudnorm=I=-14`, `acompressor=ratio=4`, presence +3 dB, `aecho=0.6:0.3:40:0.15` (arena slap) |

Only use parameters these models really expose. If a version renames a parameter, read the installed package's signature and adapt. **Do not invent SSML support** (none of these models parse SSML).

### 12.5 Pipeline
1. Backend builds a `CaptionLine {id, t, priority, profile, text, speak, audio_key}`.
2. `audio_key = sha1(f"{model_id}|{voice_id}|{profile}|{speak}")[:20]`.
3. Cache lookup: in-memory LRU (256 clips) → disk cache folder (proposed, §16.7: `backend/data/tts_cache/`; until approved, memory only).
4. Miss → enqueue to the TTS worker (bounded queue, max 24 items; when full, drop the oldest P0/P1 first; P3 never dropped).
5. Worker synthesizes WAV 24 kHz mono → FFmpeg post chain → encode Opus 48 kbps in OGG (fallback MP3 96 kbps if the browser lacks Opus decode, i.e. older Safari) → return bytes.
6. Endpoint `GET /api/franchise/live/tts/{audio_key}` returns 200 with audio, 202 `{"status":"pending"}` while queued, 503 when no worker. The client polls 202 at 250 ms up to 2.5 s, then gives up (caption only).
7. **Prerender pool** at live-game start: P3 goal calls and P2 save calls for all dressed skaters/goalies of both teams ("{shooter_last} SCORES!" variants), around 40–60 short clips. Low priority; never blocks the game.
- FFmpeg: use a system FFmpeg binary called as a subprocess. Document that redistributing FFmpeg needs an LGPL build (no `--enable-gpl` / `--enable-nonfree`) if it's ever bundled.

### 12.6 Audio formats and keys
- Opus/OGG preferred; MP3 fallback. Header `Cache-Control: private, max-age=86400`.
- Never put player names or text in URLs; only the hash key. (Text goes over the internal worker link in a POST body.)

### 12.7 Worker process (separate process, never inside the FastAPI process)
- Entry: `python -m services.player_tracking_sim --tts-worker --port 8765` (in the same backend file, guarded by `if __name__ == "__main__":`). It runs a tiny HTTP server (stdlib `http.server` + `ThreadingHTTPServer`, or FastAPI/uvicorn if that's already a dependency; check `requirements.txt` first) bound to **127.0.0.1 only**.
- Endpoints: `GET /capabilities` → `{models:[...], gpu: bool, voices:[...]}`; `POST /synth {text, profile, voice_id, model_id}` → audio bytes.
- Model load is lazy and once per process. The heavy imports (torch, kokoro, chatterbox) happen **only** in the worker; the FastAPI process must never import them (keeps startup fast and the 409/sim guard unaffected).
- The FastAPI side talks to `LIVE_TTS_WORKER_URL` with a 4 s timeout and a circuit breaker (3 failures → open for 30 s).
- Windows: document how to start the worker in a second terminal; no reliance on `fork`.
- Deployment costs: local CPU tier = free; local GPU tier = free (user's hardware); hosted GPU = paid infra (not an API fee, but still a cost; note it for Kiru).

### 12.8 TTS QA corpus (40 categories, from the master prompt)
Create the corpus as a constant `TTS_QA_CORPUS` in the backend file (moving it to a data file needs approval). Cover: routine saves, faceoffs, line changes, PP start/end, PK kills, rush chances, one-timers, rebounds, net-front goals, point shots, empty-net goals, shorthanded goals, OT winners, shootout goals/saves, hat tricks, milestone goals (100th, 500th), assists naming two players, goalie pulled, goalie changes, timeouts, late tying goals, blowouts, comebacks, rivalry intros, playoff OT, series clinchers, crowd-roar moments, booing moments, intermission summaries, injuries **(skip: the engine has none, mark N/A)**, Czech names, Finnish names, Swedish names, Russian names, German names, French-Canadian names, apostrophe names (O'Reilly), hyphenated names, numbers/scores/times, and 3-goal-lead "WE WANT" moments. Each entry: `{category, profile, text, expected_speak}`. A QA script (§16.6) renders all and reports clip length, peak level, failures.

### 12.9 Voice actor (decision D2)
If cloning is used: a **contracted** voice actor gives written consent covering synthetic use in this game, commercial release, and the specific models. Record: 48 kHz/24-bit WAV, a quiet room, 10–15 min of read hockey-style lines across all four profiles, plus 30 s of neutral reading for the reference clip. Store the consent record with the project (outside the repo if it holds personal data). **No reference clip may come from broadcasts, YouTube, or any real commentator.**

---

## §13. STYLING SPEC (CALENDAR ARCADE / DOUBLE-DASH FEEL)

### 13.1 Rules
- Reuse the existing calendar tokens in `frontend/src/styles/game.css`. **Do not define new colour hex values** except team colours from `teamPalette`. New CSS goes in **one** appended block at the end of game.css, headed:
  `/* ===== SIM TRACKER — KART PASS (live sim screen) ===== */`
- Selector specificity pattern used by the calendar pass: `html:not(#_):not(#__) .st-…`. All classes for this screen use the `st-` prefix. Root element: `<div className="st-root ccx-scope">`.
- Reuse utility classes where they fit: `ccx-panel`, `ccx-head`, `ccx-kicker`, `ccx-title`, `ccx-btn`, `ccx-btn--primary`, `ccx-btn--sm`, `ccx-tabs`/`ccx-tab`/`ccx-tab.is-active`/`ccx-tab__count`, `ccx-chip`, `ccx-stamp` (`is-good`/`is-bad`/`is-warn`), `ccx-table` (`td.is-num`, `is-good`/`is-mid`/`is-bad`), `ccx-choice`/`ccx-choice__opt`.

### 13.2 Token reference (exact values currently in game.css)
- Colours: `--ccx-ink #071233`, `--ccx-cream #ffe7c4`, `--ccx-paper #fff3dc`, `--ccx-deep #050c2c`, `--ccx-cell #0b1844`, `--ccx-cell-2 #10225a`, `--ccx-yellow #ffe23a`, `--ccx-orange #ff8a1f`, `--ccx-coral #ff6b5c`, `--ccx-lime #9be564`, `--ccx-sky #9fd8ff`, `--ccx-soft #a9bfdc`, `--ccx-muted #4b5a82`.
- Fonts: `--ccx-display` (Luckiest Guy), `--ccx-pixel` (Silkscreen), `--ccx-ui` (Nunito).
- Shadows: `--ccx-outline: inset 0 0 0 3px var(--ccx-ink)`, `--ccx-drop: 5px 5px 0 rgba(5,12,44,.7)`, `--ccx-drop-sm: 3px 3px 0 var(--ccx-ink)`.
- Background: `--ccx-bg` (the calendar's blue radial/linear stack).
- Textures: `--tx-ice-dark` (rink), `--tx-night` (the stands strip above the rink), `--tx-chalk` (intermission whiteboard card), `--tx-wood-dark` (bench panel header strip).
- Pixel icons (mask/background images): `--pxi-puck`, `--pxi-whistle` (penalty), `--pxi-flame` (hot), `--pxi-snow` (cold), `--pxi-bolt` (momentum), `--pxi-clock`, `--pxi-star` (goal), `--pxi-crown` (milestone), `--pxi-heart` (crowd), `--pxi-skull` (goalie struggle), `--pxi-check` (order accepted), `--pxi-trophy` (final win).
- Keyframes to reuse (do not redefine): `cp-pop` (cards/toasts), `cp-slam` (GOAL banner, tilted −9°), `cp-wobble` (chant bubbles), `cp-float` (HOT stamp), `cp-pulse` (on-ice row), `cp-flicker` (LIVE dot), `cp-sweep` (momentum bar sheen), `cp-blink` (buffering/clock under 1:00), `cp-spin`, `cp-confetti` (home win final), `cp-ticker` (bottom crawl), `cp-dash` (animated dashed outline of the selected token: `stroke-dasharray: 6 3; animation: cp-dash 1s linear infinite`), `cp-zamboni` (intermission: a pixel Zamboni crossing the rink).
- Motion style: stepped, arcade: `transition: transform .08s steps(2)` like `ccx-btn`. No soft blur glows.

### 13.3 Layout (1280 × 720 logical canvas; §2 diagram)
```
.st-root            display:grid; grid-template-rows:56px 1fr 180px; grid-template-columns:232px 816px 232px; width:1280px; height:720px; background:var(--ccx-bg); overflow:hidden;
.st-top             grid-column:1/-1; grid-row:1
.st-bench           grid-column:1; grid-row:2; overflow-y:auto
.st-rinkwrap        grid-column:2; grid-row:2; display:grid; grid-template-rows:72px 390px 1fr;   /* stands, rink, momentum strip */
.st-feed            grid-column:3; grid-row:2; overflow-y:auto
.st-deck            grid-column:1/-1; grid-row:3
```
All scroll areas: `scrollbar-width: thin; scrollbar-color: var(--ccx-cream) var(--ccx-deep);` and `min-height:0` on grid children (prevents the clipping bugs fixed in Stats Central). Every text cell that can overflow gets `min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap` plus a `title` attribute.

### 13.4 Components (class → look)
- `.st-top`: `ccx-deep` bar with a 4 px cream top border. Left: away chip; centre: score in `--ccx-display` 34 px yellow with a 3 px ink stroke (`-webkit-text-stroke: 3px var(--ccx-ink); paint-order: stroke fill; text-shadow: 3px 3px 0 var(--ccx-ink)`), clock in Silkscreen 16 px cream inside a `ccx-chip`, period stamp, strength stamp (`is-warn` on PP/PK, `is-bad` when empty net); right: home chip; far right: `LIVE` dot (coral, `cp-flicker`) plus speed buttons.
- Team chip `.st-teamchip`: rounded 8 px, team primary background, 3 px ink inset, abbreviation in Luckiest Guy 20 px, record in Silkscreen 9 px.
- `.st-stands`: `background: var(--tx-night)`; crowd meter as 16 pixel blocks (fill colour by mood: quiet `--ccx-muted`, buzz `--ccx-sky`, loud `--ccx-yellow`, roar `--ccx-orange`, booing `--ccx-coral`); mood label in Silkscreen; RIVALRY ribbon (coral, rotated −3°, `cp-wobble`) when tier ≥ 2.
- `.st-rink`: wrapper `background: var(--tx-ice-dark); border-radius: 18px; box-shadow: var(--ccx-outline), var(--ccx-drop);` with the SVG inside.
- `.st-chant`: cream bubble, 3 px ink border, Luckiest Guy 18 px ink text, rotated −4°, `cp-pop` in, `cp-wobble` while visible.
- `.st-goalbanner`: full-rink overlay; "GOAL!" Luckiest Guy 96 px yellow with a 5 px ink stroke, `cp-slam`; the scorer line beneath in a cream `ccx-chip`; home goals add `cp-confetti` over the rink area only.
- `.st-momentum`: a 14 px pixel bar under the rink, centre tick, home side fills right in the home colour, away side fills left; sheen `cp-sweep`; the label in Silkscreen 10 px.
- `.st-bench-row`: `ccx-deep` row, 2 px cream 30% inset; `.is-onice` → yellow 2 px inset + `cp-pulse`; energy blocks `.st-energy i` 6×10 px with a 1 px ink gap.
- `.st-feed-item`: P0 `color: var(--ccx-soft)`; P1 cream; P2 cream with an orange left border 4 px dashed (`border-left: 4px dashed var(--ccx-orange)`, the double-dash accent); P3 (goal) yellow card, ink text, `--ccx-drop-sm`.
- `.st-alert`: `ccx-panel` mini with a severity stamp and a pixel icon (`mask: var(--pxi-flame) center/contain no-repeat; background: currentColor`).
- `.st-deck`: four `ccx-panel` groups in a row (Lines | Goalie | Tactics | Special teams), each with a `ccx-kicker` title; buttons are `ccx-btn--sm`; the active selection is `ccx-choice__opt` with a yellow fill. Timeout is a coral round button with a whistle icon; disabled after use (`opacity:.45; filter:grayscale(.6)`).
- `.st-hovercard`: `ccx-panel` 220 px wide, positioned with a clamp so it never leaves the rink (no clipping: compute in JS against the 816×390 box).
- `.st-toast`: bottom-centre over the rink, `cp-pop`, auto-hide 2.2 s.
- `.st-intermission`: `background: var(--tx-chalk)`, period stats table (`ccx-table`), "Resume" primary button; `cp-zamboni` decorative sprite behind.
- `.st-final`: winner banner, three stars (from tonight's scratch ledger: points first, then goalie saves; label "Model stars"), Commit (primary) / Abort buttons.
- Double-dash accents: every panel kicker gets a `::after` of two short dashes (`content:""; width:14px; height:3px; background: repeating-linear-gradient(90deg, var(--ccx-orange) 0 5px, transparent 5px 9px)`), matching the calendar's dashed motif. The rink centre line and the selected-token outline are dashed too.

### 13.5 Responsiveness
The app scales the 1280×720 canvas. Verify at window sizes 1366×768 and 1600×900 (§16.5) that nothing clips. No breakpoints inside the canvas.

### 13.6 Calendar hooks styling
The calendar hero's new buttons reuse `ccx-btn--primary` (LIVE SIM, with a `--pxi-bolt` icon) and `ccx-btn` (FAST SIM). RESUME LIVE shows instead when a live game is open (coral fill, `cp-blink` dot).

---

## §14. API ENDPOINTS (ALL IN `backend/services/player_tracking_sim.py`)
`router = APIRouter(prefix="/api/franchise/live", tags=["live-sim"])`. Mount it in `main.py` exactly like the existing pattern near the end of the file (`from services.creation_report import router as _creation_report_router` → `app.include_router(...)`), §15.3. Every endpoint reads the session through the `x-franchise-session` header (same as `_post_franchise_advance_impl`), uses the shared `_session(sid)` helper pattern from §1.6 (400 when the header is missing, 404 when the session is unknown or expired after a code revision change), returns `json_safe(...)`, and calls `save_session` only on endpoints that change persisted franchise state (commit, abort-cleanup of `_live_score_override`).

| Method & path | Body | Returns | Notes |
|---|---|---|---|
| `POST /start` | `{game_id?: str}` (default = the user's game today from `nhl_today.user_game_today`) | `LiveSnapshot` | 409 `{reason:"no_user_game_today"}`; 409 `{reason:"sim_busy"}` if `sim_guard` is held; resumes an existing registry entry for the same slot instead of making a new one |
| `GET /state` | — | `LiveSnapshot` | cheap; no simulation |
| `GET /events?since_seq=N` | — | `{events, alerts, captions, frames, last_seq}` | read-only; used for resume/catch-up |
| `POST /advance` | `{until_t: float}` | same as `/events` plus `snapshot_lite` | simulates up to `until_t` (capped to +60 game-seconds per call); takes `exclusive_sim` for the duration of the step only |
| `POST /coach` | `CoachOrder` (without id/status) | `{order}` with status | §6 validation |
| `POST /pause` / `POST /resume` | — | `{state}` | server side just records it (pacing is client-driven) |
| `POST /commit` | — | `{ok, day_result, score}` | §4.7 order; only when `state == "final"` |
| `POST /abort` | — | `{ok}` | discards the live game; the slot stays unplayed and is fast-simmed on the next advance; clears overrides |
| `GET /tts/{audio_key}` | — | audio / 202 / 503 | §12.5 |
| `GET /tts-status` | — | `{worker: "up|down", capabilities}` | used by the settings toggle |

Errors use `HTTPException(status_code, detail={"reason": "...", "message": "...", ...})` (same `reason` key as the §4.6 advance guard). The frontend shows `message` and branches on `reason`.

---

## §15. HOOK EDITS OUTSIDE THE TWO FILES (MINIMAL; EACH IS A SMALL, REVIEWABLE DIFF)
These are the **only** allowed edits outside the two target files. Show each diff to Kiru in the PR description.

### 15.1 `backend/services/franchise_sim.py`: `_franchise_resolve_game_score` (line ~4004)
Add at the very top of the function a one-shot override check:
```
override = getattr(session, "_live_score_override", None)
if override and override.get("key") == <the same (home_id, away_id, iso_day) key the function uses>:
    session._live_score_override = None
    return override["score_result"]
```
- Read the function first to find the exact key fields; mirror them, don't guess.
- The override is consumed exactly once. When it doesn't match, the function behaves exactly as today (fast sim untouched).
- Also insert the live game's result into `session.sim._pending_event_games` in the cached-contract shape (§0.1) so `accumulate_unified_game_stats` takes the cache path and box-score stats come from the live game.

### 15.2 `backend/main.py`: `_post_franchise_advance_impl` (line ~1427) guard
After `s = _session_or_404(...)`: when the live registry holds an open (not committed/aborted) game for this session whose slot day would be simulated by this advance, raise `HTTPException(409, detail={"reason": "live_game_open", "live_id": ..., "message": "Finish or abort the live game first."})` (the contract in §4.6). Import the tiny helper `has_open_live_game(session_id) -> Optional[str]` (returns the live_id or None) from `services.player_tracking_sim`. **Do not import heavy modules at main.py import time**; the backend file's top-level imports must stay light (no torch etc., §12.7).

### 15.3 `backend/main.py`: mount the router
Append, matching the existing style:
```
# Live sim tracker (services/player_tracking_sim.py).
from services.player_tracking_sim import router as _live_sim_router  # noqa: E402

app.include_router(_live_sim_router)
```

### 15.4 `frontend/src/game/constants.js`
Add `SIM_TRACKER: "sim_tracker",` to `SCREENS` (line ~5). Do **not** add it to the nav list (line ~38). It's reached only from the calendar.

### 15.5 `frontend/src/App.js`
Add `const SimTracker = React.lazy(() => import("./screens/SimTracker"));` next to the other lazy screens (line ~23) and a route branch for `SCREENS.SIM_TRACKER` passing `sessionId`, `gameId` (from navigation state) and `onExit` (back to the calendar, then refetch the calendar/day data). Follow exactly how CalendarScreen is routed.

### 15.6 `frontend/src/screens/CalendarScreen.js`
In the hero/today card where the user's game for today is shown: replace or augment the existing sim action with **LIVE SIM** (primary) and **FAST SIM** (the existing behaviour, unchanged). When `liveState()` reports an open live game: show **RESUME LIVE** and disable advancing buttons with the tooltip "Live game in progress". Handle the 409 `live_game_open` from any advance call by showing the same message.

### 15.7 `frontend/src/services/franchiseService.js`
Add thin wrappers that use the same request helper and session header as `advanceFranchise` (line ~81): `liveStart`, `liveState`, `liveEvents`, `liveAdvance`, `liveCoach`, `livePause`, `liveResume`, `liveCommit`, `liveAbort`, `liveTtsUrl(audioKey)`, `liveTtsStatus`. No business logic in this file.

### 15.8 `frontend/src/styles/game.css`
Append the single block from §13.1. Nothing else in game.css changes.

### 15.9 Not allowed
No edits to `engine.py` (the live loop is a port that lives in the backend file and calls engine primitives), progression, contracts, awards, or any other screen. If something seems to require one, stop and report.

Note: `@babylonjs/core` is already a dependency, but **do not** use it for the rink. The spec is SVG (§11.4) to match the 2D arcade style and keep the bundle small.

---

## §16. VALIDATION (EVERY STAGE MUST PASS ITS CHECKS BEFORE THE NEXT)

### 16.1 Parity script (blocking for Stage 2)
`python -m services.player_tracking_sim --parity --games 300 --seed 7` (a CLI mode in the backend file). It runs 300 seeded matchups through the **unmodified** engine event path and through the LiveEventLoop with no coach orders, and compares distributions against §4.8 tolerances: goals/game, shots/game, PP goals, save %, OT rate, shootout rate, home win %. Prints a table and exits non-zero on failure. With no orders and the same seed, scores should match exactly game by game; report the match rate (target 100%; anything lower needs an explanation before continuing).

### 16.2 Fast-sim determinism (blocking)
Run the existing fast-sim path for 30 days on a fixed seed before and after the hooks (§15). Standings, player stats and the day results must be byte-identical (compare `json_safe` dumps). This proves §15.1 changed nothing when no override exists.

### 16.3 Commit integration
Start a franchise, live-sim the user's game to the end, commit. Check:
- the calendar shows the result
- standings updated once (not twice)
- player season stats include the live box score exactly once
- goalie GP/W/L correct, including the two-goalie case (§6.2)
- `advance` works afterwards
- abort leaves the slot playable and fast sim produces a normal result

### 16.4 Coaching effects (sanity, not exact)
Over 200 seeded live games each: shortening the bench to 3 lines raises line-1 TOI share; a goalie pull raises both GF and GA in the final 2 minutes; benching a player gives that player 0 TOI after the order; illegal orders are rejected with reasons.

### 16.5 Frontend checks (built-in browser)
At 1280×720, 1366×768 and 1600×900: no clipped text, no overflowing panels, scroll areas scroll, hover card stays inside the rink, goal banner sits centred, reduced motion is respected, the audio starts only after a gesture, leaving and returning resumes the game, the console has no errors or React key warnings, and memory stays flat over a full game at 8x (check with the Performance panel or by watching JS heap).

### 16.6 TTS checks
- `python -m services.player_tracking_sim --tts-qa` renders `TTS_QA_CORPUS` via the worker and prints per-category duration/peak/failure.
- Kill the worker mid-game: captions continue, one toast, no crash.
- Confirm the licence file of every model in use and paste its path and licence name into the PR description.

### 16.7 Proposed new files (each needs Kiru's approval before creation)
- `backend/data/commentary_pronunciations.json` (pronunciation seed, §10.6)
- `backend/data/tts_cache/` (disk clip cache, §12.5)
- `backend/tests/test_player_tracking_sim.py` (parity smoke test plus order validation tests)
- `docs/live_sim_tracker.md` (operator notes: starting the TTS worker, flags)
- TTS model weights download location (outside the repo; document the path)

---

## §17. STAGED DELIVERY (ONE PR PER STAGE; STOP AND REPORT AT EACH)
| Stage | Scope | Exit criteria |
|---|---|---|
| 1 | Backend skeleton: registry, pre-game builder, snapshot, `/start` `/state` `/abort`, `has_open_live_game`, router mount | `/start` returns a valid LiveSnapshot for the user's game; abort cleans up |
| 2 | LiveEventLoop port, `/advance`, `/events`, result build, parity CLI | §16.1 passes |
| 3 | Coach orders (§6), CPU coach, validation | §16.4 passes |
| 4 | Commit path and hooks §15.1–15.2 | §16.2 and §16.3 pass |
| 5 | Frontend skeleton: routing, calendar buttons, layout, top bar, bench, feed with text only, coach deck wired | Playable full game with text only |
| 6 | Tracking model (§7) plus the SVG rink, tokens, markers, heatmap, Tracking tab | 60 fps budget; no clipping |
| 7 | Momentum, crowd, alerts, chants, procedural audio | Crowd reacts correctly in a rivalry test game and a low-fan-meter test game |
| 8 | Commentary text engine plus linter; TTS worker with one CPU adapter; prerender pool | §16.6 passes; lint has zero false names/numbers over 50 games |
| 9 | Playoffs: playoff OT (no shootout), series state in the top bar and commentary, louder crowd base, clinch calls | A playoff game commits correctly into the bracket |
| 10 | Polish: settings, accessibility pass, Tier A TTS adapter if D1 approves a GPU | Kiru sign-off |

---

## §18. OUTSTANDING DECISIONS (ASK KIRU; DO NOT ASSUME BEYOND THE DEFAULTS)
| ID | Decision | Default until decided |
|---|---|---|
| D1 | TTS model per tier and whether a GPU worker exists | Tier B Kokoro-82M adapter only |
| D2 | Voice actor contract and recording (needed for any cloning) | No cloning; stock voice |
| D3 | Crowd/momentum home lift on the sim (§8.4) | Off |
| D4 | Shootout model: (A) reproduce the engine's single `rng < 0.52` resolution and stage a cosmetic shooter-by-shooter reveal consistent with that winner, or (B) a real round-by-round model (needs engine work, out of scope) | A |
| D5 | Penalty labels: duration-only text, or approve an engine change that records penalty types | Duration only |
| D6 | Pronunciation data file (§16.7) | In-file dict only |
| D7 | Soundtrack folder licensing: it has copyrighted game soundtrack mp3s (FIFA OST, Super Mario 64). The live screen must not play them; they are a release blocker for the whole game (§0.6) | Live screen uses procedural audio only |
| D8 | Commentator persona name ("Rinkside Reggie" placeholder) | Placeholder |

---

## FINAL CHECKLIST FOR THE AGENT (READ BEFORE EVERY PR)
- [ ] Only `backend/services/player_tracking_sim.py` and `frontend/src/screens/SimTracker.js` contain new logic; hook edits match §15 exactly.
- [ ] Engine RNG consumption unchanged with no coach orders (§0.2, §16.1).
- [ ] Fast sim byte-identical (§16.2).
- [ ] No fabricated names, numbers, penalty types, posts, fights or injuries (§0.5, §10.5).
- [ ] Tracking analytics labelled "model"/"est." everywhere (§7).
- [ ] No new hex colours; `st-` prefix; one game.css block (§13).
- [ ] No paid APIs; licences verified; no real-broadcaster voice or audio (§12.1).
- [ ] No new files without approval (§16.7).
- [ ] Session pickling still works: no locks, threads or audio objects stored on the session (§0.4).
- [ ] Windows dev flow works: backend on :8000, CRA on :3000, TTS worker on 127.0.0.1:8765 (§0.7, §12.7).
