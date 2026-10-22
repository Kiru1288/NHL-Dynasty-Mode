# Storyline Engine — Technical Reference

How every storyline feature works: what feeds it, the exact trigger math, what it writes to the game, and how it reaches the sim and the UI.

**Files**

| File | Role |
|---|---|
| `SimEngine/app/sim_engine/franchise/storyline_engine.py` (13.4k lines) | Core engine. Three generations stacked in one file: V1 data stories + cause stories, "Universe V2" (player entities, locker rooms, meetings), "Universe V3 patch" (life events, major incidents, readiness, promises). Later definitions override earlier ones with the same name. |
| `storyline_coverage.py` | Extra daily coverage: concern thresholds, rolling form, CPU rumors, org desk, league social, box-score stories, matchup → sim strength. |
| `storyline_procedural.py` | Life-event rule table + procedural headline/summary copy + "why did this fire" trigger context. |
| `storyline_copy.py` | Headline pools, goalie SV% floors, league volume caps, headline-fingerprint dedupe. |
| `storyline_stat_bridge.py` | Feeds finished box scores into postgame + matchup context. |
| `storyline_conduct.py` | Temporary OVR modifiers (`apply_temporary_ovr_modifier`, `apply_storyline_ovr_nudge`). |
| `state.py::_record_storyline` | Final write into `session.storyline_events` (the feed). |
| `backend/services/franchise_sim.py` | Calls everything once per sim day. |
| `backend/services/trade_demand_engine.py` | Formal trade demands (separate engine). |

---

## 0. The daily pipeline (how it kicks in)

Every simulated day, `franchise_sim._run_franchise_narrative_passes(session, day_idx, day_meta, full=…)` runs, in this order:

```
1. franchise_record_data_storylines()          ← §1 data stories (stats-driven)
2. franchise_cause_storyline_daily_pass()
     ├─ migrate_session_storyline_state()
     ├─ tick_franchise_storyline_modifiers()   ← count down cause-story modifiers
     ├─ _decay_trade_rumor_state()
     ├─ _expire_gm_meetings()
     ├─ _maybe_record_losing_streak_event()    ← §3 feeds locker room
     ├─ _process_locker_room_triggers()        ← §3 conflicts
     └─ (if not light bulk) narrative_universe_daily_pass()
           ├─ legacy: prospect social posts, season seal (offseason), eras
           └─ narrative_universe_v2_daily_pass()   ← the "daily director" (§0.1)
3. _decay_all_fan_heat()                        ← weekly pull of fan confidence back to standings
4. _goofy_front_office_tick()
── only when full=True ──
5. _franchise_fanout_player_storylines()
6. trade_demand_engine.process_trade_demand_day()  ← §10
7. _franchise_tick_conduct_and_resolve()
```

**Bulk sims** (sim-to-date / sim-week): the passes run with `full=False`, and if the session is flagged `_bulk_calendar_advance` + `_light_game_stat_accumulation`, the whole universe director is **deferred**. To keep it alive, `bulk_universe_catchup()` runs after the bulk loop:
- replays `_u_advance_game_based_state` for every user-team game missed (max 40), so promises still count games;
- runs the full `narrative_universe_v2_daily_pass` once every **7 days** (`UNIVERSE_BULK_PASS_EVERY_DAYS`) and at the end (`force=True`).

### 0.1 The daily director — `narrative_universe_v2_daily_pass`

Idempotent per day (`session._universe_last_daily_tick`). Seeded RNG: `_u_seed("universe_daily_v3", season, day, user_team)` unless the sim RNG is passed — so a given save + day replays the same rolls.

```
_u_sync_player_entities()          build/refresh one "entity" dict per player
_u_tick_readiness_modifiers()      readiness days_remaining -= 1, expire at 0
_u_tick_player_availability_days() suspension/leave days -= 1
for every player: _u_tick_player_life()      §4.2 drift of stress/energy/focus/concerns
for every team:  _u_rebuild_locker_room()    §4.3 culture, factions, roles
_u_generate_minor_life_events()    §5
_u_cpu_morale_management()         §11
_process_storyline_followups()     §9.4
_u_expire_interactions()           §6.5 ignored meetings
expire_press_conferences(); reconcile_requires_action()
_u_generate_daily_interactions()   §6 meeting requests
_u_maybe_create_trade_demand_from_state()   (disabled, returns 0)
_u_tick_trade_demands()            leak/deadline handling on existing demands
_u_run_major_league_event_pass()   §7
_u_generate_ambient_social()       §9
_u_tick_promises()                 §6.6
storyline_coverage.run_coverage_daily_pass()  §2
snapshot → session.universe_daily_snapshots (last 120)
```

### 0.2 Two publishing paths

| Path | Used by | What it does |
|---|---|---|
| **V1 cause path** `_emit_cause_storyline(sl)` | data stories (user team), locker-room conflicts, lineup fallout, trade fallout | `validate_storyline_before_effects` → optional temp OVR modifier → `_apply_storyline_effects` → `_record_storyline` → append to `active_cause_storylines` (last 60) → inbox notification → **popup** if user team → `_enqueue_decision` if it has action options |
| **Universe path** `_u_record_storyline(event, …)` + `_u_notify_user_event(level)` | life events, meetings, major incidents, promises, postgame, coverage | records story with heat/public flag; notification router by level: **1** inbox · **2** banner (HIGH) · **3** popup · **4** blocking BREAKING NEWS (CRITICAL). CPU-team events are skipped unless `force_league=True` |

### 0.3 Global gates that every story passes through

| Gate | Where | Rule |
|---|---|---|
| **Cooldown by stable key** | `_can_fire` / `_mark_fired` (`session._storyline_engine_cooldowns`) | Blocks the same key for **7 d** (minor) / **14 d** (major) / **10 d** (crisis) — but only if severity is the same; a worse severity breaks through and is tagged `escalating` with `repeat_count` |
| **Stable keys are bucketed** | per feature | e.g. `star_underperforming|pid|season` (once/season), `losing_skid|tid|season|4/7/10` (once per tier), `concern|role|pid|season|day//21` (once per 3-week window) |
| **Daily league cap** | `select_daily_data_stories` | Keeps **all** user-team data stories + max **7** league stories, highest heat first, headline-fingerprint deduped |
| **Volume slots** | `claim_league_story_slot` (CPU teams only) | Goalie heater max 2/day · Shutout 1/day · Superstar carry 2/day · Community lift 1/day · Reporter confrontation 1 per 3 days. Same headline shape (numbers → `#`) once per day |
| **Feed dedupe** | `_record_storyline` | `_storyline_dedupe` key list (last 400–500); feed trimmed to 480 rows, user-team rows protected |
| **Negative-on-user guard** | `validate_storyline_before_effects` | A negative story on your team **must** have a `cause_type` in `STORYLINE_CAUSE_TYPES` and a `cause_event_id` present in `decision_event_log` (stat stories exempt). Otherwise blocked and logged to `_storyline_blocked_log`. Also blocked if the player isn't on your roster |
| **Random-story guard** | `should_block_random_storyline_for_user` | Blocks uncaused scandal / locker-room / "trade rumor" / "scratched" text on your team (legal/conduct rows still allowed) |
| **Decision throttle** | `_enqueue_decision` | One blocking decision per player per **21 days**; none if that player already has an active promise |

### 0.4 How a story actually changes the game

Three mechanisms, from weakest to strongest:

1. **Storyline effects** — `_apply_storyline_effects(team, player, effects)`; integer deltas × **0.01** onto 0–1 scales:

   | Effect key | Writes to |
   |---|---|
   | `player_morale` | `player.psych.morale` |
   | `player_confidence`, `goalie_confidence` | `psych.confidence_level` (+ `traits.confidence` for player_confidence) |
   | `development_confidence` | `psych.internal_motivation` × 0.8 |
   | `media_pressure` | `psych.media_stress` × 0.9 and team `organizational_pressure` |
   | `team_morale` | `team.state.team_morale` |
   | `fan_confidence` | org pressure (−0.35×) and fan profile via `apply_fan_engagement_delta` at **0.4** strength (`STORY_FAN_EFFECT_SCALE`) |
   | `gm_trust` | universe entity `state.gm_trust`, synced to trade scales |
   | `room_tension`, `lineup_pressure`, `depth_pressure`, `trade_market_heat`, `draft_stock` | stored on the story only (display / downstream readers) |

2. **Readiness modifiers** — `_u_apply_readiness_modifier(ovr_delta, days, stat_modifiers)`. A day-counted "effective OVR" layer stored in `session.universe_readiness_modifiers` (max 20 per player). Total per player clamped to **−25 … +6**. If `|delta| ≥ 0.75` it is **mirrored onto the roster OVR** via `apply_temporary_ovr_modifier` as a rounded integer for `days/2` games (2–24) — that's what you see as "OVR 84 → 83".
3. **Temporary OVR modifier (cause path)** — stories carrying `ovr_modifier` call `apply_temporary_ovr_modifier(duration_games=12)` (negative) or `apply_storyline_ovr_nudge` (positive).

Plus permanent-ish levers: `_u_apply_potential_change` (potential, capped **±2.5 per season** per player, `UNIVERSE_POTENTIAL_SEASON_CAP`), `_u_apply_attribute_change` (temp or permanent attribute, permanent cap 3.0), and availability (§7).

### 0.5 How it reaches the game sim

Before each game, `franchise_sim` calls `storyline_coverage.apply_matchup_to_scales(home, away, h_scale, a_scale)`:
- `build_universe_game_context(team)` is built for both sides (§4.4);
- `home_win_probability_delta = home.delta − away.delta`, clamped **±0.18**;
- home strength scale × (1 + delta), away × (1 − delta), both clamped to **0.90 – 1.10**.

After each game, `storyline_stat_bridge` sends the box score to `apply_universe_postgame` (§8) and `ingest_game_box_storylines` (§2.6).

---

## 1. Data-driven stories — `run_data_storyline_pass`

**Input:** `session.player_season_stats`, rosters (NHL/AHL/IR/scratches/ECHL), standings rank map, injuries, draft board.
**Odds:** 100% when the condition is met; it's the cooldown key that limits repeats.

**Expected-production model:**

```
expected P/GP = clamp(0.12, 1.35, 0.95 × base)
  base = max(0.08, (OVR − 62) / 38)
         × 0.52 (D) or × 0.88 (F)
         × 0.72 if rookie (age ≤ 23 and ≤ 40 GP)
         × 0.92 if age ≥ 33
expected SV% = 0.870 + (OVR − 70) × 0.0018      (OVR 85 → .897)
```

| Story (stype → cause_type) | Exact trigger | Effects | Key / repeat |
|---|---|---|---|
| **Star underperforming** → PLAYER_LOW_PRODUCTION | OVR ≥ 84 and GP ≥ 6: P/GP < 62% expected (minor). GP ≥ 10: P/GP < 45% expected (major) | major: confidence −6, media +5, team morale −3, trade value −2 (minor ≈ half). Heat 78/55. User team + major → decision: *back publicly* / *challenge via media* / *reduce TOI* / *explore trade* | `pid|season` |
| **Rookie breakout** → ROOKIE_BREAKOUT | rookie, OVR ≤ 83, GP ≥ 6, P/GP ≥ max(0.55, 1.45 × expected) | positive. User team → decision: *promote (creates role promise, 6 games)* / *steady plan* / *shelter minutes* | `pid|season` |
| **Superstar carrying** → SUPERSTAR_CARRY | OVR ≥ 86, GP ≥ 10, pts ≥ 14, team rank ≥ 20th, P/GP ≥ 0.85 (contract-year flag if ≤ 1 yr left) | User team: queues `carry_followup` story **12 days later** | `pid|season` |
| **Contract pressure** → CONTRACT_DISPUTE | cap hit ≥ $6.5M, GP ≥ 10, points < 55% of expected | negative | `pid|season` |
| **Goalie meltdown** → GOALIE_BAD_FORM | GP ≥ 6, SV% < expected − .018 **and** SV% between .500 and **.890** | goalie confidence −8, team morale −2, media +3, fans −3. HIGH priority if > .028 below. User team → decision: *start backup (start promise)* / *another chance* / *call up AHL goalie (call-up promise)* | `pid|season` |
| **Goalie heater** → GOALIE_HEATER | GP ≥ 3, SV% ≥ expected + .012 **and** SV% ≥ **.915** | positive | `pid|season` |
| **Backup taking the net** | in a team's goalie group: backup GP ≥ 3, starter GP ≥ 6, backup SV% ≥ starter + .015 | | `backup_pid|season` |
| **Surprise team** | GP ≥ 8, team strength < 0.46, win% ≥ .580 | | `tid|season` |
| **Contender collapse** | GP ≥ 8, strength ≥ 0.58, win% ≤ .420 | | `tid|season` |
| **Playoff race** (user only) | GP ≥ 55, league points rank 7–11 | media +3, morale +1/−2 | `tid|season` |
| **Losing skid** → LOSING_STREAK | current L streak ≥ 4 | | tiered 4 / 7 / 10 |
| **Win streak** → WINNING_STREAK | current W streak ≥ 5 | | tiered 5 / 8 / 12 |
| **Prospect riser / faller** | draft rank moved ≥ 8 spots vs preseason baseline | `draft_stock` ±min(12/10, delta) | `pid|season` |
| **Injury ripple** → PLAYER_INJURED | injury logged ≤ 2 days ago (major if ≥ 4 games) | team morale −2, depth +4, lineup +3 | `pid|season` |

**Routing:** `franchise_record_data_storylines` sends user-team rows with a cause_type through the V1 cause path (popup + decision), everything else straight to `_record_storyline`.

> ⚠️ **Two issues found while tracing this:**
> 1. `run_data_storyline_pass.emit()` applies effects to **every** generated story *before* the 7-story league cap trims them, so CPU stories that never show up in the feed still change morale/fan numbers.
> 2. User-team data stories with a cause_type then go through `_emit_cause_storyline`, which calls `_apply_storyline_effects` **again**: their effects are applied **twice**.

---

## 2. Coverage stories — `storyline_coverage.py`

Runs at the end of the daily director.

### 2.1 Concern-threshold stories (all players)
Re-derives four concern scores per player each day:
- **role** = avg(psych.role_satisfaction, psych.ice_time_satisfaction)
- **team_belonging** = psych.locker_room_fit
- **contract** = 100 − psych.contract_pressure (min 8)
- **winning** = 25 + points% × 70

`pressure = importance × (100 − satisfaction) / 100`. Fires if pressure ≥ **28** (≥ **22** for role/winning if ambition ≥ 68).

| Concern | Cause | Base heat (+ pressure/4, max +18) |
|---|---|---|
| role | PLAYER_ROLE_FRUSTRATION | 58 |
| contract | CONTRACT_YEAR_HEAT (published as *rumour/claim*) | 62 |
| team_belonging | ROOM_BELONGING | 52 |
| winning | WINNING_CONCERN | 55 |

Key `concern|key|pid|season|day//21` → max once per 3-week window. Max 16/day. A random local/performance/contracts beat writer is credited.

### 2.2 Rolling form (last 10 games, needs ≥ 5)
`expected = max(0.15, (OVR − 62)/42)`, ×0.55 for D.
- **ROLLING_HOT:** P/GP ≥ 1.55 × expected and ≥ 5 pts → morale +3.5, confidence +2.8.
- **ROLLING_COLD:** P/GP ≤ 0.42 × expected, OVR ≥ 78, ≥ 6 GP → morale −3.8, confidence −3.0.
Key bucket `day//6`; max 6/day.

### 2.3 Org desk (per team, max 5/day)
| Story | Trigger | Repeat |
|---|---|---|
| COACH_HOT_SEAT | GP ≥ 18, rank ≥ 24th | per 21 days |
| GM_JOB_SECURITY | GP ≥ 22, rank ≥ 26th | per 21 days |
| CAPTAINCY_PULSE | team has a captain → **8%/day** | per 12 days |
| AHL_CALLUP / AHL_SENDDOWN | diff of AHL roster vs yesterday's snapshot | per player per day |

### 2.4 CPU locker-room rumors
Resolved CPU-team interactions (last 24) are published as LOCKER_ROOM_PULSE. Mentor/glue scenes only **22%** of the time (as "chatter"); conflicts always (as "rumour"). Max 10/day.

### 2.5 League social
- **42%/day:** a random agent with leak tendency ≥ 0.35 posts a vague client tweet.
- **55%/day:** a beat writer posts a "checking in on X" note.

### 2.6 Box-score stories (`ingest_game_box_storylines`, every game)
| Story | Trigger | Heat |
|---|---|---|
| HAT_TRICK | 3+ goals | 78 |
| ON_ICE_ALTERCATION | 15+ PIM | 64 |
| FIRST_GOAL | first goal of season, OVR ≤ 82 | 48 |
| SHUTOUT | opponent scored 0. Published if your team, or goalie already had one in last 7 days, or team is top-3 / rank 7–10. Repeat shutout on your team → `shutout_coach` follow-up | 46–68 |
| OT_WINNER | OT game, last goal scorer | 70 |

---

## 3. Locker-room conflicts (your team) — `_process_locker_room_triggers`

**Feed:** `session.decision_event_log`. Your actions write events through `record_decision_event`: lineup saves (§3.1), TradeHub (§3.2), contract disputes, captaincy changes. Losing streaks are auto-logged by `_maybe_record_losing_streak_event` when GP ≥ 8, L ≥ 4 and W ≤ 2 (max once per 5 days).

**Each day:**
1. Scan the last 40 events; keep those of type `PLAYER_SCRATCHED_BY_USER, PLAYER_DEMOTED, PLAYER_ROLE_REDUCED, TRADE_REJECTED, PLAYER_REPEATEDLY_SHOPPED, LOSING_STREAK, CONTRACT_DISPUTE, CAPTAINCY_CHANGED`, on your team, **≤ 3 days old**, not already spawned, with no active arc.
2. Culprit = the event's player. For LOSING_STREAK, a random roster player with character < 50.
3. Character (`_player_character_0_100`) → roll: **< 40: 55%**, **40–49: 35%**, **50–54: 18%**, **≥ 55: never**.
4. On hit → `_build_locker_room_conflict_storyline` → cause path (popup, effects, possible decision). Event marked `_locker_room_spawned`. **Max 1 per day.**
5. Active conflicts end via `resolve_culprit_traded_storylines` when the culprit is traded.

### 3.1 Lineup fallout — `record_lineup_save_decisions` (on Save Lines)
Compares old vs new even-strength line ranks (F lines 1–4, D pairs 1–3, G slots). With no previous lines it infers ranks from roster order.

| Change | Event | Effects | Temp OVR | Decision? |
|---|---|---|---|---|
| removed from lines | PLAYER_SCRATCHED_BY_USER | morale −6 (char < 50) / −3, tension +3, lineup +4 | −2 (char < 45) / −1, 6 games | yes if char < 45 |
| down ≥ 2 lines | PLAYER_DEMOTED | morale −8 / −4, tension +2, coach trust −2 | −3 / −2 | yes if char < 45 |
| down 1 line | PLAYER_ROLE_REDUCED | morale −4 (char < 55) / −2 | −1 | no |
| up ≥ 1 line | PLAYER_PROMOTED | morale +4, confidence +3, dev +2 | **+min(2, jump)** | no |

Decision options: *Restore prior role* (creates a role promise, 6 games) or *Meet privately* (morale +3, GM trust +2).

### 3.2 Trade fallout — `record_trade_hub_evaluation`
Fires only when **you submit** a TradeHub proposal containing your player and it's **rejected** (not on accept, "technical" rejections, soft blocks, or while the player's rumor cooldown is active). Each one increments `trade_attempt_count`.

```
fragility = clamp((80 − mentalOVR) / 55, 0, 1)
gm_trust_drop = round(3 + attempts×2.2 + fragility×8 + volatility×3 − professionalism×2)
morale_drop   = round(2 + attempts×2.0 + fragility×10 + volatility×2)
ovr_mod       = round((0.5 + attempts×0.85 + rumorHeat×0.018) × (0.65 + fragility×2.6))
                +2 if mental < 45; +3 more if mental < 30 and attempts ≥ 2; clamp 1–18
duration_days = clamp(7, 120, 7 + attempts×7 + fragility×52 + max(0, 50−char)×0.35)
severity: ≤2 minor · ≤6 mid · else major
```

Headline escalates: 1st = TRADE_REJECTED, 2nd–3rd = "another failed proposal", 4th+ = "camp pushing for clarity" (PLAYER_REPEATEDLY_SHOPPED). Decision required if attempts ≥ 2 or ovr_mod ≥ 4. **Mental OVR** = stored value, or `resilience×0.28 + professionalism×0.17 + confidence×0.23 + focus×0.18 + morale×0.14`.

---

## 4. The player "universe" model

### 4.1 Entities
`_u_sync_player_entities` keeps `session.universe_players[pid]`, one dict per player:
- **personality:** character, volatility, ego, ambition, competitiveness, coachability, professionalism, resilience, leadership, empathy, accountability, loyalty, money_focus, media_savvy, family_orientation…
- **state (0–100):** morale, confidence, coach_trust, gm_trust, belonging, energy, focus, personal_stress, media_stress, role_satisfaction, character_modifier
- **life:** relationship_status, dependents, home_stability, relocation_strain, community_connection, city_attachment, financial_stress, sleep_quality, home_owned
- **concerns:** role, contract, winning, development, home_life, team_belonging, each `{importance, satisfaction, trend}`
- **niches:** mentor, glue_guy, peacemaker, culture_carrier, quiet_professional, accountability_driver, self_first… (each with game effects)
- memories (last 40), social profile, potential ledger, agent

Initial values are seeded from the player's real psych/traits plus random jitter (e.g. confidence = 40 + competitiveness×0.22 ± 12).

### 4.2 Daily life drift — `_u_tick_player_life`
- role satisfaction moves 18%/day toward actual deployment (`infer_role_satisfaction_from_deployment`, real TOI vs earned slot)
- `personal_stress` → target `30 + relocation×0.30 + max(0, 60−home)×0.36 + max(0, 62−sleep)×0.28`, 14%/day (6% while a meeting "hold" is active), ±1.8 jitter
- energy from sleep and stress; focus = `48 + confidence×0.24 + energy×0.20 − stress×0.14`
- relocation strain −0.18/day; character modifier decays ×0.992/day
- contract satisfaction = `38 + years_left×18`; winning = `25 + points%×70`
- top 3 concerns by pressure are cached → these feed meetings and coverage.

### 4.3 Locker room — `_u_rebuild_locker_room` (daily, every team)
Per player:
```
room_value = character×.22 + professionalism×.18 + empathy×.17 + accountability×.16
           + leadership×.13 + belonging×.08 + 4.5 per positive niche
disruption = max(0, 50−char)×.72 + max(0, ego−62)×.35 + max(0, vol−62)×.36
           + max(0, 42−role_sat)×.35 + 12 if self_first
```
Roles: top-2 room value ≥ 67 → leadership group, glue/peacemaker → connector, mentor, disruption ≥ 50 → disruptor risk.

Culture:
- `unity = avgChem×.48 + avgMorale×.28 + 12 + positiveNiches×1.1 − disruptors×1.7`
- `tension = avgTension×.55 + disruptors×7 − positiveNiches×1.4`
- confidence, accountability, work ethic, leadership (top 5), belonging = averages.

**Factions** are connected groups where pair chemistry ≥ 67 and trust ≥ 60 (max 6). Pair relationships (trust, respect, tension, familiarity, chemistry) are seeded from compatibility ± jitter.

### 4.4 Game context — `build_universe_game_context`
Per player:
- `effort = (morale−50)×.028 + (focus−50)×.024 + (energy−65)×.018`
- `composure = (conf−50)×.028 + (resilience−50)×.022 + (mental−55)×.010`
- `discipline = (char−50)×.016 − max(0, vol−55)×.028`
- if character < 45 and role satisfaction < 46: defensive effort, passing and penalty-risk penalties, and the player is listed as a "character risk"
- niche abilities add their game effects × tier multiplier
- `overall_equivalent` (±8) = weighted sum + readiness OVR; readiness stat modifiers (shooting, stamina, positioning…) pass through

Team:
```
win_delta = (unity−50)×.00115 + (conf−50)×.00088 + (leadership−50)×.00062
          + (accountability−50)×.00048 − max(0, tension−35)×.0012
          + min(.028, glueGuys×.0032) − disruptorTax − min(.08, readinessTax)
clamped ±0.16 per team → net ±0.18 per matchup (§0.5)
```

---

## 5. Off-ice life events — `_u_generate_minor_life_events`

**Kick-in per player per day:**
1. Cooldown since the last life event: **6 days** if volatility ≥ 62, else **9** (4 in bulk catch-up).
2. Roll: your team **7%** × (0.75 + vol/100); CPU teams **3%** × same. Bulk catch-up: 22%, your team only. Your team is processed first.
3. **52% positive / 48% negative.**
4. `evaluate_life_event_rules`: collect all rules of that polarity whose predicate passes and whose family requirements are met (partnered / has kids / single). Sort by weight, take everything within **8 weight** of the top, pick randomly from that band. No match → nothing happens (the roll is wasted).
5. Same rule on the same player within **30 days** → skipped.
6. Apply, multiplied by `LIFE_EVENT_DRAMA = 1.6`: profile deltas, character modifier, readiness (`ovr × 1.6` for `max(3, days × 1.35)` days, + attribute stat modifiers). Positive + age ≤ 25 → potential roll.
7. Visibility: negative always public. Positive uses the rule's public_chance, or 35% if it has none.
8. Memory added (±2 emotion). Story recorded (`POSITIVE_LIFE_EVENT` / `MINOR_LIFE_EVENT`, category personal_life). Your team → inbox notification (level 1).
9. League cap: 36 events/day (72 in bulk).

Effective OVR after drama: only a base `|ovr| ≥ 0.47` crosses the 0.75 line that mirrors onto the roster OVR. That means only **family stress spillover (−0.88)** and **relocation family strain (−0.80)** show up as a visible −1 OVR. Every other life event (e.g. sleep hit −0.64, homesick −0.64, family milestone +0.56) lives only in the readiness layer that feeds the pre-game context.

| Rule | Polarity | Predicate | Weight | Profile Δ (pre-drama) | OVR / days | Extra |
|---|---|---|---|---|---|---|
| family_stress_spillover | − | stress > 68, family_orient > 55, kids | stress + fam×.35 | stress +4, focus −2.5, morale −2 | −0.55 / 6 | |
| sleep_recovery_hit | − | sleep < 52, energy < 58 | (62−sleep)+(58−energy) | energy −3, focus −2 | −0.4 / 2 | stamina −0.35 |
| media_backlash | − | media_stress > 58, vol > 52, char < 62 | ms + vol×.4 + (62−char)×.5 | media +3, conf −1.5 | −0.35 / 5 | char −0.15, discipline −0.2 |
| relocation_family_strain | − | relocation > 42, kids | relocation + 20 | stress +4, belonging −2 | −0.5 / 9 | |
| housing_instability | − | home_stability < 48 | 70 − home | stress +2, energy −1 | −0.25 / 4 | |
| partner_friction | − | partnered, stress > 60, home < 55 | stress + (55−home) | morale −2, stress +3 | −0.35 / 4 | |
| financial_pinch | − | fin_stress > 38, or money_focus > 62 and contract_sat < 45 | fin + max(0, money−55) | stress +2, morale −1 | −0.2 / 5 | |
| homesick_signal | − | single, relocation > 36, belonging < 50 | relocation + (50−belonging) | morale −2.5, belonging −2 | −0.4 / 7 | |
| community_lift | + | community > 58, char > 55 | community + char×.3 | morale +3, belonging +4, conf +1 | +0.25 / 7 | char +0.5, public 28% |
| family_milestone | + | partnered, stress < 42, fam > 58 | fam + (42−stress) | morale +4, belonging +2, stress −2 | +0.35 / 8 | public 45%, **potential +0.15 @ 12%** |
| belonging_reset | + | home > 68, relocation < 28 | home − relocation | belonging +4, stress −3, focus +1.5 | +0.3 / 10 | char +0.2, **potential +0.2 @ 15%** |
| confidence_surge | + | conf > 68, winning_sat > 60 | conf + win×.25 | conf +2, morale +2 | +0.2 / 4 | public 35% |
| roots_in_city | + | belonging > 62, no home owned | belonging | belonging +4, stress −2 | +0.25 / 10 | char +0.25, public 35% |
| mentorship_click | + | age ≤ 24, conf < 52, mentor tag | (52−conf) + 20 | conf +2, focus +2 | +0.25 / 6 | off. awareness +0.25, **potential +0.3 @ 20%** |

---

## 6. Player meeting requests

### 6.1 Candidate generation — `_u_interaction_candidates(team)`
For each player (on your team: skip if they already have a pending request, or are inside the 12-day request / resolved-meeting cooldown, or attention was cleared). Each passing rule adds `(score, kind, actor, target)`:

| Kind | Condition | Score |
|---|---|---|
| request_more_ice / request_pp_time / request_starting_role | role_sat (from real deployment) < 45. G → starts; offense ≥ 78 and ambition ≥ 60 → PP; else more ice | 80 − role_sat + ambition×.24 |
| contract_clarity | contract_sat < 38, importance ≥ 58 | 72 − sat + money_focus×.20 |
| winning_concern_meeting | winning_sat < 34, competitiveness ≥ 65 | 70 − sat + comp×.25 |
| development_meeting | age ≤ 24, dev_sat < 45 | 66 − sat + ambition×.20 |
| personal_check_in | stress > 65 | stress + 8 |
| reporter_confrontation | char < 68, 21-day cooldown, media_score > 84, vol ≥ 52, char < 62 (score −0.85/pt of char above 58) | media_score (capped 88 on your team; max 2 media items) |
| ↳ reporter_altercation | media_score > 122, vol > 78, char < 48 → **12%** roll | |
| unheralded_leader | OVR < 80, room_value ≥ 72 | room_value − 8 |
| mentor_session | mentor niche, target ≤ 24 (one of 3 youngest/least confident, not mentored in 30 days) | 58 + max(0, 52 − target conf) |
| glue_intervention | glue/peacemaker, room tension ≥ 44 | 62 + tension×.35 |
| blame_game (pair) | pair tension ≥ 68 and (char floor < 50, or disruptive and < 58). Skipped if a peacemaker is present and tension < 78 | tension + vol×.35 + max(0, 52−floor)×.45 (+6 disruptive) |
| ↳ teammate_fight | disruptive, tension ≥ 84, vol ≥ 74, floor < 46 → **18%** roll | |

### 6.2 Selection — `_u_generate_daily_interactions_impl`
- Your team first, then each CPU team.
- **Your team:** skip if ≥ 2 pending, or a request arrived < **4 days** ago.
- Weighted pick among the **top 5** candidates (weight = score).
- Pass if `score ± uniform(12) ≥ threshold` **and** `uniform(0,100) ≤ cap`. Your team: threshold 52, cap 60 (≈ 60%). CPU: threshold 61, cap 28.
- League max 8 CPU interactions/day.

### 6.3 Queue or auto-resolve — `_u_queue_or_resolve`
- **Your team** (`requires_action`): goes into `universe_interaction_queue` (max 12) and stamps `last_player_meeting_request_day`. Notification level by stakes: critical → popup (3), high → banner (2), else inbox (1).
- **CPU teams:** resolved immediately with the default choice. Later published as rumors (§2.4).

### 6.4 Choices and outcomes (the six "extended" request types)
Expire after **3 days**. Default = `honest`. `_u_apply_outcome` routes profile changes, relationships, readiness, potential, attributes, promises, and public coverage.

| Request | Honest | Promise / commit | Decline / dismiss |
|---|---|---|---|
| More ice | GM trust +4, focus +2, readiness +0.25/5d | morale +5, conf +3, trust +4 → **role promise 5 games** (kept: +0.6 / broken: −2.5) | trust −6, morale −5, readiness −1.0/10d |
| PP time | trust +3, focus +2 | morale +4, conf +2 → **PP promise 8 games** (kept: potential +0.5, off. awareness +0.4 / broken −2.0) | morale −3, trust −3, readiness −0.6/7d |
| Starts (G) | trust +4, focus +2 | conf +4, morale +3 → **start promise 6 games** (kept: potential +0.35 / broken −2.5) | morale −4, trust −3, readiness −0.8/10d |
| Contract clarity | trust +5, focus +2 | *commit*: trust +4, morale +3, readiness +0.4/7d | *deflect*: trust −5, media +2, readiness −0.65/10d |
| Winning concern | trust +4, focus +2 | **winning promise 12 games** (kept +0.75 / broken −3.0) | trust −8, morale −5, belonging −2, readiness −1.2/14d |
| Development | focus +4, trust +3, readiness +0.35/8d, potential +0.2 | *skills*: conf +3, off. awareness +0.4 for 8 games, potential + | *wait*: trust −2, morale −1 |

Other kinds (mentor, glue, blame game, fight, reporter, check-in, unheralded leader) use the V2 scene builder `_u_make_interaction`.

The separate **GM meetings desk** (`start_gm_player_meeting` / `resolve_gm_player_meeting`) is where you call players in yourself. Outcomes are scaled by harshness × personality (`_scale_negative_meeting_outcome`), walkouts are recorded, and it includes NTC/NMC waiver asks using `clause_consent.preview_waiver_chance`.

### 6.5 Ignoring a request — `_u_expire_interactions`
A request past `expires_day` that you never answered: **GM trust −3, morale −2**, banner "X noticed you never answered". Non-blocking ones auto-apply the default choice.

### 6.6 Promises — `_u_tick_promises` + `_u_promise_progress_tick`
- First tick stores a **baseline snapshot** (line rank, TOI, PP/PK unit, starts, roster IDs).
- Each game, progress +1 if delivered:

| Promise | Counts as delivered |
|---|---|
| role_opportunity | dressed and (better line rank, or TOI +1.5 min, or better PP unit) |
| power_play / penalty_kill | on a better PP/PK unit (unit 1 = top) |
| goalie_start | start count went up since last tick |
| call_up | on the NHL roster |
| mentor_assignment | dressed and on the same line as the mentee |
| winning_commitment | any newly added player with OVR ≥ 80 (needs just 1) |

- Required progress = due_games (1 for call-up and winning).
- When `games_remaining` hits 0:
  - **Kept:** trust +8, morale +6, success readiness for 10 days, potential/attribute rewards, story heat 32 (private).
  - **Broken:** trust −6, morale −4, failure readiness for **21 days**, **public** story heat 67, banner.
  - **Waived:** trust −1.
- Saving lines calls `credit_line_promises` immediately.

---

## 7. Major league incidents — `_u_run_major_league_event_pass`

- **Season budget:** `target = randint(3, 5)`, seeded by season (deterministic).
- **Daily roll:** needs ≥ 14 days since the last one. `chance = clamp(0.0035, 0.12, remaining_events / remaining_calendar_days × 1.25)`. The 1.25 factor front-loads so the budget is reliably spent; late in the season the chance climbs.
- **24% team incident** (random team, any club including yours):

| Type | Pick | Sanction (→ `universe_team_sanctions`, `universe_cap_penalties`, `universe_forfeited_picks`) | Heat |
|---|---|---|---|
| ILLEGAL_TEAM_WORKOUTS | 34% | fine $0.25–1.5M; **18%** next year's 2nd | 84 |
| CAP_CIRCUMVENTION | 28% | fine $1–5M, cap penalty $2–6M; **55%** next year's 1st | 94 |
| EXECUTIVE_MISCONDUCT | 20% | fine $0.5–4M | 84 |
| UNDER_TABLE_PAYMENTS | 18% | fine $2–8M, cap penalty $3–7.5M, 1st forfeited | 94 |

All: team morale −0.08, org pressure +0.18.

- **76% player incident** (uniform over every NHL roster player), weights 1.35 / 1.25 / 1.15 / 0.45 / 0.35 / 0.01:

| Type | % | Availability | Readiness OVR (scaled by mental OVR) | Duration |
|---|---|---|---|---|
| PLAYER_ARRESTED | 29.6 | investigative leave 5–14 days | −8 … −18 (`11 + (60−mental)×.12`) | 60–120 d |
| LEAGUE_SUSPENSION | 27.4 | 8–30 games | −6 … −16 | 45–90 d |
| MAJOR_PUBLIC_ALTERCATION | 25.2 | 3–8 games | −7 … −16 | 30–75 d |
| GAMBLING_VIOLATION | 9.9 | 20–50 games | −10 … −20 | 90–180 d |
| PLAYER_BANNED | 7.7 | 180–365 days | −12 … −22 | = ban |
| PLAYER_DEATH | 0.22 | permanent, `retired = career_ended = True` | — | — |

Player hit: morale −14, confidence −10, media +25, plus focus/composure and shot/assist involvement stat penalties. Team: morale −0.07 (−0.20 for death), org pressure +0.20 (+0.12).

**Availability:** `universe_player_is_available()` returns False for suspended / banned / investigative_leave / deceased. Days tick in the daily director; games tick via `tick_universe_player_availability_after_game`.

**Presentation:** level 4 with `force_league=True`, so **every** major incident shows the blocking BREAKING NEWS popup, even on other teams.

---

## 8. Postgame — `apply_universe_postgame` (every game, both teams)

Per player, from the box score:

| Situation | Morale Δ | Confidence Δ |
|---|---|---|
| base | win +1.5 / loss −1.4 | +1.1 / −1.0 |
| skater 3+ pts / 2 pts / 1 pt | +4.8 / +2.6 / +1.2 | +3.4 / +1.8 / +0.8 |
| skater 0 pts with 15+ min TOI | −0.9 (W) / −1.6 (L) | |
| 2+ goals | +1.4 | |
| 6+ shots, 0 pts | −0.8 | −0.6 |
| goalie shutout (or 0 GA on 18+ SA) | +5.5 | +4.0 |
| goalie SV% ≥ .930 on 20+ SA | +3.2 | +2.4 |
| goalie SV% ≤ .860 on 18+ SA | −4.0 | −3.2 |
| goalie 5+ GA | −2.6 | |
| 4+ PIM | media stress +1.5 | |

**Mean reversion:** morale pulls 7% toward a resting baseline `56 + (role_sat−55)×.25 + (gm_trust−58)×.15 + (resilience−55)×.12`, clamped 38–74. Confidence pulls 6% toward 55. While a meeting "hold" is active: deltas × 0.7 and a 4% pull toward the meeting anchor. Results are pushed to `player.psych`.

**Postgame stories:**
- **Selfish game moment** (losses only): `risk = max(0, 48−char)×.006 + max(0, 48−role_sat)×.004 + max(0, tension−55)×.002`, roll `< min(0.28, risk)`. Effect: coach trust −4, discipline −1 for 3 games, public story heat 52, social burst. A char-30 / role-30 player in a tense room ≈ 18–20%.
- **Winning detail** (room value ≥ 72, OVR < 80, 0 points): **8%** on a win / **3.5%** on a loss, heat 28, public 55%.

---

## 9. Media layer

### 9.1 Press conferences — `_maybe_queue_press_conference`
- Your team only; story heat ≥ 52 or priority HIGH/CRITICAL; not already queued.
- Max **3 open**: a new one only bumps the coolest pending if it is ≥ 10 heat hotter.
- Answer window **4 days** (`PRESS_CONFERENCE_WINDOW_DAYS`). Questions are built from story context (`_build_press_questions`).
- `apply_press_conference_response` looks up `_PRESS_RESPONSE_SPECS[question:response]` + tradeoffs. Generic fallbacks:
  - no-comment/cold → media +2, fans −2, heat +4
  - firm/honest/diplomatic/support → fans +1, morale +1, heat −4, GM trust +1
- Expired conferences resolve via `expire_press_conferences`.

### 9.2 Agent leaks — `_maybe_agent_leak`
Only for claim-type / trade / CONTRACT_YEAR_HEAT / ROLE_FRUSTRATION / COACH_HOT_SEAT / GM_JOB_SECURITY / WINNING_CONCERN stories. Chance = `leak_tendency × 0.35` (default 0.3 → **10.5%**). Posts a partial-truth claim (fact vs public claim are kept separate via knowledge layers).

### 9.3 Social and Reddit
- `_spawn_social_posts`: every story gets a reporter post. If heat ≥ 22 → **55%** a fan post (35% meme if heat < 45). Likes ≈ heat×45.
- Player self-posts: non-quiet personalities 48%. Ambient social: skip 52%, quiet players skip a further 65%.
- Reddit threads: subreddit by team; title 72% templated, OP 68%; 35% user-team-fan top comment; upvote ratio by tone (0.88–0.97 positive, 0.55–0.75 negative). Sentiment feeds back into fan confidence via `apply_fan_engagement_delta` (`REDDIT_ENGAGEMENT_WEIGHT 2.5`; fan conf += delta×8, org pressure −= delta×0.04).
- Knowledge layers (`_assign_knowledge_layers`): each story is tagged fact / report / claim / speculation and confirmed / reported / rumour / chatter. Market tone (`_apply_market_media_tone`) adjusts by team market size.

### 9.4 Follow-ups — `_process_storyline_followups`
A queue of `{due_day, kind}` items, deduped by kind + player + reporter:
- `reporter_response`: after a reporter confrontation
- `carry_followup`: 12 days after a superstar-carry story; contract-year angle if applicable
- `shutout_coach`: after a repeat shutout on your team

### 9.5 Arcs and eras
- Stories are grouped into arcs (`_arc_key_for_storyline`). Each beat is appended with a phase from `_arc_phase_from_beat_index`. `narrative_director_score` ranks what to surface.
- In the offseason `seal_narrative_season` archives the season, and `build_narrative_eras` builds the history view.

---

## 10. Trade demands (separate engine)

`trade_demand_engine.process_trade_demand_day` owns formal demands; it runs in the `full=True` daily pass. The storyline-side generator `_u_maybe_create_trade_demand_from_state` is **disabled** (`return 0`; the old 5.5%/day severe-state path is unreachable).

When a demand is created via `_u_create_trade_demand`, the immediate public-leak chance is:
- character < 35 → `min(0.92, 0.62 + agent_leak×0.30)`
- 35–44 → `0.18 + agent_leak×0.18`
- ≥ 45 → 0

`_u_tick_trade_demands` handles later leaks and deadlines. `resolve_universe_trade_demands_after_trade` closes demands when the player moves. Trades also call `apply_trade_universe_relocation`, which adds relocation strain 8–20 and updates city/family life.

---

## 11. CPU morale management — `_u_cpu_morale_management`

Stops CPU rosters drifting to extremes without a GM to manage them. Per CPU player, daily:
- morale < 38 → **22%**: + (2 + max(0, coach_trust − 50) × 0.06)
- morale > 88 → **10%**: −1.5

---

## 12. Debug and inspection hooks

| Function | Use |
|---|---|
| `build_storyline_debug_payload(session)` | cooldown table, active cause stories, blocked log |
| `session._storyline_blocked_log` | why a story on your team was refused (last 80) |
| `session.universe_daily_snapshots` | per-day culture + counts (last 120) |
| `session.universe_game_contexts` | last 60 pre-game contexts with full player modifiers |
| `build_story_impact_report(session)` | what each story actually changed (OVR before/after, receipts) |
| `backend/scripts/probe_story_personality_sim.py`, `overnight_franchise_soak.py`, `bench_franchise_advance.py` | offline soak / benchmark of the passes |
| env `NHL_FRANCHISE_DEBUG=1` | dev logging of daily counts and blocked stories |
