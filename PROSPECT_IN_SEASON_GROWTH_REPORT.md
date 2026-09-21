# Prospect In-Season Growth & Potential Rise Report

**Question:** Do prospects grow gradually over the season (OVR creeping up), and does their potential rating rise when they earn it?
**Short answer:** No. Neither is reliably there today.

- **OVR growth is not gradual.** Prospects get one lump of growth per season, at a random point in the season. Only NHL-roster players get a gradual, multi-pulse in-season drift.
- **Potential rises rarely and in tiny steps.** The only in-season potential movement for prospects is a small once-per-season roll (+/-1 or 2 pot, ~8% breakout chance) and a WJC bump for draft-eligible players. The performance-based ceiling raise exists but is wired to the same one-shot path and is probably starved of production data for prospects.

Findings come from reading the code. Nothing was run, so the "unverified" items below are flagged as such.

---

## 1. What exists today

| Player group | In-season OVR path | Shape | Potential path |
|---|---|---|---|
| NHL roster (`team.roster`) | `_nhl_in_season_development_tick` -> `apply_in_season_development_pulse` | **Gradual**: pulse every 8 calendar days, capped to the in-season pool (32% of the annual budget) | None in-season (year-end only) |
| AHL / ECHL, free agents, overseas FAs | `_depth_pool_progression_tick` -> `run_player_progression` | **One lump** per season | Once-per-season drift roll |
| Junior / NCAA / Europe (`league.development_leagues`) | Same `_depth_pool_progression_tick` | **One lump** per season, at a random date | Once-per-season drift roll, plus WJC bump |
| Drafted, unsigned prospects | `run_unsigned_prospect_development_pass` (offseason only) | **One lump** at season end | **None**: this path never touches potential |
| `team.prospect_pool` (SimEngine pipeline) | `progress_prospects` -> `_develop_prospect_one_year` (offseason only) | One lump at season end | Not touched here |

Key references:
- NHL-only tick: [franchise_sim.py:12468](backend/services/franchise_sim.py:12468). It loops `tm.roster` only (line 12492-12493).
- Pulse logic: [development.py:2482](SimEngine/app/sim_engine/progression/development.py:2482). Pool shares are `_IN_SEASON_POOL_SHARE = 0.32` and `_SEASON_END_POOL_SHARE = 0.58` (line 1840).
- Depth tick: [franchise_sim.py:11817](backend/services/franchise_sim.py:11817). It shuffles the whole pool and processes only the first 72 per call, every 5 calendar days (line 12343).
- Season-start OVR snapshot (`season_start_ovr`) covers `roster`, `ahl_roster` and `echl_roster` only ([franchise_sim.py:12446](backend/services/franchise_sim.py:12446)). It never covers dev-league prospects or `prospect_pool`, so there is no "growth since September" figure to show for them.
- Offseason unsigned pass: [unsigned_prospect_development.py:33](backend/services/unsigned_prospect_development.py:33), called from [franchise_offseason.py:2998](backend/services/franchise_offseason.py:2998).

## 2. Why prospect OVR is not gradual

1. **The ledger makes it a single event.** `apply_player_development` returns immediately once `ledger["development_applied"]` is set ([development.py:2182](SimEngine/app/sim_engine/progression/development.py:2182)). The first time a prospect is drawn by the depth tick, they receive the whole annual budget in one go and are locked for the rest of the season.
2. **The timing is random.** The pool is shuffled and 72 players are taken per tick. A prospect can jump in October, or in March, or not before the offseason pass. Two similar prospects can look completely different on any given date.
3. **Coverage is thin.** All AHL, ECHL, FA and dev-league players share those 72 slots per tick, so many prospects reach the season's final weeks with no growth at all. (Pool size not measured.)
4. **The NHL pulse system is the right model but is not used for prospects.** It splits growth across many small pulses, respects the age and gap gates, and tracks `_in_season_growth_spent_01`. Prospects, who grow the most, are excluded from it.

Effect on the game: a top prospect's OVR sits flat and then jumps once, or sits flat all year and jumps in the offseason report. That is the opposite of the gradual ramp the user wants.

## 3. Why potential does not rise

There are three potential mechanisms in the code. For prospects, none does the job.

| Mechanism | Where | What it does | Problem for prospects |
|---|---|---|---|
| `apply_potential_drift` | [potential.py:95](SimEngine/app/sim_engine/progression/potential.py:95) | Once per season: ~7.8% breakout (+1 or +2 pot), ~10% bust, else `no_change` | Rises are rare and tiny. Even a prospect with a huge year usually gets nothing. Potential is floored at current OVR, so it can only follow growth, never lead it. |
| `reevaluate_ceilings_from_performance` | [development.py:2020](SimEngine/app/sim_engine/progression/development.py:2020) | Overperformance raises expected/active/max ceiling by up to ~+2.5 to +4.5 | Runs only inside `apply_player_development`, so once per season, right after the lump growth. It needs `production_score`, which prospects probably lack (see below). |
| WJC bump | [franchise_sim.py:12890](backend/services/franchise_sim.py:12890) | Post-tournament skill and potential nudge for draft-eligible players with tournament GP | Works, but was limited to draft-eligible players. **Corrected:** it now covers every real prospect who played (drafted ones too). |

Other gaps:
- **Unsigned drafted prospects never get a potential update.** `develop_unsigned_prospect` grows OVR only. It never calls `apply_potential_drift` or `reevaluate_ceilings_from_performance`.
- **The offseason prospect pipeline moves OVR only.** `_develop_prospect_one_year` sets tier-based growth and bust/steal pressure but does not raise potential.
- **Prospect production is probably not reaching the ceiling logic (unverified).** `production_score` on prospects lives inside the weekly stock fields in [prospect_league_scoring.py](SimEngine/app/sim_engine/generation/prospect_league_scoring.py) (`weekly_production_score`, line ~1287, ~1475). The ceiling function reads `player.production_score` / `recent_performance_score` / `points_signal` / `production` and defaults to 0.5 when they are missing. If those attributes are not set on dev-league players, overperformance is always about zero and the breakout branch never fires. Worth confirming first.
- **Hidden vs shown potential.** The draft board shows scouted, fogged potential bands (`build_potential_intel`, [draft_ranking_logic.py:1688](backend/services/draft_ranking_logic.py:1688)). Even if true potential moves, the UI needs a deliberate path to show "potential rising" to the user, e.g. through scouting confidence or a stock-rising label. Otherwise the user will not see the change.

## 4. What "correct" should look like

Target behaviour:
- Overall rises gradually across the season: roughly 8-12 small steps for a growing prospect, most of the annual gain landing by the deadline, with the remainder in the offseason.
- Potential rises when a prospect outperforms, in visible steps (+1 to +3), and only up to the player's maximum ceiling.
- Ceiling raises come from evidence: production versus expectation for that league and age, plus development-league quality and ice time.
- Bust-style prospects can also stall or slip, so the mechanic is not one-directional.

## 5. Recommended implementation

1. **Add a prospect in-season pulse.** Add `_prospect_in_season_development_tick(session)`, called next to `_nhl_in_season_development_tick`, that iterates all dev-league prospects plus `prospect_pool`. Reuse `apply_in_season_development_pulse` (it already handles age <= 22 boosts and the pool cap). Adjust the pulse for prospects:
   - use `unsigned_prospect_development` context (league quality, ice time, coaching, org plan) as the pulse modifier;
   - skip the `age >= 32` regression branch (irrelevant to prospects).
2. **Stop the depth tick from lumping prospects.** Exclude prospects from the one-shot `run_player_progression` in `_depth_pool_progression_tick`, or make it consume only the season-end share. Otherwise the ledger blocks the pulses. Use the existing 32% / 58% split so the offseason pass grants the remaining ~58%, not a fresh full budget.
3. **Extend the season-start snapshot** ([franchise_sim.py:12446](backend/services/franchise_sim.py:12446)) to dev-league players and `prospect_pool`, so `season_start_ovr` exists for them. That gives the UI a "+N this season" figure.
4. **Add an in-season potential review.** A once-a-month check (or every N pulses) that:
   - feeds the prospect's real weekly/season production versus league-age expectation into `reevaluate_ceilings_from_performance`;
   - keeps the once-per-season ledger guard but allows several small, capped raises, each bounded by `maximum_ceiling`;
   - caps total in-season potential gain (suggest +4 pot per season, +6 for a transcendent-flag prospect).
5. **Wire potential into the unsigned offseason pass.** Call the same ceiling and drift logic from `develop_unsigned_prospect`, so drafted-but-unsigned players are not permanently frozen on potential.
6. **Surface it.** Add "potential rising" and "OVR +N since October" fields to the prospect payload, and a scouting-gated label on the draft board so fog is kept but movement is visible.
7. **Tests.** Add tests beside `backend/tests/test_prospect_league_scoring.py`:
   - monotone-ish OVR trajectory over 30 sim days for a high-runway 18-year-old;
   - total in-season gain never exceeds the 32% in-season pool;
   - an overperforming prospect's potential rises; an average one's does not;
   - no double-application of annual growth (in-season pulses plus the offseason pass together stay within the annual budget).

## 6. Risks / things to check before building

- **Double growth.** In-season pulses plus the offseason pass must together stay inside the annual budget. The `_in_season_growth_spent_01` counter and the ledger already exist for this, but the offseason unsigned pass uses its own season id and budget, so confirm they do not stack.
- **Potential inflation.** [PLAYER_DEVELOPMENT_SYSTEM_REPORT.md](PLAYER_DEVELOPMENT_SYSTEM_REPORT.md) already flags a "ceiling ratchet" where potential chases OVR with no brake (issues 1 and 3). Adding more ways to raise potential makes that worse unless every raise is bounded by `maximum_ceiling` and the per-season cap above.
- **Performance cost.** The NHL tick runs every 8 days over about 32 rosters. Adding every dev-league prospect (hundreds) is fine if pulses are gated to prospects still under about 22 with runway, as the NHL pulse already does.
- **Scoreboard consistency.** `season_start_ovr` is compared against displayed OVR in the offseason review. If prospects start using it, the year-end review numbers will change.

## 7. Suggested order

1. Confirm whether prospects have `production_score` (or an equivalent) as a player attribute. This decides whether the ceiling raise can work at all.
2. Build the prospect pulse tick and remove prospects from the lump path (items 1-3 above). This delivers the gradual OVR growth.
3. Add the in-season potential review and the payload fields (items 4-6).
4. Add tests (item 7).

---

## 8. Implemented

| Change | Where |
|---|---|
| Gradual OVR: 22 pulses/season on a steady schedule, ~85% of the annual budget in-season | `apply_prospect_in_season_pulse` in `development.py` |
| Monthly potential review from real production, independent of OVR, capped +4 pot/season (+6 transcendent) | `apply_prospect_potential_review` in `development.py` |
| Offseason (unsigned + year-end) pays only the leftover, once | `prospect_offseason_leftover`, used by `unsigned_prospect_development.py` and `apply_player_development` |
| Unsigned pass now also reviews potential | `unsigned_prospect_development.py` |
| Tick, season-start snapshot, prospect discovery, payload fields | `backend/services/prospect_in_season_growth.py`, wired in `franchise_sim.py` |
| Depth tick no longer lumps prospects | `_depth_pool_progression_tick` |
| WJC development now applies to all real prospects who played, including drafted | `_apply_wjc_development_to_prospects`, `_wjc_stock_rows` |
| Tests | `backend/tests/test_prospect_in_season_growth.py` |

Not done: scouting-gated "potential rising" label on the draft board (payload fields `ovr_change_season`, `potential_change_season`, `potential_trend` exist for the UI to use), and objects in `team.prospect_pool` that have no `ratings` dict still only develop in the offseason.
