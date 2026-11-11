# Player growth report: NHL, AHL and rights-held players

Status: code-read analysis, not yet measured. Nothing here has been confirmed by running a season. The first step is instrumentation (see "Next steps").

## 1. What the growth budget does

Core function: `calculate_season_growth_budget` in `SimEngine/app/sim_engine/progression/development.py` (around line 3368). It returns a bounded fraction of 1.0 (about 99 displayed points) for one season.

- **Gap-based approach.** The approach is `min(0.080, gap_exp * 0.50 + 0.030)`. Below a small gap it drops to `min(0.028, ...)`. So the most a player can gain in a normal year is about 8 displayed points before any modifier.
- **Step caps.** Growth is capped at `0.068` for age 25 and under and `0.038` above that, and `gap_exp` further limits the step.
- **Ceiling target.** Growth heads for the hidden ceiling (`_growth_ceiling99`). For players 25 and under whose scouted potential is at least 6 points above current OVR, the target is a fraction of the scouted number, not the hidden ceiling.
- **Multiplier stack.** Many factors multiply the budget: morale (`0.78 + 0.48 * morale`), development rate (`0.70 + 0.70 * rate`), season surplus, overperformance, runway, production and ice time. Several of these can each take 20–30% off, and the effects compound.
- **Declines.** Players past their peak take negative budgets, limited by `-0.030` and by form.

## 2. NHL players

- Growth runs once in-season (checkpoints at 20, 41 and 61 GP, `apply_in_season_performance_checkpoints`) and once at season end.
- The annual budget is split: `_IN_SEASON_POOL_SHARE = 0.40` in-season and `_SEASON_END_POOL_SHARE = 0.58` at season end. That sums to 0.98, so about 2% of each year's budget is never paid.
- `_PERF_CHECKPOINT_CAPS` limits the in-season pulse to +1, +2 and +3 across the three checkpoints, and `_PERF_CHECKPOINT_NEG_FLOOR` limits the drop to -1.5.

Why NHL growth can feel flat:
1. Pulses are capped small, so most of a player's year lands at season end. A poor end-of-year result can pull the net gain close to zero.
2. Morale, usage and rate multipliers stack multiplicatively, so a player with mixed signals gets far less than the base budget.
3. The step caps (0.068 and 0.038) and the 0.080 approach cap mean a 15-point gap takes two or more years even on a good path.

## 3. AHL players

- The AHL gets a bonus only through `_young_ahl_development_opportunity`. It applies when all of these are true:
  - age is 23 or under,
  - `_dev_growth_league` contains "AHL", and
  - the player has a top role, top power-play usage, `toi_quality` of at least 0.75, power-play share of at least 0.35, or usage of at least 1.02.
- If `_dev_growth_league` is not set when the budget is calculated, the bonus never fires, and an AHL player is treated like any other development player. **This needs verifying.** It is the most likely cause of AHL growth looking missing.
- Overperformance pays an "earned" bonus for young players (`overperf * 0.055` for age 23 and under), capped at 0.026. That only fires when production is stamped before the budget runs.

## 4. Rights-held players (drafted, not signed)

Entry point: `develop_unsigned_prospect` in `backend/services/unsigned_prospect_development.py`, run by `run_unsigned_prospect_development_pass`.

- **Who is eligible.** `_consider_unsigned` skips signed players and anyone whose status is NHL or active. A rights-held player must have a rights team or be drafted.
- **League quality.** `_league_quality` multiplies growth by 0.62 to 0.80 depending on league, with 0.65 as the default. The default applies to any league not on the list, so an unlisted league grows slower.
- **Under-challenged penalty.** If league quality is 0.65 or below and OVR is above 0.70, the player is flagged `underchallenged` and loses growth.
- **Clamp at expected ceiling.** The new OVR is `min(expected_ceiling - 0.005, ...)`, so a player can never grow past the expected ceiling. If that ceiling was set low at draft time, growth stops early.
- **Yearly payment.** The budget is either the leftover from in-season pulses or the full year cap from `prospect_year_payment`, with ±8% jitter. Growth only runs if `development_applied` is not already set for the season.
- **Dependency on the schedule.** The pass runs at a fixed point in the season. If it runs before in-season pulses are stamped, it may pay the full year and then the pulses add more, or the reverse, and the ledger hides which one happened.

Why rights-held growth can feel absent:
1. The ceiling clamp, the league-quality multiplier and the under-challenged penalty all work against the player.
2. The clamp uses a ceiling that may be stale, so a player can hit it early.
3. A player who does not play in a listed league gets the 0.65 default.

## 5. Next steps (instrument first)

1. Log, per player and per season: `budget`, every multiplier, the cap that fired, `ovr_before`, `ovr_after`, and the `_dev_growth_league` value at budget time. `develop_unsigned_prospect` already builds a `ledger`, so the rights-held path can write it out.
2. Run one simulated season and read the log. Confirm whether AHL players are getting the opportunity flag, and whether rights-held players hit the ceiling clamp.
3. Check the 2% unallocated share. If it matters, route the remainder into the season-end pool.
4. Once the numbers are confirmed, tune the multiplier stack first, then the step caps.

## Not read yet

- `prospect_year_cap` and `_prospect_payment_multiplier`, which set the yearly prospect payment.
- The main NHL season-end path in `SimEngine/app/sim_engine/engine.py`.
