# Transcendent tanking

Headless Ottawa franchise, seed 4242, generated players, one forced transcendent (Thomas Collins). The before pass logged the old rules. The after pass logged the strength-based rules through the trade deadline (calendar day 176).

## What was wrong

Contender status came from `gm_window` / `window` / `status` text. A club labelled contend stayed out of the lottery race even when its roster rank and points pace said otherwise. Pressure for that label was a flat 12. Clubs that had traded their own first were capped and then skipped for the day, so they neither tanked nor tried to buy a pick. The daily hook wrote storylines. The only roster change was a 5% scratch of the oldest veteran, and that set was never cleared at the new season. Trade interest topped out around +0.18, and the market bump was capped at +0.06. The draft lottery did not read tank pressure.

## Before (old rules, days 0–22)

The day-mode sim stopped on a user interrupt at day 22, so this sample does not reach the deadline. It is enough to see the old decisions.

| | |
| --- | --- |
| Clubs | 32 |
| Modes | none 7, soft_sell 19, hard_tank 6 |
| Labelled contender while strength rank said no | 6 (team 0 was strength rank 28, pressure 12, mode none) |
| Non-contenders with a sell, dump, or signing | 0% |
| Real roster moves | none |

Log: `docs/tank_probe_before.json`.

## After (strength rules, through the deadline)

A club is a contender only when its combined OVR and points-pace rank is inside the top group and it is within four projected points of the playoff line. Everyone else is at least a seller. Pressure rises with the playoff-line gap. Owning an unprotected first and sitting far out of the race is hard-tank. A club that does not own its first stays a seller and sets `lottery_pick_intent` to buy.

| | Before (day 22) | After (day 176) |
| --- | --- | --- |
| Mode none | 7 | 4 |
| Soft sell | 19 | 0 |
| Seller | 0 | 19 |
| Hard tank | 6 | 9 |
| Non-contenders | — | 27 |
| Non-contenders with a sell, dump, or signing | 0% | 96.3% (26 of 27) |

Actions across the season, counting a club once: sell 25, dump 7, sign 20, scratch 25, bid 3. The three bids are clubs that no longer held their own first. Storyline hooks still fire (shop, youth, rental, hard-tank, abort).

The worst 16 by projected points at the deadline were all at pressure 70–95. Lottery odds for that order, after the pressure weight:

| Slot | Base odds | Weighted odds | Pressure |
| --- | --- | --- | --- |
| Last place | 18.5% | 19.82% | 95 |
| 16th non-playoff | 0.5% | 0.37% | 70 |

The May lottery itself was not executed in the probe. The weight is what `_run_draft_lottery` passes in for non-playoff clubs only. Playoff clubs are not in that list.

Log: `docs/tank_probe_after.json`.

## What the sim does now

- Weekly log per club: label, strength rank, pressure, mode, pick ownership, actions (`TANK week=...` and `session.tank_debug_log`).
- Pressure refreshes each week, on deadline day, and when points move by four or the roster changes.
- Non-contenders are sellers in the existing trade matcher, with a higher selloff budget while a transcendent is in the class. High-salary veterans are listed for a dump even when the club is not in cap hell.
- Once a week the CPU scratches expensive veterans, tries a league-minimum signing of a player below the roster median, and runs the existing CPU trade executor. Youth and cheap depth sort above expensive veterans. Saved lines are left in place for the user's club and for contenders.
- Tank scratches clear when the season year changes and again at rollover.
- Sellers accept a worse return for players-for-picks, and a club bidding for a first gets a larger interest bump.

## Not verified here

- A full save/load round trip of `_tank_scratched_ids`. They live on the team object, so they travel with the session, and the season-change clear is unit-tested.
- The probe stopped at the deadline, so the scratch clear was not observed on a live rollover.
- One non-contender (team 5) had storylines only, no sell or dump, by the deadline.
- `tests/test_trade_foundation.py` already had failures that this change does not target (clause rejection, pick value, the 82-game ledger cap). They were not rewritten.
