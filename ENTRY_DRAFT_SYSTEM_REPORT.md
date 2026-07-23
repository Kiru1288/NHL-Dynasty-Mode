# Entry Draft System Report

How drafting actually works in **NHL Franchise Mode** today: pick execution, rights by league, unsigned prospects, pick trading around the draft, and the full set of draft entities.

**Live source of truth:** `backend/services/franchise_entry_draft.py`  
(Not the SimEngine `franchise/` mirrors, and not the universe-only `draft_sim.py` mid-draft trader.)

---

## 1. Architecture overview

```
Offseason stages
  awards → retirements → salary_cap → development
  → draft_lottery → draft_combine → draft → re_sign → free_agency → roster_cleanup

Prospect world (season-long)
  league.development_leagues
    → franchise_sim.build_draft_class_rankings()
    → draft_ranking_logic / draft_prospect_profile
    → session._cached_draft_class_rankings  (Draft Class UI)

Live Entry Draft
  franchise_entry_draft.initialize_entry_draft / _execute_pick
    → session.draft_state + draft_payload
    → player flags + team.prospect_pool + team.reserve_list
    → league.draft_pick_registry[pick].resolved = True

ELC (separate phase)
  contract_economy.sign_elc / assign_elc_contract
  auto on NHL promotion / late roster_cleanup promotion pass
```

| Layer | Job | Key files |
|--------|-----|-----------|
| Live draft engine | Pick execution | `backend/services/franchise_entry_draft.py` |
| Session bags | Flags / payloads | `backend/services/franchise_session.py` |
| Offseason orchestration | Lottery → combine → draft | `backend/services/franchise_offseason.py` |
| Rankings / board | Season scouting | `franchise_sim`, `draft_ranking_logic.py`, `draft_prospect_profile.py` |
| Pick ownership | Season trades → draft order | `SimEngine/.../trades/trade_pick_registry.py` |
| Contracts / reserve | Unsigned ledger + ELC | `backend/services/contract_economy.py` |
| Universe sim (separate) | Sim mid-draft trades | `SimEngine/.../draft/draft_sim.py` |
| UI — draft floor | Live picks | `frontend/src/events/entryDraft/EntryDraftMenu.js` |
| UI — class board | Season rankings | `frontend/src/screens/DraftClass.js` |
| UI — lottery | Odds theatre | `frontend/src/screens/DraftLottery.js` (+ backend lottery is real) |

---

## 2. The art of drafting a player (what a pick actually does)

### Preconditions

1. Offseason reaches **draft** after **draft_combine**.
2. `initialize_entry_draft` requires combine done (or raises).
3. Order is built once: lottery slots 1–16 + rest of field → 7×32 = 224 slots, with registry ownership applied per slot.

### User pick path

1. `EntryDraftMenu` → `submitDraftPick` → `POST /api/franchise/draft/pick`
2. `execute_franchise_draft_pick` → `execute_user_draft_pick` (must own the clock)
3. Shared `_execute_pick(..., user_initiated=True)`

### CPU pick path

- Single: `execute_cpu_draft_pick`
- Batch: `sim_entry_draft_to_user_pick`, `sim_entry_draft_round`, `complete_entry_draft`
- Choice via `_cpu_select_prospect` / `_cpu_draft_score` (board, philosophy, needs)

### Shared execution (`_execute_pick`)

1. Resolve eligible board entry (not already drafted; still findable in `development_leagues`).
2. Validate slot owner / user turn.
3. Find live player entity in a development-league club.
4. Classify pick (`_classify_pick`, `_pick_reason`).
5. **`_assign_drafted_prospect`** — the roster/rights write.
6. **`_mark_pick_resolved`** on pick registry.
7. Append `completed_picks` / `draft_results`; advance `overall_pick`.
8. Invalidate session caches.

### What is written at pick time (`_assign_drafted_prospect`)

| Write | Effect |
|--------|--------|
| Remove from junior/NCAA/EU club | Yanked out of `development_leagues` team roster |
| Draft identity | `drafted`, `drafted_by`, `draft_team_id`, year/round/overall |
| Rights (simple) | `rights_team_id`, `rights_status="drafted"` |
| Org status | `prospect_status="org_prospect"`, `status="prospect"`, `team_id` |
| Path labels | `development_path`, `post_draft_league` ∈ {Junior, NCAA, Europe}, `nhl_eta` |
| **Unsigned** | `signed_status="unsigned"`, `entry_level_contract_eligible=True` |
| Org lists | `team.prospect_pool` |
| Reserve list | `contract_economy.add_to_reserve_list(...)` |
| Archive | `session.draft_results_archive` |

### What is **not** written at pick time

- No ELC / AAV / term
- No NHL or AHL roster slot
- No CBA-style exclusive/non-exclusive window
- No `rights_expiry_year`

**Translation:** Drafting claims the player for the org and parks them **unsigned** on the prospect pool + reserve list. Signing is a later economic act.

---

## 3. Rights by league — intended labels vs reality

### What the code does today

League codes drive **board display** and a **post-draft path label**, not exclusive NHL rights calendars.

`_development_path_for` (`franchise_entry_draft.py`):

| Source league code | `development_path` |
|--------------------|--------------------|
| `NCAA*` | `"NCAA"` |
| `EU_*` | `"Europe"` |
| `CHL*` / OHL / WHL / QMJHL / USHL | `"Junior"` |
| Anything else | `"Junior"` (default) |

Board taxonomy also lives in `draft_ranking_logic._LEAGUE_PARENT_MAP` (CHL splits, NCAA, USHL, EU junior leagues including SHL/Liiga/DEL/KHL jr, etc.).

After the pick:

- `rights_status = "drafted"` (not “exclusive CHL”, “NCAA”, etc.)
- `rights_team_id` = drafting club
- No expiry, no forfeit-to-free-agency, no European exclusive window

### Rights matrix (implementation status)

| League / situation | Exclusive vs non-excl. | Duration | If left unsigned | Code reality |
|--------------------|------------------------|----------|------------------|--------------|
| CHL (OHL/WHL/QMJHL) | — | — | Org prospect | Path label `"Junior"` only |
| NCAA | — | — | Org prospect | Path label `"NCAA"` only |
| Europe (SHL/Liiga/KHL jr/…) | — | — | Org prospect | Path label `"Europe"` only |
| USHL | — | — | Org prospect | Treated as Junior |
| AHL | N/A at draft | — | — | Not a draft-rights class; live board comes from `development_leagues` juniors/college/EU |
| Unsigned drafted NHL claim | Soft org claim | **No expiry modeled** | Stay until ELC / promo | See §4 |

**Gap:** Real NHL rules (≈2-year CHL exclusive, NCAA until leave school + sign window, European exclusive / non-exclusive) are **not modeled**. Path labels are descriptive only.

---

## 4. Leaving a player unsigned

### Can drafted players stay unsigned?

**Yes — always start that way.** Every pick sets:

- `signed_status="unsigned"`
- `entry_level_contract_eligible=True`
- Reserve-list entry with the same status

### Where they live

1. Live player object on `team.prospect_pool`
2. Dict row on `team.reserve_list` (`player_id`, draft meta, `signed_status`, `player_ref`)
3. Session `draft_results` / `draft_results_archive`
4. Pick registry: `resolved=True`, `selected_prospect_id=...`

### Development while unsigned

Weak / incomplete for live franchise picks:

- Removed from their junior/college club → they leave the weekly prospect-league scoring path used by Draft Class.
- Live assign path does **not** set a full multi-year junior/NCAA sim while rights are held.
- Offseason development primarily ticks **NHL rosters**, not unsigned pool prospects in a league seasons sense.

### Rights expiry?

**No** draft-rights expiry / UFA re-entry found for unsigned drafted players.

### Pressure to sign

`run_prospect_promotion_pass` (late offseason / roster cleanup) re-ensures reserve entries and can **auto-assign ELCs** to ELC-eligible unsigned prospects. So “park them unsigned for years” is not a durable strategy once that stage runs.

Manual signing: `POST /api/franchise/contracts/sign-elc` → `sign_elc` / `assign_elc_contract` (3-year ELC, fixed ELC AAV, RFA after). On success: `signed_status="signed"`, leave reserve / update rights toward RFA.

---

## 5. Traded draft picks and the draft

### Pre-draft / in-season trades — **real**

- Registry: `league.draft_pick_registry[pick_id]`
  - `original_team_id`, `current_owner_team_id`, `protection`, `conditions`, `resolved`, …
- Created by `ensure_draft_pick_registry`; moved by `transfer_pick` during Trade Hub executions.
- At draft start, `_apply_registry_to_slot` sets each slot’s `team_id` = **current owner**.
  - If owner ≠ original → UI can show `is_traded` / via-team.
- On pick: `_mark_pick_resolved` → cannot transfer resolved picks.

Protection / conditions exist on registry rows and affect **trade valuation / AI**, but Entry Draft slot resolution does not currently convert NHL-style “top-10 protected → becomes 2nd” inside the pick engine.

### Mid-draft trades on the draft floor — **stub**

`EntryDraftMenu.TradePanel` looks for:

- `draft.trade_offers`
- `draft.draft_day_trade_offers`
- `draft.pick_trade_offers`

and shows that the backend has not returned live CPU offers yet. `get_entry_draft_payload` does **not** populate those fields.

### Mid-draft via Trade Hub — **risky gap**

General trades can still move registry ownership, but **`draft_state.draft_order` is snapshotted at init** and is not rebuilt after trades. Clock ownership can desync from registry unless something re-applies ownership (not present on the execute path).

### SimEngine universe mid-draft trades — **separate world**

`SimEngine/app/sim_engine/draft/draft_sim.py` (`try_trade_up`, `enable_draft_trades`) is for the **universe/sim draft**, not the FastAPI franchise Entry Draft.

---

## 6. Draft entity inventory

### `FranchiseSession` fields

| Field | Role |
|--------|------|
| `draft_lottery_done` / `draft_lottery_payload` | Lottery 1–16 |
| `draft_combine_done` / `draft_combine_payload` | Combine gate |
| `draft_completed` | Session done flag |
| `draft_payload` | Latest API-shaped draft dict |
| `draft_state` | Live runtime SOT during draft |
| `draft_rank_prev` / `draft_preseason_rank` / `draft_rank_snapshot_week` | Stock movement |
| `draft_stock_history` | Per-prospect rank history |
| `draft_results_archive` | Cross-year log |
| `scouting_state` | Profiles, combine results |
| `_cached_draft_class_rankings` | Board cache |
| `wjc_draft_score_boosts` | WJC stock effects |

### `draft_state` (live)

Includes: `draft_id`, `draft_year`, `current_round`, `current_pick`, `overall_pick`, `current_team_id`, `draft_order`, `pick_ownership`, `completed_picks`, `available_prospects`, `drafted_prospect_ids`, `user_team_id`, `is_user_pick`, `draft_started`, `draft_completed`, `event_status`, `draft_results`, `round_recaps`, `team_draft_boards`, `public_draft_board`, `team_needs_snapshot`, traded-slot flags, timestamps, etc.

### League / team entities

| Entity | Notes |
|--------|--------|
| `league.draft_pick_registry` | Ownership, protection, conditions, resolved |
| `team.owned_pick_ids` | Synced from registry |
| `team.prospect_pool` | Drafted prospects (unsigned/signed) |
| `team.reserve_list` | ELC-eligible unsigned ledger |
| `team.rfa_rights` | Post-contract RFA (not draft rights) |
| `league.development_leagues` | Pre-draft prospect world |

### Player attributes set on draft

`drafted`, `drafted_by`, `draft_team_id`, `draft_year`, `draft_round`, `draft_pick_number`, `draft_overall_pick`, `rights_team_id`, `rights_status`, `prospect_status`, `development_path`, `post_draft_league`, `nhl_eta`, `signed_status`, `entry_level_contract_eligible`, `team_id`, `status`, draft profile / stock snapshot fields.

### Supporting modules

| Module | Symbols / role |
|--------|----------------|
| `draft_lottery.py` | `run_draft_lottery`, lottery teams/results |
| `draft_sim.py` | Universe sim only (`DraftTrade`, `try_trade_up`) |
| `draft_board.py` | Sim team boards |
| `trade_asset.py` | `DraftPickTradeAsset`, `canonical_pick_id` |
| `draft_ranking_logic.py` | League maps, ETA/rank helpers |
| `draft_prospect_profile.py` | Draft Class cards |
| `draft_class_generator.py` | Class generation |
| `franchise_scouting.py` | Scout coverage / bias |

### API surface

- `GET/POST` `/api/franchise/entry-draft/{state,start,cpu-pick,sim-round,sim-to-user-pick,complete,results}`
- `POST /api/franchise/draft/pick`
- Combine routes + `POST /api/franchise/contracts/sign-elc`

---

## 7. Gaps / fake vs real

| Area | Status |
|------|--------|
| Live 7-round user/CPU picks | **Real** |
| Lottery + combine gate + stock board | **Real** (combine backend; lottery backend) |
| Season pick-trade ownership into draft order | **Real** (at init) |
| Unsigned at pick + reserve + later ELC | **Real**; long-term unsigned retention **weak** |
| Exclusive / non-exclusive rights by league | **Missing** (path labels only) |
| Stay in CHL/NCAA under NHL rights & develop | **Missing** (yanked into `prospect_pool`) |
| Mid-draft trade offers on floor | **Stub UI** |
| Mid-draft Trade Hub → refresh live order | **Gap** |
| Protected/conditional pick *slot* resolution | **Partial** (trade value only) |
| SimEngine `draft_sim` mid-draft trades | Real for **universe**, unused by franchise API |
| DraftLottery.js local simulate | Can **diverge** from backend lottery |

---

## 8. Precision takeaways

1. **Live franchise drafts are owned by** `backend/services/franchise_entry_draft.py`.
2. **Board prospects** are real players under `league.development_leagues` (draft-age), not a separate fantasy registry.
3. **“Rights” after a pick** ≈ org claim (`rights_status="drafted"` + reserve), **not** CBA league rules.
4. **Traded picks in the draft order** are real when ownership moved **before** draft init; **live mid-draft shopping is not**.
5. **Unsigned is the default**; signing is separate; auto-promotion/ELC can collapse long unpaid holds.

---

*Generated from codebase inspection of franchise entry draft, contract economy, pick registry, offseason flow, and Entry Draft / Draft Class UI.*
