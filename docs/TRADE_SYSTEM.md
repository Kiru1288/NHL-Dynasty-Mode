# Trade System Architecture

Canonical map of trade flows in NHL Franchise Mode. All user-facing and CPU trades route through the shared SimEngine `trades/*` stack unless noted.

---

## Trade pipelines

| ID | Name | Trigger | Assets | Validation / acceptance |
|----|------|---------|--------|-------------------------|
| **A** | TradeHub (user) | `POST /api/franchise/trade` | NHL players, picks, retention | `validate_trade_rules` + AI interest |
| **B** | Ambient CPU | `propose_and_execute_cpu_trades()` | Players + picks when realistic | Full stack; `cpu_ambient_trade: True` (stricter fairness + min interest) |
| **C** | Cap-casualty CPU | `run_cpu_cap_casualty_trade_pass()` | Players + picks | Full stack; `cap_casualty_trade: True` (AI interest bypass; hard validation remains) |

**Shared modules:** `trade_evaluator.evaluate_trade_package`, `trade_executor.execute_validated_trade`, `trade_rules.validate_trade_rules`, `trade_pick_registry`.

**Authoritative path for user trades:** Pipeline **A** via `backend/services/trade_service.py`.

---

## Request / response contract (Pipeline A)

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/franchise/trade/evaluate` | Dry-run validation + AI interest |
| `POST` | `/api/franchise/trade` | Evaluate + execute atomically |
| `GET` | `/api/franchise/trade/assets` | Trade pools (NHL, AHL display-only, picks) |
| `GET` | `/api/franchise/trade/market` | Partner summaries (window, needs, cap) |
| `GET` | `/api/franchise/trade/history` | `league.trade_history` slice |

### Bridge

`backend/services/trade_service.py` → SimEngine:

- `evaluate_franchise_trade()` → `trade_evaluator.evaluate_trade_package()`
- `execute_franchise_trade()` → `trade_executor.execute_validated_trade()`
- `_trade_context()` supplies `season_year`, `deadline_phase`, `calendar_cursor`, `team_by_id`, `sim`, `league`

### Payload shape

Keys in `assets_by_team` are **acquiring team IDs**. Each asset's `team` field is the **source** (current owner).

```json
{
  "assets_by_team": {
    "<acquiring_team_id>": [
      { "type": "player", "id": "<player_id>", "team": "<source_team_id>", "retained": 0 },
      { "type": "pick", "id": "<canonical_pick_id>", "team": "<current_owner_team_id>" }
    ]
  }
}
```

**TradeHub mapping:** user outgoing → `assets_by_team[partnerTeamId]`; partner outgoing → `assets_by_team[userTeamId]`.

### Evaluate response (key fields)

| Field | Meaning |
|-------|---------|
| `can_execute` | Passes `validate_trade_rules` |
| `accepted` | Non-user partners meet AI interest (cap-casualty exempt) |
| `contract_slot_impact` | Per-team before/after/limit/ok + incoming/outgoing contract counts |
| `cap_impact` / `roster_impact` / `clause_impact` | Projected post-trade state |
| `interest_level` | Per-team AI interest 0–1 |
| `fairness_gap` | Max net value spread between teams |

Execution requires `can_execute` **and** `accepted`.

---

## Module map

| File | Role |
|------|------|
| `SimEngine/.../trades/trade_asset.py` | Normalize payload; NHL vs AHL roster lookup |
| `SimEngine/.../trades/trade_rules.py` | Legality: ownership, clauses, cooldown, cap, roster, retention, contract slots |
| `SimEngine/.../trades/trade_evaluator.py` | Value + AI acceptance + verdict |
| `SimEngine/.../trades/trade_value.py` | Player/pick contextual valuation |
| `SimEngine/.../trades/trade_executor.py` | Atomic roster/pick/retention mutations; acquisition stamps |
| `SimEngine/.../trades/cpu_trade_proposer.py` | Ambient CPU package builder + execution |
| `SimEngine/.../trades/trade_pick_registry.py` | Canonical pick ownership |
| `SimEngine/.../economy/cap_engine.py` | Cap snapshots; retention cleanup; contract slots |
| `backend/services/trade_service.py` | API bridge + post-trade pick audit |
| `backend/services/contract_economy.py` | Cap-casualty builder/executor |
| `frontend/src/screens/TradeHub.js` | UX; mirrors backend tradeability (backend still authoritative) |

---

## Legality rules

Enforced in `validate_trade_rules()`:

| Rule | Enforcement |
|------|-------------|
| Player on source NHL `roster` | Required; AHL blocked |
| NMC / full NTC | Hard block |
| M-NTC | Hard block unless destination in `approved_trade_teams` / `approved_trade_team_ids` / `approved_destinations` / `no_trade_list` |
| Recently acquired (7-day default) | Hard block via `last_acquired_day` / `acquired_via_trade` |
| Retained 0–50%, max 3 active slots | Hard block |
| Pick registry ownership, not resolved | Hard block |
| Usable cap after trade | Hard block (partial relief for cap-casualty only) |
| Active roster ≤ 23 | Hard block |
| Contract slots ≤ 50 | Hard block |

---

## CPU trade intelligence

### Ambient (Pipeline B)

- Uses team needs to pair buyers/sellers (positions of need vs surplus).
- Deadline rentals: contenders target pending UFAs; rebuilders shop rentals when playoff odds are low.
- Packages may include mid/late picks; never resolved picks; bottom-five own 1sts protected when value is high.
- Acceptance: `fairness_gap ≤ 7`, partner `interest ≥ 0.50`, no AI bypass.

### Cap-casualty (Pipeline C)

- Triggered when teams fail cap compliance after offseason processing.
- May bypass AI interest so over-cap teams can shed salary; still requires cap/roster/slots/clauses/ownership.
- Protected: NMC, core stars, true ELCs.

---

## Draft pick ownership

- **Source of truth:** `league.draft_pick_registry`
- **Denormalized:** `team.owned_pick_ids`
- **Transfer:** `transfer_pick()`; post-trade `audit_pick_registry_integrity()`

---

## Retained salary lifecycle

- Created on execute when `retained > 0`; stored on retaining team `retained_salary_records`.
- `seasons_remaining` decremented at offseason salary-cap advance; expired rows removed via `cleanup_league_retained_salary_records()`.
- Cap snapshot ignores rows with `seasons_remaining <= 0`.

---

## Known limitations

- M-NTC approved lists depend on contract data quality; missing lists hard-block trades.
- CPU ambient volume is capped per sim tick; not every realistic fit is attempted.
- TradeHub clause blocking is UX-only; API always re-validates.
- Pick lottery modeling is heuristic (standings rank, points %, team window) — not a full lottery simulator.

---

## Tests

| File | Coverage |
|------|----------|
| `backend/tests/test_trade_foundation.py` | Clauses, cooldown, slots, picks, ambient CPU, execute integration |
| `backend/tests/test_contract_economy.py` | Cap-casualty pipeline, retention cleanup |
| `backend/tests/audit_draft_pick_safety.py` | Pick registry integrity script |

---

## Deprecated APIs

| Symbol | Replacement |
|--------|-------------|
| `franchise_sim.execute_trade_package()` | `trade_service.execute_franchise_trade()` |

Legacy executor moved players only with no picks, retention, cap, or AI checks.
