# Trade Value, CPU Pairing & Contract-Slot Audit

**Status:** living gate (implementation landed; this doc tracks proven truths + remaining open design)  
**Date:** 2026-07-25 (updated after critique pass)  
**Rule:** cite **function names + constants + short excerpts**, not approximate line ranges. Line numbers drift; callers matter.

Legend:

| Tag | Meaning |
|-----|---------|
| **CONFIRMED** | Proven in source / tests |
| **LIKELY** | Strongly implied; not season-measured |
| **OPEN** | Must be decided or measured before treating as done |

---

## 0. Label correction

There is **not** one league-wide “authoritative” trade-value system.

| Path | Entry | Role |
|------|-------|------|
| **Primary NHL player-and-pick package valuation** | `evaluate_player_asset_value` → `evaluate_package_value` (`trade_value.py`) | Trade Hub, legality packages, ambient `_player_trade_value` |
| Draft-day | `pick_value` / draft trade fuzz (`draft_sim.py`) | Separate stack — intentional |
| Cap-casualty partner scoring | `contract_economy` dump partner helpers | Separate partner selection — intentional vs debt: **OPEN** |
| Ambient candidate shopping | `cpu_trade_proposer._pick_trade_candidates` | Uses primary evaluator with acquiring perspective when set |

Implementation stance: keep separate stacks unless a product decision unifies draft-day into `trade_value.py`.

---

## 1. Primary package valuation — exact attach points

### 1.1 Functions (not line guesses)

| Concern | Function | Callers |
|---------|----------|---------|
| Player TV | `evaluate_player_asset_value` → `_evaluate_player_asset_value_impl` | `evaluate_asset_value`, `cpu_trade_proposer._player_trade_value`, Trade Hub / `trade_service` paths |
| Talent curve | `_talent_base(ovr)` | Inside impl |
| Needs | `TeamNeeds.evaluate(acquiring_team, …)` | Inside impl (`acquiring_team` arg) |
| Package net | `evaluate_package_value` | `trade_evaluator.evaluate_trade_package` |
| Safe failure | `reduced_trade_value_fallback` | Wrapper on `evaluate_player_asset_value`; `_player_trade_value` |

Constants:

- `TRADE_VALUE_FORMULA_VERSION`
- `TRADE_VALUE_FALLBACK_SCALE = 0.55`
- `TRADE_VALUE_FALLBACK_FLOOR = 3.0`
- `TRADE_VALUE_FALLBACK_CEIL = 45.0`

### 1.2 High-risk OVR fallback — **CONFIRMED fixed**

**Defect (pre-fix):** `_player_trade_value` / enrichment failures returned raw OVR (~70–90), which collides with TV scale 0–100 and can price depth like stars.

**Required behavior now:**

```text
except Exception:
  log exception (logger.exception / warning)
  return reduced_trade_value_fallback(...)  # scaled _talent_base, capped ≤ 45
```

Tests: `test_reduced_trade_value_fallback_never_returns_raw_ovr`, `test_cpu_trade_value_fallback_does_not_use_raw_ovr`.

**Exception swallowing inventory (valuation path):**

| Site | Catches | Logs? | Fallback |
|------|---------|-------|----------|
| `evaluate_player_asset_value` | `Exception` | yes (`logger.exception`) | reduced fallback dict |
| `_player_trade_value` | `Exception` | yes | `reduced_trade_value_fallback` |
| `_safe_float` / field readers | broad | no | numeric default (not TV) |

Season exception rate: **OPEN** (needs telemetry counter).

### 1.3 Owner vs acquiring — **CONFIRMED**

- Needs / window modifiers use **`acquiring_team`**.
- Ambient matching should pass acquiring team into `_player_trade_value(..., acquiring_team=…)`.

---

## 2. CPU pairing — diversity controls

### 2.1 Exact constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `CPU_PAIR_COOLDOWN_DAYS` | 18 | Hard skip recent pair |
| `CPU_REACQUIRE_SOFT_DAYS` | 35 | Soft reacquire friction |
| `CPU_REVERSE_TRADE_PENALTY` | 0.22 | Soft, not hard ban |
| `CPU_SEASON_PAIR_SOFT_CAP` | 2 | Hard skip after N season pair trades |
| `CPU_PEER_ATTEMPT_MODULO` | 5 | Peer path every 5th attempt (was 3) |
| `CPU_AMBIENT_FAIRNESS_GAP_MAX` | 7.0 | Quality gate |
| `CPU_DIVERSITY_TARGETS` | dict | Measurable acceptance targets |

### 2.2 Measurable targets (`CPU_DIVERSITY_TARGETS`)

- `max_pair_repetitions_per_season`: 2  
- `min_median_unique_partners_per_trading_team`: 3.0  
- `max_pct_trades_reuse_prior_pair`: 0.35  
- `max_reverse_trade_rate`: 0.12  
- `max_fairness_gap_mean`: 7.0  

Season telemetry harness: **OPEN** (constants exist; live measurement not yet wired).

### 2.3 Mid-pass pool staleness — **CONFIRMED addressed**

After each successful execute, `propose_and_execute_cpu_trades` calls `_reclassify_pools()` so seller/buyer/peer membership refreshes inside the same proposer call.

### 2.4 Peer-swap design — **OPEN product review**

Peer attempts still exist (`i % CPU_PEER_ATTEMPT_MODULO`). Measure peer-swap rate, average OVR/TV, need improvement, same-position swaps before deciding to remove.

### 2.5 Quality regression — **OPEN**

Diversity gains must not worsen fairness gap, legality, need fit, slot/cap legality, star rarity, prospect protection, or volume. Track alongside `CPU_DIVERSITY_TARGETS`.

---

## 3. Contract slots (NHL SPCs)

### 3.1 Predicate — prefer explicit category

Functions:

- `normalize_contract_dict` / `_normalize_contract_type_token`
- `does_contract_use_contract_slot`
- `uses_nhl_contract_slot`
- `iter_org_contract_players` / `_org_player_dedupe_key`
- `_count_team_contract_slots`
- `cap_engine._player_has_active_contract` / `count_team_contract_slots`
- `trade_asset.player_holds_nhl_spc`

Canonical flags written on normalize:

- `is_nhl_spc` / `nhl_spc` / `standard_player_contract`

Type normalization covers aliases (`spc`, `nhl`, `ahl only`, lowercase, spaces/underscores). Salary is **not** the sole gate for explicit SPCs.

### 3.2 Retained salary vs SPC — **CONFIRMED**

- **SPC / 50-slot** follows the **acquiring** club with the player.  
- **Retained salary** is a **cap charge + retention slot (max 3)** on the **retaining** club (`retained_salary_records`).  
- Retention records are **not** SPCs and are not scanned by slot counters.  
- Test: `test_retained_salary_does_not_create_spc_on_retaining_team`.

### 3.3 ID dedupe — **CONFIRMED**

- Stable `id:` key when present.  
- Missing/blank IDs use `obj:{id(player)}` — never collapse many `None`s into one.  
- Duplicate same ID across AHL+ECHL counts once.

### 3.4 `prospect_pool` semantics — **PARTIAL**

Iterator includes `prospect_pool` as an ownership/assignment container. Slot counting still requires `uses_nhl_contract_slot`. Rights-only / unsigned rows should not count. Full product taxonomy (overseas, duplicate refs): **OPEN** documentation in franchise bootstrap.

### 3.5 Expiry date boundary — **OPEN canonical calendar**

`is_contract_active(..., season=)` uses `expiry_year` when provided. Still need one franchise answer for:

- after last regular-season game  
- during playoffs  
- after June expiry processing  
- re-sign stage  
- before next-season generation  

Until then, prefer `years_remaining` decremented only by the official offseason expiry processor.

### 3.6 UI / API provenance — **CONFIRMED sources**

| Surface | Reads | Local calc? |
|---------|-------|-------------|
| `RosterScreen` summary “NHL SPCs” | `snap.nhl_spcs_used` → `contract_slots_used` → `franchiseState.contract_slots` | Fallback local org signed count only if backend missing |
| `CapLedger` Cap tab / compact | `buildRosterSlotSummary` → `snap.nhl_spcs_used` / `contract_slots_used` | Roster headcounts for NHL/AHL/ECHL **assignment** rows only |
| Offseason FA / re-sign menus | `payload.contract_slots` / `contract_slots_used` from office/market | No invented eligibility |
| Backend snapshot | `get_team_cap_snapshot_full` emits **both** `contract_slots_used` and alias `nhl_spcs_used` | Preserves API shape |

API policy: **do not rename** `contract_slots_used`; add aliases only.

### 3.7 Save migration — **OPEN**

Changing org-wide SPC counting can jump displayed totals on old saves. Needed: schema version detect, type normalize on load, duplicate cleanup, missing-ID policy, migration warning. `CONTRACT_SCHEMA_VERSION = 2` exists on contracts; franchise-level migration report still **OPEN**.

### 3.8 Performance — **OPEN measure**

Org scan = `roster ∪ ahl ∪ echl ∪ prospect_pool` in `_count_team_contract_slots` / `iter_org_contract_players`. Cache+invalidate vs always-scan: measure under Contract Office + CPU FA + trade projection for 32 teams.

---

## 4. Assignment / atomic moves

### 4.1 Destination after minor-league trade — **CONFIRMED rule**

`trade_executor._resolve_trade_destination_attr`:

1. Source NHL → dest `roster`  
2. Source AHL/ECHL/prospect **with NHL SPC** → dest `ahl_roster`  
3. Else → `roster`  

Flags synced via `_sync_assignment_flags` (`in_minors`, `roster_location`).

### 4.2 Rollback — **CONFIRMED improved**

`execute_validated_trade` snapshots **all** org lists + retained records + pick registry; `_rollback_all` restores on failure.

Call-up: `_call_up_best_ahl_spc` validates planned lists, then commits; on exception restores prior NHL/AHL lists.

### 4.3 Canonical assignment field — **OPEN**

Lists are authoritative containers; flags must match the list. Contradictory `roster_location` / `in_minors` vs list membership still possible on legacy data — needs one heal pass.

### 4.4 Waivers — **OPEN / blocking for send-downs**

Before enabling general send-downs: eligibility, claim sim, pending state, no dual active roster. Call-up auto-demote currently bypasses waivers (known gap).

---

## 5. Test matrix (landed + still missing)

**Landed**

- Reduced TV fallback ≠ raw OVR  
- CPU TV fallback ≠ raw OVR  
- Type normalize (`ahl only`, `spc`) + explicit `is_nhl_spc`  
- Missing IDs count separately  
- Duplicate ID AHL+ECHL once  
- Retained salary ≠ SPC on retaining team  
- Affiliate trade destination rule  

**Still missing / desired**

- Trade to exactly 50 / reject 51  
- One-for-two contract trade projection  
- Failed trade rollback integration (AHL lists)  
- Failed call-up rollback integration  
- Contract expiry stage boundary  
- Save/load SPC’d AHL player  
- Frontend refresh after org movement  
- Season diversity metrics vs `CPU_DIVERSITY_TARGETS`  
- Exception-rate telemetry for valuation fallback  

---

## 6. Implementation blockers cleared vs still open

| Blocker from critique | Status |
|-----------------------|--------|
| Exact attach points (functions/constants) | Cleared in this doc |
| “Authoritative” overstatement | Relabeled primary path |
| OVR fallback high risk | Fixed + tests |
| Exception swallowing inventory | Documented for TV path |
| UI/API slot sources | Proven in §3.6 |
| Money-only SPC predicate | Explicit flags + normalize |
| Type variant normalize | Landed |
| Retained salary slot ownership | Documented + tested |
| Atomic org/trade rollback | Improved |
| Trade destination rules | Deterministic function |
| CPU season pair + mid-pass reclassify | Landed |
| Measurable diversity targets | Constants; harness OPEN |
| Save migration | OPEN |
| Perf of org scans | OPEN |
| Waivers / canonical assignment | OPEN |
| Expiry calendar boundary | OPEN |

---

## 7. Short excerpts (supporting)

Fallback ceiling:

```python
TRADE_VALUE_FALLBACK_SCALE = 0.55
TRADE_VALUE_FALLBACK_FLOOR = 3.0
TRADE_VALUE_FALLBACK_CEIL = 45.0
```

SPC type normalize entry:

```python
def _normalize_contract_type_token(raw: Any) -> str:
    s = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    ...
```

Trade destination:

```python
def _resolve_trade_destination_attr(loc: str, player: Any) -> str:
    if loc == "nhl":
        return "roster"
    if loc in ("ahl", "echl", "prospect") and player_holds_nhl_spc(player):
        return "ahl_roster"
    return "roster"
```
