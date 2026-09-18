# Draft / Scouting / Prospect OVR — Flaw Audit

**Scope:** How prospects are created, how scouting reveals/obscures Overall (OVR), how "steals/busts" and low/high OVR emerge, and how the OVR you actually get is resolved once you draft the player.

**Verdict:** The core mechanics work, but there are **three independent, disagreeing OVR pipelines** (scouting screen, draft board, post-draft roster) plus a systematic **downward bias** in scouted estimates and **post-scout rating re-rolls**. The number you scout is not the number the board shows, which is not the number you see after you draft. That is the root cause of the "finding the overall once you draft is flawed" complaint.

---

## How it works today (data flow)

1. **Generation** — Prospects are real `Player` entities spawned in development leagues (`league_hierarchy_bootstrap.py`). True OVR is attribute-derived via `compute_ovr()` (`entities/player.py`). True potential is a **separate integer roll** stored in `ratings["dev_potential"]`.
2. **Board build** — `build_draft_class_rankings()` (`franchise_sim.py`) snapshots each player's `player.ovr()` into row `true_ovr` (via `_player_ovr99`, `franchise_sim.py:2910`), then fogs it for public display.
3. **Scouting** — Two separate fog functions add noise/bands to `true_ovr` and reveal the exact value once scouting ≥ `DRAFT_OVR_REVEAL_THRESHOLD = 72` (`franchise_scouting.py:21`).
4. **Draft pick** — `_assign_drafted_prospect()` (`franchise_entry_draft.py:1373`) attaches rights to the **same** player object. **No new OVR is computed or assigned.**
5. **Post-draft display** — Roster/org APIs show `get_effective_ovr_display()` — a **65/35 ratings-blend**, *not* `player.ovr()` and *not* the board's `true_ovr`.

The problem: steps 3, 2, and 5 each compute a different number for the "same" overall.

---

## CRITICAL issues

### C1. Three different OVR numbers for the same player, none reconciled
- **Scouting screen** builds its range around `potential`/fallback, via `_draft_ovr_range` (`franchise_scouting.py:113`, called at `:806`).
- **Draft board** builds its range around `true_ovr = player.ovr()` via the inline block (`franchise_sim.py:8606-8624`) and `compute_public_ovr_band` (`draft_ranking_logic.py:1624`).
- **Post-draft roster** shows `get_effective_ovr_display` = `ratings_blend*0.65 + ovr()*0.35` (`storyline_conduct.py:139`, `:208-214`).

Because the roster display is a 65/35 blend and the board reveal is pure `ovr()`, **even a fully-scouted, "revealed" prospect will change OVR the moment they land on your roster.** A revealed 84 can become an 81 or 87 post-draft with zero development. This is the single biggest driver of the "flawed" feel.
- **Fix:** Pick one canonical current-OVR function and use it everywhere (board reveal, scouting reveal, roster). If the 65/35 blend is the intended "true" display, snapshot `true_ovr` from that same function at board-build time.

### C2. Scouting-screen OVR range is centered on the wrong quantity
In `_normalize_prospect` (`franchise_scouting.py:795-806`):
```python
potential = float(entry.get("potential_score") or entry.get("true_ovr") or 70)
true_ovr  = float(entry.get("true_ovr") or potential or 70)
...
ovr_range = _draft_ovr_range(true_ovr, scouted)
```
But the board serialization **pops `true_ovr` unless the player is revealed** (`franchise_sim.py:8599`, `:8682-8685`). So for any un-revealed prospect, `entry.get("true_ovr")` is absent and `true_ovr` **falls back to `potential_score` (the estimated ceiling)**. The "current OVR range" on the scouting screen is therefore centered on the **ceiling**, not the current OVR — systematically too high, and in the opposite direction of the board's bias (see H1).
- **Fix:** Have the scouting screen consume the board's already-fogged `current_ovr_estimate`/`current_ovr_range` instead of recomputing from a field that isn't there.

### C3. `dev_potential` is a roll independent of current ratings, so POT can contradict OVR/production
Potential is set as a standalone `randint` per pipeline tier (e.g. franchise `88–97`), and residuals are floored at `ovr99 + 2` (`league_hierarchy_bootstrap.py`, `_assign_residual_dev_potential`). Current ratings are re-rolled separately (`build_role_shaped_ratings`). Junior box scores are then simulated *from* OVR/ratings (`prospect_league_scoring.py`), not from potential. Net effect: a prospect can carry an elite POT with mediocre tools/production, or vice-versa, with no guaranteed coherence — this is a common source of "this ranking makes no sense" cases.
- **Fix:** Derive POT as `current + headroom(age, tier, tools)` with bounded noise, or at minimum validate `POT ≥ OVR` and correlate POT tier with tools/production at generation.

---

## HIGH issues

### H1. Scouted OVR estimates are systematically biased LOW
All three fog functions use an **asymmetric spread** — wide below true, narrow above:
- `_draft_ovr_range` (`franchise_scouting.py:114-116`): `low = true - spread`, `high = true + spread*0.65`.
- Board inline (`franchise_sim.py:8610-8612`): `ovr_lo = t - ovr_gap`, `ovr_hi = t + ovr_gap*0.45`.
- `compute_public_ovr_band` (`draft_ranking_logic.py:1645-1646`): `lo = center - span`, `hi = center + span*0.65`.

The **midpoint (what's shown as `current_ovr_estimate`) sits below the true OVR** in every case. Prospects consistently scout *worse* than they are, so post-reveal/post-draft they tend to jump **up** — manufacturing far more "steals" than "busts" from fog alone, independent of any real bust logic.
- **Fix:** Make the band symmetric (`high = true + spread`) unless you deliberately want a pessimism bias, in which case document and tune it.

### H2. Board re-rolls a prospect's ratings *after* you may have scouted them
`ensure_board_prospect_ovr_floors` (`franchise_sim.py:8349-8354`) calls `_apply_shaped_player` → `build_role_shaped_ratings` + `persist_recomputed_ovr`, which **rewrites `player.ratings` in place** to force top board slots to hit OVR floors. If this runs on a board rebuild after the user already scouted (and "revealed") a player at an earlier value, the underlying player silently changes.
- **Fix:** Freeze prospect ratings once the class is generated; apply slot floors *only* at generation, never on rebuild. Or gate floors so they never move an already-revealed prospect.

### H3. Rights-card OVR can display `None` or an un-scaled value
The rights card uses `_safe_player_ovr` (`draft_rights_engine.py:763-778`), which reads `overall` → `ovr` → `current_ovr` → `true_ovr` in order and returns the first float — **without the `<=1.5 → *99` scale guard** that `_player_ovr99` applies. Then `overall: int(ovr) if ovr else None` (`draft_rights_engine.py:506`). If `player.ovr()` returns a 0–1 fraction (e.g. `0.82`), `int(0.82) == 0` → falsy → **`overall: None`**. Just-drafted prospects can show a blank/zero overall on the rights screen.
- **Fix:** Route `_safe_player_ovr` through the same 0–1→0–99 normalization used by `_player_ovr99`.

### H4. "Steal / Reach / Value" label has nothing to do with the hidden true OVR
`_selection_label_from_public` (`franchise_entry_draft.py:910-975`) and the round recap (`:1722`) compute steal/value purely from **public rank vs pick number** (`overall_pick - public_rank`). A genuine hidden gem taken at his correct public rank is labeled "Value/Expected," while a heavily over-fogged average player looks like a "Steal." The label measures market disagreement, not actual outcome — misleading given the game also has real `pipeline_steal`/`pipeline_bust` flags that it *doesn't* use here.
- **Fix:** Either rename the label to "vs. Consensus," or blend in the hidden `pipeline_steal`/`pipeline_bust`/true-OVR delta.

---

## MEDIUM issues

### M1. Two reveal thresholds / three fog implementations to keep in sync
`DRAFT_OVR_REVEAL_THRESHOLD = 72` is referenced in `franchise_scouting.py` (`:21`, `:798`) and re-imported inside `franchise_sim.py` (`:8613`), and each of the three fog functions re-implements the reveal branch separately. Any future tuning must be changed in ≥3 places or they drift.
- **Fix:** Single `public_ovr_band(true_ovr, scout_pct)` helper used by all screens.

### M2. Bust/steal is a one-time spawn coin-flip, not tied to ratings
`pipeline_bust = rng.random() < 0.06`, `pipeline_steal = ... < 0.042` (`league_hierarchy_bootstrap.py:376-387`), rolled at spawn independent of the player's tools or slot. High picks and low picks are equally likely to be flagged bust, and the flag mostly feeds narrative/progression `dev_type` rather than a draft-time OVR consequence.
- **Fix:** Weight bust probability by pick tier / character flags / tools-vs-production gap so busts concentrate where they're dramatically meaningful.

### M3. Magic default OVR of 70 for missing data
`_normalize_prospect` falls back to `70` when both `potential_score` and `true_ovr` are missing (`franchise_scouting.py:795-796`). Any data-plumbing gap silently produces a plausible-looking 70 OVR prospect instead of surfacing the error.
- **Fix:** Fall back to an explicit "unknown/unscouted" state, not a hardcoded rating.

### M4. Potential-score display floor for round 1 diverges from the player's real `dev_potential`
The board lifts first-round `potential_score` for UI/ranking (`franchise_sim.py:~8389-8404`) separately from the stored `dev_potential` in ratings. The ceiling you see can exceed the ceiling the player can actually reach.
- **Fix:** Clamp displayed ceiling to the true `dev_potential` (fogged), don't invent a higher floor.

---

## LOW / MINOR issues

- **L1. Dead generation path.** `generation/draft_class_generator.py` (0–1 attrs + `potential_ceiling_0_1`, its own bust/steal at `:288-293`) is **not** the live franchise path but still exists and looks authoritative — a trap for future edits. Consider deleting or clearly marking it non-live.
- **L2. Non-deterministic fallbacks.** Several repair paths fall back to `random.Random(42)` when no rng is passed (`franchise_sim.py:8353`, and `draft_ranking_logic.py:1219,1279`), so results differ depending on whether the caller threaded the seed. Thread the franchise seed through consistently.
- **L3. Global-pool prospect seeding uses `hash(...)`** (`engine.py:~6495`), which isn't stable across processes (PYTHONHASHSEED) — breaks reproducibility for that path.
- **L4. `_draft_scout_completion` jitter uses `abs(hash(str(key)))`** (`franchise_scouting.py:108`) — same cross-process instability for scouting completion %.
- **L5. Duplicate confidence/estimate keys.** The board entry emits many near-synonyms (`current_ovr_estimate`, `public_ovr_low/high`, `current_ovr_range`, `scout_disagreement_narrowing`) (`franchise_sim.py:8662-8668`); easy for the frontend to bind to the wrong one and show a different number than the reveal.
- **L6. `risk` label is a mixed signal** — derived from `is_bust_risk`/`character_concerns` OR a confidence threshold (`franchise_sim.py:8646-8648`), so "High risk" can mean "we haven't scouted him" rather than "he's actually risky."

---

## Summary table

| ID | Severity | Issue | Primary location |
|----|----------|-------|------------------|
| C1 | Critical | Three unreconciled OVR numbers (scout vs board vs roster) | `storyline_conduct.py:139,208`; `franchise_sim.py:8606,6774` |
| C2 | Critical | Scouting range centered on ceiling because `true_ovr` is stripped | `franchise_scouting.py:795-806`; `franchise_sim.py:8599,8682` |
| C3 | Critical | `dev_potential` rolled independent of ratings/production | `league_hierarchy_bootstrap.py` (`_assign_residual_dev_potential`) |
| H1 | High | Scouted OVR estimate biased systematically low (asymmetric span) | `franchise_scouting.py:114`; `franchise_sim.py:8610`; `draft_ranking_logic.py:1645` |
| H2 | High | Ratings re-rolled after scouting via board floors | `franchise_sim.py:8349`; `league_hierarchy_bootstrap.py` (`_apply_shaped_player`) |
| H3 | High | Rights-card OVR can be `None`/un-scaled | `draft_rights_engine.py:763,506` |
| H4 | High | Steal/Reach label ignores true OVR (public-rank only) | `franchise_entry_draft.py:910,1722` |
| M1 | Medium | Reveal threshold + fog duplicated 3× | `franchise_scouting.py:21`; `franchise_sim.py:8613`; `draft_ranking_logic.py:1624` |
| M2 | Medium | Bust/steal is spawn coin-flip, not tools-linked | `league_hierarchy_bootstrap.py:376-387` |
| M3 | Medium | Magic default OVR = 70 on missing data | `franchise_scouting.py:795` |
| M4 | Medium | R1 ceiling display floor > true `dev_potential` | `franchise_sim.py:~8389` |
| L1 | Low | Dead `draft_class_generator.py` path | `generation/draft_class_generator.py` |
| L2 | Low | `random.Random(42)` fallbacks break reproducibility | `franchise_sim.py:8353`; `draft_ranking_logic.py:1219,1279` |
| L3 | Low | `hash()`-based prospect seed unstable | `engine.py:~6495` |
| L4 | Low | `hash()`-based scout completion jitter | `franchise_scouting.py:108` |
| L5 | Minor | Duplicate/synonym OVR keys confuse frontend | `franchise_sim.py:8662-8668` |
| L6 | Minor | `risk` conflates "unscouted" with "risky" | `franchise_sim.py:8646` |

---

## Recommended fix order

1. **C1 + H3** — Unify on one current-OVR function; make board `true_ovr`, scouting reveal, and roster display all agree. This alone fixes the "OVR changes when I draft him" bug.
2. **C2 + H1** — Feed the scouting screen the board's fogged estimate; make the band symmetric. Fixes the "estimates are always off, always low" feel.
3. **H2** — Freeze ratings after generation so scouting a player is meaningful and stable.
4. **C3 + M2 + M4** — Make potential/bust/ceiling coherent with tools and production.
5. Everything else (M1, M3, L1–L6) as cleanup.
