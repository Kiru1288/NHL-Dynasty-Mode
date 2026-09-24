# Draft-floor trade down — backend report

Last updated: 2026-03-23

## What exists today

| Layer | Location | Role |
|-------|----------|------|
| Offer generation | `backend/services/draft_day_trade_offers.py` | Builds climb packages when a later team has a real board priority still available |
| Sweetener pricing | `franchise_entry_draft._draft_swap_sweeteners` | Maps chart gap → owned future picks (no free 2nds on tiny slides) |
| Accept / execute | `franchise_entry_draft.accept_draft_day_trade_offer` | Validates via `evaluate_trade_package`, executes trade, syncs draft clock, stores true target for CPU |
| Payload refresh | `get_entry_draft_payload` | Regenerates offers each state pull (up to 5 when user on clock, 3 otherwise) |
| API | `POST /api/franchise/entry-draft/accept-trade` | Accepts offer dict from client |
| Tests | `backend/tests/test_franchise_issue_audit.py::test_user_on_clock_gets_trade_down_offers` | Smoke on fogged targets + incoming assets |

### Offer fields (after this pass)

- Identity: `offer_id`, `from_team_id`, `team_name`, `partner_overall_pick`, `slots_moved`
- Package: `incoming_assets`, `outgoing_assets`, `sweetener_pick_ids`, `slot_value_gap`
- Fog: `target_candidates`, `true_target_prospect_id` (server only), `candidate_intel[]` (rank band, need fit, scout buzz)
- Narrative: `intel_lines[]`, `pitch_headline`, `reason_headline`, `urgency`, `urgency_label`, `offer_style`
- Value UI: `value_meter` (`user_send_value`, `user_receive_value`, `fairness_gap`, `can_execute`) when league + user on clock

## What was added in this pass

1. **Richer offers** — per-candidate intel, dynamic intel lines, offer styles (`standard` / `desperate` / `pay_up`), stable `offer_id`.
2. **Trade value preview** — `evaluate_trade_package` snapshot for the user’s side before accept (powers FE value bars).
3. **More partners in the window** — lookahead 14 slots; still capped by `max_offers` in `get_entry_draft_payload`.
4. **Empty desk guard** — when user is on clock, at least one climber can be forced through willingness RNG so Trade Down is not blank.

## Gaps / recommended next work

### High priority

1. **Dedicated refresh endpoint** — `GET /api/franchise/entry-draft/trade-offers` to regenerate without full `/state` heavy payload; include `generated_at` pick number for stale-offer detection.
2. **Offer staleness** — reject accept when `overall_pick` or `partner_overall_pick` no longer matches live order (partially checked via partner slot owner).
3. **Counter-offers** — user proposes “move to #X + sweetener Y”; CPU accepts/rejects via same evaluator (no generator-only packages).
4. **Trade-up desk** — symmetric offers when user wants to climb (mirror of trade down using user as climber).

### Medium priority

5. **Scouting reveal minigame** — spend scout capital to narrow `candidate_intel` (eliminate one decoy) without revealing `true_target_prospect_id` to client.
6. **Philosophy / ideology hooks** — surface `cpu_franchise_profiles` aggression in `intel_lines` and filter partners (already used in `_partner_willing_to_climb`; not yet in payload text).
7. **Social / floor buzz linkage** — on accept, push `buildDraftPickReactionTweet` + trade-down fan templates with `{movingTeam}` / `{tradedAssets}` from offer.
8. **Draft trade log UI** — expose `draft_trade_log` on recap screen with true target revealed post-pick.

### Lower priority / engine

9. **Multi-asset packages** — occasional prospect + pick overpay (requires draft-day prospect trade rules audit).
10. **Live offer decay** — urgency timer that removes offers after N seconds of clock (needs draft clock API).
11. **SimEngine parity** — `SimEngine/app/sim_engine/draft/draft_board.py` `consider_trade_down` hints not wired to franchise entry draft.

## Frontend alignment

- Trade Down modal uses team logos, expandable cards, value bar from `value_meter`, and three-column rumoured targets.
- If `value_meter` is empty (no league registry in test/dev), UI still shows chart gap chips and package lists.

## How to verify

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest tests/test_franchise_issue_audit.py::test_user_on_clock_gets_trade_down_offers -q
```

Manual: advance franchise to Entry Draft with user on clock → **Trade down** → confirm multiple clubs, logos, value bar, expandable intel.
