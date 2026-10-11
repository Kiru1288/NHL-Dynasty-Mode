# Scouting — Bug Report

Scope: `backend/services/franchise_scouting.py`, the scouting routes in `backend/main.py`,
the bulk-sim hooks in `backend/services/franchise_sim.py`, and `frontend/src/screens/Scouting.js`.
Every item was found by reading the code; line numbers are from the audit commit.
Status column: **Fixed** = addressed in the scouting overhaul commit.

## Severity key
- **Critical** — breaks the system or lets the player get scouting for free
- **High** — wrong results the player will notice
- **Medium** — misleading UI / numbers
- **Low** — polish / consistency

---

## A. Progress, files and seasons

| # | Sev | Bug | Evidence | Status |
|---|-----|-----|----------|--------|
| S1 | Critical | **Rest-of-season sims double scouting progress.** Passive growth is applied every 3 days by the incremental catch-up, then applied again for the whole span by the season-extras pass. | `franchise_sim.py` `_run_bulk_incremental_catchup` (~13358) and `_run_bulk_finalize_season_extras` (~13394) both call `apply_passive_scouting_progress`; `_run_bulk_finalize_catchup` passes `deferred + steps_n` days to the second one. | Fixed |
| S2 | Critical | **Scouting budget never resets.** `used_budget` only ever grows; after the first draft cycle the budget is spent forever and every action fails with "budget exceeded". | No code anywhere resets `scouting_state["used_budget"]`. | Fixed |
| S3 | High | **Coverage deployments never end.** Region/country deployments stay active forever and charge $48K/week each indefinitely — even after the budget is gone (used can exceed budget). | `_register_coverage_deployment` (no expiry); `_charge_deployment_salary` has no budget check. | Fixed |
| S4 | High | **Last year's class never leaves the scouting file.** Overlays for drafted players keep their assigned scout (eating assignment slots), their Spotlight (5 slots, so Spotlight can fill up permanently) and their watchlist entries. | `state["prospects"]`, `scout_spotlight`, `watchlist` are never pruned; `_count_assigned_prospects` counts every overlay ever made. | Fixed |
| S5 | High | **Assigned scouts are too slow to matter.** 0.42%/day means ~170 days to the 72% OVR reveal and ~240 days to a full file — longer than the scouting season, so most files never finish. | `PASSIVE_ASSIGNED_DAILY = 0.42` | Fixed (rebalanced) |
| S6 | High | **Scout quality and specialty do nothing.** Every assigned scout grows a file at the same rate; the per-scout "days to 72%" and Speed bar are fiction. | `_scout_pool` computes `speed_days_to_72`; `apply_passive_scouting_progress` never reads the scout. | Fixed |
| S7 | Medium | Re-assigning a prospect to a different scout restarts the 14-day full-file clock. | `_apply_scouting_meta_patch` always overwrites `assigned_at_cursor`. | Fixed |
| S8 | Medium | Timeline says a full file is "never" reachable for an assigned prospect still inside the 14-day unlock window. | `_scout_file_timeline` uses the 78% cap and ignores the pending unlock. | Fixed |
| S9 | Medium | Hidden-ceiling sleepers and public-miss markers are keyed to the season year, so the new class has last class's markers between the draft and season rollover. | `ensure_scouting_markers` (`markers_season == sy`). | Fixed |
| S10 | Medium | `weeks_until_draft` is 0 all of July (right after the draft), so the UI says the draft is this week all summer. | `_weeks_until_draft` (`month in (6, 7)`). | Fixed |
| S11 | Low | Phase labels disagree with the offseason stages: April is "Combine week" but the combine stage is May and the draft is June. | `_draft_period_label`, `_scouting_phase`. | Fixed |

## B. Actions and money

| # | Sev | Bug | Evidence | Status |
|---|-----|-----|----------|--------|
| S12 | Critical | **Over-budget actions still scout for free.** Coverage is written to the prospect before the budget check; the request then fails but the gain stays. | `apply_scouting_command`: `_apply_to_targets` runs before the `used + cost > budget` check. | Fixed |
| S13 | Critical | **The client sets the price.** The backend charges whatever `estimated_cost` the frontend sends (0 works). The frontend formula also differs from the backend one. | `_estimate_cost` returns `body["estimated_cost"]` when present. | Fixed |
| S14 | High | **Actions that hit nobody are still charged** ("applied to 0 prospect(s)"). | `apply_scouting_command` charges `cost` even when `affected` is empty. | Fixed |
| S15 | High | **Region deployments never match.** Region targets compare `slug(region)` to the raw id. | `_apply_to_targets` `target_type == "region"` branch. | Fixed |
| S16 | High | **Deep dives are infinitely repeatable.** Interview, dinner, combine, workout and medical can be bought again and again for more coverage. | No "already done" check in `_apply_to_targets`. | Fixed |
| S17 | Medium | The UI's "Analytics" deep dive maps to no backend action and silently runs Player Focus instead. | `SCOUTING_ACTIONS.ANALYTICS = "analytics"` vs backend `analytics_deep_dive`; `_resolve_action` falls back. | Fixed |
| S18 | Low | Medical review never reveals anything — it only adds "cleared pending final review" once coverage passes 75%. | `_apply_to_targets` `medical_review` branch. | Fixed |

## C. State, staff and assignments payloads

| # | Sev | Bug | Evidence | Status |
|---|-----|-----|----------|--------|
| S19 | High | **"Assignments" are a receipt log, not assignments.** Every action is stored as `completed`; nothing is ever `active`, so Cancel does nothing and "active assignments" is always 0. | `_upsert_assignment(..., status="completed")`; `get_scouting_assignments`. | Fixed |
| S20 | Medium | Staff workload is total ÷ number of scouts (identical for everyone, and always 0 because of S19). | `get_scouting_state`. | Fixed |
| S21 | Low | `date` and `draft_year` in the state payload are always empty (read keys the context never sets). | `get_scouting_state`. | Fixed |

## D. Frontend (`Scouting.js`)

| # | Sev | Bug | Evidence | Status |
|---|-----|-----|----------|--------|
| F1 | Critical | **You can't assign a scout from the Scouting screen.** The big "Assign Scout" button actually buys a one-off action; real scout assignment (what drives passive growth and the 100% file) only exists on the Draft Class screen. | No `assigned_scout` anywhere in `Scouting.js`; `CommandPanel` → `assignScouting` → `/assign` with an action. | Fixed |
| F2 | High | **Country deployments turn into single-player actions.** Selecting a country auto-selects its top prospect, and `assignScouting` prefers the prospect, so "Deploy Coverage" never deploys to the country. | `selectCountry` sets `selectedProspectId`; `assignScouting` picks `player` first. | Fixed |
| F3 | Medium | Deep-dive buttons stay enabled after the interview/dinner/combine/medical is done and show no status. | `PlayerView` `DraftActionButton`s. | Fixed |
| F4 | Medium | The Budget tile shows the total budget, not what's left. | `dashboardStats.budget`. | Fixed |
| F5 | Medium | The cost shown is a client-side estimate unrelated to what the backend charges (see S13). | `estimateAssignmentCost`. | Fixed |
| F6 | Low | Every note renders as author "Scout", date "—" (backend notes are plain strings). | `ScoutNote`. | Fixed |
| F7 | Medium | No guided path: 7 tabs, a command rail, 12 actions × 4 intensities, and the core loop (assign scout → file grows → 72% reveals OVR → 100% full file) isn't explained anywhere. | `VIEW_TABS`, `CommandPanel`. | Fixed (new Desk tab, auto-scouting) |
| F8 | Low | Visual style doesn't match the Entry Draft floor (different palette, fonts, sidebar shell, buttons). | `ScoutingStyles` | Fixed (restyled) |

---

## What changed to make scouting easier

- **Scouting Desk** (new default tab): three-step guide, your projected pick, your draft range with one-click Assign / Watch / Open, and the files your staff is working.
- **Auto-assign scouts**: one click fills every open scout slot with the best targets (your targets and watchlist first, then your draft range, then the top of the board), matching each prospect to the scout who covers that region.
- **Let staff manage assignments**: optional — every week scouts move off finished files and onto the next priority prospect.
- **Faster, meaningful scouts**: assigned files reach the 72% OVR reveal in about 6–8 weeks (faster with elite or region-specialist scouts) and can finish before the draft.
- **Prices come from the server** and are shown before you spend.
