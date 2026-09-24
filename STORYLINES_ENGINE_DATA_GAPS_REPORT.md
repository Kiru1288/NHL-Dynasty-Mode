# Storylines Engine: Missing & Insufficient Data Report

What data the storylines engine never has, invents, or is too thin on, and what that does to the stories and popups the player sees.

**Code reviewed:** `SimEngine/app/sim_engine/franchise/` (`storyline_engine.py` 11,306 lines, `storyline_coverage.py`, `storyline_procedural.py`, `storyline_copy.py`, `social_copy_engine.py`, `social_templates.py`, `storyline_stat_bridge.py`) plus the popup builders in `backend/services/franchise_sim.py`.

## How much to trust each finding

| Tag | Meaning |
|---|---|
| **[V]** | Verified by running code or a live session |
| **[C]** | Read from code; not executed |

**Limits of this audit, stated plainly:**
- The live audit I ran was stuck before the season started (a pending decision blocked day advance; calendar cursor at 2, `player_season_stats` empty). It examined **95 stories generated before any game was played**. That covers the life-event, locker-room and trade paths, and it says **nothing about the in-season stat-driven pass**. A second run with decision auto-resolution never got past startup and was stopped.
- Everything about the stat-driven pass (`run_data_storyline_pass`) is therefore **[C]** unless marked otherwise.
- The full backend test suites I started for comparison were stopped before finishing, so there is no whole-suite regression result for the earlier scoring/popup changes. The targeted storyline, trade and popup test files passed.

---

## 1. Ranked summary

| # | Finding | Severity | Verified |
|---|---|---|---|
| 1 | Social/post copy **invents values** when evidence is missing (.900 SV%, 2.80 GAA, 0.00 PPG, "Unknown player", "the club", $0M cap) | High | [V] |
| 2 | Cap hit reads as **0 for all 736 rostered players** in the storyline engine, so the contract-pressure story cannot fire | High | [V] |
| 3 | Social evidence lookup **raises `TypeError`** for rostered players (`float()` on a method) | High | [V] |
| 4 | Missing template values render as **raw `{placeholder}`** or as **empty text** depending on which of three renderers runs | High | [V] |
| 5 | **No coach, GM or captain data exists** for AI teams; coach/GM/captain stories fall back to generic text or cannot fire | High | [C] |
| 6 | Prospect stock stories carry **no player name** (they print an internal id), attribute to the user's team, and have no cause type | High | [C] |
| 7 | Streak length was **never supplied** (fixed earlier in this session); several other triggers still use record **proxies** | Medium | [V]/[C] |
| 8 | Popups carry only headline/summary/cause; no stats, OVR, age, contract or record; user-team only | Medium | [C] |
| 9 | Trigger reasons thin and dropped by the popup (see the correction in section 11) | Medium | [V]/[C] |
| 10 | Stat triggers are season-aggregate PPG / SV% against an OVR-only expectation; no minutes, role, or context | Medium | [C] |
| 11 | Dead and half-wired story types (copy with no emitter; emitters with no copy or cause mapping) | Medium | [V] |
| 12 | Reporter/outlet attribution present on only **4%** of stories; effects payload empty on **99%** of life events | Low-Med | [V] |
| 13 | Coverage gaps: no milestones, awards races, deadline, waivers, cap trouble, coach firings | Info | [C] |

---

## 2. How data flows

```
game sim / rosters / standings
  -> run_data_storyline_pass()        storyline_engine.py:862   14 stat-driven types (skater, goalie, team, prospect, injury)
  -> storyline_coverage.py            rolling last-10 form, box-score moments, org desks, league social
  -> conduct / life / locker-room     personal_life, locker_room_pulse, contract_year_heat ...
  -> _build_storyline()               storyline_engine.py:~660   the one record shape
  -> copy: storyline_copy / procedural / social_*   headline, body, posts
  -> _u_enqueue_story_impact_popup()  user-team only -> pending_ui_popups (cap 30)
  -> ShowcasePopupLayer (frontend)
```

The record shape (`_build_storyline`) has 40+ fields, but most emitters fill about ten of them.

---

## 3. Data that is invented or silently defaulted

### 3.1 Social copy fabricates facts **[V]**
`social_copy_engine.build_evidence_context` fills gaps with plausible-looking values. I called it on a storyline with no evidence:

| Field | Value produced |
|---|---|
| name / team | `Unknown player` / `the club` |
| save_pct / gaa / expected_save_pct | `.900` / `2.80` / `.905` |
| ppg, points, games_played, overall, age, cap_hit | `0.00`, `0`, `0`, `0`, `0`, `0` |
| team_record / league_rank | `—` / `—` |
| injury_type | `undisclosed` |

A template filled from it reads: *"Unknown player (the club, —) has a .900 SV% and 2.80 GAA through 0 GP; cap hit $0M. Rank —."* The save percentage and GAA are not zero or blank; they are **made-up numbers that look real**, which is worse than blank because a reader cannot tell them from data.

### 3.2 Three renderers, three failure modes **[V]/[C]**
| Renderer | Behaviour when a key is missing |
|---|---|
| `social_templates.render_template` | leaves the literal `{key}` in the text (verified: `{unsupplied_key}` printed as-is) |
| `storyline_copy._format_line` | `defaultdict(str)`: missing key becomes **empty string** (e.g. "riding a -game win streak") **[C]** |
| `social_copy_engine._safe_format` | catches `KeyError` and returns `""`, so the post is **silently dropped** **[C]** |

`render_template` also treats a value of `0` as "missing" for `requires` checks (`if not ctx.get(key)`), so a genuine 0-goal or 0-PIM value blocks a template.

### 3.3 Defaults in the stat-driven pass **[C]**
- Age defaults to **27** (`age_by_id.get(pid, 27)`); `_player_age` defaults to **26**.
- `is_rookie = age <= 23 and gp <= 40`: rookie status comes from age and games, not draft year or NHL experience. A 22-year-old sophomore with 30 GP is a "rookie".
- OVR defaults to 0, then falls back to the roster player.
- `credibility = 92 if evidence else 60` measures whether an evidence dict exists, not how good it is.
- `years_left` defaults to **99**, so "contract year" is false whenever the contract cannot be read.

---

## 4. Data that is never there

### 4.1 Cap hit is always 0 in the storyline engine **[V]**
`_cap_hit_m` (`storyline_engine.py:542`) returned 0 for **736 of 736** rostered players in a live session. It reads only `contract.cap_hit_m / aav_m / salary_m` through `getattr`. The canonical helper `economy/cap_engine.player_cap_hit_millions` also checks player-level `cap_hit_m` / `contract_aav_m` and contract keys `cap_hit`, `aav`, `salary_aav`, and handles dict contracts.

Consequences:
- **`contract_pressure` cannot fire** (requires `cap >= 6.5`).
- Every story `ctx["cap"]` is 0; the cap evidence field is 0.
- `contract_year` evidence is unreliable (see 3.3).

### 4.2 Social evidence lookup crashes for rostered players **[V]**
`social_copy_engine._lookup_session_evidence` does `float(getattr(player, "overall", None) or getattr(player, "ovr", None))`. `Player` has **no `overall`** and `ovr` is a **method**, so this raises `TypeError`. It also reads `player.cap_hit` / `player.salary`, which `Player` does not have. Same bug class as the `Player.ovr` issue fixed earlier in the prospect-scoring module. Whether the exception is swallowed upstream depends on the caller; either way the enrichment cannot succeed.

### 4.3 Coaches, GMs and captains do not exist for AI teams **[C]**
- `Team` defines no `coach_name`, `gm_name`, `captain` or `captain_id`; `Player` defines no `is_captain`. `captain_id` is only ever **cleared** (`franchise_retirement.py:634`), never set.
- `emit_org_desk_storylines` therefore falls back to `"{team} bench"` for the coach and `"the general manager"` for the GM (headlines like "Heat rising on Ottawa Senators bench").
- The **captaincy story can never fire**, and `CAPTAINCY_CHANGED` / `CAPTAIN_TRADED` cause types have nothing feeding them.

### 4.3b Prospect stories carry no identity **[C]**
Draft-rank keys are player ids (`pk = str(getattr(p, "id"))`). The prospect emitters do `pname = str(key).split("|")[-1][:40] or "Prospect"`, so the body prints an **internal id**. The headlines are fixed jokes ("Anonymous draft hopeful has rudely entered the first-round conversation"). They also:
- set `team_id=uid` (the user's team) though the prospect belongs to no NHL club yet;
- carry no `player_id`, name, league, age, position or stats (evidence is just rank numbers);
- have **no `cause_type` mapping** (the only two emitted types without one);
- scan `list(ranks.items())[:80]`, the first 80 dict entries, **not the top 80 prospects**.

### 4.4 Anonymous moves **[C]**
- AHL **send-down** story: "assigns a body to the AHL". The player is never named (`sent[0]` is never resolved).
- AHL **call-up** story: names the player but gives no OVR, age, stats or reason.

### 4.5 Fields the copy reads but nothing supplies **[V]**
Cross-check of `ctx.get(...)` keys in the copy modules against `story_ctx(...)` callers: `opponent` is read and **never supplied** (and no opponent data is passed for any story). (`streak` was in this list until fixed earlier this session.)

---

## 5. Signal that is too thin

### 5.1 Triggers **[C]**
- **Skater production** is season-aggregate PPG against `(OVR - 62) / 38`, scaled by position. No ice time, line/PP role, shooting percentage, or team context. A depth player on the third line and a first-liner with the same OVR share one expectation.
- **Goalie form** compares SV% to `0.870 + (OVR - 70) * 0.0018`. No shot quality, workload, or team defence, and the minimum samples are only 3 and 6 GP (`GOALIE_GP_MINOR/MAJOR`).
- **Team stories** use records: `hot_team`, `cold_record`, `coach_pressure`, and the press-moment "losing skid" (`l >= 4 and w <= 2`) are still record **proxies**, not verified streaks. Only the two main streak stories were converted this session.
- **Playoff race** is emitted only for the **user's** team and only in league ranks 7-11.
- **Cooldown keys** contain only `player|season` for most types, so one star slump per season is all a player can produce; `repeat_count` escalation is never exercised in the observed data.
- League-wide cap of **7 stories per day** (`select_daily_data_stories`), so a busy day drops stories with no record of what was dropped.

### 5.2 Injury story **[C]**
Only games out and a tier label. No injury type in the story (`injury_type` is filled from the tier), no return date, no replacement or depth-chart consequence, no lineup data even though the copy says "depth chart under stress".

### 5.3 Trade stories **[C]**
Rumor stories invent a target: a **random** player from a seller's roster with `trade_value = OVR * 0.72` and a hard-coded credibility of 42. Nothing links the rumor to team need, contract status, or availability. (Completed-trade popups now resolve player data; see `MINOR_AND_JUNIOR_LEAGUE_STATS_REPORT.md` history and the popup fixes in this session.)

### 5.4 Popup payload **[C]**
`_u_enqueue_story_impact_popup` copies only: title, headline, summary, cause type, team/player id and name, date, severity and rating-impact lines. **It drops the evidence dict**, so a popup about a slump cannot show the numbers that caused it. Also:
- **user-team only** (`if tid != utid: return`), so league events never pop up;
- the pending queue is truncated to **30** (`[-30:]`) while trade popups allow 500, so story popups can be silently displaced.

### 5.5 Trigger explanations are missing **[V]**
`build_data_story_trigger_context` is called with an **empty ctx** (`{}`) at creation, and in the live sample **0 of 95** stories carried `trigger_reason(s)`. The popup UI has a "Why this story fired" section that therefore never renders for these stories.

---

## 6. Observed output (live session, preseason only) **[V]**

95 stories, 140 attempted days but the calendar never left preseason (see limits).

| Measure | Result |
|---|---|
| By category | personal_life 41, locker_room 29, team 13, business 4, trade 3, other 5 |
| Stat-driven stories | **0** (no games played) |
| Stories with **no `effects`** | 87 of 88 that have the field (99%) |
| `players` list empty | 89 of 95 (94%) |
| `team_id` empty | 33 of 94 (35%) |
| `source_label` empty | 42 of 95 (44%) |
| No `evidence` dict | 22 of 95 (23%) |
| No `trigger_reason(s)` | 95 of 95 |
| Reporter / outlet / knowledge type present | **4 of 95 (4%)** |
| `cause_type` missing | 6 of 95 |

Reading it: the life-event and locker-room paths produce stories that name a player but carry no mechanical consequence (`effects` empty) and almost no attribution. 41 of 95 are personal-life items (home repairs, unexpected expense, pregnancy news), so **this desk is 43% of preseason output but attaches no data to anything else in the game**. Whether family details are persisted on the player profile was not checked.

---

## 7. Dead and half-wired story types **[V]**

| Situation | Types |
|---|---|
| Copy handler exists, **engine never emits it** | `goal_drought`, `veteran_fade` (`star_underperforming`, `win_streak`, `losing_skid` are emitted under other stypes) |
| Emitted, **no copy branch** (hard-coded strings) | `injury_ripple`, `prospect_rising`, `prospect_falling` |
| Emitted, **no `cause_type`** | `prospect_rising`, `prospect_falling` |
| Cause types defined with nothing observed feeding them | `CAPTAINCY_CHANGED`, `CAPTAIN_TRADED` (no captain data exists, §4.3) |

Emitted stat-driven types: 14 (`backup_taking_net`, `cold_streak_team`, `contender_collapse`, `contract_pressure`, `goalie_heater`, `goalie_meltdown`, `hot_streak_team`, `injury_ripple`, `playoff_race`, `prospect_falling`, `prospect_rising`, `rookie_breakout`, `superstar_carrying`, `surprise_team`).

---

## 8. Coverage gaps **[C]**
Searching the emitters and cause-type lists finds **no** story for: career milestones (the only milestone headlines are in the league sim's news feed), award races, the trade deadline, waivers, cap trouble, coach firings, or mid-season extension news (one `EXTENSION_REJECTED` cause type, tied to user actions). AHL/ECHL/junior performance never feeds a story, even though those players now have stat lines and history (`prospect_stat_history`). Rolling form covers the last 10 games only for skaters with 5+ games logged.

---

## 9. Recommended fixes, in order

1. **Stop inventing values (3.1, 3.2).** Remove every fabricated default in `build_evidence_context`. Make missing data explicit: a template that needs a value must be skipped (or a "no data" variant used), never filled. Unify the three renderers on one rule: unresolved placeholder means do not publish.
2. **Fix `_cap_hit_m` (4.1)** to call `player_cap_hit_millions`; this alone re-enables `contract_pressure` and gives every story real cap evidence. Add a test that fails if the cap is 0 for a rostered player.
3. **Fix `_lookup_session_evidence` (4.2)** to use the shared OVR helper and the cap engine.
4. **Give prospect stories an identity (4.3b):** resolve the id to name/league/age/position, attribute to the right club or none, add the cause type, and scan the top-N prospects.
5. **Decide the coach/GM/captain model (4.3).** Either generate and persist them (and feed captaincy causes) or remove those stories. Do not ship "Ottawa Senators bench".
6. **Forward `evidence` and identity fields into popups (5.4)** and build `trigger_reasons` from real evidence (5.5).
7. **Convert remaining record proxies to verified signals (5.1)** using the streak helper and recent-game log; add TOI/role to the skater expectation.
8. **Name the send-down player and add OVR/age/why to moves (4.4).**
9. **Reconcile dead types (7):** emit `goal_drought` / `veteran_fade` or delete their copy; add copy and cause mappings for the rest.
10. **Add a data-completeness gate:** a publish check that rejects any story whose headline or body contains an unresolved placeholder, "Unknown", "the club", or a numeric default, and logs what it dropped.

## 10. To close the audit gap
Run an in-season audit that resolves pending decisions and fills lines (the script used here is in the session scratchpad, not the repo), advance to about day 60, and repeat the field-completeness and suspicious-text pass on the **stat-driven** stories. That is the one area of this report that is still code-reading only.


---

## 11. Fix status (top issues addressed)

### Correction to finding #9
"`trigger_reasons` empty on 95 of 95" was overstated. Those 95 preseason stories came from the life-event and locker-room paths, which never go through `_build_storyline`. The stat-driven builder **does** pass evidence to the trigger-reason engine (`build_data_story_trigger_context` reads it), so those stories did carry reasons for the types that engine understands. What was true: the reasons were thin (score/gate rows only), evidence rows were not included, and the popup dropped them anyway.

### What changed

| # | Finding | Fix | Where |
|---|---|---|---|
| 1 | Social copy invents values | `build_evidence_context` now contains **only real values**; missing means absent. `enrich_dynasty_context` no longer adds random contract years, prior overall, draft round, rival cap hit/term, default cap space / salary cap / morale, or `rival_player` (which was the story's own player). `build_entity_context` likewise. | `social_copy_engine.py` |
| 2 | Cap hit always 0 | `_cap_hit_m` now calls `economy.cap_engine.player_cap_hit_millions` (all field names, dict and object contracts). Contract-years reads dict/object and returns **None** when unknown instead of 99. `contract_pressure` can now fire. | `storyline_engine.py` |
| 3 | Lookup crashes | `_lookup_session_evidence` uses `_player_ovr99`, the cap engine, real contract years and age; adds real `league_rank`. No `float(method)`. | `social_copy_engine.py` |
| 4 | Missing values rendered as `{key}` / empty / dropped | `render_template` returns `None` unless **every** placeholder resolves (0 is a real value; None/"" are missing); `filter_templates` uses the same rule; `_format_line` returns "" on a missing key. Reporter posts must have a fillable opener (retries, then falls back to the real headline). Publish gate extended to catch "Unknown player", `None`, `$0M`, "Rank -", "(-)". | `social_templates.py`, `storyline_copy.py`, `social_copy_engine.py` |
| 5 | Coach/GM/captain data absent | No data was invented. AI-team coach headlines no longer name a made-up bench ("Pressure building on X's coaching staff"); the GM-seat story no longer names **the user's** GM for every AI team (a latent bug: it read `session.gm_name`), it says "the front office". The captaincy story still needs real captain data (see below). | `storyline_coverage.py` |
| 6 | Prospect stories anonymous | Resolves the id to the real player; headline names them ("X climbs 12 spots to No. 5"), evidence carries age, position, league, club and current season line; scans the **top** 80; skips prospects it cannot name; no team attribution (was the user's club); `PROSPECT_RISING/FALLING` cause types added. | `storyline_engine.py` |
| 7 | Popup drops the facts | Popup payload now includes evidence, cause, trigger reasons, position, OVR and team name. Every story with evidence gets concrete "Why this story fired" rows (games played, points, P/GP, save %, record, streak, cap hit, rank...). Popup shows position and OVR. | `storyline_engine.py`, `ShowcasePopupLayer.js` |

### Also found and fixed while testing
- The reporter-post stat suffix picked a random label and, when that stat was missing, silently used the points value under the wrong label (e.g. "cap hit: 18" for an 18-point player). It now only labels stats the context has.
- Angle-specific openers each needed 5-6 fields and only ever worked because the missing ones were invented. Four low-requirement openers were added to every angle.

### Tests
`backend/tests/test_storyline_data_integrity.py` (13 tests): cap hit for every contract shape, unknown contract years, no invented context, random-free enrichment, unresolved placeholders, zero as a real value, the extended publish gate, 200 posts with and without evidence (no invented values, no mislabeled stats), the lookup on a real `Player`, evidence rows, popup payload, and the prospect stories end-to-end through the real `run_data_storyline_pass`.

### Still open
- **Captains, coaches, GMs for AI teams** do not exist as data. Generating and persisting them is a design decision, so it was not done here. Until then the captaincy story cannot fire.
- **Proxy triggers** (`hot_team`, `cold_record`, press-moment `losing_skid`) and the OVR-only production expectation are unchanged.
- **Popups** are still user-team only; the story-impact queue is still capped at 30.
- **Dead/half-wired types** (`goal_drought`, `veteran_fade` copy with no emitter; missing copy for `injury_ripple`) are unchanged.
- The **in-season stat-driven audit** is still not run against a live session; the prospect and cap fixes are covered by tests, not by observing a season.
