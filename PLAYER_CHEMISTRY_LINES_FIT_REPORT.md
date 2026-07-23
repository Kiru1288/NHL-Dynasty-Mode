# Player Chemistry, Line Deployment, Fit & Ice-Time Handling Report

**Scope:** How NHL Franchise Mode models chemistry, Edit Lines “fit,” position mismatches, and player reactions to demotion / low ice / scratches — and what actually reaches the sim.  
**Date:** 2026-07-13  
**Sources:** `editLines.js`, `ChemistryScreen.js`, `SimEngine/.../systems/chemistry.py`, `world/chemistry.py`, `engine.py`, `storyline_engine.py`, `franchise_sim.py`

---

## Executive summary

The game has **three chemistry systems** and **two lineup systems** that are only partly connected.

| Layer | What it does | Feeds games? | Tied to Edit Lines? |
|-------|--------------|--------------|---------------------|
| Edit Lines unit chemistry / fit % | UI scores for line mix & position match | No | Yes (local only) |
| Franchise room / pair chemistry API | Personality + playstyle + mood report | Reporting / psych inputs | **No** — projects lines from roster order |
| World team chemistry (`_world_chemistry`) | Small team strength & chaos modifiers | Yes | No |
| Engine playstyle line chemistry | Seasonal rating buffs / line composite | Yes | No — auto-assigns lines |
| Saved `session.lines` | Persist + demotion/scratch storylines | Storyline psych / OVR mods only | Yes on save |
| Game dress / TOI | OVR-sorted lines in `engine.py` | Yes | **No** |

**Bottom line:** Deploying someone on L4 or scratching them in Edit Lines is real for **narrative morale** (after a prior save). The **chemistry numbers and position-fit % on that screen do not set in-game ice time or line strength**. Games still auto-build lines by overall.

---

## 1. Chemistry systems

### 1.1 Franchise chemistry profiles (`systems/chemistry.py`)

Each player can carry a `chemistry_profile`:

| Field | Role |
|-------|------|
| `personality`, `playstyle` | Categorical compatibility |
| `leadership`, `ego`, `work_ethic`, `coachability`, `adaptability`, `temperament`, `competitiveness`, `loyalty`, `defensive_buy_in`, `pressure_response`, `room_presence` | 0–100 style traits |

Also: `chemistry_relationships` (familiarity by player id), `chemistry_history`.

**Pair chemistry (0–100)** roughly:

```
0.38 × personality_compat
+ 0.34 × playstyle_compat
+ 0.18 × mood          (avg morale+confidence)
+ 0.10 × familiarity
− 0.14 × ego_tension   (when ego_a + ego_b > 120)
```

**Labels:** Broken &lt;30 · Awkward &lt;45 · Neutral &lt;60 · Connected &lt;75 · Strong &lt;90 · Elite ≥90

**Forward line bonuses:** playmaker+sniper (+4). Penalties for weak average `defensive_buy_in` (&lt;44), ego clashes.  
**Defense pair:** complements (shutdown / puck-mover style) get bonuses; twin offensive-D styles get penalties.

**Team room overall** blends morale, confidence, role_satisfaction, leadership, buy-in, coach_trust, chaos resistance, minus tension.

**API:** `GET /api/franchise/chemistry` → `ChemistryScreen.js`.

**Gap:** Chemistry screen builds F1–F4 / D1–D3 via `_project_lines_from_roster` (**roster / overall order**), not your saved Edit Lines.

**Gap:** `apply_daily_chemistry_tick` / `apply_storyline_chemistry_effect` exist but are **not called** from the live franchise loop (as of this audit).

---

### 1.2 World team chemistry (`world/chemistry.py`)

- Team float `_world_chemistry` (~0.52 start).
- Updates after games (~±0.008, blowouts larger); damped by roster stability / identity.
- **Sim impact:** `team_strength_modifier ≈ 1 + 0.04 × (c − 0.5)` (~±2% at extremes), plus slightly lower chaos noise when high.

This is **not** the Chemistry Screen profile/pair model.

---

### 1.3 Engine playstyle line chemistry (`engine.py`)

Separate playstyle vocabulary (e.g. `sniper`, `playmaker`, `power_forward`, `offensive_d`, `defensive_d`).

- Auto-optimizes forward/D line assignments for chemistry.
- Applies rating multipliers on offense / pass / IQ keys from chemistry score.
- Low chemistry can ding morale.
- Composite → `_runner_line_composite_strength` → small team strength multiplier (~0.935–1.065).
- Used on yearly / preseason passes — **not** from Edit Lines slots.

---

### 1.4 Edit Lines local chemistry (UI-only)

In `editLines.js`:

- `calculateUnitChemistry` — blends overall, local profile fields, positional balance, handedness.
- Labels roughly Elite ≥90 … Poor Mix &lt;60.
- Warning when unit score &lt; 62.

**Important scale bug:** franchise roster `morale` is often **0–1**, while Edit Lines checks treat it like **0–100**, so real morale (~0.5) looks “unhappy” in those UI checks. Live roster often **lacks `chemistry_profile`**, so the UI falls back to ~75 placeholders.

This score is **never sent to the sim**.

---

## 2. Line deployment (Edit Lines)

### Structure (`editLines.js`)

| Unit | Slots |
|------|-------|
| 4 forward lines | LW / C / RW |
| 3 defense pairs | LD / RD |
| Goalies | Starter / Backup / (Third) |

Persistence:

1. `localStorage` (`nhl_franchise_even_strength_lines_v1`)
2. `POST /api/franchise/lines` with `unit_type: "even_strength"`

Backend (`save_franchise_lines`) validates unknowns/duplicates/goalie-in-skater slots and soft-warns on mismatches; stores under `session.lines[unit_type]`.

### PP / PK

Session schema mentions PP/PK `unit_type`s. There are separate Power Play / Penalty Kill screens, but **Edit Lines itself only saves even-strength**. Lineup fallout storylines also only fire for even-strength saves.

### Auto-fill

`makeInitialLines` packs C/LW/RW/D/G by roster order — not coach schemes.

---

## 3. Deployed “fit” and schemes

There is **no full coach-scheme × line-slot fit engine** for Edit Lines.

| Concept | Implementation | Meaning |
|---------|----------------|---------|
| **`chemistryFitScore(player, slot)`** | `editLines.js` | Position match only |
| Exact slot match | **92** | e.g. C in C |
| Legal family | **72** | e.g. RW in C (forward family) |
| Illegal | **28** | e.g. D in C |
| Unit chemistry | Local mix score | Not scheme-aware |
| `_system_archetype_line_bonus` | `engine.py` | Team system (`run_and_gun`, `defensive_lock`, …) nudges **auto** playstyle chem ≤ ~0.09 |
| Coach fit | Coach hiring entity | Not line placement |

So “fit when dropped on a line” in the UI = **positional legality %**, not “does this star want L3 under a defensive lock.”

---

## 4. Position mismatches

| Layer | Behavior |
|-------|----------|
| **Drag-drop** | Hard block if `posFit` fails (“cannot play {slot}”) |
| **Soft UI warnings** | Out of position (family but not exact), 3 snipers, same-handed D pair, low chem |
| **Backend save** | Soft warnings; save still succeeds |
| **In-game lines** | Only F vs D buckets matter; LW/C/RW from Edit Lines are **ignored** |

`posFit` families:

- Goalie slots → G only  
- LD/RD → D / LD / RD  
- C/LW/RW → C / LW / RW / F  

A winger at center is **allowed** (72 fit); backend may warn. Many roster rows are plain `"D"`, so LD vs RD “mismatch” is often soft.

---

## 5. Happiness: lower lines, low ice, scratched

### 5.1 What Edit Lines save actually does

`storyline_engine.py` maps units to ranks:

- F1→1 … F4→4  
- D1→1 … D3→3  
- Starter=1, Backup=2, Third=3  

On save (only if a **previous** lineup exists):

| Change | Cause | Typical morale impact* |
|--------|--------|-------------------------|
| Ranked → off roster | `PLAYER_SCRATCHED_BY_USER` | −3 to −6; room tension; temp OVR −1/−2 (~6 GP) |
| Drop ≥2 ranks | `PLAYER_DEMOTED` | −4 to −8; coach trust / tension; temp OVR −2/−3 |
| Drop 1 rank | `PLAYER_ROLE_REDUCED` | −2 to −4; lineup pressure |
| Rise in rank | `PLAYER_PROMOTED` event | Template has +morale, but **positive fallout storyline often not emitted** |

\*Display units then applied as `×0.01` onto `psych.morale` (0–1). Low-character players can escalate into locker-room arcs.

**First lineup save** has empty previous ranks → **no demotion/scratch events**.

### 5.2 What does *not* continuously run off Edit Lines

| Idea | Reality in code |
|------|-----------------|
| Happy on L1 / upset on L4 every game | Only when you **save** a rank change (with prior baseline) |
| Low TOI from your lines | Games assign TOI from **auto OVR lines**, not `session.lines` |
| Scratched in Edit Lines = scratched in game | Game scratches use tank / dress logic (`_tank_scratched_ids`), not “missing from saved lines” |
| `ice_time_satisfaction` | Behavior context uses performance/coach room; `scratched_recently` is largely **hardcoded 0** |
| `Team.role_mismatch_factor` | Based on **roster depth index**, not line slot |

### 5.3 Post-game ice-time decisions

Franchise decisions can enqueue post-day options (promote / bench message / calm) that nudge `psych` — separate from Edit Lines chemistry UI.

### 5.4 World morale in games

`world/morale.py`: small per-player performance and team strength nudges from morale / role_satisfaction proxies after games.

---

## 6. How games actually assign ice time

Rough even-strength pattern in `engine.py` (`_gm_forward_lines`, `_gm_toi_seconds_for_line`, `_gm_role_usage_mult`):

1. Sort dressed forwards/D by overall (and related scores).
2. Bucket into 4 F lines / 3 D pairs.
3. Approximate TOI bands: L1 ~18–22, L2 ~15–18, L3 ~12–15, L4 ~8–12; D higher.
4. Usage multipliers by line index (~2.06 / 1.38 / 0.96 / 0.72).

Optional `line_role` on a player object can override — **Edit Lines does not set it**.

So a player you parked on L4 can still see L1 ice if their OVR sorts them top of the dressed list.

---

## 7. Schemes (coach / system)

“Schemes” in sim sense are team **systems** (e.g. run-and-gun, defensive lock) that:

- Slightly boost matching **auto** playstyle chemistry in the engine pass.
- Are **not** shown as “this L2 winger fits your scheme 87%” in Edit Lines.

Coach fit scores exist for hiring / staff context, not for slotting Player X onto PP1.

---

## 8. Gaps (highest impact)

1. **`session.lines` do not drive game dress, TOI, or PP/PK units.**
2. Chemistry Screen lines ≠ saved Edit Lines.
3. Daily / storyline chemistry tick helpers appear unused in the live loop.
4. Two chemistry taxonomies (franchise profiles vs engine playstyles) are not unified.
5. Edit Lines chem math ≠ backend; morale scale / missing profiles skew UI.
6. PP/PK unit types are scaffolded; even-strength is what Edit Lines saves for fallout.
7. Edit Lines “scratch” ≠ in-game scratch.
8. Continuous ice-time happiness not fed by your deployed minutes.
9. Promotions under-reward narrative vs demotions; first save has no demotion baseline.
10. No true scheme×line deployed-fit model in the lines UI.

---

## 9. Key file index

| Concern | Path |
|---------|------|
| Edit Lines UI | `frontend/src/screens/editLines.js` |
| Chemistry UI | `frontend/src/screens/ChemistryScreen.js` |
| API client | `frontend/src/services/franchiseService.js` |
| Lines / chemistry routes | `backend/main.py` |
| Save lines + world hooks | `backend/services/franchise_sim.py` |
| Session `lines` | `backend/services/franchise_session.py` |
| Profile / pair / room chem | `SimEngine/app/sim_engine/systems/chemistry.py` |
| World team chem | `SimEngine/app/sim_engine/world/chemistry.py` |
| World morale | `SimEngine/app/sim_engine/world/morale.py` |
| Line chem + game TOI | `SimEngine/app/sim_engine/engine.py` |
| Lineup fallout ranks | `SimEngine/app/sim_engine/franchise/storyline_engine.py` |
| Ice-time decisions | `SimEngine/app/sim_engine/franchise/decisions.py` |
| Role mismatch proxy | `SimEngine/app/sim_engine/entities/team.py` |

---

## 10. Practical takeaway for GMs

- **Saving** a demotion or scratch can hurt morale / temporary OVR via storylines.  
- **On-ice results** still mostly follow auto-ranked overall lines + world chemistry/morale + seasonal playstyle synergy.  
- Chemistry / fit meters on Edit Lines are **guidance for you**, not the sim’s deployed lineup.  
- To make “happy on L1 / upset when scratched” fully meaningful in games, `session.lines` would need to drive dress/TOI (and chemistry reporting should read those same units).
