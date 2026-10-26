# PuckGPT: one lazy-built JSON archive and an in-house language model

Goal: type anything about the save and get a correct answer, with **no outside AI service and no cost to sim speed**.

Two rules:
1. **The sim never writes the JSON file.** The file is built only when someone uses the PuckGPT text box, from data the save already keeps.
2. **PuckGPT's understanding of the question comes from our own small language model.** It is trained in code on thousands of generated wordings of each question type, and it runs locally in a few milliseconds.

```
 SIM (unchanged speed)                          PUCKGPT (only when the text box is used)
 ─────────────────────                          ─────────────────────────────────────────
 existing stores already in the save:           1st question of the session:
   box-score book, legacy, trade_history,  ───►   build/refresh saves/<id>.puck.json (only seasons not built yet)
   draft archive, rating_history,                 load into memory indexes (lazy, per section)
   season archives, injury log, social          every question:
 + a few O(1) list appends for things              question ─► PuckLM (intent + slots) ─► handler ─► answer
   not kept today (series, suspensions…)
```

---

## 1. Zero cost during the sim

- **No file I/O, no JSON encoding and no indexing while simulating.** The sim keeps doing exactly what it does now.
- Most of what PuckGPT needs is **already stored in the save**: the box-score book (every game, every season), `legacy` (trades, seasons, h2h, HOF), `trade_history`, `waiver_history`, `draft_results_archive`, `rating_history`, `potential_history`, `player_season_archive`, `season_history`, `injury_log_all`, `player_award_history` and the social `puck_archive`.
- The few things not kept today get a **one-line list append** at the moment they happen. That's O(1), with no formatting and no lookups:
  - playoff series (seeds, winner)
  - suspension reason, incident and length
  - coaching changes
  - lottery results by year (today they get overwritten)
  - game flags (comeback deficit, hat trick, milestone)

  That's a few hundred appends per season, an unmeasurable cost.

## 2. The JSON file: built lazily, only from the text box

**When:** the first time the PuckGPT box is opened or a question is submitted in a server session. Never during the sim, and never on save load.

**What:** one file, `saves/<id>.puck.json`:

```json
{
  "meta":   {"version":1, "built_through":{"season":2029,"day":214}, "user_team":"TOR"},
  "players":{"<id>":{"name":"…","pos":"LD","shoots":"L","born":2004,"draft":{…}}},
  "teams":  {"TOR":{"name":"…"}},
  "seasons":{
    "2027":{"standings":[…],"player_lines":{…},"games":[…],"series":[…],"awards":[…],
            "draft":[…],"lottery":{…},"transactions":[…],"injuries":[…],
            "suspensions":[…],"ratings":{…},"streaks":{…}}
  }
}
```

**Incremental build:** finished seasons never change. They are built once and never touched again. Each later build only rebuilds the current season's section, from `built_through` forward. A question in season 20 costs one season of work, not twenty.

**Lazy load:** the file is split by season so a question only reads what it needs.
- "Who leads the league in goals" loads the current season only.
- "Longest win streak in franchise history" loads the streak summaries from every season, not the per-game data.
- Each season section is parsed on first use and cached in memory until the session ends.

| Step | When it happens | Cost |
|---|---|---|
| First build, season-1 save | first PuckGPT question | ~0.2–0.5 s |
| First build, 20-season backfill | first question on an old save, once ever | a few seconds, "Indexing archive…" bar, runs in a background thread |
| Current-season refresh | first question after more sim days | O(new games), ~50–200 ms |
| Load one season section | first time a question needs it | ~20–80 ms |
| Lookups after load | every question | < 10 ms |

**Live data** (current rosters, contracts and ratings right now) is read straight from the session, so it's never stale.

**Memory:** only the sections that are touched get loaded. Per-game detail for old seasons can be stored as season summaries only, so the file stays small (~1–2 MB per season).

---

## 3. PuckLM: our own small language model for question wording

PuckLM's job is to turn any phrasing into the same structured query, for example:

`"who's lit the lamp the most for us since year 1"` → `{intent:"leaders", stat:"goals", team:"<user>", scope:"since", season_from:1}`

It's built in code: no external service, no download, and it trains in seconds.

### 3.1 Training data, generated from code
For each intent, we write many **templates** with interchangeable slots and synonym lists, then expand them combinatorially.

```python
INTENTS["leaders"] = [
  "who has the most {stat} {scope}",
  "{stat} leaders {scope}",
  "who leads {team_ref} in {stat}",
  "top {n} {position} by {stat} {scope}",
  "who's {stat_verb} the most {scope}",
  "best {stat} {scope}",
]
SYN["stat"]["goals"]      = ["goals", "tallies", "snipes", "markers"]
SYN["stat_verb"]["goals"] = ["scored", "lit the lamp", "found the net", "buried"]
SYN["team_ref"]["user"]   = ["us", "my team", "our team", "we", "the club"]
SYN["scope"]              = ["this year", "this season", "all time", "ever", "since year {y}", "in {season}", "in franchise history", ""]
```

The expansion adds typos (dropped or swapped letters), casing changes, filler ("hey", "can you show me"), and reordered clauses. That yields about 2k–10k examples per intent, around 100k labelled questions in total.

### 3.2 The model
- **Intent classifier:** character n-gram (3–5) and word n-gram features fed into a **linear model** (multinomial logistic regression, trained with scikit-learn or about 150 lines of pure NumPy). Character n-grams make it robust to typos and slang.
  - Model size: about 1–5 MB.
  - Inference: about 1 ms.
  - It gives a confidence score, so low-confidence questions can show a "Did you mean…" choice of the top 3 intents.
- **Slot tagger:** a dictionary and fuzzy matcher built from the live save (player names, teams, awards, seasons), plus regex for numbers ("under 30", "$4M", "top 10", dates). It's simpler and more accurate than a learned tagger for closed vocabularies.
- **Optional upgrade later:** a tiny word-embedding layer trained on the same corpus (fastText-style, still pure CPU, still under 10 MB) to generalise further to unseen synonyms.

### 3.3 Intents (v1)
Leaders and records · streaks · player card · compare · find with filters · roster fit · transactions · draft and draft-class grade · most improved / declined · awards · injuries and suspensions · games (date, hat trick, shutout, comeback, OT, upset) · diagnose team ("why are we losing") · timeline ("when did…", "first time…") · playoff history · standings and "as of" questions.

### 3.4 Getting better over time
- Every question and its predicted intent and confidence go to an `unanswered` / `low_confidence` list in the JSON file.
- New wordings found there are added as templates. Retraining runs as a test or build step (seconds) and ships with the game.
- A unit test asserts at least 95% accuracy on a held-out set of hand-written questions, so new templates can't silently break old ones.

### 3.5 Answer handlers
Each intent maps to one handler that reads the JSON sections and the live session. Every handler returns the same shape:

`{answer, analysis[], table?, cards[], followups[]}`

The analysis lines are **number-driven templates**, for example "Your PK is 29th; that's the biggest gap." Every claim is true by construction.

---

## 4. Time complexity summary

| Path | Complexity |
|---|---|
| Sim, per event | O(1), only for the few new facts; otherwise unchanged |
| JSON build | O(new games since `built_through`), only from the PuckGPT box |
| Section load | O(section size), once per session per section |
| PuckLM inference | O(question length × features), ~1 ms |
| Handler queries | O(1) dict lookups or O(rows in one season), < 10 ms; career rollups cached |

## 5. Build order
1. The O(1) appends for the missing facts (series, suspensions, coach changes, lottery history, game flags).
2. The lazy JSON builder (incremental, season-split) and the section loader.
3. The PuckLM template corpus, training script, saved model, slot tagger and accuracy test.
4. Handlers, in order: leaders, player card, find, streaks, transactions, draft, development, awards, games, compare, fit, diagnose.
5. UI: answer, analysis, table, dossier cards, "Did you mean" chips, and the indexing progress bar on first open.
