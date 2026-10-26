# PuckGPT: intended scope and current gaps

Snapshot of `main` @ `cae68db` (2026-10-26).

## 1. What PuckGPT is supposed to be

A search desk on the hub that answers plain-language questions about **this save**. It should return four things:
search results, analysis, recommendations, and historical context, plus insights specific to the user's team.

Example questions it is meant to handle:

| Kind | Example |
|---|---|
| Roster fit | Who is the best fit for my second-pair LD? Best UFA defenseman under 30? |
| Records | Longest win streak in franchise history? Biggest playoff upset? |
| Development | Who improved the most this season? Which prospect projects best defensively? |
| Drafting | What was my best draft class? Show all first-round picks since Year 1 |
| Transactions | Show every trade involving my team |
| Diagnosis | Why is my team losing? |
| Comparison | Compare Player A and Player B |
| Market | Best trade targets |

The feature depends on the save recording history as it happens. That covers league, team, player and game history (see section 4).

## 2. What exists today

| Piece | Where | What it does |
|---|---|---|
| Hub entry + screen | `frontend/src/screens/PuckGptScreen.js` (83 lines) | A search box, 4 starter chips, a one-line answer and a flat list of hits. |
| API | `GET /api/franchise/puckgpt?q=` (`backend/main.py:419`) | Calls `answer_save_query`. |
| Engine | `backend/services/sim_history.py` (637 lines) | Keyword routing, with no language model. |
| Archive | `session.puck_archive`, `session.sim_history` | Final scores with lineups, loud social posts, and an "awards published" note. |

The router (`answer_save_query`) understands only 4 kinds of question:

1. **A date** → the games played that day, with lineups.
2. **"social" / "tweet" / "post"** → the warmest or coldest voices in the social feed.
3. **"fit" / "second pair"** → your players at a position, plus the top 8 other players aged 32 or under at that position, sorted by potential.
4. **Anything else** → a keyword match against game rows. If that finds nothing, it returns the last 6 log entries.

## 3. What is missing or broken

### 3a. Bugs in what exists
- **The streak answer is wrong.** `_streak_answer` counts runs of *home-team* wins across every game in the log, so it is not your team's streak. It also reads `kind == "game"` rows from `sim_history`. `record_final_score` writes those rows, but the log is capped at 4,000 entries, so long saves lose the early seasons.
- **The "fit" answer is not a fit.** It ignores cap space, contract and term, handedness when the player is listed as plain `D`, and the stated pair or role. It never checks whether the player is actually available. It ranks only by potential.
- **The keyword fallback returns noise.** Dumping the last 6 log rows when nothing matches looks like an answer but isn't one.
- **The hits are flat text.** There are no player or team links into the dossier, and no tables or numbers.
- **The starter chips hard-code `Feb 15 2029`**, which is an empty result in most saves.

### 3b. Question types with no handler at all
Records and streaks for your team · most-improved and development · draft classes and pick history · trade and signing history · player-vs-player comparison · free-agent and trade-target search with filters (age, position, cap hit, term, RFA/UFA) · "why is my team losing" diagnosis · playoff history and upsets · awards lookup · injuries and suspensions · milestones (hat tricks, shutouts, comebacks).

### 3c. Data the save already keeps that PuckGPT never reads
These stores exist, but the PuckGPT engine doesn't touch them:

| Store | Would answer |
|---|---|
| `session.legacy` (`trades`, `seasons`, `players`, `h2h`, `hof`, `captaincy`, `events`) | Trades involving my team, franchise season records, head-to-head, Hall of Fame |
| `trade_history`, `waiver_history`, `cpu_roster_moves_log` | Transaction history |
| `draft_results_archive`, `draft_stock_history` | Best draft class, first-round picks since Year 1 |
| `rating_history`, `potential_history` | Who improved the most |
| `player_season_archive`, `season_history` | Career totals, past standings |
| `player_award_history`, `career_awards` | Awards lookup |
| `injury_log_all`, `injury_log_major` | Injury history |
| `trade_demands_history`, `gm_meeting_history` | Morale and front-office context |

**The quickest big win is wiring these existing stores in.** Most of the example questions need no new tracking.

### 3d. Data not tracked yet
- **Streaks:** per-team win and loss streaks, recorded at the moment they end. Today they can only be rebuilt from capped logs.
- **Game flags:** comeback (largest deficit overcome), shutout, hat trick, OT or shootout, and milestone performances. These should be stored on each archived game.
- **Draft lottery results by year.** The lottery payload is overwritten each season.
- **League records:** single-season and career leaders, written each season.
- **Suspensions** with reason, incident and length (also report item 14).
- **Coaching changes.**
- **Playoff series results,** including the seeds needed to define an upset.

## 4. Recommended build order

1. **Fix the streak bug and stop the fallback noise.** This is small and stops wrong answers.
2. **Restructure the router into intents.** Each handler is a pure function `(session, parsed_query) → {answer, analysis, hits[], table?, links[]}`. Parse out entities first: player names (fuzzy-matched against all rosters), team names, position, season, and an age or cap filter.
3. **Wire in the existing stores (3c),** one handler each: trades, draft class, most improved, awards, injuries, past seasons, and the franchise record book.
4. **Add the real "fit" and target search:** position and handedness, cap room after the move, contract status, the selling team's willingness (reuse the trade-value code), and filters for age, AAV and term.
5. **Add compare and diagnose.** Compare shows two players side by side, with ratings, stats and contract. Diagnose shows team stat ranks versus the league, goaltending, special teams, injuries and line ice time, and turns them into ranked reasons.
6. **Add the new tracking (3d).** Record it at the moment it happens and keep it uncapped per season.
7. **Upgrade the UI:** answer plus analysis, result cards that link to the dossier and team pages, tables for list answers, a follow-up chip per intent, and a recent-questions list.
8. **Optional:** a Claude-backed rewriter that turns free text into the intent JSON, with the deterministic handlers still producing every fact. Answers stay grounded in the save, and the parsing copes with messy phrasing.

## 5. Size estimate
- Steps 1–3: small to medium. This is mostly read-only code over data that already exists.
- Steps 4–5: medium. They reuse the trade-value and cap code.
- Step 6: medium. It touches the sim and offseason hooks and needs save migration.
- Step 7: small to medium, frontend only.
