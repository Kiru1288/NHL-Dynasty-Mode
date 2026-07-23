# Backend services

**Source of truth for the live franchise API.** `backend/main.py` imports from here only.

`SimEngine/app/sim_engine/franchise/` is a legacy split copy (stubs/reference). Editing those files does **not** change the running game.

Franchise orchestration lives here. SimEngine packages (`entities/`, `economy/`, `trades/`, etc.) provide game rules; these modules wire them into the playable franchise API.

Start the API: `backend/start_api.ps1` — then verify `http://127.0.0.1:8000/api/health` shows `code.franchise_sim` under this folder.

| Module | Role |
|--------|------|
| `franchise_sim.py` | Core franchise engine (day advance, state, trades, decisions) |
| `franchise_session.py` | Mutable session state |
| `franchise_store.py` | In-memory session registry for HTTP API |
| `franchise_paths.py` | SimEngine `sys.path` bootstrap |
| `nhl_season_calendar.py` | NHL season calendar |
| `franchise_offseason.py` | Offseason stage machine |
| `franchise_entry_draft.py` | NHL Entry Draft execution (order, CPU picks, assignment) |
| `franchise_retirement.py` | Final Skate retirement pass |
| `franchise_scouting.py` | GM scouting assignments |
| `trade_service.py` | Trade evaluation / market API |
