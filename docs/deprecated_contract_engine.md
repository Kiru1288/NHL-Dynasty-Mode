# Contract Negotiation Engine — Source of Truth

**Franchise mode (API + UI):** `backend/services/contract_economy.py`

- `evaluate_contract_offer()` — interest score, accept/counter/reject
- `sign_player_to_team()` — cap-validated signing
- CPU passes: `run_cpu_rfa_decisions`, `run_cpu_own_ufa_resign`, `run_cpu_offer_sheet_pass`, `tick_offer_sheets`
- FA market: `backend/services/fa_market_engine.py` (calls `contract_economy`)

**Standalone sim runner only:** `SimEngine/app/sim_engine/entities/contract.py`

- PCDS multi-week `negotiate_contract()` loop
- Not wired into franchise Re-Sign / Cap Ledger / Free Agency UI

**Design note:** User and CPU share the same FA wire and acceptance model — skill matters, not hidden info asymmetry.
