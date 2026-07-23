"""Temp smoke: start franchise, verify draft board / cap / FA / lines. Run: python backend/tests/_smoke_fullstack.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.franchise_sim import (  # noqa: E402
    start_franchise,
    build_draft_class_rankings,
    get_contract_office,
    save_franchise_lines,
    build_state_payload,
)

s = start_franchise(team_query="Toronto Maple Leafs", head_coach_name="Smoke", coach_archetype="balanced", seed=42)
sim = s.sim

print("=== DRAFT BOARD ===")
board = build_draft_class_rankings(s, sim)
entries = board["entries"]
print("class_strength:", board.get("class_strength"), "| total:", board.get("total"))
top10 = entries[:10]
goalies_top10 = [e for e in top10 if e["position"] == "G"]
print("goalies in top 10:", len(goalies_top10))
countries = {}
for e in entries[:100]:
    countries[e.get("nationality") or "?"] = countries.get(e.get("nationality") or "?", 0) + 1
print("top-100 nationalities:", dict(sorted(countries.items(), key=lambda kv: -kv[1])))
e1 = entries[0]
print("1st overall:", e1["name"], e1["position"], "tier", e1["scout_tier"], "ovr", e1["true_ovr"], "pot", e1["potential_score"], "nat", e1["nationality"])
assert e1["scout_tier"] in ("A+", "A", "A-"), "1st overall must be A-tier"
with_stats = [e for e in entries[:50] if e.get("gp")]
print("top-50 entries with stats:", len(with_stats))
sk = [e for e in entries[:50] if e["position"] != "G" and e.get("ppg")]
if sk:
    print("best skater ppg in top 50:", max(float(e["ppg"]) for e in sk))
g = [e for e in entries if e["position"] == "G" and e.get("save_pct")][:1]
if g:
    print("sample goalie:", g[0]["name"], "sv%", g[0]["save_pct"], "gaa", g[0].get("gaa"))
print("stock fields on #1:", {k: e1.get(k) for k in ("stock_delta", "stock_label", "stock_reason", "trend", "scouting_confidence", "risk")})

print("=== CONTRACT OFFICE ===")
office = get_contract_office(s)
cap = office["cap"]
print("upperLimit:", cap.get("upperLimit"), "totalCapHit:", cap.get("totalCapHit"), "usableCapSpace:", cap.get("usableCapSpace"))
assert cap.get("totalCapHit", 0) > 30, "payroll should be a real contract sum"
contracts = office["contracts"]
print("contracts:", len(contracts), "| top AAV:", contracts[0]["name"], contracts[0]["aav"], "x", contracts[0]["yearsRemaining"])
missing = [c for c in contracts if not c["aav"]]
print("contracts missing aav:", len(missing))
fas = office["freeAgents"]
print("free agents:", len(fas), "| best:", fas[0]["name"], fas[0]["overall"], "ask", fas[0]["askingAav"], "x", fas[0]["askingTerm"])
print("summary:", office["summary"])

print("=== LINES ===")
team = s.team_by_id[str(s.user_team_id)]
roster = [p for p in team.roster if not getattr(p, "retired", False)]
skaters = [p for p in roster if str(getattr(p.identity, "position", "")).split(".")[-1] != "G"]
pid = str(roster[0].id)
res = save_franchise_lines(s, {"unit_type": "even_strength", "lines": {"forwards": [{"id": "f1", "name": "Line 1", "slots": {"LW": pid, "C": "", "RW": ""}}]}})
print("save ok:", res["ok"], "warnings:", res["warnings"][:3])
state = build_state_payload(s)
print("state lines keys:", list(state.get("lines", {}).keys()))
print("state team cap:", state["team"]["salary_cap"], state["team"]["cap_hit"], state["team"]["cap_space"])
print("state has offseason extras key check (free_agents):", "free_agents" in state)
print("financials_status:", state.get("financials_status"))
print("SMOKE OK")
