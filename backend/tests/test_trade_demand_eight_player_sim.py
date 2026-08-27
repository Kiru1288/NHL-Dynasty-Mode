"""Integration test: 8-player trade demand simulation across team contexts."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_sim_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_trade_demand_simulation.py"
    spec = importlib.util.spec_from_file_location("run_trade_demand_simulation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TradeDemandEightPlayerSimulation(unittest.TestCase):
    def test_eight_player_scenarios(self):
        mod = _load_sim_module()
        session, specs, _ = mod.build_scenarios()

        results = []
        for sp in specs:
            player = sp["_player"]
            team = sp["_team"]
            row = mod.update_player_stability(session, player, team)
            results.append(
                {
                    "pid": sp["pid"],
                    "stability": float(row["trade_stability_score"]),
                    "escalation": int(row["escalation_level"]),
                    "character": sp["character"],
                    "mental": sp["mental"],
                }
            )

        by_id = {r["pid"]: r for r in results}

        # Winning + high character/mental tolerate poor role (Kucherov)
        self.assertGreaterEqual(by_id["p1_kucherov"]["stability"], 70)
        self.assertEqual(by_id["p1_kucherov"]["escalation"], 0)

        # Happy star on winner stays stable (McDavid)
        self.assertGreaterEqual(by_id["p5_mcdavid"]["stability"], 70)
        self.assertEqual(by_id["p5_mcdavid"]["escalation"], 0)

        # Rebuilders should not hit full crisis on day one
        self.assertLess(by_id["p2_bedard"]["escalation"], 4)
        self.assertGreater(by_id["p2_bedard"]["stability"], 25)
        self.assertLess(by_id["p6_keller"]["escalation"], 4)

        # Daily drift erodes frustrated rebuilders from neutral baseline
        session.trade_stability_state.pop("p2_bedard", None)
        for day in range(42, 70):
            mod.apply_daily_stability_update(session, specs[1]["_player"], specs[1]["_team"], day)
        bedard_after = session.trade_stability_state.get("p2_bedard", {})
        bedard_score = float(bedard_after.get("trade_stability_score") or 100)
        self.assertLess(bedard_score, 88)
        self.assertLess(bedard_score, by_id["p1_kucherov"]["stability"])

        # Miller: questionable character on losing team — escalates with drift
        session.trade_stability_state.pop("p8_miller", None)
        for day in range(42, 70):
            mod.apply_daily_stability_update(session, specs[7]["_player"], specs[7]["_team"], day)
        miller_after = session.trade_stability_state.get("p8_miller", {})
        miller_score = float(miller_after.get("trade_stability_score") or 100)
        self.assertLess(miller_score, 88)
        self.assertLess(miller_score, by_id["p5_mcdavid"]["stability"])

        # Repeated trade exposure erodes shopped stars (not instant formal demand)
        matthews_before = float(by_id["p3_matthews"]["stability"])
        for attempt in (1, 2, 3):
            mod.apply_trade_hub_exposure(
                session,
                specs[2]["_player"],
                attempt_n=attempt,
                rejection_kind="rejected",
            )
        row = mod.update_player_stability(session, specs[2]["_player"], specs[2]["_team"])
        self.assertLess(float(row["trade_stability_score"]), matthews_before)

    def test_crisis_timer_and_agent_variation(self):
        mod = _load_sim_module()
        session, specs, _ = mod.build_scenarios()
        rng = session.sim.rng

        keller = next(s for s in specs if s["pid"] == "p6_keller")
        matthews = next(s for s in specs if s["pid"] == "p3_matthews")

        for sp in (keller, matthews):
            row = mod.update_player_stability(session, sp["_player"], sp["_team"])
            if int(row.get("escalation_level") or 0) < 3:
                row["escalation_level"] = 3
                row["trade_stability_score"] = 30

        keller_demand = mod.open_trade_demand(
            session, keller["_player"], keller["_team"],
            reason="role", calendar_idx=55, rng=rng, force_formal=True,
        )
        matthews_demand = mod.open_trade_demand(
            session, matthews["_player"], matthews["_team"],
            reason="trade_exposure", calendar_idx=55, rng=rng, force_formal=True,
        )

        keller_timer = int(keller_demand["crisis"]["initial_seconds"])
        matthews_timer = int(matthews_demand["crisis"]["initial_seconds"])

        # Blake + low character → shorter fuse than Kim + decent character
        self.assertLessEqual(keller_timer, 150)
        self.assertGreater(matthews_timer, 240)

        # Low-character pool assignment is deterministic (Keller char 55)
        mod.ensure_player_agent(keller["_player"], session)
        agent_name = keller_demand["agent"]["name"]
        self.assertIn(
            agent_name,
            {"Jordan Blake", "Allan Carter", "Daniel Kim", "Marco Rossi", "Patricia Walsh"},
        )
        self.assertIn(agent_name, {"Jordan Blake", "Allan Carter", "Daniel Kim", "Marco Rossi"})


if __name__ == "__main__":
    unittest.main()
