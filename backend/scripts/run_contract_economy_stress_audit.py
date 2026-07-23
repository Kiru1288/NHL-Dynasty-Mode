#!/usr/bin/env python3
"""
Run contract economy stress audit (5-10 simulated offseason cycles).

Usage:
  python backend/scripts/run_contract_economy_stress_audit.py
  python backend/scripts/run_contract_economy_stress_audit.py --seasons 10 --seed 42
  python backend/scripts/run_contract_economy_stress_audit.py --json > audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SIM = ROOT / "SimEngine"
for p in (str(BACKEND), str(SIM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.contract_economy_audit import (  # noqa: E402
    format_audit_report,
    report_to_dict,
    run_stress_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract economy multi-season stress audit")
    parser.add_argument("--seasons", type=int, default=8, help="Offseason cycles to simulate (default 8)")
    parser.add_argument("--seed", type=int, default=424242, help="RNG seed")
    parser.add_argument("--full", action="store_true", help="Use full start_franchise (slow)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text report")
    args = parser.parse_args()

    print(f"Running contract economy stress audit ({args.seasons} seasons, seed={args.seed})...", file=sys.stderr)
    report = run_stress_audit(seasons=args.seasons, seed=args.seed, full_franchise=args.full)

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_audit_report(report))

    return 1 if report.summary.get("errors", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
