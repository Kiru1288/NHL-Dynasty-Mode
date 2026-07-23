"""Temp test runner (no pytest installed). Run: python backend/tests/_run_tests_noptest.py"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "backend"), str(ROOT / "SimEngine")):
    if p not in sys.path:
        sys.path.insert(0, p)

fails = 0
for fname in ("test_trade_foundation.py", "test_prospect_league_scoring.py"):
    spec = importlib.util.spec_from_file_location(fname[:-3], str(ROOT / "backend" / "tests" / fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name in dir(m):
        if name.startswith("test_"):
            try:
                getattr(m, name)()
                print("PASS", fname, name)
            except Exception as e:
                fails += 1
                print("FAIL", fname, name, repr(e))
print("total fails:", fails)
sys.exit(1 if fails else 0)
