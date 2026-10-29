import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from summarize import parse, wilson


def test_wilson_known_values():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi <= 1.0 and lo > 0.70


def test_parse_counts():
    info = {"per_task": [{"task_id": 0, "metrics": {"successes": [True, False, True]}},
                         {"task_id": 1, "metrics": {"successes": [False, False, True]}}]}
    r = parse(info)
    assert (r["successes"], r["episodes"]) == (3, 6)
    assert r["per_task"]["1"]["successes"] == 1
