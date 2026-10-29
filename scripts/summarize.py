"""Turn lerobot-eval eval_info.json files into results/*.json + results/summary.csv.

Usage: python scripts/summarize.py <name>=<logs_dir> [<name>=<logs_dir> ...]
Timing files (results/timing_<name>.json) are merged in when present.
"""
import csv, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def parse(info: dict) -> dict:
    per_task = {}
    for t in info["per_task"]:
        s = t["metrics"]["successes"]
        per_task[str(t["task_id"])] = {"successes": int(sum(s)), "episodes": len(s)}
    k = sum(v["successes"] for v in per_task.values())
    n = sum(v["episodes"] for v in per_task.values())
    lo, hi = wilson(k, n)
    return {"per_task": per_task, "successes": k, "episodes": n,
            "success_rate": k / n if n else None, "ci95": [lo, hi]}


def main(args):
    rows = []
    for a in args:
        name, d = a.split("=", 1)
        res = parse(json.load(open(Path(d) / "eval_info.json")))
        res["name"] = name
        tf = ROOT / "results" / f"timing_{name}.json"
        if tf.exists():
            t = json.load(open(tf))
            res["chunk_latency_s"] = {k: t[k] for k in ("n_calls", "mean_s", "median_s", "p95_s")}
        json.dump(res, open(ROOT / "results" / f"{name}.json", "w"), indent=1)
        lat = res.get("chunk_latency_s", {})
        rows.append([name, res["successes"], res["episodes"], f"{100*res['success_rate']:.1f}",
                     f"{100*res['ci95'][0]:.1f}", f"{100*res['ci95'][1]:.1f}",
                     lat.get("n_calls", ""), f"{lat['median_s']:.3f}" if lat else ""])
    with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "successes", "episodes", "success_pct", "ci95_lo", "ci95_hi",
                    "chunk_calls", "chunk_latency_median_s"])
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
