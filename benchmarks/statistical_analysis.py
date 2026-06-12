"""
Statistical analysis + plots for the transformer AL benchmark.

Reads ``metrics.csv`` (or ``metrics_shard*.csv``) produced by ``transformer_benchmark.py`` and
produces the evidence a reviewer (or defense committee) expects and that the legacy harness
entirely lacks:

- Per ``(dataset, strategy, seed)`` Area Under the Learning Curve (AULC), x-axis normalized to
  fraction of budget so curves are comparable.
- ``alc_summary.csv``      : per (dataset, strategy) mean +/- std AULC and final-budget macro-F1,
                            plus AULC lift vs the random baseline.
- ``statistical_tests.csv``: Wilcoxon signed-rank, each strategy vs random, paired over
                            ``(dataset, seed)`` AULC, with Bonferroni-corrected p-values.
- ``bootstrap_ci.csv``     : 95% bootstrap CI of the mean AULC lift vs random per strategy.
- ``learning_curves.png``  : macro-F1 vs budget, mean over seeds with +/-1 std band, per dataset.

Usage:
    python benchmarks/statistical_analysis.py --input-dir runs/deadline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

RANDOM_BASELINE = "random"
_BOOTSTRAP_ITERS = 2000
_BOOTSTRAP_SEED = 12345


def _load_metrics(input_dir: Path) -> list[dict[str, Any]]:
    merged = input_dir / "metrics.csv"
    files = [merged] if merged.exists() else sorted(input_dir.glob("metrics_shard*.csv"))
    if not files:
        raise SystemExit(f"No metrics.csv or metrics_shard*.csv found in {input_dir}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for f in files:
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row["dataset"], row["strategy"], row["seed"], row["budget"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "strategy": row["strategy"],
                        "seed": int(row["seed"]),
                        "budget": int(row["budget"]),
                        "requested_budget": int(row.get("requested_budget") or row["budget"]),
                        "macro_f1": float(row["macro_f1"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )
    return rows


def _trapezoid_aulc(points: list[tuple[float, float]]) -> float:
    """Normalized AULC: x scaled to [0,1] over its observed range, trapezoidal integral."""
    pts = sorted(points)
    if len(pts) < 2:
        return pts[0][1] if pts else 0.0
    x0, x1 = pts[0][0], pts[-1][0]
    span = (x1 - x0) or 1.0
    area = 0.0
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        area += (xb - xa) * (ya + yb) / 2.0
    return area / span


def _aulc_by_curve(rows: list[dict[str, Any]], metric: str = "macro_f1") -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["strategy"], r["seed"])].append((float(r["requested_budget"]), float(r[metric])))
    return {key: _trapezoid_aulc(points) for key, points in grouped.items()}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _final_macro_f1(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], float]:
    best_budget: dict[tuple[str, str, int], int] = {}
    value: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r["dataset"], r["strategy"], r["seed"])
        if r["budget"] >= best_budget.get(key, -1):
            best_budget[key] = r["budget"]
            value[key] = r["macro_f1"]
    return value


def write_alc_summary(rows: list[dict[str, Any]], aulc: dict[tuple[str, str, int], float], out: Path) -> None:
    final = _final_macro_f1(rows)
    datasets = sorted({k[0] for k in aulc})
    strategies = sorted({k[1] for k in aulc})

    summary_rows: list[dict[str, Any]] = []
    for ds in datasets:
        random_aulc_by_seed = {seed: aulc[(ds, RANDOM_BASELINE, seed)]
                               for (d, s, seed) in aulc if d == ds and s == RANDOM_BASELINE}
        for strat in strategies:
            seeds = sorted(seed for (d, s, seed) in aulc if d == ds and s == strat)
            aulc_vals = [aulc[(ds, strat, seed)] for seed in seeds]
            final_vals = [final[(ds, strat, seed)] for seed in seeds if (ds, strat, seed) in final]
            lift_vals = [aulc[(ds, strat, seed)] - random_aulc_by_seed[seed]
                         for seed in seeds if seed in random_aulc_by_seed]
            a_mean, a_std = _mean_std(aulc_vals)
            f_mean, f_std = _mean_std(final_vals)
            l_mean, l_std = _mean_std(lift_vals)
            summary_rows.append(
                {
                    "dataset": ds,
                    "strategy": strat,
                    "n_seeds": len(seeds),
                    "aulc_macro_f1_mean": round(a_mean, 5),
                    "aulc_macro_f1_std": round(a_std, 5),
                    "final_macro_f1_mean": round(f_mean, 5),
                    "final_macro_f1_std": round(f_std, 5),
                    "aulc_lift_vs_random_mean": round(l_mean, 5),
                    "aulc_lift_vs_random_std": round(l_std, 5),
                }
            )
    _write_csv(out, summary_rows)
    print(f"[alc] wrote {out}")


def write_significance(aulc: dict[tuple[str, str, int], float], out: Path) -> None:
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        print("[significance] scipy not available; skipping Wilcoxon tests")
        return

    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    # Pair each strategy vs random over (dataset, seed).
    pairs_by_strategy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (ds, strat, seed), val in aulc.items():
        if strat == RANDOM_BASELINE:
            continue
        base_key = (ds, RANDOM_BASELINE, seed)
        if base_key in aulc:
            pairs_by_strategy[strat].append((val, aulc[base_key]))

    n_tests = max(1, len(strategies))
    test_rows: list[dict[str, Any]] = []
    for strat in strategies:
        pairs = pairs_by_strategy.get(strat, [])
        strat_vals = [p[0] for p in pairs]
        rand_vals = [p[1] for p in pairs]
        diffs = [a - b for a, b in pairs]
        mean_lift = sum(diffs) / len(diffs) if diffs else float("nan")
        p_value = float("nan")
        statistic = float("nan")
        note = ""
        nonzero = [d for d in diffs if d != 0.0]
        if len(nonzero) >= 6:
            try:
                res = wilcoxon(strat_vals, rand_vals, alternative="greater")
                statistic, p_value = float(res.statistic), float(res.pvalue)
            except Exception as exc:  # pragma: no cover
                note = f"wilcoxon_failed:{exc}"
        else:
            note = f"insufficient_pairs(n_nonzero={len(nonzero)}; need>=6)"
        test_rows.append(
            {
                "strategy": strat,
                "baseline": RANDOM_BASELINE,
                "n_pairs": len(pairs),
                "mean_aulc_lift": round(mean_lift, 5),
                "wilcoxon_statistic": statistic,
                "p_value_greater": p_value,
                "p_value_bonferroni": (min(1.0, p_value * n_tests) if not math.isnan(p_value) else float("nan")),
                "note": note,
            }
        )
    _write_csv(out, test_rows)
    print(f"[significance] wrote {out}")


def write_bootstrap_ci(aulc: dict[tuple[str, str, int], float], out: Path) -> None:
    try:
        import numpy as np  # type: ignore
    except Exception:
        print("[bootstrap] numpy not available; skipping")
        return

    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for strat in strategies:
        lifts = [
            val - aulc[(ds, RANDOM_BASELINE, seed)]
            for (ds, s, seed), val in aulc.items()
            if s == strat and (ds, RANDOM_BASELINE, seed) in aulc
        ]
        if not lifts:
            continue
        arr = np.asarray(lifts, dtype=float)
        means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(_BOOTSTRAP_ITERS)])
        rows.append(
            {
                "strategy": strat,
                "n": len(lifts),
                "mean_aulc_lift": round(float(arr.mean()), 5),
                "ci95_low": round(float(np.percentile(means, 2.5)), 5),
                "ci95_high": round(float(np.percentile(means, 97.5)), 5),
                "prob_positive": round(float((means > 0).mean()), 4),
            }
        )
    _write_csv(out, rows)
    print(f"[bootstrap] wrote {out}")


def plot_learning_curves(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        print(f"[plot] matplotlib/numpy unavailable; skipping ({exc})")
        return

    datasets = sorted({r["dataset"] for r in rows})
    strategies = sorted({r["strategy"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        for strat in strategies:
            by_budget: dict[int, list[float]] = defaultdict(list)
            for r in rows:
                if r["dataset"] == ds and r["strategy"] == strat:
                    by_budget[r["requested_budget"]].append(r["macro_f1"])
            if not by_budget:
                continue
            budgets = sorted(by_budget)
            means = np.array([np.mean(by_budget[b]) for b in budgets])
            stds = np.array([np.std(by_budget[b]) for b in budgets])
            style = dict(linewidth=2.4, color="black", linestyle="--") if strat == RANDOM_BASELINE else dict(linewidth=1.8)
            ax.plot(budgets, means, marker="o", label=strat, **style)
            ax.fill_between(budgets, means - stds, means + stds, alpha=0.12)
        ax.set_title(ds)
        ax.set_xlabel("labeled budget")
        ax.set_ylabel("macro-F1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Active-learning learning curves (mean ± std over seeds; DistilBERT)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot] wrote {out}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AL benchmark statistics + plots")
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    rows = _load_metrics(input_dir)
    aulc = _aulc_by_curve(rows)
    print(f"[load] {len(rows)} metric rows; {len(aulc)} curves")

    write_alc_summary(rows, aulc, input_dir / "alc_summary.csv")
    write_significance(aulc, input_dir / "statistical_tests.csv")
    write_bootstrap_ci(aulc, input_dir / "bootstrap_ci.csv")
    plot_learning_curves(rows, input_dir / "learning_curves.png")

    # Console headline.
    print("\n=== AULC lift vs random (mean over dataset x seed) ===")
    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    for strat in strategies:
        lifts = [val - aulc[(ds, RANDOM_BASELINE, seed)]
                 for (ds, s, seed), val in aulc.items()
                 if s == strat and (ds, RANDOM_BASELINE, seed) in aulc]
        if lifts:
            print(f"  {strat:28s} {sum(lifts) / len(lifts):+.4f}")
    summary = {"n_rows": len(rows), "n_curves": len(aulc)}
    (input_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
