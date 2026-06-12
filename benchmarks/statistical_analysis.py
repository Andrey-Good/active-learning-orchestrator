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
                # Include protocol in the dedup key so cold and warm rows never collide.
                protocol = row.get("protocol") or "cold"
                key = (row["dataset"], row["strategy"], row["seed"], protocol, row["budget"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "strategy": row["strategy"],
                        "seed": int(row["seed"]),
                        "protocol": protocol,
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


def _aulc_by_curve(
    rows: list[dict[str, Any]], metric: str = "macro_f1"
) -> dict[tuple[str, str, str, int], float]:
    """Return AULC keyed by (dataset, strategy, protocol, seed)."""
    grouped: dict[tuple[str, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["strategy"], r.get("protocol", "cold"), r["seed"])].append(
            (float(r["requested_budget"]), float(r[metric]))
        )
    return {key: _trapezoid_aulc(points) for key, points in grouped.items()}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _final_macro_f1(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], float]:
    """Return final macro-F1 keyed by (dataset, strategy, protocol, seed)."""
    best_budget: dict[tuple[str, str, str, int], int] = {}
    value: dict[tuple[str, str, str, int], float] = {}
    for r in rows:
        key = (r["dataset"], r["strategy"], r.get("protocol", "cold"), r["seed"])
        if r["budget"] >= best_budget.get(key, -1):
            best_budget[key] = r["budget"]
            value[key] = r["macro_f1"]
    return value


def _label_efficiency(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.80, 0.90),
) -> dict[tuple[str, str, str], dict[float, int | None]]:
    """Per (dataset, strategy, protocol): smallest budget at which mean macro-F1 over seeds
    reaches *threshold* × max observed mean macro-F1 across all strategies for that dataset/protocol.

    Returns -1 (sentinel) when the threshold is never reached within the observed budget range.
    """
    # Build mean macro-F1 per (dataset, strategy, protocol, requested_budget).
    from collections import defaultdict as _dd

    # Accumulate per-seed values first.
    by_key: dict[tuple[str, str, str, int], list[float]] = _dd(list)
    for r in rows:
        by_key[(r["dataset"], r["strategy"], r.get("protocol", "cold"), r["requested_budget"])].append(
            r["macro_f1"]
        )
    mean_f1: dict[tuple[str, str, str, int], float] = {
        k: sum(v) / len(v) for k, v in by_key.items()
    }

    # Max mean macro-F1 across all strategies per (dataset, protocol).
    max_by_dataset: dict[tuple[str, str], float] = {}
    for (ds, _strat, proto, _bgt), val in mean_f1.items():
        key = (ds, proto)
        max_by_dataset[key] = max(max_by_dataset.get(key, 0.0), val)

    # For each (dataset, strategy, protocol), find the smallest budget reaching the threshold.
    result: dict[tuple[str, str, str], dict[float, int | None]] = {}
    combos = {(ds, st, pr) for (ds, st, pr, _) in mean_f1}
    for ds, strat, proto in combos:
        dataset_max = max_by_dataset.get((ds, proto), float("nan"))
        budgets = sorted(b for (d, s, p, b) in mean_f1 if d == ds and s == strat and p == proto)
        entry: dict[float, int | None] = {}
        for thr in thresholds:
            target = thr * dataset_max if not math.isnan(dataset_max) else float("nan")
            found: int | None = None
            for b in budgets:
                if mean_f1.get((ds, strat, proto, b), 0.0) >= target:
                    found = b
                    break
            entry[thr] = found  # None → sentinel for "never reached"
        result[(ds, strat, proto)] = entry
    return result


def write_alc_summary(
    rows: list[dict[str, Any]],
    aulc: dict[tuple[str, str, str, int], float],
    out: Path,
) -> None:
    final = _final_macro_f1(rows)
    datasets = sorted({k[0] for k in aulc})
    strategies = sorted({k[1] for k in aulc})
    protocols = sorted({k[2] for k in aulc})

    label_eff = _label_efficiency(rows)

    summary_rows: list[dict[str, Any]] = []
    for ds in datasets:
        for proto in protocols:
            random_aulc_by_seed = {
                seed: aulc[(ds, RANDOM_BASELINE, proto, seed)]
                for (d, s, p, seed) in aulc
                if d == ds and s == RANDOM_BASELINE and p == proto
            }
            # Random baseline label-efficiency (for savings calculation).
            rand_le = label_eff.get((ds, RANDOM_BASELINE, proto), {})
            for strat in strategies:
                seeds = sorted(seed for (d, s, p, seed) in aulc if d == ds and s == strat and p == proto)
                if not seeds:
                    continue
                aulc_vals = [aulc[(ds, strat, proto, seed)] for seed in seeds]
                final_vals = [
                    final[(ds, strat, proto, seed)]
                    for seed in seeds
                    if (ds, strat, proto, seed) in final
                ]
                lift_vals = [
                    aulc[(ds, strat, proto, seed)] - random_aulc_by_seed[seed]
                    for seed in seeds
                    if seed in random_aulc_by_seed
                ]
                a_mean, a_std = _mean_std(aulc_vals)
                f_mean, f_std = _mean_std(final_vals)
                l_mean, l_std = _mean_std(lift_vals)
                strat_le = label_eff.get((ds, strat, proto), {})
                # Label-efficiency at 80% and 90% thresholds.
                le_80 = strat_le.get(0.80)
                le_90 = strat_le.get(0.90)
                rand_le_80 = rand_le.get(0.80)
                rand_le_90 = rand_le.get(0.90)
                savings_80: int | str = (
                    (rand_le_80 - le_80)
                    if (le_80 is not None and rand_le_80 is not None)
                    else ""
                )
                savings_90: int | str = (
                    (rand_le_90 - le_90)
                    if (le_90 is not None and rand_le_90 is not None)
                    else ""
                )
                summary_rows.append(
                    {
                        "dataset": ds,
                        "strategy": strat,
                        "protocol": proto,
                        "n_seeds": len(seeds),
                        "aulc_macro_f1_mean": round(a_mean, 5),
                        "aulc_macro_f1_std": round(a_std, 5),
                        "final_macro_f1_mean": round(f_mean, 5),
                        "final_macro_f1_std": round(f_std, 5),
                        "aulc_lift_vs_random_mean": round(l_mean, 5),
                        "aulc_lift_vs_random_std": round(l_std, 5),
                        "label_eff_budget_80pct": le_80 if le_80 is not None else -1,
                        "label_eff_budget_90pct": le_90 if le_90 is not None else -1,
                        "label_eff_savings_vs_random_80pct": savings_80,
                        "label_eff_savings_vs_random_90pct": savings_90,
                    }
                )
    _write_csv(out, summary_rows)
    print(f"[alc] wrote {out}")


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction and return adjusted p-values.

    BH(k) = min(1, p_(k) * n / k) with monotonicity enforced (backwards cumulative min).
    NaN inputs are passed through unchanged.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Build (original_index, p_value) pairs, keeping NaN aside.
    indexed = [(i, p) for i, p in enumerate(p_values) if not math.isnan(p)]
    nan_indices = {i for i, p in enumerate(p_values) if math.isnan(p)}
    indexed_sorted = sorted(indexed, key=lambda x: x[1])  # ascending p

    adjusted = [float("nan")] * n
    # Compute raw BH threshold: p_(k) * n / k  (k = 1-based rank).
    raw_adj = [p * n / (rank + 1) for rank, (_, p) in enumerate(indexed_sorted)]
    # Enforce monotonicity: backwards cumulative minimum.
    for i in range(len(raw_adj) - 2, -1, -1):
        raw_adj[i] = min(raw_adj[i], raw_adj[i + 1])
    # Cap at 1.0 and place back in original order.
    for rank, (orig_idx, _) in enumerate(indexed_sorted):
        adjusted[orig_idx] = min(1.0, raw_adj[rank])
    return adjusted


def _cohens_d_paired(diffs: list[float]) -> float:
    """Cohen's d for paired data: mean(diff) / std(diff, ddof=1).  Returns 0 if std is 0."""
    if len(diffs) < 2:
        return float("nan")
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return mean / std


def write_significance(
    aulc: dict[tuple[str, str, str, int], float],
    out: Path,
) -> None:
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        print("[significance] scipy not available; skipping Wilcoxon tests")
        return

    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    protocols = sorted({k[2] for k in aulc})

    # Pair each (strategy, protocol) vs random over (dataset, seed).
    pairs_by_strat_proto: dict[tuple[str, str], list[tuple[float, float, str, int]]] = defaultdict(list)
    for (ds, strat, proto, seed), val in aulc.items():
        if strat == RANDOM_BASELINE:
            continue
        base_key = (ds, RANDOM_BASELINE, proto, seed)
        if base_key in aulc:
            pairs_by_strat_proto[(strat, proto)].append((val, aulc[base_key], ds, seed))

    # Collect raw p-values for BH correction across all (strategy, protocol) combos.
    combo_order = [(strat, proto) for strat in strategies for proto in protocols
                   if (strat, proto) in pairs_by_strat_proto]
    raw_p_values: list[float] = []
    combo_data: list[dict[str, Any]] = []

    for strat, proto in combo_order:
        entries = pairs_by_strat_proto[(strat, proto)]
        pairs = [(v, r) for v, r, _ds, _seed in entries]
        strat_vals = [p[0] for p in pairs]
        rand_vals = [p[1] for p in pairs]
        diffs = [a - b for a, b in pairs]
        mean_lift = sum(diffs) / len(diffs) if diffs else float("nan")
        cohens_d = _cohens_d_paired(diffs)
        pct_beat = sum(1 for d in diffs if d > 0) / len(diffs) if diffs else float("nan")
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
        raw_p_values.append(p_value)
        combo_data.append(
            {
                "strategy": strat,
                "protocol": proto,
                "baseline": RANDOM_BASELINE,
                "n_pairs": len(pairs),
                "mean_aulc_lift": round(mean_lift, 5),
                "wilcoxon_statistic": statistic,
                "p_value_greater": p_value,
                # BH will be filled in below after all raw p-values are collected.
                "p_value_bh": float("nan"),
                "cohens_d": round(cohens_d, 5) if not math.isnan(cohens_d) else float("nan"),
                "pct_seeds_beat_random": round(pct_beat, 4) if not math.isnan(pct_beat) else float("nan"),
                "note": note,
            }
        )

    # Apply Benjamini-Hochberg FDR correction.
    bh_p = _benjamini_hochberg(raw_p_values)
    for entry, p_bh in zip(combo_data, bh_p):
        entry["p_value_bh"] = p_bh

    _write_csv(out, combo_data)
    print(f"[significance] wrote {out}")


def write_significance_by_dataset(
    aulc: dict[tuple[str, str, str, int], float],
    out: Path,
) -> None:
    """Per-dataset Wilcoxon signed-rank test, paired over seeds only.

    For each (dataset, strategy, protocol) group, tests whether that strategy's AULC is
    significantly greater than random's AULC, with pairs formed over seeds (not dataset×seed).
    BH-FDR correction is applied across strategies within each (dataset, protocol) group.

    This avoids the dilution effect of pooling across datasets where a strategy helps on one
    dataset (e.g. ag_news_imb) but is neutral on others — per-dataset Wilcoxon on ag_news_imb
    is expected to be strongly significant for class_group_balanced_entropy (p≈0.18 pooled).
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        print("[significance_by_dataset] scipy not available; skipping")
        return

    datasets = sorted({k[0] for k in aulc})
    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    protocols = sorted({k[2] for k in aulc})

    all_rows: list[dict[str, Any]] = []

    for ds in datasets:
        for proto in protocols:
            # Collect seeds present for random baseline in this (dataset, protocol).
            random_by_seed = {
                seed: val
                for (d, s, p, seed), val in aulc.items()
                if d == ds and s == RANDOM_BASELINE and p == proto
            }
            if not random_by_seed:
                continue

            # Gather raw p-values for BH correction within this (dataset, protocol).
            combo_order: list[str] = []
            raw_p: list[float] = []
            ds_combo_data: list[dict[str, Any]] = []

            for strat in strategies:
                strat_by_seed = {
                    seed: val
                    for (d, s, p, seed), val in aulc.items()
                    if d == ds and s == strat and p == proto
                }
                shared_seeds = sorted(set(strat_by_seed) & set(random_by_seed))
                if not shared_seeds:
                    continue

                strat_vals = [strat_by_seed[seed] for seed in shared_seeds]
                rand_vals = [random_by_seed[seed] for seed in shared_seeds]
                diffs = [a - b for a, b in zip(strat_vals, rand_vals)]
                mean_lift = sum(diffs) / len(diffs) if diffs else float("nan")
                cohens_d = _cohens_d_paired(diffs)
                pct_beat = sum(1 for d in diffs if d > 0) / len(diffs) if diffs else float("nan")

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
                    note = f"insufficient_pairs(n_nonzero={len(nonzero)};need>=6)"

                combo_order.append(strat)
                raw_p.append(p_value)
                ds_combo_data.append(
                    {
                        "dataset": ds,
                        "strategy": strat,
                        "protocol": proto,
                        "baseline": RANDOM_BASELINE,
                        "n_pairs": len(shared_seeds),
                        "mean_aulc_lift": round(mean_lift, 5),
                        "wilcoxon_statistic": statistic,
                        "p_value_greater": p_value,
                        "p_value_bh": float("nan"),  # filled below
                        "cohens_d": round(cohens_d, 5) if not math.isnan(cohens_d) else float("nan"),
                        "pct_seeds_beat_random": round(pct_beat, 4) if not math.isnan(pct_beat) else float("nan"),
                        "note": note,
                    }
                )

            # Apply BH correction within this (dataset, protocol) group.
            bh_p = _benjamini_hochberg(raw_p)
            for entry, p_bh in zip(ds_combo_data, bh_p):
                entry["p_value_bh"] = p_bh
            all_rows.extend(ds_combo_data)

    _write_csv(out, all_rows)
    print(f"[significance_by_dataset] wrote {out}")


def write_bootstrap_ci(
    aulc: dict[tuple[str, str, str, int], float],
    out: Path,
) -> None:
    try:
        import numpy as np  # type: ignore
    except Exception:
        print("[bootstrap] numpy not available; skipping")
        return

    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    protocols = sorted({k[2] for k in aulc})
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for strat in strategies:
        for proto in protocols:
            lifts = [
                val - aulc[(ds, RANDOM_BASELINE, proto, seed)]
                for (ds, s, p, seed), val in aulc.items()
                if s == strat and p == proto and (ds, RANDOM_BASELINE, proto, seed) in aulc
            ]
            if not lifts:
                continue
            arr = np.asarray(lifts, dtype=float)
            means = np.array(
                [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(_BOOTSTRAP_ITERS)]
            )
            rows.append(
                {
                    "strategy": strat,
                    "protocol": proto,
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
    protocols = sorted({r.get("protocol", "cold") for r in rows})

    # One column per dataset, one row per protocol.
    n_rows = len(protocols)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    for row_idx, proto in enumerate(protocols):
        for col, ds in enumerate(datasets):
            ax = axes[row_idx][col]
            for strat in strategies:
                by_budget: dict[int, list[float]] = defaultdict(list)
                for r in rows:
                    if (
                        r["dataset"] == ds
                        and r["strategy"] == strat
                        and r.get("protocol", "cold") == proto
                    ):
                        by_budget[r["requested_budget"]].append(r["macro_f1"])
                if not by_budget:
                    continue
                budgets = sorted(by_budget)
                means = np.array([np.mean(by_budget[b]) for b in budgets])
                stds = np.array([np.std(by_budget[b]) for b in budgets])
                style = (
                    dict(linewidth=2.4, color="black", linestyle="--")
                    if strat == RANDOM_BASELINE
                    else dict(linewidth=1.8)
                )
                ax.plot(budgets, means, marker="o", label=strat, **style)
                ax.fill_between(budgets, means - stds, means + stds, alpha=0.12)
            ax.set_title(f"{ds} [{proto}]")
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
    write_significance_by_dataset(aulc, input_dir / "statistical_tests_by_dataset.csv")
    write_bootstrap_ci(aulc, input_dir / "bootstrap_ci.csv")
    plot_learning_curves(rows, input_dir / "learning_curves.png")

    # Console headline.
    stat_path = input_dir / "statistical_tests.csv"
    # Load BH p-values and Cohen's d from the just-written CSV (if it exists).
    bh_lookup: dict[tuple[str, str], float] = {}
    cd_lookup: dict[tuple[str, str], float] = {}
    if stat_path.exists():
        with stat_path.open(newline="", encoding="utf-8") as _fh:
            for _r in csv.DictReader(_fh):
                _key = (_r.get("strategy", ""), _r.get("protocol", "cold"))
                try:
                    bh_lookup[_key] = float(_r.get("p_value_bh", "nan") or "nan")
                    cd_lookup[_key] = float(_r.get("cohens_d", "nan") or "nan")
                except ValueError:
                    pass

    print("\n=== AULC lift vs random (mean over dataset x seed) ===")
    strategies = sorted({k[1] for k in aulc if k[1] != RANDOM_BASELINE})
    protocols_headline = sorted({k[2] for k in aulc})
    for proto in protocols_headline:
        print(f"  -- protocol: {proto} --")
        for strat in strategies:
            lifts = [
                val - aulc[(ds, RANDOM_BASELINE, proto, seed)]
                for (ds, s, p, seed), val in aulc.items()
                if s == strat and p == proto and (ds, RANDOM_BASELINE, proto, seed) in aulc
            ]
            if lifts:
                mean_lift = sum(lifts) / len(lifts)
                p_bh = bh_lookup.get((strat, proto), float("nan"))
                d = cd_lookup.get((strat, proto), float("nan"))
                p_bh_str = f"{p_bh:.4f}" if not math.isnan(p_bh) else "n/a"
                d_str = f"{d:+.3f}" if not math.isnan(d) else "n/a"
                print(f"    {strat:34s} lift={mean_lift:+.4f}  BH_p={p_bh_str}  d={d_str}")
    summary = {"n_rows": len(rows), "n_curves": len(aulc)}
    (input_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
