"""Combine several evaluate_cv.py runs (different --seed/--tag) into one table: how stable are the % improvements?

Usage:  python summarize_seeds.py            (reads results_cv_vs_paper_s*.csv and results_cv_summary_s*.csv)
"""
import glob
import pandas as pd

files = sorted(glob.glob("results_cv_vs_paper_s*.csv"))
if not files:
    raise SystemExit("no results_cv_vs_paper_s*.csv files found; run evaluate_cv.py with --seed N --tag sN first")
df = pd.concat([pd.read_csv(f).assign(run=f) for f in files])
tuned = "paper trust-decay, k's+tau tuned on calibration people"
d = df[df.baseline == tuned]
g = d.groupby("ours").pct_faster.agg(["mean", "min", "max", "count"]).round(1)
print(f"{len(files)} runs. % faster than the TUNED reimplementation of the paper's trust-decay:\n")
print(g.to_string())
fa = pd.concat([pd.read_csv(f).assign(run=f) for f in sorted(glob.glob("results_cv_summary_s*.csv"))])
print("\nFalse alarms/hour per method, mean over runs (compare like with like):")
print(fa.groupby("method", sort=False).false_alarms_per_h.mean().round(3).to_string())
