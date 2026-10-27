"""Build the figures and results.json from the final cross-validation CSVs.

Reads results_final/results_cv_{summary,vs_paper,paired}_s{0,1,2}.csv
Writes results_final/figures/*.png and results_final/results.json
"""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results_final"
OUT = os.path.join(D, "figures")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
TUNED_PREFIX = "paper trust-decay, k's+tau"
FLAG = "cusum, fused + failure flag only"
OURS = ["cusum, face only (skip failures)", FLAG, "cusum, fused + quality (proposed)"]

summ = pd.concat([pd.read_csv(f"{D}/results_cv_summary_s{s}.csv").assign(seed=s) for s in SEEDS])
vs = pd.concat([pd.read_csv(f"{D}/results_cv_vs_paper_s{s}.csv").assign(seed=s) for s in SEEDS])
pair = pd.concat([pd.read_csv(f"{D}/results_cv_paired_s{s}.csv").assign(seed=s) for s in SEEDS])

tuned = [m for m in summ.method.unique() if m.startswith(TUNED_PREFIX)][0]
calib = "paper trust-decay, tau calibrated"
order = ["periodic k=2 (skip failures)", calib, tuned] + OURS
short = {
    "periodic k=2 (skip failures)": "Periodic face check (k=2)",
    calib: "Paper method, tau calibrated",
    tuned: "Paper method, tuned (baseline)",
    OURS[0]: "Ours: face only",
    OURS[1]: "Ours: + voice & failure flag",
    OURS[2]: "Ours: + quality (proposed)",
}
order = [m for m in order if m in set(summ.method)]

g = summ.groupby("method").agg(
    delay=("delay_s", "mean"), dmin=("delay_s", "min"), dmax=("delay_s", "max"),
    fa=("false_alarms_per_h", "mean"), fmin=("false_alarms_per_h", "min"), fmax=("false_alarms_per_h", "max"),
    missed=("missed", "mean")).loc[order]

def bar(ax, vals, lo, hi, labels, xlabel, color):
    err = [vals - lo, hi - vals]
    ax.barh(labels, vals, xerr=err, color=color, capsize=4)
    ax.invert_yaxis(); ax.set_xlabel(xlabel)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}" if v < 5 else f" {v:.0f}", va="center", fontsize=9)

labels = [short[m] for m in order]
colors = ["#9aa5b1" if not m.startswith("cusum") else "#2f6fdb" for m in order]

fig, ax = plt.subplots(figsize=(9, 4.5))
bar(ax, g.delay.values, g.dmin.values, g.dmax.values, labels, "Mean detection delay (seconds, lower is better)", colors)
ax.set_title("Time to detect a swapped candidate (mean of 3 seeds, bars = min-max)")
plt.tight_layout(); plt.savefig(f"{OUT}/1_detection_delay.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
bar(ax, g.fa.values, g.fmin.values, g.fmax.values, labels, "False alarms per hour (lower is better)", colors)
ax.set_title("False alarms per hour (mean of 3 seeds, bars = min-max)")
plt.tight_layout(); plt.savefig(f"{OUT}/3_false_alarms.png", dpi=200); plt.close()

b = vs[vs.baseline.str.startswith(TUNED_PREFIX)]
a = b.groupby("ours").pct_faster.agg(["mean", "min", "max"]).loc[OURS]
s0 = b[b.seed == 0].set_index("ours").loc[OURS]
fig, ax = plt.subplots(figsize=(9, 3.5))
bar(ax, a["mean"].values, a["min"].values, a["max"].values,
    [short[m] for m in OURS], "% faster than tuned paper baseline (bars = min-max over seeds)", "#2f6fdb")
ax.set_title("Where the improvement comes from")
plt.tight_layout(); plt.savefig(f"{OUT}/2_improvement_steps.png", dpi=200); plt.close()

q = pair[pair.method == FLAG].sort_values("seed")
res = {
    "baseline": tuned,
    "seeds": SEEDS,
    "n_people": int(b.n_people.iloc[0]),
    "improvement_vs_tuned_baseline_pct": {
        m: {"mean": round(a.loc[m, "mean"], 1), "min": round(a.loc[m, "min"], 1), "max": round(a.loc[m, "max"], 1),
            "seed0_ci": s0.loc[m, "ci"]} for m in OURS},
    "methods": {short[m]: {"delay_s": round(g.loc[m, "delay"], 1), "false_alarms_per_h": round(g.loc[m, "fa"], 3),
                            "missed_rate": round(g.loc[m, "missed"], 3)} for m in order},
    "quality_effect_seconds_per_seed": [round(x, 2) for x in q.mean_delay_minus_proposed_s],
    "note_as_published_false_alarms_per_h": round(summ[summ.method.str.contains("as published")].false_alarms_per_h.mean(), 1),
}
json.dump(res, open(f"{D}/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
