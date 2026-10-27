"""End-to-end SYNTHETIC evaluation: detection delay at a matched false-alarm rate.

People are split into train / calibration / test (never shared). Every method's threshold is
calibrated on held-out genuine calibration streams to the SAME false alarms per hour, then
tested on swap sessions from unseen test people.

Run:  python evaluate.py            (about a minute)
"""
import argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulator as sim
import detectors as D
from fusion import Fusion


def build_training(people, rng, n=3000):
    G, I = [], []
    for p in people:
        G.append(sim.genuine_stream(rng, n, p))
        I.append(sim.impostor_stream(rng, n // 2, p, "person"))
        I.append(sim.impostor_stream(rng, n // 4, p, "replay"))
    return np.vstack(G), np.vstack(I)


def first_detection(alarms, t0, n):
    after = [a for a in alarms if a >= t0]
    return (after[0] - t0 + 1) if after else None


def bootstrap_ci(vals, rng, B=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    means = [rng.choice(vals, len(vals)).mean() for _ in range(B)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def main(a):
    rng = np.random.default_rng(a.seed)
    train_p = sim.make_people(a.seed + 1, a.n_train)
    cal_p = sim.make_people(a.seed + 2, a.n_cal)
    test_p = sim.make_people(a.seed + 3, a.n_test)

    # --- fusion models (the only trained part) + ablations
    G, I = build_training(train_p, rng)
    fusers = {name: Fusion(name).fit(G, I) for name in ["full", "no_quality", "face_only"]}

    # --- calibration stream: long genuine data from calibration people only
    cal_streams = [sim.genuine_stream(rng, a.cal_len, p) for p in cal_p]
    cal_X = np.vstack(cal_streams)  # concatenated across people (resets between people ignored)

    methods = {
        "login_only (existing)": dict(prep=lambda X: X[:, 0], fn=D.login_only, rng=(0, 1), up=True),
        "periodic_threshold":    dict(prep=lambda X: X[:, 0], fn=D.periodic_threshold, rng=(0.0, 0.9), up=True),
        "trust_decay (face-only)": dict(prep=lambda X: X[:, 0], fn=D.trust_decay, rng=(0.0, 0.9), up=True),
    }
    for name, label in [("face_only", "cusum, face only"), ("no_quality", "cusum, fused no quality"),
                        ("full", "cusum, fused + quality (proposed)")]:
        f = fusers[name]
        methods[label] = dict(prep=(lambda X, f=f: f.llr(X)), fn=D.cusum, rng=(0.5, 200.0), up=False)

    # --- calibrate every method to the same false-alarm rate
    params = {}
    for m, c in methods.items():
        if m.startswith("login_only"):
            params[m] = None
            continue
        x = c["prep"](cal_X)
        params[m] = D.calibrate(c["fn"], x, a.target_fa_per_hour, c["rng"][0], c["rng"][1], c["up"])

    # --- achieved false alarms on FRESH genuine data from test people (sanity check)
    fa_rows = {}
    test_gen = np.vstack([sim.genuine_stream(rng, a.cal_len, p) for p in test_p])
    for m, c in methods.items():
        fa_rows[m] = 0.0 if params[m] is None else D.false_alarms_per_hour(c["fn"], c["prep"](test_gen), params[m])

    # --- swap evaluation on test people
    rows = []
    for kind in ["person", "replay"]:
        per_method = {m: [] for m in methods}      # per-person mean delay (detected only)
        missed = {m: [] for m in methods}
        for p in test_p:
            sessions = [sim.swap_session(rng, p, kind=kind) for _ in range(a.sessions_per_person)]
            for m, c in methods.items():
                delays = []
                for X, t0 in sessions:
                    al = [] if params[m] is None else c["fn"](c["prep"](X), params[m])
                    d = first_detection(al, t0, len(X))
                    delays.append(d)
                det = [d for d in delays if d is not None]
                missed[m].append(1 - len(det) / len(delays))
                per_method[m].append(np.mean(det) if det else np.nan)
        for m in methods:
            v = np.array(per_method[m]); ok = v[~np.isnan(v)]
            lo, hi = bootstrap_ci(ok, rng)
            rows.append(dict(scenario=kind, method=m,
                             mean_delay_checks=np.nan if len(ok) == 0 else ok.mean(),
                             ci_low=lo, ci_high=hi,
                             mean_delay_seconds=np.nan if len(ok) == 0 else ok.mean() * sim.CHECK_SECONDS,
                             missed_swap_rate=float(np.mean(missed[m])),
                             false_alarms_per_hour_test=fa_rows[m]))
    df = pd.DataFrame(rows)
    df.to_csv("results_synthetic.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"\nTarget false alarms/hour = {a.target_fa_per_hour}  |  people: train {a.n_train}, cal {a.n_cal}, test {a.n_test}")
    print("*** SYNTHETIC DATA: demonstrates machinery, not findings ***\n")
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, kind in zip(axes, ["person", "replay"]):
        d = df[df.scenario == kind]
        ax.barh(d.method, d.mean_delay_seconds.fillna(0),
                xerr=[(d.mean_delay_seconds - d.ci_low * sim.CHECK_SECONDS).fillna(0),
                      (d.ci_high * sim.CHECK_SECONDS - d.mean_delay_seconds).fillna(0)], color="#4C72B0")
        for i, (v, miss) in enumerate(zip(d.mean_delay_seconds, d.missed_swap_rate)):
            ax.text(0, i, f"  missed {miss:.0%}", va="center", fontsize=8, color="black")
        ax.set_title(f"swap type: {kind}"); ax.set_xlabel("mean detection delay (s), detected swaps only")
    fig.suptitle("SYNTHETIC: delay at matched false-alarm rate", fontsize=10)
    plt.tight_layout(); plt.savefig("delay_comparison_synthetic.png", dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_cal", type=int, default=60)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--cal_len", type=int, default=6000, help="genuine checks per calibration person")
    ap.add_argument("--sessions_per_person", type=int, default=10)
    ap.add_argument("--target_fa_per_hour", type=float, default=0.1, help="0.1 = one false alarm per 10 genuine hours")
    main(ap.parse_args())
