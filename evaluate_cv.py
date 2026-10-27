"""Person-disjoint K-fold evaluation with bootstrap confidence intervals OVER PEOPLE.

Why: evaluate_real.py repeats random splits of the same 91 actors, so its "10/10 splits" are not independent.
Here every actor is a TEST person exactly once. In each fold the remaining people are split into train
(fit fusion) and calibration (set every detector's threshold to the same false alarms/hour), then the test
fold is evaluated. Per-person results are pooled across folds and resampled over people to give 95% CIs.

Run:  python -W ignore evaluate_cv.py --folds 5 --cache ~/swapdetect_data/features_cache_heavy \
          --target_fa_per_hour 0.1 --p_deg 0.5
Same caveats as evaluate_real.py (simulated degradation/voice-silence, acted studio data, no attacks).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import detectors as D
import evaluate_real as R
from evaluate import first_detection
from simulator import CHECK_SECONDS, CHECKS_PER_HOUR

PROPOSED = "cusum, fused + quality (proposed)"


GRID_KW = [5, 10, 20, 45, 90]       # k_weight candidates (s); paper uses 45
GRID_KD = [60, 120, 300, 600, 1200, 2400, 6000]   # k_decay candidates (s); paper uses 60
TUNED = "paper trust-decay, k's+tau tuned on calibration people"


def tune_paper(cal, rk_cal, cal_X, rng, a):
    """Give the paper's method its best shot: pick (k_weight, k_decay) and calibrate tau using ONLY calibration people."""
    sessions = []
    for P in cal:
        for _ in range(a.tune_sessions):
            Q = R.pick_impostor(rng, P, cal, rk_cal, a.hard_frac)
            t0 = int(rng.integers(30, 120))
            g, i = R.make_stream(rng, P, P, 180, a.p_deg), R.make_stream(rng, Q, P, 180, a.p_deg)
            X = g.copy(); X[t0:] = i[t0:]
            sessions.append((X[:, 4], t0))
    best = None
    for kw in GRID_KW:
        for kd in GRID_KD:
            fn = lambda x, t, kw=kw, kd=kd: D.trust_authenticator(x, t, kw, kd)
            tau = D.calibrate(fn, cal_X[:, 4], a.target_fa_per_hour, 0.0, 1.0, True)
            ds = []
            for z, t0 in sessions:
                d = first_detection(fn(z, tau), t0, len(z))
                ds.append(d if d is not None else (len(z) - t0) + 1)      # a miss counts as the full remaining time
            score = float(np.mean(ds))
            if best is None or score < best[0]:
                best = (score, kw, kd, tau)
    return best


def run_fold(people, test_idx, rng, a):
    test_set = set(test_idx)
    test = [people[i] for i in test_idx]
    rest = [people[i] for i in range(len(people)) if i not in test_set]
    perm = rng.permutation(len(rest)); half = len(rest) // 2
    train = [rest[i] for i in perm[:half]]
    cal = [rest[i] for i in perm[half:]]
    assert not ({p["actor"] for p in train + cal} & {p["actor"] for p in test})
    assert not ({p["actor"] for p in train} & {p["actor"] for p in cal})
    rk_train, rk_test = R.similarity_rank(train), R.similarity_rank(test)

    fusers = R.fit_fusers(train, rk_train, rng, a)
    methods = R.make_methods(fusers)
    cal_X = np.vstack([R.make_stream(rng, P, P, a.cal_len, a.p_deg) for P in cal])
    if a.paper:
        methods.update(R.paper_methods())
    params = R.calibrate_all(methods, cal_X, a.target_fa_per_hour)
    if a.paper:
        _, kw, kd, tau = tune_paper(cal, R.similarity_rank(cal), cal_X, rng, a)
        edge = (kw in (min(GRID_KW), max(GRID_KW))) or (kd in (min(GRID_KD), max(GRID_KD)))
        print(f"   tuned paper baseline: k_weight={kw}s k_decay={kd}s tau={tau:.3f}"
              + ("   <-- WARNING: best setting is on the edge of the search grid" if edge else ""), flush=True)
        methods[TUNED] = dict(prep=lambda X: X[:, 4], fn=lambda x, t, kw=kw, kd=kd: D.trust_authenticator(x, t, kw, kd),
                              rng=(0.0, 1.0), up=True)
        params[TUNED] = tau

    rows = []
    for P in test:
        gen = R.make_stream(rng, P, P, a.test_gen_len, a.p_deg)               # genuine-only stream -> false alarms
        sessions = []
        for _ in range(a.sessions_per_person):
            Q = R.pick_impostor(rng, P, test, rk_test, a.hard_frac)
            t0 = int(rng.integers(30, 120))
            g, i = R.make_stream(rng, P, P, 180, a.p_deg), R.make_stream(rng, Q, P, 180, a.p_deg)
            X = g.copy(); X[t0:] = i[t0:]
            sessions.append((X, t0))
        for m, c in methods.items():
            if params[m] is None:
                alarms_gen, delays = 0, [None] * len(sessions)
            else:
                alarms_gen = len(c["fn"](c["prep"](gen), params[m]))
                delays = [first_detection(c["fn"](c["prep"](X), params[m]), t0, len(X)) for X, t0 in sessions]
            det = [d for d in delays if d is not None]
            rows.append(dict(actor=P["actor"], method=m,
                             mean_delay_s=(np.mean(det) * CHECK_SECONDS) if det else np.nan,
                             missed=1 - len(det) / len(delays),
                             genuine_alarms=alarms_gen, genuine_hours=len(gen) / CHECKS_PER_HOUR))
    return rows


def boot(rng, per_person, stat, B):
    n = len(per_person)
    vals = [stat(per_person.iloc[rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(a):
    rng = np.random.default_rng(a.seed)
    sfx = f"_{a.tag}" if a.tag else ""
    actors = R.load_cache(a.cache)
    people = [R.build_person(k, v, a.enroll, a.min_pool, rng) for k, v in sorted(actors.items())]
    people = [p for p in people if p is not None]
    print(f"{len(people)} usable people, {a.folds}-fold person-disjoint CV, target {a.target_fa_per_hour} false alarms/h, "
          f"p_deg={a.p_deg}")
    order = rng.permutation(len(people))
    folds = np.array_split(order, a.folds)
    rows = []
    for k, f in enumerate(folds):
        print(f"fold {k + 1}/{a.folds}: {len(f)} test people", flush=True)
        rows += run_fold(people, list(f), rng, a)
    df = pd.DataFrame(rows)
    df.to_csv(f"results_cv_persons{sfx}.csv", index=False)
    assert df.groupby("method").actor.nunique().eq(len(people)).all()       # each person tested once per method

    out = []
    for m, g in df.groupby("method", sort=False):
        g = g.reset_index(drop=True)
        d_lo, d_hi = boot(rng, g, lambda x: x.mean_delay_s.mean(), a.boot)
        m_lo, m_hi = boot(rng, g, lambda x: x.missed.mean(), a.boot)
        f_lo, f_hi = boot(rng, g, lambda x: x.genuine_alarms.sum() / x.genuine_hours.sum(), a.boot)
        out.append(dict(method=m, delay_s=g.mean_delay_s.mean(), delay_ci=f"[{d_lo:.1f}, {d_hi:.1f}]",
                        missed=g.missed.mean(), missed_ci=f"[{m_lo:.3f}, {m_hi:.3f}]",
                        false_alarms_per_h=g.genuine_alarms.sum() / g.genuine_hours.sum(),
                        fa_ci=f"[{f_lo:.3f}, {f_hi:.3f}]"))
    summ = pd.DataFrame(out); summ.to_csv(f"results_cv_summary{sfx}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(f"\nPERSON-DISJOINT {a.folds}-FOLD CV | {len(people)} test people (each once) | 95% bootstrap CI over people")
    print("(delay = mean over DETECTED swaps; simulated degradation; no attacks/liveness)\n")
    print(summ.round(3).to_string(index=False))

    piv = df.pivot(index="actor", columns="method", values="mean_delay_s")
    pr = []
    for m in piv.columns:
        if m == PROPOSED or piv[m].isna().all() or m.startswith("login_only"):
            continue
        d = (piv[m] - piv[PROPOSED]).dropna().reset_index(drop=True)
        if len(d) < 5:
            continue
        lo, hi = boot(rng, pd.DataFrame({"d": d}), lambda x: x.d.mean(), a.boot)
        pr.append(dict(method=m, n_people=len(d), mean_delay_minus_proposed_s=d.mean(), ci=f"[{lo:.1f}, {hi:.1f}]",
                       pct_people_proposed_faster=100 * (d > 0).mean()))
    paired = pd.DataFrame(pr); paired.to_csv(f"results_cv_paired{sfx}.csv", index=False)
    print("\nPaired per-person difference (positive = proposed faster); CI excludes 0 => reliable at 95%:")
    print(paired.round(2).to_string(index=False))

    if a.paper:
        base_names = ["paper trust-decay, as published (tau=0.5)", "paper trust-decay, tau calibrated", TUNED]
        ours = ["cusum, face only (skip failures)", "cusum, fused + failure flag only", PROPOSED]
        rows2 = []
        for b in base_names:
            for o in ours:
                pair = piv[[b, o]].dropna().reset_index(drop=True)
                if len(pair) < 5:
                    continue
                pct = lambda x: 100 * (1 - x[o].mean() / x[b].mean())
                lo, hi = boot(rng, pair, pct, a.boot)
                rows2.append(dict(baseline=b, ours=o, n_people=len(pair), baseline_delay_s=pair[b].mean(),
                                  ours_delay_s=pair[o].mean(), pct_faster=pct(pair), ci=f"[{lo:.1f}, {hi:.1f}]"))
        imp = pd.DataFrame(rows2); imp.to_csv(f"results_cv_vs_paper{sfx}.csv", index=False)
        print("\nIMPROVEMENT vs a reimplementation of Palmetshofer et al. (2026) trust-decay, same features, same people")
        print("(% faster = 1 - ours/baseline mean delay on people where both detected; CI = bootstrap over people)\n")
        print(imp.round(2).to_string(index=False))
        fa = summ.set_index("method").false_alarms_per_h
        print("\nFalse alarms/hour (compare like with like): " + "; ".join(f"{k[:40]}={fa[k]:.3f}" for k in base_names + ours if k in fa.index))

    fig, ax = plt.subplots(figsize=(9, 5))
    s2 = summ.set_index("method")
    lo = [float(x.strip("[]").split(",")[0]) for x in s2.delay_ci]
    hi = [float(x.strip("[]").split(",")[1]) for x in s2.delay_ci]
    y = s2.delay_s.fillna(0).values
    ax.barh(s2.index, y, xerr=[np.clip(y - np.array(lo), 0, None), np.clip(np.array(hi) - y, 0, None)], color="#4C72B0")
    ax.set_xlabel("mean detection delay (s), 95% CI over people"); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f"delay_comparison_cv{sfx}.png", dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/swapdetect_data/features_cache"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. s1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--enroll", type=int, default=4)
    ap.add_argument("--min_pool", type=int, default=6)
    ap.add_argument("--train_len", type=int, default=1500)
    ap.add_argument("--n_imp", type=int, default=8)
    ap.add_argument("--cal_len", type=int, default=3000)
    ap.add_argument("--test_gen_len", type=int, default=3000, help="genuine checks per test person (false alarms)")
    ap.add_argument("--no_paper", dest="paper", action="store_false", help="skip the paper-baseline reimplementation")
    ap.add_argument("--tune_sessions", type=int, default=5, help="swap sessions per calibration person for tuning the baseline")
    ap.add_argument("--sessions_per_person", type=int, default=20)
    ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--p_deg", type=float, default=0.3)
    ap.add_argument("--target_fa_per_hour", type=float, default=1.0)
    main(ap.parse_args())
