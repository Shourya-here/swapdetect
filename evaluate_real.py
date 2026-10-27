"""Evaluate on REAL CREMA-D features (face + voice from the same real person, simulated webcam degradation).

Pipeline per random PERSON split (train / calibration / test, no person shared):
  1. enrollment template per person = mean clean embedding of K clips (those clips are NOT used as checks)
  2. a "check" = one clip scored against the CLAIMED person's template:
        face = cosine(face embedding, face template), voice = cosine(voice embedding, voice template),
        qual = heuristic face quality, live = 1.0 (CREMA-D has no attacks; liveness NOT evaluated here)
  3. genuine stream: the person's own clips, in blocks of clean/degraded video (webcam conditions come in episodes)
     swap session: genuine until t0, then ANOTHER REAL person's clips (face AND voice switch together),
                   half the impostors are the 'most similar' people (hard pairs)
  4. fusion trained on train people only; every detector calibrated on calibration people's genuine streams
     to the same false alarms/hour; delay measured on swap sessions of unseen test people.

Honest limits: clips are short, acted, studio-lit; degradation and voice-silence masking are SIMULATED;
genuine streams re-sample a small pool of clips per person (so very rare events are under-represented);
independence between checks is assumed within blocks.

Run:  python evaluate_real.py --seeds 3
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import detectors as D
from fusion import Fusion
from evaluate import first_detection, bootstrap_ci
from simulator import CHECK_SECONDS


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def load_cache(cache_dir):
    actors = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        stem = os.path.basename(f).split("__")[0]
        d = np.load(f)
        actors.setdefault(str(d["actor"]), {}).setdefault(stem, {})[str(d["version"])] = dict(
            face=d["face_emb"], voice=d["voice_emb"], qual=float(d["qual"]),
            ok=bool(d["face_ok"]), emo=str(d["emotion"]))
    return actors


def build_person(actor, clips, K, min_pool, rng):
    stems = sorted(s for s in clips if "clean" in clips[s] and "degraded" in clips[s])
    neu = [s for s in stems if clips[s]["clean"]["emo"] == "NEU"]
    enroll = (neu + [s for s in stems if s not in neu])[:K]
    pool = [s for s in stems if s not in enroll]
    ef = [clips[s]["clean"]["face"] for s in enroll if clips[s]["clean"]["ok"]]
    if not ef or len(pool) < min_pool:
        return None
    P = dict(actor=actor, p_speech=float(rng.uniform(0.4, 0.8)), efaces=_unit(np.array(ef)),
             tface=_unit(np.mean(ef, 0)), tvoice=_unit(np.mean([clips[s]["clean"]["voice"] for s in enroll], 0)))
    f, v, q, ok, deg = [], [], [], [], []
    for s in pool:
        for ver in ("clean", "degraded"):
            c = clips[s][ver]
            f.append(c["face"]); v.append(c["voice"]); q.append(c["qual"]); ok.append(c["ok"]); deg.append(ver == "degraded")
    P.update(face=_unit(np.nan_to_num(np.array(f))), voice=_unit(np.array(v)), qual=np.array(q),
             ok=np.array(ok), deg=np.array(deg))
    return P


def make_stream(rng, src, claimed, n, p_deg=0.3):
    """n checks of person `src` scored against the template of person `claimed`."""
    clean, deg = np.where(~src["deg"])[0], np.where(src["deg"])[0]
    idx = []
    while len(idx) < n:                       # blocks of consistent webcam conditions
        L = int(rng.integers(5, 21))
        pool = deg if rng.random() < p_deg else clean
        idx.extend(rng.choice(pool, L))
    idx = np.array(idx[:n])
    ok = src["ok"][idx]
    face = np.where(ok, src["face"][idx] @ claimed["tface"], 0.0)       # no face found -> score 0, quality 0
    qual = np.where(ok, src["qual"][idx], 0.0)
    voice = src["voice"][idx] @ claimed["tvoice"]
    voice[rng.random(n) > src["p_speech"]] = np.nan                     # simulated: candidate not speaking
    # paper-style frame score Z_t: 90th percentile of cosine similarity to ALL enrollment embeddings (NaN = no face)
    zpct = np.where(ok, np.percentile(src["face"][idx] @ claimed["efaces"].T, 90, axis=1), np.nan)
    return np.column_stack([face, np.ones(n), voice, qual, zpct])


def pick_impostor(rng, P, people, sim_rank, hard_frac):
    others = [i for i, p in enumerate(people) if p["actor"] != P["actor"]]
    if rng.random() < hard_frac:
        top = [j for j in sim_rank[P["actor"]] if people[j]["actor"] != P["actor"]][:5]
        return people[int(rng.choice(top))]
    return people[int(rng.choice(others))]


def similarity_rank(people):
    T = np.stack([p["tface"] for p in people])
    S = T @ T.T
    return {p["actor"]: list(np.argsort(-S[i])) for i, p in enumerate(people)}


def fit_fusers(train, rank_train, rng, a):
    """Train the fusion models on TRAIN people only (genuine checks vs real impostor checks)."""
    G, I = [], []
    for P in train:
        G.append(make_stream(rng, P, P, a.train_len, a.p_deg))
        for _ in range(a.n_imp):
            Q = pick_impostor(rng, P, train, rank_train, a.hard_frac)
            I.append(make_stream(rng, Q, P, a.train_len // 8, a.p_deg))
    G, I = np.vstack(G), np.vstack(I)
    return {name: Fusion(name).fit(G, I) for name in ["full", "no_quality", "face_only", "fail_flag"]}


def make_methods(fusers):
    face_skip = lambda X: np.where(X[:, 3] > 0, X[:, 0], np.nan)     # failed detections are skipped, not scored 0
    methods = {
        "login_only (existing)":          dict(prep=face_skip, fn=D.login_only, rng=(0, 1), up=True),
        "periodic k=1 (skip failures)":   dict(prep=face_skip, fn=lambda x, t: D.periodic_threshold(x, t, k=1), rng=(0.0, 0.95), up=True),
        "periodic k=2 (skip failures)":   dict(prep=face_skip, fn=lambda x, t: D.periodic_threshold(x, t, k=2), rng=(0.0, 0.95), up=True),
        "periodic k=3 (skip failures)":   dict(prep=face_skip, fn=lambda x, t: D.periodic_threshold(x, t, k=3), rng=(0.0, 0.95), up=True),
        "trust_decay lam=0.8 (skip)":     dict(prep=face_skip, fn=lambda x, t: D.trust_decay(x, t, lam=0.8), rng=(0.0, 0.95), up=True),
        "trust_decay lam=0.5 (skip)":     dict(prep=face_skip, fn=lambda x, t: D.trust_decay(x, t, lam=0.5), rng=(0.0, 0.95), up=True),
    }
    fo = fusers["face_only"]
    methods["cusum, face only (skip failures)"] = dict(
        prep=lambda X: np.where(X[:, 3] > 0, fo.llr(X), np.nan), fn=D.cusum, rng=(0.01, 200.0), up=False)
    for name, label in [("no_quality", "cusum, fused no quality"),
                        ("fail_flag", "cusum, fused + failure flag only"),
                        ("full", "cusum, fused + quality (proposed)")]:
        fz = fusers[name]
        methods[label] = dict(prep=(lambda X, fz=fz: fz.llr(X)), fn=D.cusum, rng=(0.01, 200.0), up=False)
    return methods


def paper_methods(k_weight=45.0, k_decay=60.0):
    """Reimplementation of the temporal-decay authenticator of Palmetshofer et al. (2026) on OUR features."""
    z = lambda X: X[:, 4]
    fn = lambda x, t: D.trust_authenticator(x, t, k_weight, k_decay)
    return {
        "paper trust-decay, as published (tau=0.5)": dict(prep=z, fn=fn, rng=(0.0, 1.0), up=True, fixed=0.5),
        "paper trust-decay, tau calibrated":          dict(prep=z, fn=fn, rng=(0.0, 1.0), up=True),
    }


def calibrate_all(methods, cal_X, target):
    """Calibrate every detector to the same false alarms/hour on calibration people's genuine checks."""
    params = {}
    for m, c in methods.items():
        if "fixed" in c:                      # e.g. the paper's published tau=0.5, not recalibrated
            params[m] = c["fixed"]
        else:
            params[m] = None if m.startswith("login_only") else D.calibrate(
                c["fn"], c["prep"](cal_X), target, c["rng"][0], c["rng"][1], c["up"])
    return params


def run_one(actors, a, seed):
    rng = np.random.default_rng(seed)
    people = [build_person(k, v, a.enroll, a.min_pool, rng) for k, v in sorted(actors.items())]
    people = [p for p in people if p is not None]
    order = rng.permutation(len(people)); n = len(people)
    train = [people[i] for i in order[: n // 3]]
    cal = [people[i] for i in order[n // 3: 2 * n // 3]]
    test = [people[i] for i in order[2 * n // 3:]]
    assert not ({p["actor"] for p in train} & {p["actor"] for p in test}) \
        and not ({p["actor"] for p in cal} & {p["actor"] for p in test}) \
        and not ({p["actor"] for p in train} & {p["actor"] for p in cal})
    ranks = {id(g): similarity_rank(g) for g in (train, cal, test)}

    fusers = fit_fusers(train, ranks[id(train)], rng, a)

    cal_X = np.vstack([make_stream(rng, P, P, a.cal_len, a.p_deg) for P in cal])
    methods = make_methods(fusers)

    params = {}
    for m, c in methods.items():
        params[m] = None if m.startswith("login_only") else D.calibrate(
            c["fn"], c["prep"](cal_X), a.target_fa_per_hour, c["rng"][0], c["rng"][1], c["up"])

    at_bound = {}
    for m, c in methods.items():
        p = params[m]
        if p is None:
            at_bound[m] = False
            continue
        lo, hi = c["rng"]; tol = (hi - lo) / 2 ** 10
        at_bound[m] = bool(p >= hi - tol) if c["up"] else bool(p <= lo + tol)
        if at_bound[m]:
            print(f"[seed {seed}] WARNING: {m}: calibration hit its search bound (param={p:.3f}); "
                  f"the false-alarm target is NOT binding for this method")
    test_gen = np.vstack([make_stream(rng, P, P, a.cal_len, a.p_deg) for P in test])
    fa = {m: (0.0 if params[m] is None else D.false_alarms_per_hour(c["fn"], c["prep"](test_gen), params[m]))
          for m, c in methods.items()}

    per, miss = {m: [] for m in methods}, {m: [] for m in methods}
    for P in test:
        sessions = []
        for _ in range(a.sessions_per_person):
            Q = pick_impostor(rng, P, test, ranks[id(test)], a.hard_frac)
            t0 = int(rng.integers(30, 120))
            g, i = make_stream(rng, P, P, 180, a.p_deg), make_stream(rng, Q, P, 180, a.p_deg)
            X = g.copy(); X[t0:] = i[t0:]
            sessions.append((X, t0))
        for m, c in methods.items():
            ds = [first_detection([] if params[m] is None else c["fn"](c["prep"](X), params[m]), t0, len(X))
                  for X, t0 in sessions]
            det = [d for d in ds if d is not None]
            miss[m].append(1 - len(det) / len(ds))
            per[m].append(np.mean(det) if det else np.nan)
    rows = []
    for m in methods:
        v = np.array(per[m]); ok = v[~np.isnan(v)]
        rows.append(dict(seed=seed, method=m, n_test_people=len(test),
                         mean_delay_s=np.nan if len(ok) == 0 else ok.mean() * CHECK_SECONDS,
                         missed_swap_rate=float(np.mean(miss[m])), false_alarms_per_hour_test=fa[m],
                         threshold=np.nan if params[m] is None else params[m], at_search_bound=at_bound[m]))
    return pd.DataFrame(rows)


def main(a):
    actors = load_cache(a.cache)
    print(f"loaded {len(actors)} actors from {a.cache}")
    dfs = [run_one(actors, a, a.seed + s) for s in range(a.seeds)]
    df = pd.concat(dfs)
    df.to_csv("results_real_all_splits.csv", index=False)
    summ = df.groupby("method", sort=False).agg(
        mean_delay_s=("mean_delay_s", "mean"), delay_sd_across_splits=("mean_delay_s", "std"),
        missed_swap_rate=("missed_swap_rate", "mean"),
        false_alarms_per_hour_test=("false_alarms_per_hour_test", "mean"),
        threshold=("threshold", "mean"), at_search_bound=("at_search_bound", "max")).reset_index()
    summ.to_csv("results_real_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"\nREAL CREMA-D features | target false alarms/hour = {a.target_fa_per_hour} | p_deg={a.p_deg} | hard_frac={a.hard_frac} | {a.seeds} person split(s)")
    print("(degradation + voice-silence are simulated; no liveness/attacks here)\n")
    print(summ.round(3).to_string(index=False))
    print("\nNote: delay is the mean over DETECTED swaps only; read it together with missed_swap_rate.")
    ref = "cusum, fused + quality (proposed)"
    pv = df.pivot(index="seed", columns="method", values="mean_delay_s")
    if ref in pv.columns and a.seeds > 1:
        rows = []
        for m in pv.columns:
            if m == ref or pv[m].isna().all():
                continue
            diff = (pv[m] - pv[ref]).dropna()
            rows.append(dict(method=m, mean_delay_minus_proposed_s=diff.mean(), sd=diff.std(),
                             splits_proposed_faster=f"{int((diff > 0).sum())}/{len(diff)}"))
        paired = pd.DataFrame(rows)
        paired.to_csv("results_real_paired.csv", index=False)
        print("\nPaired comparison across splits (positive = proposed detects faster):")
        print(paired.round(2).to_string(index=False))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(summ.method, summ.mean_delay_s.fillna(0), xerr=summ.delay_sd_across_splits.fillna(0), color="#4C72B0")
    for i, m in enumerate(summ.missed_swap_rate):
        ax.text(0, i, f"  missed {m:.0%}", va="center", fontsize=8)
    ax.set_xlabel("mean detection delay (s); bars = sd across person splits")
    plt.tight_layout(); plt.savefig("delay_comparison_real.png", dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/swapdetect_data/features_cache"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="number of different random person splits")
    ap.add_argument("--enroll", type=int, default=4)
    ap.add_argument("--min_pool", type=int, default=6)
    ap.add_argument("--train_len", type=int, default=1500)
    ap.add_argument("--n_imp", type=int, default=8)
    ap.add_argument("--cal_len", type=int, default=3000)
    ap.add_argument("--sessions_per_person", type=int, default=20)
    ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--p_deg", type=float, default=0.3, help="fraction of checks with degraded video (1.0 = stress test)")
    ap.add_argument("--target_fa_per_hour", type=float, default=1.0)
    main(ap.parse_args())
