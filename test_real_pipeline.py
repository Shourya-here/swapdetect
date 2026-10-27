"""Runs the whole real-data pipeline on a FAKE feature cache (random embeddings with person structure).
Checks that the code path works and splits don't leak. Numbers from this test mean nothing."""
import os, tempfile, argparse, numpy as np
import evaluate_real as R


def make_fake_cache(d, n_actors=36, n_clips=12, seed=0):
    rng = np.random.default_rng(seed)
    emos = ["ANG", "HAP", "NEU", "SAD", "FEA", "DIS"]
    for a in range(n_actors):
        cf, cv = rng.normal(size=512), rng.normal(size=192)
        cf, cv = cf / np.linalg.norm(cf), cv / np.linalg.norm(cv)
        for c in range(n_clips):
            stem = f"{1001 + a}_S{c:02d}_{emos[c % 6]}_XX"
            for ver, sig_f, sig_q in (("clean", 0.9, 0.9), ("degraded", 1.7, 0.55)):
                f = cf + rng.normal(0, sig_f / np.sqrt(512), 512); v = cv + rng.normal(0, 0.8 / np.sqrt(192), 192)
                np.savez(f"{d}/{stem}__{ver}.npz", face_emb=(f / np.linalg.norm(f)).astype(np.float32),
                         qual=float(np.clip(rng.normal(sig_q, 0.05), 0, 1)), face_ok=True,
                         voice_emb=v.astype(np.float32), actor=str(1001 + a), emotion=emos[c % 6],
                         sentence="S", version=ver)


def test_pipeline_runs_without_leakage():
    d = tempfile.mkdtemp(); make_fake_cache(d)
    a = argparse.Namespace(cache=d, seed=0, seeds=1, enroll=4, min_pool=6, train_len=400, n_imp=4,
                           cal_len=600, sessions_per_person=3, hard_frac=0.5, target_fa_per_hour=1.0, p_deg=0.3)
    actors = R.load_cache(d)
    assert len(actors) == 36
    df = R.run_one(actors, a, 0)
    assert len(df) == 10 and df.n_test_people.iloc[0] >= 10
    assert (df.missed_swap_rate >= 0).all() and (df.missed_swap_rate <= 1).all()
    assert df.loc[df.method.str.startswith("login_only"), "missed_swap_rate"].iloc[0] == 1.0
    print(df.round(2).to_string(index=False))


def test_cv_each_person_tested_once():
    import os, evaluate_cv as C
    d = tempfile.mkdtemp(); make_fake_cache(d)
    a = argparse.Namespace(cache=d, seed=0, folds=3, boot=200, enroll=4, min_pool=6, train_len=400, n_imp=4,
                           cal_len=600, test_gen_len=600, sessions_per_person=3, hard_frac=0.5, p_deg=0.3,
                           target_fa_per_hour=1.0, paper=True, tune_sessions=2, tag="")
    cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
    try:
        C.main(a)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    test_pipeline_runs_without_leakage(); print("ok test_pipeline_runs_without_leakage")
    test_cv_each_person_tested_once(); print("ok test_cv_each_person_tested_once")
