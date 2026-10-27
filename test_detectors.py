"""Unit tests with KNOWN change points. Run: python -m pytest -q   (or python test_detectors.py)"""
import numpy as np
import detectors as D
import simulator as sim
from fusion import Fusion


def test_cusum_detects_known_change_quickly():
    rng = np.random.default_rng(0)
    llr = np.r_[rng.normal(1.0, 0.5, 200), rng.normal(-1.5, 0.5, 100)]  # genuine -> impostor at t=200
    al = D.cusum(llr, h=8.0)
    first = [a for a in al if a >= 200]
    assert first and first[0] - 200 < 15
    assert not [a for a in al if a < 200]          # no false alarm before the change


def test_cusum_quiet_on_genuine_only():
    rng = np.random.default_rng(1)
    llr = rng.normal(1.0, 0.5, 5000)
    assert D.cusum(llr, h=8.0) == []


def test_higher_threshold_means_fewer_alarms():
    rng = np.random.default_rng(2)
    llr = rng.normal(-0.05, 1.0, 20000)
    assert len(D.cusum(llr, 4.0)) >= len(D.cusum(llr, 12.0))


def test_login_only_never_alarms():
    assert D.login_only(np.zeros(100)) == []


def test_calibration_hits_target():
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 1.0, 60000)
    h = D.calibrate(D.cusum, x, 0.5, 0.5, 100.0, more_alarms_when_larger=False)
    assert D.false_alarms_per_hour(D.cusum, x, h) <= 0.5


def test_fusion_scores_genuine_higher_than_impostor():
    rng = np.random.default_rng(4)
    ps = sim.make_people(5, 6)
    G = np.vstack([sim.genuine_stream(rng, 800, p) for p in ps])
    I = np.vstack([sim.impostor_stream(rng, 800, p, "person") for p in ps])
    f = Fusion("full").fit(G, I)
    assert f.llr(G).mean() > 0 > f.llr(I).mean()


def test_failed_checks_are_skipped():
    nan = float("nan")
    assert D.periodic_threshold([0.1, nan, 0.1], 0.5, k=2) == [2]      # NaN neither resets nor counts
    assert D.cusum([-1.0, nan, -1.0], 2.0) == [2]                       # NaN adds no evidence
    assert D.trust_decay([0.7, nan, 0.7], 0.5) == []                    # state unchanged by NaN


def test_paper_trust_authenticator_matches_hand_calculation():
    import math
    fw, fd = math.exp(-10 / 45), math.exp(-10 / 60)
    x1 = 1.0 * fw + 0.12 * (1 - fw)                       # trust after one weak match (z = 0.12)
    x2 = x1 * fw + 0.12 * (1 - fw)                        # ... after a second one (no reset yet)
    assert D.trust_authenticator([0.12, 0.12], tau=x1 + 1e-9) == [0, 1]      # locks, resets to 1.0, locks again
    assert D.trust_authenticator([0.12, 0.12], tau=(x1 + x2) / 2) == [1]     # only the second update crosses
    assert D.trust_authenticator([0.12, 0.12], tau=x2 - 1e-9) == []
    nan = float("nan")
    y = 1.0 * fd                                          # no face -> pure decay (NOT skipped)
    assert D.trust_authenticator([nan], tau=y + 1e-9) == [0]
    assert D.trust_authenticator([nan], tau=y - 1e-9) == []
    assert D.trust_authenticator([0.9] * 50, tau=0.5) == []                  # steady genuine similarity never locks


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
