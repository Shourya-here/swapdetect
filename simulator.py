"""Synthetic per-check score streams for developing/testing the detector.

*** SYNTHETIC. Results from this module demonstrate the machinery, NOT scientific findings. ***

One "check" = one periodic verification (default every 10 s). Each check yields:
  face   cosine similarity to the enrolled face      (always present)
  live   liveness score (1 = live)                    (always present)
  voice  speaker similarity to enrolled voice         (NaN when candidate not speaking)
  qual   face quality in [0,1] (blur/pose/light)      (always present)

Person-level random effects, autocorrelated quality, bad-quality episodes (nuisance
changes) and heavy-tailed noise are included so that fixed thresholds misbehave the
way they do on real webcams.
"""
import numpy as np

CHECK_SECONDS = 10
CHECKS_PER_HOUR = 3600 // CHECK_SECONDS


def _t_noise(rng, n, df=5):
    return rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))  # unit variance


def make_person(rng):
    return {
        "face_off": rng.normal(0, 0.04),       # person-specific shift of genuine face score
        "voice_off": rng.normal(0, 0.05),
        "q_base": np.clip(rng.normal(0.85, 0.07), 0.5, 0.98),
        "episode_rate": rng.uniform(0.002, 0.008),
        "p_speech": np.clip(rng.normal(0.6, 0.1), 0.3, 0.9),
    }


def quality_stream(rng, n, person):
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.97 * z[t - 1] + 0.25 * rng.normal()
    q = np.clip(person["q_base"] + 0.08 * z, 0.05, 1.0)
    t = 0
    while t < n:  # bad-quality episodes: head turn, low light, occlusion
        if rng.random() < person["episode_rate"]:
            L = int(rng.geometric(0.1))
            q[t:t + L] = rng.uniform(0.10, 0.35, size=min(L, n - t))
            t += L
        else:
            t += 1
    return q


def genuine_stream(rng, n, person):
    q = quality_stream(rng, n, person)
    face = (0.68 + person["face_off"] - 0.30 * (1 - q)
            + (0.08 + 0.12 * (1 - q)) * _t_noise(rng, n))
    live = np.clip(0.92 - 0.10 * (1 - q) + 0.05 * _t_noise(rng, n), 0, 1)
    voice = 0.70 + person["voice_off"] + 0.10 * _t_noise(rng, n)
    voice[rng.random(n) > person["p_speech"]] = np.nan
    return np.column_stack([face, live, voice, q])


def impostor_stream(rng, n, person, kind="person"):
    """kind='person': a different real person takes over.
       kind='replay': genuine person's face/voice replayed on a screen (liveness low)."""
    q = quality_stream(rng, n, person)
    if kind == "person":
        pair_mu = rng.normal(0.15, 0.10)   # some impostors look similar (hard pairs)
        face = pair_mu + 0.05 * (1 - q) + (0.08 + 0.06 * (1 - q)) * _t_noise(rng, n)
        live = np.clip(0.92 - 0.10 * (1 - q) + 0.05 * _t_noise(rng, n), 0, 1)
        voice = rng.normal(0.20, 0.08) + 0.10 * _t_noise(rng, n)
    else:
        face = (0.62 + person["face_off"] - 0.30 * (1 - q)
                + (0.08 + 0.12 * (1 - q)) * _t_noise(rng, n))
        live = np.clip(0.28 + 0.12 * _t_noise(rng, n), 0, 1)
        voice = 0.65 + person["voice_off"] + 0.12 * _t_noise(rng, n)
    voice[rng.random(n) > person["p_speech"]] = np.nan
    return np.column_stack([face, live, voice, q])


def swap_session(rng, person, n=180, kind="person", t0_range=(30, 120)):
    """Genuine until t0, then impostor. Returns (X, t0)."""
    t0 = int(rng.integers(*t0_range))
    g = genuine_stream(rng, n, person)
    i = impostor_stream(rng, n, person, kind)
    X = g.copy()
    X[t0:] = i[t0:]
    return X, t0


def make_people(seed, n_people):
    rng = np.random.default_rng(seed)
    return [make_person(rng) for _ in range(n_people)]
