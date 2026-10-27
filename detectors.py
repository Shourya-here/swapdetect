"""Detectors. Each takes a 1-D array of per-check evidence and a parameter, returns the indices
of alarms. After an alarm the detector resets (the candidate would be re-checked / reviewed).

Baselines:
  login_only          never alarms after login (the existing system in walkthrough.md)
  periodic_threshold  alarm if face score < tau for k consecutive checks (fixed-threshold periodic check)
  trust_decay         exponentially-smoothed face score; alarm if it drops below tau
                      (simplified re-implementation of the 'temporal trust decay' idea, NOT the authors' code)
Proposed:
  cusum               CUSUM on the fused per-check LLR; alarm when evidence for 'impostor' reaches h
"""
import numpy as np
from simulator import CHECKS_PER_HOUR


def login_only(x, param=None):
    return []


def periodic_threshold(face, tau, k=3):
    alarms, run = [], 0
    for t, s in enumerate(face):
        if s != s:            # NaN = failed check (e.g., no face found): no evidence either way
            continue
        run = run + 1 if s < tau else 0
        if run >= k:
            alarms.append(t)
            run = 0
    return alarms


def trust_decay(face, tau, lam=0.8, init=0.7):
    alarms, trust = [], init
    for t, s in enumerate(face):
        if s != s:            # NaN = failed check: leave trust unchanged
            continue
        trust = lam * trust + (1 - lam) * s
        if trust < tau:
            alarms.append(t)
            trust = init
    return alarms


def cusum(llr_genuine, h):
    """Page's CUSUM for a change from genuine to impostor. Evidence per check = -LLR(genuine)."""
    alarms, S = [], 0.0
    for t, l in enumerate(llr_genuine):
        if l != l:            # NaN = failed check: no evidence either way
            continue
        S = max(0.0, S - l)
        if S >= h:
            alarms.append(t)
            S = 0.0
    return alarms


def false_alarms_per_hour(alarm_fn, x, param):
    return len(alarm_fn(x, param)) / len(x) * CHECKS_PER_HOUR


def calibrate(alarm_fn, x, target_per_hour, lo, hi, more_alarms_when_larger, iters=16):
    """Bisection on the detector parameter so false alarms/hour on genuine data <= target (closest)."""
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = false_alarms_per_hour(alarm_fn, x, mid)
        if rate <= target_per_hour:
            best = mid
            if more_alarms_when_larger:
                lo = mid      # can afford more alarms -> larger param
            else:
                hi = mid      # can afford more alarms -> smaller param
        else:
            if more_alarms_when_larger:
                hi = mid
            else:
                lo = mid
    return best if best is not None else (lo if more_alarms_when_larger else hi)


def trust_authenticator(z, tau, k_weight=45.0, k_decay=60.0, dt=10.0, x0=1.0):
    """Time-weighted trust decay, implemented from Eq. 3-4 of Palmetshofer, Findling & Kurz (2026),
    "Continuous Face Authentication on Mobile and Desktop Platforms: A Comparative Study" (arXiv 2608.00763).

        face found : X_t = X_{t-1} * f_wei + Z_t * (1 - f_wei),  f_wei = exp(-dt / k_weight)
        no face    : X_t = X_{t-1} * f_dec,                      f_dec = exp(-dt / k_decay)
        lock when X_t < tau; after a lock the user re-authenticates (trust reset to x0).

    z      per-check similarity Z_t (the paper uses the 90th percentile over enrollment embeddings);
           NaN means 'no face detected' (decay branch, NOT skipped).
    dt     seconds between checks (our nominal cadence is 10 s; the paper runs at 1 Hz).
    Returns indices of locks. Written independently from the published equations, not copied from their code.
    """
    import math
    fw, fd = math.exp(-dt / k_weight), math.exp(-dt / k_decay)
    x, alarms = x0, []
    for t, s in enumerate(z):
        x = x * fd if s != s else x * fw + s * (1 - fw)
        if x < tau:
            alarms.append(t)
            x = x0
    return alarms
