# Reference paper

**Palmetshofer, Findling, Kurz.** *Continuous Face Authentication on Mobile and Desktop Platforms: A Comparative Study.* arXiv 2608.00763.

## What the paper does

Continuous authentication with InsightFace embeddings. A trust score is updated with an exponential time-weighted rule. When no face is visible, trust decays toward zero instead of the frame being skipped. The session locks when trust falls below a threshold. Published settings include time constants of 45 s and 60 s, a threshold of 0.5, an initial trust of 1.0 and a 90th-percentile similarity. The study used 25 real volunteers on real devices and the compact buffalo_sc model.

## What we took

The idea of continuous authentication with a trust score, and the update equations, which we reimplemented (`detectors.py`, with a unit test against a hand calculation).

## What we did not take

Their data, their numbers, and their face model. We use buffalo_l for both our method and the reimplemented baseline, so the comparison is not confounded by model strength.

## Where we differ

See [METHOD.md](METHOD.md): a CUSUM decision rule, face and voice fusion, and a face-found flag.

## What this comparison is, and is not

It is a comparison against our reimplementation of the paper's method, on our features, our people and our simulated degradation. It is not a comparison against the paper's published results, and the paper's reported performance says nothing about ours.

## The three versions of the baseline

1. **As published** (threshold 0.5): about 17 false alarms per hour on our data. Short delay only because it alarms constantly. Not used for any claim.
2. **Threshold calibrated** to our false-alarm target.
3. **Fully tuned:** time constants and threshold chosen using only calibration people. This is the strongest baseline and the one we quote.
