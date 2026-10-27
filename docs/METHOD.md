# Method

## The task

Detect that the person in a session has changed (a swap), as quickly as possible, without raising too many false alarms on genuine candidates. Two numbers are always reported together: detection delay (seconds from the swap to the alarm) and false alarms per hour.

## Existing practice

Login-only verification checks the face once and never again. It detects no swaps.

## Baselines

- **Periodic face check:** alarm after k failed or mismatching checks (k = 1, 2, 3), skipping frames where the face check fails. Our own simple rule.
- **Simple trust decay:** our own simplified exponential decay rule. Not the published method.
- **Paper method:** reimplementation of the authenticator of Palmetshofer et al. (see REFERENCE_PAPER.md), in three versions: as published, threshold calibrated, and fully tuned on calibration people. The fully tuned version is the baseline we quote.

## Our method: three changes

### 1. Decision rule: CUSUM instead of decaying trust

The paper locks the session when a decaying trust score falls below a fixed threshold. We accumulate evidence of a change with a CUSUM detector and alarm when the total crosses a limit. The limit is calibrated on calibration people so the false-alarm rate hits a target (0.1 per hour in our runs). The alarm point is therefore set by the false-alarm budget, not by a number copied from another setting.

### 2. Evidence: face and voice fusion

The paper uses face only. We fuse face and voice evidence with weights learned on calibration people, so that when the face is poor the voice can still carry information.

### 3. Failure handling: face-found flag

In the paper, a missing face makes trust decay toward zero. We pass a face-found flag, so a frame without a face is treated as missing evidence, not as a sign of an impostor.

### Optional: continuous quality score

A face-quality score can be added to the fusion. Our results do not show that it helps.

## Attribution of the gain

The step from the tuned baseline to our face-only CUSUM bundles three things (decision rule, learned score, skipping failures). Voice fusion and the face-found flag were tested together. The gain should not be credited to any single change.

| Version | % faster than tuned baseline (mean, 3 seeds) | Range |
|---|---|---|
| Face only | 28.1 | 23.1 to 31.5 |
| + voice and failure flag | 49.3 | 47.0 to 51.9 |
| + quality (proposed) | 50.5 | 49.4 to 52.5 |
