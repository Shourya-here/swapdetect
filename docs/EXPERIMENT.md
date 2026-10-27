# Experiment

- **People:** 91 actors with recorded clips (acted studio material, not real exam sessions).
- **Degradation:** simulated webcam degradation (`--p_deg 0.5`) so face checks sometimes fail.
- **Swaps:** sessions in which another person takes over; every method's alarm time is recorded. Check cadence is one face check every 10 seconds.
- **Cross-validation:** 5 folds, each person tested exactly once. Thresholds, fusion weights and the baseline's tuned settings use calibration people only.
- **False-alarm target:** 0.1 per hour for calibrated methods.
- **Face model:** InsightFace buffalo_l for all methods.
- **Seeds:** 0, 1, 2.
- **Statistics:** delay computed on people where both methods detected the swap; 95% confidence intervals by bootstrap over people; percent faster = 1 - ours / baseline mean delay.

## Baseline tuning

Grid in `evaluate_cv.py`: k_weight 5, 10, 20, 45, 90 s; k_decay 60, 120, 300, 600, 1200, 2400, 6000 s. The best k_decay often sat at the top of the grid (6000 s, almost no decay), so we treat that as the baseline's strongest form. A warning is printed when the best setting is on a grid edge.

## Results

See the README table and `results_final/results.json`.

## Fusion model

Face and voice evidence are combined by a standardised logistic regression (see `fusion.py`), fitted on calibration people only.
