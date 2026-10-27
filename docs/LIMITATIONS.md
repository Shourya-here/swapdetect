# Limitations and what we do not claim

We do not claim:

- that we beat the paper's published numbers (we did not use their data or system);
- better recognition accuracy (we measured detection delay, false alarms and misses);
- that the continuous quality score helps (its effect flips sign across seeds);
- that results carry over to real exams, real webcams, other populations or other devices;
- robustness to photo or replay attacks, deepfakes, or attackers who try to avoid detection.

Known limitations:

- Acted studio data with simulated degradation. Real degradation may behave differently.
- Our false-alarm rate averages slightly above the tuned baseline's (0.169 vs 0.145 per hour), so part of the speed advantage may come from a looser operating point.
- The baseline's tuned decay often hit the top of the search range.
- Voice and face come from the same studio recordings.
- Privacy, consent, and fairness across demographic groups were not evaluated and need attention before any real deployment.
