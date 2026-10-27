# Questions a reviewer is likely to ask

**Did you improve the paper?** No. We improved on a reimplementation of its method, on our own data and simulation.

**Why delay and not accuracy?** The task is detection over time, so we report delay, false alarms and misses together.

**Isn't lower delay just a looser threshold?** That is why false alarms are reported next to delay and every calibrated method targets a similar rate. Ours average 0.169 per hour vs 0.145 for the baseline: comparable, not identical.

**Did you tune the baseline fairly?** Yes: time constants and threshold tuned on calibration people only, and the grid widened twice until the result stopped changing.

**Why not use the as-published baseline?** Its threshold gives about 17 false alarms per hour on our data.

**Is it stable?** Over 3 seeds the improvement ranged from 49.4% to 52.5%.

**Where does the gain come from?** Mostly voice fusion and the face-found flag. The quality score added nothing reliable.

**Would it work on real exams?** Unknown. It needs real webcams, real sessions and attack testing.
