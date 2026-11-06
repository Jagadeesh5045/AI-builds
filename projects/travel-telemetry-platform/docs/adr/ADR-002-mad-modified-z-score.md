# ADR-002: Rolling median plus MAD modified z-score for anomaly detection

Status: Accepted

## Context
Leadership dashboards need anomaly flags that survive weekly seasonality and
occasional genuine spikes. Mean and standard deviation approaches are pulled
around by the very spikes they are meant to catch, so a real crash spike would
raise the bar for detecting the next one.

## Decision
Use the Iglewicz and Hoaglin modified z-score: for each point, baseline is the
rolling median of prior values (current point excluded), scale is the rolling
median absolute deviation (MAD), z = 0.6745 * (x - median) / MAD, flag at
|z| > 3.5 by default with the threshold configurable. Windows are
frequency-aware: 7 days for daily views, 28 for weekly, 84 for monthly.
Daily telemetry also has a weekly cycle, so daily detection uses the seasonal
form of the same estimator: the baseline pool for a point is its own weekday
over the prior 12 weeks. A threshold sweep on the synthetic dataset (recorded
in RESULTS.md) set the platform operating threshold to 5.0: every injected
anomaly is still recovered, with far fewer false positives than at 3.5.

## Consequences
Detection is robust to outliers and needs no training data. Performance is
scored honestly against injected ground-truth anomalies in the test suite
(true positives, false positives and misses are reported in RESULTS.md).
Known limit: the first days of a series have no baseline and cannot be flagged.
