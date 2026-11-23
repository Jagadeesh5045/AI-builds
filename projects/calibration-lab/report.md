# Calibration Lab: results

Dataset: 900 synthetic discharges, prevalence 0.221. Split 540 train, 180 validation, 180 test.

## Test set, raw vs calibrated

Raw: ROC-AUC 0.722, Brier 0.1571, log loss 0.4824, ECE 0.0551.
Calibrated (platt): ROC-AUC 0.722, Brier 0.1586, log loss 0.4882, ECE 0.0741.

## Threshold

Chosen on validation at a false-negative cost of £450 and a false-positive cost of £60: 0.13. Test cost at the chosen threshold: £8,220.00 for the test split, versus £14,820.00 at the default 0.5. Test confusion at the chosen threshold: {'tp': 30, 'fp': 62, 'tn': 78, 'fn': 10}.

## Slices (test, calibrated)

- age<65: n=77, prevalence 0.1688, ROC-AUC 0.7512, Brier 0.1188.
- age65-79: n=74, prevalence 0.2703, ROC-AUC 0.7333, Brier 0.1743.
- prior0: n=75, prevalence 0.1333, ROC-AUC 0.6523, Brier 0.1254.
- prior1: n=70, prevalence 0.2, ROC-AUC 0.6926, Brier 0.1447.
- prior2+: n=35, prevalence 0.4571, ROC-AUC 0.6612, Brier 0.2573.

Worst slice by Brier: prior2+ at 0.2573.
