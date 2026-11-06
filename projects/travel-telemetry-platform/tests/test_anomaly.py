"""Anomaly detection scored against the injected ground truth."""
from analysis.anomaly import detect_anomalies, score_against_ground_truth


def test_crash_spikes_detected(telemetry):
    df, truth = telemetry
    flagged = detect_anomalies(df, "crash_rate_pct", threshold=3.5)
    flagged_dates = [str(d.date()) for d in flagged["date"]]
    scores = score_against_ground_truth(flagged_dates, truth["crash_rate_pct"])
    assert scores["true_positives"] == len(truth["crash_rate_pct"]), scores
    assert scores["missed"] == 0, scores


def test_booking_drop_detected(telemetry):
    df, truth = telemetry
    flagged = detect_anomalies(df, "booking_completions", threshold=3.5)
    flagged_dates = [str(d.date()) for d in flagged["date"]]
    scores = score_against_ground_truth(flagged_dates, truth["booking_completions"])
    assert scores["true_positives"] >= 1, scores
    assert scores["missed"] <= 1, scores


def test_tuned_threshold_keeps_recall_with_fewer_false_positives(telemetry):
    df, truth = telemetry
    total_fp = 0
    for metric, truth_dates in truth.items():
        flagged = detect_anomalies(df, metric, threshold=5.0)
        flagged_dates = [str(d.date()) for d in flagged["date"]]
        scores = score_against_ground_truth(flagged_dates, truth_dates)
        assert scores["missed"] == 0, (metric, scores)
        assert scores["true_positives"] == len(truth_dates), (metric, scores)
        total_fp += scores["false_positives"]
    assert total_fp <= 5, total_fp


def test_no_flags_on_constant_series():
    import pandas as pd

    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60), "value": 5.0})
    flagged = detect_anomalies(df, "value", threshold=3.5)
    assert flagged.empty
