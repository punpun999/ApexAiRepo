"""
APEX-AI — Threshold x Persistence(k) Sensitivity Sweep  [for paper Table 3]

Regenerates the "Evaluation policy sensitivity" table using the CURRENT
fixed-calibration deviation scores (same data health_deviation_score.py,
evaluate_thresholds_v2.py etc. already produced today) — this replaces
the paper's existing Table 3, which was built before the calibration-
window fix and is now stale.

Same methodology as evaluate_thresholds_v2.py (train-only thresholds,
independent 3-sigma-on-RMS degradation-start reference, EWMA-smoothed
production pipeline), generalized in ONE way: instead of a single fixed
persistence (k=3) and mean lead time, this sweeps k across several
values and reports MEDIAN + RANGE, matching the paper table's columns:

    Threshold | k | Runs reaching criterion | Median lead (h) |
    Lead range (h) | Reference-period sequences

"Reference-period sequences" = total persistent false-alarm EVENTS
(k-consecutive-window streaks) occurring in the assumed-healthy zone
[calibration end, degradation start) across the 5 evaluation bearings —
i.e. exactly what evaluate_thresholds_v2.py calls false_alarm_events,
just swept over k as well as threshold.

Default THRESHOLDS/K_VALUES below reproduce the paper's apparent table
shape (3.0 and 3.5, at k=1/3/5 -> 6 rows). Edit either list to widen the
sweep if the original table had additional rows.

Inputs : deviation_score_per_snapshot.csv, features_per_snapshot.csv
         (already regenerated today by health_deviation_score.py)
Output : table3_threshold_k_sensitivity.csv

Run:
    python sweep_threshold_k_sensitivity.py
"""

import numpy as np
import pandas as pd

CALIBRATION_WINDOWS = 200          # must match health_deviation_score.py
SECONDS_PER_SNAPSHOT = 10
ALPHA = 0.2                        # EWMA smoothing — matches decision_layer.py

# --- what to sweep — set to cover every row in the paper's Table 3 ---
# Paper Table 3 rows: (3.0,k3) (3.5,k1) (3.5,k3) (3.5,k5) (4.0,k3) (5.0,k3) (7.0,k3)
# Sweeping the full grid below covers all of them (plus extras you can drop).
THRESHOLDS = [3.0, 3.5, 4.0, 5.0, 7.0]
K_VALUES = [1, 3, 5]

TRAIN_RUNS = ["dev_run_0", "dev_run_1"]
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]

SIGMA_K = 3.0                      # 3-sigma degradation-start reference
DEG_PERSISTENCE = 5                # fixed, independent of the k being swept


def windows_to_hours(n):
    return n * SECONDS_PER_SNAPSHOT / 3600.0


def ewma(x, alpha):
    return pd.Series(x).ewm(alpha=alpha).mean().to_numpy()


def first_persistent_exceedance(above, k, start=0):
    streak = 0
    for i in range(start, len(above)):
        streak = streak + 1 if above[i] else 0
        if streak >= k:
            return i - k + 1
    return None


def alarm_events(above, k, start, end):
    """Count persistent k-consecutive-window alarm EVENTS in [start, end)."""
    events, streak, fired = 0, 0, False
    for i in range(start, end):
        if above[i]:
            streak += 1
            if streak >= k and not fired:
                events += 1
                fired = True
        else:
            streak, fired = 0, False
    return events


def degradation_reference(feats):
    """Independent 3-sigma-on-RMS degradation-start marker per run.
    Unchanged by k — this is a fixed reference, not an alarm rule."""
    deg_start = {}
    for run, g in feats.groupby("run", sort=False):
        g = g.sort_values("window_index").reset_index(drop=True)
        n = len(g)
        above_any = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            cal = g[ch].iloc[:CALIBRATION_WINDOWS]
            above_any |= (g[ch].to_numpy() > cal.mean() + SIGMA_K * cal.std())
        idx = first_persistent_exceedance(above_any, DEG_PERSISTENCE,
                                          start=CALIBRATION_WINDOWS)
        deg_start[run] = idx if idx is not None else n
    return deg_start


def main():
    scores = pd.read_csv("deviation_score_per_snapshot.csv")
    feats = pd.read_csv("features_per_snapshot.csv")
    deg_start = degradation_reference(feats)

    # Pre-compute each run's smoothed score series once (same for every
    # threshold/k combination — only the threshold comparison changes).
    smoothed_by_run = {}
    for run in EVAL_RUNS:
        s = (scores[scores["run"] == run]
             .sort_values("window_index")["deviation_score"].to_numpy())
        smoothed_by_run[run] = ewma(s, ALPHA)

    rows = []
    for thr in THRESHOLDS:
        for k in K_VALUES:
            total_ref_sequences = 0
            leads = []
            detected = 0
            for run in EVAL_RUNS:
                sm = smoothed_by_run[run]
                above = sm >= thr
                n, ds = len(sm), deg_start[run]

                # false-alarm ("reference-period") events before degradation start
                total_ref_sequences += alarm_events(above, k, CALIBRATION_WINDOWS, ds)

                # first TRUE alarm at/after degradation start
                first_alarm = first_persistent_exceedance(above, k, start=ds)
                if first_alarm is not None:
                    detected += 1
                    leads.append(windows_to_hours(n - first_alarm))

            if leads:
                median_lead = round(float(np.median(leads)), 2)
                lead_range = f"{min(leads):.2f}-{max(leads):.2f}"
            else:
                median_lead, lead_range = None, None

            rows.append({
                "threshold": thr,
                "k": k,
                "runs_reaching_criterion": f"{detected}/{len(EVAL_RUNS)}",
                "median_lead_hours": median_lead,
                "lead_range_hours": lead_range,
                "reference_period_sequences": total_ref_sequences,
            })

    table3 = pd.DataFrame(rows)
    table3.to_csv("table3_threshold_k_sensitivity.csv", index=False)

    print("Table 3 replacement — threshold x persistence(k) sensitivity")
    print("(fixed-calibration data, smoothed pipeline, 5 evaluation bearings)\n")
    print(table3.to_string(index=False))
    print("\nSaved table3_threshold_k_sensitivity.csv")
    print("\nCompare each row against the paper's current Table 3 — any cell")
    print("that differs should be updated before submission.")


if __name__ == "__main__":
    main()
