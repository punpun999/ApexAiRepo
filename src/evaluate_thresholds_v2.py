"""
APEX-AI — Clean Threshold Evaluation v2: EWMA-smoothed decision pipeline

Same clean methodology as v1 (train-only thresholds, 3-sigma RMS
degradation-start reference, evaluation on the 5 held-out bearings),
but the decision pipeline now matches what the decision layer will
actually run in production:

    deviation score -> EWMA smoothing -> threshold -> persistence(3)

EWMA (exponentially weighted moving average) with smoothing factor
ALPHA: each smoothed value = ALPHA * new_score + (1-ALPHA) * previous.
Small ALPHA = heavier smoothing = fewer fragmented alarm events, at the
price of a small reaction delay. ALPHA = 0.2 means roughly "the last
~10 windows dominate" (~ 100 s of signal).

Outputs:
  results/clean_threshold_evaluation_v2.csv  (raw vs smoothed, per threshold)
  results/clean_per_run_alarms_v2.csv        (per run x threshold, smoothed)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CALIBRATION_WINDOWS = 200
SECONDS_PER_SNAPSHOT = 10
PERSISTENCE = 3
ALPHA = 0.2
CANDIDATE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]

TRAIN_RUNS = ["dev_run_0", "dev_run_1"]
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]

SIGMA_K = 3.0
DEG_PERSISTENCE = 5

INSPECTION_COST_SAR = 1000
PREDICTIVE_COST_SAR = 80000
REACTIVE_COST_SAR = 101000
PLANNING_HORIZON_HOURS = 1.0


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
    ev, streak, fired = 0, 0, False
    for i in range(start, end):
        if above[i]:
            streak += 1
            if streak >= k and not fired:
                ev += 1
                fired = True
        else:
            streak, fired = 0, False
    return ev


def degradation_reference(feats):
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


def evaluate(scores, deg_start, smooth):
    rows, detail = [], []
    for thr in CANDIDATE_THRESHOLDS:
        total_fa, lead_hours, detected = 0, [], 0
        for run in EVAL_RUNS:
            s = (scores[scores["run"] == run]
                 .sort_values("window_index")["deviation_score"].to_numpy())
            if smooth:
                s = ewma(s, ALPHA)
            above = s >= thr
            n, ds = len(s), deg_start[run]
            fa = alarm_events(above, PERSISTENCE, CALIBRATION_WINDOWS, ds)
            first_alarm = first_persistent_exceedance(above, PERSISTENCE, start=ds)
            if first_alarm is not None:
                detected += 1
                lead_hours.append(windows_to_hours(n - first_alarm))
            total_fa += fa
            detail.append({
                "threshold": thr, "run": run, "false_alarm_events": fa,
                "first_true_alarm_window": first_alarm,
                "lead_time_hours": (None if first_alarm is None
                                    else round(windows_to_hours(n - first_alarm), 2)),
            })
        cost = total_fa * INSPECTION_COST_SAR
        for lh in lead_hours:
            cost += (PREDICTIVE_COST_SAR if lh >= PLANNING_HORIZON_HOURS
                     else REACTIVE_COST_SAR)
        cost += (len(EVAL_RUNS) - detected) * REACTIVE_COST_SAR
        rows.append({
            "pipeline": "smoothed" if smooth else "raw",
            "threshold": thr,
            "false_alarm_events_total": total_fa,
            "detected_runs": f"{detected}/{len(EVAL_RUNS)}",
            "mean_lead_time_hours": round(np.mean(lead_hours), 2) if lead_hours else None,
            "min_lead_time_hours": round(np.min(lead_hours), 2) if lead_hours else None,
            "scenario_cost_sar": cost,
        })
    return rows, detail


def main():
    scores = pd.read_csv(RESULTS / "deviation_score_per_snapshot.csv")
    feats = pd.read_csv(RESULTS / "features_per_snapshot.csv")
    deg_start = degradation_reference(feats)

    raw_rows, _ = evaluate(scores, deg_start, smooth=False)
    smo_rows, smo_detail = evaluate(scores, deg_start, smooth=True)

    both = pd.DataFrame(raw_rows + smo_rows)
    both.to_csv(RESULTS / "clean_threshold_evaluation_v2.csv", index=False)
    pd.DataFrame(smo_detail).to_csv(RESULTS / "clean_per_run_alarms_v2.csv", index=False)

    print("RAW pipeline (score -> threshold -> persistence):")
    print(pd.DataFrame(raw_rows).drop(columns="pipeline").to_string(index=False))
    print("\nSMOOTHED pipeline (score -> EWMA -> threshold -> persistence):")
    smo_df = pd.DataFrame(smo_rows).drop(columns="pipeline")
    print(smo_df.to_string(index=False))
    best = smo_df.loc[smo_df["scenario_cost_sar"].idxmin()]
    print(f"\nCost-optimal threshold (smoothed): {best['threshold']}"
          f" — {best['scenario_cost_sar']:.0f} SAR")


if __name__ == "__main__":
    main()
