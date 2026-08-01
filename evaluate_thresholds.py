"""
APEX-AI — Clean Threshold Evaluation  [v1]

Fixes the circular evaluation in the first decision-support prototype:

  OLD: thresholds derived from healthy percentiles of ALL 7 bearings,
       false alarms counted on those SAME healthy windows (in-sample).
  NEW: 1) Thresholds are derived from TRAIN bearings only
          (dev_run_0, dev_run_1) — calibration-period score percentiles.
       2) An INDEPENDENT degradation-start reference is computed per run
          with the classic 3-sigma rule on raw RMS (per channel, from the
          calibration period), so alarms can be judged against a marker
          that does not come from our composite deviation score.
       3) Each candidate threshold (with the 3-consecutive-windows
          persistence rule from the decision-support doc) is evaluated on
          the 5 EVALUATION bearings (val_run_0 + test_run_0..3):
            - false-alarm events BEFORE degradation start
            - lead time (hours) from first true alarm to failure
       4) A scenario cost is attached to every threshold so the
          cost-optimal threshold can be identified (research contribution:
          threshold derived from cost, not accuracy alone).

All cost numbers are DECLARED SCENARIO ASSUMPTIONS (team placeholder
values), editable below. FEMTO is an accelerated-life test: whole
lifetimes are a few hours, so the planning horizon is expressed in
FEMTO-time, not real plant time.

Inputs : data/deviation_score_per_snapshot.csv
         data/features_per_snapshot.csv
Outputs: clean_threshold_evaluation.csv   (one row per candidate threshold)
         clean_per_run_alarms.csv         (per run x threshold detail)
         degradation_start_reference.csv  (3-sigma marker per run)
"""

import numpy as np
import pandas as pd

# ----------------------------- parameters -----------------------------
CALIBRATION_WINDOWS = 200          # must match health_deviation_score.py
SECONDS_PER_SNAPSHOT = 10          # FEMTO: one snapshot every 10 s
PERSISTENCE = 3                    # consecutive windows required (Jory's rule)
CANDIDATE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]

TRAIN_RUNS = ["dev_run_0", "dev_run_1"]
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]

# 3-sigma degradation-start reference (independent of deviation score)
SIGMA_K = 3.0                      # classic 3-sigma bound on raw RMS
DEG_PERSISTENCE = 5                # consecutive windows above bound

# ---- scenario cost assumptions (placeholders — team-editable) ----
INSPECTION_COST_SAR = 1000         # cost of one (false-)alarm inspection
PREDICTIVE_COST_SAR = 80000        # planned maintenance incl. inspection
REACTIVE_COST_SAR = 101000         # run-to-failure total (repair+downtime)
PLANNING_HORIZON_HOURS = 1.0       # min lead time to actually plan (FEMTO-time)
# ----------------------------------------------------------------------


def windows_to_hours(n):
    return n * SECONDS_PER_SNAPSHOT / 3600.0


def first_persistent_exceedance(above: np.ndarray, k: int, start: int = 0):
    """Index of the first window where `above` has been True for k
    consecutive windows (returns the FIRST window of that streak),
    searching from `start`. None if never."""
    streak = 0
    for i in range(start, len(above)):
        streak = streak + 1 if above[i] else 0
        if streak >= k:
            return i - k + 1
    return None


def alarm_events(above: np.ndarray, k: int, start: int, end: int):
    """Count separate alarm EVENTS (persistent streaks) in [start, end).
    An event = k consecutive True windows; the event ends when the
    signal drops below threshold, so one long excursion counts once."""
    events = 0
    streak = 0
    fired = False
    for i in range(start, end):
        if above[i]:
            streak += 1
            if streak >= k and not fired:
                events += 1
                fired = True
        else:
            streak = 0
            fired = False
    return events


def main():
    scores = pd.read_csv("deviation_score_per_snapshot.csv")
    feats = pd.read_csv("features_per_snapshot.csv")

    # ---------- 1) thresholds context: train-only calibration percentiles ----------
    train_cal = scores[
        scores["run"].isin(TRAIN_RUNS)
        & (scores["window_index"] < CALIBRATION_WINDOWS)
    ]["deviation_score"]
    pct = {p: np.percentile(train_cal, p) for p in (90, 95, 99, 99.5, 99.9)}
    print("Train-only calibration score percentiles "
          f"({len(train_cal)} windows from {TRAIN_RUNS}):")
    for p, v in pct.items():
        print(f"  P{p}: {v:.3f}")

    # ---------- 2) independent degradation-start reference (3-sigma on RMS) ----------
    deg_rows = []
    deg_start = {}
    for run, g in feats.groupby("run", sort=False):
        g = g.sort_values("window_index").reset_index(drop=True)
        n = len(g)
        above_any = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            cal = g[ch].iloc[:CALIBRATION_WINDOWS]
            bound = cal.mean() + SIGMA_K * cal.std()
            above_any |= (g[ch].to_numpy() > bound)
        idx = first_persistent_exceedance(above_any, DEG_PERSISTENCE,
                                          start=CALIBRATION_WINDOWS)
        deg_start[run] = idx if idx is not None else n  # never degraded -> end
        deg_rows.append({
            "run": run,
            "n_windows": n,
            "deg_start_window": idx,
            "deg_start_pct_of_life": None if idx is None else round(100 * idx / n, 1),
            "healthy_zone_windows": (idx if idx is not None else n) - CALIBRATION_WINDOWS,
        })
    deg_df = pd.DataFrame(deg_rows)
    deg_df.to_csv("degradation_start_reference.csv", index=False)
    print("\n3-sigma RMS degradation-start reference:")
    print(deg_df.to_string(index=False))

    # ---------- 3) evaluate each candidate threshold on EVAL runs ----------
    eval_rows, detail_rows = [], []
    for thr in CANDIDATE_THRESHOLDS:
        total_fa = 0
        lead_hours = []
        detected = 0
        for run in EVAL_RUNS:
            g = scores[scores["run"] == run].sort_values("window_index")
            s = g["deviation_score"].to_numpy()
            n = len(s)
            above = s >= thr
            ds = deg_start[run]

            # false-alarm events in assumed-healthy zone [calibration, deg_start)
            fa = alarm_events(above, PERSISTENCE, CALIBRATION_WINDOWS, ds)

            # first true alarm at/after degradation start
            first_alarm = first_persistent_exceedance(above, PERSISTENCE, start=ds)
            if first_alarm is not None:
                detected += 1
                lead_w = n - first_alarm          # windows to failure (end of run)
                lead_h = windows_to_hours(lead_w)
                lead_hours.append(lead_h)
            else:
                lead_w, lead_h = None, None

            total_fa += fa
            detail_rows.append({
                "threshold": thr, "run": run,
                "false_alarm_events": fa,
                "deg_start_window": ds,
                "first_true_alarm_window": first_alarm,
                "lead_time_hours": None if lead_h is None else round(lead_h, 2),
            })

        # ---------- 4) scenario cost for this threshold ----------
        cost = total_fa * INSPECTION_COST_SAR
        for lh in lead_hours:
            cost += (PREDICTIVE_COST_SAR if lh >= PLANNING_HORIZON_HOURS
                     else REACTIVE_COST_SAR)
        cost += (len(EVAL_RUNS) - detected) * REACTIVE_COST_SAR  # missed = reactive

        eval_rows.append({
            "threshold": thr,
            "false_alarm_events_total": total_fa,
            "detected_runs": f"{detected}/{len(EVAL_RUNS)}",
            "mean_lead_time_hours": round(np.mean(lead_hours), 2) if lead_hours else None,
            "min_lead_time_hours": round(np.min(lead_hours), 2) if lead_hours else None,
            "scenario_cost_sar": cost,
        })

    eval_df = pd.DataFrame(eval_rows)
    detail_df = pd.DataFrame(detail_rows)
    eval_df.to_csv("clean_threshold_evaluation.csv", index=False)
    detail_df.to_csv("clean_per_run_alarms.csv", index=False)

    best = eval_df.loc[eval_df["scenario_cost_sar"].idxmin()]
    print("\nClean threshold evaluation (5 evaluation bearings):")
    print(eval_df.to_string(index=False))
    print(f"\nCost-optimal threshold under current scenario: {best['threshold']}"
          f" (scenario cost {best['scenario_cost_sar']:.0f} SAR)")
    print("\nPer-run detail saved to clean_per_run_alarms.csv")


if __name__ == "__main__":
    main()
