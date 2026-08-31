"""
APEX-AI — Out-of-Sample Cost Evaluation and Cost-Ratio Sweep
[reviewer revision item: R2, "Present cost as an out-of-sample sweep"]

WHAT THE REVIEWER ASKED FOR
---------------------------
R2 flagged a real validity limit: the cost-optimal threshold (4.0) and the
provisional critical threshold (7.0) were both chosen AND reported on the
same five evaluation trajectories, so those two settings are effectively
in-sample. The request was to

  (a) separate the runs used to CHOOSE the threshold from the runs used to
      REPORT the cost,
  (b) plot expected cost against the cost ratio rather than reporting a
      single declared figure, following the Susto et al. (2015) template
      already cited in the paper, and
  (c) give per-run lead at threshold 4.0 alongside 3.5.

All three are implemented here.

PART A — LEAVE-ONE-TRAJECTORY-OUT COST EVALUATION
-------------------------------------------------
With only five evaluation trajectories, a fixed train/test split of the
evaluation set would leave too few runs on either side to mean anything.
Leave-one-out is the standard resampling answer at this sample size:

    for each evaluation trajectory i:
        choose the cost-optimal threshold using the OTHER FOUR trajectories
        report the cost incurred on trajectory i at that threshold

The reported total is then the sum of five costs, each incurred on a
trajectory that took no part in selecting the threshold applied to it.
This is genuinely out-of-sample with respect to threshold selection. It is
NOT out-of-sample with respect to the development trajectories, which were
already excluded from the evaluation set upstream.

The in-sample figure (one threshold chosen and reported on all five) is
reported alongside it, so the size of the optimism is visible rather than
hidden.

PART B — COST-RATIO SWEEP
-------------------------
Rather than one declared saving, expected cost is swept against the ratio

    R = reactive cost / predictive cost

holding the predictive cost fixed and varying the reactive cost. For each
ratio the cost-optimal threshold is recomputed. This shows how the optimum
moves with the economics instead of asserting a single number, which is the
point the paper already makes qualitatively in its live dashboard.

PART C — PER-RUN LEAD AT 3.5 AND 4.0
------------------------------------
Direct answer to (c): per-trajectory lead time at both thresholds, so the
operating choice and the cost-optimal choice can be compared per asset.

COST MODEL (unchanged from the paper; all figures are declared assumptions)
  inspection        1,000 SAR per reference-period sequence
  predictive       80,000 SAR when lead >= planning horizon
  reactive        101,000 SAR when lead < planning horizon, or no alarm
  planning horizon      1.0 h of recorded (accelerated-life) time

Inputs : results/deviation_score_per_snapshot.csv
         results/features_per_snapshot.csv
Outputs: results/cost_leave_one_out.csv
         results/cost_ratio_sweep.csv
         results/lead_time_by_threshold.csv

Run:
    python src/cost_out_of_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CALIBRATION_WINDOWS = 200
SECONDS_PER_SNAPSHOT = 10
ALPHA = 0.2
PERSISTENCE = 3
SIGMA_K = 3.0
DEG_PERSISTENCE = 5

CANDIDATE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]
REPORT_THRESHOLDS = [3.5, 4.0]

INSPECTION_COST = 1000
PREDICTIVE_COST = 80000
REACTIVE_COST = 101000
PLANNING_HORIZON_H = 1.0

COST_RATIOS = [1.05, 1.15, 1.26, 1.5, 2.0, 3.0, 5.0, 10.0]

EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]
RUN_TO_BEARING = {
    "dev_run_0": "Bearing1_1", "dev_run_1": "Bearing1_2",
    "val_run_0": "Bearing1_3", "test_run_0": "Bearing1_4",
    "test_run_1": "Bearing1_5", "test_run_2": "Bearing1_6",
    "test_run_3": "Bearing1_7",
}


def windows_to_hours(n):
    return n * SECONDS_PER_SNAPSHOT / 3600.0


def ewma(x, alpha=ALPHA):
    return pd.Series(x).ewm(alpha=alpha).mean().to_numpy()


def first_persistent(above, k, start=0):
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
                ev, fired = ev + 1, True
        else:
            streak, fired = 0, False
    return ev


def degradation_reference(feats):
    deg = {}
    for run, g in feats.groupby("run", sort=False):
        g = g.sort_values("window_index").reset_index(drop=True)
        n = len(g)
        above = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            cal = g[ch].iloc[:CALIBRATION_WINDOWS]
            above |= g[ch].to_numpy() > cal.mean() + SIGMA_K * cal.std()
        idx = first_persistent(above, DEG_PERSISTENCE, start=CALIBRATION_WINDOWS)
        deg[run] = idx if idx is not None else n
    return deg


def run_outcome(smoothed, deg_start, threshold):
    """Reference-period sequences and lead time for one run at one threshold."""
    above = smoothed >= threshold
    n = len(smoothed)
    seqs = alarm_events(above, PERSISTENCE, CALIBRATION_WINDOWS, deg_start)
    fi = first_persistent(above, PERSISTENCE, start=deg_start)
    lead = None if fi is None else windows_to_hours(n - fi)
    return seqs, lead


def run_cost(seqs, lead, reactive=REACTIVE_COST, predictive=PREDICTIVE_COST,
             inspection=INSPECTION_COST, horizon=PLANNING_HORIZON_H):
    cost = seqs * inspection
    if lead is None or lead < horizon:
        cost += reactive
    else:
        cost += predictive
    return cost


def main():
    scores = pd.read_csv(RESULTS / "deviation_score_per_snapshot.csv")
    feats = pd.read_csv(RESULTS / "features_per_snapshot.csv")
    deg = degradation_reference(feats)

    smoothed = {}
    for run in EVAL_RUNS:
        s = (scores[scores["run"] == run]
             .sort_values("window_index")["deviation_score"].to_numpy())
        smoothed[run] = ewma(s)

    # outcome[(run, thr)] = (sequences, lead)
    outcome = {(run, thr): run_outcome(smoothed[run], deg[run], thr)
               for run in EVAL_RUNS for thr in CANDIDATE_THRESHOLDS}

    # ------------------------------- PART A -------------------------------
    def total_cost(runs, thr, reactive=REACTIVE_COST):
        return sum(run_cost(*outcome[(r, thr)], reactive=reactive) for r in runs)

    # in-sample: one threshold chosen and reported on all five
    in_sample_costs = {t: total_cost(EVAL_RUNS, t) for t in CANDIDATE_THRESHOLDS}
    in_sample_thr = min(in_sample_costs, key=in_sample_costs.get)
    in_sample_total = in_sample_costs[in_sample_thr]

    loo_rows, loo_total = [], 0
    for held in EVAL_RUNS:
        others = [r for r in EVAL_RUNS if r != held]
        costs = {t: total_cost(others, t) for t in CANDIDATE_THRESHOLDS}
        chosen = min(costs, key=costs.get)
        seqs, lead = outcome[(held, chosen)]
        c = run_cost(seqs, lead)
        loo_total += c
        loo_rows.append({
            "held_out_run": held,
            "bearing": RUN_TO_BEARING[held],
            "threshold_chosen_on_other_four": chosen,
            "ref_sequences": seqs,
            "lead_h": None if lead is None else round(lead, 2),
            "outcome": "planned" if (lead is not None and lead >= PLANNING_HORIZON_H)
                       else "reactive",
            "cost_sar": c,
        })

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(RESULTS / "cost_leave_one_out.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 94)
    print("PART A — LEAVE-ONE-TRAJECTORY-OUT COST EVALUATION")
    print("each trajectory's cost is incurred at a threshold chosen on the OTHER four")
    print("=" * 94)
    print(loo_df.to_string(index=False))
    print(f"\n  out-of-sample total (5 trajectories) : {loo_total:,} SAR")
    print(f"  in-sample optimum threshold          : {in_sample_thr}")
    print(f"  in-sample total at that threshold    : {in_sample_total:,} SAR")
    diff = loo_total - in_sample_total
    print(f"  optimism (in-sample understates by)  : {diff:,} SAR "
          f"({100 * diff / in_sample_total:+.1f}%)")

    # ------------------------------- PART B -------------------------------
    sweep_rows = []
    for ratio in COST_RATIOS:
        reactive = PREDICTIVE_COST * ratio
        costs = {t: total_cost(EVAL_RUNS, t, reactive=reactive)
                 for t in CANDIDATE_THRESHOLDS}
        best = min(costs, key=costs.get)

        # leave-one-out total at this ratio
        loo_at_ratio = 0
        for held in EVAL_RUNS:
            others = [r for r in EVAL_RUNS if r != held]
            oc = {t: total_cost(others, t, reactive=reactive)
                  for t in CANDIDATE_THRESHOLDS}
            ch = min(oc, key=oc.get)
            loo_at_ratio += run_cost(*outcome[(held, ch)], reactive=reactive)

        sweep_rows.append({
            "cost_ratio_reactive_over_predictive": ratio,
            "reactive_cost_sar": int(reactive),
            "cost_optimal_threshold": best,
            "in_sample_total_sar": int(costs[best]),
            "leave_one_out_total_sar": int(loo_at_ratio),
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS / "cost_ratio_sweep.csv", index=False)

    print("\n" + "=" * 94)
    print("PART B — COST-RATIO SWEEP")
    print("predictive cost held at 80,000 SAR; reactive cost varied")
    print("=" * 94)
    print(sweep_df.to_string(index=False))
    thrs = sweep_df["cost_optimal_threshold"].unique()
    print(f"\n  cost-optimal threshold takes {len(thrs)} distinct value(s) "
          f"across the swept range: {sorted(thrs)}")

    # ------------------------------- PART C -------------------------------
    lead_rows = []
    for run in EVAL_RUNS:
        row = {"run": run, "bearing": RUN_TO_BEARING[run]}
        for thr in REPORT_THRESHOLDS:
            seqs, lead = outcome[(run, thr)]
            row[f"lead_h_at_{thr}"] = None if lead is None else round(lead, 2)
            row[f"ref_seq_at_{thr}"] = seqs
        lead_rows.append(row)
    lead_df = pd.DataFrame(lead_rows)
    lead_df.to_csv(RESULTS / "lead_time_by_threshold.csv", index=False)

    print("\n" + "=" * 94)
    print("PART C — PER-RUN LEAD AT 3.5 (operating) AND 4.0 (cost-optimal)")
    print("=" * 94)
    print(lead_df.to_string(index=False))
    for thr in REPORT_THRESHOLDS:
        vals = [r[f"lead_h_at_{thr}"] for r in lead_rows
                if r[f"lead_h_at_{thr}"] is not None]
        if vals:
            print(f"  median lead at {thr}: {np.median(vals):.2f} h "
                  f"({len(vals)}/{len(EVAL_RUNS)} trajectories)")

    print("\nSaved cost_leave_one_out.csv, cost_ratio_sweep.csv, "
          "lead_time_by_threshold.csv")
    print("\nHow to read Part A: the optimism figure is the honest quantity. If it is")
    print("small, the in-sample optimum was not badly overfitted and the paper can")
    print("say so with a number attached. If large, report the out-of-sample total")
    print("as the headline instead.")


if __name__ == "__main__":
    main()
