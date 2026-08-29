"""
APEX-AI — Recompute Section 4.1 statistics and Table 2

WHY THIS IS NEEDED
------------------
The calibration fix changed the HDS values themselves (the baseline is now
the first 200 snapshots instead of the first 10% of each run). Every
statistic derived from HDS therefore needs recomputing, not just the alarm
thresholds:

  - Spearman correlation between snapshot index and HDS (paper: median 0.860)
  - Late-to-early decile ratio            (paper: median 6.56)
  - Per-bearing early/late medians         (paper Table 2 columns)
  - Per-bearing lead to end at 3.5 / k=3   (paper Table 2 last column)

These appear in the abstract, Section 4.1, Table 2, and the conclusion.

METHOD (matches the paper's stated definitions)
-----------------------------------------------
  - rho: Spearman correlation of window_index vs HDS, per trajectory
  - early median: median HDS over the FIRST 10% of the trajectory
  - late  median: median HDS over the FINAL 10% of the trajectory
    (note: the 10% here is the paper's descriptive early-vs-late statistic,
     which is separate from the calibration window and is unchanged)
  - ratio: late median / early median
  - bootstrap: 20,000 resamples of complete trajectories, seed 42
  - lead to end: EWMA-smoothed HDS, threshold 3.5, 3 consecutive windows,
    searched from the independent 3-sigma-on-RMS degradation start, then
    (n - first_alarm) converted to hours. Matches evaluate_thresholds_v2.py.

Inputs : deviation_score_per_snapshot.csv, features_per_snapshot.csv
Output : table2_per_bearing_recomputed.csv + printed summary

Run:
    python recompute_section41_table2.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CALIBRATION_WINDOWS = 200
SECONDS_PER_SNAPSHOT = 10
ALPHA = 0.2
WARNING_THRESHOLD = 3.5
PERSISTENCE = 3
SIGMA_K = 3.0
DEG_PERSISTENCE = 5
DECILE_FRACTION = 0.10
N_BOOTSTRAP = 20000
SEED = 42

# rul_datasets Condition 1 split -> original FEMTO bearing names
RUN_TO_BEARING = {
    "dev_run_0": "Bearing1_1",
    "dev_run_1": "Bearing1_2",
    "val_run_0": "Bearing1_3",
    "test_run_0": "Bearing1_4",
    "test_run_1": "Bearing1_5",
    "test_run_2": "Bearing1_6",
    "test_run_3": "Bearing1_7",
}
ROLE = {
    "dev_run_0": "Development", "dev_run_1": "Development",
    "val_run_0": "Evaluation", "test_run_0": "Evaluation",
    "test_run_1": "Evaluation", "test_run_2": "Evaluation",
    "test_run_3": "Evaluation",
}
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]


def windows_to_hours(n):
    return n * SECONDS_PER_SNAPSHOT / 3600.0


def ewma(x, alpha):
    return pd.Series(x).ewm(alpha=alpha).mean().to_numpy()


def first_persistent(above, k, start=0):
    streak = 0
    for i in range(start, len(above)):
        streak = streak + 1 if above[i] else 0
        if streak >= k:
            return i - k + 1
    return None


def degradation_reference(feats):
    deg = {}
    for run, g in feats.groupby("run", sort=False):
        g = g.sort_values("window_index").reset_index(drop=True)
        n = len(g)
        above = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            cal = g[ch].iloc[:CALIBRATION_WINDOWS]
            above |= (g[ch].to_numpy() > cal.mean() + SIGMA_K * cal.std())
        idx = first_persistent(above, DEG_PERSISTENCE, start=CALIBRATION_WINDOWS)
        deg[run] = idx if idx is not None else n
    return deg


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=SEED):
    """Run-level nonparametric bootstrap of the MEDIAN across trajectories."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    meds = [np.median(rng.choice(vals, size=len(vals), replace=True))
            for _ in range(n_boot)]
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def main():
    scores = pd.read_csv("deviation_score_per_snapshot.csv")
    feats = pd.read_csv("features_per_snapshot.csv")
    deg = degradation_reference(feats)

    rows = []
    for run in RUN_TO_BEARING:
        g = scores[scores["run"] == run].sort_values("window_index")
        hds = g["deviation_score"].to_numpy()
        n = len(hds)

        rho, _ = spearmanr(np.arange(n), hds)

        k_dec = max(1, int(round(n * DECILE_FRACTION)))
        early_med = float(np.median(hds[:k_dec]))
        late_med = float(np.median(hds[-k_dec:]))
        ratio = late_med / early_med if early_med > 0 else np.nan

        # lead to end at 3.5 / k=3, smoothed, from degradation start
        sm = ewma(hds, ALPHA)
        above = sm >= WARNING_THRESHOLD
        fa = first_persistent(above, PERSISTENCE, start=deg[run])
        lead = None if fa is None else round(windows_to_hours(n - fa), 2)

        rows.append({
            "Bearing": RUN_TO_BEARING[run],
            "Role": ROLE[run],
            "Snapshots": n,
            "rho": round(float(rho), 3),
            "Early median": round(early_med, 3),
            "Late median": round(late_med, 3),
            "Lead to end (h)": lead,
            "_run": run,
            "_ratio": ratio,
        })

    df = pd.DataFrame(rows)
    out = df.drop(columns=["_run", "_ratio"])
    out.to_csv("table2_per_bearing_recomputed.csv", index=False)

    print("=" * 74)
    print("TABLE 2 REPLACEMENT — per-bearing HDS summaries")
    print("(threshold 3.5, k = 3, fixed 200-snapshot calibration)")
    print("=" * 74)
    print(out.to_string(index=False))

    rhos = df["rho"].to_numpy()
    ratios = df["_ratio"].to_numpy()
    lo_r, hi_r = bootstrap_ci(rhos)
    lo_q, hi_q = bootstrap_ci(ratios)

    print("\n" + "=" * 74)
    print("SECTION 4.1 STATISTICS")
    print("=" * 74)
    print(f"Median Spearman rho          : {np.median(rhos):.3f}")
    print(f"  range                      : {rhos.min():.3f}-{rhos.max():.3f}")
    print(f"  bootstrap 95% interval     : {lo_r:.3f}-{hi_r:.3f}")
    print(f"Median late-to-early ratio   : {np.median(ratios):.2f}")
    print(f"  range                      : {ratios.min():.2f}-{ratios.max():.2f}")
    print(f"  bootstrap 95% interval     : {lo_q:.2f}-{hi_q:.2f}")
    print(f"All trajectories rho > 0     : {bool((rhos > 0).all())}")
    print(f"Late median > early median   : "
          f"{int((df['Late median'] > df['Early median']).sum())}/7 trajectories")

    eval_leads = [r["Lead to end (h)"] for r in rows
                  if r["_run"] in EVAL_RUNS and r["Lead to end (h)"] is not None]
    print(f"\nMedian lead, 5 evaluation runs: {np.median(eval_leads):.2f} h "
          f"(should equal 1.24 — cross-check)")

    print("\nSaved table2_per_bearing_recomputed.csv")
    print("\nCompare against the paper's current Table 2 and Section 4.1 —")
    print("the abstract and conclusion also quote the median rho and ratio.")


if __name__ == "__main__":
    main()
