"""
APEX-AI — Commissioning-Window Sensitivity and Robust Scaling
[reviewer revision items: minor polish, R2]

TWO REVIEWER REQUESTS, ONE SCRIPT
---------------------------------
R2 asked, under "minor polish":

  (a) "add a commissioning-window sensitivity check, since the
       200-snapshot window is fixed a priori and untested"
  (b) "given the noted instability of dominant-frequency z-scores, add a
       brief robust-scaling (median or MAD) comparison"

Both are the same experiment shape: rerun the HDS pipeline with one
design choice varied, then compare indicator quality and alarm behaviour
under the protocol already used in the paper. They are combined here so
the two results are directly comparable and share one code path.

PART A — COMMISSIONING-WINDOW SENSITIVITY
-----------------------------------------
Recomputes HDS for each candidate window length. Everything downstream
is recomputed consistently at each length, which matters and is easy to
get wrong: the commissioning window defines the baseline mean/std, the
calibrated threshold (P99.9 of development commissioning values), the
start of the assumed-healthy zone for counting reference-period
sequences, AND the calibration slice of the independent 3-sigma RMS
degradation reference. All five move together here.

Window cap: the shortest development trajectory (Bearing1_2) has 871
snapshots, so a 400-snapshot window is already 46% of it. The sweep
stops there; anything longer stops being a plausible commissioning
period for that asset.

PART B — ROBUST SCALING
-----------------------
Recomputes HDS with median and MAD in place of mean and standard
deviation:

    z_robust = (x - median_cal) / (1.4826 * MAD_cal)

The 1.4826 factor makes MAD a consistent estimator of the standard
deviation for normally distributed data, so the robust score stays on
roughly the same numeric scale as the standard one and the two are
comparable. Robust scaling should help most where a feature's
commissioning window contains outliers that inflate the standard
deviation — which is the concern R2 raised about dominant frequency.

Both parts report the same quantities as the main comparator table:
trendability, robustness, calibrated threshold, runs reaching the
criterion, reference-period sequences, and median lead.

Inputs : results/features_per_snapshot.csv
Outputs: results/commissioning_window_sensitivity.csv
         results/robust_scaling_comparison.csv

Run:
    python src/sensitivity_and_robust_scaling.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SECONDS_PER_SNAPSHOT = 10
ALPHA = 0.2
PERSISTENCE = 3
SIGMA_K = 3.0
DEG_PERSISTENCE = 5
MAD_SCALE = 1.4826
MAD_FLOOR_FRACTION = 0.01   # floor MAD at 1% of the feature's commissioning median

BASELINE_WINDOW = 200
CANDIDATE_WINDOWS = [100, 150, 200, 300, 400]

FEATURE_NAMES = [
    "rms_ch0", "peak_ch0", "crest_factor_ch0", "kurtosis_ch0", "dominant_freq_ch0",
    "rms_ch1", "peak_ch1", "crest_factor_ch1", "kurtosis_ch1", "dominant_freq_ch1",
]

RUN_TO_BEARING = {
    "dev_run_0": "Bearing1_1", "dev_run_1": "Bearing1_2",
    "val_run_0": "Bearing1_3", "test_run_0": "Bearing1_4",
    "test_run_1": "Bearing1_5", "test_run_2": "Bearing1_6",
    "test_run_3": "Bearing1_7",
}
TRAIN_RUNS = ["dev_run_0", "dev_run_1"]
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]


# ----------------------------------------------------------------- helpers
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


def trendability(x):
    rho, _ = spearmanr(np.arange(len(x)), x)
    return abs(float(rho))


def robustness(x):
    sm = ewma(x)
    return float(np.mean(np.exp(-np.abs(x - sm) / (np.abs(sm) + 1e-9))))


def monotonicity(x):
    d = np.diff(x)
    return abs((d > 0).sum() - (d < 0).sum()) / len(d) if len(d) else np.nan


# ------------------------------------------------------------- HDS variants
def hds_standard(mat, cal_n):
    cal = mat[:cal_n]
    mu, sd = cal.mean(axis=0), cal.std(axis=0)
    return np.mean(np.abs((mat - mu) / (sd + 1e-9)), axis=1)


def hds_robust(mat, cal_n):
    """Median/MAD standardization instead of mean/std."""
    cal = mat[:cal_n]
    med = np.median(cal, axis=0)
    mad = np.median(np.abs(cal - med), axis=0) * MAD_SCALE
    return np.mean(np.abs((mat - med) / (mad + 1e-9)), axis=1)


def hds_robust_floored(mat, cal_n):
    """Median/MAD with a floor on the scale.

    Plain median/MAD fails on this feature set: when a feature is very
    stable during commissioning its MAD approaches zero, the division
    inflates the z-score enormously, and the score becomes dominated by
    whichever feature happened to be most stable. This variant floors the
    MAD scale at a small fraction of the feature's own commissioning
    median, so a degenerate scale cannot dominate. It isolates whether the
    failure is intrinsic to robust scaling or only to the degenerate-MAD
    case.
    """
    cal = mat[:cal_n]
    med = np.median(cal, axis=0)
    mad = np.median(np.abs(cal - med), axis=0) * MAD_SCALE
    floor = MAD_FLOOR_FRACTION * np.abs(med)
    scale = np.maximum(mad, floor)
    return np.mean(np.abs((mat - med) / (scale + 1e-9)), axis=1)


def degradation_reference(feats_by_run, cal_n):
    """3-sigma-on-RMS onset marker, recomputed for this window length."""
    deg = {}
    for run, g in feats_by_run.items():
        n = len(g)
        above = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            col = g[ch].to_numpy()
            cal = col[:cal_n]
            above |= col > cal.mean() + SIGMA_K * cal.std()
        idx = first_persistent(above, DEG_PERSISTENCE, start=cal_n)
        deg[run] = idx if idx is not None else n
    return deg


def evaluate(series_by_run, deg, cal_n, threshold):
    fa_total, leads, detected = 0, [], 0
    for run in EVAL_RUNS:
        sm = ewma(series_by_run[run])
        above = sm >= threshold
        n, ds = len(sm), deg[run]
        fa_total += alarm_events(above, PERSISTENCE, cal_n, ds)
        fi = first_persistent(above, PERSISTENCE, start=ds)
        if fi is not None:
            detected += 1
            leads.append(windows_to_hours(n - fi))
    return {
        "runs": f"{detected}/{len(EVAL_RUNS)}",
        "ref_sequences": fa_total,
        "median_lead_h": round(float(np.median(leads)), 2) if leads else None,
        "min_lead_h": round(float(np.min(leads)), 2) if leads else None,
        "max_lead_h": round(float(np.max(leads)), 2) if leads else None,
    }


def run_configuration(feats_by_run, mats, cal_n, score_fn):
    """One full pipeline pass at a given window length and scoring rule."""
    series = {run: score_fn(mats[run], cal_n) for run in RUN_TO_BEARING}
    deg = degradation_reference(feats_by_run, cal_n)

    dev_cal = np.concatenate([series[r][:cal_n] for r in TRAIN_RUNS])
    thr = float(np.percentile(dev_cal, 99.9))

    res = evaluate(series, deg, cal_n, thr)
    ev_trend = np.median([trendability(series[r]) for r in EVAL_RUNS])
    ev_robust = np.median([robustness(series[r]) for r in EVAL_RUNS])
    ev_mono = np.median([monotonicity(series[r]) for r in EVAL_RUNS])

    return {
        "calibrated_threshold": round(thr, 3),
        "trendability": round(float(ev_trend), 4),
        "robustness": round(float(ev_robust), 4),
        "monotonicity": round(float(ev_mono), 4),
        **res,
    }


def main():
    feats = pd.read_csv(RESULTS / "features_per_snapshot.csv")
    feats_by_run, mats = {}, {}
    for run in RUN_TO_BEARING:
        g = feats[feats["run"] == run].sort_values("window_index").reset_index(drop=True)
        feats_by_run[run] = g
        mats[run] = g[FEATURE_NAMES].to_numpy()

    shortest_dev = min(len(feats_by_run[r]) for r in TRAIN_RUNS)
    print(f"Shortest development trajectory: {shortest_dev} snapshots\n")

    # ---------------------------- PART A ----------------------------
    rows_a = []
    for w in CANDIDATE_WINDOWS:
        if w >= shortest_dev:
            print(f"skipping window {w} (>= shortest development run)")
            continue
        r = run_configuration(feats_by_run, mats, w, hds_standard)
        rows_a.append({
            "commissioning_windows": w,
            "minutes": round(w * SECONDS_PER_SNAPSHOT / 60, 1),
            "pct_of_shortest_dev_run": round(100 * w / shortest_dev, 1),
            **r,
        })

    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(RESULTS / "commissioning_window_sensitivity.csv", index=False)

    print("=" * 104)
    print("PART A — COMMISSIONING-WINDOW SENSITIVITY (standard mean/std HDS)")
    print("everything downstream recomputed per window: baseline, threshold,")
    print("healthy-zone start, and the 3-sigma onset reference")
    print("=" * 104)
    print(df_a.to_string(index=False))

    base = df_a[df_a["commissioning_windows"] == BASELINE_WINDOW]
    if not base.empty:
        b = base.iloc[0]
        spread_lead = df_a["median_lead_h"].dropna()
        print(f"\nAt the paper's {BASELINE_WINDOW}-snapshot window: threshold "
              f"{b['calibrated_threshold']}, {b['runs']} runs, "
              f"{b['ref_sequences']} reference-period sequences, "
              f"median lead {b['median_lead_h']} h")
        if len(spread_lead):
            print(f"Median lead across all windows: "
                  f"{spread_lead.min():.2f}-{spread_lead.max():.2f} h")

    # ---------------------------- PART B ----------------------------
    rows_b = []
    for label, fn in (("standard (mean/std)", hds_standard),
                      ("robust (median/MAD)", hds_robust),
                      ("robust (median/MAD, floored)", hds_robust_floored)):
        r = run_configuration(feats_by_run, mats, BASELINE_WINDOW, fn)
        rows_b.append({"scaling": label, **r})

    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(RESULTS / "robust_scaling_comparison.csv", index=False)

    print("\n" + "=" * 104)
    print(f"PART B — ROBUST SCALING at the {BASELINE_WINDOW}-snapshot window")
    print("median/MAD standardization vs the paper's mean/std")
    print("=" * 104)
    print(df_b.to_string(index=False))

    print("\nSaved commissioning_window_sensitivity.csv, robust_scaling_comparison.csv")
    print("\nHow to read Part A: if runs reaching the criterion and median lead stay")
    print("stable across window lengths, the 200-snapshot choice is not load-bearing")
    print("and the paper can say so. Large swings would mean the opposite, which is")
    print("also worth reporting honestly.")
    print("\nHow to read Part B: robust scaling is worth adopting only if it improves")
    print("trendability/robustness or reduces reference-period sequences without")
    print("costing lead time. If it does not, report that the simpler mean/std")
    print("standardization was retained on evidence.")


if __name__ == "__main__":
    main()
