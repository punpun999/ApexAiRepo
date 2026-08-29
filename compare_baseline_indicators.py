"""
APEX-AI — Health-Indicator Baseline Comparison  [reviewer revision #1]

WHAT THE REVIEWERS ASKED FOR
----------------------------
Both reviewers identified the same primary gap: the ten-feature HDS is
never compared against a simpler indicator, so the reader cannot tell
whether aggregating ten standardized features adds value over, say, RMS
alone. Reviewer 2 additionally asked for:

  - the full Mahalanobis distance, since the mean absolute z-score used
    by HDS is its diagonal-covariance special case;
  - health-indicator quality metrics (monotonicity, trendability,
    robustness) following Lei et al. (2018), to substantiate the
    interpretability claim quantitatively;
  - all of it under the IDENTICAL protocol already used for HDS.

This script does exactly that. Every indicator below is referenced to the
same fixed 200-snapshot commissioning window, smoothed with the same
EWMA (alpha = 0.2), and evaluated with the same warning threshold logic
and three-snapshot persistence rule, against the same independent
degradation-onset reference. Only the indicator definition changes.

INDICATORS COMPARED
-------------------
  hds_10feature   The paper's indicator: mean |z-score| over 10 features.
  rms_only        z-score of rms_ch0 only. The simplest conventional
                  vibration indicator, and the baseline R1 named.
  best_single     Whichever single feature scores best on monotonicity;
                  reported so the comparison is not straw-manned by
                  picking a weak single feature.
  mahalanobis     Full Mahalanobis distance from the commissioning mean
                  using the commissioning covariance (pseudo-inverse,
                  ridge-regularized). HDS is its diagonal special case,
                  so this tests whether modelling feature correlation
                  helps.

QUALITY METRICS (Lei et al. 2018)
---------------------------------
  monotonicity  |#increasing steps - #decreasing steps| / (n-1).
                Higher = the indicator moves consistently in one
                direction as degradation proceeds.
  trendability  |Spearman rho| between indicator and snapshot index.
                Higher = stronger monotonic trend with consumed life.
  robustness    mean exp(-|residual| / |smoothed|) after EWMA smoothing.
                Higher = less erratic around its own trend.

NOTE ON THRESHOLDS
------------------
Indicators live on different scales, so a single fixed threshold (3.5)
is not comparable across them. Each indicator is therefore ALSO given a
threshold calibrated the same way the paper calibrates HDS: the P99.9 of
its own commissioning-period values on the TWO DEVELOPMENT trajectories
only. That keeps the comparison fair and keeps threshold selection out
of the evaluation set. Both the fixed-3.5 and the calibrated-threshold
results are reported.

Inputs : features_per_snapshot.csv  (already produced by
         health_deviation_score.py)
Outputs: baseline_indicator_quality.csv   (Lei et al. metrics per run)
         baseline_indicator_alarms.csv    (per-run alarm behaviour)
         baseline_indicator_summary.csv   (headline comparison table)

Run:
    python compare_baseline_indicators.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CALIBRATION_WINDOWS = 200
SECONDS_PER_SNAPSHOT = 10
ALPHA = 0.2
PERSISTENCE = 3
FIXED_THRESHOLD = 3.5
SIGMA_K = 3.0
DEG_PERSISTENCE = 5
RIDGE = 1e-6

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


# ------------------------------------------------- Lei et al. (2018) metrics
def monotonicity(x):
    d = np.diff(x)
    if len(d) == 0:
        return np.nan
    return abs((d > 0).sum() - (d < 0).sum()) / len(d)


def trendability(x):
    rho, _ = spearmanr(np.arange(len(x)), x)
    return abs(float(rho))


def robustness(x):
    sm = ewma(x)
    denom = np.abs(sm) + 1e-9
    return float(np.mean(np.exp(-np.abs(x - sm) / denom)))


# ----------------------------------------------------------- indicators
def build_indicators(feat_matrix):
    """feat_matrix: (n_windows, 10). Returns dict name -> score series."""
    cal = feat_matrix[:CALIBRATION_WINDOWS]
    mu, sd = cal.mean(axis=0), cal.std(axis=0)
    z = (feat_matrix - mu) / (sd + 1e-9)

    out = {}
    # the paper's indicator
    out["hds_10feature"] = np.mean(np.abs(z), axis=1)
    # simplest conventional baseline
    out["rms_only"] = np.abs(z[:, FEATURE_NAMES.index("rms_ch0")])
    # every single feature, so the best one can be picked afterwards
    for j, name in enumerate(FEATURE_NAMES):
        out[f"single::{name}"] = np.abs(z[:, j])

    # full Mahalanobis distance from the commissioning distribution
    cov = np.cov(cal, rowvar=False)
    cov = cov + RIDGE * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    delta = feat_matrix - mu
    out["mahalanobis"] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv, delta), 0))
    return out


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


def evaluate_alarms(series_by_run, threshold, deg):
    """Alarm behaviour on the 5 evaluation runs at a given threshold."""
    fa_total, leads, detected = 0, [], 0
    per_run = []
    for run in EVAL_RUNS:
        sm = ewma(series_by_run[run])
        above = sm >= threshold
        n, ds = len(sm), deg[run]
        fa = alarm_events(above, PERSISTENCE, CALIBRATION_WINDOWS, ds)
        fi = first_persistent(above, PERSISTENCE, start=ds)
        lead = None if fi is None else windows_to_hours(n - fi)
        if fi is not None:
            detected += 1
            leads.append(lead)
        fa_total += fa
        per_run.append({
            "run": run, "bearing": RUN_TO_BEARING[run],
            "false_alarm_events": fa,
            "lead_hours": None if lead is None else round(lead, 2),
        })
    return {
        "detected": f"{detected}/{len(EVAL_RUNS)}",
        "false_alarm_events": fa_total,
        "median_lead_h": round(float(np.median(leads)), 2) if leads else None,
        "min_lead_h": round(float(np.min(leads)), 2) if leads else None,
        "max_lead_h": round(float(np.max(leads)), 2) if leads else None,
    }, per_run


def main():
    feats = pd.read_csv("features_per_snapshot.csv")
    deg = degradation_reference(feats)

    # --- build every indicator for every run ---
    indicators_by_run = {}
    for run in RUN_TO_BEARING:
        g = feats[feats["run"] == run].sort_values("window_index")
        mat = g[FEATURE_NAMES].to_numpy()
        indicators_by_run[run] = build_indicators(mat)

    all_names = list(indicators_by_run["dev_run_0"].keys())

    # --- Lei et al. quality metrics, per run per indicator ---
    qrows = []
    for run in RUN_TO_BEARING:
        for name in all_names:
            x = indicators_by_run[run][name]
            qrows.append({
                "run": run, "bearing": RUN_TO_BEARING[run],
                "role": "development" if run in TRAIN_RUNS else "evaluation",
                "indicator": name,
                "monotonicity": monotonicity(x),
                "trendability": trendability(x),
                "robustness": robustness(x),
            })
    qdf = pd.DataFrame(qrows)
    qdf.round(4).to_csv("baseline_indicator_quality.csv", index=False)

    # --- pick the best single feature by mean monotonicity on DEV runs only ---
    dev_single = qdf[(qdf["role"] == "development")
                     & (qdf["indicator"].str.startswith("single::"))]
    best_single = (dev_single.groupby("indicator")["monotonicity"].mean()
                   .sort_values(ascending=False).index[0])
    best_single_feature = best_single.split("::")[1]
    print(f"Best single feature by development monotonicity: {best_single_feature}\n")

    COMPARED = {
        "HDS (10 features)": "hds_10feature",
        "RMS only": "rms_only",
        f"Best single ({best_single_feature})": best_single,
        "Mahalanobis (full covariance)": "mahalanobis",
    }

    # --- headline comparison ---
    rows = []
    alarm_detail = []
    for label, key in COMPARED.items():
        series = {run: indicators_by_run[run][key] for run in RUN_TO_BEARING}

        ev = qdf[(qdf["role"] == "evaluation") & (qdf["indicator"] == key)]
        # threshold calibrated on development commissioning windows only
        dev_cal = np.concatenate([series[r][:CALIBRATION_WINDOWS] for r in TRAIN_RUNS])
        cal_thr = float(np.percentile(dev_cal, 99.9))

        fixed_res, fixed_runs = evaluate_alarms(series, FIXED_THRESHOLD, deg)
        cal_res, cal_runs = evaluate_alarms(series, cal_thr, deg)

        for r in fixed_runs:
            alarm_detail.append({**r, "indicator": label, "threshold_type": "fixed 3.5",
                                 "threshold": FIXED_THRESHOLD})
        for r in cal_runs:
            alarm_detail.append({**r, "indicator": label, "threshold_type": "calibrated P99.9",
                                 "threshold": round(cal_thr, 3)})

        rows.append({
            "indicator": label,
            "monotonicity": round(ev["monotonicity"].median(), 4),
            "trendability": round(ev["trendability"].median(), 4),
            "robustness": round(ev["robustness"].median(), 4),
            "calibrated_threshold": round(cal_thr, 3),
            "cal_detected": cal_res["detected"],
            "cal_false_alarms": cal_res["false_alarm_events"],
            "cal_median_lead_h": cal_res["median_lead_h"],
            "fixed35_detected": fixed_res["detected"],
            "fixed35_false_alarms": fixed_res["false_alarm_events"],
            "fixed35_median_lead_h": fixed_res["median_lead_h"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv("baseline_indicator_summary.csv", index=False)
    pd.DataFrame(alarm_detail).to_csv("baseline_indicator_alarms.csv", index=False)

    # ------------------------------- report -------------------------------
    pd.set_option("display.width", 200)
    print("=" * 100)
    print("HEALTH-INDICATOR QUALITY (Lei et al. 2018) — median over 5 evaluation trajectories")
    print("=" * 100)
    print(summary[["indicator", "monotonicity", "trendability", "robustness"]]
          .to_string(index=False))

    print("\n" + "=" * 100)
    print("ALARM BEHAVIOUR — threshold calibrated per indicator (P99.9 of development")
    print("commissioning windows), identical EWMA + 3-snapshot persistence protocol")
    print("=" * 100)
    print(summary[["indicator", "calibrated_threshold", "cal_detected",
                   "cal_false_alarms", "cal_median_lead_h"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("ALARM BEHAVIOUR — the paper's fixed 3.5 threshold applied to every indicator")
    print("(shown for completeness; indicators are on different scales, so the")
    print("calibrated comparison above is the fair one)")
    print("=" * 100)
    print(summary[["indicator", "fixed35_detected", "fixed35_false_alarms",
                   "fixed35_median_lead_h"]].to_string(index=False))

    print("\nSaved baseline_indicator_summary.csv, baseline_indicator_quality.csv,")
    print("baseline_indicator_alarms.csv")
    print("\nHow to read this: if HDS shows higher monotonicity/trendability or fewer")
    print("false alarms at comparable lead time than RMS-only, the ten-feature")
    print("aggregation is justified. If RMS-only matches it, say so plainly — that is")
    print("still a publishable, honest finding and answers the reviewers directly.")


if __name__ == "__main__":
    main()
