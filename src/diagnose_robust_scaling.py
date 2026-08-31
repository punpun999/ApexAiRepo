"""
APEX-AI — Diagnostic: why does median/MAD scaling behave differently?

CONTEXT
-------
Replacing the commissioning mean/std with median/MAD produced a calibrated
threshold of 95.3 instead of 3.4, and reached the warning criterion in only
2 of 5 evaluation trajectories. A first hypothesis was that MAD collapses
toward zero for very stable features, inflating the z-scores. Flooring the
MAD scale at 1% of each feature's commissioning median changed NOTHING, so
that hypothesis is wrong: the floor never bound.

This script tests the remaining explanation directly. For each feature, in
each trajectory's 200-snapshot commissioning window, it reports:

    std                 the scale used by the paper's HDS
    mad_scaled          1.4826 * MAD, the scale used by the robust variant
    ratio               std / mad_scaled

If the ratio is consistently well above 1, the robust scale is
systematically smaller than the standard one, every robust z-score is
correspondingly larger, and the threshold rises accordingly. If the ratio
varies widely BETWEEN features, then the two formulations also weight the
ten features differently, which changes which feature dominates the mean
absolute z-score and therefore what the indicator responds to.

Both effects are checked. The output is descriptive only; nothing here
changes any reported result.

Inputs : results/features_per_snapshot.csv
Outputs: results/scaling_diagnostic.csv  + printed summary

Run:
    python src/diagnose_robust_scaling.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CALIBRATION_WINDOWS = 200
MAD_SCALE = 1.4826

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


def main():
    feats = pd.read_csv(RESULTS / "features_per_snapshot.csv")

    rows = []
    for run in RUN_TO_BEARING:
        g = (feats[feats["run"] == run]
             .sort_values("window_index")
             .head(CALIBRATION_WINDOWS))
        for f in FEATURE_NAMES:
            x = g[f].to_numpy()
            med = np.median(x)
            std = x.std()
            mad_scaled = np.median(np.abs(x - med)) * MAD_SCALE
            rows.append({
                "run": run,
                "bearing": RUN_TO_BEARING[run],
                "role": "development" if run in TRAIN_RUNS else "evaluation",
                "feature": f,
                "mean": float(x.mean()),
                "median": med,
                "std": std,
                "mad_scaled": mad_scaled,
                "ratio_std_over_mad": std / mad_scaled if mad_scaled > 0 else np.inf,
            })

    df = pd.DataFrame(rows)
    df.round(6).to_csv(RESULTS / "scaling_diagnostic.csv", index=False)

    pd.set_option("display.width", 200)

    print("=" * 92)
    print("PER-FEATURE SCALE COMPARISON IN THE COMMISSIONING WINDOW")
    print("median over the seven trajectories")
    print("=" * 92)
    summary = (df.groupby("feature")[["std", "mad_scaled", "ratio_std_over_mad"]]
                 .median()
                 .reindex(FEATURE_NAMES)
                 .round(4))
    print(summary.to_string())

    r = df["ratio_std_over_mad"].replace([np.inf, -np.inf], np.nan).dropna()
    print("\n" + "=" * 92)
    print("RATIO std / (1.4826 * MAD) — ACROSS ALL RUNS AND FEATURES")
    print("=" * 92)
    print(f"  median : {r.median():.3f}")
    print(f"  range  : {r.min():.3f} - {r.max():.3f}")
    print(f"  > 1    : {(r > 1).sum()} of {len(r)} feature-run pairs")
    print(f"  > 2    : {(r > 2).sum()} of {len(r)}")
    print(f"  > 5    : {(r > 5).sum()} of {len(r)}")

    print("\n" + "=" * 92)
    print("BETWEEN-FEATURE SPREAD OF THE RATIO (median across runs)")
    print("a wide spread means the two formulations weight the ten features")
    print("differently, not merely rescale them by a common factor")
    print("=" * 92)
    per_feat = summary["ratio_std_over_mad"].sort_values(ascending=False)
    print(per_feat.to_string())
    print(f"\n  max / min across features: "
          f"{per_feat.max() / per_feat.min():.2f}x")

    print("\n" + "=" * 92)
    print("SMALLEST MAD SCALES (checks the degenerate-MAD hypothesis)")
    print("=" * 92)
    smallest = df.nsmallest(5, "mad_scaled")[
        ["bearing", "feature", "median", "std", "mad_scaled", "ratio_std_over_mad"]]
    print(smallest.to_string(index=False))
    print("\nIf none of these mad_scaled values is near zero relative to its")
    print("feature median, the degenerate-scale explanation is ruled out and")
    print("the systematic std/MAD gap is the operative effect.")

    print("\nSaved scaling_diagnostic.csv")


if __name__ == "__main__":
    main()
