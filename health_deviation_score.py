"""
APEX-AI — Health Deviation Score (z-score based)  [v1.1]

CHANGES vs previous version (agreed with the team):

1. FIXED CALIBRATION PERIOD instead of "first 10% of the run".
   The old rule needed the TOTAL run length to compute 10% — i.e. it
   "looked into the future", which is impossible in real deployment.
   New rule: the healthy baseline is always the first
   CALIBRATION_WINDOWS snapshots (a fixed, known-in-advance period),
   exactly like commissioning a new machine for a fixed warm-up time.

   CALIBRATION_WINDOWS = 200 snapshots ~= 33 minutes of elapsed
   operation (FEMTO records one snapshot every 10 s). This is a team
   parameter — change it here if the industrial engineers prefer a
   different commissioning period.

2. PER-WINDOW FEATURES are now exported alongside the deviation score
   (features_per_snapshot.csv). The threshold-evaluation script needs
   raw RMS per window to compute the independent "degradation start"
   reference (3-sigma rule on RMS), without re-loading the raw dataset.

Outputs:
    deviation_score_per_snapshot.csv  (same schema as before)
    deviation_score_summary.csv       (same schema as before)
    features_per_snapshot.csv         (NEW: run, window_index, all 10 features)
    deviation_score_<run>.png         (one plot per run)

Run:
    python health_deviation_score.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq

import rul_datasets

SAMPLE_RATE_HZ = 25600
SECONDS_PER_SNAPSHOT = 10          # FEMTO: one 0.1 s recording every 10 s
CALIBRATION_WINDOWS = 200          # fixed healthy calibration period (~33 min)

FEATURE_NAMES = [
    "rms_ch0", "peak_ch0", "crest_factor_ch0", "kurtosis_ch0", "dominant_freq_ch0",
    "rms_ch1", "peak_ch1", "crest_factor_ch1", "kurtosis_ch1", "dominant_freq_ch1",
]


def extract_features(window: np.ndarray) -> np.ndarray:
    feats = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / (rms + 1e-9)
        kurt = kurtosis(signal)
        spectrum = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), d=1.0 / SAMPLE_RATE_HZ)
        dominant_freq = freqs[np.argmax(spectrum[1:]) + 1]
        feats.extend([rms, peak, crest_factor, kurt, dominant_freq])
    return np.array(feats)


def deviation_score(all_window_feats, baseline_mean, baseline_std):
    """Average absolute z-score across features. Higher = further from healthy."""
    z_scores = (all_window_feats - baseline_mean) / (baseline_std + 1e-9)
    return np.mean(np.abs(z_scores), axis=1)


def main():
    print("Loading FEMTO condition 1...")
    reader = rul_datasets.reader.FemtoReader(fd=1)
    reader.prepare_data()

    all_features, all_labels, run_names = [], [], []
    for split in ("dev", "val", "test"):
        feats, labels = reader.load_split(split)
        for i, (f, l) in enumerate(zip(feats, labels)):
            all_features.append(f)
            all_labels.append(l)
            run_names.append(f"{split}_run_{i}")

    summary_rows = []
    all_per_snapshot_rows = []
    all_feature_rows = []

    for run_name, run_windows, run_labels in zip(run_names, all_features, all_labels):
        n_windows = run_windows.shape[0]

        # Fixed calibration period — never depends on total run length.
        n_healthy = min(CALIBRATION_WINDOWS, n_windows)
        if n_healthy < CALIBRATION_WINDOWS:
            print(f"WARNING: {run_name} has only {n_windows} windows "
                  f"(< {CALIBRATION_WINDOWS}); using all of them as calibration.")

        # Features for EVERY window in this run
        all_feats = np.array([extract_features(w) for w in run_windows])

        # Baseline = mean/std over the fixed calibration period only
        baseline_mean = all_feats[:n_healthy].mean(axis=0)
        baseline_std = all_feats[:n_healthy].std(axis=0)

        scores = deviation_score(all_feats, baseline_mean, baseline_std)

        # --- per-snapshot deviation scores (same schema as before) ---
        all_per_snapshot_rows.append(pd.DataFrame({
            "run": run_name,
            "window_index": np.arange(n_windows),
            "rul_label": run_labels,
            "deviation_score": scores,
        }))

        # --- NEW: per-snapshot raw features (for the 3-sigma reference) ---
        feat_df = pd.DataFrame(all_feats, columns=FEATURE_NAMES)
        feat_df.insert(0, "window_index", np.arange(n_windows))
        feat_df.insert(0, "run", run_name)
        feat_df["n_calibration_windows"] = n_healthy
        all_feature_rows.append(feat_df)

        # --- plot ---
        plt.figure(figsize=(9, 4.5))
        plt.plot(scores)
        plt.axvline(n_healthy, color="green", linestyle="--",
                    label=f"end of calibration period ({n_healthy} windows)")
        plt.xlabel("Window index (time, 1 window = 10 s)")
        plt.ylabel("Deviation score (avg |z-score| across features)")
        plt.title(f"Health Deviation Score — {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"deviation_score_{run_name}.png", dpi=150)
        plt.close()

        summary_rows.append({
            "run": run_name,
            "n_windows": n_windows,
            "n_calibration_windows": n_healthy,
            "mean_deviation_score": scores.mean(),
            "max_deviation_score": scores.max(),
            "final_deviation_score": scores[-1],
        })
        print(f"{run_name}: mean={scores.mean():.2f}, max={scores.max():.2f}, "
              f"final={scores[-1]:.2f}")

    pd.DataFrame(summary_rows).to_csv("deviation_score_summary.csv", index=False)
    print("\nSaved deviation_score_summary.csv")

    per_snapshot_df = pd.concat(all_per_snapshot_rows, ignore_index=True)
    per_snapshot_df.to_csv("deviation_score_per_snapshot.csv", index=False)
    print(f"Saved deviation_score_per_snapshot.csv ({len(per_snapshot_df)} rows)")

    features_df = pd.concat(all_feature_rows, ignore_index=True)
    features_df.to_csv("features_per_snapshot.csv", index=False)
    print(f"Saved features_per_snapshot.csv ({len(features_df)} rows)")


if __name__ == "__main__":
    main()
