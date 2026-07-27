"""
APEX-AI — Health Deviation Score (z-score based)

Builds on compute_baseline_stats.py: instead of just knowing what
"healthy" looks like, this computes, for every single snapshot in a run,
how many standard deviations away it is from that run's own healthy
baseline (a z-score) — then combines all features into one overall
"deviation score" per snapshot.

This is the actual number you'd put on a dashboard: low and flat while
healthy, rising as the bearing degrades.

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
HEALTHY_FRACTION = 0.10

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
    """
    all_window_feats: (n_windows, n_features)
    Returns one deviation score per window: the average absolute z-score
    across all features. Higher = further from healthy.
    """
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
    for run_name, run_windows, run_labels in zip(run_names, all_features, all_labels):
        n_windows = run_windows.shape[0]
        n_healthy = max(1, int(n_windows * HEALTHY_FRACTION))

        # Extract features for EVERY window in this run
        all_feats = np.array([extract_features(w) for w in run_windows])

        # Baseline = mean/std over just the healthy period
        baseline_mean = all_feats[:n_healthy].mean(axis=0)
        baseline_std = all_feats[:n_healthy].std(axis=0)

        scores = deviation_score(all_feats, baseline_mean, baseline_std)

        # Save the per-snapshot scores (not just the run-level summary) so
        # every single window's deviation score is available, e.g. for a
        # dashboard that needs to plot or query individual points in time.
        per_snapshot_df = pd.DataFrame({
            "run": run_name,
            "window_index": np.arange(n_windows),
            "rul_label": run_labels,
            "deviation_score": scores,
        })
        all_per_snapshot_rows.append(per_snapshot_df)

        # Plot this run's deviation score over time
        plt.figure(figsize=(9, 4.5))
        plt.plot(scores)
        plt.axvline(n_healthy, color="green", linestyle="--",
                    label=f"end of healthy baseline period ({n_healthy} windows)")
        plt.xlabel("Window index (time)")
        plt.ylabel("Deviation score (avg |z-score| across features)")
        plt.title(f"Health Deviation Score — {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"deviation_score_{run_name}.png", dpi=150)
        plt.close()

        summary_rows.append({
            "run": run_name,
            "n_windows": n_windows,
            "mean_deviation_score": scores.mean(),
            "max_deviation_score": scores.max(),
            "final_deviation_score": scores[-1],
        })
        print(f"{run_name}: mean={scores.mean():.2f}, max={scores.max():.2f}, "
              f"final={scores[-1]:.2f} -> saved deviation_score_{run_name}.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("deviation_score_summary.csv", index=False)
    print("\nSaved deviation_score_summary.csv")
    print(summary_df)

    # Combine every run's per-snapshot scores into one file — this is the
    # full-resolution data (one row per snapshot, not just per-run summaries).
    per_snapshot_df = pd.concat(all_per_snapshot_rows, ignore_index=True)
    per_snapshot_df.to_csv("deviation_score_per_snapshot.csv", index=False)
    print(f"\nSaved deviation_score_per_snapshot.csv with {len(per_snapshot_df)} rows "
          f"(one row per snapshot, across all {len(run_names)} runs)")
    print(per_snapshot_df.head())


if __name__ == "__main__":
    main()
