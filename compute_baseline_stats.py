"""
APEX-AI — Healthy Baseline Statistics per Run

Jory's request: for each bearing run, compute the mean and standard
deviation of the engineered features (RMS, kurtosis, crest factor,
dominant frequency, etc.) over the FIRST 10% of that run's snapshots —
i.e. the period where the bearing is still healthy.

Why this matters: once we know what "healthy" looks like for each
bearing, we can later measure how far each later snapshot deviates from
that baseline. A rising deviation over time = a rising health/anomaly
score, which is a natural thing to show on a dashboard alongside (or
instead of) the raw RUL prediction.

Output: baseline_stats.csv — one row per run, with the healthy-period
mean and std for every feature. This is the dashboard-ready table Jory
asked for.

Run:
    python compute_baseline_stats.py
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq

import rul_datasets

SAMPLE_RATE_HZ = 25600
HEALTHY_FRACTION = 0.10  # first 10% of each run's snapshots

FEATURE_NAMES = [
    "rms_ch0", "peak_ch0", "crest_factor_ch0", "kurtosis_ch0", "dominant_freq_ch0",
    "rms_ch1", "peak_ch1", "crest_factor_ch1", "kurtosis_ch1", "dominant_freq_ch1",
]


def extract_features(window: np.ndarray) -> np.ndarray:
    """Same feature extraction as apex_v1_baseline.py — kept identical on
    purpose, so these baseline stats line up with the model's own features."""
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


def main():
    print("Loading FEMTO condition 1...")
    reader = rul_datasets.reader.FemtoReader(fd=1)
    reader.prepare_data()

    # Combine dev + val + test runs so we get baseline stats for every run,
    # not just the training ones.
    all_features, all_labels, run_names = [], [], []
    for split in ("dev", "val", "test"):
        feats, labels = reader.load_split(split)
        for i, (f, l) in enumerate(zip(feats, labels)):
            all_features.append(f)
            all_labels.append(l)
            run_names.append(f"{split}_run_{i}")

    rows = []
    for run_name, run_windows in zip(run_names, all_features):
        n_windows = run_windows.shape[0]
        n_healthy = max(1, int(n_windows * HEALTHY_FRACTION))
        healthy_windows = run_windows[:n_healthy]

        # Extract features for every window in the healthy period
        healthy_feats = np.array([extract_features(w) for w in healthy_windows])

        row = {"run": run_name, "n_windows_total": n_windows, "n_windows_healthy": n_healthy}
        for i, name in enumerate(FEATURE_NAMES):
            row[f"{name}_mean"] = healthy_feats[:, i].mean()
            row[f"{name}_std"] = healthy_feats[:, i].std()
        rows.append(row)
        print(f"Processed {run_name}: {n_healthy}/{n_windows} windows used for baseline")

    df = pd.DataFrame(rows)
    df.to_csv("baseline_stats.csv", index=False)
    print(f"\nSaved baseline_stats.csv with {len(df)} runs.")
    print(df.head())


if __name__ == "__main__":
    main()
