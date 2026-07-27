"""
APEX-AI v1 baseline — Bearing degradation / RUL estimation on FEMTO (PRONOSTIA)

What this does:
1. Downloads + loads the FEMTO bearing dataset via the `rul_datasets` library
   (first run downloads automatically — no manual dataset wrangling needed).
2. Extracts hand-crafted vibration features per window (RMS, kurtosis, crest
   factor, dominant frequency) instead of feeding raw signal to a model —
   matches the FFT/frequency-domain pipeline described in the research doc.
3. Trains a simple Random Forest baseline to predict Remaining Useful Life
   (RUL) from those features.
4. Plots predicted vs. true RUL over one held-out run, so you can see whether
   the model is actually tracking degradation.

Setup (run once in your venv):
    pip install rul_datasets scikit-learn scipy matplotlib numpy

Run:
    python apex_v1_baseline.py

Notes:
- FemtoReader fd=1/2/3 correspond to the three FEMTO operating conditions
  (different loads/speeds). Start with fd=1.
- Each "window" is a 2560-sample vibration snapshot with 2 channels
  (horizontal + vertical accelerometer).
- Labels are RUL in cycles (roughly proportional to time-to-failure).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import rul_datasets

SAMPLE_RATE_HZ = 25600  # FEMTO accelerometer sampling rate


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    window shape: (2560, 2) — 2560 samples x 2 channels (horiz, vert)
    Returns a flat feature vector combining both channels.
    """
    feats = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]

        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / (rms + 1e-9)
        kurt = kurtosis(signal)

        # Dominant frequency via FFT — the "signature" of a developing fault
        spectrum = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), d=1.0 / SAMPLE_RATE_HZ)
        dominant_freq = freqs[np.argmax(spectrum[1:]) + 1]  # skip DC component

        feats.extend([rms, peak, crest_factor, kurt, dominant_freq])
    return np.array(feats)


def build_feature_table(features_list, labels_list):
    """
    features_list / labels_list: one entry per run (list of arrays), as
    returned by reader.load_split(). Flattens into one big (X, y) table.
    """
    X, y = [], []
    for run_features, run_labels in zip(features_list, labels_list):
        for window, rul in zip(run_features, run_labels):
            X.append(extract_features(window))
            y.append(rul)
    return np.array(X), np.array(y)


def main():
    print("Loading FEMTO (PRONOSTIA) bearing dataset — condition 1...")
    reader = rul_datasets.reader.FemtoReader(fd=1)
    reader.prepare_data()  # downloads on first run, cached after that

    train_features, train_labels = reader.load_split("dev")
    val_features, val_labels = reader.load_split("val")

    print(f"Training runs: {len(train_features)}, validation runs: {len(val_features)}")
    print("Extracting features (RMS, kurtosis, crest factor, dominant frequency)...")

    X_train, y_train = build_feature_table(train_features, train_labels)
    X_val, y_val = build_feature_table(val_features, val_labels)

    print(f"Training samples: {X_train.shape[0]}, feature dim: {X_train.shape[1]}")

    print("Training baseline Random Forest regressor...")
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Validation RMSE: {rmse:.2f} cycles")

    # Plot predicted vs. true RUL for the first validation run, so you can
    # see whether the model tracks the degradation trend over time.
    n_first_run = len(val_features[0])
    raw_preds_first_run = preds[:n_first_run]

    # Smooth the raw per-window predictions with a rolling average. Each
    # window's guess is noisy on its own, but the underlying wear level
    # barely changes between neighboring windows, so averaging nearby
    # predictions should cancel out a lot of that window-to-window jitter.
    SMOOTHING_WINDOW = 15  # number of neighboring predictions to average

    def rolling_average(values, window):
        values = np.asarray(values)
        kernel = np.ones(window) / window
        # "same" mode keeps the output the same length as the input, which
        # keeps this simple to plot against the true RUL on the same x-axis.
        return np.convolve(values, kernel, mode="same")

    smoothed_preds = rolling_average(raw_preds_first_run, SMOOTHING_WINDOW)

    plt.figure(figsize=(9, 5))
    plt.plot(range(n_first_run), val_labels[0], label="True RUL", linewidth=2)
    plt.plot(range(n_first_run), raw_preds_first_run, label="Predicted RUL (raw)",
              linewidth=1, alpha=0.35, color="orange")
    plt.plot(range(n_first_run), smoothed_preds,
              label=f"Predicted RUL (smoothed, window={SMOOTHING_WINDOW})",
              linewidth=2, color="darkorange")
    plt.xlabel("Window index (time)")
    plt.ylabel("RUL (cycles)")
    plt.title("APEX-AI v1 Baseline — Predicted vs. True RUL (FEMTO, condition 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rul_baseline_result.png", dpi=150)
    print("Saved plot to rul_baseline_result.png")

    smoothed_rmse = np.sqrt(mean_squared_error(val_labels[0], smoothed_preds))
    print(f"Raw prediction RMSE (this run): {rmse:.2f} cycles")
    print(f"Smoothed prediction RMSE (this run): {smoothed_rmse:.2f} cycles")


if __name__ == "__main__":
    main()
