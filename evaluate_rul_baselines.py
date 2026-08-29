"""
APEX-AI — RUL Evaluation with Baselines  [item 6]

WHAT THIS ADDS over apex_v1_baseline.py
---------------------------------------
The old baseline script printed a single validation RMSE to the console
and saved nothing. The paper therefore cannot report RUL numbers, and the
repository snapshot has no prediction arrays to audit. This script fixes
all of that:

  1. RMSE *and* MAE, reported PER RUN (not just pooled), for validation
     and test trajectories separately.
  2. Prediction arrays saved to CSV, so every reported number is
     reproducible and auditable from the repository.
  3. Comparison against two trivial baselines, so the Random Forest has
     something to beat:

       - NAIVE   : always predict the mean RUL of the training data.
                   This is the "no model at all" floor. Any model that
                   cannot beat it has learned nothing useful.
       - LINEAR  : ordinary least squares on the same 10 features.
                   This is the "simplest real model" bar. If the Random
                   Forest cannot beat linear regression, the extra
                   complexity is not justified.

  4. A smoothed variant of each model's predictions (centred rolling
     mean), matching the smoothing already used elsewhere in the project,
     reported separately so the effect of smoothing is visible rather
     than baked in.

IMPORTANT SCOPE NOTE
--------------------
This is an EXPLORATORY RUL module. It uses the same train/eval split as
the rest of the project (dev = training, val + test = evaluation), and it
does NOT use the fixed commissioning window — RUL regression is trained on
labelled targets rather than referenced to a healthy baseline, so the
calibration-window question does not apply here. The paper's claim that
RUL accuracy remains unvalidated should be updated only to the extent
these numbers support; they are single-condition, small-sample results.

Inputs : FEMTO condition 1 via rul_datasets (uses the local cache)
Outputs: rul_predictions_per_snapshot.csv   (every prediction, all models)
         rul_metrics_per_run.csv            (MAE/RMSE per run per model)
         rul_metrics_summary.csv            (pooled eval metrics per model)

Run:
    python evaluate_rul_baselines.py
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import rul_datasets

SAMPLE_RATE_HZ = 25600
SECONDS_PER_SNAPSHOT = 10
SMOOTHING_WINDOW = 15          # matches apex_v1_baseline.py
RF_TREES = 200                 # matches apex_v1_baseline.py
RF_MAX_DEPTH = 12
RANDOM_STATE = 42

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


def extract_features(window: np.ndarray) -> np.ndarray:
    """Identical to health_deviation_score.py — kept in sync deliberately."""
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


def rolling_average(values, window):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_runs():
    """Returns list of (run_name, features_array, labels_array)."""
    reader = rul_datasets.reader.FemtoReader(fd=1)
    reader.prepare_data()
    runs = []
    for split in ("dev", "val", "test"):
        feats, labels = reader.load_split(split)
        for i, (f, l) in enumerate(zip(feats, labels)):
            runs.append((f"{split}_run_{i}", f, np.asarray(l, dtype=float)))
    return runs


def main():
    print("Loading FEMTO condition 1 (uses local cache)...")
    runs = load_runs()

    print("Extracting features for all runs...")
    run_X, run_y = {}, {}
    for name, windows, labels in runs:
        run_X[name] = np.array([extract_features(w) for w in windows])
        run_y[name] = labels
        print(f"  {name} ({RUN_TO_BEARING[name]}): {len(labels)} snapshots")

    train_runs = ["dev_run_0", "dev_run_1"]
    eval_runs = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]

    X_train = np.vstack([run_X[r] for r in train_runs])
    y_train = np.concatenate([run_y[r] for r in train_runs])
    print(f"\nTraining set: {X_train.shape[0]} snapshots x {X_train.shape[1]} features")

    # ---------------------------- fit the models ----------------------------
    print("Fitting models...")
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH,
                               random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    naive_value = float(np.mean(y_train))
    print(f"  naive baseline predicts a constant {naive_value:.1f} snapshots")

    # ------------------------- predict on every run -------------------------
    pred_rows, metric_rows = [], []
    for name in train_runs + eval_runs:
        X, y = run_X[name], run_y[name]
        n = len(y)

        preds = {
            "naive": np.full(n, naive_value),
            "linear": lin.predict(X),
            "random_forest": rf.predict(X),
        }
        preds["linear_smoothed"] = rolling_average(preds["linear"], SMOOTHING_WINDOW)
        preds["random_forest_smoothed"] = rolling_average(
            preds["random_forest"], SMOOTHING_WINDOW)

        df = pd.DataFrame({
            "run": name,
            "bearing": RUN_TO_BEARING[name],
            "role": "development" if name in train_runs else "evaluation",
            "window_index": np.arange(n),
            "true_rul": y,
        })
        for model, p in preds.items():
            df[f"pred_{model}"] = p
        pred_rows.append(df)

        for model, p in preds.items():
            metric_rows.append({
                "run": name,
                "bearing": RUN_TO_BEARING[name],
                "role": "development" if name in train_runs else "evaluation",
                "model": model,
                "n_snapshots": n,
                "mae_snapshots": mean_absolute_error(y, p),
                "rmse_snapshots": np.sqrt(mean_squared_error(y, p)),
                "mae_hours": mean_absolute_error(y, p) * SECONDS_PER_SNAPSHOT / 3600,
                "rmse_hours": np.sqrt(mean_squared_error(y, p)) * SECONDS_PER_SNAPSHOT / 3600,
            })

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df.to_csv("rul_predictions_per_snapshot.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows).round(3)
    metrics_df.to_csv("rul_metrics_per_run.csv", index=False)

    # --------------------- pooled evaluation-set summary ---------------------
    ev = preds_df[preds_df["role"] == "evaluation"]
    summary_rows = []
    for model in ["naive", "linear", "linear_smoothed",
                  "random_forest", "random_forest_smoothed"]:
        y_true = ev["true_rul"].to_numpy()
        y_pred = ev[f"pred_{model}"].to_numpy()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        summary_rows.append({
            "model": model,
            "mae_snapshots": round(mae, 2),
            "rmse_snapshots": round(rmse, 2),
            "mae_hours": round(mae * SECONDS_PER_SNAPSHOT / 3600, 3),
            "rmse_hours": round(rmse * SECONDS_PER_SNAPSHOT / 3600, 3),
        })
    summary = pd.DataFrame(summary_rows)

    naive_mae = summary.loc[summary["model"] == "naive", "mae_snapshots"].iloc[0]
    naive_rmse = summary.loc[summary["model"] == "naive", "rmse_snapshots"].iloc[0]
    summary["mae_vs_naive_pct"] = (
        100 * (naive_mae - summary["mae_snapshots"]) / naive_mae).round(1)
    summary["rmse_vs_naive_pct"] = (
        100 * (naive_rmse - summary["rmse_snapshots"]) / naive_rmse).round(1)
    summary.to_csv("rul_metrics_summary.csv", index=False)

    # -------------------------------- report --------------------------------
    print("\n" + "=" * 78)
    print("POOLED EVALUATION-SET METRICS (5 held-out bearings)")
    print("positive vs_naive_pct = better than predicting the training mean")
    print("=" * 78)
    print(summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("PER-RUN METRICS — evaluation bearings, Random Forest (smoothed)")
    print("=" * 78)
    per_run = metrics_df[(metrics_df["role"] == "evaluation")
                         & (metrics_df["model"] == "random_forest_smoothed")]
    print(per_run[["bearing", "n_snapshots", "mae_snapshots",
                   "rmse_snapshots", "mae_hours", "rmse_hours"]].to_string(index=False))

    print("\nSaved:")
    print("  rul_predictions_per_snapshot.csv  (auditable prediction arrays)")
    print("  rul_metrics_per_run.csv           (MAE/RMSE per run per model)")
    print("  rul_metrics_summary.csv           (pooled, with baseline comparison)")
    print("\nRead the vs_naive columns first: if the Random Forest does not")
    print("clearly beat the naive constant, the RUL module should stay")
    print("described as exploratory in the paper.")


if __name__ == "__main__":
    main()
