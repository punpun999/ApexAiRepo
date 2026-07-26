"""
APEX-AI — Explore the FEMTO (PRONOSTIA) bearing dataset

This doesn't train anything — it just shows you what the data actually
looks like, so you can build intuition before trusting the model.

Run:
    python explore_femto.py

It will:
1. Print the basic shape/structure of the dataset (how many bearings,
   how many snapshots per bearing, what a snapshot looks like).
2. Plot the raw vibration signal for an EARLY (healthy) snapshot vs. a
   LATE (near-failure) snapshot from the same bearing, side by side —
   you should visibly see the late one looks "louder"/"spikier".
3. Plot the RMS (overall shaking level) across the bearing's entire
   life, from healthy to failure — this is the degradation curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import rul_datasets

SAMPLE_RATE_HZ = 25600


def main():
    print("Loading FEMTO condition 1...")
    reader = rul_datasets.reader.FemtoReader(fd=1)
    reader.prepare_data()

    features, labels = reader.load_split("dev")

    print(f"\nNumber of bearing runs in this split: {len(features)}")
    for i, (run_feats, run_labels) in enumerate(zip(features, labels)):
        print(f"  Run {i}: {run_feats.shape[0]} snapshots, "
              f"starting RUL = {run_labels[0]:.0f} cycles, "
              f"ending RUL = {run_labels[-1]:.0f} cycles")

    # Pick the first run to inspect closely
    run_feats = features[0]
    run_labels = labels[0]
    print(f"\nInspecting run 0.")
    print(f"Each snapshot shape: {run_feats[0].shape}  "
          f"(samples per snapshot, channels)")
    print(f"Snapshot 0 (earliest, healthiest) — first 5 raw values, channel 0:")
    print(run_feats[0][:5, 0])
    print(f"Snapshot {len(run_feats)-1} (last, closest to failure) — first 5 raw values, channel 0:")
    print(run_feats[-1][:5, 0])

    # --- Plot 1: raw signal, early vs late ---
    early_window = run_feats[0][:, 0]       # channel 0, first snapshot
    late_window = run_feats[-1][:, 0]       # channel 0, last snapshot
    time_axis = np.arange(len(early_window)) / SAMPLE_RATE_HZ

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    axes[0].plot(time_axis, early_window, linewidth=0.5)
    axes[0].set_title(f"EARLY snapshot (RUL = {run_labels[0]:.0f} cycles) — healthy")
    axes[0].set_ylabel("Vibration amplitude")

    axes[1].plot(time_axis, late_window, linewidth=0.5, color="crimson")
    axes[1].set_title(f"LATE snapshot (RUL = {run_labels[-1]:.0f} cycles) — near failure")
    axes[1].set_ylabel("Vibration amplitude")
    axes[1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig("femto_raw_signal_early_vs_late.png", dpi=150)
    print("\nSaved femto_raw_signal_early_vs_late.png")
    print("Look at the y-axis scale and the 'spikiness' — the late one should")
    print("look visibly louder and more erratic than the early one.")

    # --- Plot 2: RMS trend across the whole life of the bearing ---
    rms_over_time = [np.sqrt(np.mean(window[:, 0] ** 2)) for window in run_feats]

    plt.figure(figsize=(9, 5))
    plt.plot(rms_over_time)
    plt.xlabel("Snapshot index (time, healthy -> failure)")
    plt.ylabel("RMS (overall shaking level)")
    plt.title("Degradation curve — RMS rising as the bearing approaches failure")
    plt.tight_layout()
    plt.savefig("femto_rms_degradation_curve.png", dpi=150)
    print("Saved femto_rms_degradation_curve.png")
    print("You should see this stay roughly flat/low for a while, then climb")
    print("and get noisier as the bearing nears the end of its life.")


if __name__ == "__main__":
    main()
