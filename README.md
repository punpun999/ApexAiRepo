# APEX-AI — Predictive Maintenance v1

Anomaly detection on PRONOSTIA/FEMTO bearing vibration data. A health
deviation score is computed from vibration features relative to a
fixed healthy calibration period, then passed through a cost-based
decision layer (EWMA smoothing, thresholds, persistence, latching) that
turns the score into an operational state and recommendation. Results
are explored in a Streamlit dashboard. The design is human-in-the-loop:
the decision layer only recommends, and only an operator acknowledgment
resets a latched WARNING/CRITICAL state.

## Folder structure

```
.
├── src/        Python scripts: feature/score computation, threshold
│               evaluation, the decision layer, and the Streamlit dashboard
├── results/    Generated CSV/JSON outputs (deviation scores, features,
│               threshold evaluations, decision timeline and states)
├── figures/    Generated PNG plots (deviation score curves, raw FEMTO
│               signal and RMS degradation plots)
├── archive/    Earlier exploration scripts and the CMAPSS baseline
│               (superseded by the FEMTO-based pipeline in src/)
└── docs/       Project documentation
```

## Setup

```
pip install -r requirements.txt
```

## Run order

```
python src/health_deviation_score.py   # first time only — downloads ~727MB (FEMTO dataset)
python src/evaluate_thresholds_v2.py
python src/decision_layer.py
streamlit run src/apex_dashboard.py
```

## Key parameters

Defined in `DEFAULTS` in `src/decision_layer.py`:

- Calibration window: 200 snapshots
- EWMA alpha: 0.2
- Persistence: 3 windows
- WARNING threshold: 3.5
- CRITICAL threshold: 7.0

## Note on costs

All cost figures (inspection, predictive maintenance, reactive/run-to-failure)
are declared scenario assumptions, not measured values. They are editable
parameters used to illustrate cost-based threshold selection, including live
in the dashboard's sensitivity sliders.
