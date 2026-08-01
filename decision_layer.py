"""
APEX-AI — Decision Layer  [v1]

The independent decision layer agreed in the architecture:
model produces a raw deviation score -> THIS layer turns it into an
operational state and recommendation -> the dashboard only displays.

Pipeline per snapshot (matches evaluate_thresholds_v2.py exactly):

    deviation score -> EWMA smoothing -> threshold check
    -> persistence rule (3 consecutive windows) -> LATCHING

Latching (human-in-the-loop): once WARNING or CRITICAL is reached, the
state never de-escalates on its own, even if the score dips back down.
Only an operator acknowledgment (ack) resets it. This prevents a
flickering dashboard and keeps the final decision with the operator.

Output follows Jory's JSON contract (Decision Support doc, section 6).
Cost figures are the declared scenario assumptions and are parameters,
not hard-coded truths.

Usage:
  as a module:   from decision_layer import DecisionLayer
  as a script:   python decision_layer.py
                 reads data/deviation_score_per_snapshot.csv, writes
                 decision_timeline.csv + decision_final_states.json
"""

import json
import numpy as np
import pandas as pd

# ---------------- default configuration (team-editable) ----------------
DEFAULTS = dict(
    warning_threshold=3.5,     # ~ P99.9 of train-only healthy scores
    critical_threshold=7.0,    # zero false criticals on eval bearings
    persistence=3,             # consecutive smoothed windows required
    alpha=0.2,                 # EWMA smoothing factor
    seconds_per_snapshot=10,
    # scenario cost assumptions (editable placeholders)
    inspection_cost_sar=1000,
    predictive_cost_sar=80000,
    reactive_cost_sar=101000,
)

ACTIONS = {
    "NORMAL": "Continue operation and monitoring",
    "WARNING": "Schedule inspection",
    "CRITICAL": "Prioritize planned maintenance",
}

STATE_RANK = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


class DecisionLayer:
    """Streaming decision layer: feed one deviation score at a time."""

    def __init__(self, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self.reset()

    def reset(self):
        self._smoothed = None
        self._streak_w = 0
        self._streak_c = 0
        self._state = "NORMAL"          # latched state
        self.window_index = -1

    def ack(self):
        """Operator acknowledgment: unlatch back to NORMAL."""
        self._state = "NORMAL"
        self._streak_w = self._streak_c = 0

    def update(self, score: float) -> dict:
        c = self.cfg
        self.window_index += 1
        # EWMA
        a = c["alpha"]
        self._smoothed = (score if self._smoothed is None
                          else a * score + (1 - a) * self._smoothed)
        s = self._smoothed
        # persistence counters
        self._streak_w = self._streak_w + 1 if s >= c["warning_threshold"] else 0
        self._streak_c = self._streak_c + 1 if s >= c["critical_threshold"] else 0
        # candidate state from current signal
        if self._streak_c >= c["persistence"]:
            candidate = "CRITICAL"
        elif self._streak_w >= c["persistence"]:
            candidate = "WARNING"
        else:
            candidate = "NORMAL"
        # latching: state can only escalate
        if STATE_RANK[candidate] > STATE_RANK[self._state]:
            self._state = candidate
        return {
            "window_index": self.window_index,
            "deviation_score": round(float(score), 4),
            "smoothed_score": round(float(s), 4),
            "status": self._state,
            "consecutive_exceedances": max(self._streak_w, self._streak_c),
            "recommended_action": ACTIONS[self._state],
        }

    def snapshot_json(self, run_id: str, rul_label=None) -> dict:
        """Full record in Jory's output contract for the current window."""
        c = self.cfg
        rec = {
            "run_id": run_id,
            "window_index": self.window_index,
            "deviation_score": round(float(self._smoothed), 2),
            "status": self._state,
            "warning_threshold": c["warning_threshold"],
            "critical_threshold": c["critical_threshold"],
            "consecutive_exceedances": max(self._streak_w, self._streak_c),
            "rul_label": None if rul_label is None else float(rul_label),
            "recommended_action": ACTIONS[self._state],
            "reactive_cost_sar": c["reactive_cost_sar"],
            "predictive_cost_sar": c["predictive_cost_sar"],
            "estimated_saving_sar": c["reactive_cost_sar"] - c["predictive_cost_sar"],
        }
        return rec


def main():
    scores = pd.read_csv("deviation_score_per_snapshot.csv")
    timeline_rows, final_states = [], {}

    for run, g in scores.groupby("run", sort=False):
        g = g.sort_values("window_index")
        layer = DecisionLayer()
        first_alarm = {}
        for _, row in g.iterrows():
            out = layer.update(row["deviation_score"])
            out["run"] = run
            out["rul_label"] = row["rul_label"]
            timeline_rows.append(out)
            st = out["status"]
            if st != "NORMAL" and st not in first_alarm:
                first_alarm[st] = {
                    "window": out["window_index"],
                    "rul_label": row["rul_label"],
                    "hours_to_failure": round(
                        row["rul_label"] * DEFAULTS["seconds_per_snapshot"] / 3600, 2),
                }
        rec = layer.snapshot_json(run, rul_label=g["rul_label"].iloc[-1])
        rec["first_warning"] = first_alarm.get("WARNING")
        rec["first_critical"] = first_alarm.get("CRITICAL")
        final_states[run] = rec

    pd.DataFrame(timeline_rows).to_csv("decision_timeline.csv", index=False)
    with open("decision_final_states.json", "w") as f:
        json.dump(final_states, f, indent=2)

    print("Saved decision_timeline.csv "
          f"({len(timeline_rows)} rows) and decision_final_states.json\n")
    print("First WARNING / CRITICAL per run (with latching):")
    for run, rec in final_states.items():
        w, c = rec["first_warning"], rec["first_critical"]
        print(f"  {run}: WARNING at window "
              f"{w['window'] if w else '—'}"
              f" ({w['hours_to_failure'] if w else '—'} h to failure), "
              f"CRITICAL at window {c['window'] if c else '—'}"
              f" ({c['hours_to_failure'] if c else '—'} h to failure), "
              f"final={rec['status']}")


if __name__ == "__main__":
    main()
