"""
APEX AI — v1 Dashboard (Streamlit)

The display layer of the vertical slice. It ONLY displays:
the model produces deviation scores, decision_layer.py turns them into
states — this app renders both, plus the cost-based threshold analysis
(the research contribution) with live sensitivity sliders.

Run from the repo root:
    streamlit run src/apex_dashboard.py

Requires: streamlit, plotly  (pip install streamlit plotly)
Reads:    results/deviation_score_per_snapshot.csv, results/features_per_snapshot.csv
Imports:  decision_layer.py (must be in the same folder)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from decision_layer import DecisionLayer, DEFAULTS, ACTIONS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---------------- constants (mirror decision layer / evaluation) ----------------
CALIBRATION_WINDOWS = 200
SECONDS_PER_SNAPSHOT = 10
PERSISTENCE = 3
ALPHA = 0.2
SIGMA_K = 3.0
DEG_PERSISTENCE = 5
CANDIDATE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]
EVAL_RUNS = ["val_run_0", "test_run_0", "test_run_1", "test_run_2", "test_run_3"]

STATE_COLOR = {"NORMAL": "#2e7d32", "WARNING": "#ef6c00", "CRITICAL": "#c62828"}
STATE_EMOJI = {"NORMAL": "🟢", "WARNING": "🟠", "CRITICAL": "🔴"}

# ---------------- operator audit log ----------------
AUDIT_LOG = RESULTS / "audit_log.csv"
AUDIT_COLUMNS = ["timestamp", "run", "status", "hds_score",
                 "recommended_action", "operator_decision", "operator_note"]
# Fixed vocabulary - the operator picks one, no free text, so the log stays
# analysable (e.g. false-alarm rate per bearing) instead of turning into prose.
OPERATOR_DECISIONS = [
    "Acknowledged - followed recommendation",
    "Acknowledged - different action scheduled",
    "Dismissed (false alarm)",
    "Escalated to immediate maintenance",
]


def windows_to_hours(n):
    return n * SECONDS_PER_SNAPSHOT / 3600.0


# ------------------------------ cached data prep ------------------------------
@st.cache_data
def load_data():
    scores = pd.read_csv(RESULTS / "deviation_score_per_snapshot.csv")
    feats = pd.read_csv(RESULTS / "features_per_snapshot.csv")
    return scores, feats


def _first_persistent(above, k, start=0):
    streak = 0
    for i in range(start, len(above)):
        streak = streak + 1 if above[i] else 0
        if streak >= k:
            return i - k + 1
    return None


def _alarm_events(above, k, start, end):
    ev, streak, fired = 0, 0, False
    for i in range(start, end):
        if above[i]:
            streak += 1
            if streak >= k and not fired:
                ev, fired = ev + 1, True
        else:
            streak, fired = 0, False
    return ev


@st.cache_data
def degradation_reference(feats):
    deg = {}
    for run, g in feats.groupby("run", sort=False):
        g = g.sort_values("window_index").reset_index(drop=True)
        n = len(g)
        above = np.zeros(n, dtype=bool)
        for ch in ("rms_ch0", "rms_ch1"):
            cal = g[ch].iloc[:CALIBRATION_WINDOWS]
            above |= (g[ch].to_numpy() > cal.mean() + SIGMA_K * cal.std())
        idx = _first_persistent(above, DEG_PERSISTENCE, start=CALIBRATION_WINDOWS)
        deg[run] = idx if idx is not None else n
    return deg


@st.cache_data
def threshold_stats(scores, _deg):
    """Per candidate threshold: false-alarm events + lead times on the
    5 evaluation bearings, using the smoothed pipeline. Cost-independent
    — costs are applied live from the sidebar sliders."""
    rows = []
    for thr in CANDIDATE_THRESHOLDS:
        fa_total, leads = 0, []
        for run in EVAL_RUNS:
            s = (scores[scores["run"] == run]
                 .sort_values("window_index")["deviation_score"].to_numpy())
            sm = pd.Series(s).ewm(alpha=ALPHA).mean().to_numpy()
            above = sm >= thr
            n, ds = len(sm), _deg[run]
            fa_total += _alarm_events(above, PERSISTENCE, CALIBRATION_WINDOWS, ds)
            fi = _first_persistent(above, PERSISTENCE, start=ds)
            leads.append(None if fi is None else windows_to_hours(n - fi))
        rows.append({"threshold": thr, "false_alarms": fa_total, "leads": leads})
    return rows


@st.cache_data
def decision_timeline(scores, run, warning, critical, ack_windows=()):
    """ack_windows: sorted tuple of window indices where the operator
    acknowledged the alarm. At each such window, the layer is reset to
    NORMAL (via layer.ack()) immediately before that window's own score
    is processed, so history before the ack is untouched and the state
    can re-escalate afterwards if the score persists above threshold."""
    g = scores[scores["run"] == run].sort_values("window_index")
    ack_set = set(ack_windows)
    layer = DecisionLayer(warning_threshold=warning, critical_threshold=critical)
    recs = []
    for i, v in enumerate(g["deviation_score"]):
        if i in ack_set:
            layer.ack()
        recs.append(layer.update(v))
    df = pd.DataFrame(recs)
    df["rul_label"] = g["rul_label"].to_numpy()
    return df


def load_audit_log():
    """Read results/audit_log.csv, creating an empty header-only file on first
    use. Deliberately NOT cached: this app appends to the same file, so it has
    to be re-read on every rerun."""
    if not AUDIT_LOG.exists():
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_LOG, index=False)
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    try:
        df = pd.read_csv(AUDIT_LOG)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    df = df.reindex(columns=AUDIT_COLUMNS)
    df["operator_note"] = df["operator_note"].fillna("")
    return df


def append_audit_row(row):
    """Append exactly one acknowledgment to the CSV. Appending (not rewriting)
    keeps the log durable across app restarts and concurrent sessions."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_header = not AUDIT_LOG.exists() or AUDIT_LOG.stat().st_size == 0
    pd.DataFrame([row], columns=AUDIT_COLUMNS).to_csv(
        AUDIT_LOG, mode="a", header=write_header, index=False)


def _acknowledge(run_name, window):
    acks = st.session_state["ack_windows"].setdefault(run_name, [])
    if window not in acks:
        acks.append(window)


def _clear_acks(run_name):
    st.session_state["ack_windows"][run_name] = []


# ----------------------------------- UI -----------------------------------
st.set_page_config(page_title="APEX AI v1", page_icon="⚙️", layout="wide")
st.markdown("""<meta name="google" content="notranslate">
<style>
/* declare content language & block auto-translate side effects */
:root {translate: no;}
.stApp {translate: no;}
/* hide only the Deploy button and the app menu — keep the header
   (it holds the sidebar collapse/expand arrow) */
[data-testid="stAppDeployButton"] {display: none;}
[data-testid="stToolbarActions"] {display: none;}
[data-testid="stMainMenu"] {display: none;}
#MainMenu {visibility: hidden;}
/* fix slider thumb/track mismatch on RTL system locales */
[data-testid="stSlider"], [data-testid="stSlider"] * {direction: ltr !important;}
section[data-testid="stSidebar"] {direction: ltr;}
</style>""", unsafe_allow_html=True)

st.title("⚙️ APEX AI — Predictive Maintenance v1")
st.caption("Anomaly detection on PRONOSTIA/FEMTO bearings · cost-based alerting · human-in-the-loop")

scores, feats = load_data()
deg = degradation_reference(feats)
runs = list(scores["run"].unique())

st.session_state.setdefault("ack_windows", {})

with st.sidebar:
    st.header("Bearing")
    run = st.selectbox("Select bearing (run)", runs, index=runs.index("test_run_2"))
    st.session_state["ack_windows"].setdefault(run, [])
    st.button("Clear acknowledgments for this bearing",
              on_click=_clear_acks, args=(run,), key="clear_acks_btn")

    st.header("Playback")
    n_windows = int((scores["run"] == run).sum())
    max_t = n_windows - 1
    # clamp stored position when switching to a shorter run
    if "t_window" not in st.session_state or st.session_state.t_window > max_t:
        st.session_state.t_window = max_t
    t = st.number_input("Current time (window)", 0, max_t, step=25,
                        key="t_window",
                        help="Simulated current moment. 1 window = 10 s of "
                             "operation. Type a value or use the jump buttons.")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("⏮", help="Jump to start", width='stretch',
              on_click=lambda: st.session_state.update(t_window=0))
    b2.button("−250", width='stretch',
              on_click=lambda: st.session_state.update(
                  t_window=max(0, st.session_state.t_window - 250)))
    b3.button("+250", width='stretch',
              on_click=lambda: st.session_state.update(
                  t_window=min(max_t, st.session_state.t_window + 250)))
    b4.button("⏭", help="Jump to end", width='stretch',
              on_click=lambda: st.session_state.update(t_window=max_t))

    st.header("Decision thresholds")
    warning = st.number_input("WARNING threshold", 1.0, 15.0,
                              float(DEFAULTS["warning_threshold"]), 0.5)
    critical = st.number_input("CRITICAL threshold", 2.0, 20.0,
                               float(DEFAULTS["critical_threshold"]), 0.5)
    if critical <= warning:
        st.error("CRITICAL must be above WARNING.")
        st.stop()

    st.header("Cost scenario (assumptions)")
    st.caption("Placeholder figures — every value is editable. "
               "The optimal threshold reacts live.")
    c_insp = st.number_input("False-alarm inspection cost (SAR)", 0, 50000,
                             DEFAULTS["inspection_cost_sar"], 500)
    c_pred = st.number_input("Planned (predictive) maintenance cost (SAR)", 0, 500000,
                             DEFAULTS["predictive_cost_sar"], 5000)
    c_react = st.number_input("Run-to-failure (reactive) cost (SAR)", 0, 1000000,
                              DEFAULTS["reactive_cost_sar"], 5000)
    horizon = st.number_input("Planning horizon (hours, FEMTO-time)",
                              0.0, 3.0, 1.0, 0.25,
                              help="Minimum lead time needed to actually "
                                   "schedule maintenance.")

# ---- decision timeline for the selected bearing / thresholds ----
ack_tuple = tuple(sorted(st.session_state["ack_windows"].get(run, [])))
tl = decision_timeline(scores, run, warning, critical, ack_tuple)
now = tl.iloc[t]
state = now["status"]

# ---------------- status row ----------------
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.4])
with col1:
    st.markdown(
        f"""<div style="background:{STATE_COLOR[state]};color:white;border-radius:12px;
        padding:18px;text-align:center;">
        <div style="font-size:2rem;">{STATE_EMOJI[state]} {state}</div>
        <div style="font-size:1rem;margin-top:6px;">{ACTIONS[state]}</div>
        </div>""", unsafe_allow_html=True)
    if state != "NORMAL":
        # The acknowledgment IS the audit event: the form's submit button is the
        # Acknowledge button, so an alarm can never be unlatched without leaving
        # a row in results/audit_log.csv. On submit we append the row, record
        # the ack window in session_state (which decision_timeline() replays
        # through layer.ack()), then rerun so the status box refreshes.
        with st.form(f"ack_form_{run}", clear_on_submit=True):
            operator_decision = st.selectbox(
                "Operator decision", OPERATOR_DECISIONS, index=None,
                placeholder="Select a decision…")
            operator_note = st.text_input("Note (optional)")
            submitted = st.form_submit_button("✅ Acknowledge alarm",
                                              width='stretch')
        if submitted:
            if operator_decision is None:
                st.warning("Select an operator decision before acknowledging.")
            else:
                append_audit_row({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "run": run,
                    "status": state,
                    "hds_score": round(float(now["smoothed_score"]), 4),
                    "recommended_action": ACTIONS[state],
                    "operator_decision": operator_decision,
                    "operator_note": operator_note.strip(),
                })
                _acknowledge(run, t)
                st.rerun()
with col2:
    st.metric("Smoothed deviation score", f"{now['smoothed_score']:.2f}",
              help="Average |z-score| across vibration features, EWMA-smoothed. "
                   "Roughly: how many 'healthy standard deviations' away we are.")
with col3:
    st.metric("Elapsed operation", f"{windows_to_hours(t):.2f} h",
              f"window {t} / {n_windows - 1}")
with col4:
    fw = tl[tl["status"] == "WARNING"]["window_index"].min()
    fc = tl[tl["status"] == "CRITICAL"]["window_index"].min()
    msg = []
    if pd.notna(fw) and fw <= t:
        msg.append(f"First WARNING at {windows_to_hours(int(fw)):.2f} h")
    if pd.notna(fc) and fc <= t:
        msg.append(f"First CRITICAL at {windows_to_hours(int(fc)):.2f} h")
    st.markdown("**Alarm log (latched)**<br>" + ("<br>".join(msg) if msg else "—"),
                unsafe_allow_html=True)
    if state != "NORMAL":
        st.caption("State stays latched until an operator acknowledges — "
                   "the final decision is always human.")

# ---------------- score chart ----------------
st.subheader("Health deviation over time")
fig = go.Figure()
fig.add_trace(go.Scatter(y=tl["deviation_score"][:t + 1], name="raw score",
                         line=dict(color="#b0bec5", width=1), opacity=0.5))
fig.add_trace(go.Scatter(y=tl["smoothed_score"][:t + 1], name="smoothed (EWMA)",
                         line=dict(color="#1565c0", width=2.5)))
fig.add_hline(y=warning, line_dash="dash", line_color=STATE_COLOR["WARNING"],
              annotation_text=f"WARNING ≥ {warning}",
              annotation_position="right", annotation_yshift=-14)
fig.add_hline(y=critical, line_dash="dash", line_color=STATE_COLOR["CRITICAL"],
              annotation_text=f"CRITICAL ≥ {critical}",
              annotation_position="right", annotation_yshift=14)
fig.add_vrect(x0=0, x1=min(CALIBRATION_WINDOWS, t + 1),
              fillcolor="#a5d6a7", opacity=0.25, line_width=0,
              annotation_text="calibration", annotation_position="top left")
fig.update_layout(height=380, margin=dict(l=50, r=140, t=40, b=50),
                  xaxis_title="window index (1 window = 10 s)",
                  yaxis_title="deviation score",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width='stretch')

# ---------------- cost analysis (research contribution) ----------------
st.subheader("Cost-based threshold selection — live sensitivity")
st.caption("For each candidate WARNING threshold, expected scenario cost over the 5 "
           "evaluation bearings = false alarms × inspection cost + per-bearing outcome "
           "(planned maintenance if lead time ≥ planning horizon, else reactive). "
           "Change the cost figures and watch the optimum shift.")

stats = threshold_stats(scores, deg)
cost_rows = []
for r in stats:
    cost = r["false_alarms"] * c_insp
    for lh in r["leads"]:
        if lh is None:
            cost += c_react
        else:
            cost += c_pred if lh >= horizon else c_react
    valid = [l for l in r["leads"] if l is not None]
    cost_rows.append({
        "threshold": r["threshold"],
        "false alarms (5 bearings)": r["false_alarms"],
        "mean lead time (h)": round(np.mean(valid), 2) if valid else None,
        "scenario cost (SAR)": cost,
    })
cost_df = pd.DataFrame(cost_rows)
best_thr = cost_df.loc[cost_df["scenario cost (SAR)"].idxmin(), "threshold"]

cc1, cc2 = st.columns([1.4, 1])
with cc1:
    bar = go.Figure(go.Bar(
        x=cost_df["threshold"].astype(str), y=cost_df["scenario cost (SAR)"],
        marker_color=["#c62828" if v == best_thr else "#90a4ae"
                      for v in cost_df["threshold"]]))
    bar.update_layout(height=320, margin=dict(l=70, r=20, t=20, b=50),
                      xaxis_title="candidate threshold",
                      yaxis_title="expected scenario cost (SAR)")
    st.plotly_chart(bar, width='stretch')
with cc2:
    st.metric("Cost-optimal threshold (current scenario)", f"{best_thr}")
    st.metric("Per-failure saving if caught in time",
              f"{c_react - c_pred:,.0f} SAR",
              help="Reactive cost minus planned-maintenance cost.")
    st.dataframe(cost_df, hide_index=True, width='stretch')

# ---------------- operator audit log ----------------
st.subheader("Audit Log")
st.caption("Every operator acknowledgment, appended to results/audit_log.csv "
           "— persists across app restarts. Most recent first.")
audit_df = load_audit_log()
if audit_df.empty:
    st.info("No decisions logged yet.")
else:
    st.dataframe(audit_df.iloc[::-1], hide_index=True, width='stretch')
