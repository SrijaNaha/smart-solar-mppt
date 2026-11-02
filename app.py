"""
app.py - Smart Solar Power Unit with MPPT : Streamlit dashboard (Person 2)
==========================================================================
Run with:   python -m streamlit run app.py

EVERY VALUE SHOWN IN THIS DASHBOARD IS SIMULATED.
It comes from Person 1's software model (mppt.py): simulate_mppt() and
summarize_result(). Nothing here is measured from real hardware.
Battery / converter figures are simple illustrative ESTIMATES only.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mppt import simulate_mppt, summarize_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_TITLE = "SMART SOLAR POWER UNIT WITH MPPT"
APP_SUBTITLE = "Intelligent Solar Energy Monitoring and Maximum Power Point Tracking"

# colour palette (explicit colours so cards/charts look right in any Streamlit theme)
CARD = "#0F2233"
CARD_HI = "#12304A"
YELLOW = "#FFC93C"
GREEN = "#3DDC84"
TEAL = "#2EC4B6"
ORANGE = "#FF8C42"
RED = "#FF5C5C"
BLUE = "#4DA3FF"
TEXT = "#E8EEF4"
MUTED = "#8FA3B5"
GRID = "rgba(143,163,181,0.18)"
FONT = "Segoe UI, Roboto, Helvetica, Arial, sans-serif"

STATUS_CONVERGED = "Converged"
STATUS_TRACKING = "Still Tracking"
STATUS_NO_INPUT = "No Solar Input"
STATUS_COLOURS = {STATUS_CONVERGED: GREEN, STATUS_TRACKING: YELLOW, STATUS_NO_INPUT: MUTED}

# default values for every widget (also used by the Reset button)
DEFAULTS = {
    "irr": 800,          # W/m^2
    "zero_sun": False,   # simulate night (0 W/m^2)
    "steps": 60,         # P&O iterations
    "temp": 25,          # cell temperature, degC
    "voc": 21.0,         # V
    "isc": 3.0,          # A
    "pert": 0.2,         # P&O perturbation step, V
    "batt_v": 12.0,      # battery voltage, V (assumed)
    "conv_eff": 85,      # converter efficiency, %
}

COMPARE_LEVELS = [(1000, "High sunlight"), (600, "Medium sunlight"), (200, "Low sunlight")]

HISTORY_COLUMNS = [
    "Simulation #",
    "Irradiance (W/m²)",
    "MPP Voltage (V)",
    "MPP Current (A)",
    "Max Power (W)",
    "Final Tracked Power (W)",
    "Tracking Efficiency (%)",
    "Tracking Status",
]


# ---------------------------------------------------------------------------
# Streamlit-version compatibility (use_container_width was replaced by width)
# ---------------------------------------------------------------------------
def _version_at_least(major: int, minor: int) -> bool:
    try:
        parts = re.findall(r"\d+", st.__version__)[:2]
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:  # unknown version string -> assume older API
        return False


_NEW_WIDTH_API = _version_at_least(1, 50)


def stretch_call(fn, *args, **kwargs):
    """Call a Streamlit element so that it fills the available width,
    using whichever API the installed Streamlit version supports."""
    if _NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except Exception:
            pass
    return fn(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Simulation (single cached call into Person 1's code - no duplicate work)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def run_simulation(irradiance: float, steps: int, voc: float, isc: float,
                   temperature: float, step: float) -> dict:
    """Call Person 1's simulate_mppt() and summarize_result() exactly once per
    unique set of inputs. Raises ValueError for invalid inputs."""
    result = simulate_mppt(irradiance=irradiance, steps=steps, voc=voc,
                           isc=isc, temperature=temperature, step=step)
    summary = summarize_result(result)
    return {"result": result, "summary": summary}


# ---------------------------------------------------------------------------
# Pure helper functions (no Streamlit calls)
# ---------------------------------------------------------------------------
def classify_status(p_mpp: float, converged: bool) -> str:
    """Tracking status from summarize_result()['converged'] and the theoretical power."""
    if p_mpp is None or not math.isfinite(p_mpp) or p_mpp <= 0:
        return STATUS_NO_INPUT
    return STATUS_CONVERGED if converged else STATUS_TRACKING


def tracking_efficiency(final_power: float, max_power: float):
    """(Final tracked power / theoretical max power) x 100.
    Returns None when there is no solar input (max power is zero) - never a fake 100 %."""
    if max_power is None or not math.isfinite(max_power) or max_power <= 0:
        return None
    return min(100.0, max(0.0, 100.0 * final_power / max_power))


def derive_metrics(bundle: dict, irradiance: float) -> dict:
    """Collect every number the dashboard shows from ONE simulation bundle."""
    result, summary = bundle["result"], bundle["summary"]
    p_mpp = result["p_mpp"]
    is_dark = not (p_mpp > 0)
    eff = tracking_efficiency(summary["final_power"], p_mpp)
    shortfall = None if is_dark else max(0.0, 100.0 * summary["power_shortfall_fraction"])
    curve = result["curve"]
    return {
        "irradiance": float(irradiance),
        "is_dark": is_dark,
        "v_mpp": result["v_mpp"],
        "i_mpp": result["i_mpp"],
        "p_mpp": p_mpp,
        "isc_actual": result["isc_actual"],
        "voc_actual": None if is_dark else float(curve["Voltage (V)"].iloc[-1]),
        "v_final": summary["final_voltage"],
        "i_final": summary["final_current"],
        "p_final": summary["final_power"],
        "efficiency": eff,
        "shortfall": shortfall,
        "tolerance_pct": 100.0 * summary["tolerance"],
        "converged": summary["converged"],
        "status": classify_status(p_mpp, summary["converged"]),
    }


def estimate_battery(p_max: float, converter_eff_pct: float, battery_v: float) -> dict:
    """Illustrative converter/battery estimate (NOT a measurement).
    Output power = solar max power x efficiency ; charging current = output power / battery voltage."""
    eff = min(100.0, max(0.0, float(converter_eff_pct))) / 100.0
    p_out = max(0.0, float(p_max)) * eff
    current = p_out / battery_v if battery_v and battery_v > 0 else 0.0
    return {
        "p_out": p_out,
        "current": current,
        "loss": max(0.0, float(p_max)) - p_out,
        "charging": p_out > 0,
    }


def fmt(value, decimals: int = 2) -> str:
    """Format a number; None / NaN become 'N/A'."""
    if value is None:
        return "N/A"
    try:
        if not math.isfinite(value):
            return "N/A"
    except TypeError:
        return "N/A"
    return f"{value:.{decimals}f}"


def history_row(number: int, m: dict) -> dict:
    eff = m["efficiency"]
    return {
        HISTORY_COLUMNS[0]: number,
        HISTORY_COLUMNS[1]: m["irradiance"],
        HISTORY_COLUMNS[2]: round(m["v_mpp"], 3),
        HISTORY_COLUMNS[3]: round(m["i_mpp"], 3),
        HISTORY_COLUMNS[4]: round(m["p_mpp"], 3),
        HISTORY_COLUMNS[5]: round(m["p_final"], 3),
        HISTORY_COLUMNS[6]: None if eff is None else round(eff, 2),
        HISTORY_COLUMNS[7]: m["status"],
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel shows the superscript in 'W/m²' correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Session state, inputs and callbacks
# ---------------------------------------------------------------------------
def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("history", [])


def current_params() -> dict:
    ss = st.session_state
    irradiance = 0.0 if ss["zero_sun"] else float(ss["irr"])
    return {
        "irradiance": irradiance,
        "steps": int(ss["steps"]),
        "voc": float(ss["voc"]),
        "isc": float(ss["isc"]),
        "temperature": float(ss["temp"]),
        "step": float(ss["pert"]),
    }


def reset_inputs() -> None:
    """Reset every input to its default (history is kept)."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["flash"] = ("info", "Inputs reset to default values.")


def log_run() -> None:
    """Run Simulation button: add the current simulation to the session history."""
    try:
        params = current_params()
        bundle = run_simulation(**params)  # cached - same call the page just used
        m = derive_metrics(bundle, params["irradiance"])
    except ValueError as exc:
        st.session_state["flash"] = ("error", f"Simulation failed: {exc}")
        return
    history = st.session_state["history"]
    history.append(history_row(len(history) + 1, m))
    st.session_state["flash"] = (
        "ok", f"Simulation #{len(history)} complete: {m['status']} "
              f"({fmt(m['efficiency'], 1)} % tracking efficiency)")


def clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["flash"] = ("info", "Simulation history cleared.")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {background: radial-gradient(circle at 12% 10%, #1b4d3e 0%, #0d2a3d 45%, #08141f 100%);
       border: 1px solid rgba(255,201,60,0.35); border-radius: 18px; padding: 24px 28px 18px 28px;
       box-shadow: 0 8px 28px rgba(0,0,0,0.35); margin-bottom: 14px;}
.hero h1 {color: #FFC93C; font-size: 2.05rem; letter-spacing: 1.5px; margin: 0 0 4px 0; padding: 0; line-height: 1.2;}
.hero .sub {color: #E8EEF4; font-size: 1.05rem; margin-bottom: 14px;}
.pill {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
       margin: 3px 8px 3px 0; white-space: nowrap;}
.simbanner {background: rgba(255,201,60,0.10); border: 1px solid rgba(255,201,60,0.55); color: #FFE08A;
            border-radius: 10px; padding: 9px 14px; margin-top: 10px; font-size: 0.9rem;}
.m-card {background: linear-gradient(145deg, #0F2233, #12304A); border: 1px solid rgba(255,255,255,0.07);
         border-left: 5px solid var(--accent); border-radius: 14px; padding: 14px 18px; min-height: 122px;
         box-shadow: 0 4px 14px rgba(0,0,0,0.25); margin-bottom: 10px;}
.m-label {color: #8FA3B5; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;}
.m-value {color: #E8EEF4; font-size: 2.0rem; font-weight: 700; line-height: 1.25; margin-top: 4px;}
.m-unit {color: var(--accent); font-size: 1.0rem; font-weight: 600; margin-left: 6px;}
.m-sub {color: #8FA3B5; font-size: 0.8rem; margin-top: 4px;}
.group-title {font-weight: 700; font-size: 0.95rem; margin: 6px 0 8px 0; padding-left: 10px;
              border-left: 4px solid #FFC93C;}
.tag {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 2px 9px; border-radius: 6px;
      background: rgba(255,140,66,0.15); border: 1px solid #FF8C42; color: #FF8C42; margin-left: 8px;
      vertical-align: middle; letter-spacing: 0.6px;}
</style>
"""


def pill(text: str, colour: str) -> str:
    return (f'<span class="pill" style="background:{colour}22;border:1px solid {colour};'
            f'color:{colour}">{text}</span>')


def metric_card(label: str, value: str, unit: str, sub: str, accent: str) -> str:
    return (f'<div class="m-card" style="--accent:{accent}">'
            f'<div class="m-label">{label}</div>'
            f'<div class="m-value">{value}<span class="m-unit">{unit}</span></div>'
            f'<div class="m-sub">{sub}</div></div>')


def show_card(column, label: str, value: str, unit: str, sub: str, accent: str) -> None:
    with column:
        st.markdown(metric_card(label, value, unit, sub, accent), unsafe_allow_html=True)


def group_title(text: str, tag: str = "") -> None:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="group-title">{text}{tag_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly figures (all data comes straight from simulate_mppt())
# ---------------------------------------------------------------------------
def base_layout(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 410) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=17, color=TEXT)),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, family=FONT),
        height=height, margin=dict(l=62, r=26, t=62, b=100),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False)
    return fig


def no_input_note(fig: go.Figure) -> None:
    fig.add_annotation(text="No solar input - the panel produces 0 A / 0 W", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False, font=dict(size=15, color=MUTED))


def star_marker(colour: str, size: int = 17) -> dict:
    return dict(symbol="star", size=size, color=colour, line=dict(color="white", width=1))


def fig_iv(result: dict, m: dict) -> go.Figure:
    curve = result["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["Voltage (V)"], y=curve["Current (A)"], mode="lines", name="I-V curve",
        line=dict(color=YELLOW, width=3),
        hovertemplate="V = %{x:.2f} V<br>I = %{y:.3f} A<extra>I-V curve</extra>"))
    if m["is_dark"]:
        no_input_note(fig)
    else:
        fig.add_trace(go.Scatter(
            x=[m["v_mpp"]], y=[m["i_mpp"]], mode="markers", name="Theoretical MPP",
            marker=star_marker(GREEN),
            hovertemplate="Theoretical MPP<br>V = %{x:.2f} V<br>I = %{y:.3f} A<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[m["v_final"]], y=[m["i_final"]], mode="markers", name="Final tracked point (P&O)",
            marker=dict(symbol="diamond", size=12, color=ORANGE, line=dict(color="white", width=1)),
            hovertemplate="Final tracked point<br>V = %{x:.2f} V<br>I = %{y:.3f} A<extra></extra>"))
        fig.update_yaxes(range=[0, max(result["isc_actual"], 1e-6) * 1.15])
    return base_layout(fig, "I-V Characteristics (Simulated)", "Voltage (V)", "Current (A)")


def fig_pv(result: dict, m: dict) -> go.Figure:
    curve = result["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["Voltage (V)"], y=curve["Power (W)"], mode="lines", name="P-V curve",
        line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(46,196,182,0.10)",
        hovertemplate="V = %{x:.2f} V<br>P = %{y:.2f} W<extra>P-V curve</extra>"))
    if m["is_dark"]:
        no_input_note(fig)
    else:
        fig.add_trace(go.Scatter(
            x=[m["v_mpp"]], y=[m["p_mpp"]], mode="markers+text", name="Theoretical MPP",
            marker=star_marker(GREEN, 18),
            text=[f"MPP: {m['v_mpp']:.2f} V, {m['p_mpp']:.2f} W"], textposition="top center",
            textfont=dict(color=GREEN, size=12),
            hovertemplate="Theoretical MPP<br>V = %{x:.2f} V<br>P = %{y:.2f} W<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[m["v_final"]], y=[m["p_final"]], mode="markers", name="Final tracked point (P&O)",
            marker=dict(symbol="diamond", size=12, color=ORANGE, line=dict(color="white", width=1)),
            hovertemplate="Final tracked point<br>V = %{x:.2f} V<br>P = %{y:.2f} W<extra></extra>"))
        fig.update_yaxes(range=[0, m["p_mpp"] * 1.25])
    return base_layout(fig, "P-V Characteristics (Simulated)", "Voltage (V)", "Power (W)")


def padded_range(values, pad_fraction: float = 0.08, min_pad: float = 0.3):
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(min_pad, pad_fraction * (hi - lo))
    return [max(0.0, lo - pad), hi + pad]


def fig_tracking_voltage(result: dict, m: dict) -> go.Figure:
    t = result["tracking"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["Iteration"], y=t["Voltage (V)"], mode="lines", name="Operating voltage",
        line=dict(color="rgba(143,163,181,0.65)", width=1.6), hoverinfo="skip"))
    directions = [
        (1, "Voltage increased (+1)", GREEN, "triangle-up"),
        (-1, "Voltage decreased (-1)", RED, "triangle-down"),
        (0, "Held / first sample (0)", MUTED, "circle"),
    ]
    for value, name, colour, symbol in directions:
        sub = t[t["Direction"] == value]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Iteration"], y=sub["Voltage (V)"], mode="markers", name=name,
            marker=dict(size=8, color=colour, symbol=symbol),
            customdata=sub[["Power (W)", "Delta P (W)"]].to_numpy(),
            hovertemplate=("Iteration %{x}<br>V = %{y:.2f} V<br>P = %{customdata[0]:.2f} W"
                           "<br>Delta P = %{customdata[1]:+.3f} W<extra></extra>")))
    last = t.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last["Iteration"]], y=[last["Voltage (V)"]], mode="markers", name="Final tracked point",
        marker=star_marker(ORANGE, 16),
        hovertemplate="Final point<br>Iteration %{x}<br>V = %{y:.2f} V<extra></extra>"))
    values = list(t["Voltage (V)"])
    if m["is_dark"]:
        no_input_note(fig)
    else:
        fig.add_trace(go.Scatter(
            x=[t["Iteration"].iloc[0], t["Iteration"].iloc[-1]], y=[m["v_mpp"], m["v_mpp"]],
            mode="lines", name=f"Theoretical MPP voltage ({m['v_mpp']:.2f} V)",
            line=dict(color=GREEN, width=2, dash="dash"), hoverinfo="skip"))
        values.append(m["v_mpp"])
    fig.update_yaxes(range=padded_range(values))
    return base_layout(fig, "MPPT Tracking: Operating Voltage per Iteration (P&O)",
                       "Iteration", "Operating voltage (V)")


def fig_tracking_power(result: dict, m: dict) -> go.Figure:
    t = result["tracking"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["Iteration"], y=t["Power (W)"], mode="lines+markers", name="Power tracked by P&O",
        line=dict(color=YELLOW, width=2.5), marker=dict(size=6, color=YELLOW),
        hovertemplate="Iteration %{x}<br>P = %{y:.3f} W<extra></extra>"))
    last = t.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last["Iteration"]], y=[last["Power (W)"]], mode="markers+text", name="Final tracked point",
        marker=star_marker(ORANGE, 17), text=[f"{last['Power (W)']:.2f} W"], textposition="bottom center",
        textfont=dict(color=ORANGE, size=12),
        hovertemplate="Final point<br>Iteration %{x}<br>P = %{y:.3f} W<extra></extra>"))
    values = list(t["Power (W)"])
    if m["is_dark"]:
        no_input_note(fig)
    else:
        fig.add_trace(go.Scatter(
            x=[t["Iteration"].iloc[0], t["Iteration"].iloc[-1]], y=[m["p_mpp"], m["p_mpp"]],
            mode="lines", name=f"Theoretical max power ({m['p_mpp']:.2f} W)",
            line=dict(color=GREEN, width=2, dash="dash"), hoverinfo="skip"))
        values.append(m["p_mpp"])
    fig.update_yaxes(range=padded_range(values, min_pad=0.5))
    return base_layout(fig, "MPPT Power Tracking: Power per Iteration (P&O)", "Iteration", "Power (W)")


def fig_compare_power(rows: list) -> go.Figure:
    ordered = sorted(rows, key=lambda r: r["irradiance"])
    x = [r["irradiance"] for r in ordered]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[r["p_mpp"] for r in ordered], mode="lines+markers+text", name="Theoretical max power",
        line=dict(color=GREEN, width=3), marker=dict(size=11, color=GREEN),
        text=[f"{r['p_mpp']:.1f} W" for r in ordered], textposition="top left",
        textfont=dict(color=GREEN),
        hovertemplate="G = %{x} W/m²<br>Max power = %{y:.2f} W<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=[r["p_final"] for r in ordered], mode="markers", name="Final tracked power (P&O)",
        marker=dict(symbol="diamond", size=11, color=ORANGE, line=dict(color="white", width=1)),
        hovertemplate="G = %{x} W/m²<br>Tracked power = %{y:.2f} W<extra></extra>"))
    top = max([r["p_mpp"] for r in ordered] + [1.0])
    fig.update_yaxes(range=[0, top * 1.2])
    fig.update_xaxes(tickvals=x)
    return base_layout(fig, "Maximum Power vs Irradiance (Simulated)", "Irradiance (W/m²)", "Power (W)", 380)


def fig_compare_pv(rows: list, bundles: list) -> go.Figure:
    colours = [YELLOW, TEAL, BLUE]
    fig = go.Figure()
    for colour, (g, label), bundle, row in zip(colours, COMPARE_LEVELS, bundles, rows):
        result = bundle["result"]
        fig.add_trace(go.Scatter(
            x=result["curve"]["Voltage (V)"], y=result["curve"]["Power (W)"], mode="lines",
            name=f"{g} W/m² ({label})", line=dict(color=colour, width=3),
            hovertemplate=f"{g} W/m²<br>V = %{{x:.2f}} V<br>P = %{{y:.2f}} W<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[row["v_mpp"]], y=[row["p_mpp"]], mode="markers", showlegend=False,
            marker=dict(symbol="star", size=14, color=colour, line=dict(color="white", width=1)),
            hovertemplate=f"MPP at {g} W/m²<br>V = %{{x:.2f}} V<br>P = %{{y:.2f}} W<extra></extra>"))
    return base_layout(fig, "P-V Curves under Different Sunlight (stars = theoretical MPP)",
                       "Voltage (V)", "Power (W)", 380)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ☀️ Control Panel")
        st.caption("Results update live as you change inputs. "
                   "Press **Run Simulation** to log the current run to the history table.")

        st.markdown("**Solar input**")
        st.checkbox("Simulate night (0 W/m²)", key="zero_sun",
                    help="Shows how the dashboard behaves with no sunlight.")
        st.slider("Solar irradiance (W/m²)", min_value=200, max_value=1000, step=50,
                  key="irr", disabled=bool(st.session_state["zero_sun"]))
        st.slider("MPPT iterations", min_value=5, max_value=200, step=5, key="steps",
                  help="Number of Perturb & Observe iterations. Use a small number to watch it still tracking.")
        st.slider("Cell temperature (°C)", min_value=-10, max_value=75, step=1, key="temp")

        st.markdown("**Solar panel specification (STC)**")
        st.number_input("Open-circuit voltage Voc (V)", min_value=6.0, max_value=60.0, step=0.5,
                        format="%.1f", key="voc")
        st.number_input("Short-circuit current Isc (A)", min_value=0.5, max_value=15.0, step=0.5,
                        format="%.1f", key="isc")

        with st.expander("Advanced"):
            st.number_input("P&O perturbation step (V)", min_value=0.05, max_value=1.0, step=0.05,
                            format="%.2f", key="pert",
                            help="Voltage change applied at every iteration (default 0.2 V).")

        st.button("▶ Run Simulation", type="primary", on_click=log_run, key="btn_run")
        st.button("↺ Reset inputs", on_click=reset_inputs, key="btn_reset")

        st.divider()
        st.caption("🟡 **SIMULATION MODE**: every value is produced by the software model in "
                   "`mppt.py`. No real panel, converter or battery is measured.")


def render_header(params: dict, m: dict) -> None:
    runs = len(st.session_state["history"])
    system = {
        STATUS_CONVERGED: ("Operating: MPP reached", GREEN),
        STATUS_TRACKING: ("Operating: tracking MPP...", YELLOW),
        STATUS_NO_INPUT: ("Standby: no solar input", MUTED),
    }[m["status"]]
    pills = "".join([
        pill("⚠ SIMULATION MODE", YELLOW),
        pill(f"☀ Irradiance: {m['irradiance']:.0f} W/m²", ORANGE),
        pill(f"⚙ System: {system[0]}", system[1]),
        pill(f"● Simulation: live · {runs} run{'s' if runs != 1 else ''} logged", TEAL),
    ])
    html = (f'<div class="hero"><h1>☀️ {APP_TITLE}</h1><div class="sub">{APP_SUBTITLE}</div>'
            f'<div>{pills}</div>'
            f'<div class="simbanner"><b>SIMULATION MODE:</b> all values on this dashboard are simulated by '
            f'a software model. They are <b>not</b> actual hardware measurements.</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def render_panel_monitoring(m: dict) -> None:
    st.header("☀️ Solar Panel Monitoring", divider="yellow")
    st.caption("Operating point = where the P&O tracker ended up after the chosen number of iterations "
               "(what the simulated panel is delivering). The theoretical MPP is the ideal peak of the curve.")
    sub_dark = "No solar input"
    cols = st.columns(4)
    show_card(cols[0], "Panel voltage (operating)", fmt(m["v_final"]), "V",
              sub_dark if m["is_dark"] else f"Theoretical MPP voltage: {m['v_mpp']:.2f} V", YELLOW)
    show_card(cols[1], "Panel current (operating)", fmt(m["i_final"], 3), "A",
              sub_dark if m["is_dark"] else f"Theoretical MPP current: {m['i_mpp']:.3f} A", TEAL)
    show_card(cols[2], "Panel power (operating)", fmt(m["p_final"]), "W",
              sub_dark if m["is_dark"] else f"Theoretical max power: {m['p_mpp']:.2f} W", GREEN)
    voc_text = "Voc at this light: N/A" if m["voc_actual"] is None else f"Voc at this light: {m['voc_actual']:.2f} V"
    show_card(cols[3], "Solar irradiance", f"{m['irradiance']:.0f}", "W/m²",
              f"Isc at this light: {m['isc_actual']:.2f} A · {voc_text}", ORANGE)


def render_mppt_section(m: dict) -> None:
    st.header("🎯 MAXIMUM POWER POINT TRACKING", divider="yellow")

    group_title("Theoretical maximum power point", "FROM MODEL")
    cols = st.columns(3)
    show_card(cols[0], "Theoretical MPP voltage", fmt(m["v_mpp"]), "V", "Peak of the simulated P-V curve", GREEN)
    show_card(cols[1], "Theoretical MPP current", fmt(m["i_mpp"], 3), "A", "Current at the peak", GREEN)
    show_card(cols[2], "Theoretical maximum power", fmt(m["p_mpp"]), "W", "Best the panel can deliver here", GREEN)

    group_title("Final tracked point", "PERTURB & OBSERVE")
    cols = st.columns(3)
    show_card(cols[0], "Final tracked voltage", fmt(m["v_final"]), "V", "Last P&O iteration", YELLOW)
    show_card(cols[1], "Final tracked current", fmt(m["i_final"], 3), "A", "Last P&O iteration", YELLOW)
    show_card(cols[2], "Final tracked power", fmt(m["p_final"]), "W", "Last P&O iteration", YELLOW)

    group_title("Tracking performance")
    colour = STATUS_COLOURS[m["status"]]
    cols = st.columns(3)
    if m["efficiency"] is None:
        show_card(cols[0], "Tracking efficiency", "N/A", "", "No solar input, nothing to track", MUTED)
        show_card(cols[1], "Power shortfall", "N/A", "", "No solar input, nothing to track", MUTED)
    else:
        show_card(cols[0], "Tracking efficiency", fmt(m["efficiency"]), "%",
                  "Final tracked power ÷ theoretical max power × 100", colour)
        show_card(cols[1], "Power shortfall", fmt(m["shortfall"]), "%",
                  "How far the final point is below the theoretical maximum", colour)
    show_card(cols[2], "Tracking status", m["status"], "",
              f"'Converged' = within {m['tolerance_pct']:.0f} % of theoretical maximum power", colour)
    if m["efficiency"] is not None:
        st.progress(min(1.0, max(0.0, m["efficiency"] / 100.0)),
                    text=f"Tracking efficiency: {m['efficiency']:.2f} %")
    if m["status"] == STATUS_TRACKING:
        st.info("The tracker has not reached the maximum power point yet. "
                "Increase the number of MPPT iterations in the sidebar to let it converge.")


def render_graphs(result: dict, m: dict) -> None:
    st.header("📈 Interactive Graphs", divider="yellow")
    st.caption("All curves and tracking histories are the real output of `simulate_mppt()`. "
               "Hover over a point for exact values; use the toolbar to zoom.")
    left, right = st.columns(2)
    with left:
        stretch_call(st.plotly_chart, fig_iv(result, m), key="fig_iv")
    with right:
        stretch_call(st.plotly_chart, fig_pv(result, m), key="fig_pv")
    left, right = st.columns(2)
    with left:
        stretch_call(st.plotly_chart, fig_tracking_voltage(result, m), key="fig_track_v")
    with right:
        stretch_call(st.plotly_chart, fig_tracking_power(result, m), key="fig_track_p")
    st.caption("In the voltage graph, ▲ / ▼ show whether P&O increased or decreased the voltage to reach that "
               "iteration. Near the maximum power point P&O keeps oscillating by one step, "
               "which is normal behaviour for this algorithm.")

    with st.expander("View raw P&O tracking data (from simulate_mppt)"):
        tracking = result["tracking"]
        stretch_call(st.dataframe, tracking, hide_index=True)
        st.download_button("⬇ Download tracking data (CSV)", data=to_csv_bytes(tracking),
                           file_name="mppt_tracking_data.csv", mime="text/csv", key="dl_tracking")


def render_battery(m: dict) -> None:
    st.header("🔋 Battery Monitoring (Estimated)", divider="yellow")
    st.markdown('<span class="tag">SIMULATED ESTIMATE · NOT A MEASUREMENT</span>', unsafe_allow_html=True)
    st.caption("Illustrative estimate for the solar panel → DC-DC converter → battery chain. "
               "It does not model battery state of charge, a real charging profile or battery protection.")
    left, right = st.columns(2)
    with left:
        st.slider("Battery voltage (V) - assumed", min_value=3.0, max_value=24.0, step=0.5, key="batt_v")
    with right:
        st.slider("Converter efficiency (%) - assumed", min_value=50, max_value=100, step=1, key="conv_eff")

    battery_v = float(st.session_state["batt_v"])
    est = estimate_battery(m["p_mpp"], st.session_state["conv_eff"], battery_v)

    cols = st.columns(4)
    show_card(cols[0], "Battery voltage (assumed)", f"{battery_v:.1f}", "V", "Set by you, not measured", BLUE)
    show_card(cols[1], "Est. converter output power", fmt(est["p_out"]), "W",
              f"Solar max power {m['p_mpp']:.2f} W × {st.session_state['conv_eff']} %", TEAL)
    show_card(cols[2], "Est. charging current", fmt(est["current"], 3), "A",
              "Estimated output power ÷ battery voltage", YELLOW)
    status_text = "Charging (estimated)" if est["charging"] else "Not charging (no solar input)"
    show_card(cols[3], "Charging status", status_text, "", "Estimated from simulated solar power",
              GREEN if est["charging"] else MUTED)

    st.caption(
        f"Estimated converter loss: {est['loss']:.2f} W. Solar *input* power is "
        f"{m['p_mpp']:.2f} W (theoretical maximum); the converter *output* power is lower because of losses. "
        f"Using the final tracked power instead would give about "
        f"{estimate_battery(m['p_final'], st.session_state['conv_eff'], battery_v)['p_out']:.2f} W.")
    if est["charging"] and battery_v >= m["v_mpp"]:
        st.info("The assumed battery voltage is at or above the panel's MPP voltage. Depending on the converter "
                "topology this may not be possible; this simple estimate does not model that.")


def render_comparison(params: dict) -> None:
    st.header("🌤️ SOLAR PERFORMANCE UNDER DIFFERENT SUNLIGHT CONDITIONS", divider="yellow")
    st.caption("Three fixed conditions simulated with the same panel settings, temperature and iteration count "
               "as the sidebar, using the real `simulate_mppt()`.")
    rows, bundles, table = [], [], []
    for irradiance, label in COMPARE_LEVELS:
        bundle = run_simulation(**{**params, "irradiance": float(irradiance)})
        metrics = derive_metrics(bundle, irradiance)
        bundles.append(bundle)
        rows.append(metrics)
        table.append({
            "Condition": label,
            "Irradiance (W/m²)": irradiance,
            "MPP Voltage (V)": round(metrics["v_mpp"], 2),
            "MPP Current (A)": round(metrics["i_mpp"], 3),
            "Max Power (W)": round(metrics["p_mpp"], 2),
            "Final Tracked Power (W)": round(metrics["p_final"], 2),
            "Tracking Efficiency (%)": None if metrics["efficiency"] is None else round(metrics["efficiency"], 2),
        })
    stretch_call(st.dataframe, pd.DataFrame(table), hide_index=True)
    left, right = st.columns(2)
    with left:
        stretch_call(st.plotly_chart, fig_compare_power(rows), key="fig_cmp_power")
    with right:
        stretch_call(st.plotly_chart, fig_compare_pv(rows, bundles), key="fig_cmp_pv")


def render_history() -> None:
    st.header("🗂️ MPPT SIMULATION HISTORY", divider="yellow")
    df = pd.DataFrame(st.session_state["history"], columns=HISTORY_COLUMNS)
    st.caption("Every press of **Run Simulation** adds one row. The table is kept for this browser session.")
    if df.empty:
        st.info("No simulations logged yet. Press ▶ Run Simulation in the sidebar to add the first row.")
    else:
        stretch_call(st.dataframe, df, hide_index=True)
    left, right, _ = st.columns([1, 1, 3])
    with left:
        st.download_button("⬇ Download CSV", data=to_csv_bytes(df), file_name="mppt_simulation_history.csv",
                           mime="text/csv", disabled=df.empty, key="dl_history")
    with right:
        st.button("🗑 Clear history", on_click=clear_history, disabled=df.empty, key="btn_clear")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Smart Solar Power Unit with MPPT", page_icon="☀️",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    flash = st.session_state.pop("flash", None)
    if flash:
        kind, message = flash
        st.toast(message, icon={"ok": "✅", "info": "ℹ️", "error": "⚠️"}.get(kind, "ℹ️"))

    params = current_params()
    try:
        bundle = run_simulation(**params)
    except ValueError as exc:
        st.error(f"The simulation could not be run with these inputs: {exc}")
        st.stop()
        return
    except Exception as exc:  # unexpected problem inside the model
        st.error(f"Unexpected error while running the simulation: {exc}")
        st.stop()
        return

    result = bundle["result"]
    m = derive_metrics(bundle, params["irradiance"])

    render_header(params, m)
    render_panel_monitoring(m)
    render_mppt_section(m)
    render_graphs(result, m)
    render_battery(m)
    try:
        render_comparison(params)
    except ValueError as exc:
        st.error(f"The irradiance comparison could not be computed: {exc}")
    render_history()

    st.divider()
    st.caption("Smart Solar Power Unit with MPPT · college project dashboard · "
               "all values are simulated (mppt.py, Perturb & Observe) and are not hardware measurements.")


if __name__ == "__main__":
    main()
