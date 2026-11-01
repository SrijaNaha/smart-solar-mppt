"""
mppt.py - Simulated solar panel + Perturb & Observe (P&O) MPPT
================================================================
Project : Smart Solar Power Unit with MPPT (college project, Person 1 module)

EVERYTHING IN THIS FILE IS A SOFTWARE SIMULATION.
No physical solar panel, DC-DC converter or battery is measured or controlled.

Dashboard integration (Person 2)
--------------------------------
    from mppt import simulate_mppt
    result = simulate_mppt(irradiance=800, steps=60, voc=21, isc=3)
    result.keys() -> curve, tracking, v_mpp, i_mpp, p_mpp, isc_actual

Solar panel model (simplified single-diode model, no series/shunt resistance)
-----------------------------------------------------------------------------
    I(V)  = I_ph - I_0 * (exp(V / a) - 1)          [A]
    P     = V * I                                  [W]

    I_ph  = Isc_stc * (G / 1000) * (1 + alpha * (T - 25))      photo-current = Isc
    a     = n * Ns * k * T_kelvin / q                          "modified" thermal voltage
    Voc   = Voc_stc * (1 + beta * (T - 25)) + a * ln(G / 1000)
    I_0   = I_ph / (exp(Voc / a) - 1)       <- chosen so that I(Voc) = 0

Assumptions / limitations (also listed in README.md):
  * Teaching model, NOT a validated model of any specific commercial panel.
  * Series resistance, shunt resistance, partial shading, bypass diodes,
    cell mismatch, ageing and panel-temperature-vs-irradiance heating are ignored.
  * Only Voc and Isc are given by the caller, so the remaining parameters
    (ideality factor, cell count, temperature coefficients) use fixed
    demonstration defaults (see constants below).
  * Temperature is the cell temperature (default 25 degC, the STC value).
"""

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants and demonstration defaults
# --------------------------------------------------------------------------
BOLTZMANN_K = 1.380649e-23        # J/K
ELECTRON_Q = 1.602176634e-19      # C
G_STC = 1000.0                    # W/m^2, standard test condition irradiance
T_STC = 25.0                      # degC, standard test condition temperature

IDEALITY_FACTOR = 1.3             # diode ideality factor n (typical silicon 1-2)
VOLTS_PER_CELL = 0.6              # used to guess number of series cells: Ns = Voc/0.6
ALPHA_ISC = 0.0005                # +0.05 %/degC  short-circuit current temp. coefficient
BETA_VOC = -0.0035                # -0.35 %/degC  open-circuit voltage temp. coefficient

MAX_IRRADIANCE = 1500.0           # W/m^2, upper limit accepted by the validation
MIN_IRRADIANCE = 1.0              # W/m^2, below this the panel is treated as "dark"
CURVE_POINTS = 501                # number of points in the I-V / P-V curve

CURVE_COLUMNS = ["Voltage (V)", "Current (A)", "Power (W)"]
TRACKING_COLUMNS = ["Iteration", "Voltage (V)", "Current (A)", "Power (W)",
                    "Direction", "Delta P (W)"]


# --------------------------------------------------------------------------
# Input validation helpers (all raise ValueError with a clear message)
# --------------------------------------------------------------------------
def _as_finite_float(value, name):
    """Convert value to float and make sure it is finite (no NaN / inf)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _as_positive_int(value, name):
    """Accept only whole numbers >= 1 (ints, numpy ints, or 60.0 style floats)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return int(number)


# --------------------------------------------------------------------------
# Solar panel model
# --------------------------------------------------------------------------
class SolarPanel:
    """
    Simplified single-diode solar panel (SIMULATED).

    Parameters
    ----------
    irradiance : float   W/m^2, 0 <= G <= 1500. G < 1 W/m^2 is treated as darkness
                         (all outputs are exactly zero; no error is raised).
    voc, isc   : float   open-circuit voltage [V] and short-circuit current [A]
                         at STC (1000 W/m^2, 25 degC). Must be > 0.
    temperature: float   cell temperature in degC (default 25).
    """

    def __init__(self, irradiance=800, voc=21, isc=3, temperature=T_STC):
        g = _as_finite_float(irradiance, "irradiance")
        voc = _as_finite_float(voc, "voc")
        isc = _as_finite_float(isc, "isc")
        temp = _as_finite_float(temperature, "temperature")

        if g < 0:
            raise ValueError(f"irradiance must be >= 0 W/m^2, got {g}")
        if g > MAX_IRRADIANCE:
            raise ValueError(f"irradiance must be <= {MAX_IRRADIANCE} W/m^2, got {g}")
        if voc <= 0:
            raise ValueError(f"voc must be > 0 V, got {voc}")
        if isc <= 0:
            raise ValueError(f"isc must be > 0 A, got {isc}")
        if not -50.0 <= temp <= 150.0:
            raise ValueError(f"temperature must be within -50..150 degC, got {temp}")

        self.irradiance = g
        self.voc_stc = voc
        self.isc_stc = isc
        self.temperature = temp
        self.is_dark = g < MIN_IRRADIANCE

        # Number of series cells guessed from Voc (e.g. 21 V -> 35 cells)
        self.n_cells = max(1, int(round(voc / VOLTS_PER_CELL)))
        t_kelvin = temp + 273.15
        # "a" = n * Ns * Vt  (modified diode thermal voltage), in volts
        self.a = (IDEALITY_FACTOR * self.n_cells * BOLTZMANN_K * t_kelvin) / ELECTRON_Q

        if self.is_dark:
            self.isc_actual = 0.0
            self.voc_actual = 0.0
            self.i0 = 0.0
        else:
            # Photo-current (= short-circuit current) scales linearly with irradiance
            self.isc_actual = isc * (g / G_STC) * (1.0 + ALPHA_ISC * (temp - T_STC))
            # Voc drops with temperature and falls logarithmically with irradiance
            self.voc_actual = (voc * (1.0 + BETA_VOC * (temp - T_STC))
                               + self.a * math.log(g / G_STC))
            if self.voc_actual <= 0:
                raise ValueError("irradiance/temperature combination gives Voc <= 0")
            # Diode saturation current chosen so that I(Voc) = 0 exactly
            self.i0 = self.isc_actual / math.expm1(self.voc_actual / self.a)

    # ---- electrical behaviour -------------------------------------------
    def current(self, voltage):
        """Panel current [A] at terminal voltage(s) [V]. Accepts scalar or array.
        Voltages outside 0..Voc give 0 A (model only valid in 0..Voc)."""
        v = np.asarray(voltage, dtype=float)
        if self.is_dark:
            i = np.zeros_like(v)
        else:
            i = self.isc_actual - self.i0 * np.expm1(v / self.a)
            i = np.where((v < 0) | (v > self.voc_actual), 0.0, i)
            i = np.clip(i, 0.0, None)          # remove tiny negative round-off at Voc
        return float(i) if i.ndim == 0 else i

    def power(self, voltage):
        """P = V * I  [W]."""
        v = np.asarray(voltage, dtype=float)
        p = v * np.asarray(self.current(v))
        return float(p) if p.ndim == 0 else p

    # ---- curves and theoretical maximum power point ----------------------
    def iv_curve(self, points=CURVE_POINTS):
        """DataFrame with columns 'Voltage (V)', 'Current (A)', 'Power (W)'.
        Voltage runs from 0 (short circuit) to Voc (open circuit)."""
        # In darkness there is no Voc; use the nominal Voc so the axis is still usable.
        v_end = self.voc_stc if self.is_dark else self.voc_actual
        v = np.linspace(0.0, v_end, int(points))
        i = np.asarray(self.current(v), dtype=float)
        p = v * i
        return pd.DataFrame({CURVE_COLUMNS[0]: v, CURVE_COLUMNS[1]: i, CURVE_COLUMNS[2]: p})

    def mpp(self):
        """Theoretical MPP (v_mpp, i_mpp, p_mpp) found by golden-section search.
        P(V) rises then falls with a single peak, so this converges reliably."""
        if self.is_dark:
            return 0.0, 0.0, 0.0
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, self.voc_actual
        x1 = hi - inv_phi * (hi - lo)
        x2 = lo + inv_phi * (hi - lo)
        f1, f2 = self.power(x1), self.power(x2)
        for _ in range(200):
            if hi - lo < 1e-10:
                break
            if f1 < f2:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + inv_phi * (hi - lo)
                f2 = self.power(x2)
            else:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - inv_phi * (hi - lo)
                f1 = self.power(x1)
        v_mpp = 0.5 * (lo + hi)
        i_mpp = self.current(v_mpp)
        return float(v_mpp), float(i_mpp), float(v_mpp * i_mpp)


# --------------------------------------------------------------------------
# Perturb & Observe MPPT
# --------------------------------------------------------------------------
def perturb_and_observe(panel, steps=60, v_init=None, step=0.2):
    """
    Genuine iterative P&O (SIMULATED measurements taken from `panel`).

    Each iteration:
      1. 'measure' current at the present voltage and compute P = V * I
      2. compare P with the previous iteration's power (delta P)
      3. if P went UP   -> keep perturbing in the same direction
         if P went DOWN -> reverse the direction
         if P unchanged -> hold the voltage (nothing to learn; e.g. darkness)
      4. apply the new voltage = V + direction * step (kept inside 0..Voc)

    The algorithm only ever sees measured V, I and P - never the theoretical MPP.
    Returns a DataFrame with one row per iteration (see TRACKING_COLUMNS).
    'Direction' is the perturbation direction used to REACH that row's voltage
    (+1 = voltage increased, -1 = decreased, 0 = held / first sample).
    """
    v_max = panel.voc_stc if panel.is_dark else panel.voc_actual
    v = min(max(float(v_init), 0.0), v_max)       # start point inside valid range
    direction = 1                                  # first perturbation goes upward
    applied_direction = 0                          # nothing applied before 1st sample
    p_prev = None
    eps = 1e-12                                    # "power unchanged" threshold [W]
    rows = []

    for k in range(1, steps + 1):
        i = panel.current(v)                       # measure current
        p = v * i                                  # P = V x I
        delta_p = 0.0 if p_prev is None else p - p_prev
        rows.append((k, v, i, p, applied_direction, delta_p))

        # ---- decide the next perturbation --------------------------------
        hold = False
        if p_prev is not None:
            if delta_p > eps:
                pass                               # power rose: same direction
            elif delta_p < -eps:
                direction = -direction             # power fell: reverse
            else:
                hold = True                        # no change: keep voltage
        p_prev = p

        if hold:
            applied_direction = 0
            continue
        v_new = min(max(v + direction * step, 0.0), v_max)
        if v_new == v:                             # stuck on a boundary: turn around
            direction = -direction
            v_new = min(max(v + direction * step, 0.0), v_max)
        applied_direction = 0 if v_new == v else direction
        v = v_new

    return pd.DataFrame(rows, columns=TRACKING_COLUMNS)


# --------------------------------------------------------------------------
# Dashboard interface (Person 2 imports this)
# --------------------------------------------------------------------------
def simulate_mppt(irradiance=800, steps=60, voc=21, isc=3,
                  temperature=T_STC, v_init=None, step=0.2):
    """
    Simulate a solar panel and track its maximum power point with P&O.

    Required (original) arguments - unchanged:
        irradiance : W/m^2 (0..1500; 0 -> dark panel, all-zero output)
        steps      : number of P&O iterations (whole number >= 1)
        voc        : STC open-circuit voltage [V]
        isc        : STC short-circuit current [A]
    Optional arguments (safe defaults, do not break the old call):
        temperature: cell temperature [degC], default 25
        v_init     : P&O starting voltage [V]; default 0.5 * voc.
                     Must be 0 <= v_init <= voc; if it exceeds the panel's
                     present Voc (low light) it is clipped to that Voc.
        step       : P&O voltage perturbation [V], default 0.2, must be > 0

    Returns dict with EXACTLY these keys:
        curve     : DataFrame ['Voltage (V)', 'Current (A)', 'Power (W)']
        tracking  : DataFrame ['Iteration', 'Voltage (V)', 'Current (A)',
                               'Power (W)', 'Direction', 'Delta P (W)']
        v_mpp, i_mpp, p_mpp : theoretical MPP (float)
        isc_actual : model short-circuit current at this irradiance (float, A)
    All values are SIMULATED.
    """
    steps = _as_positive_int(steps, "steps")
    step = _as_finite_float(step, "step")
    if step <= 0:
        raise ValueError(f"step must be > 0 V, got {step}")

    panel = SolarPanel(irradiance=irradiance, voc=voc, isc=isc, temperature=temperature)

    if v_init is None:
        v_init = 0.5 * panel.voc_stc
    v_init = _as_finite_float(v_init, "v_init")
    if v_init < 0 or v_init > panel.voc_stc:
        raise ValueError(f"v_init must be within 0..{panel.voc_stc} V, got {v_init}")

    curve = panel.iv_curve()
    tracking = perturb_and_observe(panel, steps=steps, v_init=v_init, step=step)
    v_mpp, i_mpp, p_mpp = panel.mpp()

    return {
        "curve": curve,
        "tracking": tracking,
        "v_mpp": float(v_mpp),
        "i_mpp": float(i_mpp),
        "p_mpp": float(p_mpp),
        "isc_actual": float(panel.isc_actual),
    }


def summarize_result(result, tolerance=0.02):
    """
    Compare the FINAL tracked point with the theoretical MPP.
    `tolerance` is the allowed relative power shortfall (0.02 = 2 %).
    Returns a dict (all simulated values). In darkness (p_mpp == 0) the
    tracker is reported as converged only because there is nothing to track.
    """
    last = result["tracking"].iloc[-1]
    p_final = float(last["Power (W)"])
    p_mpp = result["p_mpp"]
    if p_mpp > 0:
        shortfall = (p_mpp - p_final) / p_mpp
        converged = shortfall <= tolerance
    else:
        shortfall, converged = 0.0, True
    return {
        "final_voltage": float(last["Voltage (V)"]),
        "final_current": float(last["Current (A)"]),
        "final_power": p_final,
        "power_shortfall_fraction": float(shortfall),
        "tolerance": float(tolerance),
        "converged": bool(converged),
    }


# --------------------------------------------------------------------------
# Standalone command-line demonstration:  python mppt.py [--plot]
# --------------------------------------------------------------------------
def _demo(levels=(1000, 600, 200), steps=60, tolerance=0.02):
    print("=" * 62)
    print(" Smart Solar Power Unit - MPPT demo (ALL VALUES SIMULATED)")
    print("=" * 62)
    for g in levels:
        res = simulate_mppt(irradiance=g, steps=steps)
        s = summarize_result(res, tolerance)
        print(f"\nIrradiance            : {g} W/m^2")
        print(f"Theoretical MPP V     : {res['v_mpp']:.2f} V")
        print(f"Theoretical MPP I     : {res['i_mpp']:.3f} A")
        print(f"Theoretical max power : {res['p_mpp']:.2f} W")
        print(f"Final tracked V       : {s['final_voltage']:.2f} V")
        print(f"Final tracked I       : {s['final_current']:.3f} A")
        print(f"Final tracked power   : {s['final_power']:.2f} W")
        print(f"Power shortfall       : {100 * s['power_shortfall_fraction']:.2f} % "
              f"(tolerance {100 * tolerance:.0f} %)")
        status = "YES" if s["converged"] else "NO"
        print(f"Tracking within tol.  : {status}  (after {steps} iterations)")


def _plot(levels=(1000, 600, 200), steps=60):
    """Optional standalone plot (needs matplotlib, which is NOT in requirements)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; run: python -m pip install matplotlib")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in levels:
        res = simulate_mppt(irradiance=g, steps=steps)
        line, = ax.plot(res["curve"]["Voltage (V)"], res["curve"]["Power (W)"],
                        label=f"{g} W/m^2")
        ax.plot(res["tracking"]["Voltage (V)"], res["tracking"]["Power (W)"], "o",
                ms=3, color=line.get_color())
        ax.plot(res["v_mpp"], res["p_mpp"], "k*", ms=10)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Simulated P-V curves (dots = P&O iterations, star = theoretical MPP)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    import sys
    _demo()
    if "--plot" in sys.argv:
        _plot()
