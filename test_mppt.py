"""
Automated tests for mppt.py  (run with:  python -m pytest -v)

All tests check numerical properties of the SIMULATED solar panel and the
Perturb & Observe tracker - not just that "something is returned".
"""

import math

import numpy as np
import pandas as pd
import pytest

from mppt import SolarPanel, perturb_and_observe, simulate_mppt, summarize_result

EXPECTED_KEYS = {"curve", "tracking", "v_mpp", "i_mpp", "p_mpp", "isc_actual"}
CURVE_COLS = ["Voltage (V)", "Current (A)", "Power (W)"]
IRRADIANCES = [1000, 800, 600, 400, 200, 100]


# ---------------------------------------------------------------- 1. defaults
def test_default_call_works_and_has_expected_shape():
    result = simulate_mppt()
    assert set(result.keys()) == EXPECTED_KEYS
    assert len(result["tracking"]) == 60          # default steps
    assert len(result["curve"]) >= 200            # smooth enough to plot
    assert result["p_mpp"] > 0
    # default 800 W/m^2 -> Isc scales linearly: 3 A * 0.8 = 2.4 A
    assert result["isc_actual"] == pytest.approx(2.4, rel=1e-9)


def test_original_positional_call_still_works():
    result = simulate_mppt(800, 60, 21, 3)
    assert set(result.keys()) == EXPECTED_KEYS


# --------------------------------------------------- 2. high / medium / low G
@pytest.mark.parametrize("g", [1000, 600, 200])
def test_high_medium_low_irradiance_isc_and_mpp(g):
    r = simulate_mppt(irradiance=g)
    assert r["isc_actual"] == pytest.approx(3.0 * g / 1000.0, rel=1e-9)
    assert 0 < r["v_mpp"] < 21
    assert 0 < r["i_mpp"] < r["isc_actual"]
    # STC panel (21 V, 3 A) should give a plausible ~50 W at 1000 W/m^2
    if g == 1000:
        assert 40 < r["p_mpp"] < 55


def test_lower_irradiance_gives_lower_max_power():
    powers = [simulate_mppt(irradiance=g)["p_mpp"] for g in IRRADIANCES]
    assert all(a > b for a, b in zip(powers, powers[1:]))     # strictly decreasing
    # Power is close to proportional to irradiance, but slightly LESS than
    # proportional because Voc falls logarithmically as light decreases.
    ratio = powers[-1] / powers[0]            # 100 W/m^2 vs 1000 W/m^2
    assert 0.07 < ratio < 0.10


def test_voc_decreases_slowly_with_lower_irradiance():
    voc = [SolarPanel(irradiance=g).voc_actual for g in IRRADIANCES]
    assert all(a > b for a, b in zip(voc, voc[1:]))
    assert voc[-1] > 0.75 * voc[0]       # Voc falls only logarithmically


def test_higher_temperature_lowers_power():
    cool = simulate_mppt(irradiance=1000, temperature=15)["p_mpp"]
    hot = simulate_mppt(irradiance=1000, temperature=60)["p_mpp"]
    assert hot < cool


# ------------------------------------------------ 3. zero / invalid irradiance
def test_zero_irradiance_is_defined_as_dark_panel():
    r = simulate_mppt(irradiance=0)
    assert r["p_mpp"] == 0 and r["i_mpp"] == 0 and r["v_mpp"] == 0
    assert r["isc_actual"] == 0
    assert (r["curve"]["Current (A)"] == 0).all()
    assert (r["curve"]["Power (W)"] == 0).all()
    assert (r["tracking"]["Power (W)"] == 0).all()
    assert np.isfinite(r["tracking"].to_numpy()).all()
    assert summarize_result(r)["converged"] is True   # nothing to track


@pytest.mark.parametrize("bad", [-1, -0.001, 1501, float("nan"), float("inf"), "abc", None, True])
def test_invalid_irradiance_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(irradiance=bad)


# --------------------------------------- 4. invalid voltage / current / steps
@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "x"])
def test_invalid_voc_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(voc=bad)


@pytest.mark.parametrize("bad", [0, -3, float("nan"), float("inf"), "x"])
def test_invalid_isc_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(isc=bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, float("nan"), float("inf"), "ten", None, True])
def test_invalid_steps_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(steps=bad)


@pytest.mark.parametrize("bad", [-1.0, 25.0, float("nan"), "x"])   # voc is 21 V
def test_invalid_initial_voltage_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(v_init=bad)


@pytest.mark.parametrize("bad", [0, -0.2, float("nan"), "x"])
def test_invalid_step_size_raises(bad):
    with pytest.raises(ValueError):
        simulate_mppt(step=bad)


def test_initial_voltage_above_present_voc_is_clipped():
    # at 200 W/m^2 the real Voc is below 21 V; 20.9 V start must be clipped into range
    r = simulate_mppt(irradiance=200, v_init=20.9)
    voc_now = SolarPanel(irradiance=200).voc_actual
    assert 20.9 > voc_now
    assert r["tracking"]["Voltage (V)"].iloc[0] == pytest.approx(voc_now)
    assert r["tracking"]["Voltage (V)"].max() <= voc_now + 1e-12
    assert summarize_result(r)["converged"]       # still finds the MPP from the top


# ------------------------------------------ 5. non-negative current and power
@pytest.mark.parametrize("g", IRRADIANCES)
def test_voltage_current_power_non_negative(g):
    r = simulate_mppt(irradiance=g)
    for df in (r["curve"], r["tracking"]):
        assert (df["Voltage (V)"] >= 0).all()
        assert (df["Current (A)"] >= 0).all()
        assert (df["Power (W)"] >= 0).all()


# ---------------------------------------- 6. endpoints: Isc and Voc behaviour
@pytest.mark.parametrize("g", IRRADIANCES)
def test_curve_endpoints_short_and_open_circuit(g):
    r = simulate_mppt(irradiance=g)
    c = r["curve"]
    first, last = c.iloc[0], c.iloc[-1]
    # short circuit: V = 0 -> I = Isc, P = 0
    assert first["Voltage (V)"] == 0
    assert first["Current (A)"] == pytest.approx(r["isc_actual"], rel=1e-9)
    assert first["Power (W)"] == 0
    # open circuit: I -> 0 and P -> 0
    assert last["Voltage (V)"] == pytest.approx(SolarPanel(irradiance=g).voc_actual)
    assert last["Current (A)"] == pytest.approx(0.0, abs=1e-9)
    assert last["Power (W)"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("g", IRRADIANCES)
def test_current_never_increases_with_voltage(g):
    i = simulate_mppt(irradiance=g)["curve"]["Current (A)"].to_numpy()
    assert (np.diff(i) <= 1e-12).all()


# ----------------------------------------- 7. MPP identified from the curve
@pytest.mark.parametrize("g", IRRADIANCES)
def test_mpp_is_interior_peak_of_power_curve(g):
    c = simulate_mppt(irradiance=g)["curve"]
    k = int(c["Power (W)"].idxmax())
    assert 0 < k < len(c) - 1                       # maximum is inside the range
    # power rises before the peak and falls after it (single-peak P-V curve)
    p = c["Power (W)"].to_numpy()
    assert (np.diff(p[: k + 1]) >= -1e-12).all()
    assert (np.diff(p[k:]) <= 1e-12).all()


# --------------------------------- 8. returned MPP consistent with the curve
@pytest.mark.parametrize("g", IRRADIANCES)
def test_mpp_matches_curve_maximum(g):
    r = simulate_mppt(irradiance=g)
    c = r["curve"]
    k = int(c["Power (W)"].idxmax())
    dv = c["Voltage (V)"].iloc[1] - c["Voltage (V)"].iloc[0]
    assert abs(r["v_mpp"] - c["Voltage (V)"].iloc[k]) <= dv          # within one grid step
    assert r["p_mpp"] >= c["Power (W)"].max() - 1e-9                 # true peak >= grid peak
    assert r["p_mpp"] == pytest.approx(c["Power (W)"].max(), rel=1e-4)
    assert r["p_mpp"] == pytest.approx(r["v_mpp"] * r["i_mpp"], rel=1e-12)   # P = V x I


@pytest.mark.parametrize("g", IRRADIANCES)
def test_curve_rows_obey_p_equals_v_times_i(g):
    c = simulate_mppt(irradiance=g)["curve"]
    assert np.allclose(c["Power (W)"], c["Voltage (V)"] * c["Current (A)"], rtol=1e-12, atol=1e-12)


def test_mpp_is_not_at_other_voltages():
    """Power at 60 % and 100 % of Voc must be clearly below the MPP power."""
    g = 1000
    panel = SolarPanel(irradiance=g)
    p_mpp = simulate_mppt(irradiance=g)["p_mpp"]
    assert panel.power(0.5 * panel.voc_actual) < 0.95 * p_mpp
    assert panel.power(panel.voc_actual) < 1e-6


# ------------------------------------------ 9. P&O convergence near theory
@pytest.mark.parametrize("g", IRRADIANCES)
def test_po_converges_close_to_theoretical_mpp(g):
    r = simulate_mppt(irradiance=g, steps=60)
    s = summarize_result(r, tolerance=0.02)
    assert s["converged"], s
    assert s["final_power"] <= r["p_mpp"] + 1e-9           # cannot beat the true MPP
    assert abs(s["final_voltage"] - r["v_mpp"]) <= 0.5      # within ~2 perturbation steps


@pytest.mark.parametrize("v_start", [1.0, 5.0, 10.5, 19.0, 20.5])
def test_po_converges_from_below_and_above_the_mpp(v_start):
    r = simulate_mppt(irradiance=1000, steps=120, v_init=v_start)
    assert summarize_result(r, 0.02)["converged"]


def test_po_really_iterates_and_is_not_the_theoretical_answer():
    r = simulate_mppt(irradiance=1000, steps=60)
    t = r["tracking"]
    v = t["Voltage (V)"].to_numpy()
    assert v[0] == pytest.approx(10.5)                       # starts at 0.5 * Voc
    assert t["Power (W)"].iloc[0] < 0.9 * r["p_mpp"]         # starts far from the MPP
    steps_taken = np.abs(np.diff(v))
    # every voltage change is exactly one perturbation step (0.2 V) or a hold
    assert np.all(np.isclose(steps_taken, 0.2) | np.isclose(steps_taken, 0.0))
    # climbs monotonically while far below the MPP, then oscillates around it
    far_below = v[v < r["v_mpp"] - 1.0]
    assert np.all(np.diff(far_below) > 0)
    assert abs(v[-1] - r["v_mpp"]) < 0.5
    # tracked values are really on the panel: I and P recomputed from V match
    panel = SolarPanel(irradiance=1000)
    assert np.allclose(t["Current (A)"], panel.current(v), rtol=1e-12)
    assert np.allclose(t["Power (W)"], v * panel.current(v), rtol=1e-12)


def test_po_direction_logic_both_sides_of_mpp():
    # starting below the MPP the voltage must first go UP, above it first go DOWN
    below = simulate_mppt(irradiance=1000, v_init=10.0, steps=5)["tracking"]
    above = simulate_mppt(irradiance=1000, v_init=20.0, steps=5)["tracking"]
    assert below["Direction"].iloc[1:].tolist() == [1, 1, 1, 1]
    assert above["Direction"].iloc[1] == 1                    # first perturbation is upward
    assert above["Direction"].iloc[2] == -1                   # power fell -> reversed
    assert above["Direction"].iloc[3] == -1                   # power rose -> keep going down
    assert above["Voltage (V)"].iloc[-1] < above["Voltage (V)"].iloc[1]


def test_po_oscillates_around_mpp_after_convergence():
    t = simulate_mppt(irradiance=1000, steps=100)["tracking"]
    tail = t["Direction"].iloc[-12:].tolist()
    assert 1 in tail and -1 in tail           # steady-state oscillation (known P&O behaviour)
    assert t["Voltage (V)"].iloc[-12:].max() - t["Voltage (V)"].iloc[-12:].min() <= 0.4 + 1e-9


def test_few_iterations_do_not_reach_mpp_honest_history():
    r = simulate_mppt(irradiance=1000, steps=5)
    assert len(r["tracking"]) == 5
    assert not summarize_result(r, 0.02)["converged"]


def test_smaller_step_gives_smaller_oscillation():
    big = simulate_mppt(irradiance=1000, steps=200, step=0.5)["tracking"]["Voltage (V)"].iloc[-20:]
    small = simulate_mppt(irradiance=1000, steps=200, step=0.1)["tracking"]["Voltage (V)"].iloc[-20:]
    assert (small.max() - small.min()) < (big.max() - big.min())


def test_tracker_follows_a_change_in_irradiance():
    """Run P&O at 1000 W/m^2, then hand the final voltage to a 300 W/m^2 panel."""
    r_hi = simulate_mppt(irradiance=1000, steps=60)
    v_end = float(r_hi["tracking"]["Voltage (V)"].iloc[-1])
    panel_lo = SolarPanel(irradiance=300)
    t_lo = perturb_and_observe(panel_lo, steps=60, v_init=v_end, step=0.2)
    v_mpp_lo, _, p_mpp_lo = panel_lo.mpp()
    assert t_lo["Power (W)"].iloc[-1] >= 0.98 * p_mpp_lo
    assert abs(t_lo["Voltage (V)"].iloc[-1] - v_mpp_lo) < 0.5


# ----------------------- 10 & 11. exact names, keys and dashboard-friendly types
def test_dict_keys_and_dataframe_columns_are_exact():
    r = simulate_mppt()
    assert set(r.keys()) == EXPECTED_KEYS
    assert list(r["curve"].columns) == CURVE_COLS
    for col in CURVE_COLS:
        assert col in r["tracking"].columns


def test_output_types_for_dashboard():
    r = simulate_mppt()
    assert isinstance(r["curve"], pd.DataFrame)
    assert isinstance(r["tracking"], pd.DataFrame)
    for key in ("v_mpp", "i_mpp", "p_mpp", "isc_actual"):
        assert type(r[key]) is float            # plain Python float, not numpy
    for df in (r["curve"], r["tracking"]):
        assert all(pd.api.types.is_float_dtype(df[c]) for c in CURVE_COLS)
        assert df.index.is_unique
    assert isinstance(r["curve"].to_dict("records"), list)    # JSON-style conversion works


# ---------------------------------------- 12. no NaN / inf under valid inputs
@pytest.mark.parametrize("g", [0, 1, 50, 200, 600, 1000, 1500])
@pytest.mark.parametrize("temp", [-10, 25, 75])
def test_no_nan_or_inf(g, temp):
    r = simulate_mppt(irradiance=g, temperature=temp)
    for df in (r["curve"], r["tracking"]):
        assert np.isfinite(df.to_numpy(dtype=float)).all()
    for key in ("v_mpp", "i_mpp", "p_mpp", "isc_actual"):
        assert math.isfinite(r[key])


@pytest.mark.parametrize("voc,isc", [(12, 1.0), (21, 3), (37, 8.5), (44.2, 9.9)])
def test_other_panel_sizes_work(voc, isc):
    r = simulate_mppt(irradiance=800, steps=300, voc=voc, isc=isc, step=0.1)
    assert 0 < r["v_mpp"] < voc
    assert 0.6 * voc < r["v_mpp"]
    assert summarize_result(r, 0.03)["converged"]
    assert np.isfinite(r["tracking"].to_numpy(dtype=float)).all()
