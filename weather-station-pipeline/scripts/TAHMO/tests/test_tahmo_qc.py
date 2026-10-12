"""
test_tahmo_qc.py
----------------
Unit tests for TAHMO Quality Control & derivations module.
"""

import os
import sys
import numpy as np

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import tahmo_derivations as td


def test_temperature_range_and_fusion():
    te_slots = np.array([[25.0, 75.0], [20.0, 22.0]], dtype=np.float32)
    te_rec = td.reconcile_collocated_scalars(te_slots, "te")
    assert np.isclose(te_rec[0], 25.0), f"Expected 25.0, got {te_rec[0]}"
    assert np.isclose(te_rec[1], 21.0), f"Expected 21.0, got {te_rec[1]}"
    print("  [OK] Temperature range gate & slot fusion passed.")


def test_wind_vector_wrap_around():
    ws_slots = np.array([[5.0, 5.0]], dtype=np.float32)
    wd_slots = np.array([[359.0, 1.0]], dtype=np.float32)
    ws_r, wd_r, u_r, v_r = td.reconcile_collocated_wind(ws_slots, wd_slots)
    assert np.isclose(ws_r[0], 5.0, atol=0.1), f"Expected ws ~ 5.0, got {ws_r[0]}"
    assert np.isclose(wd_r[0], 0.0, atol=2.0) or np.isclose(wd_r[0], 360.0, atol=2.0), f"Expected wd ~ 0/360, got {wd_r[0]}"
    print("  [OK] Wind vector wrap-around passed.")


def test_ultrasonic_blockage_rejection():
    ws_block = np.array([[0.0, 5.0]], dtype=np.float32)
    wd_block = np.array([[90.0, 90.0]], dtype=np.float32)
    ws_r, wd_r, u_r, v_r = td.reconcile_collocated_wind(ws_block, wd_block)
    assert np.isclose(ws_r[0], 5.0), f"Expected ws=5.0 after blockage rejection, got {ws_r[0]}"
    print("  [OK] Ultrasonic blockage rejection passed.")


def test_dewpoint_and_derivations():
    ap = np.array([101.325], dtype=np.float32)
    te = np.array([25.0], dtype=np.float32)
    rh = np.array([60.0], dtype=np.float32)
    ws = np.array([5.0], dtype=np.float32)
    wd = np.array([90.0], dtype=np.float32)
    u = np.array([-5.0], dtype=np.float32)
    v = np.array([0.0], dtype=np.float32)
    elev = np.array([100.0], dtype=np.float32)
    ts = np.array([0.0], dtype=np.float32)

    derived = td.compute_derived_9_vars(ap, te, rh, ws, wd, u, v, elev, ts)
    assert derived["pa"][0] == 101325.0
    assert derived["2t"][0] == 298.15
    assert derived["dt"][0] <= derived["2t"][0], "Dew point must be <= 2t"
    assert derived["msl"][0] >= derived["pa"][0], "MSL must be >= pa for elevation > 0"
    print("  [OK] 9-channel meteorological derivations passed.")


if __name__ == "__main__":
    print("Running TAHMO QC unit tests:")
    test_temperature_range_and_fusion()
    test_wind_vector_wrap_around()
    test_ultrasonic_blockage_rejection()
    test_dewpoint_and_derivations()
    print("All unit tests passed successfully!")

