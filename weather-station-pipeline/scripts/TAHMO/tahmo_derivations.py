"""
tahmo_derivations.py
--------------------
Meteorological variable quality control, collocated sensor fusion,
and physical derivations for the TAHMO-to-ESFM processing pipeline.

Adopts the official SwissClim Quality Control standards:
1. Physical value range gates per sensor slot.
2. Collocated multi-sensor fusion (scalar median / vector wind).
3. Thermodynamic and kinematic derivations for the 9 ESFM target variables:
   [pa, msl, 2t, dt, 10u, 10v, wd, ws, ts].
"""

from typing import Tuple, Dict, Optional
import numpy as np


# ----------------------------------------------------------------------
# 1. Physical Range Gates (SwissClim PIPELINE_DETAILS.md standard)
# ----------------------------------------------------------------------

RANGE_BOUNDS = {
    # Variable: (min_valid, max_valid)
    "te": (-10.0, 55.0),    # Surface air temperature (Celsius)
    "ap": (65.0, 108.0),    # Atmospheric pressure (kPa)
    "rh": (1.0, 100.0),     # Relative humidity (%)
    "ws": (0.0, 50.0),      # Wind speed (m/s)
    "wd": (0.0, 360.0),     # Wind direction (degrees)
}


def apply_range_gates(slot_data: np.ndarray, var_name: str) -> np.ndarray:
    """
    Mask non-physical sensor values as NaN according to SwissClim range bounds.
    
    Args:
        slot_data: Array containing sensor readings.
        var_name: Variable identifier ('te', 'ap', 'rh', 'ws', 'wd').
        
    Returns:
        Array with out-of-range values set to NaN.
    """
    data = slot_data.copy().astype(np.float32)

    # Handle RH fraction normalization if inputs are in [0, 1]
    if var_name == "rh":
        with np.errstate(all="ignore"):
            max_val = np.nanmax(data)
            if not np.isnan(max_val) and max_val <= 1.05:
                data = data * 100.0

    if var_name in RANGE_BOUNDS:
        lo, hi = RANGE_BOUNDS[var_name]
        with np.errstate(all="ignore"):
            invalid = (data < lo) | (data > hi)
            data[invalid] = np.nan

    return data


# ----------------------------------------------------------------------
# 2. Collocated Multi-Sensor Fusion
# ----------------------------------------------------------------------

def reconcile_collocated_scalars(slot_data: np.ndarray, var_name: str) -> np.ndarray:
    """
    Reconcile multiple collocated sensor slots for continuous thermodynamic scalars (te, ap, rh).
    Applies range gates, then computes robust nanmedian across valid slots.
    """
    cleaned = apply_range_gates(slot_data, var_name)
    with np.errstate(all="ignore"):
        if cleaned.ndim > 1:
            reconciled = np.nanmedian(cleaned, axis=-1)
        else:
            reconciled = cleaned
    return reconciled


def reconcile_collocated_wind(ws_slots: np.ndarray, wd_slots: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconcile collocated wind sensors using vector decomposition and ultrasonic blockage rejection.
    Avoids scalar direction wrap-around errors across North (359° / 1°).
    """
    ws_clean = apply_range_gates(ws_slots, "ws")
    wd_clean = apply_range_gates(wd_slots, "wd")

    with np.errstate(all="ignore"):
        rad = np.radians(wd_clean)
        u_slots = -ws_clean * np.sin(rad)
        v_slots = -ws_clean * np.cos(rad)

        if ws_slots.ndim > 1:
            max_ws_across_slots = np.nanmax(ws_clean, axis=-1, keepdims=True)
            blockage_mask = (ws_clean == 0.0) & (max_ws_across_slots > 2.0)
            u_slots[blockage_mask] = np.nan
            v_slots[blockage_mask] = np.nan

            u_rec = np.nanmedian(u_slots, axis=-1)
            v_rec = np.nanmedian(v_slots, axis=-1)
        else:
            u_rec = u_slots
            v_rec = v_slots

        ws_rec = np.sqrt(u_rec**2 + v_rec**2)
        wd_rec = (np.degrees(np.arctan2(-u_rec, -v_rec))) % 360.0

        all_nan = np.isnan(u_rec) & np.isnan(v_rec)
        ws_rec[all_nan] = np.nan
        wd_rec[all_nan] = np.nan

    return ws_rec, wd_rec, u_rec, v_rec


# ----------------------------------------------------------------------
# 3. Meteorological Derivations (9 ESFM Targets)
# ----------------------------------------------------------------------

def derive_dewpoint_magnus(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """
    Calculate 2m dew point temperature in Celsius using the Magnus-Tetens formula.
    Enforces physical constraint: dewpoint <= air temperature.
    """
    a = 17.27
    b = 237.7
    rh_clipped = np.clip(rh_pct, 1e-4, 100.0)
    
    with np.errstate(all="ignore"):
        alpha = (a * temp_c) / (b + temp_c) + np.log(rh_clipped / 100.0)
        td_c = (b * alpha) / (a - alpha)
        td_c = np.minimum(td_c, temp_c)
    return td_c


def derive_vapor_pressure_hpa(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """
    Calculate actual water vapor pressure e (in hPa) using the Magnus formula.
    e_sat = 6.112 * exp(17.67 * T / (T + 243.5))
    e = (rh / 100) * e_sat
    """
    rh_clipped = np.clip(rh_pct, 0.0, 100.0)
    with np.errstate(all="ignore"):
        e_sat = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = (rh_clipped / 100.0) * e_sat
    return np.maximum(e, 0.0)


def derive_msl_pressure(
    p_surface_pa: np.ndarray,
    temp_k: np.ndarray,
    elevation_m: np.ndarray,
    rh_pct: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate Mean Sea Level (MSL) pressure in Pascals using the standard WMO
    hypsometric reduction formula (WMO-No. 8 / CIMO Guide) with vapor pressure
    and high-plateau temperature damping corrections.

    Formula:
        P_msl = P_stn * exp( (g * h) / (R_d * T_mean) )
    where:
        g = 9.80665 m/s^2 (standard gravity)
        R_d = 287.05 J/(kg*K) (gas constant for dry air)
        h = station elevation in meters
        T_mean = T_effective + (gamma * h) / 2 + e_corr
        gamma = 0.0065 K/m (standard lapse rate)
        e_corr = 0.12 K/hPa * e (vapor pressure correction term)
        T_effective: For plateau elevations (h > 500m), dampens extreme surface diurnal
                     temperature departures to prevent fictitious horizontal pressure gradients.
    """
    g = 9.80665
    r_d = 287.05
    gamma = 0.0065
    c_vp = 0.12  # K / hPa (WMO standard vapor pressure factor)

    with np.errstate(all="ignore"):
        temp_c = temp_k - 273.15

        # 1. Vapor pressure correction e_corr
        if rh_pct is not None:
            e_hpa = derive_vapor_pressure_hpa(temp_c, rh_pct)
            e_corr = c_vp * e_hpa
        else:
            e_corr = 0.0

        # 2. Plateau elevation damping for h > 500m
        # Over high terrain (e.g. East African plateau 1500m-2500m), strong diurnal heating/cooling
        # distorts sea-level reduction. Damp departures from standard 15°C (288.15K) sea-level lapse.
        h_plateau = np.maximum(0.0, elevation_m - 500.0)
        plateau_weight = np.clip(h_plateau / 1500.0, 0.0, 0.75)
        t_ref_k = 288.15 - gamma * elevation_m
        t_effective = (1.0 - plateau_weight) * temp_k + plateau_weight * t_ref_k

        # 3. Mean column virtual temperature
        t_mean = t_effective + (gamma * elevation_m) / 2.0 + e_corr
        t_mean = np.maximum(t_mean, 200.0)  # Physical safety lower bound

        # 4. WMO exponential hypsometric integration
        exponent = (g * elevation_m) / (r_d * t_mean)
        msl_pa = p_surface_pa * np.exp(exponent)

        # For stations with non-positive or negligible elevation, MSLP matches surface pressure
        at_sea_level = elevation_m <= 0.0
        msl_pa[at_sea_level] = p_surface_pa[at_sea_level]

    return msl_pa



def compute_derived_9_vars(
    ap_kpa: np.ndarray,
    te_c: np.ndarray,
    rh_pct: np.ndarray,
    ws_ms: np.ndarray,
    wd_deg: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    elevation_m: np.ndarray,
    ts_min: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Assemble the complete set of 9 ESFM target variables:
    [pa, msl, 2t, dt, 10u, 10v, wd, ws, ts].
    """
    # 1. Surface pressure pa (Pa)
    pa = ap_kpa * 1000.0

    # 2. 2m Temperature 2t (K)
    t2 = te_c + 273.15

    # 3. Mean sea level pressure msl (Pa) with WMO vapor pressure and plateau corrections
    msl = derive_msl_pressure(pa, t2, elevation_m, rh_pct=rh_pct)

    # 4. 2m Dew point temperature dt (K)
    td_c = derive_dewpoint_magnus(te_c, rh_pct)
    dt = td_c + 273.15

    with np.errstate(all="ignore"):
        dt = np.minimum(dt, t2)

    return {
        "pa": pa.astype(np.float32),
        "msl": msl.astype(np.float32),
        "2t": t2.astype(np.float32),
        "dt": dt.astype(np.float32),
        "10u": u_ms.astype(np.float32),
        "10v": v_ms.astype(np.float32),
        "wd": wd_deg.astype(np.float32),
        "ws": ws_ms.astype(np.float32),
        "ts": ts_min.astype(np.float32),
    }

