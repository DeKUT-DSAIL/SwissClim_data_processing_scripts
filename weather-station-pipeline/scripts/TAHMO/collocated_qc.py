"""
collocated_qc.py
================
Collocated-sensor quality control and reconciliation for TAHMO (ATMOS 41) data.
Produces an intermediate clean 5-minute Zarr store (tahmo_qc_fused.zarr) with the five
reconciled primitive channels: [ap, te, rh, ws, wd] and their QC flags.

Preserves station coordinates (lat, lon, elevation) directly in the output Zarr store
so downstream steps (augment_tahmo.py, build_tahmo_dataset.py) can consume the clean
data directly without re-querying raw multi-sensor slots.

Design:
  * QC and reconcile PRIMITIVE channels (ap, te, rh, ws, wd) at native 5-min cadence.
  * Collocation separates real weather (all slots see it) from sensor artefacts (one slot).
    A single-sensor value is never "corrected", only accepted / rejected.
  * Ambiguity -> NaN (ESFM replaces NaN with a learnable missing-observation token).
  * Writes block by block with per-block marker files, making execution parallel-safe
    and fully resumable across Slurm job restarts.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

try:
    import dask.array as da
    import xarray as xr
except ImportError:
    xr = None
    da = None

STEP_MIN = 5
CALM_MS = 0.3                  # wind accuracy floor: below this, direction is undefined
ESFM_VARS = ["pa", "msl", "2t", "dt", "10u", "10v", "wd", "ws", "ts"]
CORE_FOR_SNAP = ["pa", "2t", "dt", "ws"]

SPEC = {
    #        hard limits (manual)  accuracy fn (manual)           flatline steps   step test
    "ap": dict(lo=50.0, hi=110.0, acc=lambda v: np.full_like(v, 0.10), flat=36, step=0.30),
    "te": dict(lo=-50.0, hi=60.0, acc=lambda v: np.full_like(v, 0.60), flat=72, step=5.0),
    "rh": dict(lo=0.0, hi=1.0, acc=lambda v: np.full_like(v, 0.03), flat=72, step=0.40),
    "ws": dict(lo=0.0, hi=30.0, acc=lambda v: np.maximum(0.3, 0.03 * v), flat=72, step=None),
    "wd": dict(lo=0.0, hi=360.0, acc=lambda v: np.full_like(v, 5.0), flat=48, step=None),
}
K_MAD = 3.5
FLOOR_MULT = 2 ** 0.5
TOL_CAP_MULT = 6.0

F_NODATA, F_SINGLE, F_PAIR_OK, F_MAJORITY, F_MAJ_OUTLIER, F_PAIR_REF, F_CONFLICT = range(7)


# --------------------------------------------------------------------------------------
# Per-slot health checks
# --------------------------------------------------------------------------------------
def hard_limits(x, lo, hi):
    x = x.astype(np.float32, copy=True)
    x[(x < lo) | (x > hi)] = np.nan
    return x


def std_pressure_kpa(h_m):
    """ICAO standard-atmosphere station pressure [kPa] at elevation h_m [m]."""
    return 101.325 * (1.0 - 2.25577e-5 * np.asarray(h_m, float)) ** 5.25588


def flatline(x, n, exempt=None):
    """True where a slot repeats the *same* value for >= n consecutive steps.
    `exempt` marks values that may legitimately repeat (rh==1 in rain, ws==0 in calm)."""
    same = np.zeros(x.shape, bool)
    same[1:] = x[1:] == x[:-1]            # NaN == NaN is False
    if exempt is not None:
        same &= ~exempt
    run = np.ones(x.shape, np.int32)
    for t in range(1, x.shape[0]):
        run[t] = np.where(same[t], run[t - 1] + 1, 1)
    bad = run >= n
    for t in range(x.shape[0] - 2, -1, -1):   # back-propagate to the start of the run
        bad[t] |= bad[t + 1] & same[t + 1]
    return bad


def leave_one_out_median(x):
    """For each slot k: nan-median of the *other* slots. x: (T, S, K)."""
    out = np.full(x.shape, np.nan, np.float32)
    if x.shape[2] < 2:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(x.shape[2]):
            out[:, :, k] = np.nanmedian(np.delete(x, k, axis=2), axis=2)
    return out


def step_artifact(x, step_max, peer):
    """Isolated one-sample spike in one slot while its peers stay flat.
    Both the jump in and the jump out must exceed step_max with the same sign, so the good
    sample after a spike is kept. A move seen by all slots (cold pool, pressure jump) is kept.
    Without peers only an extreme (2x) isolated spike is rejected."""
    j_in = np.full(x.shape, np.nan, np.float32)
    j_out = np.full(x.shape, np.nan, np.float32)
    j_in[1:] = x[1:] - x[:-1]
    j_out[:-1] = x[:-1] - x[1:]
    p_in = np.full(x.shape, np.nan, np.float32)
    p_out = np.full(x.shape, np.nan, np.float32)
    p_in[1:] = np.abs(peer[1:] - peer[:-1])
    p_out[:-1] = np.abs(peer[:-1] - peer[1:])
    same_sign = (j_in * j_out) > 0
    has_peer = np.isfinite(p_in) & np.isfinite(p_out)
    with_peer = (np.abs(j_in) > step_max) & (np.abs(j_out) > step_max) & same_sign & has_peer \
        & (p_in < step_max / 2) & (p_out < step_max / 2)
    no_peer = (np.abs(j_in) > 2 * step_max) & (np.abs(j_out) > 2 * step_max) & same_sign & ~has_peer
    return with_peer | no_peer


# --------------------------------------------------------------------------------------
# Tolerance learning and group reconciliation
# --------------------------------------------------------------------------------------
def estimate_bias(x):
    """Persistent per-(station, slot) offset relative to the group, zero-centred over slots.
    Reference = nan-median over ALL slots (incl. self). For K=2 this gives +-d/2, so the
    offset-removed pair really agrees. Returns (S, K); zeros where undefined / K=1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmedian(x, axis=2, keepdims=True)
        b = np.nanmedian(x - g, axis=0)                     # (S, K)
        b = b - np.nanmean(b, axis=1, keepdims=True)        # relative offsets only
    return np.nan_to_num(b)


def adaptive_tol(x, peer, acc_fn):
    """Per (station, slot) offset and tolerance learned over the block.
    tol = max(K_MAD * robust_sigma(residual of debiased slot to debiased peers),
              sqrt2 * manual accuracy), capped at TOL_CAP_MULT * accuracy."""
    bias = estimate_bias(x)
    xc = x - bias[None]
    r = xc - leave_one_out_median(xc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(r, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(r - med[None]), axis=0)   # (S, K)
    level = np.where(np.isfinite(peer), peer, np.nan_to_num(x))
    floor = FLOOR_MULT * acc_fn(level)
    cap = TOL_CAP_MULT * acc_fn(level)
    tol = np.maximum(K_MAD * np.nan_to_num(sigma)[None], floor)
    return bias.astype(np.float32), np.minimum(tol, cap).astype(np.float32)


def reconcile(x, bias, tol, ref=None, ref_tol=None):
    """Collapse K collocated slots to one value per (time, station).
    x (T,S,K) health-passed values; bias (S,K); tol (T,S,K) or (S,K).
    Tests use offset-removed values; the fused value is built from the raw ones.
    ref: optional independent (T,S) tie-breaker. Returns fused (T,S) and flag (T,S)."""
    T, S, K = x.shape
    tol = np.broadcast_to(tol, x.shape)
    xc = x - bias[None]
    fin = np.isfinite(x)
    n = fin.sum(2)
    fused = np.full((T, S), np.nan, np.float32)
    flag = np.full((T, S), F_NODATA, np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m1 = n == 1
        fused[m1] = np.nanmax(np.where(fin, x, -np.inf), axis=2)[m1]
        flag[m1] = F_SINGLE

        m2 = n == 2
        spread = np.nanmax(xc, axis=2) - np.nanmin(xc, axis=2)
        tolp = np.nanmax(np.where(fin, tol, np.nan), axis=2)
        agree = m2 & (spread <= tolp)
        fused[agree] = np.nanmean(x, axis=2)[agree]
        flag[agree] = F_PAIR_OK
        conflict2 = m2 & ~agree
        if ref is not None:                       # independent tie-breaker (e.g. ht for te)
            dist = np.where(fin, np.abs(x - ref[..., None]), np.inf)
            order = np.sort(dist, axis=2)
            pick = np.argmin(dist, axis=2)
            ok = conflict2 & np.isfinite(ref) & (order[..., 0] <= ref_tol) & (order[..., 1] > ref_tol)
            chosen = np.take_along_axis(x, pick[..., None], axis=2)[..., 0]
            fused[ok] = chosen[ok]
            flag[ok] = F_PAIR_REF
            conflict2 &= ~ok
        flag[conflict2] = F_CONFLICT

        m3 = n >= 3
        med = np.nanmedian(xc, axis=2)
        mad = 1.4826 * np.nanmedian(np.abs(xc - med[..., None]), axis=2)
        inl = fin & (np.abs(xc - med[..., None]) <= np.maximum(K_MAD * mad[..., None], tol))
        n_in = inl.sum(2)
        good3 = m3 & (n_in >= 2)
        fused[good3] = np.nanmedian(np.where(inl, x, np.nan), axis=2)[good3]
        flag[good3] = np.where(n_in == n, F_MAJORITY, F_MAJ_OUTLIER)[good3]
        flag[m3 & ~good3] = F_CONFLICT
    return fused, flag


# --------------------------------------------------------------------------------------
# Wind direction (circular) reconciliation
# --------------------------------------------------------------------------------------
def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def reconcile_wd(wd, ws_slot, tol_floor=7.07, tol_cap=45.0):
    """wd (T,S,K) deg FROM; ws_slot (T,S,K) used to drop calm samples.
    Works on residuals to a per-timestep circular mean so the 359/1 deg wrap is harmless."""
    wd = np.where(ws_slot >= CALM_MS, wd, np.nan)
    z = np.exp(1j * np.deg2rad(wd))
    ref = np.rad2deg(np.angle(np.nansum(z, axis=2))) % 360.0          # (T,S)
    lin = _wrap180(wd - ref[..., None]).astype(np.float32)             # residuals
    bias = estimate_bias(lin)                                          # vane misalignment
    lc = lin - bias[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = lc - leave_one_out_median(lc)
        med = np.nanmedian(r, axis=0)
        sigma = 1.4826 * np.nan_to_num(np.nanmedian(np.abs(r - med[None]), axis=0))
    tol = np.clip(K_MAD * sigma, tol_floor, tol_cap)[None].repeat(wd.shape[0], 0)
    fused_lin, flag = reconcile(lin, bias.astype(np.float32), tol.astype(np.float32))
    return ((ref + fused_lin) % 360.0).astype(np.float32), flag


# --------------------------------------------------------------------------------------
# Block QC for the five primitives
# --------------------------------------------------------------------------------------
def qc_primitives(raw, elevation_m=None, refs=None, ref_tols=None,
                  max_tilt_deg=5.0, ap_dev_kpa=5.0):
    """raw: dict name -> (T,S,K) float arrays for ap, te, rh, ws, wd (+ optional wg,
    tilt_x, tilt_y). elevation_m: (S,) surveyed elevation (enables the pressure plausibility
    check). refs: optional independent (T,S) references, e.g. {"te": ht}, {"ap": logger_pressure}.
    Returns dict with fused values and *_flag arrays, each (T,S)."""
    refs, ref_tols = refs or {}, ref_tols or {}
    out, cleaned = {}, {}

    tilt_bad = None
    if "tilt_x" in raw and "tilt_y" in raw:
        tilt_bad = (np.abs(raw["tilt_x"]) > max_tilt_deg) | (np.abs(raw["tilt_y"]) > max_tilt_deg)

    for name in ("ap", "te", "rh", "ws"):
        s = SPEC[name]
        x0 = raw[name]
        if name == "rh":
            x0 = np.where((x0 > 1.0) & (x0 <= 1.05), 1.0, x0)
        x = hard_limits(x0, s["lo"], s["hi"])
        if name == "ap" and elevation_m is not None:
            ps = std_pressure_kpa(elevation_m)[None, :, None]
            x[np.abs(x - ps) > ap_dev_kpa] = np.nan
        if name == "ws":
            if "wg" in raw:                                            # gust invariant wg >= ws
                x[x > raw["wg"] + 0.05] = np.nan
            if tilt_bad is not None:
                x[tilt_bad] = np.nan
        exempt = {"rh": x >= 0.995, "ws": x < 0.05}.get(name)          # legit repeats
        x[flatline(x, s["flat"], exempt)] = np.nan
        peer = leave_one_out_median(x)
        if s["step"] is not None:
            x[step_artifact(x, s["step"], peer)] = np.nan
            peer = leave_one_out_median(x)
        bias, tol = adaptive_tol(x, peer, s["acc"])
        fused, flag = reconcile(x, bias, tol, refs.get(name), ref_tols.get(name))
        out[name], out[name + "_flag"] = fused, flag
        cleaned[name] = x

    wd = hard_limits(raw["wd"], 0.0, 360.0)
    if tilt_bad is not None:
        wd[tilt_bad] = np.nan
    wd[flatline(wd, SPEC["wd"]["flat"], exempt=np.nan_to_num(cleaned["ws"], nan=0.0) < 0.5)] = np.nan
    wd_f, wd_flag = reconcile_wd(wd, np.nan_to_num(cleaned["ws"], nan=0.0))
    wd_f[~(out["ws"] >= CALM_MS)] = np.nan                  # calm / unknown speed -> undefined
    out["wd"], out["wd_flag"] = wd_f, wd_flag
    return out


# --------------------------------------------------------------------------------------
# Derivation of the ESFM variables (5-min), hourly anchor snapping with ts
# --------------------------------------------------------------------------------------
def derive_esfm(f, elevation_m):
    """f: fused primitives (T,S). elevation_m: (S,) surveyed elevation of the PRESSURE SENSOR
    (not logger GPS). Returns (T,S,8) = [pa, msl, 2t, dt, 10u, 10v, wd, ws] (SI units)."""
    ap, te, rh, ws, wd = (f[k].astype(np.float64) for k in ("ap", "te", "rh", "ws", "wd"))
    pa = ap * 1000.0
    tk = te + 273.15
    msl = pa * (1.0 + 0.0065 * elevation_m[None] / tk) ** 5.255
    a, b = 17.625, 243.04                                              # Alduchov & Eskridge 1996
    g = np.log(np.clip(rh, 0.01, 1.0)) + a * te / (b + te)
    dt = b * g / (a - g) + 273.15
    calm = ws < CALM_MS
    th = np.deg2rad(wd)
    u, v = -ws * np.sin(th), -ws * np.cos(th)
    u[calm], v[calm] = 0.0, 0.0                                        # calm: zero vector, wd NaN
    arr = np.stack([pa, msl, tk, dt, u, v, wd, ws], axis=-1)
    return arr.astype(np.float32)


def rh_vp_consistency(rh, te, vp_kpa):
    """Median of (rh * es(te) - vp) in kPa."""
    es = 0.61094 * np.exp(17.625 * te / (te + 243.04))
    return float(np.nanmedian(rh * es - vp_kpa))


def _snap_anchor(times, arr8, threshold_min=15, shift_weight=1 / 5, stamp_offset_min=0.0):
    """Choose ONE anchor record per (hour, station) within +/-threshold and use it for all
    variables so that `ts` describes the whole observation."""
    t0 = times[0].astype("datetime64[m]")
    first_h = times[0].astype("datetime64[h]")
    last_h = times[-1].astype("datetime64[h]")
    hours = np.arange(first_h, last_h + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    offs = np.arange(-threshold_min, threshold_min + 1, STEP_MIN)
    base = ((hours.astype("datetime64[m]") - t0) / np.timedelta64(1, "m")).astype(int) // STEP_MIN
    idx = base[:, None] + (offs // STEP_MIN)[None]                      # (H, C)
    inside = (idx >= 0) & (idx < arr8.shape[0])
    g = arr8[np.clip(idx, 0, arr8.shape[0] - 1)]                        # (H, C, S, 8)
    g[~inside] = np.nan
    core = [ESFM_VARS.index(c) for c in CORE_FOR_SNAP]
    miss = (~np.isfinite(g[..., core])).sum(-1)                         # (H, C, S)
    cost = np.abs(offs)[None, :, None] * shift_weight + miss
    cost = np.where(miss < len(core), cost, np.inf)                    # need >=1 core variable
    best = np.argmin(cost, axis=1)                                      # (H, S)
    ok = np.isfinite(np.take_along_axis(cost, best[:, None, :], 1)[:, 0])
    pick = np.take_along_axis(g, best[:, None, :, None], 1)[:, 0]       # (H, S, 8)
    pick[~ok] = np.nan
    ts = np.where(ok, offs[best] + stamp_offset_min, np.nan).astype(np.float32)
    return hours, np.concatenate([pick, ts[..., None]], axis=-1).astype(np.float32)


# Observation groups for per-variable snapping
GROUPS = {"pa": [0], "msl": [1], "2t": [2], "dt": [3], "wind": [4, 5, 6, 7]}
GROUP_KEY = {"pa": 0, "msl": 1, "2t": 2, "dt": 3, "wind": 7}


def _snap_per_variable(times, arr8, threshold_min=15, stamp_offset_min=0.0):
    """Paper-parity rule (Ozdemir et al. 2026, App. C.3): within +/-threshold, keep for each
    (hour, station, variable group) the record with the smallest |offset| that has a valid value."""
    t0 = times[0].astype("datetime64[m]")
    hours = np.arange(times[0].astype("datetime64[h]"),
                      times[-1].astype("datetime64[h]") + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    offs = np.arange(-threshold_min, threshold_min + 1, STEP_MIN)
    base = ((hours.astype("datetime64[m]") - t0) / np.timedelta64(1, "m")).astype(int) // STEP_MIN
    idx = base[:, None] + (offs // STEP_MIN)[None]
    inside = (idx >= 0) & (idx < arr8.shape[0])
    g = arr8[np.clip(idx, 0, arr8.shape[0] - 1)]                        # (H, C, S, 8)
    g[~inside] = np.nan
    H, _, S, _ = g.shape
    out = np.full((H, S, 8), np.nan, np.float32)
    off_g = np.full((H, S, len(GROUPS)), np.nan, np.float32)
    for gi, (name, cols) in enumerate(GROUPS.items()):
        valid = np.isfinite(g[..., GROUP_KEY[name]])                    # (H, C, S)
        cost = np.where(valid, np.abs(offs)[None, :, None].astype(float), np.inf)
        best = np.argmin(cost, axis=1)                                  # (H, S)
        ok = np.isfinite(np.take_along_axis(cost, best[:, None, :], 1)[:, 0])
        sel = np.take_along_axis(g[..., cols], best[:, None, :, None], 1)[:, 0]
        out[..., cols] = np.where(ok[..., None], sel, np.nan)
        off_g[..., gi] = np.where(ok, offs[best], np.nan)
    absoff = np.where(np.isfinite(off_g), np.abs(off_g), np.inf)
    ts = np.take_along_axis(off_g, np.argmin(absoff, axis=2)[..., None], 2)[..., 0] + stamp_offset_min
    return hours, np.concatenate([out, ts[..., None].astype(np.float32)], axis=-1)


def snap_hourly(times, arr8, threshold_min=15, mode="anchor", shift_weight=1 / 5, stamp_offset_min=0.0):
    if mode == "anchor":
        return _snap_anchor(times, arr8, threshold_min, shift_weight, stamp_offset_min)
    if mode == "per_variable":
        return _snap_per_variable(times, arr8, threshold_min, stamp_offset_min)
    raise ValueError(mode)


def hourly_coverage(h):
    return np.isfinite(h).mean(axis=0)


def pair_stats(h, lag=6):
    a, b = np.isfinite(h[:-lag]), np.isfinite(h[lag:])
    return (a & b).mean(axis=(0, 1))


def audit_hourly(h):
    lim = {"pa": (50e3, 110e3), "msl": (87e3, 108.5e3), "2t": (223.15, 333.15), "dt": (200.0, 320.0),
           "10u": (-30, 30), "10v": (-30, 30), "wd": (0, 360), "ws": (0, 30), "ts": (-15, 15)}
    rep = {}
    for i, v in enumerate(ESFM_VARS):
        x = h[..., i]
        rep[v] = int(np.nansum((x < lim[v][0]) | (x > lim[v][1])))
    both = np.isfinite(h[..., 3]) & np.isfinite(h[..., 2])
    rep["dt>2t"] = int(np.sum(both & (h[..., 3] > h[..., 2] + 1e-3)))
    return rep


# --------------------------------------------------------------------------------------
# Resumable, block-wise zarr writing
# --------------------------------------------------------------------------------------
def _done_dir(path):
    return Path(str(path) + ".done")


def block_done(path, t0):
    return (_done_dir(path) / str(int(t0))).exists()


def _mark_done(path, t0):
    d = _done_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    (d / str(int(t0))).touch()


def init_store(path, times, stations, float_vars, flag_vars, chunk_t,
               lats=None, lons=None, elevations=None):
    """Create an empty store with fixed time chunks and preserved coordinates."""
    nt, ns = len(times), len(stations)
    data = {v: (("time", "station"), da.full((nt, ns), np.nan, np.float32, chunks=(chunk_t, ns)))
            for v in float_vars}
    data.update({v: (("time", "station"), da.zeros((nt, ns), np.uint8, chunks=(chunk_t, ns)))
                 for v in flag_vars})

    coords = {"time": times, "station": stations}
    if lats is not None:
        coords["lat"] = ("station", np.asarray(lats, dtype=np.float32))
    if lons is not None:
        coords["lon"] = ("station", np.asarray(lons, dtype=np.float32))
    if elevations is not None:
        coords["elevation"] = ("station", np.asarray(elevations, dtype=np.float32))
        coords["elevationmsl"] = ("station", np.asarray(elevations, dtype=np.float32))

    enc = {v: {"dtype": "float32", "_FillValue": np.nan} for v in float_vars}
    enc.update({v: {"dtype": "uint8"} for v in flag_vars})
    if lats is not None:
        enc["lat"] = {"dtype": "float32", "_FillValue": np.nan}
    if lons is not None:
        enc["lon"] = {"dtype": "float32", "_FillValue": np.nan}
    if elevations is not None:
        enc["elevation"] = {"dtype": "float32", "_FillValue": np.nan}
        enc["elevationmsl"] = {"dtype": "float32", "_FillValue": np.nan}
    ds = xr.Dataset(data, coords=coords)
    ds.to_zarr(path, mode="w", compute=False, consolidated=False, encoding=enc)
    _done_dir(path).mkdir(parents=True, exist_ok=True)


def write_block(path, arrays, t0):
    """arrays: dict name -> (nt, S) core rows only (no halo). t0 = global row of first core row."""
    nt = next(iter(arrays.values())).shape[0]
    ds = xr.Dataset({k: (("time", "station"), v) for k, v in arrays.items()})
    ds.to_zarr(path, region={"time": slice(t0, t0 + nt)}, consolidated=False)
    _mark_done(path, t0)


def load_primitives(ds, names, station_dim="station", time_dim="time"):
    out = {}
    for n in names:
        a = ds[n]
        slot = [d for d in a.dims if d not in (station_dim, time_dim)]
        a = a.transpose(time_dim, station_dim, *slot)
        out[n] = a.values.astype(np.float32).reshape(a.shape[0], a.shape[1], -1)
    return out


def run_pass1(src, out_path, elevation_m, block_steps, chunk_t, halo=72,
              names=("ap", "te", "rh", "ws", "wd"), optional=("wg", "tilt_x", "tilt_y"),
              refs_map=None, dry_run=False):
    """Pass 1: QC + reconcile at 5-min. Resumable via block markers."""
    assert block_steps % chunk_t == 0
    elevation_m = np.asarray(elevation_m, float)
    times, stations = src["time"].values, src["station"].values
    assert len(elevation_m) == len(stations), "elevation_m must follow src['station'] order"
    nt = len(times)
    fvars = ["ap", "te", "rh", "ws", "wd"]
    out_path = str(out_path)

    lats = src["lat"].values if "lat" in src else None
    lons = src["lon"].values if "lon" in src else None
    elevations = elevation_m

    if not Path(out_path).exists():
        print(f"Initializing intermediate Zarr store at: {out_path}")
        init_store(out_path, times, stations, fvars, [v + "_flag" for v in fvars], chunk_t,
                   lats=lats, lons=lons, elevations=elevations)
        print("Initialized empty Zarr structure with coordinates.")

    blocks = list(range(0, nt, block_steps))
    total_blocks = len(blocks)
    print(f"Total time steps: {nt:,} ({total_blocks} blocks of {block_steps} steps)")

    for b_idx, t0 in enumerate(blocks):
        if block_done(out_path, t0):
            print(f"Block {b_idx + 1}/{total_blocks} (t0={t0}) already done. Skipping.")
            continue

        print(f"Processing block {b_idx + 1}/{total_blocks} (time index {t0:,} -> {min(nt, t0 + block_steps):,})...")
        t_start_blk = time.time()
        lo, hi = max(0, t0 - halo), min(nt, t0 + block_steps + halo)
        blk = src.isel(time=slice(lo, hi))
        raw = load_primitives(blk, list(names) + [o for o in optional if o in blk])
        refs = {k: blk[v].values for k, v in (refs_map or {}).items()}
        res = qc_primitives(raw, elevation_m=elevation_m, refs=refs,
                            ref_tols={"te": 3.0, "ap": 0.5})
        a, b = t0 - lo, t0 - lo + min(block_steps, nt - t0)
        write_block(out_path, {k: v[a:b] for k, v in res.items()}, t0)
        elapsed = time.time() - t_start_blk
        print(f"  Block {b_idx + 1} completed and saved in {elapsed:.1f}s.")

        if dry_run:
            print("Dry-run requested: stopped after first processed block.")
            break


def main():
    parser = argparse.ArgumentParser(description="TAHMO Collocated Sensor QC and Reconciliation")
    parser.add_argument("--input-zarr", "--input_zarr", "--zarr_path", type=str,
                        default="/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_lazy.zarr",
                        help="Path to raw tahmo_lazy.zarr")
    parser.add_argument("--output-zarr", "--output_zarr", type=str,
                        default="/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_qc_fused.zarr",
                        help="Path to intermediate output tahmo_qc_fused.zarr")
    parser.add_argument("--config", type=str, default="config/tahmo_config.json",
                        help="Path to pipeline configuration JSON")
    parser.add_argument("--block-steps", "--block_steps", type=int, default=8640,
                        help="Block size in 5-minute timesteps (default 8640 = 30 days)")
    parser.add_argument("--chunk-t", "--chunk_t", type=int, default=288,
                        help="Chunk size in time dimension (default 288 = 1 day)")
    parser.add_argument("--dry-run", "--dry_run", action="store_true",
                        help="Process only a single block for verification")
    args = parser.parse_args()
    REPO_DIR = Path(__file__).resolve().parent.parent.parent

    # Resolve config file
    cfg_file = args.config
    if not os.path.exists(cfg_file) and (REPO_DIR / cfg_file).exists():
        cfg_file = str(REPO_DIR / cfg_file)

    # Load from config if available
    input_zarr = args.input_zarr
    output_zarr = args.output_zarr
    block_steps = args.block_steps
    chunk_t = args.chunk_t

    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
            qc_cfg = cfg.get("collocated_qc", {})
            if "input_zarr" in qc_cfg:
                input_zarr = qc_cfg["input_zarr"]
            elif "zarr_path" in qc_cfg:
                input_zarr = qc_cfg["zarr_path"]
            if "output_zarr" in qc_cfg:
                output_zarr = qc_cfg["output_zarr"]
            if "block_steps" in qc_cfg:
                block_steps = int(qc_cfg["block_steps"])
            if "chunk_t" in qc_cfg:
                chunk_t = int(qc_cfg["chunk_t"])
        except Exception as e:
            print(f"Warning: could not parse config {cfg_file}: {e}")

    # Resolve candidate raw zarr paths
    if not os.path.exists(input_zarr):
        candidates = [
            input_zarr,
            "/ritom/scratch/cscs/tligawa/ai-4-weather/data/tahmo_lazy.zarr",
            f"{os.environ.get('SCRATCH', '/ritom/scratch/cscs/tligawa')}/ai-4-weather/data/tahmo_lazy.zarr",
            "/users/tligawa/ritom_scratch/ai-4-weather/data/tahmo_lazy.zarr",
        ]
        for c in candidates:
            if c and os.path.exists(c):
                input_zarr = c
                break

    print("=" * 80)
    print("TAHMO COLLOCATED MULTI-SENSOR QC & RECONCILIATION")
    print("=" * 80)
    print(f"Input Raw Zarr:        {input_zarr}")
    print(f"Output Fused Zarr:     {output_zarr}")
    print(f"Block steps:           {block_steps:,} (chunk: {chunk_t:,})")
    print(f"Dry run:               {args.dry_run}")
    print("=" * 80)

    if xr is None:
        raise ImportError("xarray and dask are required for collocated QC processing.")

    if not os.path.exists(input_zarr):
        raise FileNotFoundError(f"Raw Zarr store not found at {input_zarr}")

    print(f"Opening source Zarr dataset: {input_zarr} ...")
    src = xr.open_zarr(input_zarr, consolidated=False)
    print(f"  Stations: {len(src['station']):,}")
    print(f"  Time intervals: {len(src['time']):,}")

    # Resolve elevation
    if "elevationmsl" in src:
        elevation_m = src["elevationmsl"].values
    elif "elevation" in src:
        elevation_m = src["elevation"].values
    else:
        elevation_m = np.zeros(len(src["station"]), dtype=float)

    run_pass1(
        src=src,
        out_path=output_zarr,
        elevation_m=elevation_m,
        block_steps=block_steps,
        chunk_t=chunk_t,
        dry_run=args.dry_run,
    )
    print("\n✅ Collocated QC pass 1 completed successfully.")


if __name__ == "__main__":
    main()

