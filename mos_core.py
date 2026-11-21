"""
mos_core.py — Shared Constants, Raw Observation Ingestion (Synoptic + Raw AZ_COOP),
and HRRR Predictor Feature Engineering for Northern AZ MOS.
"""
import datetime
import io
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
IEM_AZCOOP_GEOJSON = "https://mesonet.agron.iastate.edu/geojson/network/AZ_COOP.geojson"
IEM_AZCLIMATE_GEOJSON = "https://mesonet.agron.iastate.edu/geojson/network/AZCLIMATE.geojson"

# Non-physical regional/statewide aggregates in IEM Climodat to always exclude
SYNTHETIC_COOP_IDS = {"AZ0000"}
SYNTHETIC_COOP_PREFIXES = ("AZC",)

NETWORK_PRIORITY = {
    "AZ_ASOS":   1,
    "AZCLIMATE": 2,
    "USCRN":     3,
    "RAWS":      4,
    "SNOTEL":    5,
    "RWIS":      6,
    "ALERT":     7,
    "CWOP":      8,
}

# Every consecutive hourly HRRR timestep across each target diurnal window
CYCLE_PERIODS = {
    "00Z": [
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
        {"key": "min_2", "kind": "min", "label": "Morning Min (Day 2)",   "fhrs": [28, 29, 30, 31, 32, 33, 34, 35, 36]},
    ],
    "06Z": [
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
        {"key": "min_2", "kind": "min", "label": "Morning Min (Day 2)",   "fhrs": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]},
    ],
    "12Z": [
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Today)", "fhrs": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]},
        {"key": "max_2", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [29, 30, 31, 32, 33, 34, 35, 36]},
    ],
    "18Z": [
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Today)", "fhrs": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
        {"key": "max_2", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]},
    ],
}

CANDIDATE_PREDICTORS = [
    "hrrr_ext_t", "hrrr_win_dp2m", "hrrr_win_dpd", "hrrr_win_rh",
    "hrrr_tmp_500", "hrrr_tmp_700", "hrrr_hgt_500", "hrrr_thick_1000_500", "hrrr_lapse_500_700",
    "hrrr_lcdc", "hrrr_mcdc", "hrrr_hcdc", "hrrr_tcdc", "hrrr_dlwrf", "hrrr_dswrf",
    "hrrr_pwat", "hrrr_qpf_in", "hrrr_snow_in",
    "hrrr_u10_kt", "hrrr_v10_kt", "hrrr_wspd10_kt", "hrrr_wspd700_kt",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2", "norm_t",
]

_AZ_COOP_CATALOG_CACHE = None


def is_synthetic_coop_station(sid: str, name: str = "") -> bool:
    """Returns True for non-physical IEM Climodat aggregates (statewide mean / climate divisions)."""
    s = str(sid).strip().upper()
    n = str(name).strip().lower()
    if s in SYNTHETIC_COOP_IDS or any(s.startswith(p) for p in SYNTHETIC_COOP_PREFIXES):
        return True
    if "climate division" in n or "arizona average" in n:
        return True
    return False


def create_sync_session(pool_size: int = 24) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get_synoptic_token() -> str:
    token = os.environ.get("SYNOPTIC_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SYNOPTIC_TOKEN environment variable is missing. Set it in GitHub Actions Secrets.")
    return token


def synoptic_get_sequential(session: requests.Session, endpoint: str, params: dict, token: str, timeout: int = 60) -> dict:
    """Executes one sequential Synoptic GET request and scrubs the API token from any error text."""
    q_params = dict(params)
    q_params["token"] = token
    url = f"{SYNOPTIC_BASE}/{endpoint.lstrip('/')}"
    try:
        resp = session.get(url, params=q_params, timeout=timeout)
        time.sleep(0.35)
        if resp.status_code != 200:
            raise RuntimeError(f"Synoptic HTTP {resp.status_code} on {endpoint}")
        payload = resp.json()
        summary = payload.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") not in (1, 2):
            msg = str(summary.get("RESPONSE_MESSAGE", "Unknown Synoptic error")).replace(token, "***")
            raise RuntimeError(f"Synoptic API error on {endpoint}: {msg}")
        return payload
    except Exception as exc:
        raise RuntimeError(str(exc).replace(token, "***")) from None


def get_verify_date_mst(init_dt: datetime.datetime, fhrs: list) -> datetime.date:
    mid_fh = fhrs[len(fhrs) // 2]
    return (init_dt + datetime.timedelta(hours=mid_fh) - datetime.timedelta(hours=7)).date()


def eval_harmonic_norm(w: list, doy: int) -> float:
    return float(
        w[0]
        + w[1] * math.cos(2.0 * math.pi * doy / 365.25)
        + w[2] * math.sin(2.0 * math.pi * doy / 365.25)
    )


def despike_cwop_series(temps: pd.Series, threshold_f: float = 4.0) -> pd.Series:
    """Rejects isolated single-packet telemetry spikes (> threshold_f from 5-pt median)."""
    if len(temps) < 3:
        return temps
    med = temps.rolling(window=5, center=True, min_periods=3).median()
    is_spike = (temps - med).abs() > threshold_f
    return temps.where(~is_spike, med)


def parse_timeseries_diurnal_df(
    stn_obj: dict,
    apply_spike_filter: bool = False,
    min_obs_per_window: int = 6,
) -> pd.DataFrame:
    """
    Aggregates UTC air_temp time-series into standardized NWS MOS diurnal windows:
      - Morning Min (obs_min) for Date D: 00:00Z to 18:00Z on Date D
      - Afternoon Max (obs_max) for Date D: 12:00Z on Date D to 06:00Z on Date D+1
    """
    obs = stn_obj.get("OBSERVATIONS", {})
    dts = obs.get("date_time", [])
    t_key = "air_temp_set_1" if "air_temp_set_1" in obs else next((k for k in obs if k.startswith("air_temp")), None)
    if not dts or not t_key:
        return pd.DataFrame()
    temps = obs.get(t_key, [])
    if len(dts) != len(temps):
        return pd.DataFrame()

    df = pd.DataFrame({
        "ts": pd.to_datetime(dts, utc=True, errors="coerce"),
        "temp": pd.to_numeric(temps, errors="coerce"),
    }).dropna(subset=["ts", "temp"])

    df = df[(df["temp"] >= -55.0) & (df["temp"] <= 135.0)].sort_values("ts").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    if apply_spike_filter and len(df) >= 3:
        df["temp"] = despike_cwop_series(df["temp"], threshold_f=4.0)

    # 1. Morning Min window: 00:00Z <= ts <= 18:00Z on Date D
    in_min_win = (df["ts"].dt.hour < 18) | ((df["ts"].dt.hour == 18) & (df["ts"].dt.minute == 0))
    min_df = df[in_min_win].copy()
    min_df["date"] = min_df["ts"].dt.tz_localize(None).dt.normalize()
    min_agg = min_df.groupby("date")["temp"].agg(obs_min="min", min_cnt="count")
    min_agg = min_agg[min_agg["min_cnt"] >= min_obs_per_window][["obs_min"]]

    # 2. Afternoon Max window: 12:00Z on Date D <= ts <= 06:00Z on Date D+1
    ts_shift = df["ts"] - pd.Timedelta(hours=12)
    in_max_win = (ts_shift.dt.hour < 18) | ((ts_shift.dt.hour == 18) & (ts_shift.dt.minute == 0))
    max_df = df[in_max_win].copy()
    max_df["date"] = ts_shift[in_max_win].dt.tz_localize(None).dt.normalize()
    max_agg = max_df.groupby("date")["temp"].agg(obs_max="max", max_cnt="count")
    max_agg = max_agg[max_agg["max_cnt"] >= min_obs_per_window][["obs_max"]]

    out = min_agg.join(max_agg, how="outer").sort_index()
    if out.empty:
        return pd.DataFrame()

    both = out["obs_min"].notna() & out["obs_max"].notna()
    bad = both & ((out["obs_max"] < out["obs_min"]) | ((out["obs_max"] - out["obs_min"]) > 75.0))
    out.loc[bad, ["obs_min", "obs_max"]] = np.nan
    out = out.dropna(how="all").round(1)
    out.index.name = "date"
    return out


def fetch_az_coop_catalog(session: requests.Session = None) -> list:
    """Fetches physical NWS COOP stations from IEM's AZ_COOP network (no synthetic Climodat sites)."""
    global _AZ_COOP_CATALOG_CACHE
    if _AZ_COOP_CATALOG_CACHE is not None:
        return _AZ_COOP_CATALOG_CACHE

    sess = session or create_sync_session()
    resp = sess.get(IEM_AZCOOP_GEOJSON, timeout=30)
    resp.raise_for_status()
    geo = resp.json()

    entries = []
    for feat in geo.get("features", []):
        coop_id = str(feat.get("id", "")).strip().upper()
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]
        if not coop_id or lat is None or lon is None:
            continue
        elev_m = props.get("elevation")
        elev_ft = int(round(float(elev_m) * 3.28084)) if elev_m is not None else None
        entries.append({
            "az_coop_id": coop_id,
            "name": str(props.get("sname", coop_id)).strip(),
            "lat": float(lat),
            "lon": float(lon),
            "elev_ft": elev_ft,
            "wfo": str(props.get("wfo", "")).strip().upper(),
            "county": str(props.get("county", "")).strip().title(),
            "archive_begin": props.get("archive_begin"),
            "archive_end": props.get("archive_end"),
        })

    _AZ_COOP_CATALOG_CACHE = entries
    return entries


def resolve_physical_az_coop(meta: dict, catalog: list) -> dict:
    """
    Matches an AZ#### station entry to its physical 5-character NWSLI station in AZ_COOP
    by coordinate proximity (<= 6 km). Returns matched AZ_COOP metadata or None.
    """
    if meta.get("az_coop_id"):
        target_id = str(meta["az_coop_id"]).strip().upper()
        for entry in catalog:
            if entry["az_coop_id"] == target_id:
                return entry

    lat0, lon0 = float(meta["lat"]), float(meta["lon"])
    cos_lat = math.cos(math.radians(lat0))
    best_entry = None
    best_km = 999.0

    for entry in catalog:
        dlat_km = (entry["lat"] - lat0) * 111.132
        dlon_km = (entry["lon"] - lon0) * 111.320 * cos_lat
        dist_km = math.hypot(dlat_km, dlon_km)
        if dist_km < best_km:
            best_km = dist_km
            best_entry = entry

    if best_entry is not None and best_km <= 6.0:
        return best_entry
    return None


def fetch_raw_az_coop_daily_df(
    stn_id: str,
    meta: dict,
    start_date: str,
    end_date: str,
    session: requests.Session = None,
) -> pd.DataFrame:
    """
    Fetches strictly observed (NEVER IEM-estimated) daily High/Low temperatures from
    IEM's raw AZ_COOP network (/cgi-bin/request/daily.py). Missing observer days remain NaN.
    """
    if is_synthetic_coop_station(stn_id, meta.get("name", "")):
        return pd.DataFrame()

    sess = session or create_sync_session()
    catalog = fetch_az_coop_catalog(sess)
    coop_match = resolve_physical_az_coop(meta, catalog)
    if coop_match is None:
        return pd.DataFrame()

    az_coop_id = coop_match["az_coop_id"]
    s_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=2)
    e_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "network": "AZ_COOP",
        "stations": az_coop_id,
        "sts": s_dt.strftime("%Y-%m-%d"),
        "ets": e_dt.strftime("%Y-%m-%d"),
        "var": "max_temp_f,min_temp_f",
        "format": "csv",
        "na": "M",
    }
    resp = sess.get(IEM_DAILY_URL, params=params, timeout=30)
    resp.raise_for_status()
    if not resp.text or len(resp.text.strip()) < 30:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(resp.text), comment="#", na_values=["M", "T", " ", "", "None", "null"])
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in ["day", "date"] if c in df.columns), None)
    min_col = next((c for c in ["min_temp_f", "low", "tmin"] if c in df.columns), None)
    max_col = next((c for c in ["max_temp_f", "high", "tmax"] if c in df.columns), None)
    if not date_col or not min_col or not max_col:
        return pd.DataFrame()

    df["date"] = pd.DatetimeIndex(pd.to_datetime(df[date_col], errors="coerce")).tz_localize(None).normalize()
    df["obs_min"] = pd.to_numeric(df[min_col], errors="coerce")
    df["obs_max"] = pd.to_numeric(df[max_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # Enforce physical bounds per variable without fabricating missing values
    df.loc[(df["obs_min"] < -55.0) | (df["obs_min"] > 135.0), "obs_min"] = np.nan
    df.loc[(df["obs_max"] < -55.0) | (df["obs_max"] > 135.0), "obs_max"] = np.nan
    bad_pair = df["obs_min"].notna() & df["obs_max"].notna() & (
        (df["obs_max"] < df["obs_min"]) | ((df["obs_max"] - df["obs_min"]) > 75.0)
    )
    df.loc[bad_pair, ["obs_min", "obs_max"]] = np.nan

    df = df.dropna(subset=["obs_min", "obs_max"], how="all")
    if df.empty:
        return pd.DataFrame()

    out = (
        df[["date", "obs_min", "obs_max"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")
        .round(1)
    )
    return out


def extract_period_features(var_arrays: dict, period: dict, init_dt: datetime.datetime, meta: dict) -> dict:
    if "tmp_2m" not in var_arrays:
        return {}
    n_fhrs = len(var_arrays["tmp_2m"])
    fhrs = period["fhrs"]
    if any((fh - 1) >= n_fhrs for fh in fhrs):
        return {}

    def val_at(vkey: str, fh: int, default=np.nan) -> float:
        arr = var_arrays.get(vkey)
        idx = fh - 1
        if arr is None or idx < 0 or idx >= len(arr):
            return default
        v = float(arr[idx])
        return default if np.isnan(v) else v

    def k_to_f(k): return (k - 273.15) * 9.0 / 5.0 + 32.0 if (180.0 <= k <= 340.0) else np.nan
    def k_to_c(k): return (k - 273.15) if (180.0 <= k <= 340.0) else np.nan
    def ms_to_kt(ms): return ms * 1.94384

    temps = [k_to_f(val_at("tmp_2m", fh)) for fh in fhrs]
    if any(np.isnan(t) for t in temps):
        return {}

    hrrr_ext_t = min(temps) if period["kind"] == "min" else max(temps)

    def win_mean(vkey: str, conv=lambda x: x, default=0.0) -> float:
        vals = [conv(val_at(vkey, fh)) for fh in fhrs]
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else default

    win_t = float(np.mean(temps))
    win_dp = win_mean("dpt_2m", k_to_f, default=win_t - 15.0)
    win_rh = win_mean("rh_2m", default=45.0)
    u10 = win_mean("u_10m", ms_to_kt, default=0.0)
    v10 = win_mean("v_10m", ms_to_kt, default=0.0)
    u700 = win_mean("u_700", ms_to_kt, default=0.0)
    v700 = win_mean("v_700", ms_to_kt, default=0.0)

    tmp_500 = win_mean("tmp_500", k_to_c, default=-20.0)
    tmp_700 = win_mean("tmp_700", k_to_c, default=-5.0)
    hgt_500 = win_mean("hgt_500", default=5700.0)
    hgt_1000 = win_mean("hgt_1000", default=150.0)

    apcp_end = val_at("apcp", fhrs[-1], default=0.0)
    apcp_beg = val_at("apcp", max(1, fhrs[0] - 6), default=0.0)
    qpf_in = max(0.0, (apcp_end - apcp_beg) / 25.4)

    verify_date = get_verify_date_mst(init_dt, fhrs)
    doy = pd.Timestamp(verify_date).dayofyear

    w_clim = meta["clim_min_w"] if period["kind"] == "min" else meta["clim_max_w"]
    norm_t = eval_harmonic_norm(w_clim, doy)

    return {
        "hrrr_ext_t": hrrr_ext_t,
        "hrrr_win_temp": win_t,
        "hrrr_win_dp2m": win_dp,
        "hrrr_win_dpd": win_t - win_dp,
        "hrrr_win_rh": win_rh,
        "hrrr_u10_kt": u10,
        "hrrr_v10_kt": v10,
        "hrrr_wspd10_kt": math.hypot(u10, v10),
        "hrrr_wspd700_kt": math.hypot(u700, v700),
        "hrrr_lcdc": win_mean("lcdc", default=0.0),
        "hrrr_mcdc": win_mean("mcdc", default=0.0),
        "hrrr_hcdc": win_mean("hcdc", default=0.0),
        "hrrr_tcdc": win_mean("tcdc", default=0.0),
        "hrrr_tmp_500": tmp_500,
        "hrrr_tmp_700": tmp_700,
        "hrrr_hgt_500": hgt_500,
        "hrrr_thick_1000_500": hgt_500 - hgt_1000,
        "hrrr_lapse_500_700": tmp_500 - tmp_700,
        "hrrr_pwat": win_mean("pwat", default=8.0),
        "hrrr_qpf_in": qpf_in,
        "hrrr_snow_in": win_mean("snod", lambda m: m * 39.3701, default=0.0),
        "hrrr_dlwrf": win_mean("dlwrf", default=220.0),
        "hrrr_dswrf": win_mean("dswrf", default=0.0),
        "sin_doy_1": math.sin(2.0 * math.pi * doy / 365.25),
        "cos_doy_1": math.cos(2.0 * math.pi * doy / 365.25),
        "sin_doy_2": math.sin(4.0 * math.pi * doy / 365.25),
        "cos_doy_2": math.cos(4.0 * math.pi * doy / 365.25),
        "norm_t": norm_t,
    }
