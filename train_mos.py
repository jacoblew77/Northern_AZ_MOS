"""
train_mos.py — Plug-and-Play Multi-Station HRRRv4 MOS Trainer (Pure Model + Climatology)
- Excludes obs_prev_min / obs_prev_max so equations never rely on 1-2 day lagged COOP/ASOS reports.
- Includes --discover-coop to automatically query IEM's AZCLIMATE network metadata, verify
  every active WFO Flagstaff CWA COOP station (including Bagdad, Bellemont, Jerome, Seligman,
  Grand Canyon, etc.) across the 2020-2026 training window and current month, and add them
  to stations.json before training.
- Supports incremental training for newly added stations in stations.json, or --retrain-all.
"""
import argparse
import asyncio
import datetime
import io
import json
import logging
import math
import os
import pickle
import re
import warnings

import aiohttp
import fsspec
import numcodecs
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util import Retry
import zarr

warnings.filterwarnings("ignore")
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

START_DATE = "2020-12-03"
END_DATE = (datetime.date.today() - datetime.timedelta(days=4)).strftime("%Y-%m-%d")
SAMPLE_STEP_DAYS = 3
MAX_CONCURRENT_RUNS = 180
OUTPUT_EQ_FILE = "mos_equations.json"
STATIONS_FILE = "stations.json"
CACHE_DIR = "zarr_pixel_cache"

CYCLE_PERIODS = {
    "00Z": [
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [9, 12, 14]},
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [20, 21, 23]},
        {"key": "min_2", "kind": "min", "label": "Morning Min (Day 2)",   "fhrs": [33, 36]},
    ],
    "06Z": [
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [3, 6, 8]},
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [14, 15, 17]},
        {"key": "min_2", "kind": "min", "label": "Morning Min (Day 2)",   "fhrs": [27, 30, 32]},
    ],
    "12Z": [
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Today)", "fhrs": [8, 9, 11]},
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [21, 24, 26]},
        {"key": "max_2", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [32, 33, 35]},
    ],
    "18Z": [
        {"key": "max_1", "kind": "max", "label": "Afternoon Max (Today)", "fhrs": [2, 3, 5]},
        {"key": "min_1", "kind": "min", "label": "Morning Min (Day 1)",   "fhrs": [15, 18, 20]},
        {"key": "max_2", "kind": "max", "label": "Afternoon Max (Day 1)", "fhrs": [26, 27, 29]},
    ],
}

ZARR_VAR_CANDIDATES = {
    "tmp_2m":   ["2m_above_ground/TMP"],
    "dpt_2m":   ["2m_above_ground/DPT"],
    "rh_2m":    ["2m_above_ground/RH"],
    "u_10m":    ["10m_above_ground/UGRD"],
    "v_10m":    ["10m_above_ground/VGRD"],
    "lcdc":     ["low_cloud_layer/LCDC"],
    "mcdc":     ["middle_cloud_layer/MCDC"],
    "hcdc":     ["high_cloud_layer/HCDC"],
    "tcdc":     ["entire_atmosphere/TCDC"],
    "tmp_500":  ["500mb/TMP"],
    "tmp_700":  ["700mb/TMP"],
    "hgt_500":  ["500mb/HGT"],
    "hgt_1000": ["1000mb/HGT"],
    "u_700":    ["700mb/UGRD"],
    "v_700":    ["700mb/VGRD"],
    "pwat":     ["entire_atmosphere_considered_as_a_single_layer/PWAT", "entire_atmosphere/PWAT"],
    "apcp":     ["surface/APCP"],
    "snod":     ["surface/SNOD"],
    "dlwrf":    ["surface/DLWRF"],
    "dswrf":    ["surface/DSWRF"],
}


def get_verify_date_mst(init_dt: datetime.datetime, fhrs: list) -> datetime.date:
    """Computes the exact Arizona (MST = UTC-7) calendar date on which the window verifies."""
    mid_fh = fhrs[len(fhrs) // 2]
    valid_mst = init_dt + datetime.timedelta(hours=mid_fh) - datetime.timedelta(hours=7)
    return valid_mst.date()


def create_sync_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def format_coop_name(raw_name: str) -> str:
    """Formats uppercase IEM COOP names cleanly while preserving meteorological/park acronyms."""
    words = raw_name.strip().title().split()
    acronyms = {"Wfo", "Np", "Nm", "Rs", "Ap", "Us", "Az", "Nne", "Nnw", "Sse", "Ssw", "Ene", "Ese", "Wnw", "Wsw", "Ne", "Nw", "Se", "Sw"}
    cleaned = []
    for w in words:
        if w in acronyms:
            cleaned.append(w.upper())
        elif re.match(r"^\d+(N|S|E|W|Ne|Nw|Se|Sw|Nne|Nnw|Sse|Ssw|Ene|Ese|Wnw|Wsw)$", w):
            cleaned.append(w.upper())
        else:
            cleaned.append(w)
    name = " ".join(cleaned)
    if not any(tag in name for tag in ("COOP", "NP", "NM", "WFO", "Ranch", "Monument", "Mon")):
        name = f"{name} COOP"
    return name


def discover_active_northern_az_coop(existing_stations: dict) -> dict:
    """
    Queries IEM's official AZCLIMATE GeoJSON metadata feed, filters for all stations in
    WFO Flagstaff's CWA (wfo == 'FGZ' or Northern AZ counties), and verifies each candidate
    has >= 1,400 valid daily High/Low rows since 2020-12-03 AND active observations within
    the last 30 days.
    """
    print("Querying IEM AZCLIMATE metadata to discover all active Northern AZ COOP sites...")
    session = create_sync_session()
    geo_url = "https://mesonet.agron.iastate.edu/geojson/network/AZCLIMATE.geojson"
    resp = session.get(geo_url, timeout=30)
    resp.raise_for_status()
    geo = resp.json()

    fgz_counties = {"Coconino", "Yavapai", "Navajo", "Apache", "Gila"}
    recent_cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=30))
    updated = dict(existing_stations)
    added_count = 0

    for feat in geo.get("features", []):
        stn_id = str(feat.get("id", "")).strip().upper()
        if not stn_id.startswith("AZ") or stn_id in updated:
            continue

        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]
        if lat is None or lon is None:
            continue

        wfo = str(props.get("wfo", "")).strip().upper()
        county = str(props.get("county", "")).strip().title()
        archive_end = props.get("archive_end")

        if archive_end and str(archive_end) < "2026-01-01":
            continue

        in_fgz = (wfo == "FGZ") or (county in fgz_counties and float(lat) >= 33.75)
        if not in_fgz:
            continue

        raw_name = str(props.get("sname", stn_id))
        elev_m = props.get("elevation")
        elev_ft = int(round(float(elev_m) * 3.28084)) if elev_m is not None else 5000

        meta_cand = {
            "name": format_coop_name(raw_name),
            "network": "AZCLIMATE",
            "iem_id": stn_id,
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "elev_ft": elev_ft,
        }

        try:
            obs_df = fetch_station_daily_obs(stn_id, meta_cand, START_DATE, datetime.date.today().strftime("%Y-%m-%d"))
            if len(obs_df) < 1400:
                print(f"  [-] Skipping {stn_id} ({meta_cand['name']}): insufficient 2020-2026 days ({len(obs_df)}).")
                continue
            if obs_df.index.max() < recent_cutoff:
                print(f"  [-] Skipping {stn_id} ({meta_cand['name']}): inactive (last obs {obs_df.index.max().date()}).")
                continue

            updated[stn_id] = meta_cand
            added_count += 1
            print(f"  [+] Verified active COOP site: {stn_id} — {meta_cand['name']} ({elev_ft} ft, {len(obs_df)} days, latest {obs_df.index.max().date()})")
        except Exception as exc:
            print(f"  [-] Skipping {stn_id} ({raw_name}): error checking IEM ({exc})")

    print(f"COOP discovery complete: added {added_count} new active Northern AZ COOP station(s).")
    return updated


def fit_harmonic_climatology(obs_df: pd.DataFrame) -> tuple:
    doys = obs_df.index.dayofyear.values
    X = np.column_stack([
        np.ones(len(doys)),
        np.cos(2.0 * math.pi * doys / 365.25),
        np.sin(2.0 * math.pi * doys / 365.25),
    ])
    w_min, _, _, _ = np.linalg.lstsq(X, obs_df["obs_min"].values, rcond=None)
    w_max, _, _, _ = np.linalg.lstsq(X, obs_df["obs_max"].values, rcond=None)
    return [round(float(x), 4) for x in w_min], [round(float(x), 4) for x in w_max]


def fetch_station_daily_obs(stn_id: str, meta: dict, start_date: str, end_date: str) -> pd.DataFrame:
    s_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=5)
    e_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    session = create_sync_session()

    if meta["network"] == "AZCLIMATE":
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/coop.py?"
            f"network=AZCLIMATE&station={meta['iem_id']}&"
            f"year1={s_dt.year}&month1={s_dt.month}&day1={s_dt.day}&"
            f"year2={e_dt.year}&month2={e_dt.month}&day2={e_dt.day}&"
            f"vars=high,low&what=download&delim=comma&with_header=true"
        )
    else:
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
            f"network={meta['network']}&stations={meta['iem_id']}&"
            f"year1={s_dt.year}&month1={s_dt.month}&day1={s_dt.day}&"
            f"year2={e_dt.year}&month2={e_dt.month}&day2={e_dt.day}&"
            f"var=max_temp_f&var=min_temp_f&format=csv"
        )

    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), comment="#", na_values=["M", "T", " ", "", "None"])
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next(c for c in ["day", "date"] if c in df.columns)
    min_col = next(c for c in ["low", "min_temp_f", "tmin"] if c in df.columns)
    max_col = next(c for c in ["high", "max_temp_f", "tmax"] if c in df.columns)

    df["date"] = pd.DatetimeIndex(pd.to_datetime(df[date_col])).tz_localize(None).normalize()
    df["obs_min"] = pd.to_numeric(df[min_col], errors="coerce")
    df["obs_max"] = pd.to_numeric(df[max_col], errors="coerce")

    return (
        df[["date", "obs_min", "obs_max"]]
        .dropna(subset=["obs_min", "obs_max"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")
    )


def resolve_zarr_pixels(stations: dict):
    mapper = fsspec.get_mapper("https://hrrrzarr.s3.amazonaws.com/grid/HRRR_chunk_index.zarr")
    g_root = zarr.open_group(mapper, mode="r")
    lats = np.array(g_root["latitude"][:])
    lons = np.array(g_root["longitude"][:])
    lons_180 = np.where(lons > 180.0, lons - 360.0, lons)

    station_pixels = {}
    unique_tiles = set()
    for stn_id, meta in stations.items():
        t_lat, t_lon = meta["lat"], meta["lon"]
        dist2 = (lats - t_lat) ** 2 + ((lons_180 - t_lon) * math.cos(math.radians(t_lat))) ** 2
        iy, ix = np.unravel_index(int(np.argmin(dist2)), lats.shape)
        tile_id = f"0.{int(g_root['chunk_y'][iy])}.{int(g_root['chunk_x'][ix])}"
        ly, lx = int(g_root["in_chunk_y"][iy]), int(g_root["in_chunk_x"][ix])
        station_pixels[stn_id] = {"tile_id": tile_id, "ly": ly, "lx": lx}
        unique_tiles.add(tile_id)

    session = create_sync_session()
    meta_resp = session.get("https://hrrrzarr.s3.amazonaws.com/sfc/20230115/20230115_00z_fcst.zarr/.zmetadata", timeout=20)
    meta_resp.raise_for_status()
    metadata = meta_resp.json().get("metadata", {})

    resolved_specs = {}
    for var_key, candidates in ZARR_VAR_CANDIDATES.items():
        for cand in candidates:
            lvl, name = cand.split("/")
            for path in (f"{lvl}/{name}/{lvl}/{name}", f"{lvl}/{name}"):
                zmeta = metadata.get(f"{path}/.zarray")
                if zmeta is not None:
                    resolved_specs[var_key] = {
                        "path": path,
                        "dtype": np.dtype(zmeta["dtype"]),
                        "chunks": tuple(zmeta["chunks"]),
                        "fill_value": zmeta.get("fill_value", 0),
                    }
                    break
            if var_key in resolved_specs:
                break

    return station_pixels, sorted(unique_tiles), resolved_specs


def eval_harmonic_norm(w: list, doy: int) -> float:
    return float(w[0] + w[1] * math.cos(2.0 * math.pi * doy / 365.25) + w[2] * math.sin(2.0 * math.pi * doy / 365.25))


def extract_period_features(var_arrays: dict, period: dict, init_dt: datetime.datetime, meta: dict) -> dict:
    """Builds the predictor dictionary from HRRR atmospheric/surface fields and seasonal climatology."""
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


def forward_screening_regression(df: pd.DataFrame, target_col: str, candidate_cols: list, max_predictors=7, min_r2_gain=0.004):
    y = df[target_col].values
    n = len(y)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    selected = ["hrrr_ext_t"]
    remaining = [c for c in candidate_cols if c != "hrrr_ext_t"]
    X_init = np.column_stack([np.ones(n), df[selected].values])
    w_init, _, _, _ = np.linalg.lstsq(X_init, y, rcond=None)
    best_r2 = 1.0 - (np.sum((y - X_init @ w_init) ** 2) / ss_tot)

    for _ in range(2, max_predictors + 1):
        best_cand, trial_best = None, best_r2
        for cand in remaining:
            X_t = np.column_stack([np.ones(n), df[selected + [cand]].values])
            try:
                w_t, _, _, _ = np.linalg.lstsq(X_t, y, rcond=None)
                r2 = 1.0 - (np.sum((y - X_t @ w_t) ** 2) / ss_tot)
                if r2 > trial_best:
                    trial_best, best_cand = r2, cand
            except np.linalg.LinAlgError:
                continue
        if best_cand is not None and (trial_best - best_r2) >= min_r2_gain:
            selected.append(best_cand)
            remaining.remove(best_cand)
            best_r2 = trial_best
        else:
            break

    X_f = np.column_stack([np.ones(n), df[selected].values])
    weights, _, _, _ = np.linalg.lstsq(X_f, y, rcond=None)
    return selected, weights


async def fetch_tile_var(session, compressor, url, spec, pixels_in_tile):
    dtype = spec["dtype"]
    ny, nx = spec["chunks"][-2], spec["chunks"][-1]
    fill_val = spec["fill_value"]
    for attempt in range(3):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    raw = compressor.decode(await resp.read())
                    arr = np.frombuffer(raw, dtype=dtype).reshape(-1, ny, nx)
                    out = {}
                    for stn_id, (ly, lx) in pixels_in_tile.items():
                        ts = arr[:, ly, lx].astype(np.float32)
                        if fill_val is not None and not np.isnan(float(fill_val)):
                            ts[ts == float(fill_val)] = np.nan
                        out[stn_id] = ts
                    return out
                if resp.status == 404:
                    return None
        except Exception:
            await asyncio.sleep(0.15 * (attempt + 1))
    return None


async def download_all_stations_zarr(unique_runs, station_pixels, unique_tiles, resolved_specs):
    os.makedirs(CACHE_DIR, exist_ok=True)
    data_store = {}
    uncached_stations = {}

    for stn_id, info in station_pixels.items():
        cache_path = os.path.join(CACHE_DIR, f"{stn_id}_step{SAMPLE_STEP_DAYS}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data_store[stn_id] = pickle.load(f)
            print(f"  -> Loaded cached Zarr runs for {stn_id} ({len(data_store[stn_id])} runs).")
        else:
            data_store[stn_id] = {}
            uncached_stations[stn_id] = info

    if not uncached_stations:
        return data_store

    active_tiles = sorted({info["tile_id"] for info in uncached_stations.values()})
    tile_to_pixels = {t: {} for t in active_tiles}
    for stn_id, info in uncached_stations.items():
        tile_to_pixels[info["tile_id"]][stn_id] = (info["ly"], info["lx"])

    queue = asyncio.Queue()
    for r in unique_runs:
        queue.put_nowait(r)

    connector = aiohttp.TCPConnector(limit=350, ttl_dns_cache=600)
    timeout = aiohttp.ClientTimeout(total=15, sock_connect=5, sock_read=8)
    compressor = numcodecs.Blosc()
    pbar = tqdm(total=len(unique_runs), desc="Multi-Station HRRR-Zarr Ingestion", unit="run", smoothing=0.05)

    async def worker(session):
        while True:
            try:
                init_dt = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                d_str = init_dt.strftime("%Y%m%d")
                h_str = f"{init_dt.hour:02d}"
                base_zarr = f"https://hrrrzarr.s3.amazonaws.com/sfc/{d_str}/{d_str}_{h_str}z_fcst.zarr"
                tmp_spec = resolved_specs["tmp_2m"]

                for tile_id, pix_map in tile_to_pixels.items():
                    tmp_res = await fetch_tile_var(session, compressor, f"{base_zarr}/{tmp_spec['path']}/{tile_id}", tmp_spec, pix_map)
                    if tmp_res is None:
                        continue
                    for stn_id, ts in tmp_res.items():
                        data_store[stn_id].setdefault(init_dt, {})["tmp_2m"] = ts

                    other_keys = [k for k in resolved_specs if k != "tmp_2m"]
                    coros = [
                        fetch_tile_var(session, compressor, f"{base_zarr}/{resolved_specs[k]['path']}/{tile_id}", resolved_specs[k], pix_map)
                        for k in other_keys
                    ]
                    res_list = await asyncio.gather(*coros)
                    for k, res in zip(other_keys, res_list):
                        if res is not None:
                            for stn_id, ts in res.items():
                                data_store[stn_id].setdefault(init_dt, {})[k] = ts
            finally:
                pbar.update(1)
                queue.task_done()

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            workers = [asyncio.create_task(worker(session)) for _ in range(MAX_CONCURRENT_RUNS)]
            await asyncio.gather(*workers)
    finally:
        pbar.close()

    for stn_id in uncached_stations:
        cache_path = os.path.join(CACHE_DIR, f"{stn_id}_step{SAMPLE_STEP_DAYS}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(data_store[stn_id], f)

    return data_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-all", action="store_true", help="Retrain all stations instead of only newly added ones.")
    parser.add_argument("--discover-coop", action="store_true", help="Auto-discover all active Northern AZ COOP sites from IEM and add to stations.json.")
    args = parser.parse_args()

    with open(STATIONS_FILE, "r") as f:
        all_stations = json.load(f)

    if args.discover_coop:
        all_stations = discover_active_northern_az_coop(all_stations)
        with open(STATIONS_FILE, "w") as f:
            json.dump(all_stations, f, indent=2)

    if os.path.exists(OUTPUT_EQ_FILE) and not args.retrain_all:
        with open(OUTPUT_EQ_FILE, "r") as f:
            equations_bundle = json.load(f)
        existing_ids = set(equations_bundle.get("stations", {}).keys())
        target_stations = {k: v for k, v in all_stations.items() if k not in existing_ids}
        for old_id in list(existing_ids):
            if old_id not in all_stations:
                del equations_bundle["stations"][old_id]
        if not target_stations:
            print(f"All {len(all_stations)} stations in {STATIONS_FILE} are already trained in {OUTPUT_EQ_FILE}!")
            return
        print(f"Incremental mode: training {len(target_stations)} newly added station(s): {list(target_stations.keys())}")
    else:
        equations_bundle = {"stations": {}}
        target_stations = all_stations
        print(f"Full training mode: training all {len(target_stations)} stations...")

    obs_by_stn = {}
    valid_targets = {}
    for stn_id, meta in target_stations.items():
        print(f"Fetching COOP/ASOS observations for {stn_id} ({meta['name']})...")
        try:
            obs_df = fetch_station_daily_obs(stn_id, meta, START_DATE, END_DATE)
            if len(obs_df) < 200:
                print(f"  [!] Skipping {stn_id}: only {len(obs_df)} daily records available.")
                continue
            w_min, w_max = fit_harmonic_climatology(obs_df)
            meta_enriched = dict(meta)
            meta_enriched.setdefault("elev_ft", 5000)
            meta_enriched["clim_min_w"] = w_min
            meta_enriched["clim_max_w"] = w_max
            obs_by_stn[stn_id] = obs_df
            valid_targets[stn_id] = meta_enriched
        except Exception as exc:
            print(f"  [!] Skipping {stn_id} due to observation fetch error: {exc}")

    if not valid_targets:
        print("No valid stations to train.")
        return

    station_pixels, unique_tiles, resolved_specs = resolve_zarr_pixels(valid_targets)
    init_dates = pd.date_range(START_DATE, END_DATE, freq=f"{SAMPLE_STEP_DAYS}D")
    unique_runs = [
        datetime.datetime(d.year, d.month, d.day, hr)
        for d in init_dates
        for hr in (0, 6, 12, 18)
    ]

    data_store = asyncio.run(download_all_stations_zarr(unique_runs, station_pixels, unique_tiles, resolved_specs))

    candidate_cols = [
        "hrrr_ext_t", "hrrr_win_dp2m", "hrrr_win_dpd", "hrrr_win_rh",
        "hrrr_tmp_500", "hrrr_tmp_700", "hrrr_hgt_500", "hrrr_thick_1000_500", "hrrr_lapse_500_700",
        "hrrr_lcdc", "hrrr_mcdc", "hrrr_hcdc", "hrrr_tcdc", "hrrr_dlwrf", "hrrr_dswrf",
        "hrrr_pwat", "hrrr_qpf_in", "hrrr_snow_in",
        "hrrr_u10_kt", "hrrr_v10_kt", "hrrr_wspd10_kt", "hrrr_wspd700_kt",
        "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2", "norm_t",
    ]

    equations_bundle["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    equations_bundle["training_window"] = f"{START_DATE} to {END_DATE}"

    for stn_id, meta in valid_targets.items():
        obs_df = obs_by_stn[stn_id]
        stn_runs = data_store[stn_id]
        equations_bundle["stations"][stn_id] = {"meta": meta, "cycles": {}}

        for cyc, periods in CYCLE_PERIODS.items():
            cyc_hr = int(cyc.replace("Z", ""))
            equations_bundle["stations"][stn_id]["cycles"][cyc] = {}

            for period in periods:
                rows = []
                for init_dt, var_arrays in stn_runs.items():
                    if init_dt.hour != cyc_hr:
                        continue
                    v_date = pd.Timestamp(get_verify_date_mst(init_dt, period["fhrs"]))
                    if v_date not in obs_df.index:
                        continue

                    feats = extract_period_features(var_arrays, period, init_dt, meta)
                    if not feats:
                        continue
                    feats["target"] = float(obs_df.loc[v_date, "obs_min" if period["kind"] == "min" else "obs_max"])
                    feats["verify_date"] = v_date
                    rows.append(feats)

                df_p = pd.DataFrame(rows).sort_values("verify_date")
                split_idx = int(len(df_p) * 0.75)
                tr_df, te_df = df_p.iloc[:split_idx], df_p.iloc[split_idx:]

                sel_vars, wts = forward_screening_regression(tr_df, "target", candidate_cols)
                te_pred = np.column_stack([np.ones(len(te_df)), te_df[sel_vars].values]) @ wts
                mos_mae = float(np.mean(np.abs(te_pred - te_df["target"].values)))
                raw_mae = float(np.mean(np.abs(te_df["hrrr_ext_t"].values - te_df["target"].values)))

                equations_bundle["stations"][stn_id]["cycles"][cyc][period["key"]] = {
                    "kind": period["kind"],
                    "label": period["label"],
                    "fhrs": period["fhrs"],
                    "intercept": round(float(wts[0]), 6),
                    "predictors": sel_vars,
                    "weights": [round(float(w), 6) for w in wts[1:]],
                    "holdout_raw_mae": round(raw_mae, 2),
                    "holdout_mos_mae": round(mos_mae, 2),
                    "sample_size": len(df_p),
                }
                print(f"[{stn_id} | {cyc} | {period['key']}] Raw MAE: {raw_mae:.2f}°F -> MOS MAE: {mos_mae:.2f}°F (n={len(df_p)})")

    with open(OUTPUT_EQ_FILE, "w") as f:
        json.dump(equations_bundle, f, indent=2)
    print(f"\nSaved {len(equations_bundle['stations'])} trained stations to {OUTPUT_EQ_FILE}!")


if __name__ == "__main__":
    main()
