"""
train_mos.py — Multi-Station HRRRv4 MOS Trainer (Synoptic Data API + Raw IEM AZ_COOP)
- Trains on EVERY available HRRRv4 day (SAMPLE_STEP_DAYS = 1) across 00Z, 06Z, 12Z, 18Z
  while preserving the chronological 75% train / 25% holdout validation split.
- Uses raw observed NWS COOP daily reports (network=AZ_COOP) rather than estimated
  Climodat (AZCLIMATE) series, and removes non-physical aggregates (AZ0000, AZC002).
"""
import argparse
import asyncio
import datetime
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
from tqdm.auto import tqdm
import zarr

from mos_core import (
    CANDIDATE_PREDICTORS,
    CYCLE_PERIODS,
    IEM_AZCLIMATE_GEOJSON,
    NETWORK_PRIORITY,
    create_sync_session,
    extract_period_features,
    fetch_az_coop_catalog,
    fetch_raw_az_coop_daily_df,
    get_synoptic_token,
    get_verify_date_mst,
    is_synthetic_coop_station,
    parse_timeseries_diurnal_df,
    resolve_physical_az_coop,
    synoptic_get_sequential,
)

warnings.filterwarnings("ignore")
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

START_DATE = "2020-12-03"
END_DATE = (datetime.date.today() - datetime.timedelta(days=4)).strftime("%Y-%m-%d")
SAMPLE_STEP_DAYS = 1  # Daily sampling across full HRRRv4 archive (~3x more data)
MAX_CONCURRENT_RUNS = 200
OUTPUT_EQ_FILE = "mos_equations.json"
STATIONS_FILE = "stations.json"
CACHE_DIR = "zarr_pixel_cache"
OBS_CACHE_DIR = "obs_cache"

DEFAULT_MIN_POR_START = "2021-03-01"
DEFAULT_MIN_VALID_DAYS = 1400
PINNED_STATIONS = set()

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


def get_obs_cache_path(sid: str) -> str:
    safe_sid = "".join(c for c in sid if c.isalnum() or c in ("_", "-"))
    return os.path.join(OBS_CACHE_DIR, f"{safe_sid}_raw_diurnal_v2.pkl")


def load_cached_obs(sid: str, target_end_date: str, refresh: bool = False):
    if refresh:
        return None
    path = get_obs_cache_path(sid)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        if isinstance(df, pd.DataFrame) and len(df) >= 200 and isinstance(df.index, pd.DatetimeIndex):
            if df.index.max() >= (pd.Timestamp(target_end_date) - pd.Timedelta(days=14)):
                return df
    except Exception:
        return None
    return None


def save_cached_obs(sid: str, df: pd.DataFrame):
    if df is None or len(df) < 200 or not isinstance(df.index, pd.DatetimeIndex):
        return
    os.makedirs(OBS_CACHE_DIR, exist_ok=True)
    with open(get_obs_cache_path(sid), "wb") as f:
        pickle.dump(df, f)


def format_station_name(raw_name: str, network: str) -> str:
    words = raw_name.strip().title().split()
    acronyms = {
        "Wfo", "Np", "Nm", "Rs", "Ap", "Us", "Az", "Uscrn", "Raws", "Snotel", "Rwis", "Adot", "Alert", "Cwop",
        "Nne", "Nnw", "Sse", "Ssw", "Ene", "Ese", "Wnw", "Wsw", "Ne", "Nw", "Se", "Sw", "Sr", "I",
    }
    cleaned = []
    for w in words:
        if w in acronyms:
            cleaned.append(w.upper())
        elif re.match(r"^\d+(N|S|E|W|Ne|Nw|Se|Sw|Nne|Nnw|Sse|Ssw|Ene|Ese|Wnw|Wsw)$", w):
            cleaned.append(w.upper())
        else:
            cleaned.append(w)
    name = " ".join(cleaned)
    tag_map = {
        "AZCLIMATE": ("COOP", "NP", "NM", "WFO", "Ranch", "Monument", "Mon"),
        "USCRN": ("USCRN",),
        "RAWS": ("RAWS",),
        "SNOTEL": ("SNOTEL",),
        "RWIS": ("RWIS", "ADOT"),
        "ALERT": ("ALERT", "FCD"),
        "CWOP": ("CWOP",),
    }
    if network in tag_map and not any(t.lower() in name.lower() for t in tag_map[network]):
        name = f"{name} {tag_map[network][0]}"
    return name


def sanitize_stations_dict(raw_stations: dict, session: requests.Session) -> dict:
    """
    Removes non-physical COOP aggregates (AZ0000, AZC002), links physical AZ#### COOP
    sites to their real 5-character AZ_COOP station ID, and fixes double-converted elevations.
    """
    coop_catalog = fetch_az_coop_catalog(session)
    cleaned = {}
    for sid, meta in raw_stations.items():
        if is_synthetic_coop_station(sid, meta.get("name", "")):
            print(f"  [-] Dropping non-physical aggregate station: {sid} ({meta.get('name', '')})")
            continue

        m = dict(meta)
        if m.get("network") == "AZCLIMATE":
            match = resolve_physical_az_coop(m, coop_catalog)
            if match is None:
                print(f"  [-] Dropping unmapped/non-physical COOP station: {sid} ({m.get('name', '')})")
                continue
            m["source"] = "IEM_COOP"
            m["iem_id"] = sid
            m["az_coop_id"] = match["az_coop_id"]
            if match.get("elev_ft") is not None:
                m["elev_ft"] = match["elev_ft"]
            elif m.get("elev_ft", 5000) > 13000:
                m["elev_ft"] = int(round(m["elev_ft"] / 3.28084))
        else:
            m["source"] = "SYNOPTIC"
            m["synoptic_id"] = sid

        cleaned[sid] = m
    return cleaned


def fetch_synoptic_timeseries_daily(
    session: requests.Session,
    stid: str,
    network: str,
    start_date: str,
    end_date: str,
    token: str,
) -> pd.DataFrame:
    s_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=1)
    e_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)
    payload = synoptic_get_sequential(
        session,
        "stations/timeseries",
        {
            "stid": stid,
            "vars": "air_temp",
            "start": s_dt.strftime("%Y%m%d") + "0000",
            "end": e_dt.strftime("%Y%m%d") + "0600",
            "obtimezone": "UTC",
            "units": "english",
            "hfmetars": "1",
            "qc_checks": "synopticlabs",
            "restricted": "false",
        },
        token=token,
        timeout=90,
    )
    stn_list = payload.get("STATION", [])
    if not stn_list or stn_list[0].get("RESTRICTED"):
        return pd.DataFrame()
    return parse_timeseries_diurnal_df(stn_list[0], apply_spike_filter=(network == "CWOP"), min_obs_per_window=6)


def discover_all_public_northern_az_stations(
    existing_stations: dict,
    token: str,
    min_por_start: str = DEFAULT_MIN_POR_START,
    min_valid_days: int = DEFAULT_MIN_VALID_DAYS,
    refresh_obs: bool = False,
) -> tuple:
    session = create_sync_session()
    fgz_counties = {"Coconino", "Yavapai", "Navajo", "Apache", "Gila"}
    recent_cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=21))
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    candidates = sanitize_stations_dict(existing_stations, session)
    coop_catalog = fetch_az_coop_catalog(session)

    print("Querying Synoptic public network registry...")
    net_payload = synoptic_get_sequential(session, "networks", {"restricted": "false"}, token=token, timeout=30)
    mnet_to_tier = {}
    for net in net_payload.get("MNET", []):
        if net.get("RESTRICTED"):
            continue
        nid = str(net.get("ID", "")).strip()
        short = str(net.get("SHORTNAME", "")).upper()
        long_n = str(net.get("LONGNAME", "")).upper()
        combined = f"{short} {long_n}"
        if nid == "1" or "ASOS" in short or "AWOS" in short:
            mnet_to_tier[nid] = "AZ_ASOS"
        elif "USCRN" in combined:
            mnet_to_tier[nid] = "USCRN"
        elif nid == "2" or "RAWS" in short:
            mnet_to_tier[nid] = "RAWS"
        elif "SNOTEL" in combined:
            mnet_to_tier[nid] = "SNOTEL"
        elif "RWIS" in combined or "ADOT" in combined or "DOT" in short:
            mnet_to_tier[nid] = "RWIS"
        elif "ALERT" in combined or "FLOOD CONTROL" in combined or "FCD" in short:
            mnet_to_tier[nid] = "ALERT"
        elif nid == "65" or "APRSWXNET" in combined or "CWOP" in combined:
            mnet_to_tier[nid] = "CWOP"

    meta_payload = synoptic_get_sequential(
        session,
        "stations/metadata",
        {
            "cwa": "FGZ",
            "status": "active",
            "restricted": "false",
            "complete": "1",
            "vars": "air_temp",
            "network": ",".join(sorted(mnet_to_tier.keys())),
        },
        token=token,
        timeout=45,
    )

    for stn in meta_payload.get("STATION", []):
        if stn.get("RESTRICTED"):
            continue
        sid = str(stn.get("STID", "")).strip().upper()
        if not sid or sid in candidates:
            continue
        tier = mnet_to_tier.get(str(stn.get("MNET_ID", "")).strip())
        if not tier:
            continue
        try:
            lat = float(stn.get("LATITUDE"))
            lon = float(stn.get("LONGITUDE"))
        except (TypeError, ValueError):
            continue

        period = stn.get("PERIOD_OF_RECORD", {})
        por_start = str(period.get("start", ""))
        por_end = str(period.get("end", ""))
        if sid not in PINNED_STATIONS and (not por_start or por_start[:10] > min_por_start):
            continue
        if not por_end or por_end[:10] < (datetime.date.today() - datetime.timedelta(days=21)).isoformat():
            continue

        elev_raw = stn.get("ELEVATION")
        elev_ft = int(round(float(elev_raw))) if elev_raw is not None else 5000
        candidates[sid] = {
            "name": format_station_name(str(stn.get("NAME", sid)), tier),
            "network": tier,
            "source": "SYNOPTIC",
            "synoptic_id": sid,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "elev_ft": elev_ft,
        }

    print("Querying IEM metadata for active physical Northern AZ COOP sites...")
    coop_resp = session.get(IEM_AZCLIMATE_GEOJSON, timeout=30)
    coop_resp.raise_for_status()
    for feat in coop_resp.json().get("features", []):
        sid = str(feat.get("id", "")).strip().upper()
        props = feat.get("properties", {})
        raw_name = str(props.get("sname", sid))
        if is_synthetic_coop_station(sid, raw_name) or not re.match(r"^AZ\d{4}$", sid) or sid in candidates:
            continue
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        lon, lat = coords[0], coords[1]
        if lat is None or lon is None:
            continue
        wfo = str(props.get("wfo", "")).strip().upper()
        county = str(props.get("county", "")).strip().title()
        archive_end = props.get("archive_end")
        if archive_end and str(archive_end) < (datetime.date.today() - datetime.timedelta(days=30)).isoformat():
            continue
        if not ((wfo == "FGZ") or (county in fgz_counties and float(lat) >= 33.75)):
            continue

        cand_meta = {
            "name": format_station_name(raw_name, "AZCLIMATE"),
            "network": "AZCLIMATE",
            "source": "IEM_COOP",
            "iem_id": sid,
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "elev_ft": 5000,
        }
        match = resolve_physical_az_coop(cand_meta, coop_catalog)
        if match is None:
            continue
        cand_meta["az_coop_id"] = match["az_coop_id"]
        if match.get("elev_ft") is not None:
            cand_meta["elev_ft"] = match["elev_ft"]
        candidates[sid] = cand_meta

    new_syn_ids = [sid for sid, m in candidates.items() if sid not in existing_stations and m.get("source") == "SYNOPTIC"]
    new_coop_ids = [sid for sid, m in candidates.items() if sid not in existing_stations and m.get("source") == "IEM_COOP"]

    verified = {sid: candidates[sid] for sid in existing_stations if sid in candidates}
    prefetched_obs = {}

    for i, sid in enumerate(new_syn_ids, 1):
        try:
            tier = candidates[sid]["network"]
            df = load_cached_obs(sid, END_DATE, refresh=refresh_obs)
            if df is None:
                df = fetch_synoptic_timeseries_daily(session, sid, tier, START_DATE, today_str, token)
                save_cached_obs(sid, df)
            req_days = 200 if sid in PINNED_STATIONS else min_valid_days
            if df is None or df.empty or len(df) < req_days or df.index.max() < recent_cutoff:
                continue
            verified[sid] = candidates[sid]
            prefetched_obs[sid] = df
            print(f"  [{i}/{len(new_syn_ids)}] Verified {tier}: {sid} — {candidates[sid]['name']} ({len(df)} days)")
        except Exception as exc:
            print(f"  [{i}/{len(new_syn_ids)}] Skipping {sid}: {exc}")

    for sid in new_coop_ids:
        try:
            df = load_cached_obs(sid, END_DATE, refresh=refresh_obs)
            if df is None:
                df = fetch_raw_az_coop_daily_df(sid, candidates[sid], START_DATE, today_str, session=session)
                save_cached_obs(sid, df)
            req_days = 200 if sid in PINNED_STATIONS else min_valid_days
            if df is not None and not df.empty and len(df) >= req_days and df.index.max() >= recent_cutoff:
                verified[sid] = candidates[sid]
                prefetched_obs[sid] = df
                print(f"  [+] Verified Raw COOP ({candidates[sid]['az_coop_id']}): {sid} — {candidates[sid]['name']} ({len(df)} days)")
        except Exception:
            continue

    ordered = dict(sorted(verified.items(), key=lambda kv: (NETWORK_PRIORITY.get(kv[1].get("network", ""), 99), kv[0])))
    print(f"Discovery complete: {len(ordered)} verified physical stations.")
    return ordered, prefetched_obs


def fit_harmonic_climatology(obs_df: pd.DataFrame) -> tuple:
    min_sub = obs_df.dropna(subset=["obs_min"])
    doys_min = min_sub.index.dayofyear.values
    X_min = np.column_stack([
        np.ones(len(doys_min)),
        np.cos(2.0 * math.pi * doys_min / 365.25),
        np.sin(2.0 * math.pi * doys_min / 365.25),
    ])
    w_min, _, _, _ = np.linalg.lstsq(X_min, min_sub["obs_min"].values, rcond=None)

    max_sub = obs_df.dropna(subset=["obs_max"])
    doys_max = max_sub.index.dayofyear.values
    X_max = np.column_stack([
        np.ones(len(doys_max)),
        np.cos(2.0 * math.pi * doys_max / 365.25),
        np.sin(2.0 * math.pi * doys_max / 365.25),
    ])
    w_max, _, _, _ = np.linalg.lstsq(X_max, max_sub["obs_max"].values, rcond=None)
    return [round(float(x), 4) for x in w_min], [round(float(x), 4) for x in w_max]


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

    connector = aiohttp.TCPConnector(limit=400, ttl_dns_cache=600)
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
    parser.add_argument("--retrain-all", action="store_true", help="Retrain all stations from scratch.")
    parser.add_argument("--discover-networks", action="store_true", help="Discover active public Northern AZ stations.")
    parser.add_argument("--refresh-obs", action="store_true", help="Force fresh download of historical observations.")
    parser.add_argument("--min-por-start", default=DEFAULT_MIN_POR_START, help="Earliest record start date cutoff (YYYY-MM-DD).")
    parser.add_argument("--min-valid-days", type=int, default=DEFAULT_MIN_VALID_DAYS, help="Minimum valid daily observation days required.")
    parser.add_argument("--prune-stations", action="store_true", help="Remove stations with fewer than --min-valid-days.")
    args = parser.parse_args()

    token = get_synoptic_token()
    os.makedirs(OBS_CACHE_DIR, exist_ok=True)
    session = create_sync_session()

    with open(STATIONS_FILE, "r") as f:
        raw_stations = json.load(f)

    all_stations = sanitize_stations_dict(raw_stations, session)
    with open(STATIONS_FILE, "w") as f:
        json.dump(all_stations, f, indent=2)

    prefetched_obs = {}
    if args.discover_networks:
        all_stations, prefetched_obs = discover_all_public_northern_az_stations(
            all_stations,
            token,
            min_por_start=args.min_por_start,
            min_valid_days=args.min_valid_days,
            refresh_obs=args.refresh_obs,
        )
        with open(STATIONS_FILE, "w") as f:
            json.dump(all_stations, f, indent=2)

    if args.prune_stations:
        kept_stations = {}
        pruned_ids = []
        for sid, meta in all_stations.items():
            if sid in PINNED_STATIONS:
                kept_stations[sid] = meta
                continue
            obs_df = prefetched_obs.get(sid) or load_cached_obs(sid, END_DATE, refresh=args.refresh_obs)
            if obs_df is None:
                try:
                    if meta.get("source") == "SYNOPTIC":
                        obs_df = fetch_synoptic_timeseries_daily(session, sid, meta.get("network", ""), START_DATE, END_DATE, token)
                    else:
                        obs_df = fetch_raw_az_coop_daily_df(sid, meta, START_DATE, END_DATE, session=session)
                    save_cached_obs(sid, obs_df)
                except Exception:
                    obs_df = pd.DataFrame()
            valid_days = 0 if (obs_df is None or obs_df.empty) else len(obs_df[obs_df.index <= pd.Timestamp(END_DATE)])
            if valid_days >= args.min_valid_days:
                kept_stations[sid] = meta
                if obs_df is not None and not obs_df.empty:
                    prefetched_obs[sid] = obs_df
            else:
                pruned_ids.append((sid, meta.get("name", sid), valid_days))
        if pruned_ids:
            print(f"Pruned {len(pruned_ids)} short-record/ended station(s) (< {args.min_valid_days} un-estimated valid days):")
            for sid, name, cnt in pruned_ids:
                print(f"  [-] Removed {sid} — {name} ({cnt} valid days)")
        all_stations = kept_stations
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
            with open(OUTPUT_EQ_FILE, "w") as f:
                json.dump(equations_bundle, f, indent=2)
            return
        print(f"Incremental mode: training {len(target_stations)} newly added station(s)...")
    else:
        equations_bundle = {"stations": {}}
        target_stations = all_stations
        print(f"Full daily training mode (step={SAMPLE_STEP_DAYS}d): training {len(target_stations)} stations...")

    obs_by_stn = {}
    valid_targets = {}
    cached_hits = 0
    api_fetches = 0

    syn_targets = [sid for sid, m in target_stations.items() if m.get("source") == "SYNOPTIC"]
    coop_targets = [sid for sid, m in target_stations.items() if m.get("source") == "IEM_COOP"]

    for idx, sid in enumerate(syn_targets, 1):
        meta = target_stations[sid]
        obs_df = prefetched_obs.get(sid) or load_cached_obs(sid, END_DATE, refresh=args.refresh_obs)
        if obs_df is not None:
            cached_hits += 1
        else:
            print(f"[{idx}/{len(syn_targets)}] Fetching Synoptic time-series for {sid} ({meta['name']})...")
            try:
                obs_df = fetch_synoptic_timeseries_daily(session, sid, meta.get("network", ""), START_DATE, END_DATE, token)
                save_cached_obs(sid, obs_df)
                api_fetches += 1
            except Exception as exc:
                print(f"  [!] Skipping {sid}: {exc}")
                continue

        if obs_df is None or obs_df.empty or not isinstance(obs_df.index, pd.DatetimeIndex):
            continue
        obs_df = obs_df[obs_df.index <= pd.Timestamp(END_DATE)]
        if len(obs_df) < 200:
            continue

        w_min, w_max = fit_harmonic_climatology(obs_df)
        meta_enriched = dict(meta)
        meta_enriched.setdefault("elev_ft", 5000)
        meta_enriched["clim_min_w"] = w_min
        meta_enriched["clim_max_w"] = w_max
        obs_by_stn[sid] = obs_df
        valid_targets[sid] = meta_enriched

    for idx, sid in enumerate(coop_targets, 1):
        meta = target_stations[sid]
        obs_df = prefetched_obs.get(sid) or load_cached_obs(sid, END_DATE, refresh=args.refresh_obs)
        if obs_df is not None:
            cached_hits += 1
        else:
            print(f"[{idx}/{len(coop_targets)}] Fetching raw AZ_COOP ({meta.get('az_coop_id')}) observations for {sid} ({meta['name']})...")
            try:
                obs_df = fetch_raw_az_coop_daily_df(sid, meta, START_DATE, END_DATE, session=session)
                save_cached_obs(sid, obs_df)
            except Exception as exc:
                print(f"  [!] Skipping {sid} due to raw COOP fetch error: {exc}")
                continue

        if obs_df is None or obs_df.empty or not isinstance(obs_df.index, pd.DatetimeIndex):
            print(f"  [!] Skipping {sid}: no valid un-estimated AZ_COOP records returned.")
            continue
        obs_df = obs_df[obs_df.index <= pd.Timestamp(END_DATE)]
        if len(obs_df) < 200:
            print(f"  [!] Skipping {sid}: only {len(obs_df)} valid un-estimated COOP days available.")
            continue

        w_min, w_max = fit_harmonic_climatology(obs_df)
        meta_enriched = dict(meta)
        meta_enriched.setdefault("elev_ft", 5000)
        meta_enriched["clim_min_w"] = w_min
        meta_enriched["clim_max_w"] = w_max
        obs_by_stn[sid] = obs_df
        valid_targets[sid] = meta_enriched

    print(f"Observation ingestion complete: {cached_hits} from obs_cache, {api_fetches} fetched from Synoptic API.")

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

    equations_bundle["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    equations_bundle["training_window"] = f"{START_DATE} to {END_DATE}"
    equations_bundle["sample_step_days"] = SAMPLE_STEP_DAYS

    for stn_id, meta in valid_targets.items():
        obs_df = obs_by_stn[stn_id]
        stn_runs = data_store[stn_id]
        equations_bundle["stations"][stn_id] = {"meta": meta, "cycles": {}}

        for cyc, periods in CYCLE_PERIODS.items():
            cyc_hr = int(cyc.replace("Z", ""))
            equations_bundle["stations"][stn_id]["cycles"][cyc] = {}

            for period in periods:
                obs_col = "obs_min" if period["kind"] == "min" else "obs_max"
                rows = []
                for init_dt, var_arrays in stn_runs.items():
                    if init_dt.hour != cyc_hr:
                        continue
                    v_date = pd.Timestamp(get_verify_date_mst(init_dt, period["fhrs"]))
                    if v_date not in obs_df.index:
                        continue
                    target_val = obs_df.loc[v_date, obs_col]
                    if pd.isna(target_val):
                        continue

                    feats = extract_period_features(var_arrays, period, init_dt, meta)
                    if not feats:
                        continue
                    feats["target"] = float(target_val)
                    feats["verify_date"] = v_date
                    rows.append(feats)

                df_p = pd.DataFrame(rows).sort_values("verify_date")
                split_idx = int(len(df_p) * 0.75)
                tr_df, te_df = df_p.iloc[:split_idx], df_p.iloc[split_idx:]

                sel_vars, wts = forward_screening_regression(tr_df, "target", CANDIDATE_PREDICTORS)
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
                flag = " [RAW FALLBACK]" if mos_mae > raw_mae else ""
                print(f"[{stn_id} | {cyc} | {period['key']}] Raw MAE: {raw_mae:.2f}°F -> MOS MAE: {mos_mae:.2f}°F (n={len(df_p)}){flag}")

    with open(OUTPUT_EQ_FILE, "w") as f:
        json.dump(equations_bundle, f, indent=2)
    print(f"\nSaved {len(equations_bundle['stations'])} trained stations to {OUTPUT_EQ_FILE}!")


if __name__ == "__main__":
    main()
