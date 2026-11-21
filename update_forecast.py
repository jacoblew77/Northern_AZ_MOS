"""
update_forecast.py — Operational 6-Hourly HRRR MOS Runner (Synoptic + Raw IEM AZ_COOP)
- Fetches latest 48-hr synoptic HRRR run (00Z, 06Z, 12Z, 18Z) from AWS S3.
- Verifies against Synoptic UTC diurnal windows and raw un-estimated AZ_COOP daily reports.
- Writes:
  * docs/all_forecasts.json    — Full set of trained stations (consumed by docs/index.html).
  * docs/latest_forecasts.json — Stripped-down curated list controlled by control_stations.json
                                 with exact formatting and station schema preserved.
"""
import datetime
import json
import logging
import math
import os
import warnings
import zoneinfo
from concurrent.futures import ThreadPoolExecutor

import eccodes
import numpy as np
import pandas as pd
import requests

from mos_core import (
    create_sync_session,
    despike_cwop_series,
    extract_period_features,
    fetch_raw_az_coop_daily_df,
    get_synoptic_token,
    get_verify_date_mst,
    is_synthetic_coop_station,
    synoptic_get_sequential,
)

warnings.filterwarnings("ignore")
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

EQ_FILE = "mos_equations.json"
CONTROL_FILE = "control_stations.json"
ALL_OUT_JSON = "docs/all_forecasts.json"
OUT_JSON = "docs/latest_forecasts.json"
HIST_JSON = "docs/verification_history.json"
ALL_CYCLES = ("00Z", "06Z", "12Z", "18Z")

HRRR_IDX_PATTERNS = {
    "tmp_2m":   ":TMP:2 m above ground:",
    "dpt_2m":   ":DPT:2 m above ground:",
    "rh_2m":    ":RH:2 m above ground:",
    "u_10m":    ":UGRD:10 m above ground:",
    "v_10m":    ":VGRD:10 m above ground:",
    "lcdc":     ":LCDC:low cloud layer:",
    "mcdc":     ":MCDC:middle cloud layer:",
    "hcdc":     ":HCDC:high cloud layer:",
    "tcdc":     ":TCDC:entire atmosphere:",
    "tmp_500":  ":TMP:500 mb:",
    "tmp_700":  ":TMP:700 mb:",
    "hgt_500":  ":HGT:500 mb:",
    "hgt_1000": ":HGT:1000 mb:",
    "u_700":    ":UGRD:700 mb:",
    "v_700":    ":VGRD:700 mb:",
    "pwat":     ":PWAT:entire atmosphere",
    "apcp":     ":APCP:surface:0-",
    "snod":     ":SNOD:surface:",
    "dlwrf":    ":DLWRF:surface:",
    "dswrf":    ":DSWRF:surface:",
}


def build_curated_latest_forecasts(all_stations_out: dict, control_path: str = CONTROL_FILE) -> dict:
    """
    Filters `all_stations_out` using `control_stations.json` ({"City Label": "STATION_ID" or null}).
    Preserves exact station order, skips null entries, and injects `control_city`.
    """
    if not os.path.exists(control_path):
        print(f"[!] {control_path} not found — writing all {len(all_stations_out)} stations to {OUT_JSON}.")
        return all_stations_out

    with open(control_path, "r") as f:
        control_map = json.load(f)

    curated = {}
    for city_label, raw_sid in control_map.items():
        if not raw_sid:
            continue
        sid = str(raw_sid).strip().upper()
        if sid not in all_stations_out:
            print(f"  [!] Control station '{sid}' for '{city_label}' is not in {EQ_FILE} — skipping.")
            continue
        if sid in curated:
            print(f"  [!] Duplicate station '{sid}' for '{city_label}' already assigned to '{curated[sid]['control_city']}' — skipping.")
            continue
        stn_copy = dict(all_stations_out[sid])
        stn_copy["control_city"] = city_label
        curated[sid] = stn_copy

    return curated


def fetch_synoptic_recent_obs_batch(
    session: requests.Session,
    stations: dict,
    syn_ids: list,
    start_dt: datetime.date,
    end_dt: datetime.date,
    token: str,
) -> dict:
    obs_maps = {sid: {} for sid in syn_ids}
    start_str = start_dt.strftime("%Y%m%d") + "0000"
    end_str = (end_dt + datetime.timedelta(days=1)).strftime("%Y%m%d") + "0600"

    batch_size = 150
    for idx in range(0, len(syn_ids), batch_size):
        chunk = syn_ids[idx:idx + batch_size]
        payload = synoptic_get_sequential(
            session,
            "stations/timeseries",
            {
                "stid": ",".join(chunk),
                "vars": "air_temp",
                "start": start_str,
                "end": end_str,
                "obtimezone": "UTC",
                "units": "english",
                "hfmetars": "1",
                "qc_checks": "synopticlabs",
                "restricted": "false",
            },
            token=token,
            timeout=60,
        )
        for stn_obj in payload.get("STATION", []):
            if stn_obj.get("RESTRICTED"):
                continue
            sid = str(stn_obj.get("STID", "")).strip().upper()
            obs = stn_obj.get("OBSERVATIONS", {})
            dts = obs.get("date_time", [])
            t_key = "air_temp_set_1" if "air_temp_set_1" in obs else next((k for k in obs if k.startswith("air_temp")), None)
            if not sid or not dts or not t_key:
                continue
            temps = obs.get(t_key, [])
            if len(dts) != len(temps):
                continue

            df = pd.DataFrame({
                "ts": pd.to_datetime(dts, utc=True, errors="coerce"),
                "temp": pd.to_numeric(temps, errors="coerce"),
            }).dropna(subset=["ts", "temp"])
            df = df[(df["temp"] >= -55.0) & (df["temp"] <= 135.0)].sort_values("ts").reset_index(drop=True)
            if df.empty:
                continue

            net = stations.get(sid, {}).get("meta", {}).get("network", "")
            if net == "CWOP" and len(df) >= 3:
                df["temp"] = despike_cwop_series(df["temp"], threshold_f=4.0)

            s_map = obs_maps.setdefault(sid, {})

            in_min = (df["ts"].dt.hour < 18) | ((df["ts"].dt.hour == 18) & (df["ts"].dt.minute == 0))
            min_sub = df[in_min].copy()
            min_sub["d_iso"] = min_sub["ts"].dt.strftime("%Y-%m-%d")
            for d_iso, grp in min_sub.groupby("d_iso"):
                if len(grp) >= 6:
                    s_map.setdefault(d_iso, {"min": None, "max": None})["min"] = round(float(grp["temp"].min()), 1)

            ts_shift = df["ts"] - pd.Timedelta(hours=12)
            in_max = (ts_shift.dt.hour < 18) | ((ts_shift.dt.hour == 18) & (ts_shift.dt.minute == 0))
            max_sub = df[in_max].copy()
            max_sub["d_iso"] = ts_shift[in_max].dt.strftime("%Y-%m-%d")
            for d_iso, grp in max_sub.groupby("d_iso"):
                if len(grp) >= 6:
                    s_map.setdefault(d_iso, {"min": None, "max": None})["max"] = round(float(grp["temp"].max()), 1)

            for d_iso, vals in s_map.items():
                mn_v, mx_v = vals.get("min"), vals.get("max")
                if mn_v is not None and mx_v is not None and (mx_v < mn_v or (mx_v - mn_v) > 75.0):
                    vals["min"], vals["max"] = None, None

    return obs_maps


def fetch_raw_coop_obs_window(stn_id: str, meta: dict, start_dt: datetime.date, end_dt: datetime.date) -> tuple:
    """Fetches raw, un-estimated daily High/Low observations from IEM AZ_COOP for verification."""
    obs_map = {}
    try:
        df = fetch_raw_az_coop_daily_df(stn_id, meta, start_dt.isoformat(), end_dt.isoformat())
        if df is not None and not df.empty:
            for dt_idx, row in df.iterrows():
                mn = None if pd.isna(row["obs_min"]) else round(float(row["obs_min"]), 1)
                mx = None if pd.isna(row["obs_max"]) else round(float(row["obs_max"]), 1)
                obs_map[dt_idx.strftime("%Y-%m-%d")] = {"min": mn, "max": mx}
    except Exception:
        pass
    return stn_id, obs_map


def get_latest_obs_display(obs_map: dict, before_date: datetime.date) -> tuple:
    valid_dates = sorted(
        d_iso for d_iso, vals in obs_map.items()
        if d_iso < before_date.isoformat() and vals.get("min") is not None and vals.get("max") is not None
    )
    if valid_dates:
        best_iso = valid_dates[-1]
        best_dt = datetime.date.fromisoformat(best_iso)
        return float(obs_map[best_iso]["min"]), float(obs_map[best_iso]["max"]), best_dt.strftime("%a, %b %d")
    return None, None, None


def fetch_fhr_all_stations(init_dt: datetime.datetime, fhr: int, stations: dict) -> dict:
    session = create_sync_session()
    d_str = init_dt.strftime("%Y%m%d")
    c_str = f"{init_dt.hour:02d}"
    f_str = f"{fhr:02d}"
    base_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d_str}/conus/hrrr.t{c_str}z.wrfsfcf{f_str}.grib2"

    idx_resp = session.get(base_url + ".idx", timeout=15)
    if idx_resp.status_code != 200:
        return {}

    records = []
    for line in idx_resp.text.strip().splitlines():
        parts = line.split(":")
        if len(parts) >= 6:
            try:
                records.append((int(parts[1]), line))
            except ValueError:
                continue
    records.sort(key=lambda x: x[0])

    matched = []
    for i, (start_b, line) in enumerate(records):
        end_b = (records[i + 1][0] - 1) if (i + 1 < len(records)) else None
        for key, pat in HRRR_IDX_PATTERNS.items():
            if pat in line:
                matched.append((key, start_b, end_b))
                break

    if not matched:
        return {}

    matched.sort(key=lambda x: x[1])
    ranges = []
    cur_s, cur_e, cur_items = matched[0][1], matched[0][2], [(matched[0][0], matched[0][1])]
    for key, s_b, e_b in matched[1:]:
        if cur_e is not None and (s_b - cur_e) <= 262144:
            cur_e = e_b
            cur_items.append((key, s_b))
        else:
            ranges.append((cur_s, cur_e, cur_items))
            cur_s, cur_e, cur_items = s_b, e_b, [(key, s_b)]
    ranges.append((cur_s, cur_e, cur_items))

    station_grid_indices = None
    stn_out = {stn_id: {} for stn_id in stations}

    for r_start, r_end, items in ranges:
        hdr = {"Range": f"bytes={r_start}-{r_end}" if r_end is not None else f"bytes={r_start}-"}
        r = session.get(base_url, headers=hdr, timeout=20)
        if r.status_code not in (200, 206):
            continue
        buf = r.content
        for key, rec_start in items:
            offset = rec_start - r_start
            if offset < 0 or offset >= len(buf):
                continue
            gid = eccodes.codes_new_from_message(buf[offset:])
            if gid is None:
                continue
            try:
                if station_grid_indices is None:
                    lats = eccodes.codes_get_array(gid, "latitudes")
                    lons = eccodes.codes_get_array(gid, "longitudes")
                    lons_180 = np.where(lons > 180.0, lons - 360.0, lons)
                    station_grid_indices = {}
                    for stn_id, s_info in stations.items():
                        t_lat, t_lon = s_info["meta"]["lat"], s_info["meta"]["lon"]
                        dist2 = (lats - t_lat) ** 2 + ((lons_180 - t_lon) * math.cos(math.radians(t_lat))) ** 2
                        station_grid_indices[stn_id] = int(np.argmin(dist2))

                vals = eccodes.codes_get_values(gid)
                for stn_id, g_idx in station_grid_indices.items():
                    stn_out[stn_id][key] = float(vals[g_idx])
            finally:
                eccodes.codes_release(gid)

    return stn_out


def eval_period_guidance(eq: dict, feats: dict) -> tuple:
    raw_val = float(feats["hrrr_ext_t"])
    mos_eq_val = float(eq["intercept"] + sum(w * feats[v] for v, w in zip(eq["predictors"], eq["weights"])))
    use_raw_fallback = float(eq["holdout_mos_mae"]) > float(eq["holdout_raw_mae"])
    active_val = raw_val if use_raw_fallback else mos_eq_val
    return active_val, raw_val, mos_eq_val, use_raw_fallback


def format_fhr_window_desc(init_dt: datetime.datetime, fhrs: list, az_tz: zoneinfo.ZoneInfo) -> str:
    fh0, fh1 = fhrs[0], fhrs[-1]
    v0_utc = (init_dt + datetime.timedelta(hours=fh0)).replace(tzinfo=datetime.timezone.utc)
    v1_utc = (init_dt + datetime.timedelta(hours=fh1)).replace(tzinfo=datetime.timezone.utc)
    v0_mst = v0_utc.astimezone(az_tz).strftime("%I:%M %p").lstrip("0")
    v1_mst = v1_utc.astimezone(az_tz).strftime("%I:%M %p").lstrip("0")
    return f"f{fh0:02d}–f{fh1:02d} ({v0_utc.strftime('%HZ')}–{v1_utc.strftime('%HZ')} / {v0_mst}–{v1_mst} MST, {len(fhrs)} hourly steps)"


def find_init_for_verify_date(target_v_date: datetime.date, cyc: str, fhrs: list) -> datetime.datetime:
    cyc_hr = int(cyc.replace("Z", ""))
    for day_shift in (-2, -1, 0):
        cand_d = target_v_date + datetime.timedelta(days=day_shift)
        cand_init = datetime.datetime(cand_d.year, cand_d.month, cand_d.day, cyc_hr)
        if get_verify_date_mst(cand_init, fhrs) == target_v_date:
            return cand_init
    raise ValueError(f"Could not map verify date {target_v_date} for cycle {cyc} fhrs={fhrs}")


def backfill_past_7_days(stations: dict, hist_records: dict, today_mst: datetime.date):
    past_dates = [today_mst - datetime.timedelta(days=offset) for offset in range(7, 0, -1)]
    sample_stn = next(iter(stations.values()))

    needed_runs = {}
    missing_jobs = []

    for v_dt in past_dates:
        v_iso = v_dt.isoformat()
        for cyc in ALL_CYCLES:
            for p_key in ("min_1", "max_1"):
                eq = sample_stn["cycles"][cyc][p_key]
                kind = eq["kind"]
                any_missing = any(
                    kind not in hist_records.get(sid, {}).get(v_iso, {}).get(cyc, {})
                    for sid in stations
                )
                if any_missing:
                    init_dt = find_init_for_verify_date(v_dt, cyc, eq["fhrs"])
                    missing_jobs.append((v_dt, cyc, p_key, init_dt))
                    req_fhrs = set(eq["fhrs"] + [max(1, eq["fhrs"][0] - 6)])
                    needed_runs.setdefault(init_dt, set()).update(req_fhrs)

    if not needed_runs:
        return

    tasks = [(init_dt, fh) for init_dt, fhr_set in sorted(needed_runs.items()) for fh in sorted(fhr_set)]
    print(f"Backfilling multi-cycle 7-day verification ({len(missing_jobs)} slots across {len(tasks)} GRIB slices)...")

    with ThreadPoolExecutor(max_workers=24) as pool:
        fetched = list(pool.map(
            lambda pair: (pair[0], pair[1], fetch_fhr_all_stations(pair[0], pair[1], stations)),
            tasks,
        ))

    by_init_fhr = {}
    for init_dt, fh, stn_dict in fetched:
        by_init_fhr.setdefault(init_dt, {})[fh] = stn_dict

    for v_dt, cyc, p_key, init_dt in missing_jobs:
        v_iso = v_dt.isoformat()
        for stn_id, s_info in stations.items():
            meta = s_info["meta"]
            eq = s_info["cycles"][cyc][p_key]
            kind = eq["kind"]
            cyc_slot = hist_records.setdefault(stn_id, {}).setdefault(v_iso, {}).setdefault(cyc, {})
            if kind in cyc_slot:
                continue

            var_arrays = {}
            for vkey in HRRR_IDX_PATTERNS:
                arr = np.full(36, np.nan, dtype=np.float32)
                for fh, stn_dict in by_init_fhr.get(init_dt, {}).items():
                    val = stn_dict.get(stn_id, {}).get(vkey)
                    if val is not None:
                        arr[fh - 1] = val
                var_arrays[vkey] = arr

            feats = extract_period_features(var_arrays, eq, init_dt, meta)
            if not feats:
                continue
            active_val, raw_val, _, used_fallback = eval_period_guidance(eq, feats)
            cyc_slot[kind] = {
                "mos_f": round(active_val, 1),
                "raw_f": round(raw_val, 1),
                "used_raw_fallback": used_fallback,
                "init_utc": init_dt.strftime("%Y-%m-%d %H:%MZ"),
            }


def build_station_7d_verification(stn_id: str, obs_map: dict, hist_records: dict, today_mst: datetime.date) -> dict:
    past_dates = [today_mst - datetime.timedelta(days=offset) for offset in range(7, 0, -1)]
    stn_hist = hist_records.get(stn_id, {})

    by_cycle = {}
    for cyc in ALL_CYCLES:
        cyc_out = {"min": [], "max": [], "min_mos_mae": None, "min_raw_mae": None, "max_mos_mae": None, "max_raw_mae": None}
        for kind in ("min", "max"):
            mos_errs = []
            raw_errs = []
            for dt in past_dates:
                d_iso = dt.isoformat()
                fcst = stn_hist.get(d_iso, {}).get(cyc, {}).get(kind)
                obs_val = obs_map.get(d_iso, {}).get(kind)
                mos_f = fcst["mos_f"] if fcst else None
                raw_f = fcst["raw_f"] if fcst else None
                if mos_f is not None and obs_val is not None:
                    mos_errs.append(abs(mos_f - obs_val))
                if raw_f is not None and obs_val is not None:
                    raw_errs.append(abs(raw_f - obs_val))
                cyc_out[kind].append({
                    "date": d_iso,
                    "label": dt.strftime("%a %m/%d"),
                    "obs_f": obs_val,
                    "mos_f": mos_f,
                    "raw_f": raw_f,
                })
            if mos_errs:
                cyc_out[f"{kind}_mos_mae"] = round(float(np.mean(mos_errs)), 2)
            if raw_errs:
                cyc_out[f"{kind}_raw_mae"] = round(float(np.mean(raw_errs)), 2)
        by_cycle[cyc] = cyc_out

    return by_cycle


def main():
    token = get_synoptic_token()
    with open(EQ_FILE, "r") as f:
        eq_bundle = json.load(f)

    stations = {
        sid: s for sid, s in eq_bundle["stations"].items()
        if not is_synthetic_coop_station(sid, s.get("meta", {}).get("name", ""))
    }
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    az_tz = zoneinfo.ZoneInfo("America/Phoenix")
    now_az = datetime.datetime.now(az_tz)
    today_mst = now_az.date()

    latest_syn = now_utc.replace(minute=0, second=0, microsecond=0)
    latest_syn -= datetime.timedelta(hours=(latest_syn.hour % 6))

    chosen_init = None
    chosen_cyc = None

    for back_hrs in range(0, 24, 6):
        cand_init = latest_syn - datetime.timedelta(hours=back_hrs)
        cyc_name = f"{cand_init.hour:02d}Z"
        probe = fetch_fhr_all_stations(cand_init, 36, stations)
        if probe and "tmp_2m" in next(iter(probe.values())):
            chosen_init = cand_init
            chosen_cyc = cyc_name
            break

    if chosen_init is None:
        raise RuntimeError("Could not find a complete synoptic HRRR run on AWS S3.")

    print(f"Active 48-hr Synoptic HRRR Cycle: {chosen_init.strftime('%Y-%m-%d %H:%MZ')} ({chosen_cyc})")

    fhrs_to_fetch = list(range(1, 37))
    with ThreadPoolExecutor(max_workers=16) as pool:
        fhr_results = dict(zip(fhrs_to_fetch, pool.map(lambda fh: fetch_fhr_all_stations(chosen_init, fh, stations), fhrs_to_fetch)))

    session = create_sync_session()
    syn_ids = [sid for sid, s in stations.items() if s["meta"].get("network") != "AZCLIMATE"]
    coop_ids = [sid for sid, s in stations.items() if s["meta"].get("network") == "AZCLIMATE"]

    obs_start = today_mst - datetime.timedelta(days=10)
    obs_maps = {}
    if syn_ids:
        print(f"Fetching 10-day UTC diurnal High/Low verification from Synoptic for {len(syn_ids)} stations...")
        obs_maps.update(fetch_synoptic_recent_obs_batch(session, stations, syn_ids, obs_start, today_mst, token))

    if coop_ids:
        coop_start = today_mst - datetime.timedelta(days=14)
        with ThreadPoolExecutor(max_workers=8) as pool:
            coop_futures = [pool.submit(fetch_raw_coop_obs_window, sid, stations[sid]["meta"], coop_start, today_mst) for sid in coop_ids]
            for sid, omap in (f.result() for f in coop_futures):
                obs_maps[sid] = omap

    hist_records = {}
    if os.path.exists(HIST_JSON):
        with open(HIST_JSON, "r") as f:
            hist_bundle = json.load(f)
        if hist_bundle.get("schema_version") == 2:
            hist_records = hist_bundle.get("records", {})

    backfill_past_7_days(stations, hist_records, today_mst)

    base_meta = {
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "updated_mst": now_az.strftime("%Y-%m-%d %I:%M %p MST"),
        "hrrr_init_utc": chosen_init.strftime("%Y-%m-%d %H:%MZ"),
        "synoptic_cycle": chosen_cyc,
    }

    all_stations_out = {}

    for stn_id, s_info in stations.items():
        meta = s_info["meta"]
        obs_map = obs_maps.get(stn_id, {})
        last_min, last_max, obs_date_lbl = get_latest_obs_display(obs_map, today_mst)

        var_arrays = {}
        for vkey in HRRR_IDX_PATTERNS:
            arr = np.full(36, np.nan, dtype=np.float32)
            for fh in fhrs_to_fetch:
                val = fhr_results.get(fh, {}).get(stn_id, {}).get(vkey)
                if val is not None:
                    arr[fh - 1] = val
            var_arrays[vkey] = arr

        hourly_curve = []
        for fh in fhrs_to_fetch:
            k_val = var_arrays["tmp_2m"][fh - 1]
            if not np.isnan(k_val):
                t_f = round((float(k_val) - 273.15) * 9.0 / 5.0 + 32.0, 1)
                valid_az = (chosen_init + datetime.timedelta(hours=fh)).replace(tzinfo=datetime.timezone.utc).astimezone(az_tz)
                hourly_curve.append({
                    "fhr": fh,
                    "time_mst": valid_az.strftime("%a %I %p"),
                    "raw_temp_f": t_f,
                })

        periods_out = []
        cycle_eqs = s_info["cycles"][chosen_cyc]
        stn_hist = hist_records.setdefault(stn_id, {})

        for p_key, eq in cycle_eqs.items():
            feats = extract_period_features(var_arrays, eq, chosen_init, meta)
            if not feats:
                continue
            active_val, raw_val, mos_eq_val, used_fallback = eval_period_guidance(eq, feats)
            verify_dt = get_verify_date_mst(chosen_init, eq["fhrs"])
            v_iso = verify_dt.isoformat()
            kind = eq["kind"]

            cyc_slot = stn_hist.setdefault(v_iso, {}).setdefault(chosen_cyc, {})
            if p_key in ("min_1", "max_1") or kind not in cyc_slot:
                cyc_slot[kind] = {
                    "mos_f": round(active_val, 1),
                    "raw_f": round(raw_val, 1),
                    "used_raw_fallback": used_fallback,
                    "init_utc": chosen_init.strftime("%Y-%m-%d %H:%MZ"),
                }

            breakdown = [{"term": "Intercept", "value": 1.0, "weight": round(eq["intercept"], 4), "contrib": round(eq["intercept"], 2)}]
            for v, w in zip(eq["predictors"], eq["weights"]):
                breakdown.append({
                    "term": v,
                    "value": round(feats[v], 2),
                    "weight": round(w, 4),
                    "contrib": round(feats[v] * w, 2),
                })

            periods_out.append({
                "key": p_key,
                "kind": kind,
                "label": eq["label"],
                "fhrs": eq["fhrs"],
                "window_desc": format_fhr_window_desc(chosen_init, eq["fhrs"], az_tz),
                "verify_date": verify_dt.strftime("%a, %b %d"),
                "raw_hrrr_f": round(raw_val, 1),
                "mos_f": round(active_val, 1),
                "mos_eq_f": round(mos_eq_val, 1),
                "used_raw_fallback": used_fallback,
                "delta_f": round(active_val - raw_val, 1),
                "holdout_mos_mae": eq["holdout_mos_mae"],
                "holdout_raw_mae": eq["holdout_raw_mae"],
                "breakdown": breakdown,
            })

        prune_cutoff = (today_mst - datetime.timedelta(days=21)).isoformat()
        for old_iso in [d for d in stn_hist if d < prune_cutoff]:
            del stn_hist[old_iso]

        verif_7d = build_station_7d_verification(stn_id, obs_map, hist_records, today_mst)

        all_stations_out[stn_id] = {
            "id": stn_id,
            "name": meta["name"],
            "network": meta["network"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "elev_ft": meta.get("elev_ft", 5000),
            "obs_date_label": obs_date_lbl,
            "prev_obs_min": round(last_min, 1) if last_min is not None else None,
            "prev_obs_max": round(last_max, 1) if last_max is not None else None,
            "periods": periods_out,
            "hourly_curve": hourly_curve,
            "past_7d_verification": verif_7d,
        }

    curated_latest_out = build_curated_latest_forecasts(all_stations_out)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(HIST_JSON, "w") as f:
        json.dump({"schema_version": 2, "updated_utc": base_meta["updated_utc"], "records": hist_records}, f, indent=2)
    with open(ALL_OUT_JSON, "w") as f:
        json.dump({**base_meta, "stations": all_stations_out}, f, indent=2)
    with open(OUT_JSON, "w") as f:
        json.dump({**base_meta, "stations": curated_latest_out}, f, indent=2)

    print(
        f"Wrote {len(all_stations_out)} total stations to {ALL_OUT_JSON} and "
        f"{len(curated_latest_out)} curated control stations to {OUT_JSON}."
    )
    for sid, obj in curated_latest_out.items():
        print(f"  [{obj['control_city']}] -> {sid} ({obj['name']})")


if __name__ == "__main__":
    main()
