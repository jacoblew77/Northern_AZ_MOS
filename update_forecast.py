"""
update_forecast.py — Operational 6-Hourly HRRR MOS Runner + Multi-Cycle 7-Day Verification
1. Fetches the latest complete 48-hr synoptic HRRR run (00Z, 06Z, 12Z, 18Z) from AWS S3.
2. Computes MOS forecasts using pure HRRR predictors + seasonal harmonic climatology across
   Airports (AZ_ASOS), NWS COOP (AZCLIMATE), NOAA USCRN (USCRN), and RAWS (AZ_DCP).
3. Maintains docs/verification_history.json keyed by station, date, and synoptic cycle
   (00Z, 06Z, 12Z, 18Z), auto-backfilling missing cycles across the past 7 days from AWS S3,
   and exports cycle-selectable 7-day verification metrics into docs/latest_forecasts.json.
"""
import datetime
import io
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
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

warnings.filterwarnings("ignore")
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

EQ_FILE = "mos_equations.json"
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


def get_verify_date_mst(init_dt: datetime.datetime, fhrs: list) -> datetime.date:
    mid_fh = fhrs[len(fhrs) // 2]
    valid_mst = init_dt + datetime.timedelta(hours=mid_fh) - datetime.timedelta(hours=7)
    return valid_mst.date()


def create_sync_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=24, pool_maxsize=24)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def eval_harmonic_norm(w: list, doy: int) -> float:
    return float(w[0] + w[1] * math.cos(2.0 * math.pi * doy / 365.25) + w[2] * math.sin(2.0 * math.pi * doy / 365.25))


def fetch_station_obs_window(stn_id: str, meta: dict, start_dt: datetime.date, end_dt: datetime.date) -> tuple:
    """Fetches daily High/Low observations from IEM across [start_dt, end_dt] for ASOS, COOP, USCRN, or RAWS."""
    session = create_sync_session()
    if meta["network"] == "AZCLIMATE":
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/coop.py?"
            f"network=AZCLIMATE&station={meta['iem_id']}&"
            f"year1={start_dt.year}&month1={start_dt.month}&day1={start_dt.day}&"
            f"year2={end_dt.year}&month2={end_dt.month}&day2={end_dt.day}&"
            f"vars=high,low&what=download&delim=comma&with_header=true"
        )
    else:
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
            f"network={meta['network']}&stations={meta['iem_id']}&"
            f"year1={start_dt.year}&month1={start_dt.month}&day1={start_dt.day}&"
            f"year2={end_dt.year}&month2={end_dt.month}&day2={end_dt.day}&"
            f"var=max_temp_f&var=min_temp_f&format=csv"
        )

    obs_map = {}
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.text.strip()) > 40:
            df = pd.read_csv(io.StringIO(resp.text), comment="#", na_values=["M", "T", " ", "", "None"])
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = next(c for c in ["day", "date"] if c in df.columns)
            min_col = next(c for c in ["low", "min_temp_f", "tmin"] if c in df.columns)
            max_col = next(c for c in ["high", "max_temp_f", "tmax"] if c in df.columns)
            df["dt"] = pd.to_datetime(df[date_col]).dt.date
            df["obs_min"] = pd.to_numeric(df[min_col], errors="coerce")
            df["obs_max"] = pd.to_numeric(df[max_col], errors="coerce")
            df = df.sort_values("dt")
            for _, row in df.iterrows():
                if pd.isna(row["dt"]):
                    continue
                mn = None if pd.isna(row["obs_min"]) else round(float(row["obs_min"]), 1)
                mx = None if pd.isna(row["obs_max"]) else round(float(row["obs_max"]), 1)
                if mn is not None and not (-55.0 <= mn <= 135.0):
                    mn = None
                if mx is not None and not (-55.0 <= mx <= 135.0):
                    mx = None
                if mn is not None and mx is not None and mx < mn:
                    mn, mx = None, None
                obs_map[row["dt"].isoformat()] = {"min": mn, "max": mx}
    except Exception:
        pass
    return stn_id, obs_map


def get_latest_obs_display(obs_map: dict, before_date: datetime.date) -> tuple:
    """Returns the most recent completed daily High/Low before today_mst for display in the station header."""
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


def extract_period_features(var_arrays: dict, period: dict, init_dt: datetime.datetime, meta: dict) -> dict:
    fhrs = period["fhrs"]
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


def format_fhr_window_desc(init_dt: datetime.datetime, fhrs: list, az_tz: zoneinfo.ZoneInfo) -> str:
    """Formats the exact forecast hours, UTC hours, and MST local times for UI tooltips."""
    parts = []
    for fh in fhrs:
        v_utc = (init_dt + datetime.timedelta(hours=fh)).replace(tzinfo=datetime.timezone.utc)
        v_mst = v_utc.astimezone(az_tz)
        parts.append(f"f{fh:02d} ({v_utc.strftime('%HZ')} / {v_mst.strftime('%I:%M %p').lstrip('0')} MST)")
    return ", ".join(parts)


def find_init_for_verify_date(target_v_date: datetime.date, cyc: str, fhrs: list) -> datetime.datetime:
    """Finds the exact UTC initialization datetime for `cyc` whose `fhrs` verify on `target_v_date`."""
    cyc_hr = int(cyc.replace("Z", ""))
    for day_shift in (-2, -1, 0):
        cand_d = target_v_date + datetime.timedelta(days=day_shift)
        cand_init = datetime.datetime(cand_d.year, cand_d.month, cand_d.day, cyc_hr)
        if get_verify_date_mst(cand_init, fhrs) == target_v_date:
            return cand_init
    raise ValueError(f"Could not map verify date {target_v_date} for cycle {cyc} fhrs={fhrs}")


def backfill_past_7_days(stations: dict, hist_records: dict, today_mst: datetime.date):
    """
    Backfills any missing (date, cycle, kind) slots across the past 7 completed MST days
    for all 4 synoptic cycles (00Z, 06Z, 12Z, 18Z) using archived AWS S3 HRRR runs.
    """
    past_dates = [today_mst - datetime.timedelta(days=offset) for offset in range(7, 0, -1)]
    sample_stn = next(iter(stations.values()))

    needed_runs = {}   # init_dt -> set of required fhrs
    missing_jobs = []  # (v_date, cyc, p_key, init_dt)

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
            mos_val = float(eq["intercept"] + sum(w * feats[v] for v, w in zip(eq["predictors"], eq["weights"])))
            raw_val = float(feats["hrrr_ext_t"])
            cyc_slot[kind] = {
                "mos_f": round(mos_val, 1),
                "raw_f": round(raw_val, 1),
                "init_utc": init_dt.strftime("%Y-%m-%d %H:%MZ"),
            }


def build_station_7d_verification(stn_id: str, obs_map: dict, hist_records: dict, today_mst: datetime.date) -> dict:
    """Builds cycle-specific (00Z, 06Z, 12Z, 18Z) 7-day verification series and MAE stats."""
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
    with open(EQ_FILE, "r") as f:
        eq_bundle = json.load(f)

    stations = eq_bundle["stations"]
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

    obs_start = today_mst - datetime.timedelta(days=14)
    with ThreadPoolExecutor(max_workers=12) as pool:
        obs_futures = [pool.submit(fetch_station_obs_window, sid, sinfo["meta"], obs_start, today_mst) for sid, sinfo in stations.items()]
        obs_maps = {sid: omap for sid, omap in (f.result() for f in obs_futures)}

    hist_records = {}
    if os.path.exists(HIST_JSON):
        with open(HIST_JSON, "r") as f:
            hist_bundle = json.load(f)
        if hist_bundle.get("schema_version") == 2:
            hist_records = hist_bundle.get("records", {})

    backfill_past_7_days(stations, hist_records, today_mst)

    output_payload = {
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "updated_mst": now_az.strftime("%Y-%m-%d %I:%M %p MST"),
        "hrrr_init_utc": chosen_init.strftime("%Y-%m-%d %H:%MZ"),
        "synoptic_cycle": chosen_cyc,
        "stations": {},
    }

    for stn_id, s_info in stations.items():
        meta = s_info["meta"]
        obs_map = obs_maps[stn_id]
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
            mos_val = float(eq["intercept"] + sum(w * feats[v] for v, w in zip(eq["predictors"], eq["weights"])))
            raw_val = float(feats["hrrr_ext_t"])
            verify_dt = get_verify_date_mst(chosen_init, eq["fhrs"])
            v_iso = verify_dt.isoformat()
            kind = eq["kind"]

            cyc_slot = stn_hist.setdefault(v_iso, {}).setdefault(chosen_cyc, {})
            if p_key in ("min_1", "max_1") or kind not in cyc_slot:
                cyc_slot[kind] = {
                    "mos_f": round(mos_val, 1),
                    "raw_f": round(raw_val, 1),
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
                "mos_f": round(mos_val, 1),
                "delta_f": round(mos_val - raw_val, 1),
                "holdout_mos_mae": eq["holdout_mos_mae"],
                "holdout_raw_mae": eq["holdout_raw_mae"],
                "breakdown": breakdown,
            })

        prune_cutoff = (today_mst - datetime.timedelta(days=21)).isoformat()
        for old_iso in [d for d in stn_hist if d < prune_cutoff]:
            del stn_hist[old_iso]

        verif_7d = build_station_7d_verification(stn_id, obs_map, hist_records, today_mst)

        output_payload["stations"][stn_id] = {
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

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(HIST_JSON, "w") as f:
        json.dump({"schema_version": 2, "updated_utc": output_payload["updated_utc"], "records": hist_records}, f, indent=2)
    with open(OUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Wrote updated forecasts and multi-cycle 7-day verification for {len(stations)} stations.")


if __name__ == "__main__":
    main()
