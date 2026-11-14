"""
update_forecast.py — Operational 6-Hourly HRRR MOS Runner for GitHub Actions
Uses exact MST verification dates and strict MST completed-day observation cutoffs
matching train_mos.py with zero look-ahead leakage.
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


def get_obs_cutoff_date_mst(init_dt: datetime.datetime) -> datetime.date:
    init_mst = init_dt - datetime.timedelta(hours=7)
    return init_mst.date() - datetime.timedelta(days=1)


def create_sync_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=24, pool_maxsize=24)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def eval_harmonic_norm(w: list, doy: int) -> float:
    return float(w[0] + w[1] * math.cos(2.0 * math.pi * doy / 365.25) + w[2] * math.sin(2.0 * math.pi * doy / 365.25))


def fetch_latest_obs_persistence(stn_id: str, meta: dict, cutoff_date: datetime.date) -> tuple:
    """Fetches the latest completed MST daily High/Low on or before cutoff_date."""
    session = create_sync_session()
    s_dt = cutoff_date - datetime.timedelta(days=10)
    e_dt = cutoff_date
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
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.text.strip()) > 40:
            df = pd.read_csv(io.StringIO(resp.text), comment="#", na_values=["M", "T", " ", "", "None"])
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = next(c for c in ["day", "date"] if c in df.columns)
            min_col = next(c for c in ["low", "min_temp_f", "tmin"] if c in df.columns)
            max_col = next(c for c in ["high", "max_temp_f", "tmax"] if c in df.columns)
            df["dt"] = pd.to_datetime(df[date_col]).dt.date
            df["obs_min"] = pd.to_numeric(df[min_col], errors="coerce")
            df["obs_max"] = pd.to_numeric(df[max_col], errors="coerce")
            df = df[df["dt"] <= cutoff_date].dropna(subset=["obs_min", "obs_max"]).sort_values("dt")
            if not df.empty:
                last_row = df.iloc[-1]
                return stn_id, float(last_row["obs_min"]), float(last_row["obs_max"]), last_row["dt"].strftime("%a, %b %d")
    except Exception:
        pass
    doy = cutoff_date.timetuple().tm_yday
    fallback_min = eval_harmonic_norm(meta.get("clim_min_w", [35.0, -18.0, -5.0]), doy)
    fallback_max = eval_harmonic_norm(meta.get("clim_max_w", [65.0, -20.0, -5.0]), doy)
    return stn_id, fallback_min, fallback_max, cutoff_date.strftime("%a, %b %d (Clim)")


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


def extract_period_features(var_arrays: dict, period: dict, init_dt: datetime.datetime, meta: dict, prev_min: float, prev_max: float) -> dict:
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
        "obs_prev_min": float(prev_min),
        "obs_prev_max": float(prev_max),
        "sin_doy_1": math.sin(2.0 * math.pi * doy / 365.25),
        "cos_doy_1": math.cos(2.0 * math.pi * doy / 365.25),
        "sin_doy_2": math.sin(4.0 * math.pi * doy / 365.25),
        "cos_doy_2": math.cos(4.0 * math.pi * doy / 365.25),
        "norm_t": norm_t,
    }


def main():
    with open(EQ_FILE, "r") as f:
        eq_bundle = json.load(f)

    stations = eq_bundle["stations"]
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    az_tz = zoneinfo.ZoneInfo("America/Phoenix")

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

    obs_cutoff_dt = get_obs_cutoff_date_mst(chosen_init)
    print(f"Active 48-hr Synoptic HRRR Cycle: {chosen_init.strftime('%Y-%m-%d %H:%MZ')} ({chosen_cyc}) | Obs Cutoff: {obs_cutoff_dt}")

    fhrs_to_fetch = list(range(1, 37))
    with ThreadPoolExecutor(max_workers=16) as pool:
        fhr_results = dict(zip(fhrs_to_fetch, pool.map(lambda fh: fetch_fhr_all_stations(chosen_init, fh, stations), fhrs_to_fetch)))

    with ThreadPoolExecutor(max_workers=12) as pool:
        obs_futures = [pool.submit(fetch_latest_obs_persistence, sid, sinfo["meta"], obs_cutoff_dt) for sid, sinfo in stations.items()]
        obs_pers = {sid: (pmn, pmx, p_lbl) for sid, pmn, pmx, p_lbl in (f.result() for f in obs_futures)}

    output_payload = {
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "updated_mst": datetime.datetime.now(az_tz).strftime("%Y-%m-%d %I:%M %p MST"),
        "hrrr_init_utc": chosen_init.strftime("%Y-%m-%d %H:%MZ"),
        "synoptic_cycle": chosen_cyc,
        "stations": {},
    }

    for stn_id, s_info in stations.items():
        meta = s_info["meta"]
        prev_min, prev_max, obs_date_lbl = obs_pers[stn_id]

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
        for p_key, eq in cycle_eqs.items():
            feats = extract_period_features(var_arrays, eq, chosen_init, meta, prev_min, prev_max)
            if not feats:
                continue
            mos_val = float(eq["intercept"] + sum(w * feats[v] for v, w in zip(eq["predictors"], eq["weights"])))
            raw_val = float(feats["hrrr_ext_t"])
            verify_dt = get_verify_date_mst(chosen_init, eq["fhrs"])

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
                "kind": eq["kind"],
                "label": eq["label"],
                "verify_date": verify_dt.strftime("%a, %b %d"),
                "raw_hrrr_f": round(raw_val, 1),
                "mos_f": round(mos_val, 1),
                "delta_f": round(mos_val - raw_val, 1),
                "holdout_mos_mae": eq["holdout_mos_mae"],
                "holdout_raw_mae": eq["holdout_raw_mae"],
                "breakdown": breakdown,
            })

        output_payload["stations"][stn_id] = {
            "id": stn_id,
            "name": meta["name"],
            "network": meta["network"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "elev_ft": meta.get("elev_ft", 5000),
            "obs_date_label": obs_date_lbl,
            "prev_obs_min": round(prev_min, 1),
            "prev_obs_max": round(prev_max, 1),
            "periods": periods_out,
            "hourly_curve": hourly_curve,
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Wrote updated forecasts for {len(stations)} stations to {OUT_JSON}.")


if __name__ == "__main__":
    main()
