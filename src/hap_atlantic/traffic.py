"""Air traffic: loading OpenSky state vectors, closing oceanic coverage gaps, snapshots.

Canonical state-vector table (one row per aircraft and time):

    time [s, UNIX], icao24 [str], lat [deg], lon [deg], alt [m], interpolated [bool]

Terrestrial ADS-B receivers do not cover the central North Atlantic. Positions inside a
coverage gap are reconstructed along the great circle between the last position before
and the first position after the gap; this matches the organised oceanic tracks to within
a few tens of km, which is small compared with the link ranges considered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import great_circle, slerp_latlon

COLUMNS = ["time", "icao24", "lat", "lon", "alt"]


def load_states(paths) -> pd.DataFrame:
    """Load one or several Parquet/CSV files with (raw or canonical) state vectors."""
    if isinstance(paths, str | Path):
        p = Path(paths)
        paths = sorted(p.glob("*.parquet")) + sorted(p.glob("*.csv*")) if p.is_dir() else [p]
    frames = []
    for path in paths:
        path = Path(path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frames.append(normalise(df))
    if not frames:
        raise FileNotFoundError(f"no state-vector files found in {paths}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["icao24", "time"])


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map OpenSky REST / Trino column names to the canonical schema."""
    rename = {
        "latitude": "lat",
        "longitude": "lon",
        "geo_altitude": "alt",
        "geoaltitude": "alt",
        "time_position": "time_pos",
        "timestamp": "time",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "alt" not in df.columns:
        for c in ("baro_altitude", "baroaltitude"):
            if c in df.columns:
                df = df.rename(columns={c: "alt"})
                break
    elif "baro_altitude" in df.columns or "baroaltitude" in df.columns:
        baro = df["baro_altitude"] if "baro_altitude" in df.columns else df["baroaltitude"]
        df["alt"] = df["alt"].fillna(baro)
    if "time_pos" in df.columns:  # REST: prefer the time of the last position update
        df["time"] = df["time_pos"].fillna(df["time"])
    if "onground" in df.columns:
        df = df.rename(columns={"onground": "on_ground"})
    if "on_ground" in df.columns:
        df = df[~df["on_ground"].astype(bool)]
    if np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = df["time"].astype("int64") // 10**9
    df = df.dropna(subset=COLUMNS)
    out = df[COLUMNS].copy()
    out["icao24"] = out["icao24"].astype(str).str.strip().str.lower()
    out["time"] = out["time"].astype(float)
    return out


def interpolate_to_grid(
    states: pd.DataFrame,
    dt: float,
    max_gap: float = 5 * 3600.0,
    min_speed: float = 120.0,
    max_speed: float = 340.0,
    radius: float = 6_371_008.8,
) -> pd.DataFrame:
    """Resample every aircraft onto a common time grid with spacing dt.

    Gaps longer than max_gap, or whose implied ground speed is outside
    [min_speed, max_speed] m/s while longer than 10 min, are not bridged (aircraft landed,
    two different flights with the same transponder, ...).
    """
    out = []
    for icao, g in states.sort_values("time").groupby("icao24", sort=False):
        t = g["time"].to_numpy()
        if len(t) < 2:
            continue
        lat, lon, alt = (g[c].to_numpy() for c in ("lat", "lon", "alt"))
        gap = np.diff(t)
        dist = great_circle(lat[:-1], lon[:-1], lat[1:], lon[1:], radius)
        speed = dist / np.maximum(gap, 1e-9)
        ok = (gap > 0) & (gap <= max_gap) & (
            (gap <= 600.0) | ((speed >= min_speed) & (speed <= max_speed))
        )
        grid = np.arange(np.ceil(t[0] / dt) * dt, t[-1] + 1e-9, dt)
        if grid.size == 0:
            continue
        i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
        valid = ok[i]
        if not valid.any():
            continue
        grid, i = grid[valid], i[valid]
        f = (grid - t[i]) / gap[i]
        glat, glon = slerp_latlon(lat[i], lon[i], lat[i + 1], lon[i + 1], f)
        galt = alt[i] + f * (alt[i + 1] - alt[i])
        out.append(
            pd.DataFrame(
                {
                    "time": grid,
                    "icao24": icao,
                    "lat": glat,
                    "lon": glon,
                    "alt": galt,
                    "interpolated": gap[i] > 120.0,
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=[*COLUMNS, "interpolated"])
    return pd.concat(out, ignore_index=True)


def snapshots(grid_states: pd.DataFrame):
    """Yield (time, DataFrame) per grid time step."""
    yield from grid_states.groupby("time", sort=True)


# ----------------------------------------------------------------------------------------
# Synthetic traffic (only for tests and for developing the pipeline before real data).
# ----------------------------------------------------------------------------------------

EU_AIRPORTS = {
    "LHR": (51.47, -0.45), "FRA": (50.03, 8.57), "CDG": (49.01, 2.55), "AMS": (52.31, 4.76),
    "MUC": (48.35, 11.79), "MAD": (40.47, -3.56), "DUB": (53.42, -6.27), "ZRH": (47.46, 8.55),
    "FCO": (41.80, 12.25), "CPH": (55.62, 12.65), "MAN": (53.35, -2.27), "LIS": (38.77, -9.13),
}
NA_AIRPORTS = {
    "JFK": (40.64, -73.78), "EWR": (40.69, -74.17), "BOS": (42.36, -71.01),
    "IAD": (38.95, -77.46), "ORD": (41.98, -87.90), "ATL": (33.64, -84.43),
    "YYZ": (43.68, -79.63), "YUL": (45.47, -73.74), "PHL": (39.87, -75.24),
    "MIA": (25.79, -80.29), "DFW": (32.90, -97.04), "LAX": (33.94, -118.41),
}


def synthetic_states(
    n_flights: int = 1400,
    n_regional: int = 3000,
    start: float = 0.0,
    duration: float = 86400.0,
    dt: float = 60.0,
    cruise_alt: float = 11_000.0,
    speed: float = 250.0,
    seed: int = 0,
    radius: float = 6_371_008.8,
) -> pd.DataFrame:
    """Great-circle flights with a North-Atlantic-like diurnal departure pattern.

    Westbound flights leave Europe mainly 08-15 UTC, eastbound flights leave North America
    mainly 21-03 UTC, which reproduces the two daily traffic waves over the ocean.
    n_regional continental flights per continent provide relays near the endpoints.
    """
    rng = np.random.default_rng(seed)
    eu, na = list(EU_AIRPORTS.values()), list(NA_AIRPORTS.values())
    rows = []
    for k in range(n_flights + 2 * n_regional):
        if k >= n_flights:  # continental background traffic, 06-21 local time
            pool, utc_offset = (eu, -1.0) if k < n_flights + n_regional else (na, 5.0)
            i, j = rng.choice(len(pool), size=2, replace=False)
            o, d = pool[i], pool[j]
            dep_h = rng.uniform(6.0, 21.0) + utc_offset
        elif rng.random() < 0.5:  # westbound
            o, d = eu[rng.integers(len(eu))], na[rng.integers(len(na))]
            dep_h = rng.normal(11.5, 2.0)
        else:  # eastbound
            o, d = na[rng.integers(len(na))], eu[rng.integers(len(eu))]
            dep_h = rng.normal(24.0, 2.0)
        # flights departing on the previous day are still airborne at the start
        dep = start + (dep_h % 24.0) * 3600.0 - rng.choice([0.0, 86400.0])
        dist = great_circle(o[0], o[1], d[0], d[1], radius)
        t_flight = dist / speed
        t = np.arange(np.ceil(dep / dt) * dt, dep + t_flight, dt)
        t = t[(t >= start) & (t < start + duration)]
        if t.size == 0:
            continue
        f = (t - dep) / t_flight
        lat, lon = slerp_latlon(np.full_like(f, o[0]), np.full_like(f, o[1]),
                                np.full_like(f, d[0]), np.full_like(f, d[1]), f)
        ramp = 200e3 / dist  # climb / descent over the first / last 200 km
        alt = cruise_alt * np.clip(np.minimum(f, 1 - f) / ramp, 0.0, 1.0)
        alt += rng.choice([-600.0, 0.0, 600.0])  # flight levels
        icao = f"{rng.integers(16**6):06x}"
        rows.append(pd.DataFrame({"time": t, "icao24": icao, "lat": lat, "lon": lon,
                                  "alt": alt, "interpolated": False}))
    return pd.concat(rows, ignore_index=True)
