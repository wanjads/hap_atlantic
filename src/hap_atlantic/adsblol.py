"""Reader for the adsb.lol daily archives (github.com/adsblol/globe_history_<year>, ODbL).

Each release is a tar file with one gzip-compressed readsb trace per aircraft
(`traces/<xx>/trace_full_<icao>.json`). Trace entries are arrays

    [dt, lat, lon, alt_baro (ft | "ground" | null), gs, track, flags, vrate, details,
     source, alt_geom (ft), ...]

relative to the top-level `timestamp` (README-json.md of wiedehopf/readsb). Positions keep
the plain ICAO address so that flights crossing midnight join up across daily files;
landings are not bridged because `interpolate_to_grid` rejects implausible ground speeds.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

FT = 0.3048


def parse_trace(data: dict, bbox, min_dt: float) -> pd.DataFrame | None:
    """Airborne points of one trace inside bbox, thinned to at most one point per min_dt."""
    lamin, lamax, lomin, lomax = bbox
    t0 = float(data["timestamp"])
    icao = str(data["icao"]).lower()
    rows, last_t = [], -np.inf
    for e in data.get("trace", []):
        alt = e[3]
        if alt == "ground" or alt is None:
            continue
        lat, lon = e[1], e[2]
        if not (lamin <= lat <= lamax and lomin <= lon <= lomax):
            continue
        t = t0 + e[0]
        if t - last_t < min_dt:
            continue
        geom = e[10] if len(e) > 10 and e[10] is not None else None
        rows.append((t, icao, lat, lon, (geom if geom is not None else alt) * FT))
        last_t = t
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["time", "icao24", "lat", "lon", "alt"])


def read_archive(path: str | Path, bbox, min_dt: float = 30.0, progress: bool = True):
    """All airborne positions inside bbox from one adsb.lol tar archive."""
    frames, n = [], 0
    with tarfile.open(path) as tar:
        for member in tar:
            if not (member.isfile() and "trace_full_" in member.name):
                continue
            raw = tar.extractfile(member).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            df = parse_trace(json.loads(raw), bbox, min_dt)
            if df is not None:
                frames.append(df)
            n += 1
            if progress and n % 20000 == 0:
                print(f"  {n} traces read, {len(frames)} in the area", flush=True)
    return pd.concat(frames, ignore_index=True)
