"""Hourly ERA5 cloud cover and precipitation at the ground stations (via Open-Meteo).

Open-Meteo serves the ERA5 reanalysis (0.25 deg, hourly) without registration:
https://open-meteo.com/en/docs/historical-weather-api. Responses are cached as JSON.

Line-of-sight model per station and hour (random cloud overlap):

    P(clear) = (1 - c_low) * (1 - c_mid) * (1 - f_cirrus * c_high)

where f_cirrus in [0, 1] is the fraction of high (cirrus) cloud that blocks the beam.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "precipitation"]
CACHE = Path("data/weather")


def fetch_hourly(lat: float, lon: float, start: str, end: str, cache: Path = CACHE):
    """Hourly ERA5 data (UTC) for one location; start/end as YYYY-MM-DD (inclusive)."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"era5_{lat:.4f}_{lon:.4f}_{start}_{end}.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:
        r = requests.get(URL, params={
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": ",".join(VARS), "models": "era5", "timezone": "GMT",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        f.write_text(json.dumps(data))
    h = data["hourly"]
    df = pd.DataFrame({v: h[v] for v in VARS})
    t = pd.to_datetime(h["time"], utc=True)
    df["time"] = ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    return df


def p_clear(df: pd.DataFrame, f_cirrus: float) -> np.ndarray:
    c = {k: df[k].to_numpy(dtype=float) / 100.0 for k in VARS[:3]}
    return (1 - c["cloud_cover_low"]) * (1 - c["cloud_cover_mid"]) * (
        1 - f_cirrus * c["cloud_cover_high"])


class WeatherProcess:
    """Per station: optical line of sight drawn once per hour from ERA5 cloud cover."""

    def __init__(self, stations, t_start: float, t_end: float, f_cirrus: float,
                 rain_max: float, rng):
        start = pd.Timestamp(t_start, unit="s", tz="UTC").strftime("%Y-%m-%d")
        end = pd.Timestamp(t_end, unit="s", tz="UTC").strftime("%Y-%m-%d")
        self.hours, self.p, self.rain = None, [], []
        for gs in stations:
            df = fetch_hourly(gs.site.lat, gs.site.lon, start, end)
            self.hours = df["time"].to_numpy()
            self.p.append(p_clear(df, f_cirrus))
            self.rain.append(df["precipitation"].to_numpy(dtype=float))
        self.p, self.rain = np.array(self.p), np.array(self.rain)
        self.rain_max = rain_max
        self.draw = rng.random(self.p.shape)  # one draw per station and hour

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(optical clear, RF available) per station at time t."""
        k = int(np.clip(np.searchsorted(self.hours, t, side="right") - 1, 0,
                        len(self.hours) - 1))
        return self.draw[:, k] < self.p[:, k], self.rain[:, k] <= self.rain_max
