"""Cloud blockage of optical links from hourly ERA5 cloud cover.

ERA5 (Hersbach et al. 2020, 0.25 deg, hourly) is read through the Open-Meteo archive API
(https://open-meteo.com/en/docs/historical-weather-api, no registration) on a regular
lat/lon grid that covers the corridor. Responses are cached as JSON in data/weather.

ERA5 gives the cloud *fraction* per layer (low < 2 km, mid 2-6 km, high > 6 km). Whether a
given ray is blocked is decided by a latent standard Gaussian field Z(x, t) per layer:
a point x in layer l is cloudy at time t if

    Phi(Z_l(x, t)) < c_l(x, t)            (high layer: f_cirrus * c_high)

so the cloudy area matches the ERA5 fraction, and nearby points / times are cloudy
together. Z has the covariance exp(-|dx|^2 / 2 L^2) * exp(-dt^2 / 2 T^2) and is drawn with
random Fourier features, i.e. it is a deterministic function of (x, t, seed). Layers are
independent (random overlap, the same assumption as P_clear = prod(1 - c_l)).

* air-to-air rays stay above h_chord_min (7 km) and only cross the high layer: a ray is
  blocked if any of its sample points lies in a cloudy cell of the high-layer mask;
* ground links cross all three layers at the heights `layer_heights`;
* RF fallback of a ground station fails when ERA5 precipitation exceeds rf_rain_max.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.special import ndtr

from . import geometry as geo
from .config import NORTH_ATLANTIC_BBOX, CloudParams

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "precipitation"]
CACHE = Path("data/weather")
CHUNK = 50  # locations per request
LOW, MID, HIGH = 0, 1, 2


def _day(t: float) -> str:
    return pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%d")


def fetch_grid(lats, lons, start: str, end: str, cache: Path = CACHE, pause: float = 6.0):
    """Hourly ERA5 data on the grid lats x lons, start/end as YYYY-MM-DD (inclusive).

    Returns (hours [s, UNIX], data[var] of shape (n_hours, n_lat, n_lon)).
    """
    folder = cache / (f"era5_grid_{lats[0]:g}_{lats[-1]:g}_{len(lats)}_{lons[0]:g}_{lons[-1]:g}_"
                 f"{len(lons)}_{start}_{end}")
    folder.mkdir(parents=True, exist_ok=True)
    pts = [(la, lo) for la in lats for lo in lons]
    raw = []
    for i in range(0, len(pts), CHUNK):
        f = folder / f"chunk_{i // CHUNK:04d}.json"
        if not f.exists():
            chunk = pts[i: i + CHUNK]
            params = {
                "latitude": ",".join(f"{la:g}" for la, _ in chunk),
                "longitude": ",".join(f"{lo:g}" for _, lo in chunk),
                "start_date": start, "end_date": end, "hourly": ",".join(VARS),
                "models": "era5", "timezone": "GMT",
            }
            for attempt in range(6):
                try:
                    r = requests.get(URL, params=params, timeout=180)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 5:
                        raise
                    time.sleep(30 * (attempt + 1))
            out = r.json()
            out = out if isinstance(out, list) else [out]
            f.write_text(json.dumps([{"hourly": o["hourly"]} for o in out]))
            time.sleep(pause)  # free tier: 600 location-calls per minute
        raw += json.loads(f.read_text())
    t = pd.to_datetime(raw[0]["hourly"]["time"], utc=True)
    hours = ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(float)
    data = {}
    for v in VARS:
        a = np.array([d["hourly"][v] for d in raw], dtype=float)  # (n_pts, n_hours)
        a = np.nan_to_num(a, nan=0.0)
        data[v] = a.T.reshape(len(hours), len(lats), len(lons))
    return hours, data


class LatentField:
    """Standard Gaussian space-time field via random Fourier features (deterministic)."""

    def __init__(self, l_corr: float, t_corr: float, n_features: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal((3, n_features)) / l_corr
        self.nu = rng.standard_normal(n_features) / t_corr
        self.phi = rng.uniform(0, 2 * np.pi, n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, xyz: np.ndarray, t: float) -> np.ndarray:
        """Z at Cartesian points xyz (..., 3) [m] and time t [s]."""
        arg = xyz @ self.w + (self.nu * t + self.phi)
        return self.scale * np.cos(arg).sum(axis=-1)


class CloudField:
    """ERA5 cloud fractions + latent fields; `at(t)` gives the cloud state at time t."""

    def __init__(self, t_start: float, t_end: float, cp: CloudParams, seed: int,
                 radius: float, bbox=NORTH_ATLANTIC_BBOX):
        lamin, lamax, lomin, lomax = bbox
        self.lats = np.arange(lamin, lamax + 1e-9, cp.grid_step)
        self.lons = np.arange(lomin, lomax + 1e-9, cp.grid_step)
        self.hours, self.data = fetch_grid(self.lats, self.lons, _day(t_start), _day(t_end))
        self.cp, self.radius = cp, radius
        # one independent latent field per layer, seeds derived from the run seed
        self.z = [LatentField(cp.l_corr, cp.t_corr, cp.n_features, 1000 * seed + k)
                  for k in range(3)]
        # high-layer mask grid for air-to-air rays
        self.mlats = np.arange(lamin, lamax + 1e-9, cp.mask_step)
        self.mlons = np.arange(lomin, lomax + 1e-9, cp.mask_step)
        mlat, mlon = np.meshgrid(self.mlats, self.mlons, indexing="ij")
        self.mxyz = geo.to_ecef(mlat, mlon, 0.0, radius)

    def cover(self, layer: int, lat, lon, t: float) -> np.ndarray:
        """ERA5 cover fraction [0, 1] of a layer, bilinear in space, linear in time."""
        v = VARS[layer]
        return self._interp(self.data[v], lat, lon, t) / 100.0

    def rain(self, lat, lon, t: float) -> np.ndarray:
        return self._interp(self.data["precipitation"], lat, lon, t)

    def _interp(self, a, lat, lon, t):
        k = np.clip(np.searchsorted(self.hours, t, side="right") - 1, 0, len(self.hours) - 2)
        ft = np.clip((t - self.hours[k]) / (self.hours[k + 1] - self.hours[k]), 0.0, 1.0)
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        fi = np.clip((lat - self.lats[0]) / self.cp.grid_step, 0, len(self.lats) - 1.000001)
        fj = np.clip((lon - self.lons[0]) / self.cp.grid_step, 0, len(self.lons) - 1.000001)
        i, j = fi.astype(int), fj.astype(int)
        di, dj = fi - i, fj - j
        out = 0.0
        for kk, wt in ((k, 1 - ft), (k + 1, ft)):
            g = a[kk]
            out = out + wt * ((1 - di) * (1 - dj) * g[i, j] + (1 - di) * dj * g[i, j + 1]
                              + di * (1 - dj) * g[i + 1, j] + di * dj * g[i + 1, j + 1])
        return out

    def cloudy(self, layer: int, lat, lon, t: float) -> np.ndarray:
        c = self.cover(layer, lat, lon, t)
        if layer == HIGH:
            c = self.cp.f_cirrus * c
        u = ndtr(self.z[layer](geo.to_ecef(lat, lon, 0.0, self.radius), t))
        return u < c

    def at(self, t: float) -> CloudState:
        mlat, mlon = np.meshgrid(self.mlats, self.mlons, indexing="ij")
        c = self.cp.f_cirrus * self.cover(HIGH, mlat, mlon, t)
        mask = ndtr(self.z[HIGH](self.mxyz, t)) < c
        return CloudState(self, t, mask)


class CloudState:
    """Cloud state at one time step."""

    def __init__(self, field: CloudField, t: float, high_mask: np.ndarray):
        self.f, self.t, self.mask = field, t, high_mask

    def _mask_lookup(self, lat, lon):
        f = self.f
        i = np.clip(np.rint((lat - f.mlats[0]) / f.cp.mask_step).astype(int), 0, len(f.mlats) - 1)
        j = np.clip(np.rint((lon - f.mlons[0]) / f.cp.mask_step).astype(int), 0, len(f.mlons) - 1)
        return self.mask[i, j]

    def air_clear(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Air-to-air rays a->b (n, 3) that do not cross a cloudy high-layer cell."""
        if len(a) == 0:
            return np.zeros(0, dtype=bool)
        s = (np.arange(self.f.cp.ray_samples) + 0.5) / self.f.cp.ray_samples
        p = a[:, None, :] + s[:, None] * (b - a)[:, None, :]
        lat, lon = geo.to_latlon(p)
        return ~self._mask_lookup(lat, lon).any(axis=1)

    def ground_clear(self, g: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Ground (3,) -> aircraft (n, 3) rays that cross no cloudy layer."""
        hg = np.linalg.norm(g) - self.f.radius
        ht = np.linalg.norm(targets, axis=-1) - self.f.radius
        clear = np.ones(len(targets), dtype=bool)
        for layer, h in enumerate(self.f.cp.layer_heights):
            frac = np.clip((h - hg) / np.maximum(ht - hg, 1.0), 0.0, 1.0)
            p = g + frac[:, None] * (targets - g)
            lat, lon = geo.to_latlon(p)
            blocked = (self._mask_lookup(lat, lon) if layer == HIGH
                       else self.f.cloudy(layer, lat, lon, self.t))
            clear &= ~(blocked & (ht > h))
        return clear

    def zenith_clear(self, lat, lon) -> bool:
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        ok = ~self._mask_lookup(lat, lon)
        for layer in (LOW, MID):
            ok &= ~self.f.cloudy(layer, lat, lon, self.t)
        return bool(ok[0])

    def rain_ok(self, lat, lon, rain_max: float) -> bool:
        return bool(self.f.rain(np.atleast_1d(lat), np.atleast_1d(lon), self.t)[0] <= rain_max)
