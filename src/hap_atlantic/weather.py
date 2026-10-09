"""Cloud attenuation of optical links from hourly 3D ERA5 cloud fields.

Data: ERA5 (Hersbach et al. 2020) on 0.25 deg and 37 pressure levels from the public,
analysis-ready ARCO-ERA5 store on Google Cloud (Carver & Merose 2023; no account needed):
fraction of cloud cover cc, specific cloud ice / liquid water content q_i / q_l, and the
hourly total precipitation (for the RF fallback). Levels 1000-150 hPa (0.1-13.6 km in the
standard atmosphere) over the corridor are cached in data/weather as one .npz per period.

Model per time step t:

* cloud occurrence: a point x (lat, lon, height) is inside a cloud if
  Phi(Z(x, t)) < cc(x, t), where Z is a standard Gaussian field with covariance
  exp(-dh^2 / 2 L^2) exp(-dz^2 / 2 L_z^2) exp(-dt^2 / 2 T^2) (random Fourier features,
  deterministic given the seed). The cloudy volume therefore matches ERA5, nearby points
  and consecutive time steps are correlated, and the vertical decorrelation length L_z
  gives the usual exponential-random overlap of cloud layers.
* in-cloud extinction: ice (Fu 1996) beta = IWC (a0 + a1 / D_e), IWC in g/m^3, D_e in um;
  liquid (geometric optics) beta = 3 LWC / (2 rho_w r_e). In-cloud water contents are the
  grid-box contents divided by cc.
* a ray loses 10 log10(e) * integral(beta ds) dB; the loss enters the FSO link budget.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import geometry as geo
from .config import NORTH_ATLANTIC_BBOX, CloudParams

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CACHE = Path("data/weather")
P_TOP = 150  # highest level used [hPa]
FU_A0, FU_A1 = -6.656e-3, 3.686  # Fu (1996), beta [1/m] = IWC [g/m^3] (a0 + a1 / De [um])
RHO_W = 1000.0  # [kg/m^3]
R_DRY = 287.05  # [J/(kg K)]
DB = 10 / np.log(10)


def isa_height(p_hpa):
    """Standard-atmosphere height [m] of a pressure level."""
    return 44_330.8 * (1 - (np.asarray(p_hpa, float) / 1013.25) ** 0.190263)


def isa_density(p_hpa):
    h = isa_height(p_hpa)
    temp = np.maximum(288.15 - 0.0065 * h, 216.65)
    return np.asarray(p_hpa, float) * 100 / (R_DRY * temp)


def fetch_arco(t_start: float, t_end: float, bbox=NORTH_ATLANTIC_BBOX, cache: Path = CACHE,
               threads: int = 8) -> Path:
    """Download the hourly 3D cloud fields covering [t_start, t_end] [s, UNIX] into one .npz."""
    h0, h1 = np.floor(t_start / 3600) * 3600, np.ceil(t_end / 3600) * 3600
    start = pd.Timestamp(h0, unit="s").strftime("%Y%m%dT%H")
    end = pd.Timestamp(h1, unit="s").strftime("%Y%m%dT%H")
    f = cache / f"arco_{start}_{end}.npz"
    if f.exists():
        return f
    import xarray as xr

    ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    lamin, lamax, lomin, lomax = bbox
    times = pd.date_range(pd.Timestamp(h0, unit="s"), pd.Timestamp(h1, unit="s"), freq="h").values
    it = np.searchsorted(ds.time.values, times)
    lat = ds.latitude.values
    ilat = np.flatnonzero((lat >= lamin) & (lat <= lamax))[::-1]  # ascending latitude
    lon = (ds.longitude.values + 180) % 360 - 180
    ilon = np.flatnonzero((lon >= lomin) & (lon <= lomax))
    ilon = ilon[np.argsort(lon[ilon])]
    lev = ds.level.values
    ilev = np.flatnonzero(lev >= P_TOP)[::-1]  # 1000 hPa first = ascending height

    def one(i):
        out = {}
        for v in ("fraction_of_cloud_cover", "specific_cloud_ice_water_content",
                  "specific_cloud_liquid_water_content"):
            a = ds[v].isel(time=int(i)).values  # one chunk = all levels, whole globe
            out[v] = a[np.ix_(ilev, ilat, ilon)]
        tp = ds["total_precipitation"].isel(time=int(i)).values
        out["tp"] = tp[np.ix_(ilat, ilon)]
        return out

    with ThreadPoolExecutor(threads) as ex:
        hours = list(ex.map(one, it))
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f,
        hours=times.astype("datetime64[s]").astype(float),
        levels=lev[ilev].astype(float), lats=lat[ilat], lons=lon[ilon],
        cc=np.stack([h["fraction_of_cloud_cover"] for h in hours]).astype(np.float16),
        qi=np.stack([h["specific_cloud_ice_water_content"] for h in hours]).astype(np.float32),
        ql=np.stack([h["specific_cloud_liquid_water_content"] for h in hours]).astype(np.float32),
        tp=np.stack([h["tp"] for h in hours]).astype(np.float32) * 1e3,  # m -> mm per hour
    )
    return f


class LatentField:
    """Standard Gaussian space-time field via random Fourier features (deterministic).

    Anisotropic: horizontal length l_corr, vertical length lz_corr, time t_corr.
    """

    def __init__(self, l_corr: float, lz_corr: float, t_corr: float, n_features: int,
                 seed: int, radius: float = 6_371_008.8):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal((3, n_features)) / l_corr
        self.wz = rng.standard_normal(n_features) / lz_corr
        self.nu = rng.standard_normal(n_features) / t_corr
        self.phi = rng.uniform(0, 2 * np.pi, n_features)
        self.scale = np.sqrt(2.0 / n_features)
        self.radius = radius

    def __call__(self, xyz: np.ndarray, t: float, z=0.0, chunk: int = 20_000) -> np.ndarray:
        """Z at surface points xyz (n, 3) [m] (horizontal position), heights z [m], time t."""
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        z = np.broadcast_to(np.asarray(z, float), (len(xyz),))
        out = np.empty(len(xyz))
        for s in range(0, len(xyz), chunk):
            arg = (xyz[s: s + chunk] @ self.w + np.outer(z[s: s + chunk], self.wz)
                   + (self.nu * t + self.phi))
            out[s: s + chunk] = self.scale * np.cos(arg).sum(axis=-1)
        return out


class CloudField:
    """3D ERA5 cloud fields + latent occurrence field; `at(t)` gives the state at time t."""

    def __init__(self, t_start: float, t_end: float, cp: CloudParams, seed: int,
                 radius: float):
        d = np.load(fetch_arco(t_start, t_end))
        self.hours, self.lats, self.lons = d["hours"], d["lats"], d["lons"]
        self.levels = d["levels"]
        self.heights = isa_height(self.levels)
        self.cc = d["cc"]
        # in-cloud extinction [1/m], hour by hour to keep the memory small (~70 MB per day)
        rho = isa_density(self.levels)[:, None, None]
        qi, ql = d["qi"], d["ql"]
        self.beta = np.empty(self.cc.shape, dtype=np.float16)
        for k in range(len(self.hours)):
            cc = np.maximum(self.cc[k].astype(np.float32), 0.01)
            iwc = qi[k] * rho / cc * 1e3  # in-cloud [g/m^3]
            lwc = ql[k] * rho / cc  # in-cloud [kg/m^3]
            self.beta[k] = (iwc * (FU_A0 + FU_A1 / (cp.de_ice * 1e6))
                            + 3 * lwc / (2 * RHO_W * cp.re_liquid))
        del qi, ql
        self.tp = d["tp"]
        self.cp, self.radius = cp, radius
        self.z = LatentField(cp.l_corr, cp.lz_corr, cp.t_corr, cp.n_features, 1000 + seed,
                             radius)
        glat, glon = np.meshgrid(self.lats, self.lons, indexing="ij")
        self.gxyz = geo.to_ecef(glat, glon, 0.0, radius)  # (ny, nx, 3)

    def _time(self, t):
        k = int(np.clip(np.searchsorted(self.hours, t, side="right") - 1, 0,
                        len(self.hours) - 2))
        f = float(np.clip((t - self.hours[k]) / (self.hours[k + 1] - self.hours[k]), 0, 1))
        return k, f

    def at(self, t: float) -> CloudState:
        k, f = self._time(t)
        cc = (1 - f) * self.cc[k].astype(np.float32) + f * self.cc[k + 1].astype(np.float32)
        b0, b1 = self.beta[k].astype(np.float32), self.beta[k + 1].astype(np.float32)
        beta = (1 - f) * b0 + f * b1
        ext = np.zeros_like(beta)
        sel = np.nonzero(cc > 1e-3)  # latent field only where clouds are possible
        if sel[0].size:
            xyz = self.gxyz[sel[1], sel[2]]
            u = ndtr(self.z(xyz, t, self.heights[sel[0]]))
            inside = u < cc[sel]
            ext[sel[0][inside], sel[1][inside], sel[2][inside]] = beta[sel][inside]
        rain = (1 - f) * self.tp[k] + f * self.tp[k + 1]
        return CloudState(self, ext, rain)


class CloudState:
    """Cloud extinction field (level, lat, lon) [1/m] at one time step."""

    def __init__(self, field: CloudField, ext: np.ndarray, rain: np.ndarray):
        self.f, self.ext, self.rain_field = field, ext, rain
        h = field.heights
        self._hedges = (h[1:] + h[:-1]) / 2

    def _lookup(self, lat, lon, h):
        f = self.f
        i = np.clip(np.rint((lat - f.lats[0]) / 0.25).astype(int), 0, len(f.lats) - 1)
        j = np.clip(np.rint((lon - f.lons[0]) / 0.25).astype(int), 0, len(f.lons) - 1)
        lev = np.searchsorted(self._hedges, h)
        return np.where(h <= f.heights[-1] + 500, self.ext[lev, i, j], 0.0)

    def ray_loss_db(self, a: np.ndarray, b: np.ndarray, n: int | None = None) -> np.ndarray:
        """Cloud loss [dB] of straight rays a -> b (m, 3), positions incl. radius + height."""
        if len(a) == 0:
            return np.zeros(0)
        n = n or self.f.cp.ray_samples
        out = np.empty(len(a))
        s = (np.arange(n) + 0.5) / n
        step = max(1, 100_000 // n)
        for c in range(0, len(a), step):
            aa, bb = a[c: c + step], b[c: c + step]
            p = aa[:, None, :] + s[:, None] * (bb - aa)[:, None, :]
            lat, lon = geo.to_latlon(p)
            h = np.linalg.norm(p, axis=-1) - self.f.radius
            seg = np.linalg.norm(bb - aa, axis=-1) / n
            out[c: c + step] = DB * self._lookup(lat, lon, h).sum(axis=1) * seg
        return out

    def zenith_loss_db(self, lat, lon) -> float:
        """Cloud loss of a vertical ray from the ground to 13 km."""
        R = self.f.radius
        a = geo.to_ecef(lat, lon, 0.0, R)[None]
        b = geo.to_ecef(lat, lon, 13e3, R)[None]
        return float(self.ray_loss_db(a, b, n=64)[0])

    def rain_ok(self, lat, lon, rain_max: float) -> bool:
        f = self.f
        i = int(np.clip(np.rint((lat - f.lats[0]) / 0.25), 0, len(f.lats) - 1))
        j = int(np.clip(np.rint((lon - f.lons[0]) / 0.25), 0, len(f.lons) - 1))
        return bool(self.rain_field[i, j] <= rain_max)
