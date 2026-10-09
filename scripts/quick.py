"""Small sanity-check experiments (a few minutes in total) with figures in figures/quick/.

9 July 2023 only, one snapshot every 30 min, 100 Gbit/s links, K = 2, one ground station
per endpoint.

    python scripts/quick.py elevation   # clear sky, minimum elevation 10 / 5 / 3 deg
    python scripts/quick.py clouds      # 5 deg, ERA5 clouds, f_cirrus x RF fallback
    python scripts/quick.py newyork     # map: airliners around New York, elevation rings
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hap_atlantic import geometry as geo
from hap_atlantic.config import NEW_YORK, scenario
from hap_atlantic.simulate import run, tech_config
from hap_atlantic.traffic import select_fleet

STATES = "data/processed/20230708-10.parquet"
T0 = pd.Timestamp("2023-07-09", tz="UTC").timestamp()
T1 = T0 + 86400.0
EVERY = 30  # minutes
OUT, FIG = Path("results/quick"), Path("figures/quick")
WORKERS = 15

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def _run(p, label, metrics=("ocean", "e2e"), milp_time=10.0):
    tic = time.time()
    r = run(pd.read_parquet(STATES), p, [tech_config("leo", 2, p)], T0, T1, every=EVERY,
            metrics=metrics, workers=WORKERS, milp_time=milp_time)
    r["hour"] = (r["time"] - T0) / 3600.0
    print(f"  {label}: {time.time() - tic:.0f} s, e2e available "
          f"{(r['e2e_leo_k2'] > 0).mean():.0%}, latency path {r['connected'].mean():.0%}",
          flush=True)
    return r.assign(variant=label)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"figures/quick/{name}.png")


def elevation():
    p0 = scenario("leo")
    rows = [_run(replace(p0, geometry=replace(p0.geometry, elev_min_ground=e)), f"{e:g} deg")
            for e in (10.0, 5.0, 3.0)]
    df = pd.concat(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "elevation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    ref = rows[0]
    axes[0].step(ref["hour"], ref["ocean_leo_k2"] / 1e9, where="post", color="0.6", lw=2,
                 label="ocean (independent of elevation)")
    for r, col in zip(rows, ("#d95f02", "#1b9e77", "#7570b3"), strict=True):
        axes[0].step(r["hour"], r["e2e_leo_k2"] / 1e9, where="post", color=col,
                     label=f"Frankfurt–New York, min. elevation {r['variant'].iloc[0]}")
        axes[1].step(r["hour"], r["dst_ground_links"], where="post", color=col)
    axes[0].set_ylabel("capacity [Gbit/s] (100 Gbit/s links, K=2)")
    axes[1].set_ylabel("usable ground links at New York")
    for ax in axes:
        ax.set_xlabel("time [h UTC], 9 Jul 2023")
        ax.set_xlim(0, 24)
    axes[0].legend(fontsize=6)
    _save(fig, "q1_elevation")


def clouds():
    p0 = scenario("leo")
    rows = []
    for f_ci in (0.0, 0.5, 1.0):
        for rf in (0.0, 10e9):
            p = replace(p0, clouds=replace(p0.clouds, model="era5", f_cirrus=f_ci),
                        ground=replace(p0.ground, rf_rate=rf))
            rows.append(_run(p, f"f_ci={f_ci:g}, RF {rf / 1e9:g} Gbit/s").assign(
                f_cirrus=f_ci, rf=rf / 1e9))
    clear = _run(p0, "clear sky")
    df = pd.concat([*rows, clear])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "clouds.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    labels = [r["variant"].iloc[0] for r in rows] + ["clear sky"]
    allr = [*rows, clear]
    x = np.arange(len(allr))
    axes[0].bar(x - 0.2, [(r["ocean_leo_k2"] > 0).mean() for r in allr], 0.4,
                label="ocean chain exists")
    axes[0].bar(x + 0.2, [(r["e2e_leo_k2"] > 0).mean() for r in allr], 0.4,
                label="Frankfurt–New York connected")
    axes[0].set_ylabel("share of the day")
    axes[0].legend(fontsize=6)
    axes[1].bar(x - 0.2, [r["ocean_leo_k2"].mean() / 1e9 for r in allr], 0.4, label="ocean")
    axes[1].bar(x + 0.2, [r["e2e_leo_k2"].mean() / 1e9 for r in allr], 0.4,
                label="Frankfurt–New York")
    axes[1].set_ylabel("mean capacity [Gbit/s]")
    axes[1].legend(fontsize=6)
    for ax in axes[:2]:
        ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=6)
    for r in allr:
        d = r.loc[r["connected"], "min_delay"] * 1e3
        axes[2].plot(np.sort(d), np.arange(1, len(d) + 1) / len(r), label=r["variant"].iloc[0])
    axes[2].set_xlabel("one-way latency [ms]")
    axes[2].set_ylabel("share of the day with latency ≤ x")
    axes[2].legend(fontsize=5)
    _save(fig, "q2_clouds")


def newyork():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t = T0 + 14 * 3600
    st = pd.read_parquet(STATES)
    snap = select_fleet(st[st.time == t], True, 1.0, 0)
    R = 6_371_008.8
    g = geo.to_ecef(NEW_YORK.lat, NEW_YORK.lon, NEW_YORK.alt, R)
    pos = geo.to_ecef(snap.lat.to_numpy(), snap.lon.to_numpy(), snap.alt.to_numpy(), R)
    el = geo.elevation_deg(np.broadcast_to(g, pos.shape), pos)
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(-73, 41))
    ax.set_extent([-78.5, -69, 38.5, 43.5], crs=pc)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, color="0.5")
    sc = ax.scatter(snap.lon, snap.lat, c=snap.alt / 1e3, s=12, cmap="viridis", vmin=0,
                    vmax=12, transform=pc, zorder=3)
    hi = (snap.alt.to_numpy() >= 7000)
    for e, ls in ((10, "-"), (5, "--"), (3, ":")):
        vis = (el >= e) & hi
        ax.scatter(snap.lon[vis], snap.lat[vis], s=60, facecolor="none", edgecolor="r",
                   lw=0.6 if e != 10 else 1.2, transform=pc, zorder=4)
        # ring: horizontal distance at which an aircraft at 10.5 km has elevation e
        d = next(dk for dk in np.arange(1e3, 400e3, 1e3)
                 if geo.elevation_deg(g, geo.to_ecef(*geo.destination(
                     NEW_YORK.lat, NEW_YORK.lon, 0.0, dk, R), 10.5e3, R)) <= e)
        brg = np.arange(0, 361, 3)
        la, lo = geo.destination(NEW_YORK.lat, NEW_YORK.lon, brg, d, R)
        ax.plot(lo, la, ls=ls, color="r", lw=0.8, transform=pc,
                label=f"{e}° elevation at 10.5 km ({d / 1e3:.0f} km)")
    ax.plot(NEW_YORK.lon, NEW_YORK.lat, "r*", ms=10, transform=pc)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="altitude [km]")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Airliners around NY4, 9 Jul 2023 14:00 UTC (red circles: usable, ≥7 km)",
                 fontsize=8)
    _save(fig, "q3_newyork")


EXPERIMENTS = {"elevation": elevation, "clouds": clouds, "newyork": newyork}

if __name__ == "__main__":
    for name in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[name]()
