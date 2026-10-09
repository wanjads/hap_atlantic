"""Small sanity-check experiments (a few minutes each) with figures in figures/quick/.

9 July 2023 only, one snapshot every 30 min, 100 Gbit/s links, K = 2, one ground station
per endpoint, minimum elevation 5 deg, minimum air-to-air ray height 3 km.

    python scripts/quick.py hmin      # min. ray height 2/3/5/7 km, clear sky vs. ERA5 clouds
    python scripts/quick.py clouds    # ERA5 clouds: seeds, RF fallback, ice crystal size
    python scripts/quick.py tech      # ERA5 clouds: rate per link x terminals per aircraft
    python scripts/quick.py newyork   # map: airliners around New York, elevation rings
    python scripts/quick.py map       # map of one snapshot with cloud-blocked links
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
from hap_atlantic.config import NEW_YORK, TECHS, scenario
from hap_atlantic.reference import benchmarks
from hap_atlantic.simulate import run, tech_config
from hap_atlantic.traffic import select_fleet

STATES = "data/processed/20230708-10.parquet"
T0 = pd.Timestamp("2023-07-09", tz="UTC").timestamp()
T1 = T0 + 86400.0
EVERY = 30  # minutes
OUT, FIG = Path("results/quick"), Path("figures/quick")
WORKERS = 15
ERA5 = {"model": "era5"}

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def _p(h_min=3e3, clouds=False, **kw):
    p = scenario("leo")
    p = replace(p, geometry=replace(p.geometry, h_chord_min=h_min))
    if clouds:
        p = replace(p, clouds=replace(p.clouds, model="era5"))
    return replace(p, **kw)


def _run(p, label, configs=None, metrics=("ocean", "e2e"), milp_time=10.0):
    tic = time.time()
    configs = configs or [tech_config("leo", 2, p)]
    r = run(pd.read_parquet(STATES), p, configs, T0, T1, every=EVERY, metrics=metrics,
            workers=WORKERS, milp_time=milp_time)
    r["hour"] = (r["time"] - T0) / 3600.0
    c = configs[0].label
    print(f"  {label}: {time.time() - tic:.0f} s, ocean {(r[f'ocean_{c}'] > 0).mean():.0%}, "
          f"e2e {(r[f'e2e_{c}'] > 0).mean():.0%}, mean e2e {r[f'e2e_{c}'].mean() / 1e9:.0f} "
          f"Gbit/s", flush=True)
    return r.assign(variant=label)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"figures/quick/{name}.png")


def _bars(ax, rows, col, fn, ylabel):
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [fn(r[f"ocean_{col}"]) for r in rows], 0.4, label="ocean")
    ax.bar(x + 0.2, [fn(r[f"e2e_{col}"]) for r in rows], 0.4, label="Frankfurt–New York")
    ax.set_xticks(x, [r["variant"].iloc[0] for r in rows], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6)


def hmin():
    rows = []
    for clouds in (False, True):
        for h in (2e3, 3e3, 5e3, 7e3):
            rows.append(_run(_p(h, clouds), f"{h / 1e3:g} km, {'ERA5' if clouds else 'clear'}"))
    pd.concat(rows).to_csv(_out("hmin"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    _bars(axes[0], rows, "leo_k2", lambda s: (s > 0).mean(), "share of the day connected")
    _bars(axes[1], rows, "leo_k2", lambda s: s.mean() / 1e9, "mean capacity [Gbit/s]")
    axes[2].bar(range(len(rows)), [r["n_air_links"].mean() for r in rows])
    axes[2].set_xticks(range(len(rows)), [r["variant"].iloc[0] for r in rows], rotation=35,
                       ha="right", fontsize=6)
    axes[2].set_ylabel("usable air-to-air links (after Yao)")
    fig.suptitle("min. height of air-to-air rays; 100 Gbit/s links, K=2", fontsize=8)
    _save(fig, "q1_hmin")


def clouds():
    rows = [_run(_p(), "clear sky")]
    for seed in (0, 1, 2):
        rows.append(_run(_p(clouds=True, seed=seed), f"ERA5 seed {seed}"))
    p = _p(clouds=True)
    rows.append(_run(replace(p, ground=replace(p.ground, rf_rate=10e9)), "ERA5 + RF 10 Gbit/s"))
    for de in (25e-6, 100e-6):
        rows.append(_run(replace(p, clouds=replace(p.clouds, de_ice=de)),
                         f"ERA5, ice D_e={de * 1e6:g} um"))
    pd.concat(rows).to_csv(_out("clouds"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    _bars(axes[0], rows, "leo_k2", lambda s: (s > 0).mean(), "share of the day connected")
    _bars(axes[1], rows, "leo_k2", lambda s: s.mean() / 1e9, "mean capacity [Gbit/s]")
    ref = benchmarks()
    for r in rows[:5]:
        d = np.sort(r.loc[r["connected"], "min_delay"] * 1e3)
        axes[2].step(d, np.arange(1, len(d) + 1) / len(r), where="post",
                     label=r["variant"].iloc[0])
    for lab, v in (("vacuum", "vacuum"), ("HF", "hf_scaled"), ("fibre+MW",
                                                                "fibre_microwave_published")):
        axes[2].axvline(ref[v] * 1e3, color="k", lw=0.6, ls=":")
        axes[2].text(ref[v] * 1e3 + 0.1, 0.02, lab, fontsize=6, rotation=90)
    axes[2].set_xlabel("one-way latency [ms]")
    axes[2].set_ylabel("share of the day with latency ≤ x")
    axes[2].set_xlim(20.5, 32.5)
    axes[2].legend(fontsize=5)
    _save(fig, "q2_clouds")


def tech():
    p = _p(clouds=True)
    configs = [tech_config(t, k, p) for t in ("leo", "dwdm", "visionary") for k in (2, 4, 6)]
    r = _run(p, "ERA5 seed 0", configs=configs)
    r.to_csv(_out("tech"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, m in zip(axes, ("ocean", "e2e"), strict=True):
        for k, col in ((2, "#1b9e77"), (4, "#d95f02"), (6, "#7570b3")):
            rates = [TECHS[t].r_max for t in ("leo", "dwdm", "visionary")]
            caps = [r[f"{m}_{t}_k{k}"].mean() for t in ("leo", "dwdm", "visionary")]
            chains = r[f"{m}_leo_k{k}"].mean() / TECHS["leo"].r_max
            ax.loglog([TECHS["demo"].r_max, 1e11], [chains * TECHS["demo"].r_max, chains * 1e11],
                      color=col, ls="--")
            ax.loglog(rates, caps, "o-", color=col, label=f"K={k}")
        ax.axhline(250e12, color="k", lw=0.6, ls=":")
        ax.text(3e9, 280e12, "one cable (250 Tbit/s)", fontsize=6)
        ax.axvspan(1e9, 2.6e9, color="0.85")
        ax.set_title({"ocean": "oceanic section", "e2e": "Frankfurt–New York"}[m], fontsize=8)
        ax.set_xlabel("rate per link R_max [bit/s]")
    axes[0].set_ylabel("mean capacity [bit/s], ERA5 clouds")
    axes[0].legend(fontsize=6)
    _save(fig, "q4_tech")


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
    vis = (el >= 5) & (snap.alt.to_numpy() >= 3000)
    ax.scatter(snap.lon[vis], snap.lat[vis], s=60, facecolor="none", edgecolor="r", lw=0.8,
               transform=pc, zorder=4, label="visible at ≥5° and above 3 km")
    for e, ls in ((10, "-"), (5, "--")):
        d = next(dk for dk in np.arange(1e3, 400e3, 1e3)
                 if geo.elevation_deg(g, geo.to_ecef(*geo.destination(
                     NEW_YORK.lat, NEW_YORK.lon, 0.0, dk, R), 10.5e3, R)) <= e)
        la, lo = geo.destination(NEW_YORK.lat, NEW_YORK.lon, np.arange(0, 361, 3), d, R)
        ax.plot(lo, la, ls=ls, color="r", lw=0.8, transform=pc,
                label=f"{e}° elevation at 10.5 km ({d / 1e3:.0f} km)")
    ax.plot(NEW_YORK.lon, NEW_YORK.lat, "r*", ms=10, transform=pc)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="altitude [km]")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Airliners around NY4, 9 Jul 2023 14:00 UTC", fontsize=8)
    _save(fig, "q3_newyork")


def snapshot_map():
    sys.path.insert(0, str(Path(__file__).parent))
    import figures

    figures.FIG = FIG
    figures.fig_map()


def _out(name):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}.csv"


EXPERIMENTS = {"hmin": hmin, "clouds": clouds, "tech": tech, "newyork": newyork,
               "map": snapshot_map}

if __name__ == "__main__":
    for name in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[name]()
