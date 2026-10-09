"""Mini sanity-check experiments (seconds each) with figures in figures/quick/.

9 July 2023, only 4 snapshots (04, 09, 14, 19 UTC), always with ERA5 clouds, 100 Gbit/s
links, K = 2, minimum elevation 5 deg, minimum air-to-air ray height 3 km.

* capacity: ground stations everywhere on land, Europe -> North America
* latency:  Frankfurt -> New York (NY4), one station per endpoint, RF fallback

    python scripts/quick.py overview  # capacity and latency per snapshot
    python scripts/quick.py clouds    # RF fallback on/off
    python scripts/quick.py tech      # rate per link x terminals per aircraft
    python scripts/quick.py newyork   # map: airliners around New York, elevation rings
    python scripts/quick.py map       # map of one snapshot

Results are stored in results/quick/<name>.csv; `--plot` only redraws the figures from
these files (no simulation):

    python scripts/quick.py --plot overview clouds tech
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
DAY = pd.Timestamp("2023-07-09", tz="UTC").timestamp()
T0, T1 = DAY + 4 * 3600.0, DAY + 86400.0
EVERY = 300  # minutes -> 04, 09, 14, 19 UTC
OUT, FIG = Path("results/quick"), Path("figures/quick")
WORKERS = 4

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 200})
_CACHE = {}


def _states():
    """The 4 snapshots only (read once)."""
    if "s" not in _CACHE:
        st = pd.read_parquet(STATES, filters=[("time", ">=", T0), ("time", "<", T1)])
        times = np.sort(st["time"].unique())[::EVERY]
        _CACHE["s"] = st[st["time"].isin(times)]
    return _CACHE["s"]


def _run(p, label, configs=None, milp_time=2.0):
    tic = time.time()
    configs = configs or [tech_config("leo", 2, p)]
    r = run(_states(), p, configs, T0, T1, every=1, workers=WORKERS, milp_time=milp_time)
    r["hour"] = (r["time"] - DAY) / 3600.0
    c = f"cap_{configs[0].label}"
    gap = 1 - r[c].sum() / r[f"{c}_bound"].sum()
    print(f"  {label}: {time.time() - tic:.0f} s, capacity {r[c].mean() / 1e9:.0f} Gbit/s "
          f"(min {r[c].min() / 1e9:.0f}, gap to bound {gap:.0%}), Frankfurt-New York connected "
          f"{r['connected'].sum()}/{len(r)}", flush=True)
    return r.assign(variant=label)


def _out(name):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}.csv"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"figures/quick/{name}.png")


def _latency(ax, rows):
    ref = benchmarks()
    for i, r in enumerate(rows):
        ax.plot(r["hour"] + 0.3 * i, r["min_delay"] * 1e3, "o", label=r["variant"].iloc[0])
    for lab, v in (("vacuum", "vacuum"), ("HF (scaled)", "hf_scaled"),
                   ("fibre + microwave", "fibre_microwave_published")):
        ax.axhline(ref[v] * 1e3, color="k", lw=0.6, ls=":")
        ax.text(3.6, ref[v] * 1e3 + 0.15, lab, fontsize=6)
    ax.set_xlabel("time [h UTC] (missing dot = no path)")
    ax.set_ylabel("Frankfurt–New York one-way latency [ms]")
    ax.set_ylim(20, 33)


def overview():
    _run(scenario("leo"), "ERA5").to_csv(_out("overview"), index=False)


def plot_overview():
    r = pd.read_csv(_out("overview"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    axes[0].bar(r["hour"], r["cap_leo_k2"] / 1e9, width=2)
    axes[0].set_ylabel("capacity Europe → N. America [Gbit/s]")
    axes[1].bar(r["hour"] - 0.5, r["eu_access"], width=1, label="aircraft with access, Europe")
    axes[1].bar(r["hour"] + 0.5, r["na_access"], width=1, label="… North America")
    axes[1].set_ylabel("aircraft with land access")
    axes[1].legend(fontsize=6)
    for ax in axes[:2]:
        ax.set_xlabel("time [h UTC]")
        ax.set_xticks(r["hour"])
    _latency(axes[2], [r])
    fig.suptitle("9 Jul 2023, ERA5 clouds, 100 Gbit/s links, K=2", fontsize=8)
    _save(fig, "q1_overview")


def clouds():
    p = scenario("leo")
    rows = [_run(p, "RF fallback (default)"),
            _run(replace(p, ground=replace(p.ground, rf_rate=0.0)), "optical only")]
    pd.concat(rows).to_csv(_out("clouds"), index=False)


def plot_clouds():
    df = pd.read_csv(_out("clouds"))
    df = df[~df["variant"].str.startswith("ice")]  # crystal-size test dropped
    rows = [g for _, g in df.groupby("variant", sort=False)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    x = np.arange(len(rows))
    labels = [r["variant"].iloc[0] for r in rows]
    axes[0].bar(x, [r["cap_leo_k2"].mean() / 1e9 for r in rows])
    axes[0].set_ylabel("mean capacity Europe → N. America [Gbit/s]")
    axes[1].bar(x, [r["connected"].mean() for r in rows])
    axes[1].set_ylabel("share of snapshots: Frankfurt–NY path")
    axes[1].set_ylim(0, 1.05)
    for ax in axes[:2]:
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=7)
    _latency(axes[2], rows)
    axes[2].legend(fontsize=6)
    _save(fig, "q2_clouds")


def tech():
    p = scenario("leo")
    names = ("leo", "dwdm")
    configs = [tech_config(t, k, p) for t in names for k in (2, 4)]
    _run(p, "ERA5", configs=configs).to_csv(_out("tech"), index=False)


def plot_tech():
    r = pd.read_csv(_out("tech"))
    names = ("leo", "dwdm")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for k, col in ((2, "#1b9e77"), (4, "#d95f02")):
        rates = [TECHS[t].r_max for t in names]
        caps = [r[f"cap_{t}_k{k}"].mean() for t in names]
        chains = caps[0] / TECHS["leo"].r_max
        ax.loglog([TECHS["demo"].r_max, 1e11], [chains * TECHS["demo"].r_max, chains * 1e11],
                  color=col, ls="--")
        ax.loglog(rates, caps, "o-", color=col, label=f"K={k}")
    ax.axhline(250e12, color="k", lw=0.6, ls=":")
    ax.text(3e9, 280e12, "one cable (250 Tbit/s)", fontsize=6)
    ax.axvspan(1e9, 2.6e9, color="0.85")
    ax.text(1.1e9, 3e9, "demon-\nstrated", fontsize=6)
    ax.set_xlabel("rate per link R_max [bit/s]")
    ax.set_ylabel("mean capacity Europe → N. America [bit/s]")
    ax.legend(fontsize=6)
    _save(fig, "q4_tech")


def newyork():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t = DAY + 14 * 3600
    st = pd.read_parquet(STATES, filters=[("time", "==", t)])
    snap = select_fleet(st, True, 1.0, 0)
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


EXPERIMENTS = {"overview": overview, "clouds": clouds, "tech": tech, "newyork": newyork,
               "map": snapshot_map}
PLOTS = {"overview": plot_overview, "clouds": plot_clouds, "tech": plot_tech}

if __name__ == "__main__":
    args = sys.argv[1:]
    plot_only = "--plot" in args
    names = [a for a in args if a != "--plot"] or list(PLOTS if plot_only else EXPERIMENTS)
    for name in names:
        if not plot_only:
            EXPERIMENTS[name]()
        if name in PLOTS:
            PLOTS[name]()
