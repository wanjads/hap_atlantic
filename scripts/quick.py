"""Experiments of the paper (seconds to minutes each); figures in figures/quick/.

Two days: 9 July 2023 (summer, all-time Shanwick record) and 16 January 2024 (winter
weekday). One snapshot per hour (24 per day), always with ERA5 clouds and RF fallback of
the ground links, minimum elevation 5 deg, minimum air-to-air ray height 3 km;
capacities within 1 % of the upper bound (MILP limit 30 s; the bound is plotted).

* capacity: ground stations everywhere on land, Europe -> North America
* latency:  Frankfurt -> New York (NY4), stations at the endpoints and on rings of 75 and
            150 km around them (land points only)

    python scripts/quick.py overview   # capacity and latency per hour (100 Gbit/s, K=2)
    python scripts/quick.py tech       # rate per link 1 Gbit/s - 1 Tbit/s x K = 2, 4
    python scripts/quick.py map        # world map of one snapshot (12 UTC)
    python scripts/quick.py newyork    # airliners and ground stations around New York
    python scripts/quick.py frankfurt  # the same around Frankfurt
    python scripts/quick.py --day winter overview      # only one day (default: both)
    python scripts/quick.py --plot overview tech       # redraw from results/quick/*.csv

Results are stored in results/quick/<name>_<day>.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hap_atlantic import flow
from hap_atlantic import geometry as geo
from hap_atlantic.config import FRANKFURT, NEW_YORK, scenario
from hap_atlantic.reference import benchmarks
from hap_atlantic.simulate import Config, run
from hap_atlantic.topology import build_graph, ground_stations
from hap_atlantic.traffic import select_fleet
from hap_atlantic.weather import CloudField

DAYS = {
    "summer": ("data/processed/20230708-10.parquet", "2023-07-09", "9 Jul 2023"),
    "winter": ("data/processed/20240115-17.parquet", "2024-01-16", "16 Jan 2024"),
}
COLORS = {"summer": "#d95f02", "winter": "#1f78b4"}
OUT, FIG = Path("results/quick"), Path("figures/quick")
WORKERS = 4
RATES = (1e9, 1e10, 1e11, 1e12)  # bit/s per link; 1 W up to 100 Gbit/s, 5 W at 1 Tbit/s

# Times, as in the paper (IEEEtran); STIX is the Times clone shipped with matplotlib
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 200})
CABLE_ONE = 250e12  # Dunant (Dunant2021)
CABLE_ALL = 1.2e15  # potential capacity of all trans-Atlantic cables (TeleGeography, TODO verify)


def _t0(day):
    return pd.Timestamp(DAYS[day][1], tz="UTC").timestamp()


def _states(day):
    """One snapshot per hour of the day."""
    t0 = _t0(day)
    st = pd.read_parquet(DAYS[day][0], filters=[("time", ">=", t0), ("time", "<", t0 + 86400)])
    return st[(st["time"] - t0) % 3600 == 0]


def _config(r_max, k, p):
    return Config(f"r{r_max:.0e}_k{k}", r_max, (5.0 if r_max > 1e11 else 1.0) / p.terminal.p_tx,
                  k)


def _run(day, configs, milp_time=30.0):
    tic = time.time()
    t0 = _t0(day)
    r = run(_states(day), scenario("leo"), configs, t0, t0 + 86400, every=1, workers=WORKERS,
            milp_time=milp_time)
    r["hour"] = (r["time"] - t0) / 3600.0
    c = f"cap_{configs[0].label}"
    gap = 1 - r[c].sum() / r[f"{c}_bound"].sum()
    print(f"  {day}: {time.time() - tic:.0f} s, capacity {r[c].mean() / 1e9:.0f} Gbit/s "
          f"(gap to bound {gap:.1%}), Frankfurt-New York connected "
          f"{r['connected'].sum()}/{len(r)}", flush=True)
    return r.assign(day=day)


def _out(name, day):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}_{day}.csv"


def _read(name, days):
    return {d: pd.read_csv(_out(name, d)) for d in days if _out(name, d).exists()}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"figures/quick/{name}.png")


def _time_axis(ax):
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("time (UTC)")


# --- capacity and latency over the day ---------------------------------------------------


def overview(day):
    p = scenario("leo")
    _run(day, [_config(1e11, 2, p)]).to_csv(_out("overview", day), index=False)


def plot_overview(days):
    res = _read("overview", days)
    col = "cap_r1e+11_k2"
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    w = 0.8 / max(len(res), 1)
    for i, (d, r) in enumerate(res.items()):
        x = r["hour"] + (i - (len(res) - 1) / 2) * w
        ax.bar(x, r[col] / 1e9, width=w, color=COLORS[d], label=DAYS[d][2])
        ax.plot(x, r[f"{col}_bound"] / 1e9, "k_", ms=4, mew=1.2)
    ax.plot([], [], "k_", ms=6, mew=1.2, label="upper bound")
    ax.set_ylabel("capacity Europe → N. America [Gbit/s]")
    _time_axis(ax)
    ax.legend(fontsize=6)
    _save(fig, "q1_capacity")

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ref = benchmarks()
    for i, (d, r) in enumerate(res.items()):
        ax.plot(r["hour"] + 0.2 * i, r["min_delay"] * 1e3, "o", ms=3, color=COLORS[d],
                label=DAYS[d][2])
    for lab, v in (("vacuum", "vacuum"), ("HF (scaled)", "hf_scaled"),
                   ("fibre + microwave", "fibre_microwave_published")):
        ax.axhline(ref[v] * 1e3, color="k", lw=0.6, ls=":")
        ax.text(0.2, ref[v] * 1e3 + 0.15, lab, fontsize=6)
    _time_axis(ax)
    ax.set_ylabel("Frankfurt–New York one-way latency [ms]")
    ax.set_ylim(20, 33)
    ax.legend(fontsize=6, loc="center right")
    _save(fig, "q2_latency")


# --- rate per link x terminals ------------------------------------------------------------


def tech(day):
    p = scenario("leo")
    configs = [_config(r, k, p) for k in (2, 4) for r in RATES]
    _run(day, configs).to_csv(_out("tech", day), index=False)


def plot_tech(days):
    res = _read("tech", days)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for d, r in res.items():
        for k, ls, mk in ((2, "-", "o"), (4, "--", "s")):
            cap = [r[f"cap_r{x:.0e}_k{k}"].mean() for x in RATES]
            ax.loglog(RATES, cap, ls=ls, marker=mk, ms=4, color=COLORS[d],
                      label=f"{DAYS[d][2]}, $K={k}$")
    for v, lab in ((CABLE_ONE, "one cable (250 Tbit/s)"),
                   (CABLE_ALL, "all trans-Atlantic cables (≈1.2 Pbit/s)")):
        ax.axhline(v, color="k", lw=0.6, ls=":")
        ax.text(2.9e9, v * 1.15, lab, fontsize=6)
    ax.axvspan(9e8, 2.6e9, color="0.85")
    ax.text(1.0e9, 4e13, "demon-\nstrated", fontsize=6)
    ax.set_xlim(7e8, 1.5e12)
    ax.set_ylim(2e9, 3e15)
    ax.set_xlabel(r"rate per link $R_\mathrm{max}$ [bit/s]")
    ax.set_ylabel("mean capacity Europe → N. America [bit/s]")
    ax.legend(fontsize=6, loc="lower right")
    _save(fig, "q4_tech")


# --- maps --------------------------------------------------------------------------------


def snapshot_map(day, hour=12):
    """Airliners (dots), cloud at cruise level (8-12.5 km), min-latency path Frankfurt-NY."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t0 = _t0(day)
    t = t0 + hour * 3600
    p = scenario("leo")
    snap = select_fleet(pd.read_parquet(DAYS[day][0], filters=[("time", "==", t)]))
    field = CloudField(t0 - 3600, t0 + 86400 + 3600, p.clouds, p.seed, p.geometry.earth_radius)
    cs = field.at(t)
    G = build_graph(snap, p, cs)
    delay, path = flow.min_latency_path(G)
    print(f"  {day} {hour:02d}:00 UTC: min. delay {delay * 1e3:.2f} ms, {len(path) - 1} hops")
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.16, 3.1))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-33, central_latitude=50, standard_parallels=(40, 60)))
    ax.set_extent([-80, 14, 34, 66], crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeeee", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4, color="0.5")
    cruise = (field.heights >= 8e3) & (field.heights <= 12.5e3)
    cloud = (cs.ext[cruise] > 0).any(axis=0)
    ax.pcolormesh(field.lons, field.lats, np.where(cloud, 1.0, np.nan), transform=pc,
                  cmap=mpl.colors.ListedColormap(["#9ecae1"]), alpha=0.6, shading="nearest",
                  rasterized=True)
    pos = {v: (dd["lon"], dd["lat"]) for v, dd in G.nodes(data=True) if "lon" in dd}
    xy = np.array([pos[v] for v, k in G.nodes(data="kind") if k == "aircraft"])
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="k", transform=pc, zorder=3, lw=0)
    if path:
        pts = [pos[v] for v in path if v in pos]
        ax.plot(*zip(*pts, strict=True), color="#ff7f00", lw=1.3, transform=pc, zorder=6,
                path_effects=[mpl.patheffects.withStroke(linewidth=2.2, foreground="k")])
    for site in (FRANKFURT, NEW_YORK):
        ax.plot(site.lon, site.lat, marker="*", ms=9, color="#e6550d", mec="k", mew=0.5,
                transform=pc, zorder=7)
    _save(fig, f"map_{day}")


def _station_map(day, site, extent, name, hour=14):
    """Airliners around an endpoint with the latency-service ground stations."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t = _t0(day) + hour * 3600
    snap = select_fleet(pd.read_parquet(DAYS[day][0], filters=[("time", "==", t)]))
    p = scenario("leo")
    R = p.geometry.earth_radius
    stations = [g.site for g in ground_stations(p) if g.site.name.startswith(site.name)]
    pos = geo.to_ecef(snap.lat.to_numpy(), snap.lon.to_numpy(), snap.alt.to_numpy(), R)
    vis = np.zeros(len(snap), dtype=bool)
    for st in stations:
        g = geo.to_ecef(st.lat, st.lon, 0.0, R)
        vis |= geo.elevation_deg(np.broadcast_to(g, pos.shape), pos) >= 5
    vis &= snap.alt.to_numpy() >= 3000
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(site.lon, site.lat))
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f2f2", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, color="0.5")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, color="0.7")
    sc = ax.scatter(snap.lon, snap.lat, c=snap.alt / 1e3, s=12, cmap="viridis", vmin=0,
                    vmax=12, transform=pc, zorder=3)
    ax.scatter(snap.lon[vis], snap.lat[vis], s=60, facecolor="none", edgecolor="r", lw=0.8,
               transform=pc, zorder=4, label="usable: ≥5° from a station, above 3 km")
    ax.scatter([s.lon for s in stations], [s.lat for s in stations], marker="^", s=40,
               color="#e6550d", edgecolor="k", lw=0.5, transform=pc, zorder=5,
               label=f"ground stations ({len(stations)})")
    g = geo.to_ecef(site.lat, site.lon, 0.0, R)
    d = next(dk for dk in np.arange(1e3, 400e3, 1e3)
             if geo.elevation_deg(g, geo.to_ecef(*geo.destination(
                 site.lat, site.lon, 0.0, dk, R), 10.5e3, R)) <= 5)
    la, lo = geo.destination(site.lat, site.lon, np.arange(0, 361, 3), d, R)
    ax.plot(lo, la, ls="--", color="r", lw=0.8, transform=pc,
            label=f"5° elevation at 10.5 km from the centre ({d / 1e3:.0f} km)")
    ax.plot(site.lon, site.lat, "r*", ms=11, mec="k", mew=0.5, transform=pc, zorder=6)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="altitude [km]")
    ax.legend(loc="lower right", fontsize=6)
    _save(fig, f"{name}_{day}")


def newyork(day):
    # New York slightly up and left of the centre of the map
    _station_map(day, NEW_YORK, [-76.9, -70.0, 38.6, 42.4], "q3_newyork")


def frankfurt(day):
    _station_map(day, FRANKFURT, [5.9, 12.4, 48.2, 51.8], "q3_frankfurt")


EXPERIMENTS = {"overview": overview, "tech": tech, "map": snapshot_map, "newyork": newyork,
               "frankfurt": frankfurt}
PLOTS = {"overview": plot_overview, "tech": plot_tech}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--day", choices=list(DAYS), action="append")
    ap.add_argument("--plot", action="store_true", help="only redraw from stored results")
    a = ap.parse_args()
    days = a.day or list(DAYS)
    for name in a.names:
        if not a.plot:
            for d in days:
                EXPERIMENTS[name](d)
        if name in PLOTS:
            PLOTS[name](list(DAYS))
