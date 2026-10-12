"""Experiments of the paper; results in results/quick/, figures in figures/quick/.

Two days: 9 July 2023 (summer, all-time Shanwick record) and 16 January 2024 (winter
weekday). One snapshot per hour (24 per day), always with ERA5 clouds and RF fallback of
the ground links, minimum elevation 5 deg, minimum air-to-air ray height 3 km. The only
random element is the latent cloud field (seed 0; more seeds via --seed).
Capacities are solved to within 1 % of the upper bound where possible: MILP limit 30 s,
capacities still more than 1 % below the bound are re-solved with 300 s (the bound is
stored with every value, so the remaining gap is known).

* capacity: ground stations everywhere on land, Europe -> North America
* latency:  Frankfurt -> New York (NY4), stations at the endpoints and on rings of 75 and
            150 km around them (land points only)

    python scripts/quick.py sweep      # all rates x K = 2, 4 per day and seed (main run)
    python scripts/quick.py rings      # latency without the ring stations
    python scripts/quick.py --seed 1 --seed 2 --seed 3 --seed 4 latency  # latency, more seeds
    python scripts/quick.py map        # maps of one snapshot (12 UTC, seed 0)
    python scripts/quick.py newyork    # airliners and ground stations around New York
    python scripts/quick.py frankfurt  # the same around Frankfurt
    python scripts/quick.py summary    # numbers quoted in the paper -> results/quick/summary.txt
    python scripts/quick.py --day winter --seed 0 sweep   # one day / seed only
    python scripts/quick.py --plot sweep                  # redraw from results/quick/*.csv

Existing result files are not recomputed (delete them or use --force); every day is run in
blocks of 4 hours that are stored in results/quick/parts/, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hap_atlantic import flow
from hap_atlantic import geometry as geo
from hap_atlantic.config import FRANKFURT, NEW_YORK, Params
from hap_atlantic.reference import benchmarks
from hap_atlantic.simulate import config, run
from hap_atlantic.topology import build_graph, ground_stations
from hap_atlantic.traffic import select_fleet
from hap_atlantic.weather import CloudField

DAYS = {
    "summer": ("data/processed/20230708-10.parquet", "2023-07-09", "9 Jul 2023"),
    "winter": ("data/processed/20240115-17.parquet", "2024-01-16", "16 Jan 2024"),
}
SEEDS = (0,)  # seeds of the latent cloud field (one: a day takes about an hour)
COLORS = {"summer": "#d95f02", "winter": "#1f78b4"}
OUT, FIG = Path("results/quick"), Path("figures/quick")
WORKERS = 4  # further limited by the free memory (simulate.safe_workers)
RATES = (1e9, 2.5e9, 1e10, 1e11, 1e12)  # bit/s per link
KS = (2, 4)
REF = (1e11, 2)  # reference configuration: 100 Gbit/s, K = 2
COLW = 3.5  # IEEE column width [in]

# Times, as in the paper (IEEEtran); STIX is the Times clone shipped with matplotlib.
# Figures are drawn at their final size, so 8 pt here is 8 pt in the paper.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "legend.fontsize": 7,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "savefig.dpi": 300})
CABLE_ONE = 250e12  # Dunant (Dunant2021)
# lower bound for all transatlantic cables: Dunant 250 + Grace Hopper 352 + Amitie >320 Tbit/s
# (Dunant2021, Papapavlou2022); 17 cables in total (TeleGeography2022)
CABLE_ALL = (250 + 352 + 320) * 1e12


def power(r_max: float) -> float:
    """Transmit power [W] assumed for a rate per link: 1 W up to 100 Gbit/s, 5 W above."""
    return 5.0 if r_max > 1e11 else 1.0


def label(r_max: float, k: int) -> str:
    return config(r_max, k, power(r_max), Params()).label


def _t0(day):
    return pd.Timestamp(DAYS[day][1], tz="UTC").timestamp()


def _states(day):
    """One snapshot per hour of the day."""
    t0 = _t0(day)
    st = pd.read_parquet(DAYS[day][0], filters=[("time", ">=", t0), ("time", "<", t0 + 86400)])
    return st[(st["time"] - t0) % 3600 == 0]


def _run(day, p, configs, milp_time=30.0, retry_time=300.0, hours=range(24)):
    """Hourly snapshots; capacities that are still > 1 % from the bound are re-solved."""
    tic = time.time()
    t0 = _t0(day)
    t_a, t_b = t0 + 3600 * min(hours), t0 + 3600 * (max(hours) + 1)
    states = _states(day)
    states = states[(states["time"] >= t_a) & (states["time"] < t_b)]
    # the time window of `run` stays the whole day: it also selects the cached weather file
    r = run(states, p, configs, t0, t0 + 86400, every=1, workers=WORKERS, milp_time=milp_time)
    cols = [f"cap_{c.label}" for c in configs]
    # re-solve only the (snapshot, configuration) pairs that are still > 1 % from the bound
    r = r.set_index("time")
    for c in configs:
        col = f"cap_{c.label}"
        hard = r.index[r[col] < 0.99 * r[f"{col}_bound"]]
        if len(hard):
            print(f"  {day} seed {p.seed}: re-solving {len(hard)} snapshots for {c.label} "
                  f"with {retry_time:.0f} s", flush=True)
            r2 = run(states[states["time"].isin(hard)], p, [c], t0, t0 + 86400, every=1,
                     workers=WORKERS, milp_time=retry_time).set_index("time")
            for k in (col, f"{col}_heur", f"{col}_bound"):
                r.loc[r2.index, k] = r2[k]
    r = r.reset_index()
    r["hour"] = (r["time"] - t0) / 3600.0
    msg = (f"  {day} seed {p.seed} {min(hours):02d}-{max(hours) + 1:02d} h: "
           f"{time.time() - tic:.0f} s")
    if cols:
        c = f"cap_{label(*REF)}"
        msg += f", capacity {r[c].mean() / 1e9:.0f} Gbit/s"
    print(msg + f", Frankfurt-New York connected {r['connected'].sum()}/{len(r)}", flush=True)
    return r.assign(day=day, seed=p.seed)


def _out(name, day, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}_{day}_s{seed}.csv"


def _read(name, days):
    """{day: DataFrame of all seeds} for the result files that exist."""
    res = {}
    for d in days:
        files = sorted(OUT.glob(f"{name}_{d}_s*.csv"))
        if files:
            res[d] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return res


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"figures/quick/{name}.pdf")


def _time_axis(ax):
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("time (UTC)")


# --- main run: all rates x K, per day and seed --------------------------------------------


BLOCK = 4  # hours per checkpoint: an interrupted run loses at most one block


def _blocks(name, day, seed, p, configs, force):
    """Run the day in blocks of BLOCK hours (checkpointed), then merge into one file.

    After every block the elapsed time and a naive estimate of the remaining time of this
    day (mean block time x remaining blocks) are printed.
    """
    f = _out(name, day, seed)
    if f.exists() and not force:
        print(f"  {f} exists")
        return
    parts, spent, done = [], 0.0, 0
    starts = list(range(0, 24, BLOCK))
    for n, h0 in enumerate(starts):
        part = OUT / "parts" / f"{name}_{day}_s{seed}_h{h0:02d}.csv"
        if force or not part.exists():
            part.parent.mkdir(parents=True, exist_ok=True)
            tic = time.time()
            _run(day, p, configs, hours=range(h0, h0 + BLOCK)).to_csv(part, index=False)
            spent, done = spent + time.time() - tic, done + 1
            left = len(starts) - n - 1
            print(f"  {name} {day} seed {seed}: {spent / 60:.1f} min so far, about "
                  f"{spent / done * left / 60:.0f} min left for this day", flush=True)
        parts.append(pd.read_csv(part))
    pd.concat(parts, ignore_index=True).to_csv(f, index=False)


def sweep(day, seed, force=False):
    p = replace(Params(), seed=seed)
    configs = [config(r, k, power(r), p) for k in KS for r in RATES]
    _blocks("sweep", day, seed, p, configs, force)


def latency(day, seed, force=False):
    """Latency only (no capacity), for the cloud seeds beyond seed 0 (cheap: ~2 min/day)."""
    _blocks("latency", day, seed, replace(Params(), seed=seed), [], force)


def rings(day, seed, force=False):
    """Latency only, without the ring stations (stations at the endpoints only)."""
    p = replace(Params(), seed=seed)
    p = replace(p, ground=replace(p.ground, site_rings=()))
    _blocks("rings", day, seed, p, [], force)


def plot_sweep(days):
    res = _read("sweep", days)
    if not res:
        return
    col = f"cap_{label(*REF)}"

    # capacity per hour: mean over seeds, whiskers min-max over seeds
    fig, ax = plt.subplots(figsize=(COLW, 2.1))
    w = 0.8 / max(len(res), 1)
    for i, (d, r) in enumerate(res.items()):
        g = r.groupby("hour")
        m, lo, hi = (g[col].mean() / 1e9, g[col].min() / 1e9, g[col].max() / 1e9)
        x = m.index + (i - (len(res) - 1) / 2) * w
        ax.bar(x, m, width=w, color=COLORS[d], label=DAYS[d][2],
               yerr=[m - lo, hi - m], error_kw={"elinewidth": 0.6, "capsize": 0})
        ax.plot(x, g[f"{col}_bound"].mean() / 1e9, "k_", ms=3, mew=0.8)
    ax.plot([], [], "k_", ms=5, mew=0.8, label="upper bound")
    ax.set_ylabel("capacity [Gbit/s]")
    ax.set_ylim(0, 1.2 * ax.get_ylim()[1])
    _time_axis(ax)
    ax.legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.5)
    _save(fig, "q1_capacity")

    # latency per hour: markers = median over all cloud seeds (no path counts as infinite
    # latency, so an hour has a marker if most seeds have a path), whiskers = min-max over
    # the seeds with a path
    fig, ax = plt.subplots(figsize=(COLW, 2.1))
    ref = benchmarks()
    lat = _read("latency", days)
    for i, (d, r) in enumerate(res.items()):
        allseeds = pd.concat([r, lat[d]]) if d in lat else r
        ms_ = np.where(allseeds["connected"], allseeds["min_delay"] * 1e3, np.inf)
        med = pd.Series(ms_, index=allseeds["hour"].to_numpy()).groupby(level=0).median()
        med = med[np.isfinite(med)]
        g = allseeds[allseeds["connected"]].groupby("hour")["min_delay"]
        lo, hi = (g.min() * 1e3).reindex(med.index), (g.max() * 1e3).reindex(med.index)
        ax.errorbar(med.index + 0.25 * (i - 0.5), med, yerr=[med - lo, hi - med], fmt="o",
                    ms=2.5, color=COLORS[d], elinewidth=0.6, capsize=0, label=DAYS[d][2])
    for lab, v, dy in (("vacuum", "vacuum", -0.55), ("LEO bound", "leo_lower_bound", 0.12),
                       ("shortwave (scaled)", "hf_scaled", 0.12),
                       ("fiber + microwave", "fibre_microwave_published", 0.12)):
        ax.axhline(ref[v] * 1e3, color="k", lw=0.6, ls=":")
        ax.text(0.0, ref[v] * 1e3 + dy, lab, fontsize=7)
    _time_axis(ax)
    ax.set_ylabel("one-way latency [ms]")
    ax.set_ylim(20, 33)
    ax.legend(loc="center right")
    _save(fig, "q2_latency")

    # mean capacity versus rate per link
    fig, ax = plt.subplots(figsize=(COLW, 2.15))
    for d, r in res.items():
        for k, ls, mk in ((2, "-", "o"), (4, "--", "s")):
            per_seed = np.array([[r.loc[r["seed"] == s, f"cap_{label(x, k)}"].mean()
                                  for x in RATES] for s in sorted(r["seed"].unique())])
            m = per_seed.mean(axis=0)
            ax.errorbar(RATES, m, yerr=[m - per_seed.min(axis=0), per_seed.max(axis=0) - m],
                        ls=ls, marker=mk, ms=3, color=COLORS[d], elinewidth=0.6,
                        label=f"{DAYS[d][2]}, $K={k}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for v, lab in ((CABLE_ONE, "one cable (250 Tbit/s)"),
                   (CABLE_ALL, "all transatlantic cables (>0.9 Pbit/s)")):
        ax.axhline(v, color="k", lw=0.6, ls=":")
        ax.text(3.2e9, v * 1.25, lab, fontsize=7)
    ax.axvspan(9e8, 2.7e9, color="0.88", lw=0)
    ax.text(1.0e9, 1.5e13, "demon-\nstrated", fontsize=7)
    ax.set_xlim(7e8, 1.5e12)
    ax.set_ylim(1e9, 4e15)
    ax.set_xlabel(r"rate per link $R_\mathrm{max}$ [bit/s]")
    ax.set_ylabel("mean capacity [bit/s]")
    ax.legend(loc="lower right", fontsize=6.5)
    _save(fig, "q4_tech")


# --- numbers for the paper ---------------------------------------------------------------


def summary(days=tuple(DAYS)):
    res, rg = _read("sweep", days), _read("rings", days)
    lat = {d: pd.concat([r, *([_read("latency", [d])[d]] if _read("latency", [d]) else [])],
                        ignore_index=True) for d, r in res.items()}
    ref = benchmarks()
    lines = ["benchmarks [ms]: " + ", ".join(f"{k}={v * 1e3:.2f}" for k, v in ref.items()
                                              if k != "great_circle_km")]
    col = f"cap_{label(*REF)}"
    for d, r in res.items():
        seeds = sorted(r["seed"].unique())
        lines.append(f"\n== {d} ({len(seeds)} seeds)")
        pm = r.groupby("seed")[col].mean() / 1e9
        lines.append(f"ref capacity mean {pm.mean():.1f} Gbit/s "
                     f"(seeds {pm.min():.1f}-{pm.max():.1f})")
        hm = r.groupby("hour")[col].mean() / 1e9
        lines.append(f"hourly mean over seeds: max {hm.max():.0f} at {hm.idxmax():.0f}h; "
                     + " ".join(f"{h:.0f}:{v:.0f}" for h, v in hm.items()))
        z = r.groupby("hour")[col].max()
        lines.append(f"hours with zero capacity in all seeds: {list(z[z == 0].index)}; "
                     f"in some seed: {list(r.loc[r[col] == 0, 'hour'].unique())}")
        lines.append(f"access EU {r.eu_access.min()}-{r.eu_access.max()}, "
                     f"NA {r.na_access.min()}-{r.na_access.max()}; "
                     f"aircraft in corridor mean {r.n_aircraft.mean():.0f}")
        for k in KS:
            for x in RATES:
                c = f"cap_{label(x, k)}"
                ps = r.groupby("seed")[c].mean()
                gap = 1 - r[c] / r[f"{c}_bound"].where(r[f"{c}_bound"] > 0)
                lines.append(f"K={k} R={x:.1e}: mean {ps.mean() / 1e9:.2f} Gbit/s "
                             f"(seeds {ps.min() / 1e9:.2f}-{ps.max() / 1e9:.2f}), chains "
                             f"{ps.mean() / x:.2f}, max gap {np.nanmax(gap):.2%}, "
                             f"sum gap {1 - r[c].sum() / r[f'{c}_bound'].sum():.2%}")
        for x in RATES:
            lines.append(f"K4/K2 at {x:.1e}: "
                         f"{r[f'cap_{label(x, 4)}'].mean() / r[f'cap_{label(x, 2)}'].mean():.2f}")
        con = r.groupby("seed")["connected"].sum()
        lines.append(f"connected hours per seed: {dict(con)}")
        rc = r[r["connected"]]
        lines.append(f"latency min {rc.min_delay.min() * 1e3:.2f}, "
                     f"max {rc.min_delay.max() * 1e3:.2f}, "
                     f"median {rc.min_delay.median() * 1e3:.2f} ms, hops median "
                     f"{rc.min_delay_hops.median():.0f}")
        nc = r[~r["connected"]]
        lines.append(f"not connected but capacity > 0: {int((nc[col] > 0).sum())} of {len(nc)}")
        if d in lat:
            g = lat[d]
            con = g.groupby("seed")["connected"].sum()
            gc = g[g["connected"]]
            lines.append(f"latency over {g.seed.nunique()} seeds: connected hours {dict(con)}, "
                         f"median {gc.min_delay.median() * 1e3:.2f} ms, range "
                         f"{gc.min_delay.min() * 1e3:.2f}-{gc.min_delay.max() * 1e3:.2f} ms")
        if d in rg:
            g = rg[d]
            lines.append(f"without rings: connected hours per seed "
                         f"{dict(g.groupby('seed')['connected'].sum())}, median latency "
                         f"{g.loc[g.connected, 'min_delay'].median() * 1e3:.2f} ms")
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


# --- maps --------------------------------------------------------------------------------


def snapshot_map(day, hour=12):
    """Airliners (dots), cloud at cruise level (8-12.5 km), min-latency path (seed 0)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t0 = _t0(day)
    t = t0 + hour * 3600
    p = Params()
    snap = select_fleet(pd.read_parquet(DAYS[day][0], filters=[("time", "==", t)]))
    field = CloudField(t0 - 3600, t0 + 86400 + 3600, p.clouds, p.seed, p.geometry.earth_radius)
    cs = field.at(t)
    G = build_graph(snap, p, cs)
    delay, path = flow.min_latency_path(G)
    print(f"  {day} {hour:02d}:00 UTC: min. delay {delay * 1e3:.2f} ms, {len(path) - 1} hops")
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COLW, 1.35))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-33, central_latitude=50, standard_parallels=(40, 60)))
    ax.set_extent([-80, 14, 34, 66], crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeeee", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, color="0.5")
    cruise = (field.heights >= 8e3) & (field.heights <= 12.5e3)
    cloud = (cs.ext[cruise] > 0).any(axis=0)
    ax.pcolormesh(field.lons, field.lats, np.where(cloud, 1.0, np.nan), transform=pc,
                  cmap=mpl.colors.ListedColormap(["#9ecae1"]), alpha=0.6, shading="nearest",
                  rasterized=True)
    pos = {v: (dd["lon"], dd["lat"]) for v, dd in G.nodes(data=True) if "lon" in dd}
    xy = np.array([pos[v] for v, k in G.nodes(data="kind") if k == "aircraft"])
    ax.scatter(xy[:, 0], xy[:, 1], s=0.4, color="k", transform=pc, zorder=3, lw=0,
               rasterized=True)
    if path:
        pts = [pos[v] for v in path if v in pos]
        ax.plot(*zip(*pts, strict=True), color="#ff7f00", lw=0.9, transform=pc, zorder=6,
                path_effects=[mpl.patheffects.withStroke(linewidth=1.6, foreground="k")])
    for site in (FRANKFURT, NEW_YORK):
        ax.plot(site.lon, site.lat, marker="*", ms=6, color="#e6550d", mec="k", mew=0.4,
                transform=pc, zorder=7)
    _save(fig, f"map_{day}")


def _station_map(day, site, extent, name, hour=14):
    """Airliners around an endpoint with the latency-service ground stations."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t = _t0(day) + hour * 3600
    snap = select_fleet(pd.read_parquet(DAYS[day][0], filters=[("time", "==", t)]))
    p = Params()
    R = p.geometry.earth_radius
    stations = [g.site for g in ground_stations(p) if g.site.name.startswith(site.name)]
    pos = geo.to_ecef(snap.lat.to_numpy(), snap.lon.to_numpy(), snap.alt.to_numpy(), R)
    vis = np.zeros(len(snap), dtype=bool)
    for st in stations:
        g = geo.to_ecef(st.lat, st.lon, 0.0, R)
        vis |= geo.elevation_deg(np.broadcast_to(g, pos.shape), pos) >= p.geometry.elev_min_ground
    # aircraft below h_chord_min cannot have links to other aircraft, i.e. cannot relay
    vis &= snap.alt.to_numpy() >= p.geometry.h_chord_min
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(3.0, 2.6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(site.lon, site.lat))
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f2f2", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, color="0.5")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, color="0.7")
    sc = ax.scatter(snap.lon, snap.lat, c=snap.alt / 1e3, s=5, cmap="viridis", vmin=0,
                    vmax=12, transform=pc, zorder=3, lw=0)
    ax.scatter(snap.lon[vis], snap.lat[vis], s=22, facecolor="none", edgecolor="r", lw=0.6,
               transform=pc, zorder=4, label="can relay to a station")
    ax.scatter([s.lon for s in stations], [s.lat for s in stations], marker="^", s=18,
               color="#e6550d", edgecolor="k", lw=0.4, transform=pc, zorder=5,
               label=f"ground stations ({len(stations)})")
    g = geo.to_ecef(site.lat, site.lon, 0.0, R)
    d = next(dk for dk in np.arange(1e3, 400e3, 1e3)
             if geo.elevation_deg(g, geo.to_ecef(*geo.destination(
                 site.lat, site.lon, 0.0, dk, R), 10.5e3, R)) <= p.geometry.elev_min_ground)
    la, lo = geo.destination(site.lat, site.lon, np.arange(0, 361, 3), d, R)
    ax.plot(lo, la, ls="--", color="r", lw=0.6, transform=pc,
            label=f"5° elevation at 10.5 km ({d / 1e3:.0f} km)")
    ax.plot(site.lon, site.lat, "r*", ms=7, mec="k", mew=0.4, transform=pc, zorder=6)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("altitude [km]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.5, frameon=False)
    _save(fig, f"{name}_{day}")


def newyork(day):
    # New York slightly up and left of the centre of the map
    _station_map(day, NEW_YORK, [-76.9, -70.0, 38.6, 42.4], "q3_newyork")


def frankfurt(day):
    _station_map(day, FRANKFURT, [5.9, 12.4, 48.2, 51.8], "q3_frankfurt")


PER_SEED = {"sweep": sweep, "rings": rings, "latency": latency}
PER_DAY = {"map": snapshot_map, "newyork": newyork, "frankfurt": frankfurt}
PLOTS = {"sweep": plot_sweep}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=[*PER_SEED, *PER_DAY, "summary"])
    ap.add_argument("--day", choices=list(DAYS), action="append")
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--plot", action="store_true", help="only redraw from stored results")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    a = ap.parse_args()
    days = a.day or list(DAYS)
    seeds = a.seed or list(SEEDS)
    for name in a.names:
        if name == "summary":
            summary()
            continue
        if not a.plot:
            if name in PER_SEED:
                for s in seeds:
                    for d in days:
                        PER_SEED[name](d, s, a.force)
            else:
                for d in days:
                    PER_DAY[name](d)
        if name in PLOTS:
            PLOTS[name](list(DAYS))
