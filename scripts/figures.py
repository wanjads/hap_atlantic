"""Figures for the paper (PDF + PNG in figures/), from the CSVs of scripts/experiments.py.

    python scripts/figures.py            # all figures whose inputs exist
    python scripts/figures.py map rate   # selected figures
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hap_atlantic import flow
from hap_atlantic.config import FRANKFURT, K_VALUES, NEW_YORK, TECHS, scenario
from hap_atlantic.reference import benchmarks
from hap_atlantic.simulate import OCEAN_LON_EAST, OCEAN_LON_WEST
from hap_atlantic.topology import build_graph
from hap_atlantic.traffic import select_fleet
from hap_atlantic.weather import CloudField

RES, FIG = Path("results"), Path("figures")
COL, FULL = 3.5, 7.16  # IEEE column / page width [in]
DAYS = {"summer": "9 Jul 2023", "winter": "16 Jan 2024"}
DAY_FILES = {"summer": ("data/processed/20230708-10.parquet", "2023-07-09"),
             "winter": ("data/processed/20240115-17.parquet", "2024-01-16")}
KCOL = {2: "#1b9e77", 4: "#d95f02", 6: "#7570b3"}
CABLE_DUNANT, USED_BW = 250e12, 571e12  # [bit/s] Dunant2021, TeleGeography via Amitie2023

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "savefig.dpi": 300, "lines.linewidth": 1.2,
})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"figures/{name}.pdf")


def _read(name):
    f = RES / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def _hours(df, day):
    return (df["time"] - pd.Timestamp(DAY_FILES[day][1], tz="UTC").timestamp()) / 3600.0


# --- Fig. model: map of one snapshot ----------------------------------------------------


def fig_map(day="summer", hour=12.0):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    path, d = DAY_FILES[day]
    t = pd.Timestamp(d, tz="UTC").timestamp() + hour * 3600
    p = scenario("leo")
    p = replace(p, clouds=replace(p.clouds, model="era5"))
    states = pd.read_parquet(path)
    snap = select_fleet(states[states.time == t], True, 1.0, 0)
    t0 = pd.Timestamp(d, tz="UTC").timestamp()
    field = CloudField(t0 - 3600, t0 + 86400 + 3600, p.clouds, p.seed, p.geometry.earth_radius)
    cs = field.at(t)
    G = build_graph(snap, p, cs)
    G_clear = build_graph(snap, replace(p, clouds=replace(p.clouds, model="clear")))
    delay, path_nodes = flow.min_latency_path(G)
    path_label = "min. one-way delay"
    if not path_nodes:  # no path under clouds: show the clear-sky path instead
        delay, path_nodes = flow.min_latency_path(G_clear)
        path_label = "no path under clouds; clear-sky min. delay"

    proj = ccrs.LambertConformal(central_longitude=-33, central_latitude=50,
                                 standard_parallels=(40, 60))
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(FULL, 3.1))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-80, 14, 34, 66], crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeeee", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4, color="0.5")
    cruise = (field.heights >= 8e3) & (field.heights <= 12.5e3)
    cloud = (cs.ext[cruise] > 0).any(axis=0)  # cloud somewhere at cruise level
    ax.pcolormesh(field.lons, field.lats, np.where(cloud, 1.0, np.nan), transform=pc,
                  cmap=mpl.colors.ListedColormap(["#9ecae1"]), alpha=0.6, shading="nearest",
                  rasterized=True)
    pos = {v: (dd["lon"], dd["lat"]) for v, dd in G.nodes(data=True) if "lon" in dd}
    blocked = [(u, v) for u, v, k in G_clear.edges(data="kind")
               if k == "air" and not G.has_edge(u, v)]
    for edges, kw in ((blocked, {"color": "#de2d26", "lw": 0.2, "alpha": 0.15}),
                      ([(u, v) for u, v, k in G.edges(data="kind") if k == "air"],
                       {"color": "0.35", "lw": 0.25, "alpha": 0.5})):
        for u, v in edges:
            ax.plot(*zip(pos[u], pos[v], strict=True), transform=pc, **kw, rasterized=True)
    xy = np.array([pos[v] for v, k in G.nodes(data="kind") if k == "aircraft"])
    ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color="k", transform=pc, zorder=3, lw=0)
    if path_nodes:
        pts = [pos.get(v) for v in path_nodes if v in pos]
        ax.plot(*zip(*pts, strict=True), color="#ff7f00", lw=1.3, transform=pc, zorder=6,
                path_effects=[mpl.patheffects.withStroke(linewidth=2.2, foreground="k")])
    for site in (FRANKFURT, NEW_YORK):
        ax.plot(site.lon, site.lat, marker="*", ms=9, color="#e6550d", mec="k", mew=0.5,
                transform=pc, zorder=5)
    for lon in (OCEAN_LON_EAST, OCEAN_LON_WEST):
        ax.plot([lon, lon], [34, 66], ls="--", lw=0.8, color="#3182bd", transform=pc)
    ax.plot([FRANKFURT.lon, NEW_YORK.lon], [FRANKFURT.lat, NEW_YORK.lat], transform=ccrs.Geodetic(),
            color="#e6550d", lw=0.6, ls=":")
    n_clear = sum(1 for *_, k in G.edges(data="kind") if k == "air")
    ax.text(0.01, 0.02, f"{DAYS[day]}, {hour:02.0f}:00 UTC: {len(xy)} airliners\n"
            f"{n_clear} usable (grey) / {len(blocked)} cloud-blocked (red) links, "
            f"blue: cloud at 8-12.5 km\n{path_label} {delay * 1e3:.2f} ms "
            f"({len(path_nodes) - 1} hops, orange)", transform=ax.transAxes,
            fontsize=6, bbox={"fc": "white", "ec": "none", "alpha": 0.85}, zorder=7)
    _save(fig, "fig_map")


# --- Fig. traffic and ocean capacity over the day ----------------------------------------


def fig_daily():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    for ax, day in zip(axes, DAYS, strict=True):
        df = _read(f"tech_{day}")
        if df is None:
            continue
        h = _hours(df, day)
        for k in K_VALUES:
            ax.step(h, df[f"ocean_leo_k{k}"] / 100e9, where="post", color=KCOL[k],
                    label=f"$K={k}$")
        st = select_fleet(pd.read_parquet(DAY_FILES[day][0]), True, 1.0, 0)
        oc = st[(st.lon > OCEAN_LON_WEST) & (st.lon < OCEAN_LON_EAST) & (st.alt > 8000)]
        n = oc.groupby("time").size()
        ax2 = ax.twinx()
        ax2.plot(_hours(n.reset_index(), day), n.to_numpy(), color="0.55", lw=0.8, ls="--")
        ax2.set_ylim(0, 450)
        ax2.grid(False)
        if day == "winter":
            ax2.set_ylabel("airliners 55\u00b0W–15\u00b0W (dashed)")
        else:
            ax2.set_yticklabels([])
        ax.set_title(DAYS[day], fontsize=8)
        ax.set_xlabel("time [h UTC]")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
    axes[0].set_ylabel("disjoint oceanic chains\n(= capacity / $R_\\mathrm{max}$)")
    axes[0].legend(loc="upper left", ncol=3)
    _save(fig, "fig_daily")


# --- Fig. capacity vs transceiver rate ---------------------------------------------------


def _sweep_curve(df, metric, k):
    """Mean capacity vs r_max: linear in r_max up to the smallest link rate, then MILP points."""
    rmin = df["min_link_rate"].min()
    chains = (df[f"{metric}_leo_k{k}"] / TECHS["leo"].r_max).mean()
    r = np.logspace(9, np.log10(min(rmin, 1e12)), 30)
    pts = [(t.r_max, df[f"{metric}_{name}_k{k}"].mean()) for name, t in TECHS.items()
           if f"{metric}_{name}_k{k}" in df and name != "leo"]
    return r, chains * r, pts


def fig_rate():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    for ax, metric, title in zip(axes, ("ocean", "e2e"),
                                 ("oceanic section", "Frankfurt–New York, 1 station each"),
                                 strict=True):
        for day, ls in (("summer", "-"), ("winter", "--")):
            df = _read(f"tech_{day}")
            if df is None:
                continue
            for k in K_VALUES:
                r, c, pts = _sweep_curve(df, metric, k)
                ax.loglog(r, c, ls=ls, color=KCOL[k],
                          label=f"$K={k}$, {DAYS[day]}" if metric == "ocean" else None)
                if pts:
                    x, y = zip(*sorted(pts), strict=True)
                    ax.loglog([r[-1], *x], [c[-1], *y], ls=ls, color=KCOL[k], marker="o",
                              ms=2.5)
        ax.axvspan(1e9, 2.6e9, color="0.85", zorder=0)
        ax.text(1.1e9, 1.3e9, "demon-\nstrated", fontsize=6, va="bottom")
        for v, lab, va in ((CABLE_DUNANT, "one cable (Dunant, 250 Tbit/s)", "top"),
                           (USED_BW, "used NA–EU bandwidth (571 Tbit/s)", "bottom")):
            ax.axhline(v, color="k", lw=0.6, ls=":")
            ax.text(3e9, v * (0.85 if va == "top" else 1.15), lab, fontsize=6, va=va)
        ax.set_xlim(1e9, 1.2e13)
        ax.set_ylim(1e9, 2e15)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("transceiver rate per link $R_\\mathrm{max}$ [bit/s]")
    axes[0].set_ylabel("mean capacity [bit/s]")
    axes[0].legend(ncol=2, loc="lower right", fontsize=6)
    axes[1].text(3e9, 4e13, "lines: chains $\\times R_\\mathrm{max}$ (all links capped)\n"
                 "dots: MILP with link budget\n(1 W up to 100 Gbit/s, 5 W at 1 Tbit/s, "
                 "10 W at 10 Tbit/s)", fontsize=5.5, va="top")
    _save(fig, "fig_rate")


# --- Fig. latency ------------------------------------------------------------------------


def fig_latency():
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ref = benchmarks()
    for day, col in (("summer", "#e6550d"), ("winter", "#3182bd")):
        for wx, ls in (("clear", "-"), ("era5", "--")):
            df = _read(f"paths_{day}_{wx}")
            if df is None:
                continue
            d = np.sort(df["delay_10us"].to_numpy() * 1e3)
            frac = np.arange(1, len(d) + 1) / len(d)
            ax.step(d, frac, where="post", color=col, ls=ls,
                    label=f"{DAYS[day]}, {'clear sky' if wx == 'clear' else 'ERA5 clouds'}")
    lines = [("vacuum", ref["vacuum"]), ("LEO bound", ref["leo_lower_bound"]),
             ("HF (scaled)", ref["hf_scaled"]), ("fibre bound", ref["fibre_great_circle"]),
             ("fibre+MW", ref["fibre_microwave_published"]),
             ("fibre", ref["fibre_best_published"])]
    for lab, v in lines:
        ax.axvline(v * 1e3, color="0.3", lw=0.6, ls=":")
        ax.text(v * 1e3 + 0.12, 0.98, lab, fontsize=5.5, rotation=90, va="top")
    ax.set_xlim(20, 35)
    ax.set_ylim(0, 1)
    ax.set_xlabel("one-way latency Frankfurt–New York [ms]")
    ax.legend(loc="center right", fontsize=5.5, bbox_to_anchor=(0.88, 0.5))
    ax.set_ylabel("fraction of time with latency $\\leq x$\n(plateau = availability)")
    _save(fig, "fig_latency")


# --- Fig. weather ------------------------------------------------------------------------


def fig_weather():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    variants = [(1, 0.0, "1 station"), (5, 0.0, "5 stations"), (1, 10.0, "1 st. + RF"),
                (5, 10.0, "5 st. + RF")]
    for di, day in enumerate(DAYS):
        df = _read(f"weather_{day}")
        if df is None:
            continue
        df["avail"] = df["e2e_leo_k2"] > 0
        g = df.groupby(["sites", "rf_gbps", "seed"])["avail"].mean()
        m = [g.loc[s, rf].mean() for s, rf, _ in variants]
        sd = [g.loc[s, rf].std() for s, rf, _ in variants]
        ax.bar(np.arange(4) + (di - 0.5) * 0.4, m, 0.4, yerr=sd, capsize=1.5, label=DAYS[day])
    ax.set_xticks(range(4), [v[2] for v in variants])
    ax.set_ylabel("e2e availability (capacity > 0)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)
    _save(fig, "fig_weather")


# --- Fig. equipage ------------------------------------------------------------------------


def fig_equipage():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))
    for day, ls in (("summer", "-"), ("winter", "--")):
        eq, full = _read(f"equipage_{day}"), _read(f"tech_{day}")
        if eq is None or full is None:
            continue
        full = full.assign(equipage=1.0, seed=0)
        df = pd.concat([eq, full[eq.columns.intersection(full.columns)]])
        for k in (2, 4):
            col = f"ocean_leo_k{k}"
            g = df.groupby(["equipage", "seed"]).agg(cap=(col, "mean"),
                                                     av=(col, lambda x: (x > 0).mean()))
            g = g.groupby("equipage").agg(["mean", "std"])
            x = g.index.to_numpy()
            axes[0].errorbar(x, g[("cap", "mean")] / 1e12, g[("cap", "std")] / 1e12, ls=ls,
                             color=KCOL[k], marker="o", ms=2.5, capsize=1.5,
                             label=f"$K={k}$, {DAYS[day]}")
            axes[1].errorbar(x, g[("av", "mean")], g[("av", "std")], ls=ls, color=KCOL[k],
                             marker="o", ms=2.5, capsize=1.5)
    axes[0].set_ylabel("mean oceanic capacity [Tbit/s]\n($R_\\mathrm{max}=100$ Gbit/s)")
    axes[1].set_ylabel("oceanic availability")
    for ax in axes:
        ax.set_xlabel("share of equipped airliners")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=6)
    _save(fig, "fig_equipage")


# --- Fig. path stability -----------------------------------------------------------------


def _runs(series):
    """Durations [min] of runs of identical non-empty values."""
    s = series.fillna("").to_numpy()
    out, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if s[start]:
                out.append(i - start)
            start = i
    return np.array(out)


def fig_paths():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for day, col in (("summer", "#e6550d"), ("winter", "#3182bd")):
        df = _read(f"paths_{day}_clear")
        if df is None:
            continue
        for c, ls, lab in (("path", ":", "min-latency path"),
                           ("sticky_path", "-", "sticky (+0.1 ms)")):
            r = np.sort(_runs(df[c]))
            ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post", color=col, ls=ls,
                    label=f"{lab}, {DAYS[day]}")
    ax.set_xscale("log")
    ax.set_xlabel("path lifetime [min]")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=5.5)
    _save(fig, "fig_paths")


FIGS = {"map": fig_map, "daily": fig_daily, "rate": fig_rate, "latency": fig_latency,
        "weather": fig_weather, "equipage": fig_equipage, "paths": fig_paths}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
