"""All simulation runs of the paper. Results go to results/<experiment>_<day>.csv.

    python scripts/experiments.py tech      # clear sky, technology tiers x K
    python scripts/experiments.py equipage  # clear sky, share of equipped airliners
    python scripts/experiments.py weather   # ERA5 clouds, f_cirrus x sites x RF x seeds
    python scripts/experiments.py paths     # min-latency path every minute, stability
    python scripts/experiments.py solver    # heuristic vs MILP vs bounds, Yao vs full graph

Days: 9 July 2023 (all-time Shanwick record, summer peak) and 16 January 2024 (winter
weekday). Neighbouring days are part of the traffic files to avoid edge effects.
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from hap_atlantic import flow
from hap_atlantic.config import K_VALUES, scenario
from hap_atlantic.simulate import ocean_graph, run, tech_config
from hap_atlantic.topology import DST, SRC, build_graph, with_tech
from hap_atlantic.traffic import select_fleet
from hap_atlantic.weather import CloudField

DAYS = {
    "summer": ("data/processed/20230708-10.parquet", "2023-07-09"),
    "winter": ("data/processed/20240115-17.parquet", "2024-01-16"),
}
OUT = Path("results")
SEEDS = (0, 1, 2)
WORKERS = 16


def _day(name):
    path, day = DAYS[name]
    t0 = pd.Timestamp(day, tz="UTC").timestamp()
    return pd.read_parquet(path), t0, t0 + 86400.0


def _save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"wrote {OUT / name}.csv ({len(df)} rows)", flush=True)


def exp_tech(days):
    """Clear sky; leo results scale linearly to any r_max <= min_link_rate (all capped)."""
    p = scenario("leo")
    configs = [tech_config(t, k, p) for k in K_VALUES for t in ("leo", "dwdm", "visionary")]
    for d in days:
        states, t0, t1 = _day(d)
        r = run(states, p, configs, t0, t1, workers=WORKERS)
        _save(r.assign(day=d), f"tech_{d}")


def exp_equipage(days):
    p0 = scenario("leo")
    configs = [tech_config("leo", k, p0) for k in (2, 4)]
    for d in days:
        states, t0, t1 = _day(d)
        rows = []
        for eq, seed in itertools.product((0.25, 0.5, 0.75), SEEDS):
            p = replace(p0, equipage=eq, seed=seed)
            r = run(states, p, configs, t0, t1, workers=WORKERS)
            rows.append(r.assign(equipage=eq, seed=seed))
            print(f"  {d} equipage {eq} seed {seed}", flush=True)
        _save(pd.concat(rows).assign(day=d), f"equipage_{d}")


def exp_weather(days):
    p0 = scenario("leo")
    configs = [tech_config("leo", 2, p0)]
    for d in days:
        states, t0, t1 = _day(d)
        rows = []
        for f_ci, sites, rf, seed in itertools.product((0.0, 0.5, 1.0), (1, 5), (0.0, 10e9),
                                                     SEEDS):
            p = replace(p0, seed=seed,
                        clouds=replace(p0.clouds, model="era5", f_cirrus=f_ci),
                        ground=replace(p0.ground, n_sites=sites, rf_rate=rf))
            tic = time.time()
            r = run(states, p, configs, t0, t1, workers=WORKERS)
            rows.append(r.assign(f_cirrus=f_ci, sites=sites, rf_gbps=rf / 1e9, seed=seed))
            print(f"  {d} f_ci {f_ci} sites {sites} rf {rf / 1e9:g} seed {seed} "
                  f"({time.time() - tic:.0f} s)", flush=True)
        _save(pd.concat(rows).assign(day=d), f"weather_{d}")


# --- path stability ---------------------------------------------------------------------

HOP_DELAYS = (1e-6, 10e-6, 100e-6)
STICKY_SLACK = 0.1e-3  # keep the current path while it is at most 0.1 ms slower


def _path_delay(G, path):
    try:
        return sum(G[u][v]["delay"] for u, v in itertools.pairwise(path))
    except KeyError:  # a link of the path no longer exists
        return np.inf


def _paths_chunk(args):
    snaps, p, t_range = args
    clouds = (CloudField(*t_range, p.clouds, p.seed, p.geometry.earth_radius)
              if p.clouds.model == "era5" else None)
    rows, sticky = [], None
    for t, snap in snaps:
        G = build_graph(snap, p, clouds.at(t) if clouds is not None else None)
        H = nx.subgraph_view(G, filter_edge=lambda u, v, G=G: G[u][v]["cap"] >= 1e9)
        row = {"time": t}
        for hd in HOP_DELAYS:
            w = {(u, v): d["delay"] if d["kind"] == "fiber"
                 else d["delay"] - p.terminal.hop_delay + hd
                 for u, v, d in H.edges(data=True)}
            try:
                delay, path = nx.single_source_dijkstra(
                    H, SRC, DST, weight=lambda u, v, _, w=w: w.get((u, v), w.get((v, u))))
            except nx.NetworkXNoPath:
                delay, path = np.inf, []
            row[f"delay_{hd * 1e6:g}us"] = delay
            if hd == p.terminal.hop_delay:
                row["hops"] = max(len(path) - 1, 0)
                row["path"] = ";".join(map(str, path))
                # sticky routing: keep the old path while valid and within the slack
                d_old = _path_delay(H, sticky) if sticky else np.inf
                if not path:
                    sticky = None
                elif d_old > delay + STICKY_SLACK:
                    sticky = path
                row["sticky_path"] = ";".join(map(str, sticky)) if sticky else ""
                row["sticky_delay"] = _path_delay(H, sticky) if sticky else np.inf
        rows.append(row)
    return rows


def exp_paths(days):
    # full candidate graph: the Yao sparsification picks a different link set every minute,
    # which would end paths whose links are still physically available
    p0 = scenario("leo")
    p0 = replace(p0, geometry=replace(p0.geometry, yao_cones=0))
    for d, wx in itertools.product(days, ("clear", "era5")):
        p = replace(p0, clouds=replace(p0.clouds, model=wx))
        states, t0, t1 = _day(d)
        states = select_fleet(states, p.airliners_only, p.equipage, p.seed)
        states = states[(states.time >= t0) & (states.time < t1)]
        snaps = list(states.groupby("time"))
        n = WORKERS  # contiguous blocks keep the sticky state
        blocks = np.array_split(np.arange(len(snaps)), n)
        chunks = [([snaps[i] for i in b], p, (t0 - 3600.0, t1 + 3600.0)) for b in blocks]
        with ProcessPoolExecutor(n) as ex:
            rows = [r for part in ex.map(_paths_chunk, chunks) for r in part]
        _save(pd.DataFrame(rows).assign(day=d, clouds=wx), f"paths_{d}_{wx}")


# --- solver check -----------------------------------------------------------------------


def _solver_one(args):
    snap, t, k = args
    out = []
    for cones in (8, 0):
        p = scenario("leo", k)
        p = replace(p, geometry=replace(p.geometry, yao_cones=cones))
        G = with_tech(build_graph(snap, p), p.terminal.r_max, k)
        for metric, H in (("ocean", ocean_graph(G, p.terminal.r_max)), ("e2e", G)):
            tic = time.time()
            heur = flow.widest_path_heuristic(H)[0]
            t_heur = time.time() - tic
            tic = time.time()
            milp = flow.milp_max_flow(flow.prune(H), time_limit=300.0)
            out.append({
                "time": t, "k": k, "yao_cones": cones, "metric": metric,
                "n_edges": H.number_of_edges(), "heuristic": heur, "t_heuristic": t_heur,
                "milp": milp["value"], "milp_bound": milp["bound"], "milp_status": milp["status"],
                "t_milp": time.time() - tic,
                "bound_relaxed": flow.max_flow_node_capacity(H, p.terminal.r_max),
                "bound_unconstrained": flow.max_flow_unconstrained(H)[0],
            })
    return out


def exp_solver(days):
    for d in days:
        states, t0, t1 = _day(d)
        states = select_fleet(states, True, 1.0, 0)
        times = np.arange(t0, t1, 3600.0)
        jobs = [(states[states.time == t], t, k) for t in times for k in (2, 4)]
        with ProcessPoolExecutor(WORKERS) as ex:
            rows = [r for part in ex.map(_solver_one, jobs) for r in part]
        _save(pd.DataFrame(rows).assign(day=d), f"solver_{d}")


EXPERIMENTS = {"tech": exp_tech, "equipage": exp_equipage, "weather": exp_weather,
               "paths": exp_paths, "solver": exp_solver}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=EXPERIMENTS)
    ap.add_argument("--days", nargs="+", default=list(DAYS), choices=list(DAYS))
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    WORKERS = a.workers
    EXPERIMENTS[a.experiment](a.days)
