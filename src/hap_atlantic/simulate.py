"""Time-stepped simulation: one candidate graph and one set of metrics per snapshot.

Per snapshot the candidate graph is built once (uncapped link-budget rates, clouds
applied) and then evaluated for several technology configurations (r_max, transmit-power
scale, terminals per aircraft K) with `topology.with_tech`. Three capacities are reported:

* ocean     the oceanic section alone: aircraft east of 15 W inject, west of 55 W extract
* backbone  Frankfurt -> New York with unlimited terminals at the ground stations
* e2e       Frankfurt -> New York with the configured ground stations
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import flow
from .config import TECHS, Params
from .topology import DST, SRC, build_graph, with_tech
from .traffic import select_fleet
from .weather import CloudField

OCEAN_LON_EAST, OCEAN_LON_WEST = -15.0, -55.0  # Shanwick / Gander oceanic boundaries
METRICS = ("ocean", "backbone", "e2e")


@dataclass(frozen=True)
class Config:
    """Technology configuration evaluated on a candidate graph."""

    label: str
    r_max: float
    p_scale: float  # transmit power relative to the one the graph was built with
    k: int


def tech_config(name: str, k: int, p: Params) -> Config:
    t = TECHS[name]
    return Config(f"{name}_k{k}", t.r_max, t.p_tx / p.terminal.p_tx, k)


def ocean_graph(G: nx.Graph, r_access: float, lon_east: float = OCEAN_LON_EAST,
                lon_west: float = OCEAN_LON_WEST) -> nx.Graph:
    """Graph for the oceanic capacity.

    Every aircraft east of lon_east may inject traffic and every aircraft west of lon_west
    may extract it; such an access link costs one terminal and carries r_access, as if it
    came from a continental neighbour. Links between two aircraft on the same side are
    dropped: a direct access link dominates them.
    """
    side = {v: (1 if d["lon"] > lon_east else -1 if d["lon"] < lon_west else 0)
            for v, d in G.nodes(data=True) if d.get("kind") == "aircraft"}
    H = nx.Graph()
    H.add_node(SRC, kind="endpoint", terminals=np.inf)
    H.add_node(DST, kind="endpoint", terminals=np.inf)
    for u, v, d in G.edges(data=True):
        if d["kind"] != "air" or (side[u] != 0 and side[u] == side[v]):
            continue
        H.add_edge(u, v, **d)
    for v in list(H.nodes):
        if v in (SRC, DST):
            continue
        H.nodes[v].update(G.nodes[v])
        if side[v]:
            H.add_edge(SRC if side[v] == 1 else DST, v, kind="air", rate=r_access,
                       cap=r_access, delay=0.0)
    return H


def _backbone(G: nx.Graph) -> nx.Graph:
    B = G.copy()
    for v, kind in B.nodes(data="kind"):
        if kind == "ground":
            B.nodes[v]["terminals"] = np.inf
    return B


def _ground_links(G, endpoint) -> int:
    return sum(1 for gs in G.neighbors(endpoint) for v in G.neighbors(gs) if v != endpoint)


def evaluate_snapshot(G: nx.Graph, p: Params, configs, metrics=METRICS,
                      milp_time: float = 60.0) -> dict:
    n_air = sum(1 for _, k in G.nodes(data="kind") if k == "aircraft")
    delay, path = flow.min_latency_path(G)
    m = {
        "n_aircraft": n_air,
        "n_air_links": sum(1 for *_, k in G.edges(data="kind") if k == "air"),
        "src_ground_links": _ground_links(G, SRC),
        "dst_ground_links": _ground_links(G, DST),
        "src_rf_links": sum(1 for gs in G.neighbors(SRC) for _, _, md in
                            G.edges(gs, data="medium") if md == "rf"),
        "n_clear_sites_src": sum(1 for gs in G.neighbors(SRC) if G.nodes[gs].get("clear")),
        "n_clear_sites_dst": sum(1 for gs in G.neighbors(DST) if G.nodes[gs].get("clear")),
        "min_link_rate": min((d["rate"] for *_, d in G.edges(data=True)
                              if d["kind"] != "fiber"), default=np.nan),
        "connected": bool(np.isfinite(delay)),
        "min_delay": delay,
        "min_delay_hops": max(len(path) - 1, 0),
    }
    for c in configs:
        H = with_tech(G, c.r_max, c.k, c.p_scale)
        graphs = {"ocean": lambda H=H, c=c: ocean_graph(H, c.r_max),
                  "backbone": lambda H=H: _backbone(H), "e2e": lambda H=H: H}
        for name in metrics:
            r = flow.capacity(graphs[name](), c.r_max, time_limit=milp_time)
            m[f"{name}_{c.label}"] = r["value"]
            m[f"{name}_{c.label}_heur"] = r["heuristic"]
            m[f"{name}_{c.label}_bound"] = r["bound"]
    return m


def _chunk(args):
    snaps, p, configs, metrics, t_range, milp_time = args
    clouds = (CloudField(*t_range, p.clouds, p.seed, p.geometry.earth_radius)
              if p.clouds.model == "era5" else None)
    rows = []
    for t, snap in snaps:
        G = build_graph(snap, p, clouds.at(t) if clouds is not None else None)
        rows.append({"time": t, **evaluate_snapshot(G, p, configs, metrics, milp_time)})
    return rows


def _mem_available() -> float:
    """MemAvailable from /proc/meminfo [bytes] (inf if unknown)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    return float("inf")


def safe_workers(workers: int, clouds: bool) -> int:
    """Limit the number of processes so that they use at most 60 % of the free memory.

    Measured peak per worker: ~0.8 GB clear sky, ~1.8 GB with 3D ERA5 clouds.
    """
    per_worker = (2.0 if clouds else 1.0) * 1024**3
    return max(1, min(workers, int(0.6 * _mem_available() / per_worker)))


def run(states: pd.DataFrame, p: Params, configs, t0: float, t1: float, every: int = 10,
        metrics=METRICS, workers: int = 1, milp_time: float = 60.0) -> pd.DataFrame:
    """Evaluate every `every`-th grid time step in [t0, t1).

    milp_time: time limit per MILP; if it is hit, the value is the best solution found and
    the `_bound` column holds the upper bound.
    """
    states = select_fleet(states, p.airliners_only, p.equipage, p.seed)
    times = np.sort(states["time"].unique())
    times = times[(times >= t0) & (times < t1)][::every]
    by_t = {t: g for t, g in states[states["time"].isin(times)].groupby("time")}
    snaps = [(t, by_t.get(t, states.iloc[:0])) for t in times]
    t_range = (t0 - 3600.0, t1 + 3600.0)  # weather window (whole days are fetched)
    n = max(1, min(safe_workers(workers, p.clouds.model == "era5"), len(snaps)))
    chunks = [(snaps[i::n], p, configs, metrics, t_range, milp_time) for i in range(n)]
    if n == 1:
        rows = _chunk(chunks[0])
    else:
        with ProcessPoolExecutor(n) as ex:
            rows = [r for part in ex.map(_chunk, chunks) for r in part]
    return pd.DataFrame(rows).sort_values("time", ignore_index=True)
