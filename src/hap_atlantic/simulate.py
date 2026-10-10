"""Time-stepped simulation: candidate graphs and metrics per snapshot.

Per snapshot two graphs are built once (uncapped link-budget rates incl. cloud loss):

* latency graph   Frankfurt -> New York (NY4) via the optical / RF ground stations at the
                  endpoints: minimum one-way latency and availability of a path;
* capacity graph  ground stations everywhere on land (Europe -> North America): the
                  transatlantic capacity, for several technology configurations (r_max,
                  transmit-power scale, terminals per aircraft K, see `topology.with_tech`).
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import flow
from .config import Params
from .topology import DST, SRC, build_graph, land_graph, with_tech
from .traffic import select_fleet
from .weather import CloudField


@dataclass(frozen=True)
class Config:
    """Technology configuration evaluated on a candidate graph."""

    label: str
    r_max: float  # transceiver rate per link [bit/s]
    p_scale: float  # transmit power relative to the one the graph was built with
    k: int  # terminals per aircraft


def config(r_max: float, k: int, p_tx: float, p: Params) -> Config:
    """Configuration with rate r_max per link, k terminals and transmit power p_tx [W]."""
    return Config(f"r{r_max:.1e}_k{k}", r_max, p_tx / p.terminal.p_tx, k)


def _ground_links(G, endpoint) -> int:
    return sum(1 for gs in G.neighbors(endpoint) for v in G.neighbors(gs) if v != endpoint)


def evaluate_snapshot(G: nx.Graph, L: nx.Graph, p: Params, configs,
                      milp_time: float = 60.0) -> dict:
    """G: latency graph (Frankfurt - New York), L: capacity graph (land - land)."""
    delay, path = flow.min_latency_path(G)
    m = {
        "n_aircraft": sum(1 for _, k in G.nodes(data="kind") if k == "aircraft"),
        "n_air_links": sum(1 for *_, k in G.edges(data="kind") if k == "air"),
        "eu_access": L.degree(SRC),
        "na_access": L.degree(DST),
        "na_access_rf": sum(1 for *_, md in L.edges(DST, data="medium") if md == "rf"),
        "fra_ground_links": _ground_links(G, SRC),
        "ny_ground_links": _ground_links(G, DST),
        "ny_rf_links": sum(1 for gs in G.neighbors(DST) for *_, md in G.edges(gs, data="medium")
                           if md == "rf"),
        "min_link_rate": min((d["rate"] for *_, d in L.edges(data=True)), default=np.nan),
        "connected": bool(np.isfinite(delay)),
        "min_delay": delay,
        "min_delay_hops": max(len(path) - 1, 0),
        "path": ";".join(map(str, path)),
    }
    for c in configs:
        r = flow.capacity(with_tech(L, c.r_max, c.k, c.p_scale), c.r_max, time_limit=milp_time)
        m[f"cap_{c.label}"] = r["value"]
        m[f"cap_{c.label}_heur"] = r["heuristic"]
        m[f"cap_{c.label}_bound"] = r["bound"]
    return m


def _chunk(args):
    snaps, p, configs, t_range, milp_time = args
    clouds = (CloudField(*t_range, p.clouds, p.seed, p.geometry.earth_radius)
              if p.clouds.model == "era5" else None)
    rows = []
    for t, snap in snaps:
        cs = clouds.at(t) if clouds is not None else None
        G = build_graph(snap, p, cs)
        rows.append({"time": t, **evaluate_snapshot(G, land_graph(G, p, cs), p, configs,
                                                     milp_time)})
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
    """Limit the number of processes so that they use at most 75 % of the free memory.

    Measured peak per worker: ~0.8 GB clear sky, ~1.8 GB with 3D ERA5 clouds.
    """
    per_worker = (2.0 if clouds else 1.0) * 1024**3
    return max(1, min(workers, int(0.75 * _mem_available() / per_worker)))


def run(states: pd.DataFrame, p: Params, configs, t0: float, t1: float, every: int = 10,
        workers: int = 1, milp_time: float = 60.0) -> pd.DataFrame:
    """Evaluate every `every`-th grid time step in [t0, t1).

    milp_time: time limit per MILP; if it is hit, the value is the best solution found and
    the `_bound` column holds the upper bound.
    """
    times = np.sort(states["time"].unique())
    times = times[(times >= t0) & (times < t1)][::every]
    states = states[states["time"].isin(times)]  # only the evaluated snapshots
    states = select_fleet(states, p.airliners_only)
    by_t = {t: g for t, g in states.groupby("time")}
    snaps = [(t, by_t.get(t, states.iloc[:0])) for t in times]
    t_range = (t0 - 3600.0, t1 + 3600.0)  # weather window (whole days are fetched)
    n = max(1, min(safe_workers(workers, p.clouds.model == "era5"), len(snaps)))
    chunks = [(snaps[i::n], p, configs, t_range, milp_time) for i in range(n)]
    if n == 1:
        rows = _chunk(chunks[0])
    else:
        # spawn: fresh workers that neither inherit (and copy) the memory of the caller nor
        # its thread state (fork after pyarrow threads can deadlock)
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
            rows = [r for part in ex.map(_chunk, chunks) for r in part]
    return pd.DataFrame(rows).sort_values("time", ignore_index=True)
