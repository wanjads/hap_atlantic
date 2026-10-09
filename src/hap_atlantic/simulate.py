"""Time-stepped simulation: one candidate graph and one set of metrics per snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import flow
from .config import Params
from .topology import DST, SRC, CloudProcess, build_graph, ground_stations
from .traffic import snapshots


def _ground_links(G, endpoint) -> int:
    return sum(1 for gs in G.neighbors(endpoint) for v in G.neighbors(gs) if v != endpoint)


def evaluate_snapshot(G, p: Params, run_milp: bool = False, milp_time: float = 30.0) -> dict:
    n_air = sum(1 for _, k in G.nodes(data="kind") if k == "aircraft")
    n_links = sum(1 for *_, k in G.edges(data="kind") if k == "air")
    ub1, cut = flow.max_flow_unconstrained(G)
    ub2 = flow.max_flow_node_capacity(G, p.terminal.r_max)
    heur, paths = flow.widest_path_heuristic(G)
    delay, path = flow.min_latency_path(G)
    m = {
        "n_aircraft": n_air,
        "n_air_links": n_links,
        "connected": bool(np.isfinite(delay)),
        "flow_unconstrained": ub1,
        "flow_relaxed": ub2,
        "flow_heuristic": heur,
        "n_paths": len(paths),
        "min_delay": delay,
        "min_delay_hops": max(len(path) - 1, 0),
        "mincut_aircraft": sum(1 for v in cut if G.nodes[v].get("kind") == "aircraft"),
        "src_ground_links": _ground_links(G, SRC),
        "dst_ground_links": _ground_links(G, DST),
    }
    if run_milp:
        r = flow.milp_max_flow(G, time_limit=milp_time)
        m["flow_milp"], m["flow_milp_bound"] = r["value"], r["bound"]
    return m


def run(grid_states: pd.DataFrame, p: Params, run_milp: bool = False,
        progress: bool = True, every: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(p.seed)
    clouds = CloudProcess(len(ground_stations(p)), p.ground.p_cloud_free,
                          p.ground.cloud_corr_time, p.dt, rng)
    rows = []
    for k, (t, snap) in enumerate(snapshots(grid_states)):
        clear = clouds.step()
        if k % every:
            continue
        G = build_graph(snap, p, clear)
        rows.append({"time": t, "n_clear_sites_src": int(clear[: p.ground.n_sites].sum()),
                     "n_clear_sites_dst": int(clear[p.ground.n_sites:].sum()),
                     **evaluate_snapshot(G, p, run_milp)})
        if progress and len(rows) % 50 == 0:
            print(f"  {len(rows)} snapshots, t={t:.0f}, heur={rows[-1]['flow_heuristic']/1e9:.1f}"
                  " Gbit/s", flush=True)
    return pd.DataFrame(rows)
