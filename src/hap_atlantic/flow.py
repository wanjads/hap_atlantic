"""Throughput and latency between SRC and DST on a candidate graph.

Every aircraft carries K optical terminals, so at most K of its candidate links can be
active at the same time (degree constraint). Methods, from loosest to tightest:

* `max_flow_unconstrained`  ignores the terminal limit (upper bound 1)
* `max_flow_node_capacity`  relaxes it to a relay capacity of K/2 * r_max per node (bound 2)
* `milp_max_flow`           exact terminal assignment + routing (MILP)
* `widest_path_heuristic`   successive widest paths with terminal bookkeeping (scalable)

`capacity` is what the evaluation reports: the heuristic value if it already meets bound 2
(then it is optimal), otherwise the MILP on the graph pruned to nodes that lie on some
SRC-DST path. The bounds refer to the candidate graph they are given; on a Yao-sparsified
graph they bound the sparsified problem only.
"""

from __future__ import annotations

import heapq
import itertools

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .topology import DST, SRC

EPS = 1e-6


def _digraph(G: nx.Graph, node_cap=None) -> nx.DiGraph:
    D = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        cap = d["cap"]
        D.add_edge(u, v, capacity=cap)
        D.add_edge(v, u, capacity=cap)
    if node_cap is not None:  # node splitting: v_in -> v_out with capacity node_cap(v)
        S = nx.DiGraph()
        for u, v, d in D.edges(data=True):
            S.add_edge((u, "out"), (v, "in"), capacity=d["capacity"])
        for v in D.nodes:
            S.add_edge((v, "in"), (v, "out"), capacity=node_cap(v))
        return S
    return D


def _inf_to_large(D: nx.DiGraph) -> None:
    finite = [c for *_, c in D.edges(data="capacity") if np.isfinite(c)]
    big = 10 * (sum(finite) + 1.0)
    for *_, d in D.edges(data=True):
        if not np.isfinite(d["capacity"]):
            d["capacity"] = big


def max_flow_unconstrained(G: nx.Graph) -> tuple[float, set]:
    """Max flow ignoring terminal limits. Returns (value, source side of the min cut)."""
    if not (G.has_node(SRC) and G.has_node(DST)) or not nx.has_path(G, SRC, DST):
        return 0.0, {SRC}
    D = _digraph(G)
    _inf_to_large(D)
    value, (side, _) = nx.minimum_cut(D, SRC, DST, flow_func=nx.algorithms.flow.preflow_push)
    return float(value), side


def max_flow_node_capacity(G: nx.Graph, r_max: float) -> float:
    """Max flow with relay capacity floor(K/2)*r_max per aircraft and K*r_max per station."""
    if not nx.has_path(G, SRC, DST):
        return 0.0

    def cap(v):
        attrs = G.nodes[v]
        k = attrs.get("terminals", np.inf)
        if attrs.get("kind") == "aircraft":
            return (k // 2) * r_max
        if attrs.get("kind") == "ground":
            return k * r_max
        return np.inf

    D = _digraph(G, node_cap=cap)
    _inf_to_large(D)
    value, _ = nx.maximum_flow(D, (SRC, "out"), (DST, "in"),
                               flow_func=nx.algorithms.flow.preflow_push)
    return float(value)


def min_latency_path(G: nx.Graph, min_rate: float = 1e9) -> tuple[float, list]:
    """Lowest one-way delay path using only links with capacity >= min_rate."""
    H = nx.subgraph_view(G, filter_edge=lambda u, v: G[u][v]["cap"] >= min_rate)
    try:
        delay, path = nx.single_source_dijkstra(H, SRC, DST, weight="delay")
    except nx.NetworkXNoPath:
        return np.inf, []
    return float(delay), path


def prune(G: nx.Graph) -> nx.Graph:
    """Subgraph of the nodes that lie on at least one simple SRC-DST path.

    With a virtual SRC-DST edge these are exactly the nodes of the biconnected component
    that contains it.
    """
    H = nx.Graph(G.edges)
    H.add_edge(SRC, DST)
    for comp in nx.biconnected_components(H):
        if SRC in comp and DST in comp:
            return G.subgraph(comp).copy()
    return G.subgraph([SRC, DST]).copy()


GAP = 0.01  # accepted relative gap to the upper bound


def capacity(G: nx.Graph, r_max: float, time_limit: float = 60.0) -> dict:
    """Max SRC-DST flow under the terminal constraint (see module docstring).

    Stops as soon as a solution within GAP (1 %) of the upper bound is known: the heuristic
    if it already meets the relaxation bound, otherwise the MILP with mip_rel_gap = GAP (or
    the best solution found within time_limit). Returns value, upper bound, heuristic value
    and the method that produced the value.
    """
    if not (G.has_node(SRC) and G.has_node(DST)) or not nx.has_path(G, SRC, DST):
        return {"value": 0.0, "bound": 0.0, "heuristic": 0.0, "method": "no path"}
    heur, _ = widest_path_heuristic(G)
    bound = max_flow_node_capacity(G, r_max)
    if heur >= bound * (1 - GAP):
        return {"value": heur, "bound": bound, "heuristic": heur, "method": "heuristic"}
    r = milp_max_flow(prune(G), time_limit=time_limit, rel_gap=GAP)
    value = max(r["value"], heur)
    ub = min(bound, r["bound"]) if np.isfinite(r["bound"]) else bound
    return {"value": value, "bound": ub, "heuristic": heur, "method": "milp"}


# ----------------------------------------------------------------------------------------
# Heuristic
# ----------------------------------------------------------------------------------------


def _terminal_cost(G, u, v) -> int:
    return 0 if G[u][v]["kind"] == "fiber" else 1


def widest_path_heuristic(G: nx.Graph, max_paths: int = 10_000):
    """Successive widest (max-bottleneck) paths under the terminal constraint.

    A link is either active (terminals already pointed at each other, residual capacity
    left) or inactive; an inactive link may be activated if both ends have a free terminal.
    Ties in the bottleneck are broken by delay. Returns (total flow, list of
    (rate, delay, path)).
    """
    free = {v: G.nodes[v].get("terminals", np.inf) for v in G.nodes}
    active: set[frozenset] = set()
    res: dict[tuple, float] = {}
    for u, v, d in G.edges(data=True):
        res[(u, v)] = res[(v, u)] = d["cap"]
        if d["kind"] == "fiber":
            active.add(frozenset((u, v)))
    paths, total = [], 0.0
    banned: set[frozenset] = set()

    for _ in range(max_paths):
        found = _widest_path(G, free, active, res, banned)
        if found is None:
            break
        width, delay, path = found
        need: dict = {}
        new_edges = []
        for u, v in itertools.pairwise(path):
            e = frozenset((u, v))
            if e not in active:
                new_edges.append(e)
                need[u] = need.get(u, 0) + 1
                need[v] = need.get(v, 0) + 1
        bad = [n for n, k in need.items() if k > free[n]]
        if bad:  # node needs two new terminals but has one: ban one of its new links
            n = bad[0]
            banned.add(next(e for e in new_edges if n in e))
            continue
        banned.clear()
        for e in new_edges:
            active.add(e)
        for n, k in need.items():
            free[n] -= k
        for u, v in itertools.pairwise(path):
            res[(u, v)] -= width
        paths.append((width, delay, path))
        total += width
    return total, paths


def _widest_path(G, free, active, res, banned):
    best = {SRC: (np.inf, 0.0)}
    prev = {}
    heap = [(-np.inf, 0.0, 0, SRC)]
    counter = itertools.count(1)
    done = set()
    while heap:
        negw, delay, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == DST:
            path = [DST]
            while path[-1] != SRC:
                path.append(prev[path[-1]])
            return -negw, delay, path[::-1]
        for v in G.neighbors(u):
            if v in done:
                continue
            e = frozenset((u, v))
            r = res[(u, v)]
            if r <= EPS or e in banned:
                continue
            if e not in active and (free[u] < 1 or free[v] < 1):
                continue
            w = min(-negw, r)
            dl = delay + G[u][v]["delay"]
            bw, bd = best.get(v, (0.0, np.inf))
            if w > bw + EPS or (abs(w - bw) <= EPS and dl < bd):
                best[v] = (w, dl)
                prev[v] = u
                heapq.heappush(heap, (-w, dl, next(counter), v))
    return None


# ----------------------------------------------------------------------------------------
# Exact MILP
# ----------------------------------------------------------------------------------------


def milp_max_flow(G: nx.Graph, time_limit: float = 60.0, scale: float = 1e9,
                  rel_gap: float = 1e-3) -> dict:
    """Exact joint terminal assignment and routing (HiGHS via scipy.optimize.milp).

    Variables: x_e in {0,1} per non-fibre link, f_uv, f_vu >= 0 per link, F total flow.
    max F  s.t.  f_uv, f_vu <= c_e x_e;  flow conservation;  sum_{e at v} x_e <= K_v.
    Capacities are scaled to Gbit/s for numerical conditioning.
    """
    if not nx.has_path(G, SRC, DST):
        return {"value": 0.0, "bound": 0.0, "status": "no path"}
    edges = list(G.edges(data=True))
    nodes = list(G.nodes)
    term_edges = [k for k, (_, _, d) in enumerate(edges) if d["kind"] != "fiber"]
    x_of = {k: j for j, k in enumerate(term_edges)}
    n_x, n_e = len(term_edges), len(edges)
    f0 = n_x  # f_uv at f0 + 2k, f_vu at f0 + 2k + 1
    iF = f0 + 2 * n_e
    n_var = iF + 1

    finite = [d["cap"] for *_, d in edges if np.isfinite(d["cap"])]
    big = 10 * sum(finite) / scale + 1.0
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # capacity coupling
    for k, (_, _, d) in enumerate(edges):
        if k not in x_of:
            continue
        c = d["cap"] / scale
        for s in (0, 1):
            rows += [r, r]
            cols += [f0 + 2 * k + s, x_of[k]]
            vals += [1.0, -c]
            lo.append(-np.inf)
            hi.append(0.0)
            r += 1
    # flow conservation: out - in = F at SRC, -F at DST, 0 elsewhere
    cons_row = {v: r + i for i, v in enumerate(nodes)}
    for k, (u, v, _) in enumerate(edges):
        for s, (a, b) in enumerate(((u, v), (v, u))):
            rows += [cons_row[a], cons_row[b]]
            cols += [f0 + 2 * k + s, f0 + 2 * k + s]
            vals += [1.0, -1.0]
    for v in nodes:
        if v in (SRC, DST):
            rows.append(cons_row[v])
            cols.append(iF)
            vals.append(-1.0 if v == SRC else 1.0)
        lo.append(0.0)
        hi.append(0.0)
    r += len(nodes)
    # terminal limits
    for v in nodes:
        K = G.nodes[v].get("terminals", np.inf)
        if not np.isfinite(K):
            continue
        ks = [x_of[k] for k, (a, b, _) in enumerate(edges) if k in x_of and v in (a, b)]
        if len(ks) <= K:
            continue
        rows += [r] * len(ks)
        cols += ks
        vals += [1.0] * len(ks)
        lo.append(-np.inf)
        hi.append(float(K))
        r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    ub = np.full(n_var, np.inf)
    ub[:n_x] = 1.0
    for k, (_, _, d) in enumerate(edges):
        if k not in x_of:
            ub[f0 + 2 * k: f0 + 2 * k + 2] = big if not np.isfinite(d["cap"]) else d["cap"] / scale
    integrality = np.zeros(n_var)
    integrality[:n_x] = 1
    c = np.zeros(n_var)
    c[iF] = -1.0
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(np.zeros(n_var), ub),
               options={"time_limit": time_limit, "mip_rel_gap": rel_gap})
    value = -res.fun * scale if res.x is not None else 0.0
    bound = getattr(res, "mip_dual_bound", None)
    return {
        "value": float(value),
        "bound": float(-bound * scale) if bound is not None else np.nan,
        "status": res.message,
    }
