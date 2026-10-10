import itertools

import networkx as nx
import numpy as np
import pytest

from hap_atlantic import flow
from hap_atlantic.topology import DST, SRC


def _toy_graph(K=2):
    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    for v in "abcd":
        G.add_node(v, kind="aircraft", terminals=K)
    G.add_edge(SRC, "a", kind="fiber", cap=np.inf, delay=0.0)
    G.add_edge(SRC, "b", kind="fiber", cap=np.inf, delay=0.0)
    for u, v in [("a", "c"), ("b", "d"), ("a", "d"), ("b", "c")]:
        G.add_edge(u, v, kind="air", cap=10.0, delay=1.0)
    G.add_edge("c", DST, kind="fiber", cap=np.inf, delay=0.0)
    G.add_edge("d", DST, kind="fiber", cap=np.inf, delay=0.0)
    return G


def _random_graph(seed, n=7, p=0.45, K=2):
    """Random capacity graph (5 or 10 Gbit/s links): SRC/DST unlimited, aircraft with K."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    nodes = list(range(n))
    for v in nodes:
        G.add_node(v, kind="aircraft", terminals=K)
    for u, v in itertools.combinations(nodes, 2):
        if rng.random() < p:
            G.add_edge(u, v, kind="air", cap=float(rng.choice([5e9, 10e9])),
                       delay=float(rng.uniform(1, 3)))
    for v in nodes[:3]:
        G.add_edge(SRC, v, kind="ground", cap=10e9, delay=1.0)
    for v in nodes[-3:]:
        G.add_edge(v, DST, kind="ground", cap=10e9, delay=1.0)
    return G


def _brute_force(G):
    """Max flow over all link subsets that respect the terminal limits (tiny graphs only)."""
    edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] != "fiber"]
    best = 0.0
    for mask in itertools.product([0, 1], repeat=len(edges)):
        act = [e for e, m in zip(edges, mask, strict=True) if m]
        deg = {}
        for u, v in act:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        if any(k > G.nodes[n].get("terminals", np.inf) for n, k in deg.items()):
            continue
        H = G.edge_subgraph(act + [(u, v) for u, v, d in G.edges(data=True)
                                   if d["kind"] == "fiber"]).copy()
        if SRC in H and DST in H:
            best = max(best, flow.max_flow_unconstrained(H)[0])
    return best


def test_flow_methods_agree_on_toy():
    ub, _ = flow.max_flow_unconstrained(_toy_graph())
    assert ub == pytest.approx(40.0)
    G1 = _toy_graph(K=1)  # one terminal: only a perfect matching (2 links) can be active
    assert flow.widest_path_heuristic(G1)[0] == pytest.approx(20.0)
    assert flow.milp_max_flow(G1, scale=1.0)["value"] == pytest.approx(20.0)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("K", [2, 4])
def test_bound_ordering_on_random_graphs(seed, K):
    G = _random_graph(seed, K=K)
    heur = flow.widest_path_heuristic(G)[0]
    milp = flow.milp_max_flow(G, rel_gap=0.0)["value"]
    bound = flow.max_flow_node_capacity(G, r_max=10e9)
    unc = flow.max_flow_unconstrained(G)[0]
    tol = 1e3  # bit/s
    assert heur <= milp + tol and milp <= bound + tol and bound <= unc + tol


@pytest.mark.parametrize("seed", range(2))
def test_milp_matches_brute_force(seed):
    G = _random_graph(seed, n=5, p=0.5)
    assert flow.milp_max_flow(G, rel_gap=0.0)["value"] == pytest.approx(_brute_force(G),
                                                                         rel=1e-6)


@pytest.mark.parametrize("seed", range(2))
def test_prune_keeps_exactly_nodes_on_simple_paths(seed):
    G = _random_graph(seed, n=8, p=0.3)
    on_path = {SRC, DST}
    for path in nx.all_simple_paths(G, SRC, DST):
        on_path |= set(path)
    if not nx.has_path(G, SRC, DST):
        on_path = {SRC, DST}
    assert set(flow.prune(G).nodes) == on_path


@pytest.mark.parametrize("seed", range(2))
def test_capacity_within_gap_of_bound(seed):
    G = _random_graph(seed, n=8)
    r = flow.capacity(G, r_max=10e9)
    assert r["heuristic"] <= r["value"] <= r["bound"] * (1 + 1e-9)
    assert r["value"] >= (1 - flow.GAP) * r["bound"] * (1 - 1e-9)


def test_capacity_bound_never_below_value():
    # tiny capacities (far below the 1 Gbit/s scaling unit of the MILP)
    G = _random_graph(3, n=8)
    for *_, d in G.edges(data=True):
        d["cap"] /= 1e9
    r = flow.capacity(G, r_max=10.0)
    assert r["value"] <= r["bound"] * (1 + 1e-9)


def test_capacity_certified_and_prune():
    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    for gs in ("gsS0", "gsD0"):
        G.add_node(gs, kind="ground", terminals=2)
    for v in "abcde":
        G.add_node(v, kind="aircraft", terminals=2)
    G.add_edge(SRC, "gsS0", kind="fiber", cap=np.inf, delay=0.0)
    G.add_edge(DST, "gsD0", kind="fiber", cap=np.inf, delay=0.0)
    for u, v in [("gsS0", "a"), ("gsS0", "b"), ("c", "gsD0"), ("d", "gsD0")]:
        G.add_edge(u, v, kind="ground", cap=10.0, delay=1.0)
    for u, v in [("a", "c"), ("b", "d"), ("a", "d"), ("b", "c"), ("a", "e")]:
        G.add_edge(u, v, kind="air", cap=10.0, delay=1.0)
    r = flow.capacity(G, r_max=10.0)
    assert r["value"] == pytest.approx(20.0) and r["bound"] >= r["value"] - 1e-9
    assert flow.milp_max_flow(G, scale=1.0)["value"] == pytest.approx(20.0)
    assert "e" not in flow.prune(G)  # dead end


def test_node_bound_single_relay():
    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    for k in (2, 3, 4):
        G.add_node("x", kind="aircraft", terminals=k)
        for i in range(4):
            G.add_edge(SRC, f"s{i}", kind="fiber", cap=np.inf, delay=0.0)
            G.add_node(f"s{i}", kind="endpoint", terminals=np.inf)
            G.add_edge(f"s{i}", "x", kind="air", cap=10.0, delay=1.0)
            G.add_node(f"d{i}", kind="endpoint", terminals=np.inf)
            G.add_edge("x", f"d{i}", kind="air", cap=10.0, delay=1.0)
            G.add_edge(f"d{i}", DST, kind="fiber", cap=np.inf, delay=0.0)
        assert flow.max_flow_node_capacity(G, 10.0) == pytest.approx((k // 2) * 10.0)
        assert flow.milp_max_flow(G, scale=1.0)["value"] == pytest.approx((k // 2) * 10.0)


def test_no_path():
    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    assert flow.capacity(G, 10.0)["method"] == "no path"
    assert flow.min_latency_path(G) == (np.inf, [])
    assert flow.milp_max_flow(G)["value"] == 0.0


def test_min_latency_path_respects_rate_and_delay():
    G = nx.Graph()
    G.add_edge(SRC, "a", cap=10e9, delay=1.0)
    G.add_edge("a", DST, cap=10e9, delay=1.0)
    G.add_edge(SRC, "b", cap=0.5e9, delay=0.1)  # fast but too slow a link
    G.add_edge("b", DST, cap=10e9, delay=0.1)
    delay, path = flow.min_latency_path(G, min_rate=1e9)
    assert delay == pytest.approx(2.0) and path == [SRC, "a", DST]
    assert flow.min_latency_path(G, min_rate=0.1e9)[1] == [SRC, "b", DST]


def test_heuristic_paths_respect_terminals():
    G = _random_graph(3, n=8, K=2)
    total, paths = flow.widest_path_heuristic(G)
    assert paths
    used = {}
    seen = set()
    for _, _, path in paths:
        for u, v in itertools.pairwise(path):
            e = frozenset((u, v))
            if e in seen or G[u][v]["kind"] == "fiber":
                continue
            seen.add(e)
            for n in (u, v):
                used[n] = used.get(n, 0) + 1
    assert all(k <= G.nodes[n]["terminals"] for n, k in used.items())
    assert total == pytest.approx(sum(w for w, _, _ in paths))
