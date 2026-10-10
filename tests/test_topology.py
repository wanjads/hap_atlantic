from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hap_atlantic import geometry as geo
from hap_atlantic.config import C0, N_AIR, N_FIBER, Params
from hap_atlantic.topology import (
    DST,
    SRC,
    build_graph,
    ground_stations,
    land_graph,
    with_tech,
    yao_sparsify,
)

P = Params()
R = P.geometry.earth_radius


def _snapshot(n=60, seed=0, alt=11e3):
    """Aircraft spread along the Frankfurt-New York great circle (cruise altitude)."""
    rng = np.random.default_rng(seed)
    f = np.sort(rng.uniform(0, 1, n))
    lat, lon = geo.slerp_latlon(P.source.lat, P.source.lon, P.sink.lat, P.sink.lon, f)
    lat = lat + rng.normal(0, 1.0, n)
    return pd.DataFrame({"time": 0.0, "icao24": [f"{i:06x}" for i in range(n)], "lat": lat,
                         "lon": lon, "alt": np.full(n, alt)})


def test_ground_stations_counts_and_delays():
    gs = ground_stations(P)
    for end in (SRC, DST):
        own = [g for g in gs if g.endpoint == end]
        assert 1 <= len(own) <= 1 + len(P.ground.site_rings) * P.ground.site_bearings
        assert own[0].fiber_delay == 0.0
    ring = [g for g in gs if g.fiber_delay > 0]
    allowed = {r * P.ground.fiber_detour * N_FIBER / C0 for r in P.ground.site_rings}
    assert all(min(abs(g.fiber_delay - a) for a in allowed) < 1e-12 for g in ring)


def test_ground_stations_only_on_land():
    # synthetic mask: no land between 60 and 10 deg W
    assert all(not (-60.0 < g.site.lon < -10.0) for g in ground_stations(P))


def test_ground_stations_without_rings():
    p = replace(P, ground=replace(P.ground, site_rings=()))
    assert len(ground_stations(p)) == 2


def test_yao_keeps_nearest_and_farthest_per_cone():
    # node 0 at the centre, four nodes to the east at increasing distance
    lat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    lon = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pairs = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    d = geo.great_circle(0.0, 0.0, lat[pairs[:, 1]], lon[pairs[:, 1]], R)
    keep = yao_sparsify(pairs, d, lat, lon, cones=8)
    # from node 0, (0,1) is nearest and (0,4) farthest; (0,2), (0,3) are kept only if they
    # are extreme for their other end, which they are (single neighbour)
    assert keep[0] and keep[3]
    assert keep.sum() == 4


def test_yao_drops_middle_links_of_a_dense_cone():
    lon = np.array([0.0, 1.0, 1.1, 1.2, 1.3])
    lat = np.zeros(5)
    pairs = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)])
    d = geo.great_circle(lat[pairs[:, 0]], lon[pairs[:, 0]], lat[pairs[:, 1]], lon[pairs[:, 1]], R)
    keep = yao_sparsify(pairs, d, lat, lon, cones=8)
    assert 0 < keep.sum() < len(pairs)


def test_build_graph_link_constraints():
    G = build_graph(_snapshot(), P)
    air = [(u, v, d) for u, v, d in G.edges(data=True) if d["kind"] == "air"]
    assert air
    for u, v, d in air:
        a = geo.to_ecef(G.nodes[u]["lat"], G.nodes[u]["lon"], G.nodes[u]["alt"], R)
        b = geo.to_ecef(G.nodes[v]["lat"], G.nodes[v]["lon"], G.nodes[v]["alt"], R)
        assert geo.chord_min_height(a, b, R) >= P.geometry.h_chord_min - 1e-6
        assert d["dist"] <= P.geometry.d_max_air
        assert d["rate"] >= P.terminal.r_min and d["cap"] <= P.terminal.r_max
        assert d["delay"] == pytest.approx(d["dist"] * N_AIR / C0 + P.terminal.hop_delay)


def test_build_graph_filters_low_and_far_aircraft():
    snap = _snapshot(n=10)
    extra = pd.DataFrame({"time": 0.0, "icao24": ["low", "far"], "lat": [48.0, 20.0],
                          "lon": [-30.0, -30.0], "alt": [500.0, 11e3]})
    G = build_graph(pd.concat([snap, extra]), P)
    assert "low" not in G and "far" not in G


def test_aircraft_below_ray_height_cannot_relay():
    snap = _snapshot(n=40)
    snap.loc[0, "alt"] = 2_000.0  # between h_node_min and h_chord_min
    G = build_graph(snap, P)
    assert snap.icao24[0] in G
    assert all(d["kind"] != "air" for *_, d in G.edges(snap.icao24[0], data=True))


def test_build_graph_endpoints_and_fiber_edges():
    G = build_graph(_snapshot(), P)
    assert G.nodes[SRC]["terminals"] == np.inf
    fib = [d for *_, d in G.edges(data=True) if d["kind"] == "fiber"]
    assert len(fib) == len(ground_stations(P))
    assert all(d["cap"] == np.inf for d in fib)


def test_build_graph_without_aircraft():
    G = build_graph(_snapshot().iloc[:0], P)
    assert set(nx.node_connected_component(G, SRC)) & {DST} == set()


def test_with_tech_scales_and_caps():
    G = nx.Graph()
    G.add_node("a", kind="aircraft", terminals=2)
    G.add_node("b", kind="aircraft", terminals=2)
    G.add_node("g", kind="ground", terminals=4)
    G.add_edge("a", "b", kind="air", rate=50.0, cap=50.0)
    G.add_edge("g", "a", kind="ground", medium="rf", rate=10.0, cap=10.0)
    H = with_tech(G, r_max=100.0, k=4, p_scale=3.0)
    assert H["a"]["b"]["cap"] == 100.0 and H.nodes["a"]["terminals"] == 4
    assert H["g"]["a"]["cap"] == 10.0  # RF is not scaled with the optical power
    assert H.nodes["g"]["terminals"] == 4 and G["a"]["b"]["cap"] == 50.0  # copy


def test_land_graph_one_access_link_per_side():
    G = build_graph(_snapshot(n=80), P)
    L = land_graph(G, P)
    assert L.degree(SRC) > 0 and L.degree(DST) > 0
    for end in (SRC, DST):
        for _, v, d in L.edges(end, data=True):
            assert d["kind"] == "ground" and L.nodes[v]["kind"] == "aircraft"
    # an aircraft in mid-ocean (synthetic mask: water from 60 to 10 deg W) has no access
    mid = [v for v, d in L.nodes(data=True) if d.get("kind") == "aircraft"
           and -45.0 < d["lon"] < -25.0]
    assert mid and all(not L.has_edge(SRC, v) and not L.has_edge(DST, v) for v in mid)
