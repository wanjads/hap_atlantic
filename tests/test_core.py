import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hap_atlantic import flow
from hap_atlantic import geometry as geo
from hap_atlantic.config import FRANKFURT, NEW_YORK, AtmosphereParams, Params, TerminalParams
from hap_atlantic.link import link_rate, scintillation_index
from hap_atlantic.topology import DST, SRC, build_graph
from hap_atlantic.traffic import interpolate_to_grid, synthetic_states

R = 6_371_008.8


def test_great_circle_fra_ny():
    d = geo.great_circle(FRANKFURT.lat, FRANKFURT.lon, NEW_YORK.lat, NEW_YORK.lon, R)
    assert 6150e3 < d < 6250e3


def test_line_of_sight_horizon():
    # two aircraft at 11 km see each other up to 2*sqrt(2 R h) ~ 749 km (grazing ray)
    h = 11e3
    d_horizon = 2 * np.sqrt(2 * R * h)
    lat2 = np.degrees(d_horizon / R) * 0.99
    a = geo.to_ecef(0.0, 0.0, h, R)
    b = geo.to_ecef(lat2, 0.0, h, R)
    assert 0 < geo.chord_min_height(a, b, R) < 500
    b_far = geo.to_ecef(np.degrees(d_horizon / R) * 1.05, 0.0, h, R)
    assert geo.chord_min_height(a, b_far, R) < 0


def test_slerp_endpoints():
    lat, lon = geo.slerp_latlon(50.0, 8.0, 40.0, -74.0, np.array([0.0, 1.0]))
    assert np.allclose(lat, [50.0, 40.0]) and np.allclose(lon, [8.0, -74.0])


def test_scintillation_weak_limit():
    assert scintillation_index(1e-4) == pytest.approx(1e-4, rel=1e-2)


def test_link_rate_monotone_and_capped():
    term, atm = TerminalParams(), AtmosphereParams()
    d = np.array([50e3, 300e3, 700e3])
    heights = np.full((3, 16), 11e3)
    r = link_rate(d, heights, term, atm)
    assert np.all(np.diff(r) <= 0) and r.max() <= term.r_max


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


def test_flow_methods_agree_on_toy():
    ub, _ = flow.max_flow_unconstrained(_toy_graph())
    assert ub == pytest.approx(40.0)
    # with one terminal per aircraft only a perfect matching (2 links) can be active
    G1 = _toy_graph(K=1)
    assert flow.widest_path_heuristic(G1)[0] == pytest.approx(20.0)
    assert flow.milp_max_flow(G1, scale=1.0)["value"] == pytest.approx(20.0)


def test_pipeline_on_synthetic_traffic():
    states = synthetic_states(n_flights=300, duration=3600, dt=300, seed=1)
    raw = states.drop(columns="interpolated").iloc[::2]  # drop half to force interpolation
    grid = interpolate_to_grid(raw, dt=300)
    assert grid["icao24"].nunique() > 0
    p = Params()
    t, snap = next(iter(grid.groupby("time")))
    G = build_graph(snap, p)
    assert G.has_node(SRC) and G.has_node(DST)
    assert flow.widest_path_heuristic(G)[0] <= flow.max_flow_unconstrained(G)[0] + 1e-6


def test_capacity_certified_and_milp():
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


def test_with_tech_scales_and_caps():
    from hap_atlantic.topology import with_tech

    G = _toy_graph()
    for _, _, d in G.edges(data=True):
        d["rate"] = np.inf if d["kind"] == "fiber" else 50.0
    H = with_tech(G, r_max=100.0, k=4, p_scale=3.0)
    assert H["a"]["c"]["cap"] == 100.0 and H.nodes["a"]["terminals"] == 4
    assert with_tech(G, r_max=100.0, k=2)["a"]["c"]["cap"] == 50.0


def test_select_fleet():
    from hap_atlantic.traffic import select_fleet

    df = pd.DataFrame({"icao24": list("abcd"), "type": ["B77W", "GLF6", "A20N", "C17"],
                       "military": [False, False, False, True], "time": 0.0})
    assert list(select_fleet(df, True)["icao24"]) == ["a", "c"]
    assert len(select_fleet(df, False)) == 4


def test_latent_field_statistics():
    from hap_atlantic.weather import LatentField

    z = LatentField(50e3, 2e3, 3600.0, 2048, seed=3)
    rng = np.random.default_rng(0)
    x = rng.uniform(-5e6, 5e6, (4000, 3))
    v = z(x, 0.0)
    assert abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.1
    # strongly correlated at 5 km, nearly independent at 200 km
    near = z(x + [5e3, 0, 0], 0.0)
    far = z(x + [200e3, 0, 0], 0.0)
    assert np.corrcoef(v, near)[0, 1] > 0.9 and abs(np.corrcoef(v, far)[0, 1]) < 0.15
    # vertical: correlated at 0.5 km, nearly independent at 8 km
    assert np.corrcoef(v, z(x, 0.0, 500.0))[0, 1] > 0.9
    assert abs(np.corrcoef(v, z(x, 0.0, 8e3))[0, 1]) < 0.15


def test_cloud_extinction_fu():
    from hap_atlantic.weather import FU_A0, FU_A1, isa_height

    # 0.01 g/m^3 of ice with De = 50 um: about 0.67 / km, i.e. ~2.9 dB/km
    beta = 0.01 * (FU_A0 + FU_A1 / 50.0)
    assert 2.5 < 10 / np.log(10) * beta * 1e3 < 3.3
    assert abs(isa_height(250.0) - 10_363) < 50


def test_adsblol_parse_trace():
    from hap_atlantic.adsblol import parse_trace

    data = {"icao": "3C6444", "timestamp": 1000.0, "trace": [
        [0, 50.0, 8.0, "ground", None, None, 0, None, None, "adsb_icao", None],
        [10, 50.1, 8.0, 10000, 300, 270, 0, 0, None, "adsb_icao", 10100],
        [20, 50.2, 8.0, 11000, 300, 270, 0, 0, None, "adsb_icao", None],
        [100, 50.3, 8.0, 12000, 300, 270, 0, 0, None, "adsb_icao", None],
        [200, 10.0, 8.0, 35000, 450, 270, 0, 0, None, "adsb_icao", None],  # outside bbox
        [300, 51.0, 8.0, 5000, 250, 270, 2, 0, None, "adsb_icao", None],
    ]}
    df = parse_trace(data, (35.0, 66.0, -80.0, 15.0), min_dt=30.0)
    assert list(df["time"]) == [1010.0, 1100.0, 1300.0]
    assert set(df["icao24"]) == {"3c6444"}
    assert df["alt"].iloc[0] == pytest.approx(10100 * 0.3048)
