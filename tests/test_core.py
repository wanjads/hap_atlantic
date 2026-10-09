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


def test_normalise_rest_columns():
    from hap_atlantic.traffic import normalise

    df = pd.DataFrame({"icao24": ["ABC123"], "time": [100], "time_position": [98],
                       "latitude": [50.0], "longitude": [-20.0], "geo_altitude": [None],
                       "baro_altitude": [11000.0], "on_ground": [False]})
    out = normalise(df)
    assert out.iloc[0].to_dict() == {"time": 98.0, "icao24": "abc123", "lat": 50.0,
                                     "lon": -20.0, "alt": 11000.0}
