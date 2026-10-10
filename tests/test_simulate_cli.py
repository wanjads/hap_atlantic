from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hap_atlantic import cli, simulate
from hap_atlantic.config import Params
from hap_atlantic.topology import DST, SRC, build_graph, land_graph
from hap_atlantic.traffic import synthetic_states

CLEAR = replace(Params(), clouds=replace(Params().clouds, model="clear"))


def test_safe_workers_bounds():
    assert simulate.safe_workers(1, clouds=True) == 1
    assert 1 <= simulate.safe_workers(64, clouds=True) <= 64


def test_config_label_and_power_scale():
    c = simulate.config(1e12, 4, 5.0, Params())
    assert c.label == "r1.0e+12_k4" and c.p_scale == 5.0 and c.k == 4


def _snapshot():
    s = synthetic_states(n_flights=600, n_regional=600, duration=600, dt=600, seed=4)
    t = s.time.iloc[len(s) // 2]
    return s[s.time == t]


def test_evaluate_snapshot_keys():
    G = build_graph(_snapshot(), CLEAR)
    L = land_graph(G, CLEAR)
    cfg = simulate.config(1e11, 2, 1.0, CLEAR)
    m = simulate.evaluate_snapshot(G, L, CLEAR, [cfg])
    for k in ("n_aircraft", "connected", "min_delay", "min_delay_hops", "path",
              f"cap_{cfg.label}", f"cap_{cfg.label}_bound", f"cap_{cfg.label}_heur"):
        assert k in m
    assert m[f"cap_{cfg.label}"] <= m[f"cap_{cfg.label}_bound"] + 1e-6
    if m["connected"]:
        assert m["path"].startswith(SRC) and m["path"].endswith(DST)


def test_evaluate_snapshot_hops_count_fiber_edges():
    G = nx.Graph()
    for n in (SRC, DST):
        G.add_node(n, kind="endpoint", terminals=np.inf)
    G.add_node("gs", kind="ground", terminals=4)
    G.add_node("a", kind="aircraft", terminals=2)
    G.add_edge(SRC, "gs", kind="fiber", cap=np.inf, delay=0.0, rate=np.inf)
    G.add_edge("gs", "a", kind="ground", medium="optical", cap=1e11, rate=1e12, delay=1e-3)
    G.add_edge("a", DST, kind="ground", medium="optical", cap=1e11, rate=1e12, delay=1e-3)
    m = simulate.evaluate_snapshot(G, G, CLEAR, [])
    assert m["connected"] and m["min_delay_hops"] == 3


def test_run_on_synthetic_traffic():
    s = synthetic_states(n_flights=400, n_regional=400, duration=1200, dt=600, seed=5)
    cfg = simulate.config(1e10, 2, 1.0, CLEAR)
    r = simulate.run(s, CLEAR, [cfg], 0.0, 1200.0, every=1, workers=1)
    assert len(r) == 2 and list(r.time) == sorted(r.time)


def test_cli_bounds(capsys):
    cli.main(["bounds"])
    out = capsys.readouterr().out
    assert "leo_lower_bound" in out and "22.85" in out


def test_cli_synth(tmp_path):
    f = tmp_path / "s.parquet"
    cli.main(["synth", "--out", str(f), "--flights", "20"])
    assert len(pd.read_parquet(f)) > 0


def test_cli_prepare(tmp_path):
    raw = tmp_path / "raw.parquet"
    s = synthetic_states(n_flights=30, n_regional=10, duration=3600, dt=300, seed=1)
    s.drop(columns="interpolated").to_parquet(raw)
    out = tmp_path / "grid.parquet"
    cli.main(["prepare", str(raw), "--out", str(out), "--dt", "300"])
    g = pd.read_parquet(out)
    assert (g.time % 300 == 0).all() and "interpolated" in g


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        cli.main([])
