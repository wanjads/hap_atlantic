"""Sensitivity of ocean capacity / end-to-end connectivity to geometric assumptions.

Usage: python scripts/sensitivity.py data/processed/20230708-10.parquet results/sensitivity.csv
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from hap_atlantic.config import scenario
from hap_atlantic.simulate import run

T0, T1 = 1688860800, 1688860800 + 86400  # 9 July 2023 UTC
EVERY = 10  # every 10th minute


def one(args):
    path, h_chord, elev = args
    p = scenario("conservative")
    p = replace(p, geometry=replace(p.geometry, h_chord_min=h_chord, elev_min_ground=elev),
                ground=replace(p.ground, weather="markov", p_cloud_free=1.0))
    st = pd.read_parquet(path)
    st = st[(st.time >= T0) & (st.time < T1)]
    r = run(st, p, every=EVERY, progress=False)
    return {
        "h_chord_min_km": h_chord / 1e3,
        "elev_min_deg": elev,
        "ocean_mean_gbps": r.flow_ocean.mean() / 1e9,
        "ocean_max_gbps": r.flow_ocean.max() / 1e9,
        "ocean_available": (r.flow_ocean > 0).mean(),
        "e2e_available": r.connected.mean(),
        "e2e_mean_gbps": r.flow_heuristic.mean() / 1e9,
        "delay_median_ms": r.loc[r.connected, "min_delay"].median() * 1e3,
        "air_links_mean": r.n_air_links.mean(),
    }


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    grid = [(path, h * 1e3, e) for h in (5, 6, 7, 8, 9) for e in (5.0, 10.0, 15.0)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, grid))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
