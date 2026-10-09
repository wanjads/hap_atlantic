"""End-to-end results with ERA5 weather at the ground stations, optical vs. hybrid FSO/RF.

Usage: python scripts/weather_runs.py data/processed/20230708-10.parquet results/weather.csv
"""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from hap_atlantic.config import scenario
from hap_atlantic.simulate import run

T0, T1 = 1688860800, 1688860800 + 86400  # 9 July 2023 UTC
EVERY = 10
SEEDS = (0, 1, 2)


def one(args):
    path, sc, sites, f_cirrus, rf_gbps, seed = args
    p = scenario(sc)
    p = replace(p, seed=seed, ground=replace(p.ground, weather="era5", n_sites=sites,
                                            f_cirrus=f_cirrus, rf_rate=rf_gbps * 1e9))
    st = pd.read_parquet(path)
    st = st[(st.time >= T0 - 3600) & (st.time < T1 + 3600)]  # weather fetch covers the day
    r = run(st, p, every=EVERY, progress=False)
    r = r[(r.time >= T0) & (r.time < T1)]
    return {"scenario": sc, "sites": sites, "f_cirrus": f_cirrus, "rf_gbps": rf_gbps,
            "seed": seed, "e2e_available": (r.flow_heuristic > 0).mean(),
            "e2e_mean_gbps": r.flow_heuristic.mean() / 1e9,
            "latency_available": r.connected.mean(),
            "delay_median_ms": r.loc[r.connected, "min_delay"].median() * 1e3}


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    grid = [(path, *c) for c in itertools.product(("conservative", "advanced"), (1, 5),
                                                  (0.0, 1.0), (0.0, 10.0), SEEDS)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, grid))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    agg = df.groupby(["scenario", "sites", "f_cirrus", "rf_gbps"]).mean(numeric_only=True)
    print(agg.drop(columns="seed").round(3).to_string())
