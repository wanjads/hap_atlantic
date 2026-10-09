"""Command line interface.

    hap-atlantic collect  --out data/raw/live --hours 24          # record OpenSky REST data
    hap-atlantic trino    --start "2025-07-11 00:00" --stop "2025-07-12 00:00"
    hap-atlantic adsblol  data/raw/adsblol_20230709/*.tar --out data/raw/adsblol_20230709.parquet
    hap-atlantic prepare  data/raw/live --out data/processed/day.parquet
    hap-atlantic synth    --out data/processed/synthetic.parquet
    hap-atlantic simulate data/processed/day.parquet --scenario conservative
    hap-atlantic bounds                                            # back-of-envelope numbers
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import SCENARIOS, scenario


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hap-atlantic")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record OpenSky state vectors via the REST API")
    c.add_argument("--out", default="data/raw/live")
    c.add_argument("--hours", type=float, default=24.0)
    c.add_argument("--interval", type=float, default=90.0)
    c.add_argument("--credentials", default=None)
    c.add_argument("--yes", action="store_true", help="skip the credit confirmation")

    t = sub.add_parser("trino", help="download historical state vectors (research access)")
    t.add_argument("--start", required=True)
    t.add_argument("--stop", required=True)
    t.add_argument("--out", default="data/raw/trino")

    al = sub.add_parser("adsblol", help="extract positions from an adsb.lol daily archive")
    al.add_argument("archive")
    al.add_argument("--out", required=True)
    al.add_argument("--min-dt", type=float, default=30.0)

    pr = sub.add_parser("prepare", help="clean raw states and resample onto a time grid")
    pr.add_argument("raw", nargs="+", help="raw files (neighbouring days avoid edge effects)")
    pr.add_argument("--out", required=True)
    pr.add_argument("--dt", type=float, default=60.0)

    s = sub.add_parser("synth", help="generate synthetic North Atlantic traffic")
    s.add_argument("--out", default="data/processed/synthetic.parquet")
    s.add_argument("--flights", type=int, default=1400)
    s.add_argument("--seed", type=int, default=0)

    sim = sub.add_parser("simulate", help="run the capacity / latency simulation")
    sim.add_argument("states")
    sim.add_argument("--scenario", choices=SCENARIOS, default="conservative")
    sim.add_argument("--every", type=int, default=1, help="evaluate every n-th snapshot")
    sim.add_argument("--milp", action="store_true")
    sim.add_argument("--sites", type=int, default=None, help="ground stations per endpoint")
    sim.add_argument("--clear-sky", action="store_true", help="no cloud blockage")
    sim.add_argument("--rf-rate", type=float, default=None, help="RF fallback rate [Gbit/s]")
    sim.add_argument("--f-cirrus", type=float, default=None)
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--start", default=None, help="UTC, e.g. '2023-07-09 00:00'")
    sim.add_argument("--stop", default=None, help="UTC, exclusive")
    sim.add_argument("--out", default=None)

    sub.add_parser("bounds", help="print reference delays and a min-cut estimate")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "collect":
        from .opensky import CREDITS_PER_REQUEST, collect_states

        n_req = int(args.hours * 3600 / args.interval)
        print(f"This makes ~{n_req} requests = ~{n_req * CREDITS_PER_REQUEST} OpenSky credits "
              f"(standard accounts: 4000/day).")
        if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
            return
        collect_states(args.out, args.interval, args.hours * 3600,
                       credentials_path=args.credentials)
    elif args.cmd == "trino":
        from .opensky import fetch_history_trino

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        df = fetch_history_trino(args.start, args.stop)
        name = f"trino_{args.start}_{args.stop}".replace(" ", "T").replace(":", "")
        df.to_parquet(out / f"{name}.parquet", index=False)
    elif args.cmd == "adsblol":
        from .adsblol import read_archive
        from .opensky import NORTH_ATLANTIC_BBOX

        df = read_archive(args.archive, NORTH_ATLANTIC_BBOX, args.min_dt)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"{df['icao24'].nunique()} aircraft, {len(df)} positions")
    elif args.cmd == "prepare":
        from .traffic import interpolate_to_grid, load_states

        grid = interpolate_to_grid(load_states(args.raw), args.dt)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(args.out, index=False)
        print(f"{grid['icao24'].nunique()} aircraft, {grid['time'].nunique()} time steps, "
              f"{grid['interpolated'].mean():.0%} interpolated positions")
    elif args.cmd == "synth":
        from .traffic import synthetic_states

        df = synthetic_states(args.flights, seed=args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"{df['icao24'].nunique()} flights, {len(df)} positions")
    elif args.cmd == "simulate":
        from .simulate import run

        p = scenario(args.scenario)
        if args.sites is not None:
            p = replace(p, ground=replace(p.ground, n_sites=args.sites))
        if args.clear_sky:
            p = replace(p, ground=replace(p.ground, weather="markov", p_cloud_free=1.0))
        if args.rf_rate is not None:
            p = replace(p, ground=replace(p.ground, rf_rate=args.rf_rate * 1e9))
        if args.f_cirrus is not None:
            p = replace(p, ground=replace(p.ground, f_cirrus=args.f_cirrus))
        if args.seed is not None:
            p = replace(p, seed=args.seed)
        states = pd.read_parquet(args.states)
        if args.start:
            states = states[states["time"] >= pd.Timestamp(args.start, tz="UTC").timestamp()]
        if args.stop:
            states = states[states["time"] < pd.Timestamp(args.stop, tz="UTC").timestamp()]
        res = run(states, p, run_milp=args.milp, every=args.every)
        out = args.out or f"results/{Path(args.states).stem}_{args.scenario}.csv"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        cols = ["flow_heuristic", "flow_backbone", "flow_ocean"]
        summary = (res[cols] / 1e9).describe()
        print(summary.round(1).to_string())
        print(f"connected {res['connected'].mean():.1%}, median min delay "
              f"{res.loc[res['connected'], 'min_delay'].median() * 1e3:.2f} ms")
    elif args.cmd == "bounds":
        from .topology import fiber_reference_delay

        ref = fiber_reference_delay(scenario("conservative"))
        print(f"great circle FRA-NY4: {ref['great_circle_km']:.0f} km")
        print(f"  vacuum:             {ref['vacuum_great_circle'] * 1e3:.2f} ms one-way")
        print(f"  fibre (n=1.468):    {ref['fiber_great_circle'] * 1e3:.2f} ms one-way")


if __name__ == "__main__":
    main()
