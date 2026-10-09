"""Command line interface.

    hap-atlantic adsblol  data/raw/adsblol_20230709/*.tar --out data/raw/adsblol_20230709.parquet
    hap-atlantic prepare  data/raw/adsblol_2023070*.parquet --out data/processed/day.parquet
    hap-atlantic synth    --out data/processed/synthetic.parquet
    hap-atlantic simulate data/processed/day.parquet --scenario leo --k 2 \
        --start 2023-07-09 --stop 2023-07-10 --clouds era5 --workers 16
    hap-atlantic bounds                                            # latency benchmarks
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
    sim.add_argument("--scenario", choices=SCENARIOS, default="leo")
    sim.add_argument("--k", type=int, default=2, help="terminals per aircraft")
    sim.add_argument("--every", type=int, default=10, help="evaluate every n-th time step")
    sim.add_argument("--sites", type=int, default=None, help="ground stations per endpoint")
    sim.add_argument("--clouds", choices=["clear", "era5"], default="clear")
    sim.add_argument("--rf-rate", type=float, default=None, help="RF fallback rate [Gbit/s]")
    sim.add_argument("--equipage", type=float, default=None)
    sim.add_argument("--all-aircraft", action="store_true", help="not only jet airliners")
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--start", required=True, help="UTC, e.g. '2023-07-09 00:00'")
    sim.add_argument("--stop", required=True, help="UTC, exclusive")
    sim.add_argument("--workers", type=int, default=1)
    sim.add_argument("--out", default=None)

    w = sub.add_parser("weather", help="download and cache 3D ERA5 cloud fields (ARCO-ERA5)")
    w.add_argument("--start", required=True, help="UTC, e.g. '2023-07-08 23:00'")
    w.add_argument("--end", required=True, help="UTC, e.g. '2023-07-10 01:00'")

    sub.add_parser("bounds", help="print latency benchmarks Frankfurt - New York")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "adsblol":
        from .adsblol import read_archive
        from .config import NORTH_ATLANTIC_BBOX
        from .traffic import aircraft_table

        df, aircraft = read_archive(args.archive, NORTH_ATLANTIC_BBOX, args.min_dt)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        aircraft.to_parquet(aircraft_table(out), index=False)
        print(f"{df['icao24'].nunique()} aircraft, {len(df)} positions")
    elif args.cmd == "prepare":
        from .traffic import interpolate_to_grid, load_aircraft, load_states

        grid = interpolate_to_grid(load_states(args.raw), args.dt)
        aircraft = load_aircraft(args.raw)
        if aircraft is not None:
            grid = grid.merge(aircraft, on="icao24", how="left")
            grid["type"] = grid["type"].fillna("")
            grid["military"] = grid["military"].fillna(False).astype(bool)
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
        from .simulate import run, tech_config

        p = scenario(args.scenario, args.k)
        p = replace(p, clouds=replace(p.clouds, model=args.clouds),
                    airliners_only=not args.all_aircraft)
        if args.sites is not None:
            p = replace(p, ground=replace(p.ground, n_sites=args.sites))
        if args.rf_rate is not None:
            p = replace(p, ground=replace(p.ground, rf_rate=args.rf_rate * 1e9))
        if args.equipage is not None:
            p = replace(p, equipage=args.equipage)
        if args.seed is not None:
            p = replace(p, seed=args.seed)
        t0 = pd.Timestamp(args.start, tz="UTC").timestamp()
        t1 = pd.Timestamp(args.stop, tz="UTC").timestamp()
        cfg = tech_config(args.scenario, args.k, p)
        res = run(pd.read_parquet(args.states), p, [cfg], t0, t1, every=args.every,
                  workers=args.workers)
        out = args.out or f"results/{Path(args.states).stem}_{cfg.label}_{args.clouds}.csv"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        cols = [f"{m}_{cfg.label}" for m in ("ocean", "backbone", "e2e")]
        print((res[cols] / 1e9).describe().round(1).to_string())
        print(f"connected {res['connected'].mean():.1%}, median min delay "
              f"{res.loc[res['connected'], 'min_delay'].median() * 1e3:.2f} ms")
    elif args.cmd == "weather":
        from .weather import fetch_arco

        f = fetch_arco(pd.Timestamp(args.start, tz="UTC").timestamp(),
                       pd.Timestamp(args.end, tz="UTC").timestamp())
        print(f"cached in {f}")
    elif args.cmd == "bounds":
        from .reference import benchmarks

        for k, v in benchmarks().items():
            print(f"{k:28s} {v:9.2f} km" if k.endswith("km")
                  else f"{k:28s} {v * 1e3:9.2f} ms one-way")


if __name__ == "__main__":
    main()
