# hap_atlantic

Simulation code for the paper on the capacity and latency of an optical backbone of
airliners across the North Atlantic.

Two questions:

* **Capacity** (complement to submarine cables): optical ground stations anywhere on land
  in Europe and North America, the ocean is the bottleneck. Max flow from all European to
  all North American land stations.
* **Latency** (high-frequency trading): Frankfurt (Eurex/DE-CIX) to New York (Equinix NY4),
  minimum one-way delay and availability of a path.

Only fully public data are used:

* **Air traffic:** adsb.lol daily archives
  ([globe_history_2023/2024](https://github.com/adsblol), ODbL, attribution required),
  crowd-sourced terrestrial ADS-B; jet airliners only (ICAO type list in `traffic.py`),
  all of them carry terminals. The central Atlantic has no receivers; positions in coverage
  gaps are reconstructed on the great circle (14–15 % of all grid positions). Gaps are
  bridged up to 10 min always, up to 1 h at 120–340 m/s and up to 8 h at 190–340 m/s
  implied ground speed: the ocean crossings take 4–7 h at 200–300 m/s, slower long gaps
  are aircraft that leave the area and return on a later flight (`traffic.py`).
* **Weather:** ERA5 (cloud fraction, cloud ice and liquid water on 25 pressure levels,
  precipitation, land–sea mask) from the public
  [ARCO-ERA5](https://github.com/google-research/arco-era5) store on Google Cloud (no account).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,plots]"
.venv/bin/pytest -q
```

## Workflow

```bash
# 1. traffic: download the daily releases (about 1-1.5 GB each), extract, resample
gh release download v2023.07.09-planes-readsb-prod-0 -R adsblol/globe_history_2023 \
    -D data/raw/adsblol_20230709
hap-atlantic adsblol data/raw/adsblol_20230709/*.tar --out data/raw/adsblol_20230709.parquet
#    (same for 2023-07-08/10 and 2024-01-15/16/17)
hap-atlantic prepare data/raw/adsblol_2023070{8,9}.parquet data/raw/adsblol_20230710.parquet \
    --out data/processed/20230708-10.parquet
hap-atlantic prepare data/raw/adsblol_2024011{5,6,7}.parquet --out data/processed/20240115-17.parquet

# 2. clouds (about 35-50 MB per day; also downloaded automatically when needed)
hap-atlantic weather --start "2023-07-08 23:00" --end "2023-07-10 01:00"
hap-atlantic weather --start "2024-01-15 23:00" --end "2024-01-17 01:00"

# 3. experiments and figures (results/quick/*.csv, figures/quick/*.pdf|png)
./run_all.sh                                # everything: 2 days x 5 seeds, maps, summary
python scripts/quick.py sweep               # main run (all rates x K, every day and seed)
python scripts/quick.py --day winter --seed 0 sweep   # one day / seed
python scripts/quick.py rings               # latency without the ring stations
python scripts/quick.py --plot sweep        # redraw only, no simulation
python scripts/quick.py summary             # numbers of the paper -> results/quick/summary.txt

hap-atlantic bounds                         # latency benchmarks
```

Existing result files are skipped, so an interrupted run can simply be restarted.

Days: 9 July 2023 (all-time record of Shanwick Oceanic, summer) and 16 January 2024
(winter weekday), one snapshot per hour. The only random element is the latent cloud
field; every experiment is repeated for five seeds (0–4). Only aircraft within 1000 km of
the Frankfurt–New York great circle are considered (also for the capacity). Long runs
can be started with a memory cap, e.g.
`systemd-run --user --scope -p MemoryMax=8G ./run_all.sh`; the number of worker processes
is also limited by the free memory (`simulate.safe_workers`).

## Model in brief

* **Air-to-air links:** if the straight ray stays above `h_chord_min` = 3 km and within
  700 km. Rate = photon-counting FSO budget (Gaussian beam, clear-air extinction,
  Hufnagel-Valley fade margin, cloud loss) capped at the transceiver rate `r_max`; links
  below 1 Gbit/s are dropped. Each aircraft has `K` terminals (degree constraint).
* **Clouds (`weather.py`):** a correlated latent Gaussian field (50 km, 2 km vertical, 1 h;
  modelling assumptions) decides which 0.25° grid boxes are cloudy, matching the ERA5 cloud
  fraction on every level in expectation; the in-cloud extinction follows from the ice and
  liquid water content in the geometric-optics limit (Francis et al. 1994) and is
  integrated along every ray.
* **Ground links:** optical above 5° elevation; RF fallback (10 Gbit/s) when the optical
  ray is blocked, unless it rains heavily. Capacity: stations on every land cell (0.25°),
  Europe east of 12° W, North America west of 52° W (Iceland, Greenland, Azores excluded).
  Latency: stations at Frankfurt / NY4 and on rings of 75 and 150 km around them, so that
  a cruising aircraft is always within reach of a station (aircraft below 3 km cannot
  link to other aircraft and thus cannot relay; near the endpoints these are mostly the
  arrivals and departures).
* **Capacity (`flow.py`):** degree-constrained max flow; the widest-path heuristic is
  accepted if it is within 1 % of the relaxation bound, otherwise the MILP (HiGHS) is
  solved to a 1 % gap (time limit). The upper bound is stored with every value; it refers
  to the sparsified candidate graph (nearest and farthest neighbour in each of 8 sectors).
  Each aircraft has at most one ground link per continent, which uses one of its `K`
  terminals (also the RF fallback).
* **Latency:** shortest path over links ≥ 1 Gbit/s with 10 µs processing per hop;
  benchmarks (cables, microwave, shortwave, LEO lower bound) in `reference.py`. The LEO
  bound: up at 25° elevation to a 550 km shell, one straight inter-satellite link, down
  at 25°.

All parameters are in `src/hap_atlantic/config.py`.
