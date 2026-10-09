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
  gaps are reconstructed on the great circle (11–12 % of all grid positions).
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

# 3. experiments and figures (results/quick/*.csv, figures/quick/*.png)
python scripts/quick.py                     # everything, both days
python scripts/quick.py overview tech       # selected experiments
python scripts/quick.py --day winter map    # one day
python scripts/quick.py --plot overview tech  # redraw only, no simulation

hap-atlantic bounds                         # latency benchmarks
```

Days: 9 July 2023 (all-time record of Shanwick Oceanic, summer) and 16 January 2024
(winter weekday), one snapshot per hour. Long runs are started with a memory cap, e.g.
`systemd-run --user --scope -p MemoryMax=8G python scripts/quick.py`; the number of
worker processes is also limited by the free memory (`simulate.safe_workers`).

## Model in brief

* **Air-to-air links:** if the straight ray stays above `h_chord_min` = 3 km and within
  700 km. Rate = photon-counting FSO budget (Gaussian beam, clear-air extinction,
  Hufnagel-Valley fade margin, cloud loss) capped at the transceiver rate `r_max`; links
  below 1 Gbit/s are dropped. Each aircraft has `K` terminals (degree constraint).
* **Clouds (`weather.py`):** a correlated latent Gaussian field (50 km, 2 km vertical, 1 h)
  decides where clouds are, matching the ERA5 cloud fraction on every level; the in-cloud
  extinction follows from the ice (Fu 1996) and liquid water content and is integrated
  along every ray.
* **Ground links:** optical above 5° elevation; RF fallback (10 Gbit/s) when the optical
  ray is blocked, unless it rains heavily. Capacity: stations on every land cell (0.25°),
  Europe east of 12° W, North America west of 52° W (Iceland, Greenland, Azores excluded).
  Latency: stations at Frankfurt / NY4 and on rings of 75 and 150 km around them.
* **Capacity (`flow.py`):** degree-constrained max flow; the widest-path heuristic is
  accepted if it is within 1 % of the relaxation bound, otherwise the MILP (HiGHS) is
  solved to a 1 % gap (time limit). The upper bound is stored with every value.
* **Latency:** shortest path over links ≥ 1 Gbit/s with 10 µs processing per hop;
  benchmarks (cables, microwave, shortwave, LEO bound) in `reference.py`.

All parameters are in `src/hap_atlantic/config.py`.
