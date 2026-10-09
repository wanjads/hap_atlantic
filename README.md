# hap_atlantic

Simulation code for the paper on the capacity and latency of an optical backbone of
airliners between Frankfurt and New York.

Only fully public data are used:

* **Air traffic:** adsb.lol daily archives
  ([globe_history_2023/2024](https://github.com/adsblol), ODbL, attribution required),
  crowd-sourced terrestrial ADS-B. The central Atlantic has no receivers; positions in
  coverage gaps are reconstructed on the great circle (11–12 % of all grid positions).
* **Weather:** ERA5 hourly reanalysis (cloud cover low/mid/high, precipitation) via the
  [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api)
  (CC BY 4.0, no registration), on a 1° grid over the corridor.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,plots]"
.venv/bin/pytest -q
```

## Workflow

```bash
# 1. traffic: download the daily releases (about 1-1.5 GB each), then extract positions
gh release download v2023.07.09-planes-readsb-prod-0 -R adsblol/globe_history_2023 \
    -D data/raw/adsblol_20230709
hap-atlantic adsblol data/raw/adsblol_20230709/*.tar --out data/raw/adsblol_20230709.parquet
#    (same for 2023-07-08/10 and 2024-01-15/16/17)
hap-atlantic prepare data/raw/adsblol_2023070{8,9}.parquet data/raw/adsblol_20230710.parquet \
    --out data/processed/20230708-10.parquet
hap-atlantic prepare data/raw/adsblol_2024011{5,6,7}.parquet --out data/processed/20240115-17.parquet

# 2. experiments of the paper (ERA5 data are downloaded and cached in data/weather)
python scripts/experiments.py tech        # clear sky: technology tiers x K
python scripts/experiments.py equipage    # share of equipped airliners
python scripts/experiments.py weather     # ERA5 clouds: f_cirrus x stations x RF x seeds
python scripts/experiments.py paths       # min-latency path every minute, stability
python scripts/experiments.py solver      # heuristic vs MILP vs bounds, Yao vs full graph

# 3. figures (PDF + PNG in figures/)
python scripts/figures.py

# single runs and reference latencies
hap-atlantic simulate data/processed/20230708-10.parquet --scenario leo --k 2 \
    --start 2023-07-09 --stop 2023-07-10 --clouds era5 --workers 16
hap-atlantic bounds
```

## Model in brief

* **Nodes:** jet airliners (ICAO type list in `traffic.py`; no business jets, turboprops or
  military) within 1000 km of the Frankfurt–New York arc; a random share `equipage`
  carries `K` optical terminals. Optical ground stations at the endpoints (optionally 5 per
  endpoint on a 50 km ring), connected by fibre.
* **Links:** air-to-air if the ray stays above `h_chord_min` = 7 km and within 700 km;
  ground-to-air above 10° elevation. Rate = photon-counting FSO budget (Gaussian beam,
  clear-air extinction, Hufnagel-Valley fade margin) capped at the transceiver rate
  `r_max`. At cruise altitude the cap binds for every link up to ~430 Gbit/s (1 W), so the
  capacity is (number of terminal-disjoint chains) × `r_max`.
* **Clouds (`weather.py`):** ERA5 gives the cloud fraction per layer; a latent Gaussian
  space-time field (correlation 50 km, 1 h, random Fourier features) decides which rays
  are blocked, so nearby stations and consecutive minutes are correlated. Air-to-air rays
  cross only the high (cirrus) layer, scaled by `f_cirrus`; ground links cross all layers.
  Optional RF fallback (fixed rate) for ground links only; it fails in heavy rain.
* **Technology tiers (`config.TECHS`):** demo 2.5 Gbit/s, leo 100 Gbit/s, dwdm 1 Tbit/s,
  visionary 10 Tbit/s per link; evaluated for K = 2, 4, 6.
* **Metrics (`simulate.py`):** oceanic capacity (aircraft east of 15° W inject, west of
  55° W extract), backbone capacity (unlimited ground terminals) and end-to-end capacity,
  each the degree-constrained max flow: heuristic value if it meets the relaxation bound,
  otherwise the MILP (HiGHS) on the pruned graph. Minimum one-way latency over links
  ≥ 1 Gbit/s with 10 µs processing per hop. Benchmarks in `reference.py`.

All parameters are in `src/hap_atlantic/config.py`.
