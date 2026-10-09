"""Latency benchmarks for Frankfurt <-> New York (one-way, seconds).

Measured / published values (bib keys of the paper in brackets):

* Hibernia Express, NY4 (Secaucus) <-> LD4 (Slough): round trip below 58.95 ms, measured
  between cable stations [Equinix2015].
* London <-> Frankfurt fibre: 4.29 ms one-way exchange-to-exchange, 4.56 ms from Slough
  [euNetworks2011]; microwave Slough <-> Frankfurt: 4.64 ms round trip [McKay2015].
* Shortwave radio Chicago -> London: 24.8 ms mean one-way, 6-bit payloads every 5 ms
  [STAC2023]; the great circle is 6 350 km, i.e. 3.6 ms above the vacuum limit.

Derived values: vacuum and fibre (n = 1.4682) on the great circle, and an idealised LEO
lower bound (straight up to a 550 km shell, along the shell, straight down).
"""

from __future__ import annotations

from . import geometry as geo
from .config import C0, N_FIBER, Params

HIBERNIA_RTT = 58.95e-3
LON_FRA_FIBRE = 4.56e-3  # Slough -> Frankfurt
LON_FRA_MICROWAVE = 4.64e-3 / 2
HF_CHI_LON = 24.8e-3
HF_CHI_LON_KM = 6350.0
LEO_ALT = 550e3


def benchmarks(p: Params | None = None) -> dict[str, float]:
    p = p or Params()
    R = p.geometry.earth_radius
    gc = float(geo.great_circle(p.source.lat, p.source.lon, p.sink.lat, p.sink.lon, R))
    hf_overhead = HF_CHI_LON / (HF_CHI_LON_KM * 1e3 / C0)
    return {
        "great_circle_km": gc / 1e3,
        "vacuum": gc / C0,
        "fibre_great_circle": gc * N_FIBER / C0,
        "fibre_best_published": HIBERNIA_RTT / 2 + LON_FRA_FIBRE,
        "fibre_microwave_published": HIBERNIA_RTT / 2 + LON_FRA_MICROWAVE,
        "leo_lower_bound": (gc * (R + LEO_ALT) / R + 2 * LEO_ALT) / C0,
        "hf_scaled": hf_overhead * gc / C0,  # Chicago-London HF overhead applied to FRA-NY
    }
