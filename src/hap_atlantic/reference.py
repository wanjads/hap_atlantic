"""Latency benchmarks for Frankfurt <-> New York (one-way, seconds).

Measured / published values (bib keys of the paper in brackets):

* Hibernia Express, NY4 (Secaucus) <-> LD4 (Slough): round trip below 58.95 ms
  [Equinix2015].
* London <-> Frankfurt fibre: 4.29 ms one-way exchange-to-exchange, 4.56 ms from Slough
  [euNetworks2011]; microwave Slough <-> Frankfurt: 4.64 ms round trip [McKay2015].
* Shortwave radio Chicago -> London: 24.8 ms mean one-way, 6-bit payloads every 5 ms
  [STAC2023]; the great circle is 6 350 km, i.e. 3.6 ms above the vacuum limit.

Derived values: vacuum and fibre (n = 1.4682) on the great circle, and a lower bound for
a LEO constellation (see `leo_lower_bound`).
"""

from __future__ import annotations

import numpy as np

from . import geometry as geo
from .config import C0, N_FIBER, Params

HIBERNIA_RTT = 58.95e-3
LON_FRA_FIBRE = 4.56e-3  # Slough -> Frankfurt
LON_FRA_MICROWAVE = 4.64e-3 / 2
HF_CHI_LON = 24.8e-3
HF_CHI_LON_KM = 6350.0
LEO_ALT = 550e3  # Starlink shell [m]
LEO_ELEV_MIN = 25.0  # minimum elevation of the user terminals [deg]


def slant_range(alt: float, elev_deg: float, radius: float) -> float:
    """Distance [m] from the ground to a point at altitude alt seen at elevation elev."""
    e = np.radians(elev_deg)
    return float(np.sqrt((radius + alt) ** 2 - (radius * np.cos(e)) ** 2) - radius * np.sin(e))


def leo_lower_bound(gc: float, radius: float, alt: float = LEO_ALT,
                    elev_min: float = LEO_ELEV_MIN) -> float:
    """Shortest ground-satellite-...-satellite-ground path [m] via a shell at altitude alt.

    Every satellite path starts with an up-link at elevation >= elev_min. The shortest
    such path goes up at exactly elev_min towards the destination, follows one straight
    inter-satellite chord and comes down at elev_min; by the triangle inequality any other
    path via satellites of this shell (more hops, steeper links) is longer. For
    Frankfurt-New York the chord is about 4 800 km long (Starlink links reach 5 400 km)
    and its lowest point is about 160 km above the ground.
    """
    s = slant_range(alt, elev_min, radius)
    e = np.radians(elev_min)
    phi = np.arcsin(s * np.cos(e) / (radius + alt))  # central angle covered by one access
    rest = max(gc / radius - 2 * phi, 0.0)
    chord = 2 * (radius + alt) * np.sin(rest / 2)
    return 2 * s + chord


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
        "leo_lower_bound": leo_lower_bound(gc, R) / C0,
        "hf_scaled": hf_overhead * gc / C0,  # Chicago-London HF overhead applied to FRA-NY
    }
