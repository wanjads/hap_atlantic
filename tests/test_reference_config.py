import dataclasses

import numpy as np
import pytest

from hap_atlantic.config import C0, Params
from hap_atlantic.reference import LEO_ALT, benchmarks, leo_lower_bound, slant_range

R = 6_371_008.8


def test_slant_range_zenith_and_monotone():
    assert slant_range(550e3, 90.0, R) == pytest.approx(550e3)
    s = [slant_range(550e3, e, R) for e in (10, 25, 45, 90)]
    assert np.all(np.diff(s) < 0)


def test_leo_bound_above_vacuum_and_below_old_model():
    gc = 6_202e3
    b = leo_lower_bound(gc, R)
    old = gc * (R + LEO_ALT) / R + 2 * LEO_ALT  # vertical up, along the shell, down
    assert gc < b < old


def test_leo_bound_monotone_in_elevation_and_altitude():
    gc = 6_202e3
    assert leo_lower_bound(gc, R, elev_min=10) < leo_lower_bound(gc, R, elev_min=25)
    assert leo_lower_bound(gc, R, alt=340e3) < leo_lower_bound(gc, R, alt=550e3)


def test_leo_bound_short_distance_reduces_to_two_slants():
    # endpoints so close that both see the same satellite: chord length 0
    assert leo_lower_bound(10e3, R) == pytest.approx(2 * slant_range(LEO_ALT, 25.0, R))


def test_benchmark_ordering_and_values():
    b = benchmarks()
    assert b["vacuum"] < b["leo_lower_bound"] < b["hf_scaled"] < b["fibre_great_circle"]
    assert b["fibre_great_circle"] < b["fibre_microwave_published"] < b["fibre_best_published"]
    assert b["vacuum"] * 1e3 == pytest.approx(20.69, abs=0.01)
    assert b["leo_lower_bound"] * 1e3 == pytest.approx(22.85, abs=0.01)
    assert b["great_circle_km"] * 1e3 / C0 == pytest.approx(b["vacuum"])


def test_params_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Params().seed = 3


PAPER_TABLE = [
    ("terminal.p_tx", 1.0), ("terminal.r_max", 100e9), ("terminal.d_rx", 0.10),
    ("terminal.divergence", 20e-6), ("terminal.eta_optics", 0.5),
    ("terminal.pointing_loss_db", 3.0), ("terminal.photons_per_bit", 8.0),
    ("terminal.hop_delay", 10e-6), ("terminal.r_min", 1e9), ("terminal.outage_prob", 1e-3),
    ("geometry.h_chord_min", 3e3), ("geometry.d_max_air", 700e3),
    ("geometry.elev_min_ground", 5.0), ("geometry.corridor_half_width", 1000e3),
    ("geometry.h_node_min", 1e3), ("geometry.yao_cones", 8),
    ("ground.rf_rate", 10e9), ("ground.rf_rain_max", 5.0), ("ground.fiber_detour", 1.3),
    ("ground.terminals_per_site", 4), ("ground.site_rings", (75e3, 150e3)),
    ("ground.land_eu_lon", -12.0), ("ground.land_na_lon", -52.0),
    ("ground.land_candidates", 6), ("clouds.l_corr", 50e3), ("clouds.lz_corr", 2e3),
    ("clouds.t_corr", 3600.0), ("clouds.re_ice", 25e-6), ("clouds.re_liquid", 10e-6),
    ("atmosphere.alpha0_db_per_km", 0.2), ("terminals_per_aircraft", 2),
]


def test_defaults_match_paper_table():
    for path, value in PAPER_TABLE:
        obj = Params()
        for part in path.split("."):
            obj = getattr(obj, part)
        assert obj == value, path
