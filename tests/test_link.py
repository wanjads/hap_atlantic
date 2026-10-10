from dataclasses import replace

import numpy as np
import pytest

from hap_atlantic.config import C0, H_PLANCK, AtmosphereParams, TerminalParams
from hap_atlantic.link import (
    cn2_hufnagel_valley,
    extinction_db,
    fade_margin_db,
    geometric_gain,
    link_rate,
    rytov_spherical,
    scintillation_index,
)

TERM, ATM = TerminalParams(), AtmosphereParams()


def test_hv_ground_value_and_clipping():
    expected = ATM.hv_a + 2.7e-16
    assert cn2_hufnagel_valley(0.0, ATM) == pytest.approx(expected)
    assert cn2_hufnagel_valley(-100.0, ATM) == pytest.approx(expected)


def test_hv_decreases_in_boundary_layer_and_has_tropopause_bump():
    h = np.array([0.0, 500.0, 1000.0, 3000.0])
    assert np.all(np.diff(cn2_hufnagel_valley(h, ATM)) < 0)
    # HV 5/7: the high-altitude term peaks near 10 km
    assert cn2_hufnagel_valley(10e3, ATM) > cn2_hufnagel_valley(5e3, ATM)


def test_geometric_gain_limits():
    assert geometric_gain(1.0, TERM) == pytest.approx(1.0)
    g = geometric_gain(np.array([10e3, 100e3, 700e3]), TERM)
    assert np.all(np.diff(g) < 0)
    # far field: fraction ~ 2 a^2 / w^2
    w = TERM.divergence / 2 * 700e3
    assert g[-1] == pytest.approx(2 * (TERM.d_rx / 2) ** 2 / w**2, rel=1e-3)


def test_extinction_sea_level_and_altitude():
    assert extinction_db(np.zeros((1, 8)), 1e3, ATM)[0] == pytest.approx(ATM.alpha0_db_per_km)
    assert extinction_db(np.full((1, 8), 20e3), 100e3, ATM)[0] < 1e-5
    assert extinction_db(np.zeros((1, 8)), 0.0, ATM)[0] == 0.0


def test_rytov_constant_cn2_matches_closed_form():
    # constant Cn^2 (only the 2.7e-16 term at h = 0): beta0^2 = 0.4964 Cn^2 k^(7/6) L^(11/6)
    atm = replace(ATM, hv_a=0.0, hv_wind=0.0)
    L = 50e3
    b = rytov_spherical(np.zeros((1, 2000)), L, TERM, atm)[0]
    k = 2 * np.pi / TERM.wavelength
    assert b == pytest.approx(0.4964 * 2.7e-16 * k ** (7 / 6) * L ** (11 / 6), rel=5e-3)


def test_scintillation_weak_limit():
    assert scintillation_index(1e-4) == pytest.approx(1e-4, rel=1e-2)


def test_scintillation_saturates_near_one():
    assert scintillation_index(1e6) == pytest.approx(1.0, abs=0.05)


def test_fade_margin_properties():
    assert fade_margin_db(0.0, 1e-3) == pytest.approx(0.0)
    m = fade_margin_db(np.array([0.1, 0.5, 1.5]), 1e-3)
    assert np.all(m > 0) and np.all(np.diff(m) > 0)
    assert fade_margin_db(0.5, 1e-2) < fade_margin_db(0.5, 1e-4)


def test_fade_margin_lognormal_quantile():
    # P(I < I_th) = p for a unit-mean log-normal I
    from scipy.stats import lognorm

    s2 = np.log1p(0.7)
    i_th = 10 ** (-fade_margin_db(0.7, 1e-3) / 10)
    assert lognorm(s=np.sqrt(s2), scale=np.exp(-s2 / 2)).cdf(i_th) == pytest.approx(1e-3)


def test_link_rate_monotone_and_capped():
    d = np.array([50e3, 300e3, 700e3])
    heights = np.full((3, 16), 11e3)
    r = link_rate(d, heights, TERM, ATM)
    assert np.all(np.diff(r) <= 0) and r.max() <= TERM.r_max


def test_link_rate_linear_in_power_and_loss():
    d, h = np.array([400e3]), np.full((1, 16), 11e3)
    r1 = link_rate(d, h, TERM, ATM, capped=False)
    r5 = link_rate(d, h, replace(TERM, p_tx=5.0), ATM, capped=False)
    r_loss = link_rate(d, h, TERM, ATM, extra_loss_db=10.0, capped=False)
    assert r5[0] == pytest.approx(5 * r1[0]) and r_loss[0] == pytest.approx(r1[0] / 10)


def test_link_rate_photon_budget():
    # no losses at all: R = P eta G / (n_ppb h nu)
    term = replace(TERM, pointing_loss_db=0.0, outage_prob=0.5)
    atm = replace(ATM, alpha0_db_per_km=0.0, hv_a=0.0, hv_wind=0.0)
    d = np.array([100e3])
    h = np.full((1, 16), 30e3)  # Cn^2 practically zero up there
    r = link_rate(d, h, term, atm, capped=False)[0]
    expected = (term.p_tx * term.eta_optics * geometric_gain(d, term)[0]
                / (term.photons_per_bit * H_PLANCK * C0 / term.wavelength))
    assert r == pytest.approx(expected, rel=1e-3)


def test_link_budget_examples_quoted_in_paper():
    from hap_atlantic import geometry as geo

    R = 6_371_008.8
    out = []
    for d in (100e3, 600e3):
        ang = 2 * np.arcsin(d / 2 / (R + 11e3))
        a, b = geo.to_ecef(0, 0, 11e3, R), geo.to_ecef(np.degrees(ang), 0, 11e3, R)
        h = geo.chord_heights(a[None], b[None], R, 16)
        out.append(link_rate(np.array([d]), h, TERM, ATM, capped=False)[0])
    assert 100e12 < out[0] < 140e12  # "about 120 Tbit/s at 100 km"
    assert 0.6e12 < out[1] < 0.8e12  # "0.7 Tbit/s at 600 km"
