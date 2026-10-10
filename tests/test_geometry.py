import numpy as np
import pytest

from hap_atlantic import geometry as geo
from hap_atlantic.config import FRANKFURT, NEW_YORK

R = 6_371_008.8
POINTS = [(50.11, 8.68, 11e3), (-33.9, 151.2, 0.0)]


@pytest.mark.parametrize("lat,lon,alt", POINTS)
def test_ecef_radius_and_roundtrip(lat, lon, alt):
    xyz = geo.to_ecef(lat, lon, alt, R)
    assert np.linalg.norm(xyz) == pytest.approx(R + alt)
    la, lo = geo.to_latlon(xyz)
    assert la == pytest.approx(lat, abs=1e-9) and lo == pytest.approx(lon, abs=1e-9)


def test_great_circle_one_degree_on_equator():
    assert geo.great_circle(0.0, 0.0, 0.0, 1.0, R) == pytest.approx(R * np.pi / 180)


def test_great_circle_symmetric_zero_and_antipodal():
    d1 = geo.great_circle(50.0, 8.0, 40.0, -74.0, R)
    assert d1 == pytest.approx(geo.great_circle(40.0, -74.0, 50.0, 8.0, R))
    assert geo.great_circle(12.0, 34.0, 12.0, 34.0, R) == pytest.approx(0.0, abs=1e-6)
    assert geo.great_circle(0.0, 0.0, 0.0, 180.0, R) == pytest.approx(np.pi * R)


def test_great_circle_fra_ny():
    d = geo.great_circle(FRANKFURT.lat, FRANKFURT.lon, NEW_YORK.lat, NEW_YORK.lon, R)
    assert 6150e3 < d < 6250e3


@pytest.mark.parametrize("bearing", [45.0, 225.0])
def test_destination_distance_and_bearing(bearing):
    lat2, lon2 = geo.destination(50.0, 8.0, bearing, 150e3, R)
    assert geo.great_circle(50.0, 8.0, lat2, lon2, R) == pytest.approx(150e3, rel=1e-9)
    assert geo.initial_bearing(50.0, 8.0, lat2, lon2) == pytest.approx(bearing % 360, abs=1e-6)


def test_initial_bearing_cardinal():
    assert geo.initial_bearing(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)
    assert geo.initial_bearing(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0)
    assert geo.initial_bearing(0.0, 0.0, -10.0, 0.0) == pytest.approx(180.0)
    assert geo.initial_bearing(0.0, 10.0, 0.0, 0.0) == pytest.approx(270.0)


def test_slerp_endpoints():
    lat, lon = geo.slerp_latlon(50.0, 8.0, 40.0, -74.0, np.array([0.0, 1.0]))
    assert np.allclose(lat, [50.0, 40.0]) and np.allclose(lon, [8.0, -74.0])


def test_slerp_constant_speed():
    f = np.linspace(0, 1, 11)
    lat, lon = geo.slerp_latlon(50.0, 8.0, 40.0, -74.0, f)
    total = geo.great_circle(50.0, 8.0, 40.0, -74.0, R)
    d = geo.great_circle(50.0, 8.0, lat, lon, R)
    assert np.allclose(d, f * total, rtol=1e-9, atol=1e-3)


def test_line_of_sight_horizon():
    # two aircraft at 11 km see each other up to 2*sqrt(2 R h) ~ 749 km (grazing ray)
    h = 11e3
    d_horizon = 2 * np.sqrt(2 * R * h)
    a = geo.to_ecef(0.0, 0.0, h, R)
    b = geo.to_ecef(np.degrees(d_horizon / R) * 0.99, 0.0, h, R)
    assert 0 < geo.chord_min_height(a, b, R) < 500
    b_far = geo.to_ecef(np.degrees(d_horizon / R) * 1.05, 0.0, h, R)
    assert geo.chord_min_height(a, b_far, R) < 0


def test_chord_min_height_includes_endpoints():
    # the lowest point of a segment is never above its lower endpoint
    a = geo.to_ecef(45.0, -30.0, 2e3, R)
    b = geo.to_ecef(45.1, -30.0, 11e3, R)
    assert geo.chord_min_height(a, b, R) == pytest.approx(2e3, abs=1.0)


def test_chord_min_height_degenerate_segment():
    a = geo.to_ecef(10.0, 20.0, 5e3, R)
    assert geo.chord_min_height(a, a, R) == pytest.approx(5e3)


def test_chord_heights_shape_and_bounds():
    a = geo.to_ecef(np.zeros(3), np.zeros(3), np.full(3, 11e3), R)
    b = geo.to_ecef(np.array([1.0, 3.0, 5.0]), np.zeros(3), np.full(3, 11e3), R)
    h = geo.chord_heights(a, b, R, 16)
    assert h.shape == (3, 16)
    assert np.all(h <= 11e3 + 1e-6)
    assert np.all(h.min(axis=1) >= geo.chord_min_height(a, b, R) - 1e-6)


def test_elevation_overhead_and_horizon():
    g = geo.to_ecef(30.0, 10.0, 0.0, R)
    assert geo.elevation_deg(g, geo.to_ecef(30.0, 10.0, 10e3, R)) == pytest.approx(90.0)
    far = geo.to_ecef(30.0 + 5.0, 10.0, 10e3, R)  # 556 km away: below the horizon
    assert geo.elevation_deg(g, far) < 0


def test_corridor_distance():
    lat_a, lon_a, lat_b, lon_b = 0.0, 0.0, 0.0, 40.0
    on = geo.corridor_distance(np.array([0.0]), np.array([20.0]), lat_a, lon_a, lat_b, lon_b, R)
    off = geo.corridor_distance(np.array([1.0]), np.array([20.0]), lat_a, lon_a, lat_b, lon_b, R)
    beyond = geo.corridor_distance(np.array([0.0]), np.array([50.0]), lat_a, lon_a, lat_b,
                                   lon_b, R)
    assert on[0] == pytest.approx(0.0, abs=1e-6)
    assert off[0] == pytest.approx(R * np.pi / 180, rel=1e-6)
    assert beyond[0] == pytest.approx(10 * R * np.pi / 180, rel=1e-6)
