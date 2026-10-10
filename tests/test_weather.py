from types import SimpleNamespace

import numpy as np
import pytest

from hap_atlantic import geometry as geo
from hap_atlantic.config import CloudParams
from hap_atlantic.weather import DB, RHO_ICE, CloudState, LatentField, isa_density, isa_height

R = 6_371_008.8


def test_isa_height_reference_points():
    assert isa_height(1013.25) == pytest.approx(0.0, abs=1e-6)
    assert abs(isa_height(250.0) - 10_363) < 50
    assert np.all(np.diff(isa_height(np.array([1000.0, 700.0, 500.0, 300.0, 150.0]))) > 0)


def test_isa_density_sea_level():
    assert isa_density(1013.25) == pytest.approx(1.225, rel=2e-3)


def test_cloud_extinction_ice():
    # 0.01 g/m^3 of ice with r_e = 25 um: about 0.65 / km, i.e. ~2.8 dB/km
    beta = 3 * 1e-5 / (2 * RHO_ICE * 25e-6)
    assert 2.5 < DB * beta * 1e3 < 3.3


def test_latent_field_deterministic():
    x = np.random.default_rng(1).uniform(-5e6, 5e6, (100, 3))
    a = LatentField(50e3, 2e3, 3600.0, 256, seed=7)(x, 10.0)
    b = LatentField(50e3, 2e3, 3600.0, 256, seed=7)(x, 10.0)
    c = LatentField(50e3, 2e3, 3600.0, 256, seed=8)(x, 10.0)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_latent_field_statistics():
    z = LatentField(50e3, 2e3, 3600.0, 2048, seed=3)
    x = np.random.default_rng(0).uniform(-5e6, 5e6, (4000, 3))
    v = z(x, 0.0)
    assert abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.1
    near, far = z(x + [5e3, 0, 0], 0.0), z(x + [200e3, 0, 0], 0.0)
    assert np.corrcoef(v, near)[0, 1] > 0.9 and abs(np.corrcoef(v, far)[0, 1]) < 0.15
    assert np.corrcoef(v, z(x, 0.0, 500.0))[0, 1] > 0.9
    assert abs(np.corrcoef(v, z(x, 0.0, 8e3))[0, 1]) < 0.15


def test_latent_field_time_correlation():
    z = LatentField(50e3, 2e3, 3600.0, 2048, seed=5)
    x = np.random.default_rng(2).uniform(-5e6, 5e6, (4000, 3))
    v = z(x, 0.0)
    assert np.corrcoef(v, z(x, 300.0))[0, 1] > 0.9
    assert abs(np.corrcoef(v, z(x, 6 * 3600.0))[0, 1]) < 0.15


def _state(beta=1e-4, top=11e3, bottom=9e3, rain=0.0):
    """Synthetic cloud state: uniform extinction beta [1/m] between bottom and top."""
    heights = np.arange(0.0, 13_001.0, 500.0)
    lats, lons = np.arange(30.0, 70.001, 0.25), np.arange(-80.0, 15.001, 0.25)
    field = SimpleNamespace(heights=heights, lats=lats, lons=lons, radius=R,
                            cp=CloudParams(ray_samples=200))
    ext = np.zeros((len(heights), len(lats), len(lons)))
    ext[(heights >= bottom) & (heights <= top)] = beta
    return CloudState(field, ext, np.full((len(lats), len(lons)), rain))


def test_ray_loss_horizontal_inside_cloud():
    cs = _state(beta=1e-4)
    a = geo.to_ecef(50.0, -30.0, 10e3, R)[None]
    b = geo.to_ecef(50.0, -29.0, 10e3, R)[None]
    d = np.linalg.norm(b - a)
    assert cs.ray_loss_db(a, b)[0] == pytest.approx(DB * 1e-4 * d, rel=0.02)


def test_ray_loss_above_cloud_is_zero():
    cs = _state(beta=1e-4, top=8e3, bottom=5e3)
    a = geo.to_ecef(50.0, -30.0, 11e3, R)[None]
    b = geo.to_ecef(50.0, -29.5, 11e3, R)[None]
    assert cs.ray_loss_db(a, b)[0] == 0.0
    assert len(cs.ray_loss_db(np.zeros((0, 3)), np.zeros((0, 3)))) == 0


def test_zenith_loss_thin_layer():
    cs = _state(beta=1e-3, top=10_000.0, bottom=9_000.0)
    # the layer occupies the levels 9.0, 9.5 and 10.0 km, i.e. heights 8.75-10.25 km;
    # 64 samples over 13 km resolve this to about one sample (203 m)
    assert cs.zenith_loss_db(50.0, 8.0) == pytest.approx(DB * 1e-3 * 1500.0, rel=0.1)


def test_rain_ok_threshold():
    assert _state(rain=4.0).rain_ok(50.0, 8.0, 5.0)
    assert not _state(rain=6.0).rain_ok(50.0, 8.0, 5.0)
