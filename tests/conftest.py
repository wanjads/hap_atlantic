"""Shared fixtures: the tests must not depend on downloaded data (data/ is not in git)."""

import numpy as np
import pytest

from hap_atlantic import topology


def synthetic_land_sea_mask():
    """0.25 deg grid; land east of 10 deg W ("Europe") and west of 60 deg W ("America")."""
    lat = np.arange(30.0, 70.001, 0.25)
    lon = np.arange(-90.0, 20.001, 0.25)
    glon = np.broadcast_to(lon, (len(lat), len(lon)))
    lsm = ((glon >= -10.0) | (glon <= -60.0)).astype(float)
    return lat, lon, lsm


@pytest.fixture(autouse=True)
def _synthetic_land(monkeypatch):
    monkeypatch.setattr(topology, "land_sea_mask", synthetic_land_sea_mask)
    topology.land_points.cache_clear()
    yield
    topology.land_points.cache_clear()
