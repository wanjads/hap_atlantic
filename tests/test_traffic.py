import numpy as np
import pandas as pd
import pytest

from hap_atlantic import geometry as geo
from hap_atlantic.traffic import (
    aircraft_table,
    interpolate_to_grid,
    load_aircraft,
    load_states,
    normalise,
    select_fleet,
    snapshots,
    synthetic_states,
)

R = 6_371_008.8


def _two_reports(gap_s, speed, alt=(11e3, 11e3), lat=50.0):
    """Two position reports of one aircraft flying west at `speed` with a gap of gap_s."""
    d = speed * gap_s
    lat2, lon2 = geo.destination(lat, -20.0, 270.0, d, R)
    return pd.DataFrame({"time": [0.0, gap_s], "icao24": "abc123", "lat": [lat, float(lat2)],
                         "lon": [-20.0, float(lon2)], "alt": list(alt)})


@pytest.mark.parametrize("gap_h,speed,bridged", [
    (5.5, 230.0, True),    # typical ocean crossing (previously lost with max_gap = 5 h)
    (7.5, 150.0, False),   # too slow for a continuous crossing
    (9.0, 250.0, False),   # longer than max_gap
    (0.5, 150.0, True),    # < 1 h: 120 m/s suffices
    (0.1, 20.0, True),     # <= 10 min: always bridged
    (2.0, 360.0, False),   # faster than any airliner
])
def test_gap_rules(gap_h, speed, bridged):
    g = interpolate_to_grid(_two_reports(gap_h * 3600, speed), dt=60.0)
    assert (len(g) > 0) == bridged


def test_interpolated_flag_and_count():
    g = interpolate_to_grid(_two_reports(5.5 * 3600, 230.0), dt=60.0)
    assert len(g) == 5.5 * 60 + 1 and g["interpolated"].all()
    g = interpolate_to_grid(_two_reports(300.0, 230.0), dt=60.0)
    assert not g["interpolated"].any()


def test_interpolation_on_great_circle_at_constant_speed():
    raw = _two_reports(3 * 3600, 240.0)
    g = interpolate_to_grid(raw, dt=600.0)
    d = geo.great_circle(raw.lat[0], raw.lon[0], g.lat.to_numpy(), g.lon.to_numpy(), R)
    assert np.allclose(d, 240.0 * g.time.to_numpy(), rtol=1e-6)


def test_altitude_linear():
    g = interpolate_to_grid(_two_reports(1200.0, 230.0, alt=(10e3, 12e3)), dt=600.0)
    assert np.allclose(g.alt, [10e3, 11e3, 12e3])


def test_single_report_and_empty_input():
    one = _two_reports(600.0, 230.0).iloc[:1]
    assert len(interpolate_to_grid(one, dt=60.0)) == 0
    empty = interpolate_to_grid(one.iloc[:0], dt=60.0)
    assert list(empty.columns) == ["time", "icao24", "lat", "lon", "alt", "interpolated"]


def test_normalise_datetime_and_icao():
    df = pd.DataFrame({"time": pd.to_datetime([0, 60], unit="s", utc=True),
                       "icao24": [" ABC123 ", "abc123"], "lat": [1.0, np.nan],
                       "lon": [2.0, 3.0], "alt": [100.0, 200.0]})
    out = normalise(df)
    assert list(out.time) == [0.0] and list(out.icao24) == ["abc123"]


def test_load_states_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"time": [0.0, 0.0, 60.0], "icao24": ["a", "a", "a"],
                       "lat": [1.0, 1.0, 2.0], "lon": [1.0, 1.0, 2.0], "alt": [1e3] * 3})
    f = tmp_path / "s.csv"
    df.to_csv(f, index=False)
    out = load_states(tmp_path)
    assert len(out) == 2  # duplicate (icao24, time) removed
    with pytest.raises(FileNotFoundError):
        load_states(tmp_path / "empty_dir_that_does_not_exist" / "x")


def test_aircraft_table_path_and_loading(tmp_path):
    raw = tmp_path / "adsblol_20230709.parquet"
    assert aircraft_table(raw).name == "adsblol_20230709_aircraft.parquet"
    assert load_aircraft([raw]) is None
    pd.DataFrame({"icao24": ["a"], "type": ["B77W"], "military": [False]}).to_parquet(
        aircraft_table(raw))
    assert list(load_aircraft([raw]).type) == ["B77W"]


def test_select_fleet():
    df = pd.DataFrame({"icao24": list("abcd"), "type": ["B77W", "GLF6", "A20N", "C17"],
                       "military": [False, False, False, True], "time": 0.0})
    assert list(select_fleet(df, True)["icao24"]) == ["a", "c"]
    assert len(select_fleet(df, False)) == 4


def test_select_fleet_military_airliner_and_missing_type():
    df = pd.DataFrame({"icao24": ["a", "b"], "type": ["A332", "A332"],
                       "military": [True, False]})
    assert list(select_fleet(df)["icao24"]) == ["b"]
    no_type = pd.DataFrame({"icao24": ["a"], "time": [0.0]})
    assert len(select_fleet(no_type)) == 1


def test_synthetic_states_and_snapshots():
    s = synthetic_states(n_flights=50, n_regional=20, duration=3600, dt=600, seed=2)
    assert {"time", "icao24", "lat", "lon", "alt", "interpolated"} <= set(s.columns)
    assert s.alt.max() <= 11_600.0 + 1e-6 and s.time.between(0, 3600).all()
    times = [t for t, _ in snapshots(s)]
    assert times == sorted(times)


def test_pipeline_drop_half_and_reinterpolate():
    states = synthetic_states(n_flights=300, duration=3600, dt=300, seed=1)
    raw = states.drop(columns="interpolated").iloc[::2]
    grid = interpolate_to_grid(raw, dt=300)
    assert grid["icao24"].nunique() > 0 and (grid.time % 300 == 0).all()
