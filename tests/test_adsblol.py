import gzip
import io
import json
import tarfile

import pytest

from hap_atlantic.adsblol import aircraft_info, parse_trace, read_archive

BBOX = (35.0, 66.0, -80.0, 15.0)


def _trace():
    return {"icao": "3C6444", "timestamp": 1000.0, "t": "A359", "dbFlags": 0, "trace": [
        [0, 50.0, 8.0, "ground", None, None, 0, None, None, "adsb_icao", None],
        [10, 50.1, 8.0, 10000, 300, 270, 0, 0, None, "adsb_icao", 10100],
        [20, 50.2, 8.0, 11000, 300, 270, 0, 0, None, "adsb_icao", None],
        [100, 50.3, 8.0, 12000, 300, 270, 0, 0, None, "adsb_icao", None],
        [150, 50.3, 8.0, None, 300, 270, 0, 0, None, "adsb_icao", None],
        [200, 10.0, 8.0, 35000, 450, 270, 0, 0, None, "adsb_icao", None],  # outside bbox
        [300, 51.0, 8.0, 5000, 250, 270, 2, 0, None, "adsb_icao", None],
    ]}


def test_parse_trace():
    df = parse_trace(_trace(), BBOX, min_dt=30.0)
    assert list(df["time"]) == [1010.0, 1100.0, 1300.0]
    assert set(df["icao24"]) == {"3c6444"}
    assert df["alt"].iloc[0] == pytest.approx(10100 * 0.3048)  # geometric altitude preferred
    assert df["alt"].iloc[1] == pytest.approx(12000 * 0.3048)  # barometric as fallback


def test_parse_trace_nothing_inside():
    data = _trace()
    data["trace"] = [[0, 0.0, 0.0, 30000, 0, 0, 0, 0, None, "x", None]]
    assert parse_trace(data, BBOX, 30.0) is None


@pytest.mark.parametrize("flags,military", [(0, False), (3, True)])
def test_aircraft_info_military_bit(flags, military):
    data = dict(_trace(), dbFlags=flags)
    assert aircraft_info(data) == ("3c6444", "A359", military)


def test_aircraft_info_missing_type():
    data = _trace()
    del data["t"]
    assert aircraft_info(data)[1] == ""


def test_read_archive(tmp_path):
    f = tmp_path / "day.tar"
    with tarfile.open(f, "w") as tar:
        for name, payload in (("traces/44/trace_full_3c6444.json",
                               gzip.compress(json.dumps(_trace()).encode())),
                              ("README.txt", b"not a trace")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    pos, ac = read_archive(f, BBOX, progress=False)
    assert len(pos) == 3 and list(ac.columns) == ["icao24", "type", "military"]
    assert ac.iloc[0].type == "A359"
