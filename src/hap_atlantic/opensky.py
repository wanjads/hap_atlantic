"""OpenSky Network data access.

Two routes:

1. REST API (any account, OAuth2 client credentials). State vectors are available only
   for the last hour, so we *record* them: `collect_states` polls a bounding box over the
   North Atlantic and writes one Parquet file per hour. A bounding box larger than
   400 deg^2 costs 4 credits per request; with 4000 credits/day this allows one request
   every ~90 s (8000 credits/day for active feeders).
2. Trino historical database (research access, `pip install pyopensky`): arbitrary past
   days in one query, see `fetch_history_trino`.

Credentials are read from the environment (OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET) or
from a credentials.json downloaded from the OpenSky account page. Never commit them.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)
API_URL = "https://opensky-network.org/api"

# lat_min, lat_max, lon_min, lon_max: covers Frankfurt, New York and the oceanic tracks
NORTH_ATLANTIC_BBOX = (35.0, 66.0, -80.0, 15.0)
CREDITS_PER_REQUEST = 4  # /states/all with a bounding box > 400 deg^2

STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact", "longitude",
    "latitude", "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate",
    "sensors", "geo_altitude", "squawk", "spi", "position_source", "category",
]


def load_credentials(path: str | Path | None = None) -> tuple[str, str]:
    cid, secret = os.environ.get("OPENSKY_CLIENT_ID"), os.environ.get("OPENSKY_CLIENT_SECRET")
    if cid and secret:
        return cid, secret
    candidates = [Path(path)] if path else [
        Path("credentials.json"),
        Path.home() / ".config" / "opensky" / "credentials.json",
    ]
    for c in candidates:
        if c.is_file():
            data = json.loads(c.read_text())
            cid = data.get("clientId") or data.get("client_id")
            secret = data.get("clientSecret") or data.get("client_secret")
            if cid and secret:
                return cid, secret
    raise RuntimeError(
        "OpenSky credentials not found: set OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET or put "
        "credentials.json into ~/.config/opensky/"
    )


class OpenSkyClient:
    def __init__(self, credentials_path=None, session: requests.Session | None = None):
        self.client_id, self.client_secret = load_credentials(credentials_path)
        self.session = session or requests.Session()
        self._token, self._expires = None, 0.0

    def _auth_header(self) -> dict:
        if self._token is None or time.time() > self._expires - 60:
            r = self.session.post(TOKEN_URL, data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }, timeout=30)
            r.raise_for_status()
            tok = r.json()
            self._token = tok["access_token"]
            self._expires = time.time() + float(tok.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        for _ in range(5):
            r = self.session.get(f"{API_URL}{endpoint}", params=params,
                                 headers=self._auth_header(), timeout=60)
            if r.status_code == 401:  # token expired early
                self._token = None
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
                log.warning("rate limited, sleeping %.0f s", wait)
                time.sleep(min(wait, 3600))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"OpenSky request failed repeatedly: {endpoint} {params}")

    def credits_remaining(self, r: requests.Response) -> int | None:
        v = r.headers.get("X-Rate-Limit-Remaining")
        return int(v) if v is not None else None

    def states(self, bbox=NORTH_ATLANTIC_BBOX, t: int | None = None) -> pd.DataFrame:
        lamin, lamax, lomin, lomax = bbox
        params = {"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax, "extended": 1}
        if t is not None:
            params["time"] = int(t)
        r = self._get("/states/all", params)
        data = r.json()
        rows = data.get("states") or []
        df = pd.DataFrame([row + [None] * (len(STATE_FIELDS) - len(row)) for row in rows],
                          columns=STATE_FIELDS)
        df["time"] = data.get("time")
        df = df.drop(columns=["sensors"])
        df.attrs["credits_remaining"] = self.credits_remaining(r)
        return df


def collect_states(out_dir: str | Path, interval: float = 90.0, duration: float = 86400.0,
                   bbox=NORTH_ATLANTIC_BBOX, credentials_path=None) -> None:
    """Poll /states/all and append to hourly Parquet files in out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = OpenSkyClient(credentials_path)
    t_end = time.time() + duration
    buffer, hour = [], None
    while time.time() < t_end:
        t0 = time.time()
        try:
            df = client.states(bbox)
        except (requests.RequestException, RuntimeError) as exc:
            log.error("request failed: %s", exc)
            time.sleep(interval)
            continue
        h = time.strftime("%Y%m%d_%H", time.gmtime(t0))
        if hour is not None and h != hour and buffer:
            _flush(buffer, out / f"states_{hour}.parquet")
            buffer = []
        hour = h
        buffer.append(df)
        log.info("%s: %d aircraft, credits left %s", h, len(df), df.attrs["credits_remaining"])
        time.sleep(max(0.0, interval - (time.time() - t0)))
    if buffer:
        _flush(buffer, out / f"states_{hour}.parquet")


def _flush(frames: list[pd.DataFrame], path: Path) -> None:
    df = pd.concat(frames, ignore_index=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def fetch_history_trino(start: str, stop: str, bbox=NORTH_ATLANTIC_BBOX) -> pd.DataFrame:
    """Historical state vectors from the OpenSky Trino database (needs research access).

    pyopensky reads its credentials from its own config file, see
    https://open-aviation.github.io/pyopensky/
    """
    from pyopensky.trino import Trino  # optional dependency

    lamin, lamax, lomin, lomax = bbox
    return Trino().history(start, stop, bounds=(lomin, lamin, lomax, lamax))
