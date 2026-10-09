"""Candidate link graph for one traffic snapshot.

Nodes
    "SRC", "DST"          endpoints (Frankfurt, New York), terrestrial
    "gsS<i>", "gsD<i>"    optical ground stations around the endpoints
    <icao24>              aircraft
Edges (undirected, full-duplex; attribute `cap` per direction)
    kind="fiber"   endpoint <-> ground station, unlimited capacity, no terminal used
    kind="ground"  ground station <-> aircraft
    kind="air"     aircraft <-> aircraft
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import geometry as geo
from .config import C0, N_AIR, N_FIBER, Params, Site
from .link import link_rate

SRC, DST = "SRC", "DST"


@dataclass(frozen=True)
class GroundStation:
    name: str
    endpoint: str  # SRC or DST
    site: Site
    fiber_delay: float  # one-way delay to the endpoint [s]


def ground_stations(p: Params) -> list[GroundStation]:
    """Ground stations: the endpoint itself plus (n_sites - 1) sites on a ring around it."""
    out = []
    for endpoint, site, tag in ((SRC, p.source, "S"), (DST, p.sink, "D")):
        n = p.ground.n_sites
        for i in range(n):
            if i == 0:
                lat, lon, dist = site.lat, site.lon, 0.0
            else:
                brg = 360.0 * (i - 1) / (n - 1)
                lat, lon = geo.destination(site.lat, site.lon, brg, p.ground.ring_radius,
                                           p.geometry.earth_radius)
                dist = p.ground.ring_radius
            delay = dist * p.ground.fiber_detour * N_FIBER / C0
            out.append(GroundStation(f"gs{tag}{i}", endpoint,
                                     Site(f"{site.name} {i}", float(lat), float(lon), site.alt),
                                     delay))
    return out


class CloudProcess:
    """Two-state Markov chain per ground station: cloud-free line of sight or blocked."""

    def __init__(self, n: int, p_clear: float, corr_time: float, dt: float, rng):
        self.rng = rng
        # transition probabilities with stationary P(clear) = p_clear and
        # mean sojourn ~ corr_time in the rarer state
        mean_clear = corr_time * max(p_clear / (1 - p_clear), 1.0) if p_clear < 1 else np.inf
        mean_cloud = corr_time * max((1 - p_clear) / max(p_clear, 1e-9), 1.0)
        self.p_leave_clear = 0.0 if np.isinf(mean_clear) else min(dt / mean_clear, 1.0)
        self.p_leave_cloud = min(dt / mean_cloud, 1.0)
        self.state = rng.random(n) < p_clear

    def step(self) -> np.ndarray:
        u = self.rng.random(self.state.size)
        leave = np.where(self.state, u < self.p_leave_clear, u < self.p_leave_cloud)
        self.state = np.where(leave, ~self.state, self.state)
        return self.state.copy()


def build_graph(snap: pd.DataFrame, p: Params, clear: np.ndarray | None = None) -> nx.Graph:
    """Candidate graph for one snapshot (aircraft positions in `snap`)."""
    g_par, term, atm = p.geometry, p.terminal, p.atmosphere
    r_eff = g_par.earth_radius * g_par.k_refraction
    snap = snap[snap["alt"] >= g_par.h_node_min]
    ids = snap["icao24"].to_numpy()
    pos = geo.to_ecef(snap["lat"].to_numpy(), snap["lon"].to_numpy(), snap["alt"].to_numpy(),
                      r_eff)

    G = nx.Graph()
    G.add_node(SRC, kind="endpoint", terminals=np.inf)
    G.add_node(DST, kind="endpoint", terminals=np.inf)
    for icao, row in zip(ids, snap.itertuples(index=False), strict=True):
        G.add_node(icao, kind="aircraft", terminals=p.terminals_per_aircraft,
                   lat=row.lat, lon=row.lon, alt=row.alt)

    # --- aircraft <-> aircraft -----------------------------------------------------------
    if len(ids) > 1:
        pairs = cKDTree(pos).query_pairs(g_par.d_max_air, output_type="ndarray")
        if len(pairs):
            a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
            ok = geo.chord_min_height(a, b, r_eff) >= g_par.h_chord_min
            pairs, a, b = pairs[ok], a[ok], b[ok]
            d = np.linalg.norm(b - a, axis=-1)
            rate = link_rate(d, geo.chord_heights(a, b, r_eff), term, atm)
            for (i, j), dist, r in zip(pairs, d, rate, strict=True):
                if r > 0:
                    G.add_edge(ids[i], ids[j], kind="air", cap=float(r), dist=float(dist),
                               delay=float(dist * N_AIR / C0 + term.hop_delay))

    # --- ground stations ------------------------------------------------------------------
    stations = ground_stations(p)
    if clear is None:
        clear = np.ones(len(stations), dtype=bool)
    for gs, is_clear in zip(stations, clear, strict=True):
        G.add_node(gs.name, kind="ground", terminals=p.ground.terminals_per_site,
                   lat=gs.site.lat, lon=gs.site.lon, alt=gs.site.alt, clear=bool(is_clear))
        G.add_edge(gs.endpoint, gs.name, kind="fiber", cap=np.inf, dist=0.0,
                   delay=gs.fiber_delay)
        if not is_clear or len(ids) == 0:
            continue
        gpos = geo.to_ecef(gs.site.lat, gs.site.lon, gs.site.alt, r_eff)
        el = geo.elevation_deg(np.broadcast_to(gpos, pos.shape), pos)
        sel = np.flatnonzero(el >= g_par.elev_min_ground)
        if sel.size == 0:
            continue
        a = np.broadcast_to(gpos, (sel.size, 3))
        d = np.linalg.norm(pos[sel] - a, axis=-1)
        rate = link_rate(d, geo.chord_heights(a, pos[sel], r_eff), term, atm)
        for k, dist, r in zip(sel, d, rate, strict=True):
            if r > 0:
                G.add_edge(gs.name, ids[k], kind="ground", cap=float(r), dist=float(dist),
                           delay=float(dist * N_AIR / C0 + term.hop_delay))
    return G


def fiber_reference_delay(p: Params) -> dict[str, float]:
    """One-way reference delays [s] between the endpoints."""
    gc = float(geo.great_circle(p.source.lat, p.source.lon, p.sink.lat, p.sink.lon,
                                p.geometry.earth_radius))
    return {
        "great_circle_km": gc / 1e3,
        "vacuum_great_circle": gc / C0,
        "fiber_great_circle": gc * N_FIBER / C0,  # lower bound for any fibre route
    }
