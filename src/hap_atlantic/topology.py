"""Candidate link graph for one traffic snapshot.

Nodes
    "SRC", "DST"          endpoints (Frankfurt, New York), terrestrial
    "gsS<i>", "gsD<i>"    optical ground stations around the endpoints
    <icao24>              aircraft
Edges (undirected, full-duplex; attribute `cap` per direction)
    kind="fiber"   endpoint <-> ground station, unlimited capacity, no terminal used
    kind="ground"  ground station <-> aircraft (medium "optical" or "rf")
    kind="air"     aircraft <-> aircraft
`rate` is the uncapped link-budget rate at the configured transmit power, `cap` is
min(rate, r_max). `with_tech` re-caps a graph for another technology tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import geometry as geo
from .config import C0, N_AIR, N_FIBER, NORTH_ATLANTIC_BBOX, Params, Site
from .link import link_rate
from .weather import CloudState, land_sea_mask

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


def yao_sparsify(pairs: np.ndarray, d: np.ndarray, lat, lon, cones: int) -> np.ndarray:
    """Mask of pairs that are the nearest or farthest neighbour of either end in some cone.

    The nearest-neighbour-per-cone (Yao) graph is a spanner, i.e. it preserves short paths;
    the farthest neighbour per cone keeps the long hops that make low-latency paths.
    """
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dd = np.concatenate([d, d])
    cone = (geo.initial_bearing(lat[i], lon[i], lat[j], lon[j]) * cones / 360.0).astype(int)
    key = i.astype(np.int64) * cones + np.minimum(cone, cones - 1)
    df = pd.DataFrame({"key": key, "d": dd, "pair": np.concatenate([np.arange(len(d))] * 2)})
    g = df.groupby("key")["d"]
    sel = np.union1d(df.loc[g.idxmin(), "pair"].to_numpy(), df.loc[g.idxmax(), "pair"].to_numpy())
    keep = np.zeros(len(d), dtype=bool)
    keep[sel] = True
    return keep


def build_graph(snap: pd.DataFrame, p: Params, clouds: CloudState | None = None) -> nx.Graph:
    """Candidate graph for one snapshot (aircraft positions in `snap`).

    clouds: cloud state at the snapshot time (None = clear sky). The cloud loss of every ray
    enters its link budget; links below r_min are dropped before the Yao sparsification,
    so that clear alternatives survive.
    """
    g_par, term, atm = p.geometry, p.terminal, p.atmosphere
    r_eff = g_par.earth_radius * g_par.k_refraction
    snap = snap[snap["alt"] >= g_par.h_node_min]
    corr = geo.corridor_distance(snap["lat"].to_numpy(), snap["lon"].to_numpy(), p.source.lat,
                                 p.source.lon, p.sink.lat, p.sink.lon, g_par.earth_radius)
    snap = snap[corr <= g_par.corridor_half_width]
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
            loss = clouds.ray_loss_db(a, b) if clouds is not None else 0.0
            rate = link_rate(d, geo.chord_heights(a, b, r_eff, g_par.ray_samples), term, atm,
                             extra_loss_db=loss, capped=False)
            ok = rate >= term.r_min  # cloud-blocked rays drop out before the sparsification
            pairs, d, rate = pairs[ok], d[ok], rate[ok]
            if g_par.yao_cones and len(pairs):
                keep = yao_sparsify(pairs, d, snap["lat"].to_numpy(), snap["lon"].to_numpy(),
                                    g_par.yao_cones)
                pairs, d, rate = pairs[keep], d[keep], rate[keep]
            for (i, j), dist, r in zip(pairs, d, rate, strict=True):
                G.add_edge(ids[i], ids[j], kind="air", rate=float(r),
                           cap=float(min(r, term.r_max)), dist=float(dist),
                           delay=float(dist * N_AIR / C0 + term.hop_delay))

    # --- ground stations ------------------------------------------------------------------
    for gs in ground_stations(p):
        zen = 0.0 if clouds is None else clouds.zenith_loss_db(gs.site.lat, gs.site.lon)
        G.add_node(gs.name, kind="ground", terminals=p.ground.terminals_per_site,
                   lat=gs.site.lat, lon=gs.site.lon, alt=gs.site.alt, zenith_loss_db=zen,
                   clear=zen < 3.0)
        G.add_edge(gs.endpoint, gs.name, kind="fiber", rate=np.inf, cap=np.inf, dist=0.0,
                   delay=gs.fiber_delay)
        if len(ids) == 0:
            continue
        gpos = geo.to_ecef(gs.site.lat, gs.site.lon, gs.site.alt, r_eff)
        el = geo.elevation_deg(np.broadcast_to(gpos, pos.shape), pos)
        sel = np.flatnonzero(el >= g_par.elev_min_ground)
        if sel.size == 0:
            continue
        a = np.broadcast_to(gpos, (sel.size, 3))
        d = np.linalg.norm(pos[sel] - a, axis=-1)
        loss = clouds.ray_loss_db(np.ascontiguousarray(a), pos[sel]) if clouds is not None else 0.0
        rate = link_rate(d, geo.chord_heights(a, pos[sel], r_eff, g_par.ray_samples), term, atm,
                         extra_loss_db=loss, capped=False)
        rf = p.ground.rf_rate > 0 and (clouds is None or clouds.rain_ok(
            gs.site.lat, gs.site.lon, p.ground.rf_rain_max))
        for k, dist, r in zip(sel, d, rate, strict=True):
            if r >= term.r_min:
                medium, r = "optical", float(r)
            elif rf:
                medium, r = "rf", p.ground.rf_rate
            else:
                continue
            G.add_edge(gs.name, ids[k], kind="ground", medium=medium, rate=r,
                       cap=min(r, term.r_max), dist=float(dist),
                       delay=float(dist * N_AIR / C0 + term.hop_delay))
    return G


def with_tech(G: nx.Graph, r_max: float, k: int, p_scale: float = 1.0) -> nx.Graph:
    """Copy of G for another technology tier: optical rates scaled by p_scale (transmit
    power ratio; the budget is linear in P_tx), capped at r_max; k terminals per aircraft."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        if d["kind"] == "fiber":
            continue
        rate = d["rate"] * (p_scale if d.get("medium", "optical") == "optical" else 1.0)
        d["cap"] = min(rate, r_max)
    for _, d in H.nodes(data=True):
        if d["kind"] == "aircraft":
            d["terminals"] = k
    return H



@lru_cache(maxsize=4)
def land_points(lon_eu: float, lon_na: float, radius: float):
    """Land grid points (lat, lon, ECEF) of Europe and North America inside the corridor box."""
    lat, lon, lsm = land_sea_mask()
    lamin, lamax, lomin, lomax = NORTH_ATLANTIC_BBOX
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    box = (glat >= lamin) & (glat <= lamax) & (glon >= lomin) & (glon <= lomax) & (lsm >= 0.5)
    greenland = (glat > 59.5) & (glon > -62.0)
    out = []
    for sel in (box & (glon >= lon_eu), box & (glon <= lon_na) & ~greenland):
        la, lo = glat[sel], glon[sel]
        out.append((la, lo, geo.to_ecef(la, lo, 0.0, radius)))
    return out


def land_graph(G: nx.Graph, p: Params, clouds: CloudState | None = None) -> nx.Graph:
    """Capacity graph: aircraft of G with access links from ground stations everywhere on land.

    SRC = all European land stations, DST = all North American land stations (unlimited
    terminals). An aircraft gets one access link per side if a land point sees it at
    >= elev_min_ground; of the land_candidates nearest points the best link budget (incl.
    cloud loss) is used, else the RF fallback of the nearest point (no heavy rain).
    """
    g_par, term, atm = p.geometry, p.terminal, p.atmosphere
    r_eff = g_par.earth_radius * g_par.k_refraction
    ac = [(v, d) for v, d in G.nodes(data=True) if d.get("kind") == "aircraft"]
    H = G.subgraph([v for v, _ in ac]).copy()
    H.add_node(SRC, kind="endpoint", terminals=np.inf)
    H.add_node(DST, kind="endpoint", terminals=np.inf)
    if not ac:
        return H
    ids = np.array([v for v, _ in ac], dtype=object)
    pos = geo.to_ecef(np.array([d["lat"] for _, d in ac]), np.array([d["lon"] for _, d in ac]),
                      np.array([d["alt"] for _, d in ac]), r_eff)
    e = np.radians(g_par.elev_min_ground)
    h_max = max(d["alt"] for _, d in ac)
    s_max = np.sqrt((r_eff + h_max) ** 2 - (r_eff * np.cos(e)) ** 2) - r_eff * np.sin(e)
    sides = land_points(p.ground.land_eu_lon, p.ground.land_na_lon, g_par.earth_radius)
    for end, (la, lo, xyz) in zip((SRC, DST), sides, strict=True):
        if len(xyz) == 0:
            continue
        k = p.ground.land_candidates
        dist, idx = cKDTree(xyz).query(pos, k=k, distance_upper_bound=s_max)
        i_ac, i_c = np.nonzero(np.isfinite(dist))
        if i_ac.size == 0:
            continue
        cell = idx[i_ac, i_c]
        a, b = xyz[cell], pos[i_ac]
        ok = geo.elevation_deg(a, b) >= g_par.elev_min_ground
        i_ac, cell, a, b = i_ac[ok], cell[ok], a[ok], b[ok]
        d = np.linalg.norm(b - a, axis=-1)
        loss = clouds.ray_loss_db(a, b) if clouds is not None else 0.0
        rate = link_rate(d, geo.chord_heights(a, b, r_eff, g_par.ray_samples), term, atm,
                         extra_loss_db=loss, capped=False)
        best = pd.DataFrame({"ac": i_ac, "rate": rate, "cell": cell, "d": d})
        for i, grp in best.groupby("ac"):
            r = grp["rate"].max()
            if r >= term.r_min:
                medium = "optical"
            elif p.ground.rf_rate > 0 and (clouds is None or clouds.rain_ok(
                    la[grp["cell"].iloc[grp["d"].argmin()]],
                    lo[grp["cell"].iloc[grp["d"].argmin()]], p.ground.rf_rain_max)):
                medium, r = "rf", p.ground.rf_rate
            else:
                continue
            H.add_edge(end, ids[i], kind="ground", medium=medium, rate=float(r),
                       cap=float(min(r, term.r_max)), dist=0.0, delay=0.0)
    return H
