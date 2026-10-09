"""Spherical-Earth geometry: positions, distances, line of sight.

A spherical Earth with an optional effective-radius factor is used throughout. Over the
distances considered (<1000 km per hop) the error with respect to WGS84 is well below
the uncertainty of the altitude constraints.
"""

from __future__ import annotations

import numpy as np


def to_ecef(lat_deg, lon_deg, alt, radius: float) -> np.ndarray:
    """Cartesian coordinates [m], shape (..., 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    r = radius + np.asarray(alt, dtype=float)
    return np.stack(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)], axis=-1
    )


def great_circle(lat1, lon1, lat2, lon2, radius: float) -> np.ndarray:
    """Great-circle distance on the sphere [m] (haversine)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def chord_min_height(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Minimum height above the sphere of the straight segments a->b, shape (...,)."""
    d = b - a
    dd = np.einsum("...i,...i->...", d, d)
    t = np.where(dd > 0, -np.einsum("...i,...i->...", a, d) / np.where(dd > 0, dd, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    p = a + t[..., None] * d
    return np.linalg.norm(p, axis=-1) - radius


def chord_heights(a: np.ndarray, b: np.ndarray, radius: float, n: int = 32) -> np.ndarray:
    """Heights above the sphere of n equally spaced points on each segment, shape (..., n)."""
    s = (np.arange(n) + 0.5) / n
    p = a[..., None, :] + s[:, None] * (b - a)[..., None, :]
    return np.linalg.norm(p, axis=-1) - radius


def elevation_deg(ground: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Elevation angle of target seen from ground [deg]."""
    v = target - ground
    up = ground / np.linalg.norm(ground, axis=-1, keepdims=True)
    s = np.einsum("...i,...i->...", v, up) / np.linalg.norm(v, axis=-1)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def destination(lat_deg, lon_deg, bearing_deg, dist, radius: float):
    """Point reached from (lat, lon) after dist [m] on the given initial bearing."""
    lat, lon, brg = np.radians(lat_deg), np.radians(lon_deg), np.radians(bearing_deg)
    ang = dist / radius
    lat2 = np.arcsin(np.sin(lat) * np.cos(ang) + np.cos(lat) * np.sin(ang) * np.cos(brg))
    lon2 = lon + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat), np.cos(ang) - np.sin(lat) * np.sin(lat2)
    )
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def slerp_latlon(lat1, lon1, lat2, lon2, f):
    """Point at fraction f along the great circle from 1 to 2 (vectorised)."""
    a = to_ecef(lat1, lon1, 0.0, 1.0)
    b = to_ecef(lat2, lon2, 0.0, 1.0)
    omega = np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1.0, 1.0))
    so = np.sin(omega)
    f = np.asarray(f, dtype=float)
    safe = so > 1e-12
    w1 = np.where(safe, np.sin((1 - f) * omega) / np.where(safe, so, 1), 1 - f)
    w2 = np.where(safe, np.sin(f * omega) / np.where(safe, so, 1), f)
    p = w1[..., None] * a + w2[..., None] * b
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(p[..., 2])), np.degrees(np.arctan2(p[..., 1], p[..., 0]))
