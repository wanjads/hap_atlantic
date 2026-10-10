"""Free-space optical link budget -> achievable data rate.

Model (per link, evaluated vectorised over many links):

    P_rx = P_tx * eta * G_geo(d) * 10^(-(L_point + L_atm + M_turb)/10)
    R    = min(R_max, P_rx / (n_ppb * h * nu))

* G_geo: fraction of a Gaussian beam (full-angle divergence theta) collected by a circular
  aperture of diameter D_rx at distance d.
* L_atm: clear-air extinction, integrated along the ray with an exponential aerosol profile.
* M_turb: fade margin for a target outage probability, from the scintillation index of a
  spherical wave (Andrews & Phillips, strong-fluctuation expression) with the
  Hufnagel-Valley 5/7 C_n^2 profile integrated along the ray. Aperture averaging is
  neglected, which overestimates the margin, while the log-normal model underestimates
  deep fades in strong turbulence.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .config import C0, H_PLANCK, AtmosphereParams, TerminalParams


def cn2_hufnagel_valley(h, atm: AtmosphereParams) -> np.ndarray:
    """Refractive-index structure parameter C_n^2(h) [m^-2/3], h in metres."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    return (
        0.00594 * (atm.hv_wind / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + atm.hv_a * np.exp(-h / 100.0)
    )


def geometric_gain(d, term: TerminalParams) -> np.ndarray:
    """Fraction of transmitted power collected by the receive aperture."""
    w = np.maximum(term.divergence / 2 * np.asarray(d, dtype=float), 1e-9)  # 1/e^2 radius
    return 1.0 - np.exp(-2.0 * (term.d_rx / 2) ** 2 / w**2)


def extinction_db(heights: np.ndarray, d, atm: AtmosphereParams) -> np.ndarray:
    """Clear-air loss [dB] along rays sampled at `heights` (..., n) with length d (...)."""
    alpha = atm.alpha0_db_per_km * np.exp(-np.maximum(heights, 0.0) / atm.aerosol_scale_height)
    return alpha.mean(axis=-1) * np.asarray(d) / 1e3


def rytov_spherical(heights: np.ndarray, d, term: TerminalParams, atm: AtmosphereParams):
    """Spherical-wave Rytov variance beta_0^2 along rays sampled at `heights` (..., n)."""
    n = heights.shape[-1]
    s = (np.arange(n) + 0.5) / n
    k = 2 * np.pi / term.wavelength
    w = (s * (1 - s)) ** (5 / 6)
    integral = (cn2_hufnagel_valley(heights, atm) * w).mean(axis=-1)  # int over s in [0, 1]
    d = np.asarray(d, dtype=float)
    return 2.25 * k ** (7 / 6) * d ** (11 / 6) * integral


def scintillation_index(beta2) -> np.ndarray:
    """Point-receiver scintillation index of a spherical wave, weak to strong turbulence."""
    b = np.asarray(beta2, dtype=float)
    b125 = b ** (6 / 5)  # beta^(12/5) = (beta^2)^(6/5)
    return (
        np.exp(
            0.49 * b / (1 + 0.56 * b125) ** (7 / 6) + 0.51 * b / (1 + 0.69 * b125) ** (5 / 6)
        )
        - 1.0
    )


def fade_margin_db(sigma_i2, outage_prob: float) -> np.ndarray:
    """Log-normal fade margin [dB] such that P(I < I_th) = outage_prob."""
    s2 = np.log1p(np.asarray(sigma_i2, dtype=float))
    s = np.sqrt(s2)
    ratio = np.exp(s * ndtri(outage_prob) - s2 / 2)
    return -10 * np.log10(ratio)


def link_rate(
    d,
    heights: np.ndarray,
    term: TerminalParams,
    atm: AtmosphereParams,
    extra_loss_db=0.0,
    capped: bool = True,
) -> np.ndarray:
    """Achievable data rate [bit/s] for links of length d with ray heights (..., n).

    The photon-counting budget has no bandwidth limit; with capped=True the rate is
    limited to the transceiver line rate r_max, which is binding for almost all links.
    """
    d = np.asarray(d, dtype=float)
    beta2 = rytov_spherical(heights, d, term, atm)
    loss_db = (
        term.pointing_loss_db
        + extinction_db(heights, d, atm)
        + fade_margin_db(scintillation_index(beta2), term.outage_prob)
        + extra_loss_db
    )
    p_rx = term.p_tx * term.eta_optics * geometric_gain(d, term) * 10 ** (-loss_db / 10)
    photon_energy = H_PLANCK * C0 / term.wavelength
    rate = p_rx / (term.photons_per_bit * photon_energy)
    return np.minimum(rate, term.r_max) if capped else rate
