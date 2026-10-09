"""Central simulation parameters.

All values that end up in the paper (Table "Simulation Parameters") live here, so that
the LaTeX table and the code cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Physical constants
C0 = 299_792_458.0  # speed of light in vacuum [m/s]
H_PLANCK = 6.626_070_15e-34  # [J s]
N_FIBER = 1.4682  # group index of Corning SMF-28 Ultra at 1550 nm
N_AIR = 1.0003  # refractive index of air (upper bound, sea level)


@dataclass(frozen=True)
class Site:
    name: str
    lat: float  # [deg]
    lon: float  # [deg]
    alt: float = 100.0  # [m]


# Endpoints. Frankfurt: Eurex / DE-CIX area; New York: Equinix NY4 (Secaucus, NJ),
# the usual reference point for transatlantic low-latency trading routes.
FRANKFURT = Site("Frankfurt", 50.1109, 8.6821, 110.0)
NEW_YORK = Site("New York (NY4 Secaucus)", 40.7769, -74.0699, 10.0)


@dataclass(frozen=True)
class TerminalParams:
    """Optical terminal on an aircraft (one per gimbal)."""

    wavelength: float = 1550e-9  # [m]
    p_tx: float = 1.0  # optical transmit power [W]
    divergence: float = 20e-6  # full-angle 1/e^2 beam divergence [rad]
    d_rx: float = 0.10  # receive aperture diameter [m]
    eta_optics: float = 0.5  # combined Tx/Rx optics efficiency [-]
    pointing_loss_db: float = 3.0  # mean pointing / tracking loss [dB]
    photons_per_bit: float = 20.0  # receiver sensitivity (coherent, incl. FEC) [photons/bit]
    r_max: float = 100e9  # transceiver line-rate limit per link [bit/s]
    outage_prob: float = 1e-3  # target outage probability for the turbulence fade margin
    hop_delay: float = 1e-6  # per-hop processing / OEO delay [s]


@dataclass(frozen=True)
class GeometryParams:
    earth_radius: float = 6_371_008.8  # mean Earth radius [m]
    k_refraction: float = 1.0  # effective Earth radius factor (1 = straight rays)
    h_chord_min: float = 7_000.0  # min. height of an air-to-air ray above ground [m]
    d_max_air: float = 700e3  # max. air-to-air link range (terminal acquisition limit) [m]
    h_node_min: float = 1_000.0  # aircraft below this altitude are ignored [m]
    elev_min_ground: float = 10.0  # min. elevation of ground-to-air links [deg]
    corridor_half_width: float = 1000e3  # aircraft farther from the FRA-NY arc are ignored [m]
    yao_cones: int = 8  # keep nearest + farthest neighbour per cone (0 = keep all links)
    ray_samples: int = 16  # points per ray for extinction / turbulence integrals


@dataclass(frozen=True)
class GroundParams:
    """Optical ground stations around each endpoint (site diversity)."""

    n_sites: int = 1  # number of ground stations per endpoint
    ring_radius: float = 50e3  # sites are placed on a ring of this radius [m]
    terminals_per_site: int = 4
    weather: str = "era5"  # "era5" (hourly reanalysis at the sites) or "markov" (synthetic)
    f_cirrus: float = 1.0  # fraction of high (cirrus) cloud cover that blocks the beam [-]
    p_cloud_free: float = 0.35  # markov only: stationary P(cloud-free line of sight) [-]
    cloud_corr_time: float = 3 * 3600.0  # markov only: mean duration of a cloud state [s]
    rf_rate: float = 0.0  # RF fallback rate per ground link when cloudy, 0 = optical only [bit/s]
    rf_rain_max: float = 5.0  # RF ground links fail above this rain rate [mm/h]
    fiber_detour: float = 1.3  # terrestrial fibre route length / great-circle distance [-]


@dataclass(frozen=True)
class AtmosphereParams:
    alpha0_db_per_km: float = 0.2  # clear-air extinction at sea level, 1550 nm [dB/km]
    aerosol_scale_height: float = 1_200.0  # [m]
    hv_wind: float = 21.0  # Hufnagel-Valley rms high-altitude wind [m/s]
    hv_a: float = 1.7e-14  # Hufnagel-Valley ground-level Cn^2 [m^-2/3]


@dataclass(frozen=True)
class Params:
    terminal: TerminalParams = field(default_factory=TerminalParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    ground: GroundParams = field(default_factory=GroundParams)
    atmosphere: AtmosphereParams = field(default_factory=AtmosphereParams)
    terminals_per_aircraft: int = 2
    source: Site = FRANKFURT
    sink: Site = NEW_YORK
    dt: float = 60.0  # simulation time step [s]
    seed: int = 0


def scenario(name: str) -> Params:
    """Named technology scenarios used in the evaluation."""
    base = Params()
    if name == "conservative":  # today's LEO laser terminals (~100 Gbit/s)
        return base
    if name == "advanced":  # DWDM terminal, 1 Tbit/s, 4 gimbals
        return replace(
            base,
            terminals_per_aircraft=4,
            terminal=replace(base.terminal, r_max=1e12, p_tx=5.0),
        )
    if name == "visionary":  # ground-demo class DWDM (>10 Tbit/s), 6 gimbals
        return replace(
            base,
            terminals_per_aircraft=6,
            terminal=replace(base.terminal, r_max=10e12, p_tx=10.0),
            ground=replace(base.ground, n_sites=5),
        )
    raise ValueError(f"unknown scenario {name!r}")


SCENARIOS = ("conservative", "advanced", "visionary")
