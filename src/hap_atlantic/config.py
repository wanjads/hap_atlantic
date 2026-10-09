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
# Area from which positions are read (lat_min, lat_max, lon_min, lon_max): covers
# Frankfurt, New York and the oceanic tracks.
NORTH_ATLANTIC_BBOX = (35.0, 66.0, -80.0, 15.0)

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
    hop_delay: float = 10e-6  # per-hop processing delay (coherent DSP + FEC + switch) [s]
    r_min: float = 1e9  # links with a lower link-budget rate are not used [bit/s]


@dataclass(frozen=True)
class GeometryParams:
    earth_radius: float = 6_371_008.8  # mean Earth radius [m]
    k_refraction: float = 1.0  # effective Earth radius factor (1 = straight rays)
    h_chord_min: float = 3_000.0  # min. height of an air-to-air ray above ground [m] (terrain,
    # boundary layer; clouds are modelled explicitly along the ray)
    d_max_air: float = 700e3  # max. air-to-air link range (terminal acquisition limit) [m]
    h_node_min: float = 1_000.0  # aircraft below this altitude are ignored [m]
    elev_min_ground: float = 5.0  # min. elevation of ground-to-air links [deg] (obstacles,
    # ground turbulence; the link budget alone would allow ~2 deg)
    corridor_half_width: float = 1000e3  # aircraft farther from the FRA-NY arc are ignored [m]
    yao_cones: int = 8  # keep nearest + farthest neighbour per cone (0 = keep all links)
    ray_samples: int = 16  # points per ray for extinction / turbulence integrals


@dataclass(frozen=True)
class GroundParams:
    """Optical ground stations around each endpoint (site diversity)."""

    # latency service: stations at the endpoint and on rings around it (land points only)
    site_rings: tuple = (75e3, 150e3)  # ring radii [m]
    site_bearings: int = 8  # stations per ring
    terminals_per_site: int = 4
    rf_rate: float = 10e9  # RF fallback rate per ground link when cloudy, 0 = optical only [bit/s]
    rf_rain_max: float = 5.0  # RF ground links fail above this rain rate [mm/h]
    fiber_detour: float = 1.3  # terrestrial fibre route length / great-circle distance [-]
    # capacity: optical ground stations everywhere on land (ERA5 land-sea mask, 0.25 deg);
    # Europe incl. British Isles east of land_eu_lon, North America west of land_na_lon;
    # Iceland, Greenland and the Azores are excluded
    land_eu_lon: float = -12.0
    land_na_lon: float = -52.0
    land_candidates: int = 6  # nearest land points tried per aircraft


@dataclass(frozen=True)
class CloudParams:
    """Cloud attenuation of optical links from 3D ERA5 cloud fields (see weather.py)."""

    model: str = "era5"  # "era5" or "clear" (no clouds, only for tests)
    l_corr: float = 50e3  # horizontal correlation length of cloud occurrence [m]
    lz_corr: float = 2e3  # vertical correlation length (cloud overlap) [m]
    t_corr: float = 3600.0  # correlation time [s] (~ l_corr / wind speed)
    de_ice: float = 50e-6  # effective diameter of cirrus ice crystals [m] (Fu 1996)
    re_liquid: float = 10e-6  # effective radius of cloud droplets [m]
    n_features: int = 256  # random Fourier features of the latent field
    ray_samples: int = 48  # points per ray for the cloud integral


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
    clouds: CloudParams = field(default_factory=CloudParams)
    atmosphere: AtmosphereParams = field(default_factory=AtmosphereParams)
    terminals_per_aircraft: int = 2
    airliners_only: bool = True  # all jet airliners (ICAO type list in traffic.py) carry terminals
    source: Site = FRANKFURT
    sink: Site = NEW_YORK
    dt: float = 60.0  # simulation time step [s]
    seed: int = 0


@dataclass(frozen=True)
class Tech:
    """Terminal technology tier: line rate per link and the transmit power it needs."""

    r_max: float  # [bit/s]
    p_tx: float  # [W]
    note: str


# Airborne optical links have been demonstrated at 1-2.6 Gbit/s (Moll2015, FALCON2010,
# GAASI2022, ESA2026); everything above is an extrapolation from space / ground systems.
TECHS = {
    "demo": Tech(2.5e9, 1.0, "demonstrated airborne rate"),
    "leo": Tech(100e9, 1.0, "LEO laser crosslink class (Starlink2024)"),
    "dwdm": Tech(1e12, 5.0, "DWDM terminal, 10 x 100 Gbit/s (extrapolation)"),
    "visionary": Tech(10e12, 10.0, "ground-demo DWDM class (DLR2018), static link only"),
}
K_VALUES = (2, 4, 6)  # terminals per aircraft
SCENARIOS = tuple(TECHS)


def scenario(name: str, k: int = 2) -> Params:
    """Params for a technology tier with k terminals per aircraft."""
    if name not in TECHS:
        raise ValueError(f"unknown scenario {name!r}")
    base = Params()
    t = TECHS[name]
    return replace(base, terminals_per_aircraft=k,
                   terminal=replace(base.terminal, r_max=t.r_max, p_tx=t.p_tx))
