"""Fidelity Configuration Module.

Defines the FidelityConfig dataclass and preset fidelity levels for
controlled degradation experiments across 4 components:
  - Geometry (position/height noise, building removal)
  - Material (uniform material, disable scattering)
  - Ray Tracing (reflection depth, ray count, diffraction)
  - Hardware (antenna pattern, array size)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FidelityConfig:
    """Configuration for a single fidelity experiment variant.

    Each config represents one controlled perturbation of the digital twin.
    The baseline config uses default values (no degradation).
    """

    name: str  # Unique identifier, e.g. "baseline", "geo_noise_5m"

    # --- Geometry degradation ---
    position_noise_std: float = 0.0  # Gaussian noise std on building vertex XY positions [m]
    height_noise_std: float = 0.0  # Gaussian noise std on building heights (Z) [m]
    remove_buildings_below: float = 0.0  # Remove buildings shorter than this [m]
    remove_buildings_fraction: float = 0.0  # Fraction of buildings to randomly remove [0, 1]

    # --- Material simplification ---
    uniform_material: str | None = None  # Override ALL building materials with this
    # e.g. "itu_concrete" -> everything becomes concrete
    disable_scattering: bool = False  # Set scattering_coefficient=0 for all materials

    # --- Ray Tracing parameters ---
    max_reflections: int = 5
    max_diffractions: int = 0
    ds_enable: bool = False  # Diffuse scattering
    n_samples_per_src: int = 1_000_000
    los: bool = True

    # --- Hardware (antenna) ---
    antenna_pattern: str = "iso"  # "iso", "dipole", "hw_dipole", "tr38901"
    num_tx_ant_rows: int = 1
    num_tx_ant_cols: int = 1
    num_tx_ant_cols: int = 1
    num_rx_ant_rows: int = 1
    num_rx_ant_cols: int = 1
    antenna_spacing: float = 0.5  # Wavelength spacing between elements
    polarization: str = "V"  # "V", "H", "VH", "slant"

    # --- Metadata ---
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_rt_params(self) -> dict[str, Any]:
        """Convert to ray-tracing parameter overrides for raytrace_sionna().

        Returns a dict that can be merged into the base pipeline_params `p`.
        """
        return {
            "max_reflections": self.max_reflections,
            "max_diffractions": self.max_diffractions,
            "ds_enable": self.ds_enable,
            "n_samples_per_src": self.n_samples_per_src,
            "los": self.los,
        }

    @property
    def has_geometry_degradation(self) -> bool:
        """Whether this config modifies scene geometry."""
        return (
            self.position_noise_std > 0
            or self.height_noise_std > 0
            or self.remove_buildings_below > 0
            or self.remove_buildings_fraction > 0
        )

    @property
    def has_material_degradation(self) -> bool:
        """Whether this config simplifies materials."""
        return self.uniform_material is not None or self.disable_scattering

    @property
    def has_hardware_change(self) -> bool:
        """Whether this config changes antenna/hardware settings."""
        return (
            self.antenna_pattern != "iso"
            or self.num_tx_ant_rows != 1
            or self.num_tx_ant_cols != 1
            or self.num_rx_ant_rows != 1
            or self.num_rx_ant_cols != 1
        )


# ============================================================================
# Preset fidelity configurations
# ============================================================================

BASELINE = FidelityConfig(
    name="baseline",
    description="Full fidelity: 5 reflections, 1M rays, real materials, iso antenna",
    tags=["baseline"],
)

# --- Geometry presets ---

GEO_NOISE_0_1M = FidelityConfig(
    name="geo_noise_0_1m", position_noise_std=0.1, description="Geometry: add 0.1m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_0_5M = FidelityConfig(
    name="geo_noise_0_5m", position_noise_std=0.5, description="Geometry: add 0.5m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_1M = FidelityConfig(
    name="geo_noise_1m",
    position_noise_std=1.0,
    description="Geometry: add 1m Gaussian noise to building positions",
    tags=["geometry"],
)

GEO_NOISE_2M = FidelityConfig(
    name="geo_noise_2m", position_noise_std=2.0, description="Geometry: add 2m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_3M = FidelityConfig(
    name="geo_noise_3m", position_noise_std=3.0, description="Geometry: add 3m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_4M = FidelityConfig(
    name="geo_noise_4m", position_noise_std=4.0, description="Geometry: add 4m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_5M = FidelityConfig(
    name="geo_noise_5m",
    position_noise_std=5.0,
    description="Geometry: add 5m Gaussian noise to building positions",
    tags=["geometry"],
)

GEO_NOISE_7M = FidelityConfig(
    name="geo_noise_7m", position_noise_std=7.0, description="Geometry: add 7m Gaussian noise", tags=["geometry"]
)

GEO_NOISE_10M = FidelityConfig(
    name="geo_noise_10m",
    position_noise_std=10.0,
    description="Geometry: add 10m Gaussian noise to building positions",
    tags=["geometry"],
)

GEO_HEIGHT_NOISE_3M = FidelityConfig(
    name="geo_height_noise_3m",
    height_noise_std=3.0,
    description="Geometry: add 3m noise to building heights",
    tags=["geometry"],
)

GEO_REMOVE_SMALL = FidelityConfig(
    name="geo_remove_small",
    remove_buildings_below=5.0,
    description="Geometry: remove buildings shorter than 5m",
    tags=["geometry"],
)

GEO_REMOVE_30PCT = FidelityConfig(
    name="geo_remove_30pct",
    remove_buildings_fraction=0.3,
    description="Geometry: randomly remove 30% of buildings",
    tags=["geometry"],
)

# --- Material presets ---
# NOTE: All material configs use ds_enable=True so that diffuse scattering
# is active and material scattering coefficients actually matter.

BASELINE_DS = FidelityConfig(
    name="baseline_ds",
    ds_enable=True,
    description="Baseline with diffuse scattering enabled (reference for material experiments)",
    tags=["material", "baseline"],
)

MAT_ALL_CONCRETE = FidelityConfig(
    name="mat_all_concrete",
    uniform_material="itu_concrete",
    ds_enable=True,
    description="Material: all buildings set to ITU concrete (with scattering)",
    tags=["material"],
)

MAT_ALL_GLASS = FidelityConfig(
    name="mat_all_glass",
    uniform_material="itu_glass",
    ds_enable=True,
    description="Material: all buildings set to ITU glass (with scattering)",
    tags=["material"],
)

MAT_ALL_METAL = FidelityConfig(
    name="mat_all_metal",
    uniform_material="itu_metal",
    ds_enable=True,
    description="Material: all buildings set to ITU metal (with scattering)",
    tags=["material"],
)

MAT_ALL_WOOD = FidelityConfig(
    name="mat_all_wood",
    uniform_material="itu_wood",
    ds_enable=True,
    description="Material: all buildings set to ITU wood (with scattering)",
    tags=["material"],
)

MAT_ALL_MARBLE = FidelityConfig(
    name="mat_all_marble",
    uniform_material="itu_marble",
    ds_enable=True,
    description="Material: all buildings set to ITU marble (with scattering)",
    tags=["material"],
)

MAT_ALL_BRICK = FidelityConfig(
    name="mat_all_brick",
    uniform_material="itu_brick",
    ds_enable=True,
    description="Material: all buildings set to ITU brick (with scattering)",
    tags=["material"],
)

MAT_NO_SCATTERING = FidelityConfig(
    name="mat_no_scattering",
    disable_scattering=True,
    ds_enable=True,
    description="Material: scattering enabled but all scattering coefficients set to 0",
    tags=["material"],
)

# --- Ray Tracing presets ---

RT_DEPTH_0 = FidelityConfig(
    name="rt_depth_0",
    max_reflections=0,
    description="Ray Tracing: max 0 reflections (LOS only)",
    tags=["ray_tracing"],
)

RT_DEPTH_1 = FidelityConfig(
    name="rt_depth_1",
    max_reflections=1,
    description="Ray Tracing: max 1 reflection",
    tags=["ray_tracing"],
)

RT_DEPTH_2 = FidelityConfig(
    name="rt_depth_2",
    max_reflections=2,
    description="Ray Tracing: max 2 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_3 = FidelityConfig(
    name="rt_depth_3",
    max_reflections=3,
    description="Ray Tracing: max 3 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_4 = FidelityConfig(
    name="rt_depth_4",
    max_reflections=4,
    description="Ray Tracing: max 4 reflections",
    tags=["ray_tracing"],
)
RT_DEPTH_5 = FidelityConfig(
    name="rt_depth_5",
    max_reflections=5,
    description="Ray Tracing: max 5 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_6 = FidelityConfig(
    name="rt_depth_6",
    max_reflections=6,
    description="Ray Tracing: max 6 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_7 = FidelityConfig(
    name="rt_depth_7",
    max_reflections=7,
    description="Ray Tracing: max 7 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_8 = FidelityConfig(
    name="rt_depth_8",
    max_reflections=8,
    description="Ray Tracing: max 8 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_9 = FidelityConfig(
    name="rt_depth_9",
    max_reflections=9,
    description="Ray Tracing: max 9 reflections",
    tags=["ray_tracing"],
)

RT_DEPTH_10 = FidelityConfig(
    name="rt_depth_10",
    max_reflections=10,
    description="Ray Tracing: max 10 reflections (New Baseline)",
    tags=["ray_tracing"],
)

RT_500K_RAYS = FidelityConfig(
    name="rt_500k_rays", n_samples_per_src=500_000, description="Ray Tracing: 500K rays", tags=["ray_tracing"]
)

RT_200K_RAYS = FidelityConfig(
    name="rt_200k_rays", n_samples_per_src=200_000, description="Ray Tracing: 200K rays", tags=["ray_tracing"]
)

RT_LOW_RAYS = FidelityConfig(
    name="rt_low_rays",
    n_samples_per_src=100_000,
    description="Ray Tracing: 100K rays (10x fewer than baseline)",
    tags=["ray_tracing"],
)

RT_50K_RAYS = FidelityConfig(
    name="rt_50k_rays", n_samples_per_src=50_000, description="Ray Tracing: 50K rays", tags=["ray_tracing"]
)

RT_20K_RAYS = FidelityConfig(
    name="rt_20k_rays", n_samples_per_src=20_000, description="Ray Tracing: 20K rays", tags=["ray_tracing"]
)

RT_VERY_LOW_RAYS = FidelityConfig(
    name="rt_very_low_rays",
    n_samples_per_src=10_000,
    description="Ray Tracing: 10K rays (100x fewer than baseline)",
    tags=["ray_tracing"],
)

RT_5K_RAYS = FidelityConfig(
    name="rt_5k_rays", n_samples_per_src=5_000, description="Ray Tracing: 5K rays", tags=["ray_tracing"]
)

RT_1K_RAYS = FidelityConfig(
    name="rt_1k_rays", n_samples_per_src=1_000, description="Ray Tracing: 1K rays", tags=["ray_tracing"]
)

RT_WITH_DIFFRACTION = FidelityConfig(
    name="rt_with_diffraction",
    max_diffractions=1,
    description="Ray Tracing: enable 1 diffraction event",
    tags=["ray_tracing"],
)

# --- Hardware presets ---
# To properly measure DT hardware fidelity impact on MIMO applications, we fix the
# array size to a 4x4 UPA. The baseline uses high-fidelity 3GPP patterns, and we
# degrade fidelity by simplifying the element pattern to dipole or isotropic.

HW_BASELINE_4X4 = FidelityConfig(
    name="hw_baseline_4x4",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    antenna_pattern="tr38901",
    description="Hardware: 4x4 UPA with 3GPP TR38901 pattern (Hardware Baseline)",
    tags=["hardware", "baseline"],
)

HW_4X4_DIPOLE = FidelityConfig(
    name="hw_4x4_dipole",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    antenna_pattern="dipole",
    description="Hardware: 4x4 UPA simplified to Dipole elements",
    tags=["hardware"],
)

HW_4X4_ISO = FidelityConfig(
    name="hw_4x4_iso",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    antenna_pattern="iso",
    description="Hardware: 4x4 UPA simplified to Isotropic elements",
    tags=["hardware"],
)

HW_4X4_SPACING_04 = FidelityConfig(
    name="hw_4x4_spacing04",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    antenna_pattern="tr38901",
    antenna_spacing=0.4,
    description="Hardware: 4x4 UPA with spacing error (0.4 vs 0.5)",
    tags=["hardware"],
)

HW_4X4_POL_H = FidelityConfig(
    name="hw_4x4_polh",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    antenna_pattern="tr38901",
    polarization="H",
    description="Hardware: 4x4 UPA with polarization error (H vs V)",
    tags=["hardware"],
)

# All presets in execution order
ALL_CONFIGS: list[FidelityConfig] = [
    BASELINE,
    # Geometry
    GEO_NOISE_0_1M,
    GEO_NOISE_0_5M,
    GEO_NOISE_1M,
    GEO_NOISE_2M,
    GEO_NOISE_3M,
    GEO_NOISE_4M,
    GEO_NOISE_5M,
    GEO_NOISE_7M,
    GEO_NOISE_10M,
    GEO_HEIGHT_NOISE_3M,
    GEO_REMOVE_SMALL,
    GEO_REMOVE_30PCT,
    # Material
    BASELINE_DS,
    MAT_ALL_CONCRETE,
    MAT_ALL_GLASS,
    MAT_ALL_METAL,
    MAT_ALL_WOOD,
    MAT_ALL_MARBLE,
    MAT_ALL_BRICK,
    MAT_NO_SCATTERING,
    # Ray Tracing
    RT_DEPTH_0,
    RT_DEPTH_1,
    RT_DEPTH_2,
    RT_DEPTH_3,
    RT_DEPTH_4,
    RT_DEPTH_5,
    RT_DEPTH_6,
    RT_DEPTH_7,
    RT_DEPTH_8,
    RT_DEPTH_9,
    RT_DEPTH_10,
    RT_500K_RAYS,
    RT_200K_RAYS,
    RT_LOW_RAYS,
    RT_50K_RAYS,
    RT_20K_RAYS,
    RT_VERY_LOW_RAYS,
    RT_5K_RAYS,
    RT_1K_RAYS,
    RT_WITH_DIFFRACTION,
    # Hardware
    HW_BASELINE_4X4,
    HW_4X4_DIPOLE,
    HW_4X4_ISO,
    HW_4X4_SPACING_04,
    HW_4X4_POL_H,
]

CONFIGS_BY_NAME: dict[str, FidelityConfig] = {c.name: c for c in ALL_CONFIGS}


def get_configs_by_tag(tag: str) -> list[FidelityConfig]:
    """Get all configs matching a tag (e.g. 'geometry', 'material')."""
    return [c for c in ALL_CONFIGS if tag in c.tags]
