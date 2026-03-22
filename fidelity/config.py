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
    num_rx_ant_rows: int = 1
    num_rx_ant_cols: int = 1

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

GEO_NOISE_1M = FidelityConfig(
    name="geo_noise_1m",
    position_noise_std=1.0,
    description="Geometry: add 1m Gaussian noise to building positions",
    tags=["geometry"],
)

GEO_NOISE_5M = FidelityConfig(
    name="geo_noise_5m",
    position_noise_std=5.0,
    description="Geometry: add 5m Gaussian noise to building positions",
    tags=["geometry"],
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

MAT_ALL_CONCRETE = FidelityConfig(
    name="mat_all_concrete",
    uniform_material="itu_concrete",
    description="Material: all buildings set to ITU concrete",
    tags=["material"],
)

MAT_NO_SCATTERING = FidelityConfig(
    name="mat_no_scattering",
    disable_scattering=True,
    description="Material: disable scattering for all materials",
    tags=["material"],
)

# --- Ray Tracing presets ---

RT_DEPTH_1 = FidelityConfig(
    name="rt_depth_1",
    max_reflections=1,
    description="Ray Tracing: max 1 reflection",
    tags=["ray_tracing"],
)

RT_DEPTH_3 = FidelityConfig(
    name="rt_depth_3",
    max_reflections=3,
    description="Ray Tracing: max 3 reflections",
    tags=["ray_tracing"],
)

RT_LOW_RAYS = FidelityConfig(
    name="rt_low_rays",
    n_samples_per_src=100_000,
    description="Ray Tracing: 100K rays (10x fewer than baseline)",
    tags=["ray_tracing"],
)

RT_VERY_LOW_RAYS = FidelityConfig(
    name="rt_very_low_rays",
    n_samples_per_src=10_000,
    description="Ray Tracing: 10K rays (100x fewer than baseline)",
    tags=["ray_tracing"],
)

RT_WITH_DIFFRACTION = FidelityConfig(
    name="rt_with_diffraction",
    max_diffractions=1,
    description="Ray Tracing: enable 1 diffraction event",
    tags=["ray_tracing"],
)

# --- Hardware presets ---

HW_DIPOLE = FidelityConfig(
    name="hw_dipole",
    antenna_pattern="dipole",
    description="Hardware: half-wave dipole antenna pattern",
    tags=["hardware"],
)

HW_TR38901 = FidelityConfig(
    name="hw_tr38901",
    antenna_pattern="tr38901",
    description="Hardware: 3GPP TR 38.901 antenna pattern",
    tags=["hardware"],
)

HW_4X4_ARRAY = FidelityConfig(
    name="hw_4x4_array",
    num_tx_ant_rows=4,
    num_tx_ant_cols=4,
    description="Hardware: 4x4 UPA at TX",
    tags=["hardware"],
)

# All presets in execution order
ALL_CONFIGS: list[FidelityConfig] = [
    BASELINE,
    # Geometry
    GEO_NOISE_1M,
    GEO_NOISE_5M,
    GEO_NOISE_10M,
    GEO_HEIGHT_NOISE_3M,
    GEO_REMOVE_SMALL,
    GEO_REMOVE_30PCT,
    # Material
    MAT_ALL_CONCRETE,
    MAT_NO_SCATTERING,
    # Ray Tracing
    RT_DEPTH_1,
    RT_DEPTH_3,
    RT_LOW_RAYS,
    RT_VERY_LOW_RAYS,
    RT_WITH_DIFFRACTION,
    # Hardware
    HW_DIPOLE,
    HW_TR38901,
    HW_4X4_ARRAY,
]

CONFIGS_BY_NAME: dict[str, FidelityConfig] = {c.name: c for c in ALL_CONFIGS}


def get_configs_by_tag(tag: str) -> list[FidelityConfig]:
    """Get all configs matching a tag (e.g. 'geometry', 'material')."""
    return [c for c in ALL_CONFIGS if tag in c.tags]
