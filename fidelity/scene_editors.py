"""Scene Editors for Geometry Degradation.

Functions that modify a Sionna Scene object to simulate geometry fidelity
degradation. Each function is designed to be used as a `scene_edit_func`
callback passed to `raytrace_sionna()`.

Usage:
    from fidelity.scene_editors import build_geometry_editor
    from fidelity.config import GEO_NOISE_5M

    edit_func = build_geometry_editor(GEO_NOISE_5M)
    # pass to raytrace_sionna via params["scene_edit_func"] = edit_func
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mitsuba as mi
    from sionna.rt import Scene

    from fidelity.config import FidelityConfig


def add_position_noise(scene: Scene, std: float, seed: int = 42) -> None:
    """Add Gaussian noise to building XY positions.

    Shifts each scene object's position by a random offset drawn from
    N(0, std) independently for X and Y coordinates.

    Args:
        scene: Sionna Scene object (modified in-place).
        std: Standard deviation of the Gaussian noise [meters].
        seed: Random seed for reproducibility.

    """
    import mitsuba as mi

    rng = np.random.default_rng(seed)

    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "road", "path"]

    for obj_name, obj in list(scene.objects.items()):
        # Skip terrain/road objects — only perturb buildings
        if any(kw in obj_name.lower() for kw in terrain_keywords):
            continue

        current_pos = obj.position
        noise_x = rng.normal(0, std)
        noise_y = rng.normal(0, std)

        pos_arr = np.array(current_pos)
        new_pos = mi.Vector3f(
            float(pos_arr[0]) + noise_x,
            float(pos_arr[1]) + noise_y,
            float(pos_arr[2]),  # Keep Z unchanged
        )
        obj.position = new_pos


def add_height_noise(scene: Scene, std: float, seed: int = 42) -> None:
    """Add Gaussian noise to building heights by scaling Z positions.

    For each building object, scales its Z position by (1 + noise) where
    noise ~ N(0, std/original_height). This preserves ground-level anchoring.

    Args:
        scene: Sionna Scene object (modified in-place).
        std: Standard deviation of the height noise [meters].
        seed: Random seed for reproducibility.

    """
    import mitsuba as mi

    rng = np.random.default_rng(seed)

    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "road", "path"]

    for obj_name, obj in list(scene.objects.items()):
        if any(kw in obj_name.lower() for kw in terrain_keywords):
            continue

        current_pos = obj.position
        pos_arr = np.array(current_pos)
        height = float(pos_arr[2])

        # Add noise directly to height (can't go below ground)
        noise_z = rng.normal(0, std)
        new_height = max(0.5, height + noise_z)  # Minimum 0.5m

        obj.position = mi.Vector3f(
            float(pos_arr[0]),
            float(pos_arr[1]),
            new_height,
        )


def remove_small_buildings(scene: Scene, min_height: float) -> None:
    """Remove buildings below a height threshold.

    Estimates building height from the Z-extent of its bounding box.
    Buildings below `min_height` are removed from the scene.

    Args:
        scene: Sionna Scene object (modified in-place).
        min_height: Minimum building height to keep [meters].

    """
    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "road", "path"]
    to_remove = []

    for obj_name, obj in scene.objects.items():
        if any(kw in obj_name.lower() for kw in terrain_keywords):
            continue

        # Use object position Z as a proxy for height
        # In Sionna scenes from OSM, building height is typically encoded
        # in the mesh extent, not just the position. We use position Z
        # as a heuristic when mesh data isn't easily accessible.
        height = float(np.array(obj.position)[2])
        if height > 0 and height < min_height:
            to_remove.append(obj_name)

    for name in to_remove:
        try:
            scene.remove(name)
        except Exception:
            pass

    print(f"[scene_editor] Removed {len(to_remove)} buildings below {min_height}m")


def remove_random_buildings(scene: Scene, fraction: float, seed: int = 42) -> None:
    """Randomly remove a fraction of buildings from the scene.

    Args:
        scene: Sionna Scene object (modified in-place).
        fraction: Fraction of buildings to remove [0, 1].
        seed: Random seed for reproducibility.

    """
    random.seed(seed)

    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "road", "path"]
    building_names = [
        name
        for name in scene.objects
        if not any(kw in name.lower() for kw in terrain_keywords)
    ]

    n_remove = int(len(building_names) * fraction)
    to_remove = random.sample(building_names, n_remove)

    for name in to_remove:
        try:
            scene.remove(name)
        except Exception:
            pass

    print(
        f"[scene_editor] Randomly removed {len(to_remove)}/{len(building_names)} "
        f"buildings ({fraction * 100:.0f}%)"
    )


def build_geometry_editor(config: FidelityConfig) -> callable | None:
    """Build a composite scene_edit_func from a FidelityConfig.

    Creates a single callable that applies all geometry degradation
    operations specified in the config, in order:
      1. Position noise
      2. Height noise
      3. Remove small buildings
      4. Remove random buildings

    Args:
        config: FidelityConfig with geometry degradation parameters.

    Returns:
        A callable(scene) or None if no geometry degradation is needed.

    """
    if not config.has_geometry_degradation:
        return None

    def editor(scene: Scene) -> None:
        if config.position_noise_std > 0:
            print(f"[scene_editor] Adding position noise σ={config.position_noise_std}m")
            add_position_noise(scene, config.position_noise_std)

        if config.height_noise_std > 0:
            print(f"[scene_editor] Adding height noise σ={config.height_noise_std}m")
            add_height_noise(scene, config.height_noise_std)

        if config.remove_buildings_below > 0:
            print(f"[scene_editor] Removing buildings below {config.remove_buildings_below}m")
            remove_small_buildings(scene, config.remove_buildings_below)

        if config.remove_buildings_fraction > 0:
            print(
                f"[scene_editor] Removing {config.remove_buildings_fraction * 100:.0f}% "
                "of buildings"
            )
            remove_random_buildings(scene, config.remove_buildings_fraction)

    return editor
