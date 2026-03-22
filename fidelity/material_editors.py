"""Material Editors for Material Fidelity Degradation.

Functions that modify RadioMaterial properties on Sionna scene objects
to simulate material fidelity degradation.

Usage:
    from fidelity.material_editors import build_material_editor
    from fidelity.config import MAT_ALL_CONCRETE

    edit_func = build_material_editor(MAT_ALL_CONCRETE)
    # pass to raytrace_sionna via params["scene_edit_func"] = edit_func
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sionna.rt import Scene

    from fidelity.config import FidelityConfig


def set_uniform_material(scene: Scene, material_name: str) -> None:
    """Set all building objects to use a single uniform material.

    This simulates the scenario where material information is unavailable
    or inaccurate, and a single default material (e.g., concrete) is used
    for all surfaces.

    Args:
        scene: Sionna Scene object (modified in-place).
        material_name: Name of the Sionna RadioMaterial to use, e.g. "itu_concrete".

    """
    terrain_keywords = ["plane", "floor", "terrain", "roads", "paths", "road", "path"]

    # Get reference to the target material from the scene
    if material_name not in scene.radio_materials:
        print(
            f"[material_editor] Warning: material '{material_name}' not found. "
            f"Available: {list(scene.radio_materials.keys())}"
        )
        return

    target_material = scene.radio_materials[material_name]
    count = 0

    for obj_name, obj in scene.objects.items():
        if any(kw in obj_name.lower() for kw in terrain_keywords):
            continue

        obj.radio_material = target_material
        count += 1

    print(f"[material_editor] Set {count} objects to material '{material_name}'")


def disable_scattering(scene: Scene) -> None:
    """Disable diffuse scattering for all materials in the scene.

    Sets scattering_coefficient=0 for every RadioMaterial, effectively
    removing diffuse scattering from the simulation.

    Args:
        scene: Sionna Scene object (modified in-place).

    """
    count = 0
    for mat_name, mat in scene.radio_materials.items():
        try:
            mat.scattering_coefficient = 0.0
            count += 1
        except (AttributeError, TypeError):
            print(f"[material_editor] Could not disable scattering for '{mat_name}'")

    print(f"[material_editor] Disabled scattering for {count} materials")


def build_material_editor(config: FidelityConfig) -> callable | None:
    """Build a material editing function from a FidelityConfig.

    Creates a callable that applies all material degradation operations
    specified in the config:
      1. Uniform material override
      2. Disable scattering

    Args:
        config: FidelityConfig with material degradation parameters.

    Returns:
        A callable(scene) or None if no material degradation is needed.

    """
    if not config.has_material_degradation:
        return None

    def editor(scene: Scene) -> None:
        if config.uniform_material is not None:
            print(
                f"[material_editor] Setting all materials to '{config.uniform_material}'"
            )
            set_uniform_material(scene, config.uniform_material)

        if config.disable_scattering:
            print("[material_editor] Disabling scattering for all materials")
            disable_scattering(scene)

    return editor
