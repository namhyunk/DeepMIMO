"""Sionna Ray Tracing Materials Module.

This module handles loading and converting material data from Sionna's format to DeepMIMO's format.
"""

from pathlib import Path
from typing import Any

import numpy as np

from deepmimo.core.materials import Material, MaterialList
from deepmimo.utils import load_pickle


def _sf(val: Any, default: float = 0.0) -> float:
    """Safely convert a scalar or 0-d/1-element numpy array to float."""
    if val is None:
        return default
    return float(np.asarray(val).flat[0])


def read_materials(load_folder: str) -> tuple[dict, dict[str, int]]:
    """Read materials from a Sionna RT simulation folder.

    Args:
        load_folder: Path to simulation folder containing material files

    Returns:
        Tuple of (Dict containing materials and their categorization,
                 Dict mapping object names to material indices)

    """
    # Load Sionna materials
    material_properties = load_pickle(str(Path(load_folder) / "sionna_materials.pkl"))
    material_indices = load_pickle(str(Path(load_folder) / "sionna_material_indices.pkl"))

    # Initialize material list
    material_list = MaterialList()

    # Attribute matching for scattering models
    scat_model = {
        "LambertianPattern": Material.SCATTERING_LAMBERTIAN,
        "DirectivePattern": Material.SCATTERING_DIRECTIVE,
        "BackscatteringPattern": Material.SCATTERING_DIRECTIVE,  # directive = backscattering
    }

    # Convert each Sionna material to DeepMIMO Material
    materials = []
    for i, mat_property in enumerate(material_properties):
        # Get scattering model type and handle case where scattering is disabled
        scattering_model = scat_model[mat_property["scattering_pattern"]]
        scat_coeff = mat_property["scattering_coefficient"]
        scattering_model = Material.SCATTERING_NONE if not scat_coeff else scattering_model

        # Create Material object
        material = Material(
            id=i,
            name=f"material_{i}",  # Default name if not provided
            permittivity=_sf(mat_property["relative_permittivity"]),
            conductivity=_sf(mat_property["conductivity"]),
            scattering_model=scattering_model,
            scattering_coefficient=_sf(scat_coeff),
            cross_polarization_coefficient=_sf(mat_property["xpd_coefficient"]),
            alpha_r=_sf(mat_property["alpha_r"]),
            alpha_i=_sf(mat_property["alpha_i"]),
            lambda_param=_sf(mat_property["lambda_"]),
        )
        materials.append(material)

    # Add all materials to buildings category by default
    # This can be modified if Sionna provides material categorization
    material_list.add_materials(materials)

    return material_list.to_dict(), material_indices
