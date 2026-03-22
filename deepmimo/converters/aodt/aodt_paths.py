"""AODT Ray Paths Module.

This module handles reading and processing:
1. Ray path data from raypaths.parquet
2. Channel Impulse Response (CIR) from cirs.parquet
"""

from pathlib import Path
from typing import Any

import numpy as np

from deepmimo import consts as c
from deepmimo import utils as gu
from deepmimo.converters import converter_utils as cu

from . import aodt_utils as au
from .safe_import import pd

# AODT interaction type mapping
AODT_TYPE_TO_NUM = {
    "emission": 0,
    "reflection": 1,
    "diffraction": 2,
    "scattering": 3,
    "diffuse": 3,  # alias for scattering
    "reception": 4,
    "transmission": 5,
}

AODT_INTERACTIONS_MAP = {
    0: None,  # emission - not counted as interaction
    1: c.INTERACTION_REFLECTION,  # reflection
    2: c.INTERACTION_DIFFRACTION,  # diffraction
    3: c.INTERACTION_SCATTERING,  # diffuse scattering
    4: None,  # reception - not counted as interaction
    5: c.INTERACTION_TRANSMISSION,  # transmission
}

MIN_INTERACTIONS = 2


def _transform_interaction_types(types: np.ndarray) -> float:
    """Transform AODT interaction types array into a single DeepMIMO interaction code.

    Args:
        types: Array of AODT interaction types where:
              - First element is always 'emission'
              - Last element is always 'reception'
              - Middle elements can be:
                'reflection', 'diffraction', 'scattering'/'diffuse', 'transmission'

    Returns:
        float: Single number representing concatenated interaction types.
               For example: [0, 1, 2, 1, 4] -> 121 (reflection-diffraction-reflection)
               LoS paths return c.INTERACTION_LOS

    Example:
        ['emission', 'reflection', 'reflection', 'reception'] -> 11 (two reflections)
        ['emission', 'diffraction', 'reception'] -> 2 (single diffraction)
        ['emission', 'reception'] -> 0 (LoS)

    """
    # If only emission and reception, it's LoS
    if len(types) <= MIN_INTERACTIONS:
        return c.INTERACTION_LOS

    # Take only middle interactions (exclude first and last)
    interactions = types[1:-1]

    # Convert string types to numeric codes
    if interactions.dtype == "O":  # str
        numeric_types = [AODT_TYPE_TO_NUM[t.lower()] for t in interactions]
    else:
        numeric_types = interactions

    # Map AODT types to DeepMIMO types and concatenate
    mapped = [
        str(AODT_INTERACTIONS_MAP[t]) for t in numeric_types if AODT_INTERACTIONS_MAP[t] is not None
    ]
    if not mapped:  # If all interactions were mapped to None
        return c.INTERACTION_LOS

    return float("".join(mapped))


# Public alias for external callers and tests.
transform_interaction_types = _transform_interaction_types


def _preallocate_data(n_rx: int, n_paths: int = c.MAX_PATHS) -> dict:
    """Pre-allocate data for path conversion.

    Args:
        n_rx: Number of RXs
        n_paths: Number of paths to allocate. Defaults to c.MAX_PATHS.

    Returns:
        data: Dictionary containing pre-allocated data

    """
    return {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.DELAY_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.POWER_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.PHASE_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_rx, n_paths), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros(
            (n_rx, n_paths, c.MAX_INTER_PER_PATH, 3),
            dtype=c.FP_TYPE,
        )
        * np.nan,
    }


def _col_unique_values(df: Any, col: str) -> np.ndarray:
    """Return unique values for a dataframe column or empty array if missing."""
    try:
        series = df[col]
    except Exception:
        return np.array([])
    try:
        values = series.unique()
    except Exception:
        try:
            values = np.unique(series.to_numpy())
        except Exception:
            return np.array([])
    return np.array(values)


def _fill_path_geometry(
    data: dict[str, np.ndarray],
    rx_row_idx: int,
    paths: Any,
    *,
    set_tx_pos: bool,
) -> None:
    """Fill angles/positions/interactions for one RX row from raypaths."""
    tx_pos = None
    rx_pos = None
    for path_idx, path in enumerate(paths.itertuples()):
        # Process interaction points
        interaction_points = au.process_points(path.points)

        # Convert from cm to m
        interaction_points = interaction_points / 100.0

        # First point is TX, last point is RX
        if path_idx == 0:
            tx_pos = interaction_points[0]
            rx_pos = interaction_points[-1]
            if set_tx_pos:
                data[c.TX_POS_PARAM_NAME][0] = tx_pos
            data[c.RX_POS_PARAM_NAME][rx_row_idx] = rx_pos

        # Calculate angles
        departure_vector = interaction_points[1] - tx_pos
        departure_angles = gu.cartesian_to_spherical(departure_vector.reshape(1, -1))[0]
        data[c.AOD_AZ_PARAM_NAME][rx_row_idx, path_idx] = np.rad2deg(departure_angles[1])
        data[c.AOD_EL_PARAM_NAME][rx_row_idx, path_idx] = np.rad2deg(departure_angles[2])

        arrival_vector = rx_pos - interaction_points[-2]
        arrival_angles = gu.cartesian_to_spherical(arrival_vector.reshape(1, -1))[0]
        data[c.AOA_AZ_PARAM_NAME][rx_row_idx, path_idx] = np.rad2deg(arrival_angles[1])
        data[c.AOA_EL_PARAM_NAME][rx_row_idx, path_idx] = np.rad2deg(arrival_angles[2])

        # Store interaction data - skip first (TX) and last (RX) points
        actual_interactions = interaction_points[1:-1]
        if len(actual_interactions) > 0:
            data[c.INTERACTIONS_POS_PARAM_NAME][
                rx_row_idx,
                path_idx,
                : len(actual_interactions),
                :,
            ] = actual_interactions

        # Transform interaction types to DeepMIMO format
        data[c.INTERACTIONS_PARAM_NAME][rx_row_idx, path_idx] = _transform_interaction_types(
            path.interaction_types,
        )


def read_paths(  # noqa: C901, PLR0912, PLR0915
    rt_folder: str, output_folder: str, txrx_dict: dict[str, Any]
) -> None:
    """Read and process ray paths and channel responses.

    Args:
        rt_folder (str): Path to folder containing parquet files.
        output_folder (str): Path to folder where processed paths will be saved.
        txrx_dict (dict[str, Any]): Dictionary containing TX/RX configurations.

    Raises:
        FileNotFoundError: If required files are not found.
        ValueError: If required parameters are missing.

    """
    # Read both parquet files
    paths_file = str(Path(rt_folder) / "raypaths.parquet")
    cirs_file = str(Path(rt_folder) / "cirs.parquet")

    if not Path(paths_file).exists() or not Path(cirs_file).exists():
        msg = "Both raypaths.parquet and cirs.parquet are required"
        raise FileNotFoundError(msg)

    paths_df = pd.read_parquet(paths_file)
    cirs_df = pd.read_parquet(cirs_file)

    if len(paths_df) == 0 or len(cirs_df) == 0:
        msg = "Empty parquet files"
        raise ValueError(msg)

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Prepare for multi-antenna RT outputs (similar constraint philosophy as Sionna RT):
    # - If RU has multiple antenna elements, treat each element as a TX "point" (tx_idx)
    # - If UE has multiple antenna elements, treat each element as an RX "point" (row in arrays)
    # - Multi-antenna + multi-user is not supported in this mode

    # Build mapping from RU/UE IDs to txrx_set IDs
    tx_id_map = {}  # ru_id -> (tx_set_id, tx_idx)

    # Map transmitters
    for txrx_set in txrx_dict.values():
        if txrx_set["is_tx"]:
            tx_id_map[txrx_set["id_orig"]] = (
                txrx_set["id"],
                0,
            )  # tx_idx is always 0 since each TX is its own set

    # Get the single receiver set
    rx_set = next(v for v in txrx_dict.values() if v["is_rx"])
    rx_set_id = rx_set["id"]

    time_idx = paths_df["time_idx"].unique()[0]  # take first time index
    paths_time_df = paths_df[paths_df["time_idx"] == time_idx]
    cirs_time_df = cirs_df[cirs_df["time_idx"] == time_idx]

    # Determine whether antenna element indices exist
    has_ru_ant = len(_col_unique_values(cirs_time_df, "ru_ant_el")) > 0
    has_ue_ant = len(_col_unique_values(cirs_time_df, "ue_ant_el")) > 0

    for ru_id in paths_time_df["ru_id"].unique():
        paths_ru_df = paths_time_df[paths_time_df["ru_id"] == ru_id]
        cirs_ru_df = cirs_time_df[cirs_time_df["ru_id"] == ru_id]

        # Get number of UEs (receivers) for this RU
        ue_ids = paths_ru_df["ue_id"].unique()

        ru_ant_els = _col_unique_values(cirs_ru_df, "ru_ant_el") if has_ru_ant else np.array([0])
        ue_ant_els = _col_unique_values(cirs_ru_df, "ue_ant_el") if has_ue_ant else np.array([0])
        ru_ant_els = np.sort(ru_ant_els) if len(ru_ant_els) else np.array([0])
        ue_ant_els = np.sort(ue_ant_els) if len(ue_ant_els) else np.array([0])

        multi_tx_ant = len(ru_ant_els) > 1
        multi_rx_ant = len(ue_ant_els) > 1

        # Multi-antenna + multi-user limitations (same spirit as Sionna paths converter)
        if multi_tx_ant and len(paths_time_df["ru_id"].unique()) > 1:
            msg = "Multi-antenna & multi-RU not supported yet"
            raise ValueError(msg)
        if multi_rx_ant and len(ue_ids) > 1:
            msg = "Multi-antenna & multi-UE not supported yet"
            raise ValueError(msg)

        # Update txrx_dict metadata for multi-antenna-as-points representation
        if multi_tx_ant:
            # Update the corresponding TX set (same set_id, multiple tx_idx)
            for txrx_set in txrx_dict.values():
                if txrx_set.get("is_tx") and txrx_set.get("id_orig") == ru_id:
                    txrx_set["num_points"] = len(ru_ant_els)
                    txrx_set["num_active_points"] = len(ru_ant_els)
                    txrx_set["num_ant"] = 1
                    txrx_set["dual_pol"] = False
                    break

        if multi_rx_ant:
            # Update RX set to reflect per-element "points"
            rx_set = next(v for v in txrx_dict.values() if v["is_rx"])
            rx_set["num_points"] = len(ue_ant_els)
            rx_set["num_active_points"] = len(ue_ant_els)
            rx_set["num_ant"] = 1
            rx_set["dual_pol"] = False

        n_rx_points = len(ue_ant_els) if multi_rx_ant else len(ue_ids)
        n_paths = c.MAX_PATHS

        # Get TX/RX set IDs for saving
        tx_set_id, _ = tx_id_map[ru_id]

        for tx_ant_idx, ru_ant_el in enumerate(ru_ant_els):
            data = _preallocate_data(n_rx_points, n_paths)

            if multi_rx_ant:
                # Single UE, multiple antenna elements => treat antenna elements as receiver points
                ue_id = ue_ids[0]
                paths = paths_ru_df[paths_ru_df["ue_id"] == ue_id]

                for rx_ant_idx, ue_ant_el in enumerate(ue_ant_els):
                    _fill_path_geometry(
                        data,
                        rx_ant_idx,
                        paths,
                        set_tx_pos=(rx_ant_idx == 0),
                    )

                    cirs = cirs_ru_df[cirs_ru_df["ue_id"] == ue_id]
                    if has_ru_ant:
                        cirs = cirs[cirs["ru_ant_el"] == ru_ant_el]
                    if has_ue_ant:
                        cirs = cirs[cirs["ue_ant_el"] == ue_ant_el]
                    if len(cirs) == 0:
                        print(
                            f"Warning: No CIR data for RU {ru_id} UE {ue_id} "
                            f"(ru_ant_el={ru_ant_el}, ue_ant_el={ue_ant_el})",
                        )
                        continue

                    ant_cirs = cirs.iloc[[0]]
                    cir_data = ant_cirs["cir_re"].to_numpy()[0] + 1j * ant_cirs["cir_im"].to_numpy()[0]
                    cir_power = 20 * np.log10(np.abs(cir_data))
                    data[c.POWER_PARAM_NAME][rx_ant_idx, : len(cir_data)] = cir_power
                    data[c.PHASE_PARAM_NAME][rx_ant_idx, : len(cir_data)] = np.angle(
                        cir_data,
                        deg=True,
                    )
                    data[c.DELAY_PARAM_NAME][rx_ant_idx, : len(cir_data)] = ant_cirs[
                        "cir_delay"
                    ].to_numpy()[0]
            else:
                # One antenna element per UE => treat UEs as receiver points
                for rx_idx, ue_id in enumerate(ue_ids):
                    paths = paths_ru_df[paths_ru_df["ue_id"] == ue_id]
                    cirs = cirs_ru_df[cirs_ru_df["ue_id"] == ue_id]
                    if has_ru_ant:
                        cirs = cirs[cirs["ru_ant_el"] == ru_ant_el]
                    if has_ue_ant:
                        cirs = cirs[cirs["ue_ant_el"] == ue_ant_els[0]]

                    if len(cirs) == 0:
                        print(f"Warning: No CIR data for RU {ru_id} UE {ue_id}")
                        continue

                    _fill_path_geometry(
                        data,
                        rx_idx,
                        paths,
                        set_tx_pos=(rx_idx == 0),
                    )

                    ant_cirs = cirs.iloc[[0]]
                    cir_data = ant_cirs["cir_re"].to_numpy()[0] + 1j * ant_cirs["cir_im"].to_numpy()[0]
                    cir_power = 20 * np.log10(np.abs(cir_data))
                    data[c.POWER_PARAM_NAME][rx_idx, : len(cir_data)] = cir_power
                    data[c.PHASE_PARAM_NAME][rx_idx, : len(cir_data)] = np.angle(
                        cir_data,
                        deg=True,
                    )
                    data[c.DELAY_PARAM_NAME][rx_idx, : len(cir_data)] = ant_cirs[
                        "cir_delay"
                    ].to_numpy()[0]

            # Compress data before saving
            data = cu.compress_path_data(data)

            # Save data: treat RU antenna element index as tx_idx (same philosophy as Sionna paths)
            tx_idx = tx_ant_idx if multi_tx_ant else 0
            for key in data:
                mat_filename = gu.get_mat_filename(key, tx_set_id, tx_idx, rx_set_id)
                gu.save_mat(data[key], key, str(Path(output_folder) / mat_filename))
