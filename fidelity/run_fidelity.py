"""Fidelity Experiment Runner.

Main script to generate digital twin datasets at varying fidelity levels.
Iterates over FidelityConfig presets and runs Sionna ray tracing for each.

Usage:
    # Run all presets on a built-in scene
    python -m fidelity.run_fidelity --scene builtin:simple_street_canyon

    # Run specific config(s)
    python -m fidelity.run_fidelity --configs baseline geo_noise_5m mat_all_concrete

    # Run all configs for a component
    python -m fidelity.run_fidelity --tag geometry

    # Run on an OSM scene folder
    python -m fidelity.run_fidelity --scene /path/to/osm_folder
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deepmimo as dm  # noqa: E402

from fidelity.config import (  # noqa: E402
    ALL_CONFIGS,
    BASELINE,
    CONFIGS_BY_NAME,
    FidelityConfig,
    get_configs_by_tag,
)
from fidelity.material_editors import build_material_editor  # noqa: E402
from fidelity.scene_editors import build_geometry_editor  # noqa: E402

# GPU configuration
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"


# ============================================================================
# Scene Edit Function Builder
# ============================================================================


def build_scene_editor(config: FidelityConfig):
    """Build a composite scene_edit_func from geometry + material editors.

    The returned callable applies geometry edits first, then material edits.
    Returns None if no edits are needed.
    """
    geo_editor = build_geometry_editor(config)
    mat_editor = build_material_editor(config)

    if geo_editor is None and mat_editor is None:
        return None

    def combined_editor(scene):
        if geo_editor is not None:
            geo_editor(scene)
        if mat_editor is not None:
            mat_editor(scene)

    return combined_editor


def build_antenna_arrays(config: FidelityConfig):
    """Build TX and RX antenna array configurations from config.

    Returns kwargs to override in the scene creation step.
    """
    from sionna.rt import PlanarArray

    tx_array = PlanarArray(
        num_rows=config.num_tx_ant_rows,
        num_cols=config.num_tx_ant_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=config.antenna_pattern,
        polarization="V",
    )
    rx_array = PlanarArray(
        num_rows=config.num_rx_ant_rows,
        num_cols=config.num_rx_ant_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=config.antenna_pattern,
        polarization="V",
    )
    return tx_array, rx_array


# ============================================================================
# Base RT Parameters (shared across all configs)
# ============================================================================

BASE_RT_PARAMS = {
    # Scenario
    "carrier_freq": 3.5e9,
    "bandwidth": 10e6,
    "max_paths": 10,
    "ray_spacing": 0.25,
    "max_transmissions": 0,
    "ds_max_reflections": 2,
    "ds_max_transmissions": 0,
    "ds_max_diffractions": 1,
    "ds_final_interaction_only": True,
    "conform_to_terrain": False,
    "bs2bs": False,
    # Sionna specific
    "los": True,
    "synthetic_array": True,
    "batch_size": 15,
    "use_builtin_scene": False,
    "builtin_scene_path": "",
    "path_inspection_func": None,
    "scene_edit_func": None,
    "create_scene_folder": False,
    # Sionna 0.x
    "scat_random_phases": True,
    "edge_diffraction": False,
    "scat_keep_prob": 0.001,
    # Sionna 1.x
    "n_samples_per_src": 1_000_000,
    "max_paths_per_src": 1_000_000,
    "refraction": False,
    "cpu_offload": True,
    "rx_ori": None,
    "rx_vel": None,
    "tx_ori": None,
    "tx_vel": None,
    "obj_idx": None,
    "obj_pos": None,
    "obj_ori": None,
    "obj_vel": None,
    # RT knobs (overridden per config)
    "max_reflections": 5,
    "max_diffractions": 0,
    "ds_enable": False,
}


def build_rt_params(config: FidelityConfig) -> dict:
    """Merge a FidelityConfig's RT params into the base params."""
    params = BASE_RT_PARAMS.copy()
    params.update(config.to_rt_params())
    params["scene_edit_func"] = build_scene_editor(config)
    return params


# ============================================================================
# Main Runner
# ============================================================================


def run_single_config(
    config: FidelityConfig,
    scene_folder: str,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    output_root: str,
    *,
    use_builtin: bool = False,
    builtin_scene_path: str = "",
) -> str:
    """Run ray tracing for a single fidelity configuration.

    Args:
        config: FidelityConfig defining the fidelity level.
        scene_folder: Path to the scene folder (with scene.xml).
        tx_pos: TX positions array [N_TX, 3].
        rx_pos: RX positions array [N_RX, 3].
        output_root: Root directory for results.
        use_builtin: Whether to use a Sionna built-in scene.
        builtin_scene_path: Name of built-in scene (if use_builtin=True).

    Returns:
        Path to the generated DeepMIMO scenario.

    """
    from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

    print(f"\n{'=' * 60}")
    print(f"  Running: {config.name}")
    print(f"  Description: {config.description}")
    print(f"{'=' * 60}\n")

    # Build RT parameters
    params = build_rt_params(config)

    # Configure scene source
    if use_builtin:
        params["use_builtin_scene"] = True
        params["builtin_scene_path"] = builtin_scene_path

    # Configure antenna arrays if non-default
    if config.has_hardware_change:
        # Hardware changes require modifying the scene after creation
        # We inject this via scene_edit_func
        original_editor = params["scene_edit_func"]

        def hw_editor(scene):
            if original_editor is not None:
                original_editor(scene)
            tx_array, rx_array = build_antenna_arrays(config)
            scene.tx_array = tx_array
            scene.rx_array = rx_array
            print(
                f"[hw_editor] Set antenna: pattern={config.antenna_pattern}, "
                f"TX={config.num_tx_ant_rows}x{config.num_tx_ant_cols}, "
                f"RX={config.num_rx_ant_rows}x{config.num_rx_ant_cols}"
            )

        params["scene_edit_func"] = hw_editor

    # Output folder
    config_output = os.path.join(output_root, config.name)
    os.makedirs(config_output, exist_ok=True)

    # Run ray tracing
    t_start = time.time()
    rt_path = raytrace_sionna(scene_folder, tx_pos, rx_pos, **params)
    t_elapsed = time.time() - t_start

    # Convert to DeepMIMO format
    scen_name = dm.convert(rt_path, scenario_name=config.name, overwrite=True)

    # Save metadata
    metadata = {
        "config_name": config.name,
        "description": config.description,
        "tags": config.tags,
        "rt_params": {k: v for k, v in config.to_rt_params().items()},
        "geometry": {
            "position_noise_std": config.position_noise_std,
            "height_noise_std": config.height_noise_std,
            "remove_buildings_below": config.remove_buildings_below,
            "remove_buildings_fraction": config.remove_buildings_fraction,
        },
        "material": {
            "uniform_material": config.uniform_material,
            "disable_scattering": config.disable_scattering,
        },
        "hardware": {
            "antenna_pattern": config.antenna_pattern,
            "tx_array": f"{config.num_tx_ant_rows}x{config.num_tx_ant_cols}",
            "rx_array": f"{config.num_rx_ant_rows}x{config.num_rx_ant_cols}",
        },
        "runtime_seconds": t_elapsed,
        "scenario_name": scen_name,
        "scene_folder": scene_folder,
    }

    metadata_path = os.path.join(config_output, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  ✓ {config.name} completed in {t_elapsed:.1f}s")
    print(f"  ✓ Scenario: {scen_name}")
    print(f"  ✓ Metadata: {metadata_path}")

    return scen_name


def run_experiment(
    configs: list[FidelityConfig],
    scene_folder: str,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    output_root: str,
    *,
    use_builtin: bool = False,
    builtin_scene_path: str = "",
) -> dict[str, str]:
    """Run the full fidelity experiment across multiple configurations.

    Args:
        configs: List of FidelityConfig objects to run.
        scene_folder: Path to scene folder or built-in scene name.
        tx_pos: TX positions [N_TX, 3].
        rx_pos: RX positions [N_RX, 3].
        output_root: Root output directory.
        use_builtin: Whether to use a Sionna built-in scene.
        builtin_scene_path: Name of built-in scene.

    Returns:
        Dict mapping config.name -> scenario_name.

    """
    os.makedirs(output_root, exist_ok=True)
    results = {}

    print(f"\n{'#' * 60}")
    print(f"  FIDELITY EXPERIMENT")
    print(f"  Configs: {len(configs)}")
    print(f"  Scene: {scene_folder}")
    print(f"  Output: {output_root}")
    print(f"{'#' * 60}\n")

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] ", end="")
        try:
            scen_name = run_single_config(
                config,
                scene_folder,
                tx_pos,
                rx_pos,
                output_root,
                use_builtin=use_builtin,
                builtin_scene_path=builtin_scene_path,
            )
            results[config.name] = scen_name
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n  ✗ {config.name} FAILED: {e}")
            results[config.name] = f"ERROR: {e}"

    # Save summary
    summary_path = os.path.join(output_root, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results: {len([v for v in results.values() if not v.startswith('ERROR')])}"
          f"/{len(configs)} succeeded")
    print(f"  Summary: {summary_path}")
    print(f"{'#' * 60}\n")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_scene_arg(scene_str: str) -> tuple[str, bool, str]:
    """Parse the --scene argument.

    Returns:
        (scene_folder, use_builtin, builtin_scene_path)
    """
    if scene_str.startswith("builtin:"):
        builtin_name = scene_str.split(":", 1)[1]
        return "", True, builtin_name
    return scene_str, False, ""


def main():
    parser = argparse.ArgumentParser(
        description="Run fidelity degradation experiments using Sionna RT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline on built-in scene
  python -m fidelity.run_fidelity --scene builtin:simple_street_canyon --configs baseline

  # Run all geometry configs
  python -m fidelity.run_fidelity --scene /path/to/osm_folder --tag geometry

  # Run specific configs
  python -m fidelity.run_fidelity --scene /path/to/scene --configs baseline geo_noise_5m rt_depth_1

  # List all available configs
  python -m fidelity.run_fidelity --list
        """,
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="builtin:simple_street_canyon",
        help="Scene folder path or 'builtin:<name>' for Sionna built-in scenes",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        default=None,
        help="Config names to run (space-separated)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run all configs with this tag (geometry, material, ray_tracing, hardware)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: fidelity/results)",
    )
    parser.add_argument(
        "--tx-height",
        type=float,
        default=10.0,
        help="TX antenna height [m] (default: 10.0)",
    )
    parser.add_argument(
        "--rx-spacing",
        type=float,
        default=2.0,
        help="RX grid spacing [m] (default: 2.0)",
    )
    parser.add_argument(
        "--rx-height",
        type=float,
        default=1.5,
        help="RX antenna height [m] (default: 1.5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configs and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configs",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable Fidelity Configs:")
        print("-" * 70)
        for c in ALL_CONFIGS:
            tags_str = ", ".join(c.tags) if c.tags else "baseline"
            print(f"  {c.name:25s} [{tags_str:15s}]  {c.description}")
        return

    # Select configs
    if args.configs:
        configs = []
        for name in args.configs:
            if name not in CONFIGS_BY_NAME:
                print(f"Error: unknown config '{name}'. Use --list to see available configs.")
                return
            configs.append(CONFIGS_BY_NAME[name])
    elif args.tag:
        configs = get_configs_by_tag(args.tag)
        if not configs:
            print(f"Error: no configs found with tag '{args.tag}'")
            return
        # Always include baseline for comparison
        if BASELINE not in configs:
            configs.insert(0, BASELINE)
    elif args.all:
        configs = ALL_CONFIGS
    else:
        configs = [BASELINE]

    # Parse scene
    scene_folder, use_builtin, builtin_scene_path = parse_scene_arg(args.scene)

    # Default output directory
    if args.output is None:
        output_root = str(Path(__file__).resolve().parent / "results")
    else:
        output_root = args.output

    # Generate positions
    from deepmimo.pipelines.txrx_placement import gen_plane_grid

    rx_pos = gen_plane_grid(
        0, 40, 0, 40,
        args.rx_spacing,
        args.rx_height,
    )
    tx_pos = np.array([[20, 20, args.tx_height]])

    print(f"TX positions: {tx_pos.shape[0]} stations")
    print(f"RX positions: {rx_pos.shape[0]} users")

    # Run experiment
    run_experiment(
        configs,
        scene_folder,
        tx_pos,
        rx_pos,
        output_root,
        use_builtin=use_builtin,
        builtin_scene_path=builtin_scene_path,
    )


if __name__ == "__main__":
    main()
