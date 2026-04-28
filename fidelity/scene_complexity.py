"""Scene-level geometric complexity descriptors.

Computes per-scene descriptors used by aggregate.py and
plot_complexity_vs_sensitivity.py to relate fidelity sensitivity to scene
complexity:

  building_count           : non-terrain object count
  building_volume_fraction : sum(building bbox volumes) / scene bbox volume
  los_probability          : fraction of TX-RX samples with line-of-sight
  angular_spread_proxy     : Shannon entropy (bits) of building-azimuth
                             histogram from the scene centroid; saturates
                             toward log2(n_bins) for uniform azimuth coverage

Usage:
    python -m fidelity.scene_complexity \
        --scenes builtin:simple_street_canyon builtin:munich \
        --output fidelity/results/_aggregate/scene_complexity.json

Each --scenes entry is either ``builtin:<name>`` (Sionna built-in scene) or a
path to a Mitsuba XML.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Force a CPU-capable Mitsuba variant before importing sionna.rt — the LLVM
# fallback path is enough for object iteration, bbox math, and the few
# ray_test() calls we do for the LoS probability descriptor.
os.environ.setdefault("DRJIT_LLVM_OPTIONAL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mitsuba as mi  # noqa: E402

_VARIANT_CANDIDATES = (
    "cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb",
)
for _v in _VARIANT_CANDIDATES:
    if _v in mi.variants():
        try:
            mi.set_variant(_v)
            break
        except Exception:
            continue

from sionna.rt import load_scene, scene as _sc  # noqa: E402

TERRAIN_KEYWORDS = ("plane", "floor", "terrain", "roads", "paths", "road", "path")


def _resolve_scene_arg(arg: str) -> tuple[str, str]:
    """Return (display_name, xml_path) for a CLI scene entry."""
    if arg.startswith("builtin:"):
        builtin = arg.split(":", 1)[1]
        path = getattr(_sc, builtin, None)
        if path is None:
            raise ValueError(f"Unknown built-in scene '{builtin}'")
        return builtin, str(path)
    p = Path(arg)
    if not p.exists():
        raise FileNotFoundError(f"Scene XML not found: {arg}")
    # Use the immediate folder name if present, else the file stem.
    name = p.parent.name if p.parent.name else p.stem
    return name, str(p)


def _is_terrain(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in TERRAIN_KEYWORDS)


def _building_base_name(name: str) -> str:
    """Strip the `-<material>` suffix so '<bldg>-itu_marble' and
    '<bldg>-itu_metal' both map to '<bldg>'. Sionna built-in scenes
    register one object per (building, material) pair."""
    if "-itu_" in name:
        return name.split("-itu_")[0]
    return name


def _bbox_extents(bbox) -> np.ndarray:
    """Return a length-3 numpy array of bbox extents (max-min along each axis)."""
    try:
        ext = bbox.extents()
        return np.array([float(ext[0]), float(ext[1]), float(ext[2])], dtype=np.float64)
    except Exception:
        pass
    try:
        mn = bbox.min
        mx = bbox.max
        return np.array(
            [float(mx[0]) - float(mn[0]),
             float(mx[1]) - float(mn[1]),
             float(mx[2]) - float(mn[2])],
            dtype=np.float64,
        )
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _bbox_center(bbox) -> np.ndarray:
    try:
        mn = bbox.min
        mx = bbox.max
        return np.array(
            [(float(mn[0]) + float(mx[0])) / 2.0,
             (float(mn[1]) + float(mx[1])) / 2.0,
             (float(mn[2]) + float(mx[2])) / 2.0],
            dtype=np.float64,
        )
    except Exception:
        return np.zeros(3, dtype=np.float64)


def _bbox_volume(bbox) -> float:
    return float(np.prod(np.maximum(_bbox_extents(bbox), 0.0)))


def _azimuth_entropy_bits(positions: np.ndarray, n_bins: int = 36) -> float:
    """Shannon entropy of building azimuths (XY-plane) about their centroid."""
    if positions.shape[0] < 2:
        return 0.0
    centroid = positions[:, :2].mean(axis=0)
    deltas = positions[:, :2] - centroid
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # in (-pi, pi]
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def _los_probability(scene, n_samples: int = 256, rx_height: float = 1.5,
                     tx_height: float = 10.0) -> float:
    """Sample n_samples TX–RX pairs uniformly over the scene's XY bbox and
    return the fraction with unobstructed line-of-sight."""
    mi_scene = scene.mi_scene
    bbox = mi_scene.bbox()
    mn = np.asarray([bbox.min[0], bbox.min[1]])
    mx = np.asarray([bbox.max[0], bbox.max[1]])

    rng = np.random.default_rng(0)
    tx_xy = rng.uniform(mn, mx, size=(n_samples, 2))
    rx_xy = rng.uniform(mn, mx, size=(n_samples, 2))
    tx = np.column_stack([tx_xy, np.full(n_samples, tx_height, dtype=np.float64)])
    rx = np.column_stack([rx_xy, np.full(n_samples, rx_height, dtype=np.float64)])

    los_count = 0
    valid = 0
    for o, d_pt in zip(tx, rx):
        direction = d_pt - o
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            continue
        valid += 1
        try:
            ray = mi.Ray3f(
                mi.Point3f(*o.tolist()),
                mi.Vector3f(*(direction / dist).tolist()),
            )
            si = mi_scene.ray_intersect(ray)
            t = float(si.t[0]) if hasattr(si.t, "__getitem__") else float(si.t)
            if t > dist - 1e-3:  # no hit before reaching RX
                los_count += 1
        except Exception:
            # Fall back to ray_test if ray_intersect signature differs.
            try:
                hit = mi_scene.ray_test(ray)
                if not bool(hit):
                    los_count += 1
            except Exception:
                continue
    return los_count / max(valid, 1)


def describe_scene(arg: str) -> dict:
    name, xml_path = _resolve_scene_arg(arg)
    print(f"[scene_complexity] {name}: loading {xml_path}")
    scene = load_scene(xml_path)
    mi_scene = scene.mi_scene

    scene_bbox = mi_scene.bbox()
    scene_extents = np.asarray(scene_bbox.max) - np.asarray(scene_bbox.min)
    scene_volume = float(np.prod(np.maximum(scene_extents, 0.0)))

    # Sionna registers one object per (building, material) pair; fold them
    # back to "buildings" by stripping the '-itu_*' suffix.
    buildings: dict[str, dict] = {}
    for obj_name, obj in scene.objects.items():
        if _is_terrain(obj_name):
            continue
        base = _building_base_name(obj_name)
        mesh = getattr(obj, "mi_mesh", None) or getattr(obj, "mi_shape", None)
        if mesh is None:
            continue
        try:
            bb = mesh.bbox()
        except Exception as e:
            print(f"[scene_complexity]   skip bbox for '{obj_name}': {e}")
            continue
        ext = _bbox_extents(bb)
        ctr = _bbox_center(bb)
        b = buildings.setdefault(base, {"min": np.full(3, np.inf),
                                        "max": np.full(3, -np.inf)})
        b["min"] = np.minimum(b["min"], ctr - ext / 2.0)
        b["max"] = np.maximum(b["max"], ctr + ext / 2.0)

    n_buildings = len(buildings)
    building_volume_sum = float(sum(
        np.prod(np.maximum(b["max"] - b["min"], 0.0)) for b in buildings.values()
    ))
    positions = (
        np.array([(b["min"] + b["max"]) / 2.0 for b in buildings.values()])
        if buildings else np.zeros((0, 3), dtype=np.float64)
    )
    azimuth_entropy = _azimuth_entropy_bits(positions)

    print(f"[scene_complexity]   sampling {256} TX–RX pairs for LoS probability")
    los_p = _los_probability(scene)

    bvf = (
        building_volume_sum / scene_volume if scene_volume > 0 else 0.0
    )
    return {
        "scene_name": name,
        "xml_path": xml_path,
        "building_count": int(n_buildings),
        "building_volume_fraction": float(bvf),
        "los_probability": float(los_p),
        "angular_spread_proxy": float(azimuth_entropy),
        "scene_bbox_extents_m": [float(x) for x in scene_extents.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+", required=True,
        help="Scene IDs: 'builtin:<name>' or path to a Mitsuba XML.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    descriptors = {}
    for s in args.scenes:
        try:
            d = describe_scene(s)
            descriptors[d["scene_name"]] = d
        except FileNotFoundError as e:
            print(f"[scene_complexity] skipping '{s}': {e}")
            continue

    with open(out_path, "w") as f:
        json.dump(descriptors, f, indent=2)
    print(f"[scene_complexity] wrote {out_path} ({len(descriptors)} scenes)")


if __name__ == "__main__":
    main()
