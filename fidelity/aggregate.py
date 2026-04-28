"""Aggregate per-scene fidelity metrics into a cross-scene threshold table.

Reads ``fidelity/results/<scene>/fidelity_metrics.json`` and (optionally)
``ml_task_results.json`` for each scene, and emits a single JSON keyed by
scene name containing per-tag sensitivity summaries that
plot_complexity_vs_sensitivity.py consumes.

Per (scene, tag) we report:

  mean_drop          : 1 - mean(unified_fidelity_score) over the tag's configs
                       — higher = more sensitive in this tag.
  worst_drop         : 1 - min(unified_fidelity_score) over the tag's configs.
  configs            : count of configs that contributed.
  sensitivity_slope  : least-squares slope of (1 - unified) vs. ordered
                       perturbation magnitude, when a magnitude can be
                       parsed from the config name (e.g. ``geo_noise_5m``,
                       ``rt_depth_3``). NaN otherwise.

Usage:
    python -m fidelity.aggregate \
        --scenes simple_street_canyon munich \
        --output fidelity/results/_aggregate/threshold_table.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

TAG_OF_CONFIG: dict[str, str] = {}

CONFIG_TAG_PREFIXES = (
    ("geo_", "geometry"),
    ("mat_", "material"),
    ("rt_",  "ray_tracing"),
    ("hw_",  "hardware"),
)


def _tag_of(cfg: str) -> str | None:
    if cfg in TAG_OF_CONFIG:
        return TAG_OF_CONFIG[cfg]
    for pfx, tag in CONFIG_TAG_PREFIXES:
        if cfg.startswith(pfx):
            return tag
    return None


_NUM_RE = re.compile(r"(\d+(?:_\d+)?)")


def _parse_magnitude(cfg: str) -> float | None:
    """Best-effort numeric magnitude from the config name.

    For sweeps that span an ordered axis we want a scalar to regress against:
      geo_noise_0_5m  -> 0.5    (meters)
      geo_noise_10m   -> 10.0
      rt_depth_3      -> 3.0    (reflections)
      rt_5k_rays      -> 5000   (rays)
      rt_500k_rays    -> 500000
      rt_low_rays     -> NaN    (no number)
    """
    if "_rays" in cfg:
        m = re.search(r"(\d+)([kKmM]?)_rays", cfg)
        if m:
            base = float(m.group(1))
            mult = {"": 1, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}[m.group(2)]
            return base * mult
        return None
    if "rt_depth_" in cfg:
        m = re.search(r"rt_depth_(\d+)", cfg)
        if m:
            return float(m.group(1))
    if "geo_noise_" in cfg:
        m = re.search(r"geo_noise_(\d+(?:_\d+)?)m", cfg)
        if m:
            return float(m.group(1).replace("_", "."))
    if "geo_height_noise_" in cfg:
        m = re.search(r"geo_height_noise_(\d+(?:_\d+)?)m", cfg)
        if m:
            return float(m.group(1).replace("_", "."))
    return None


def _per_tag_summary(metrics: dict) -> dict:
    """Group fidelity_metrics entries by tag → summary stats."""
    tags: dict[str, list] = {"geometry": [], "material": [], "ray_tracing": [], "hardware": []}
    for cfg_name, res in metrics.items():
        if not isinstance(res, dict):
            continue
        unified = res.get("unified_fidelity_score")
        if unified is None:
            continue
        tag = _tag_of(cfg_name)
        if tag is None:
            continue
        mag = _parse_magnitude(cfg_name)
        tags[tag].append((cfg_name, float(unified), mag))

    summary = {}
    for tag, entries in tags.items():
        if not entries:
            summary[tag] = {
                "configs": 0,
                "mean_drop": float("nan"),
                "worst_drop": float("nan"),
                "sensitivity_slope": float("nan"),
            }
            continue
        unified_vals = np.array([u for _, u, _ in entries], dtype=np.float64)
        drops = 1.0 - unified_vals
        mean_drop = float(np.mean(drops))
        worst_drop = float(np.max(drops))

        # Slope of drop vs magnitude (only configs with a parseable magnitude).
        ordered = [(m, d) for (_, _, m), d in zip(entries, drops) if m is not None]
        slope = float("nan")
        if len(ordered) >= 2:
            xs = np.array([m for m, _ in ordered], dtype=np.float64)
            ys = np.array([d for _, d in ordered], dtype=np.float64)
            # log-x for ray counts (orders of magnitude span)
            if tag == "ray_tracing" and "rays" in entries[0][0] and xs.min() > 0:
                xs = np.log10(xs)
            if np.std(xs) > 0:
                slope = float(np.polyfit(xs, ys, 1)[0])

        summary[tag] = {
            "configs": len(entries),
            "mean_drop": mean_drop,
            "worst_drop": worst_drop,
            "sensitivity_slope": slope,
            "config_names": [c for c, _, _ in entries],
        }
    return summary


def _ml_task_summary(ml: list[dict]) -> dict:
    """Per-tag mean drop in beam_top1 (in pp) and loc_rmse (in m), relative
    to the matching baseline within each tag."""
    if not ml:
        return {}
    by_name = {r["config"]: r for r in ml}
    base_top1 = {r["config"]: r.get("beam_top1") for r in ml if r.get("is_baseline")}
    base_rmse = {r["config"]: r.get("loc_rmse") for r in ml if r.get("is_baseline")}

    by_tag: dict[str, list[tuple[float, float]]] = {}
    for r in ml:
        if r.get("is_baseline"):
            continue
        tag = r.get("tag")
        if tag not in by_tag:
            by_tag[tag] = []
        bl = r.get("baseline")
        bl_top1 = base_top1.get(bl)
        bl_rmse = base_rmse.get(bl)
        top1 = r.get("beam_top1")
        rmse = r.get("loc_rmse")
        if top1 is None or bl_top1 is None or rmse is None or bl_rmse is None:
            continue
        try:
            d_top1 = float(bl_top1) - float(top1)
            d_rmse = float(rmse) - float(bl_rmse)
            by_tag[tag].append((d_top1, d_rmse))
        except (TypeError, ValueError):
            continue

    out = {}
    for tag, pairs in by_tag.items():
        if not pairs:
            out[tag] = {"mean_beam_top1_drop_pp": float("nan"),
                        "mean_loc_rmse_increase_m": float("nan"),
                        "configs": 0}
            continue
        a = np.array(pairs, dtype=np.float64)
        out[tag] = {
            "mean_beam_top1_drop_pp": float(np.mean(a[:, 0])),
            "mean_loc_rmse_increase_m": float(np.mean(a[:, 1])),
            "configs": len(pairs),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="Scene subdir names under fidelity/results/")
    parser.add_argument("--results-root", default="fidelity/results",
                        help="Root for per-scene results (default fidelity/results)")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    root = Path(args.results_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = {"scenes": {}}
    for scene in args.scenes:
        scene_dir = root / scene
        fid_path = scene_dir / "fidelity_metrics.json"
        ml_path = scene_dir / "ml_task_results.json"
        if not fid_path.exists():
            print(f"[aggregate] missing: {fid_path} (skipping {scene})")
            continue
        with open(fid_path) as f:
            metrics = json.load(f)
        per_tag = _per_tag_summary(metrics)

        ml_summary: dict = {}
        if ml_path.exists():
            with open(ml_path) as f:
                ml = json.load(f)
            ml_summary = _ml_task_summary(ml)

        # Merge ml summaries onto per_tag dict by tag.
        for tag, ml_row in ml_summary.items():
            per_tag.setdefault(tag, {}).update(
                {f"ml_{k}": v for k, v in ml_row.items()}
            )

        table["scenes"][scene] = {
            "n_configs_total": sum(per_tag[t].get("configs", 0) for t in per_tag),
            "tags": per_tag,
        }
        print(f"[aggregate] {scene}: tags={list(per_tag)}, "
              f"mean_drops="
              f"geo={per_tag['geometry']['mean_drop']:.3f} "
              f"mat={per_tag['material']['mean_drop']:.3f} "
              f"rt={per_tag['ray_tracing']['mean_drop']:.3f} "
              f"hw={per_tag['hardware']['mean_drop']:.3f}")

    with open(out_path, "w") as f:
        json.dump(table, f, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print(f"[aggregate] wrote {out_path}")


if __name__ == "__main__":
    main()
