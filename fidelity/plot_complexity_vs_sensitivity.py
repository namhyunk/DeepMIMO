"""Main paper figure: scene-complexity descriptor → fidelity sensitivity slope.

Reads:
  --complexity  scene_complexity.json  (per scene: building_count, los_probability, ...)
  --thresholds  threshold_table.json   (per scene × tag: mean_drop, worst_drop, slope, ML drops)

Plots, for one scene-complexity descriptor:
  - 4 series (geometry / material / ray_tracing / hardware)
  - X = the chosen descriptor (e.g. building_count)
  - Y = mean_drop (1 - mean unified_fidelity_score) for that tag in that scene
  - One marker per scene; lines connect markers in descriptor-sorted order.
  - Optional secondary panel for ML beam_top1 drop (pp) — same X, ML-derived Y.

The single-descriptor mode mirrors the canonical call in
``fidelity/experiments/05_complexity_vs_sensitivity.sh``:

    python -m fidelity.plot_complexity_vs_sensitivity \
        --complexity fidelity/results/_aggregate/scene_complexity.json \
        --thresholds fidelity/results/_aggregate/threshold_table.json \
        --descriptor building_count \
        --output     fidelity/results/_figures/complexity_vs_sensitivity_building_count

A bare ``--all-descriptors`` flag generates the full set in one go.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAGS = ("geometry", "material", "ray_tracing", "hardware")
TAG_COLORS = {
    "geometry":    "#1f77b4",
    "material":    "#2ca02c",
    "ray_tracing": "#d62728",
    "hardware":    "#9467bd",
}
TAG_MARKERS = {
    "geometry":    "o",
    "material":    "s",
    "ray_tracing": "^",
    "hardware":    "D",
}

DESCRIPTOR_LABELS = {
    "building_count":           "Number of buildings",
    "los_probability":          "Line-of-sight probability",
    "building_volume_fraction": "Building volume / scene-bbox volume",
    "angular_spread_proxy":     "Building-azimuth entropy [bits]",
}


def _scene_descriptor(complexity: dict, scene: str, descriptor: str) -> float | None:
    if scene not in complexity:
        return None
    v = complexity[scene].get(descriptor)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _tag_y(thresholds: dict, scene: str, tag: str, y_key: str) -> float | None:
    sc = thresholds.get("scenes", {}).get(scene, {})
    tag_row = sc.get("tags", {}).get(tag, {})
    v = tag_row.get(y_key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _plot_one(complexity: dict, thresholds: dict, descriptor: str,
              y_key: str, y_label: str, ax) -> None:
    scenes = list(thresholds.get("scenes", {}))
    points_per_tag: dict[str, list[tuple[float, float, str]]] = {}
    for scene in scenes:
        x = _scene_descriptor(complexity, scene, descriptor)
        if x is None:
            continue
        for tag in TAGS:
            y = _tag_y(thresholds, scene, tag, y_key)
            if y is None:
                continue
            points_per_tag.setdefault(tag, []).append((x, y, scene))

    for tag in TAGS:
        pts = sorted(points_per_tag.get(tag, []), key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys,
                marker=TAG_MARKERS[tag],
                color=TAG_COLORS[tag],
                linewidth=1.6, markersize=8,
                label=tag.replace("_", " "))
        for x, y, scene in pts:
            ax.annotate(scene, (x, y),
                        textcoords="offset points",
                        xytext=(6, 4),
                        fontsize=8, color="#444444")

    ax.set_xlabel(DESCRIPTOR_LABELS.get(descriptor, descriptor))
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_descriptor(complexity: dict, thresholds: dict, descriptor: str,
                    output_prefix: Path, with_ml: bool = True) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if with_ml:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        _plot_one(complexity, thresholds, descriptor,
                  "mean_drop",
                  r"$1 - \overline{\mathrm{unified\ fidelity}}$ (channel-level)",
                  axes[0])
        _plot_one(complexity, thresholds, descriptor,
                  "ml_mean_beam_top1_drop_pp",
                  "Mean beam-top1 drop [pp]  (ML task)",
                  axes[1])
        for ax in axes:
            ax.legend(loc="best", fontsize=9, frameon=True)
        fig.suptitle(
            f"Cross-scene fidelity sensitivity vs "
            f"{DESCRIPTOR_LABELS.get(descriptor, descriptor)}",
            fontsize=11,
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        _plot_one(complexity, thresholds, descriptor,
                  "mean_drop",
                  r"$1 - \overline{\mathrm{unified\ fidelity}}$",
                  ax)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(
            f"Sensitivity vs {DESCRIPTOR_LABELS.get(descriptor, descriptor)}",
            fontsize=10,
        )

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    paths.extend([png, pdf])
    print(f"[plot] wrote {png} and {pdf}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complexity", required=True,
                        help="Path to scene_complexity.json (output of fidelity.scene_complexity)")
    parser.add_argument("--thresholds", required=True,
                        help="Path to threshold_table.json (output of fidelity.aggregate)")
    parser.add_argument("--descriptor", default=None,
                        help="Single descriptor to plot (e.g. building_count). "
                             "Ignored if --all-descriptors is set.")
    parser.add_argument("--all-descriptors", action="store_true",
                        help="Generate one plot per known descriptor.")
    parser.add_argument("--output", default=None,
                        help="Output prefix (no suffix). Required unless --all-descriptors.")
    parser.add_argument("--output-dir", default="fidelity/results/_figures",
                        help="Used with --all-descriptors as the destination dir.")
    parser.add_argument("--no-ml", action="store_true",
                        help="Skip the ML-task panel.")
    args = parser.parse_args()

    with open(args.complexity) as f:
        complexity = json.load(f)
    with open(args.thresholds) as f:
        thresholds = json.load(f)

    if args.all_descriptors:
        out_dir = Path(args.output_dir)
        for d in DESCRIPTOR_LABELS:
            prefix = out_dir / f"complexity_vs_sensitivity_{d}"
            plot_descriptor(complexity, thresholds, d, prefix,
                            with_ml=not args.no_ml)
        return

    if not args.descriptor:
        raise SystemExit("Either --descriptor or --all-descriptors must be set.")
    if not args.output:
        raise SystemExit("--output (prefix) required without --all-descriptors.")
    plot_descriptor(complexity, thresholds, args.descriptor,
                    Path(args.output), with_ml=not args.no_ml)


if __name__ == "__main__":
    main()
