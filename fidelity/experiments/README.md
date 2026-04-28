# Fidelity Experiments — Geometry-Complexity vs Required Fidelity

## Research direction

Reframed primary question:

> Can scene-level geometry complexity predict the fidelity needed for wireless ML?

We replace the narrower "do hand-crafted geometry features correlate with channel
statistics?" framing (which over-relies on per-UE ray-derived quantities like
`n_paths`, `mean_bounces`) with a scene-level view: how does increasing geometry
complexity shift the required digital-twin fidelity, both for channel reconstruction
NMSE and downstream ML tasks (beam prediction, localization)?

## Scene complexity ladder

| Scene                  | Role                                 | Source                 |
|------------------------|--------------------------------------|------------------------|
| `simple_street_canyon` | Controlled toy scene, physics check  | Sionna built-in        |
| `munich`               | High-complexity outdoor, sim-to-sim  | Sionna built-in        |
| `DICHASUS dc41`        | Real measurement, sim-to-real       | DICHASUS + diff-rt-cal |

For each scene we run the same 4-axis fidelity sweep (geometry / material /
ray-tracing / hardware) and the same downstream ML tasks, then compare the
"sufficient fidelity threshold" across scenes.

## Bash scripts

Run from the DeepMIMO repo root unless a script says otherwise.

| Script                                  | Purpose                                                                |
|-----------------------------------------|------------------------------------------------------------------------|
| `00_canyon_full_sweep.sh`               | Reproduce canyon sweep (sanity check the existing results)             |
| `01_munich_full_sweep.sh`               | Replicate the canyon sweep on Munich — fills the Munich gap            |
| `02_dichasus_full_sweep.sh`             | DICHASUS sim-to-real fidelity sweep                                    |
| `03_extract_scene_complexity.sh`        | Compute scene-level geometry descriptors per scene                     |
| `04_aggregate_three_scenes.sh`          | Build the canyon-vs-munich-vs-dichasus comparison table                |
| `05_complexity_vs_sensitivity.sh`       | Main figure: geometry complexity → fidelity sensitivity slope          |
| `99_per_ue_correlation_appendix.sh`     | Per-UE feature correlation (kept as appendix/diagnostic)               |

Output goes to `fidelity/results/<scene>/` (gitignored). Aggregated artifacts
land in `fidelity/results/_aggregate/` and `fidelity/results/_figures/`.

## What's intentionally not in this folder

- The per-UE feature correlation work in `April/` (top-level scratch dir) is
  diagnostic, not the main story. `99_per_ue_correlation_appendix.sh` documents
  how to regenerate those artifacts if needed.
- Geo-LWM / learned geometry embedding is the *next* phase, not this one.
  This phase establishes whether hand-crafted scene-complexity descriptors
  already explain cross-scene fidelity sensitivity. If they do, Geo-LWM is the
  natural follow-up; if they don't, Geo-LWM is the proposed remedy.
