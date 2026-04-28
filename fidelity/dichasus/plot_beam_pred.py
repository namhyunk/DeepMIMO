"""
Plot beam prediction results: fidelity (pwr_corr) vs ML task metrics.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "beam_pred" / "beam_pred_results.json"
OUT_DIR = BASE_DIR.parent / "beamer" / "dichasus_fidelity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    results = [r for r in results if r.get("pwr_corr_fidelity") is not None]
    results.sort(key=lambda r: r["pwr_corr_fidelity"])

    configs = [r["config"] for r in results]
    pwr_corr = np.array([r["pwr_corr_fidelity"] for r in results])
    agreement = np.array([r["label_agreement_pct"] for r in results])
    sim_top1 = np.array([r["sim_trained_real_top1"] for r in results])
    sim_top3 = np.array([r["sim_trained_real_top3"] for r in results])
    oracle_top1 = np.array([r["oracle_real_top1"] for r in results])
    sim_gain = np.array([r["sim_trained_real_gain"] for r in results])
    oracle_gain = np.array([r["oracle_real_gain"] for r in results])
    random_top1 = np.array([r["random_top1"] for r in results])
    random_gain = np.array([r["random_gain"] for r in results])

    # ============================================================
    # Figure 1: Fidelity vs ML performance (scatter + overlay)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(pwr_corr, sim_top1, 'o-', label='Sim-trained → Real top-1', color='tab:blue', markersize=9, linewidth=1.8)
    ax.plot(pwr_corr, sim_top3, 's--', label='Sim-trained → Real top-3', color='tab:cyan', markersize=7, linewidth=1.5)
    ax.plot(pwr_corr, oracle_top1, '^-', label='Oracle (train on real)', color='tab:green', markersize=9, linewidth=1.8)
    ax.axhline(random_top1.mean(), color='gray', linestyle=':', alpha=0.7, label=f'Random ({random_top1.mean():.1f}%)')
    for i, c in enumerate(configs):
        ax.annotate(c, (pwr_corr[i], sim_top1[i]), fontsize=7.5,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Channel Fidelity (pwr_corr from 30-pos experiment)', fontsize=11)
    ax.set_ylabel('Beam Prediction Accuracy (%)', fontsize=11)
    ax.set_title('Does higher channel fidelity → better sim-to-real transfer?', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(pwr_corr, sim_gain, 'o-', label='Sim-trained → Real gain ratio', color='tab:blue', markersize=9, linewidth=1.8)
    ax.plot(pwr_corr, oracle_gain, '^-', label='Oracle gain ratio', color='tab:green', markersize=9, linewidth=1.8)
    ax.axhline(random_gain.mean(), color='gray', linestyle=':', alpha=0.7, label=f'Random ({random_gain.mean():.1f}%)')
    for i, c in enumerate(configs):
        ax.annotate(c, (pwr_corr[i], sim_gain[i]), fontsize=7.5,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Channel Fidelity (pwr_corr)', fontsize=11)
    ax.set_ylabel('Effective Beam Gain Ratio (%)', fontsize=11)
    ax.set_title('Beam gain ratio vs fidelity', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.suptitle('DICHASUS Beam Prediction: Sim-to-Real Transfer vs Channel Fidelity',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out1 = OUT_DIR / "beam_pred_vs_fidelity.png"
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")
    plt.close()

    # ============================================================
    # Figure 2: Per-config bar comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(configs))
    w = 0.28

    ax = axes[0]
    ax.bar(x - w, sim_top1, w, label='Sim→Real (top-1)', color='tab:blue', alpha=0.85)
    ax.bar(x, oracle_top1, w, label='Oracle (top-1)', color='tab:green', alpha=0.85)
    ax.bar(x + w, random_top1, w, label='Random', color='gray', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Beam Top-1 Accuracy by Config')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - w, sim_gain, w, label='Sim→Real gain', color='tab:blue', alpha=0.85)
    ax.bar(x, oracle_gain, w, label='Oracle gain', color='tab:green', alpha=0.85)
    ax.bar(x + w, random_gain, w, label='Random', color='gray', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Gain Ratio (%)')
    ax.set_title('Effective Beam Gain by Config')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DICHASUS Beam Prediction by Fidelity Config', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out2 = OUT_DIR / "beam_pred_per_config.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close()

    # ============================================================
    # Correlation stats
    # ============================================================
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(pwr_corr, sim_top1)
    sr, sp = spearmanr(pwr_corr, sim_top1)
    print(f"\nFidelity (pwr_corr) vs Sim→Real top-1:")
    print(f"  Pearson r={pr:.3f} (p={pp:.3f})")
    print(f"  Spearman r={sr:.3f} (p={sp:.3f})")


if __name__ == "__main__":
    main()
