"""
Correlation Matrix Heatmap
- Computes Pearson correlation between all geometry features and channel features
- Displays a single heatmap with annotations for quick overview
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("files/extended_experiment_results.csv")
df = df[df['max_dist'] == 300].reset_index(drop=True)

# Filter: BS height = 6m only (US scenarios)
df = df[df['bs_height'] == 6.0].reset_index(drop=True)
print(f"Using {len(df)} samples (300m radius, BS height=6m only)\n")

# ---------------------------------------------------------------------------
# 2. Define feature groups
# ---------------------------------------------------------------------------
geometry_features = [
    'solid_angle_blockage',
    'bs_relative_height_penalty',
    'dist_weighted_surf',
    'skyline_roughness',
    'avg_surface_volume_ratio',
    'num_buildings',
    'avg_building_height',
    'max_building_height',
    'avg_footprint',
    'min_building_dist',
    'height_ratio',
]

channel_features = [
    'los_prob',
    'avg_pathloss',
    'rms_delay_spread',
    'k_factor_db',
    'avg_num_paths',
    'angular_spread_az',
    'std_pathloss',
    'std_delay_spread',
    'reflection_ratio',
    'diffraction_ratio',
]

# Readable labels
geo_labels = [
    'Solid Angle Blockage',
    'BS-Relative Height Penalty',
    'Dist-Weighted Surf Area',
    'Skyline Roughness',
    'Surface/Volume Ratio',
    'Num Buildings',
    'Avg Height',
    'Max Height',
    'Avg Footprint',
    'Min Building Dist',
    'Height Ratio',
]

ch_labels = [
    'LOS Prob',
    'Avg Pathloss',
    'RMS Delay Spread',
    'K-Factor',
    'Avg Num Paths',
    'Angular Spread',
    'Std Pathloss',
    'Std Delay Spread',
    'Reflection Ratio',
    'Diffraction Ratio',
]

# ---------------------------------------------------------------------------
# 3. Compute correlation matrix (Pearson & Spearman) + p-values
# ---------------------------------------------------------------------------
n_geo = len(geometry_features)
n_ch = len(channel_features)

# Pearson
pearson_corr_matrix = np.zeros((n_geo, n_ch))
pearson_pval_matrix = np.zeros((n_geo, n_ch))

# Spearman
spearman_corr_matrix = np.zeros((n_geo, n_ch))
spearman_pval_matrix = np.zeros((n_geo, n_ch))

for i, gf in enumerate(geometry_features):
    for j, cf in enumerate(channel_features):
        x, y = df[gf].values, df[cf].values
        # Skip constant columns (e.g. diffraction_ratio = 0 for all 6m cities)
        if np.std(x) == 0 or np.std(y) == 0:
            pearson_corr_matrix[i, j]  = np.nan
            pearson_pval_matrix[i, j]  = np.nan
            spearman_corr_matrix[i, j] = np.nan
            spearman_pval_matrix[i, j] = np.nan
            continue
        # Pearson
        r_p, p_p = stats.pearsonr(x, y)
        pearson_corr_matrix[i, j] = r_p
        pearson_pval_matrix[i, j] = p_p

        # Spearman
        r_s, p_s = stats.spearmanr(x, y)
        spearman_corr_matrix[i, j] = r_s
        spearman_pval_matrix[i, j] = p_s

# ---------------------------------------------------------------------------
# 4. Build annotation strings: r value + significance stars
# ---------------------------------------------------------------------------
# Pearson annotations
pearson_annot = np.empty((n_geo, n_ch), dtype=object)
for i in range(n_geo):
    for j in range(n_ch):
        r = pearson_corr_matrix[i, j]
        p = pearson_pval_matrix[i, j]
        stars = ''
        if np.isnan(r):
            pearson_annot[i, j] = 'N/A'
        else:
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            pearson_annot[i, j] = f'{r:.2f}{stars}'

# Spearman annotations
spearman_annot = np.empty((n_geo, n_ch), dtype=object)
for i in range(n_geo):
    for j in range(n_ch):
        r = spearman_corr_matrix[i, j]
        p = spearman_pval_matrix[i, j]
        stars = ''
        if np.isnan(r):
            spearman_annot[i, j] = 'N/A'
        else:
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            spearman_annot[i, j] = f'{r:.2f}{stars}'

# ---------------------------------------------------------------------------
# 5. Plot heatmaps (Pearson and Spearman side-by-side)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)) # Two subplots side-by-side

# Pearson Heatmap
sns.heatmap(
    pearson_corr_matrix,
    annot=pearson_annot,
    fmt='',
    xticklabels=ch_labels,
    yticklabels=geo_labels,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8},
    ax=ax1,
)

ax1.set_title(
    'Pearson Correlation (BS height=6m only, US)\n'
    '(* p<0.05, ** p<0.01, *** p<0.001)',
    fontsize=14, fontweight='bold', pad=15,
)
ax1.set_xlabel('Channel Features', fontsize=12, labelpad=10)
ax1.set_ylabel('Geometry Features', fontsize=12, labelpad=10)
plt.setp(ax1.get_xticklabels(), rotation=35, ha='right', fontsize=10)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)


# Spearman Heatmap
sns.heatmap(
    spearman_corr_matrix,
    annot=spearman_annot,
    fmt='',
    xticklabels=ch_labels,
    yticklabels=geo_labels,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Spearman Correlation (rho)', 'shrink': 0.8},
    ax=ax2,
)

ax2.set_title(
    'Spearman Correlation (BS height=6m only, US)\n'
    '(* p<0.05, ** p<0.01, *** p<0.001)',
    fontsize=14, fontweight='bold', pad=15,
)
ax2.set_xlabel('Channel Features', fontsize=12, labelpad=10)
ax2.set_ylabel('Geometry Features', fontsize=12, labelpad=10)
plt.setp(ax2.get_xticklabels(), rotation=35, ha='right', fontsize=10)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=10)


plt.tight_layout()
plt.savefig('figures/correlation_matrix_heatmap_pearson_spearman_bs6m.png', dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 6. Print summary table to console
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PEARSON CORRELATION MATRIX SUMMARY (r)")
print("=" * 80)

pearson_corr_df = pd.DataFrame(pearson_corr_matrix, index=geo_labels, columns=ch_labels)
print(pearson_corr_df.round(3).to_string())

print("\n" + "=" * 80)
print("SPEARMAN CORRELATION MATRIX SUMMARY (rho)")
print("=" * 80)

spearman_corr_df = pd.DataFrame(spearman_corr_matrix, index=geo_labels, columns=ch_labels)
print(spearman_corr_df.round(3).to_string())


print("\n" + "-" * 80)
print("STRONGEST CORRELATIONS (|r| > 0.4) - Pearson:")
print("-" * 80)
for i in range(n_geo):
    for j in range(n_ch):
        r = pearson_corr_matrix[i, j]
        p = pearson_pval_matrix[i, j]
        if abs(r) > 0.4:
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {geo_labels[i]:25s} ↔ {ch_labels[j]:20s}  r = {r:+.3f}  p = {p:.4f} {sig}")

print("\n" + "-" * 80)
print("STRONGEST CORRELATIONS (|rho| > 0.4) - Spearman:")
print("-" * 80)
for i in range(n_geo):
    for j in range(n_ch):
        r = spearman_corr_matrix[i, j]
        p = spearman_pval_matrix[i, j]
        if abs(r) > 0.4:
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {geo_labels[i]:25s} ↔ {ch_labels[j]:20s}  rho = {r:+.3f}  p = {p:.4f} {sig}")


print("\n" + "-" * 80)
print("SIGNIFICANT DISAGREEMENTS BETWEEN PEARSON AND SPEARMAN (|r_p - r_s| > 0.2 AND both p < 0.05):")
print("-" * 80)
disagreements_found = False
for i in range(n_geo):
    for j in range(n_ch):
        r_p = pearson_corr_matrix[i, j]
        p_p = pearson_pval_matrix[i, j]
        r_s = spearman_corr_matrix[i, j]
        p_s = spearman_pval_matrix[i, j]

        # Check for significant difference and if both are statistically significant
        if abs(r_p - r_s) > 0.2 and p_p < 0.05 and p_s < 0.05:
            print(f"  {geo_labels[i]:25s} ↔ {ch_labels[j]:20s}  Pearson r = {r_p:+.3f} (p={p_p:.4f}) "
                  f"Spearman rho = {r_s:+.3f} (p={p_s:.4f})  Diff = {abs(r_p - r_s):.3f}")
            disagreements_found = True
if not disagreements_found:
    print("  No significant disagreements found based on criteria.")


print(f"\nTotal feature pairs: {n_geo * n_ch}")
strong_pearson = np.sum(np.abs(pearson_corr_matrix) > 0.4)
strong_spearman = np.sum(np.abs(spearman_corr_matrix) > 0.4)
print(f"Strong Pearson correlations (|r| > 0.4): {strong_pearson}")
print(f"Strong Spearman correlations (|rho| > 0.4): {strong_spearman}")
print(f"\nHeatmap saved to: figures/correlation_matrix_heatmap_pearson_spearman_bs6m.png")
