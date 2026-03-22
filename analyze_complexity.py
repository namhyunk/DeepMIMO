"""
Deep analysis of geometric complexity metrics:
1. Geometry-Geometry inter-correlation (which features are redundant?)
2. Partial correlations controlling for confounders
3. New intensive (ratio-based) complexity metrics
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
print(f"Using {len(df)} samples\n")

# ---------------------------------------------------------------------------
# 2. Geometry-Geometry correlation matrix
# ---------------------------------------------------------------------------
geo_cols = [
    'solid_angle_blockage', 'bs_relative_height_penalty',
    'dist_weighted_surf', 'skyline_roughness',
    'avg_surface_volume_ratio', 'num_buildings',
    'avg_building_height', 'max_building_height',
    'avg_footprint', 'min_building_dist', 'height_ratio',
]
geo_labels = [
    'Solid Angle', 'BS-Height Penalty',
    'Dist-Wt Surf', 'Skyline Rough',
    'S/V Ratio', 'Num Buildings',
    'Avg Height', 'Max Height',
    'Avg Footprint', 'Min Bldg Dist', 'Height Ratio',
]

ch_cols = ['los_prob', 'avg_pathloss', 'rms_delay_spread', 'k_factor_db',
           'avg_num_paths', 'angular_spread_az']
ch_labels = ['LOS Prob', 'Avg Pathloss', 'RMS Delay Spread',
             'K-Factor', 'Avg Num Paths', 'Angular Spread']

# --- Geo-Geo correlation ---
n_geo = len(geo_cols)
geo_corr = np.zeros((n_geo, n_geo))
for i in range(n_geo):
    for j in range(n_geo):
        r, _ = stats.pearsonr(df[geo_cols[i]], df[geo_cols[j]])
        geo_corr[i, j] = r

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(geo_corr, annot=True, fmt='.2f',
            xticklabels=geo_labels, yticklabels=geo_labels,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax)
ax.set_title('Geometry Feature Inter-Correlation\n(Which features are redundant?)',
             fontsize=14, fontweight='bold')
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('figures/geo_geo_correlation.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved figures/geo_geo_correlation.png")

# ---------------------------------------------------------------------------
# 3. Partial correlations: S/V Ratio → Channel, controlling for Avg Height
# ---------------------------------------------------------------------------
def partial_corr(x, y, z):
    """Partial correlation between x and y, controlling for z."""
    # Regress x on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    # Regress y on z
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p

print("\n" + "=" * 90)
print("PARTIAL CORRELATION ANALYSIS")
print("=" * 90)

# For each geometry feature, compute:
# (a) Raw correlation with channel features
# (b) Partial correlation controlling for avg_building_height
# (c) Partial correlation controlling for num_buildings
print(f"\n{'Geo Feature':>25s} | {'Channel':>18s} | {'Raw r':>7s} | {'Partial (ctrl Height)':>22s} | {'Partial (ctrl NumBldg)':>22s}")
print("-" * 105)

key_geo = ['avg_surface_volume_ratio', 'dist_weighted_surf', 'skyline_roughness',
           'solid_angle_blockage', 'bs_relative_height_penalty', 'num_buildings', 'avg_building_height']
key_geo_labels = ['S/V Ratio', 'Dist-Wt Surf', 'Skyline Rough',
                  'Solid Angle', 'BS-Height Penalty', 'Num Buildings', 'Avg Height']

for gi, gc in enumerate(key_geo):
    for ci, cc in enumerate(ch_cols):
        raw_r, raw_p = stats.pearsonr(df[gc], df[cc])
        
        # Partial controlling for avg_building_height
        pr_h, pp_h = partial_corr(df[gc].values, df[cc].values, df['avg_building_height'].values)
        
        # Partial controlling for num_buildings  
        pr_n, pp_n = partial_corr(df[gc].values, df[cc].values, df['num_buildings'].values)
        
        sig_raw = '***' if raw_p < 0.001 else ('**' if raw_p < 0.01 else ('*' if raw_p < 0.05 else ''))
        sig_h = '***' if pp_h < 0.001 else ('**' if pp_h < 0.01 else ('*' if pp_h < 0.05 else ''))
        sig_n = '***' if pp_n < 0.001 else ('**' if pp_n < 0.01 else ('*' if pp_n < 0.05 else ''))
        
        print(f"{key_geo_labels[gi]:>25s} | {ch_labels[ci]:>18s} | {raw_r:+.3f}{sig_raw:4s}| {pr_h:+.3f}{sig_h:4s} (p={pp_h:.4f})  | {pr_n:+.3f}{sig_n:4s} (p={pp_n:.4f})")
    print("-" * 105)

# ---------------------------------------------------------------------------
# 4. Analyze the "Intensive vs Extensive" property
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("INTENSIVE vs EXTENSIVE ANALYSIS")
print("=" * 90)
print("\nCorrelation of each geo feature with 'num_buildings' (proxy for city density/size):")
for gi, gc in enumerate(geo_cols):
    r, p = stats.pearsonr(df[gc], df['num_buildings'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    label = "INTENSIVE (low r)" if abs(r) < 0.4 else "EXTENSIVE (high r)"
    print(f"  {geo_labels[gi]:>25s}: r = {r:+.3f} {sig:4s} → {label}")

# ---------------------------------------------------------------------------
# 5. Suggest new metrics: Intensive complexity measures
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("NEW INTENSIVE METRIC CANDIDATES")
print("=" * 90)

# 5a. Height-weighted S/V: S/V * (mean_height / bs_height)
# This captures BOTH shape complexity AND whether buildings are tall enough to matter
df['height_weighted_sv'] = df['avg_surface_volume_ratio'] * df['height_ratio']

# 5b. Effective Scattering Complexity: S/V * avg_height (intensive * intensive)
df['effective_scatter_complexity'] = df['avg_surface_volume_ratio'] * df['avg_building_height']

# 5c. Normalized Height Variance: std_height / bs_height
# (We need to reconstruct std from skyline_roughness * avg_height)
df['normalized_height_var'] = df['skyline_roughness'] * df['height_ratio']

# 5d. Compactness-weighted complexity: S/V / avg_footprint
# Small footprint + high S/V = very fragmented environment
df['fragmentation_index'] = df['avg_surface_volume_ratio'] / (df['avg_footprint'] + 1)

new_metrics = {
    'height_weighted_sv': 'Height-Weighted S/V',
    'effective_scatter_complexity': 'Effective Scatter Complexity',
    'normalized_height_var': 'Normalized Height Variance',
    'fragmentation_index': 'Fragmentation Index',
}

print(f"\n{'New Metric':>30s} | {'Channel':>18s} | {'r':>7s} | {'p-value':>10s} | Sign OK?")
print("-" * 85)
for mk, ml in new_metrics.items():
    for ci, cc in enumerate(ch_cols):
        r, p = stats.pearsonr(df[mk], df[cc])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        
        # Check if sign is physically correct
        # Higher complexity should → lower LOS, higher pathloss, higher delay spread, lower K-factor
        expected_sign = {
            'los_prob': '-', 'avg_pathloss': '+', 'rms_delay_spread': '+',
            'k_factor_db': '-', 'avg_num_paths': '+', 'angular_spread_az': '+'
        }
        actual_sign = '+' if r > 0 else '-'
        sign_ok = '✓' if actual_sign == expected_sign[cc] else '✗'
        
        print(f"{ml:>30s} | {ch_labels[ci]:>18s} | {r:+.3f}{sig:4s}| {p:10.4f}   | {sign_ok}")
    print("-" * 85)

# ---------------------------------------------------------------------------
# 6. Summary: Best metric by sign-correctness AND strength
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL COMPARISON: ALL COMPLEXITY METRICS")
print("=" * 90)

all_complexity = {
    'avg_surface_volume_ratio': 'S/V Ratio (original)',
    'height_weighted_sv': 'Height-Weighted S/V',
    'effective_scatter_complexity': 'Effective Scatter Cmplx',
    'normalized_height_var': 'Norm Height Variance',
    'fragmentation_index': 'Fragmentation Index',
}

expected_signs = {
    'los_prob': -1, 'avg_pathloss': +1, 'rms_delay_spread': +1,
    'k_factor_db': -1, 'avg_num_paths': +1, 'angular_spread_az': +1
}

print(f"\n{'Metric':>28s} | Avg|r| | Sign-correct | Best channel pair")
print("-" * 80)
for mk, ml in all_complexity.items():
    rs = []
    sign_correct = 0
    best_r = 0
    best_ch = ''
    for ci, cc in enumerate(ch_cols):
        r, p = stats.pearsonr(df[mk], df[cc])
        rs.append(abs(r))
        if np.sign(r) == expected_signs[cc]:
            sign_correct += 1
        if abs(r) > abs(best_r):
            best_r = r
            best_ch = ch_labels[ci]
    avg_r = np.mean(rs)
    print(f"{ml:>28s} | {avg_r:.3f}  | {sign_correct}/{len(ch_cols)}         | {best_ch}: r={best_r:+.3f}")

print("\nSaved analysis plots. Done!")
