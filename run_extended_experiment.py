"""Extended Correlation Experiment for DeepMIMO Datasets.

Extracts comprehensive geometry and channel features from 10 city scenarios,
saves results to CSV, and generates correlation plots for ALL combinations.

Geometry Features (9): density, complexity, num_buildings, avg/max/std height,
                       avg_footprint, min_building_dist, height_ratio
Channel Features (10): los_prob, avg_pathloss, rms_delay_spread, k_factor,
                       avg_num_paths, angular_spread_az, std_pathloss,
                       std_delay_spread, reflection_ratio, diffraction_ratio
"""

import os
import deepmimo as dm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)
os.makedirs("files", exist_ok=True)

# ============================================================================
# Feature Definitions (for plotting)
# ============================================================================

GEO_FEATURES = [
    ("solid_angle_blockage",    "Solid Angle Blockage"),
    ("bs_relative_height_penalty","BS-Relative Height Penalty"),
    ("dist_weighted_surf",      "Distance-Weighted Surf Area"),
    ("skyline_roughness",       "Skyline Roughness"),
    ("avg_surface_volume_ratio","Surface/Volume Ratio"),
    ("num_buildings",           "Number of Buildings"),
    ("avg_building_height",     "Avg Building Height (m)"),
    ("max_building_height",     "Max Building Height (m)"),
    ("avg_footprint",           "Avg Footprint Area (m²)"),
    ("min_building_dist",       "Min Building Dist (m)"),
    ("height_ratio",            "Height Ratio (Bldg/BS)"),
]

CHANNEL_FEATURES = [
    ("los_prob",            "LoS Probability",         lambda y: y),
    ("avg_pathloss",        "Avg Pathloss (dB)",       lambda y: y),
    ("rms_delay_spread",    "RMS Delay Spread (ns)",   lambda y: y * 1e9),
    ("k_factor_db",         "K-Factor (dB)",           lambda y: y),
    ("avg_num_paths",       "Avg Num Paths",           lambda y: y),
    ("angular_spread_az",   "Angular Spread Az (°)",   lambda y: y),
    ("std_pathloss",        "Std Pathloss (dB)",       lambda y: y),
    ("std_delay_spread",    "Std Delay Spread (ns)",   lambda y: y * 1e9),
    ("reflection_ratio",    "Reflection Ratio",        lambda y: y),
    ("diffraction_ratio",   "Diffraction Ratio",       lambda y: y),
]

# ============================================================================
# Feature Extraction
# ============================================================================

def get_extended_binned_results(dataset, scen_name):
    """Extract all geometry and channel features per cumulative spatial bin."""
    
    # --- Setup ---
    tx_pos_arr = np.array(dataset.tx_pos).reshape(-1, 3)
    bs_pos = tx_pos_arr[0] if len(tx_pos_arr) > 0 else np.array([0, 0, 0])
    bs_height = bs_pos[2]
    
    rx_pos = np.array(dataset.rx_pos).reshape(-1, 3)
    los = np.array(dataset.los).flatten()
    
    # Pathloss
    pathloss = np.array(dataset.compute_pathloss()).flatten()
    
    # Per-path power and delay
    n_paths = np.array(dataset.power).shape[-1]
    powers_linear = (10 ** (np.array(dataset.power) / 10)).reshape(-1, n_paths)
    powers_linear[np.isnan(powers_linear)] = 0
    delays = np.array(dataset.delay).reshape(-1, n_paths)
    delays[np.isnan(delays)] = 0
    total_power = np.sum(powers_linear, axis=1)
    
    # RMS delay spread
    mean_delay = np.sum(powers_linear * delays, axis=1) / (total_power + 1e-12)
    delay_var = np.sum(powers_linear * (delays - mean_delay[:, None])**2, axis=1) / (total_power + 1e-12)
    rms_delay_spread_arr = np.sqrt(delay_var)
    
    # AoA azimuth angles (degrees)
    aoa_az = np.array(dataset.aoa_az).reshape(-1, n_paths)
    
    # Interaction codes
    inter_raw = np.array(dataset.inter).reshape(-1, n_paths)
    
    # Number of valid paths per user
    num_paths_arr = np.array(dataset.num_paths).flatten()
    
    # Frequency
    freq_str = scen_name.split('_')[-1]
    freq_GHz = float(freq_str.replace('p', '.'))
    
    # --- Building Data ---
    scene = dataset.scene
    buildings = scene.get_objects(label="buildings")
    
    building_data = []
    for b in buildings:
        try:
            vol = b.volume
            footprint = b.footprint_area
            if vol > 0 and footprint > 0:
                dist_to_bs = np.linalg.norm(b.position[:2] - bs_pos[:2])
                building_data.append({
                    "dist": dist_to_bs,
                    "footprint": footprint,
                    "volume": vol,
                    "surf_area": b.hull_surface_area,
                    "height": b.height if hasattr(b, 'height') else vol / footprint,
                })
        except Exception:
            pass
    
    # --- Per-bin Processing ---
    bin_edges = np.arange(0, 350, 50)
    results = []
    
    for i in range(len(bin_edges) - 1):
        max_dist = bin_edges[i + 1]
        
        # ===================== GEOMETRY FEATURES =====================
        bin_b = [b for b in building_data if b["dist"] < max_dist]
        if not bin_b:
            continue
            
        circle_area = np.pi * (max_dist ** 2)

        # 1. Effective Solid Angle Blockage
        solid_angle_blockage = sum((np.sqrt(b["footprint"]) * b["height"]) / (b["dist"]**2 + 1) for b in bin_b)
        
        # 2. BS-Relative Height Penalty (volume of buildings above BS height)
        bs_relative_height_penalty = sum(max(0, b["height"] - bs_height) * b["footprint"] for b in bin_b) / circle_area
        
        # 3. Distance-Weighted Surface Area for Scattering
        dist_weighted_surf = sum(b["surf_area"] / (b["dist"] + 1) for b in bin_b)
        
        # 4. Skyline Roughness
        heights = [b["height"] for b in bin_b]
        mean_h = np.mean(heights)
        std_h = np.std(heights) if len(heights) > 1 else 0
        skyline_roughness = std_h / mean_h if mean_h > 0 else 0
        
        # Keep basic complexity (surf/vol) for comparison
        total_surf = sum(b["surf_area"] for b in bin_b)
        total_vol = sum(b["volume"] for b in bin_b)
        complexity = total_surf / total_vol if total_vol > 0 else 0
        
        # Keep basic counts
        n_buildings = len(bin_b)
        avg_h = mean_h
        max_h = np.max(heights)
        avg_fp = np.mean([b["footprint"] for b in bin_b])
        min_b_dist = min(b["dist"] for b in bin_b)
        h_ratio = avg_h / bs_height if bs_height > 0 else 0
        
        # ===================== CHANNEL FEATURES =====================
        ue_dists = np.linalg.norm(rx_pos[:, :2] - bs_pos[:2], axis=1)
        ue_mask = (ue_dists < max_dist) & (total_power > 0)
        n_ues = np.sum(ue_mask)
        if n_ues < 10:
            continue
        
        # 1. LoS probability
        local_los = los[ue_mask]
        los_prob = np.sum(local_los == 1) / n_ues
        
        # 2. Avg pathloss
        local_pl = pathloss[ue_mask]
        valid_pl = local_pl[~np.isnan(local_pl) & ~np.isinf(local_pl)]
        avg_pl = np.mean(valid_pl) if len(valid_pl) > 0 else np.nan
        
        # 3. RMS delay spread (mean across users)
        local_ds = rms_delay_spread_arr[ue_mask]
        avg_ds = np.mean(local_ds)
        
        # 4. K-factor (dB) — ratio of LoS path power to NLoS power
        local_pwr = powers_linear[ue_mask]
        los_users = local_los == 1
        if np.sum(los_users) > 5:
            los_pwr = local_pwr[los_users]
            p_los = los_pwr[:, 0]
            p_nlos = np.sum(los_pwr[:, 1:], axis=1)
            k_linear = p_los / (p_nlos + 1e-15)
            k_factor_db = float(10 * np.log10(np.mean(k_linear) + 1e-15))
        else:
            k_factor_db = np.nan
        
        # 5. Avg number of valid paths
        local_npaths = num_paths_arr[ue_mask]
        avg_npaths = np.mean(local_npaths)
        
        # 6. Angular spread (AoA azimuth, power-weighted RMS)
        local_aoa = aoa_az[ue_mask]
        local_aoa_pwr = powers_linear[ue_mask]
        aoa_total_pwr = np.sum(local_aoa_pwr, axis=1, keepdims=True) + 1e-15
        # Replace NaN angles with 0 weight
        aoa_valid = ~np.isnan(local_aoa)
        local_aoa_clean = np.where(aoa_valid, local_aoa, 0)
        local_aoa_pwr_clean = np.where(aoa_valid, local_aoa_pwr, 0)
        aoa_total_pwr_clean = np.sum(local_aoa_pwr_clean, axis=1, keepdims=True) + 1e-15
        mean_aoa = np.sum(local_aoa_pwr_clean * local_aoa_clean, axis=1, keepdims=True) / aoa_total_pwr_clean
        aoa_var = np.sum(local_aoa_pwr_clean * (local_aoa_clean - mean_aoa) ** 2, axis=1) / aoa_total_pwr_clean.squeeze()
        angular_spread = float(np.sqrt(np.nanmean(aoa_var)))
        
        # 7. Std pathloss
        std_pl = float(np.std(valid_pl)) if len(valid_pl) > 1 else 0
        
        # 8. Std delay spread
        std_ds = float(np.std(local_ds)) if len(local_ds) > 1 else 0
        
        # 9-10. Interaction type ratios (reflection / diffraction)
        local_inter = inter_raw[ue_mask]
        flat_codes = local_inter.flatten()
        valid_codes_mask = ~np.isnan(flat_codes) & (flat_codes >= 0)
        valid_codes = flat_codes[valid_codes_mask].astype(int)
        n_total_paths = len(valid_codes)
        
        if n_total_paths > 0:
            # Vectorized digit check
            non_los = valid_codes[valid_codes > 0]
            has_refl = np.zeros(len(non_los), dtype=bool)
            has_diffr = np.zeros(len(non_los), dtype=bool)
            temp = non_los.copy()
            while np.any(temp > 0):
                digit = temp % 10
                has_refl |= (digit == 1)
                has_diffr |= (digit == 2)
                temp //= 10
            refl_ratio = float(np.sum(has_refl)) / n_total_paths
            diffr_ratio = float(np.sum(has_diffr)) / n_total_paths
        else:
            refl_ratio = 0
            diffr_ratio = 0
        
        results.append({
            "scenario": scen_name,
            "max_dist": max_dist,
            "freq_GHz": freq_GHz,
            "bs_height": bs_height,
            # Geometry
            # Geometry
            "solid_angle_blockage": solid_angle_blockage,
            "bs_relative_height_penalty": bs_relative_height_penalty,
            "dist_weighted_surf": dist_weighted_surf,
            "skyline_roughness": skyline_roughness,
            "avg_surface_volume_ratio": complexity,
            "num_buildings": n_buildings,
            "avg_building_height": avg_h,
            "max_building_height": max_h,
            "avg_footprint": avg_fp,
            "min_building_dist": min_b_dist,
            "height_ratio": h_ratio,
            # Channel
            "los_prob": los_prob,
            "avg_pathloss": avg_pl,
            "rms_delay_spread": avg_ds,
            "k_factor_db": k_factor_db,
            "avg_num_paths": avg_npaths,
            "angular_spread_az": angular_spread,
            "std_pathloss": std_pl,
            "std_delay_spread": std_ds,
            "reflection_ratio": refl_ratio,
            "diffraction_ratio": diffr_ratio,
        })
    
    return results

# ============================================================================
# Plotting
# ============================================================================

def plot_correlation_group(df, geo_group, channel_features, filename):
    """Plot a group of geometry features vs all channel features.
    
    Args:
        df: DataFrame with all features
        geo_group: list of (col_name, label) for geometry features (rows)
        channel_features: list of (col_name, label, transform) (columns)
        filename: output PNG filename
    """
    n_rows = len(geo_group)
    n_cols = len(channel_features)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    
    unique_dists = sorted(df['max_dist'].unique())
    palette = sns.color_palette("viridis", n_colors=len(unique_dists))
    
    for row, (geo_col, geo_label) in enumerate(geo_group):
        for col, (ch_col, ch_label, ch_transform) in enumerate(channel_features):
            ax = axes[row, col]
            
            # Drop NaN/inf for this pair
            plot_df = df[[geo_col, ch_col, 'max_dist']].replace([np.inf, -np.inf], np.nan).dropna()
            if plot_df.empty:
                ax.set_title(f"{geo_label.split('(')[0].strip()}\nvs {ch_label.split('(')[0].strip()}")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue
            
            y_data = ch_transform(plot_df[ch_col])
            
            # Scatter
            sns.scatterplot(
                x=plot_df[geo_col], y=y_data, hue=plot_df['max_dist'],
                palette=palette, ax=ax, alpha=0.7, legend=(row == 0 and col == n_cols - 1)
            )
            
            # Regression per distance
            for dist in unique_dists:
                mask = plot_df['max_dist'] == dist
                subset_x = plot_df.loc[mask, geo_col]
                subset_y = ch_transform(plot_df.loc[mask, ch_col])
                if len(subset_x) > 1:
                    sns.regplot(
                        x=subset_x, y=subset_y,
                        scatter=False, ax=ax,
                        color=palette[unique_dists.index(dist)]
                    )
            
            ax.set_xlabel(geo_label)
            ax.set_ylabel(ch_label)
            ax.set_title(f"{geo_label.split('(')[0].strip()} vs {ch_label.split('(')[0].strip()}")
            
            # Handle legend
            if row == 0 and col == n_cols - 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                fig.legend(
                    handles, [f"{int(float(l))}m Radius" for l in labels],
                    title="Cumulative Radius", loc='center right',
                    bbox_to_anchor=(0.99, 0.5)
                )
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
    
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(f"figures/{filename}", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figures/{filename}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    all_results = []
    import deepmimo as dm
    available = dm.get_available_scenarios()
    # Filter for standard 'city_X_name_3p5'
    cities_3p5 = [c for c in available if c.startswith('city_') and c.endswith('_3p5')]
    cities_3p5 = sorted(cities_3p5, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999)
    scenarios = cities_3p5
    
    success_count = 0
    target_count = 50
    
    for scen in scenarios:
        if success_count >= target_count:
            break
        print(f"\n--- Processing {scen} ---")
        try:
            dataset = dm.load(scen, max_paths=10)
            
            # Force max_scattering to 1 if it exists
            if hasattr(dataset, 'rt_params'):
                dataset.rt_params.max_scattering = 1
                print(f"Scattering explicitly verified (max_scattering={dataset.rt_params.max_scattering})")
                
            dataset.compute_channels()
            binned = get_extended_binned_results(dataset, scen)
            if len(binned) > 0:
                all_results.extend(binned)
                success_count += 1
                print(f"Extracted {len(binned)} spatial bins. (Total successful cities: {success_count}/{target_count})")
            else:
                print("No valid spatial bins extracted. Skipping.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    print(f"\nSuccessfully processed {success_count} scenarios.")
    
    df = pd.DataFrame(all_results)
    df = df.dropna(subset=[g[0] for g in GEO_FEATURES] + [c[0] for c in CHANNEL_FEATURES], how='all')
    
    # Save CSV
    csv_path = "files/extended_experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n--- Saved {len(df)} samples to {csv_path} ---")
    print(df.head())
    
    if df.empty:
        print("No data to plot.")
        exit()
    
    # --- Generate Plots: 3 figures, each 3 geometry rows × 10 channel cols ---
    
    # Group 1: Density & Structure
    plot_correlation_group(
        df,
        geo_group=[GEO_FEATURES[0], GEO_FEATURES[1], GEO_FEATURES[2]],
        channel_features=CHANNEL_FEATURES,
        filename="correlation_plots_density_structure.png"
    )
    
    # Group 2: Height features
    plot_correlation_group(
        df,
        geo_group=[GEO_FEATURES[3], GEO_FEATURES[4], GEO_FEATURES[5]],
        channel_features=CHANNEL_FEATURES,
        filename="correlation_plots_height.png"
    )
    
    # Group 3: Spatial features
    plot_correlation_group(
        df,
        geo_group=[GEO_FEATURES[6], GEO_FEATURES[7], GEO_FEATURES[8]],
        channel_features=CHANNEL_FEATURES,
        filename="correlation_plots_spatial.png"
    )
    
    print("\n--- All plots saved ---")
