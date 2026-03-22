import os
import deepmimo as dm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("files", exist_ok=True)

def get_binned_results(dataset, scen_name):
    # Flatten shapes handling MacroDataset arrays (e.g., shape=(n_BS, n_users, n_paths))
    tx_pos_arr = np.array(dataset.tx_pos).reshape(-1, 3)
    bs_pos = tx_pos_arr[0] if len(tx_pos_arr) > 0 else np.array([0, 0, 0])
    
    rx_pos = np.array(dataset.rx_pos).reshape(-1, 3)
    los = np.array(dataset.los).flatten()
    
    # Compute pathloss and RMS delay spread arrays upfront
    pathloss = np.array(dataset.compute_pathloss()).flatten()
    
    # Delay spread logic
    powers_linear = (10 ** (np.array(dataset.power) / 10)).reshape(-1, np.array(dataset.power).shape[-1])
    powers_linear[np.isnan(powers_linear)] = 0
    delays = np.array(dataset.delay).reshape(-1, np.array(dataset.delay).shape[-1])
    delays[np.isnan(delays)] = 0
    total_power = np.sum(powers_linear, axis=1)
    mean_delay = np.sum(powers_linear * delays, axis=1) / (total_power + 1e-12)
    delay_variance = np.sum(powers_linear * (delays - mean_delay[:, None])**2, axis=1) / (total_power + 1e-12)
    rms_delay_spread_arr = np.sqrt(delay_variance)
    
    # Pre-calculate building metrics (distances, footprint, volume, area)
    scene = dataset.scene
    buildings = scene.get_objects(label="buildings")
    
    building_data = []
    for b in buildings:
        try:
            vol = b.volume
            if vol > 0:
                dist_to_bs = np.linalg.norm(b.position[:2] - bs_pos[:2])
                building_data.append({
                    "dist": dist_to_bs,
                    "footprint": b.footprint_area,
                    "volume": vol,
                    "surf_area": b.hull_surface_area
                })
        except Exception:
            pass

    # Process by Bins
    bin_edges = np.arange(0, 350, 50) # 0, 50, 100, 150, 200, 250, 300
    results = []
    
    for i in range(len(bin_edges)-1):
        min_dist = bin_edges[i]
        max_dist = bin_edges[i+1]
        
        # 1. Spatial Bin Geometry Metrics
        # Change to CUMULATIVE distance: 0 to max_dist
        bin_b_data = [b for b in building_data if b["dist"] < max_dist]
        
        if not bin_b_data:
            # Skip empty bins
            continue
            
        bin_total_surf = sum(b["surf_area"] for b in bin_b_data)
        bin_total_vol = sum(b["volume"] for b in bin_b_data)
        avg_surf_vol_ratio = bin_total_surf / bin_total_vol if bin_total_vol > 0 else 0
        
        bin_total_footprint = sum(b["footprint"] for b in bin_b_data)
        # Calculate full circle area
        circle_area = np.pi * (max_dist**2)
        bin_density = bin_total_footprint / circle_area
        
        # 2. Local Channel Metrics
        # Get distances of all receivers from BS (cumulative: 0 to max_dist)
        ue_dists = np.linalg.norm(rx_pos[:, :2] - bs_pos[:2], axis=1)
        ue_in_bin_mask = (ue_dists < max_dist) & (total_power > 0)
        
        num_valid_ues = np.sum(ue_in_bin_mask)
        if num_valid_ues < 10:
            continue # insufficient statistics
            
        local_los = los[ue_in_bin_mask]
        los_prob = np.sum(local_los == 1) / num_valid_ues
        
        local_pl = pathloss[ue_in_bin_mask]
        valid_local_pl = local_pl[~np.isnan(local_pl) & ~np.isinf(local_pl)]
        avg_pathloss = np.mean(valid_local_pl) if len(valid_local_pl) > 0 else np.nan
        
        local_ds = rms_delay_spread_arr[ue_in_bin_mask]
        avg_ds = np.mean(local_ds)
        
        results.append({
            "scenario": scen_name,
            "distance_bin": f"0-{max_dist}m",
            "max_dist": max_dist,
            "avg_surface_volume_ratio": avg_surf_vol_ratio,
            "local_building_density": bin_density,
            "num_buildings_in_bin": len(bin_b_data),
            "los_prob": los_prob,
            "avg_pathloss": avg_pathloss,
            "rms_delay_spread": avg_ds,
            "num_users": num_valid_ues
        })
        
    return results

if __name__ == "__main__":
    all_results = []
    scenarios_to_test = [
        'city_0_newyork_3p5', 'city_1_losangeles_3p5', 
        'city_2_chicago_3p5', 'city_3_houston_3p5', 
        'city_4_phoenix_3p5', 'city_5_philadelphia_3p5', 
        'city_6_miami_3p5', 'city_7_sandiego_3p5', 
        'city_8_dallas_3p5', 'city_9_sanfrancisco_3p5'
    ]

    for scen in scenarios_to_test:
        print(f"\n--- Processing {scen} ---")
        try:
            # Load the dataset directly (this loads the scene geometry)
            dataset = dm.load(scen, max_paths=10)
            dataset.compute_channels()
            
            binned_results = get_binned_results(dataset, scen)
            all_results.extend(binned_results)
            print(f"Extracted {len(binned_results)} spatial bins.")
            
        except Exception as e:
            print(f"Error processing {scen}: {e}")

    df = pd.DataFrame(all_results)
    df = df.dropna()
    print("\n--- Final Results ---")
    print(df.head())
    
    if not df.empty:
        df.to_csv("files/experiment_results_binned.csv", index=False)
        
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Color palette depending on number of unique distances
        unique_dists = sorted(list(df['max_dist'].unique()))
        palette = sns.color_palette("viridis", n_colors=len(unique_dists))
        
        # Row 1: X = Building Density
        # Density vs Delay Spread
        sns.scatterplot(data=df, x='local_building_density', y=df['rms_delay_spread']*1e9, hue='max_dist', palette=palette, ax=axes[0,0], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='local_building_density', y=subset['rms_delay_spread']*1e9, scatter=False, ax=axes[0,0], color=palette[unique_dists.index(dist)])
        axes[0,0].set_xlabel('Cumulative Building Density')
        axes[0,0].set_ylabel('RMS Delay Spread (ns)')
        axes[0,0].set_title('Density vs Delay Spread')
        axes[0,0].get_legend().remove()
        
        # Density vs LoS
        sns.scatterplot(data=df, x='local_building_density', y='los_prob', hue='max_dist', palette=palette, ax=axes[0,1], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='local_building_density', y='los_prob', scatter=False, ax=axes[0,1], color=palette[unique_dists.index(dist)])
        axes[0,1].set_xlabel('Cumulative Building Density')
        axes[0,1].set_ylabel('Cumulative LoS Probability')
        axes[0,1].set_title('Density vs LoS Prob')
        axes[0,1].get_legend().remove()

        # Density vs Pathloss
        sns.scatterplot(data=df, x='local_building_density', y='avg_pathloss', hue='max_dist', palette=palette, ax=axes[0,2], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='local_building_density', y='avg_pathloss', scatter=False, ax=axes[0,2], color=palette[unique_dists.index(dist)])
        axes[0,2].set_xlabel('Cumulative Building Density')
        axes[0,2].set_ylabel('Cumulative Avg Pathloss (dB)')
        axes[0,2].set_title('Density vs Pathloss')
        
        # Fix legend on row 1 rightmost plot
        handles, labels = axes[0,2].get_legend_handles_labels()
        fig.legend(handles, [f"{int(float(l))}m Radius" for l in labels], title="Cumulative Radius", loc='center right', bbox_to_anchor=(0.98, 0.75))
        axes[0,2].get_legend().remove()

        # Row 2: X = Surface/Volume Ratio (Complexity)
        # Complexity vs Delay Spread
        sns.scatterplot(data=df, x='avg_surface_volume_ratio', y=df['rms_delay_spread']*1e9, hue='max_dist', palette=palette, ax=axes[1,0], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='avg_surface_volume_ratio', y=subset['rms_delay_spread']*1e9, scatter=False, ax=axes[1,0], color=palette[unique_dists.index(dist)])
        axes[1,0].set_xlabel('Cumulative Total Surface / Total Volume')
        axes[1,0].set_ylabel('RMS Delay Spread (ns)')
        axes[1,0].set_title('Complexity vs Delay Spread')
        axes[1,0].get_legend().remove()
        
        # Complexity vs LoS
        sns.scatterplot(data=df, x='avg_surface_volume_ratio', y='los_prob', hue='max_dist', palette=palette, ax=axes[1,1], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='avg_surface_volume_ratio', y='los_prob', scatter=False, ax=axes[1,1], color=palette[unique_dists.index(dist)])
        axes[1,1].set_xlabel('Cumulative Total Surface / Total Volume')
        axes[1,1].set_ylabel('Cumulative LoS Probability')
        axes[1,1].set_title('Complexity vs LoS Prob')
        axes[1,1].get_legend().remove()

        # Complexity vs Pathloss
        sns.scatterplot(data=df, x='avg_surface_volume_ratio', y='avg_pathloss', hue='max_dist', palette=palette, ax=axes[1,2], alpha=0.7)
        for dist in unique_dists:
            subset = df[df['max_dist'] == dist]
            if len(subset) > 1:
                sns.regplot(data=subset, x='avg_surface_volume_ratio', y='avg_pathloss', scatter=False, ax=axes[1,2], color=palette[unique_dists.index(dist)])
        axes[1,2].set_xlabel('Cumulative Total Surface / Total Volume')
        axes[1,2].set_ylabel('Cumulative Avg Pathloss (dB)')
        axes[1,2].set_title('Complexity vs Pathloss')
        axes[1,2].get_legend().remove()
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig('figures/correlation_plots_binned.png', dpi=300)
        print("Distance-controlled plots saved to figures/correlation_plots_binned.png")
