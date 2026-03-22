
import pandas as pd
df = pd.read_csv('extended_experiment_results.csv')
df = df[df['max_dist'] == 300]
cols = ['scenario', 'local_building_density', 'k_factor_db', 'rms_delay_spread', 'avg_building_height', 'los_prob', 'avg_pathloss']
df_subset = df[cols].sort_values('local_building_density', ascending=False)
print('===== HIGH DENSITY CITIES =====')
print(df_subset.head(10).to_string(index=False))
print('\n===== LOW DENSITY CITIES =====')
print(df_subset.tail(10).to_string(index=False))

