import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm

def main():
    configs = [
        "geo_noise_5m",
        "mat_all_concrete",
        "rt_5k_rays",
        "hw_dipole",
        "hw_tr38901",
        "hw_4x4_array"
    ]
    baseline = dm.load("baseline")
    print(f"Baseline shape: {baseline.channels.shape}")
    for c in configs:
        try:
            dg = dm.load(c)
            if baseline.channels.shape != dg.channels.shape:
                print(f"ANOMALY: {c} shape {dg.channels.shape} != {baseline.channels.shape}")
            else:
                print(f"{c} shape Matches: {dg.channels.shape}")
        except Exception as e:
            print(f"Failed to load {c}: {e}")

if __name__ == "__main__":
    main()
