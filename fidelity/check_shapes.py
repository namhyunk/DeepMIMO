import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm
from quantify_fidelity import compute_hardware_metrics

def main():
    names = [
        "baseline",
        "geo_noise_5m",
        "rt_1k_rays",
        "mat_all_concrete",
        "hw_4x4_array",
        "hw_dipole", 
        "hw_tr38901"
    ]
    for n in names:
        try:
            ds = dm.load(n)
            print(f"{n}: channels shape {ds.channels.shape}")
        except Exception as e:
            print(f"Failed to load {n}: {e}")

if __name__ == "__main__":
    main()
