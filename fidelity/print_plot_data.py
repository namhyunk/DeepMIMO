import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm
from quantify_fidelity import compute_hardware_metrics, compute_ray_tracing_metrics

def main():
    rt_names = [
        "baseline", "rt_500k_rays", "rt_200k_rays", "rt_low_rays", 
        "rt_50k_rays", "rt_20k_rays", "rt_very_low_rays", "rt_5k_rays", "rt_1k_rays"
    ]
    rt_x = [1_000_000, 500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000, 1_000] 
    
    rt_nmse = [0.0]
    
    for idx, r in enumerate(rt_names[1:]):
        try:
            bl_ds = dm.load("baseline")
            dg_ds = dm.load(r)
            hw = compute_hardware_metrics(bl_ds, dg_ds)
            rt_nmse.append(hw["capacity_ratio"])
        except Exception as e:
            rt_nmse.append(float('nan'))
            print(f"Error loading {r}: {e}")
            
    print("RT X:", rt_x)
    print("RT NMSE:", rt_nmse)

if __name__ == "__main__":
    main()
