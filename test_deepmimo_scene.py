import deepmimo as dm
import numpy as np

if __name__ == "__main__":
    dataset = dm.load('city_0_newyork_3p5', max_paths=1)
    scene = dataset.scene
    buildings = scene.get_objects(label="buildings")
    print(f"Number of buildings: {len(buildings)}")
    if len(buildings) > 0:
        b = buildings[0]
        print(dir(b))
        print("vol:", b.volume)
        print("footprint:", b.footprint_area)
        print("surf_area:", b.hull_surface_area)
        print("pos:", b.position)
        print("bbox:", b.bounds if hasattr(b, 'bounds') else "no bounds attribute")
