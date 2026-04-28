import sionna
from sionna.rt import load_scene
import sys

try:
    print("Loading Simple Street Canyon...")
    scene1 = load_scene(sionna.rt.scene.simple_street_canyon)
    scene1.add_camera("my_cam", position=[0, -50, 20], look_at=[0, 0, 0])
    scene1.render_to_file(camera="my_cam", filename="simple_canyon.png")
except Exception as e:
    print("Error scene 1:", e)

try:
    print("Loading Munich...")
    scene2 = load_scene(sionna.rt.scene.munich)
    scene2.add_camera("my_cam", position=[0, -1000, 500], look_at=[0, 0, 0])
    scene2.render_to_file(camera="my_cam", filename="munich.png")
    print("Done")
except Exception as e:
    print("Error scene 2:", e)
