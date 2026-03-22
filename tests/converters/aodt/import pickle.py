import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os

# ==================== Sionna RT imports (new API) ====================
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, BackscatteringPattern, PathSolver, RadioMaterial, subcarrier_frequencies

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow sees the following GPUs:")
    for gpu in gpus:
        print(gpu.name)
else:
    print("No GPUs detected.")

class DataLoader:
    def __init__(self, data, batch_size):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.data[batch_indices]

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    
    scenarios = ['city_0_newyork_3p5', 'city_1_losangeles_3p5', 'city_2_chicago_3p5', 'city_3_houston_3p5', 'city_4_phoenix_3p5', 'city_5_philadelphia_3p5', 'city_6_miami_3p5', 'city_7_sandiego_3p5', 'city_8_dallas_3p5', 'city_9_sanfrancisco_3p5', 'city_10_austin_3p5']
    # scenarios = ['city_5_philadelphia_3p5', 'city_6_miami_3p5', 'city_7_sandiego_3p5']
    
    # ------------------- 1. Define Scattering Patterns -------------------
    # Assuming BackscatteringPattern is a custom class you defined or 
    # a specific Sionna pattern function. 
    # (Ensure lambda_ is cast to a float or tensor as needed)

    concrete_params = {
    "scattering_coefficient": 0.25,
    "xpd_coefficient": 0.4,
    }

    concrete_pattern = BackscatteringPattern(alpha_r=6, alpha_i=6, lambda_=0.5)
    # asphalt_pattern  = BackscatteringPattern(alpha_r=4, alpha_i=4, lambda_=0.75)
    # floor_pattern    = BackscatteringPattern(alpha_r=4, alpha_i=4, lambda_=0.75)

    asphalt_material = RadioMaterial("asphalt_material",
                                thickness=1.0,
                                relative_permittivity=5.72,
                                conductivity=0.0005,
                                scattering_coefficient=0.4,
                                xpd_coefficient=0.4,
                                scattering_pattern="lambertian") # custom material only supports lambertian pattern

    floor_material = RadioMaterial("floor_material",
                                thickness=1.0,
                                relative_permittivity=18.175819,
                                conductivity=0.7645,
                                scattering_coefficient=0.4,
                                xpd_coefficient=0.4,
                                scattering_pattern="lambertian") # custom material only supports lambertian pattern

    # Process each scenario
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Processing scenario: {scenario}")
        print(f"{'='*60}\n")
        
        scene_path = os.path.join(root_path, scenario, f'{scenario}.xml')

        # Load scene
        scene = load_scene(scene_path)
        scene.frequency = 3.5e9

        for obj in scene.objects.values():
            if obj.radio_material.name == 'itu_concrete':
                obj.radio_material.thickness = 1.0
                obj.radio_material.scattering_coefficient = concrete_params["scattering_coefficient"]
                obj.radio_material.xpd_coefficient = concrete_params["xpd_coefficient"]
                obj.radio_material.scattering_pattern = concrete_pattern
            elif obj.radio_material.name == 'asphalt':
                obj.radio_material = asphalt_material
            elif obj.radio_material.name == 'floor':
                obj.radio_material = floor_material

        scene.remove("floor")
        scene.remove("asphalt")

        # ------------------- Antenna arrays (SISO, isotropic) -------------------
        scene.tx_array = PlanarArray(num_rows=1, num_cols=32,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")

        # ------------------- Transmitter (fixed) -------------------
        tx_position = np.load(os.path.join(root_path, scenario, 'wireless insite/tx_location.npy'))
        tx = Transmitter(name="tx", position=tx_position.flatten().tolist())
        scene.add(tx)

        # ------------------- Receiver positions -------------------
        rx_position = np.load(os.path.join(root_path, scenario, 'wireless insite/rx_location.npy'))

        user_grid = rx_position
        indices = np.arange(user_grid.shape[0])
        batch_size = 32  # Reduced from 128 to avoid tensor size limit
        data_loader = DataLoader(indices, batch_size)

        # ------------------- PathSolver (new, fast, supports all interactions) -------------------
        path_solver = PathSolver()
        
        for max_depth in range(1, 5):
            print(f"Running for max depth {max_depth}...")
            dict_list = []
            for batch_idx, batch in tqdm(enumerate(data_loader), desc=f"Generating Batches - {scenario}", unit="batch"):
                for i in batch:
                    # Create receiver
                    rx = Receiver(name=f"rx_{i}", position=user_grid[i].flatten().tolist())
                    scene.add(rx)

                # Compute paths (includes reflection + scattering + diffraction)
                paths = path_solver(scene, 
                    max_depth=max_depth,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=True,
                    refraction=True,
                    diffraction=True,
                )
                paths.normalize_delays = False

                # OFDM system parameters
                num_subcarriers = 32
                subcarrier_spacing=30e3

                # Compute frequencies of subcarriers relative to the carrier frequency
                frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

                # Compute channel frequency response
                h_freq = paths.cfr(frequencies=frequencies,
                                normalize=False, # Normalize energy
                                normalize_delays=False,
                                out_type="numpy")

                # a, tau = paths.cir(out_type="numpy")

                # valid = paths.valid # original mask in version 0.19.2

                # doa_phi = np.array(paths.phi_r)
                # doa_theta = np.array(paths.theta_r)
                # dod_phi = np.array(paths.phi_t)
                # dod_theta = np.array(paths.theta_t)

                # power = np.abs(a)**2
                # phase = np.angle(a, deg=True)


                # when running on deepmimo scenario, don't save everything into dict. 
                # too many user and the data is too large makes the server crash.
                dict_item = {
                    # 'a': a,
                    # 'valid': valid,
                    # 'phase': phase,
                    # 'delay': tau,
                    # 'power': power,
                    # 'doa_phi': doa_phi,
                    # 'doa_theta': doa_theta,
                    # 'dod_phi': dod_phi,
                    # 'dod_theta': dod_theta,
                    # 'Tx_loc': tx_position,
                    # 'Rx_locs': user_grid[batch],
                    'CFR': h_freq,
                }
                dict_list.append(dict_item)

                for i in batch:
                    # Remove receiver
                    scene.remove(f"rx_{i}")

            
            
            # Concatenate all data from dict_list along the first dimension (users)
            concatenated_data = {}
            for key in dict_list[0].keys():
                arrays = [item[key] for item in dict_list]
                concatenated_data[key] = np.concatenate(arrays, axis=0)
            
            # Save the results
            output_path = os.path.join(root_path, scenario, f'raytracing_results_{scenario}_maxdepth{max_depth}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(concatenated_data, f)

            print(f"Done with max depth {max_depth}! Data saved to {output_path} with {concatenated_data['CFR'].shape[0]} users")
        
        print(f"\nCompleted processing scenario: {scenario}\n")