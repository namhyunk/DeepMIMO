import deepmimo as dm
import numpy as np

if __name__ == "__main__":
    dataset = dm.load('city_0_newyork_3p5', max_paths=1)
    tx_pos_arr = np.array(dataset.tx_pos).reshape(-1, 3)
    rx_pos = np.array(dataset.rx_pos).reshape(-1, 3)
    print("tx z:", tx_pos_arr[0][2] if len(tx_pos_arr)>0 else None)
    print("rx z:", rx_pos[0][2] if len(rx_pos)>0 else None)
    print("frequency:", getattr(dataset, 'frequency', None))
    # Or in dataset.params?
    print("params:", getattr(dataset, 'params', "No params"))
