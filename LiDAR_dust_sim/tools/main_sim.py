__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import numpy as np

from tqdm import tqdm #type: ignore
from pathlib import Path
from simulation import augment
#from sampling import compute_occupancy, snowfall_rate_to_rainfall_rate
import os
import time
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map 


#JSON_PATH = Path ('/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps.json')

# SNOWFALL_RATES = [0.5, 1.0, 2.0, 2.5, 1.5]       # mm/h
# TERMINAL_VELOCITIES = [2.0, 1.6, 2.0, 1.6, 0.6]  # m/s


def process_one_file(file_name, main_folder, save_folder):
    
    file_path = os.path.join(main_folder, file_name)
    save_path = os.path.join(save_folder, file_name)
    if os.path.exists(save_path):
        #print(f'Skipping the point cloud {file_name} as it already exists.')
        return

    points = np.fromfile(file_path, dtype = np.float32).reshape(-1,5)
    pc = copy.deepcopy(points)
    
    augmented_pc = augment(pc = pc, beam_divergence= float(np.degrees(3e-3)))
    augmented_pc.astype(np.float32).tofile(save_path)


if __name__ == '__main__':

    # Set parameters
    SNOWFALL_RATE = 1.5
    TERMINAL_VELOCITY = 1.6
    SHOW_PBAR = True
    samples_or_sweeps = 'samples'

    # Make dirs
    MAIN_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes')
    SAVE_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes_dust')
    MAIN_FOLDER = MAIN_FOLDER / samples_or_sweeps / 'LIDAR_TOP'
    SAVE_FOLDER = SAVE_FOLDER / f'{samples_or_sweeps}_{SNOWFALL_RATE}' / 'LIDAR_TOP'
    SAVE_FOLDER.mkdir(parents= True, exist_ok= True)

    file_names = os.listdir(MAIN_FOLDER)[:10]

    partial_func = partial(process_one_file, main_folder = MAIN_FOLDER, save_folder=SAVE_FOLDER)

    for file_name in file_names:
        print("Processing: ", file_name)
        partial_func(file_name = file_name)

