__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import numpy as np

from tqdm import tqdm #type: ignore
from pathlib import Path
from simulation_v2 import augment
from lib.OpenPCDet.pcdet.utils import calibration_kitti
from sampling import compute_occupancy, snowfall_rate_to_rainfall_rate
import os
import time
import json


SPLIT_FOLDER = Path(__file__).parent.parent.parent.resolve() / 'lib' / 'LiDAR_fog_sim' / 'SeeingThroughFog' / 'splits'
#LIDAR_FOLDER = Path.home() / 'datasets' / 'DENSE' / 'SeeingThroughFog' / 'lidar_hdl64_strongest'
LIDAR_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps/LIDAR_TOP')
JSON_PATH = Path ('/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps.json')
SAVE_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes_snow/sweeps_2.0/LIDAR_TOP')
SAVE_FOLDER.mkdir(parents= True, exist_ok= True)

NUSCENES_ROOT = '/home/saksham/samsad/mtech-project/datasets/nuscenes'

#SPLIT = SPLIT_FOLDER / 'train_clear.txt'

# SNOWFALL_RATES = [0.5, 1.0, 2.0, 2.5, 1.5]       # mm/h
# TERMINAL_VELOCITIES = [2.0, 1.6, 2.0, 1.6, 0.6]  # m/s

SNOWFALL_RATES = [2.0]       # mm/h
TERMINAL_VELOCITIES = [1.6]  # m/s

def load_json(json_path, sweep_folder):
    with open(json_path, 'r') as f:
        file_dict = json.load(f)
    
    print(f'Got json with {len(file_dict)} files')
    # Check if the files exist in the sweep folder
    for key in file_dict:
        file_name = file_dict[key]
        file_path = os.path.join(sweep_folder, file_name)
        if not os.path.exists(file_path):
            print(f'file: {file_name} doesnt exist.')
            break
    
    print('Checking existense of file complete.\n')
    return file_dict


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.parent.parent.resolve() / \
                 'lib' / 'OpenPCDet' / 'data' / 'dense' / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def process_one_file():
    pass


if __name__ == '__main__':

    rainfall_rates = []
    occupancy_ratios = []
    np.random.seed(seed = 42)

    for i in range(len(SNOWFALL_RATES)):
        rainfall_rate = snowfall_rate_to_rainfall_rate(SNOWFALL_RATES[i], TERMINAL_VELOCITIES[i])
        occupancy_ratio = compute_occupancy(SNOWFALL_RATES[i], TERMINAL_VELOCITIES[i])
        rainfall_rates.append(rainfall_rate)
        occupancy_ratios.append(occupancy_ratio)
    
    rain_occu_combined = np.column_stack((rainfall_rates, occupancy_ratios))
    calibration = get_calib()
    print('Number of LIDAR point clouds: ',len(os.listdir(LIDAR_FOLDER)))

    # Load the json dict of file_names
    file_dict = load_json(json_path = JSON_PATH, sweep_folder = LIDAR_FOLDER)

    # Beginning 500 samples are running on dgx server, we will run snow simulation
    # for sweeps from sweeps_500 to sweeps_1199 (Total 600 sweeps)
    sweeps_idx = list(range(500, 1200))

    # generate file names
    file_names = []
    existence_flag = True
    for sweep_number in sweeps_idx:
        file_name = file_dict[f"sweep_{sweep_number}"]
        file_names.append(file_name)
        # Also keep on checking if the files exist
        existence_flag = os.path.exists(os.path.join(LIDAR_FOLDER, file_name))
        if not existence_flag:
            print(f'File {file_name} doesnt exist. Abort the process!')
            raise FileNotFoundError ('Could not find the file.')
    
    print('generated file names having length = ', len(file_names))
    
    for mode in ['gunn']:
        max_points = len(file_names)
        #pbar = tqdm(file_names[:max_points], desc = f'{mode} running...')
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(LIDAR_FOLDER, file_name)
            save_path = os.path.join(SAVE_FOLDER, file_name)
            if os.path.exists(save_path):
                print(f'Skipping the point cloud {file_name} as it already exists.')
                continue

            print('Converting the point cloud: ', f'it: [{sweeps_idx[i]}]', file_name)
            points = np.fromfile(file_path, dtype = np.float32).reshape(-1,5)
            #print('Stuck here')
            
            for params in rain_occu_combined:
                print('\t-Runnning with Parameters: ', params) #, end= '\t')
                #print('stuck here')
                start = time.perf_counter()
                rainfall_rate, occupancy_ratio = params
                pc = copy.deepcopy(points)
                snowflakes_file_prefix = f'{mode}_{rainfall_rate}_{occupancy_ratio}'
                augmented_pc = augment(pc = pc, particle_file_prefix= snowflakes_file_prefix,
                                              beam_divergence= float(np.degrees(3e-3)))
                augmented_pc.astype(np.float32).tofile(save_path)
                end = time.perf_counter()
                print(f"\t-Elapsed time: {end - start:.6f} seconds")
    
    print('Snowfalkes Simulated Successfully...')
