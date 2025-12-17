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


SPLIT_FOLDER = Path(__file__).parent.parent.parent.resolve() / 'lib' / 'LiDAR_fog_sim' / 'SeeingThroughFog' / 'splits'
#LIDAR_FOLDER = Path.home() / 'datasets' / 'DENSE' / 'SeeingThroughFog' / 'lidar_hdl64_strongest'
LIDAR_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps/LIDAR_TOP')
SAVE_FOLDER = Path('/home/saksham/samsad/mtech-project/datasets/nuscenes_snow/sweeps_2.0/LIDAR_TOP')
SAVE_FOLDER.mkdir(parents= True, exist_ok= True)


#SPLIT = SPLIT_FOLDER / 'train_clear.txt'

# SNOWFALL_RATES = [0.5, 1.0, 2.0, 2.5, 1.5]       # mm/h
# TERMINAL_VELOCITIES = [2.0, 1.6, 2.0, 1.6, 0.6]  # m/s

SNOWFALL_RATES = [2.0]       # mm/h
TERMINAL_VELOCITIES = [1.6]  # m/s


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
    
    # #max_count = 10
    # for file_name in os.listdir(LIDAR_FOLDER):
    #     file_path = os.path.join(LIDAR_FOLDER, file_name)
    #     print(file_name, "  |   Exists: ", os.path.exists(file_path))

    # Due to time related issues, we will convert only 100 points out of 3531 points for snow simulation
    max_points = 600
    for mode in ['gunn']:
        file_names = list(os.listdir(LIDAR_FOLDER))
        np.random.shuffle(file_names)
        max_points = len(file_names)
        #pbar = tqdm(file_names[:max_points], desc = f'{mode} running...')
        for i, file_name in enumerate(file_names[:max_points]):
            file_path = os.path.join(LIDAR_FOLDER, file_name)
            save_path = os.path.join(SAVE_FOLDER, file_name)
            if os.path.exists(save_path):
                print(f'Skipping the point cloud {file_name} as it already exists.')
                continue

            print('Converting the point cloud: ', f'it: [{i+1}]', file_name)
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