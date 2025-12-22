import sys
import os
#import LiDAR_fog_sim

# # Add the current directory to sys.path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Adjust this for dataset
PC_FEATURE_DIMS = 5
MAX_INTENSITY_VAL = True # (True: Range-(0,255), False: Range-(0,1))

input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP"
output_folder = "/home/saksham/samsad/mtech-project/datasets"
fault_sim_path = "/home/saksham/samsad/mtech-project/fault-sim"   # WARNING: Do not change this, this is required to get precomputed integration in fog and dust sim

fog_params = {

    "alpha": 0.06, #available options for alpha = [0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.12, 0.15, 0.2]
    "gamma": 1e-6,
    "simulation_options": dict(
        noise = 10,
        gain = False,
        noise_variant = 'v1',
        hard = True,
        soft = True
    ),
    "input_dir": input_dir,
    "output_dir": output_folder

}

snow_params = {
    "snowfall_rate": 2.5,               #in mm/hr #Available arguments: [0.5, 1.0, 2.0, 2.5, 1.5]
    "terminal_velocity": 1.6,            #in m/s Available arguments: [2.0, 1.6, 2.0, 1.6, 0.6]  # m/s
    "mode": "gunn",
    "input_dir": input_dir,
    "output_dir": output_folder
}

rain_params = {
    "rainfall_rate": 30, # in mm/hr Available range 5-100
    "input_dir": input_dir,
    "output_dir": output_folder
}

dust_params = {
    "mode": "light", #Available options: light, moderate, heavy
    "simulation_options": dict(
        noise = 10,
        gain = False,
        noise_variant = 'v1',
        hard = True,
        soft = True
    ),
    "integration_path": fault_sim_path + "/LiDAR_dust_sim/fog_sim_method/integral_lookup_tables",
    "input_dir": input_dir,
    "output_dir": output_folder
}
