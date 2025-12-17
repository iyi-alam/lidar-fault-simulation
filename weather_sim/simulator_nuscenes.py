"""
This file is meant to simulate faults in all the files of an input folder and save the simulated output in an output folder
"""
#%% Main
import os
import sys
# Add the parent directory (fault-sim) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
from tqdm import tqdm #type:ignore
from functools import partial
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm.contrib.concurrent import process_map  #type:ignore
from nuscenes.nuscenes import NuScenes

import config_params
from visualization_utils import plot_bev, plot_with_anns
import LiDAR_fog_sim.fog_simulation as fogsim
from LiDAR_snow_sim.tools.snowfall import sampling, simulation
from LiDAR_rain_sim import rain_sim
from LiDAR_dust_sim.fog_sim_method import mie_scatt, precompute_integration, dust_params
from pyquaternion import Quaternion
import argparse


class BaseSim:
    def __init__(self):
        self.save = False

    def simulate(self, pc: np.ndarray):
        return pc


class FogSim(BaseSim):
    
    def __init__(self, input_dir, output_dir, 
                 paramset: fogsim.ParameterSet,
                 simulation_options, save = True):
        self.input_dir = input_dir 
        self.output_dir = output_dir
        self.pset = paramset
        self.simulation_options = simulation_options
        self.save = save

    def simulate(self, pc: np.ndarray):
        foggified_pc, _, _ = fogsim.simulate_fog(self.pset, pc, **self.simulation_options)
        return foggified_pc


class SnowSim(BaseSim):
    def __init__(self, input_dir, output_dir, rainfall_rate, 
                     occupancy_ratio, mode, save = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rainfall_rate = rainfall_rate
        self.occupancy_ratio = occupancy_ratio
        self.mode = mode
        self.save = save
    
    def simulate(self, pc: np.ndarray):
        pc = deepcopy(pc)
        snowflakes_file_prefix = f'{self.mode}_{self.rainfall_rate}_{self.occupancy_ratio}'
        augmented_pc = simulation.augment(pc = pc, particle_file_prefix= snowflakes_file_prefix,
                                        beam_divergence= float(np.degrees(3e-3)))
        
        return augmented_pc


class RainSim(BaseSim):
    def __init__(self, input_dir, output_dir, driver: rain_sim.rainSim, save = True):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.driver = driver
        self.save = save

    def simulate(self, pc: np.ndarray):
        pc = deepcopy(pc)
        rain_pc, labels, stats_arr = self.driver.simulate(pc)
        return rain_pc

class DustSim(BaseSim):
    
    def __init__(self, input_dir, output_dir, 
                 paramset: fogsim.ParameterSet,
                 simulation_options, save = True):
        self.input_dir = input_dir 
        self.output_dir = output_dir
        self.pset = paramset
        self.simulation_options = simulation_options
        self.save = save

    def simulate(self, pc: np.ndarray):
        foggified_pc, _, _ = fogsim.simulate_fog(self.pset, pc, **self.simulation_options)
        return foggified_pc


class NuscenesSimulator:
    def __init__(self, dataroot,  simulator: BaseSim, 
                 save_path = None,dataversion="v1.0-mini",
                 show_pbar = True):
        
        self.dataroot = dataroot
        self.nusc = NuScenes(version=dataversion, dataroot=dataroot, verbose=True)
        self.simulator = simulator
        self.save_path = save_path
        self.show_pbar = show_pbar

    def process_one_sample(self, sample):
        data_token = sample["data"]["LIDAR_TOP"]
        data_record = self.nusc.get("sample_data", data_token)
        data_path = os.path.join(self.dataroot, data_record["filename"])

        orig_pc = np.fromfile(data_path, dtype=np.float32).reshape((-1,5))
        pc = deepcopy(orig_pc)

        if self.simulator:
            sim_pc = self.simulator.simulate(pc)  
        else:
            sim_pc = pc                                                           

        if self.save_path:
            file_name = data_path.split("/")[-1]
            output_path = os.path.join(self.save_path, file_name)
            sim_pc.astype(np.float32).tofile(output_path)

        # Get transformation matrices
        cs_record = self.nusc.get('calibrated_sensor', data_record['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', data_record['ego_pose_token'])

        # Translation and rotation from global to LiDAR frame
        lidar_translation = np.array(cs_record['translation'])
        lidar_rotation = Quaternion(cs_record['rotation'])

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        bounding_box = []

        for ann_token in sample['anns']:
    
            # and ann['category_name'].startswith('vehicle'):
            box = self.nusc.get_box(ann_token)

            # Transform box from global -> ego -> LiDAR
            box.translate(-ego_translation)
            box.rotate(ego_rotation.inverse)

            box.translate(-lidar_translation)
            box.rotate(lidar_rotation.inverse)

            corners = box.corners()  # (3, 8)
            #print(corners)
            x = corners[0, [0, 1, 5, 4, 0]]
            y = corners[1, [0, 1, 5, 4, 0]]

            bounding_box.append((x,y))

        return orig_pc, sim_pc, bounding_box
        

    def simulate_fault(self, max_samples = None):
        num_samples = len(self.nusc.sample)
        if max_samples:
            num_samples = min(max_samples, num_samples)

        indx = list(range(len(self.nusc.sample)))
        np.random.shuffle(indx)
        nusc_samples = [self.nusc.sample[i] for i in indx[:num_samples]]

        if self.show_pbar:
            results = process_map(self.process_one_sample, nusc_samples, chunksize=1)
        else:
            with Pool() as p:
                results = list(p.imap(self.process_one_sample, nusc_samples, chunksize=1))
        
        return results


def get_rain_simulator(rain_rate, save = True):
    rain_config = config_params.rain_params
    rain_cls_setter = rain_sim.RainParameters()
    rain_cls_setter.rain_rate = rain_rate #, rain_config["rainfall_rate"]
    sensor_cls_setter = rain_sim.SensorParameters()
    rain_cls_driver = rain_sim.rainSim(r_set = rain_cls_setter, 
                                       s_set = sensor_cls_setter,
                                       use_tqdm=False)
    
    return RainSim(
        input_dir=rain_config["input_dir"],
        output_dir=rain_config["output_dir"],
        driver=rain_cls_driver,
        save = save
    )

def get_fog_simulator(fog_alpha, save = True):
    config = config_params.fog_params
    parameter_set = fogsim.ParameterSet(alpha= fog_alpha, gamma=config["gamma"])
    return FogSim(input_dir= config["input_dir"],
                       output_dir= config["output_dir"],
                       paramset= parameter_set,
                       simulation_options=config["simulation_options"],
                       save = save)

def get_snow_simulator(snowfall_rate, save):
    snow_params = config_params.snow_params
    #snowfall_rate = snow_params["snowfall_rate"]
    terminal_velocity = snow_params["terminal_velocity"]
    rainfall_rate = sampling.snowfall_rate_to_rainfall_rate(snowfall_rate, terminal_velocity)
    occupancy_ratio = sampling.compute_occupancy(snowfall_rate, terminal_velocity)

    rain_sim_args = {
        "input_dir": snow_params["input_dir"],
        "output_dir": snow_params["output_dir"],
        "rainfall_rate": rainfall_rate,        
        "occupancy_ratio": occupancy_ratio,      
        "mode": snow_params["mode"],            
        "save": save
    }

    return SnowSim(**rain_sim_args)

def get_dust_simulator(dust_level = "moderate", save=True):
    dust_config = config_params.dust_params
    dust_pset = dust_params.DustParams(mode = dust_level, compute_alpha_beta = True)
    # dust_pset.alpha = 0.051
    # dust_pset.beta = 0.029
    print(f"Dust params: alpha = {dust_pset.alpha}, beta = {dust_pset.beta}")
    
    # Now precompute integration
    print("Precomputing integration lookup tables...")
    integration_args = {
        "n_steps": 2000,
        "r_0_max": 200,
        "save_path": dust_config["integration_path"],
        "n_cpus": cpu_count()
    }

    precompute_integration.compute_integration(alpha=dust_pset.alpha, beta=dust_pset.beta, arguments=integration_args)

    print("Simulating dust in lidar point cloud")
    # Now initiate parameter set equivalent of fog
    parameter_set = fogsim.ParameterSet(alpha= dust_pset.alpha, beta=dust_pset.beta ,gamma=0.000001)
    return DustSim(input_dir= dust_config["input_dir"],
                       output_dir= dust_config["output_dir"],
                       paramset= parameter_set,
                       simulation_options=dust_config["simulation_options"],
                       save = save)

def get_fault_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fault', type=str, default="all")
    parser.add_argument("--fog_alpha", type = float, default=0.04)
    parser.add_argument("--rain_rate", type=float, default=50.0)
    parser.add_argument("--snow_rate", type=float, default=1.5)
    parser.add_argument("--dust_level", type=str, default="moderate")
    parser.add_argument("--nusc_root", type=str, default=None)
    parser.add_argument("--rain_save_path", type = str, default = None)
    parser.add_argument("--fog_save_path", type=str, default=None)
    parser.add_argument("--snow_save_path", type=str, default=None)
    parser.add_argument("--dust_save_path", type=str, default=None)
    parser.add_argument("--plot_save_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save", action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_fault_args()
    nusc_sim = NuscenesSimulator(dataroot=args.nusc_root,
                                 simulator=None,
                                 save_path=None,
                                 show_pbar=True)
    
    
    if not args.plot_save_path:
        args.plot_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    #%% Simulate Rain in Nuscenes Samples
    if args.fault == "rain" or args.fault == "all":
        print("Simulating rain in point cloud")
        nusc_sim.simulator = get_rain_simulator(args.rain_rate, save = args.save)
        os.makedirs(args.rain_save_path, exist_ok=True)
        nusc_sim.save_path = args.rain_save_path
        
        rain_results = nusc_sim.simulate_fault(max_samples=args.max_samples) 
        pc1, pc2, bbox = rain_results[0]
        plot_with_anns(pc1, pc2, xlim=50, ylim=50,
                save_dir=args.plot_save_path, 
                save_name=f"rain_sim_anns_{args.rain_rate}.png",
                custom_title="Rain Simulated Point Cloud",
                save=True,
                box_corners=bbox)
        
    
    #%% Simulate Fog in Nuscenes Samples
    if args.fault == "fog" or args.fault == "all":
        print("Simulating fog in point cloud")
        nusc_sim.simulator = get_fog_simulator(args.fog_alpha, save = args.save)
        os.makedirs(args.fog_save_path, exist_ok=True)
        nusc_sim.save_path = args.fog_save_path
        fog_results = nusc_sim.simulate_fault(max_samples=args.max_samples) 
        pc1, pc2, bbox = fog_results[0]
        plot_with_anns(pc1, pc2, xlim=50, ylim=50,
                save_dir=args.plot_save_path, 
                save_name=f"fog_sim_anns_{args.fog_alpha}.png",
                custom_title="Fog Simulated Point Cloud",
                save=True,
                box_corners=bbox)

    #%% Simulate Snow in Nuscenes Samples
    if args.fault == "snow" or args.fault == "all":
        print("Simulating snow in point cloud")
        nusc_sim.simulator = get_snow_simulator(args.snow_rate, save = args.save)
        os.makedirs(args.snow_save_path, exist_ok=True)
        nusc_sim.save_path = args.snow_save_path
        snow_results = nusc_sim.simulate_fault(max_samples=args.max_samples) 
        pc1, pc2, bbox = snow_results[0]
        plot_with_anns(pc1, pc2, xlim=50, ylim=50,
                save_dir=args.plot_save_path, 
                save_name=f"snow_sim_anns_{args.snow_rate}.png",
                custom_title="Snow Simulated Point Cloud",
                save=True,
                box_corners=bbox)

    #%% Simulate Dust in Nuscenes Samples
    if args.fault == "dust" or args.fault == "all":
        print("Simulating dust in point cloud")
        nusc_sim.simulator = get_dust_simulator(args.dust_level, save=args.save)
        os.makedirs(args.dust_save_path, exist_ok=True)
        nusc_sim.save_path = args.dust_save_path
        dust_results = nusc_sim.simulate_fault(max_samples=args.max_samples)
        pc1, pc2, bbox = dust_results[0]
        plot_with_anns(pc1, pc2, xlim=50 , ylim=50,
                save_dir=args.plot_save_path, 
                save_name=f"dust_sim_anns_{args.dust_level}.png",
                custom_title="Dust Simulated Point Cloud",
                save=True,
                box_corners=bbox)
    
    if not args.fault in ["fog", "snow", "rain", "dust", "all"]:
        print("Fault not implemented. Choose any one from: ", ["fog", "snow", "rain", "dust", "all"])