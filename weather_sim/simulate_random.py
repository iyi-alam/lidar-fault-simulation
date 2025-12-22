"""
This file is meant to simulate faults in all the files of an input folder and save the simulated output in an output folder
"""
#%% Main
import os
import sys
# Add the parent directory (fault-sim) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pathlib import Path

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm.contrib.concurrent import process_map 
import argparse
import config_params
from nuscenes.nuscenes import NuScenes

import weather_sim.config_params as config_params
from weather_sim.visualization_utils import plot_bev
import LiDAR_fog_sim.fog_simulation as fogsim

GLOBAL_RESIZE_N = config_params.PC_FEATURE_DIMS #Some datasets can be resized with (-1,5) while some need to resize to (-1,4). This variable takes care of this.
MAX_INTENSITY_VAL = config_params.MAX_INTENSITY_VAL
ALPHA_LIST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]


class FogSim:
    
    def __init__(self, nusc_root, output_dir, 
                 paramset: fogsim.ParameterSet,
                 simulation_options, save = True):
        self.nusc_root = nusc_root 
        self.output_dir = output_dir
        self.pset = paramset
        self.simulation_options = simulation_options
        self.save = save
        self.samples_dict = {}

    def _process_file(self, file_name_alphas, pset: fogsim.ParameterSet, fog_params):
        file_name, alpha = file_name_alphas
        file_path = os.path.join(self.nusc_root , file_name)
        pc = np.fromfile(file_path, dtype=np.float32).reshape((-1, GLOBAL_RESIZE_N))
        _pc = deepcopy(pc)
        if not MAX_INTENSITY_VAL:
            _pc[:,3] = _pc[:,3]*255

        pset.alpha = alpha
        foggified_pc, _, _ = fogsim.simulate_fog(pset, _pc, **fog_params)

        if not MAX_INTENSITY_VAL:
            foggified_pc[:,3] /= 255.0
        
        assert (foggified_pc.shape[1] == GLOBAL_RESIZE_N)
        #if self.save:
        out_path = os.path.join(self.output_dir, file_name)
        foggified_pc.astype('float32').tofile(out_path)
        if not os.path.exists(out_path):
            raise RuntimeError("Error saving file at location: ", out_path)
        # return pc, foggified_pc, file_name

    def simulate(self, show_pbar, max_samples):
        filename_alphas = []
        nusc = NuScenes(version = 'v1.0-mini', dataroot=self.nusc_root, verbose=False)
        tokens = set()
        for scene_index, scene in enumerate(nusc.scene):
            first_sample_token = scene['first_sample_token']
            sample_token = first_sample_token
            alpha = np.random.choice(ALPHA_LIST)
            # iterate samples in this scene
            while sample_token:
                sample = nusc.get('sample', sample_token)

                # LIDAR_TOP keyframe sample_data token
                sd_token = sample['data']['LIDAR_TOP']

                # go backwards across sweeps
                is_a_sample = True
                while sd_token:
                    sd = nusc.get('sample_data', sd_token)
                    filename = sd['filename']
                    full_path = os.path.join(self.nusc_root, filename)

                    if os.path.exists(full_path) and sd_token not in tokens:
                        tokens.add(sd_token)
                        filename_alphas.append((filename, alpha))
                        self.samples_dict[filename] = is_a_sample
                        is_a_sample = False

                    sd_token = sd['prev']

                sample_token = sample['next']
        
        print("Total samples and sweeps to process: ", len(tokens))
        # Partial function with fixed pset and fog_params
        process_fn = partial(self._process_file, pset=self.pset, fog_params=self.simulation_options)

        with Pool(cpu_count()) as pool:
            if show_pbar:
                results = list(tqdm(pool.imap(process_fn, filename_alphas, chunksize=1), total=len(filename_alphas)))
            else:
                results = list(pool.imap(process_fn, filename_alphas, chunksize=1))
        
        # print("Number of simulated samples: ", len(os.listdir(self.output_dir, 'samples/LIDAR')))
        # print("Number of simulated sweeps: ", len(os.listdir(self.output_dir, 'sweeps')))
        return results



def simulate_fog(args, fog_params, save = False, show_pbar = True, max_samples = None):
    parameter_set = fogsim.ParameterSet(alpha= fog_params["alpha"], gamma=fog_params["gamma"])
    simulator = FogSim(nusc_root= args.nusc_root,
                       output_dir= args.output_dir,
                       paramset= parameter_set,
                       simulation_options=fog_params["simulation_options"],
                       save = save)

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)


def compare_plot(results, save_dir, save_name, plot_title, save = False):
    choice = np.random.choice(len(results))
    orig_pc, sim_pc, _ = results[choice]
    plot_bev(orig_pc, sim_pc, xlim=75, ylim=75,
             save = save, save_dir=save_dir,
             save_name=save_name, custom_title=plot_title)

def get_fault_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fault', type=str, default="all")
    parser.add_argument("--fog_alpha", type = float, default=0.04)
    parser.add_argument("--fog_gamma", type=float, default=1e-6)
    parser.add_argument("--nusc_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--plot_save_path", default=None)
    #parser.add_argument("--max_samples")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_fault_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.plot_save_path:
        args.plot_save_path = os.path.join(args.output_dir, "plots")

    #%% Simulate fog with random alpha values
    if args.fault == "fog" or args.fault == "all":
        args.output_dir = os.path.join(args.output_dir, f"{args.fault}_random")
        os.makedirs(args.output_dir, exist_ok=True)
        fog_params = {
            "alpha": args.fog_alpha,
            "gamma": args.fog_gamma,
            "simulation_options": dict(
                noise = 2,
                gain = True,
                noise_variant = 'v1',
                hard = True,
                soft = True
            ),

        }
        fog_results = simulate_fog(args, fog_params = fog_params, max_samples=None, save=args.save, show_pbar=True)
