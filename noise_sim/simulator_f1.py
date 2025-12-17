#%% Main Cell
import os
import sys

# Add the directory containing simulation files to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from LiDAR_noise_corruptions import sampling_noise, displacement_noise, object_level_noise
import LiDAR_object_failure
from LiDAR_cross_talk import crosstalk
import putils
import noise_config

import numpy as np


class RandomPositionNoise:
    def __init__(self, noise_params, noise_type):
        if noise_type == "uniform":
            self.noise_sim = displacement_noise.Uniform(**noise_params)
        elif noise_type == "gaussian":
            self.noise_sim = displacement_noise.Gaussian(**noise_params)
        elif noise_type == "impulse":
            self.noise_sim = displacement_noise.Impulse(**noise_params)
        else:
            raise NotImplementedError(f"Given noise type {noise_type} is not implemented")
    
    def simulate(self, pc: np.ndarray) -> np.ndarray:
        return self.noise_sim.simulate(pc)

class RandomSamplingNoise:
    def __init__(self, sampling_params, sampling_type):
        if sampling_type == "upsample":
            self.sim = sampling_noise.Upsample(**sampling_params)
        elif sampling_type == "downsample":
            self.sim = sampling_noise.Downsample(**sampling_params)
        else:
            raise NotImplementedError(f"Given sampling type: {sampling_type} is not implemented")
        
    def simulate(self, pc: np.ndarray) -> np.ndarray:
        return self.sim.simulate(pc)

class ObjectLevelNoise:
    def __init__(self, nuscenes_root, noise_params, noise_type,num_samples):

        if noise_type == "object_fail":
            self.sim = object_level_noise.ObjectFail(**noise_params)
        elif noise_type == "gaussian":
            self.sim = object_level_noise.Gaussian(**noise_params)
        else:
            raise NotImplementedError(f"Noise type {noise_type} is not implemented")
        
        self.nuscenes_root = nuscenes_root
        self.num_samples = num_samples
    
    def simulate(self):
        return object_level_noise.simulate_object_level_fault(
            nuscenes_root=self.nuscenes_root,
            simulator=self.sim,
            save_path=None,
            num_samples=self.num_samples
        )



def simulate_random_pos_noise(noise_config, num_files, save = False):
    file_lst = os.listdir(noise_config["input_dir"])
    file_choices = np.random.choice(file_lst, size=num_files, replace=False)

    noise_params = noise_config["noise_params"]
    noise_type = noise_config["noise_type"]

    pos_noise_sim = RandomPositionNoise(noise_params=noise_params, noise_type=noise_type)
    results = putils.process_files(file_names=file_choices,
                                   main_folder=noise_config["input_dir"], 
                                   simulator=pos_noise_sim)
    return results


def simulate_random_sampling_noise(sampling_params, num_files, sampling_type, save = False):
    file_lst = os.listdir(sampling_params["input_dir"])
    file_choices = np.random.choice(file_lst, size=num_files, replace=False)
    sampler = RandomSamplingNoise(sampling_params=sampling_params[sampling_type],
                                  sampling_type=sampling_type)
    results = putils.process_files(file_names=file_choices,
                                   main_folder=sampling_params["input_dir"],
                                   simulator=sampler)
    return results



if __name__ == "__main__":
    PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True)

    #%% Simulate Gaussian Noise in Spherical Co-rodinates of point cloud
    noise_params = noise_config.gaussian_pos_noise
    results = simulate_random_pos_noise(noise_config=noise_params, num_files=10, save=False)
    putils.compare_results(results=results,
                           save_dir=PLOT_DIR,
                           save_name=f"{noise_params['noise_type']}_noise.png",
                           title = f"{noise_params['noise_type']} Simulated Point Cloud".title(),
                           save = False)

    #%% Simulate Uniform Noise in Spherical Co-rodinates of point cloud
    noise_params = noise_config.uniform_pos_noise
    results = simulate_random_pos_noise(noise_config=noise_params, num_files=10, save=False)
    putils.compare_results(results=results,
                           save_dir=PLOT_DIR,
                           save_name=f"{noise_params['noise_type']}_noise.png",
                           title = f"{noise_params['noise_type']} Simulated Point Cloud".title(),
                           save = False)
    
    #%% Simulate Impulse Noise in Spherical Co-rodinates of point cloud
    noise_params = noise_config.impulse_pos_noise
    results = simulate_random_pos_noise(noise_config=noise_params, num_files=10, save=False)
    putils.compare_results(results=results,
                           save_dir=PLOT_DIR,
                           save_name=f"{noise_params['noise_type']}_noise.png",
                           title = f"{noise_params['noise_type']} Simulated Point Cloud".title(),
                           save = False)

    #%% Simulate Upsampling Noise
    sampling_params = noise_config.sampling_noise
    sampling_type = "upsample"
    results = simulate_random_sampling_noise(sampling_params=sampling_params,
                                             num_files=10,
                                             sampling_type=sampling_type)
    
    putils.compare_results(results=results,
                           save_dir=PLOT_DIR,
                           save_name=f"{sampling_type}_noise.png",
                           title = f"{sampling_type.title()} Noise in Point Cloud",
                           save=False)
    

    # %% Simulate Downsampling Noise
    sampling_params = noise_config.sampling_noise
    sampling_type = "downsample"
    results = simulate_random_sampling_noise(sampling_params=sampling_params,
                                             num_files=10,
                                             sampling_type=sampling_type)
    
    putils.compare_results(results=results,
                           save_dir=PLOT_DIR,
                           save_name=f"{sampling_type}_noise.png",
                           title = f"{sampling_type.title()} Noise in Point Cloud",
                           save=False)
    
    # %% Simulate Object Level Noise: No-Echo
    object_fail_args = noise_config.object_fail_params
    simulator = ObjectLevelNoise(nuscenes_root=object_fail_args["input_dir"],
                                 noise_params={"drop_prob": object_fail_args["drop_prob"]},
                                 num_samples=10,
                                 noise_type="object_fail")
    results = simulator.simulate()
    print(len(results))
    putils.compare_results_anns(results, 
                                save_dir=PLOT_DIR, 
                                save_name="failed_echo.png",
                                title="Failure of Echo from Object",
                                save=False)

# %%
