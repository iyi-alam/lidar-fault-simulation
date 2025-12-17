# Ref paper: [Characterization and simulation of the effect of road dirt on the performance of a laser scanner]
# (https://ieeexplore.ieee.org/document/8317784)

import numpy as np
import os
from dust_lookup import data, samples
from typing import List
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


class DustSim:
    def __init__(self, total_channels: List[int], mode = "moderate", ):
        """
        mode = ["light", "moderate", "heavy"]
        depending upon the mode, more channels will be affected with dust
        """
        if mode == "light":
            num_channels = 20 #%
            self.position_noise = 2
        elif mode == "heavy":
            num_channels = 70 #%
            self.position_noise = 10
        else: # mode == "moderate":
            num_channels = 40 #%
            self.position_noise = 5
        
        #self.R_max = 120
        self.data = data
        self.samples = samples
        self.channel_choice = np.random.choice(total_channels, size = int(num_channels * len(total_channels)/100), replace=False)
    
    def simulate(self, pc):
        
        for channel in self.channel_choice:
            channel_flag = pc[:,4] == channel
            pc_channel = pc[channel_flag, :3]
            noise_sample = np.random.choice(self.samples)
            del_r = self.data[noise_sample]["del_R"]
            noise = np.random.randint(0, 1, size = pc_channel.shape[0]) * del_r/100
            pc[channel_flag, 3] = pc[channel_flag, 3] * (1 - noise)

            # Add position change noise
            pc[channel_flag,:3] = np.random.randn(pc_channel.shape[0], 3) * (1 - self.position_noise/100) * pc[channel_flag,:3]

            #print(f"Max Intensity: {pc[:,3].max()}, Minimum Intensity: {pc[:,3].min()}")
            
        return pc
    

def process_one_file(file_name, load_folder, save_folder, simulator: DustSim):
    file_path = os.path.join(load_folder, file_name)
    save_path = os.path.join(save_folder, file_name)
    pc = np.fromfile(file_path, dtype = np.float32).reshape((-1, 5))
    pc = deepcopy(pc)
    pc = simulator.simulate(pc=pc)
    pc.astype(np.float32).tofile(save_path)
    return

            

if __name__ == "__main__":

    dust_sim = DustSim(total_channels=list(range(32)), mode = "moderate")
    #print(dust_sim.channel_choice)
    LOAD_FOLDER = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP"
    SAVE_FOLDER = "/home/saksham/samsad/mtech-project/datasets/nuscenes_dust/samples/LIDAR_TOP"
    USE_MULTIPROCESS = False
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    partial_process = partial(process_one_file, load_folder = LOAD_FOLDER, save_folder = SAVE_FOLDER, simulator = dust_sim)
    file_names = os.listdir(LOAD_FOLDER)

    if USE_MULTIPROCESS:
        with Pool(processes=10) as p:
            for _ in tqdm(p.imap_unordered(partial_process, file_names), total=len(file_names)):
                pass
    
    else:
        for file_name in tqdm(file_names, total=len(file_names)):
            partial_process(file_name=file_name)

