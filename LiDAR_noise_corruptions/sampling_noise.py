import os
import sys

# Add the directory containing simulation files to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from visualize_bev import plot_bev
from copy import deepcopy

class Upsample:
    def __init__(self, del_x, del_y, del_z, int_factor, sample_prob, num_samples):
        self.disp = np.array([del_x, del_y, del_z])
        self.intensity_factor = int_factor
        self.sample_prob = sample_prob
        self.num_samples = num_samples

    def simulate(self, pc: np.ndarray):
        num_points = int(pc.shape[0] * self.sample_prob)
        indx = list(range(pc.shape[0]))
        np.random.shuffle(indx)
        affected_points = indx[:num_points]
        new_points = []

        for point in affected_points:
            for i in range(self.num_samples):
                new_point = np.zeros_like(pc[point])
                new_point[:3] = pc[point, :3] + np.random.rand(3) * self.disp
                new_point[3] = pc[point, 3] * (np.random.rand() * self.intensity_factor)
                new_point[4:] = pc[point, 4:]
                new_points.append(new_point)
        
        new_points = np.stack(new_points, axis = 0)

        return np.concatenate((pc, new_points), axis = 0)

class Downsample:
    def __init__(self, percent_points):
        self.percent_points = percent_points

    def simulate(self, pc: np.ndarray):
        num_points = int(pc.shape[0] * self.percent_points)
        indx = list(range(pc.shape[0]))
        np.random.shuffle(indx)
        affected_points = indx[:num_points]
        for point in affected_points:
            pc[point, :4] = np.array([0,0,0,0])

        return pc
        


def process_files(file_names, main_folder, simulator):
    """
    simulates the desired fault and returns a list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    This can be used for visual comparison
    """

    results = [None]*len(file_names)
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(main_folder, file_name)
        orig_pc = np.fromfile(file_path, dtype = np.float32).reshape((-1,5))
        pc = deepcopy(orig_pc)
        simulated_pc = simulator.simulate(pc)
        results[i] = (orig_pc, simulated_pc)
    
    return results

def compare_results(results, save_dir, save_name, title, save = True):
    """
    :results should be list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    """
    pc1, pc2 = results[0]

    plot_params  = {
        "pc1": pc1,                  # First point cloud (e.g., np.ndarray of shape (N, 3))
        "pc2": pc2,                  # Second point cloud (e.g., np.ndarray of shape (N, 3))
        "masking": True,           
        "xlim": 50,                 # X-axis limit for visualization
        "ylim": 50,                 # Y-axis limit for visualization
        "save": save,             
        "save_dir": save_dir,             
        "save_name": save_name,  
        "show_error": False,         
        "custom_title": title        
    }

    plot_bev(**plot_params)


if __name__ == "__main__":

    upsample_params = {
        "del_x": 5,          # Shift in X direction (float)
        "del_y": 5,         # Shift in Y direction (float)
        "del_z": 5,          # Shift in Z direction (float)
        "int_factor": 0.2,    # Maximum intensity gain of randomly upsampled points
        "sample_prob": 0.01,    # Probability of sampling (float between 0 and 1)
        "num_samples": 10    # Number of samples (integer)
    }

    downsample_params = {"percent_points": 0.2}

    up_sampler = Upsample(**upsample_params)
    dwn_sampler = Downsample(**downsample_params)
    
    main_folder = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP"
    num_files = 10
    file_lst = os.listdir(main_folder)
    file_choices = np.random.choice(file_lst, size=num_files, replace=False)

    # simulated_results = process_files(file_choices, main_folder, simulator=up_sampler)
    # save_dir = "/home/saksham/samsad/mtech-project/fault-sim/LiDAR_noise_corruptions/outputs"
    # save_name = "random_upsampling.png"
    # title = "Randomly Upsampled Point Cloud"
    # compare_results(results=simulated_results, save_dir=save_dir, save_name=save_name, title=title)

    simulated_results = process_files(file_choices, main_folder, simulator=dwn_sampler)
    save_dir = "/home/saksham/samsad/mtech-project/fault-sim/LiDAR_noise_corruptions/outputs"
    save_name = "random_downsample.png"
    title = "Randomly Downsampled Point Cloud"
    compare_results(results=simulated_results, save_dir=save_dir, save_name=save_name, title=title)


    
    