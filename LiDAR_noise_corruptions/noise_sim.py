import numpy as np
import os
from copy import deepcopy
from visualize_bev import plot_bev

def polar2cart(point: np.ndarray):
    R, theta, phi = point[:, 0], point[:, 1], point[:, 2]
    z = R * np.cos(theta)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    return np.stack((x, y, z), axis = 1)

def cart2polar(point: np.ndarray):
    x,y, z = point[:, 0], point[:, 1], point[:, 2]
    R = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/R)
    phi = np.arctan2(y, x)
    return np.stack([R, theta, phi], axis = 1)

class Uniform:
    def __init__(self, pos_noise, noise_probs):
        self.pos_noise = np.array(pos_noise)
        self.noise_probs = noise_probs

    def simulate(self, pc: np.ndarray):
        num_points = int(self.noise_probs * pc.shape[0])

        # Randomly select affected points
        indices = np.arange(pc.shape[0])
        np.random.shuffle(indices)
        affected_indices = indices[:num_points]

        # Simulate noise in xyz co-ordinates
        affected_xyz = pc[affected_indices, :3]
        affected_polar = cart2polar(affected_xyz)
        #breakpoint()
        noisy_polar = affected_polar + np.random.rand(num_points, 3) * self.pos_noise
        noisy_xyz = polar2cart(noisy_polar)
        pc[affected_indices, :3] = noisy_xyz

        return pc


class Gaussian:
    def __init__(self, pos_noise, noise_probs):
        self.pos_noise = np.array(pos_noise)
        self.noise_probs = noise_probs

    def simulate(self, pc: np.ndarray):
        num_points = int(self.noise_probs * pc.shape[0])

        # Randomly select affected points
        indices = np.arange(pc.shape[0])
        np.random.shuffle(indices)
        affected_indices = indices[:num_points]

        # Simulate noise in xyz co-ordinates
        affected_xyz = pc[affected_indices, :3]
        affected_polar = cart2polar(affected_xyz)
        #breakpoint()
        noisy_polar = affected_polar + np.random.normal(0, 1, size=(num_points, 3)) * self.pos_noise
        noisy_xyz = polar2cart(noisy_polar)
        pc[affected_indices, :3] = noisy_xyz

        return pc
    

class Impulse:
    def __init__(self, min_prob, min_val, max_prob, max_val):
        self.min_prob = min_prob
        self.min_val = min_val
        self.max_prob = max_prob
        self.max_val = max_val

    def simulate(self, pc: np.ndarray):
        rnd_num = np.random.rand(pc.shape[0])

        pc[rnd_num < self.min_prob, 3] = self.min_val
        pc[rnd_num > self.max_prob, 3] = self.max_val

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

    noise_type = "gaussian"

    if noise_type == "uniform":
        noise_sim = Uniform(pos_noise=(0.5, 0.01 * np.pi/180, 0.01 * np.pi/180), noise_probs=0.2)
    elif noise_type == "gaussian":
        noise_sim = Gaussian(pos_noise=(0.5, 0.01 * np.pi/180, 0.01 * np.pi/180), noise_probs=0.2)
    elif noise_type == "impulse":
        noise_sim = Impulse(min_prob=0.1, min_val=0, max_prob=0.8, max_val=255)
    else:
        raise NotImplementedError(f"Given noise type {noise_type} is not implemented")

    main_folder = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP"
    num_files = 10
    file_lst = os.listdir(main_folder)
    file_choices = np.random.choice(file_lst, size=num_files, replace=False)

    simulated_results = process_files(file_choices, main_folder, simulator=noise_sim)
    save_dir = "/home/saksham/samsad/mtech-project/fault-sim/LiDAR_noise_corruptions/outputs"
    save_name = "impulse_noise.png"
    title = "Impulse Noise Corrupted Point Cloud"
    compare_results(results=simulated_results, save_dir=save_dir, save_name=save_name, title=title)



