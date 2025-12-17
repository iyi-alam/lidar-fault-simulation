import os
import sys

# Add the directory containing simulation files to the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from tqdm import tqdm
import random
from pyquaternion import Quaternion
from copy import deepcopy
from typing import List
import matplotlib.pyplot as plt
from visualize_bev import plot_bev, plot_with_anns

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


def simulate_object_level_fault(nuscenes_root, simulator, save_path=None, num_samples = 10):
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root, verbose=False)
    breakpoint()
    indices = list(range(len(nusc.sample)))
    random.shuffle(indices)
    indx_choice = indices[:num_samples]
    results = []
    drop_prob = simulator.drop_prob
    for choice in indx_choice:
        # Get LiDAR point cloud
        sample = nusc.sample[choice]
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(nuscenes_root, lidar_data['filename'])

        # Load point cloud
        orig_points = np.fromfile(lidar_filepath, dtype = np.float32).reshape((-1,5))
        points = deepcopy(orig_points)

        # Get transformation matrices
        cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Translation and rotation from global to LiDAR frame
        lidar_translation = np.array(cs_record['translation'])
        lidar_rotation = Quaternion(cs_record['rotation'])

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # Create mask
        mask = np.ones(points.shape[0], dtype=bool)
        box_corners = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            # and ann['category_name'].startswith('vehicle'):
            box = nusc.get_box(ann_token)

            # Transform box from global -> ego -> LiDAR
            box.translate(-ego_translation)
            box.rotate(ego_rotation.inverse)

            box.translate(-lidar_translation)
            box.rotate(lidar_rotation.inverse)

            # Check which points are inside the box
            inside_mask = points_in_box(box, points[:, :3].T)

            if random.random() < drop_prob:
                mask[inside_mask] = False  # Drop those points

            # Plot 2D projection of the box (X–Y BEV)
            corners = box.corners()  # (3, 8)
            #print(corners)
            x = corners[0, [0, 1, 5, 4, 0]]
            y = corners[1, [0, 1, 5, 4, 0]]
            box_corners.append((x, y))
  
        # Simulate fault wherever the flag is False
        #print(f"Affected points: {np.sum(mask)}/{len(mask)}")
        filtered_points = simulator.simulate(points, mask)
        results.append((orig_points, filtered_points, box_corners))

        # Save corrupted point cloud
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, os.path.basename(lidar_filepath))
            #print(out_file)
            filtered_points.tofile(out_file)

    return results
            

class Gaussian:
    def __init__(self, pos_sigma, drop_prob = 0.5):
        self.r_sigma, self.theta_sigma, self.phi_sigma = pos_sigma
        self.drop_prob = drop_prob
    

    def simulate(self, pc: np.ndarray, mask: List[bool]):

        pc_affected = pc[~mask, :]
        #print("Affected points shape: ", pc_affected.shape)
        r_noise = np.random.normal(0, self.r_sigma, size = pc_affected.shape[0])
        theta_noise = np.random.normal(0, self.theta_sigma, size = pc_affected.shape[0])
        phi_noise = np.random.normal(0, self.phi_sigma, size=pc_affected.shape[0])

        r_theta_phi = cart2polar(pc_affected)
        r_theta_phi[:,0] = r_theta_phi[:,0] + r_noise
        r_theta_phi[:,1] = r_theta_phi[:,1] + theta_noise
        r_theta_phi[:,2] = r_theta_phi[:,2] + phi_noise
        pc[~mask, :3] = polar2cart(r_theta_phi)

        return pc
    
class ObjectFail:
    def __init__(self, drop_prob):
        self.drop_prob=drop_prob

    def simulate(self, pc: np.ndarray, mask: List[bool]) -> np.ndarray:
        return pc[mask]


def compare_results(results, save_dir, save_name, title, save = True):
    """
    :results should be list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    """
    pc1, pc2, box_corners = results[0]
    print(len(box_corners))
    print(type(box_corners[0]))
    x, y = box_corners[0]
    print(type(x), type(y))
    

    plot_params  = {
        "pc1": pc1,                  # First point cloud (e.g., np.ndarray of shape (N, 3))
        "pc2": pc2,                  # Second point cloud (e.g., np.ndarray of shape (N, 3))
        "masking": False,           
        "xlim": 30,                 # X-axis limit for visualization
        "ylim": 30,                 # Y-axis limit for visualization
        "save": save,             
        "save_dir": save_dir,             
        "save_name": save_name,  
        "show_error": False,         
        "custom_title": title,
        "box_corners": box_corners        
    }

    plot_with_anns(**plot_params)



if __name__ == '__main__':
    
    root_path = "/home/saksham/samsad/mtech-project/datasets/nuscenes"
    #print(os.listdir(root_path))
    #save_path = '/home/saksham/samsad/mtech-project/datasets/nuscenes_obj_fail/samples/LIDAR_TOP'
    gss= Gaussian(pos_sigma=(0.5, 0.01 * np.pi/180, 0.01 * np.pi/180))
    drop_prob = 0.5
    # simulated_results = simulate_object_level_fault(nuscenes_root= root_path, simulator = gss, save_path= None, drop_prob= drop_prob,)

    # compare_results(
    #     results=simulated_results,
    #     save_dir="/home/saksham/samsad/mtech-project/fault-sim/LiDAR_noise_corruptions/outputs",
    #     save_name="object_level_gaussian_noise.png",
    #     title="Object Level Gaussian Noise",
    #     save=False
    # )

    object_fail = ObjectFail(drop_prob=0.5)
    simulated_results = simulate_object_level_fault(nuscenes_root= root_path, simulator = object_fail, save_path= None)
    compare_results(
        results=simulated_results,
        save_dir="/home/saksham/samsad/mtech-project/fault-sim/LiDAR_noise_corruptions/outputs",
        save_name="object_no_echo.png",
        title="Failed Echo from Objects",
        save=False
    )


    
    # orig_pc = np.random.rand(1000, 5) * 100
    # pc = deepcopy(orig_pc)
    # mask = np.ones(pc.shape[0], dtype=bool)
    # indices = list(range(pc.shape[0]))
    # np.random.shuffle(indices)
    # mask[indices[:pc.shape[0]//2]] = False
    # print(np.sum(mask))

    # affected_pc =  gss.simulate(pc, mask)

    # plt.scatter(orig_pc[:,0], affected_pc[:, 0], label = "x-coordinate comparison", s=5, alpha=0.4)
    # plt.show()
