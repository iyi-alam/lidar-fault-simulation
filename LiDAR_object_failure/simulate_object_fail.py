import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
import os
from tqdm import tqdm
import random
from pyquaternion import Quaternion

def simulate_object_drop_fault(nuscenes_root, save_path=None, drop_prob=0.5, verbose=False):
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root, verbose=verbose)

    for sample in tqdm(nusc.sample):
        # Get LiDAR point cloud
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_filepath = os.path.join(nuscenes_root, lidar_data['filename'])

        # Load point cloud
        points = np.fromfile(lidar_filepath, dtype = np.float32).reshape((-1,5))
        #points = pc.points.T  # (N, 5)

        # Get transformation matrices
        cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Translation and rotation from global to LiDAR frame
        lidar_translation = np.array(cs_record['translation'])
        lidar_rotation = Quaternion(cs_record['rotation'])

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # Create mask
        keep_mask = np.ones(points.shape[0], dtype=bool)

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if ann['category_name'].startswith('vehicle') and random.random() < drop_prob:
                box = nusc.get_box(ann_token)

                # Transform box from global -> ego -> LiDAR
                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)

                box.translate(-lidar_translation)
                box.rotate(lidar_rotation.inverse)

                # Check which points are inside the box
                inside_mask = points_in_box(box, points[:, :3].T)
                keep_mask[inside_mask] = False  # Drop those points

        filtered_points = points[keep_mask]
        #print(filtered_points.shape)

        if verbose:
            print(f"Original points: {points.shape[0]}, After drop: {filtered_points.shape[0]}")
        

        # Save corrupted point cloud
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, os.path.basename(lidar_filepath))
            #print(out_file)
            filtered_points.tofile(out_file)
            


if __name__ == '__main__':
    root_path = '/home/saksham/samsad/mtech-project/datasets/nuscenes-mini'
    save_path = '/home/saksham/samsad/mtech-project/datasets/nuscenes_obj_fail/samples/LIDAR_TOP'
    drop_prob = 1.0
    simulate_object_drop_fault(nuscenes_root= root_path, save_path= save_path, drop_prob= drop_prob, verbose= False)