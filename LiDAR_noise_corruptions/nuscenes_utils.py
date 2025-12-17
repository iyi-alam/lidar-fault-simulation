import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import os

def draw_bev_with_boxes(nusc, sample, lidar_bin_path, drop_prob=0.0, draw_boxes=True, figsize=(8, 8)):
    # Load point cloud from file
    points = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 5)

    # Get transforms for LiDAR to global
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # Translation and rotation from global to LiDAR frame
    lidar_translation = np.array(cs_record['translation'])
    lidar_rotation = Quaternion(cs_record['rotation'])

    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot point cloud in BEV (X–Y)
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c='k', label='LiDAR points')

    if draw_boxes:
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if ann['category_name'].startswith('vehicle'):
                # if drop_prob == 1.0:
                #     continue  # simulate full drop condition for plotting

                box = nusc.get_box(ann_token)
                # Transform box from global -> ego -> LiDAR
                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)

                box.translate(-lidar_translation)
                box.rotate(lidar_rotation.inverse)

                # Plot 2D projection of the box (X–Y BEV)
                corners = box.corners()  # (3, 8)
                #print(corners)
                x = corners[0, [0, 1, 5, 4, 0]]
                y = corners[1, [0, 1, 5, 4, 0]]
                print(type(x), y)
                ax.plot(x, y, 'r-', linewidth = 0.5)  # Red box outline

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_aspect('equal')
    ax.set_title("BEV Point Cloud with Vehicle Bounding Boxes")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Paths
    nuscenes_root = "/home/saksham/samsad/mtech-project/datasets/nuscenes"
    corrupted_root = "/home/saksham/samsad/mtech-project/datasets/nuscenes_obj_fail/samples/LIDAR_TOP"

    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root, verbose=True)
    sample_index = 10
    sample = nusc.sample[sample_index]

    # Original file
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    orig_path = os.path.join(nuscenes_root, lidar_data['filename'])

    # Corrupted file (with dropped points)
    corrupt_path = os.path.join(corrupted_root, os.path.basename(orig_path))

    print("Original Point Cloud with Boxes:")
    draw_bev_with_boxes(nusc, sample, orig_path, draw_boxes=True)

    print("Corrupted Point Cloud with Remaining Boxes:")
    draw_bev_with_boxes(nusc, sample, corrupt_path, draw_boxes=True, drop_prob=1.0)  # hide dropped boxes if needed

