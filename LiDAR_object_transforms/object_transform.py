import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from copy import deepcopy


class Transform:
    def __init__(self, pos = 0.2, angle = 10, transform_prob = 0.5):
        """
        :pos in meters
        :angle in degrees
        """
        self.translation, self.rotation = self.get_transformation_params(pos, angle)
        self.transform_prob = transform_prob

    def get_transformation_params(self, max_translate = 1, max_rotate = 10):
        translation = np.random.rand(3)
        axis = np.random.randn(3)
        axis = axis/np.linalg.norm(axis)
        theta = np.random.uniform(low=-max_rotate, high=max_rotate) * np.pi/180
        rot_vec = theta * axis

        rotation = R.from_rotvec(rot_vec)
        return translation, rotation

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.rotation.apply(points) + self.translation


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


                keep_mask[inside_mask] = False

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
            
def simulate_object_transform(nuscenes_root, simulator: Transform, save_path=None, num_samples = 5):
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_root, verbose=False)
    #breakpoint()
    indices = list(range(len(nusc.sample)))
    random.shuffle(indices)
    indx_choice = indices[:num_samples]
    results = []
    transform_prob = simulator.transform_prob

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
        orig_box = []
        transf_box = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            if random.random() < transform_prob:

                # and ann['category_name'].startswith('vehicle'):
                box = nusc.get_box(ann_token)

                # Transform box from global -> ego -> LiDAR
                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)

                box.translate(-lidar_translation)
                box.rotate(lidar_rotation.inverse)

                # Check which points are inside the box
                inside_mask = points_in_box(box, points[:, :3].T)

                corners = box.corners()  # (3, 8)
                #print(corners)
                x = corners[0, [0, 1, 5, 4, 0]]
                y = corners[1, [0, 1, 5, 4, 0]]

                # Transform points inside the box
                if sum(inside_mask) > 0:
                    points[inside_mask,:3] = simulator.apply(points[inside_mask,:3])

                    # Transform bounding boxes too
                    z = np.zeros(5)
                    bbox_points = np.stack((np.array(x), np.array(y), z), axis = 1)
                    tbbox_points = simulator.apply(bbox_points)

                    orig_box.append(bbox_points)
                    transf_box.append(tbbox_points)

  
        # Simulate fault wherever the flag is False
        #print(f"Affected points: {np.sum(mask)}/{len(mask)}")
        #filtered_points = simulator.simulate(points, mask)
        results.append((orig_points, points, orig_box, transf_box))

        # Save corrupted point cloud
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, os.path.basename(lidar_filepath))
            #print(out_file)
            points.tofile(out_file)

    return results


def plot_with_anns(pc1, pc2, xlim = 75, ylim = 75, 
             save = False, save_dir = "", save_name = "boxtransf_point_cloud.png",
             show_error = True, custom_title = None,
             box_corners = None):
    
    x1, y1, i1 = pc1[:,0], pc1[:,1], pc1[:,3]
    x2, y2, i2 = pc2[:,0], pc2[:,1], pc2[:,3]
    orig_box, trf_box = box_corners

    # if pc1.shape == pc2.shape:
    #     # Compute MSE between intensities
    #     mse_pos = np.sqrt(np.linalg.norm(x1-x2)**2 + np.linalg.norm(y1-y2)**2)
    #     mse_i = np.linalg.norm(i1-i2)
    # else:
    #     print(f"MSE computation not possible due to size mismatch, Size of pc1: {pc1.shape} and Size of pc2: {pc2.shape}")
        

    fig, ax = plt.subplots(1,2, figsize = (36,18))

    sc1 = ax[0].scatter(x1, y1, c = 'gray', s=1.0)
    #ax[0].set_title("Original point cloud")
    ax[0].set_xlabel("X[meter]", fontsize = 36)
    ax[0].set_ylabel("Y[meter]", fontsize = 36)
    ax[0].set_xticks(ticks = np.linspace(-xlim, xlim, 6))
    ax[0].set_yticks(ticks = np.linspace(-ylim, ylim, 6))
    ax[0].tick_params(axis='both', labelsize=30)
    ax[0].axis('equal')
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)

    if orig_box is not None:
        for box in orig_box:
            ax[0].plot(box[:, 0], box[:, 1], 'r-', linewidth = 2.0)  # Red box outline

    # cbar1 = fig.colorbar(sc1, ax=ax[0], shrink=0.8)
    # cbar1.set_label("Intensity")

    sc2 = ax[1].scatter(x2, y2, c = 'gray', s=1.0)
    if custom_title is None:
        custom_title = "Modified point cloud"

    # if show_error:
    #     ax[1].set_title(f"{custom_title}\npos_mse = {mse_pos:.4f} | i_mse = {mse_i:.4f}")
    # else:
    #     ax[1].set_title(custom_title)

    ax[1].set_xlabel("X[meter]", fontsize = 36)
    ax[1].set_ylabel("Y[meter]", fontsize = 36)
    ax[1].set_xticks(ticks = np.linspace(-xlim, xlim, 6))
    ax[1].set_yticks(ticks = np.linspace(-ylim, ylim, 6))
    ax[1].tick_params(axis='both', labelsize=30)
    ax[1].axis('equal')
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)
    
    if orig_box is not None:
        for box in orig_box:
            ax[1].plot(box[:, 0], box[:, 1], 'r-', linewidth = 2.0)  # Red box outline

    if trf_box is not None:
        for box in trf_box:
            ax[1].plot(box[:, 0], box[:, 1], 'g-', linewidth = 2.0)  # Green box outline

    # cbar2 = fig.colorbar(sc2, ax=ax[1], shrink=0.8)
    # cbar2.set_label("Intensity")

    plt.tight_layout()
    if save:
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi = 150)
    else:
        plt.show()
    
    return


if __name__ == "__main__":

    dataroot = "/home/saksham/samsad/mtech-project/datasets/nuscenes-mini"
    transformer = Transform(pos=5.0, angle=20)

    results = simulate_object_transform(nuscenes_root=dataroot,
                                        simulator=transformer,
                                        save_path=None,
                                        num_samples=5)
    
    print(len(results))

    pc1, pc2, orig_box, trf_box = results[0]
    # orig_box = results[0][2]
    # trf_box = results[0][3]

    assert len(orig_box) == len(trf_box)

    # for i in range(len(orig_box)):
    #     a = orig_box[i]
    #     b = trf_box[i]

    #     plt.plot(a[:, 0], a[:, 1], '-r')
    #     plt.plot(b[:,0], b[:, 1], '-g')
    
    # plt.show()

    plot_with_anns(pc1, pc2, save=True, box_corners=(orig_box, trf_box))
    



