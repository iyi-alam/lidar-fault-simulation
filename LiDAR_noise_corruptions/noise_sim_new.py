import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


# IMPORT NOISE FUNCTIONS

from add_noise_pc import MAP   # from your second file



# BOX → BEV POLYGON

def get_bev_polygon(box):
    w, l, h = box.wlh
    local = np.array([
        [ l/2,  w/2, 0],
        [ l/2, -w/2, 0],
        [-l/2, -w/2, 0],
        [-l/2,  w/2, 0],
    ])
    R = box.orientation.rotation_matrix
    c = np.array(box.center)

    corners = (R @ local.T).T + c
    x = [corners[i,0] for i in [0,1,2,3,0]]
    y = [corners[i,1] for i in [0,1,2,3,0]]
    return x, y



# PLOT FUNCTION

def plot_one(pc, box_corners, save_path, title="Original", xlim=25, ylim=25):
    x, y = pc[:,0], pc[:,1]

    plt.figure(figsize=(14,12))
    plt.scatter(x, y, c='gray', s=1.0)

    plt.xlabel("X (meters)", fontsize=32)
    plt.ylabel("Y (meters)", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.axis("equal")
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    for bx, by in box_corners:
        plt.plot(bx, by, "g-", linewidth=2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()



# LOAD POINT CLOUD

def load_pc_and_boxes(nusc, sample):
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    filepath = os.path.join(nusc.dataroot, sd["filename"])

    points = np.fromfile(filepath, dtype=np.float32).reshape((-1,5))

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    lidar_t = np.array(cs["translation"])
    lidar_R = Quaternion(cs["rotation"])
    ego_t = np.array(ego["translation"])
    ego_R = Quaternion(ego["rotation"])

    box_corners = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith("vehicle"):
            continue

        box = deepcopy(nusc.get_box(ann_token))

        box.translate(-ego_t)
        box.rotate(ego_R.inverse)
        box.translate(-lidar_t)
        box.rotate(lidar_R.inverse)

        box_corners.append(get_bev_polygon(box))

    return points, box_corners



# NOISE WRAPPER

def apply_noise(pc, noise_type="gaussian", severity=3):
    """
    pc: (N,5) → x,y,z,intensity,channel

    noise_type:
        gaussian, uniform, background, impulse, upsampling,
        gaussian_rad, uniform_rad, impulse_rad
    """

    assert noise_type in MAP, f"{noise_type} not supported"

    xyz = pc[:, :3]
    rest = pc[:, 3:]   # intensity + channel

    # apply noise
    noisy_xyz = MAP[noise_type](xyz.copy(), severity)

    # Handle point count change (e.g., background, upsampling)
    if noisy_xyz.shape[0] != xyz.shape[0]:
        # pad or truncate rest
        diff = noisy_xyz.shape[0] - xyz.shape[0]

        if diff > 0:
            pad = np.repeat(rest[:1], diff, axis=0)
            rest = np.vstack([rest, pad])
        else:
            rest = rest[:noisy_xyz.shape[0]]

    return np.hstack([noisy_xyz, rest])



# MAIN

if __name__ == "__main__":

    ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes-mini/"
    nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)

    save_dir = "noise_results"
    os.makedirs(save_dir, exist_ok=True)

    noise_types = [
        # "gaussian",
        "background",
        'upsampling',
        "gaussian_rad"
    ]

    noise_severities = dict(
        gaussian = 5,
        background = 5,
        upsampling = 5,
        gaussian_rad = 5
    )

    for k, sample in enumerate(nusc.sample[:5]):

        pc_clean, box_corners = load_pc_and_boxes(nusc, sample)

        # Save clean
        plot_one(
            pc_clean,
            box_corners,
            save_path=f"{save_dir}/sample{k}_clean.png",
            xlim=75, ylim=75
        )

        # Apply multiple noise types
        for noise in noise_types:

            pc_noisy = apply_noise(
                pc_clean,
                noise_type=noise,
                severity=noise_severities[noise]
            )

            noise_folder = os.path.join(save_dir, noise)
            os.makedirs(noise_folder, exist_ok=True)

            plot_one(
                pc_noisy,
                box_corners,
                save_path=f"{noise_folder}/sample{k}_{noise}.png",
                xlim=75, ylim=75
            )

    print(f"Done. Results saved to {save_dir}")