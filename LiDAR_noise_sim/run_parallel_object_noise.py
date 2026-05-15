import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from copy import deepcopy
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from object_noise_simulate import OBJECT_NOISE_MAP



# IO
def load_pc(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc.astype(np.float32).tofile(path)



# POINT-IN-BOX (VECTORIZED)
def points_in_box_mask(points, box):
    """
    points: (N,3)
    box: nuScenes Box (already in lidar frame)
    """

    # translate to box center
    pts = points - box.center

    # rotate to box frame
    R = box.orientation.rotation_matrix
    pts = pts @ R  # inverse rotation (R is orthonormal)

    w, l, h = box.wlh

    mask = (
        (np.abs(pts[:, 0]) <= l / 2) &
        (np.abs(pts[:, 1]) <= w / 2) &
        (np.abs(pts[:, 2]) <= h / 2)
    )

    return mask



# BUILD OBJECT FLAG
def get_object_flag(nusc, sample, pc_xyz):
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    lidar_t = np.array(cs["translation"])
    lidar_R = Quaternion(cs["rotation"])
    ego_t = np.array(ego["translation"])
    ego_R = Quaternion(ego["rotation"])

    object_mask = np.zeros(pc_xyz.shape[0], dtype=bool)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # restrict to vehicles (modify if needed)
        if not ann["category_name"].startswith("vehicle"):
            continue

        box = deepcopy(nusc.get_box(ann_token))

        # global → ego → lidar
        box.translate(-ego_t)
        box.rotate(ego_R.inverse)
        box.translate(-lidar_t)
        box.rotate(lidar_R.inverse)

        mask = points_in_box_mask(pc_xyz, box)
        object_mask |= mask

    return object_mask.astype(np.uint8)

def get_object_flag_for_sd(nusc, sample, sd, pc_xyz):

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    lidar_t = np.array(cs["translation"])
    lidar_R = Quaternion(cs["rotation"])
    ego_t = np.array(ego["translation"])
    ego_R = Quaternion(ego["rotation"])

    object_mask = np.zeros(pc_xyz.shape[0], dtype=bool)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        if not ann["category_name"].startswith("vehicle"):
            continue

        box = deepcopy(nusc.get_box(ann_token))

        # transform annotation → THIS sweep frame
        box.translate(-ego_t)
        box.rotate(ego_R.inverse)
        box.translate(-lidar_t)
        box.rotate(lidar_R.inverse)

        mask = points_in_box_mask(pc_xyz, box)
        object_mask |= mask

    return object_mask.astype(np.uint8)

def get_sweeps(nusc, start_token, max_sweeps=10):
    sweeps = []
    token = start_token

    for _ in range(max_sweeps):
        sd = nusc.get("sample_data", token)
        sweeps.append(sd)

        if sd["prev"] == "":
            break

        token = sd["prev"]

    return sweeps

def process_one(args):
    (
        nusc_root,
        sample,
        save_root,
        noise_dict,
        noise_params,
        max_sweeps
    ) = args

    global nusc

    lidar_token = sample["data"]["LIDAR_TOP"]
    sweeps = get_sweeps(nusc, lidar_token, max_sweeps)

    for sd in sweeps:

        in_path = os.path.join(nusc_root, sd["filename"])
        pc = load_pc(in_path)

        pc_xyz = pc[:, :3]

        # compute object mask for THIS sweep
        object_flag = get_object_flag_for_sd(nusc, sample, sd, pc_xyz)

        for noise_name, enabled in noise_dict.items():
            if not enabled:
                continue

            fn = OBJECT_NOISE_MAP[noise_name]

            pc_noisy = fn(
                pc[:, :4].copy(),
                object_flag,
                **noise_params.get(noise_name, {})
            )

            # handle extra channel
            rest = pc[:, 4:]

            if pc_noisy.shape[0] != pc.shape[0]:
                diff = pc_noisy.shape[0] - pc.shape[0]

                if diff > 0:
                    pad = np.repeat(rest[:1], diff, axis=0)
                    rest = np.vstack([rest, pad])
                else:
                    rest = rest[:pc_noisy.shape[0]]

            pc_out = np.hstack([pc_noisy, rest])

            noise_vals = list(noise_params[noise_name].values())
            postfix = '_'.join(str(v) for v in noise_vals) 


            out_path = os.path.join(
                save_root,
                f"object_{noise_name}_{postfix}",
                sd["filename"]
            )

            save_pc(pc_out, out_path)

    return 1

def build_tasks(nusc, nusc_root, save_root, noise_dict, noise_params, max_sweeps):
    tasks = []

    for sample in nusc.sample:
        tasks.append((
            nusc_root,
            sample,
            save_root,
            noise_dict,
            noise_params,
            max_sweeps
        ))

    return tasks

def simulate_object_noise_mp(
    nusc_root,
    save_root,
    noise_dict,
    noise_params,
    num_workers=None,
    max_sweeps=10
):

    global nusc
    nusc = NuScenes(version="v1.0-mini", dataroot=nusc_root, verbose=False)

    tasks = build_tasks(
        nusc,
        nusc_root,
        save_root,
        noise_dict,
        noise_params,
        max_sweeps

    )

    print(f"Total samples: {len(tasks)}")
    #raise RuntimeError("Debug Stop")

    num_workers = num_workers or cpu_count()

    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks)
        ))

    print(f"Finished. Saved to {save_root}")


if __name__ == "__main__":

    NUSC_ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes-mini/"
    SAVE_ROOT = "/home/saksham/samsad/mtech-project/datasets/nusc-mini-sim/lidar_corrupt/noise/"

    noise_dict = {
        "gaussian_cart": False,
        "gaussian_rad": True,
        "background": False,
        "upsample": False
    }

    # c is percentage affected points and jitter is point shift in upsampling
    noise_params = {
        "gaussian_cart": {"scale": 0.01},     # 1 cm
        "gaussian_rad": {"scale": 0.02},      # 2 cm
        "background": {"percentage": 0.05},   # +5% around objects
        "upsample": {"percentage": 0.05, "jitter": 0.01}
    }

    
    for gaussian_rad_scale in [0.01, 0.02, 0.05]:
        noise_params["gaussian_rad"]["scale"] = gaussian_rad_scale
        simulate_object_noise_mp(
            nusc_root=NUSC_ROOT,
            save_root=SAVE_ROOT,
            noise_dict=noise_dict,
            noise_params=noise_params,
            num_workers=8,
            max_sweeps=10
        )