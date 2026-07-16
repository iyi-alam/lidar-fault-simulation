import os
import sys
import argparse
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import pickle

import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import weather_sim.config_params as config_params
from LiDAR_rain_sim import rain_sim


GLOBAL_NUSC = None
GLOBAL_RAIN_SIM = None
GLOBAL_ARGS = None


class RainSim:
    def __init__(self, rain_rate):
        rain_cls_setter = rain_sim.RainParameters()
        rain_cls_setter.rain_rate = rain_rate

        sensor_cls_setter = rain_sim.SensorParameters()

        self.driver = rain_sim.rainSim(
            r_set=rain_cls_setter,
            s_set=sensor_cls_setter,
            use_tqdm=False,
        )

    def simulate(self, pc: np.ndarray):
        pc = deepcopy(pc)
        rain_pc, _, _ = self.driver.simulate(pc)
        return rain_pc


def init_worker(dataroot, dataversion, rain_rate, save_root):
    global GLOBAL_NUSC
    global GLOBAL_RAIN_SIM
    global GLOBAL_ARGS

    GLOBAL_NUSC = NuScenes(
        version=dataversion,
        dataroot=dataroot,
        verbose=False,
    )

    GLOBAL_RAIN_SIM = RainSim(rain_rate)

    GLOBAL_ARGS = {
        "dataroot": dataroot,
        "save_root": save_root,
    }


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_pc(abs_path):
    return np.fromfile(abs_path, dtype=np.float32).reshape((-1, 5))


def save_pc(abs_path, pc):
    ensure_parent_dir(abs_path)
    pc.astype(np.float32).tofile(abs_path)


def get_sd(token):
    return GLOBAL_NUSC.get("sample_data", token)


def get_lidar_to_global(sample_data_token):
    sd = get_sd(sample_data_token)
    cs = GLOBAL_NUSC.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = GLOBAL_NUSC.get("ego_pose", sd["ego_pose_token"])

    T_sensor = np.eye(4, dtype=np.float32)
    T_sensor[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    T_sensor[:3, 3] = np.array(cs["translation"], dtype=np.float32)

    T_ego = np.eye(4, dtype=np.float32)
    T_ego[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
    T_ego[:3, 3] = np.array(ego["translation"], dtype=np.float32)

    return T_ego @ T_sensor


def transform_xyz(xyz, T):
    if xyz.shape[0] == 0:
        return xyz

    pts = np.ones((xyz.shape[0], 4), dtype=np.float32)
    pts[:, :3] = xyz
    transformed = (T @ pts.T).T
    return transformed[:, :3]


def compute_spherical_bins(xyz, az_bins, el_bins, range_bins, max_range):
    r = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    el = np.arctan2(xyz[:, 2], xy)

    az_idx = np.clip(((az + np.pi) / (2 * np.pi) * az_bins).astype(np.int32), 0, az_bins - 1)
    el_idx = np.clip(((el + np.pi / 2) / np.pi * el_bins).astype(np.int32), 0, el_bins - 1)
    r_idx = np.clip((r / max_range * range_bins).astype(np.int32), 0, range_bins - 1)

    return az_idx, el_idx, r_idx


def build_drop_probability_map(orig_pc, sim_pc, az_bins=180, el_bins=32, range_bins=20, max_range=80.0):
    dropped = sim_pc[:, 4] == 0
    xyz = orig_pc[:, :3]

    az_idx, el_idx, r_idx = compute_spherical_bins(
        xyz,
        az_bins,
        el_bins,
        range_bins,
        max_range,
    )

    total = np.zeros((az_bins, el_bins, range_bins), dtype=np.float32)
    dropped_count = np.zeros_like(total)

    for i in range(xyz.shape[0]):
        total[az_idx[i], el_idx[i], r_idx[i]] += 1.0
        if dropped[i]:
            dropped_count[az_idx[i], el_idx[i], r_idx[i]] += 1.0

    prob = np.zeros_like(total)
    valid = total > 0
    prob[valid] = dropped_count[valid] / total[valid]
    return prob


def apply_drop_probability_map(pc, prob_map, max_range=80.0):
    if pc.shape[0] == 0:
        return pc

    az_bins, el_bins, range_bins = prob_map.shape

    az_idx, el_idx, r_idx = compute_spherical_bins(
        pc[:, :3],
        az_bins,
        el_bins,
        range_bins,
        max_range,
    )

    probs = prob_map[az_idx, el_idx, r_idx]
    rand = np.random.rand(pc.shape[0])
    drop_mask = rand < probs

    out = pc.copy()
    out[drop_mask, 0:4] = 0.0
    out[drop_mask, 4] = 0
    return out


def transfer_scatter_particles(sample_scatter_pts, sample_token, sweep_token, jitter_std=0.05):
    if sample_scatter_pts.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)

    T_sample = get_lidar_to_global(sample_token)
    T_sweep = get_lidar_to_global(sweep_token)
    T_global_to_sweep = np.linalg.inv(T_sweep)

    scatter_global_xyz = transform_xyz(sample_scatter_pts[:, :3], T_sample)
    scatter_sweep_xyz = transform_xyz(scatter_global_xyz, T_global_to_sweep)
    scatter_sweep_xyz += np.random.normal(0.0, jitter_std, scatter_sweep_xyz.shape).astype(np.float32)

    out = sample_scatter_pts.copy()
    out[:, :3] = scatter_sweep_xyz
    out[:, 4] = 1
    return out


def build_unique_groups(nusc, samples_list=None, max_sweeps=None, max_samples=None):
    if max_samples is not None:
        samples_token_list = samples_list[: max_samples]
    else:
        samples_token_list = samples_list

    visited_tokens = set()
    groups = []

    for sample_token in samples_token_list:
        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]

        if lidar_token in visited_tokens:
            continue

        owned_chain = []
        token = lidar_token
        count = 0

        while True:
            sd = nusc.get("sample_data", token)
            rel_path = sd["filename"]

            if token != lidar_token and rel_path.startswith("samples/LIDAR_TOP"):
                break

            if sd["token"] not in visited_tokens:
                visited_tokens.add(sd["token"])
                owned_chain.append(sd["token"])

            count += 1

            if max_sweeps is not None and count >= max_sweeps:
                break

            if sd["prev"] == "":
                break

            token = sd["prev"]

        if len(owned_chain) == 0:
            continue

        groups.append({
            "key_token": lidar_token,
            "owned_chain": owned_chain,
        })

    return groups


def process_group(group):
    dataroot = GLOBAL_ARGS["dataroot"]
    save_root = GLOBAL_ARGS["save_root"]

    key_token = group["key_token"]
    owned_tokens = group["owned_chain"]

    key_sd = get_sd(key_token)
    key_rel = key_sd["filename"]
    key_abs = os.path.join(dataroot, key_rel)

    key_orig = load_pc(key_abs)
    key_sim = GLOBAL_RAIN_SIM.simulate(key_orig.copy())

    save_pc(os.path.join(save_root, key_rel), key_sim)

    sample_scatter_pts = key_sim[key_sim[:, 4] == 1].copy()
    drop_prob_map = build_drop_probability_map(key_orig, key_sim)

    for token in owned_tokens:
        if token == key_token:
            continue

        sweep_sd = get_sd(token)
        sweep_rel = sweep_sd["filename"]
        sweep_abs = os.path.join(dataroot, sweep_rel)

        sweep_pc = load_pc(sweep_abs)
        dropped_sweep = apply_drop_probability_map(sweep_pc, drop_prob_map)

        transferred_scatter = transfer_scatter_particles(
            sample_scatter_pts,
            key_token,
            token,
        )

        if transferred_scatter.shape[0] > 0:
            final_sweep = np.vstack([dropped_sweep, transferred_scatter])
        else:
            final_sweep = dropped_sweep

        save_pc(os.path.join(save_root, sweep_rel), final_sweep)

    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_root", required=True, type=str)
    parser.add_argument("--save_root", required=True, type=str)
    parser.add_argument("--samples_pkl_dir", required=True, type=str)
    parser.add_argument("--dataversion", default="v1.0-trainval", type=str)
    parser.add_argument("--rain_rate", default=50.0, type=float)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_sweeps", default=10, type=int)
    parser.add_argument("--num_workers", default=min(1, os.cpu_count()//2), type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    nusc = NuScenes(
        version=args.dataversion,
        dataroot=args.nusc_root,
        verbose=True,
    )

    with open(args.samples_pkl_dir, "rb") as f:
        samples_list = pickle.load(f)

    groups = build_unique_groups(
        nusc,
        samples_list=samples_list,
        max_sweeps=args.max_sweeps,
        max_samples=args.max_samples,
    )

    print(f"Unique processing groups: {len(groups)}")
    #raise RuntimeError("Stop here for testing")

    with Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(
            args.nusc_root,
            args.dataversion,
            args.rain_rate,
            args.save_root,
        ),
    ) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_group, groups),
                total=len(groups),
            )
        )

    print("Rain simulation complete.")
