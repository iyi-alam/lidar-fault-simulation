import os
import argparse
from multiprocessing import Pool, cpu_count
import pickle

import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

GLOBAL_NUSC = None
GLOBAL_ARGS = None


def init_worker(dataroot, save_root, simulated_samples_root, dataversion):
    global GLOBAL_NUSC
    global GLOBAL_ARGS

    GLOBAL_NUSC = NuScenes(
        version=dataversion,
        dataroot=dataroot,
        verbose=False,
    )

    GLOBAL_ARGS = {
        "dataroot": dataroot,
        "save_root": save_root,
        "simulated_samples_root": simulated_samples_root,
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


# FIX 1: Replaced np.add.at with np.bincount via a flat index.
# np.add.at is unbuffered and allocates significant temporaries at scale.
# np.bincount is a single C-level pass — minimal memory, much faster.
def build_drop_probability_map(orig_pc, sim_pc, az_bins=180, el_bins=32, range_bins=20, max_range=80.0):
    dropped = sim_pc[:, 4] == 0
    xyz = orig_pc[:, :3]

    az_idx, el_idx, r_idx = compute_spherical_bins(xyz, az_bins, el_bins, range_bins, max_range)

    total_size = az_bins * el_bins * range_bins
    flat_idx = az_idx * (el_bins * range_bins) + el_idx * range_bins + r_idx

    total = np.bincount(flat_idx, minlength=total_size).reshape(az_bins, el_bins, range_bins).astype(np.float32)
    dropped_count = np.bincount(flat_idx[dropped], minlength=total_size).reshape(az_bins, el_bins, range_bins).astype(np.float32)

    prob = np.zeros_like(total)
    valid = total > 0
    prob[valid] = dropped_count[valid] / total[valid]
    return prob


def apply_drop_probability_map(pc, prob_map, max_range=100.0):
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


# FIX 2: Converted to a generator so groups are yielded one at a time.
# The original function built the entire group list in RAM before processing
# began. This generator yields one group at a time, letting memory be freed
# as each group is consumed by the pool instead of holding everything at once.
def iter_unique_groups(nusc, samples_list, max_sweeps=None, max_samples=None):
    if max_samples is not None:
        samples_list = samples_list[:max_samples]

    visited_tokens = set()

    for sample_token in samples_list:
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

        if owned_chain:
            yield {"key_token": lidar_token, "owned_chain": owned_chain}


def process_group(group):
    dataroot = GLOBAL_ARGS["dataroot"]
    save_root = GLOBAL_ARGS["save_root"]
    simulated_samples_root = GLOBAL_ARGS["simulated_samples_root"]

    key_token = group["key_token"]
    owned_tokens = group["owned_chain"]

    key_sd = get_sd(key_token)
    key_rel = key_sd["filename"]

    key_orig = load_pc(os.path.join(dataroot, key_rel))
    key_sim = load_pc(os.path.join(simulated_samples_root, key_rel))

    save_pc(os.path.join(save_root, key_rel), key_sim)

    sample_scatter_pts = key_sim[key_sim[:, 4] == 1].copy()
    drop_prob_map = build_drop_probability_map(key_orig, key_sim)

    # FIX 3: Explicitly delete large arrays as soon as they are no longer needed
    # so the worker's RSS drops before processing the sweep chain.
    del key_orig, key_sim

    for token in owned_tokens:
        if token == key_token:
            continue

        sweep_sd = get_sd(token)
        sweep_rel = sweep_sd["filename"]
        sweep_pc = load_pc(os.path.join(dataroot, sweep_rel))

        dropped_sweep = apply_drop_probability_map(sweep_pc, drop_prob_map)
        del sweep_pc  # free raw sweep immediately after applying drop map

        transferred_scatter = transfer_scatter_particles(
            sample_scatter_pts,
            key_token,
            token,
        )

        # FIX 4: np.concatenate instead of np.vstack — avoids intermediate copies.
        if transferred_scatter.shape[0] > 0:
            final_sweep = np.concatenate([dropped_sweep, transferred_scatter], axis=0)
        else:
            final_sweep = dropped_sweep

        save_pc(os.path.join(save_root, sweep_rel), final_sweep)
        del final_sweep, dropped_sweep, transferred_scatter  # free each sweep before the next

    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_root", required=True, type=str)
    parser.add_argument("--save_root", required=True, type=str)
    parser.add_argument("--simulated_samples_root", required=True, type=str)
    parser.add_argument("--samples_pkl_dir", required=True, type=str)
    parser.add_argument("--dataversion", default="v1.0-trainval", type=str)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_sweeps", default=2, type=int)
    # FIX 5: Workers capped at 2 by default.
    # Each worker loads a full NuScenes metadata object (~500MB–1GB).
    # More than 2–4 workers on a typical workstation exhausts RAM and triggers
    # the Linux OOM killer (SIGKILL / exit code 9). Override via --num_workers.
    parser.add_argument("--num_workers", default=min(8, cpu_count()), type=int)
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

    # Count groups for tqdm without materialising the full list in RAM.
    print("Counting processing groups...")
    total_groups = sum(1 for _ in iter_unique_groups(
        nusc,
        samples_list,
        max_sweeps=args.max_sweeps,
        max_samples=args.max_samples,
    ))
    print(f"Unique processing groups: {total_groups}")

    # Recreate the generator for actual processing.
    group_gen = iter_unique_groups(
        nusc,
        samples_list,
        max_sweeps=args.max_sweeps,
        max_samples=args.max_samples,
    )

    with Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(
            args.nusc_root,
            args.save_root,
            args.simulated_samples_root,
            args.dataversion,
        ),
    ) as pool:
        list(
            tqdm(
                # FIX 6: chunksize=1 prevents imap_unordered from pre-fetching
                # and queuing large batches of groups into worker memory at once.
                pool.imap_unordered(process_group, group_gen, chunksize=1),
                total=total_groups,
            )
        )

    print("Rain sweep propagation complete.")