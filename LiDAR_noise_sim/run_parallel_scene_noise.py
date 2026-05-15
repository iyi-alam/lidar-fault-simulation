import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from scene_noise_simulate import NOISE_MAP


# assumes NOISE_MAP exists



# IO
def load_pc(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc.astype(np.float32).tofile(path)



# NOISE
def apply_noise(pc, noise_name, noise_params):
    fn = NOISE_MAP[noise_name]

    xyz_i = pc[:, :4]
    rest = pc[:, 4:]

    noisy = fn(xyz_i.copy(), **noise_params.get(noise_name, {}))

    if noisy.shape[0] != xyz_i.shape[0]:
        diff = noisy.shape[0] - xyz_i.shape[0]

        if diff > 0:
            pad = np.repeat(rest[:1], diff, axis=0)
            rest = np.vstack([rest, pad])
        else:
            rest = rest[:noisy.shape[0]]

    return np.hstack([noisy, rest])



# SWEEP CHAIN
def get_sweeps(nusc, start_token, max_sweeps=None):
    sweeps = []
    token = start_token
    count = 0

    while True:
        sd = nusc.get("sample_data", token)
        sweeps.append(sd)

        count += 1

        # stop if reached limit
        if max_sweeps is not None and count >= max_sweeps:
            break

        # stop if no more previous sweeps
        if sd["prev"] == "":
            break

        token = sd["prev"]

    return sweeps



# WORKER
def process_one(args):
    (
        in_path,
        rel_path,
        save_root,
        noise_dict,
        noise_params
    ) = args

    pc = load_pc(in_path)

    for noise_name, enabled in noise_dict.items():
        if not enabled:
            continue

        pc_noisy = apply_noise(pc, noise_name, noise_params)
        noise_vals = list(noise_params[noise_name].values())
        postfix = '_'.join(str(v) for v in noise_vals)
        out_path = os.path.join(save_root, f"scene_{noise_name}_{postfix}", rel_path)
        save_pc(pc_noisy, out_path)
    
    return 1  # progress unit



# BUILD TASK LIST
def build_tasks(nusc, nusc_root, save_root, noise_dict, noise_params, max_sweeps):

    tasks = []
    visited_tokens = set()

    for sample in nusc.sample:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sweeps = get_sweeps(nusc, lidar_token, max_sweeps)

        for sd in sweeps:

            token = sd["token"]

            # skip already processed sweep
            if token in visited_tokens:
                continue

            visited_tokens.add(token)

            rel_path = sd["filename"]
            in_path = os.path.join(nusc_root, rel_path)

            tasks.append((
                in_path,
                rel_path,
                save_root,
                noise_dict,
                noise_params
            ))

    return tasks



# MAIN

def simulate_and_save_mp(
    nusc_root,
    save_root,
    noise_dict,
    noise_params,
    max_sweeps=10,
    num_workers=None
):

    nusc = NuScenes(version="v1.0-mini", dataroot=nusc_root, verbose=False)

    tasks = build_tasks(
        nusc,
        nusc_root,
        save_root,
        noise_dict,
        noise_params,
        max_sweeps
    )

    print(f"Total sweeps to process: {len(tasks)}")
    #raise RuntimeError("Debug")

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
    MAX_SWEEPS = None
    NUM_WORKERS = 18

    noise_dict = {
        "gaussian_cart": False,
        "gaussian_rad": True,
        "background": False,
        "upsample": False
    }

    # c is percentage affected points and jitter is point shift in upsampling
    noise_params = {
        "gaussian_cart": {"scale": 0.1},     # 1 cm std
        "gaussian_rad": {"scale": 0.1},      # 1 cm range noise
        "background": {"percentage": 0.10},   # +10% clutter
        "upsample": {"percentage": 0.10, "jitter": 0.1}
    }

    for gaussian_rad_scale in [0.01, 0.02, 0.05, 1.0, 2.0]:
        noise_params["gaussian_rad"]["scale"] = gaussian_rad_scale
        simulate_and_save_mp(
            nusc_root=NUSC_ROOT,
            save_root=SAVE_ROOT,
            noise_dict=noise_dict,
            noise_params=noise_params,
        max_sweeps=MAX_SWEEPS,
        num_workers= min(NUM_WORKERS, cpu_count())
    )