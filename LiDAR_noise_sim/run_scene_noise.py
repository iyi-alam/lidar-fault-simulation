import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from scene_noise_simulate import NOISE_MAP

# assumes NOISE_MAP already defined
# fn: (pc: Nx4, **kwargs) -> Nx4 or Mx4



# LOAD SINGLE SWEEP
def load_pc(filepath):
    pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return pc



# SAVE POINT CLOUD
def save_pc(pc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pc.astype(np.float32).tofile(save_path)



# APPLY NOISE (Nx5 safe)
def apply_noise(pc, noise_name, noise_params):
    fn = NOISE_MAP[noise_name]

    xyz_i = pc[:, :4]   # x,y,z,intensity
    rest = pc[:, 4:]    # ring index or channel

    noisy = fn(xyz_i.copy(), **noise_params.get(noise_name, {}))

    # handle size mismatch
    if noisy.shape[0] != xyz_i.shape[0]:
        diff = noisy.shape[0] - xyz_i.shape[0]

        if diff > 0:
            pad = np.repeat(rest[:1], diff, axis=0)
            rest = np.vstack([rest, pad])
        else:
            rest = rest[:noisy.shape[0]]

    return np.hstack([noisy, rest])



# GET ALL SWEEPS (CHAIN)
def get_sweeps(nusc, start_token, max_sweeps=10):
    sweeps = []
    current_token = start_token

    for _ in range(max_sweeps):
        sd = nusc.get("sample_data", current_token)
        sweeps.append(sd)

        if sd["prev"] == "":
            break

        current_token = sd["prev"]

    return sweeps



# MAIN PIPELINE
def simulate_and_save(
    nusc_root,
    save_root,
    noise_dict,
    noise_params,
    max_sweeps=10
):

    nusc = NuScenes(version="v1.0-mini", dataroot=nusc_root, verbose=False)

    for sample in nusc.sample:

        lidar_token = sample["data"]["LIDAR_TOP"]

        sweeps = get_sweeps(nusc, lidar_token, max_sweeps)

        for sweep_idx, sd in enumerate(sweeps):

            in_path = os.path.join(nusc_root, sd["filename"])
            pc = load_pc(in_path)

            for noise_name, enabled in noise_dict.items():
                if not enabled:
                    continue

                pc_noisy = apply_noise(pc, noise_name, noise_params)

                # build save path
                rel_path = sd["filename"]  # keeps folder structure
                save_path = os.path.join(
                    save_root,
                    noise_name,
                    rel_path
                )

                save_pc(pc_noisy, save_path)

    print(f"Finished. Saved noisy dataset to: {save_root}")


if __name__ == "__main__":

    NUSC_ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes-mini/"
    SAVE_ROOT = "/home/saksham/samsad/mtech-project/datasets/nusc-mini-sim/lidar_corrupt/noise/scene_gaussian_noise/"

    noise_dict = {
        "gaussian_cart": False,
        "gaussian_rad": True,
        "background": False,
        "upsample": False
    }

    noise_params = {
        "gaussian_cart": {"c": 0.05},
        "gaussian_rad": {"c": 0.20},
        "background": {"c": 0.15},
    }

    simulate_and_save(
        nusc_root=NUSC_ROOT,
        save_root=SAVE_ROOT,
        noise_dict=noise_dict,
        noise_params=noise_params,
        max_sweeps=10
    )