import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import pickle


val_samples_file = "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/nuscenes_p1_val_samples.pkl"
with open(val_samples_file, "rb") as f:
    VAL_SAMPLES = pickle.load(f)


def apply_channel_drop(pc, drop_ch):
    channels = pc[:, 4].astype(int)
    keep_mask = ~np.isin(channels, drop_ch)
    return pc[keep_mask]


def apply_beam_drop(pc, center_deg, beam_width_deg):
    x, y = pc[:, 0], pc[:, 1]
    az = np.degrees(np.arctan2(y, x))

    half = beam_width_deg / 2
    lo = center_deg - half
    hi = center_deg + half

    if lo < -180:
        keep = ~((az > lo + 360) | (az < hi))
    elif hi > 180:
        keep = ~((az > lo) | (az < hi - 360))
    else:
        keep = ~((az >= lo) & (az <= hi))

    return pc[keep]


def load_pc(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc.astype(np.float32).tofile(path)


def process_one(args):
    (
        in_path,
        rel_path,
        save_root,
        noise_dict,
        noise_params,
        fault_spec
    ) = args

    pc = load_pc(in_path)

    for noise_name, enabled in noise_dict.items():
        if not enabled:
            continue

        if noise_name == "channel_drop":
            pc_out = apply_channel_drop(
                pc,
                drop_ch=fault_spec["channel_drop"]
            )
            postfix = f"_{noise_params['channel_drop']['num_drop']}"

        elif noise_name == "beam_drop":
            pc_out = apply_beam_drop(
                pc,
                center_deg=fault_spec["beam_drop_center"],
                beam_width_deg=noise_params["beam_drop"]["beam_width_deg"]
            )
            postfix = f"_{noise_params['beam_drop']['beam_width_deg']}"

        else:
            continue

        out_path = os.path.join(
            save_root,
            f"{noise_name}{postfix}",
            rel_path
        )

        save_pc(pc_out, out_path)

    return 1


def get_sweeps(nusc, start_token, max_sweeps=None):
    sweeps = []
    token = start_token
    count = 0

    while True:
        sd = nusc.get("sample_data", token)
        sweeps.append(sd)

        count += 1

        if max_sweeps is not None and count >= max_sweeps:
            break

        if sd["prev"] == "":
            break

        token = sd["prev"]

    return sweeps


def build_fault_spec(noise_dict, noise_params):
    spec = {}

    if noise_dict.get("channel_drop", False):
        all_channels = np.arange(32)
        num_drop = noise_params["channel_drop"]["num_drop"]

        spec["channel_drop"] = np.random.choice(
            all_channels,
            size=min(num_drop, len(all_channels)),
            replace=False
        )

    if noise_dict.get("beam_drop", False):
        spec["beam_drop_center"] = np.random.uniform(-180, 180)

    return spec


def build_tasks(nusc, nusc_root, save_root, noise_dict, noise_params, max_sweeps):
    tasks = []
    visited_tokens = set()

    for sample in VAL_SAMPLES:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sweeps = get_sweeps(nusc, lidar_token, max_sweeps)

        fault_spec = build_fault_spec(noise_dict, noise_params)

        for sd in sweeps:
            token = sd["token"]

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
                noise_params,
                fault_spec
            ))

    return tasks


def simulate_density_mp(
    nusc_root,
    save_root,
    version,
    noise_dict,
    noise_params,
    max_sweeps=10,
    num_workers=None
):
    nusc = NuScenes(
        version=version,
        dataroot=nusc_root,
        verbose=False
    )

    tasks = build_tasks(
        nusc,
        nusc_root,
        save_root,
        noise_dict,
        noise_params,
        max_sweeps
    )

    print(f"Total sweeps: {len(tasks)}")

    num_workers = num_workers or cpu_count()

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks)
        ))

    print(f"Finished. Saved to {save_root}")

if __name__ == "__main__":

    NUSC_ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval"
    SAVE_ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_channel_drop", action="store_true")
    parser.add_argument("--enable_beam_drop", action="store_true")
    parser.add_argument("--beam_width_deg", type=int, default=180)
    parser.add_argument("--channel_drop", type=float, default=16)
    args = parser.parse_args()

    noise_dict = {
        "channel_drop": args.enable_channel_drop,
        "beam_drop": args.enable_beam_drop,
        # "echo_loss": False
    }

    noise_params = {
        "channel_drop": {"num_drop": args.channel_drop},
        "beam_drop": {"beam_width_deg": args.beam_width_deg},
        # "echo_loss": {"drop_ratio": 0.1}
    }

    # for beam_width in [30, 60, 90, 120, 180]:
    #     noise_params["beam_drop"]["beam_width_deg"] = beam_width

    simulate_density_mp(
        nusc_root=NUSC_ROOT,
        save_root=SAVE_ROOT,
        version = "v1.0-trainval",
        noise_dict=noise_dict,
        noise_params=noise_params,
        max_sweeps=None,
        num_workers=18
    )  