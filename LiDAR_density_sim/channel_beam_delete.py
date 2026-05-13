import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import pickle


val_samples_file= "/home/saksham/samsad/mtech-project/datasets/nuscenes_part1/nuscenes_p1_val_samples.pkl"
with open(val_samples_file, 'rb') as f:
    VAL_SAMPLES = pickle.load(f)

# NOISE FUNCTIONS
def apply_channel_drop(pc, num_drop=16):
    channels = pc[:, 4].astype(int)
    unique_ch = np.unique(channels)

    drop_ch = np.random.choice(
        unique_ch,
        size=min(num_drop, len(unique_ch)),
        replace=False
    )

    keep_mask = ~np.isin(channels, drop_ch)
    return pc[keep_mask]


def apply_beam_drop(pc, beam_width_deg=5):
    x, y = pc[:, 0], pc[:, 1]
    az = np.degrees(np.arctan2(y, x))

    center = np.random.uniform(-180, 180)
    half = beam_width_deg / 2

    lo = center - half
    hi = center + half

    if lo < -180:
        keep = ~((az > lo + 360) | (az < hi))
    elif hi > 180:
        keep = ~((az > lo) | (az < hi - 360))
    else:
        keep = ~((az >= lo) & (az <= hi))

    return pc[keep]


# Optional: simple echo loss (random thinning)
def apply_echo_loss(pc, drop_ratio=0.1):
    keep = np.random.rand(len(pc)) > drop_ratio
    return pc[keep]



# IO
def load_pc(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc.astype(np.float32).tofile(path)



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

        if noise_name == "channel_drop":
            pc_out = apply_channel_drop(
                pc,
                **noise_params.get(noise_name, {})
            )

            postfix = f"_{noise_params['channel_drop']['num_drop']}"

        elif noise_name == "beam_drop":
            pc_out = apply_beam_drop(
                pc,
                **noise_params.get(noise_name, {})
            )

            postfix = f"_{noise_params['beam_drop']['beam_width_deg']}"

        elif noise_name == "echo_loss":
            pc_out = apply_echo_loss(
                pc,
                **noise_params.get(noise_name, {})
            )

            postfix = f"_{noise_params['echo_loss']['drop_ratio']}"

        else:
            continue

        out_path = os.path.join(
            save_root,
            f"{noise_name}{postfix}",
            rel_path
        )

        save_pc(pc_out, out_path)

    return 1


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


# BUILD TASK LIST
def build_tasks(nusc, nusc_root, save_root, noise_dict, noise_params, max_sweeps):

    tasks = []
    visited_tokens = set()

    for sample in VAL_SAMPLES: #nusc.sample:
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

    noise_dict = {
        "channel_drop": True,
        #"beam_drop": True,
        # "echo_loss": False
    }

    noise_params = {
        "channel_drop": {"num_drop": 24},
        #"beam_drop": {"beam_width_deg": 30},
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