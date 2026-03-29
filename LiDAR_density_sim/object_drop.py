import os
import numpy as np
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from copy import deepcopy
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes



# IO
def load_pc(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def save_pc(pc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc.astype(np.float32).tofile(path)



# POINT-IN-BOX (VECTOR FORM)
def points_in_box_mask(points, box):
    pts = points - box.center

    R = box.orientation.rotation_matrix
    pts = pts @ R  # inverse rotation

    w, l, h = box.wlh

    mask = (
        (np.abs(pts[:, 0]) <= l / 2) &
        (np.abs(pts[:, 1]) <= w / 2) &
        (np.abs(pts[:, 2]) <= h / 2)
    )

    return mask



# OBJECT DROP CORE
def apply_object_drop(nusc, sample, pc, drop_prob):

    lidar_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_token)

    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ego = nusc.get('ego_pose', sd['ego_pose_token'])

    lidar_t = np.array(cs['translation'])
    lidar_R = Quaternion(cs['rotation'])

    ego_t = np.array(ego['translation'])
    ego_R = Quaternion(ego['rotation'])

    keep_mask = np.ones(pc.shape[0], dtype=bool)

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        if not ann['category_name'].startswith('vehicle'):
            continue

        if random.random() > drop_prob:
            continue

        box = deepcopy(nusc.get_box(ann_token))

        # global → ego → lidar
        box.translate(-ego_t)
        box.rotate(ego_R.inverse)

        box.translate(-lidar_t)
        box.rotate(lidar_R.inverse)

        inside = points_in_box_mask(pc[:, :3], box)
        keep_mask[inside] = False

    return pc[keep_mask]

def process_one(args):
    (
        nusc_root,
        sample,
        save_root,
        drop_prob
    ) = args

    global nusc

    lidar_token = sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_token)

    in_path = os.path.join(nusc_root, sd['filename'])
    pc = load_pc(in_path)

    pc_out = apply_object_drop(nusc, sample, pc, drop_prob)

    out_path = os.path.join(
        save_root,
        sd['filename']
    )

    save_pc(pc_out, out_path)

    return 1

def build_tasks(nusc, nusc_root, save_root, drop_prob):
    tasks = []

    for sample in nusc.sample:
        tasks.append((
            nusc_root,
            sample,
            save_root,
            drop_prob
        ))

    return tasks

def simulate_object_drop_fault_mp(
    nusc_root,
    save_root,
    drop_prob=0.5,
    num_workers=None
):

    global nusc
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=nusc_root,
        verbose=False
    )

    tasks = build_tasks(nusc, nusc_root, save_root, drop_prob)

    print(f"Total samples: {len(tasks)}")

    num_workers = num_workers or cpu_count()

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks)
        ))

    print(f"Finished. Saved to {save_root}")

if __name__ == "__main__":

    NUSC_ROOT = "/home/saksham/samsad/mtech-project/datasets/nuscenes-mini/"
    SAVE_ROOT = "/home/saksham/samsad/mtech-project/datasets/nusc-mini-sim/lidar_corrupt/density/object_drop/"

    simulate_object_drop_fault_mp(
        nusc_root=NUSC_ROOT,
        save_root=SAVE_ROOT,
        drop_prob=0.5,
        num_workers=18
    )