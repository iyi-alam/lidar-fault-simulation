from nuscenes.nuscenes import NuScenes
import os
import pickle
from nuscenes.utils.splits import train, val

all_splits = train + val
dataroot = "/data/home/samsadalam/mtech_project/datasets/v1.0-trainval"
save_dir = "/data/home/samsadalam/mtech_project/datasets/v1.0-trainval"
nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=True)

trainval_p1_samples = []
for scene in nusc.scene:
    if not scene['name'] in all_splits:
        continue
    sample_token = scene['first_sample_token']
    while True:
        trainval_p1_samples.append(sample_token)
        sample = nusc.get('sample', sample_token)
        if scene['last_sample_token'] == sample_token:
            break   
        sample_token = sample['next']

print(f"Total trainval samples collected: {len(trainval_p1_samples)}")
with open(os.path.join(save_dir, "trainval_p1_samples.pkl"), "wb") as f:
    pickle.dump(trainval_p1_samples, f)