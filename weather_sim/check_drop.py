import numpy as np
import os
from tqdm import tqdm

lidar_folder = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP_weather/rain/rain_rate_5.0/"
filenames = os.listdir(lidar_folder)

label_0_count = 0
label_1_count = 0
all_count = 0
unique_vals = set()
for filename in tqdm(filenames, desc = "processing"):
    filepath = os.path.join(lidar_folder, filename)
    pc = np.fromfile(filepath, dtype=np.float32).reshape(-1,5)
    label_0_count +=  np.sum(pc[:,4] == 0)
    label_1_count +=  np.sum(pc[:,4] == 1)
    unique_vals.update(pc[:,4])
    all_count += pc.shape[0]

print(f"Label 0 count: {label_0_count}/{all_count} ({label_0_count/all_count*100:.2f}%)")
print(f"Label 1 count: {label_1_count}/{all_count} ({label_1_count/all_count*100:.2f}%)")
print(f"Unique labels: {sorted(unique_vals)}")