import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

SAVE_FOLDER = Path.home() / 'samsad/mtech-project/datasets/nuscenes_snow/samples_1.5/LIDAR_TOP'
LIDAR_FOLDER = Path.home() / 'samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP'

snow_file_paths = []
nosnow_file_paths = []
for file_name in os.listdir(SAVE_FOLDER):
    if file_name.endswith('_snow_1.5.bin'):
        snow_file = os.path.join(SAVE_FOLDER, file_name)
        nosnow_file = os.path.join(LIDAR_FOLDER, file_name.replace('_snow_1.5.bin', '.bin'))
        if os.path.exists(snow_file) and os.path.exists(nosnow_file):
            print(f'Found the file: {file_name}')
            snow_file_paths.append(snow_file)
            nosnow_file_paths.append(nosnow_file)


#print(nosnow_file_paths)
# Open any example file and plot
N = len(snow_file_paths)
k = np.random.choice(N)
k = min(k, N-1)
snow_pc = np.fromfile(snow_file_paths[k], dtype = np.float32).reshape((-1,5))
nosnow_pc = np.fromfile(nosnow_file_paths[k], dtype = np.float32).reshape((-1,5))

# Plot the clear and foggy point cloud
# Extract the point cloud co-ordinates
x_snow = snow_pc[:,0]
y_snow = snow_pc[:,1]
z_snow = snow_pc[:,2]
i_snow = snow_pc[:,3]
# Stack into (N, 4) format
# ==== Filter points within x and y range [-25, 25] ====
mask = (x_snow >= -25) & (x_snow <= 25) & (y_snow >= -25) & (y_snow <= 25)
x_snow_f = x_snow[mask]
y_snow_f = y_snow[mask]
i_snow_f = i_snow[mask]

x_nosnow = nosnow_pc[:,0]
y_nosnow = nosnow_pc[:,1]
z_nosnow = nosnow_pc[:,2]
i_nosnow = nosnow_pc[:,3]
# Stack into (N, 4) format
# ==== Filter points within x and y range [-25, 25] ====
mask = (x_nosnow >= -25) & (x_nosnow <= 25) & (y_nosnow >= -25) & (y_nosnow <= 25)
x_nosnow_f = x_nosnow[mask]
y_nosnow_f = y_nosnow[mask]
i_nosnow_f = i_nosnow[mask]

fig, axs1 = plt.subplots(1, 2, figsize=(16, 6))

 
# No Snow
sc = axs1[0].scatter(x_nosnow_f, y_nosnow_f, c=i_nosnow_f, cmap='viridis', s=0.3)
axs1[0].set_title("LiDAR Top View (BEV)")
axs1[0].set_xlabel("X [m]")
axs1[0].set_ylabel("Y [m]")
axs1[0].axis("equal")
plt.colorbar(sc, ax=axs1[0], label="Intensity")
 
# With Snow
sc_f = axs1[1].scatter(x_snow_f, y_snow_f, c=i_snow_f, cmap='viridis', s=0.3)
axs1[1].set_title(f"Snowy LiDAR Top View (BEV), snowfall = {0.06} mm/h")
axs1[1].set_xlabel("X [m]")
axs1[1].set_ylabel("Y [m]")
axs1[1].axis("equal")
plt.colorbar(sc_f, ax=axs1[1], label="Intensity")

plt.tight_layout()
plt.savefig('snow_simulation_comp.png')


