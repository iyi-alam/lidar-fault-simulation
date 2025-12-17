import numpy as np
from rain_sim import RainParamaters, SensorParameters, rainSim
import os
import matplotlib.pyplot as plt
from copy import deepcopy

OUTPUT_PATH = "/home/saksham/samsad/mtech-project/Output"

def plot(orig_i, rain_i, labels):
    colormap = {0:'black', 1:'red', 2:'blue'}
    colors = [colormap[i] for i in labels]
    plt.scatter(orig_i, rain_i, c = colors, s = 10, alpha = 0.6)
    plt.xlabel('Original Intensity')
    plt.ylabel('Rainy Intensity')
    plt.title('Original vs Rainy Intensity Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def convert_intensity(lidar_file):
    rset = RainParamaters()
    pset = SensorParameters()
    sim = rainSim(rset = rset, pset = pset, use_tqdm = False)
    pc = np.fromfile(lidar_file, dtype = np.float32).reshape((-1,5))
    rain_pc, labels, _ = sim.simulate(pc)
    return pc[:,3], rain_pc[:,3], labels

def simulate_rain(lidar_file):
    rset = RainParamaters()
    pset = SensorParameters()
    sim = rainSim(rset = rset, pset = pset, use_tqdm = False)
    pc = np.fromfile(lidar_file, dtype = np.float32).reshape((-1,5))
    pc = deepcopy(pc)
    print(pc.shape)
    rain_pc, labels, _ = sim.simulate(pc)
    print(rain_pc.shape)
    return pc, rain_pc

def plot_4_bev_maps(clean, faulty, savepath, xlim=(-50, 50), ylim=(-50, 50), figsize=(12, 12)):
    """
    Plot 2 clean and 2 faulty BEV maps in a 2x2 grid.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Define a list of titles and point clouds
    titles = ['Original BEV', 'Original BEV', 'Rainy BEV', 'Rainy BEV']
    pcs = [clean[0], clean[1], faulty[0], faulty[1]]
    cmaps = ['viridis', 'viridis', 'plasma', 'plasma']  # different colormaps for visual clarity

    for ax, pc, title, cmap in zip(axs.flat, pcs, titles, cmaps):
        sc = ax.scatter(pc[:, 0], pc[:, 1], s=0.5, c=pc[:, 2], cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{savepath}")

if __name__ == "__main__":
    lidar_folder = 'datasets/nuscenes/samples/LIDAR_TOP'
    file_names = os.listdir(lidar_folder)
    file_names = np.random.choice(file_names, 2)
    file_paths = [os.path.join(lidar_folder, file_name) for file_name in file_names]
    simulated_pc = [simulate_rain(file_path) for file_path in file_paths]
    orig = [x[0] for x in simulated_pc]
    rainy = [x[1] for x in simulated_pc]

    print(orig[0].shape, orig[1].shape)
    print(rainy[0].shape, rainy[1].shape)
    #plot(orig, rainy, labels)
    plot_4_bev_maps(orig, rainy, savepath="compare_rain.png")