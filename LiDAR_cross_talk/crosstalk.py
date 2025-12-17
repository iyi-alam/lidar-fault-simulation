import numpy as np
from copy import deepcopy
import open3d as o3d
import os
import matplotlib.pyplot as plt

OUTPUT_PATH = "/home/saksham/samsad/mtech-project/Output"

class CrossTalk:
    def __init__(self, kt, sigma_c):
        self.kt = kt
        self.sigma_c = sigma_c
        
    
    def simulate(self, pc):
        points = deepcopy(pc)
        N = points.shape[0]
        mask = np.zeros(shape = N)
        L = self.kt * N // 100
        rng = list(range(N))
        np.random.shuffle(rng)
        zeta_c = np.random.randn(self.sigma_c.shape[0]) * self.sigma_c
        print(zeta_c)
        points[rng[:L],:4] = points[rng[:L], :4] + zeta_c
        mask[rng[:L]] = 1
        return points, mask

class Visualize:
    def __init__(self, pc, mask):
        self.pc = pc
        self.mask = mask

    def label_to_color(self):
        # labels: numpy array of 0s and 1s
        colors = np.zeros((self.mask.shape[0], 3))  # Initialize with black
        colors[self.mask == 1] = [1.0, 0.0, 0.0]     # Red for label 1
        return colors

    def visualize_point_cloud(self):
        points = self.pc[:, :3]
        intensity = self.pc[:, 3]  # Unused now, can be removed
        mask = np.logical_and.reduce((
            points[:, 0] <= 25, points[:, 0] >= -25,
            points[:, 1] <= 25, points[:, 1] >= -25
        ))

        mod_points = points[mask]
        mod_mask = self.mask[mask]  # Filter mask to match mod_points
        colors = np.zeros((mod_mask.shape[0], 3))
        colors[mod_mask == 1] = [1.0, 0.0, 0.0]  # Red for label 1, else stays black

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mod_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        o3d.visualization.draw_geometries([pcd], window_name="LiDAR Viewer")

def plot_bev(clean_pc, faulty_pc, savepath, xlim=(-50, 50), ylim=(-50, 50), figsize=(12, 6)):
    """
    Plot BEV maps of clean and faulty point clouds side-by-side.
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    axs[0].scatter(clean_pc[:, 0], clean_pc[:, 1], s=0.5, c=clean_pc[:, 2], cmap='viridis')
    axs[0].set_title('Clean BEV Map')
    axs[0].set_xlabel('X (meters)')
    axs[0].set_ylabel('Y (meters)')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].set_aspect('equal')
    
    axs[1].scatter(faulty_pc[:, 0], faulty_pc[:, 1], s=0.5, c=faulty_pc[:, 2], cmap='plasma')
    axs[1].set_title('Faulty BEV Map')
    axs[1].set_xlabel('X (meters)')
    axs[1].set_ylabel('Y (meters)')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
    axs[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/{savepath}")
    

def plot_4_bev_maps(clean, faulty, savepath, xlim=(-50, 50), ylim=(-50, 50), figsize=(12, 12)):
    """
    Plot 2 clean and 2 faulty BEV maps in a 2x2 grid.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Define a list of titles and point clouds
    titles = ['Original BEV', 'Original BEV', 'Cross-talk', 'Cross-talk']
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
    import os
    lidar_folder = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP"
    num_samples = 10
    kt = 5
    sigma_c = np.array([5,5,5,20])
    ctalk = CrossTalk(kt = kt, sigma_c= sigma_c)
    
    original_points = []
    simulated_points= []
    
    # Randomly convert some 10 points for visualization
    file_names = os.listdir(lidar_folder)
    np.random.shuffle(file_names)
    for file_name in file_names[:num_samples]:
        file_path = os.path.join(lidar_folder, file_name)
        pc = np.fromfile(file_path, dtype = np.float32).reshape((-1,5))
        simulated_pc, mask = ctalk.simulate(pc = pc)
        original_points.append(pc)
        simulated_points.append((simulated_pc, mask))
    
    print('Cross Talk simulated successfully! ')

    # Plot a random sample
    choice = np.random.choice(num_samples, 2)
    opc = [original_points[k] for k in choice]
    spc  = [simulated_points[k][0] for k in choice]
    # vis = Visualize(spc, mask)
    # vis.visualize_point_cloud()
    plot_4_bev_maps(opc, spc, "compare_cross_talk.png")




