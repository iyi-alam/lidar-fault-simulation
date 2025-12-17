import numpy as np
import matplotlib.pyplot as plt
import os


def plot(pc_n, pc_sn, save_path):
    """
    Parameters:
    - pc_n: numpy array of shape (4, N), normal point cloud
    - pc_sn: numpy array of shape (4, N), simulated point cloud
    """
    #assert pc_n.shape[0] == 4 and pc_sn.shape[0] == 4, "Input point clouds must be of shape (4, N)"
    def filter_range(pc):
        x, y = pc[:, 0], pc[:, 1]
        mask = (x >= -10) & (x <= 10) & (y >= -10) & (y <= 10)
        return pc[mask, :]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #pc_n,pc_sn = pc_n.T, pc_sn.T
    pc_n = filter_range(pc_n)
    pc_sn = filter_range(pc_sn)

    # Normal Point Cloud
    sc1 = axs[0].scatter(pc_n[:,0], pc_n[:, 1], c=pc_n[:,3], cmap='viridis', s=1)
    axs[0].set_title('Normal Point Cloud')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].axis('equal')
    fig.colorbar(sc1, ax=axs[0], label='Intensity')

    # Simulated Point Cloud
    sc2 = axs[1].scatter(pc_sn[:,0], pc_sn[:,1], c=pc_sn[:,3], cmap='viridis', s=1)
    axs[1].set_title('Snowy Point Cloud')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].axis('equal')
    fig.colorbar(sc2, ax=axs[1], label='Intensity')

    plt.tight_layout()
    plt.savefig(save_path)


def load_pc(lidar_path, fault_path_extension):
    file_names = os.listdir(lidar_path)
    pc_n = []
    pc_sn = []
    for item in file_names[:10]:
        
        normal_path = os.path.join(lidar_path, item)
        faulty_path = normal_path.replace('nuscenes', 'nuscenes_obj_fail')
        pc_n.append(np.fromfile(normal_path, dtype = np.float32).reshape((-1,5)))
        pc_sn.append(np.fromfile(faulty_path, dtype = np.float32).reshape((-1,5)))
    
    k = np.random.choice(len(pc_n))
    return pc_n[k], pc_sn[k]


if __name__ == '__main__':
    lidar_path = '/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP'
    fault_path_extension = 'nuscenes_obj_fail'

    pc_n, pc_sn = load_pc(lidar_path, fault_path_extension)
    plot(pc_n, pc_sn, 'object_fail_comparison.png')