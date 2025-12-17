import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt

def getMask(x,y, xlim, ylim):
    return np.logical_and.reduce([
        x > -xlim, x < xlim,
        y> -ylim, y < ylim
    ])

def applyMask(x, y, i, mask):
    return x[mask], y[mask], i[mask]


def plot_bev(pc1, pc2, masking = True, xlim = 25, ylim = 25, 
             save = False, save_dir = "", save_name = "modified_point_cloud.png",
             show_error = True, custom_title = None):
    
    # pc1 = np.fromfile(pc_file1, dtype = np.float32).reshape((-1,5))
    # pc2 = np.fromfile(pc_file2, dtype = np.float32).reshape((-1,5))
    
    x1, y1, i1 = pc1[:,0], pc1[:,1], pc1[:,3]
    x2, y2, i2 = pc2[:,0], pc2[:,1], pc2[:,3]

    if pc1.shape == pc2.shape:
        # Compute MSE between intensities
        mse_pos = np.sqrt(np.linalg.norm(x1-x2)**2 + np.linalg.norm(y1-y2)**2)
        mse_i = np.linalg.norm(i1-i2)
    else:
        print(f"MSE computation not possible due to size mismatch, Size of pc1: {pc1.shape} and Size of pc2: {pc2.shape}")
        

    # if masking:
    #     mask1 = getMask(x1, y1, xlim,ylim)
    #     mask2 = getMask(x2, y2, xlim, ylim)

    #     x1, y1, i1 = applyMask(x1, y1, i1, mask1)
    #     x2, y2, i2 = applyMask(x2, y2, i2, mask2)

    fig, ax = plt.subplots(1,2, figsize = (16,6))

    sc1 = ax[0].scatter(x1, y1, c = i1, cmap='viridis', s=0.1)
    ax[0].set_title("Original point cloud")
    ax[0].set_xlabel("X[m]")
    ax[0].set_ylabel("Y[m]")
    ax[0].axis('equal')
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)
    cbar1 = fig.colorbar(sc1, ax=ax[0], shrink=0.8)
    cbar1.set_label("Intensity")

    sc2 = ax[1].scatter(x2, y2, c = i2, cmap='viridis', s=0.1)
    if custom_title is None:
        custom_title = "Modified point cloud"

    if show_error:
        ax[1].set_title(f"{custom_title}\npos_mse = {mse_pos:.4f} | i_mse = {mse_i:.4f}")
    else:
        ax[1].set_title(custom_title)

    ax[1].set_xlabel("X[m]")
    ax[1].set_ylabel("Y[m]")
    ax[1].axis('equal')
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)
    cbar2 = fig.colorbar(sc2, ax=ax[1], shrink=0.8)
    cbar2.set_label("Intensity")

    plt.tight_layout()
    if save:
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
    
    return

def plot_with_anns(pc1, pc2, masking = True, xlim = 25, ylim = 25, 
             save = False, save_dir = "", save_name = "modified_point_cloud.png",
             show_error = True, custom_title = None,
             box_corners = None):
    
    x1, y1, i1 = pc1[:,0], pc1[:,1], pc1[:,3]
    x2, y2, i2 = pc2[:,0], pc2[:,1], pc2[:,3]

    if pc1.shape == pc2.shape:
        # Compute MSE between intensities
        mse_pos = np.sqrt(np.linalg.norm(x1-x2)**2 + np.linalg.norm(y1-y2)**2)
        mse_i = np.linalg.norm(i1-i2)
    else:
        print(f"MSE computation not possible due to size mismatch, Size of pc1: {pc1.shape} and Size of pc2: {pc2.shape}")
        

    if masking:
        mask1 = getMask(x1, y1, xlim,ylim)
        mask2 = getMask(x2, y2, xlim, ylim)

        x1, y1, i1 = applyMask(x1, y1, i1, mask1)
        x2, y2, i2 = applyMask(x2, y2, i2, mask2)

    fig, ax = plt.subplots(1,2, figsize = (16,6))

    sc1 = ax[0].scatter(x1, y1, c = i1, cmap='viridis', s=0.1)
    ax[0].set_title("Original point cloud")
    ax[0].set_xlabel("X[m]")
    ax[0].set_ylabel("Y[m]")
    ax[0].axis('equal')
    ax[0].set_xlim(-xlim, xlim)
    ax[0].set_ylim(-ylim, ylim)

    if box_corners is not None:
        for box_x, box_y in box_corners:
            ax[0].plot(box_x, box_y, 'r-', linewidth = 0.5)  # Red box outline

    cbar1 = fig.colorbar(sc1, ax=ax[0], shrink=0.8)
    cbar1.set_label("Intensity")

    sc2 = ax[1].scatter(x2, y2, c = i2, cmap='viridis', s=0.1)
    if custom_title is None:
        custom_title = "Modified point cloud"

    if show_error:
        ax[1].set_title(f"{custom_title}\npos_mse = {mse_pos:.4f} | i_mse = {mse_i:.4f}")
    else:
        ax[1].set_title(custom_title)

    ax[1].set_xlabel("X[m]")
    ax[1].set_ylabel("Y[m]")
    ax[1].axis('equal')
    ax[1].set_xlim(-xlim, xlim)
    ax[1].set_ylim(-ylim, ylim)

    if box_corners is not None:
        for box_x, box_y in box_corners:
            ax[1].plot(box_x, box_y, 'r-', linewidth = 0.5)  # Red box outline

    cbar2 = fig.colorbar(sc2, ax=ax[1], shrink=0.8)
    cbar2.set_label("Intensity")

    plt.tight_layout()
    if save:
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
    
    return

def process_files(file_names, main_folder, simulator):
    """
    simulates the desired fault and returns a list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    This can be used for visual comparison
    """

    results = [None]*len(file_names)
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(main_folder, file_name)
        orig_pc = np.fromfile(file_path, dtype = np.float32).reshape((-1,5))
        pc = deepcopy(orig_pc)
        simulated_pc = simulator.simulate(pc)
        results[i] = (orig_pc, simulated_pc)
    
    return results

def compare_results(results, save_dir, save_name, title, save = True, box_corners = None):
    """
    :results should be list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    """
    pc1, pc2 = results[0]

    plot_params  = {
        "pc1": pc1,                  # First point cloud (e.g., np.ndarray of shape (N, 3))
        "pc2": pc2,                  # Second point cloud (e.g., np.ndarray of shape (N, 3))
        "masking": True,           
        "xlim": 50,                 # X-axis limit for visualization
        "ylim": 50,                 # Y-axis limit for visualization
        "save": save,             
        "save_dir": save_dir,             
        "save_name": save_name,  
        "show_error": False,         
        "custom_title": title        
    }

    plot_bev(**plot_params)

def compare_results_anns(results, save_dir, save_name, title, save = True):
    """
    :results should be list of tuples where the tuple elements are:
    0: Original point cloud
    1: Simulated point cloud
    """
    #print(len(results))
    pc1, pc2, box_corners = results[0]
    
    
    plot_params  = {
        "pc1": pc1,                  # First point cloud (e.g., np.ndarray of shape (N, 3))
        "pc2": pc2,                  # Second point cloud (e.g., np.ndarray of shape (N, 3))
        "masking": False,           
        "xlim": 30,                 # X-axis limit for visualization
        "ylim": 30,                 # Y-axis limit for visualization
        "save": save,             
        "save_dir": save_dir,             
        "save_name": save_name,  
        "show_error": False,         
        "custom_title": title,
        "box_corners": box_corners        
    }

    plot_with_anns(**plot_params)