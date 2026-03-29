import os
import numpy as np
import matplotlib.pyplot as plt


# IMPORT YOUR TRANSFORMATIONS

from object_transforms import MAP as OBJ_MAP
from object_transforms import get_lidar, process_1_scan



# PLOT FUNCTION (BEV)

def plot_one(pc, save_path, title="Point Cloud", xlim=50, ylim=50):

    x = pc[:, 0]
    y = pc[:, 1]

    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, c='gray', s=1.0)

    plt.axis("equal")
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    plt.xlabel("X (meters)", fontsize=32)
    plt.ylabel("Y (meters)", fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()



# LOAD CLEAN POINT CLOUD

def load_clean_pc(root_path, sample_idx):
    pc = get_lidar(root_path, sample_idx)
    return pc



# APPLY TRANSFORMATION

def apply_transformation(root_path, sample_idx, transform_type, severity):

    assert transform_type in OBJ_MAP, f"{transform_type} not supported"

    pc_transformed = OBJ_MAP[transform_type](root_path, sample_idx, severity)

    return pc_transformed



# MAIN

if __name__ == "__main__":

    ROOT = "/home/saksham/samsad/mtech-project/datasets/kitti/"  # <-- update

    save_dir = "kitti_object_transform_results"
    os.makedirs(save_dir, exist_ok=True)

    transform_types = [
        "shear_obj",
        "scale_obj",
        "translocate_obj",
        "rotation_obj",
        # "distortion_ffd_obj",
        # "distortion_rbf_obj",
        # "distortion_rbf_inv_obj"
    ]

    # KITTI sample indices (assuming training split)
    velodyne_dir = os.path.join(ROOT, "training", "velodyne")
    sample_files = sorted(os.listdir(velodyne_dir))

    # process first N samples
    NUM_SAMPLES = 10

    for k, file_name in enumerate(sample_files[:NUM_SAMPLES]):

        sample_idx = file_name.split(".")[0]

        print(f"Processing sample {sample_idx}")

        
        # Load clean
        
        pc_clean = load_clean_pc(ROOT, sample_idx)

        plot_one(
            pc_clean,
            save_path=f"{save_dir}/{sample_idx}_clean.png",
            xlim=75, ylim=75
        )

        
        # Apply transformations
        
        for transform in transform_types:

            try:
                pc_transformed = apply_transformation(
                    ROOT,
                    sample_idx,
                    transform,
                    severity=3
                )

                plot_one(
                    pc_transformed,
                    save_path=f"{save_dir}/{sample_idx}_{transform}.png",
                    xlim=75, ylim=75
                )

            except Exception as e:
                print(f"Skipping {transform} for {sample_idx}: {e}")

    print(f"Done. Results saved to: {save_dir}")