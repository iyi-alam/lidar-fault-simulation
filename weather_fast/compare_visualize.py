import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from weather_sim import visualization_utils
import numpy as np

main_folder = "/data/home/samsadalam/mtech_project/lidar-fault-simulation/data/nuscenes-mini"
sim_folder = "/data/home/samsadalam/mtech_project/lidar-fault-simulation/data/nuscenes-mini_snow"
plot_save_dir = "/data/home/samsadalam/mtech_project/lidar-fault-simulation/weather_fast/plots"

main_samples = os.path.join(main_folder, "samples/LIDAR_TOP")
sim_samples = os.path.join(sim_folder, "samples/LIDAR_TOP")

for i, sample_file in enumerate(os.listdir(sim_samples)):
    main_file = os.path.join(main_samples, sample_file)
    sim_file = os.path.join(sim_samples, sample_file)

    main_pc = np.fromfile(main_file, dtype=np.float32).reshape(-1, 5)
    sim_pc = np.fromfile(sim_file, dtype=np.float32).reshape(-1, 5)

    # Check label 2 and 3 counts
    label2_count_sim = np.sum(sim_pc[:, 4] == 2)
    label3_count_sim = np.sum(sim_pc[:, 4] == 3)
    total_count_sim = sim_pc.shape[0]
    print(f"Sample Index: {i+1} | Label 2 count: {label2_count_sim/total_count_sim*100:.2f}% | Label 3 count: {label3_count_sim/total_count_sim*100:.2f}%")
    visualization_utils.plot_bev(main_pc, sim_pc, save_dir=plot_save_dir, 
                                 save_name=sample_file.replace(".bin", ".png"), save=True,
                                 xlim=60, ylim=60, special_label=2)