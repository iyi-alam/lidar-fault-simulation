import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from weather_sim import visualization_utils
import numpy as np

main_folder = "/data/home/samsadalam/mtech_project/datasets/v1.0-trainval"
sim_folder = "/data/home/samsadalam/mtech_project/datasets/v1.0-trainval-sim/rain_50"
plot_save_dir = "/data/home/samsadalam/mtech_project/lidar-fault-simulation/weather_fast/plots"

main_samples = os.path.join(main_folder, "samples/LIDAR_TOP")
sim_samples = os.path.join(sim_folder, "samples/LIDAR_TOP")

for i, sample_file in enumerate(os.listdir(sim_samples)):
    main_file = os.path.join(main_samples, sample_file)
    sim_file = os.path.join(sim_samples, sample_file)

    main_pc = np.fromfile(main_file, dtype=np.float32).reshape(-1, 5)
    sim_pc = np.fromfile(sim_file, dtype=np.float32).reshape(-1, 5)

    # Check label 0 and 1 counts
    label0_count_sim = np.sum(sim_pc[:, 4] == 0)
    label1_count_sim = np.sum(sim_pc[:, 4] == 1)
    total_count_sim = sim_pc.shape[0]
    print(f"Sample Index: {i+1} | Drop count: {label0_count_sim/total_count_sim*100:.2f}% | Scatter count: {label1_count_sim/total_count_sim*100:.2f}%")
    visualization_utils.plot_bev(main_pc, sim_pc, save_dir=plot_save_dir, 
                                 save_name=sample_file.replace(".bin", ".png"), save=True,
                                 xlim=60, ylim=60)