import numpy as np

gaussian_pos_noise = dict(
    noise_params = {
        "pos_noise": (0.5, 0.01 * np.pi/180, 0.01 * np.pi/180),
        "noise_probs": 0.2
    },

    noise_type = "gaussian",
    input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP",
    output_dir = None
)

uniform_pos_noise = dict(
    noise_params = {
        "pos_noise": (0.5, 0.01 * np.pi/180, 0.01 * np.pi/180),
        "noise_probs": 0.2
    },

    noise_type = "uniform",
    input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP",
    output_dir = None
)

impulse_pos_noise = dict(
    noise_params = dict(min_prob=0.1, min_val=0, max_prob=0.8, max_val=255),
    noise_type = "impulse",
    input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP",
    output_dir = None
)

sampling_noise = dict(
    upsample  = {
        "del_x": 5,          # Shift in X direction (float)
        "del_y": 5,         # Shift in Y direction (float)
        "del_z": 5,          # Shift in Z direction (float)
        "int_factor": 0.2,    # Maximum intensity gain of randomly upsampled points
        "sample_prob": 0.01,    # Probability of sampling (float between 0 and 1)
        "num_samples": 10    # Number of samples (integer)
    },

    downsample = {"percent_points": 0.9},
    input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP",
    output_dir = None
)

object_fail_params = dict(
    drop_prob = 1.0,
    input_dir = "/home/saksham/samsad/mtech-project/datasets/nuscenes"
)