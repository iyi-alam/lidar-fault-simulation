import numpy as np
import os
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map #type:ignore
from multiprocessing import Pool

def compute_occupancy(density, mean_radius):
    mean_radius = mean_radius * 100
    mean_vol = (4 * np.pi/3) * (mean_radius**3) # in cm3
    return mean_vol * density


def sample_r(mean_radius, std_radius, occupancy_ratio, R_0):

    total_area = (np.pi * R_0**2) * occupancy_ratio
    area = 0
    num_particles = 0
    lst = []

    var_dist = np.log(1 + std_radius**2/mean_radius**2)
    mean_dist = np.log(mean_radius) - var_dist/2
    sigma_dist = np.sqrt(var_dist)

    while (area < total_area):
        r = np.random.lognormal(mean=mean_dist, sigma=sigma_dist)
        d = np.random.uniform(0, R_0)
        theta = np.random.uniform(0, 2 * np.pi)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        lst.append(np.array([x, y, r]))
        area += np.pi * r**2
        num_particles += 1


        if num_particles % 10000 == 0:
            print(f"\rSampled Area: {area}/{total_area}, Num Particles: {num_particles}", end='', flush=True)

    print("\n")
    lst = np.stack(lst, axis=0)
    return num_particles, lst

def lognormal(mean, std):
    mean = np.log(mean)
    return np.random.lognormal(mean=mean, sigma=std)


def dart_throwing(mean_radius, std_radius, occupancy_ratio, R_0, show_progressbar, reduction_factor = 0.0005) -> np.ndarray:

    # Initialize samples to empty set.
    samples = []

    # Initialize occupied area to 0.
    area_occupied = 0.0

    # Calculate global occupied area across entire domain.
    area_occupied_global = (occupancy_ratio * reduction_factor) * np.pi * R_0 ** 2

    large_number = 1 / occupancy_ratio
    total = area_occupied_global * large_number + 1

    if show_progressbar:

        pbar = tqdm(total=total, desc='sampling particles',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    else:

        pbar = None

    var_dist = np.log(1 + std_radius**2/mean_radius**2)
    mean_dist = np.log(mean_radius) - var_dist/2
    sigma_dist = np.sqrt(var_dist)

    # Main sampling loop.
    while area_occupied < area_occupied_global:

        # Sample center of particle.
        length = np.sqrt(np.random.uniform(0, R_0 ** 2))
        angle = np.random.uniform(0,2) * np.pi

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        disk_radius = np.random.lognormal(mean = mean_dist, sigma = sigma_dist)

        # # Convert diameter to meters.
        # particle_diameter = 1 #particle_diameter / 1000

        # Sample height of particle center relative to examined plane.
        #height = np.random.uniform(-particle_diameter / 2, particle_diameter / 2)

        # Calculate radius of disk that constitutes the intersection of the sampled ball with the examined plane.
        # disk_radius = np.sqrt((particle_diameter / 2) ** 2 - height ** 2)
        #disk_radius = particle_diameter/2

        # If the disk includes the origin, reject the sample and continue.
        if x ** 2 + y ** 2 <= disk_radius ** 2:
            continue

        # # Check whether current particle overlaps with any particle that has already been sampled.
        # sample_has_overlap = (samples[:, 0] - x) ** 2 + (samples[:, 1] - y) ** 2 <= (samples[:, 2] + disk_radius) ** 2

        # # If yes, reject the sample and continue.
        # if np.any(sample_has_overlap):
        #     continue

        else:

            area = np.pi * disk_radius ** 2
            area_occupied += area
            samples.append(np.array([x, y, disk_radius]))
            # samples = np.concatenate((samples, ))

            if pbar:
                pbar.update(area * large_number)
                pbar.set_postfix({'n_sampled': len(samples)})

    if pbar:
        pbar.n = total
        pbar.close()

    return np.stack(samples, axis = 0)

def parallel_process(channel, output_dir, **sampling_params):
    print("Processing channel: ", channel)
    file_name = f"particles_{channel}.npy"
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        return
    else:
        samples = dart_throwing(**sampling_params)
        #assert samples.dim == 2, "Insufficient number of dimensions"
        np.save(file_path, arr=samples)
    return


if __name__ == "__main__":

    OUTPUT_DIR = "/home/saksham/samsad/mtech-project/fault-sim/LiDAR_dust_sim/particles"
    USE_MP = True
    density = 6082 # units/cm3
    mean_radius = 20e-6
    std_radius = mean_radius*0.35  # 0.35 factor is value chosen to match the distribution. The paper doesn't mention standard deviation
    occupancy_ratio = compute_occupancy(density=density, mean_radius=mean_radius)
    print(occupancy_ratio)

    R_0 = 80
    # num_particles, lst = sample_r(mean_radius, std_radius, occupancy_ratio*0.0005, R_0=R_0)
    # print(num_particles)
    # print("Max Radius: ", np.max(lst[:, 2]))
    # print("Min Radius: ", np.min(lst[:, 2]))
    # print("Mean Radius: ", np.mean(lst[:, 2]))
    # print("Total Samples: ", lst.shape[0])
    # np.save("sampled_particles.npy", arr=lst)

    channels = list(range(0, 64))
    sampling_params = {
        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "occupancy_ratio": occupancy_ratio,
        "R_0": R_0,
        "show_progressbar": False
    }

    partial_func = partial(parallel_process, output_dir=OUTPUT_DIR, **sampling_params)
    #process_map(partial_func, channels, max_workers=1, chunksize=1)

    if USE_MP:
        with Pool(processes=16) as p:
            for _ in p.imap_unordered(partial_func, channels):
                pass

    else: 
        for channel in channels:
            print("Processing Channel : ", channel+1)
            partial_func(channel=channel)



 