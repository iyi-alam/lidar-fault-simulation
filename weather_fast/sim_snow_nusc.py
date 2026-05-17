import os
import sys
import argparse
from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
from tqdm.contrib.concurrent import process_map
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import weather_sim.config_params as config_params
import LiDAR_fog_sim.fog_simulation as fogsim
from LiDAR_snow_sim.tools.snowfall import sampling, simulation
from LiDAR_rain_sim import rain_sim
from LiDAR_dust_sim.fog_sim_method import precompute_integration, dust_params


class BaseSim:
    def simulate(self, pc: np.ndarray):
        return pc


class SnowSim(BaseSim):
    def __init__(self, snowfall_rate, terminal_velocity, mode = "gunn"):
        terminal_velocity = terminal_velocity
        self.rainfall_rate = sampling.snowfall_rate_to_rainfall_rate(
            snowfall_rate,
            terminal_velocity,
        )
        self.occupancy_ratio = sampling.compute_occupancy(
            snowfall_rate,
            terminal_velocity,
        )
        self.mode = mode

    def simulate(self, pc: np.ndarray):
        pc = deepcopy(pc)
        snowflakes_file_prefix = f'{self.mode}_{self.rainfall_rate}_{self.occupancy_ratio}'
        return simulation.augment(
            pc=pc,
            particle_file_prefix=snowflakes_file_prefix,
            beam_divergence=float(np.degrees(3e-3)),
        )


class NuscenesSnowTemporalSimulator:
    def __init__(self, dataroot, save_root, 
                 dataversion="v1.0-trainval", 
                 snowfall_rate=1.5, terminal_velocity=0.5, mode="gunn",
                 sweeps_per_sample=10):
        self.dataroot = dataroot
        self.save_root = save_root
        self.nusc = NuScenes(version=dataversion, dataroot=dataroot, verbose=True)
        self.snow_sim = SnowSim(snowfall_rate, terminal_velocity, mode=mode)
        self.sweeps_per_sample = sweeps_per_sample

    def ensure_parent_dir(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load_pc(self, rel_path):
        abs_path = os.path.join(self.dataroot, rel_path)
        return np.fromfile(abs_path, dtype=np.float32).reshape((-1, 5))

    def save_pc(self, rel_path, pc):
        abs_path = os.path.join(self.save_root, rel_path)
        self.ensure_parent_dir(abs_path)
        pc.astype(np.float32).tofile(abs_path)

    def get_sd_record(self, token):
        return self.nusc.get("sample_data", token)

    def get_lidar_to_global(self, sample_data_token):
        sd = self.get_sd_record(sample_data_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])

        T_sensor = np.eye(4, dtype=np.float32)
        T_sensor[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
        T_sensor[:3, 3] = np.array(cs["translation"], dtype=np.float32)

        T_ego = np.eye(4, dtype=np.float32)
        T_ego[:3, :3] = Quaternion(ego["rotation"]).rotation_matrix
        T_ego[:3, 3] = np.array(ego["translation"], dtype=np.float32)

        return T_ego @ T_sensor

    def transform_points_xyz(self, xyz, T):
        if xyz.shape[0] == 0:
            return xyz
        pts = np.ones((xyz.shape[0], 4), dtype=np.float32)
        pts[:, :3] = xyz
        transformed = (T @ pts.T).T
        return transformed[:, :3]

    def transfer_snow_particles(self, sample_snow_pts, sample_token, sweep_token, jitter_std=0.1):
        if sample_snow_pts.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float32)

        T_sample = self.get_lidar_to_global(sample_token)
        T_sweep = self.get_lidar_to_global(sweep_token)
        T_global_to_sweep = np.linalg.inv(T_sweep)

        snow_global_xyz = self.transform_points_xyz(sample_snow_pts[:, :3], T_sample)
        snow_sweep_xyz = self.transform_points_xyz(snow_global_xyz, T_global_to_sweep)
        snow_sweep_xyz += np.random.normal(0.0, jitter_std, snow_sweep_xyz.shape).astype(np.float32)

        out = sample_snow_pts.copy()
        out[:, :3] = snow_sweep_xyz
        out[:, 4] = 2
        return out

    def compute_spherical_bins(self, xyz, az_bins, el_bins, range_bins, max_range):
        r = np.linalg.norm(xyz, axis=1)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        el = np.arctan2(xyz[:, 2], xy)

        az_idx = np.clip(((az + np.pi) / (2 * np.pi) * az_bins).astype(np.int32), 0, az_bins - 1)
        el_idx = np.clip(((el + np.pi / 2) / np.pi * el_bins).astype(np.int32), 0, el_bins - 1)
        r_idx = np.clip((r / max_range * range_bins).astype(np.int32), 0, range_bins - 1)
        return az_idx, el_idx, r_idx

    def build_drop_probability_map(self, orig_pc, sim_pc, az_bins=180, el_bins=32, range_bins=20, max_range=80.0):
        dropped = sim_pc[:, 4] == 3
        xyz = orig_pc[:, :3]
        az_idx, el_idx, r_idx = self.compute_spherical_bins(
            xyz,
            az_bins,
            el_bins,
            range_bins,
            max_range,
        )

        total = np.zeros((az_bins, el_bins, range_bins), dtype=np.float32)
        dropped_count = np.zeros_like(total)

        for i in range(xyz.shape[0]):
            total[az_idx[i], el_idx[i], r_idx[i]] += 1.0
            if dropped[i]:
                dropped_count[az_idx[i], el_idx[i], r_idx[i]] += 1.0

        prob = np.zeros_like(total)
        valid = total > 0
        prob[valid] = dropped_count[valid] / total[valid]
        return prob

    def apply_drop_probability_map(self, sweep_pc, prob_map, max_range=80.0):
        if sweep_pc.shape[0] == 0:
            return sweep_pc

        az_bins, el_bins, range_bins = prob_map.shape
        xyz = sweep_pc[:, :3]
        az_idx, el_idx, r_idx = self.compute_spherical_bins(
            xyz,
            az_bins,
            el_bins,
            range_bins,
            max_range,
        )

        probs = prob_map[az_idx, el_idx, r_idx]
        rand = np.random.rand(sweep_pc.shape[0])
        drop_mask = rand < probs

        out = sweep_pc.copy()
        out[drop_mask, 0:4] = 0.0
        out[drop_mask, 4] = 3
        return out

    def process_sample(self, sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        key_sd = self.get_sd_record(lidar_token)

        key_orig = self.load_pc(key_sd["filename"])
        key_sim = self.snow_sim.simulate(key_orig.copy())
        self.save_pc(key_sd["filename"], key_sim)

        sample_snow_pts = key_sim[key_sim[:, 4] == 2].copy()
        drop_prob_map = self.build_drop_probability_map(key_orig, key_sim)

        prev_token = key_sd["prev"]
        sweep_count = 0
        while prev_token != "" and sweep_count < self.sweeps_per_sample:
            sweep_sd = self.get_sd_record(prev_token)
            sweep_pc = self.load_pc(sweep_sd["filename"])

            dropped_sweep = self.apply_drop_probability_map(sweep_pc, drop_prob_map)
            transferred_snow = self.transfer_snow_particles(
                sample_snow_pts,
                lidar_token,
                sweep_sd["token"],
            )

            if transferred_snow.shape[0] > 0:
                final_sweep = np.vstack([dropped_sweep, transferred_snow])
            else:
                final_sweep = dropped_sweep

            self.save_pc(sweep_sd["filename"], final_sweep)
            prev_token = sweep_sd["prev"]
            sweep_count += 1

    def run(self, max_samples=None):
        samples = self.nusc.sample
        if max_samples is not None:
            idx = np.random.permutation(len(samples))[:max_samples]
            samples = [samples[i] for i in idx]

        print("Simulating Snow on {} samples...".format(len(samples)))
        process_map(
            self.process_sample,
            samples,
            chunksize=1,
            max_workers=20 #cpu_count(),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_root", required=True, type=str)
    parser.add_argument("--save_root", required=True, type=str)
    parser.add_argument("--dataversion", default="v1.0-trainval", type=str)
    parser.add_argument("--snowfall_rate", default=1.5, type=float)
    parser.add_argument("--terminal_velocity", default=0.5, type=float)
    parser.add_argument("--mode", default="gunn", type=str)

    parser.add_argument("--max_samples", default=None, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    simulator = NuscenesSnowTemporalSimulator(
        dataroot=args.nusc_root,
        save_root=args.save_root,
        dataversion=args.dataversion,
        snowfall_rate=args.snowfall_rate,
        terminal_velocity=args.terminal_velocity,
        mode=args.mode
    )

    simulator.run(max_samples=args.max_samples)
