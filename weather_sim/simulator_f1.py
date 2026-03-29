"""
This file is meant to simulate faults in all the files of an input folder and save the simulated output in an output folder
"""
#%% Main
import os
import sys
# Add the parent directory (fault-sim) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pathlib import Path

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm.contrib.concurrent import process_map 
import argparse
import config_params

import weather_sim.config_params as config_params
from weather_sim.visualization_utils import plot_bev
import LiDAR_fog_sim.fog_simulation as fogsim
from LiDAR_snow_sim.tools.snowfall import sampling, simulation
from LiDAR_rain_sim import rain_sim
from LiDAR_dust_sim.fog_sim_method import mie_scatt, precompute_integration, dust_params
from LISA.pylisa.lisa import Lisa

GLOBAL_RESIZE_N = config_params.PC_FEATURE_DIMS #Some datasets can be resized with (-1,5) while some need to resize to (-1,4). This variable takes care of this.
MAX_INTENSITY_VAL = config_params.MAX_INTENSITY_VAL


class FogSim:
    
    def __init__(self, input_dir, output_dir, 
                 paramset: fogsim.ParameterSet,
                 simulation_options, save = True):
        self.input_dir = input_dir 
        self.output_dir = output_dir
        self.pset = paramset
        self.simulation_options = simulation_options
        self.save = save

    def _process_file(self, file_name, pset, fog_params):
        file_path = os.path.join(self.input_dir , file_name)
        pc = np.fromfile(file_path, dtype=np.float32).reshape((-1, GLOBAL_RESIZE_N))
        _pc = deepcopy(pc)
        if not MAX_INTENSITY_VAL:
            _pc[:,3] = _pc[:,3]*255.0

        # Simulate fog
        foggified_pc, _, fog_mask, _ = fogsim.simulate_fog(pset, _pc, **fog_params)

        if not MAX_INTENSITY_VAL:
            foggified_pc[:,3] /= 255.0
        assert (foggified_pc.shape[1] == GLOBAL_RESIZE_N)

        # Add fog mask as last column and remove channel's column
        foggified_pc = np.hstack([foggified_pc[:,:-1], fog_mask.reshape(-1, 1)])

        if self.save:
            save_path = os.path.join(self.output_dir, file_name)
            foggified_pc.astype('float32').tofile(save_path)
        #return pc, foggified_pc, file_name

    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)
        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        # Partial function with fixed pset and fog_params
        process_fn = partial(self._process_file, pset=self.pset, fog_params=self.simulation_options)

        with Pool(cpu_count()) as pool:
            if show_pbar:
                results = list(tqdm(pool.imap(process_fn, file_names[:N], chunksize=1), total=N))
            else:
                results = list(pool.imap(process_fn, file_names[:N], chunksize=1))

        # if self.save:
        #     for pc, foggified_pc, file_name in results:
        #         out_path = os.path.join(self.output_dir, file_name)
        #         foggified_pc.astype('float32').tofile(out_path)
        
        #return results


class SnowSim:
    def __init__(self, input_dir, output_dir, rainfall_rate, 
                     occupancy_ratio, mode, save = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rainfall_rate = rainfall_rate
        self.occupancy_ratio = occupancy_ratio
        self.mode = mode
        self.save = save
    
    def process_one_file(self, file_name):
    
        file_path = os.path.join(self.input_dir, file_name)
        save_path = os.path.join(self.output_dir, file_name)

        #print('Converting the point cloud: ', f'it: [{sweep_idx}]', file_name)
        points = np.fromfile(file_path, dtype = np.float32).reshape(-1,GLOBAL_RESIZE_N)
        pc = deepcopy(points)

        if not MAX_INTENSITY_VAL:
            pc[:,3] = pc[:,3]*255
        
        # Some of the datasets do not have a channel record. In that case add a dummy channel record and then later delete it
        # The simulator has been modified to not use channel record though
        if GLOBAL_RESIZE_N < 5:
            pc = np.concatenate((pc, np.zeros(shape=(pc.shape[0],1))), axis=1)
        snowflakes_file_prefix = f'{self.mode}_{self.rainfall_rate}_{self.occupancy_ratio}'
        augmented_pc = simulation.augment(pc = pc, particle_file_prefix= snowflakes_file_prefix,
                                        beam_divergence= float(np.degrees(3e-3)))
        
        if not MAX_INTENSITY_VAL:
            augmented_pc[:,3] /= 255.0

        if GLOBAL_RESIZE_N < 5:
            augmented_pc = augmented_pc[:,:GLOBAL_RESIZE_N]
        
        if self.save:
            augmented_pc.astype(np.float32).tofile(save_path)
        #return points, augmented_pc, file_name

    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)

        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        if show_pbar:
            results = process_map(self.process_one_file, file_names[:N], chunksize = 1)
        else:
            with Pool(processes= cpu_count()) as p:
                results = list(p.imap(self.process_one_file, file_names[:N]))
            p.close()
        
        # if self.save:
        #     for orig_pc, sim_pc, file_name in results:
        #         file_path = os.path.join(self.output_dir, file_name)
        #         sim_pc.astype(np.float32).tofile(file_path)
        # return results


class RainSim:
    def __init__(self, input_dir, output_dir, driver: rain_sim.rainSim, save = True):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.driver = driver
        self.save = save

    def process_one_file(self, file_name):
        file_path = os.path.join(self.input_dir, file_name)
        save_path = os.path.join(self.output_dir, file_name)
        points = np.fromfile(file_path, dtype = np.float32).reshape((-1,GLOBAL_RESIZE_N))
        pc = deepcopy(points)
        rain_pc, labels, stats_arr = self.driver.simulate(pc)
        if self.save:
            file_path = os.path.join(self.output_dir, file_name)
            rain_pc.astype(np.float32).tofile(save_path)
        return points, rain_pc, file_name
    
    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)

        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        if show_pbar:
            results = process_map(self.process_one_file, file_names[:N], chunksize = 1)
        else:
            with Pool(processes= cpu_count()) as p:
                results = list(p.imap(self.process_one_file, file_names[:N]))
            p.close()
        
        return results
    
class RainSim_New:
    def __init__(self, input_dir, output_dir, driver: Lisa, save = True):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.driver = driver
        self.save = save

    def process_one_file(self, file_name):
        file_path = os.path.join(self.input_dir, file_name)
        save_path = os.path.join(self.output_dir, file_name)
        points = np.fromfile(file_path, dtype = np.float32).reshape((-1,GLOBAL_RESIZE_N))
        pc = deepcopy(points)

        last_dims = pc[:, 4:]
        # We only need to send the first four features
        if GLOBAL_RESIZE_N > 4:
            pc = pc[:,:4]
        
        if MAX_INTENSITY_VAL:
            pc[:,3] = pc[:,3]/255.0 
            
        rain_pc = self.driver.simulate(pc)
        # rain_pc = rain_pc[:,:4]

        # if MAX_INTENSITY_VAL:
        #     rain_pc[:,3] = rain_pc[:,3]*255.0
        
        # # remove labels column and append the extra channel info
        # if GLOBAL_RESIZE_N > 4:
        #     rain_pc = np.concatenate((rain_pc, last_dims), axis = -1)

        if self.save:
            file_path = os.path.join(self.output_dir, file_name)
            rain_pc.astype(np.float32).tofile(save_path)
        return points, rain_pc, file_name
    
    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)

        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        if show_pbar:
            results = process_map(self.process_one_file, file_names[:N], chunksize = 1)
        else:
            with Pool(processes= cpu_count()) as p:
                results = list(p.imap(self.process_one_file, file_names[:N]))
            p.close()
        
        return results
    
class SnowSim_New:
    def __init__(self, input_dir, output_dir, driver: Lisa, save = True):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.driver = driver
        self.save = save

    def process_one_file(self, file_name):
        file_path = os.path.join(self.input_dir, file_name)
        save_path = os.path.join(self.output_dir, file_name)
        points = np.fromfile(file_path, dtype = np.float32).reshape((-1,GLOBAL_RESIZE_N))
        pc = deepcopy(points)

        last_dims = pc[:, 4:]
        # We only need to send the first four features
        if GLOBAL_RESIZE_N > 4:
            pc = pc[:,:4]
        
        if MAX_INTENSITY_VAL:
            pc[:,3] = pc[:,3]/255.0 
            
        rain_pc = self.driver.simulate(pc)
        rain_pc = rain_pc[:,:4]

        if MAX_INTENSITY_VAL:
            rain_pc[:,3] = rain_pc[:,3]*255.0
        
        # remove labels column and append the extra channel info
        if GLOBAL_RESIZE_N > 4:
            rain_pc = np.concatenate((rain_pc, last_dims), axis = -1)

        if self.save:
            file_path = os.path.join(self.output_dir, file_name)
            rain_pc.astype(np.float32).tofile(save_path)
        return points, rain_pc, file_name
    
    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)

        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        if show_pbar:
            results = process_map(self.process_one_file, file_names[:N], chunksize = 1)
        else:
            with Pool(processes= cpu_count()) as p:
                results = list(p.imap(self.process_one_file, file_names[:N]))
            p.close()
        
        return results


class DustSim:
    
    def __init__(self, input_dir, output_dir, 
                 paramset: fogsim.ParameterSet,
                 simulation_options, save = True):
        self.input_dir = input_dir 
        self.output_dir = output_dir
        self.pset = paramset
        self.simulation_options = simulation_options
        self.save = save

    def _process_file(self, file_name, pset, fog_params):
        file_path = os.path.join(self.input_dir , file_name)
        pc = np.fromfile(file_path, dtype=np.float32).reshape((-1, GLOBAL_RESIZE_N))
        foggified_pc, _, _ = fogsim.simulate_fog(pset, pc, **fog_params)
        return pc, foggified_pc, file_name

    def simulate(self, show_pbar, max_samples):
        file_names = os.listdir(self.input_dir)
        if max_samples:
            N = min(len(file_names), max_samples)
        else:
            N = len(file_names)

        # Partial function with fixed pset and fog_params
        process_fn = partial(self._process_file, pset=self.pset, fog_params=self.simulation_options)

        with Pool(cpu_count()) as pool:
            if show_pbar:
                results = list(tqdm(pool.imap(process_fn, file_names[:N], chunksize=1), total=N))
            else:
                results = list(pool.imap(process_fn, file_names[:N], chunksize=1))

        if self.save:
            for pc, foggified_pc, file_name in results:
                out_path = os.path.join(self.output_dir, file_name)
                foggified_pc.astype('float32').tofile(out_path)
        
        return results


def simulate_fog(args, fog_params, save = False, show_pbar = True, max_samples = None):
    parameter_set = fogsim.ParameterSet(alpha= fog_params["alpha"], gamma=fog_params["gamma"])
    simulator = FogSim(input_dir= args.input_dir,
                       output_dir= args.output_dir,
                       paramset= parameter_set,
                       simulation_options=fog_params["simulation_options"],
                       save = save)

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)

def simulate_snow(args, snow_params, save=False, show_pbar = True, max_samples = None):
    rainfall_rate = sampling.snowfall_rate_to_rainfall_rate(args.snow_rate, args.terminal_velocity)
    occupancy_ratio = sampling.compute_occupancy(args.snow_rate, args.terminal_velocity)

    rain_sim_args = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "rainfall_rate": rainfall_rate,        
        "occupancy_ratio": occupancy_ratio,      
        "mode": snow_params["mode"],            
        "save": save
    }

    simulator = SnowSim(**rain_sim_args)
    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)

def simulate_rain(args, show_pbar = True, max_samples = None, save = False):
    rain_cls_setter = rain_sim.RainParameters()
    rain_cls_setter.rain_rate = args.rain_rate
    sensor_cls_setter = rain_sim.SensorParameters()
    rain_cls_driver = rain_sim.rainSim(r_set = rain_cls_setter, 
                                       s_set = sensor_cls_setter,
                                       use_tqdm=False)
    
    simulator = RainSim(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        driver=rain_cls_driver,
        save = save
    )

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)

def simulate_rain_new(args, show_pbar = True, max_samples = None, save = False):
    '''
    Initialize LISA class
    Parameters
    ----------
    m           : refractive index contrast
    lam         : wavelength (nm)
    rmax        : max lidar range (m)
    rmin        : min lidar range (m)
    bdiv        : beam divergence angle (rad)
    dst         : droplet diameter starting point (mm)
    dR          : range accuracy (m)
    saved_model : use saved mie coefficients (bool)
    atm_model   : atmospheric model type
    mode        : lidar return mode: "strongest" or "last"

    '''
    lisa_driver = Lisa(
        rain_rate= args.rain_rate,
        m=1.328,
        rmax = 120,
        dst = 0.05,
        dR = 0.09,
        atm_model='rain'
    )

    simulator = RainSim_New(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        driver= lisa_driver,
        save = args.save
    )

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)

def simulate_snow_new(args, show_pbar = True, max_samples = None, save = False):
    '''
    Initialize LISA class
    Parameters
    ----------
    m           : refractive index contrast
    lam         : wavelength (nm)
    rmax        : max lidar range (m)
    rmin        : min lidar range (m)
    bdiv        : beam divergence angle (rad)
    dst         : droplet diameter starting point (mm)
    dR          : range accuracy (m)
    saved_model : use saved mie coefficients (bool)
    atm_model   : atmospheric model type
    mode        : lidar return mode: "strongest" or "last"

    '''
    lisa_driver = Lisa(
        snow_rate= args.snow_rate,
        m=1.328,
        rmax = 120,
        dst = 0.05,
        dR = 0.09,
        atm_model='snow'
    )

    simulator = SnowSim_New(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        driver= lisa_driver,
        save = args.save
    )

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)



def simulate_dust(args, show_pbar = True, max_samples = None, save = False):
    dust_config = config_params.dust_params
    dust_pset = dust_params.DustParams(mode = args.dust_level, compute_alpha_beta = True)
    print(f"Dust params: alpha = {dust_pset.alpha}, beta = {dust_pset.beta}")
    
    # Now precompute integration
    print("Precomputing integration lookup tables...")
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    integration_args = {
        "n_steps": 2000,
        "r_0_max": 200,
        "save_path": SCRIPT_DIR / "LiDAR_dust_sim" / "fog_sim_method" / "integral_lookup_tables",
        "n_cpus": cpu_count()
    }

    precompute_integration.compute_integration(alpha=dust_pset.alpha, beta=dust_pset.beta, arguments=integration_args)

    print("Simulating dust in lidar point cloud")
    # Now initiate parameter set equivalent of fog
    parameter_set = fogsim.ParameterSet(alpha= dust_pset.alpha, beta=dust_pset.beta ,gamma=0.000001)
    simulator = DustSim(input_dir= args.input_dir,
                       output_dir= args.output_dir,
                       paramset= parameter_set,
                       simulation_options=dust_config["simulation_options"],
                       save = save)

    return simulator.simulate(show_pbar=show_pbar, max_samples=max_samples)


def compare_plot(results, save_dir, save_name, plot_title, save = False):
    choice = np.random.choice(len(results))
    orig_pc, sim_pc, _ = results[choice]
    plot_bev(orig_pc, sim_pc, xlim=75, ylim=75,
             save = save, save_dir=save_dir,
             save_name=save_name, custom_title=plot_title)

def get_fault_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fault', type=str, default="all")
    parser.add_argument("--fog_alpha", type = float, default=0.04)
    parser.add_argument("--fog_gamma", type=float, default=1e-6)
    parser.add_argument("--rain_rate", type=float, default=50.0)
    parser.add_argument("--snow_rate", type=float, default=1.5)
    parser.add_argument("--lisa_snow", action='store_true', default=False)
    parser.add_argument("--terminal_velocity", type=float, default=1.6)
    parser.add_argument("--dust_level", type=str, default="moderate")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--plot_save_path")
    #parser.add_argument("--max_samples")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_fault_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #%% Simulate fog
    if args.fault == "fog" or args.fault == "all":
        #args.output_dir = os.path.join(args.output_dir, f"{args.fault}", f"fog_alpha_{args.fog_alpha}")
        os.makedirs(args.output_dir, exist_ok=True)
        fog_params = {
            "alpha": args.fog_alpha,
            "gamma": args.fog_gamma,
            "simulation_options": dict(
                noise = 10,
                gain = True,
                noise_variant = 'v1',
                hard = True,
                soft = True
            ),

        }
        fog_results = simulate_fog(args, fog_params = fog_params, max_samples=None, save=args.save, show_pbar=True)
        # print("Number of fog simulated results: ", len(fog_results))
        # fog_alpha = args.fog_alpha
        # print(f"Fog simulation with alpha = {fog_alpha}")
        # compare_plot(fog_results, 
        #              save_dir=args.plot_save_path,
        #              save_name=f"fog_simulated_{fog_alpha}.png",
        #              plot_title="Fog Simulated Point Cloud",
        #              save = args.save)

    #%% Simulate snow
    if args.fault == "snow" or args.fault == "all":
        snow_params = {
            "snowfall_rate": args.snow_rate,               #in mm/hr
            "terminal_velocity": args.terminal_velocity,            #in m/s
            "mode": "gunn"
        }
        if not args.lisa_snow:
            args.output_dir = os.path.join(args.output_dir, f"{args.fault}", f"snow_rate_{args.snow_rate}")
            os.makedirs(args.output_dir, exist_ok=True)
            snow_results = simulate_snow(args=args, snow_params=snow_params, save=args.save, show_pbar=True, max_samples=None)
        else:
            args.output_dir = os.path.join(args.output_dir, f"{args.fault}_lisa", f"snow_rate_{args.snow_rate}")
            os.makedirs(args.output_dir, exist_ok=True)
            # # Here we need to find equivalent rainfall rate first
            # equivalent_rain_rate = sampling.snowfall_rate_to_rainfall_rate(snowfall_rate=args.snow_rate, terminal_velocity=args.terminal_velocity)
            # args.snow_rate = equivalent_rain_rate
            snow_results = simulate_snow_new(args, max_samples=None, save=args.save)
        print("Number of snow simulated results: ", len(snow_results))
        compare_plot(snow_results, 
                    save_dir=args.plot_save_path,
                    save_name=f"snow_simulated_{args.snow_rate}.png",
                    plot_title="Snow Simulated Point Cloud",
                    save = args.save)

    #%% Simulate Rain
    if args.fault == "rain" or args.fault == "all":
        args.output_dir = os.path.join(args.output_dir, f"{args.fault}", f"rain_rate_{args.rain_rate}")
        os.makedirs(args.output_dir, exist_ok=True)
        rain_results = simulate_rain_new(args, max_samples=None, save=args.save)
        print("Number of rain simulated results: ", len(rain_results))
        compare_plot(rain_results, 
                    save_dir=args.plot_save_path,
                    save_name=f"rain_simulated_{args.rain_rate}.png",
                    plot_title="Rain Simulated Point Cloud",
                    save = args.save)

    #%% Simulate Homogeneous Dust
    if args.fault == "dust" or args.fault == "all":
        args.output_dir = os.path.join(args.output_dir, f"{args.fault}", f"dust_{args.dust_level}")
        os.makedirs(args.output_dir, exist_ok=True)
        dust_results = simulate_dust(args, max_samples=None, save=args.save)
        print("Number of dust simulated results: ", len(dust_results))
        compare_plot(dust_results, 
                    save_dir=args.plot_save_path,
                    save_name="dust_simulated.png",
                    plot_title="Dust Simulated Point Cloud",
                    save = args.save)
# %%
