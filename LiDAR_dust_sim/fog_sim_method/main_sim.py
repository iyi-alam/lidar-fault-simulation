import numpy as np
import fog_simulation as fogsim
from tqdm import tqdm
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from functools import partial
import mie_scatt
import precompute_integration as integration
from scipy.stats import lognorm
import multiprocessing as mp


class DustParams:
    def __init__(self, mode = "moderate", compute_alpha_beta = True):
        m_real = 1.5
        m_img = 0.008j
        self.m = m_real + m_img

        if mode == "light":
            mean_dia = 15 * 1e-4 #in cm
            std_dia = 0.35 * mean_dia
            N_0 = 3505
        
        if mode == "heavy":
            mean_dia = 25 * 1e-4 #in cm
            std_dia = 0.35 * mean_dia
            N_0 = 8803

        else: # mode == "moderate"
            mean_dia = 20 * 1e-4 #in cm
            std_dia = 0.35 * mean_dia
            N_0 = 6082
        
        self.N_0 = N_0
        dist_sigma = np.sqrt(np.log(1 + std_dia**2/mean_dia**2))
        dist_mean = np.log(mean_dia) - dist_sigma**2/2

        self.dist = lognorm(s=dist_sigma, scale=np.exp(dist_mean))
        self.wavelength = 905 # in nm
        self.alpha = 0.02
        self.beta = 0.001

        if compute_alpha_beta:
            d_arr = d_arr = np.linspace(0.1, 100, 10000) * 1e-4  # in cm
            print("Computing extinction and scattering coefficient...")
            self.compute_alpha_beta(d_arr)
            
    
    def compute_alpha_beta(self, d_arr):
        nd_arr = mie_scatt.compute_nd(d_arr, self.dist, N_0 = self.N_0)
        qext_arr, qsca_arr = mie_scatt.compute_qext_qsca(m=self.m, wavelength=self.wavelength, d_arr=d_arr)

        print(d_arr.shape, nd_arr.shape, qext_arr.shape)
        self.alpha = mie_scatt.compute_alpha(qext_arr=qext_arr, nd_arr=nd_arr, d_arr=d_arr)
        self.beta = mie_scatt.compute_beta(qsca_arr=qsca_arr, nd_arr=nd_arr, d_arr=d_arr)
        return 


class DustSim:
    DATA_PATH = '/home/saksham/samsad/mtech-project/datasets/nuscenes'
    
    def __init__(self, new_extension, paramset: fogsim.ParameterSet, use_tqdm = False):
        self.new_extension = new_extension # only the change in extension, not the complete path
        self.use_tqdm = use_tqdm
        self.pset = paramset

    def _process_file(self, file_path, pset, fog_params):
        pc = np.fromfile(file_path, dtype=np.float32).reshape((-1, 5))
        foggified_pc, _, _ = fogsim.simulate_fog(pset, pc, **fog_params)
        return foggified_pc, file_path

    def simulate(self,  fog_params, save = False, max_samples = None, mode = 'samples'):
        lidar_path = os.path.join (self.DATA_PATH , mode , 'LIDAR_TOP')
        file_names = os.listdir(lidar_path)
        N = len(file_names)

        if max_samples is not None:
            N = min(max_samples, N)

        if self.use_tqdm:
            itr = tqdm(range(N), desc=f'Foggifying point cloud - {mode}...')
        else:
            itr = range(N)

        file_paths = [os.path.join(lidar_path, file_names[i]) for i in range(N)]

        # Partial function with fixed pset and fog_params
        process_fn = partial(self._process_file, pset=self.pset, fog_params=fog_params)

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_fn, file_paths), total=len(file_paths)))

        foggified_samples = []
        for foggified_pc, file_path in results:
            foggified_samples.append(foggified_pc)
            if save:
                out_path = file_path.replace('nuscenes', self.new_extension)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                foggified_pc.astype('float32').tofile(out_path)

        return foggified_samples
            
    
if __name__ == '__main__':
    
    ## WE ARE REUSING THE SAME CODE FOR DUST SIMULATION WITH LITTLE BIT CHANGES ONLY
    ## WE NEED TO RECOMPUTE ALPHA AND BETA FOR DUST USING SCATTERING THEORY
    ## CREATE A SEPARATE PARAMETER SET FOR DUST

    dust_pset = DustParams(mode = "heavy", compute_alpha_beta = False)
    dust_pset.alpha = 0.051
    dust_pset.beta = 0.029
    print(f"Dust params: alpha = {dust_pset.alpha}, beta = {dust_pset.beta}")
    
    # Now precompute integration
    print("Precomputing integration lookup tables...")
    integration_args = {
        "n_steps": 2000,
        "r_0_max": 200,
        "save_path": "/home/saksham/samsad/mtech-project/fault-sim/LiDAR_dust_sim/fog_sim_method/integral_lookup_tables",
        "n_cpus": mp.cpu_count()
    }

    integration.compute_integration(alpha=dust_pset.alpha, beta=dust_pset.beta, arguments=integration_args)

    print("Simulating dust in lidar point cloud")
    # Now initiate parameter set equivalent of fog
    parameter_set = fogsim.ParameterSet(alpha= dust_pset.alpha, beta=dust_pset.beta ,gamma=0.000001)
    simulator = DustSim(new_extension = 'nuscenes_dust', paramset= parameter_set, use_tqdm= True)

    fog_params = dict(
        noise = 10,
        gain = False,
        noise_variant = 'v1',
        hard = True,
        soft = True
    )

    # converted_pc = nusc_conv.convert(fog_params= fog_params, save = True, mode = 'sweeps')
    # print('Successfully converted the NUSC LIDAR TOP SWEEP')

    converted_pc = simulator.simulate(fog_params= fog_params, save = True, mode = 'samples')
    print("Dust simulation complete!")