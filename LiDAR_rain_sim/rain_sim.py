import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
from copy import deepcopy
from LiDAR_rain_sim import miescatt # to compute extinction coefficients using Mie Scattering Theory


class SensorParameters:
    '''
    Parameters:
        R_max: Maximum lidar range
        R_min: Minimum lidar range
        beam_div: beam divergence in radians
        del_R: Range accuracy
        D_st: Smallest particle diameter in rain to be considered to have effective scattering of lidar
        pmin: empirical minimum power of lidar
        tan_theta: value of tangent of beam divergence angle
    '''
    def __init__(self):
        self.wavelength = 905 # in nm
        self.R_max = 120 #m
        self.R_min = 1.5 #m
        self.beam_div = 3e-3 #beam divergence in radians
        self.del_R = 0.09 # in meters
        self.D_st = 5e-5 # in m
        self.pmin = 0.9 / self.R_max**2
        self.tan_theta = np.tan(self.beam_div)
        self.dR = 0.09 # Range accuracy in meter


class RainParameters:
    '''
    Parameters:
        rain_rate: rainfall rate in mm/hr
        n0: value of N_0 in Marshall Palmer Distribution of Rain particle diameters given by N(D) = N_0.exp(-lamda.D)
        lamda:P value of lamda in Marshall Palmer distribution
        alpha: value of extinction co-efficient
        rho: reflectivity of rain particles from Fresnel equation
    '''
    def __init__(self):
        self.rain_rate = 35 #unit mm/hr
        self.wavelength = SensorParameters().wavelength
        self.m = 1.328 #refrective index
        self.dia_arr = np.logspace(-5, 1, 1000)
        self.scatt = miescatt.MieScattering(m=self.m, wavelength=self.wavelength, rain_rate=self.rain_rate, dia_arr=self.dia_arr)
        self.alpha = self.compute_alpha() # in dB/meter
        self.ds = 0.05 # smallest particle diameter to consider in the units of mm
        self.n0 = 8000 #numbers per m^3.mm
        self.lamda = 4.1 * np.power(self.rain_rate, -0.21)
        self.particle_integ = (self.n0/self.lamda)*np.exp(-self.lamda * self.ds) #approximated value of integration coming in to estimate number of rain droplets
        self.rho = np.abs((self.m-1)/(self.m+1))**2
        self.drop_probs = 0.8

    def compute_alpha(self):
        return self.scatt.compute_alpha()
    

class rainSim:
    def __init__(self, r_set: RainParameters, s_set: SensorParameters, use_tqdm = False):
        self.r_set = r_set
        self.s_set = s_set
        self.snr = np.exp(-2*self.r_set.alpha*self.s_set.R_max)
        self.sigma_R = self.s_set.del_R/np.sqrt(2 * self.snr) * 0.1
        self.use_tqdm = use_tqdm
        self.max_i_1 = [] # Save maximum intensity of scattered points 
        self.max_i_2 = [] # Save maximum intsneity of original points attenuated
        self.num_1 = 0
        self.num_2 = 0
        self.perc_1 = []
        self.perc_2 = []
    
    
    def simulate_onept(self, pc):
        '''
        Args:
            pc: one point cloud having structure [x,y,z, intensity, ring_index] where x,y,z are spatial co-rodinates
            make sure the incoming intensity is already scaled in the range [0,1]
        '''
        x,y,z,i = pc[0], pc[1], pc[2],pc[3]
        R = np.sqrt(x**2+y**2+z**2)
        P_0 = i * np.exp(-2*self.r_set.alpha*R) / R**2
        snr = P_0/self.s_set.pmin

        beam_dia = R * self.s_set.tan_theta #beam dia is in meter
        Nt = 0
        if (R > self.s_set.R_min):
            bvol = (np.pi/3)*R*(beam_dia/2)**2
            Nt = int(bvol * self.r_set.particle_integ)

        R_rand = self.sample_r(R, Nt) #output samples are in meter
        valid_indx  = np.where(R_rand>self.s_set.R_min)[0]         # keep points where ranges larger than rmin
        Nt    = len(valid_indx)                       # new particle number

        if Nt > 0:
            R_rand = R_rand[valid_indx]
            D_rand = self.sample_d(Nt) / 1000   #output samples are in mm so make them in meter

            P_rand = self.r_set.rho * np.exp(-2*self.r_set.alpha*R_rand) /R_rand**2
            beam_dia_rand = R_rand * self.s_set.tan_theta

            #breakpoint()
            P_rand = P_rand * np.minimum((D_rand/beam_dia_rand)**2, 1)
            index = np.argmax(P_rand)

            if P_0<self.s_set.pmin and P_rand[index]<self.s_set.pmin: # if all smaller than Pmin, do nothing
                R_new = 0
                i_new = 0
                label    = 0 # label for lost point

            elif P_0<P_rand[index]: # scatterer has larger power
                R_new = R_rand[index] # new range is scatterer range
                i_new = (self.r_set.rho*np.exp(-2*self.r_set.alpha*R_new) * 
                        np.minimum((D_rand[index]/beam_dia_rand[index])**2,1)) # new reflectance biased by scattering
                label    = 1 # label for randomly scattered point 

            else: # object return has larger power
                sig     = self.s_set.dR/np.sqrt(2*snr)        # std of range uncertainty
                R_new = R + np.random.normal(0,sig) # range with uncertainty added
                i_new = i*np.exp(-2*self.r_set.alpha*R)      # new reflectance modified by scattering
                label    = 2 

        else:
            if P_0<self.s_set.pmin:
                R_new = 0
                i_new = 0
                label    = 0 # label for lost point
            else:
                sig     = self.s_set.dR/np.sqrt(2*snr)        # std of range uncertainty
                R_new = R + np.random.normal(0,sig) # range with uncertainty added
                i_new = i*np.exp(-2*self.r_set.alpha*R)      # new reflectance modified by scattering
                label    = 2                              # label for a non-scattering point
        
        theta = np.arccos(z/R)
        phi = np.arctan2(y, x)
        x,y,z = self.polar2cart(R_new, theta, phi)

        return np.array([x,y,z, i_new, pc[4]]), label


    def normalize_intensity(self, pc):
        imin, imax = np.min(pc[:,3]), np.max(pc[:,3])
        pc[:,3] = (pc[:,3] - imin)/(imax - imin)
        return pc
    
    def simulate(self, pc):
        # scale intensity to be in range [0,1]
        max_pc = np.max(pc[:, 3])
        pc[:,3] = pc[:,3] / max_pc
        simulated_pc = []
        labels = []
        if self.use_tqdm:
            itr = tqdm(range(pc.shape[0]), desc = 'Simulating rain...')
        else:
            itr = range(pc.shape[0])
        for i in itr:
            sim_pc, label = self.simulate_onept(pc[i])
            # Rescale simulated pc
            sim_pc[3] = np.clip(sim_pc[3]*max_pc, 0, max_pc)
            simulated_pc.append(sim_pc)
            labels.append(label)
        
        simulated_pc = np.stack(simulated_pc, axis = 0)
        labels = np.array(labels)

        # Save some statistics before returning
        max_i1 = np.max(simulated_pc[labels == 1, 3]) if len(simulated_pc[labels == 1, 3]) > 0 else 0.0
        max_i2 = np.max(simulated_pc[labels == 2 , 3])
        perc1 = sum(labels == 1) *100/ simulated_pc.shape[0]
        perc2 = sum(labels == 2) *100/ simulated_pc.shape[0]

        return simulated_pc, labels, np.array([max_i1, max_i2, perc1, perc2])
    
    def polar2cart(self, R, theta, phi):
        z = R * np.cos(theta)
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        return x, y, z
    
    def sample_r(self, R, num_samples):
        r = np.random.uniform(0,1, num_samples)
        return R * np.power(r, 1/3)
    
    def sample_d(self, num_samples):
        r = np.random.uniform(0,1, num_samples)
        return -np.log(1-r)/self.r_set.lamda + self.s_set.D_st # The output is in mm

def process_file(file_name, lidar_folder, save_folder, save = False):
    file_path = os.path.join(lidar_folder, file_name)
    save_path = os.path.join(save_folder, file_name)
    points = np.fromfile(file_path, dtype = np.float32).reshape((-1,5))
    pc = deepcopy(points)
    rain_pc, labels, stats_arr = sim.simulate(pc)
    if save:
        rain_pc.astype(np.float32).tofile(save_path)
        np.save(save_path.replace('samples', 'samples_labels').replace('.bin', '.npy'), arr = labels)
    return stats_arr

if __name__ == "__main__":
    import os

    USE_MP = True
    r_set = RainParameters()
    s_set = SensorParameters()
    sim = rainSim(r_set = r_set, s_set = s_set, use_tqdm = False)

    # Print some stats
    print("Lambda: ", r_set.lamda)
    print("Alpha: ", r_set.alpha)
    print("SNR: ", sim.snr)
    print("sigma_R: ", sim.sigma_R)

    lidar_folder = '/home/saksham/samsad/mtech-project/datasets/nuscenes/samples/LIDAR_TOP'
    save_folder = '/home/saksham/samsad/mtech-project/datasets/nuscenes_rain/samples/LIDAR_TOP'
    partial_process = partial(process_file, lidar_folder = lidar_folder, save_folder = save_folder, save = False)
    os.makedirs(save_folder, exist_ok= True)
    itr = os.listdir(lidar_folder)[:10] # Process only 100 files for experimentation
    stats_arr = []

    if USE_MP:
        with Pool(processes=10) as p:
            for return_val in tqdm(p.imap_unordered(partial_process, itr), total=len(itr)):
                stats_arr.append(return_val)
    else:
        for file_name in itr:
            return_val = partial_process(file_name=file_name)
            stats_arr.append(return_val)

    stats_arr = np.stack(stats_arr, axis = 0)

    # Post processing print maximum intensitities
    fig, _ = plt.subplots(1,1, figsize = (8, 6))
    plt.plot(stats_arr[:, 0], c = 'blue', label = "Maximum intensity of rain reflections")
    plt.plot(stats_arr[:,1], c = 'green', label = "Maximum intensity of original attenuated points")
    plt.xlabel("point cloud file")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    fig1, _ = plt.subplots(1,1, figsize = (8, 6))
    plt.plot(stats_arr[:, 2], c = 'blue', label = "Percentage of rain refleted points")
    plt.plot(stats_arr[:,3], c = 'green', label = "Percentage of of original attenuated points")
    plt.xlabel("point cloud file")
    plt.ylabel("Number of points in point cloud (%)")
    plt.legend()
    plt.show()