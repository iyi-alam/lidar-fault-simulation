import numpy as np
from scipy.stats import lognorm
from . import mie_scatt


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