import PyMieScatt as ps #type:ignore
import numpy as np
from scipy.integrate import trapezoid

#from scipy.integrate import trapz 


class MieScattering:
    def __init__(self, m, wavelength, rain_rate, dia_arr):
        self.d_arr = dia_arr # in mm
        self.lamda = 4.1 * np.power(rain_rate, -0.21) # in per mm
        self.m = m
        self.wavelength = wavelength


    def particle_size_dist(self):
        return (8/1000) * np.exp(-self.lamda * self.d_arr) # in per m. mm^3

    def compute_Qext(self):
        """
        D: particle diameter in mm
        """
        result = np.zeros_like(self.d_arr)
        for i, d in enumerate(self.d_arr):
            result[i] = ps.MieQ(m=self.m, wavelength = self.wavelength, diameter=d*1e6)[0]
        return result

    def compute_alpha(self):
        qext = self.compute_Qext()
        nd = self.particle_size_dist()
        y = self.d_arr**2 * qext * nd
        return trapezoid(y=y, x=self.d_arr)

if __name__ == "__main__":
    m =1.328 #refrective index
    wavelen = 905 # in nm
    rain_rate = 35 #in mm/hr
    dia = np.logspace(-5, 1, 10000, base=10) # in mm
    lamda = 4.1 * np.power(rain_rate, -0.21) # in per mm

    miescatt = MieScattering(m=m, wavelength=wavelen, rain_rate=rain_rate, dia_arr=dia)
    # nd = miescatt.particle_size_dist(dia)
    # print("Computing extinction co-efficients")
    # qext = miescatt.compute_Qext(dia)
    # print(qext.shape)
    print("Doing integration: ")
    alpha = miescatt.compute_alpha()
    print("alpha = ", alpha)

