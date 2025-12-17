import numpy as np
import PyMieScatt as ps #type:ignore
from scipy.stats import lognorm
from scipy.integrate import trapezoid

def compute_alpha(qext_arr, nd_arr, d_arr):
    integrand = d_arr**2 * qext_arr * nd_arr
    result = trapezoid(y=integrand, x=d_arr)
    return result * np.pi/8

def compute_beta(qsca_arr, nd_arr, d_arr):
    integrand = d_arr**2 * qsca_arr * nd_arr
    result = trapezoid(y=integrand, x=d_arr)
    return result * np.pi/8

def compute_nd(d_arr, dist, N_0):
    nd_arr = N_0 * dist.pdf(d_arr)
    return nd_arr

def compute_qext_qsca(m, wavelength, d_arr):
    d_arr = 1e7 * d_arr           # converted to nano meter
    qext_arr = np.zeros_like(d_arr)
    qsca_arr = np.zeros_like(d_arr)
    for i in range(len(qext_arr)):
        x = ps.MieQ(m=m, wavelength=wavelength, diameter = d_arr[i], asDict = True)
        qext_arr[i] = x["Qext"]
        qsca_arr[i] = x["Qsca"]

    return qext_arr, qsca_arr
    

if __name__ == "__main__":
    dia = 2 * 1000 #in nm
    m = 1.5+0.008j
    wavelength = 905 # in nm
    N_0 = 8803

    sample_mean = 25 * 1e-4 # in cm
    sample_std = 0.35 * sample_mean

    dist_sigma = np.sqrt(np.log(1 + sample_std**2/sample_mean**2))
    dist_mean = np.log(sample_mean) - dist_sigma**2/2

    dist = lognorm(s=dist_sigma, scale=np.exp(dist_mean))

    print(dist.pdf(x=sample_mean))

    d_arr = np.linspace(0.1, 100, 10000) * 1e-4  # in cm
    nd_arr = compute_nd(d_arr, dist, N_0 = N_0)
    qext_arr, qsca_arr = compute_qext_qsca(m=m, wavelength=wavelength, d_arr=d_arr)

    print(d_arr.shape, nd_arr.shape, qext_arr.shape)
    alpha = compute_alpha(qext_arr=qext_arr, nd_arr=nd_arr, d_arr=d_arr)
    beta = compute_beta(qsca_arr=qsca_arr, nd_arr=nd_arr, d_arr=d_arr)
    print(alpha, beta)

