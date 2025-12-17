import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import scipy.integrate as scipy_integrate #type:ignore
from scipy.constants import speed_of_light as c #type:ignore
from pathlib import Path
import multiprocessing as mp
from LiDAR_dust_sim.fog_sim_method.fog_simulation import ParameterSet
from tqdm import tqdm
import functools
import pickle
import os


def zeta(p, R: np.ndarray = None) -> np.ndarray:

    if R is None:
        R = p.r_0
    
    return np.clip((R - p.r_1)/(p.r_2 - p.r_1), 0, 1)



def inverse_square_modified(p, R: float, t: np.ndarray) -> np.ndarray:

    arr = np.zeros(t.shape)

    zero_pos = (t >= 2 * (R - p.r_1) / p.c)
    arr[~zero_pos] = 1 / ((R - (p.c * t[~zero_pos]) / 2) ** 2)

    return arr



def P_R_fog_soft(p, R: float, n: int = None) -> float:

    if n is None:
        n = p.n

    def integrand(t: np.ndarray) -> np.ndarray:
        arr = (np.sin(np.pi / (2 * p.tau_h) * t) ** 2) \
              * (np.exp(-2 * p.alpha * (R - ((c * t) / 2)))) \
              * inverse_square_modified(p, R, t) \
              * zeta(p, R - ((c * t) / 2)) \
              * np.heaviside(p.r_0 - R + (c * t) / 2, 0)

        return arr

    start = 0
    stop = 2 * p.tau_h

    x = np.linspace(start, stop, n)
    y = integrand(x)

    integral = scipy_integrate.simpson(y, x)

    return p.c_a * p.p_0 * p.beta * integral


def P_R_fog_soft_wrapper(p, R: float, n: int = None) -> float:

    if R > p.r_0:
        return 0            # skip unnecessary computation
    else:
        return P_R_fog_soft(p, R, n)


def compute_integration(alpha, beta, arguments):

    n = arguments["n_steps"]
    r_0_max = arguments["r_0_max"]

    save_path = Path(arguments["save_path"])
    pool = mp.Pool(arguments["n_cpus"])
    granularity = r_0_max / n

    filename = f'integral_0m_to_{r_0_max}m_stepsize_{granularity}m_tau_h_20ns_alpha_{np.round(alpha, 3)}.pickle'
    filepath = save_path / filename

    if os.path.exists(filepath):
        print("Integration lookup table already exists.")
        return 
    print(f'generating {filepath}')

    p = ParameterSet(n=n, r_range=r_0_max, alpha=alpha, beta=beta)

    integral = {}
    r_0 = 0

    steps = int(r_0_max / granularity)

    for _ in tqdm(range(steps + 1)):

        p.r_0 = round(r_0, 2)

        x_list = np.linspace(0, p.r_range, p.n)
        y_list = pool.map(functools.partial(P_R_fog_soft_wrapper, p), x_list)


        argmax = np.argmax(y_list)

        fog_distance = x_list[argmax]
        fog_response = y_list[argmax]

        fog_integral = fog_response / (p.c_a * p.p_0 * p.beta)

        integral[p.r_0] = (fog_distance, fog_integral)

        r_0 += granularity

    with open(filepath, 'wb') as f:
        pickle.dump(integral, f, protocol=pickle.HIGHEST_PROTOCOL)