import numpy as np



def car2sph(pc_xyz):
    r = np.linalg.norm(pc_xyz, axis=1)
    phi = np.arctan2(pc_xyz[:,1], pc_xyz[:,0])
    theta = np.arccos(pc_xyz[:,2] / (r + 1e-8))
    return np.stack([r, phi, theta], axis=1)

def sph2car(pc_sph):
    r, phi, theta = pc_sph[:,0], pc_sph[:,1], pc_sph[:,2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)

# Gaussian Noise
def gaussian_noise_cartesian(pc, scale):
    pc_new = pc.copy()
    noise = np.random.normal(0, scale, size=(pc.shape[0], 3))
    pc_new[:, :3] += noise
    return pc_new

def gaussian_noise_radial(pc, scale):
    pc_new = pc.copy()
    sph = car2sph(pc[:, :3])
    noise = np.random.normal(0, scale, size=sph.shape[0])
    sph[:, 0] += noise
    pc_new[:, :3] = sph2car(sph)
    return pc_new

# Uniform Noise
def uniform_noise_cartesian(pc, lims):
    pc_new = pc.copy()
    noise = np.random.uniform(-lims, lims, size=(pc.shape[0], 3))
    pc_new[:, :3] += noise
    return pc_new

def uniform_noise_radial(pc, lims):
    pc_new = pc.copy()
    sph = car2sph(pc[:, :3])
    noise = np.random.uniform(-lims, lims, size=sph.shape[0])
    sph[:, 0] += noise
    pc_new[:, :3] = sph2car(sph)
    return pc_new

# Impulse Noise
def impulse_noise_cartesian(pc, percentage, magnitude=0.2):
    pc_new = pc.copy()
    N = pc.shape[0]
    num = int(percentage * N)
    idx = np.random.choice(N, num, replace=False)
    signs = np.random.choice([-1, 1], size=(num, 3))
    pc_new[idx, :3] += signs * magnitude
    return pc_new

def impulse_noise_radial(pc, percentage, magnitude=0.4):
    pc_new = pc.copy()
    N = pc.shape[0]
    num = int(percentage * N)

    idx = np.random.choice(N, num, replace=False)
    sph = car2sph(pc[:, :3])

    signs = np.random.choice([-1, 1], size=num)
    sph[idx, 0] += signs * magnitude

    pc_new[:, :3] = sph2car(sph)
    return pc_new

# Background Noise
def background_noise(pc, percentage):
    pc_new = pc.copy()
    N = pc.shape[0]
    num = int(percentage * N)

    xyz_min = pc[:, :3].min(axis=0)
    xyz_max = pc[:, :3].max(axis=0)

    noise_xyz = np.random.uniform(xyz_min, xyz_max, size=(num, 3))
    noise_intensity = np.zeros((num, 1))

    noise_pc = np.hstack([noise_xyz, noise_intensity])
    return np.vstack([pc_new, noise_pc])

# Upsample Noise
def upsampling(pc, percentage, jitter=0.01):
    pc_new = pc.copy()
    N = pc.shape[0]
    num = int(percentage * N)

    idx = np.random.randint(0, N, size=num)
    sampled = pc[idx].copy()

    noise = np.random.uniform(-jitter, jitter, size=(num, 3))
    sampled[:, :3] += noise

    return np.vstack([pc_new, sampled])



NOISE_MAP = {
    "gaussian_cart": gaussian_noise_cartesian,
    "gaussian_rad": gaussian_noise_radial,
    "uniform_cart": uniform_noise_cartesian,
    "uniform_rad": uniform_noise_radial,
    "impulse_cart": impulse_noise_cartesian,
    "impulse_rad": impulse_noise_radial,
    "background": background_noise,
    "upsample": upsampling,
}