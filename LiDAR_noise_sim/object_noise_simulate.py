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


def gaussian_noise_object(pc, object_flag, scale):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    noise = np.random.normal(0, scale, size=(mask.sum(), 3))

    pc_new[mask, :3] += noise
    return pc_new

def uniform_noise_object(pc, object_flag, lims):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    noise = np.random.uniform(-lims, lims, size=(mask.sum(), 3))

    pc_new[mask, :3] += noise
    return pc_new

def impulse_noise_object(pc, object_flag, percentage, magnitude=0.2):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    idx = np.where(mask)[0]

    num = int(percentage * len(idx))
    if num == 0:
        return pc_new

    chosen = np.random.choice(idx, num, replace=False)
    signs = np.random.choice([-1, 1], size=(num, 3))

    pc_new[chosen, :3] += signs * magnitude
    return pc_new

def gaussian_noise_radial_object(pc, object_flag, scale):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    sph = car2sph(pc[:, :3])

    noise = np.random.normal(0, scale, size=mask.sum())
    sph[mask, 0] += noise

    pc_new[:, :3] = sph2car(sph)
    return pc_new

def uniform_noise_radial_object(pc, object_flag, lims):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    sph = car2sph(pc[:, :3])

    noise = np.random.uniform(-lims, lims, size=mask.sum())
    sph[mask, 0] += noise

    pc_new[:, :3] = sph2car(sph)
    return pc_new

def impulse_noise_radial_object(pc, object_flag, percentage, magnitude=0.4):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    idx = np.where(mask)[0]

    num = int(percentage * len(idx))
    if num == 0:
        return pc_new

    chosen = np.random.choice(idx, num, replace=False)

    sph = car2sph(pc[:, :3])
    signs = np.random.choice([-1, 1], size=num)

    sph[chosen, 0] += signs * magnitude

    pc_new[:, :3] = sph2car(sph)
    return pc_new

def background_noise_object(pc, object_flag, percentage):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    obj_points = pc[mask, :3]

    if obj_points.shape[0] == 0:
        return pc_new

    num = int(percentage * obj_points.shape[0])

    xyz_min = obj_points.min(axis=0)
    xyz_max = obj_points.max(axis=0)

    noise_xyz = np.random.uniform(xyz_min, xyz_max, size=(num, 3))
    noise_intensity = np.zeros((num, 1))

    noise_pc = np.hstack([noise_xyz, noise_intensity])

    return np.vstack([pc_new, noise_pc])


def upsample_object(pc, object_flag, percentage, jitter=0.01):
    pc_new = pc.copy()

    mask = object_flag.astype(bool)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return pc_new

    num = int(percentage * len(idx))
    chosen = np.random.choice(idx, num, replace=True)

    sampled = pc[chosen].copy()
    noise = np.random.uniform(-jitter, jitter, size=(num, 3))

    sampled[:, :3] += noise

    return np.vstack([pc_new, sampled])

OBJECT_NOISE_MAP = {
    "gaussian_cart": gaussian_noise_object,
    "uniform_cart": uniform_noise_object,
    "impulse_cart": impulse_noise_object,
    "gaussian_rad": gaussian_noise_radial_object,
    "uniform_rad": uniform_noise_radial_object,
    "impulse_rad": impulse_noise_radial_object,
    "background": background_noise_object,
    "upsample": upsample_object,
}