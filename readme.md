# LiDAR Fault Simulation

A collection of **LiDAR point cloud corruption / fault simulation utilities** for generating degraded LiDAR data under sensor faults and adverse conditions. The goal of this repository is to help you **stress-test perception pipelines** (detection, tracking, mapping, segmentation) by injecting realistic and controllable failure modes into point clouds.

This repo is organized as a set of standalone simulators/scripts. Many components are designed to work with **nuScenes** via `nuscenes-devkit`.

---

## Repository structure

Top-level directories currently include:

- `weather_sim/`  
  Weather simulation tools and helper scripts. Notable files:
  - `simulator_nuscenes.py` – nuScenes-oriented simulation
  - `simulator_f1.py` – an alternative simulator implementation/variant
  - `simulate_random.py` – run randomized simulations
  - `visualization_utils.py` – plotting/visualization helpers
  - `run_weather_sim_*.sh` – convenience scripts for running experiments/sweeps
  - `config_params.py` – configuration parameters
  - `nuscenes_auto_sim.sh` – automation helper for nuScenes simulation runs

- `LiDAR_noise_sim/`  
  Noise simulation utilities. Notable files:
  - `scene_noise_simulate.py` – core scene noise simulator
  - `object_noise_simulate.py` – core object noise simulator
  - `run_scene_noise.py` – run scene noise (single run entry script)
  - `run_parallel_scene_noise.py` – parallel execution for scene noise
  - `run_parallel_object_noise.py` – parallel execution for object noise

- Additional modules (directories exist at repo root):
  - `LiDAR_fog_sim/`
  - `LiDAR_rain_sim/`
  - `LiDAR_snow_sim/`
  - `LiDAR_dust_sim/`
  - `LiDAR_density_sim/`
  - `LiDAR_cross_talk/`
  - `LiDAR_noise_corruptions/`
  - `LiDAR_object_failure/`
  - `LiDAR_object_transforms/`
  - `noise_sim/`
  - `LISA/`

---

## Environment / dependencies

This repo provides a Conda environment file:

- `environment.yml` (env name: `lidar-fault-sim`)
  - Python 3.10
  - Core scientific stack: NumPy, SciPy, pandas, matplotlib, scikit-learn, tqdm
  - Geometry/vision: shapely, opencv-python
  - 3D tooling: open3d
  - Quaternion utils: pyquaternion
  - Dataset tooling: nuscenes-devkit==1.1.10

### Setup (recommended)

```bash
# 1) clone
git clone https://github.com/iyi-alam/lidar-fault-simulation.git
cd lidar-fault-simulation

# 2) create and activate environment
conda env create -f environment.yml
conda activate lidar-fault-sim
```

If you do not use Conda, install equivalents via pip/venv, but note that packages like **Open3D** can be platform-sensitive.

---

## Quickstart

Because this repository contains multiple independent modules, the most direct way to start is to run the module scripts from their folders.

### 1) Weather simulation (shell helpers)

Explore the scripts under `weather_sim/`:

```bash
cd weather_sim
ls *.sh
```

Common patterns:
- `run_weather_sim_general.sh` – run a general simulation workflow
- `run_weather_sim_nuscenes.sh` – run with nuScenes integration
- `run_weather_sim_nusc_sweep.sh` – run parameter sweeps (multiple settings)

Run one (example):
```bash
bash run_weather_sim_general.sh
```

> These scripts may expect dataset paths (e.g., nuScenes root) or output directories to be set inside the script or via environment variables. Open the `.sh` file to confirm required arguments.

### 2) Noise simulation (Python entry scripts)

Scene-level noise:
```bash
python LiDAR_noise_sim/run_scene_noise.py
```

Parallel variants:
```bash
python LiDAR_noise_sim/run_parallel_scene_noise.py
python LiDAR_noise_sim/run_parallel_object_noise.py
```

> If these scripts require configuration (paths, split selection, parameter ranges), it will typically be defined in the script itself or in helper modules in the same directory.

---

## Using nuScenes

Some components in this repo depend on `nuscenes-devkit`. You will typically need:

- nuScenes dataset downloaded locally
- correct dataset root path configured (commonly via constants or CLI args in scripts)

If you plan to run nuScenes-based simulations:
1. Install the environment (see above)
2. Locate `weather_sim/simulator_nuscenes.py` and any `run_weather_sim_nuscenes*.sh` scripts
3. Update dataset paths and output paths accordingly

---

## Output conventions (general guidance)

Different simulators may output one or more of:
- Corrupted point clouds (e.g., `.bin`, `.pcd`, `.ply`, or NumPy arrays)
- Logs/metadata describing injected faults and their parameters
- Visualizations (images/plots) if enabled

Check the specific module scripts you run to confirm:
- output directory layout
- file formats
- naming conventions

---

## Extending the repo (adding a new fault)

A typical pattern for adding a new corruption module:

1. Create a new directory (e.g., `LiDAR_<fault>_sim/`)
2. Implement:
   - a core simulator function/module (pure Python, operates on point arrays)
   - a runnable script (`run_<fault>.py`) that loads data, applies corruption, saves output
3. (Optional) add:
   - parallel runner for batch processing
   - visualization helpers
   - `.sh` wrappers for reproducible experiments

If you do this, consider documenting:
- required input format
- parameter definitions and recommended ranges
- expected output format

---

## Contributing

Issues and pull requests are welcome. If you submit a PR, please include:
- what fault model you implemented/changed
- a minimal command to reproduce
- example output artifacts (or screenshots) if relevant

---


