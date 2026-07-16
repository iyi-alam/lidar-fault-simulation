# Simulate Snow
cd /home/saksham/samsad/mtech-project/fault-sim/weather_fast_new

# # Simulate Light Snow
# echo "Simulating Light Snow..."
# python sim_snow_nusc_new.py \
#     --nusc_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval \
#     --save_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/light \
#     --samples_pkl_dir /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval/trainval_p1_samples.pkl \
#     --dataversion v1.0-trainval \
#     --snowfall_rate 1.0 \
#     --terminal_velocity 2.0 \
#     --mode gunn --alpha 0.0 --r_max 160.0 --reflectivity 0.8 \
#     --num_workers 40

# # Simulate Moderate Snow
# echo "Simulating Moderate Snow..."
# python sim_snow_nusc_new.py \
#     --nusc_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval \
#     --save_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/moderate_snow \
#     --samples_pkl_dir /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval/trainval_p1_samples.pkl \
#     --dataversion v1.0-trainval \
#     --snowfall_rate 2.5 \
#     --terminal_velocity 2.0 \
#     --mode gunn --alpha 0.04 --r_max 100.0 --reflectivity 0.85 \
#     --num_workers 20

# # Simulate Heavy Snow
# echo "Simulating Heavy Snow..."
# python sim_snow_nusc_new.py \
#     --nusc_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval \
#     --save_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/heavy_snow \
#     --samples_pkl_dir /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval/trainval_p1_samples.pkl \
#     --dataversion v1.0-trainval \
#     --snowfall_rate 5.0 \
#     --terminal_velocity 2.0 \
#     --mode gunn --alpha 0.12 --r_max 100 --reflectivity 0.95 \
#     --num_workers 20

# Simulate Rain
echo "Simulating Rain..."
python sim_rain_nusc_sweeps.py \
    --nusc_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval \
    --save_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/rain_50_all \
    --samples_pkl_dir /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval/trainval_p1_samples.pkl \
    --simulated_samples_root /home/saksham/samsad/mtech-project/datasets/nuscenes_part1/v1.0-trainval-sim/rain_50 \
    --dataversion v1.0-trainval \
    --num_workers 8
 