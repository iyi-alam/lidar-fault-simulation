#!/bin/bash
# Shell script

FAULT="fog"
NUSC_DATA_ROOT="/home/saksham/samsad/mtech-project/datasets/nuscenes"
PLOT_SAVE_PATH="/home/saksham/samsad/mtech-project/fault-sim/weather_sim/outputs"

FOG_ALPHA=0.04
RAIN_RATE=50.0
SNOW_RATE=1.5
DUST_LEVEL="moderate"

# Dynamically constructed paths
FOG_SAVE_PATH="${NUSC_DATA_ROOT}/${FAULT}/fog_alpha_${FOG_ALPHA}"
# RAIN_SAVE_PATH="${NUSC_DATA_ROOT}/${FAULT}/rain_rate_${RAIN_RATE}"
# SNOW_SAVE_PATH="${NUSC_DATA_ROOT}/${FAULT}/snow_rate_${SNOW_RATE}"
# DUST_SAVE_PATH="${NUSC_DATA_ROOT}/${FAULT}/dust_level_${DUST_LEVEL}"


 python3 simulator_nuscenes.py --fault "$FAULT" --nusc_root "$NUSC_DATA_ROOT" \
 --fog_alpha "$FOG_ALPHA" --fog_save_path "$FOG_SAVE_PATH" \
 --rain_rate "$RAIN_RATE" --rain_save_path "$RAIN_SAVE_PATH" \
 --snow_rate "$SNOW_RATE" --snow_save_path "$SNOW_SAVE_PATH" \
 --dust_level "$DUST_LEVEL" --dust_save_path "$DUST_SAVE_PATH" \
 --plot_save_path "$PLOT_SAVE_PATH" --save \
#  --max_samples 10 \
# --save

