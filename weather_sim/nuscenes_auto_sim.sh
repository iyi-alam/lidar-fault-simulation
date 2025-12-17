#!/bin/bash
# Automated NuScenes Weather Fault Simulation Runner
# Author: (MD Samsad Alam)

######## USER SETTINGS ########
NUSC_DATA_ROOT="/home/saksham/samsad/mtech-project/datasets/nuscenes"
PLOT_SAVE_PATH="/home/saksham/samsad/mtech-project/fault-sim/weather_sim/outputs"

# Sweep ranges for each weather type
FOG_ALPHAS=(0.01 0.02 0.04 0.06 0.1)
RAIN_RATES=(5 20 50 80 100)
SNOW_RATES=(0.5 1.0 2.0 2.5 1.5)
DUST_LEVELS=("light" "moderate" "heavy")

######## SCRIPT START ########

echo "=== Starting Automated Weather Simulation ==="
echo "NuScenes Root: $NUSC_DATA_ROOT"
echo "Plots Save Path: $PLOT_SAVE_PATH"
echo ""

mkdir -p "$PLOT_SAVE_PATH"

###############################################
# 1. FOG SIMULATION LOOP
###############################################
echo "---- FOG Simulation ----"

for alpha in "${FOG_ALPHAS[@]}"; do
    SAVE_PATH="${NUSC_DATA_ROOT}/fog/fog_alpha_${alpha}"
    mkdir -p "$SAVE_PATH"

    echo "[FOG] alpha = $alpha"
    python3 simulator_nuscenes.py \
        --fault fog \
        --nusc_root "$NUSC_DATA_ROOT" \
        --fog_alpha "$alpha" \
        --fog_save_path "$SAVE_PATH" \
        --plot_save_path "$PLOT_SAVE_PATH" \
        --save
done


###############################################
# 2. RAIN SIMULATION LOOP
###############################################
echo ""
echo "---- RAIN Simulation ----"

for rate in "${RAIN_RATES[@]}"; do
    SAVE_PATH="${NUSC_DATA_ROOT}/rain/rain_rate_${rate}"
    mkdir -p "$SAVE_PATH"

    echo "[RAIN] rate = $rate"
    python3 simulator_nuscenes.py \
        --fault rain \
        --nusc_root "$NUSC_DATA_ROOT" \
        --rain_rate "$rate" \
        --rain_save_path "$SAVE_PATH" \
        --plot_save_path "$PLOT_SAVE_PATH" \
        --save
done


###############################################
# 3. SNOW SIMULATION LOOP
###############################################
echo ""
echo "---- SNOW Simulation ----"

for rate in "${SNOW_RATES[@]}"; do
    SAVE_PATH="${NUSC_DATA_ROOT}/snow/snow_rate_${rate}"
    mkdir -p "$SAVE_PATH"

    echo "[SNOW] rate = $rate"
    python3 simulator_nuscenes.py \
        --fault snow \
        --nusc_root "$NUSC_DATA_ROOT" \
        --snow_rate "$rate" \
        --snow_save_path "$SAVE_PATH" \
        --plot_save_path "$PLOT_SAVE_PATH" \
        --save
done


###############################################
# 4. DUST SIMULATION LOOP
###############################################
echo ""
echo "---- DUST Simulation ----"

for level in "${DUST_LEVELS[@]}"; do
    SAVE_PATH="${NUSC_DATA_ROOT}/dust/dust_${level}"
    mkdir -p "$SAVE_PATH"

    echo "[DUST] level = $level"
    python3 simulator_nuscenes.py \
        --fault dust \
        --nusc_root "$NUSC_DATA_ROOT" \
        --dust_level "$level" \
        --dust_save_path "$SAVE_PATH" \
        --plot_save_path "$PLOT_SAVE_PATH" \
        --save
done

echo ""
echo "=== All Weather Simulations Complete ==="
