#!/bin/bash

###############################################
# USER CONFIGURATION
###############################################

# NuScenes data root
DATA_ROOT="/home/saksham/samsad/mtech-project/datasets/nuscenes"

# Input directory = nuScenes sweeps for LIDAR_TOP
INPUT_DIR="${DATA_ROOT}/samples/LIDAR_TOP"

# Output directory (will contain fog/, snow/, rain/, dust/)
OUTPUT_BASE="${DATA_ROOT}-weather/samples/"

# Plot directory
PLOT_SAVE_PATH="${DATA_ROOT}/plots"
mkdir -p "$PLOT_SAVE_PATH"

###############################################
# FAULT LEVELS TO SWEEP
###############################################

# Sweep ranges
FOG_ALPHAS=(0.01 0.02 0.04 0.06 0.08 0.1)   # 0.04 SKIPPED
FOG_GAMMA=1e-6

RAIN_RATES=(5.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0)          # 50.0 SKIPPED
#RAIN_RATES=(30.0)   

SNOW_RATES=(0.5 1.0 1.5 2.0 2.5)      # 1.5 SKIPPED
TERMINAL_VELOCITY=2.0              # Always same

DUST_LEVELS=("light" "heavy")      # "moderate" SKIPPED


###############################################
# PYTHON SIMULATOR SCRIPT FILE
###############################################
SIM_SCRIPT="simulator_f1.py"   # ← Make sure this matches your updated file name


###############################################
# SIMULATION LOOP FUNCTIONS
###############################################

simulate_fog () {
    for FOG_ALPHA in "${FOG_ALPHAS[@]}"; do
        echo ">>> Simulating FOG with alpha=${FOG_ALPHA}"
        OUTPUT_DIR="${OUTPUT_BASE}"
        mkdir -p "$OUTPUT_DIR"

        python3 "$SIM_SCRIPT" \
            --fault "fog" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --plot_save_path "$PLOT_SAVE_PATH" \
            --fog_alpha "$FOG_ALPHA" \
            --fog_gamma "$FOG_GAMMA" \
            --rain_rate 0 \
            --snow_rate 0 \
            --terminal_velocity "$TERMINAL_VELOCITY" \
            --dust_level "none" \
            --save
    done
}

simulate_rain () {
    for RAIN_RATE in "${RAIN_RATES[@]}"; do
        echo ">>> Simulating RAIN with rate=${RAIN_RATE}"
        OUTPUT_DIR="${OUTPUT_BASE}"
        mkdir -p "$OUTPUT_DIR"

        python3 "$SIM_SCRIPT" \
            --fault "rain" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --plot_save_path "$PLOT_SAVE_PATH" \
            --fog_alpha 0 \
            --fog_gamma "$FOG_GAMMA" \
            --rain_rate "$RAIN_RATE" \
            --snow_rate 0 \
            --terminal_velocity "$TERMINAL_VELOCITY" \
            --dust_level "none" \
            --save
    done
}

simulate_snow () {
    local LISA_SNOW="$1"   # true / false

    for SNOW_RATE in "${SNOW_RATES[@]}"; do
        echo ">>> Simulating SNOW with rate=${SNOW_RATE}"

        OUTPUT_DIR="${OUTPUT_BASE}"
        mkdir -p "$OUTPUT_DIR"

        CMD=(
            python3 "$SIM_SCRIPT"
            --fault "snow"
            --input_dir "$INPUT_DIR"
            --output_dir "$OUTPUT_DIR"
            --plot_save_path "$PLOT_SAVE_PATH"
            --fog_alpha 0
            --fog_gamma "$FOG_GAMMA"
            --rain_rate 0
            --snow_rate "$SNOW_RATE"
            --terminal_velocity "$TERMINAL_VELOCITY"
            --dust_level "none"
            --save
        )

        # Conditionally enable LISA snow
        if [ "$LISA_SNOW" = true ]; then
            CMD+=(--lisa_snow)
        fi

        "${CMD[@]}"
    done
}


simulate_dust () {
    for DUST_LEVEL in "${DUST_LEVELS[@]}"; do
        echo ">>> Simulating DUST with level=${DUST_LEVEL}"
        OUTPUT_DIR="${OUTPUT_BASE}"
        mkdir -p "$OUTPUT_DIR"

        python3 "$SIM_SCRIPT" \
            --fault "dust" \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --plot_save_path "$PLOT_SAVE_PATH" \
            --fog_alpha 0 \
            --fog_gamma "$FOG_GAMMA" \
            --rain_rate 0 \
            --snow_rate 0 \
            --terminal_velocity "$TERMINAL_VELOCITY" \
            --dust_level "$DUST_LEVEL" \
            --save
    done
}


###############################################
# RUN ALL SIMULATIONS
###############################################

echo "=============================="
echo " Running ALL weather faults..."
echo "=============================="

#simulate_fog
#simulate_rain
simulate_snow true
# simulate_dust

echo "======================================="
echo " All fault simulations completed! "
echo " Output stored in: ${OUTPUT_BASE}"
echo "======================================="
