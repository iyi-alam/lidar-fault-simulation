#!/bin/bash
# Shell script

FAULT="snow"
# KITTY_INPUT_DIR="/home/saksham/samsad/mtech-project/datasets/kitty/data_object_velodyne/training/velodyne/"
# KITTY_OUTPUT_DIR="/home/saksham/samsad/mtech-project/datasets/kitty/data_object_velodyne_fog/training/velodyne/"

# python3 simulator_f1.py --fault "$FAULT" --fog_alpha 0.04 \
#  --input_dir "$KITTY_INPUT_DIR" --output_dir "$KITTY_OUTPUT_DIR"

KITTY_INPUT_DIR="/home/saksham/samsad/mtech-project/datasets/kitty/data_object_velodyne/testing/velodyne/"
KITTY_OUTPUT_DIR="home/saksham/samsad/mtech-project/datasets/kitty/data_object_velodyne/testing/"

python3 simulator_f1.py --fault "$FAULT" --rain_rate 5.0 \
 --input_dir "$KITTY_INPUT_DIR" --output_dir "$KITTY_OUTPUT_DIR"