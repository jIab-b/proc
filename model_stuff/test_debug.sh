#!/bin/bash

# Test run for voxel SDS training
cd /home/beed/splats/proc && python -m model_stuff.train \
  --dataset_id 1 \
  --prompt "a beautiful mountain landscape with trees and a lake" \
  --steps 50 \
  --learning_rate 0.01 \
  --cfg_scale 7.5 \
  --temperature_start 0.1 \
  --temperature_end 0.1 \
  --sds_weight 1.0 \
  --photo_weight 1.0 \
  --output_dir "out_local/voxel_sds" \
  --fuckdump \
  --seed 42
