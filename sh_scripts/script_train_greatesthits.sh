#!/bin/bash

pip install -e .

N_GPUS=8
OUT_DIR="out"

accelerate launch --use_deepspeed --gpu_ids="all" --num_processes=${N_GPUS} ./py_scripts/svg/train_svg.py --dataset_name="greatesthits" --train_batch_size=1 --gradient_accumulation_steps=2 --duration_per_sample=4 --num_frames_per_sample=32 --num_train_epochs=1000 --output_dir="${OUT_DIR}" --use_ema --mixed_precision="fp16" --validation_epochs=50
