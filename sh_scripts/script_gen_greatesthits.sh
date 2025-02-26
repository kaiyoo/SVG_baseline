#!/bin/bash

# pip install -e .

MODEL_DIR="./trained_model/pipe"
OUT_DIR="./out/gen_svg"
BATCH_SIZE=2 #8
N_STEPS=25 #50
FPS=2 #8
python ./py_scripts/svg/test_svg_with_dataset.py --model_path $MODEL_DIR --dataset_name "greatesthits" --n_steps $N_STEPS --fps $FPS --save_original --save_jpeg --out_dir $OUT_DIR --batch_size $BATCH_SIZE
