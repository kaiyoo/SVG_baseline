#!/bin/bash

pip install -e .

MODEL_DIR="./out/pipe"
OUT_DIR="./out/gen_svg"

python ./py_scripts/svg/test_svg_with_dataset.py --model_path $MODEL_DIR --dataset_name "greatesthits" --n_steps 50 --fps 8 --save_original --save_jpeg --out_dir $OUT_DIR --batch_size 8
