#!/bin/bash

TARGET_PATH="./out/gen_svg"
FLIST="${TARGET_PATH}/gen_flist.txt"


python ./py_scripts/evaluation/av_align.py --audio_dir ${TARGET_PATH}/gen_wav --video_dir ${TARGET_PATH}/gen_mp4 > ${TARGET_PATH}/av_align.txt

# python ./py_scripts/evaluation/compute_fad.py --source_dir ${TARGET_PATH}/orig_wav --target_dir ${TARGET_PATH}/gen_wav > ${TARGET_PATH}/fad.txt

# python ./py_scripts/evaluation/compute_fvd.py --ref_dir ${TARGET_PATH}/orig_mp4 --gen_dir ${TARGET_PATH}/gen_mp4 > ${TARGET_PATH}/fvd.txt

# python ./py_scripts/evaluation/compute_languagebind_score.py --csv_path $FLIST > ${TARGET_PATH}/language_bind.txt