#!/bin/bash

INPUT_DIR="/groups/gce50978/dataset/GreatestHits/vis-data-256"
OUTPUT_DIR_SEG="/groups/gce50978/dataset/GreatestHits/vis-data-256-segment-8fps"
OUTPUT_DIR_CROP="/groups/gce50978/dataset/GreatestHits/vis-data-256-segment-8fps-crop"

for file in "$INPUT_DIR"/*_denoised_thumb.mp4
do
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    ffmpeg -i "$file" -c:v libx264 -crf 18 -r 8 -g 32 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*4)" -vf "crop=256:256:(in_w-256)/2:(in_h-256)/2" -c:a copy -map 0 -f segment -segment_time 4 -segment_list "flist.csv" -reset_timestamps 1 "$OUTPUT_DIR_CROP/$filename-%03d.mp4"
    ffmpeg -i "$file" -c:v libx264 -crf 18 -r 8 -g 32 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*4)" -c:a copy -map 0 -f segment -segment_time 4 -segment_list "flist.csv" -reset_timestamps 1 "$OUTPUT_DIR_SEG/$filename-%03d.mp4"
done

# Check if the duration is correct.
for file in "$OUTPUT_DIR_CROP"/*.mp4
do
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    duration=${duration%.*}
    if [ "$duration" -ne 4 ]; then
        filename=$(basename -- "$file")
        echo $filename
        rm $OUTPUT_DIR_CROP/$filename
        rm $OUTPUT_DIR_SEG/$filename
    fi
done
