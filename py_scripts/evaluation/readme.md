# Evaluation metrics

## Single-modal

### FAD

FAD is one of the most popular metric to evaluate audio quality.

First, you need to clone [audioldm_eval](https://github.com/haoheliu/audioldm_eval) somewhere accessible (assumed `../../third_party/audioldm_eval` by default). The source and target audio files need to be stored in respective directories. Then, run the following command to compute FAD:
```
python compute_fad.py --source_dir path_to_source_wav_dir --target_dir path_to_target_wav_dir
```

### FVD

FVD is one of the most popular metric to evaluate video quality.

First, you need to clone [StyleGAN-V](https://github.com/universome/stylegan-v) somewhere accessible (assumed `../../third_party/stylegan-v` by default). The generated and real videos need to be stored as jpeg files in respective directories. This is automatically done if you use `py_scripts/svg/test_svg_with_dataset.py` with `--save_jpeg` option. Then, run the following command to compute FVD:
```
python compute_fvd.py --real_data_path path_to_real_videos --fake_data_path path_to_gen_videos --mirror 1 --gpus 1 --resolution 256 --metrics fvd2048_16f --verbose 0 --use_cache 0
```


## Cross-modal

### AV-Align score

AV-Align score is for evaluating how much audio and video are aligned with each other. This was originally proposed in [TempoToken](https://arxiv.org/abs/2309.16429), and we slightly modified how to compute the score from the official implementation. Specifically, we compute IoU after rewriting it with precision and recall to mitigate an issue caused by the difference of temporal resolution between video and audio.

The target audio and video files need to be stored in respective directories. Note that the corresponding audio and video should have the same file name except for its extension. Then, run the following command to compute the averaged AV-Align score:
```
python av_align.py --audio_dir path_to_audio_dir --video_dir path_to_video_dir
```

### ImageBind score

The ImageBind score is the average of cosine similarities between two ImageBind features, each of which is extracted from a different modality. Here, we use this score to evaluate semantic alignment between text-audio, text-video, and audio-video pairs.

First, you need to clone [ImageBind](https://github.com/facebookresearch/ImageBind) somewhere accessible (assumed `../../third_party/ImageBind` by default). You also need a csv file to specify a list of text-audio-video triplets, which is automatically generated if you use `py_scripts/svg/test_svg_with_captionlist.py`. An each line of the csv file should describe a path to mp4 file, a path to wav file, and the corresponding caption separated by a comma as shown below:
```
"gen_mp4/2015-02-16-16-49-06_denoised.mp4","gen_wav/2015-02-16-16-49-06_denoised.wav","hitting dirty dishes with a drumstick"
```

Then, run the following command to compute ImageBind scores:
```
python compute_imagebind_score.py --csv_file path_to_csv_file
```

### LanguageBind score

The LanguageBind score is the average of cosine similarities between two LanguageBind features, each of which is extracted from a different modality. Here, we use this score to evaluate semantic alignment between text-audio, text-video, and audio-video pairs.

First, you need to clone [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) somewhere accessible (assumed `../../third_party/LanguageBind` by default). You also need a csv file to specify a list of text-audio-video triplets, which is automatically generated if you use `py_scripts/svg/test_svg_with_captionlist.py`. An each line of the csv file should describe a path to mp4 file, a path to wav file, and the corresponding caption separated by a comma as shown below:
```
"gen_mp4/2015-02-16-16-49-06_denoised.mp4","gen_wav/2015-02-16-16-49-06_denoised.wav","hitting dirty dishes with a drumstick"
```

Then, run the following command to compute ImageBind scores:
```
python compute_languagebind_score.py --csv_file path_to_csv_file
```

### CAVP similarity score

The score is the average cosine similarity between video and audio features, which are computed by the [CAVP from Diff-Foley](https://arxiv.org/abs/2306.17203v1). We directly use their pretrained model which is trained on VGGSound and AudioSet. We refer to this score as CAVP score. The CAVP score on the provided GreatestHits dataset is 0.8232. Note this score should only be used as reference instead of the main metric, as it is not sensitive to measure the alignment between audio and video. 

The generated sounding video files ('.mp4') are expected to store in one folder, then just run the following code to compute CAVP score. The CAVP checkpoint could be downloaded in [here](https://huggingface.co/SimianLuo/Diff-Foley/blob/main/diff_foley_ckpt/cavp_epoch66.ckpt) through huggingface provided by the authors.

```
python cavp_score.py --path path_to_video_folder --checkpoints path_to_cavp_checkpoint
```
