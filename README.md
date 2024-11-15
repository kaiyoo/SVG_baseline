# Simple Sounding Video Generation

This is the official implementation of the paper ["A Simple but Strong Baseline for Sounding Video Generation: Effective Adaptation of Audio and Video Diffusion Models for Joint Generation."](https://arxiv.org/abs/2409.17550)

## Setup

Run the command below from the project root directory.
```
pip install -r requirements.txt

pip install -e .
```

## Training

Here, we provide an exmaple of training with GreatestHits dataset.

### Dataset preparation

Download a dataset from [the official webpage](https://andrewowens.com/vis/) and run `./src/va_core/dataset/greatesthits/script_create_segmented_mp4.sh` to obtain preprocessed video clips. Or you can download the preprocessed dataset from [this link](https://drive.google.com/file/d/1ypxJaAwFH4eiz2nEn_w2Tu6o99DdSO7L/view?usp=drive_link). 

In the default setting, the mp4 files are assumed to be stored in `./src/va_core/dataset/greatesthits/vis-data-256-segment-8fps-crop`. If you use another directory, you need to modify `flist_*.txt` in `src/va_core/dataset/greatesthits` accordingly. An each line of the text file should describe a path to mp4 file and the corresponding caption separated by a comma as shown below:
```
./src/va_core/dataset/greatesthits/vis-data-256-segment-8fps-crop/2015-09-23-16-13-51-97_denoised_thumb-003.mp4,"A person is hitting a drumstick on a drum that is surrounded by leaves and foliage, creating a rhythmic sound in the midst of a natural environment."
```


### Run the training script

```
./sh_scripts/script_train_greatesthits.sh
```

## Sounding video generation with the trained model

You can download the trained model from [this link](https://drive.google.com/file/d/1pQdhl_j_hjdIMJq9waVTvM3baj5xqeF3/view?usp=drive_link). 

Then, run the following script after setting `$MODEL_DIR` in the script properly:
```
./sh_scripts/script_gen_greatesthits.sh
```

## Evaluate the generated sounding videos

### Setup

To compute LanguageBind scores, you need to clone its official repo as shown below:
```
cd third_party
git clone https://github.com/PKU-YuanGroup/LanguageBind
```

To compute FAD, you need to clone audioldm_eval as shown below:
```
cd third_party
git clone https://github.com/haoheliu/audioldm_eval
```

### Evaluation

```
./sh_scripts/script_eval.sh
```



## Credits

This repository is built on the following open-source projects:
- [Diffusers](https://github.com/huggingface/diffusers)
- [DiffFoley](https://github.com/luosiallen/Diff-Foley)
- [TempoToken](https://github.com/guyyariv/TempoTokens)
- [CoDi](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)
