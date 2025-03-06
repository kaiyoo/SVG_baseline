import logging
import math
import os
import random
from pathlib import Path
import itertools
import scipy
import shutil

import datasets
import numpy as np
import gc
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from torchvision import transforms
from pytorchvideo.transforms import UniformTemporalSubsample
from tqdm.auto import tqdm
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan, CLIPTextModel, CLIPTokenizer, SpeechT5FeatureExtractor

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel, MotionAdapter, UNetMotionModel, UNet3DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from args import parse_args

from va_core.dataset import get_dataset_info
from va_core.model.svg.unet_svg import UNetModelSVG
from va_core.model.svg.utils import TrainingModel, logmel_extractor
from va_core.model.svg.pipeline_svg import VideoAudioGenWithSVGPipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# minimal diffusers version check
check_min_version("0.17.0.dev0")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def log_validation(test_dataloader,
                    vae_a,
                    text_encoder_a,
                    tokenizer_a,
                    vocoder,
                    scheduler_a,
                    unet_a,
                    vae_v,
                    text_encoder_v,
                    tokenizer_v,
                    scheduler_v,
                    unet_v,
                    unet_svg,
                    args, 
                    epoch,
                    device,
                    weight_dtype
                    ):
    logger.info("Running validation... ")
    torch.cuda.empty_cache()

    pipeline = VideoAudioGenWithSVGPipeline(
            vae_a=vae_a,
            text_encoder_a=text_encoder_a,
            tokenizer_a=tokenizer_a,
            vocoder=vocoder,
            scheduler_a=scheduler_a,
            unet_a=unet_a,
            vae_v=vae_v,
            text_encoder_v=text_encoder_v,
            tokenizer_v=tokenizer_v,
            scheduler_v=scheduler_v,
            unet_v=unet_v,
            unet_svg=unet_svg,
    )
    # pipeline = pipeline.to(device, torch_dtype=weight_dtype)
    pipeline = pipeline.to("cuda", torch.float16)
    # CPU 오프로딩 활성화 (필요할 때만 GPU로 로드)
    pipeline.enable_sequential_cpu_offload(gpu_id=0)
    pipeline.set_progress_bar_config(disable=True)
    # pipeline.eval()        


    pipeline.unet_svg.eval()
    pipeline.unet_a.eval()
    pipeline.unet_v.eval()
    pipeline.text_encoder_a.eval()
    pipeline.text_encoder_v.eval()
    pipeline.vae_a.eval()
    pipeline.vae_v.eval()
    pipeline.vocoder.eval()

    # 내부 모델 eval 전환
    def disable_checkpointing(model):
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                 module.use_checkpoint = False

    # checkpointing 비활성화
    disable_checkpointing(pipeline.unet_svg)
    disable_checkpointing(pipeline.unet_a)
    disable_checkpointing(pipeline.unet_v)
    

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    dirname = os.path.join(args.output_dir, "val", str(epoch))
    os.makedirs(dirname, exist_ok=True)
    negative_prompt = "bad quality, worse quality"
    for step, batch in enumerate(test_dataloader):
        # Assume bs=1 so we access batch["text"][0] explicitly.
        prompt = batch["text"][0]
        prompt_a = prompt
        prompt_v = prompt


        with torch.no_grad():            
            with torch.cuda.amp.autocast():
                output = pipeline(
                    prompt_a=prompt_a,
                    prompt_v=prompt_v,
                    negative_prompt_v=negative_prompt,
                    negative_prompt_a=negative_prompt,
                    length_in_s=args.duration_per_sample,
                    frame_rate=args.num_frames_per_sample // args.duration_per_sample,
                    height_v=args.resolution,
                    width_v=args.resolution,
                    generator=generator
                )

        video_to_save = (output.videos[0] * 255.0).astype(np.uint8)
        audio_to_save = output.audios[0]
        scipy.io.wavfile.write(os.path.join(dirname, f"{step}_gen_{prompt}.wav"), rate=16000, data=audio_to_save)
        # fps=args.num_frames_per_sample / args.duration_per_sample => AttributeError: 'numpy.float64' object has no attribute 'numerator'
        fps_value = float(args.num_frames_per_sample) / float(args.duration_per_sample)
        prompt = prompt.replace(".", "_")
        torchvision.io.write_video(os.path.join(dirname, f"{step}_gen_{prompt}.mp4"), video_to_save, fps=int(np.round(fps_value)))

        if step >= args.num_validation_samples - 1:
            break

    del pipeline
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    # 단일 GPU 혹은 CPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    logger.info("Starting training...")

    # seed 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    # 1) (현재) have best model, but don't have checkpoints
    if (not args.resume_from_checkpoint) and (args.have_best_model):
        logger.info("No checkpoint, but found best_model_dir -> Load from best_model/pipe/")

        # best_model_dir = args.best_model_dir (예: "best_model/pipe")
        best_pipe_path = args.best_model_dir

        # (A) unet_a, unet_v, unet_svg를 best_model 폴더에서 불러오기
        unet_a = UNet2DConditionModel.from_pretrained(best_pipe_path, subfolder="unet_a")
        unet_v = UNetMotionModel.from_pretrained(best_pipe_path, subfolder="unet_v")
        unet_svg = UNetModelSVG.from_pretrained(best_pipe_path, subfolder="unet_svg")

        # (B) 스케줄러, 토크나이저, vocoder 등도 동일 폴더에서 불러올 수 있으면 불러옴
        #     (없으면 default 로드)
        try:
            scheduler_v = DDIMScheduler.from_pretrained(best_pipe_path, subfolder="scheduler_v")
            tokenizer_v = CLIPTokenizer.from_pretrained(best_pipe_path, subfolder="tokenizer_v")
            text_encoder_v = CLIPTextModel.from_pretrained(best_pipe_path, subfolder="text_encoder_v")
            vae_v = AutoencoderKL.from_pretrained(best_pipe_path, subfolder="vae_v")
        except:
            logger.warning("Could not load video-related subfolders from best_model. Using default video model.")
            # 필요하면 animatediff/modelscope/zeroscope 등 다시 로드
            raise NotImplementedError("원하는 방식으로 fallback 처리하세요.")

        try:
            scheduler_a = DDIMScheduler.from_pretrained(best_pipe_path, subfolder="scheduler_a")
            tokenizer_a = RobertaTokenizer.from_pretrained(best_pipe_path, subfolder="tokenizer_a")
            text_encoder_a = ClapTextModelWithProjection.from_pretrained(best_pipe_path, subfolder="text_encoder_a")
            vae_a = AutoencoderKL.from_pretrained(best_pipe_path, subfolder="vae_a")
            vocoder = SpeechT5HifiGan.from_pretrained(best_pipe_path, subfolder="vocoder")
        except:
            logger.warning("Could not load audio-related subfolders from best_model. Using default AudioLDM.")
            raise NotImplementedError("원하는 방식으로 fallback 처리하세요.")

        # (C) dummy timesteps
        # scheduler_v.set_timesteps(100, device=device)
        # scheduler_a.set_timesteps(100, device=device)

    
    # 2) Train from scratch: use base model or use checkpoints
    elif (not args.resume_from_checkpoint) and (not args.have_best_model):
        logger.info("No checkpoint, no best model -> Load default pretrained models.")

        # 모델 및 스케줄러 로드
        path_video_model = ""
        if args.video_model == "animatediff":
            # AnimateDiff
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
            path_video_model = "SG161222/Realistic_Vision_V5.1_noVAE"
            scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1, beta_schedule="linear")
            tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
            unet_i = UNet2DConditionModel.from_pretrained(path_video_model, subfolder="unet")
            unet_v = UNetMotionModel.from_unet2d(unet_i, adapter)
        elif args.video_model == "modelscope":
            path_video_model = "damo-vilab/text-to-video-ms-1.7b"
            scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler")
            tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
            text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
            vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
            unet_v = UNet3DConditionModel.from_pretrained(path_video_model, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
        elif args.video_model == "zeroscope":
            path_video_model = "cerspense/zeroscope_v2_576w"
            scheduler_v = DDIMScheduler.from_pretrained(path_video_model, subfolder="scheduler")
            tokenizer_v = CLIPTokenizer.from_pretrained(path_video_model, subfolder="tokenizer")
            text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder", torch_dtype=torch.float16)
            vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae", torch_dtype=torch.float16)
            unet_v = UNet3DConditionModel.from_pretrained(path_video_model, subfolder="unet", torch_dtype=torch.float16)
        else:
            raise ValueError(f"Unknown video model: {args.video_model}")
        
        # scheduler_v.set_timesteps(100, device=device)  # dummy for x0 prediction

        # AudioLDM 관련 모델
        path_audioldm = "cvssp/audioldm-m-full"
        scheduler_a = DDIMScheduler.from_pretrained(path_audioldm, subfolder="scheduler")
        tokenizer_a = RobertaTokenizer.from_pretrained(path_audioldm, subfolder="tokenizer")
        unet_a = UNet2DConditionModel.from_pretrained(path_audioldm, subfolder="unet")
        vocoder = SpeechT5HifiGan.from_pretrained(path_audioldm, subfolder="vocoder")
        
        # scheduler_a.set_timesteps(100, device=device)  # dummy for x0 prediction

        # Deepspeed 관련 context manager 제거하고, 바로 모델 로드
        text_encoder_a = ClapTextModelWithProjection.from_pretrained(path_audioldm, subfolder="text_encoder")
        vae_a = AutoencoderKL.from_pretrained(path_audioldm, subfolder="vae")
        # video 모델의 경우, modelscope와 zeroscope는 text_encoder_v와 vae_v가 이미 로드되었으므로,
        # animatediff의 경우에도 별도 text_encoder_v와 vae_v가 필요하면 추가 로드 필요
        if args.video_model == "animatediff":
            text_encoder_v = CLIPTextModel.from_pretrained(path_video_model, subfolder="text_encoder")
            vae_v = AutoencoderKL.from_pretrained(path_video_model, subfolder="vae")


    ## 공통      
    scheduler_v.set_timesteps(100, device=device)  # dummy for x0 prediction
    scheduler_a.set_timesteps(100, device=device)  # dummy for x0 prediction

    unet_a_upsample_factor = 2 ** unet_a.num_upsamplers    

    sampling_rate = 16000
    mel_extractor = logmel_extractor(sampling_rate)

    # SVG 모델 생성
    connector_config = {}
    if args.connector_in_type is not None:
        connector_config["connector_in_type"] = args.connector_in_type
    if args.use_x0_pred_for_connectors:
        connector_config["connector_out_input"] = "x_pred"
    # 여기서는 unet_v는 위에서 초기화된 것으로 사용합니다.
    if any(connector_config):
        unet_svg = UNetModelSVG(unet_a.config, unet_v.config, connector_audio_config=connector_config, connector_video_config=connector_config)
    else:
        unet_svg = UNetModelSVG(unet_a.config, unet_v.config)


    # Freeze params
    text_encoder_a.requires_grad_(False)
    vae_a.requires_grad_(False)
    text_encoder_v.requires_grad_(False)
    vae_v.requires_grad_(False)

    # # unet_a와 unet_v의 모든 파라미터를 freeze
    # for param in unet_a.parameters():
    #     param.requires_grad = False
    # for param in unet_v.parameters():
    #     param.requires_grad = False
    train_model = TrainingModel(unet_svg, unet_a, unet_v)

    if not args.finetune_unet:
        for k, v in train_model.named_parameters():            
            if "connector" not in k:
                v.requires_grad = False
                # phrase_temp = "motion" if isinstance(unet_v, UNetMotionModel) else "temp_"
                # if args.fix_temporal or phrase_temp not in k:
                #     v.requires_grad_(False)
                #     print(f"{k} is fixed.")

    # EMA 사용 시 EMA 모델 생성
    if args.use_ema:
        if any(connector_config):
            ema_unet = UNetModelSVG(unet_a.config, unet_v.config, connector_audio_config=connector_config, connector_video_config=connector_config)
        else:
            ema_unet = UNetModelSVG(unet_a.config, unet_v.config)
        tmp_config = {k: v for k, v in ema_unet.config.items() if k != "_use_default_values"}
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNetModelSVG, model_config=tmp_config)
        ema_unet.to(device)

    if args.gradient_checkpointing:
        unet_a.enable_gradient_checkpointing()
        unet_v.enable_gradient_checkpointing()
        unet_svg.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 단일 GPU이므로 accelerator.num_processes는 1로 가정
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_batch_size

    # optimizer 설정 (8bit Adam 선택 시 bitsandbytes 사용)
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. Run `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [x for x in train_model.parameters() if x.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def tokenize_captions(batched_captions, is_train=True):
        captions = []
        for caption in batched_captions:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError("Caption column `text` should contain either strings or lists of strings.")
        return captions

    ds_common_args = {
        "dataset_names": args.dataset_name,
        "duration_per_sample": args.duration_per_sample,
        "sampling_rate": sampling_rate,
        "sample_random_segments": True,
    }
    train_dataset_info = get_dataset_info(dataset_types="train",
                                          max_samples=args.max_train_samples,
                                          tokenizer=tokenize_captions,
                                          **ds_common_args)
    train_dataset = train_dataset_info.dataset
    train_collate_fn = train_dataset_info.collate_fn

    test_dataset_info = get_dataset_info(dataset_types="test",
                                         max_samples=args.num_validation_samples,
                                         tokenizer=lambda x: x,
                                         **ds_common_args)
    test_dataset = test_dataset_info.dataset
    test_collate_fn = test_dataset_info.collate_fn

    dataset_size = train_dataset_info.size
    if args.max_train_samples is not None and dataset_size > args.max_train_samples:
        dataset_size = args.max_train_samples

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    tf_video = torch.nn.Sequential(
        UniformTemporalSubsample(args.num_frames_per_sample, temporal_dim=1)
    ).to(device)
    tf_audio = torch.nn.Sequential(
        UniformTemporalSubsample(sampling_rate * args.duration_per_sample, temporal_dim=-1)
    ).to(device)

    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: 2.0 * (x / 255.0) - 1.0),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataset_size / total_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # mixed precision을 위해 weight_dtype 설정 (torch.cuda.amp 사용)
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_a.to(device, dtype=weight_dtype)
    text_encoder_v.to(device, dtype=weight_dtype)
    vae_a.to(device, dtype=weight_dtype)
    vae_v.to(device, dtype=weight_dtype)

    if args.short_memory:
        vae_v_cpu = vae_v.to("cpu", dtype=torch.float32)

    train_model.to(device, dtype=weight_dtype)
    vocoder.to(device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(dataset_size / total_batch_size)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  The number of parameters to be trained = {sum(p.numel() for p in train_model.parameters() if p.requires_grad)}")
    logger.info(f"  The number of parameters in unet_a = {sum(p.numel() for p in unet_a.parameters())}")
    logger.info(f"  The number of parameters in unet_v = {sum(p.numel() for p in unet_v.parameters())}")
    logger.info(f"  The number of parameters in unet_svg = {sum(p.numel() for p in unet_svg.parameters())}")
    logger.info(f"  The total number of parameters = {sum(p.numel() for p in train_model.parameters())}")
    global_step = 0
    first_epoch = 0

    # (체크포인트로부터 이어서 학습하는 로직은 필요한 경우 별도 구현)
    # 체크포인트에서 state_dict 불러오기
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

        # model
        train_model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        global_step = checkpoint.get("global_step", 0)
        global_step = int(path.split("-")[1])
        start_epoch = checkpoint.get("epoch", 0)   
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps

        if args.use_ema and "ema_unet_state_dict" in checkpoint:
            ema_unet.load_state_dict(checkpoint["ema_unet_state_dict"])
        else:
            logger.info("No EMA state in checkpoint or EMA not used.")

        # 이후 학습 루프에서 global_step, start_epoch부터 이어감
        logger.info(f"Checkpoint loaded (global_step={global_step}, epoch={start_epoch}).")


    # First validation
    # train_model.eval()
    # log_validation(test_dataloader,
    #                vae_a=vae_a,
    #                text_encoder_a=text_encoder_a,
    #                tokenizer_a=tokenizer_a,
    #                vocoder=vocoder,
    #                scheduler_a=scheduler_a,
    #                unet_a=train_model.unet_a,
    #                vae_v=vae_v,
    #                text_encoder_v=text_encoder_v,
    #                tokenizer_v=tokenizer_v,
    #                scheduler_v=scheduler_v,
    #                unet_v=train_model.unet_v,
    #                unet_svg=train_model.unet_svg,
    #                args=args,        
    #                epoch=0,
    #                device=device,
    #                weight_dtype=weight_dtype)

    # mixed precision를 위한 GradScaler 설정 (fp16 사용 시)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None

    xtype = ["audio", "video"]
    scheduler_list = [scheduler_a, scheduler_v]

    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")
    train_loss_accum = [0.0] * len(xtype)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_model.train()
        for step, batch in enumerate(train_dataloader):
            # resume 시 필요한 step 건너뛰기 로직 (필요한 경우 구현)
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            text_input = batch["text"]
            input_data = [tf_audio(batch["audio"].to("cpu")), tf_video(batch["video"].to("cpu"))]

            latents = []
            timesteps = []
            timesteps_con = []
            timesteps_shared = torch.randint(0, 1000, (args.train_batch_size,), device=device)
            for i, typ in enumerate(xtype):
                if typ == "audio":
                    audio_target = input_data[i].to(weight_dtype).cpu().numpy()
                    mel_features = mel_extractor(audio_target, args.duration_per_sample)
                    mel_features = torch.from_numpy(mel_features).to(device, dtype=weight_dtype).unsqueeze(1)
                    latents_a = vae_a.encode(mel_features).latent_dist.sample()
                    latents_a = latents_a * vae_a.config.scaling_factor
                    latents.append(latents_a)
                    timesteps_a = timesteps_shared * (scheduler_a.config.num_train_timesteps // 1000)
                    timesteps_a = timesteps_a.long()
                    timesteps.append(timesteps_a)
                    t_con = torch.randint(0, scheduler_a.config.num_train_timesteps, (args.train_batch_size,), device=device)
                    timesteps_con.append(t_con)
                elif typ == "video":
                    B, T = input_data[i].shape[:2]
                    video_target = train_transforms(input_data[i].flatten(end_dim=1).to(device, dtype=weight_dtype))
                    
                    ## Walkaround to avoid CUDA MEMORY ERROR
                    if args.short_memory:
                        # vae_v 모델을 CPU로 옮기고, float32로 변환 (CPU에서 fp16 연산은 지원되지 않으므로)
                        # vae_v_cpu = vae_v.to("cpu", dtype=torch.float32)
                        # video_target도 CPU로 옮기고 float32로 변환
                        video_target_cpu = video_target.cpu().float()
                        latents_v = vae_v_cpu.encode(video_target_cpu).latent_dist.sample()
                        # 이후 결과를 GPU로 옮기고 원하는 dtype으로 변환
                        latents_v = latents_v.to(device, dtype=weight_dtype)
                    else:
                        with torch.cuda.amp.autocast():
                            latents_v = vae_v.encode(video_target).latent_dist.sample()

                    # 이후 scaling 및 재구성 작업은 GPU에서 진행
                    ################################################################

                    latents_v = latents_v * vae_v.config.scaling_factor
                    latents_v = latents_v.reshape((B, T) + latents_v.shape[1:])
                    latents_v = latents_v.permute((0, 2, 1, 3, 4))
                    latents.append(latents_v)
                    timesteps_v = timesteps_shared * (scheduler_v.config.num_train_timesteps // 1000)
                    timesteps_v = timesteps_v.long()
                    timesteps.append(timesteps_v)
                    t_con = torch.randint(0, scheduler_v.config.num_train_timesteps, (args.train_batch_size,), device=device)
                    timesteps_con.append(t_con)

            noise = [torch.randn_like(tmp, device=tmp.device) for tmp in latents]

            noisy_latents = []
            noisy_latents_con = []
            for i, typ in enumerate(xtype):
                nl = scheduler_list[i].add_noise(latents[i], noise[i], timesteps[i])
                nlc = scheduler_list[i].add_noise(latents[i], noise[i], timesteps_con[i])
                if typ == "audio":
                    original_length = nl.shape[2]
                    padded_length = int(np.ceil(original_length / unet_a_upsample_factor)) * unet_a_upsample_factor
                    if original_length != padded_length:
                        nl = F.pad(nl, (0, 0, 0, padded_length - original_length))
                        nlc = F.pad(nlc, (0, 0, 0, padded_length - original_length))
                noisy_latents.append(nl)
                noisy_latents_con.append(nlc)

            # 텍스트 임베딩 생성
            context = []
            class_labels = []
            text_input_processed = text_input
            if args.drop_rate_cfg > 0:
                null_tokens = ""
                index_null = np.where(np.random.rand(args.train_batch_size) < args.drop_rate_cfg)[0]
                if np.any(index_null):
                    text_input_processed = [null_tokens if i in list(index_null) else text_input[i] for i in range(args.train_batch_size)]
            for i, typ in enumerate(xtype):
                if typ == "audio":
                    text_inputs_a = tokenizer_a(text_input_processed, padding="max_length", max_length=tokenizer_a.model_max_length, truncation=True, return_tensors="pt")
                    prompt_embeds_a = text_encoder_a(text_inputs_a.input_ids.to(device), text_inputs_a.attention_mask.to(device))
                    prompt_embeds_a_pooled = F.normalize(prompt_embeds_a.text_embeds, dim=-1)
                    context.append(None)
                    class_labels.append(prompt_embeds_a_pooled)
                elif typ == "video":
                    text_inputs_v = tokenizer_v(text_input_processed, padding="max_length", max_length=tokenizer_v.model_max_length, truncation=True, return_tensors="pt")
                    prompt_embeds_v = text_encoder_v(text_inputs_v.input_ids.to(device), text_inputs_v.attention_mask.to(device))
                    context.append(prompt_embeds_v[0])
                    class_labels.append(None)


            # 영상 branch에 대해 청크 처리를 적용 (예: 프레임 단위 처리)
            # 기존 noisy_latents, timesteps, context, class_labels, x_con_dummy는, noisy_latents_con은 list
            def compute_joint_noise_pred_with_video_chunking(noisy_latents, timesteps, context, class_labels, xtype, x_con_dummy, chunk_size, train_model, device, dummy_downsample_factor=8):
                """
                Args:
                noisy_latents: 리스트, 각 모달리티별 latent tensor.
                    - 오디오: [B, Ca, H, W]
                    - 영상: [B, Cv, T, H, W]
                timesteps: 리스트, 모달리티별 timestep tensor.
                    - 영상: [B, T] 또는 [T]; 오디오는 시간 축이 없으므로 그대로 사용.
                context: 리스트, 각 모달리티별 condition tensor.
                class_labels: 리스트, 각 모달리티별 class label tensor.
                xtype: 리스트, 모달리티 이름 (예: ["audio", "video"]).
                x_con_dummy: 리스트, 각 모달리티별 x_con dummy tensor (미리 채널 확장이 적용된 값).
                chunk_size: int, 영상 branch의 시간 축(T)을 나눌 청크 크기 (예: 4).
                train_model: TrainingModel 인스턴스 (joint forward pass를 수행).
                device: torch.device, 최종 결과를 보낼 장치.
                dummy_downsample_factor: int, dummy video의 공간 해상도를 얼마나 줄일지 결정 (예: 4).
                
                Returns:
                noise_pred: 리스트, [audio_noise_pred, video_noise_pred] 형태.
                    - audio_noise_pred: 오디오 branch에 대해, 영상 branch 입력을 최소한의 (예: 첫 프레임) 값으로 대체한 후 forward pass로 계산한 결과.
                    - video_noise_pred: 영상 branch는 noisy_latents를 chunk_size 단위로 나누어 여러 번 forward pass한 후 시간(dim=2) 기준으로 이어 붙인 결과.
                """
                # 모달리티 인덱스 설정
                audio_idx = xtype.index("audio") if "audio" in xtype else None
                video_idx = xtype.index("video") if "video" in xtype else None

                # 1. 오디오 branch 계산
                # 영상 branch는 dummy video로 대체하여 오디오 branch를 계산합니다.
                B, Cv, T, H, W = noisy_latents[video_idx].shape
                # 4D로 변환: 첫 프레임 추출
                first_frame = noisy_latents[video_idx][:, :, 0, :, :]  # shape: [B, Cv, H, W]
                # 공간 downsample: 예를 들어 (H // dummy_downsample_factor, W // dummy_downsample_factor)
                import torch.nn.functional as F
                dummy_video = F.interpolate(first_frame, size=(H // dummy_downsample_factor, W // dummy_downsample_factor),
                                            mode='bilinear', align_corners=False)
                # 다시 시간 축을 추가: [B, Cv, 1, new_H, new_W]
                dummy_video = dummy_video.unsqueeze(2)
                
                # 복사본 생성
                noisy_latents_audio_input = noisy_latents.copy()
                timesteps_audio_input = timesteps.copy()
                # 영상 branch를 dummy video로 대체
                noisy_latents_audio_input[video_idx] = dummy_video
                if timesteps[video_idx].ndim == 2:
                    timesteps_audio_input[video_idx] = timesteps[video_idx][:, :1]
                else:
                    timesteps_audio_input[video_idx] = timesteps[video_idx][:1]
                
                noise_pred_audio_full = train_model(noisy_latents_audio_input, timesteps_audio_input, context, class_labels, xtype, x_con_dummy)
                audio_noise_pred = noise_pred_audio_full[audio_idx] if audio_idx is not None else None

                # 2. 영상 branch 계산: 청크 단위로 처리
                num_chunks = math.ceil(T / chunk_size)
                video_preds = []
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, T)
                    new_noisy_latents = noisy_latents.copy()
                    new_timesteps = timesteps.copy()
                    new_noisy_latents[video_idx] = noisy_latents[video_idx][:, :, start:end, :, :]
                    if timesteps[video_idx].ndim == 2:
                        new_timesteps[video_idx] = timesteps[video_idx][:, start:end]
                    else:
                        new_timesteps[video_idx] = timesteps[video_idx][start:end]
                    
                    noise_pred_chunk = train_model(new_noisy_latents, new_timesteps, context, class_labels, xtype, x_con_dummy)
                    video_preds.append(noise_pred_chunk[video_idx])
                    
                    del new_noisy_latents, new_timesteps, noise_pred_chunk
                    torch.cuda.empty_cache()
                
                final_video_pred = torch.cat(video_preds, dim=2)
                
                return [audio_noise_pred, final_video_pred]

            
            if args.use_x0_pred_for_connectors:
                block_con = None
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        torch.cuda.empty_cache()
                        gc.collect()
                        x_con_dummy = [torch.cat([tmp, torch.zeros_like(tmp, device=tmp.device)], dim=1) for tmp in noisy_latents_con]
                        
                        if args.short_memory:
                            noise_pred = compute_joint_noise_pred_with_video_chunking(
                                noisy_latents, timesteps, context, class_labels, xtype, x_con_dummy,
                                chunk_size=1,  # 예: 4 프레임씩 처리
                                train_model=train_model, device=device
                            )
                        else:                            
                            noise_pred = train_model(noisy_latents, timesteps, context, class_labels, xtype, x_con_dummy)

                        for i, typ in enumerate(xtype):
                            x_pred_list = []
                            for j in range(noise_pred[i].shape[0]):
                                if typ == "audio":
                                    x_pred = scheduler_a.step(noise_pred[i][j], timesteps[i][j], noisy_latents[i][j]).pred_original_sample.unsqueeze(0)
                                elif typ == "video":
                                    x_pred = scheduler_v.step(noise_pred[i][j], timesteps[i][j], noisy_latents[i][j]).pred_original_sample.unsqueeze(0)
                                x_pred_list.append(x_pred)
                            x_pred_tensor = torch.cat(x_pred_list, dim=0).to(device)
                            mask = (torch.rand(x_pred_tensor.shape[0], device=device) > 0.5).reshape([-1] + [1]*(len(x_pred_tensor.shape)-1))
                            x_pred_tensor = torch.where(mask, x_pred_tensor, torch.zeros_like(x_pred_tensor))
                            noisy_latents_con[i] = torch.cat([noisy_latents_con[i], x_pred_tensor], dim=1)

            optimizer.zero_grad()            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    model_pred = train_model(noisy_latents, timesteps, context, class_labels, xtype, noisy_latents_con, timesteps_con)
                    loss_list = []
                    for i in range(len(xtype)):
                        if scheduler_list[i].config.prediction_type == "epsilon":
                            target = noise[i]
                        elif scheduler_list[i].config.prediction_type == "v_prediction":
                            target = scheduler_list[i].get_velocity(latents[i], noise[i], timesteps[i])
                        else:
                            raise ValueError(f"Unknown prediction type {scheduler_list[i].config.prediction_type}")
                        loss_list.append(F.mse_loss(model_pred[i].float(), target.float(), reduction="none"))
                    loss_scalar = sum(l.mean() for l in loss_list)
                scaler.scale(loss_scalar).backward()
            else:
                with torch.cuda.amp.autocast():
                    model_pred = train_model(noisy_latents, timesteps, context, class_labels, xtype, noisy_latents_con, timesteps_con)
                    loss_list = []
                    for i in range(len(xtype)):
                        if scheduler_list[i].config.prediction_type == "epsilon":
                            target = noise[i]
                        elif scheduler_list[i].config.prediction_type == "v_prediction":
                            target = scheduler_list[i].get_velocity(latents[i], noise[i], timesteps[i])
                        else:
                            raise ValueError(f"Unknown prediction type {scheduler_list[i].config.prediction_type}")
                        loss_list.append(F.mse_loss(model_pred[i].float(), target.float(), reduction="none"))
                    loss_scalar = sum(l.mean() for l in loss_list)
                loss_scalar.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.use_ema:
                    ema_unet.step(train_model.unet_svg.parameters())

                progress_bar.update(1)
                for i in range(len(xtype)):
                    logger.info(f"Step {global_step} - Loss {i}: {loss_list[i].mean().item()}")

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': train_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    }, save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

        if args.validation_epochs > 0 and (epoch + 1) % args.validation_epochs == 0:
            if args.use_ema:
                ema_unet.store(train_model.unet_svg.parameters())
                ema_unet.copy_to(train_model.unet_svg.parameters())
            log_validation(test_dataloader,
                           vae_a=vae_a,
                           text_encoder_a=text_encoder_a,
                           tokenizer_a=tokenizer_a,
                           vocoder=vocoder,
                           scheduler_a=scheduler_a,
                           unet_a=train_model.unet_a,
                           vae_v=vae_v,
                           text_encoder_v=text_encoder_v,
                           tokenizer_v=tokenizer_v,
                           scheduler_v=scheduler_v,
                           unet_v=train_model.unet_v,
                           unet_svg=train_model.unet_svg,
                           args=args,                           
                           epoch=epoch+1,
                           device=device,
                           weight_dtype=weight_dtype)
            if args.use_ema:
                ema_unet.restore(train_model.unet_svg.parameters())

    # 최종 파이프라인 저장
    pipe = VideoAudioGenWithSVGPipeline(
        vae_a=vae_a,
        text_encoder_a=text_encoder_a,
        tokenizer_a=tokenizer_a,
        vocoder=vocoder,
        scheduler_a=scheduler_a,
        unet_a=train_model.unet_a,
        vae_v=vae_v,
        text_encoder_v=text_encoder_v,
        tokenizer_v=tokenizer_v,
        scheduler_v=scheduler_v,
        unet_v=train_model.unet_v,
        unet_svg=train_model.unet_svg,
    )
    pipe.save_pretrained(os.path.join(args.output_dir, "pipe"), safe_serialization=False)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
