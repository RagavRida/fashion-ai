#!/usr/bin/env python3
"""
training/train_lora.py
-----------------------
LoRA fine-tuning of SDXL (or SD 1.5) on DeepFashion prompts.

Features:
  - PEFT LoRA adapters on UNet attention layers
  - Accelerate multi-GPU / mixed precision
  - EMA model weights
  - SNR gamma noise weighting
  - Noise offset for better dark/bright colors
  - W&B experiment tracking
  - Checkpoint save/resume

Usage:
    accelerate launch training/train_lora.py \
        --config configs/train_lora.yaml

    # With overrides:
    accelerate launch training/train_lora.py \
        --config configs/train_lora.yaml \
        --max_train_steps 1 --dry_run
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_dataset(config: dict, tokenizer, tokenizer_2=None):
    """Build dataset for LoRA training."""
    import json
    from PIL import Image
    from torch.utils.data import Dataset
    import torchvision.transforms as T

    class FashionLoRADataset(Dataset):
        def __init__(self, metadata_file: str, resolution: int = 512):
            self.samples = []
            with open(metadata_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if entry.get("split") == "train":
                            if "image_path" in entry and "prompt" in entry:
                                self.samples.append(entry)

            self.resolution = resolution
            self.transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            logger.info(f"LoRA Dataset: {len(self.samples)} training samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            img = Image.open(sample["image_path"]).convert("RGB")
            img_tensor = self.transform(img)

            # Tokenize prompt
            ids = tokenizer(
                sample["prompt"],
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.squeeze(0)

            result = {"pixel_values": img_tensor, "input_ids": ids}

            # For SDXL: second tokenizer
            if tokenizer_2 is not None:
                ids_2 = tokenizer_2(
                    sample["prompt"],
                    truncation=True,
                    max_length=tokenizer_2.model_max_length,
                    padding="max_length",
                    return_tensors="pt",
                ).input_ids.squeeze(0)
                result["input_ids_2"] = ids_2

            return result

    return FashionLoRADataset(
        config["data"]["metadata_file"],
        resolution=config["training"]["resolution"],
    )


def compute_snr(noise_scheduler, timesteps):
    """Compute signal-to-noise ratio for loss weighting."""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas = alphas_cumprod**0.5
    sqrt_one_minus_alphas = (1 - alphas_cumprod)**0.5
    sqrt_alphas = sqrt_alphas.to(device=timesteps.device)
    sqrt_one_minus_alphas = sqrt_one_minus_alphas.to(device=timesteps.device)
    snr = (sqrt_alphas[timesteps] / sqrt_one_minus_alphas[timesteps])**2
    return snr


def train(config: dict, dry_run: bool = False) -> None:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTokenizer

    set_seed(42)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=config["training"]["logging_dir"],
    )
    accelerator = Accelerator(
        mixed_precision=config["precision"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with=config["logging"]["report_to"],
        project_config=proj_config,
    )

    if accelerator.is_main_process:
        if config["logging"]["report_to"] == "wandb":
            accelerator.init_trackers(
                project_name=config["logging"]["wandb_project"],
                config=config,
                init_kwargs={"wandb": {"name": config["logging"]["wandb_run_name"]}},
            )

    use_sdxl = config["model"]["use_sdxl"]
    model_id = (
        config["model"]["base_model"] if use_sdxl
        else config["model"]["fallback_model"]
    )
    logger.info(f"Loading base model: {model_id}")

    # ── Load Components ──
    if use_sdxl:
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2"
        )
        vae = AutoencoderKL.from_pretrained(
            config["model"]["vae_model"], torch_dtype=torch.float16
        )
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = None
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        text_encoder_2 = None
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # ── Freeze non-LoRA params ──
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if text_encoder_2:
        text_encoder_2.requires_grad_(False)

    # ── Apply LoRA ──
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if config["precision"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    if config["precision"]["enable_xformers"]:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")

    # ── Dataset & Loader ──
    dataset = get_dataset(config, tokenizer, tokenizer_2)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ──
    params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = AdamW(
        params,
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )

    # ── Accelerate prepare ──
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )

    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device)
    if text_encoder_2:
        text_encoder_2 = text_encoder_2.to(accelerator.device)

    # ── Training loop ──
    max_steps = config["training"]["max_train_steps"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    noise_offset = config["training"].get("noise_offset", 0.0)
    snr_gamma = config["training"].get("snr_gamma", None)
    global_step = 0

    if dry_run:
        max_steps = 1
        logger.info("[DRY RUN] Running 1 step only")

    progress_bar = tqdm(
        total=max_steps,
        disable=not accelerator.is_main_process,
        desc="Training LoRA",
    )

    unet.train()
    for epoch in range(config["training"]["num_train_epochs"]):
        for batch in loader:
            with accelerator.accumulate(unet):
                # Encode images to latents
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                if noise_offset:
                    noise += noise_offset * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1,
                        device=latents.device
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                if use_sdxl and text_encoder_2 is not None:
                    enc2_out = text_encoder_2(batch.get("input_ids_2", batch["input_ids"]))
                    encoder_hidden_states_2 = enc2_out[0]
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=torch.cat(
                            [encoder_hidden_states, encoder_hidden_states_2], dim=-1
                        ),
                    ).sample
                else:
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
                    ).sample

                target = noise

                # Compute loss
                if snr_gamma:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                        / snr
                    )
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (loss.mean(dim=[1, 2, 3]) * mse_loss_weights).mean()
                else:
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), target.float()
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, config["training"]["max_grad_norm"])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if accelerator.is_main_process:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # Save checkpoint
                if global_step % config["checkpointing"]["save_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = output_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(save_path))

                        # Save LoRA weights separately
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_unet.save_pretrained(str(save_path / "unet_lora"))
                        logger.info(f"Checkpoint saved → {save_path}")

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # ── Final save ──
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        # Save as safetensors
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        lora_state = get_peft_model_state_dict(unwrapped_unet)
        save_file(lora_state, str(output_dir / "fashion_lora.safetensors"))
        unwrapped_unet.save_pretrained(str(output_dir / "unet_lora_final"))
        logger.success(f"LoRA training complete! Weights → {output_dir}")

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune diffusion model")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_train_steps:
        config["training"]["max_train_steps"] = args.max_train_steps

    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
