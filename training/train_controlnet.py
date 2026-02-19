#!/usr/bin/env python3
"""
training/train_controlnet.py
------------------------------
Train ControlNet on top of the LoRA fine-tuned diffusion model.
Conditions on Canny edge maps for garment structure consistency.

Features:
  - Loads LoRA weights from Phase 1
  - Initializes ControlNet from UNet weights
  - Edge conditioning dropout for classifier-free guidance
  - Accelerate multi-GPU / mixed precision
  - W&B logging

Usage:
    accelerate launch training/train_controlnet.py \
        --config configs/train_controlnet.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_controlnet_dataset(config: dict, tokenizer):
    """Dataset with (image, edge_map, prompt) triples."""
    import json
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import Dataset

    class ControlNetDataset(Dataset):
        def __init__(self, metadata_file: str, resolution: int = 512):
            self.samples = []
            with open(metadata_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if (
                            entry.get("split") == "train"
                            and "image_path" in entry
                            and "edge_path" in entry
                            and "prompt" in entry
                        ):
                            self.samples.append(entry)

            self.resolution = resolution
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.edge_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
            ])
            logger.info(f"ControlNet Dataset: {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            img = Image.open(sample["image_path"]).convert("RGB")
            edge = Image.open(sample["edge_path"]).convert("RGB")  # 3-channel for ControlNet

            img_tensor = self.img_transform(img)
            edge_tensor = self.edge_transform(edge)

            input_ids = tokenizer(
                sample["prompt"],
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.squeeze(0)

            return {
                "pixel_values": img_tensor,
                "conditioning_pixel_values": edge_tensor,
                "input_ids": input_ids,
            }

    return ControlNetDataset(
        config["data"]["metadata_file"],
        resolution=config["training"]["resolution"],
    )


def train(config: dict) -> None:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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
        accelerator.init_trackers(
            project_name=config["logging"]["wandb_project"],
            config=config,
            init_kwargs={"wandb": {"name": config["logging"]["wandb_run_name"]}},
        )

    model_id = config["model"]["base_model"]

    # ── Load UNet + tokenizer ──
    logger.info(f"Loading base model: {model_id}")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # ── Load LoRA weights onto UNet ──
    lora_weights = config["model"].get("lora_weights")
    if lora_weights and Path(lora_weights).exists():
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, lora_weights)
        unet = unet.merge_and_unload()   # bake LoRA into base weights → plain UNet
        logger.info(f"LoRA weights merged into UNet from {lora_weights}")

    # ── Initialize ControlNet from UNet ──
    controlnet_path = config["model"].get("controlnet_model_name_or_path")
    if controlnet_path and Path(str(controlnet_path)).exists():
        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        logger.info(f"ControlNet loaded from {controlnet_path}")
    else:
        controlnet = ControlNetModel.from_unet(unet)
        logger.info("ControlNet initialized from (LoRA-merged) UNet weights")

    # ── Freeze non-ControlNet weights ──
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    controlnet.train()

    if config["precision"]["gradient_checkpointing"]:
        controlnet.enable_gradient_checkpointing()

    if config["precision"]["enable_xformers"]:
        try:
            controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # ── Dataset ──
    dataset = get_controlnet_dataset(config, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ──
    optimizer = AdamW(
        controlnet.parameters(),
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

    # ── Accelerate prep ──
    controlnet, optimizer, loader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, loader, lr_scheduler
    )
    vae = vae.half().to(accelerator.device)
    unet = unet.half().to(accelerator.device)
    text_encoder = text_encoder.half().to(accelerator.device)
    text_encoder_2 = text_encoder_2.half().to(accelerator.device)

    global_step = 0
    max_steps = config["training"]["max_train_steps"]
    conditioning_dropout = config["training"].get("conditioning_dropout_prob", 0.05)

    progress_bar = tqdm(
        total=max_steps,
        disable=not accelerator.is_main_process,
        desc="Training ControlNet",
    )

    for epoch in range(config["training"]["num_train_epochs"]):
        for batch in loader:
            with accelerator.accumulate(controlnet):
                # Encode to latents
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(device=accelerator.device, dtype=torch.float16)
                    ).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.float()  # back to fp32 for UNet

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Conditioning dropout for CFG training
                cond_images = batch["conditioning_pixel_values"].to(
                    device=accelerator.device, dtype=torch.float32
                )
                if conditioning_dropout > 0:
                    drop_mask = torch.rand(bsz) < conditioning_dropout
                    cond_images = cond_images.clone()
                    cond_images[drop_mask] = 0.0  # zero out conditioning

                # Text encoding — SDXL dual encoder
                with torch.no_grad():
                    enc1_out = text_encoder(batch["input_ids"])
                    encoder_hidden_states = enc1_out.last_hidden_state  # [B, 77, 768]

                    # Tokenize with tokenizer_2 on the fly
                    prompts = tokenizer.batch_decode(
                        batch["input_ids"], skip_special_tokens=True
                    )
                    ids2 = tokenizer_2(
                        prompts, truncation=True,
                        max_length=tokenizer_2.model_max_length,
                        padding="max_length", return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                    enc2_out = text_encoder_2(ids2)
                    encoder_hidden_states_2 = enc2_out.last_hidden_state  # [B, 77, 1280]
                    pooled_output_2 = enc2_out.text_embeds               # [B, 1280]

                    combined_hidden_states = torch.cat(
                        [encoder_hidden_states, encoder_hidden_states_2], dim=-1
                    )  # [B, 77, 2048]
                    bs = latents.shape[0]
                    add_time_ids = torch.tensor(
                        [[512, 512, 0, 0, 512, 512]], dtype=torch.float32,
                        device=latents.device
                    ).repeat(bs, 1)
                    added_cond_kwargs = {
                        "text_embeds": pooled_output_2,
                        "time_ids": add_time_ids,
                    }

                # ControlNet forward (runs in fp16 via accelerate autocast)
                nl_fp16 = noisy_latents.half()
                hs_fp16 = combined_hidden_states.half()
                cond_fp16 = cond_images.half()
                acond_fp16 = {
                    "text_embeds": pooled_output_2.half(),
                    "time_ids": add_time_ids.half(),
                }

                down_block_res, mid_block_res = controlnet(
                    nl_fp16,
                    timesteps,
                    encoder_hidden_states=hs_fp16,
                    controlnet_cond=cond_fp16,
                    added_cond_kwargs=acond_fp16,
                    return_dict=False,
                )

                # UNet forward — also feed fp16 so attention layers match
                model_pred = unet(
                    nl_fp16,
                    timesteps,
                    encoder_hidden_states=hs_fp16,
                    down_block_additional_residuals=[r.half() for r in down_block_res],
                    mid_block_additional_residual=mid_block_res.half(),
                    added_cond_kwargs=acond_fp16,
                ).sample.float()  # back to fp32 for loss

                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), noise.float()
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        controlnet.parameters(), config["training"]["max_grad_norm"]
                    )

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

                # Checkpoint
                if global_step % config["checkpointing"]["save_steps"] == 0:
                    if accelerator.is_main_process:
                        ckpt_path = output_dir / f"checkpoint-{global_step}"
                        accelerator.unwrap_model(controlnet).save_pretrained(
                            str(ckpt_path)
                        )
                        logger.info(f"ControlNet checkpoint → {ckpt_path}")

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # Final save
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(str(output_dir))
        logger.success(f"ControlNet trained and saved → {output_dir}")

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet for fashion edges")
    parser.add_argument("--config", default="configs/train_controlnet.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
