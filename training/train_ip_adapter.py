#!/usr/bin/env python3
"""
training/train_ip_adapter.py
------------------------------
Train IP-Adapter for reference image conditioning.

Architecture:
  - Frozen CLIP image encoder extracts reference embeddings
  - IP-Adapter cross-attention layers project image tokens into UNet attention
  - Only IP-Adapter projection weights are trained

Usage:
    accelerate launch training/train_ip_adapter.py \
        --config configs/train_ip_adapter.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── IP-Adapter Module ────────────────────────────────────────────────────────
class ImageProjModel(nn.Module):
    """
    Projects CLIP image embeddings to match text embedding dimensionality.
    Used as the IP-Adapter image projection layer.
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        clip_embeddings_dim: int = 1024,
        clip_extra_context_tokens: int = 16,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(
            clip_embeddings_dim,
            self.clip_extra_context_tokens * cross_attention_dim,
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        embeds = image_embeds
        clip_extra = (
            self.proj(embeds)
            .reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        )
        return self.norm(clip_extra)


class IPAdapterAttnProcessor(nn.Module):
    """
    Attention processor for IP-Adapter.
    Extends standard cross-attention with additional image cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 16,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        # Linear projections for image cross-attention
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # Split encoder_hidden_states into text + image tokens
        if encoder_hidden_states is not None:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states_text = encoder_hidden_states[:, :end_pos]
            encoder_hidden_states_img = encoder_hidden_states[:, end_pos:]
        else:
            encoder_hidden_states_text = hidden_states
            encoder_hidden_states_img = None

        if attn.norm_cross:
            encoder_hidden_states_text = attn.norm_encoder_hidden_states(encoder_hidden_states_text)

        key = attn.to_k(encoder_hidden_states_text)
        value = attn.to_v(encoder_hidden_states_text)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # IP-Adapter image cross-attention
        if encoder_hidden_states_img is not None:
            ip_key = self.to_k_ip(encoder_hidden_states_img)
            ip_value = self.to_v_ip(encoder_hidden_states_img)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)
            ip_attention_probs = attn.get_attention_scores(query, ip_key)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
            hidden_states = hidden_states + self.scale * ip_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def get_ip_dataset(config: dict, tokenizer, clip_processor):
    """Dataset for IP-Adapter: (reference_image, target_image, prompt)."""
    import random
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import Dataset

    class IPAdapterDataset(Dataset):
        def __init__(self, metadata_file: str, resolution: int = 512):
            self.samples = []
            with open(metadata_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        if entry.get("split") == "train" and "image_path" in entry:
                            self.samples.append(entry)

            self.resolution = resolution
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            logger.info(f"IP-Adapter Dataset: {len(self.samples)} samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            target_img = Image.open(sample["image_path"]).convert("RGB")
            target_tensor = self.img_transform(target_img)

            # Use same image as reference (self-supervised) or random paired image
            uncond_drop = random.random() < config["training"].get("uncond_ratio", 0.1)
            if uncond_drop:
                # Drop reference: use blank image
                ref_tensor = torch.zeros(3, self.resolution, self.resolution)
            else:
                ref_tensor = self.img_transform(target_img)

            # CLIP process reference image
            clip_input = clip_processor(images=target_img, return_tensors="pt")
            clip_pixel_values = clip_input.pixel_values.squeeze(0)

            input_ids = tokenizer(
                sample.get("prompt", "a garment, fashion photography"),
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.squeeze(0)

            return {
                "pixel_values": target_tensor,
                "ref_pixel_values": ref_tensor,
                "clip_pixel_values": clip_pixel_values,
                "input_ids": input_ids,
                "is_uncond": uncond_drop,
            }

    return IPAdapterDataset(
        metadata_file=config["data"]["metadata_file"],
        resolution=config["training"]["resolution"],
    )


def train(config: dict) -> None:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
    )

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
    image_encoder_id = config["model"]["image_encoder"]

    # ── Load components ──
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # CLIP image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_id)
    clip_processor = CLIPImageProcessor.from_pretrained(image_encoder_id)

    # ── Load LoRA (merge into UNet so from_unet works cleanly) ──
    lora_weights = config["model"].get("lora_weights")
    if lora_weights and Path(lora_weights).exists():
        from peft import PeftModel
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet = PeftModel.from_pretrained(unet, lora_weights)
        unet = unet.merge_and_unload()
        # Reset attn processors to standard fp32-compatible ones
        unet.set_attn_processor(AttnProcessor2_0())
        logger.info(f"LoRA weights merged into UNet from {lora_weights}")

    # ── IP-Adapter modules ──
    num_tokens = config["model"]["num_tokens"]
    cross_attn_dim = unet.config.cross_attention_dim
    clip_embed_dim = image_encoder.config.projection_dim

    image_proj = ImageProjModel(
        cross_attention_dim=cross_attn_dim,
        clip_embeddings_dim=clip_embed_dim,
        clip_extra_context_tokens=num_tokens,
    )

    # Replace UNet attention processors with IP-Adapter processors
    # For SDXL, attention_head_dim is a tuple — derive hidden_size per block from to_k weight shape
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        if "attn2" in name:
            # hidden_size = output dim of the to_k projection = num_heads * head_dim
            to_k_key = name.replace(".processor", ".to_k.weight")
            if to_k_key in unet_sd:
                hidden_size = unet_sd[to_k_key].shape[0]
            else:
                hidden_size = cross_attn_dim  # fallback
            attn_procs[name] = IPAdapterAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attn_dim,
                num_tokens=num_tokens,
            )
        else:
            from diffusers.models.attention_processor import AttnProcessor2_0
            attn_procs[name] = AttnProcessor2_0()
    unet.set_attn_processor(attn_procs)


    # ── Freeze everything except IP-Adapter modules ──
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Unfreeze IP-Adapter projection weights
    image_proj.requires_grad_(True)
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, IPAdapterAttnProcessor):
            proc.requires_grad_(True)

    # ── Trainable params ──
    params_to_train = list(image_proj.parameters()) + [
        p for proc in unet.attn_processors.values()
        if isinstance(proc, IPAdapterAttnProcessor)
        for p in proc.parameters()
    ]
    logger.info(f"Trainable IP-Adapter params: {sum(p.numel() for p in params_to_train):,}")

    # ── Dataset ──
    dataset = get_ip_dataset(config, tokenizer, clip_processor)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    optimizer = AdamW(
        params_to_train,
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
    )

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )

    image_proj, optimizer, loader, lr_scheduler = accelerator.prepare(
        image_proj, optimizer, loader, lr_scheduler
    )

    # Keep all frozen models in fp32 — avoids Half/Float conflicts
    vae = vae.to(accelerator.device)          # fp32, frozen
    unet = unet.to(accelerator.device)        # fp32, frozen params
    text_encoder = text_encoder.to(accelerator.device)   # fp32, frozen
    image_encoder = image_encoder.to(accelerator.device) # fp32, frozen

    global_step = 0
    max_steps = config["training"]["max_train_steps"]

    progress_bar = tqdm(
        total=max_steps,
        disable=not accelerator.is_main_process,
        desc="Training IP-Adapter",
    )

    for epoch in range(config["training"]["num_train_epochs"]):
        image_proj.train()
        for batch in loader:
            with accelerator.accumulate(image_proj):
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                    ).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CLIP image embeddings
                with torch.no_grad():
                    clip_feats = image_encoder(
                        batch["clip_pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                    ).image_embeds.float()

                # Project to IP tokens (trainable — stays in grad graph)
                ip_tokens = image_proj(clip_feats)  # [B, num_tokens, cross_attn_dim]

                # Text encoding
                with torch.no_grad():
                    text_embeds = text_encoder(
                        batch["input_ids"].to(accelerator.device)
                    )[0].float()

                # Concatenate text + image tokens (fp32)
                encoder_hidden_states = torch.cat([text_embeds, ip_tokens], dim=1).float()

                # UNet forward — NO no_grad: ip_tokens must stay in computation graph
                model_pred = unet(
                    noisy_latents.float(),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at step {global_step}, skipping")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_train, config["training"]["max_grad_norm"])

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

                if global_step % config["checkpointing"]["save_steps"] == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(exist_ok=True)
                        torch.save(
                            accelerator.unwrap_model(image_proj).state_dict(),
                            str(ckpt_dir / "image_proj.pth"),
                        )
                        logger.info(f"IP-Adapter checkpoint → {ckpt_dir}")

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # ── Final save ──
    if accelerator.is_main_process:
        torch.save(
            accelerator.unwrap_model(image_proj).state_dict(),
            str(output_dir / "image_proj.pth"),
        )
        # Save IP-Adapter attention processor weights
        ip_attn_state = {
            name: proc.state_dict()
            for name, proc in unet.attn_processors.items()
            if isinstance(proc, IPAdapterAttnProcessor)
        }
        torch.save(ip_attn_state, str(output_dir / "ip_adapter_attn.pth"))
        logger.success(f"IP-Adapter training complete → {output_dir}")

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for reference conditioning")
    parser.add_argument("--config", default="configs/train_ip_adapter.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
