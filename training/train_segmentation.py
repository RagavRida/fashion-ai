#!/usr/bin/env python3
"""
training/train_segmentation.py
---------------------------------
Trains the GarmentUNet segmentation model on DeepFashion masked images.

Features:
  - Accelerate-based distributed training
  - Mixed precision (fp16)
  - W&B logging
  - Checkpoint save/resume
  - DiceBCE loss
  - IoU tracking

Usage:
    accelerate launch training/train_segmentation.py \
        --config configs/train_segmentation.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.segmentation.unet import DiceBCELoss, GarmentUNet, compute_iou


# ─── Dataset ─────────────────────────────────────────────────────────────────
class GarmentSegDataset(Dataset):
    """Dataset of (image, mask) pairs from metadata JSONL."""

    def __init__(
        self,
        metadata_file: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
    ):
        self.samples = []
        with open(metadata_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("split") == split:
                    if "image_path" in entry and "mask_path" in entry:
                        self.samples.append(entry)

        self.image_size = image_size
        self.augment = augment and (split == "train")

        # Image transforms
        img_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        if self.augment:
            img_transforms.insert(1, transforms.RandomHorizontalFlip())
            img_transforms.insert(2, transforms.ColorJitter(brightness=0.2, contrast=0.2))
        self.img_transform = transforms.Compose(img_transforms)

        # Mask transforms (preserve binary values)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1] float
        ])

        logger.info(f"GarmentSegDataset [{split}]: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")
        img_tensor = self.img_transform(img)

        # Load mask
        mask = Image.open(sample["mask_path"]).convert("L")
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()  # binarize

        return img_tensor, mask_tensor


# ─── Training Loop ────────────────────────────────────────────────────────────
def train(config: dict) -> None:
    from accelerate import Accelerator

    train_cfg = config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=config["precision"]["mixed_precision"],
        log_with=train_cfg.get("report_to", None),
        project_config=None,
    )

    if accelerator.is_main_process:
        if train_cfg.get("report_to") == "wandb":
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                name=config["logging"]["wandb_run_name"],
                config=config,
            )

    # ── Model ──
    model = GarmentUNet(
        encoder=config["model"]["encoder"],
        pretrained=config["model"]["pretrained_encoder"],
        num_classes=config["model"]["num_classes"],
    )
    loss_fn = DiceBCELoss(
        dice_weight=config["training"]["dice_weight"],
        bce_weight=config["training"]["bce_weight"],
    )

    # ── Data ──
    train_dataset = GarmentSegDataset(
        config["data"]["metadata_file"],
        split="train",
        image_size=config["training"]["resolution"],
        augment=True,
    )
    val_dataset = GarmentSegDataset(
        config["data"]["metadata_file"],
        split="val",
        image_size=config["training"]["resolution"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["dataloader_num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Optimizer ──
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["adam_weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=train_cfg["lr_scheduler_patience"],
        factor=train_cfg["lr_scheduler_factor"],
    )

    # ── Accelerate prep ──
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ── Training loop ──
    best_iou = 0.0
    early_stop_counter = 0
    patience = train_cfg.get("early_stopping_patience", 10)

    for epoch in range(1, train_cfg["num_train_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        n_batches = 0

        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch} [train]", disable=not accelerator.is_main_process
        ):
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            accelerator.backward(loss)

            nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["max_grad_norm"]
            )
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                batch_iou = compute_iou(pred, masks)

            train_loss += loss.item()
            train_iou += batch_iou
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_iou = train_iou / max(n_batches, 1)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        n_val = 0

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch} [val]", disable=not accelerator.is_main_process
            ):
                logits = model(images)
                loss = loss_fn(logits, masks)
                pred = (torch.sigmoid(logits) > 0.5).float()
                val_loss += loss.item()
                val_iou += compute_iou(pred, masks)
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_iou = val_iou / max(n_val, 1)
        scheduler.step(avg_val_loss)

        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} "
                f"train_iou={avg_train_iou:.4f} | "
                f"val_loss={avg_val_loss:.4f} val_iou={avg_val_iou:.4f}"
            )

            if train_cfg.get("report_to") == "wandb":
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/iou": avg_train_iou,
                    "val/loss": avg_val_loss,
                    "val/iou": avg_val_iou,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            # ── Save best ──
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                early_stop_counter = 0
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_checkpoint(
                    str(output_dir / "best_model.pth"),
                    epoch=epoch,
                    metrics={"val_iou": avg_val_iou},
                )
                logger.success(f"✓ New best IoU: {best_iou:.4f} → saved")
            else:
                early_stop_counter += 1

            # ── Periodic save ──
            save_freq = config["checkpointing"].get("save_epoch_frequency", 5)
            if epoch % save_freq == 0:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_checkpoint(
                    str(output_dir / f"epoch_{epoch:04d}.pth"),
                    epoch=epoch,
                    metrics={"val_iou": avg_val_iou},
                )

        # ── Early stopping ──
        if early_stop_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    accelerator.end_training()
    logger.success(f"Training complete. Best val IoU: {best_iou:.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train garment segmentation U-Net")
    parser.add_argument("--config", default="configs/train_segmentation.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
