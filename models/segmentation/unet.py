#!/usr/bin/env python3
"""
models/segmentation/unet.py
-----------------------------
Lightweight U-Net for binary garment segmentation (foreground / background).

Architecture:
  - Encoder: ResNet34 pretrained backbone (torchvision)
  - Decoder: Progressive upsampling with skip connections
  - Output: binary segmentation mask [B, 1, H, W]

Usage:
    from models.segmentation.unet import GarmentUNet

    model = GarmentUNet(encoder="resnet34", pretrained=True)
    mask = model(image_tensor)   # [B, 1, 512, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loguru import logger


# ─── Building Blocks ──────────────────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two ConvBnRelu blocks."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_c, out_c),
            ConvBnRelu(out_c, out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + skip connection + double conv."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c // 2 + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatch due to odd input dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── Garment U-Net ────────────────────────────────────────────────────────────
class GarmentUNet(nn.Module):
    """
    Garment segmentation U-Net with ResNet encoder.

    Args:
        encoder: "resnet18" | "resnet34" | "resnet50"
        pretrained: use ImageNet pretrained encoder
        num_classes: 1 for binary, >1 for multi-class
    """

    ENCODER_CHANNELS = {
        "resnet18":  [64, 64, 128, 256, 512],
        "resnet34":  [64, 64, 128, 256, 512],
        "resnet50":  [64, 256, 512, 1024, 2048],
    }

    def __init__(
        self,
        encoder: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_name = encoder
        channels = self.ENCODER_CHANNELS[encoder]

        # ── Encoder (ResNet backbone) ──
        if encoder == "resnet18":
            backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
        elif encoder == "resnet34":
            backbone = models.resnet34(weights="DEFAULT" if pretrained else None)
        elif encoder == "resnet50":
            backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        # Extract encoder stages
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)   # 512 → 256, 64ch
        self.pool = backbone.maxpool                                               # 256 → 128
        self.enc1 = backbone.layer1   # 128, channels[1]
        self.enc2 = backbone.layer2   # 64,  channels[2]
        self.enc3 = backbone.layer3   # 32,  channels[3]
        self.enc4 = backbone.layer4   # 16,  channels[4] (bottleneck)

        # ── Bridge ──
        self.bridge = DoubleConv(channels[4], channels[4])

        # ── Decoder ──
        self.dec4 = DecoderBlock(channels[4], channels[3], 256)
        self.dec3 = DecoderBlock(256,         channels[2], 128)
        self.dec2 = DecoderBlock(128,         channels[1], 64)
        self.dec1 = DecoderBlock(64,          channels[0], 32)

        # Extra upsample to get back to full resolution
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        logger.info(
            f"GarmentUNet initialized | encoder={encoder} | pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalized garment image
        Returns:
            logits: [B, num_classes, H, W]
        """
        # Encoder
        e0 = self.enc0(x)    # stride 2
        p  = self.pool(e0)   # stride 4
        e1 = self.enc1(p)    # stride 4
        e2 = self.enc2(e1)   # stride 8
        e3 = self.enc3(e2)   # stride 16
        e4 = self.enc4(e3)   # stride 32

        # Bridge
        b = self.bridge(e4)

        # Decoder
        d4 = self.dec4(b,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        out = self.final_up(d1)
        out = self.dropout(out)
        out = self.head(out)
        return out

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary mask: [B, 1, H, W] with values {0, 1}."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "GarmentUNet":
        """Load model from checkpoint."""
        import torch
        state = torch.load(checkpoint_path, map_location="cpu")
        model_kwargs = state.get("model_kwargs", kwargs)
        model = cls(**model_kwargs)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"GarmentUNet loaded from {checkpoint_path}")
        return model

    def save_checkpoint(
        self, path: str, epoch: int = 0, metrics: dict = None
    ) -> None:
        """Save model checkpoint."""
        import torch
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_kwargs": {
                    "encoder": self.encoder_name,
                    "num_classes": self.head.out_channels,
                },
                "epoch": epoch,
                "metrics": metrics or {},
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")


# ─── Loss Functions ──────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        return 1 - (2.0 * intersection + smooth) / (
            probs_flat.sum() + targets_flat.sum() + smooth
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ─── IoU Metric ──────────────────────────────────────────────────────────────
def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Compute IoU between binary masks."""
    pred = (pred_mask > 0.5).float()
    true = (true_mask > 0.5).float()
    intersection = (pred * true).sum().item()
    union = (pred + true).clamp(0, 1).sum().item()
    return intersection / (union + 1e-8)
