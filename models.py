"""
models.py — Steganography Network Architectures
================================================
Three competing networks:

  Encoder  : Embeds a secret payload into a cover image  (the Stegger)
  Decoder  : Extracts the payload back out               (proves payload survived)
  Warden   : Binary classifier — clean vs stego          (the Detective)

Architecture based on HiDDeN (Zhu et al., ECCV 2018), scaled for CPU training.
See REFERENCES.md for full citations.
"""

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Embeds a binary payload into a cover image as an imperceptible residual.

    The payload is tiled spatially and concatenated with image features,
    letting the network learn WHERE and HOW to hide each bit on its own.

    Input:
        cover    : (B, 3, H, W)  — normalized cover image in [-1, 1]
        payload  : (B, L)        — binary bits (0.0 or 1.0)
    Output:
        stego    : (B, 3, H, W)  — visually similar to cover
    """
    def __init__(self, payload_len=32, image_ch=3, hidden_dim=64):
        super().__init__()
        self.payload_len = payload_len

        self.cover_branch = nn.Sequential(
            ConvBnRelu(image_ch, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
        )
        self.joint_branch = nn.Sequential(
            ConvBnRelu(hidden_dim + payload_len, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, image_ch, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, cover, payload):
        B, _, H, W = cover.shape
        payload_map = payload.unsqueeze(-1).unsqueeze(-1).expand(B, self.payload_len, H, W)
        features = self.cover_branch(cover)
        combined = torch.cat([features, payload_map], dim=1)
        residual = self.joint_branch(combined)
        stego = torch.clamp(cover + 0.1 * residual, -1.0, 1.0)
        return stego


class Decoder(nn.Module):
    """
    Extracts the hidden payload from a stego image.

    Low BER (< 0.1) means payload survived. High BER means the encoder
    sacrificed recoverability to evade the warden — a key research tension.

    Input:
        stego    : (B, 3, H, W)
    Output:
        logits   : (B, L)  — raw logits; use BCEWithLogitsLoss in training,
                             sigmoid + threshold at 0.5 for inference
    """
    def __init__(self, payload_len=32, image_ch=3, hidden_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBnRelu(image_ch, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
            ConvBnRelu(hidden_dim, hidden_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(hidden_dim, payload_len)

    def forward(self, stego):
        features = self.conv_layers(stego).squeeze(-1).squeeze(-1)
        return self.fc(features)


class Warden(nn.Module):
    """
    Binary steganalysis classifier: clean (0) vs stego (1).

    Trained adversarially against the Encoder. When accuracy > 0.7,
    the Encoder is pressured to adapt. When fooled to ~0.5, the Warden
    must retrain. This oscillation IS the arms race.

    Input:
        image    : (B, 3, H, W)
    Output:
        logit    : (B, 1)  — raw logit; sigmoid > 0.5 means stego detected
    """
    def __init__(self, image_ch=3, hidden_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(image_ch, hidden_dim),
            nn.MaxPool2d(2),
            ConvBnRelu(hidden_dim, hidden_dim * 2),
            nn.MaxPool2d(2),
            ConvBnRelu(hidden_dim * 2, hidden_dim * 2),
            nn.MaxPool2d(2),
            ConvBnRelu(hidden_dim * 2, hidden_dim * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, image):
        features = self.features(image).squeeze(-1).squeeze(-1)
        return self.classifier(features)
