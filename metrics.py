"""
metrics.py — Steganography Evaluation Metrics
===============================================
Four metrics together tell the full story of the arms race each round:

  SSIM    : Perceptual quality — how invisible is the stego image?  [0→1, higher better]
  PSNR    : Classical distortion measure in dB                      [higher better]
  BER     : Bit Error Rate — can the payload be recovered?          [0→1, lower better]
  DetAcc  : Warden detection accuracy                               [0.5=fooled, 1.0=perfect]

References:
  Wang et al., "Image Quality Assessment: From Error Visibility to Structural
  Similarity," IEEE Transactions on Image Processing, 2004.
"""

import torch
import torch.nn.functional as F
import math


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, data_range: float = 2.0) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.avg_pool2d(img1*img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2*img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12   = F.avg_pool2d(img1*img2, window_size, stride=1, padding=pad) - mu1_mu2

    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean().item()


def psnr(cover: torch.Tensor, stego: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(cover, stego).item()
    if mse < 1e-12:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))


def bit_error_rate(payload: torch.Tensor, decoded_logits: torch.Tensor) -> float:
    decoded_bits = (decoded_logits.detach() > 0).float()
    true_bits    = (payload > 0.5).float()
    return (decoded_bits != true_bits).float().mean().item()


def detection_accuracy(warden_logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds  = (warden_logits.detach().squeeze() > 0).float()
    labels = labels.float().squeeze()
    return (preds == labels).float().mean().item()
