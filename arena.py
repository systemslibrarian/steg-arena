"""
arena.py — Adversarial Steganography Training Loop
====================================================
Orchestrates the arms race between:

  Stegger (Encoder + Decoder) : Embeds secrets, trains to evade the Warden
  Warden                      : Detects stego images, trains to catch the Stegger

Training alternates every round:
  1. Warden trains on Encoder's current stego output
  2. Encoder trains to fool the updated Warden while keeping SSIM high and BER low

After training, run export_onnx.py to convert weights to browser-ready .onnx files.

Usage:
    python arena.py                          # synthetic data, quick test
    python arena.py --image_dir ./images     # real images (BOSS Base recommended)
    python arena.py --rounds 30 --epochs_per_round 5 --image_dir ./images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
from pathlib import Path

from models import Encoder, Decoder, Warden
from dataset import get_dataloader
from metrics import ssim, psnr, bit_error_rate, detection_accuracy
from visualize import plot_arms_race

# ── Loss weights ──────────────────────────────────────────────────────────────
# These are the core research hyperparameters.
# Raise LAMBDA_ADVERSARIAL → encoder more aggressive at evasion (hurts BER/SSIM)
# Raise LAMBDA_IMAGE       → encoder stays more invisible (may reduce evasion)
# Raise LAMBDA_DECODE      → encoder prioritises payload recovery
LAMBDA_IMAGE       = 1.0
LAMBDA_DECODE      = 1.0
LAMBDA_ADVERSARIAL = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def train_warden_epoch(warden, encoder, loader, optimizer, device, payload_len):
    """Warden trains for one epoch: learn to classify clean vs stego."""
    warden.train(); encoder.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for cover in loader:
        cover = cover.to(device)
        B = cover.size(0)

        with torch.no_grad():
            payload = torch.randint(0, 2, (B, payload_len), dtype=torch.float32, device=device)
            stego = encoder(cover, payload)

        clean_logits = warden(cover)
        stego_logits = warden(stego)
        loss = criterion(clean_logits, torch.zeros(B, 1, device=device)) + \
               criterion(stego_logits, torch.ones(B, 1, device=device))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(warden.parameters(), 1.0)
        optimizer.step()

        all_logits = torch.cat([clean_logits, stego_logits])
        all_labels = torch.cat([torch.zeros(B, device=device), torch.ones(B, device=device)])
        total_loss += loss.item(); total_acc += detection_accuracy(all_logits, all_labels); n += 1

    return total_loss / n, total_acc / n


def train_encoder_epoch(encoder, decoder, warden, loader, enc_opt, dec_opt, device, payload_len):
    """Encoder+Decoder train for one epoch: hide well, recover accurately, fool warden."""
    encoder.train(); decoder.train(); warden.eval()
    img_crit = nn.MSELoss(); dec_crit = nn.BCEWithLogitsLoss()
    total_loss, total_ber, total_ssim, total_psnr, n = 0.0, 0.0, 0.0, 0.0, 0

    for cover in loader:
        cover = cover.to(device)
        B = cover.size(0)
        payload = torch.randint(0, 2, (B, payload_len), dtype=torch.float32, device=device)

        stego = encoder(cover, payload)
        decoded_logits = decoder(stego)
        warden_logits  = warden(stego)

        img_loss    = img_crit(stego, cover)
        decode_loss = dec_crit(decoded_logits, payload)
        adv_loss    = nn.BCEWithLogitsLoss()(warden_logits, torch.zeros(B, 1, device=device))
        loss = LAMBDA_IMAGE * img_loss + LAMBDA_DECODE * decode_loss + LAMBDA_ADVERSARIAL * adv_loss

        enc_opt.zero_grad(); dec_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        enc_opt.step(); dec_opt.step()

        total_loss += loss.item()
        total_ber  += bit_error_rate(payload, decoded_logits)
        total_ssim += ssim(cover.detach(), stego.detach())
        total_psnr += psnr(cover.detach(), stego.detach())
        n += 1

    return total_loss / n, total_ber / n, total_ssim / n, total_psnr / n


def status(w_acc, ber, sim):
    if w_acc > 0.80:   return "🔴 Warden winning   — stego is detectable"
    if w_acc < 0.60 and ber < 0.15 and sim > 0.90:
                       return "🟢 Encoder winning  — hidden and recoverable"
    if w_acc < 0.60:   return "🟡 Encoder evading  — but payload recovery weak"
    return                    "🟡 Arms race active — neither side dominant"


def run_arena(args):
    device    = torch.device('cpu')
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  ADVERSARIAL STEGANOGRAPHY ARENA")
    print(f"  {args.image_size}x{args.image_size} images | {args.payload_len}-bit payload")
    print(f"  {args.rounds} rounds x {args.epochs_per_round} epochs/side | CPU")
    print(f"{'='*62}\n")

    encoder = Encoder(payload_len=args.payload_len).to(device)
    decoder = Decoder(payload_len=args.payload_len).to(device)
    warden  = Warden().to(device)

    print(f"  Encoder params : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Warden  params : {sum(p.numel() for p in warden.parameters()):,}\n")

    enc_opt = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=args.lr)
    war_opt = optim.Adam(warden.parameters(),  lr=args.lr)

    loader  = get_dataloader(args.image_dir, args.batch_size, args.image_size)
    history = []

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        print(f"-- Round {rnd:>3}/{args.rounds} " + "-"*40)

        print(f"   [Warden ] training...", end='', flush=True)
        for _ in range(args.epochs_per_round):
            w_loss, w_acc = train_warden_epoch(warden, encoder, loader, war_opt, device, args.payload_len)
        print(f" det_acc={w_acc:.3f}  loss={w_loss:.4f}")

        print(f"   [Encoder] training...", end='', flush=True)
        for _ in range(args.epochs_per_round):
            e_loss, e_ber, e_ssim, e_psnr = train_encoder_epoch(
                encoder, decoder, warden, loader, enc_opt, dec_opt, device, args.payload_len)
        print(f" BER={e_ber:.3f}  SSIM={e_ssim:.4f}  PSNR={e_psnr:.1f}dB  loss={e_loss:.4f}")
        print(f"   {status(w_acc, e_ber, e_ssim)}  ({time.time()-t0:.1f}s)")

        if rnd % args.checkpoint_every == 0:
            torch.save({'round': rnd, 'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(), 'warden': warden.state_dict()},
                       out_dir / f'checkpoint_round{rnd:03d}.pt')
            print(f"   [checkpoint saved]")

        history.append({'round': rnd, 'warden_acc': round(w_acc, 4),
                        'warden_loss': round(w_loss, 4), 'encoder_ber': round(e_ber, 4),
                        'encoder_ssim': round(e_ssim, 4), 'encoder_psnr': round(e_psnr, 2),
                        'encoder_loss': round(e_loss, 4)})

    # Save everything
    torch.save(encoder.state_dict(), out_dir / 'encoder_final.pt')
    torch.save(decoder.state_dict(), out_dir / 'decoder_final.pt')
    torch.save(warden.state_dict(),  out_dir / 'warden_final.pt')

    with open(out_dir / 'arms_race_history.json', 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)

    plot_arms_race(history, out_dir / 'arms_race_plot.png')

    final = history[-1]
    print(f"\n{'='*62}")
    print(f"  COMPLETE  |  Warden acc: {final['warden_acc']:.3f}  "
          f"SSIM: {final['encoder_ssim']:.4f}  BER: {final['encoder_ber']:.3f}")
    print(f"  {status(final['warden_acc'], final['encoder_ber'], final['encoder_ssim'])}")
    print(f"\n  Next step: python export_onnx.py --weights_dir {args.output_dir}")
    verse = '"Whatever you do, do it all for the glory of God." \u2014 1 Corinthians 10:31'
    print(f"\n  {verse}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Adversarial Steganography Arena')
    p.add_argument('--image_dir',        default='./images')
    p.add_argument('--output_dir',       default='./output')
    p.add_argument('--image_size',       type=int,   default=64)
    p.add_argument('--payload_len',      type=int,   default=32)
    p.add_argument('--batch_size',       type=int,   default=4)
    p.add_argument('--rounds',           type=int,   default=20)
    p.add_argument('--epochs_per_round', type=int,   default=3)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--checkpoint_every', type=int,   default=5)
    run_arena(p.parse_args())
