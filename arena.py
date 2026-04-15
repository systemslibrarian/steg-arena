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

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
from pathlib import Path

# ── Use all available CPU cores for PyTorch operations ────────────────────────
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(os.cpu_count() or 4)

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
LAMBDA_ADVERSARIAL = 1.5

# Label smoothing: prevents the warden from becoming so confident that
# its gradients vanish, which would starve the encoder of evasion signal.
# The warden still learns to detect — it just can't reach 100% certainty.
SMOOTH_REAL = 0.9   # stego target: 0.9 instead of 1.0
SMOOTH_FAKE = 0.1   # clean target: 0.1 instead of 0.0
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
        loss = criterion(clean_logits, torch.full((B, 1), SMOOTH_FAKE, device=device)) + \
               criterion(stego_logits, torch.full((B, 1), SMOOTH_REAL, device=device))

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
        # Least-squares adversarial loss: MSE toward 0 ("clean").
        # Unlike BCE, MSE never saturates — encoder always gets useful
        # gradient signal even when the warden is highly confident.
        adv_loss    = nn.MSELoss()(torch.sigmoid(warden_logits), torch.zeros(B, 1, device=device))
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
    print(f"  Warden  params : {sum(p.numel() for p in warden.parameters()):,}")
    print(f"  CPU threads    : {torch.get_num_threads()}")

    history = []
    start_round = 1

    # ── Resume from checkpoint (load BEFORE torch.compile) ───────────────────
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder.load_state_dict(ckpt['encoder'])
            decoder.load_state_dict(ckpt['decoder'])
            warden.load_state_dict(ckpt['warden'])
            start_round = ckpt['round'] + 1
            # Load history if it exists
            hist_path = out_dir / 'arms_race_history.json'
            if hist_path.exists():
                with open(hist_path) as f:
                    saved = json.load(f)
                    history = saved.get('history', [])
            print(f"  [resumed from round {ckpt['round']}  →  continuing from round {start_round}]")
    # ─────────────────────────────────────────────────────────────────────────

    # ── JIT-compile models for faster CPU execution ───────────────────────────
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)
    warden  = torch.compile(warden)
    print(f"  torch.compile  : enabled\n")

    enc_opt = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=args.lr)
    war_opt = optim.Adam(warden.parameters(),  lr=args.lr)

    loader  = get_dataloader(args.image_dir, args.batch_size, args.image_size,
                             num_workers=args.num_workers)
    # ─────────────────────────────────────────────────────────────────────────

    for rnd in range(start_round, args.rounds + 1):
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
            # Use _orig_mod when compiled to strip the torch.compile wrapper prefix
            enc_sd = getattr(encoder, '_orig_mod', encoder).state_dict()
            dec_sd = getattr(decoder, '_orig_mod', decoder).state_dict()
            war_sd = getattr(warden,  '_orig_mod', warden).state_dict()
            torch.save({'round': rnd, 'encoder': enc_sd,
                        'decoder': dec_sd, 'warden': war_sd},
                       out_dir / f'checkpoint_round{rnd:03d}.pt')
            print(f"   [checkpoint saved]")

        history.append({'round': rnd, 'warden_acc': round(w_acc, 4),
                        'warden_loss': round(w_loss, 4), 'encoder_ber': round(e_ber, 4),
                        'encoder_ssim': round(e_ssim, 4), 'encoder_psnr': round(e_psnr, 2),
                        'encoder_loss': round(e_loss, 4)})

    # Save everything
    torch.save(getattr(encoder, '_orig_mod', encoder).state_dict(), out_dir / 'encoder_final.pt')
    torch.save(getattr(decoder, '_orig_mod', decoder).state_dict(), out_dir / 'decoder_final.pt')
    torch.save(getattr(warden,  '_orig_mod', warden).state_dict(),  out_dir / 'warden_final.pt')

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
    p.add_argument('--batch_size',       type=int,   default=8)
    p.add_argument('--rounds',           type=int,   default=20)
    p.add_argument('--epochs_per_round', type=int,   default=3)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--checkpoint_every', type=int,   default=5)
    p.add_argument('--num_workers',      type=int,   default=2)
    p.add_argument('--resume',           default=None,
                   help='Path to a checkpoint .pt file to resume from')
    run_arena(p.parse_args())
