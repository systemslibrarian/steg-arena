"""
export_onnx.py — Export Trained Models to Browser-Ready ONNX
==============================================================
Converts the three trained PyTorch .pt weights into .onnx files that
ONNX Runtime Web can run entirely client-side in the browser.

Run this AFTER arena.py has finished training:

    python export_onnx.py
    python export_onnx.py --weights_dir ./output --out_dir ./demo --image_size 64 --payload_len 32

The three output files land in demo/ next to index.html:
    demo/encoder.onnx   (Stegger — embeds the payload)
    demo/decoder.onnx   (Extractor — recovers the payload)
    demo/warden.onnx    (Detector — classifies clean vs stego)

Then push demo/ to GitHub Pages and the web demo loads them automatically.

Notes:
  - image_size and payload_len MUST match what you trained with in arena.py
  - opset 17 is supported by onnxruntime-web >= 1.17
  - Warden outputs a sigmoid probability so JS can threshold at 0.5
  - Decoder outputs sigmoid probabilities so JS bits = prob > 0.5
"""

import torch
import argparse
from pathlib import Path
from models import Encoder, Decoder, Warden


# ── ONNX-friendly wrappers ────────────────────────────────────────────────────
# These expose clean named inputs/outputs and apply sigmoid where needed
# so the JavaScript side gets probabilities, not raw logits.

class EncoderExport(torch.nn.Module):
    """
    Inputs  : cover   float32 (1, 3, H, W)  normalised [-1, 1]
              payload float32 (1, L)         binary bits 0.0 / 1.0
    Output  : stego   float32 (1, 3, H, W)  normalised [-1, 1]
    """
    def __init__(self, encoder): super().__init__(); self.e = encoder
    def forward(self, cover, payload): return self.e(cover, payload)


class DecoderExport(torch.nn.Module):
    """
    Input   : stego   float32 (1, 3, H, W)
    Output  : bits    float32 (1, L)  — sigmoid probs; JS threshold at 0.5
    """
    def __init__(self, decoder): super().__init__(); self.d = decoder
    def forward(self, stego): return torch.sigmoid(self.d(stego))


class WardenExport(torch.nn.Module):
    """
    Input   : image   float32 (1, 3, H, W)
    Output  : prob    float32 (1, 1)  — P(stego); JS threshold at 0.5
    """
    def __init__(self, warden): super().__init__(); self.w = warden
    def forward(self, image): return torch.sigmoid(self.w(image))


# ─────────────────────────────────────────────────────────────────────────────

def export_model(model, dummy, input_names, output_names, path):
    torch.onnx.export(
        model, dummy, str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    kb = path.stat().st_size / 1024
    print(f"  [OK] {path.name:<22}  {kb:>7.1f} KB")
    return kb


def run(args):
    w_dir   = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H = args.image_size
    L = args.payload_len

    print(f"\n{'='*52}")
    print(f"  ONNX EXPORT — steg-arena")
    print(f"  image_size={H}  payload_len={L}")
    print(f"  weights: {w_dir}  →  output: {out_dir}")
    print(f"{'='*52}\n")

    # ── Encoder ──────────────────────────────────────────────────────────────
    enc_pt = w_dir / 'encoder_final.pt'
    if not enc_pt.exists():
        print(f"  [SKIP] encoder_final.pt not found — run arena.py first")
    else:
        enc = Encoder(payload_len=L); enc.load_state_dict(torch.load(enc_pt, map_location='cpu'))
        enc.eval()
        model  = EncoderExport(enc)
        dummy  = (torch.zeros(1, 3, H, H), torch.zeros(1, L))
        export_model(model, dummy, ['cover', 'payload'], ['stego'], out_dir / 'encoder.onnx')

    # ── Decoder ──────────────────────────────────────────────────────────────
    dec_pt = w_dir / 'decoder_final.pt'
    if not dec_pt.exists():
        print(f"  [SKIP] decoder_final.pt not found — run arena.py first")
    else:
        dec = Decoder(payload_len=L); dec.load_state_dict(torch.load(dec_pt, map_location='cpu'))
        dec.eval()
        model  = DecoderExport(dec)
        dummy  = torch.zeros(1, 3, H, H)
        export_model(model, dummy, ['stego'], ['bits'], out_dir / 'decoder.onnx')

    # ── Warden ───────────────────────────────────────────────────────────────
    war_pt = w_dir / 'warden_final.pt'
    if not war_pt.exists():
        print(f"  [SKIP] warden_final.pt not found — run arena.py first")
    else:
        war = Warden(); war.load_state_dict(torch.load(war_pt, map_location='cpu'))
        war.eval()
        model  = WardenExport(war)
        dummy  = torch.zeros(1, 3, H, H)
        export_model(model, dummy, ['image'], ['prob'], out_dir / 'warden.onnx')

    print(f"\n  All .onnx files written to: {out_dir}/")
    print(f"\n  Next steps:")
    print(f"    1. Push {out_dir}/ contents to your crypto-lab GitHub repo")
    print(f"    2. GitHub Pages will serve encoder.onnx, decoder.onnx, warden.onnx")
    print(f"    3. index.html loads them automatically from the same folder")
    verse = '"Whatever you do, do it all for the glory of God." \u2014 1 Corinthians 10:31'
    print(f"\n  {verse}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Export steg-arena models to ONNX for browser use')
    p.add_argument('--weights_dir', default='./output', help='Folder containing *_final.pt files')
    p.add_argument('--out_dir',     default='./demo',   help='Where to write .onnx files')
    p.add_argument('--image_size',  type=int, default=64,  help='Must match arena.py --image_size')
    p.add_argument('--payload_len', type=int, default=32,  help='Must match arena.py --payload_len')
    run(p.parse_args())
