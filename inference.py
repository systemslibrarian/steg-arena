"""
inference.py — CLI: Embed, Detect, and Extract with Trained Models
==================================================================
After training with arena.py, use the trained weights directly in Python.

Usage:
    python inference.py embed   --image cover.png --message "Hello" --encoder output/encoder_final.pt --out stego.png
    python inference.py detect  --image stego.png --warden output/warden_final.pt
    python inference.py extract --image stego.png --decoder output/decoder_final.pt
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
from pathlib import Path
from models import Encoder, Decoder, Warden

IMAGE_SIZE  = 64   # must match training
PAYLOAD_LEN = 32   # must match training


def load_image(path):
    img = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return t(img).unsqueeze(0)


def save_image(tensor, path):
    tensor = (tensor.squeeze(0).clamp(-1, 1) + 1) / 2
    TF.to_pil_image(tensor).save(path)
    print(f"[inference] Saved → {path}")


def msg_to_bits(message, length):
    bits = []
    for byte in message.encode('utf-8'):
        for i in range(7, -1, -1):
            bits.append(float((byte >> i) & 1))
    bits = (bits + [0.0] * length)[:length]
    return torch.tensor(bits).unsqueeze(0)


def bits_to_msg(bits):
    bits = bits.squeeze().tolist()
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = sum(int(bits[i+j] > 0.5) << (7-j) for j in range(8))
        if byte == 0: break
        chars.append(chr(byte))
    return ''.join(chars)


def cmd_embed(args):
    enc = Encoder(payload_len=args.payload_len)
    enc.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    enc.eval()
    cover   = load_image(args.image)
    payload = msg_to_bits(args.message, args.payload_len)
    with torch.no_grad():
        stego = enc(cover, payload)
    save_image(stego, args.out)
    mse = ((cover - stego)**2).mean().item()
    print(f"[inference] Message: '{args.message}'  MSE distortion: {mse:.6f}")


def cmd_detect(args):
    war = Warden()
    war.load_state_dict(torch.load(args.warden, map_location='cpu'))
    war.eval()
    image = load_image(args.image)
    with torch.no_grad():
        prob = torch.sigmoid(war(image)).item()
    verdict = "STEGO DETECTED" if prob > 0.5 else "CLEAN — no stego found"
    print(f"[inference] P(stego) = {prob:.4f}  →  {verdict}")


def cmd_extract(args):
    dec = Decoder(payload_len=args.payload_len)
    dec.load_state_dict(torch.load(args.decoder, map_location='cpu'))
    dec.eval()
    image = load_image(args.image)
    with torch.no_grad():
        bits = torch.sigmoid(dec(image))
    print(f"[inference] Decoded: '{bits_to_msg(bits)}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('embed')
    p.add_argument('--image', required=True); p.add_argument('--message', required=True)
    p.add_argument('--encoder', required=True); p.add_argument('--out', default='stego.png')
    p.add_argument('--payload_len', type=int, default=PAYLOAD_LEN)

    p = sub.add_parser('detect')
    p.add_argument('--image', required=True); p.add_argument('--warden', required=True)

    p = sub.add_parser('extract')
    p.add_argument('--image', required=True); p.add_argument('--decoder', required=True)
    p.add_argument('--payload_len', type=int, default=PAYLOAD_LEN)

    args = parser.parse_args()
    {'embed': cmd_embed, 'detect': cmd_detect, 'extract': cmd_extract}[args.command](args)
