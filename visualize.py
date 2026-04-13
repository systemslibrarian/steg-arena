"""
visualize.py — Arms Race Visualization
========================================
Plots the three core tensions each round:
  1. Warden detection accuracy  (can it catch the encoder?)
  2. Encoder SSIM               (is the stego image invisible?)
  3. Encoder BER                (is the payload recoverable?)

Standalone usage:
    python visualize.py --history ./output/arms_race_history.json
"""

import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {'warden': '#e63946', 'ssim': '#457b9d', 'ber': '#2a9d8f', 'grid': '#e0e0e0'}


def plot_arms_race(history: list, save_path=None, show: bool = False):
    rounds = [r['round']        for r in history]
    w_acc  = [r['warden_acc']   for r in history]
    e_ssim = [r['encoder_ssim'] for r in history]
    e_ber  = [r['encoder_ber']  for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Adversarial Steganography — Arms Race Progress',
                 fontsize=14, fontweight='bold', y=1.02)

    # Panel 1: Warden accuracy
    ax = axes[0]
    ax.plot(rounds, w_acc, color=COLORS['warden'], linewidth=2.5, marker='o', markersize=5)
    ax.axhline(0.50, color='gray',           linestyle='--', linewidth=1.2, label='Random (0.50)')
    ax.axhline(0.70, color=COLORS['warden'], linestyle=':',  linewidth=1.2, alpha=0.6, label='Danger (0.70)')
    ax.axhspan(0.0, 0.60, alpha=0.06, color='green')
    ax.set_xlabel('Round'); ax.set_ylabel('Detection Accuracy')
    ax.set_title('Warden: Can It Catch the Stegger?', fontsize=12)
    ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(True, color=COLORS['grid'])

    # Panel 2: SSIM
    ax = axes[1]
    ax.plot(rounds, e_ssim, color=COLORS['ssim'], linewidth=2.5, marker='s', markersize=5)
    ax.axhline(0.95, color=COLORS['ssim'], linestyle='--', linewidth=1.2, alpha=0.7, label='Good (0.95)')
    ax.axhspan(0.95, 1.05, alpha=0.06, color='blue')
    ax.set_xlabel('Round'); ax.set_ylabel('SSIM Score')
    ax.set_title('Encoder: How Invisible Is the Payload?', fontsize=12)
    ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(True, color=COLORS['grid'])

    # Panel 3: BER
    ax = axes[2]
    ax.plot(rounds, e_ber, color=COLORS['ber'], linewidth=2.5, marker='^', markersize=5)
    ax.axhline(0.50, color='gray',         linestyle='--', linewidth=1.2, label='Random (0.50)')
    ax.axhline(0.10, color=COLORS['ber'],  linestyle=':',  linewidth=1.2, alpha=0.7, label='Good (< 0.10)')
    ax.axhspan(0.0, 0.10, alpha=0.06, color='teal')
    ax.set_xlabel('Round'); ax.set_ylabel('Bit Error Rate')
    ax.set_title('Encoder: Is the Payload Recoverable?', fontsize=12)
    ax.set_ylim(-0.02, 0.55); ax.legend(fontsize=8); ax.grid(True, color=COLORS['grid'])

    fig.text(0.5, -0.04,
             '"Whatever you do, do it all for the glory of God." \u2014 1 Corinthians 10:31',
             ha='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize] Plot saved \u2192 {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_from_file(json_path: str, show: bool = False):
    p = Path(json_path)
    with open(p) as f:
        data = json.load(f)
    history = data['history'] if 'history' in data else data
    out_path = p.parent / 'arms_race_plot.png'
    plot_arms_race(history, save_path=out_path, show=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    plot_from_file(args.history, args.show)
