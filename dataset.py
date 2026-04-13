"""
dataset.py — Cover Image Dataset Loader
=========================================
Loads cover images for steganography training.

Recommended free datasets:
  BOSS Base 1.01 : http://agents.fel.cvut.cz/boss/
  BOWS2          : http://bows2.ec-lille.fr/
  ALASKA2        : https://www.kaggle.com/c/alaska2-image-steganalysis

Falls back to synthetic noise images if no folder is found, so you can
verify the full pipeline before acquiring a dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image


class CoverImageDataset(Dataset):
    EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')

    def __init__(self, image_dir, image_size=64, split='train', split_ratio=0.8):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        image_dir = Path(image_dir)
        all_images = []
        if image_dir.exists():
            for ext in self.EXTENSIONS:
                all_images.extend(sorted(image_dir.glob(f'**/{ext}')))

        if all_images:
            cut = int(len(all_images) * split_ratio)
            self.images = all_images[:cut] if split == 'train' else all_images[cut:]
            self.synthetic = False
            print(f"[dataset] {split}: {len(self.images)} images from {image_dir}")
        else:
            self.images = []
            self.synthetic = True
            self.n_synthetic = 500 if split == 'train' else 100
            print(f"[dataset] No images found in '{image_dir}' — using {self.n_synthetic} synthetic images for {split}.")
            print(f"[dataset] Download BOSS Base for real steganalysis research.")

    def __len__(self):
        return len(self.images) if not self.synthetic else self.n_synthetic

    def __getitem__(self, idx):
        if self.synthetic:
            return torch.rand(3, self.image_size, self.image_size) * 2 - 1
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            return self.transform(img)
        except Exception:
            return torch.rand(3, self.image_size, self.image_size) * 2 - 1


def get_dataloader(image_dir, batch_size=4, image_size=64, split='train', num_workers=0):
    dataset = CoverImageDataset(image_dir, image_size, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                      num_workers=num_workers, drop_last=True)
