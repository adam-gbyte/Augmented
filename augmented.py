#!/usr/bin/env python3
"""
augment.py
Simple image augmentation toolkit using Pillow + numpy.

Fitur:
- rotasi (degree)
- flip horizontal / vertical
- zoom in / zoom out (scale)
- translasi (geser)
- perubahan brightness / contrast
- tambah noise gaussian
- pipeline random untuk memperbanyak data
- batch processing dari folder

Usage examples:
- python augment.py --input path/to/image.jpg --outdir augmented/ --n 20
- python augment.py --indir dataset/images --outdir dataset/augmented --n-per-image 5
"""

import os
import sys
import argparse
import random
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# -----------------------
# Basic transforms
# -----------------------

def rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by angle degrees (keeps full image; expands canvas)."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

def flip_horizontal(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)

def flip_vertical(img: Image.Image) -> Image.Image:
    return ImageOps.flip(img)

def zoom(img: Image.Image, scale: float) -> Image.Image:
    """
    Zooms image.
      - scale > 1.0 -> zoom in (crop center then resize to original size)
      - scale < 1.0 -> zoom out (pad around then resize to original size)
    """
    if scale == 1.0:
        return img.copy()

    w, h = img.size
    if scale > 1.0:
        # crop center
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)
    else:
        # scale < 1: place image on larger canvas (zoom out)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new(img.mode, (w, h))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

def translate(img: Image.Image, tx: int, ty: int, fillcolor=(0,0,0)) -> Image.Image:
    """
    Translate image by tx, ty pixels.
    Positive tx -> move right; Positive ty -> move down.
    """
    w, h = img.size
    canvas = Image.new(img.mode, (w, h), fillcolor)
    canvas.paste(img, (tx, ty))
    return canvas

def adjust_brightness_contrast(img: Image.Image, brightness: float=1.0, contrast: float=1.0) -> Image.Image:
    """
    brightness: 1.0 = original, <1 darker, >1 brighter
    contrast: 1.0 = original, <1 less contrast, >1 more contrast
    """
    out = ImageEnhance.Brightness(img).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return out

def add_gaussian_noise(img: Image.Image, mean: float=0.0, std: float=10.0, clip: bool=True) -> Image.Image:
    """
    Add Gaussian noise to the image.
    std is in pixel value units (0-255). For color images applied on each channel.
    """
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape).astype(np.float32)
    noised = arr + noise
    if clip:
        noised = np.clip(noised, 0, 255)
    noised = noised.astype(np.uint8)
    return Image.fromarray(noised)

# -----------------------
# Compose/Random pipeline
# -----------------------

def random_augment(
    img: Image.Image,
    rotate_range: Tuple[float,float]=(-10,10),
    flip_prob: float=0.5,
    zoom_range: Tuple[float,float]=(0.9,1.1),
    translate_frac: Tuple[float,float]=(-0.01,0.01),
    brightness_range: Tuple[float,float]=(0.8,1.2),
    contrast_range: Tuple[float,float]=(0.8,1.2),
    noise_std_range: Tuple[float,float]=(0.0,15.0),
    seed: Optional[int]=None
) -> Image.Image:
    """
    Apply a random combination of augmentations.
    translate_frac is fraction of width/height to move.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out = img

    # random rotation
    angle = random.uniform(rotate_range[0], rotate_range[1])
    out = rotate(out, angle)

    # random flip
    r = random.random()
    if r < flip_prob/2:
        out = flip_horizontal(out)
    elif r < flip_prob:
        out = flip_vertical(out)

    # random zoom
    scale = random.uniform(zoom_range[0], zoom_range[1])
    out = zoom(out, scale)

    # random translate
    w, h = out.size
    tx = int(round(random.uniform(translate_frac[0], translate_frac[1]) * w))
    ty = int(round(random.uniform(translate_frac[0], translate_frac[1]) * h))
    out = translate(out, tx, ty, fillcolor=(0,0,0))

    # brightness/contrast
    b = random.uniform(brightness_range[0], brightness_range[1])
    c = random.uniform(contrast_range[0], contrast_range[1])
    out = adjust_brightness_contrast(out, brightness=b, contrast=c)

    # noise
    noise_std = random.uniform(noise_std_range[0], noise_std_range[1])
    if noise_std > 0:
        out = add_gaussian_noise(out, std=noise_std)

    return out

# -----------------------
# Utilities for batch processing
# -----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def augment_image_file(
    infile: str,
    outdir: str,
    n: int = 10,
    prefix: str = None,
    seed: Optional[int] = None
) -> List[str]:
    """
    Augment a single image file n times and save to outdir.
    Returns list of output paths.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = Image.open(infile).convert("RGB")
    base = os.path.splitext(os.path.basename(infile))[0]
    ensure_dir(outdir)

    out_paths = []
    for i in range(n):
        subseed = None if seed is None else (seed + i + 1)
        aug = random_augment(img, seed=subseed)
        fname = f"{prefix or base}_{i+1}.jpg"
        out_path = os.path.join(outdir, fname)
        aug.save(out_path, quality=95)
        out_paths.append(out_path)
    return out_paths

def augment_folder(
    indir: str,
    outdir: str,
    n_per_image: int = 5,
    recursive: bool = False,
    seed: Optional[int] = None
) -> None:
    """
    Augment all images in indir. Save augmented images to outdir/<original_subfolder>.
    """
    supported = ('.jpg','.jpeg','.png','.bmp','.tiff')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for root, dirs, files in os.walk(indir):
        rel = os.path.relpath(root, indir)
        if rel == ".":
            rel = ""
        target_dir = os.path.join(outdir, rel)
        ensure_dir(target_dir)

        for f in files:
            if f.lower().endswith(supported):
                infile = os.path.join(root, f)
                try:
                    augment_image_file(infile, target_dir, n=n_per_image, prefix=None, seed=None)
                except Exception as e:
                    print(f"Failed augment {infile}: {e}", file=sys.stderr)

        if not recursive:
            break

# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", help="Input image file")
    p.add_argument("--indir", help="Input folder (batch mode)")
    p.add_argument("--outdir", required=True, help="Output folder")
    p.add_argument("--n", type=int, default=10, help="Number of augmented images for single input")
    p.add_argument("--n-per-image", type=int, default=5, help="Number per image when --indir used")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--recursive", action="store_true", help="Process subfolders in --indir")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    if args.input:
        if not os.path.isfile(args.input):
            print("Input file not found.", file=sys.stderr)
            sys.exit(1)
        out_paths = augment_image_file(args.input, args.outdir, n=args.n, seed=args.seed)
        print(f"Saved {len(out_paths)} augmented images to {args.outdir}")
    elif args.indir:
        if not os.path.isdir(args.indir):
            print("Input directory not found.", file=sys.stderr)
            sys.exit(1)
        augment_folder(args.indir, args.outdir, n_per_image=args.n_per_image, recursive=args.recursive, seed=args.seed)
        print(f"Augmented images saved to {args.outdir}")
    else:
        print("Provide --input <file> or --indir <folder>.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
