#!/usr/bin/env python3
"""
prepare_local_pairs_with_metrics.py

Usage:
  Ensure your extracted dataset (containing 'intact' and 'damaged' folders) is available at --raw-dir.
  Install dependencies:
    py -3 -m pip install opencv-python scikit-image pillow numpy pandas tqdm

  Run:
    py -3 prepare_local_pairs_with_metrics.py --raw-dir "C:\path\to\damaged-and-intact-packages" --out-dir .\data --num-pairs 2000

Outputs:
  - <out_dir>/pairs.csv         (ref_path,test_path,label,score)
  - <out_dir>/extra_metrics.csv (ref_path,test_path,ssim,pixel_diff_pct,damage_area_pct,mean_delta_e)
"""

import os
import argparse
from pathlib import Path
import random
import csv
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# -------------------------
# Gather images
# -------------------------
def gather_images(raw_dir: Path):
    intact = []
    damaged = []
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")
    for root, dirs, files in os.walk(raw_dir):
        name = Path(root).name.lower()
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fp = str(Path(root) / f)
                # heuristics: folder names containing 'intact' or 'damaged'
                if 'intact' in name:
                    intact.append(fp)
                elif 'damaged' in name or 'damage' in name:
                    damaged.append(fp)
                else:
                    # check parent path parts
                    parts = [pp.lower() for pp in Path(root).parts]
                    if any('intact' in pp for pp in parts):
                        intact.append(fp)
                    elif any('damaged' in pp or 'damage' in pp for pp in parts):
                        damaged.append(fp)
    # final fallback: direct children named intact/damaged
    if not intact:
        cand = raw_dir / 'intact'
        if cand.exists():
            intact = [str(p) for p in cand.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    if not damaged:
        cand = raw_dir / 'damaged'
        if cand.exists():
            damaged = [str(p) for p in cand.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    return intact, damaged

# -------------------------
# Image metric computations
# -------------------------
def load_and_resize(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img  # RGB uint8

def compute_ssim(imgA, imgB):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    # handle constant images (avoid div by zero)
    try:
        if grayA.max() == grayA.min() or grayB.max() == grayB.min():
            return float(0.0)
        score = ssim(grayA, grayB, data_range=grayA.max() - grayA.min())
        return float(score)
    except Exception:
        return float(0.0)

def compute_pixel_diff_pct(imgA, imgB, diff_thresh=25):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(grayA, grayB)
    mask = (diff > diff_thresh).astype(np.uint8)
    pct = 100.0 * (mask.sum() / mask.size)
    return float(pct), mask

def compute_damage_area_pct_from_mask(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    if mask_u8.sum() == 0:
        return 0.0
    num_labels, labels_im = cv2.connectedComponents(mask_u8)
    areas = []
    for lab in range(1, num_labels):
        area = int((labels_im == lab).sum())
        areas.append(area)
    total = sum(areas)
    pct = 100.0 * (total / mask_u8.size)
    return float(pct)

def compute_mean_delta_e(imgA, imgB):
    try:
        labA = cv2.cvtColor(imgA, cv2.COLOR_RGB2LAB).astype(np.float32)
        labB = cv2.cvtColor(imgB, cv2.COLOR_RGB2LAB).astype(np.float32)
        diff = labA - labB
        de = np.linalg.norm(diff, axis=2)
        return float(de.mean())
    except Exception:
        return float(0.0)

# -------------------------
# Pair generation + metrics
# -------------------------
def make_pairs_and_metrics(intact, damaged, num_pairs, ratios, out_csv, metrics_csv, img_size=256, seed=42):
    random.seed(seed)
    rows = []
    metrics_rows = []
    n_ii = int(num_pairs * ratios.get('intact_intact', 0.2))
    n_id = int(num_pairs * ratios.get('intact_damaged', 0.7))
    n_dd = num_pairs - n_ii - n_id

    def samp(lst):
        return random.choice(lst) if lst else None

    print(f"Generating {n_ii} intact-intact, {n_id} intact-damaged, {n_dd} damaged-damaged")

    pair_specs = []
    # create list of specs (type, ref, test, label, score)
    for _ in range(n_ii):
        a = samp(intact); b = samp(intact)
        pair_specs.append(('ii', a, b, 0, 0.0))
    for _ in range(n_id):
        a = samp(intact); b = samp(damaged)
        severity = round(random.uniform(30.0, 95.0), 2)
        pair_specs.append(('id', a, b, 1, severity))
    for _ in range(n_dd):
        a = samp(damaged); b = samp(damaged)
        severity = round(random.uniform(55.0, 100.0), 2)
        pair_specs.append(('dd', a, b, 1, severity))

    random.shuffle(pair_specs)

    for typ, a, b, label, score in tqdm(pair_specs, desc="Pairs"):
        if a is None or b is None:
            continue
        rows.append((a, b, label, score))
        try:
            imgA = load_and_resize(a, img_size)
            imgB = load_and_resize(b, img_size)

            s = compute_ssim(imgA, imgB)
            pd_pct, mask = compute_pixel_diff_pct(imgA, imgB, diff_thresh=25)
            damage_pct = compute_damage_area_pct_from_mask(mask)
            mean_de = compute_mean_delta_e(imgA, imgB)
        except Exception as e:
            # don't crash whole run for one bad image
            print("Metric compute error for pair:", a, b, "->", e)
            s = None; pd_pct = None; damage_pct = None; mean_de = None

        metrics_rows.append({
            'ref_path': a,
            'test_path': b,
            'ssim': s,
            'pixel_diff_pct': pd_pct,
            'damage_area_pct': damage_pct,
            'mean_delta_e': mean_de
        })

    # write pairs.csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ref_path','test_path','label','score'])
        for r in rows:
            w.writerow(r)

    df = pd.DataFrame(metrics_rows)
    df.to_csv(metrics_csv, index=False)
    print("Wrote:", out_csv, "and", metrics_csv)
    return len(rows)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=str, default='./data/raw', help='path to extracted dataset (contains intact/ and damaged/)')
    parser.add_argument('--out-dir', type=str, default='./data', help='output directory for pairs and metrics')
    parser.add_argument('--num-pairs', type=int, default=2000)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ii-ratio', type=float, default=0.2)
    parser.add_argument('--id-ratio', type=float, default=0.7)
    parser.add_argument('--dd-ratio', type=float, default=0.1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / 'pairs.csv'
    metrics_csv = out_dir / 'extra_metrics.csv'

    intact, damaged = gather_images(raw_dir)
    print(f"Found {len(intact)} intact images and {len(damaged)} damaged images in {raw_dir}")
    if not intact or not damaged:
        print("Could not find 'intact' or 'damaged' images automatically. Inspect the raw-dir path.")
        return

    ratios = {'intact_intact': args.ii_ratio, 'intact_damaged': args.id_ratio, 'damaged_damaged': args.dd_ratio}
    total = make_pairs_and_metrics(intact, damaged, args.num_pairs, ratios, pairs_csv, metrics_csv, img_size=args.img_size, seed=args.seed)
    print("Total pairs:", total)
    print("You can now run the siamese training script with:")
    print(f"python train_siamese.py --data_csv {pairs_csv} --epochs 12 --batch-size 16 --img-size {args.img_size} --out-dir ./ckpts")

if __name__ == '__main__':
    main()
