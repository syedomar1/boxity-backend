"""
train_siamese_with_metrics.py
Trains a Siamese CNN model using image pairs and extra visual metrics.

Usage:
py -3 train_siamese_with_metrics.py --pairs_csv .\data\pairs.csv --metrics_csv .\data\extra_metrics.csv --epochs 1 --batch-size 16 --img-size 256 --out-dir .\ckpts
"""

import argparse
import random
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models

# ---------------- Dataset ----------------
class PairDatasetWithMetrics(Dataset):
    def __init__(self, pairs_csv, metrics_csv=None, img_size=256, training=True):
        df = pd.read_csv(pairs_csv)
        self.rows = df.to_dict('records')
        self.img_size = img_size
        self.training = training

        self.metrics_lookup = {}
        if metrics_csv and Path(metrics_csv).exists():
            md = pd.read_csv(metrics_csv)
            for _, r in md.iterrows():
                key = (str(r['ref_path']), str(r['test_path']))
                feats = [
                    float(r.get('ssim', 0) or 0),
                    float(r.get('pixel_diff_pct', 0) or 0),
                    float(r.get('damage_area_pct', 0) or 0),
                    float(r.get('mean_delta_e', 0) or 0)
                ]
                self.metrics_lookup[key] = feats

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ref, test = r['ref_path'], r['test_path']
        ref_img = self.tf(Image.open(ref).convert('RGB'))
        test_img = self.tf(Image.open(test).convert('RGB'))
        label = torch.tensor(float(r['label']), dtype=torch.float32)
        score = torch.tensor(float(r['score']), dtype=torch.float32)
        metrics = torch.tensor(self.metrics_lookup.get((ref, test), [0,0,0,0]), dtype=torch.float32)
        return ref_img, test_img, label, score, metrics

# ---------------- Model ----------------
class SiameseWithMetrics(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        modules = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(512, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.regressor = nn.Sequential(nn.Linear(emb_dim+4, 128), nn.ReLU(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(emb_dim+4, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ref, test, metrics):
        ref_f = self.projector(self.encoder(ref))
        test_f = self.projector(self.encoder(test))
        diff = torch.abs(ref_f - test_f)
        x = torch.cat([diff, metrics], dim=1)
        sev = self.regressor(x)
        cls = self.classifier(x)
        return sev, cls

# ---------------- Train Loop ----------------
def train_one_epoch(model, loader, opt, device):
    model.train(); total=0
    for ref, test, label, score, metrics in tqdm(loader, desc="Training"):
        ref, test, label, score, metrics = ref.to(device), test.to(device), label.to(device), score.to(device), metrics.to(device)
        opt.zero_grad()
        sev, cls = model(ref, test, metrics)
        loss_cls = F.binary_cross_entropy_with_logits(cls.squeeze(1), label)
        loss_reg = F.l1_loss(torch.sigmoid(sev.squeeze(1)), score/100)
        loss = loss_cls + loss_reg
        loss.backward(); opt.step()
        total += loss.item()
    return total/len(loader)

# ---------------- Validation ----------------
def validate(model, loader, device):
    model.eval(); total=0
    with torch.no_grad():
        for ref, test, label, score, metrics in tqdm(loader, desc="Validation"):
            ref, test, label, score, metrics = ref.to(device), test.to(device), label.to(device), score.to(device), metrics.to(device)
            sev, cls = model(ref, test, metrics)
            loss_cls = F.binary_cross_entropy_with_logits(cls.squeeze(1), label)
            loss_reg = F.l1_loss(torch.sigmoid(sev.squeeze(1)), score/100)
            total += (loss_cls+loss_reg).item()
    return total/len(loader)

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs_csv', required=True)
    p.add_argument('--metrics_csv', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--out-dir', default='./ckpts')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    ds = PairDatasetWithMetrics(args.pairs_csv, args.metrics_csv, args.img_size, training=True)
    n = len(ds); val_n = int(0.1*n)
    idx = list(range(n)); random.shuffle(idx)
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SiameseWithMetrics().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best = 9999
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = validate(model, val_loader, device)
        print(f"Train {train_loss:.4f} | Val {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print("✅ Saved best model")

if __name__ == "__main__":
    main()
