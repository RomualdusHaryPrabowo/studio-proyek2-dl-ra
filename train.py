"""Training ConvNeXt-Tiny (pretrained ImageNet) untuk Studio 2 - Klasifikasi Sampah.

Contoh:
    python train.py --data ./train --epochs 20                 # eksperimen, validasi di my_val/
    python train.py --data ./train --epochs 20 --full          # FINAL: latih di semua data train
    python train.py --data ./train --epochs 20 --backbone resnet18   # pembanding baseline

Keluaran: submission/model.pth (disimpan fp16 agar < 100 MB; evaluate.py memuatnya
ke model fp32 tanpa masalah) + log.csv. Checkpoint terbaik dipilih berdasar Macro-F1.
"""
import argparse, csv, math, random, shutil, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).parent / "submission"))
from submission import model as sub


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_local_split(src, dst, frac, seed):
    """Salin frac gambar per kelas ke my_val/, sisanya ke my_train/ (sekali saja)."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS)
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def macro_f1(y_true, y_pred, k):
    f1s = []
    for c in range(k):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / k


def build(backbone, pretrained):
    """Kembalikan (model, head_param_prefix). ConvNeXt = arsitektur submission."""
    if backbone == "convnext_tiny":
        m = sub.get_model()
        if pretrained:
            pre = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).state_dict()
            pre = {k: v for k, v in pre.items() if not k.startswith("classifier.2.")}
            print("load pretrained:", m.load_state_dict(pre, strict=False))
        return m, "classifier.2."
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, sub.NUM_CLASSES)
        return m, "fc."
    raise ValueError(backbone)


def cutmix_mixup(x, y, k, alpha_mix=0.2, alpha_cut=1.0, p=0.5):
    """Dengan probabilitas p terapkan MixUp atau CutMix (50:50). Return x, y_soft."""
    y1 = F.one_hot(y, k).float()
    if random.random() > p:
        return x, y1
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:  # mixup
        lam = np.random.beta(alpha_mix, alpha_mix)
        x = lam * x + (1 - lam) * x[idx]
    else:  # cutmix
        lam = np.random.beta(alpha_cut, alpha_cut)
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H), random.randint(0, W)
        y0, y1_, x0, x1_ = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y0:y1_, x0:x1_] = x[idx][:, :, y0:y1_, x0:x1_]
        lam = 1 - (y1_ - y0) * (x1_ - x0) / (H * W)
    return x, lam * y1 + (1 - lam) * y1[idx]


def soft_ce(logits, target, smoothing, weight=None):
    k = logits.size(1)
    target = target * (1 - smoothing) + smoothing / k
    logp = F.log_softmax(logits, 1)
    if weight is not None:
        logp = logp * weight
    return -(target * logp).sum(1).mean()


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    yt, yp = [], []
    for x, y in dl:
        yp += model(x.to(device)).argmax(1).cpu().tolist()
        yt += y.tolist()
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder train/ dari dataset")
    ap.add_argument("--backbone", default="convnext_tiny", choices=["convnext_tiny", "resnet18"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="LR backbone (head = 5x)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1, help="epoch warmup")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--mix-p", type=float, default=0.5, help="peluang MixUp/CutMix per batch (0 = mati)")
    ap.add_argument("--balanced", action="store_true", help="WeightedRandomSampler untuk kelas minoritas")
    ap.add_argument("--scratch", action="store_true", help="tanpa bobot pretrained")
    ap.add_argument("--full", action="store_true", help="latih di SEMUA data train (final); simpan epoch terakhir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission/model.pth")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda"
    S = sub.IMG_SIZE

    tr_dir, va_dir = make_local_split(a.data, ".", 0.15, a.seed)
    if a.full:
        tr_dir = Path(a.data)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(sub.MEAN, sub.STD),
        transforms.RandomErasing(p=0.25),
    ])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf, is_valid_file=lambda p: Path(p).suffix.lower() in EXTS)
    va_ds = datasets.ImageFolder(va_dir, sub.get_transform())
    assert tr_ds.classes == va_ds.classes and len(tr_ds.classes) == sub.NUM_CLASSES, tr_ds.classes
    print("kelas:", tr_ds.classes, "| train:", len(tr_ds), "| val:", len(va_ds), "| device:", device)

    counts = np.bincount(tr_ds.targets, minlength=sub.NUM_CLASSES)
    sampler = None
    if a.balanced:
        w = (1.0 / counts)[tr_ds.targets]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=sampler is None, sampler=sampler,
                       num_workers=a.workers, drop_last=True, pin_memory=amp)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=a.workers)

    model, head = build(a.backbone, not a.scratch)
    model.to(device)
    heads = [p for n, p in model.named_parameters() if n.startswith(head)]
    body = [p for n, p in model.named_parameters() if not n.startswith(head)]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": heads, "lr": a.lr * 5}], weight_decay=a.wd)
    base = [g["lr"] for g in opt.param_groups]
    steps_ep = len(tr_dl)
    total, warm = a.epochs * steps_ep, a.warmup * steps_ep
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log = open("log.csv", "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_macro_f1", "val_acc"])
    best, step = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0
        for x, y in tr_dl:
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x, y = x.to(device), y.to(device)
            x, ys = cutmix_mixup(x, y, sub.NUM_CLASSES, p=a.mix_p)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = soft_ce(model(x).float(), ys, a.smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); step += 1

        yt, yp = predict(model, va_dl, device)
        f1 = macro_f1(yt, yp, sub.NUM_CLASSES)
        acc = sum(t == p for t, p in zip(yt, yp)) / len(yt)
        print(f"epoch {ep:2d}  loss={run/steps_ep:.4f}  val_macroF1={f1:.4f}  val_acc={acc:.4f}", flush=True)
        lw.writerow([ep, f"{run/steps_ep:.4f}", f"{f1:.4f}", f"{acc:.4f}"]); log.flush()
        if (a.full and ep == a.epochs) or (not a.full and f1 > best):
            best = f1
            sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, a.out)
    note = "(mode --full: val bocor ke train, angka ini TIDAK valid)" if a.full else ""
    print(f"selesai. best val_macroF1={best:.4f} -> {a.out} {note}")


if __name__ == "__main__":
    main()