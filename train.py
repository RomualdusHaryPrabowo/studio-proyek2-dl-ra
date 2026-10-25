"""Latih ConvNeXt-Tiny (pretrained ImageNet) -> submission/model.pth. Tanpa combine, satu model.

    uv run train.py --data ./train --epochs 25                       # eksperimen, validasi di my_val/
    uv run train.py --data ./train --exclude ./review                 # buang kandidat label salah (find_noisy.py)
    uv run train.py --data ./train --seed 1 --out weights/s1.pth      # untuk soup.py (beberapa seed)
    uv run train.py --data ./train --init model_terbaik.pth --epochs 8 --lr 5e-5 --warmup 0   # fine-tune resolusi lebih tinggi
    uv run train.py --data ./train --epochs 25 --full                 # FINAL: semua data train
    uv run evaluate.py --submission ./submission --data ./my_val

GPU kecil / OOM:  --batch-size 8 --accum 4 --workers 0   (batch efektif 32)
Arsitektur & resolusi diatur di submission/model.py (ARCH, IMG_SIZE) agar selalu konsisten.
"""
import argparse, copy, csv, math, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms

from submission import model as sub

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Imgs(Dataset):
    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf
        self.targets = [t for _, t in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, t = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), t


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


def build(pretrained):
    m = sub.get_model()
    if pretrained:
        W = {"convnext_tiny": models.ConvNeXt_Tiny_Weights, "convnext_small": models.ConvNeXt_Small_Weights}[sub.ARCH]
        pre = getattr(models, sub.ARCH)(weights=W.IMAGENET1K_V1).state_dict()
        pre = {f"net.{k}": v for k, v in pre.items() if not k.startswith("classifier.2.")}
        print("load pretrained:", m.load_state_dict(pre, strict=False))
    return m


class EMA:
    """Rata-rata bergerak bobot: menstabilkan skor antar epoch."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def cutmix_mixup(x, y, k, alpha_mix=0.2, alpha_cut=1.0, p=0.5):
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


def soft_ce(logits, target, smoothing):
    k = logits.size(1)
    target = target * (1 - smoothing) + smoothing / k
    return -(target * F.log_softmax(logits, 1)).sum(1).mean()


@torch.no_grad()
def predict(model, dl, device):
    model.eval()  # mode eval = flip-TTA aktif (sama seperti evaluate.py)
    yt, yp = [], []
    for x, y in dl:
        yp += model(x.to(device)).argmax(1).cpu().tolist()
        yt += y.tolist()
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder train/ dari dataset")
    ap.add_argument("--exclude", default=None, help="folder hasil find_noisy.py (review/): gambar di dalamnya TIDAK dipakai training")
    ap.add_argument("--init", default=None, help="mulai dari model.pth ini (fine-tune lanjutan, mis. resolusi lebih tinggi)")
    ap.add_argument("--lld", type=float, default=1.0, help="layer-wise LR decay (mis. 0.85; 1.0 = mati)")
    ap.add_argument("--init-seed", type=int, default=0, help="seed inisialisasi head (samakan antar-run agar bisa di-soup)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="akumulasi gradien (batch efektif = batch-size x accum)")
    ap.add_argument("--lr", type=float, default=2e-4, help="LR backbone (head = 5x)")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1, help="epoch warmup")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--mix-p", type=float, default=0.5, help="peluang MixUp/CutMix per batch (0 = mati)")
    ap.add_argument("--ema", type=float, default=0.995, help="decay EMA (0 = mati)")
    ap.add_argument("--balanced", action="store_true", help="WeightedRandomSampler")
    ap.add_argument("--scratch", action="store_true", help="tanpa bobot pretrained")
    ap.add_argument("--full", action="store_true", help="latih di SEMUA data train (final); simpan epoch terakhir")
    ap.add_argument("--save", choices=["last", "best"], default="last", help="checkpoint yang disimpan (non-full)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission/model.pth")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda"
    S = sub.IMG_SIZE

    tr_dir, va_dir = make_local_split(a.data, ".", 0.15, 42)  # split tetap (seed 42) agar hasil antar-eksperimen sebanding
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
    base_ds = datasets.ImageFolder(tr_dir, is_valid_file=lambda p: Path(p).suffix.lower() in EXTS)
    va_ds = datasets.ImageFolder(va_dir, sub.get_transform())
    assert base_ds.classes == va_ds.classes == sub.CLASSES, base_ds.classes
    samples = list(base_ds.samples)
    n_excl = 0
    if a.exclude:  # nama file review: <label>__<pred>__<conf>__<nama asli>
        bad = set()
        for p in Path(a.exclude).rglob("*"):
            if p.suffix.lower() in EXTS and p.name.count("__") >= 3:
                label, _, _, name = p.name.split("__", 3)
                bad.add(f"{label}/{name}")
        keep = [(p, t) for p, t in samples if f"{Path(p).parent.name}/{Path(p).name}" not in bad]
        n_excl = len(samples) - len(keep)
        samples = keep
    tr_ds = Imgs(samples, train_tf)
    counts = np.bincount(tr_ds.targets, minlength=sub.NUM_CLASSES)
    print(f"arch={sub.ARCH} size={S} | train: {len(tr_ds)} (dibuang: {n_excl}) per kelas: {counts.tolist()} "
          f"| val: {len(va_ds)} | device: {device}")

    sampler = WeightedRandomSampler((1.0 / counts)[tr_ds.targets], len(tr_ds), replacement=True) if a.balanced else None
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=sampler is None, sampler=sampler,
                       num_workers=a.workers, drop_last=True, pin_memory=amp)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=a.workers)

    torch.manual_seed(a.init_seed)  # head identik antar-run
    model = build(not a.scratch and not a.init).to(device)
    if a.init:
        print("init dari", a.init, model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True), strict=True))
    torch.manual_seed(a.seed)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    def layer_id(n):  # net.features.{0..7} -> 1..8 ; classifier -> 9
        return int(n.split(".")[2]) + 1 if n.startswith("net.features.") else 9
    groups = {}
    for n, p in model.named_parameters():
        if n.startswith(sub.HEAD_PREFIX):
            g = groups.setdefault("head", {"params": [], "lr": a.lr * 5})
        else:
            g = groups.setdefault(layer_id(n), {"params": [], "lr": a.lr * a.lld ** (9 - layer_id(n))})
        g["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), weight_decay=a.wd)
    base_lr = [g["lr"] for g in opt.param_groups]
    steps_ep = max(1, len(tr_dl) // a.accum)
    total, warm = a.epochs * steps_ep, a.warmup * steps_ep
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log = open("log.csv", "w", newline=""); lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_macro_f1", "val_acc"])
    best, step = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); run, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tr_dl):
            x, y = x.to(device), y.to(device)
            x, ys = cutmix_mixup(x, y, sub.NUM_CLASSES, p=a.mix_p)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = soft_ce(model(x).float(), ys, a.smoothing)
            scaler.scale(loss / a.accum).backward()
            run += loss.item(); nb += 1
            if (i + 1) % a.accum == 0:
                f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g, b in zip(opt.param_groups, base_lr):
                    g["lr"] = b * f
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
                step += 1

        eval_model = ema.module if ema else model
        yt, yp = predict(eval_model, va_dl, device)
        f1 = macro_f1(yt, yp, sub.NUM_CLASSES)
        acc = sum(t == p for t, p in zip(yt, yp)) / len(yt)
        print(f"epoch {ep:2d}  loss={run/nb:.4f}  val_macroF1={f1:.4f}  val_acc={acc:.4f}", flush=True)
        lw.writerow([ep, f"{run/nb:.4f}", f"{f1:.4f}", f"{acc:.4f}"]); log.flush()
        save_now = ep == a.epochs if (a.full or a.save == "last") else f1 > best
        best = max(best, f1)
        if save_now:
            # fp16 hanya untuk bobot besar (2D/4D); bias/norm/layer_scale tetap fp32
            sd = {k: v.half() if v.is_floating_point() and v.ndim in (2, 4) else v
                  for k, v in eval_model.state_dict().items()}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, a.out)
    mb = Path(a.out).stat().st_size / 1e6
    note = "(mode --full: val bocor ke train, angka TIDAK valid)" if a.full else ""
    print(f"selesai. val_macroF1 terakhir={f1:.4f} terbaik={best:.4f} -> {a.out} ({mb:.1f} MB"
          f"{'' if mb < 100 else ', >100 MB: butuh Git LFS/Release!'}) {note}")


if __name__ == "__main__":
    main()