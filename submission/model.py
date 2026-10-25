"""Submission Studio 2: ConvNeXt-Tiny (torchvision) + flip-TTA, satu model, satu file bobot.

Kontrak:
  - get_model() hanya membangun arsitektur (weights=None, tanpa unduhan, tanpa memuat model.pth).
  - get_transform() = transform validasi yang SAMA dengan saat training.
  - Output: logits [batch, 5], urutan kelas alfabetis (sama dengan ImageFolder):
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
NUM_CLASSES = 5
ARCH = "convnext_tiny"   # alternatif: "convnext_small" (fp16 ±99 MB, mepet batas 100 MB, 2x lebih lambat)
IMG_SIZE = 288           # jika OOM saat training: pakai --accum, atau turunkan ke 256 (latih ulang)
RESIZE = 320
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TTA = True               # flip-TTA saat evaluasi: ~2x lebih lambat. Set False jika evaluasi CPU > 8 menit
EVAL_FULL = False        # True: seluruh gambar di-resize ke IMG_SIZE x IMG_SIZE (tanpa crop tengah)
ZOOM = False             # True: tambah tampilan "zoom" 87,5% bagian tengah (inferensi ~2x lebih lambat)
DROP_PATH = 0.1          # stochastic depth saat training (default torchvision); tidak berpengaruh saat evaluasi
HEAD_PREFIX = "net.classifier.2."


def build_backbone():
    m = getattr(models, ARCH)(weights=None, stochastic_depth_prob=DROP_PATH)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    return m


class Net(nn.Module):
    def __init__(self, tta=TTA, zoom=ZOOM):
        super().__init__()
        self.net = build_backbone()
        self.tta, self.zoom = tta, zoom

    @staticmethod
    def _zoom(x, frac=0.875):
        h, w = x.shape[-2:]
        oh, ow = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
        return F.interpolate(x[..., oh:h - oh, ow:w - ow], size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x):
        if self.training or not (self.tta or self.zoom):   # training: logit biasa
            return self.net(x)
        views = [x] + ([self._zoom(x)] if self.zoom else [])
        if self.tta:
            views += [torch.flip(v, dims=[3]) for v in views]
        p = sum(self.net(v).float().softmax(1) for v in views) / len(views)
        return torch.log(p.clamp_min(1e-8))


def get_model():
    return Net()


def get_transform(full=EVAL_FULL):
    geo = [transforms.Resize((IMG_SIZE, IMG_SIZE))] if full else \
          [transforms.Resize(RESIZE), transforms.CenterCrop(IMG_SIZE)]
    return transforms.Compose(geo + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])