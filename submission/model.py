"""Submission Studio 2: ConvNeXt-Tiny (torchvision) dengan head 5 kelas.

Kontrak:
  - get_model() hanya membangun arsitektur (weights=None, tanpa unduhan, tanpa memuat model.pth).
  - get_transform() = transform validasi yang SAMA dengan saat training.
  - Output: logits [batch, 5], urutan kelas alfabetis (sama dengan ImageFolder):
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5
IMG_SIZE = 256
RESIZE = 292  # IMG_SIZE / 0.875
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_model():
    model = models.convnext_tiny(weights=None)
    in_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_f, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])