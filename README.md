# Studio 2 — Klasifikasi Sampah · Kelompok <12/Titut>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |        Romualdus Hary Prabowo      |   123140083  |                  |
| 2  |        Memory Simanjuntak      |123140095     |                  |
| 3  |       Juliani Leony Putri Melati Manalu       |    123140029 |                  |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (torchvision) pretrained ImageNet-1K,
  classifier akhir diganti Linear(768, 5). Semua layer ikut dilatih (fine-tuning penuh).
- **Strategi pelatihan**: AdamW (weight decay 0,05), LR backbone 2e-4 dan head 1e-3,
  warmup 1 epoch lalu cosine decay, 20 epoch, batch size 4 (keterbatasan memori GPU),
  label smoothing 0,1, gradient clipping 5,0. Augmentasi: RandomResizedCrop (skala 0,5–1,0),
  flip horizontal, TrivialAugmentWide, RandomErasing, serta MixUp/CutMix (p=0,5).
  Checkpoint dipilih berdasarkan Macro-F1 pada split validasi lokal 15%.
- **Data tambahan**: Tidak ada. <ubah jika ada: sumber dan jumlah gambar per kelas>
- **Ukuran input & preprocessing**: Validasi: Resize 292 → CenterCrop 256 → Normalisasi
  ImageNet (mean 0,485/0,456/0,406; std 0,229/0,224/0,225). Training: RandomResizedCrop ke 256.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) |- | -|- |
| Preview 2 (21 Okt) | | **1.0000 (Epoch 20)** | |
| Final (25 Okt)     | | | |

### Eksperimen (21 Oktober 2026)

Model ConvNeXt-Tiny dilatih selama 20 epoch dengan batch size 4 dan learning rate 2e-4. Pada validasi lokal 15% (sekitar 410 gambar), model memperoleh Macro-F1 terbaik sebesar **0,9537** dan akurasi **0,9561** pada epoch ke-16.

Untuk submission final, model dilatih ulang menggunakan seluruh data `train` dengan mode `--full`. Skor 1,0000 yang muncul saat evaluasi pada `my_val` tidak digunakan sebagai ukuran performa karena data tersebut ikut digunakan dalam pelatihan. Performa model pada data baru akan dinilai melalui validasi tersembunyi.

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
