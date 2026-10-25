# Studio 2 — Klasifikasi Sampah · Kelompok 12/Titut

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |        Romualdus Hary Prabowo      |   123140083  | Menyusun strategi pengerjaan proyek, mengembangkan dan melatih model klasifikasi ConvNeXt-Tiny berbasis pretrained ImageNet, membangun pipeline training dan evaluasi, serta melakukan eksperimen. |
| 2  |        Memory Simanjuntak      |123140095     |                  |
| 3  |       Juliani Leony Putri Melati Manalu       |    123140029 | Menyusun dan melatih model klasifikasi ConvNeXt-Tiny (fine-tuning pretrained ImageNet, EMA, flip-TTA), membangun pipeline training dan evaluasi (`train.py`, `submission/model.py`), menjalankan eksperimen (ensemble, data tambahan TrashNet, variasi inferensi, pencarian label salah), dan mencapai Macro-F1 lokal 0,9790.         |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (torchvision) pretrained ImageNet-1K, classifier akhir diganti Linear(768, 5). Semua layer ikut dilatih (fine-tuning penuh). Stochastic depth 0,1. Pada inferensi, model merata-ratakan probabilitas gambar asli dan gambar yang dicerminkan secara horizontal (flip-TTA) di dalam `forward`, dengan output tetap logits [batch, 5].
- **Strategi pelatihan**: AdamW (weight decay 0,05), LR backbone 2e-4 dan head 1e-3, warmup 1 epoch lalu cosine decay, 25 epoch, batch size 8 dengan akumulasi gradien 4 (batch efektif 32), label smoothing 0,1, gradient clipping 5,0, mixed precision. Augmentasi: RandomResizedCrop (skala 0,5–1,0), flip horizontal, TrivialAugmentWide, RandomErasing, serta MixUp/CutMix (p=0,5). Bobot dirata-ratakan dengan EMA (decay 0,995) dan bobot EMA epoch terakhir yang disimpan. Evaluasi lokal pada split validasi 15% (410 gambar, seed 42). Model final dilatih tanpa `--full`, yaitu hanya pada 85% data train, agar skornya bisa divalidasi.
- **Data tambahan**: Tidak ada. (TrashNet kelas cardboard dan paper sempat dicoba dan tidak meningkatkan skor validasi lokal, sehingga tidak dipakai.)
- **Ukuran input & preprocessing**: Validasi: Resize 320 → CenterCrop 288 → Normalisasi ImageNet (mean 0,485/0,456/0,406; std 0,229/0,224/0,225). Training: RandomResizedCrop ke 288.
- **Hasil validasi lokal**: Macro-F1 0,9790, akurasi 0,9805 (8 salah dari 410 gambar).


## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | - | - | - |
| Preview 2 (21 Okt) | 3991db6 | 0,9537  | 0,9514 |
| Final (25 Okt)     | d4943fa | 0,9790 | |

### Eksperimen

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):


1. **ConvNeXt-Tiny awal** (256 px, batch 4, 20 epoch): Macro-F1 lokal terbaik 0,9537
   (epoch terakhir 0,9513), 18–19 salah. Skor validasi naik-turun antar epoch karena batch kecil.
   Skor dari asdos untuk model ini: Macro-F1 0,9514, akurasi 0,9541.
2. **Ensemble ConvNeXt-Tiny + EfficientNetV2-S**: Macro-F1 0,9518, 19 salah, tidak lebih baik
   dari model tunggal dan dua kali lebih lambat, sehingga tidak dipakai.
3. **Data tambahan TrashNet** (kelas cardboard dan paper → kertas_kardus): skor lokal terakhir
   0,9652 vs 0,9703 tanpa data tambahan. Tidak membantu, sehingga tidak dipakai.
4. **Model final: ConvNeXt-Tiny, 288 px, EMA bobot, batch efektif 32, flip-TTA**:
   Macro-F1 **0,9790**, akurasi **0,9805**, 8 salah dari 410. F1 terendah: kertas_kardus 0,9615.


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
