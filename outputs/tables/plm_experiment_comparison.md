# Bảng Tổng hợp Kết quả Thực nghiệm Pretrained PLM (PhoBERT & XLM-R)

| Exp ID | Nhóm Thử nghiệm | Mô hình & Cấu hình | Trainable Params | VRAM Đỉnh | Thời gian | Dev F1 (%) | Test P (%) | Test R (%) | Test F1 (%) | Test Macro-F1 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exp01_phobert_baseline` | Baseline | **PhoBERT-base-v2** (Full fine-tuning (0 freeze), pooling: `last`) | 135,005,193 (100.00%) | 4482.4 MB | 413.8s | 94.56% | 93.74% | 94.68% | **94.21%** | 89.76% |
| `exp02_xlmr_baseline` | Baseline | **XLM-RoBERTa-base** (Full fine-tuning (0 freeze), pooling: `last`) | 278,050,569 (100.00%) | 7249.5 MB | 1479.8s | 93.57% | 92.98% | 93.96% | **93.47%** | 89.25% |
| `exp03_phobert_freeze_backbone` | Nhóm A - Freezing | **PhoBERT-base-v2** (Frozen backbone (Head-only), pooling: `last`) | 6,921 (0.01%) | 1349.2 MB | 172.0s | 20.40% | 55.38% | 11.97% | **19.69%** | 8.78% |
| `exp05_phobert_freeze_embeddings` | Nhóm A - Freezing | **PhoBERT-base-v2** (Freeze Embeddings only, pooling: `last`) | 85,651,977 (63.44%) | 4089.8 MB | 222.3s | 94.89% | 93.47% | 94.46% | **93.96%** | 88.26% |
| `exp08_phobert_sched_cosine` | Nhóm C - Scheduler | **PhoBERT-base-v2** (Full fine-tuning (0 freeze), pooling: `last`) | 135,005,193 (100.00%) | 4482.4 MB | 568.1s | 94.55% | 93.93% | 94.40% | **94.17%** | 89.55% |
| `exp10_phobert_pooling_concat4` | Nhóm E - Pooling | **PhoBERT-base-v2** (Full fine-tuning (0 freeze), pooling: `concat_4`) | 135,025,929 (100.00%) | 4559.8 MB | 610.4s | 94.63% | 94.14% | 95.34% | **94.74%** | 91.14% |