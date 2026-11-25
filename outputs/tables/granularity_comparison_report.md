# BÁO CÁO SO SÁNH MỨC TỪ (WORD-LEVEL) VÀ MỨC ÂM TIẾT (SYLLABLE-LEVEL) — NHÁNH PRETRAINED PLM

## 1. Bảng Kết quả Tổng hợp Ma trận 2×2 — VLSP 2016 (1,872 câu test, 1,804 thực thể)

| Model | Granularity | Precision (%) | Recall (%) | Micro-F1 (%) | Macro-F1 (%) | Char-Unified F1 (%) | Runtime (s) | Peak VRAM (MB) | Avg Tokens | Avg Subwords (Fertility) | Max SW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PhoBERT-base-v2** (`exp01_phobert_baseline`) | `word` | 93.74 | 94.68 | **94.21** | 89.76 | 94.21 | 419.5 | 4482.4 | 22.79 | 25.59 (1.03×) | 108 |
| **PhoBERT-base-v2** (`exp13_phobert_vlsp_syllable`) | `syllable` | 91.56 | 93.79 | **92.66** | 86.78 | 92.66 | 443.0 | 4877.2 | 27.04 | 29.46 (1.02×) | 139 |
| **XLM-RoBERTa-base** (`exp02_xlmr_baseline`) | `syllable` | 92.98 | 93.96 | **93.47** | 89.25 | 93.47 | 1479.8 | 7249.5 | 27.04 | 33.85 (1.18×) | 150 |
| **XLM-RoBERTa-base** (`exp14_xlmr_vlsp_word`) | `word` | 92.02 | 93.90 | **92.95** | 88.53 | 92.95 | 2664.5 | 9380.3 | 22.79 | 42.41 (1.77×) | 241 |

## 2. Bảng Kết quả Tổng hợp Ma trận 2×2 & Kiểm định chéo BiLSTM-CRF — PhoNER_COVID19 (3,000 câu test chung, 11,735 thực thể)

| Model | Granularity | Precision (%) | Recall (%) | Micro-F1 (%) | Macro-F1 (%) | Char-Unified F1 (%) | Runtime (s) | Peak VRAM (MB) | Avg Tokens | Avg Subwords (Fertility) | Max SW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM (TV1)** (`baseline_bilstm_word`) | `word` | 86.61 | 89.71 | **88.13** | 83.00 | 88.08 | 236.3 | N/A (CPU/TV1) | 28.56 | N/A (Word+Char) | 162 |
| **BiLSTM-CNN-CRF (TV1)** (`bilstm_cnn_crf_phoner_word`) | `word` | 93.23 | 92.09 | **92.66** | 89.16 | 92.57 | 952.6 | N/A (CPU/TV1) | 28.56 | N/A (Word+Char) | 162 |
| **PhoBERT-base-v2** (`exp15_phobert_phoner_word`) | `word` | 92.75 | 94.90 | **93.81** | 89.05 | 93.69 | 327.1 | 5910.6 | 28.56 | 32.20 (1.06×) | 177 |
| **PhoBERT-base-v2** (`exp16_phobert_phoner_syllable`) | `syllable` | 91.61 | 93.93 | **92.76** | 86.14 | 92.76 | 286.1 | 6765.5 | 36.12 | 39.52 (1.04×) | 194 |
| **XLM-RoBERTa-base** (`exp17_xlmr_phoner_syllable`) | `syllable` | 93.16 | 95.07 | **94.11** | 91.45 | 94.11 | 1599.0 | 10372.8 | 36.12 | 45.06 (1.19×) | 233 |
| **XLM-RoBERTa-base** (`exp18_xlmr_phoner_word`) | `word` | 93.05 | 94.69 | **93.86** | 91.04 | 93.74 | 3577.1 | 10563.2 | 28.56 | 60.19 (2.04×) | 256 |

## 3. Kiểm định Ý nghĩa Thống kê (Paired Bootstrap — 1,000 resamples)

| Cặp đối chiếu (A vs B) | F1(A) (%) | F1(B) (%) | ΔF1 (A - B) | 95% CI | p-value (2-sided) | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `vlsp_phobert_word_vs_syllable` | 94.21 | 92.66 | **+1.55%** | [+0.68%, +2.43%] | 0.0000 | Có ý nghĩa (p < 0.05) |
| `vlsp_xlmr_syllable_vs_word` | 93.47 | 92.95 | **+0.52%** | [-0.35%, +1.37%] | 0.2340 | Không có ý nghĩa |
| `vlsp_phobert_word_vs_xlmr_syllable` | 94.21 | 93.47 | **+0.74%** | [-0.31%, +1.78%] | 0.1400 | Không có ý nghĩa |
| `phoner_phobert_word_vs_syllable` | 93.81 | 92.76 | **+1.05%** | [+0.64%, +1.48%] | 0.0000 | Có ý nghĩa (p < 0.05) |
| `phoner_xlmr_syllable_vs_word` | 94.11 | 93.86 | **+0.24%** | [-0.08%, +0.61%] | 0.1500 | Không có ý nghĩa |
| `phoner_phobert_word_vs_xlmr_syllable` | 93.81 | 94.11 | **-0.30%** | [-0.69%, +0.07%] | 0.1160 | Không có ý nghĩa |
| `phoner_phobert_word_vs_bilstm_crf` | 93.81 | 92.66 | **+1.15%** | [+0.69%, +1.61%] | 0.0000 | Có ý nghĩa (p < 0.05) |
| `phoner_xlmr_syllable_vs_bilstm_crf` | 94.11 | 92.66 | **+1.45%** | [+1.01%, +1.88%] | 0.0000 | Có ý nghĩa (p < 0.05) |
| `phoner_xlmr_word_vs_bilstm_crf` | 93.86 | 92.66 | **+1.21%** | [+0.73%, +1.65%] | 0.0000 | Có ý nghĩa (p < 0.05) |
