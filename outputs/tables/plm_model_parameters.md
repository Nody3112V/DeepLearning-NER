# Thống kê Kiến trúc và Tham số Mô hình Pretrained PLM (PhoBERT & XLM-R)

Tài liệu chi tiết về số lượng tham số, kích thước ma trận và phân tích tác động của các chiến lược kết hợp tầng ẩn (Pooling) và đóng băng tham số (Freezing).

---

## 1. Tổng quan Cấu hình Backbone

| Thuộc tính | PhoBERT-base-v2 (`vinai/phobert-base-v2`) | XLM-RoBERTa-base (`xlm-roberta-base`) |
| :--- | :--- | :--- |
| **Kiến trúc cơ sở** | RoBERTa Architecture (Monolingual Vietnamese) | RoBERTa Architecture (Multilingual 100+ langs) |
| **Kích thước từ vựng (Vocab Size)** | 64,001 tokens (BPE) | 250,002 tokens (SentencePiece) |
| **Hidden Size ($H$)** | 768 | 768 |
| **Số tầng Encoder ($L$)** | 12 | 12 |
| **Số Attention Heads** | 12 | 12 |
| **Intermediate Size (FFN)** | 3,072 | 3,072 |
| **Tổng tham số Backbone** | **134,998,272** (~135M) | **278,043,648** (~278M) |
| **Tham số Embeddings** | 49,353,216 (36.56% backbone) | 192,398,592 (69.20% backbone) |
| **Tham số mỗi tầng Transformer** | 7,087,872 (12 tầng = 85,054,464) | 7,087,872 (12 tầng = 85,054,464) |
| **Tham số Pooler Layer** | 590,592 | 590,592 |

> **Nhận xét quan trọng**:
> - Phần thân Transformer (12 encoder layers) của PhoBERT và XLM-R có cấu trúc và số lượng tham số hoàn toàn tương đương nhau (85,054,464 tham số).
> - Sự khác biệt lớn nhất về tổng dung lượng (~135M so với ~278M) nằm ở tầng **Embeddings**: XLM-R chứa bộ từ vựng đa ngữ 250k từ (chiếm gần 70% tổng mô hình), trong khi PhoBERT tối ưu riêng cho tiếng Việt với từ vựng 64k từ (chiếm 36.56%).

---

## 2. Phân tích Các Chiến lược Pooling (Đầu vào Classifier)

Tập nhãn VLSP 2016 gồm **9 nhãn BIO** (`num_labels = 9`): `O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC`.

| Chiến lược Pooling | Kích thước Đầu vào Head | Công thức Feature | Tham số Head ($W \times H + b$) | Tham số Bổ sung | Tổng Tham số Mô hình (PhoBERT) | Tổng Tham số Mô hình (XLM-R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `last` | $[B, L, 768]$ | $h^{(12)}$ | $768 \times 9 + 9 = \mathbf{6,921}$ | $0$ | **135,005,193** | **278,050,569** |
| `concat_4` | $[B, L, 3072]$ | $[h^{(9)}; h^{(10)}; h^{(11)}; h^{(12)}]$ | $3072 \times 9 + 9 = \mathbf{27,657}$ | $0$ | **135,025,929** | **278,071,305** |
| `mean_4` | $[B, L, 768]$ | $\frac{1}{4}\sum_{i=9}^{12} h^{(i)}$ | $768 \times 9 + 9 = \mathbf{6,921}$ | $0$ | **135,005,193** | **278,050,569** |
| `weighted` | $[B, L, 768]$ | $\sum_{i=0}^{12} \alpha_i h^{(i)}$ với $\alpha = \text{softmax}(w)$ | $768 \times 9 + 9 = \mathbf{6,921}$ | $13$ ($w \in \mathbb{R}^{13}$) | **135,005,206** | **278,050,582** |

---

## 3. Phân tích Các Chiến lược Đóng băng Tham số (Freezing Strategies)

Dựa trên cấu hình `pooling_strategy = "last"` và `num_labels = 9`:

### A. PhoBERT-base-v2

| Chiến lược Đóng băng | Tham số Huấn luyện (Trainable) | Tham số Đóng băng (Frozen) | Tỷ lệ Trainable (%) | Ghi chú Tối ưu GPU VRAM & Hội tụ |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-tuning (0 freeze)** | **135,005,193** | 0 | **100.00%** | Fine-tune toàn bộ, học biểu diễn tối ưu nhất nhưng tốn VRAM nhất. |
| **Freeze Embeddings only** | **85,651,977** | 49,353,216 | **63.44%** | Tiết kiệm ~36.5% bộ nhớ gradient, bảo tồn không gian vector từ điển gốc. |
| **Freeze Embeddings + 3 Layers** | **64,388,361** | 70,616,832 | **47.69%** | Đóng băng 3 tầng cú pháp/hình thái học thấp nhất. |
| **Freeze Embeddings + 6 Layers** | **43,124,745** | 91,880,448 | **31.94%** | Chỉ fine-tune nửa trên (6 tầng trên cùng) của Transformer. |
| **Freeze Embeddings + 9 Layers** | **21,861,129** | 113,144,064 | **16.19%** | Chỉ fine-tune 3 tầng ngữ nghĩa cao nhất. |
| **Freeze Entire Encoder (Head-only)**| **6,921** | 134,998,272 | **0.005%** | Linear probing: Huấn luyện cực nhanh (<1 phút/epoch), dùng làm baseline so sánh. |

### B. XLM-RoBERTa-base

| Chiến lược Đóng băng | Tham số Huấn luyện (Trainable) | Tham số Đóng băng (Frozen) | Tỷ lệ Trainable (%) | Ghi chú Tối ưu GPU VRAM & Hội tụ |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-tuning (0 freeze)** | **278,050,569** | 0 | **100.00%** | Fine-tune toàn bộ mô hình đa ngữ. |
| **Freeze Embeddings only** | **85,651,977** | 192,398,592 | **30.80%** | **Cắt giảm tới 69.2% tham số cần gradient!** Giảm tải VRAM vượt bậc. |
| **Freeze Embeddings + 3 Layers** | **64,388,361** | 213,662,208 | **23.16%** | Đóng băng embeddings và 3 tầng dưới. |
| **Freeze Embeddings + 6 Layers** | **43,124,745** | 234,925,824 | **15.51%** | Đóng băng một nửa số tầng encoder. |
| **Freeze Embeddings + 9 Layers** | **21,861,129** | 256,189,440 | **7.86%** | Chỉ cập nhật 3 tầng trên cùng. |
| **Freeze Entire Encoder (Head-only)**| **6,921** | 278,043,648 | **0.002%** | Linear probing trên representation đa ngữ sẵn có. |

---

## 4. Sơ đồ Kiến trúc Luồng Dữ liệu (ASCII Data Flow)

```
                       [ Chuỗi Token Từ / Âm tiết ]
                                    │
                                    ▼
                     [ Subword Tokenizer & Alignment ]
             (PhoBERT: slow tokenizer / XLM-R: fast tokenizer)
                                    │
                                    ▼
                      [ input_ids, attention_mask ]
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      [ Token Embeddings ]                     [ Position Embeddings ]
    (PhoBERT: 64k x 768)                    (PhoBERT / XLM-R: 514 x 768)
    (XLM-R: 250k x 768)
               └────────────────────┬────────────────────┘
                                    │
                     [ LayerNorm + Dropout ] -> h^(0) [B, L, 768]
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      [ Layer 01-04 ]        [ Layer 05-08 ]        [ Layer 09-12 ]
    (Transformer Layers)   (Transformer Layers)   (Transformer Layers)
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
             Tuple of 13 Hidden States: (h^(0), h^(1), ..., h^(12))
                                    │
                                    ▼
                    [ Multi-Layer Pooling Module ]
      ┌─────────────────┬──────────────────┬─────────────────┬─────────────────┐
      │     "last"      │    "concat_4"    │     "mean_4"    │   "weighted"    │
      │    h^(12)       │  [h^(9)..h^(12)] │  Mean(h^(9..12))│  Sum(w_i * h^i) │
      │   [B, L, 768]   │   [B, L, 3072]   │   [B, L, 768]   │   [B, L, 768]   │
      └─────────────────┴──────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
                       [ Dropout (p = 0.2) ]
                                    │
                                    ▼
                    [ Linear Classification Head ]
                      (W: [in_dim, 9], b: [9])
                                    │
                                    ▼
                      Logits: [B, L, 9] (Real Tokens)
                                    │
                         (Khi có nhãn mục tiêu)
                                    ▼
                     [ CrossEntropyLoss (ignore=-100) ]
        (Bỏ qua hoàn toàn: Padding, Special Tokens, Subwords sau token đầu)
```
