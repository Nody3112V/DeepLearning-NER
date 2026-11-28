# DeepLearning-NER — Nhận dạng Thực thể có Tên (NER) Tiếng Việt

Bài tập lớn môn **Học sâu (Deep Learning)** — Đề tài 07: Nhận dạng thực thể có tên tiếng Việt theo chuẩn báo cáo hội nghị ACL.

## 1. Tổng quan đề tài
Dự án so sánh 3 hướng tiếp cận gán nhãn chuỗi (định dạng BIO) trên 2 bộ dữ liệu tiếng Việt (**PhoNER_COVID19** và **VLSP 2016/2018**) ở cả mức từ (*word-level*) và mức âm tiết (*syllable-level*):
1. **Nhánh 1 — BiLSTM-CRF (tự cài đặt từ đầu):** Kết hợp biểu diễn ký tự (Char-CNN/Char-LSTM) và tầng CRF (Forward-Backward & Viterbi).
2. **Nhánh 2 — Pretrained Language Models:** Tinh chỉnh **PhoBERT** và **XLM-R** với căn chỉnh nhãn từ phụ (`-100`).
3. **Nhánh 3 — LLM Few-shot:** Học từ vài mẫu (Zero-shot / Few-shot In-Context Learning) kèm hậu xử lý về chuẩn BIO.

## 2. Phân công thành viên
| Thành viên | Họ tên & MSV | Phụ trách chính | Tỷ lệ đóng góp |
| :--- | :--- | :--- | :---: |
| Thành viên 1 | Nguyễn Quang Vinh (24022490) | Nhánh BiLSTM-CRF, xử lý PhoNER_COVID19 | 33.3% |
| Thành viên 2 | Tạ Duy Khánh (24022364) | Nhánh PhoBERT & XLM-R, xử lý VLSP | 33.3% |
| Thành viên 3 | Nguyễn Thế Hiển (24022323) | Hạ tầng chung (`src/common/`), nhánh LLM, phân tích lỗi & ghép báo cáo ACL | 33.3% |

## 3. Cấu trúc thư mục
```text
DeepLearning-NER/
├── configs/                # Cấu hình siêu tham số (YAML) cho 3 nhánh
├── data/                   # Kịch bản tải dữ liệu & danh sách chỉ số test cố định (không lưu data nặng trên Git)
├── docs/                   # Tài liệu quy ước định dạng dữ liệu chung
├── examples/               # Kịch bản ví dụ minh họa suy luận và căn chỉnh từ phụ
├── outputs/                # Thống kê, bảng biểu, hình vẽ, dự đoán (checkpoints được loại trừ bởi .gitignore)
├── report/acl_latex/       # Mã nguồn báo cáo LaTeX chuẩn ACL & slide thuyết trình
├── scripts/                # Kịch bản tái tạo bảng kết quả chính
├── src/
│   ├── common/             # Schema JSONL, Entity-level F1 (seqeval), bộ tách từ, logger
│   ├── data_prep/          # Tiền xử lý PhoNER_COVID19 và VLSP về chuẩn JSONL chung
│   ├── models/             # 3 nhánh mô hình độc lập: bilstm_crf/, pretrained/, llm_ner/
│   └── analysis/           # Phân tích lỗi chéo và trực quan hóa
├── tests/                  # Bộ kiểm thử đơn vị (unit tests) cho cả 3 nhánh và hạ tầng chung
├── requirements.txt        # Danh sách thư viện Python
└── README.md
```

## 4. Hướng dẫn cài đặt môi trường
Yêu cầu **Python 3.10+**.

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows (Git Bash):
source venv/Scripts/activate
# Linux / macOS:
# source venv/bin/activate

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Tải và chuẩn hóa dữ liệu
Dữ liệu gốc và dữ liệu đã xử lý không được đẩy lên GitHub. Chạy các kịch bản sau để tải và chuẩn hóa về định dạng JSONL chung:

```bash
# 1. Tải dữ liệu
bash data/download_data.sh

# 2. Tiền xử lý về chuẩn JSONL (mức từ & mức âm tiết)
PYTHONUTF8=1 python src/data_prep/process_phoner.py
PYTHONUTF8=1 python src/data_prep/process_vlsp.py
```

## 6. Kiểm thử hạ tầng (Unit Tests)
```bash
# Chạy toàn bộ bộ kiểm thử unittest (Nhánh 1, Nhánh 2 & hạ tầng chung)
PYTHONUTF8=1 python -m unittest discover -s tests -p "tests_*.py" -v

# Hoặc chạy toàn bộ 16 tệp (gồm cả 2 tệp pytest của Nhánh 3) nếu đã cài pytest:
PYTHONUTF8=1 python -m pytest tests/ -v
```

## 7. Huấn luyện, đánh giá và theo dõi thí nghiệm
- **Nhật ký huấn luyện (Weights & Biases):** https://wandb.ai/nody3112v-vietnam-national-university-hanoi/vietnamese-ner-acl
- **Nhánh 1 — BiLSTM & BiLSTM-CNN-CRF (Tuần 2–3):**
```bash
# 1. Huấn luyện Baseline BiLSTM (Tuần 2 - không CharEncoder, không CRF)
PYTHONUTF8=1 python -m src.models.bilstm_crf.train \
  --config configs/bilstm_crf.yaml \
  --exp-name baseline_bilstm_word \
  --no-char --no-crf

# 2. Huấn luyện BiLSTM-CNN-CRF hoàn chỉnh (Tuần 3 - tự cài đặt từ đầu)
PYTHONUTF8=1 python -m src.models.bilstm_crf.train \
  --config configs/bilstm_crf.yaml

# 3. Kiểm tra lại Entity-level F1 trên tệp dự đoán tập test bằng hàm chấm chung
PYTHONUTF8=1 python -m src.common.metrics \
  --gold data/processed/phoner_covid19/word/test.jsonl \
  --pred outputs/predictions/bilstm_crf_test.jsonl
```

| Mô hình (PhoNER_COVID19 — word-level) | Best Dev F1 | Test Precision | Test Recall | Test Micro-F1 | Test Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline BiLSTM (Tuần 2) | 89.05% | 86.61% | 89.71% | 88.13% | 83.00% |
| **BiLSTM-CNN-CRF tự cài đặt (Tuần 3)** | **93.54%** | **93.23%** | **92.09%** | **92.66%** | **89.16%** |
- **Tái tạo kết quả toàn dự án:** Kịch bản tái tạo bảng kết quả chính (`scripts/reproduce_main_table.sh`) sẽ được hoàn thiện sau khi các nhánh hoàn tất thực nghiệm.
