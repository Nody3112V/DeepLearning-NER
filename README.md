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
├── outputs/                # Thống kê, bảng biểu, hình vẽ, dự đoán (checkpoints được loại trừ bởi .gitignore)
├── report/acl_latex/       # Mã nguồn báo cáo LaTeX chuẩn ACL & slide thuyết trình
├── scripts/                # Kịch bản tái tạo bảng kết quả chính
├── src/
│   ├── common/             # Schema JSONL, Entity-level F1 (seqeval), bộ tách từ, logger
│   ├── data_prep/          # Tiền xử lý PhoNER_COVID19 và VLSP về chuẩn JSONL chung
│   ├── models/             # 3 nhánh mô hình độc lập: bilstm_crf/, pretrained/, llm_ner/
│   └── analysis/           # Phân tích lỗi chéo và trực quan hóa
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
PYTHONUTF8=1 python -m unittest tests_schema.py tests_metrics.py tests_logger.py tests_word_segmenter.py tests_process_vlsp.py -v
```

## 7. Huấn luyện, đánh giá và theo dõi thí nghiệm
- **Nhật ký huấn luyện (Weights & Biases):** *(Cập nhật link project W&B công khai/chia sẻ với giảng viên tại đây)*
- **Huấn luyện & Tái tạo kết quả:** Các lệnh chạy huấn luyện từng nhánh và kịch bản tái tạo bảng kết quả chính (`scripts/reproduce_main_table.sh`) sẽ được cập nhật đầy đủ từ Tuần 2–4.
