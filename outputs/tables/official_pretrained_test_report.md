# Báo cáo Đánh giá Chính thức - Nhánh Pretrained PLM

- **Mô hình**: `vinai/phobert-base-v2` (Pooling: `concat_4`)
- **Tập dữ liệu**: `vlsp` (Mức: `word`)
- **Số lượng câu test**: `1,872`
- **Tốc độ suy luận**: `4.62s` (`2.47 ms/câu` trên GPU)

## 1. Bảng Điểm Entity-level Chuẩn (src.common.metrics)

| Loại Thực thể | Thực tế (Support) | Dự đoán (TP+FP) | Đúng (TP) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LOC** | 770 | 771 | 726 | 94.16% | 94.29% | **94.22%** |
| **MISC** | 30 | 34 | 29 | 85.29% | 96.67% | **90.62%** |
| **ORG** | 159 | 171 | 135 | 78.95% | 84.91% | **81.82%** |
| **PER** | 845 | 851 | 830 | 97.53% | 98.22% | **97.88%** |
| **MICRO OVERALL** | **1804** | **1827** | **1720** | **94.14%** | **95.34%** | **94.74%** |
| **MACRO-F1** | - | - | - | - | - | **91.14%** |

## 2. Kiểm định Tính Toàn vẹn & Error Handling
- [x] 100% mẫu test (1,872 câu) được phục hồi nhãn ở mức token gốc.
- [x] Độ dài `tokens` và `tags` khớp 1-1 trên mọi câu.
- [x] Mọi ID khớp 100% với file chuẩn `data/processed/vlsp/word/test.jsonl`.
- [x] Xử lý an toàn các câu không có thực thể và câu có nhiều thực thể.