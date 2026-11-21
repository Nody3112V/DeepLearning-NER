# BÁO CÁO THỐNG KÊ DỮ LIỆU VLSP 2016 NER

## 1. Tổng quan các tập dữ liệu

| Mức biểu diễn | Phân chia | Số câu | Số token | Min len | Max len | Avg len | Tổng thực thể | Tỷ lệ nhãn O (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Word** | TRAIN | 13,486 | 294,501 | 1 | 118 | 21.84 | 12,083 | 93.58% |
| **Word** | DEV | 1,500 | 32,233 | 1 | 83 | 21.49 | 1,333 | 93.46% |
| **Word** | TEST | 1,872 | 42,666 | 1 | 104 | 22.79 | 1,804 | 93.33% |
| **Syllable** | TRAIN | 13,486 | 348,783 | 1 | 137 | 25.86 | 12,083 | 93.32% |
| **Syllable** | DEV | 1,500 | 38,279 | 1 | 94 | 25.52 | 1,333 | 93.16% |
| **Syllable** | TEST | 1,872 | 50,612 | 1 | 135 | 27.04 | 1,804 | 93.05% |

## 2. Phân bố thực thể (Entity Distribution)

| Tập | Loại | Số lượng | Tỷ lệ (%) |
| :--- | :--- | :---: | :---: |
| TRAIN | **PER** | 5,977 | 49.47% |
| TRAIN | **LOC** | 4,932 | 40.82% |
| TRAIN | **ORG** | 946 | 7.83% |
| TRAIN | **MISC** | 228 | 1.89% |
| DEV | **PER** | 657 | 49.29% |
| DEV | **LOC** | 545 | 40.89% |
| DEV | **ORG** | 107 | 8.03% |
| DEV | **MISC** | 24 | 1.80% |
| TEST | **PER** | 845 | 46.84% |
| TEST | **LOC** | 770 | 42.68% |
| TEST | **ORG** | 159 | 8.81% |
| TEST | **MISC** | 30 | 1.66% |

## 3. Kiểm tra rò rỉ dữ liệu gốc (Data Leakage)

- Câu trùng giữa Train và Dev: **9 câu**
- Câu trùng giữa Train và Test: **10 câu**
- Câu trùng giữa Dev và Test: **1 câu**

> *Ghi chú:* Các câu trùng lặp trong dữ liệu gốc là các tiêu đề ngắn hoặc chữ ký phóng viên (`MINH LUẬN - HOÀNG KHƯƠNG`, `Đi về đâu ?`) thuộc kho ngữ liệu báo chí nguyên bản của VLSP.
