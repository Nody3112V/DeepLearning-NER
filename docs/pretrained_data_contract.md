# HỢP ĐỒNG DỮ LIỆU NỘI BỘ (DATA CONTRACT)
## Nhánh Pretrained Language Models (PhoBERT & XLM-R)
**Người phụ trách:** Thành viên 2 (Senior ML Engineer)  
**Bên liên quan:** Thành viên 3 (Hạ tầng, LLM & Đánh giá), Thành viên 1 (BiLSTM-CRF & PhoNER)  
**Trạng thái:** ĐÃ XÁC NHẬN & ĐỒNG THUẬN (CONFIRMED)  
**Ngày cập nhật:** 21/11/2026  

---

### 1. Mục đích và Phạm vi
Tài liệu này xác lập quy chuẩn kỹ thuật bất biến giữa nhánh **Pretrained PLM (Thành viên 2)** và toàn bộ hệ thống chung của dự án `DeepLearning-NER`. Đảm bảo toàn bộ quy trình từ chuẩn bị dữ liệu, huấn luyện, đánh giá độc lập đến bàn giao tích hợp demo và phân tích lỗi chéo diễn ra thông suốt, không xảy ra xung đột kiểu dữ liệu hoặc sai lệch số liệu báo cáo ACL.

---

### 2. Định dạng Dữ liệu Đầu vào (Input Data Contract)
Tuân thủ lớp chuẩn `NERSample` trong `src/common/schema.py`:

```json
{
  "id": "vlsp_word_train_00001",
  "dataset": "vlsp",
  "granularity": "word",
  "tokens": ["Không_khí", "thật", "náo_nhiệt", "."],
  "tags": ["O", "O", "O", "O"]
}
```

* **Quy ước Sentence ID:** `<prefix>_<granularity>_<split>_<05d>`
  * `prefix`: `"vlsp"` hoặc `"phoner"`
  * `granularity`: `"word"` hoặc `"syllable"`
  * `split`: `"train"`, `"dev"`, hoặc `"test"`
  * `index`: Số thứ tự 5 chữ số đệm số 0 (vd: `00001`, `01872`).
* **Quy ước Tokens & Tags:**
  * `tokens: List[str]` và `tags: List[str]` bắt buộc có cùng độ dài 1-1 (`len(tokens) == len(tags)`).
  * Nhãn BIO phẳng chuẩn: `"O"`, `"B-<TYPE>"`, `"I-<TYPE>"`.

---

### 3. Tập Nhãn Chuẩn (Label Sets)

#### A. Tập dữ liệu VLSP (Flat NER CoNLL 2016)
Gồm đúng **4 loại thực thể** (9 nhãn BIO):
* `PER`: Tên người (ví dụ: `Nguyễn_Văn_A`, `Ngô_Bảo_Châu`).
* `LOC`: Địa danh (ví dụ: `Hà_Nội`, `Việt_Nam`).
* `ORG`: Tổ chức (ví dụ: `Bộ_Y_tế`, `Đại_học_Bách_khoa`).
* `MISC`: Thực thể khác (ví dụ: `SEA_Games`).
* Tagset: `["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]`.

#### B. Tập dữ liệu PhoNER_COVID19
Gồm đúng **10 loại thực thể** (21 nhãn BIO):
* `PATIENT_ID`, `NAME`, `AGE`, `GENDER`, `JOB`, `LOCATION`, `ORGANIZATION`, `SYMPTOM_AND_DISEASE`, `TRANSPORTATION`, `DATE`.

---

### 4. Quy ước Tập Kiểm thử (Test Split Agreement)

1. **Đối với PhoNER_COVID19:**
   * Cố định đúng 3.000 câu theo mảng chỉ số trong `data/splits/test_indices.json`.
2. **Đối với VLSP 2016:**
   * **Quyết định chốt cùng TV3:** Đọc trực tiếp toàn bộ 1.872 câu từ `data/processed/vlsp/word/test.jsonl` (và `syllable`). 
   * **Không sửa `test_indices.json`** để bảo vệ cấu trúc hiện tại của PhoNER và tránh lỗi `IndexError`.
   * Khi thực hiện phân tích lỗi chéo (error analysis) giữa 3 mô hình, khớp câu trực tiếp theo trường `id` (`vlsp_word_test_00001` đến `vlsp_word_test_01872`).

---

### 5. Tiền xử lý & Tách từ (Word Segmentation Contract)

* **Công cụ chuẩn duy nhất:** `VnCoreNLP RDRsegmenter` bọc trong `src/common/word_segmenter.py`.
* **Quy ước mức từ (`word`):** Các âm tiết cấu thành một từ ghép được nối với nhau bằng dấu gạch dưới `_` (ví dụ: `["Đại_học", "Quốc_gia", "Hà_Nội"]`).
* **Quy ước mức âm tiết (`syllable`):** Từng âm tiết là một token độc lập không chứa `_` (ví dụ: `["Đại", "học", "Quốc", "gia", "Hà", "Nội"]`).
* Cả 2 bản đều được chuẩn hóa và lưu tại `data/processed/vlsp/{word,syllable}/`.

---

### 6. Đánh giá & Metrics (Evaluation Contract)

* **Hàm chuẩn:** `compute_metrics(gold_tags, pred_tags)` từ `src/common/metrics.py`.
* **Định nghĩa Entity-level:** Khớp chính xác (Exact Match) đồng thời cả ba tiêu chí: vị trí bắt đầu (`start`), vị trí kết thúc (`end_exclusive`), và loại thực thể (`type`).
* **Chỉ số chính:** Micro-F1 (được báo cáo trong Bảng kết quả chính).
* **Chỉ số phụ:** Macro-F1 (trung bình cộng F1 của các thực thể có `support > 0`).
* **Xử lý Subwords trong PLM:**
  * Mô hình Transformer chia từ thành nhiều subwords (BPE). Subword đầu tiên nhận nhãn gốc; các subword phía sau và special tokens nhận nhãn bỏ qua `-100`.
  * **Trách nhiệm của TV2:** Phải lọc bỏ toàn bộ nhãn `-100` khi trích xuất `pred_tags` trước khi truyền vào `compute_metrics`, đảm bảo chuỗi nhãn dự đoán có độ dài bằng chính xác số lượng tokens gốc.

---

### 7. Định dạng Đầu ra Dự đoán (Predictions Contract)

* **Đường dẫn xuất tệp:** `outputs/predictions/pretrained_test.jsonl`
* **Cấu trúc:** Mỗi dòng là một `NERSample` chuẩn JSONL:
  ```json
  {"id": "vlsp_word_test_00001", "dataset": "vlsp", "granularity": "word", "tokens": ["Nhiều", "gia_đình", "thực"], "tags": ["O", "O", "O"]}
  ```
  *(Trong đó `tags` là nhãn BIO do mô hình dự đoán).*
* **Kiểm định tương thích:** File xuất ra phải chạy thành công lệnh đánh giá của TV3:
  ```bash
  python -m src.common.metrics --gold data/processed/vlsp/word/test.jsonl \
                               --pred outputs/predictions/pretrained_test.jsonl
  ```

---

### 8. Giao diện Suy luận Tích hợp Demo (`demo/app.py`)

Lớp `PLMPredictor` trong `src/models/pretrained/predictor.py` cung cấp 2 phương thức phân tách rõ ràng:

1. **`predict_tokens(tokens: List[str]) -> List[str]`**:
   * Dùng cho luồng đánh giá (eval), kiểm thử nội bộ và phân tích lỗi.
   * Nhận danh sách token, trả về danh sách nhãn BIO.
2. **`predict(raw_text: str) -> List[Dict[str, Any]]`**:
   * Dùng cho giao diện tương tác (`demo/app.py`).
   * Tự động gọi `WordSegmenter` bên trong (nếu là mô hình word-level) để giao diện `app.py` hoàn toàn độc lập với logic tiền xử lý.
   * Hỗ trợ truyền chung instance `segmenter` khi khởi tạo (`create_predictor(ckpt, segmenter=shared_seg)`) để tối ưu hóa bộ nhớ RAM và không khởi tạo lại JVM.
   * Kết quả trả về:
     ```python
     [
       {"text": "Hà Nội", "type": "LOC", "start": 10, "end": 17},
       {"text": "Bộ Y tế", "type": "ORG", "start": 25, "end": 32}
     ]
     ```

---
*Tài liệu này đóng vai trò căn cứ kỹ thuật chính thức cho nhánh Pretrained PLM kể từ ngày 21/11/2026.*
