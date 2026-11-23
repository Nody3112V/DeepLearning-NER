# HƯỚNG DẪN KỸ THUẬT CĂN CHỈNH NHÃN SUBWORD (SUBWORD ALIGNMENT GUIDE)
## Nhánh Pretrained Language Models (PhoBERT & XLM-R)
**Người phụ trách:** Thành viên 2 (Senior ML Engineer)  
**Trạng thái:** ĐÃ HOÀN THIỆN & KIỂM ĐỊNH 100%  

---

### 1. Mục tiêu Kỹ thuật
Trong bài toán Nhận dạng Thực thể có Tên (Named Entity Recognition – NER), các mô hình Transformer (PhoBERT, XLM-R) không xử lý văn bản ở mức từ (word) nguyên vẹn mà sử dụng các thuật toán chia nhỏ từ thành **subwords** (Byte-Pair Encoding - BPE hoặc SentencePiece). 

Điều này dẫn đến bài toán: **Một từ gốc có thể bị phân rã thành nhiều subwords**. Ví dụ:
* `Ngô_Bảo_Châu` (`B-PER`) khi qua XLM-R bị phân rã thành 9 subwords: `▁Ngô`, `_`, `B`, `ả`, `o`, `_`, `Ch`, `â`, `u`.
* `Hà_Nội` (`B-LOC`) khi qua XLM-R bị phân rã thành 4 subwords: `▁Hà`, `_`, `N`, `ội`.

Để tính hàm mất mát `CrossEntropyLoss` chính xác mà không làm sai lệch tần suất thực thể và không làm lệch biên đánh giá `seqeval`, dự án áp dụng chiến lược căn chỉnh nhãn chuẩn mực: **First-Subword Labeling with Ignore Index (-100)**.

---

### 2. Quy tắc Căn chỉnh (Alignment Rules)

1. **Subword đầu tiên (First Subword):** Nhận ID nhãn gốc của từ đó: `label2id[tag]`.
2. **Các subword tiếp theo (Subsequent Subwords):** Nhận nhãn bỏ qua `ignore_index = -100`. PyTorch `CrossEntropyLoss` tự động bỏ qua các vị trí này khi tính gradient và loss.
3. **Special Tokens (`<s>`, `</s>`, `[CLS]`, `[SEP]`):** Nhận nhãn `-100`.
4. **Padding Tokens (`<pad>`):** Nhận nhãn `-100`.
5. **Truncation:** Khi chuỗi vượt quá `max_length`, các từ phía sau bị cắt bớt; đảm bảo `len(labels) == len(input_ids)`.
6. **Kiểm tra chặt chẽ:** Nếu gặp nhãn lạ không có trong `label2id`, ném `KeyError`. Nếu số lượng từ và số lượng nhãn đầu vào lệch nhau, ném `ValueError`.

---

### 3. Phân tích Khả năng Tương thích của Tokenizer (Tokenizer Compatibility)

| Đặc tính | **XLM-RoBERTa** (`xlm-roberta-base`) | **PhoBERT** (`vinai/phobert-base-v2`) |
| :--- | :--- | :--- |
| **Lớp Tokenizer** | `XLMRobertaTokenizerFast` | `PhobertTokenizer` |
| **Fast Tokenizer (`is_fast`)** | **True** (dựa trên Rust) | **False** (Slow Tokenizer thuần Python) |
| **Hỗ trợ `encoding.word_ids()`** | **Có sẵn** (Native hỗ trợ `is_split_into_words=True`) | **KHÔNG hỗ trợ** (gọi `word_ids()` gây ngoại lệ) |
| **Cơ chế phân tách** | SentencePiece (ký hiệu tiền tố `▁`) | Byte-Pair Encoding (BPE) trên từ ghép tiếng Việt nối `_` |

#### Giải pháp Đa năng trong `subword_align.py`:
Hàm `tokenize_and_get_word_ids`:
* **Với Fast Tokenizer:** Sử dụng trực tiếp `encoding.word_ids()` từ Rust backend.
* **Với Slow Tokenizer:** Tự động mô phỏng và tái lập mảng `word_ids` chính xác bằng cách mã hóa từng từ gốc qua `tokenizer.encode(token, add_special_tokens=False)`, gán chỉ số từ `w_idx` tương ứng cho từng subword và `None` cho `bos`/`eos`. Nhờ vậy, pipeline hoạt động thông suốt trên cả hai kiến trúc mà không phụ thuộc vào Fast Tokenizer.

---

### 4. Thiết kế API (`subword_align.py`)

1. **`align_labels_with_subwords(word_labels, word_ids, label2id, ignore_index=-100) -> List[int]`**
   Hàm hạt nhân: Nhận danh sách nhãn từ gốc và mảng `word_ids`, xuất mảng nhãn số nguyên tương ứng cho từng subword.
2. **`align_tokens_and_labels(tokens, tags, tokenizer, label2id, max_length=256) -> Dict[str, Any]`**
   Căn chỉnh hoàn chỉnh cho 1 câu: trả về `input_ids`, `attention_mask`, `labels`, `word_ids`.
3. **`align_batch_tokens_and_labels(batch_tokens, batch_tags, tokenizer, label2id, max_length=256, padding=True) -> Dict[str, Any]`**
   Căn chỉnh cho cả một lô (batch) câu có độ dài khác nhau, tự động thực hiện dynamic padding trả về PyTorch Tensors `[B, L]`.
4. **`create_ner_collate_fn(pad_token_id, ignore_index=-100)`**
   Collate function phục vụ `DataLoader` với dynamic padding tối ưu VRAM.

---

### 5. Minh họa Kết quả Thực tế

#### Câu đầu vào:
* Tokens: `['Giáo_sư', 'Ngô_Bảo_Châu', 'sinh', 'tại', 'Hà_Nội', '.']`
* Nhãn BIO: `['O', 'B-PER', 'O', 'O', 'B-LOC', 'O']`

#### Kết quả căn chỉnh trên PhoBERT (`vinai/phobert-base-v2`):
```text
Idx Subword           Input ID  Word ID   Label ID  Label Tag 
--------------------------------------------------------------
0   <s>               0         None      -100      -100 (Bỏ qua)
1   Giáo_sư           3626      0         0         O         
2   Ngô_Bảo_Châu      17052     1         1         B-PER     
3   sinh              418       2         0         O         
4   tại               35        3         0         O         
5   Hà_Nội            120       4         3         B-LOC     
6   .                 5         5         0         O         
7   </s>              2         None      -100      -100 (Bỏ qua)
```

#### Kết quả căn chỉnh trên XLM-R (`xlm-roberta-base`):
```text
Idx Subword           Input ID  Word ID   Label ID  Label Tag 
--------------------------------------------------------------
0   <s>               0         None      -100      -100 (Bỏ qua)
1   ▁Giáo             39005     0         0         O         
2   _                 454       0         -100      -100 (Bỏ qua)
3   s                 7         0         -100      -100 (Bỏ qua)
4   ư                 11479     0         -100      -100 (Bỏ qua)
5   ▁Ngô              90542     1         1         B-PER     
6   _                 454       1         -100      -100 (Bỏ qua)
7   B                 571       1         -100      -100 (Bỏ qua)
8   ả                 17195     1         -100      -100 (Bỏ qua)
9   o                 31        1         -100      -100 (Bỏ qua)
10  _                 454       1         -100      -100 (Bỏ qua)
11  Ch                15982     1         -100      -100 (Bỏ qua)
12  â                 5791      1         -100      -100 (Bỏ qua)
13  u                 34        1         -100      -100 (Bỏ qua)
14  ▁sinh             3811      2         0         O         
15  ▁tại              2251      3         0         O         
16  ▁Hà               8548      4         3         B-LOC     
17  _                 454       4         -100      -100 (Bỏ qua)
18  N                 839       4         -100      -100 (Bỏ qua)
19  ội                65166     4         -100      -100 (Bỏ qua)
20  ▁                 6         5         0         O         
21  .                 5         5         -100      -100 (Bỏ qua)
22  </s>              2         None      -100      -100 (Bỏ qua)
```

---

### 6. Kết quả Kiểm thử Toàn diện (`tests_subword_align.py`)

Tất cả 10 ca kiểm thử đơn vị độc lập đều **ĐẠT 100%** trong `0.002s`:
1. `test_01_single_word_single_subword`: ĐẠT
2. `test_02_single_word_multiple_subwords`: ĐẠT
3. `test_03_special_tokens_receive_ignore_index`: ĐẠT
4. `test_04_padding_receives_ignore_index`: ĐẠT
5. `test_05_truncation_handling`: ĐẠT
6. `test_06_batch_alignment_and_shapes`: ĐẠT
7. `test_07_diverse_bio_tags`: ĐẠT
8. `test_08_unknown_label_raises_key_error`: ĐẠT (Bắt lỗi nhãn lạ chính xác)
9. `test_09_mismatch_tokens_tags_raises_value_error`: ĐẠT (Bắt lỗi lệch độ dài chính xác)
10. `test_10_slow_tokenizer_compatibility`: ĐẠT (Mô phỏng PhobertTokenizer hoạt động chuẩn xác)
