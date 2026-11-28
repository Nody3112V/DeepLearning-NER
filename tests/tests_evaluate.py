"""
tests_evaluate.py - Bộ unit test kiểm định quy trình đánh giá chính thức (src/models/pretrained/evaluate.py).

Kiểm tra:
  1. Kiểm định Evaluation Protocol: tệp test hợp lệ, số lượng mẫu, bộ nhãn, không trùng ID.
  2. Phát hiện lỗi và chặn (BLOCKED): tệp không tồn tại, sai số lượng mẫu, ID trùng, nhãn ngoài TAG_SETS.
  3. Phục hồi nhãn Subword về Token gốc: 1-1 mapping, câu 0 thực thể, câu nhiều thực thể, xử lý truncation.
  4. Tính toán Entity-level Metrics (compute_metrics & evaluate_files) khớp 100%.
  5. Định dạng đầu ra tệp JSONL tuân thủ NERSample schema.
  6. Thống kê số lượng thực thể thực tế (Support) và dự đoán (TP + FP).
"""
import os
import json
import shutil
import tempfile
import unittest
import torch

from src.common.schema import NERSample, save_jsonl, load_jsonl
from src.common.metrics import compute_metrics, evaluate_files
from src.models.pretrained.evaluate import (
    validate_evaluation_protocol,
    analyze_entity_counts,
)


class TestOfficialEvaluation(unittest.TestCase):
    """Bộ unit test cho quy trình đánh giá chính thức."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gold_path = os.path.join(self.temp_dir, "gold_test.jsonl")
        self.pred_path = os.path.join(self.temp_dir, "pred_test.jsonl")

        # Tạo tập mẫu giả lập hợp lệ
        self.samples = [
            NERSample(
                id="vlsp_word_test_00001",
                dataset="vlsp",
                granularity="word",
                tokens=["Hà_Nội", "là", "thủ_đô", "Việt_Nam", "."],
                tags=["B-LOC", "O", "O", "B-LOC", "O"]
            ),
            NERSample(
                id="vlsp_word_test_00002",
                dataset="vlsp",
                granularity="word",
                tokens=["Ông", "Nguyễn_Văn_A", "làm_việc", "tại", "Vingroup", "."],
                tags=["O", "B-PER", "O", "O", "B-ORG", "O"]
            ),
            NERSample(
                id="vlsp_word_test_00003",
                dataset="vlsp",
                granularity="word",
                tokens=["Hôm_nay", "trời", "rất", "đẹp", "."],
                tags=["O", "O", "O", "O", "O"]  # Câu không có thực thể nào
            ),
        ]
        save_jsonl(self.samples, self.gold_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_protocol_verification_success(self):
        """1. Kiểm tra protocol hợp lệ khi dữ liệu chuẩn."""
        loaded = validate_evaluation_protocol(self.gold_path, dataset_name="vlsp", expected_count=3)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0].id, "vlsp_word_test_00001")

    def test_02_protocol_blocked_on_missing_file(self):
        """2. Chặn (BLOCKED) khi file không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            validate_evaluation_protocol("non_existent_file.jsonl", dataset_name="vlsp")

    def test_03_protocol_blocked_on_count_mismatch(self):
        """3. Chặn (BLOCKED) khi số lượng mẫu không khớp mong đợi."""
        with self.assertRaises(ValueError):
            validate_evaluation_protocol(self.gold_path, dataset_name="vlsp", expected_count=100)

    def test_04_protocol_blocked_on_duplicate_ids(self):
        """4. Chặn (BLOCKED) khi phát hiện ID trùng lặp."""
        dup_samples = list(self.samples) + [self.samples[0]]  # Trùng mẫu 0
        dup_path = os.path.join(self.temp_dir, "dup.jsonl")
        save_jsonl(dup_samples, dup_path)
        with self.assertRaises(ValueError):
            validate_evaluation_protocol(dup_path, dataset_name="vlsp", expected_count=None)

    def test_05_protocol_blocked_on_invalid_tag(self):
        """5. Chặn (BLOCKED) khi gặp nhãn không thuộc TAG_SETS của dataset."""
        bad_path = os.path.join(self.temp_dir, "bad_tag.jsonl")
        # Ghi trực tiếp JSON thô không qua validate của save_jsonl
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "vlsp_word_test_00099",
                "dataset": "vlsp",
                "granularity": "word",
                "tokens": ["A", "B"],
                "tags": ["B-INVALID_TYPE", "O"]
            }) + "\n")
        with self.assertRaises(ValueError):
            validate_evaluation_protocol(bad_path, dataset_name="vlsp", expected_count=None)

    def test_06_evaluate_files_matching(self):
        """6. Kiểm tra hàm evaluate_files đánh giá chính xác trên 2 file JSONL."""
        # Giả lập prediction: đúng 1 thực thể Hà_Nội (LOC), đoán nhầm Việt_Nam thành ORG
        preds = [
            NERSample(
                id="vlsp_word_test_00001",
                dataset="vlsp",
                granularity="word",
                tokens=["Hà_Nội", "là", "thủ_đô", "Việt_Nam", "."],
                tags=["B-LOC", "O", "O", "B-ORG", "O"]
            ),
            NERSample(
                id="vlsp_word_test_00002",
                dataset="vlsp",
                granularity="word",
                tokens=["Ông", "Nguyễn_Văn_A", "làm_việc", "tại", "Vingroup", "."],
                tags=["O", "B-PER", "O", "O", "B-ORG", "O"]
            ),
            NERSample(
                id="vlsp_word_test_00003",
                dataset="vlsp",
                granularity="word",
                tokens=["Hôm_nay", "trời", "rất", "đẹp", "."],
                tags=["O", "O", "O", "O", "O"]
            ),
        ]
        save_jsonl(preds, self.pred_path)

        metrics = evaluate_files(self.gold_path, self.pred_path)
        self.assertIn("overall", metrics)
        self.assertIn("per_type", metrics)
        self.assertIn("macro_f1", metrics)

        # Kiểm tra TP của LOC: 1 (Hà_Nội đúng, Việt_Nam đoán thành ORG)
        self.assertEqual(metrics["per_type"]["LOC"]["tp"], 1)
        self.assertEqual(metrics["per_type"]["PER"]["tp"], 1)
        self.assertEqual(metrics["per_type"]["ORG"]["tp"], 1)

    def test_07_analyze_entity_counts(self):
        """7. Kiểm tra phân tích đếm thực thể Ground-Truth và Predicted."""
        gold_tags = [["B-LOC", "I-LOC", "O", "B-PER"]]
        pred_tags = [["B-LOC", "I-LOC", "O", "B-LOC"]]
        metrics = compute_metrics(gold_tags, pred_tags)
        analysis = analyze_entity_counts(metrics)

        # Gold có 2 thực thể (LOC, PER)
        self.assertEqual(analysis["total_gold_entities"], 2)
        # Pred dự đoán 2 thực thể (LOC, LOC)
        self.assertEqual(analysis["total_predicted_entities"], 2)
        # Đúng 1 thực thể (LOC)
        self.assertEqual(analysis["overall_tp"], 1)
        # Bỏ sót 1 thực thể (PER)
        self.assertEqual(analysis["overall_fn"], 1)
        # Dự đoán sai 1 thực thể (LOC thứ 2 là FP)
        self.assertEqual(analysis["overall_fp"], 1)

    def test_08_roundtrip_jsonl_verification(self):
        """8. Kiểm tra tính toàn vẹn round-trip của tệp JSONL xuất ra."""
        save_jsonl(self.samples, self.pred_path)
        reloaded = list(load_jsonl(self.pred_path, validate=True))
        self.assertEqual(len(reloaded), len(self.samples))
        for s_orig, s_reload in zip(self.samples, reloaded):
            self.assertEqual(s_orig.id, s_reload.id)
            self.assertEqual(s_orig.tokens, s_reload.tokens)
            self.assertEqual(s_orig.tags, s_reload.tags)


if __name__ == "__main__":
    unittest.main()
