"""
tests_predictor.py - Bộ kiểm thử toàn diện cho src/models/pretrained/predictor.py.

Kiểm chứng đầy đủ 9 nhóm kịch bản yêu cầu:
  1. Câu không có thực thể.
  2. Câu có 1 thực thể.
  3. Câu có nhiều thực thể (gồm cả thực thể lặp lại nhiều lần trong câu).
  4. Câu có dấu câu sát thực thể ('(Hà Nội)', 'TP.HCM', '22/6', dấu ngoặc kép).
  5. Câu tiếng Việt có dấu đầy đủ.
  6. Câu rất dài vượt quá max_seq_len (kiểm tra sliding-window chunking).
  7. Input rỗng, khoảng trắng, và input sai kiểu (None, int, list) không làm sập demo.
  8. Error handling cho checkpoint thiếu/hỏng/lệch schema, model chưa load, và fallback CUDA -> CPU.
  9. Kiểm thử tích hợp trên checkpoint thực (PhoBERT & XLM-R) chạy trên cả CPU và GPU (nếu có CUDA).
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

from src.common.schema import NERSample, TAG_SETS
from src.models.pretrained.predictor import (
    PLMPredictor,
    PRESET_CHECKPOINTS,
    _align_tokens_to_raw_text,
    _resolve_device,
    _tokenize_syllables,
    create_predictor,
    load_predictor,
)


# --------------------------------------------------------------------------- #
# Mock deterministic model & tokenizer (test nhanh không phụ thuộc weights)
# --------------------------------------------------------------------------- #
class DeterministicTokenizer:
    """Tokenizer giả lập có hỗ trợ từ đa subword (từ dài > 6 ký tự tách thành 2 subwords)."""

    def __init__(self):
        self.is_fast = False
        self.bos_token_id = 0
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 2
        self.unk_token_id = 3
        self._tok2id: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {0: "<s>", 1: "<pad>", 2: "</s>", 3: "<unk>"}

    def _get_id(self, piece: str) -> int:
        if piece not in self._tok2id:
            new_id = len(self._id2tok)
            self._tok2id[piece] = new_id
            self._id2tok[new_id] = piece
        return self._tok2id[piece]

    def encode(self, token: str, add_special_tokens: bool = False) -> List[int]:
        # Giả lập đa subword cho từ có '_'
        if "_" in token:
            parts = [p for p in token.split("_") if p]
            return [self._get_id(f"{token}::part{i}") for i in range(len(parts))]
        return [self._get_id(token)]

    def decode_id(self, idx: int) -> str:
        return self._id2tok.get(idx, "")


class RuleBasedMockNERModel(nn.Module):
    """Mô hình PyTorch giả lập trả về logits cố định theo từ điển token để kiểm chứng decode/span."""

    def __init__(self, tokenizer: DeterministicTokenizer, token_tag_rules: Dict[str, str], tag2id: Dict[str, int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_tag_rules = token_tag_rules
        self.tag2id = tag2id
        self.num_labels = len(tag2id)
        self.model_name_or_path = "mock/phobert-rule"
        self.pooling_strategy = "last"
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False):
        self.forward_calls += 1
        bsz, seq_len = input_ids.shape
        logits = torch.zeros((bsz, seq_len, self.num_labels), device=input_ids.device)
        logits[:, :, self.tag2id["O"]] = 5.0
        for b in range(bsz):
            for l in range(seq_len):
                piece = self.tokenizer.decode_id(int(input_ids[b, l].item()))
                base_tok = piece.split("::part")[0] if "::part" in piece else piece
                tag = self.token_tag_rules.get(base_tok, "O")
                if tag in self.tag2id:
                    logits[b, l, self.tag2id["O"]] = 0.0
                    logits[b, l, self.tag2id[tag]] = 10.0
        return {"logits": logits, "loss": None}


class FakeWordSegmenter:
    """Bộ tách từ giả lập không cần JVM để kiểm thử word-level."""

    def __init__(self, compounds: Dict[str, str]):
        self.compounds = compounds

    def segment_text(self, text: str) -> List[str]:
        s = text
        for src, dst in self.compounds.items():
            s = s.replace(src, dst)
        return _tokenize_syllables(s)


class TestPLMPredictorUnit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_predictor_"))
        self.tags = TAG_SETS["vlsp"]
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.id2tag = {i: t for i, t in enumerate(self.tags)}
        self.tokenizer = DeterministicTokenizer()
        self.rules = {
            "Nguyễn_Văn_A": "B-PER",
            "Hà_Nội": "B-LOC",
            "Đà_Nẵng": "B-LOC",
            "TP.HCM": "B-LOC",
            "Bộ_Y_tế": "B-ORG",
        }
        self.mock_model = RuleBasedMockNERModel(self.tokenizer, self.rules, self.tag2id)
        self.fake_seg = FakeWordSegmenter({
            "Nguyễn Văn A": "Nguyễn_Văn_A",
            "Hà Nội": "Hà_Nội",
            "Đà Nẵng": "Đà_Nẵng",
            "Bộ Y tế": "Bộ_Y_tế",
        })
        self.predictor = PLMPredictor.from_pretrained_components(
            model=self.mock_model,
            tokenizer=self.tokenizer,
            id2tag=self.id2tag,
            dataset="vlsp",
            granularity="word",
            device="cpu",
            segmenter=self.fake_seg,
            max_seq_len=32,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_sentence_with_no_entities(self):
        """Câu không có thực thể trả về danh sách rỗng []."""
        ents = self.predictor.predict("Hôm nay trời rất đẹp và mát mẻ.")
        self.assertEqual(ents, [])

    def test_02_sentence_with_single_entity(self):
        """Câu có đúng 1 thực thể trả về đúng schema {text, type, start, end}."""
        sent = "Tôi đang sinh sống tại Hà Nội."
        ents = self.predictor.predict(sent)
        self.assertEqual(len(ents), 1)
        e = ents[0]
        self.assertEqual(set(e.keys()), {"text", "type", "start", "end"})
        self.assertEqual(e["text"], "Hà Nội")
        self.assertEqual(e["type"], "LOC")
        self.assertEqual(sent[e["start"] : e["end"]], "Hà Nội")

    def test_03_sentence_with_multiple_and_repeated_entities(self):
        """Câu nhiều thực thể và thực thể lặp lại 2 lần có 2 vị trí (start, end) riêng biệt."""
        sent = "Ông Nguyễn Văn A công tác tại Bộ Y tế ở Hà Nội rồi bay vào Đà Nẵng và trở về Hà Nội."
        ents = self.predictor.predict(sent)
        self.assertEqual([(e["text"], e["type"]) for e in ents], [
            ("Nguyễn Văn A", "PER"),
            ("Bộ Y tế", "ORG"),
            ("Hà Nội", "LOC"),
            ("Đà Nẵng", "LOC"),
            ("Hà Nội", "LOC"),
        ])
        for e in ents:
            self.assertEqual(sent[e["start"] : e["end"]], e["text"])
        # Hai thực thể 'Hà Nội' bắt buộc có start khác nhau
        self.assertNotEqual(ents[2]["start"], ents[4]["start"])

    def test_04_punctuation_and_vietnamese_diacritics(self):
        """Xử lý đúng dấu câu dính liền thực thể và dấu tiếng Việt UTF-8."""
        sent = 'Chiều nay, ông Nguyễn Văn A (Bộ Y tế) từ "Hà Nội" đến TP.HCM!'
        ents = self.predictor.predict(sent)
        self.assertEqual([(e["text"], e["type"]) for e in ents], [
            ("Nguyễn Văn A", "PER"),
            ("Bộ Y tế", "ORG"),
            ("Hà Nội", "LOC"),
            ("TP.HCM", "LOC"),
        ])
        for e in ents:
            self.assertEqual(sent[e["start"] : e["end"]], e["text"])

    def test_05_long_sentence_exceeding_max_seq_len(self):
        """Câu rất dài vượt quá max_seq_len=32 không crash và vẫn nhận diện được thực thể ở cuối câu."""
        filler = " ".join(["hôm", "nay", "thời", "tiết", "rất", "đẹp"] * 8)  # 48 từ (> max_seq_len=32)
        sent = f"Ông Nguyễn Văn A nói rằng {filler} tại Hà Nội và Đà Nẵng."
        ents = self.predictor.predict(sent)
        found = [(e["text"], e["type"]) for e in ents]
        self.assertIn(("Nguyễn Văn A", "PER"), found)
        self.assertIn(("Hà Nội", "LOC"), found)
        self.assertIn(("Đà Nẵng", "LOC"), found)
        for e in ents:
            self.assertEqual(sent[e["start"] : e["end"]], e["text"])

    def test_06_empty_and_invalid_inputs_do_not_crash_demo(self):
        """Input rỗng, khoảng trắng hoặc sai kiểu không làm sập demo."""
        self.assertEqual(self.predictor.predict(""), [])
        self.assertEqual(self.predictor.predict("   \n\t  "), [])
        self.assertEqual(self.predictor.predict(None), [])     # type: ignore[arg-type]
        self.assertEqual(self.predictor.predict(12345), [])    # type: ignore[arg-type]
        self.assertEqual(self.predictor.predict(["Hà"]), [])   # type: ignore[arg-type]

        # Chế độ strict=True phải báo TypeError rõ ràng
        with self.assertRaises(TypeError):
            self.predictor.predict(12345, strict=True)  # type: ignore[arg-type]

    def test_07_error_handling_unloaded_and_corrupted_checkpoints(self):
        """Kiểm tra lỗi model chưa nạp, checkpoint thiếu, checkpoint hỏng, checkpoint lệch schema."""
        # 1. Model chưa load
        empty_pred = PLMPredictor(checkpoint_path=None, device="cpu")
        self.assertFalse(empty_pred.is_loaded)
        with self.assertRaises(RuntimeError):
            empty_pred.predict("Hà Nội")

        # 2. Checkpoint không tồn tại
        with self.assertRaises(FileNotFoundError):
            PLMPredictor(checkpoint_path=self.temp_dir / "missing.pt", device="cpu")

        # 3. Checkpoint hỏng (file rác không phải pickle/pt hợp lệ)
        corrupt = self.temp_dir / "corrupt.pt"
        corrupt.write_bytes(b"not a valid pytorch checkpoint")
        with self.assertRaises(ValueError):
            PLMPredictor(checkpoint_path=corrupt, device="cpu")

        # 4. Checkpoint thiếu trường hoặc lệch bộ nhãn schema
        bad_schema = self.temp_dir / "bad_schema.pt"
        torch.save({
            "config": {"data": {"dataset": "vlsp", "granularity": "word"}, "model": {"pooling_strategy": "last"}},
            "id2tag": {0: "O", 1: "B-FAKE_TAG"},
            "model_state_dict": {"classifier.weight": torch.zeros(2, 768)},
        }, bad_schema)
        with self.assertRaises(ValueError):
            PLMPredictor(checkpoint_path=bad_schema, device="cpu")

        # 5. Checkpoint lệch số nhãn giữa classifier.weight và id2tag
        bad_dim = self.temp_dir / "bad_dim.pt"
        torch.save({
            "config": {"data": {"dataset": "vlsp", "granularity": "word"}, "model": {"pooling_strategy": "last"}},
            "id2tag": {0: "O", 1: "B-PER"},
            "model_state_dict": {"classifier.weight": torch.zeros(9, 768)},
        }, bad_dim)
        with self.assertRaises(ValueError):
            PLMPredictor(checkpoint_path=bad_dim, device="cpu")

    def test_08_cuda_unavailable_fallback_and_eval_no_grad(self):
        """Khi yêu cầu CUDA nhưng máy không có CUDA -> tự fallback CPU; inference không tạo grad."""
        with patch("torch.cuda.is_available", return_value=False):
            dev = _resolve_device("cuda")
            self.assertEqual(dev.type, "cpu")
            dev_auto = _resolve_device("auto")
            self.assertEqual(dev_auto.type, "cpu")

        # Kiểm tra hiệu năng: model luôn eval(), không giữ grad, không tạo lại model giữa các lần predict
        model_id_before = id(self.predictor.model)
        _ = self.predictor.predict("Tôi đi Hà Nội.")
        _ = self.predictor.predict("Ông Nguyễn Văn A.")
        self.assertEqual(id(self.predictor.model), model_id_before)
        self.assertFalse(self.predictor.model.training)
        self.assertIsNone(self.mock_model.dummy_param.grad)

    def test_09_predict_sample_matches_schema_contract(self):
        """predict_sample trả về NERSample hợp lệ theo src/common/schema.py."""
        sample = NERSample(
            id="vlsp_word_test_00001",
            dataset="vlsp",
            granularity="word",
            tokens=["Ông", "Nguyễn_Văn_A", "ở", "Hà_Nội", "."],
            tags=["O", "O", "O", "O", "O"],
        )
        pred_sample = self.predictor.predict_sample(sample)
        self.assertEqual(pred_sample.id, sample.id)
        self.assertEqual(pred_sample.tags, ["O", "B-PER", "O", "B-LOC", "O"])


class TestPLMPredictorRealCheckpoints(unittest.TestCase):
    """Kiểm thử tích hợp trên checkpoint thực đã huấn luyện (PhoBERT & XLM-R, CPU và GPU nếu có)."""

    PHOBERT_CKPT = Path(PRESET_CHECKPOINTS["phobert_vlsp_best"])
    XLMR_CKPT = Path(PRESET_CHECKPOINTS["xlmr_vlsp_syllable"])

    def _run_standard_checks(self, predictor: PLMPredictor, model_label: str):
        # 1. Input rỗng
        self.assertEqual(predictor.predict(""), [])
        self.assertEqual(predictor.predict("   "), [])

        # 2. Câu không có thực thể
        ents_none = predictor.predict("Hôm nay trời mưa rất to nên mọi người ở nhà.")
        self.assertIsInstance(ents_none, list)

        # 3. Câu có nhiều thực thể + dấu tiếng Việt + dấu câu
        sent_multi = "Ban biên tập báo Tuổi Trẻ cử nữ phóng viên Uyên Ly sang Mỹ công tác tại Hà Nội."
        ents_multi = predictor.predict(sent_multi)
        self.assertGreaterEqual(len(ents_multi), 2, f"{model_label} phải tìm được >=2 thực thể: {ents_multi}")
        for e in ents_multi:
            self.assertEqual(set(e.keys()), {"text", "type", "start", "end"})
            self.assertIn(e["type"], {"PER", "LOC", "ORG", "MISC"})
            self.assertEqual(sent_multi[e["start"] : e["end"]], e["text"])

        # 4. Câu dài (>260 tokens) không crash
        long_sent = " ".join(["Hôm nay trời nắng đẹp và trong xanh."] * 45) + " Ông Nguyễn Văn An đến Hà Nội."
        ents_long = predictor.predict(long_sent)
        self.assertIsInstance(ents_long, list)
        self.assertTrue(any(e["type"] in ("PER", "LOC") for e in ents_long), f"{model_label} bỏ sót thực thể ở cuối câu dài!")

    @unittest.skipUnless(PHOBERT_CKPT.exists(), "Chưa có checkpoint PhoBERT cục bộ")
    def test_10_real_phobert_cpu_and_gpu(self):
        """Kiểm thử PhoBERT (word-level, concat_4) trên CPU và GPU (nếu có)."""
        fake_seg = FakeWordSegmenter({
            "biên tập": "biên_tập",
            "Tuổi Trẻ": "Tuổi_Trẻ",
            "phóng viên": "phóng_viên",
            "công tác": "công_tác",
            "Hà Nội": "Hà_Nội",
            "Hôm nay": "Hôm_nay",
            "mọi người": "mọi_người",
            "trong xanh": "trong_xanh",
        })
        # CPU
        pred_cpu = create_predictor(self.PHOBERT_CKPT, device="cpu", segmenter=fake_seg)
        self.assertEqual(pred_cpu.device.type, "cpu")
        self._run_standard_checks(pred_cpu, "PhoBERT-CPU")

        # GPU (nếu có CUDA)
        if torch.cuda.is_available():
            pred_gpu = create_predictor(self.PHOBERT_CKPT, device="cuda", segmenter=fake_seg)
            self.assertEqual(pred_gpu.device.type, "cuda")
            self._run_standard_checks(pred_gpu, "PhoBERT-GPU")
            # Kết quả trên CPU và GPU phải nhất quán trên cùng câu
            sent = "Phóng viên Uyên Ly làm việc tại Hà Nội."
            self.assertEqual(pred_cpu.predict(sent), pred_gpu.predict(sent))

    @unittest.skipUnless(XLMR_CKPT.exists(), "Chưa có checkpoint XLM-R cục bộ")
    def test_11_real_xlmr_cpu_and_gpu(self):
        """Kiểm thử XLM-RoBERTa (syllable-level) trên CPU và GPU (nếu có)."""
        pred_cpu = load_predictor("xlmr_vlsp_syllable", device="cpu")
        self.assertEqual(pred_cpu.device.type, "cpu")
        self._run_standard_checks(pred_cpu, "XLM-R-CPU")

        if torch.cuda.is_available():
            pred_gpu = load_predictor("xlmr_vlsp_syllable", device="cuda")
            self.assertEqual(pred_gpu.device.type, "cuda")
            self._run_standard_checks(pred_gpu, "XLM-R-GPU")
            sent = "Phóng viên Uyên Ly làm việc tại Hà Nội."
            self.assertEqual(pred_cpu.predict(sent), pred_gpu.predict(sent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
