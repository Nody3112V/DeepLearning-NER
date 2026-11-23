"""
tests_subword_align.py - Bộ kiểm định toàn diện cho module subword_align.py.

Bao gồm 10 bài kiểm thử kỹ thuật:
  1. Một từ tương ứng đúng một subword.
  2. Một từ tương ứng nhiều subwords (subword đầu nhận nhãn, các subwords sau nhận -100).
  3. Special token đầu và cuối (<s>, </s>, [CLS], [SEP]) nhận -100.
  4. Padding nhận -100.
  5. Câu dài bị truncation (cắt ngắn chính xác, len(labels) == len(input_ids)).
  6. Batch nhiều câu có độ dài khác nhau với dynamic padding (tensor shape [B, L]).
  7. Nhãn BIO đa dạng (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, B-MISC).
  8. Nhãn không tồn tại trong label2id (ném ngoại lệ KeyError).
  9. Mismatch giữa số từ và số nhãn (ném ngoại lệ ValueError).
  10. Tương thích với Tokenizer không hỗ trợ word_ids (Slow Tokenizer: Mock & PhobertTokenizer).
"""
import unittest
from typing import List, Optional, Dict, Any
import torch

from src.models.pretrained.subword_align import (
    align_labels_with_subwords,
    tokenize_and_get_word_ids,
    align_tokens_and_labels,
    align_batch_tokens_and_labels,
    create_ner_collate_fn,
)


class MockFastTokenizer:
    """Mock Fast Tokenizer mô phỏng XLM-RoBERTa (hỗ trợ word_ids())."""
    is_fast = True
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {
            "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
            "Hà": 10, "Nội": 11, "đẹp": 12, "lắm": 13,
            "Việt": 14, "_": 15, "Nam": 16,
        }

    def __call__(
        self,
        tokens: List[str],
        is_split_into_words: bool = True,
        max_length: int = 256,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ):
        input_ids = [self.bos_token_id]
        word_ids: List[Optional[int]] = [None]

        for idx, tok in enumerate(tokens):
            if tok == "Việt_Nam":
                # Mô phỏng 1 từ bị tách thành 3 subwords: 'Việt', '_', 'Nam'
                sub_ids = [14, 15, 16]
            elif tok in self.vocab:
                sub_ids = [self.vocab[tok]]
            else:
                sub_ids = [3]  # unk

            for sid in sub_ids:
                if len(input_ids) >= max_length - 1:
                    break
                input_ids.append(sid)
                word_ids.append(idx)
            if len(input_ids) >= max_length - 1:
                break

        if len(input_ids) < max_length:
            input_ids.append(self.eos_token_id)
            word_ids.append(None)

        attention_mask = [1] * len(input_ids)

        class EncodingMock:
            def __init__(self, iids, amask, wids):
                self._data = {"input_ids": iids, "attention_mask": amask}
                self._word_ids = wids

            def __getitem__(self, item):
                return self._data[item]

            def word_ids(self):
                return self._word_ids

        return EncodingMock(input_ids, attention_mask, word_ids)


class MockSlowTokenizer:
    """Mock Slow Tokenizer mô phỏng PhoBERT (is_fast=False, không có word_ids())."""
    is_fast = False
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2
    unk_token_id = 3

    def encode(self, token: str, add_special_tokens: bool = False) -> List[int]:
        if token == "Việt_Nam":
            # 1 từ ghép thành 2 subwords
            return [101, 102]
        elif token == "siêu_dài_nhiều_từ":
            return [201, 202, 203, 204]
        return [50]


class TestSubwordAlign(unittest.TestCase):

    def setUp(self):
        self.label2id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-LOC": 3,
            "I-LOC": 4,
            "B-ORG": 5,
            "I-ORG": 6,
            "B-MISC": 7,
            "I-MISC": 8,
        }
        self.fast_tok = MockFastTokenizer()
        self.slow_tok = MockSlowTokenizer()

    def test_01_single_word_single_subword(self):
        """1. Một từ tương ứng đúng một subword."""
        word_labels = ["B-LOC", "O"]
        word_ids = [None, 0, 1, None]  # <s>, 'Hà', 'đẹp', </s>
        aligned = align_labels_with_subwords(word_labels, word_ids, self.label2id)

        self.assertEqual(aligned, [-100, 3, 0, -100])

    def test_02_single_word_multiple_subwords(self):
        """2. Một từ tương ứng nhiều subwords: subword đầu nhận nhãn, các subwords sau nhận -100."""
        # Từ #0 ('Việt_Nam') chia thành 3 subwords: wid=0, 0, 0
        word_labels = ["B-LOC", "O"]
        word_ids = [None, 0, 0, 0, 1, None]
        aligned = align_labels_with_subwords(word_labels, word_ids, self.label2id)

        # wid=0 nhận 3 tại vị trí đầu, -100 tại 2 vị trí sau
        self.assertEqual(aligned, [-100, 3, -100, -100, 0, -100])

    def test_03_special_tokens_receive_ignore_index(self):
        """3. Special tokens đầu và cuối (<s>, </s>, [CLS], [SEP]) nhận -100."""
        word_labels = ["O"]
        word_ids = [None, None, 0, None, None]
        aligned = align_labels_with_subwords(word_labels, word_ids, self.label2id)

        self.assertEqual(aligned[0], -100)
        self.assertEqual(aligned[1], -100)
        self.assertEqual(aligned[3], -100)
        self.assertEqual(aligned[4], -100)
        self.assertEqual(aligned[2], 0)

    def test_04_padding_receives_ignore_index(self):
        """4. Dynamic Padding nhận nhãn -100."""
        batch_items = [
            {"id": "c1", "input_ids": [0, 10, 2], "attention_mask": [1, 1, 1], "labels": [-100, 3, -100], "tokens": ["a"], "tags": ["B-LOC"]},
            {"id": "c2", "input_ids": [0, 10, 11, 12, 2], "attention_mask": [1, 1, 1, 1, 1], "labels": [-100, 3, -100, 0, -100], "tokens": ["a", "b"], "tags": ["B-LOC", "O"]},
        ]
        collate_fn = create_ner_collate_fn(pad_token_id=1, ignore_index=-100)
        batch = collate_fn(batch_items)

        # Shape [2, 5]
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["attention_mask"].shape, (2, 5))
        self.assertEqual(batch["labels"].shape, (2, 5))

        # Câu 1 được pad 2 vị trí cuối: input_ids=1, mask=0, labels=-100
        self.assertEqual(batch["input_ids"][0, 3:].tolist(), [1, 1])
        self.assertEqual(batch["attention_mask"][0, 3:].tolist(), [0, 0])
        self.assertEqual(batch["labels"][0, 3:].tolist(), [-100, -100])

    def test_05_truncation_handling(self):
        """5. Câu dài bị truncation: cắt ngắn chính xác, len(labels) == len(input_ids)."""
        tokens = ["từ_" + str(i) for i in range(50)]
        tags = ["O"] * 50
        max_len = 16

        res = align_tokens_and_labels(
            tokens=tokens,
            tags=tags,
            tokenizer=self.fast_tok,
            label2id=self.label2id,
            max_length=max_len,
        )

        self.assertEqual(len(res["input_ids"]), max_len)
        self.assertEqual(len(res["attention_mask"]), max_len)
        self.assertEqual(len(res["labels"]), max_len)
        self.assertEqual(len(res["word_ids"]), max_len)

    def test_06_batch_alignment_and_shapes(self):
        """6. Batch nhiều câu có độ dài khác nhau (Tensor shape [B, L])."""
        batch_tokens = [
            ["Hà", "Nội"],
            ["Việt_Nam", "đẹp", "lắm"],
        ]
        batch_tags = [
            ["B-LOC", "I-LOC"],
            ["B-LOC", "O", "O"],
        ]

        batch = align_batch_tokens_and_labels(
            batch_tokens=batch_tokens,
            batch_tags=batch_tags,
            tokenizer=self.fast_tok,
            label2id=self.label2id,
            max_length=32,
            padding=True,
        )

        self.assertIsInstance(batch["input_ids"], torch.Tensor)
        self.assertIsInstance(batch["attention_mask"], torch.Tensor)
        self.assertIsInstance(batch["labels"], torch.Tensor)

        B, L = batch["input_ids"].shape
        self.assertEqual(B, 2)
        self.assertEqual(batch["attention_mask"].shape, (B, L))
        self.assertEqual(batch["labels"].shape, (B, L))

    def test_07_diverse_bio_tags(self):
        """7. Nhãn BIO đa dạng (B-PER, I-PER, B-ORG, B-MISC,...)."""
        tokens = ["Nguyễn", "Văn_A", "làm_tại", "FPT", "ở", "Việt_Nam"]
        tags = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"]

        res = align_tokens_and_labels(
            tokens=tokens,
            tags=tags,
            tokenizer=self.fast_tok,
            label2id=self.label2id,
        )

        # Kiểm tra nhãn tại các vị trí đầu tiên của từ
        seen_words = set()
        for wid, label_id in zip(res["word_ids"], res["labels"]):
            if wid is not None and wid not in seen_words:
                seen_words.add(wid)
                expected_tag = tags[wid]
                self.assertEqual(label_id, self.label2id[expected_tag])

    def test_08_unknown_label_raises_key_error(self):
        """8. Nhãn không tồn tại trong label2id ném ngoại lệ KeyError."""
        word_labels = ["B-UNKNOWN_TAG"]
        word_ids = [None, 0, None]

        with self.assertRaises(KeyError) as ctx:
            align_labels_with_subwords(word_labels, word_ids, self.label2id)
        self.assertIn("B-UNKNOWN_TAG", str(ctx.exception))

    def test_09_mismatch_tokens_tags_raises_value_error(self):
        """9. Mismatch giữa số từ và số nhãn ném ngoại lệ ValueError."""
        tokens = ["Hà", "Nội"]
        tags = ["B-LOC"]  # thiếu 1 nhãn

        with self.assertRaises(ValueError) as ctx:
            align_tokens_and_labels(
                tokens=tokens,
                tags=tags,
                tokenizer=self.fast_tok,
                label2id=self.label2id,
            )
        self.assertIn("Không khớp kích thước", str(ctx.exception))

    def test_10_slow_tokenizer_compatibility(self):
        """10. Tương thích với Slow Tokenizer (không có word_ids())."""
        tokens = ["Việt_Nam", "đẹp"]
        tags = ["B-LOC", "O"]

        # Chạy với MockSlowTokenizer (is_fast=False)
        res = align_tokens_and_labels(
            tokens=tokens,
            tags=tags,
            tokenizer=self.slow_tok,
            label2id=self.label2id,
            max_length=16,
        )

        # 'Việt_Nam' có 2 subwords [101, 102] -> wid=0, 0
        # 'đẹp' có 1 subword [50] -> wid=1
        # Kèm <s> (bos) và </s> (eos)
        # Tổng length = 1 + 2 + 1 + 1 = 5
        self.assertEqual(len(res["input_ids"]), 5)
        self.assertEqual(res["word_ids"], [None, 0, 0, 1, None])
        # Nhãn: [-100, 3 (B-LOC), -100, 0 (O), -100]
        self.assertEqual(res["labels"], [-100, 3, -100, 0, -100])


def run_unit_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSubwordAlign)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_unit_tests()
