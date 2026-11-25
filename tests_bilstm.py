"""
tests_bilstm.py - Bộ unit test cho nhánh 1 (BiLSTM Baseline - Thành viên 1).

Kiểm tra:
  1. Vocabulary: xây từ điển word/char/tag từ JSONL, giữ nguyên viết hoa, <PAD>/<UNK>, save/load.
  2. NERDataset & collate_ner_batch: đệm động theo lô, mask nhị phân, lengths, char_ids.
  3. BiLSTMCRF (Baseline): kích thước emissions, Masked CrossEntropyLoss bỏ qua padding,
     lan truyền ngược gradient (padding_idx không đổi), và giải mã decode đúng độ dài từng câu.
"""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.bilstm_crf.dataset import create_dataloader
from src.models.bilstm_crf.model import BiLSTMCRF
from src.models.bilstm_crf.vocab import PAD_IDX, UNK_IDX, Vocabulary


SAMPLE_RECORDS = [
    {
        "id": "phoner_word_train_00001",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "tokens": ["Bệnh_nhân", "18", "tại", "Hà_Nội"],
        "tags": ["O", "B-PATIENT_ID", "O", "B-LOCATION"],
    },
    {
        "id": "phoner_word_train_00002",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "tokens": ["Bệnh_nhân", "nam", "25", "tuổi", "điều_trị", "tại", "Bệnh_viện", "Nhiệt_đới"],
        "tags": ["O", "B-GENDER", "B-AGE", "O", "O", "O", "B-LOCATION", "I-LOCATION"],
    },
]


class TestBiLSTMBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.jsonl_path = str(Path(self.tmpdir.name) / "sample.jsonl")
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for rec in SAMPLE_RECORDS:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.vocab = Vocabulary()
        self.vocab.build_from_jsonl(self.jsonl_path, min_word_freq=1)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_vocabulary_build_encode_and_roundtrip(self) -> None:
        self.assertEqual(self.vocab.word2idx["<PAD>"], PAD_IDX)
        self.assertEqual(self.vocab.word2idx["<UNK>"], UNK_IDX)
        self.assertIn("Hà_Nội", self.vocab.word2idx)
        self.assertNotIn("hà_nội", self.vocab.word2idx)  # Giữ nguyên hoa/thường

        # Encode từ đã biết và OOV
        encoded = self.vocab.encode_words(["Hà_Nội", "Từ_Ngoài_Từ_Điển"])
        self.assertEqual(encoded[1], UNK_IDX)
        self.assertNotEqual(encoded[0], UNK_IDX)

        # Round-trip encode/decode tags
        tags = ["O", "B-LOCATION", "I-LOCATION"]
        self.assertEqual(self.vocab.decode_tags(self.vocab.encode_tags(tags)), tags)

        # Round-trip save/load
        vocab_file = str(Path(self.tmpdir.name) / "vocab.json")
        self.vocab.save(vocab_file)
        loaded = Vocabulary.load(vocab_file)
        self.assertEqual(loaded.word2idx, self.vocab.word2idx)
        self.assertEqual(loaded.char2idx, self.vocab.char2idx)
        self.assertEqual(loaded.tag2idx, self.vocab.tag2idx)
        self.assertEqual(loaded.idx2tag, self.vocab.idx2tag)

    def test_dataloader_dynamic_padding_and_mask(self) -> None:
        loader = create_dataloader(
            self.jsonl_path, self.vocab, batch_size=2, shuffle=False, max_seq_len=32, max_word_len=15
        )
        batch = next(iter(loader))

        # Câu 1 dài 4, câu 2 dài 8 -> max_seq_len trong batch phải là 8
        self.assertEqual(tuple(batch["word_ids"].shape), (2, 8))
        self.assertEqual(tuple(batch["tag_ids"].shape), (2, 8))
        self.assertEqual(tuple(batch["mask"].shape), (2, 8))
        self.assertEqual(batch["lengths"].tolist(), [4, 8])

        # Kiểm tra mask và padding của câu ngắn (câu 0)
        self.assertTrue(batch["mask"][0, :4].all().item())
        self.assertFalse(batch["mask"][0, 4:].any().item())
        self.assertTrue((batch["word_ids"][0, 4:] == PAD_IDX).all().item())

        # Kiểm tra char_ids có chiều (B, T, W)
        self.assertEqual(batch["char_ids"].ndim, 3)
        self.assertEqual(batch["char_ids"].shape[:2], (2, 8))

    def test_bilstm_forward_masked_loss_backward_and_decode(self) -> None:
        torch.manual_seed(42)
        loader = create_dataloader(self.jsonl_path, self.vocab, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = BiLSTMCRF(
            vocab_size=len(self.vocab.word2idx),
            num_tags=len(self.vocab.tag2idx),
            word_emb_dim=32,
            bilstm_hidden_dim=64,
            bilstm_num_layers=1,
            dropout=0.1,
        )

        # 1. Shape emissions
        emissions = model.get_emissions(batch["word_ids"], batch["lengths"])
        self.assertEqual(tuple(emissions.shape), (2, 8, len(self.vocab.tag2idx)))

        # 2. Masked CrossEntropyLoss phải bỏ qua hoàn toàn các vị trí padding (~mask)
        model.eval()
        loss_orig = model.compute_loss(batch["word_ids"], batch["tag_ids"], batch["mask"], batch["lengths"])
        modified_tags = batch["tag_ids"].clone()
        modified_tags[0, 4:] = len(self.vocab.tag2idx) - 1  # Đổi nhãn tại vị trí padding
        loss_mod = model.compute_loss(batch["word_ids"], modified_tags, batch["mask"], batch["lengths"])
        self.assertAlmostEqual(loss_orig.item(), loss_mod.item(), places=6)

        # 3. Backward pass: gradient lan truyền tốt, padding_idx=0 giữ grad bằng 0
        model.train()
        loss = model.compute_loss(batch["word_ids"], batch["tag_ids"], batch["mask"], batch["lengths"])
        loss.backward()
        self.assertIsNotNone(model.hidden2tag.weight.grad)
        self.assertIsNotNone(model.word_embedding.weight.grad)
        self.assertEqual(model.word_embedding.weight.grad[PAD_IDX].abs().sum().item(), 0.0)

        # 4. Decode trả về đúng độ dài gốc (loại bỏ padding)
        model.eval()
        decoded = model.decode(batch["word_ids"], batch["mask"], batch["lengths"])
        self.assertEqual([len(seq) for seq in decoded], [4, 8])

    def test_dataloader_no_truncation_when_max_seq_len_none(self) -> None:
        loader_full = create_dataloader(
            self.jsonl_path, self.vocab, batch_size=2, shuffle=False, max_seq_len=None
        )
        batch_full = next(iter(loader_full))
        self.assertEqual(batch_full["lengths"].tolist(), [4, 8])

        loader_trunc = create_dataloader(
            self.jsonl_path, self.vocab, batch_size=2, shuffle=False, max_seq_len=3
        )
        batch_trunc = next(iter(loader_trunc))
        self.assertEqual(batch_trunc["lengths"].tolist(), [3, 3])

    def test_odd_hidden_dim_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BiLSTMCRF(vocab_size=10, num_tags=5, bilstm_hidden_dim=65)


if __name__ == "__main__":
    unittest.main()
