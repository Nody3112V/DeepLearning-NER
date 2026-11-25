"""
tests_bilstm.py - Bộ unit test cho nhánh 1 (BiLSTM-CNN-CRF - Thành viên 1).

Kiểm tra:
  1. Vocabulary: xây từ điển word/char/tag từ JSONL, giữ nguyên viết hoa, <PAD>/<UNK>, save/load.
  2. NERDataset & collate_ner_batch: đệm động theo lô, mask nhị phân, lengths, char_ids, max_seq_len=None.
  3. BiLSTMCRF (Baseline): kích thước emissions, Masked CrossEntropyLoss bỏ qua padding,
     lan truyền ngược gradient (padding_idx không đổi), và giải mã decode đúng độ dài từng câu.
  4. CharEncoder (CharCNN & CharLSTM): kích thước đầu ra, mask token đệm bằng 0, lan truyền gradient.
  5. CRF: đối chiếu Forward (log-partition) và Viterbi với vét cạn trên câu ngắn,
     bất biến với padding, NLL >= 0, và tích hợp toàn bộ BiLSTM-Char-CRF.
"""
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.models.bilstm_crf.char_encoder import CharCNN, CharLSTM, build_char_encoder
from src.models.bilstm_crf.crf import CRF
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


class TestCharEncoderAndCRF(unittest.TestCase):
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

    def test_char_cnn_and_char_lstm_shapes_and_padding_mask(self) -> None:
        torch.manual_seed(42)
        loader = create_dataloader(self.jsonl_path, self.vocab, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        char_ids = batch["char_ids"]  # (2, 8, W)

        for enc_type, expected_dim in (("cnn", 16), ("lstm", 32)):
            cfg = {
                "char_encoder_type": enc_type,
                "char_emb_dim": 12,
                "char_hidden_dim": 16,
                "char_kernel_size": 3,
                "dropout": 0.1,
            }
            enc, out_dim = build_char_encoder(cfg, len(self.vocab.char2idx))
            self.assertEqual(out_dim, expected_dim)
            self.assertIsInstance(enc, CharCNN if enc_type == "cnn" else CharLSTM)

            enc.eval()
            out = enc(char_ids)
            self.assertEqual(tuple(out.shape), (2, 8, expected_dim))

            # Câu 0 có 4 token thật -> vị trí 4..7 là <PAD>, đầu ra phải bằng 0 tuyệt đối
            self.assertEqual(out[0, 4:].abs().sum().item(), 0.0)

            # Kiểm tra gradient lan truyền và PAD_IDX giữ grad = 0
            enc.train()
            loss = enc(char_ids).sum()
            loss.backward()
            self.assertIsNotNone(enc.char_embedding.weight.grad)
            self.assertEqual(enc.char_embedding.weight.grad[PAD_IDX].abs().sum().item(), 0.0)

    def test_crf_forward_and_viterbi_match_brute_force(self) -> None:
        torch.manual_seed(7)
        num_tags = 4
        crf = CRF(num_tags=num_tags)
        with torch.no_grad():
            crf.start_transitions.copy_(torch.randn(num_tags))
            crf.end_transitions.copy_(torch.randn(num_tags))
            crf.transitions.copy_(torch.randn(num_tags, num_tags))

        # Batch 3 câu với độ dài thật khác nhau: 1, 3, 4 (max_len = 4)
        lengths = [1, 3, 4]
        batch_size, max_len = len(lengths), max(lengths)
        emissions = torch.randn(batch_size, max_len, num_tags)
        tags = torch.randint(0, num_tags, (batch_size, max_len))
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for b, L in enumerate(lengths):
            mask[b, :L] = True

        # Tính vét cạn log Z(x) và chuỗi tối ưu cho từng câu
        brute_log_z = []
        brute_best_paths = []
        brute_gold_scores = []

        def seq_score(b_idx: int, seq: tuple) -> float:
            s = crf.start_transitions[seq[0]].item() + emissions[b_idx, 0, seq[0]].item()
            for t in range(1, len(seq)):
                s += crf.transitions[seq[t - 1], seq[t]].item() + emissions[b_idx, t, seq[t]].item()
            s += crf.end_transitions[seq[-1]].item()
            return s

        for b, L in enumerate(lengths):
            all_scores = []
            best_s, best_seq = -float("inf"), None
            for seq in itertools.product(range(num_tags), repeat=L):
                sc = seq_score(b, seq)
                all_scores.append(sc)
                if sc > best_s:
                    best_s, best_seq = sc, list(seq)
            m = max(all_scores)
            log_z = m + math.log(sum(math.exp(sc - m) for sc in all_scores))
            brute_log_z.append(log_z)
            brute_best_paths.append(best_seq)
            brute_gold_scores.append(seq_score(b, tuple(tags[b, :L].tolist())))

        # So sánh CRF với vét cạn
        nll_per_sent = crf(emissions, tags, mask, reduction="none")
        for b in range(batch_size):
            expected_nll = brute_log_z[b] - brute_gold_scores[b]
            self.assertAlmostEqual(nll_per_sent[b].item(), expected_nll, places=5)
            self.assertGreaterEqual(nll_per_sent[b].item(), -1e-6)

        decoded = crf.decode(emissions, mask)
        self.assertEqual(decoded, brute_best_paths)

    def test_crf_mask_ignores_padding(self) -> None:
        torch.manual_seed(11)
        num_tags = 5
        crf = CRF(num_tags=num_tags)
        emissions = torch.randn(2, 6, num_tags)
        tags = torch.randint(0, num_tags, (2, 6))
        mask = torch.tensor(
            [
                [True, True, False, False, False, False],
                [True, True, True, True, True, False],
            ]
        )

        loss_orig = crf(emissions, tags, mask)
        pred_orig = crf.decode(emissions, mask)

        # Thay đổi mạnh emissions và tags ở các vị trí padding (~mask)
        mod_emissions = emissions.clone()
        mod_tags = tags.clone()
        mod_emissions[~mask] = 999.0
        mod_tags[~mask] = num_tags - 1

        loss_mod = crf(mod_emissions, mod_tags, mask)
        pred_mod = crf.decode(mod_emissions, mask)

        self.assertAlmostEqual(loss_orig.item(), loss_mod.item(), places=6)
        self.assertEqual(pred_orig, pred_mod)

    def test_full_bilstm_char_crf_end_to_end(self) -> None:
        torch.manual_seed(42)
        loader = create_dataloader(self.jsonl_path, self.vocab, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        for enc_type in ("cnn", "lstm"):
            char_enc, char_out_dim = build_char_encoder(
                {
                    "char_encoder_type": enc_type,
                    "char_emb_dim": 12,
                    "char_hidden_dim": 16,
                    "char_kernel_size": 3,
                    "dropout": 0.1,
                },
                len(self.vocab.char2idx),
            )
            crf = CRF(num_tags=len(self.vocab.tag2idx))
            model = BiLSTMCRF(
                vocab_size=len(self.vocab.word2idx),
                num_tags=len(self.vocab.tag2idx),
                word_emb_dim=32,
                bilstm_hidden_dim=64,
                bilstm_num_layers=1,
                dropout=0.1,
                char_encoder=char_enc,
                char_out_dim=char_out_dim,
                crf=crf,
            )

            model.train()
            loss = model.compute_loss(
                batch["word_ids"],
                batch["tag_ids"],
                batch["mask"],
                batch["lengths"],
                char_ids=batch["char_ids"],
            )
            self.assertGreaterEqual(loss.item(), 0.0)
            loss.backward()
            self.assertIsNotNone(model.word_embedding.weight.grad)
            self.assertIsNotNone(model.char_encoder.char_embedding.weight.grad)
            self.assertIsNotNone(model.crf.transitions.grad)
            self.assertIsNotNone(model.crf.start_transitions.grad)
            self.assertIsNotNone(model.crf.end_transitions.grad)

            model.eval()
            preds = model.decode(
                batch["word_ids"],
                batch["mask"],
                batch["lengths"],
                char_ids=batch["char_ids"],
            )
            self.assertEqual([len(p) for p in preds], [4, 8])


if __name__ == "__main__":
    unittest.main()
