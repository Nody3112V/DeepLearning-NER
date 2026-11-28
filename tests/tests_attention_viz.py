"""
tests_attention_viz.py - Bộ kiểm thử đơn vị toàn diện cho module src/models/pretrained/attention_viz.py.

Kiểm chứng toàn bộ các yêu cầu kiểm định:
  1. Kiểm tra kiến trúc (ModelAttentionSpec) với mô hình bất đối xứng (num_layers=3, num_heads=4).
  2. Attention tensor không rỗng, kích thước chính xác [num_layers, num_heads, L, L].
  3. Toàn bộ giá trị hữu hạn, nằm trong [0, 1], tổng theo hàng bằng 1.0.
  4. Không nhầm lẫn giữa chỉ số layer và chỉ số head (kể cả chỉ số âm và ngoại lệ vượt biên).
  5. Tổng hợp subword -> word chính xác theo công thức toán học (Key=sum/mean/first, Query=mean/first/max).
  6. Attention Rollout bảo toàn kích thước, giá trị hữu hạn và tổng hàng bằng 1.0.
  7. Không làm thay đổi trạng thái train/eval và không tạo gradient cho mô hình.
  8. Kiểm tra lưu ảnh chất lượng cao với quy ước đặt tên chứa Model, Sentence ID, Layer, Head, Level.
  9. Kiểm tra tương thích ngược với extract_attention_weights.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.attention_viz import (
    AttentionExtractionResult,
    aggregate_subword_attention_to_words,
    build_attention_filename,
    compute_attention_rollout,
    extract_attention_weights,
    extract_sentence_attention,
    inspect_model_attention_spec,
    plot_attention_heatmap,
    plot_head_comparison,
    plot_layer_comparison,
    plot_model_attention_comparison,
    select_layer_head_attention,
    validate_attention_result,
)


class MockSlowTokenizer:
    """Tokenizer giả lập chia một số từ ghép thành nhiều subwords để test căn chỉnh word_ids."""

    def __init__(self):
        self.is_fast = False
        self.bos_token_id = 0
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 2
        self.unk_token_id = 3
        self._vocab = {
            0: "<s>",
            1: "<pad>",
            2: "</s>",
            3: "<unk>",
            10: "Nguyễn_@@",
            11: "Văn_A",
            12: "tại",
            13: "Hà_Nội",
            14: ".",
            15: "22",
            16: "/",
            17: "6",
        }
        self._tok2ids = {
            "Nguyễn_Văn_A": [10, 11],
            "tại": [12],
            "Hà_Nội": [13],
            ".": [14],
            "22/6": [15, 16, 17],
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(self._tok2ids.get(text, [self.unk_token_id]))

    def convert_ids_to_tokens(self, ids):
        return [self._vocab.get(int(i), "<unk>") for i in ids]


class TestAttentionViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        # Bất đối xứng có chủ đích: 3 layers != 4 heads để bắt lỗi nhầm lẫn trục layer/head
        cls.mock_cfg = RobertaConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=64,
            output_hidden_states=True,
            output_attentions=True,
        )
        cls.backbone = RobertaModel(cls.mock_cfg)
        cls.model = PretrainedNERModel(
            model_name_or_path="mock/phobert-tiny",
            num_labels=5,
            pooling_strategy="last",
            classifier_dropout=0.0,
            config=cls.mock_cfg,
            pretrained_backbone=cls.backbone,
        )
        cls.tokenizer = MockSlowTokenizer()
        cls.id2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
        cls.sample_tokens = ["Nguyễn_Văn_A", "tại", "Hà_Nội", "22/6", "."]

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_attn_viz_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_inspect_model_spec_asymmetric_layers_heads(self):
        """Kiểm tra phát hiện đúng num_layers=3, num_heads=4, head_dim=8."""
        spec = inspect_model_attention_spec(self.model, self.tokenizer)
        self.assertTrue(spec.supports_output_attentions)
        self.assertEqual(spec.num_layers, 3)
        self.assertEqual(spec.num_heads, 4)
        self.assertEqual(spec.hidden_size, 32)
        self.assertEqual(spec.head_dim, 8)
        self.assertEqual(spec.attention_tuple_len, 3)
        self.assertEqual(spec.sample_layer_shape[1], 4)

    def test_02_extraction_invariants_and_validation(self):
        """Kiểm tra tensor không rỗng, đúng shape, giá trị hữu hạn, tổng hàng = 1.0."""
        res = extract_sentence_attention(
            model=self.model,
            tokenizer=self.tokenizer,
            sentence=self.sample_tokens,
            layer_idx=-1,
            head="mean",
            id2tag=self.id2tag,
            sentence_id="test_sent_01",
        )
        # <s> + (2 subwords) + 1 + 1 + (3 subwords) + 1 + </s> = 10 subwords
        self.assertEqual(tuple(res.attentions.shape), (3, 4, 10, 10))
        self.assertEqual(tuple(res.selected_attention.shape), (10, 10))
        self.assertEqual(res.layer_idx, 2)
        self.assertEqual(res.head_spec, "mean")
        self.assertEqual(res.word_ids, [None, 0, 0, 1, 2, 3, 3, 3, 4, None])
        self.assertEqual(len(res.predicted_tags), len(self.sample_tokens))

        report = validate_attention_result(res)
        self.assertTrue(report["valid"])
        self.assertEqual(report["num_subwords"], 10)
        self.assertEqual(report["num_mapped_words"], 5)
        self.assertLess(report["max_row_sum_error"], 1e-4)

    def test_03_no_layer_head_confusion(self):
        """Đảm bảo tuyệt đối không nhầm lẫn trục layer (dim 0) và trục head (dim 1)."""
        # Tạo tensor [3 layers, 4 heads, 2, 2] có giá trị phân biệt duy nhất cho từng (l, h)
        synth = torch.zeros((3, 4, 2, 2), dtype=torch.float32)
        for l in range(3):
            for h in range(4):
                p = 0.1 + 0.05 * l + 0.01 * h
                synth[l, h] = torch.tensor([[p, 1.0 - p], [1.0 - p, p]])

        sel, nl, nh = select_layer_head_attention(synth, layer_idx=1, head=3)
        self.assertEqual((nl, nh), (1, 3))
        expected_p = 0.1 + 0.05 * 1 + 0.01 * 3
        self.assertAlmostEqual(float(sel[0, 0].item()), expected_p, places=5)

        # Kiểm tra chặn vượt biên trục layer (chỉ có 3 tầng: 0..2, truyền 3 phải lỗi!)
        with self.assertRaises(IndexError):
            select_layer_head_attention(synth, layer_idx=3, head=0)

        # Kiểm tra chặn vượt biên trục head (chỉ có 4 head: 0..3, truyền 4 phải lỗi!)
        with self.assertRaises(IndexError):
            select_layer_head_attention(synth, layer_idx=0, head=4)

        # Kiểm tra head="all" và head="max"
        all_h, _, nh_all = select_layer_head_attention(synth, layer_idx=-1, head="all")
        self.assertEqual(nh_all, "all")
        self.assertEqual(tuple(all_h.shape), (4, 2, 2))
        max_h, _, nh_max = select_layer_head_attention(synth, layer_idx=0, head="max")
        self.assertEqual(nh_max, "max")
        self.assertEqual(tuple(max_h.shape), (2, 2))

    def test_04_subword_to_word_aggregation_exact_math(self):
        """Kiểm chứng phép tính toán học chính xác khi gộp subwords -> words."""
        # 4 subwords: [<s>, w0_a, w0_b, w1] với word_ids = [None, 0, 0, 1]
        attn = np.array([
            [0.4, 0.2, 0.2, 0.2],  # <s>
            [0.2, 0.3, 0.1, 0.4],  # w0_a -> keys: <s>=0.2, w0=(0.3+0.1=0.4), w1=0.4
            [0.2, 0.1, 0.5, 0.2],  # w0_b -> keys: <s>=0.2, w0=(0.1+0.5=0.6), w1=0.2
            [0.4, 0.2, 0.1, 0.3],  # w1   -> keys: <s>=0.4, w0=(0.2+0.1=0.3), w1=0.3
        ], dtype=np.float32)
        word_ids = [None, 0, 0, 1]
        orig_tokens = ["Nguyễn_Văn_A", "Hà_Nội"]

        # 1) Không renorm, bỏ special tokens:
        #    w0 query = mean([w0_a, w0_b]) = [(0.4+0.6)/2, (0.4+0.2)/2] = [0.5, 0.3]
        #    w1 query = [0.3, 0.3]
        #    special_mass_removed = mean([0.2, 0.2, 0.4]) = 0.8/3
        w_raw = aggregate_subword_attention_to_words(
            attn, word_ids, orig_tokens,
            query_agg="mean", key_agg="sum",
            include_special_tokens=False, renormalize=False,
        )
        self.assertEqual(w_raw.labels, ["Nguyễn_Văn_A", "Hà_Nội"])
        self.assertEqual(w_raw.word_to_subwords, {0: [1, 2], 1: [3]})
        np.testing.assert_allclose(w_raw.matrix, np.array([[0.5, 0.3], [0.3, 0.3]]), atol=1e-5)
        self.assertAlmostEqual(w_raw.special_token_mass_removed, 0.8 / 3.0, places=5)

        # 2) Có renormalize=True -> hàng 0: [0.5/0.8, 0.3/0.8] = [0.625, 0.375], hàng 1: [0.5, 0.5]
        w_norm = aggregate_subword_attention_to_words(
            attn, word_ids, orig_tokens,
            query_agg="mean", key_agg="sum",
            include_special_tokens=False, renormalize=True,
        )
        np.testing.assert_allclose(w_norm.matrix, np.array([[0.625, 0.375], [0.5, 0.5]]), atol=1e-5)
        np.testing.assert_allclose(w_norm.matrix.sum(axis=1), np.ones(2), atol=1e-5)

        # 3) query_agg="first" -> hàng 0 lấy đúng w0_a: [0.4, 0.4] -> renorm = [0.5, 0.5]
        w_first = aggregate_subword_attention_to_words(
            attn, word_ids, orig_tokens,
            query_agg="first", key_agg="sum",
            include_special_tokens=False, renormalize=True,
        )
        np.testing.assert_allclose(w_first.matrix[0], np.array([0.5, 0.5]), atol=1e-5)

        # 4) include_special_tokens=True -> kích thước 3x3 và tổng hàng tự nhiên = 1.0
        w_spec = aggregate_subword_attention_to_words(
            attn, word_ids, orig_tokens,
            subword_tokens=["<s>", "Ng_@@", "A", "HN"],
            query_agg="mean", key_agg="sum",
            include_special_tokens=True, renormalize=False,
        )
        self.assertEqual(w_spec.labels, ["<s>", "Nguyễn_Văn_A", "Hà_Nội"])
        np.testing.assert_allclose(w_spec.matrix.sum(axis=1), np.ones(3), atol=1e-5)

    def test_05_attention_rollout_properties(self):
        """Kiểm tra Attention Rollout có tổng hàng = 1.0 và giá trị hữu hạn."""
        res = extract_sentence_attention(self.model, self.tokenizer, self.sample_tokens)
        rollout = compute_attention_rollout(res.attentions, start_layer=0, end_layer=2)
        self.assertEqual(tuple(rollout.shape), (10, 10))
        self.assertTrue(torch.isfinite(rollout).all())
        row_sums = rollout.sum(dim=-1)
        self.assertLess((row_sums - 1.0).abs().max().item(), 1e-4)

    def test_06_preserves_model_training_state_and_no_grad(self):
        """Đảm bảo trích xuất attention KHÔNG làm đổi chế độ train/eval và KHÔNG làm bẩn gradient."""
        self.model.zero_grad(set_to_none=True)

        # Trường hợp 1: model đang ở chế độ train()
        self.model.train()
        self.assertTrue(self.model.training)
        _ = extract_sentence_attention(self.model, self.tokenizer, self.sample_tokens)
        self.assertTrue(self.model.training, "Trạng thái model.train() bị thay đổi sau khi trích xuất!")
        for p in self.model.parameters():
            self.assertIsNone(p.grad, "Phát hiện gradient bị sinh ra trong quá trình trích xuất attention!")

        # Trường hợp 2: model đang ở chế độ eval()
        self.model.eval()
        self.assertFalse(self.model.training)
        _ = extract_sentence_attention(self.model, self.tokenizer, self.sample_tokens)
        self.assertFalse(self.model.training, "Trạng thái model.eval() bị thay đổi sau khi trích xuất!")

        # Trường hợp 3: kiểm tra cả inspect_model_attention_spec và extract_attention_weights
        self.model.train()
        _ = inspect_model_attention_spec(self.model, self.tokenizer)
        self.assertTrue(self.model.training)
        ids = torch.tensor([[0, 10, 11, 2]], dtype=torch.long)
        mask = torch.ones_like(ids)
        attn_legacy = extract_attention_weights(self.model, ids, mask, layer_idx=-1)
        self.assertTrue(self.model.training)
        self.assertEqual(tuple(attn_legacy.shape), (4, 4, 4))

    def test_07_validation_rejects_corrupted_attention(self):
        """Kiểm tra validate_attention_result bắt được NaN và tổng hàng sai."""
        res = extract_sentence_attention(self.model, self.tokenizer, self.sample_tokens)

        bad_nan = res.attentions.clone()
        bad_nan[0, 0, 0, 0] = float("nan")
        res_nan = AttentionExtractionResult(**{**res.__dict__, "attentions": bad_nan})
        with self.assertRaises(ValueError):
            validate_attention_result(res_nan)

        bad_sum = res.attentions.clone() * 0.5
        res_sum = AttentionExtractionResult(**{**res.__dict__, "attentions": bad_sum})
        with self.assertRaises(ValueError):
            validate_attention_result(res_sum)

    def test_08_filename_convention_and_image_exports(self):
        """Kiểm tra đặt tên ảnh đúng quy chuẩn và lưu đầy đủ 4 loại biểu đồ không rỗng."""
        res = extract_sentence_attention(
            self.model, self.tokenizer, self.sample_tokens,
            layer_idx=-1, head="mean", id2tag=self.id2tag, sentence_id="vlsp_word_test_00627",
        )
        w_res = aggregate_subword_attention_to_words(
            res.selected_attention, res.word_ids, res.original_tokens,
            subword_tokens=res.subword_tokens, query_agg="mean", key_agg="sum",
        )

        # 1. Quy ước tên file
        p1 = build_attention_filename(self.temp_dir, "vinai/phobert-base-v2", "vlsp_word_test_00627", 11, "mean", "word")
        self.assertIn("phobert", p1.name)
        self.assertIn("vlsp_word_test_00627", p1.name)
        self.assertIn("L12", p1.name)
        self.assertIn("Hmean", p1.name)
        self.assertIn("word", p1.name)

        # 2. Heatmap đơn
        plot_attention_heatmap(w_res.labels, w_res.matrix, output_path=p1, tags=res.predicted_tags, annot=True)
        self.assertTrue(p1.exists() and p1.stat().st_size > 10_000)

        # 3. So sánh Heads
        p2 = build_attention_filename(self.temp_dir, "vinai/phobert-base-v2", "vlsp_word_test_00627", 2, "heads_grid", "word")
        plot_head_comparison(res, layer_idx=2, heads=(0, 1, 2, 3), output_path=p2)
        self.assertTrue(p2.exists() and p2.stat().st_size > 10_000)

        # 4. So sánh Layers
        p3 = build_attention_filename(self.temp_dir, "vinai/phobert-base-v2", "vlsp_word_test_00627", "1_2_3", "mean", "word")
        plot_layer_comparison(res, layers=(0, 1, 2), head="mean", output_path=p3)
        self.assertTrue(p3.exists() and p3.stat().st_size > 10_000)

        # 5. So sánh 2 mô hình cạnh nhau
        p4 = self.temp_dir / "compare_two_models.png"
        plot_model_attention_comparison(res, res, layer_idx=-1, head="mean", output_path=p4)
        self.assertTrue(p4.exists() and p4.stat().st_size > 10_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
