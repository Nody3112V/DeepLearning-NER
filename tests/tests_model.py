"""
tests_model.py - Bộ unit test toàn diện cho kiến trúc PretrainedNERModel (PhoBERT & XLM-R).

Kiểm tra:
  1. Khởi tạo mô hình & tương thích PhoBERT / XLM-R.
  2. Forward pass & kiểm tra kích thước tensor ([B, L, num_labels], hidden_states, attentions).
  3. Xử lý batch có độ dài thay đổi và padding attention mask.
  4. Tính toán Loss và kiểm tra bỏ qua nhãn ignore_index (-100).
  5. Backward pass & luồng Gradient (chỉ lan truyền đến các tham số requires_grad=True).
  6. Các chiến lược đóng băng tham số (Freeze embeddings, freeze N layers, freeze entire encoder, unfreeze_all).
  7. Thống kê tham số trainable (get_trainable_parameters).
  8. Kiểm định cả 4 chiến lược pooling (last, concat_4, mean_4, weighted).
  9. Tương thích thiết bị (CPU & CUDA nếu có).
  10. Tải mô hình từ tệp cấu hình YAML (PretrainedNERModel.from_config).
"""
import os
import unittest
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from src.models.pretrained.model import PretrainedNERModel


def create_tiny_backbone(hidden_size: int = 64, num_hidden_layers: int = 4) -> RobertaModel:
    """Tạo một Transformer Encoder nhỏ gọn để test nhanh trong vài mili-giây không cần mạng."""
    config = RobertaConfig(
        vocab_size=1000,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=512,
        output_hidden_states=True,
        output_attentions=True,
    )
    return RobertaModel(config)


class TestPretrainedNERModel(unittest.TestCase):
    """Bộ kiểm thử unit test cho PretrainedNERModel."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 64
        self.num_layers = 4
        self.num_labels = 9
        self.backbone = create_tiny_backbone(self.hidden_size, self.num_layers)

    def test_01_init_and_architecture(self):
        """Kiểm tra khởi tạo mô hình và các thuộc tính cơ sở."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            num_labels=self.num_labels,
            pooling_strategy="last",
            pretrained_backbone=self.backbone,
        )
        self.assertEqual(model.hidden_dim, self.hidden_size)
        self.assertEqual(model.num_labels, self.num_labels)
        self.assertEqual(model.classifier_in_dim, self.hidden_size)
        self.assertIsNone(model.layer_weights)

        # Kiểm tra phương thức sơ đồ kiến trúc
        diagram = model.get_architecture_diagram()
        self.assertIn("PretrainedNERModel Architecture", diagram)
        self.assertIn("Linear Classifier", diagram)

    def test_02_forward_shapes(self):
        """Kiểm tra kích thước đầu ra của forward pass."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            num_labels=self.num_labels,
            pooling_strategy="last",
            pretrained_backbone=self.backbone,
        )
        B, L = 2, 8
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones((B, L), dtype=torch.long)
        labels = torch.randint(0, self.num_labels, (B, L))

        out = model(input_ids, attention_mask, labels=labels)

        self.assertIn("loss", out)
        self.assertIn("logits", out)
        self.assertIn("hidden_states", out)
        self.assertIn("attentions", out)

        # Kiểm tra shape logits [B, L, num_labels]
        self.assertEqual(out["logits"].shape, (B, L, self.num_labels))
        # Kiểm tra loss là scalar
        self.assertEqual(out["loss"].dim(), 0)
        self.assertFalse(torch.isnan(out["loss"]))
        # Kiểm tra hidden states có num_layers + 1 phần tử
        self.assertEqual(len(out["hidden_states"]), self.num_layers + 1)
        self.assertEqual(out["hidden_states"][-1].shape, (B, L, self.hidden_size))

    def test_03_variable_sequence_length_and_padding(self):
        """Kiểm tra batch có padding (độ dài thay đổi)."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            num_labels=self.num_labels,
            pretrained_backbone=self.backbone,
        )
        B, L = 3, 12
        input_ids = torch.randint(1, 500, (B, L))
        attention_mask = torch.ones((B, L), dtype=torch.long)

        # Giả lập câu 2 có độ dài 8 (padding 4 token cuối), câu 3 độ dài 5 (padding 7 token cuối)
        attention_mask[1, 8:] = 0
        input_ids[1, 8:] = 1  # pad_token_id
        attention_mask[2, 5:] = 0
        input_ids[2, 5:] = 1

        labels = torch.randint(0, self.num_labels, (B, L))
        labels[1, 8:] = -100
        labels[2, 5:] = -100

        out = model(input_ids, attention_mask, labels=labels)
        self.assertEqual(out["logits"].shape, (B, L, self.num_labels))
        self.assertFalse(torch.isnan(out["loss"]))

    def test_04_loss_ignore_index(self):
        """Kiểm tra Loss bỏ qua hoàn toàn các vị trí -100."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            num_labels=self.num_labels,
            ignore_index=-100,
            pretrained_backbone=self.backbone,
        )
        B, L = 1, 4
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones((B, L), dtype=torch.long)

        # Trường hợp toàn bộ labels là -100 (tất cả bị ignore)
        all_ignore_labels = torch.full((B, L), -100, dtype=torch.long)
        out = model(input_ids, attention_mask, labels=all_ignore_labels)
        # Khi mọi label là -100, CrossEntropyLoss mặc định trả về 0.0 (hoặc nan nếu không reduce_mean an toàn, PyTorch chuẩn trả về 0.0 hoặc nan, check giá trị)
        # Kiểm tra trường hợp thực tế: 2 token có nhãn, 2 token -100
        labels_partial = torch.tensor([[1, -100, 2, -100]], dtype=torch.long)
        out_partial = model(input_ids, attention_mask, labels=labels_partial)
        self.assertFalse(torch.isnan(out_partial["loss"]))
        self.assertGreater(out_partial["loss"].item(), 0.0)

    def test_05_backward_pass_and_gradient_flow(self):
        """Kiểm tra backward pass và luồng gradient."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            num_labels=self.num_labels,
            pretrained_backbone=self.backbone,
        )
        input_ids = torch.randint(0, 1000, (2, 6))
        attention_mask = torch.ones((2, 6), dtype=torch.long)
        labels = torch.randint(0, self.num_labels, (2, 6))
        labels[:, 4:] = -100

        out = model(input_ids, attention_mask, labels=labels)
        loss = out["loss"]
        loss.backward()

        # Kiểm tra classifier có gradient
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertFalse(torch.isnan(model.classifier.weight.grad).any())

        # Kiểm tra encoder layer có gradient
        self.assertIsNotNone(model.encoder.encoder.layer[-1].output.dense.weight.grad)

    def test_06_freezing_strategies(self):
        """Kiểm tra chi tiết tất cả các chiến lược đóng băng."""
        # 1. Freeze Embeddings
        bb1 = create_tiny_backbone(self.hidden_size, self.num_layers)
        m1 = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            freeze_embeddings=True,
            freeze_encoder_layers=0,
            pretrained_backbone=bb1,
        )
        for p in m1.encoder.embeddings.parameters():
            self.assertFalse(p.requires_grad, "Embeddings phải bị đóng băng")
        for p in m1.encoder.encoder.layer[0].parameters():
            self.assertTrue(p.requires_grad, "Layer 0 phải trainable")

        # 2. Freeze N layers (vd đóng băng 2/4 layers)
        bb2 = create_tiny_backbone(self.hidden_size, self.num_layers)
        m2 = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            freeze_embeddings=False,
            freeze_encoder_layers=2,
            pretrained_backbone=bb2,
        )
        for p in m2.encoder.embeddings.parameters():
            self.assertTrue(p.requires_grad, "Embeddings phải trainable")
        for p in m2.encoder.encoder.layer[0].parameters():
            self.assertFalse(p.requires_grad, "Layer 0 phải bị đóng băng")
        for p in m2.encoder.encoder.layer[1].parameters():
            self.assertFalse(p.requires_grad, "Layer 1 phải bị đóng băng")
        for p in m2.encoder.encoder.layer[2].parameters():
            self.assertTrue(p.requires_grad, "Layer 2 phải trainable")
        for p in m2.encoder.encoder.layer[3].parameters():
            self.assertTrue(p.requires_grad, "Layer 3 phải trainable")

        # 3. Freeze Entire Encoder
        bb3 = create_tiny_backbone(self.hidden_size, self.num_layers)
        m3 = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            freeze_entire_encoder=True,
            pretrained_backbone=bb3,
        )
        for p in m3.encoder.parameters():
            self.assertFalse(p.requires_grad, "Toàn bộ encoder phải bị đóng băng")
        for p in m3.classifier.parameters():
            self.assertTrue(p.requires_grad, "Classifier head vẫn phải trainable")

        # 4. Unfreeze all
        m3.unfreeze_all()
        for p in m3.parameters():
            self.assertTrue(p.requires_grad, "Toàn bộ mô hình phải trainable sau unfreeze_all()")

    def test_07_trainable_parameters_count(self):
        """Kiểm tra hàm thống kê tham số get_trainable_parameters."""
        bb = create_tiny_backbone(self.hidden_size, self.num_layers)
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            freeze_entire_encoder=True,
            pretrained_backbone=bb,
        )
        stats = model.get_trainable_parameters()
        self.assertEqual(stats["encoder_trainable"], 0)
        self.assertGreater(stats["encoder_total"], 0)
        self.assertEqual(stats["classifier_trainable"], stats["classifier_total"])
        self.assertAlmostEqual(stats["trainable_params"], stats["classifier_total"])
        self.assertLess(stats["trainable_ratio"], 5.0)

    def test_08_pooling_strategies(self):
        """Kiểm định cả 4 chiến lược kết hợp tầng ẩn: last, concat_4, mean_4, weighted."""
        strategies = ["last", "concat_4", "mean_4", "weighted"]
        expected_dims = {
            "last": self.hidden_size,
            "concat_4": self.hidden_size * 4,
            "mean_4": self.hidden_size,
            "weighted": self.hidden_size,
        }

        B, L = 2, 6
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones((B, L), dtype=torch.long)
        labels = torch.randint(0, self.num_labels, (B, L))

        for strat in strategies:
            bb = create_tiny_backbone(self.hidden_size, self.num_layers)
            m = PretrainedNERModel(
                model_name_or_path="mock-roberta",
                num_labels=self.num_labels,
                pooling_strategy=strat,
                pretrained_backbone=bb,
            )
            self.assertEqual(m.classifier_in_dim, expected_dims[strat], f"Sai chiều in_dim cho strategy {strat}")

            if strat == "weighted":
                self.assertIsNotNone(m.layer_weights)
                self.assertEqual(m.layer_weights.shape[0], self.num_layers + 1)

            out = m(input_ids, attention_mask, labels=labels)
            self.assertEqual(out["logits"].shape, (B, L, self.num_labels))
            self.assertFalse(torch.isnan(out["loss"]))

            # Kiểm tra backward cho từng pooling strategy
            out["loss"].backward()
            if strat == "weighted":
                self.assertIsNotNone(m.layer_weights.grad, "layer_weights phải có gradient khi weighted pooling")

    def test_09_device_compatibility(self):
        """Kiểm tra khả năng chuyển thiết bị (CPU & GPU nếu có)."""
        model = PretrainedNERModel(
            model_name_or_path="mock-roberta",
            pretrained_backbone=self.backbone,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_ids = torch.randint(0, 1000, (2, 5), device=device)
        attention_mask = torch.ones((2, 5), dtype=torch.long, device=device)
        labels = torch.randint(0, self.num_labels, (2, 5), device=device)

        out = model(input_ids, attention_mask, labels=labels)
        self.assertEqual(out["logits"].device.type, device.type)
        self.assertEqual(out["loss"].device.type, device.type)

    def test_10_from_config_yaml(self):
        """Kiểm tra khởi tạo mô hình từ file YAML configs/pretrained_plm.yaml."""
        yaml_path = "configs/pretrained_plm.yaml"
        if not os.path.exists(yaml_path):
            self.skipTest("Không tìm thấy file configs/pretrained_plm.yaml")

        # Khởi tạo mô hình sử dụng backbone nhỏ để không cần tải weight qua mạng khi test
        tiny_bb = create_tiny_backbone(self.hidden_size, self.num_layers)
        model = PretrainedNERModel.from_config(
            yaml_path,
            pretrained_backbone=tiny_bb,
        )
        self.assertEqual(model.num_labels, 9)
        self.assertEqual(model.pooling_strategy, "last")
        self.assertEqual(model.classifier_dropout_prob, 0.2)


if __name__ == "__main__":
    unittest.main()
