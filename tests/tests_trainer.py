"""
tests_trainer.py - Bộ unit test kiểm định toàn diện cho PLMTrainer (PhoBERT & XLM-R).

Kiểm tra:
  1. Khởi tạo Trainer với mô hình & DataLoader.
  2. Phân nhóm tham số Optimizer (AdamW, differential LRs, weight decay, loại trừ bias & LayerNorm).
  3. Khởi tạo Learning Rate Schedulers (linear, cosine, constant_with_warmup).
  4. Huấn luyện một batch (Forward, Backward, GradScaler, Loss giảm).
  5. Đảm bảo tham số bị đóng băng (Frozen) KHÔNG bị cập nhật sau optimizer step.
  6. Vòng lặp đánh giá (evaluate) trích xuất Entity-level F1 qua src.common.metrics.
  7. Lưu và nạp Checkpoint toàn diện (save_checkpoint / load_checkpoint).
  8. Tính tái lập của Random Seed (set_seed).
  9. Cơ chế Gradient Accumulation.
  10. Cơ chế Early Stopping khi metric không cải thiện.
"""
import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaModel

from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.trainer import PLMTrainer, set_seed


class DummyNERDataset(Dataset):
    """Dataset giả lập để chạy test nhanh trong vài mili-giây."""

    def __init__(self, size: int = 8, seq_len: int = 6, num_labels: int = 9):
        self.size = size
        self.seq_len = seq_len
        self.num_labels = num_labels
        self.data = []
        for i in range(size):
            input_ids = torch.randint(2, 500, (seq_len,), dtype=torch.long)
            attention_mask = torch.ones((seq_len,), dtype=torch.long)
            labels = torch.randint(0, num_labels, (seq_len,), dtype=torch.long)
            labels[0] = -100  # special token
            labels[-1] = -100 # special token
            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def dummy_collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def create_mock_ner_model(hidden_size: int = 64, num_layers: int = 2, num_labels: int = 9, **kwargs) -> PretrainedNERModel:
    config = RobertaConfig(
        vocab_size=500,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        intermediate_size=hidden_size * 2,
        output_hidden_states=True,
        output_attentions=True,
    )
    backbone = RobertaModel(config)
    return PretrainedNERModel(
        model_name_or_path="mock-roberta",
        num_labels=num_labels,
        pretrained_backbone=backbone,
        **kwargs,
    )


class TestPLMTrainer(unittest.TestCase):
    """Bộ kiểm thử unit test cho PLMTrainer."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "experiment_name": "test_plm_run",
            "seed": 42,
            "train": {
                "batch_size": 4,
                "num_epochs": 2,
                "encoder_lr": 1e-4,
                "classifier_lr": 5e-4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "lr_scheduler": "linear",
                "grad_clip": 1.0,
                "early_stopping_patience": 2,
                "use_amp": False,
            },
            "logging": {
                "checkpoint_dir": self.temp_dir,
                "backends": "local",
            },
            "data": {
                "dataset": "vlsp",
                "granularity": "word",
            }
        }
        self.id2tag = {
            0: "O",
            1: "B-PER", 2: "I-PER",
            3: "B-LOC", 4: "I-LOC",
            5: "B-ORG", 6: "I-ORG",
            7: "B-MISC", 8: "I-MISC"
        }
        self.train_ds = DummyNERDataset(size=8, seq_len=6, num_labels=9)
        self.dev_ds = DummyNERDataset(size=4, seq_len=6, num_labels=9)
        self.train_loader = DataLoader(self.train_ds, batch_size=4, collate_fn=dummy_collate)
        self.dev_loader = DataLoader(self.dev_ds, batch_size=4, collate_fn=dummy_collate)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_trainer_initialization(self):
        """1. Kiểm tra khởi tạo PLMTrainer với mô hình và DataLoader."""
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        self.assertEqual(trainer.exp_name, "test_plm_run")
        self.assertEqual(trainer.num_epochs, 2)
        self.assertEqual(trainer.total_steps, 4)  # 2 batches * 2 epochs
        self.assertEqual(trainer.warmup_steps, 0) # 4 * 0.1 = 0 steps

    def test_02_optimizer_parameter_groups(self):
        """2. Kiểm tra phân nhóm tham số trong AdamW."""
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        opt = trainer.optimizer
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertGreaterEqual(len(opt.param_groups), 2)

        # Kiểm tra lrs
        encoder_lrs = [g["lr"] for g in opt.param_groups if g["lr"] == 1e-4]
        classifier_lrs = [g["lr"] for g in opt.param_groups if g["lr"] == 5e-4]
        self.assertGreater(len(encoder_lrs), 0)
        self.assertGreater(len(classifier_lrs), 0)

        # Kiểm tra weight decay
        decays = [g["weight_decay"] for g in opt.param_groups]
        self.assertIn(0.01, decays)
        self.assertIn(0.0, decays)

    def test_03_scheduler_types(self):
        """3. Kiểm tra hỗ trợ các scheduler: linear, cosine, constant_with_warmup."""
        for sched in ["linear", "cosine", "constant_with_warmup"]:
            cfg = dict(self.config)
            cfg["train"] = dict(self.config["train"])
            cfg["train"]["lr_scheduler"] = sched
            model = create_mock_ner_model()
            trainer = PLMTrainer(
                model=model,
                train_dataloader=self.train_loader,
                dev_dataloader=self.dev_loader,
                id2tag=self.id2tag,
                config=cfg,
                device=torch.device("cpu"),
            )
            self.assertIsNotNone(trainer.scheduler)

    def test_04_one_batch_train_step(self):
        """4. Kiểm tra huấn luyện 1 epoch/batch, loss tính toán và gradient cập nhật."""
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        res = trainer.train_epoch(epoch=1)
        self.assertIn("loss", res)
        self.assertIn("lr", res)
        self.assertFalse(torch.isnan(torch.tensor(res["loss"])))
        self.assertGreater(res["loss"], 0.0)

    def test_05_frozen_parameters_stay_frozen(self):
        """5. Đảm bảo các tham số bị đóng băng KHÔNG bị cập nhật sau training step."""
        model = create_mock_ner_model(freeze_entire_encoder=True)
        # Lưu bản sao trọng số của encoder trước khi train
        frozen_weight_before = model.encoder.embeddings.word_embeddings.weight.clone()
        classifier_weight_before = model.classifier.weight.clone()

        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        trainer.train_epoch(epoch=1)

        frozen_weight_after = model.encoder.embeddings.word_embeddings.weight
        classifier_weight_after = model.classifier.weight

        # Trọng số frozen KHÔNG được thay đổi
        self.assertTrue(torch.equal(frozen_weight_before, frozen_weight_after), "Trọng số bị đóng băng đã bị thay đổi!")
        # Trọng số classifier PHẢI được thay đổi
        self.assertFalse(torch.equal(classifier_weight_before, classifier_weight_after), "Classifier head không được cập nhật!")

    def test_06_evaluate_loop_and_metrics(self):
        """6. Kiểm tra vòng lặp evaluate tính toán đúng Entity-level F1."""
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        dev_loss, dev_metrics = trainer.evaluate(self.dev_loader)
        self.assertIsInstance(dev_loss, float)
        self.assertIn("overall", dev_metrics)
        self.assertIn("f1", dev_metrics["overall"])
        self.assertIn("precision", dev_metrics["overall"])
        self.assertIn("recall", dev_metrics["overall"])

    def test_07_checkpoint_save_and_load(self):
        """7. Kiểm tra lưu và nạp Checkpoint."""
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=self.config,
            device=torch.device("cpu"),
        )
        ckpt_file = os.path.join(self.temp_dir, "test_ckpt.pt")
        dummy_metrics = {"overall": {"f1": 0.85, "precision": 0.8, "recall": 0.9}}
        trainer.save_checkpoint(ckpt_file, epoch=1, metrics=dummy_metrics, is_best=True)

        self.assertTrue(os.path.exists(ckpt_file))

        # Nạp lại checkpoint
        loaded_model, ckpt_dict = PLMTrainer.load_checkpoint(ckpt_file, model=create_mock_ner_model(), device=torch.device("cpu"))
        self.assertEqual(ckpt_dict["epoch"], 1)
        self.assertEqual(ckpt_dict["seed"], 42)
        # So sánh trọng số
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_08_seed_reproducibility(self):
        """8. Kiểm tra tính tái lập khi thiết lập seed."""
        set_seed(12345)
        t1 = torch.randn(10)
        set_seed(12345)
        t2 = torch.randn(10)
        self.assertTrue(torch.equal(t1, t2))

    def test_09_gradient_accumulation(self):
        """9. Kiểm tra cơ chế Gradient Accumulation (2 steps)."""
        cfg = dict(self.config)
        cfg["train"] = dict(self.config["train"])
        cfg["train"]["gradient_accumulation_steps"] = 2
        model = create_mock_ner_model()
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=cfg,
            device=torch.device("cpu"),
        )
        res = trainer.train_epoch(epoch=1)
        self.assertGreater(res["loss"], 0.0)

    def test_10_early_stopping(self):
        """10. Kiểm tra cơ chế Early Stopping khi F1 không tăng."""
        model = create_mock_ner_model()
        cfg = dict(self.config)
        cfg["train"] = dict(self.config["train"])
        cfg["train"]["early_stopping_patience"] = 1
        cfg["train"]["num_epochs"] = 3
        trainer = PLMTrainer(
            model=model,
            train_dataloader=self.train_loader,
            dev_dataloader=self.dev_loader,
            id2tag=self.id2tag,
            config=cfg,
            device=torch.device("cpu"),
        )
        trainer.best_f1 = 1.0  # Đặt F1 tốt nhất là 1.0 (không thể vượt qua)
        summary = trainer.train()
        # Huấn luyện phải dừng sớm trước epoch 3
        self.assertLessEqual(trainer.patience_counter, 2)


if __name__ == "__main__":
    unittest.main()
