"""
trainer.py - Quy trình huấn luyện và đánh giá mô hình PLM (PhoBERT & XLM-R).

Tuân thủ nghiêm ngặt các quy tắc kỹ thuật trong .agent/skills/dl-engineer.md:
  * Thiết lập seed cố định đảm bảo tính tái lập.
  * Tối ưu VRAM: Mixed Precision (torch.amp.autocast + GradScaler), pin_memory=True.
  * Đánh giá bọc trong with torch.no_grad(): và model.eval().
  * Tích hợp src.common.logger (WandB/TensorBoard/local) và src.common.metrics (Entity-level F1).
  * Hỗ trợ lưu checkpoint tốt nhất và early stopping.
"""
from typing import Dict, List, Optional, Tuple, Any
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.common.metrics import compute_metrics, format_report
from src.common.logger import ExperimentLogger
from src.models.pretrained.model import PretrainedNERModel


def set_seed(seed: int = 42) -> None:
    """Thiết lập seed ngẫu nhiên cố định để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PLMTrainer:
    """Trainer cho mô hình Pretrained Language Models Token Classification."""

    def __init__(
        self,
        model: PretrainedNERModel,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        id2tag: Dict[int, str],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.train_cfg = config.get("train", {})
        self.log_cfg = config.get("logging", {})
        self.exp_name = config.get("experiment_name", "phobert_ner")

        # Thiết lập seed
        set_seed(config.get("seed", 42))

        # Thiết bị
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.id2tag = id2tag

        # Thư mục lưu checkpoint
        self.checkpoint_dir = Path(self.log_cfg.get("checkpoint_dir", "outputs/checkpoints/pretrained")) / self.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Cấu hình Optimizer với Differential Learning Rates và Weight Decay
        self.optimizer = self._build_optimizer()

        # Cấu hình Scheduler
        num_epochs = self.train_cfg.get("num_epochs", 10)
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * self.train_cfg.get("warmup_ratio", 0.1))
        self.scheduler = self._build_scheduler(total_steps, warmup_steps)

        # Mixed Precision (AMP)
        self.use_amp = self.train_cfg.get("use_amp", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_f1 = -1.0
        self.early_stopping_patience = self.train_cfg.get("early_stopping_patience", 3)
        self.patience_counter = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Tạo AdamW optimizer với learning rate riêng cho encoder và classifier."""
        encoder_lr = float(self.train_cfg.get("encoder_lr", 2.0e-5))
        classifier_lr = float(self.train_cfg.get("classifier_lr", 1.0e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.01))

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            # 1. Encoder params có weight decay
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            # 2. Encoder params không có weight decay
            {
                "params": [
                    p for n, p in self.model.encoder.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            # 3. Classifier params có weight decay
            {
                "params": [
                    p for n, p in self.model.classifier.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ] + ([self.model.layer_weights] if hasattr(self.model, "layer_weights") and self.model.layer_weights.requires_grad else []),
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            },
            # 4. Classifier params không có weight decay
            {
                "params": [
                    p for n, p in self.model.classifier.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "lr": classifier_lr,
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(optimizer_grouped_parameters)

    def _build_scheduler(self, total_steps: int, warmup_steps: int):
        sched_type = self.train_cfg.get("lr_scheduler", "linear").lower()
        if sched_type == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        elif sched_type == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        else:
            return None

    def train_epoch(self, epoch: int, logger: ExperimentLogger) -> float:
        """Huấn luyện 1 epoch với Mixed Precision."""
        self.model.train()
        total_loss = 0.0
        grad_clip = float(self.train_cfg.get("grad_clip", 1.0))

        for step, batch in enumerate(self.train_dataloader):
            input_ids = batch["input_ids"].to(self.device)         # [B, L]
            attention_mask = batch["attention_mask"].to(self.device) # [B, L]
            labels = batch["labels"].to(self.device)                 # [B, L]

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]

            # Backward pass với GradScaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            global_step = epoch * len(self.train_dataloader) + step
            if step % 20 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.log({
                    "train/loss": loss_val,
                    "train/lr": current_lr,
                }, step=global_step)

        return total_loss / max(len(self.train_dataloader), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Đánh giá mô hình trên tập kiểm tra / validation."""
        self.model.eval()
        total_loss = 0.0

        all_gold_tags: List[List[str]] = []
        all_pred_tags: List[List[str]] = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)         # [B, L]
            attention_mask = batch["attention_mask"].to(self.device) # [B, L]
            labels = batch["labels"].to(self.device)                 # [B, L]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]                         # [B, L, num_labels]

            if loss is not None:
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()     # [B, L]
            labels_np = labels.cpu().numpy()                       # [B, L]

            for b in range(len(preds)):
                gold_seq = []
                pred_seq = []
                for l_id, p_id in zip(labels_np[b], preds[b]):
                    if l_id != -100:  # Bỏ qua special tokens và subwords đứng sau
                        gold_seq.append(self.id2tag.get(int(l_id), "O"))
                        pred_seq.append(self.id2tag.get(int(p_id), "O"))

                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

        avg_loss = total_loss / max(len(dataloader), 1)
        metrics = compute_metrics(all_gold_tags, all_pred_tags)
        return avg_loss, metrics

    def train(self) -> Dict[str, Any]:
        """Vòng lặp huấn luyện chính."""
        num_epochs = self.train_cfg.get("num_epochs", 10)
        backends = self.log_cfg.get("backends", "auto")
        project_name = self.log_cfg.get("project_name", "vietnamese-ner-acl")

        print(f"\n==========================================")
        print(f"BẮT ĐẦU HUẤN LUYỆN PLM: {self.exp_name}")
        print(f"Thiết bị: {self.device} | AMP: {self.use_amp} | Epochs: {num_epochs}")
        print(f"==========================================\n")

        with ExperimentLogger(
            run_name=self.exp_name,
            member="tv2",
            config=self.config,
            project=project_name,
            backends=backends,
            tags=["plm", self.config.get("data", {}).get("dataset", "vlsp"), self.config.get("data", {}).get("granularity", "word")],
        ) as logger:

            for epoch in range(1, num_epochs + 1):
                train_loss = self.train_epoch(epoch, logger)
                dev_loss, dev_metrics = self.evaluate(self.dev_dataloader)

                f1_micro = dev_metrics.get("f1_micro", 0.0)
                f1_macro = dev_metrics.get("f1_macro", 0.0)

                # Ghi nhận log
                logger.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "dev/loss": dev_loss,
                    "dev/f1_micro": f1_micro,
                    "dev/f1_macro": f1_macro,
                }, step=epoch)
                logger.log_ner_metrics(dev_metrics, prefix="dev", step=epoch)

                print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
                      f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
                      f"Dev F1 (micro): {f1_micro:.4f} | Dev F1 (macro): {f1_macro:.4f}")

                # Lưu checkpoint tốt nhất theo f1_micro
                if f1_micro > self.best_f1:
                    self.best_f1 = f1_micro
                    self.patience_counter = 0
                    best_ckpt_path = self.checkpoint_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                        "id2tag": self.id2tag,
                        "best_f1": self.best_f1,
                    }, best_ckpt_path)
                    print(f"  >>> Đã lưu checkpoint mới tốt nhất: {best_ckpt_path} (F1: {self.best_f1:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\n[Early Stopping] Dừng sớm tại epoch {epoch} vì F1 không tăng sau {self.early_stopping_patience} epochs.")
                        break

            logger.log_summary({"best_dev_f1_micro": self.best_f1})

        return {"best_f1_micro": self.best_f1}
