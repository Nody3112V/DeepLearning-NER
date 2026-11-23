"""
trainer.py - Quy trình huấn luyện và đánh giá mô hình Pretrained PLM (PhoBERT & XLM-R).

Tuân thủ nghiêm ngặt các quy tắc kỹ thuật trong .agent/skills/dl-engineer.md:
  * Thiết lập seed cố định đảm bảo tính tái lập (set_seed).
  * Tối ưu VRAM: Mixed Precision (torch.amp.autocast + GradScaler), pin_memory=True.
  * Hỗ trợ Gradient Accumulation để mở rộng batch size hiệu quả.
  * Hỗ trợ Differential Learning Rates & Weight Decay tách nhóm (không áp dụng decay cho bias, LayerNorm).
  * Đánh giá bọc trong with torch.no_grad(): và model.eval().
  * Đánh giá chuẩn Entity-level F1 dùng chung qua src.common.metrics (micro-F1, macro-F1, per-type).
  * Ghi nhật ký chuẩn qua src.common.logger (ExperimentLogger với W&B / TensorBoard / Local).
  * Hỗ trợ Early Stopping và lưu Checkpoint toàn diện (model, optimizer, scheduler, metadata, tokenizer).
  * Xử lý ngoại lệ GPU OutOfMemory (OOM) an toàn.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import time
import random
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    PreTrainedTokenizerBase,
)

from src.common.metrics import compute_metrics, format_report
from src.common.logger import ExperimentLogger
from src.models.pretrained.model import PretrainedNERModel


def set_seed(seed: int = 42) -> None:
    """Thiết lập seed ngẫu nhiên cố định để đảm bảo tính tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PLMTrainer:
    """
    Huấn luyện viên chuẩn hóa cho mô hình Pretrained Language Models (PhoBERT & XLM-R).
    """

    def __init__(
        self,
        model: PretrainedNERModel,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        id2tag: Dict[int, str],
        config: Dict[str, Any],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Khởi tạo PLMTrainer.

        Args:
            model: Instance của PretrainedNERModel.
            train_dataloader: DataLoader tập huấn luyện.
            dev_dataloader: DataLoader tập kiểm thử validation.
            id2tag: Bảng ánh xạ ID số nguyên -> Tên nhãn BIO.
            config: Cấu hình đầy đủ từ file YAML.
            tokenizer: Tokenizer dùng để lưu kèm checkpoint (nếu có).
            device: Thiết bị thực thi (CPU hoặc CUDA).
        """
        self.config = config
        self.train_cfg = config.get("train", {})
        self.log_cfg = config.get("logging", {})
        self.data_cfg = config.get("data", {})
        self.exp_name = config.get("experiment_name", "phobert_ner")
        self.seed = int(config.get("seed", 42))

        # Thiết lập seed cố định
        set_seed(self.seed)

        # Thiết bị phần cứng
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.id2tag = {int(k): v for k, v in id2tag.items()}
        self.tag2id = {v: k for k, v in self.id2tag.items()}

        # Thư mục lưu checkpoint
        base_ckpt_dir = self.log_cfg.get("checkpoint_dir", "outputs/checkpoints/pretrained")
        self.checkpoint_dir = Path(base_ckpt_dir) / self.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tham số tối ưu & Phần cứng
        self.num_epochs = int(self.train_cfg.get("num_epochs", 10))
        self.gradient_accumulation_steps = max(1, int(self.train_cfg.get("gradient_accumulation_steps", 1)))
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.early_stopping_patience = int(self.train_cfg.get("early_stopping_patience", 3))

        # Mixed Precision (AMP)
        self.use_amp = bool(self.train_cfg.get("use_amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Xây dựng Optimizer
        self.optimizer = self._build_optimizer()

        # Xây dựng Learning Rate Scheduler
        num_update_steps_per_epoch = max(1, len(self.train_dataloader) // self.gradient_accumulation_steps)
        self.total_steps = num_update_steps_per_epoch * self.num_epochs

        warmup_steps_cfg = self.train_cfg.get("warmup_steps", None)
        if warmup_steps_cfg is not None:
            self.warmup_steps = int(warmup_steps_cfg)
        else:
            warmup_ratio = float(self.train_cfg.get("warmup_ratio", 0.1))
            self.warmup_steps = int(self.total_steps * warmup_ratio)

        self.scheduler = self._build_scheduler(self.total_steps, self.warmup_steps)

        # Trạng thái theo dõi Early Stopping & Best Checkpoint
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.patience_counter = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Tạo AdamW optimizer với phân nhóm tham số:
          - Tách riêng Encoder và Classifier (hỗ trợ Differential Learning Rates).
          - Tách riêng tham số có weight decay và tham số không áp dụng decay (bias, LayerNorm).
          - ĐẢM BẢO CHỈ đưa các tham số requires_grad=True vào optimizer (không cập nhật tham số bị đóng băng).
        """
        # Đọc learning rate (hỗ trợ differential lr hoặc single lr)
        default_lr = float(self.train_cfg.get("learning_rate", 2e-5))
        encoder_lr = float(self.train_cfg.get("encoder_lr", default_lr))
        classifier_lr = float(self.train_cfg.get("classifier_lr", default_lr * 5))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.01))

        no_decay_names = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]

        # Lọc tham số của encoder
        encoder_decay_params = []
        encoder_no_decay_params = []
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in ["bias", "layernorm"]):
                    encoder_no_decay_params.append(param)
                else:
                    encoder_decay_params.append(param)

        # Lọc tham số của classifier & các tham số phụ (layer_weights nếu có)
        classifier_decay_params = []
        classifier_no_decay_params = []

        # Tầng linear classifier
        for name, param in self.model.classifier.named_parameters():
            if param.requires_grad:
                if "bias" in name.lower():
                    classifier_no_decay_params.append(param)
                else:
                    classifier_decay_params.append(param)

        # Trọng số layer_weights của weighted pooling (nếu có)
        if hasattr(self.model, "layer_weights") and self.model.layer_weights is not None:
            if self.model.layer_weights.requires_grad:
                classifier_no_decay_params.append(self.model.layer_weights)

        optimizer_grouped_parameters = []
        if encoder_decay_params:
            optimizer_grouped_parameters.append({
                "params": encoder_decay_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            })
        if encoder_no_decay_params:
            optimizer_grouped_parameters.append({
                "params": encoder_no_decay_params,
                "lr": encoder_lr,
                "weight_decay": 0.0,
            })
        if classifier_decay_params:
            optimizer_grouped_parameters.append({
                "params": classifier_decay_params,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            })
        if classifier_no_decay_params:
            optimizer_grouped_parameters.append({
                "params": classifier_no_decay_params,
                "lr": classifier_lr,
                "weight_decay": 0.0,
            })

        # Trường hợp ngoại lệ nếu toàn bộ encoder bị freeze hoàn toàn (head-only)
        if not optimizer_grouped_parameters:
            # Fallback mọi tham số trainable
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [{
                "params": trainable_params,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            }]

        return torch.optim.AdamW(optimizer_grouped_parameters)

    def _build_scheduler(self, total_steps: int, warmup_steps: int) -> Optional[Any]:
        """Tạo lịch trình thay đổi tốc độ học (LR Scheduler)."""
        sched_type = str(self.train_cfg.get("lr_scheduler", "linear")).lower()

        if sched_type == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        elif sched_type == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        elif sched_type in ("constant_with_warmup", "constant"):
            return get_constant_schedule_with_warmup(self.optimizer, warmup_steps)
        elif sched_type == "none":
            return None
        else:
            print(f"Cảnh báo: Scheduler '{sched_type}' không xác định, sử dụng linear scheduler mặc định.")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

    def train_epoch(self, epoch: int, logger: Optional[ExperimentLogger] = None) -> Dict[str, float]:
        """
        Huấn luyện 1 epoch với Mixed Precision và Gradient Accumulation.

        Returns:
            Dict chứa loss trung bình, learning rate và thời gian epoch.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad()

        num_batches = len(self.train_dataloader)

        for step, batch in enumerate(self.train_dataloader, start=1):
            input_ids = batch["input_ids"].to(self.device)           # [B, L]
            attention_mask = batch["attention_mask"].to(self.device) # [B, L]
            labels = batch["labels"].to(self.device)                 # [B, L]

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            try:
                # Forward pass với mixed precision
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                    loss = outputs["loss"]
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps

                # Backward pass có hỗ trợ GradScaler
                self.scaler.scale(loss).backward()

            except torch.cuda.OutOfMemoryError as oom_err:
                print(f"\n[CRITICAL ERROR] Hết bộ nhớ GPU (CUDA OutOfMemory) tại step {step}/{num_batches}!")
                print(f"Cấu hình hiện tại: Batch size = {batch['input_ids'].shape[0]}, "
                      f"Seq len = {batch['input_ids'].shape[1]}")
                torch.cuda.empty_cache()
                raise oom_err

            # Cập nhật gradient sau mỗi accumulation steps hoặc batch cuối cùng
            if (step % self.gradient_accumulation_steps == 0) or (step == num_batches):
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            loss_val = loss.item() * (self.gradient_accumulation_steps if self.gradient_accumulation_steps > 1 else 1.0)
            total_loss += loss_val

            # Logging định kỳ trong epoch
            if step % max(1, (num_batches // 10)) == 0 or step == num_batches:
                current_lr = self.optimizer.param_groups[0]["lr"]
                global_step = (epoch - 1) * num_batches + step
                if logger is not None:
                    log_payload = {
                        "train/step_loss": loss_val,
                        "train/lr": current_lr,
                    }
                    if torch.cuda.is_available():
                        log_payload["train/gpu_mem_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
                    logger.log(log_payload, step=global_step)

        duration = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": avg_loss,
            "lr": current_lr,
            "duration": duration,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """
        Đánh giá mô hình trên tập kiểm thử / validation.
        Tính toán Entity-level F1 (micro, macro, per-type) thông qua src.common.metrics.

        Returns:
            Tuple gồm (loss_trung_bình, dict_metrics)
        """
        self.model.eval()
        total_loss = 0.0

        all_gold_tags: List[List[str]] = []
        all_pred_tags: List[List[str]] = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)           # [B, L]
            attention_mask = batch["attention_mask"].to(self.device) # [B, L]
            labels = batch["labels"].to(self.device)                 # [B, L]

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]                           # [B, L, num_labels]

            if loss is not None:
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()       # [B, L]
            labels_np = labels.cpu().numpy()                         # [B, L]

            # Khôi phục nhãn thực thể tại các vị trí không phải ignore_index (-100)
            for b in range(len(preds)):
                gold_seq: List[str] = []
                pred_seq: List[str] = []
                for l_id, p_id in zip(labels_np[b], preds[b]):
                    if l_id != -100:  # Bỏ qua padding, special tokens, subwords phụ
                        gold_seq.append(self.id2tag.get(int(l_id), "O"))
                        pred_seq.append(self.id2tag.get(int(p_id), "O"))

                all_gold_tags.append(gold_seq)
                all_pred_tags.append(pred_seq)

        avg_loss = total_loss / max(len(dataloader), 1)

        # Tính toán Entity-level metrics chuẩn theo quy ước nhóm
        metrics = compute_metrics(all_gold_tags, all_pred_tags)
        return avg_loss, metrics

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        save_optimizer: bool = False,
    ) -> None:
        """Lưu toàn diện trạng thái mô hình và metadata phục vụ tái lập."""
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if (save_optimizer and self.scheduler is not None) else None,
            "scaler_state_dict": self.scaler.state_dict() if (save_optimizer and self.scaler is not None) else None,
            "config": self.config,
            "id2tag": self.id2tag,
            "tag2id": self.tag2id,
            "best_f1": self.best_f1,
            "metrics": metrics,
            "seed": self.seed,
            "pooling_strategy": self.model.pooling_strategy,
            "pytorch_version": str(torch.__version__),
        }

        tmp_path = ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint_dict, tmp_path)
        if tmp_path.exists():
            if ckpt_path.exists():
                try:
                    ckpt_path.unlink()
                except Exception:
                    pass
            try:
                tmp_path.rename(ckpt_path)
            except Exception:
                # Fallback nếu rename trên Windows bị lock
                torch.save(checkpoint_dict, ckpt_path)
                if tmp_path.exists():
                    tmp_path.unlink()

        # Nếu có tokenizer, lưu kèm vào thư mục checkpoint
        if self.tokenizer is not None and is_best:
            try:
                self.tokenizer.save_pretrained(self.checkpoint_dir)
            except Exception as e:
                print(f"Cảnh báo: Không lưu được tokenizer: {e}")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model: Optional[PretrainedNERModel] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[PretrainedNERModel, Dict[str, Any]]:
        """
        Nạp lại checkpoint đã lưu.

        Args:
            checkpoint_path: Đường dẫn tệp .pt.
            model: Instance mô hình (nếu None sẽ tự khởi tạo dựa trên config trong checkpoint).
            device: Thiết bị nạp checkpoint.

        Returns:
            Tuple gồm (model, checkpoint_dict).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)

        if model is None:
            cfg = ckpt.get("config", {})
            model = PretrainedNERModel.from_config(cfg)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt

    def train(self) -> Dict[str, Any]:
        """
        Vòng lặp huấn luyện chính (Toàn bộ các epoch).
        Tích hợp ExperimentLogger, Early Stopping, lưu Checkpoint tốt nhất.
        """
        backends = self.log_cfg.get("backends", "auto")
        project_name = self.log_cfg.get("project_name", "vietnamese-ner-acl")

        print(f"\n=======================================================")
        print(f"  BẮT ĐẦU HUẤN LUYỆN: {self.exp_name}")
        print(f"  Thiết bị: {self.device} | AMP: {self.use_amp} | Epochs: {self.num_epochs}")
        print(f"  Model: {self.model.model_name_or_path} | Pooling: {self.model.pooling_strategy}")
        param_stats = self.model.get_trainable_parameters()
        print(f"  Tham số Trainable: {param_stats['trainable_params']:,} / {param_stats['total_params']:,} "
              f"({param_stats['trainable_ratio']:.2f}%)")
        print(f"=======================================================\n")

        total_training_start = time.time()

        with ExperimentLogger(
            run_name=self.exp_name,
            member="tv2",
            config=self.config,
            project=project_name,
            backends=backends,
            tags=["plm", self.config.get("data", {}).get("dataset", "vlsp"), self.config.get("data", {}).get("granularity", "word")],
        ) as logger:

            for epoch in range(1, self.num_epochs + 1):
                # 1. Huấn luyện 1 epoch
                train_res = self.train_epoch(epoch, logger=logger)
                train_loss = train_res["loss"]

                # 2. Đánh giá trên tập Validation
                dev_loss, dev_metrics = self.evaluate(self.dev_dataloader)

                # Trích xuất Entity-level F1 chuẩn từ dict metrics
                f1_micro = dev_metrics["overall"]["f1"]
                p_micro = dev_metrics["overall"]["precision"]
                r_micro = dev_metrics["overall"]["recall"]
                f1_macro = dev_metrics.get("macro_f1", 0.0)

                # 3. Ghi nhận log đầy đủ
                log_data = {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/lr": train_res["lr"],
                    "train/epoch_duration_s": train_res["duration"],
                    "dev/loss": dev_loss,
                    "dev/precision": p_micro,
                    "dev/recall": r_micro,
                    "dev/f1_micro": f1_micro,
                    "dev/f1_macro": f1_macro,
                }
                if torch.cuda.is_available():
                    log_data["gpu/max_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

                logger.log(log_data, step=epoch)
                logger.log_ner_metrics(dev_metrics, prefix="dev", step=epoch)

                print(
                    f"Epoch {epoch:02d}/{self.num_epochs:02d} [{train_res['duration']:.1f}s] | "
                    f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | "
                    f"Dev P: {p_micro:.4f} | Dev R: {r_micro:.4f} | "
                    f"Dev F1 (micro): {f1_micro:.4f} | Dev F1 (macro): {f1_macro:.4f}"
                )

                # 4. Kiểm tra và lưu checkpoint tốt nhất
                best_ckpt_path = self.checkpoint_dir / "best_model.pt"

                if f1_micro > self.best_f1:
                    improvement = f1_micro - self.best_f1
                    self.best_f1 = f1_micro
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    self.save_checkpoint(best_ckpt_path, epoch, dev_metrics, is_best=True)
                    print(f"  >>> [*] Kỷ lục mới! Đã lưu checkpoint: {best_ckpt_path} (F1: {self.best_f1:.4f}, +{improvement:.4f})")
                else:
                    self.patience_counter += 1
                    print(f"  [-] F1 không tăng ({self.patience_counter}/{self.early_stopping_patience} epochs)")
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\n[Early Stopping] Dừng sớm tại epoch {epoch} vì metric không cải thiện sau {self.early_stopping_patience} epochs.")
                        break

            total_duration = time.time() - total_training_start
            logger.log_summary({
                "best_dev_f1_micro": self.best_f1,
                "best_epoch": self.best_epoch,
                "total_training_time_s": total_duration,
            })

            print(f"\n=======================================================")
            print(f"HOÀN THÀNH HUẤN LUYỆN: {self.exp_name}")
            print(f"F1 (micro) Tốt nhất: {self.best_f1:.4f} (tại epoch {self.best_epoch})")
            print(f"Tổng thời gian: {total_duration:.1f}s")
            print(f"=======================================================\n")

        return {
            "best_f1_micro": self.best_f1,
            "best_epoch": self.best_epoch,
            "total_duration": total_duration,
        }
