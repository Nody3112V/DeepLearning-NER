"""
predictor.py - Đóng gói mô hình PLM thành hàm dự đoán chuẩn phục vụ tích hợp Demo và Evaluation.

Tuân thủ giao diện thống nhất đã chốt với Thành viên 3:
  1. predict_tokens(tokens: List[str]) -> List[str]:
     Nhận danh sách token (âm tiết hoặc từ đã tách), trả về chuỗi nhãn BIO tương ứng.
     Dùng cho luồng đánh giá (eval), kiểm thử và phân tích lỗi chéo.
  2. predict(raw_text: str) -> List[Dict[str, Any]]:
     Nhận chuỗi văn bản thô, tự động tách từ (nếu mô hình ở cấp độ từ - word-level)
     qua VnCoreNLP, trả về danh sách các thực thể đã trích xuất:
     [
         {"text": "Hà Nội", "type": "LOC", "start": 10, "end": 17},
         {"text": "Nguyễn Văn A", "type": "PER", "start": 25, "end": 37}
     ]
     Dùng cho giao diện demo (demo/app.py).
"""
from typing import List, Dict, Any, Optional
import os
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.models.pretrained.model import PretrainedNERModel
from src.common.word_segmenter import WordSegmenter, get_segmenter


class PLMPredictor:
    """Wrapper phục vụ suy luận mô hình Pretrained Language Model (PhoBERT / XLM-R)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        segmenter: Optional[WordSegmenter] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Nạp checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt["config"]
        self.id2tag: Dict[int, str] = {int(k): v for k, v in ckpt["id2tag"].items()}
        self.tag2id: Dict[str, int] = {v: k for k, v in self.id2tag.items()}

        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})
        model_name = model_cfg.get("model_name_or_path", "vinai/phobert-base-v2")
        self.granularity = data_cfg.get("granularity", "word")

        # Khởi tạo Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False if "phobert" in model_name.lower() else True,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False if "phobert" in model_name.lower() else True,
            )

        # Khởi tạo mô hình
        self.model = PretrainedNERModel(
            model_name_or_path=model_name,
            num_labels=len(self.id2tag),
            pooling_strategy=model_cfg.get("pooling_strategy", "last"),
            classifier_dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Bộ tách từ (nếu mô hình ở mức word và cần cho predict(raw_text))
        self._segmenter = segmenter

    def _get_segmenter(self) -> WordSegmenter:
        """Khởi tạo lười hoặc dùng singleton WordSegmenter để tiết kiệm RAM."""
        if self._segmenter is None:
            self._segmenter = get_segmenter()
        return self._segmenter

    @torch.no_grad()
    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """
        Dự đoán nhãn BIO cho danh sách tokens đầu vào (luồng eval / batch evaluation).

        Args:
            tokens: Danh sách token (từ đã ghép bằng '_' hoặc âm tiết).

        Returns:
            List[str]: Danh sách nhãn BIO tương ứng với từng token, độ dài 1-1.
        """
        if not tokens:
            return []

        # Căn chỉnh subwords theo chuẩn subword_align.py
        subword_ids = []
        token_to_subword_indices = []

        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id

        if cls_id is not None:
            subword_ids.append(cls_id)

        for tok in tokens:
            sw_ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if not sw_ids:
                sw_ids = [self.tokenizer.unk_token_id]
            token_to_subword_indices.append(len(subword_ids))  # Vị trí subword đầu tiên
            subword_ids.extend(sw_ids)

        if sep_id is not None:
            subword_ids.append(sep_id)

        # Chuyển thành tensor
        input_ids = torch.tensor([subword_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # [1, L, num_labels]
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()  # [L]

        # Lấy nhãn tại các vị trí subword đầu tiên của từng token gốc
        pred_tags = []
        for first_sw_idx in token_to_subword_indices:
            if first_sw_idx < len(preds):
                pred_id = int(preds[first_sw_idx])
                pred_tags.append(self.id2tag.get(pred_id, "O"))
            else:
                pred_tags.append("O")

        return pred_tags

    def predict(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Dự đoán thực thể từ chuỗi văn bản thô phục vụ giao diện Demo (demo/app.py).

        Args:
            raw_text: Chuỗi văn bản tiếng Việt chưa tách từ.

        Returns:
            Danh sách thực thể: [{"text": str, "type": str, "start": int, "end": int}]
        """
        if not raw_text or not raw_text.strip():
            return []

        # 1. Tách từ nếu mô hình ở cấp độ từ (word-level)
        if self.granularity == "word":
            seg = self._get_segmenter()
            tokens = seg.segment_text(raw_text)
        else:
            tokens = raw_text.split()

        # 2. Dự đoán nhãn BIO trên các tokens
        tags = self.predict_tokens(tokens)

        # 3. Trích xuất Span thực thể kèm vị trí ký tự trong raw_text
        entities: List[Dict[str, Any]] = []
        cur_type: Optional[str] = None
        cur_tokens: List[str] = []

        def flush_entity():
            nonlocal cur_type, cur_tokens
            if cur_type and cur_tokens:
                # Đổi '_' thành khoảng trắng để ra văn bản hiển thị tự nhiên
                ent_text = " ".join(t.replace("_", " ") for t in cur_tokens)
                # Tìm vị trí ký tự trong raw_text
                # Tìm kiếm chuỗi tương ứng (case-insensitive hoặc tìm dải ký tự đầu tiên khớp)
                start = raw_text.find(ent_text)
                end = start + len(ent_text) if start != -1 else -1
                entities.append({
                    "text": ent_text,
                    "type": cur_type,
                    "start": start,
                    "end": end,
                })
            cur_type = None
            cur_tokens = []

        for tok, tag in zip(tokens, tags):
            if tag == "O":
                flush_entity()
            elif tag.startswith("B-"):
                flush_entity()
                cur_type = tag[2:]
                cur_tokens.append(tok)
            elif tag.startswith("I-"):
                etype = tag[2:]
                if cur_type == etype:
                    cur_tokens.append(tok)
                else:
                    flush_entity()
                    cur_type = etype
                    cur_tokens.append(tok)

        flush_entity()
        return entities


def create_predictor(checkpoint_path: str, segmenter: Optional[WordSegmenter] = None) -> PLMPredictor:
    """Factory function khởi tạo PLMPredictor."""
    return PLMPredictor(checkpoint_path=checkpoint_path, segmenter=segmenter)
