"""
attention_viz.py - Trích xuất, kiểm định và trực quan hóa Attention Weights cho nhánh Pretrained PLM
(PhoBERT-base-v2 và XLM-RoBERTa-base).

Tính năng chính:
  1. Kiểm tra kiến trúc mô hình (inspect_model_attention_spec):
     - Xác nhận hỗ trợ output_attentions, số layers, số heads, hidden_size, head_dim, tokenizer.
  2. Trích xuất Attention không làm thay đổi trạng thái train/eval của mô hình (extract_sentence_attention):
     - Hỗ trợ đầu vào là danh sách token đã tách (List[str]) hoặc câu tiếng Việt thô (str).
     - Chọn tầng bất kỳ (hỗ trợ chỉ số âm như -1), chọn head cụ thể hoặc gộp heads ("mean", "max", "all").
     - Tích hợp phục hồi nhãn NER dự đoán (recover_word_predictions) nếu truyền id2tag.
  3. Tổng hợp Attention từ mức Subword về mức Từ gốc (aggregate_subword_attention_to_words):
     - Sử dụng word_ids chuẩn từ subword_align.py.
     - Công thức minh bạch: Key aggregation ("sum" | "mean" | "first") + Query aggregation ("mean" | "first" | "max")
       kèm tùy chọn chuẩn hóa lại tổng hàng (renormalize) và ghi nhận tỷ lệ khối lượng chú ý vào special tokens.
     - Hỗ trợ Attention Rollout đa tầng (Abnar & Zuidema, 2020).
  4. Trực quan hóa chất lượng cao (300 DPI) lưu tại outputs/figures/attention/:
     - Attention Heatmap (mức subword hoặc mức từ, gắn nhãn thực thể BIO trên trục).
     - So sánh nhiều Attention Heads trong cùng 1 tầng (plot_head_comparison).
     - So sánh nhiều Transformer Layers (plot_layer_comparison).
     - Đối chiếu trực tiếp PhoBERT vs XLM-R trên cùng một câu kiểm thử (plot_model_attention_comparison).
  5. Kiểm định tính hợp lệ toàn diện (validate_attention_result).

Lưu ý học thuật:
  Attention weights chỉ phản ánh phân bố trọng số tổng hợp ngữ cảnh bên trong cơ chế self-attention
  và KHÔNG phải bằng chứng nhân quả hay lời giải thích hoàn chỉnh cho dự đoán của mô hình
  (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.common.schema import NERSample, load_jsonl
from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.subword_align import (
    recover_word_predictions,
    tokenize_and_get_word_ids,
)

DEFAULT_OUTPUT_DIR = Path("outputs/figures/attention")

# Bảng màu hiển thị nhãn thực thể trên trục heatmap (dễ phân biệt cho cả VLSP và PhoNER)
ENTITY_LABEL_COLORS: Dict[str, str] = {
    "PER": "#b91c1c",
    "NAME": "#b91c1c",
    "LOC": "#1d4ed8",
    "LOCATION": "#1d4ed8",
    "ORG": "#15803d",
    "ORGANIZATION": "#15803d",
    "MISC": "#7e22ce",
    "PATIENT_ID": "#0f766e",
    "DATE": "#b45309",
    "AGE": "#9a3412",
    "GENDER": "#be185d",
    "JOB": "#6d28d9",
    "SYMPTOM_AND_DISEASE": "#c2410c",
    "TRANSPORTATION": "#0369a1",
}


# --------------------------------------------------------------------------- #
# 1. Cấu trúc dữ liệu & Kiểm tra kiến trúc mô hình
# --------------------------------------------------------------------------- #
@dataclass
class ModelAttentionSpec:
    """Thông số kiến trúc liên quan đến cơ chế Self-Attention của mô hình PLM."""
    model_name: str
    encoder_class: str
    tokenizer_class: str
    is_fast_tokenizer: bool
    num_layers: int
    num_heads: int
    hidden_size: int
    head_dim: int
    supports_output_attentions: bool
    attention_tuple_len: int
    sample_layer_shape: Tuple[int, ...]


@dataclass
class AttentionExtractionResult:
    """Kết quả trích xuất attention hoàn chỉnh cho một câu tiếng Việt."""
    attentions: torch.Tensor               # [num_layers, num_heads, L, L] (CPU float32)
    selected_attention: torch.Tensor       # [L, L] nếu gộp/chọn 1 head, hoặc [num_heads, L, L] nếu head="all"
    layer_idx: int                         # Chỉ số tầng chuẩn hóa 0-based (0..num_layers-1)
    head_spec: Union[int, str]             # Chỉ số head 0-based hoặc "mean" | "max" | "all"
    subword_tokens: List[str]              # Danh sách subword độ dài L
    word_ids: List[Optional[int]]          # Ánh xạ từng subword -> chỉ số từ gốc (None cho special tokens)
    original_tokens: List[str]             # Danh sách token gốc độ dài W
    model_name: str
    sentence_id: str
    num_layers: int
    num_heads: int
    predicted_tags: Optional[List[str]] = None
    gold_tags: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WordAttentionResult:
    """Ma trận Attention đã tổng hợp từ mức subword về mức từ/âm tiết gốc."""
    matrix: np.ndarray                     # [W', W'] float32
    labels: List[str]                      # Nhãn hiển thị từng vị trí độ dài W'
    word_to_subwords: Dict[int, List[int]] # word_id -> danh sách vị trí subword tương ứng
    query_agg: str                         # "mean" | "first" | "max"
    key_agg: str                           # "sum" | "mean" | "first"
    include_special_tokens: bool
    renormalized: bool
    special_token_mass_removed: float      # Tỷ lệ khối lượng chú ý trung bình rơi vào <s>/</s> trước khi loại bỏ
    method_description: str


def _resolve_model_device(model: nn.Module, device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def inspect_model_attention_spec(
    model: nn.Module,
    tokenizer: Any,
    device: Optional[torch.device] = None,
) -> ModelAttentionSpec:
    """
    Kiểm tra thực nghiệm kiến trúc attention của mô hình (không làm thay đổi trạng thái train/eval).
    """
    dev = _resolve_model_device(model, device)
    was_training = model.training
    model.eval()

    config = getattr(model, "config", None)
    encoder = getattr(model, "encoder", model)
    if config is None and hasattr(encoder, "config"):
        config = encoder.config

    model_name = getattr(model, "model_name_or_path", getattr(config, "_name_or_path", type(model).__name__))
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    num_heads = int(getattr(config, "num_attention_heads", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    head_dim = hidden_size // num_heads if num_heads > 0 else 0

    probe_tokens = ["Hà_Nội", "đẹp", "."]
    input_ids_list, attention_mask_list, _ = tokenize_and_get_word_ids(tokenizer, probe_tokens, max_length=32)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=dev)          # [1, L]
    attention_mask = torch.tensor([attention_mask_list], dtype=torch.long, device=dev)  # [1, L]

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    finally:
        model.train(was_training)

    attentions = outputs["attentions"] if isinstance(outputs, dict) else getattr(outputs, "attentions", None)
    supports = attentions is not None and len(attentions) > 0
    sample_shape = tuple(attentions[0].shape) if supports else ()
    attn_len = len(attentions) if supports else 0

    if num_layers == 0 and attn_len > 0:
        num_layers = attn_len
    if num_heads == 0 and len(sample_shape) == 4:
        num_heads = int(sample_shape[1])

    return ModelAttentionSpec(
        model_name=str(model_name),
        encoder_class=type(encoder).__name__,
        tokenizer_class=type(tokenizer).__name__,
        is_fast_tokenizer=bool(getattr(tokenizer, "is_fast", False)),
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        supports_output_attentions=supports,
        attention_tuple_len=attn_len,
        sample_layer_shape=sample_shape,
    )


# --------------------------------------------------------------------------- #
# 2. Chức năng trích xuất Attention & Kiểm định
# --------------------------------------------------------------------------- #
def _normalize_index(idx: int, total: int, axis_name: str) -> int:
    """Chuẩn hóa chỉ số (hỗ trợ chỉ số âm) và kiểm tra chặt chẽ không vượt biên."""
    norm = idx + total if idx < 0 else idx
    if norm < 0 or norm >= total:
        raise IndexError(
            f"Chỉ số {axis_name}={idx} không hợp lệ! Phạm vi hợp lệ: [{-total}, {total - 1}] (tổng={total})."
        )
    return norm


def select_layer_head_attention(
    all_attentions: torch.Tensor,
    layer_idx: int = -1,
    head: Union[int, str] = "mean",
) -> Tuple[torch.Tensor, int, Union[int, str]]:
    """
    Trích xuất ma trận chú ý từ tensor chuẩn hóa [num_layers, num_heads, L, L].

    Args:
        all_attentions: Tensor [num_layers, num_heads, L, L].
        layer_idx: Chỉ số tầng (0-based hoặc số âm, vd -1 là tầng cuối).
        head: Chỉ số head (int, hỗ trợ số âm) hoặc chiến lược gộp ("mean", "max", "all").

    Returns:
        (selected_tensor, normalized_layer_idx, normalized_head_spec)
    """
    if all_attentions.dim() != 4:
        raise ValueError(
            f"all_attentions phải có 4 chiều [num_layers, num_heads, L, L], nhận {tuple(all_attentions.shape)}"
        )

    num_layers, num_heads, _, _ = all_attentions.shape
    norm_layer = _normalize_index(int(layer_idx), num_layers, axis_name="layer_idx")
    layer_tensor = all_attentions[norm_layer]  # [num_heads, L, L]

    if isinstance(head, (int, np.integer)) and not isinstance(head, bool):
        norm_head = _normalize_index(int(head), num_heads, axis_name="head")
        return layer_tensor[norm_head], norm_layer, norm_head

    head_mode = str(head).lower().strip()
    if head_mode == "mean":
        return layer_tensor.mean(dim=0), norm_layer, "mean"
    elif head_mode == "max":
        return layer_tensor.max(dim=0).values, norm_layer, "max"
    elif head_mode == "all":
        return layer_tensor, norm_layer, "all"
    else:
        raise ValueError(
            f"Giá trị head='{head}' không hợp lệ! Dùng chỉ số int trong [0, {num_heads - 1}] "
            f"hoặc một trong ('mean', 'max', 'all')."
        )


def extract_attention_weights(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    API tương thích ngược: trích xuất ma trận chú ý [num_heads, L, L] của 1 tầng từ input_ids [1, L].
    Bảo toàn nguyên vẹn trạng thái train/eval của mô hình.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            attentions = outputs["attentions"] if isinstance(outputs, dict) else getattr(outputs, "attentions", None)
            if attentions is None or len(attentions) == 0:
                raise ValueError(
                    "Mô hình không trả về attentions. Kiểm tra output_attentions=True trong cấu hình mô hình."
                )
            norm_layer = _normalize_index(layer_idx, len(attentions), axis_name="layer_idx")
            # attentions[norm_layer]: [B, num_heads, L, L] -> lấy mẫu đầu [num_heads, L, L]
            return attentions[norm_layer][0].detach().float().cpu()
    finally:
        model.train(was_training)


def extract_sentence_attention(
    model: nn.Module,
    tokenizer: Any,
    sentence: Union[str, Sequence[str]],
    layer_idx: int = -1,
    head: Union[int, str] = "mean",
    max_length: int = 256,
    id2tag: Optional[Dict[int, str]] = None,
    gold_tags: Optional[Sequence[str]] = None,
    sentence_id: str = "sample",
    segmenter: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> AttentionExtractionResult:
    """
    Trích xuất toàn bộ attention weights ([num_layers, num_heads, L, L]) và ánh xạ subword-to-word
    cho một câu tiếng Việt mà không làm thay đổi trạng thái huấn luyện (train/eval) của mô hình.

    Args:
        model: PretrainedNERModel (PhoBERT hoặc XLM-R).
        tokenizer: Tokenizer tương ứng (PhobertTokenizer hoặc XLMRobertaTokenizerFast).
        sentence: Câu tiếng Việt dạng chuỗi thô (str) hoặc danh sách token gốc (List[str]).
        layer_idx: Chỉ số tầng muốn chọn (hỗ trợ số âm, vd -1 = tầng cuối).
        head: Chỉ số head (int) hoặc cách gộp ("mean", "max", "all").
        max_length: Độ dài subword tối đa (mặc định 256).
        id2tag: Nếu cung cấp, tự động phục hồi nhãn BIO dự đoán ở mức token gốc.
        gold_tags: Nhãn BIO chuẩn tương ứng (tùy chọn, dùng để gắn trên biểu đồ).
        sentence_id: Định danh câu dùng cho báo cáo và đặt tên file ảnh.
        segmenter: Instance WordSegmenter (tùy chọn nếu sentence là chuỗi thô cần tách từ cho PhoBERT).
        device: Thiết bị thực thi (mặc định tự phát hiện theo tham số của model).
    """
    # 1. Chuẩn hóa danh sách token đầu vào
    if isinstance(sentence, str):
        text = sentence.strip()
        if not text:
            raise ValueError("Câu đầu vào rỗng!")
        if segmenter is not None:
            original_tokens = list(segmenter.segment_text(text))
        else:
            original_tokens = text.split()
    else:
        original_tokens = [str(t) for t in sentence]
        if not original_tokens:
            raise ValueError("Danh sách tokens đầu vào rỗng!")

    # 2. Tokenize và lấy word_ids chuẩn thống nhất với quá trình huấn luyện (subword_align.py)
    input_ids_list, attention_mask_list, word_ids = tokenize_and_get_word_ids(
        tokenizer=tokenizer,
        tokens=original_tokens,
        max_length=max_length,
    )
    subword_tokens = list(tokenizer.convert_ids_to_tokens(input_ids_list))
    seq_len = len(input_ids_list)

    dev = _resolve_model_device(model, device)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=dev)          # [1, L]
    attention_mask = torch.tensor([attention_mask_list], dtype=torch.long, device=dev)  # [1, L]

    # 3. Forward trong torch.no_grad() và khôi phục tuyệt đối trạng thái train/eval
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
    finally:
        model.train(was_training)

    attentions_tuple = outputs["attentions"] if isinstance(outputs, dict) else getattr(outputs, "attentions", None)
    if attentions_tuple is None or len(attentions_tuple) == 0:
        raise RuntimeError(
            "Model không trả về attentions! Kiểm tra `output_attentions=True` trong cấu hình encoder."
        )

    # Xếp chồng thành tensor 4 chiều [num_layers, num_heads, L, L] trên CPU float32
    stacked_attns = torch.stack(
        [layer_attn[0].detach().float().cpu() for layer_attn in attentions_tuple],
        dim=0,
    )  # [num_layers, num_heads, L, L]

    num_layers, num_heads, attn_l1, attn_l2 = stacked_attns.shape
    if attn_l1 != seq_len or attn_l2 != seq_len:
        raise RuntimeError(
            f"Kích thước ma trận attention ({attn_l1}x{attn_l2}) không khớp số subword ({seq_len})!"
        )

    selected_attn, norm_layer, norm_head = select_layer_head_attention(
        stacked_attns, layer_idx=layer_idx, head=head
    )

    # 4. Phục hồi nhãn NER dự đoán nếu có logits và id2tag
    predicted_tags: Optional[List[str]] = None
    logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
    if id2tag is not None and logits is not None:
        pred_ids = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()  # [L]
        predicted_tags = recover_word_predictions(
            pred_ids=pred_ids,
            word_ids=word_ids,
            num_tokens=len(original_tokens),
            id2tag=id2tag,
            fallback_tag="O",
        )

    model_name = getattr(model, "model_name_or_path", type(model).__name__)

    return AttentionExtractionResult(
        attentions=stacked_attns,
        selected_attention=selected_attn,
        layer_idx=norm_layer,
        head_spec=norm_head,
        subword_tokens=subword_tokens,
        word_ids=list(word_ids),
        original_tokens=original_tokens,
        model_name=str(model_name),
        sentence_id=sentence_id,
        num_layers=num_layers,
        num_heads=num_heads,
        predicted_tags=predicted_tags,
        gold_tags=list(gold_tags) if gold_tags is not None else None,
        metadata={
            "seq_len_subwords": seq_len,
            "num_original_tokens": len(original_tokens),
            "subword_to_word_ratio": round(
                sum(1 for w in word_ids if w is not None) / max(len(original_tokens), 1), 3
            ),
            "truncated": len({w for w in word_ids if w is not None}) < len(original_tokens),
        },
    )


def validate_attention_result(result: AttentionExtractionResult, atol: float = 1e-4) -> Dict[str, Any]:
    """
    Kiểm định toàn diện kết quả trích xuất attention (Yêu cầu mục 6):
      - Tensor không rỗng, đúng 4 chiều [num_layers, num_heads, L, L].
      - Toàn bộ giá trị hữu hạn (không NaN/Inf) và nằm trong [0, 1].
      - Tổng trọng số theo hàng (mỗi query phân bổ lên tất cả keys) bằng 1.0 (± atol).
      - Khớp tuyệt đối số tầng, số head, số subword và độ dài word_ids.
    """
    attns = result.attentions
    if attns is None or attns.numel() == 0:
        raise ValueError("Attention tensor rỗng!")
    if attns.dim() != 4:
        raise ValueError(f"Attention tensor phải có 4 chiều, nhận {tuple(attns.shape)}")

    n_l, n_h, l1, l2 = attns.shape
    if n_l != result.num_layers:
        raise ValueError(f"Lệch số tầng: tensor dim0={n_l} != num_layers={result.num_layers}")
    if n_h != result.num_heads:
        raise ValueError(f"Lệch số head: tensor dim1={n_h} != num_heads={result.num_heads}")
    if l1 != l2 or l1 != len(result.subword_tokens) or l1 != len(result.word_ids):
        raise ValueError(
            f"Lệch độ dài chuỗi: attn={l1}x{l2}, subwords={len(result.subword_tokens)}, word_ids={len(result.word_ids)}"
        )

    if not torch.isfinite(attns).all():
        raise ValueError("Phát hiện giá trị NaN hoặc Inf trong attention tensor!")

    min_val = float(attns.min().item())
    max_val = float(attns.max().item())
    if min_val < -atol or max_val > 1.0 + atol:
        raise ValueError(f"Giá trị attention nằm ngoài đoạn [0, 1]: [{min_val}, {max_val}]")

    row_sums = attns.sum(dim=-1)  # [num_layers, num_heads, L]
    max_row_err = float((row_sums - 1.0).abs().max().item())
    if max_row_err > atol:
        raise ValueError(f"Tổng hàng attention sai lệch vượt ngưỡng {atol}: max_err={max_row_err}")

    if result.predicted_tags is not None and len(result.predicted_tags) != len(result.original_tokens):
        raise ValueError(
            f"Độ dài predicted_tags ({len(result.predicted_tags)}) != original_tokens ({len(result.original_tokens)})"
        )

    return {
        "valid": True,
        "shape": tuple(attns.shape),
        "min_val": min_val,
        "max_val": max_val,
        "max_row_sum_error": max_row_err,
        "num_subwords": l1,
        "num_original_tokens": len(result.original_tokens),
        "num_mapped_words": len({w for w in result.word_ids if w is not None}),
    }


# --------------------------------------------------------------------------- #
# 3. Xử lý Subword -> Word & Attention Rollout
# --------------------------------------------------------------------------- #
def aggregate_subword_attention_to_words(
    attn_matrix: Union[torch.Tensor, np.ndarray],
    word_ids: Sequence[Optional[int]],
    original_tokens: Sequence[str],
    subword_tokens: Optional[Sequence[str]] = None,
    query_agg: str = "mean",
    key_agg: str = "sum",
    include_special_tokens: bool = False,
    renormalize: bool = True,
) -> WordAttentionResult:
    """
    Tổng hợp ma trận attention mức subword [L, L] sang mức từ gốc [W', W'] bằng ánh xạ word_ids.

    Phương pháp tổng hợp toán học minh bạch (theo chuẩn Clark et al., 2019; Vig, 2019):
      - Gọi S(u) là tập các chỉ số subword thuộc về từ thứ u (dựa trên word_ids).
      - Bước 1 (Trục Key - cột): Tổng hợp chú ý mà subword truy vấn i hướng tới tất cả subwords
        thuộc từ khóa v:
          * key_agg="sum" (mặc định):  A_tilde[i, v] = sum_{j in S(v)} A[i, j]
            (bảo toàn tổng xác suất phân bổ vào toàn bộ từ ghép v).
          * key_agg="mean":            A_tilde[i, v] = mean_{j in S(v)} A[i, j]
          * key_agg="first":           A_tilde[i, v] = A[i, min(S(v))]
      - Bước 2 (Trục Query - hàng): Tổng hợp phân bố chú ý phát ra từ các subwords thuộc từ u:
          * query_agg="mean" (mặc định): W[u, v] = mean_{i in S(u)} A_tilde[i, v]
          * query_agg="first":           W[u, v] = A_tilde[min(S(u)), v] (khớp subword đầu dùng phân loại NER)
          * query_agg="max":             W[u, v] = max_{i in S(u)} A_tilde[i, v]
      - Bước 3 (Chuẩn hóa hàng): Nếu renormalize=True (nhất là khi bỏ <s>/</s>), chuẩn hóa
        mỗi hàng W[u, :] / sum_v W[u, v] để biểu diễn phân bố chú ý tương đối giữa các từ nội dung.
    """
    attn = attn_matrix.detach().float().cpu().numpy() if isinstance(attn_matrix, torch.Tensor) else np.asarray(attn_matrix, dtype=np.float32)
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"attn_matrix phải là ma trận vuông 2D [L, L], nhận {attn.shape}")
    seq_len = attn.shape[0]
    if len(word_ids) != seq_len:
        raise ValueError(f"len(word_ids)={len(word_ids)} != attn_matrix.shape[0]={seq_len}")
    if query_agg not in ("mean", "first", "max"):
        raise ValueError(f"query_agg không hợp lệ: '{query_agg}'. Chọn trong ('mean', 'first', 'max').")
    if key_agg not in ("sum", "mean", "first"):
        raise ValueError(f"key_agg không hợp lệ: '{key_agg}'. Chọn trong ('sum', 'mean', 'first').")

    # Gom nhóm các chỉ số subword theo từng word_id
    word_to_sw: Dict[int, List[int]] = {}
    special_indices: List[int] = []
    for sw_i, wid in enumerate(word_ids):
        if wid is None:
            special_indices.append(sw_i)
        else:
            word_to_sw.setdefault(int(wid), []).append(sw_i)

    # Tính tỷ lệ khối lượng chú ý trung bình rơi vào special tokens (<s>, </s>) từ các token nội dung
    content_sw_indices = [i for i, w in enumerate(word_ids) if w is not None]
    if special_indices and content_sw_indices:
        special_mass = float(attn[np.ix_(content_sw_indices, special_indices)].sum(axis=1).mean())
    else:
        special_mass = 0.0

    # Xây dựng danh sách nhóm (label, list_of_sw_indices) theo đúng thứ tự xuất hiện
    groups: List[Tuple[str, List[int]]] = []
    visited_words = set()
    for sw_i, wid in enumerate(word_ids):
        if wid is None:
            if include_special_tokens:
                lbl = subword_tokens[sw_i] if (subword_tokens is not None and sw_i < len(subword_tokens)) else "[SPEC]"
                groups.append((lbl, [sw_i]))
        else:
            w_int = int(wid)
            if w_int not in visited_words:
                visited_words.add(w_int)
                tok_lbl = original_tokens[w_int] if w_int < len(original_tokens) else f"w{w_int}"
                groups.append((tok_lbl, word_to_sw[w_int]))

    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("Không có token hợp lệ nào để tổng hợp từ word_ids!")

    # Bước 1: Tổng hợp trục Key (cột) -> [L, n_groups]
    key_collapsed = np.zeros((seq_len, n_groups), dtype=np.float32)
    for g_col, (_, sw_cols) in enumerate(groups):
        if key_agg == "sum":
            key_collapsed[:, g_col] = attn[:, sw_cols].sum(axis=1)
        elif key_agg == "mean":
            key_collapsed[:, g_col] = attn[:, sw_cols].mean(axis=1)
        elif key_agg == "first":
            key_collapsed[:, g_col] = attn[:, sw_cols[0]]

    # Bước 2: Tổng hợp trục Query (hàng) -> [n_groups, n_groups]
    word_attn = np.zeros((n_groups, n_groups), dtype=np.float32)
    for g_row, (_, sw_rows) in enumerate(groups):
        if query_agg == "mean":
            word_attn[g_row, :] = key_collapsed[sw_rows, :].mean(axis=0)
        elif query_agg == "first":
            word_attn[g_row, :] = key_collapsed[sw_rows[0], :]
        elif query_agg == "max":
            word_attn[g_row, :] = key_collapsed[sw_rows, :].max(axis=0)

    # Bước 3: Chuẩn hóa hàng nếu yêu cầu
    if renormalize:
        row_sums = word_attn.sum(axis=1, keepdims=True)
        word_attn = np.divide(word_attn, row_sums, out=np.zeros_like(word_attn), where=(row_sums > 1e-12))

    spec_note = "có giữ <s>/</s>" if include_special_tokens else f"loại bỏ <s>/</s> (khối lượng sink TB={special_mass*100:.1f}%)"
    renorm_note = ", chuẩn hóa tổng hàng=1" if renormalize else ", giữ nguyên tỷ lệ gốc"
    method_desc = f"Key={key_agg}, Query={query_agg} ({spec_note}{renorm_note})"

    return WordAttentionResult(
        matrix=word_attn,
        labels=[g[0] for g in groups],
        word_to_subwords=word_to_sw,
        query_agg=query_agg,
        key_agg=key_agg,
        include_special_tokens=include_special_tokens,
        renormalized=renormalize,
        special_token_mass_removed=special_mass,
        method_description=method_desc,
    )


def compute_attention_rollout(
    all_attentions: torch.Tensor,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """
    Tính Attention Rollout (Abnar & Zuidema, 2020) tích hợp kết nối dư (residual connection)
    qua nhiều tầng Transformer:
        A_hat^(l) = 0.5 * A^(l) + 0.5 * I
        Rollout   = A_hat^(end) @ ... @ A_hat^(start)
    """
    if all_attentions.dim() != 4:
        raise ValueError(f"Cần tensor 4 chiều [num_layers, num_heads, L, L], nhận {tuple(all_attentions.shape)}")
    num_layers, _, seq_len, _ = all_attentions.shape
    s_l = _normalize_index(start_layer, num_layers, "start_layer")
    e_l = _normalize_index(num_layers - 1 if end_layer is None else end_layer, num_layers, "end_layer")
    if s_l > e_l:
        raise ValueError(f"start_layer ({s_l}) phải <= end_layer ({e_l})")

    eye = torch.eye(seq_len, dtype=torch.float32)
    rollout = eye.clone()
    for l in range(s_l, e_l + 1):
        layer_a = all_attentions[l].float()  # [H, L, L]
        if head_fusion == "mean":
            fused = layer_a.mean(dim=0)
        elif head_fusion == "max":
            fused = layer_a.max(dim=0).values
        else:
            raise ValueError(f"head_fusion không hợp lệ: '{head_fusion}'")
        aug = 0.5 * fused + 0.5 * eye
        aug = aug / aug.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        rollout = torch.matmul(aug, rollout)
    return rollout


# --------------------------------------------------------------------------- #
# 4. Tiện ích đặt tên file & Trực quan hóa chất lượng cao (300 DPI)
# --------------------------------------------------------------------------- #
def _sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()


def short_model_tag(model_name: str) -> str:
    low = model_name.lower()
    if "phobert" in low:
        return "phobert"
    if "xlm" in low:
        return "xlmr"
    return _sanitize_slug(model_name.split("/")[-1])


def build_attention_filename(
    output_dir: Union[str, Path],
    model_name: str,
    sentence_id: str,
    layer_spec: Union[int, str],
    head_spec: Union[int, str],
    level: str = "word",
) -> Path:
    """
    Quy ước đặt tên file ảnh chứa đầy đủ thông tin:
      <model>_<sentence_id>_L<layer>_H<head>_<level>.png
    Ví dụ:
      outputs/figures/attention/phobert_vlsp_word_test_00627_L12_Hmean_word.png
    """
    m_tag = short_model_tag(model_name)
    sid = _sanitize_slug(sentence_id)
    l_tag = f"L{int(layer_spec) + 1:02d}" if isinstance(layer_spec, (int, np.integer)) else f"L{_sanitize_slug(str(layer_spec))}"
    h_tag = f"H{int(head_spec):02d}" if isinstance(head_spec, (int, np.integer)) and not isinstance(head_spec, bool) else f"H{_sanitize_slug(str(head_spec))}"
    lv_tag = _sanitize_slug(level)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{m_tag}_{sid}_{l_tag}_{h_tag}_{lv_tag}.png"


def _format_axis_labels_with_tags(
    tokens: Sequence[str],
    tags: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Tạo nhãn hiển thị kèm tag thực thể và danh sách màu sắc cho trục tọa độ."""
    labels: List[str] = []
    colors: List[str] = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("\u2581", "▁")
        if tags is not None and i < len(tags) and tags[i] != "O":
            tag = tags[i]
            etype = tag.split("-", 1)[1] if "-" in tag else tag
            labels.append(f"{clean_tok} [{tag}]")
            colors.append(ENTITY_LABEL_COLORS.get(etype, "#b91c1c"))
        else:
            labels.append(clean_tok)
            colors.append("#1f2937")
    return labels, colors


def _apply_tick_colors(ax: plt.Axes, colors: Sequence[str]) -> None:
    for tick_lbl, col in zip(ax.get_xticklabels(), colors):
        tick_lbl.set_color(col)
        if col != "#1f2937":
            tick_lbl.set_fontweight("bold")
    for tick_lbl, col in zip(ax.get_yticklabels(), colors):
        tick_lbl.set_color(col)
        if col != "#1f2937":
            tick_lbl.set_fontweight("bold")


def plot_attention_heatmap(
    tokens: Sequence[str],
    attention_matrix: Union[torch.Tensor, np.ndarray],
    head_idx: int = 0,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "Blues",
    annot: bool = False,
) -> Path | None:
    """
    Vẽ Attention Heatmap chất lượng cao (300 DPI) cho một ma trận [L, L] (hoặc [H, L, L] chọn head_idx).
    Trục tung (Y): Query token (từ đang truy vấn ngữ cảnh).
    Trục hoành (X): Key token (từ được chú ý tới).
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    if isinstance(attention_matrix, torch.Tensor):
        mat = attention_matrix[head_idx].detach().float().cpu().numpy() if attention_matrix.dim() == 3 else attention_matrix.detach().float().cpu().numpy()
    else:
        arr = np.asarray(attention_matrix, dtype=np.float32)
        mat = arr[head_idx] if arr.ndim == 3 else arr

    seq_len = min(len(tokens), mat.shape[0], mat.shape[1])
    mat = mat[:seq_len, :seq_len]
    disp_tokens = list(tokens[:seq_len])
    disp_tags = list(tags[:seq_len]) if tags is not None else None
    axis_labels, axis_colors = _format_axis_labels_with_tags(disp_tokens, disp_tags)

    if figsize is None:
        side = max(7.2, min(14.0, seq_len * 0.52 + 2.5))
        figsize = (side + 1.0, side)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=0.0, vmax=max(float(mat.max()), 1e-6))

    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(axis_labels, rotation=45, ha="right", fontsize=8.8)
    ax.set_yticklabels(axis_labels, rotation=0, fontsize=8.8)
    _apply_tick_colors(ax, axis_colors)

    ax.set_xlabel("Key Token (được chú ý tới)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Query Token (đang truy vấn)", fontsize=10, fontweight="bold", labelpad=8)

    # Kẻ lưới mảnh phân tách ô
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which="minor", color="#e5e7eb", linestyle="-", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annot and seq_len <= 20:
        thresh = mat.max() * 0.55
        for i in range(seq_len):
            for j in range(seq_len):
                val = mat[i, j]
                if val >= 0.05:
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=6.8,
                        color="white" if val > thresh else "#111827",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Attention Weight", fontsize=9.5)

    main_title = title or f"Transformer Attention Map (Head {head_idx})"
    if subtitle:
        ax.set_title(f"{main_title}\n{subtitle}", fontsize=11, fontweight="bold", pad=12)
    else:
        ax.set_title(main_title, fontsize=11.5, fontweight="bold", pad=12)

    plt.tight_layout()
    saved_path: Optional[Path] = None
    if output_path:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saved_path


def plot_head_comparison(
    result: AttentionExtractionResult,
    layer_idx: int = -1,
    heads: Sequence[int] = (0, 2, 4, 6, 8, 11),
    word_level: bool = True,
    query_agg: str = "mean",
    key_agg: str = "sum",
    include_special_tokens: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = "Blues",
) -> Path | None:
    """
    Trực quan hóa so sánh nhiều Attention Heads trong cùng 1 tầng Transformer (dạng lưới).
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    norm_layer = _normalize_index(layer_idx, result.num_layers, "layer_idx")
    norm_heads = [_normalize_index(h, result.num_heads, "head") for h in heads]

    n_plots = len(norm_heads)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))

    matrices: List[np.ndarray] = []
    sink_masses: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    method_note = "Subword-level raw attention"

    for h in norm_heads:
        raw_mat = result.attentions[norm_layer, h]  # [L, L]
        if word_level:
            w_res = aggregate_subword_attention_to_words(
                raw_mat,
                result.word_ids,
                result.original_tokens,
                subword_tokens=result.subword_tokens,
                query_agg=query_agg,
                key_agg=key_agg,
                include_special_tokens=include_special_tokens,
                renormalize=True,
            )
            matrices.append(w_res.matrix)
            sink_masses.append(w_res.special_token_mass_removed)
            if not labels:
                tags_use = result.predicted_tags if not include_special_tokens else None
                labels, colors = _format_axis_labels_with_tags(w_res.labels, tags_use)
                spec_desc = "giữ <s>/</s>" if include_special_tokens else "loại bỏ <s>/</s>, chuẩn hóa tổng hàng=1"
                method_note = f"Key={key_agg}, Query={query_agg} ({spec_desc})"
        else:
            matrices.append(raw_mat.numpy())
            sink_masses.append(0.0)
            if not labels:
                labels, colors = _format_axis_labels_with_tags(result.subword_tokens, None)

    seq_len = len(labels)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.2, nrows * 4.9 + 0.8),
        dpi=300,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (ax, h, mat, sink) in enumerate(zip(axes_flat, norm_heads, matrices, sink_masses)):
        im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=0.0, vmax=max(float(mat.max()), 1e-6))
        sink_txt = f" | sink={sink*100:.1f}%" if (word_level and not include_special_tokens) else ""
        ax.set_title(f"Layer {norm_layer + 1} — Head {h}{sink_txt}", fontsize=10.2, fontweight="bold")
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7.0)
        ax.set_yticklabels(labels if idx % ncols == 0 else [], fontsize=7.0)
        if idx % ncols == 0:
            _apply_tick_colors(ax, colors)
        else:
            for tick_lbl, col in zip(ax.get_xticklabels(), colors):
                tick_lbl.set_color(col)
                if col != "#1f2937":
                    tick_lbl.set_fontweight("bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"So sánh Attention Heads — {result.model_name} | {result.sentence_id} | Layer {norm_layer + 1}/{result.num_layers}\n"
        f"({method_note})",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    saved_path: Optional[Path] = None
    if output_path:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saved_path


def plot_layer_comparison(
    result: AttentionExtractionResult,
    layers: Sequence[int] = (0, 3, 7, 11),
    head: Union[int, str] = "mean",
    word_level: bool = True,
    query_agg: str = "mean",
    key_agg: str = "sum",
    include_special_tokens: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = "Blues",
) -> Path | None:
    """
    Trực quan hóa diễn biến Attention qua các tầng Transformer (đầu -> giữa -> cuối).
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    norm_layers = [_normalize_index(l, result.num_layers, "layer_idx") for l in layers]

    n_plots = len(norm_layers)
    ncols = min(2, n_plots) if n_plots == 4 else min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))

    matrices: List[np.ndarray] = []
    sink_masses: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    method_note = "Subword-level"

    for l in norm_layers:
        sel, _, norm_h = select_layer_head_attention(result.attentions, layer_idx=l, head=head)
        if sel.dim() != 2:
            raise ValueError("plot_layer_comparison yêu cầu head là chỉ số int, 'mean' hoặc 'max' (không dùng 'all').")
        if word_level:
            w_res = aggregate_subword_attention_to_words(
                sel,
                result.word_ids,
                result.original_tokens,
                subword_tokens=result.subword_tokens,
                query_agg=query_agg,
                key_agg=key_agg,
                include_special_tokens=include_special_tokens,
                renormalize=True,
            )
            matrices.append(w_res.matrix)
            sink_masses.append(w_res.special_token_mass_removed)
            if not labels:
                tags_use = result.predicted_tags if not include_special_tokens else None
                labels, colors = _format_axis_labels_with_tags(w_res.labels, tags_use)
                method_note = f"Key={key_agg}, Query={query_agg}"
        else:
            matrices.append(sel.numpy())
            sink_masses.append(0.0)
            if not labels:
                labels, colors = _format_axis_labels_with_tags(result.subword_tokens, None)

    seq_len = len(labels)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.8, nrows * 5.2 + 0.8),
        dpi=300,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (ax, l, mat, sink) in enumerate(zip(axes_flat, norm_layers, matrices, sink_masses)):
        im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=0.0, vmax=max(float(mat.max()), 1e-6))
        sink_txt = f" | <s>/</s> mass={sink*100:.1f}%" if (word_level and not include_special_tokens) else ""
        ax.set_title(f"Layer {l + 1}/{result.num_layers} (Head={head}){sink_txt}", fontsize=10.2, fontweight="bold")
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=7.6)
        ax.set_yticklabels(labels, fontsize=7.6)
        _apply_tick_colors(ax, colors)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"Diễn biến Attention qua các tầng — {result.model_name} | {result.sentence_id}\n"
        f"({method_note}, {'Word-level' if word_level else 'Subword-level'})",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    saved_path: Optional[Path] = None
    if output_path:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saved_path


def plot_model_attention_comparison(
    res_left: AttentionExtractionResult,
    res_right: AttentionExtractionResult,
    layer_idx: int = -1,
    head: Union[int, str] = "mean",
    query_agg: str = "mean",
    key_agg: str = "sum",
    include_special_tokens: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = "Blues",
) -> Path | None:
    """
    Đối chiếu trực tiếp bản đồ Attention của 2 mô hình (ví dụ PhoBERT mức từ vs XLM-R mức âm tiết)
    trên cùng một câu tiếng Việt.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8), dpi=300)

    for ax, res in zip(axes, (res_left, res_right)):
        sel, norm_l, _ = select_layer_head_attention(res.attentions, layer_idx=layer_idx, head=head)
        w_res = aggregate_subword_attention_to_words(
            sel,
            res.word_ids,
            res.original_tokens,
            subword_tokens=res.subword_tokens,
            query_agg=query_agg,
            key_agg=key_agg,
            include_special_tokens=include_special_tokens,
            renormalize=True,
        )
        tags_use = res.predicted_tags if not include_special_tokens else None
        labels, colors = _format_axis_labels_with_tags(w_res.labels, tags_use)
        n = len(labels)

        im = ax.imshow(w_res.matrix, cmap=cmap, aspect="equal", vmin=0.0, vmax=max(float(w_res.matrix.max()), 1e-6))
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=7.8)
        ax.set_yticklabels(labels, fontsize=7.8)
        _apply_tick_colors(ax, colors)
        ax.set_xlabel("Key Token", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Query Token", fontsize=9.5, fontweight="bold")
        ax.set_title(
            f"{res.model_name} ({res.sentence_id})\n"
            f"Layer {norm_l + 1} | Head={head} | {len(res.original_tokens)} tokens -> {len(res.subword_tokens)} subwords "
            f"(sink={w_res.special_token_mass_removed*100:.1f}%)",
            fontsize=9.8,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)

    fig.suptitle(
        f"Đối chiếu Attention: {short_model_tag(res_left.model_name).upper()} vs {short_model_tag(res_right.model_name).upper()} "
        f"(Key={key_agg}, Query={query_agg}, chuẩn hóa hàng)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    saved_path: Optional[Path] = None
    if output_path:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saved_path


# --------------------------------------------------------------------------- #
# 5. Tiện ích nạp Checkpoint & CLI
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer_for_viz(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[PretrainedNERModel, Any, Dict[int, str], Dict[str, Any]]:
    """Nạp PretrainedNERModel (hỗ trợ cả checkpoint fp32 và fp16) kèm Tokenizer chuẩn."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_file}")

    try:
        ckpt = torch.load(ckpt_file, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_file, map_location=dev)

    cfg = ckpt["config"]
    id2tag = {int(k): v for k, v in ckpt["id2tag"].items()}
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("model_name_or_path", "vinai/phobert-base-v2")
    pooling_strategy = model_cfg.get("pooling_strategy", "last")
    use_fast = False if "phobert" in model_name.lower() else True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    model = PretrainedNERModel(
        model_name_or_path=model_name,
        num_labels=len(id2tag),
        pooling_strategy=pooling_strategy,
        classifier_dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()
    return model, tokenizer, id2tag, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất và trực quan hóa Attention Map cho PhoBERT & XLM-R")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt",
        help="Đường dẫn tệp checkpoint .pt",
    )
    parser.add_argument("--gold", type=str, default=None, help="Đường dẫn file JSONL chứa câu cần phân tích")
    parser.add_argument("--sentence-id", type=str, default=None, help="ID câu cần lấy từ --gold (vd: vlsp_word_test_00627)")
    parser.add_argument("--text", type=str, default=None, help="Chuỗi token cách nhau bằng khoảng trắng (nếu không dùng --gold)")
    parser.add_argument("--layer", type=int, default=-1, help="Chỉ số tầng (0-based hoặc âm, mặc định -1)")
    parser.add_argument("--head", type=str, default="mean", help="Chỉ số head (0..11) hoặc 'mean'/'max'")
    parser.add_argument("--level", choices=["word", "subword"], default="word", help="Mức hiển thị heatmap")
    parser.add_argument("--query-agg", choices=["mean", "first", "max"], default="mean")
    parser.add_argument("--key-agg", choices=["sum", "mean", "first"], default="sum")
    parser.add_argument("--compare-heads", action="store_true", help="Xuất thêm biểu đồ so sánh nhiều heads")
    parser.add_argument("--compare-layers", action="store_true", help="Xuất thêm biểu đồ so sánh nhiều layers")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else None
    model, tokenizer, id2tag, _ = load_model_and_tokenizer_for_viz(args.checkpoint, device=dev)

    gold_tags: Optional[List[str]] = None
    sid = args.sentence_id or "custom_sent"
    if args.gold:
        samples = list(load_jsonl(args.gold))
        target: Optional[NERSample] = None
        if args.sentence_id:
            for s in samples:
                if s.id == args.sentence_id:
                    target = s
                    break
            if target is None:
                raise ValueError(f"Không tìm thấy sentence_id='{args.sentence_id}' trong {args.gold}")
        else:
            target = samples[0]
            sid = target.id
        tokens_input: Union[str, List[str]] = target.tokens
        gold_tags = target.tags
    elif args.text:
        tokens_input = args.text.split()
    else:
        tokens_input = ["Ban", "biên_tập", "Tuổi_Trẻ", "quyết_định", "cử", "Uyên", "Ly", "sang", "Mỹ", "."]
        sid = "demo_sample"

    head_arg: Union[int, str] = int(args.head) if args.head.lstrip("-").isdigit() else args.head
    res = extract_sentence_attention(
        model=model,
        tokenizer=tokenizer,
        sentence=tokens_input,
        layer_idx=args.layer,
        head=head_arg,
        id2tag=id2tag,
        gold_tags=gold_tags,
        sentence_id=sid,
        device=dev,
    )
    val_report = validate_attention_result(res)
    print(f"[*] Validation OK: {val_report}")

    out_dir = Path(args.output_dir)
    if args.level == "word":
        w_res = aggregate_subword_attention_to_words(
            res.selected_attention,
            res.word_ids,
            res.original_tokens,
            subword_tokens=res.subword_tokens,
            query_agg=args.query_agg,
            key_agg=args.key_agg,
            include_special_tokens=False,
            renormalize=True,
        )
        out_path = build_attention_filename(out_dir, res.model_name, sid, res.layer_idx, res.head_spec, level="word")
        plot_attention_heatmap(
            tokens=w_res.labels,
            attention_matrix=w_res.matrix,
            output_path=out_path,
            title=f"{res.model_name} | {sid} | Layer {res.layer_idx + 1}/{res.num_layers} | Head={res.head_spec}",
            subtitle=w_res.method_description,
            tags=res.predicted_tags,
            annot=(len(w_res.labels) <= 16),
        )
    else:
        out_path = build_attention_filename(out_dir, res.model_name, sid, res.layer_idx, res.head_spec, level="subword")
        plot_attention_heatmap(
            tokens=res.subword_tokens,
            attention_matrix=res.selected_attention,
            output_path=out_path,
            title=f"{res.model_name} | {sid} | Layer {res.layer_idx + 1}/{res.num_layers} | Head={res.head_spec} (Subword)",
            annot=False,
        )
    print(f"[*] Đã lưu heatmap: {out_path}")

    if args.compare_heads:
        h_path = build_attention_filename(out_dir, res.model_name, sid, res.layer_idx, "heads_grid", level=args.level)
        plot_head_comparison(
            res,
            layer_idx=res.layer_idx,
            word_level=(args.level == "word"),
            query_agg=args.query_agg,
            key_agg=args.key_agg,
            output_path=h_path,
        )
        print(f"[*] Đã lưu so sánh heads: {h_path}")

    if args.compare_layers:
        l_path = build_attention_filename(out_dir, res.model_name, sid, "1_4_8_12", res.head_spec, level=args.level)
        plot_layer_comparison(
            res,
            layers=(0, 3, 7, res.num_layers - 1),
            head=res.head_spec,
            word_level=(args.level == "word"),
            query_agg=args.query_agg,
            key_agg=args.key_agg,
            output_path=l_path,
        )
        print(f"[*] Đã lưu so sánh layers: {l_path}")


if __name__ == "__main__":
    main()
