"""
subword_align.py - Căn chỉnh nhãn BIO của token gốc với chuỗi subword của Transformer Tokenizer.

Hỗ trợ đầy đủ:
  - Fast Tokenizers (XLM-RoBERTa, BERT,... có sẵn `encoding.word_ids()`).
  - Slow Tokenizers (PhoBERT / `PhobertTokenizer` không hỗ trợ `word_ids()` dạng fast).

Quy tắc căn chỉnh:
  1. Special tokens (<s>, </s>, [CLS], [SEP], padding) nhận nhãn bỏ qua: `ignore_index` (-100).
  2. Subword đầu tiên của mỗi từ/âm tiết nhận id nhãn tương ứng: `label2id[tag]`.
  3. Các subword tiếp theo của cùng từ đó nhận nhãn bỏ qua: `ignore_index` (-100).
  4. Padding token trong batch nhận: `ignore_index` (-100).
  5. Truncation được xử lý rõ ràng, đảm bảo len(labels) == len(input_ids).
  6. Kiểm tra nghiêm ngặt: mismatch độ dài tokens-tags, nhãn ngoài label2id.
"""
from typing import Dict, List, Optional, Tuple, Any, Sequence, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.common.schema import NERSample, load_jsonl


def align_labels_with_subwords(
    word_labels: Sequence[str],
    word_ids: Sequence[Optional[int]],
    label2id: Dict[str, int],
    ignore_index: int = -100,
) -> List[int]:
    """
    Hàm cốt lõi: Căn chỉnh danh sách nhãn từ gốc theo mảng word_ids do tokenizer cung cấp.

    Args:
        word_labels: Danh sách nhãn BIO của từng từ gốc trong câu (ví dụ: ['B-PER', 'I-PER', 'O']).
        word_ids: Danh sách chỉ số từ tương ứng với từng subword (None cho special/pad tokens).
        label2id: Bảng ánh xạ nhãn chuỗi -> ID số nguyên (ví dụ: {'O': 0, 'B-PER': 1, ...}).
        ignore_index: Giá trị nhãn bỏ qua trong CrossEntropyLoss (mặc định -100).

    Returns:
        List[int]: Danh sách label IDs có độ dài bằng chính xác len(word_ids).

    Raises:
        KeyError: Nếu gặp nhãn không tồn tại trong label2id.
    """
    aligned_labels: List[int] = []
    current_word: Optional[int] = None

    for wid in word_ids:
        if wid is None:
            # Special tokens (<s>, </s>, [CLS], [SEP], <pad>)
            aligned_labels.append(ignore_index)
        elif wid != current_word:
            # Subword đầu tiên của từ mới
            current_word = wid
            if wid >= len(word_labels):
                # Trường hợp đặc biệt (wid vượt quá số nhãn do lỗi dữ liệu ngoài luồng)
                aligned_labels.append(ignore_index)
                continue

            tag = word_labels[wid]
            if tag not in label2id:
                raise KeyError(
                    f"Nhãn '{tag}' tại vị trí từ #{wid} không tồn tại trong bảng label2id! "
                    f"Các nhãn hợp lệ: {list(label2id.keys())}"
                )
            aligned_labels.append(label2id[tag])
        else:
            # Subword thứ 2 trở đi của cùng một từ
            aligned_labels.append(ignore_index)

    return aligned_labels


def tokenize_and_get_word_ids(
    tokenizer: Any,
    tokens: List[str],
    max_length: int = 256,
) -> Tuple[List[int], List[int], List[Optional[int]]]:
    """
    Tokenize danh sách tokens đầu vào và trích xuất word_ids một cách an toàn.
    Tự động tương thích với cả Fast Tokenizer (XLM-R) và Slow Tokenizer (PhoBERT).

    Args:
        tokenizer: Tokenizer của HuggingFace (Fast hoặc Slow).
        tokens: Danh sách token từ/âm tiết gốc.
        max_length: Độ dài tối đa cho phép.

    Returns:
        Tuple gồm (input_ids, attention_mask, word_ids).
    """
    # 1. Thử phương thức Fast Tokenizer (có sẵn word_ids())
    is_fast = getattr(tokenizer, "is_fast", False)
    if is_fast:
        try:
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            word_ids = encoding.word_ids()
            return input_ids, attention_mask, word_ids
        except (TypeError, NotImplementedError, AttributeError):
            pass

    # 2. Xử lý cho Slow Tokenizer (như PhobertTokenizer hoặc tokenizer không hỗ trợ word_ids)
    input_ids: List[int] = []
    word_ids: List[Optional[int]] = []

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    # Token đầu câu (<s> / [CLS])
    if bos_id is not None:
        input_ids.append(bos_id)
        word_ids.append(None)

    for w_idx, token in enumerate(tokens):
        # Mã hóa từng từ gốc thành danh sách subwords
        token_subwords = tokenizer.encode(token, add_special_tokens=False)
        if not token_subwords:
            token_subwords = [tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3]

        # Kiểm tra giới hạn độ dài (để dành 1 vị trí cuối cho eos_id)
        reserve_for_eos = 1 if eos_id is not None else 0
        if len(input_ids) + len(token_subwords) > max_length - reserve_for_eos:
            allowed_len = (max_length - reserve_for_eos) - len(input_ids)
            if allowed_len > 0:
                input_ids.extend(token_subwords[:allowed_len])
                word_ids.extend([w_idx] * allowed_len)
            break

        input_ids.extend(token_subwords)
        word_ids.extend([w_idx] * len(token_subwords))

    # Token cuối câu (</s> / [SEP])
    if eos_id is not None and len(input_ids) < max_length:
        input_ids.append(eos_id)
        word_ids.append(None)

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, word_ids


def align_tokens_and_labels(
    tokens: List[str],
    tags: List[str],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 256,
    ignore_index: int = -100,
) -> Dict[str, Any]:
    """
    Căn chỉnh cấp độ câu hoàn chỉnh cho 1 mẫu dữ liệu.

    Args:
        tokens: Danh sách token từ/âm tiết gốc.
        tags: Danh sách nhãn BIO tương ứng.
        tokenizer: HuggingFace Tokenizer.
        label2id: Bản đồ nhãn -> id.
        max_length: Độ dài tối đa.
        ignore_index: Giá trị nhãn bỏ qua.

    Returns:
        Dict chứa:
            "input_ids": List[int]
            "attention_mask": List[int]
            "labels": List[int]
            "word_ids": List[Optional[int]]
    """
    if len(tokens) != len(tags):
        raise ValueError(
            f"Không khớp kích thước: số lượng tokens ({len(tokens)}) != số lượng tags ({len(tags)})!"
        )

    input_ids, attention_mask, word_ids = tokenize_and_get_word_ids(
        tokenizer=tokenizer,
        tokens=tokens,
        max_length=max_length,
    )

    labels = align_labels_with_subwords(
        word_labels=tags,
        word_ids=word_ids,
        label2id=label2id,
        ignore_index=ignore_index,
    )

    # Đảm bảo invariant về độ dài
    assert len(input_ids) == len(attention_mask) == len(labels) == len(word_ids), (
        f"Lỗi bất biến: len(input_ids)={len(input_ids)}, len(mask)={len(attention_mask)}, "
        f"len(labels)={len(labels)}, len(word_ids)={len(word_ids)}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_ids": word_ids,
    }


def align_labels_with_tokens(
    tokenizer: Any,
    tokens: List[str],
    tags: List[str],
    tag2id: Dict[str, int],
    max_seq_len: int = 256,
    ignore_index: int = -100,
) -> Tuple[List[int], List[int], List[int]]:
    """
    API kế thừa (backward-compatible) phục vụ NERDataset và trainer.
    Trả về Tuple (input_ids, attention_mask, labels).
    """
    res = align_tokens_and_labels(
        tokens=tokens,
        tags=tags,
        tokenizer=tokenizer,
        label2id=tag2id,
        max_length=max_seq_len,
        ignore_index=ignore_index,
    )
    return res["input_ids"], res["attention_mask"], res["labels"]


def align_batch_tokens_and_labels(
    batch_tokens: List[List[str]],
    batch_tags: List[List[str]],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 256,
    padding: bool = True,
    ignore_index: int = -100,
) -> Dict[str, Any]:
    """
    Xử lý và căn chỉnh nhãn cho một lô (batch) gồm nhiều câu có độ dài khác nhau.

    Args:
        batch_tokens: Danh sách các câu, mỗi câu là List[str] tokens.
        batch_tags: Danh sách các câu, mỗi câu là List[str] nhãn BIO.
        tokenizer: HuggingFace Tokenizer.
        label2id: Bảng ánh xạ nhãn -> id.
        max_length: Độ dài chuỗi tối đa.
        padding: Có thực hiện padding các câu trong batch về cùng độ dài không.
        ignore_index: Giá trị gán cho các vị trí bỏ qua / padding.

    Returns:
        Dict chứa input_ids, attention_mask, labels dạng Tensor (nếu padding=True)
        hoặc dạng List lồng nhau (nếu padding=False).
    """
    if len(batch_tokens) != len(batch_tags):
        raise ValueError(
            f"Số câu trong batch_tokens ({len(batch_tokens)}) != batch_tags ({len(batch_tags)})!"
        )

    processed_items = []
    for toks, tgs in zip(batch_tokens, batch_tags):
        item = align_tokens_and_labels(
            tokens=toks,
            tags=tgs,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=max_length,
            ignore_index=ignore_index,
        )
        processed_items.append(item)

    if not padding:
        return {
            "input_ids": [x["input_ids"] for x in processed_items],
            "attention_mask": [x["attention_mask"] for x in processed_items],
            "labels": [x["labels"] for x in processed_items],
        }

    # Dynamic padding theo câu dài nhất trong batch
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(x["input_ids"]) for x in processed_items)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in processed_items:
        cur_len = len(item["input_ids"])
        pad_len = max_len - cur_len

        padded_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        padded_attention_mask.append(item["attention_mask"] + [0] * pad_len)
        padded_labels.append(item["labels"] + [ignore_index] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


class NERDataset(Dataset):
    """Dataset cho Transformer Token Classification."""

    def __init__(
        self,
        samples: List[NERSample],
        tokenizer: Any,
        tag2id: Dict[str, int],
        max_seq_len: int = 256,
        max_length: Optional[int] = None,
        ignore_index: int = -100,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_seq_len = max_length if max_length is not None else max_seq_len
        self.ignore_index = ignore_index

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str,
        tokenizer: Any,
        tag2id: Dict[str, int],
        max_seq_len: int = 256,
        ignore_index: int = -100,
    ) -> "NERDataset":
        samples = list(load_jsonl(jsonl_path))
        return cls(samples, tokenizer, tag2id, max_seq_len, ignore_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        aligned = align_tokens_and_labels(
            tokens=sample.tokens,
            tags=sample.tags,
            tokenizer=self.tokenizer,
            label2id=self.tag2id,
            max_length=self.max_seq_len,
            ignore_index=self.ignore_index,
        )
        return {
            "id": sample.id,
            "input_ids": aligned["input_ids"],
            "attention_mask": aligned["attention_mask"],
            "labels": aligned["labels"],
            "tokens": sample.tokens,
            "tags": sample.tags,
        }


class NERDataCollator:
    """
    Callable class thực hiện dynamic padding cho DataLoader.
    Được thiết kế dạng class top-level để tương thích hoàn toàn với multiprocessing (spawn) trên Windows.
    Tối ưu VRAM: Chỉ pad tới độ dài chuỗi dài nhất trong batch hiện tại.
    """

    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)

        batch_ids = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_tokens = []
        batch_tags = []

        for item in batch:
            batch_ids.append(item["id"])
            batch_tokens.append(item["tokens"])
            batch_tags.append(item["tags"])

            cur_len = len(item["input_ids"])
            pad_len = max_len - cur_len

            # Pad input_ids với pad_token_id, attention_mask với 0, labels với ignore_index
            padded_ids = item["input_ids"] + [self.pad_token_id] * pad_len
            padded_mask = item["attention_mask"] + [0] * pad_len
            padded_labels = item["labels"] + [self.ignore_index] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        # [B, L]
        return {
            "ids": batch_ids,
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "tokens": batch_tokens,
            "tags": batch_tags,
        }


def create_ner_collate_fn(pad_token_id: int, ignore_index: int = -100) -> NERDataCollator:
    """Hàm bao bọc tương thích ngược trả về instance của NERDataCollator."""
    return NERDataCollator(pad_token_id=pad_token_id, ignore_index=ignore_index)

    return collate_fn
