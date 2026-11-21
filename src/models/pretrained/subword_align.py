"""
subword_align.py - Căn chỉnh nhãn giữa từ gốc/âm tiết với subwords của Transformer Tokenizer.

Hỗ trợ PhoBERT (BPE) và XLM-R (SentencePiece).
Quy tắc:
  * Subword đầu tiên của một từ/âm tiết nhận id của nhãn gốc.
  * Các subword đứng sau (subsequent subwords) nhận nhãn bỏ qua: -100.
  * Các special tokens (<s>, </s>, <pad>) nhận nhãn: -100.
"""
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from src.common.schema import NERSample, load_jsonl


def align_labels_with_tokens(
    tokenizer: Any,
    tokens: List[str],
    tags: List[str],
    tag2id: Dict[str, int],
    max_seq_len: int = 256,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Căn chỉnh danh sách tokens và tags sang input_ids, attention_mask, labels.
    Hỗ trợ cả Fast Tokenizer (sử dụng word_ids) và Slow Tokenizer.
    
    Args:
        tokenizer: HuggingFace PreTrainedTokenizer hoặc PreTrainedTokenizerFast
        tokens: Danh sách token từ/âm tiết gốc
        tags: Danh sách nhãn BIO tương ứng
        tag2id: Bản đồ nhãn -> id (ví dụ: {"O": 0, "B-PER": 1, ...})
        max_seq_len: Độ dài chuỗi tối đa
        
    Returns:
        input_ids: List[int]
        attention_mask: List[int]
        labels: List[int] (gán -100 cho subwords tiếp theo và special tokens)
    """
    # Thử phương thức Fast Tokenizer (có sẵn word_ids())
    try:
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_seq_len,
            truncation=True,
            return_offsets_mapping=False,
        )
        word_ids = encoding.word_ids()
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        labels = []
        current_word = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != current_word:
                current_word = wid
                tag = tags[wid] if wid < len(tags) else "O"
                labels.append(tag2id.get(tag, tag2id.get("O", 0)))
            else:
                labels.append(-100)  # Subword đứng sau
        return input_ids, attention_mask, labels

    except (TypeError, NotImplementedError, AttributeError):
        # Fallback cho Slow Tokenizer (chưa hỗ trợ Fast hoặc không có word_ids)
        subwords = []
        labels = []
        
        # Thêm token bắt đầu (<s> / [CLS])
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        
        subwords.append(bos_id)
        labels.append(-100)

        for token, tag in zip(tokens, tags):
            token_subwords = tokenizer.encode(token, add_special_tokens=False)
            if not token_subwords:
                continue
            for i, sw in enumerate(token_subwords):
                if len(subwords) >= max_seq_len - 1:  # chừa chỗ cho eos_token
                    break
                subwords.append(sw)
                if i == 0:
                    labels.append(tag2id.get(tag, tag2id.get("O", 0)))
                else:
                    labels.append(-100)
            if len(subwords) >= max_seq_len - 1:
                break

        # Thêm token kết thúc (</s> / [SEP])
        subwords.append(eos_id)
        labels.append(-100)
        
        attention_mask = [1] * len(subwords)
        return subwords, attention_mask, labels


class NERDataset(Dataset):
    """Dataset cho Transformer Token Classification."""

    def __init__(
        self,
        samples: List[NERSample],
        tokenizer: Any,
        tag2id: Dict[str, int],
        max_seq_len: int = 256,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_seq_len = max_seq_len

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str,
        tokenizer: Any,
        tag2id: Dict[str, int],
        max_seq_len: int = 256,
    ) -> "NERDataset":
        samples = list(load_jsonl(jsonl_path))
        return cls(samples, tokenizer, tag2id, max_seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        input_ids, attention_mask, labels = align_labels_with_tokens(
            tokenizer=self.tokenizer,
            tokens=sample.tokens,
            tags=sample.tags,
            tag2id=self.tag2id,
            max_seq_len=self.max_seq_len,
        )
        return {
            "id": sample.id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tokens": sample.tokens,
            "tags": sample.tags,
        }


def create_ner_collate_fn(pad_token_id: int):
    """
    Collate function thực hiện dynamic padding cho DataLoader.
    Tối ưu VRAM: Chỉ pad tới độ dài chuỗi dài nhất trong batch hiện tại.
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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

            # Pad input_ids với pad_token_id, attention_mask với 0, labels với -100
            padded_ids = item["input_ids"] + [pad_token_id] * pad_len
            padded_mask = item["attention_mask"] + [0] * pad_len
            padded_labels = item["labels"] + [-100] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        # Chuyển đổi sang PyTorch Tensors
        # [B, L]
        return {
            "ids": batch_ids,
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),         # [B, L]
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long), # [B, L]
            "labels": torch.tensor(batch_labels, dtype=torch.long),               # [B, L]
            "tokens": batch_tokens,
            "tags": batch_tags,
        }

    return collate_fn
