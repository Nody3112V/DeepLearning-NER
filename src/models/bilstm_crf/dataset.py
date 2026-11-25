import json
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from src.models.bilstm_crf.vocab import Vocabulary, PAD_IDX


class NERDataset(Dataset):
    """Đọc tệp JSONL chuẩn hóa và mã hóa thành chỉ số word, char, tag."""

    def __init__(
        self,
        jsonl_path: str,
        vocab: Vocabulary,
        max_seq_len: Optional[int] = 256,
        max_word_len: int = 25,
    ) -> None:
        self.samples: List[Dict] = []
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                tokens = (
                    record["tokens"][:max_seq_len]
                    if max_seq_len is not None
                    else record["tokens"]
                )
                tags = (
                    record["tags"][:max_seq_len]
                    if max_seq_len is not None
                    else record["tags"]
                )

                self.samples.append(
                    {
                        "id": record["id"],
                        "tokens": tokens,
                        "word_ids": vocab.encode_words(tokens),
                        "char_ids": vocab.encode_chars(tokens, max_word_len=max_word_len),
                        "tag_ids": vocab.encode_tags(tags),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_ner_batch(batch: List[Dict], min_word_len: int = 3) -> Dict[str, object]:
    """Đệm động theo lô cho word_ids, char_ids, tag_ids và tạo mask + lengths."""
    batch_size = len(batch)
    lengths = [len(item["word_ids"]) for item in batch]
    max_seq_len = max(lengths)

    max_word_len = max(
        min_word_len,
        max(len(ch) for item in batch for ch in item["char_ids"]),
    )

    word_ids = torch.full((batch_size, max_seq_len), PAD_IDX, dtype=torch.long)
    char_ids = torch.full((batch_size, max_seq_len, max_word_len), PAD_IDX, dtype=torch.long)
    tag_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    ids = []
    tokens_list = []

    for i, item in enumerate(batch):
        seq_len = len(item["word_ids"])
        ids.append(item["id"])
        tokens_list.append(item["tokens"])

        word_ids[i, :seq_len] = torch.tensor(item["word_ids"], dtype=torch.long)
        tag_ids[i, :seq_len] = torch.tensor(item["tag_ids"], dtype=torch.long)
        mask[i, :seq_len] = True

        for j, chars in enumerate(item["char_ids"]):
            char_ids[i, j, : len(chars)] = torch.tensor(chars, dtype=torch.long)

    return {
        "ids": ids,
        "tokens": tokens_list,
        "word_ids": word_ids,
        "char_ids": char_ids,
        "tag_ids": tag_ids,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def create_dataloader(
    jsonl_path: str,
    vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = False,
    max_seq_len: Optional[int] = 256,
    max_word_len: int = 25,
) -> DataLoader:
    dataset = NERDataset(
        jsonl_path=jsonl_path,
        vocab=vocab,
        max_seq_len=max_seq_len,
        max_word_len=max_word_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ner_batch,
    )
