import json
from collections import Counter
from pathlib import Path
from typing import Dict, List
from src.common.schema import PHONER_BIO_TAGS

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    """Quản lý ánh xạ Word, Character và nhãn BIO cho nhánh BiLSTM-CRF."""

    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.char2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.tag2idx: Dict[str, int] = {}
        self.idx2tag: Dict[int, str] = {}

    def build_from_jsonl(self, train_path: str, min_word_freq: int = 1) -> None:
        word_counter = Counter()
        char_set = set()
        tag_set = set()

        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                for token in sample["tokens"]:
                    word_counter[token] += 1
                    char_set.update(token)
                tag_set.update(sample["tags"])

        # 1. Xây từ điển từ (giữ nguyên viết hoa/thường vì chữ hoa rất quan trọng trong NER)
        for word, freq in sorted(word_counter.items()):
            if freq >= min_word_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        # 2. Xây từ điển ký tự (phục vụ Char-CNN / Char-LSTM ở Tuần 3)
        for ch in sorted(char_set):
            if ch not in self.char2idx:
                self.char2idx[ch] = len(self.char2idx)

        # 3. Xây từ điển nhãn BIO (ưu tiên thứ tự chuẩn PHONER_BIO_TAGS nếu khớp)
        ordered_tags = list(PHONER_BIO_TAGS) if tag_set.issubset(PHONER_BIO_TAGS) else []
        for t in sorted(tag_set):
            if t not in ordered_tags:
                ordered_tags.append(t)

        self.tag2idx = {tag: idx for idx, tag in enumerate(ordered_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

    def encode_words(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(tok, UNK_IDX) for tok in tokens]

    def encode_chars(self, tokens: List[str], max_word_len: int = 25) -> List[List[int]]:
        encoded = []
        for tok in tokens:
            chars = [self.char2idx.get(ch, UNK_IDX) for ch in tok[:max_word_len]]
            encoded.append(chars)
        return encoded

    def encode_tags(self, tags: List[str]) -> List[int]:
        return [self.tag2idx[t] for t in tags]

    def decode_tags(self, tag_ids: List[int]) -> List[str]:
        return [self.idx2tag[i] for i in tag_ids]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word2idx": self.word2idx,
            "char2idx": self.char2idx,
            "tag2idx": self.tag2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab.word2idx = payload["word2idx"]
        vocab.char2idx = payload["char2idx"]
        vocab.tag2idx = payload["tag2idx"]
        vocab.idx2tag = {int(v): k for k, v in vocab.tag2idx.items()}
        return vocab
