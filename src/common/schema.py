"""
schema.py - Định nghĩa dữ liệu NER dùng chung cho cả nhóm (src/common).

Dựa trên bản schema.py của TV1 (giữ nguyên NERSample, PHONER_ENTITY_TYPES,
PHONER_BIO_TAGS, to_json_line), bổ sung:
  * Bộ nhãn theo từng tập dữ liệu (TAG_SETS) và kiểm tra nhãn có thuộc đúng bộ nhãn.
  * Kiểm tra dataset / granularity hợp lệ; tùy chọn kiểm tra BIO chặt (strict_bio).
  * Chiều ĐỌC: NERSample.from_json_line, load_jsonl, save_jsonl.
  * Quy ước đặt id: parse_sample_id / make_sample_id  (vd: phoner_word_train_00001).
  * validate() ném ValueError thay vì assert (assert bị bỏ qua khi chạy python -O).

Định dạng một dòng JSONL ghi ra:
  {"id", "dataset", "granularity", "tokens", "tags"}
Khi ĐỌC, chỉ cần 3 trường {"id", "tokens", "tags"} (đúng chuẩn đã thống nhất);
dataset và granularity thiếu thì được suy ra từ id.
"""
from dataclasses import dataclass, asdict
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional
import json
import re

# --------------------------------------------------------------------------- #
# 1. Bộ nhãn
# --------------------------------------------------------------------------- #

# Các loại thực thể có tên
PHONER_ENTITY_TYPES = [
    "PATIENT_ID",
    "NAME",
    "AGE",
    "GENDER",
    "JOB",
    "LOCATION",
    "ORGANIZATION",
    "SYMPTOM_AND_DISEASE",
    "TRANSPORTATION",
    "DATE",
]

# Khai báo bộ nhãn BIO cho các loại thực thể
PHONER_BIO_TAGS = ["O"] + [
    f"{prefix}-{etype}"
    for etype in PHONER_ENTITY_TYPES
    for prefix in ("B", "I")
]

# VLSP: TV2 điền danh sách loại thực thể thật vào đây (vd: ["PER", "LOC", ...]).
# Khi còn để trống, validate() chỉ kiểm tra ĐỊNH DẠNG B-/I- mà chưa kiểm tra loại.
VLSP_ENTITY_TYPES: List[str] = []

VLSP_BIO_TAGS = ["O"] + [
    f"{prefix}-{etype}"
    for etype in VLSP_ENTITY_TYPES
    for prefix in ("B", "I")
]

ALLOWED_DATASETS = ("phoner_covid19", "vlsp")
ALLOWED_GRANULARITIES = ("word", "syllable")

# Bộ nhãn theo từng tập dữ liệu (rỗng/chưa khai báo loại -> không kiểm tra loại)
TAG_SETS: Dict[str, List[str]] = {
    "phoner_covid19": PHONER_BIO_TAGS,
    "vlsp": VLSP_BIO_TAGS,
}


def get_tag_set(dataset: str) -> Optional[FrozenSet[str]]:
    """Trả về tập nhãn hợp lệ của dataset, hoặc None nếu chưa khai báo loại thực thể."""
    tags = TAG_SETS.get(dataset)
    if not tags or len(tags) <= 1:  # chỉ có "O" -> chưa khai báo loại nào
        return None
    return frozenset(tags)


# --------------------------------------------------------------------------- #
# 2. Quy ước id:  <tiền tố tập>_<granularity>_<split>_<số thứ tự>
#    ví dụ: phoner_word_train_00001, vlsp_syllable_test_00123
# --------------------------------------------------------------------------- #
DATASET_ID_PREFIX = {"phoner_covid19": "phoner", "vlsp": "vlsp"}
_PREFIX_TO_DATASET = {v: k for k, v in DATASET_ID_PREFIX.items()}

_ID_RE = re.compile(
    r"^(?P<prefix>[a-z0-9]+)_(?P<granularity>word|syllable)_"
    r"(?P<split>[a-z0-9]+)_(?P<index>\d+)$"
)


def parse_sample_id(sample_id: str) -> Dict[str, str]:
    """'phoner_word_train_00001' -> {dataset, granularity, split, index}."""
    m = _ID_RE.match(sample_id)
    if not m:
        raise ValueError(f"id không đúng quy ước <tập>_<granularity>_<split>_<số>: {sample_id!r}")
    prefix = m.group("prefix")
    if prefix not in _PREFIX_TO_DATASET:
        raise ValueError(f"Tiền tố tập dữ liệu lạ trong id {sample_id!r}: {prefix!r}")
    return {"dataset": _PREFIX_TO_DATASET[prefix], "granularity": m.group("granularity"),
            "split": m.group("split"), "index": m.group("index")}


def make_sample_id(dataset: str, granularity: str, split: str, index: int, width: int = 5) -> str:
    if dataset not in DATASET_ID_PREFIX:
        raise ValueError(f"dataset không hợp lệ: {dataset!r}")
    return f"{DATASET_ID_PREFIX[dataset]}_{granularity}_{split}_{index:0{width}d}"


# --------------------------------------------------------------------------- #
# 3. Mẫu dữ liệu NER
# --------------------------------------------------------------------------- #

# Biểu diễn 1 mẫu dữ liệu NER
@dataclass
class NERSample:
    id: str
    dataset: str          # "phoner_covid19" hoặc "vlsp"
    granularity: str      # "word" hoặc "syllable"
    tokens: List[str]
    tags: List[str]

    # Kiểm tra tính hợp lệ của mẫu dữ liệu NER
    def validate(self, strict_bio: bool = False) -> None:
        """
        strict_bio=True: cấm "I-X" đứng đầu câu / sau "O" / sau nhãn loại khác.
        (metrics.py vẫn chấm được các trường hợp này, nên mặc định không chặn.)
        """
        if self.dataset not in ALLOWED_DATASETS:
            raise ValueError(f"Sample {self.id}: dataset không hợp lệ ({self.dataset!r})")
        if self.granularity not in ALLOWED_GRANULARITIES:
            raise ValueError(f"Sample {self.id}: granularity không hợp lệ ({self.granularity!r})")
        if len(self.tokens) == 0:
            raise ValueError(f"Sample {self.id}: danh sách tokens rỗng!")
        if len(self.tokens) != len(self.tags):
            raise ValueError(
                f"Sample {self.id}: số lượng tokens ({len(self.tokens)}) != tags ({len(self.tags)})"
            )

        tag_set = get_tag_set(self.dataset)
        prev = "O"
        for t in self.tags:
            if t != "O":
                if not (len(t) > 2 and t[:2] in ("B-", "I-")):
                    raise ValueError(f"Sample {self.id}: nhãn BIO không hợp lệ ({t})")
                if tag_set is not None and t not in tag_set:
                    raise ValueError(
                        f"Sample {self.id}: nhãn {t} không thuộc bộ nhãn của {self.dataset}")
                if strict_bio and t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
                    raise ValueError(f"Sample {self.id}: {t} không có B-/I- cùng loại đứng trước")
            prev = t

    # Chuyển NERSample -> dictionary -> chuỗi json
    def to_json_line(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False)

    # Chuỗi json -> NERSample (chiều ngược lại của to_json_line)
    @classmethod
    def from_json_line(cls, line: str, validate: bool = True,
                       strict_bio: bool = False) -> "NERSample":
        rec = json.loads(line)
        missing = {"id", "tokens", "tags"} - set(rec)
        if missing:
            raise ValueError(f"Thiếu trường bắt buộc: {sorted(missing)}")
        dataset, granularity = rec.get("dataset"), rec.get("granularity")
        if dataset is None or granularity is None:   # bản ghi chuẩn 3 trường -> suy ra từ id
            info = parse_sample_id(rec["id"])
            dataset = dataset or info["dataset"]
            granularity = granularity or info["granularity"]
        sample = cls(id=rec["id"], dataset=dataset, granularity=granularity,
                     tokens=list(rec["tokens"]), tags=list(rec["tags"]))
        if validate:
            sample.validate(strict_bio=strict_bio)
        return sample


# --------------------------------------------------------------------------- #
# 4. Đọc / ghi file JSONL
# --------------------------------------------------------------------------- #
def load_jsonl(path: str, validate: bool = True, strict_bio: bool = False) -> Iterator[NERSample]:
    """Đọc file JSONL thành các NERSample; lỗi được báo kèm 'file:số dòng'."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield NERSample.from_json_line(line, validate=validate, strict_bio=strict_bio)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e


def save_jsonl(samples: Iterable[NERSample], path: str) -> int:
    """Ghi các NERSample ra file JSONL (có validate). Trả về số dòng đã ghi."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json_line() + "\n")
            n += 1
    return n