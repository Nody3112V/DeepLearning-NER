from dataclasses import dataclass, asdict
from typing import List, Dict
import json

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

# Biểu diễn 1 mẫu dữ liệu NER
@dataclass
class NERSample:
    id: str
    dataset: str          # "phoner_covid19" hoặc "vlsp"
    granularity: str      # "word" hoặc "syllable"
    tokens: List[str]
    tags: List[str]

    # Kiểm tra tính hợp lệ của mẫu dữ liệu NER
    def validate(self) -> None:
        assert len(self.tokens) > 0, f"Sample {self.id}: danh sách tokens rỗng!"
        assert len(self.tokens) == len(self.tags), (
            f"Sample {self.id}: số lượng tokens ({len(self.tokens)}) != tags ({len(self.tags)})"
        )
        for t in self.tags:
            assert t == "O" or t.startswith(("B-", "I-")), f"Sample {self.id}: nhãn BIO không hợp lệ ({t})"

    # Chuyển NERSample -> dictionary -> chuỗi json
    def to_json_line(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False)
