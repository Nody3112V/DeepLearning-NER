#!/usr/bin/env python3
"""
download_vlsp.py - Tải và xuất tập dữ liệu chuẩn VLSP 2016 NER từ HuggingFace Hub vào data/raw/vlsp/.

Nguồn dữ liệu: datnth1709/VLSP2016-NER-data (gồm 13.486 câu train và 3.372 câu valid).
Định dạng xuất: CoNLL-like text format (mỗi dòng: Token\tTag, phân tách câu bằng dòng trắng).
Tập nhãn 4 loại: PER, LOC, ORG, MISC theo chuẩn BIO.
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent / "raw" / "vlsp"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TAG_MAP = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def export_split_to_conll(dataset_split, out_path: Path):
    """Ghi một split sang định dạng CoNLL (Token\tTag)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in dataset_split:
            tokens = sample["tokens"]
            tags = sample["ner_tags"]
            for tok, tag_id in zip(tokens, tags):
                tag_str = TAG_MAP.get(tag_id, "O")
                f.write(f"{tok}\t{tag_str}\n")
            f.write("\n")
    print(f"  -> Đã lưu {len(dataset_split)} câu vào: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def download_and_extract_vlsp():
    print("=== ĐANG TẢI VÀ CHUẨN BỊ TẬP DỮ LIỆU VLSP 2016 NER ===")
    print("Nguồn: HuggingFace Hub (datnth1709/VLSP2016-NER-data)...")

    ds = load_dataset("datnth1709/VLSP2016-NER-data")
    train_data = ds["train"]
    valid_data = ds["valid"]

    # Phân chia valid split thành dev (1.500 câu) và test (1.872 câu)
    dev_data = valid_data.select(range(1500))
    test_data = valid_data.select(range(1500, len(valid_data)))

    print(f"Tổng số câu: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")

    # Ghi vào data/raw/vlsp/
    export_split_to_conll(train_data, RAW_DIR / "train.txt")
    export_split_to_conll(dev_data, RAW_DIR / "dev.txt")
    export_split_to_conll(test_data, RAW_DIR / "test.txt")

    print("\n=== HOÀN TẤT TẢI DỮ LIỆU VLSP 2016 VÀO data/raw/vlsp/ ===")


if __name__ == "__main__":
    download_and_extract_vlsp()
