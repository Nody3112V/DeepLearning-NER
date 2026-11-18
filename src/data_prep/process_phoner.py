import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
from src.common.schema import NERSample, PHONER_ENTITY_TYPES

RAW_BASE = Path("data/raw/phoner_covid19/data")
OUT_BASE = Path("data/processed/phoner_covid19")
SPLIT_FILE = Path("data/splits/test_indices.json")
STATS_FILE = Path("outputs/tables/phoner_stats.json")

# Kiểm tra, sửa chuỗi nhãn BIO đảm bảo không xuất hiện nhãn I- không hợp lệ (vd: B phải đứng trước I)
def sanitize_bio_sequence(tags: List[str]) -> Tuple[List[str], int]:
    fixed = [] # Chuỗi nhãn sau khi chuẩn hoá
    num_fixed = 0 # Số lỗi được sửa
    prev_prefix = "O" # Nhãn trước đó
    prev_type = "" # Loại thực thể trước đó

    for tag in tags:
        # Input:  ["B-PER", "I-PER", "UNKNOWN", "I-ORG"]
        # Output: ["B-PER", "I-PER", "O", "B-ORG"]
        if tag == "O" or "-" not in tag:
            fixed.append("O")
            prev_prefix, prev_type = "O", ""
            continue

        prefix, etype = tag.split("-", 1)
        if prefix == "I" and (prev_prefix == "O" or prev_type != etype):
            fixed.append(f"B-{etype}")
            num_fixed += 1
            prev_prefix, prev_type = "B", etype
        else:
            fixed.append(tag)
            prev_prefix, prev_type = prefix, etype

    return fixed, num_fixed

# Đếm số lượng thực thể
def count_entities(tags: List[str]) -> Counter:
    counts = Counter()
    for tag in tags:
        if tag.startswith("B-"):
            counts[tag.split("-", 1)[1]] += 1
    return counts


def process_split(granularity: str, split: str) -> Dict:
    in_path = RAW_BASE / granularity / f"{split}_{granularity}.json"
    out_dir = OUT_BASE / granularity
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    samples = []
    entity_counter = Counter()
    tag_counter = Counter()
    total_tokens = 0
    total_fixed = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tokens = record["words"]
            raw_tags = record["tags"]

            clean_tags, n_fix = sanitize_bio_sequence(raw_tags)
            total_fixed += n_fix

            sample_id = f"phoner_{granularity}_{split}_{idx:05d}"
            sample = NERSample(
                id=sample_id,
                dataset="phoner_covid19",
                granularity=granularity,
                tokens=tokens,
                tags=clean_tags,
            )
            fout.write(sample.to_json_line() + "\n")
            samples.append(sample)

            total_tokens += len(tokens)
            tag_counter.update(clean_tags)
            entity_counter.update(count_entities(clean_tags))

    return {
        "num_sentences": len(samples), # Số lượng câu
        "total_tokens": total_tokens, # Tổng số token
        "avg_length": round(total_tokens / max(len(samples), 1), 2), # Độ dài trung bình
        "bio_fixes": total_fixed, # Số lỗi BIO được sửa
        "entity_counts": dict(entity_counter), # Số lượng thực thể
        "o_token_ratio": round(tag_counter["O"] / max(total_tokens, 1) * 100, 2), # Tỷ lệ nhãn "O"
        "sample_ids": [s.id for s in samples], # Danh sách ID của các câu
    }


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    test_indices = {}

    for granularity in ("word", "syllable"):
        all_stats[granularity] = {}
        for split in ("train", "dev", "test"):
            stats = process_split(granularity, split)
            if split == "test":
                test_indices[granularity] = [
                    int(sid.rsplit("_", 1)[-1]) for sid in stats["sample_ids"]
                ]
            del stats["sample_ids"]
            all_stats[granularity][split] = stats

    # Lưu danh sách chỉ số câu test cố định
    with SPLIT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "phoner_covid19",
                "num_test_sentences": len(test_indices["word"]),
                "indices": test_indices["word"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Lưu bảng thống kê chi tiết
    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("=== HOÀN TẤT CHUẨN HÓA PHONER_COVID19 ===")
    for gran in ("word", "syllable"):
        print(f"\n--- Cấp độ: {gran.upper()} ---")
        for sp in ("train", "dev", "test"):
            st = all_stats[gran][sp]
            print(
                f"  {sp:5s}: {st['num_sentences']:5d} câu | "
                f"{st['total_tokens']:7d} tokens | "
                f"TB: {st['avg_length']:5.2f} tokens/câu | "
                f"Tỷ lệ nhãn 'O': {st['o_token_ratio']}%"
            )

    print("\n--- Phân bố 10 loại thực thể trên tập TRAIN (Word-level) ---")
    train_ent = all_stats["word"]["train"]["entity_counts"]
    total_ent = sum(train_ent.values())
    for etype in PHONER_ENTITY_TYPES:
        cnt = train_ent.get(etype, 0)
        pct = cnt / max(total_ent, 1) * 100
        print(f"  {etype:20s}: {cnt:6d} ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
