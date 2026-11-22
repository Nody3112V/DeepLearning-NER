"""
process_vlsp.py - Tiền xử lý và chuẩn hóa dữ liệu VLSP 2016 NER.

Nhiệm vụ (Thành viên 2):
  1. Đọc dữ liệu CoNLL/txt từ data/raw/vlsp/ (mặc định đã ở cấp độ từ word-level có dấu '_').
  2. Chuẩn hóa chuỗi nhãn BIO (sửa lỗi I- mồ côi không có B- hoặc I- cùng loại đi trước).
  3. Xuất tập dữ liệu chuẩn hóa cấp độ từ (word-level) theo chuẩn NERSample (JSONL).
  4. Chuyển đổi sang cấp độ âm tiết (syllable-level) bảo toàn 100% ranh giới và số lượng thực thể.
  5. Kiểm tra rò rỉ dữ liệu (data leakage) và tính toán toàn diện các chỉ số thống kê:
     - Số câu, số tokens, độ dài min/max/avg
     - Phân bố nhãn BIO, số lượng và tỷ lệ từng loại thực thể
     - Tỷ lệ nhãn O
  6. Xuất bảng thống kê chi tiết dạng JSON và Markdown vào outputs/tables/.
"""
from typing import List, Tuple, Dict, Any, Optional, Set
import os
import json
import argparse
from pathlib import Path
from collections import Counter

from src.common.schema import NERSample, make_sample_id, save_jsonl, load_jsonl, TAG_SETS

RAW_BASE = Path("data/raw/vlsp")
OUT_BASE = Path("data/processed/vlsp")
STATS_JSON = Path("outputs/tables/vlsp_stats.json")
STATS_MD = Path("outputs/tables/vlsp_stats.md")

VLSP_TYPES = ["PER", "LOC", "ORG", "MISC"]


def sanitize_bio_sequence(tags: List[str]) -> Tuple[List[str], int]:
    """
    Chuẩn hóa chuỗi nhãn BIO, đảm bảo không có nhãn I- đứng đầu hoặc đi sau O/nhãn khác.
    Nếu phát hiện I-X mồ côi, tự động sửa thành B-X và đếm số lượng sửa đổi.
    """
    fixed = []
    num_fixed = 0
    prev_prefix = "O"
    prev_type = ""

    for tag in tags:
        tag = tag.strip().upper()
        if tag == "O" or "-" not in tag:
            fixed.append("O")
            prev_prefix, prev_type = "O", ""
            continue

        parts = tag.split("-", 1)
        prefix = parts[0]
        etype = parts[1].strip()

        if prefix not in ("B", "I"):
            prefix = "B"

        if prefix == "I" and (prev_prefix == "O" or prev_type != etype):
            fixed.append(f"B-{etype}")
            num_fixed += 1
            prev_prefix, prev_type = "B", etype
        else:
            fixed.append(f"{prefix}-{etype}")
            prev_prefix, prev_type = prefix, etype

    return fixed, num_fixed


def word_to_syllables(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str]]:
    """
    Chuyển đổi câu từ cấp độ từ (word-level, có '_') sang cấp độ âm tiết (syllable-level).
    
    Quy tắc chuyển đổi nhãn:
      - Từ có nhãn 'O' -> tất cả các âm tiết con mang nhãn 'O'.
      - Từ có nhãn 'B-X' -> âm tiết đầu mang nhãn 'B-X', các âm tiết sau mang nhãn 'I-X'.
      - Từ có nhãn 'I-X' -> tất cả các âm tiết con mang nhãn 'I-X'.
    Bảo toàn 100% ranh giới thực thể và tổng số lượng thực thể.
    """
    syl_tokens: List[str] = []
    syl_tags: List[str] = []

    for tok, tag in zip(tokens, tags):
        sylls = [s for s in tok.split("_") if s]
        if not sylls:
            syl_tokens.append(tok)
            syl_tags.append(tag)
            continue

        syl_tokens.extend(sylls)
        if tag == "O" or "-" not in tag:
            syl_tags.extend(["O"] * len(sylls))
        else:
            prefix, etype = tag.split("-", 1)
            if prefix == "B":
                syl_tags.append(tag)
                syl_tags.extend([f"I-{etype}"] * (len(sylls) - 1))
            else:
                syl_tags.extend([f"I-{etype}"] * len(sylls))

    return syl_tokens, syl_tags


def parse_conll_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    """
    Đọc tệp định dạng CoNLL (mỗi dòng: Token [POS] [Chunk] Tag; phân cách câu bằng dòng trắng).
    """
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[-1]
                current_tokens.append(token)
                current_tags.append(tag)
            else:
                print(f"[Cảnh báo] Dòng không hợp lệ tại {file_path}:{line_num}: '{line}'")

    if current_tokens:
        sentences.append((current_tokens, current_tags))

    return sentences


def count_entities(tags: List[str]) -> Counter:
    """Đếm số lượng thực thể theo loại dựa trên nhãn B-."""
    counts = Counter()
    for tag in tags:
        if tag.startswith("B-"):
            counts[tag.split("-", 1)[1]] += 1
    return counts


def analyze_split_stats(samples: List[NERSample], bio_fixes: int = 0) -> Dict[str, Any]:
    """Tính toán bộ thống kê toàn diện cho một tập dữ liệu."""
    total_sentences = len(samples)
    if total_sentences == 0:
        return {}

    lengths = [len(s.tokens) for s in samples]
    total_tokens = sum(lengths)
    tag_counter = Counter()
    entity_counter = Counter()

    for s in samples:
        tag_counter.update(s.tags)
        entity_counter.update(count_entities(s.tags))

    total_entities = sum(entity_counter.values())
    entity_ratios = {
        etype: round(cnt / max(total_entities, 1) * 100, 2)
        for etype, cnt in sorted(entity_counter.items())
    }

    return {
        "num_sentences": total_sentences,
        "total_tokens": total_tokens,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": round(total_tokens / total_sentences, 2),
        "bio_fixes": bio_fixes,
        "total_entities": total_entities,
        "entity_counts": dict(sorted(entity_counter.items())),
        "entity_ratios": entity_ratios,
        "o_token_ratio": round(tag_counter["O"] / max(total_tokens, 1) * 100, 2),
        "tag_distribution": dict(sorted(tag_counter.items())),
    }


def check_data_leakage(split_samples: Dict[str, List[NERSample]]) -> Dict[str, Any]:
    """Kiểm tra câu trùng lặp giữa các tập train, dev và test."""
    texts = {}
    for split, samples in split_samples.items():
        texts[split] = [" ".join(s.tokens) for s in samples]

    train_set = set(texts.get("train", []))
    dev_set = set(texts.get("dev", []))
    test_set = set(texts.get("test", []))

    overlap_train_dev = train_set & dev_set
    overlap_train_test = train_set & test_set
    overlap_dev_test = dev_set & test_set

    return {
        "unique_sentences": {split: len(set(txts)) for split, txts in texts.items()},
        "overlap_train_dev": len(overlap_train_dev),
        "overlap_train_test": len(overlap_train_test),
        "overlap_dev_test": len(overlap_dev_test),
    }


def export_markdown_report(stats: Dict[str, Any], leakage: Dict[str, Any], out_path: Path) -> None:
    """Tạo báo cáo markdown chi tiết từ số liệu thống kê."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BÁO CÁO THỐNG KÊ DỮ LIỆU VLSP 2016 NER",
        "",
        "## 1. Tổng quan các tập dữ liệu",
        "",
        "| Mức biểu diễn | Phân chia | Số câu | Số token | Min len | Max len | Avg len | Tổng thực thể | Tỷ lệ nhãn O (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for gran in ["word", "syllable"]:
        for split in ["train", "dev", "test"]:
            s = stats.get(gran, {}).get(split, {})
            lines.append(
                f"| **{gran.capitalize()}** | {split.upper()} | {s.get('num_sentences', 0):,} | "
                f"{s.get('total_tokens', 0):,} | {s.get('min_length', 0)} | {s.get('max_length', 0)} | "
                f"{s.get('avg_length', 0):.2f} | {s.get('total_entities', 0):,} | {s.get('o_token_ratio', 0):.2f}% |"
            )

    lines.extend([
        "",
        "## 2. Phân bố thực thể (Entity Distribution)",
        "",
        "| Tập | Loại | Số lượng | Tỷ lệ (%) |",
        "| :--- | :--- | :---: | :---: |",
    ])

    for split in ["train", "dev", "test"]:
        s = stats.get("word", {}).get(split, {})
        counts = s.get("entity_counts", {})
        ratios = s.get("entity_ratios", {})
        for etype in VLSP_TYPES:
            cnt = counts.get(etype, 0)
            rat = ratios.get(etype, 0.0)
            lines.append(f"| {split.upper()} | **{etype}** | {cnt:,} | {rat:.2f}% |")

    lines.extend([
        "",
        "## 3. Kiểm tra rò rỉ dữ liệu gốc (Data Leakage)",
        "",
        f"- Câu trùng giữa Train và Dev: **{leakage.get('overlap_train_dev', 0)} câu**",
        f"- Câu trùng giữa Train và Test: **{leakage.get('overlap_train_test', 0)} câu**",
        f"- Câu trùng giữa Dev và Test: **{leakage.get('overlap_dev_test', 0)} câu**",
        "",
        "> *Ghi chú:* Các câu trùng lặp trong dữ liệu gốc là các tiêu đề ngắn hoặc chữ ký phóng viên (`MINH LUẬN - HOÀNG KHƯƠNG`, `Đi về đâu ?`) thuộc kho ngữ liệu báo chí nguyên bản của VLSP.",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> Đã lưu báo cáo Markdown vào: {out_path}")


def process_vlsp_dataset(raw_dir: Path = RAW_BASE, out_dir: Path = OUT_BASE) -> Dict[str, Any]:
    """Quy trình tiền xử lý chính cho VLSP cả 2 mức word và syllable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, Any] = {"word": {}, "syllable": {}, "metadata": {}}
    split_word_samples: Dict[str, List[NERSample]] = {}

    word_out_dir = out_dir / "word"
    syl_out_dir = out_dir / "syllable"
    word_out_dir.mkdir(parents=True, exist_ok=True)
    syl_out_dir.mkdir(parents=True, exist_ok=True)

    print("=== BẮT ĐẦU TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU VLSP 2016 ===")

    for split in ("train", "dev", "test"):
        raw_file = raw_dir / f"{split}.txt"
        if not raw_file.exists():
            raw_file = raw_dir / f"{split}.conll"

        if not raw_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file raw: {raw_file}")

        print(f"\nĐang đọc và xử lý {raw_file.name}...")
        raw_sentences = parse_conll_file(raw_file)

        word_samples: List[NERSample] = []
        syl_samples: List[NERSample] = []
        w_total_fixed = 0

        for idx, (tokens, raw_tags) in enumerate(raw_sentences, start=1):
            # 1. Chuẩn hóa nhãn BIO (sửa I- mồ côi)
            clean_tags, n_fix = sanitize_bio_sequence(raw_tags)
            w_total_fixed += n_fix

            # 2. Tạo bản Word-level
            w_sid = make_sample_id(dataset="vlsp", granularity="word", split=split, index=idx)
            w_sample = NERSample(
                id=w_sid,
                dataset="vlsp",
                granularity="word",
                tokens=tokens,
                tags=clean_tags,
            )
            w_sample.validate()
            word_samples.append(w_sample)

            # 3. Tạo bản Syllable-level
            syl_toks, syl_tags = word_to_syllables(tokens, clean_tags)
            s_sid = make_sample_id(dataset="vlsp", granularity="syllable", split=split, index=idx)
            s_sample = NERSample(
                id=s_sid,
                dataset="vlsp",
                granularity="syllable",
                tokens=syl_toks,
                tags=syl_tags,
            )
            s_sample.validate()
            syl_samples.append(s_sample)

        # Lưu các tệp JSONL
        w_out_path = word_out_dir / f"{split}.jsonl"
        s_out_path = syl_out_dir / f"{split}.jsonl"
        save_jsonl(word_samples, str(w_out_path))
        save_jsonl(syl_samples, str(s_out_path))

        split_word_samples[split] = word_samples

        all_stats["word"][split] = analyze_split_stats(word_samples, bio_fixes=w_total_fixed)
        all_stats["syllable"][split] = analyze_split_stats(syl_samples, bio_fixes=0)

        print(f"  -> [{split.upper()}] Đã xuất {len(word_samples):,} câu:")
        print(f"     Word    : {all_stats['word'][split]['total_tokens']:,} tokens | Entities: {all_stats['word'][split]['total_entities']:,} | BIO fixes: {w_total_fixed}")
        print(f"     Syllable: {all_stats['syllable'][split]['total_tokens']:,} tokens | Entities: {all_stats['syllable'][split]['total_entities']:,}")

    # Kiểm tra rò rỉ dữ liệu
    leakage_stats = check_data_leakage(split_word_samples)
    all_stats["leakage"] = leakage_stats

    # Ghi file thống kê JSON
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu bảng thống kê JSON vào: {STATS_JSON}")

    # Ghi file thống kê Markdown
    export_markdown_report(all_stats, leakage_stats, STATS_MD)

    print("=== HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU VLSP 2016 ===")
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý tập dữ liệu VLSP")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_BASE))
    parser.add_argument("--out_dir", type=str, default=str(OUT_BASE))
    args = parser.parse_args()

    process_vlsp_dataset(raw_dir=Path(args.raw_dir), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
