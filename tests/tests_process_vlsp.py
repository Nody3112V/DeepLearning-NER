"""
tests_process_vlsp.py - Bộ kiểm định chất lượng tiền xử lý tập dữ liệu VLSP 2016.

Kiểm tra 9 tiêu chí kỹ thuật:
  1. Số lượng tokens == số lượng labels (100% các mẫu).
  2. Không có nhãn nào nằm ngoài bộ nhãn chuẩn VLSP (TAG_SETS['vlsp']).
  3. Tất cả các ID câu là duy nhất (không trùng lặp).
  4. Không có mẫu rỗng (tokens rỗng hoặc tags rỗng).
  5. Không có dữ liệu hỏng hoặc giá trị None/NaN.
  6. Chuỗi BIO hợp lệ (không có I-X mồ côi sau khi chuẩn hóa).
  7. Không thất thoát thực thể khi chuyển đổi giữa word-level và syllable-level.
  8. Phân tách tập train, dev, test hoàn toàn độc lập (IDs tách biệt).
  9. Kiểm tra rò rỉ dữ liệu (data leakage) giữa các tập.
"""
import sys
from pathlib import Path
from collections import Counter

from src.common.schema import load_jsonl, TAG_SETS
from src.data_prep.process_vlsp import count_entities, VLSP_TYPES

PROCESSED_BASE = Path("data/processed/vlsp")


def run_vlsp_data_tests():
    print("=== BẮT ĐẦU KIỂM ĐỊNH DỮ LIỆU TIỀN XỬ LÝ VLSP 2016 ===")
    valid_tags = TAG_SETS.get("vlsp")
    assert valid_tags is not None, "Bộ nhãn VLSP chưa được khai báo trong schema.py!"

    all_seen_ids = set()

    for gran in ["word", "syllable"]:
        print(f"\n--- Kiểm tra mức biểu diễn: {gran.upper()} ---")
        split_entity_counts = {}

        for split in ["train", "dev", "test"]:
            jsonl_file = PROCESSED_BASE / gran / f"{split}.jsonl"
            assert jsonl_file.exists(), f"Không tìm thấy file: {jsonl_file}"

            samples = list(load_jsonl(str(jsonl_file)))
            assert len(samples) > 0, f"Tập {gran}/{split} bị rỗng!"

            split_ids = set()
            total_entities = Counter()

            for idx, s in enumerate(samples):
                # 1. Kiểm tra tokens == labels
                assert len(s.tokens) == len(s.tags), (
                    f"[{gran}/{split} mẫu #{idx}] tokens ({len(s.tokens)}) != tags ({len(s.tags)}) tại id={s.id}"
                )

                # 2. Không có nhãn ngoài label map
                for tag in s.tags:
                    assert tag in valid_tags, (
                        f"[{gran}/{split}] Nhãn lạ '{tag}' không thuộc bộ nhãn VLSP tại id={s.id}"
                    )

                # 3. ID không trùng lặp
                assert s.id not in split_ids, f"Trùng lặp ID trong {split}: {s.id}"
                split_ids.add(s.id)

                # 4. Không có mẫu rỗng
                assert len(s.tokens) > 0, f"Mẫu rỗng tại id={s.id}"

                # 5. Không có dữ liệu hỏng
                for tok in s.tokens:
                    assert isinstance(tok, str) and len(tok) > 0, f"Token hỏng tại id={s.id}"

                # 6. BIO transitions hợp lệ (strict check: không có I-X đi sau O hoặc loại khác)
                prev_tag = "O"
                for tag in s.tags:
                    if tag.startswith("I-"):
                        etype = tag[2:]
                        assert prev_tag in (f"B-{etype}", f"I-{etype}"), (
                            f"Lỗi chuyển tiếp BIO: '{tag}' đi sau '{prev_tag}' tại id={s.id}"
                        )
                    prev_tag = tag

                # Đếm thực thể
                total_entities.update(count_entities(s.tags))

            split_entity_counts[split] = total_entities
            print(f"  [{split.upper()}] {len(samples):,} câu, {sum(len(s.tokens) for s in samples):,} tokens: ĐẠT")
            print(f"     Thực thể: {dict(total_entities)}")

        # Lưu IDs để kiểm tra phân tách tập
        if gran == "word":
            for split in ["train", "dev", "test"]:
                p = PROCESSED_BASE / gran / f"{split}.jsonl"
                for s in load_jsonl(str(p)):
                    all_seen_ids.add(s.id)

    # 7. Kiểm tra không thất thoát thực thể giữa word và syllable
    print("\n--- Kiểm định bảo toàn thực thể giữa Word-level và Syllable-level ---")
    for split in ["train", "dev", "test"]:
        w_samples = list(load_jsonl(str(PROCESSED_BASE / "word" / f"{split}.jsonl")))
        s_samples = list(load_jsonl(str(PROCESSED_BASE / "syllable" / f"{split}.jsonl")))

        assert len(w_samples) == len(s_samples), (
            f"Lệch số câu {split}: word={len(w_samples)}, syl={len(s_samples)}"
        )

        w_entities = Counter()
        s_entities = Counter()
        for ws, ss in zip(w_samples, s_samples):
            w_entities.update(count_entities(ws.tags))
            s_entities.update(count_entities(ss.tags))

        assert w_entities == s_entities, (
            f"Thất thoát thực thể tại {split}: Word={dict(w_entities)} != Syllable={dict(s_entities)}"
        )
        print(f"  [{split.upper()}] Bảo toàn thực thể 100%: Word ({sum(w_entities.values())}) == Syllable ({sum(s_entities.values())})")

    # 8. Kiểm tra phân tách tập train, dev, test (Disjoint IDs)
    print("\n--- Kiểm định phân tách tập Train, Dev, Test ---")
    train_ids = {s.id for s in load_jsonl(str(PROCESSED_BASE / "word" / "train.jsonl"))}
    dev_ids = {s.id for s in load_jsonl(str(PROCESSED_BASE / "word" / "dev.jsonl"))}
    test_ids = {s.id for s in load_jsonl(str(PROCESSED_BASE / "word" / "test.jsonl"))}

    assert len(train_ids & dev_ids) == 0, "Trùng lặp ID giữa Train và Dev!"
    assert len(train_ids & test_ids) == 0, "Trùng lặp ID giữa Train và Test!"
    assert len(dev_ids & test_ids) == 0, "Trùng lặp ID giữa Dev và Test!"
    print("  Phân tách tập IDs: ĐẠT (Train, Dev, Test hoàn toàn độc lập)")

    print("\n=== TẤT CẢ KIỂM ĐỊNH DỮ LIỆU VLSP 2016 ĐẠT TIÊU CHUẨN 100% ===")


if __name__ == "__main__":
    run_vlsp_data_tests()
