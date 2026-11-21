"""Kiểm thử schema.py."""
import json, os, tempfile
from src.common.schema import (NERSample, load_jsonl, save_jsonl, parse_sample_id,
                               make_sample_id, get_tag_set)

def ok(**kw):
    d = dict(id="phoner_word_train_00001", dataset="phoner_covid19", granularity="word",
             tokens=["Bệnh_nhân", "số", "204", "ở", "Hà_Nội"],
             tags=["O", "O", "B-PATIENT_ID", "O", "B-LOCATION"])
    d.update(kw); return NERSample(**d)

def must_fail(fn):
    try: fn()
    except ValueError: return
    raise SystemExit("phải ném ValueError")

# --- validate ---
ok().validate()
must_fail(lambda: ok(tokens=[]).validate())
must_fail(lambda: ok(tags=["O"]).validate())                          # lệch độ dài
must_fail(lambda: ok(tags=["O", "O", "B-ABC", "O", "O"]).validate())  # loại không thuộc PhoNER
must_fail(lambda: ok(tags=["O", "O", "B-", "O", "O"]).validate())     # thiếu loại
must_fail(lambda: ok(tags=["O", "O", "X-NAME", "O", "O"]).validate()) # sai tiền tố
must_fail(lambda: ok(dataset="abc").validate())
must_fail(lambda: ok(granularity="char").validate())
ok(tags=["O", "O", "I-NAME", "O", "O"]).validate()                    # mặc định: cho phép I- lẻ
must_fail(lambda: ok(tags=["O", "O", "I-NAME", "O", "O"]).validate(strict_bio=True))
ok(tags=["B-NAME", "I-NAME", "O", "O", "O"]).validate(strict_bio=True)
# VLSP chưa khai báo loại -> chỉ kiểm tra định dạng
assert get_tag_set("vlsp") is None
ok(id="vlsp_word_train_00001", dataset="vlsp", tags=["O", "O", "B-PER", "O", "B-LOC"]).validate()

# --- id ---
assert parse_sample_id("phoner_word_train_00001") == {
    "dataset": "phoner_covid19", "granularity": "word", "split": "train", "index": "00001"}
assert make_sample_id("vlsp", "syllable", "test", 12) == "vlsp_syllable_test_00012"
must_fail(lambda: parse_sample_id("foo_bar"))
must_fail(lambda: parse_sample_id("zzz_word_train_00001"))

# --- đọc/ghi, gồm cả định dạng 3 trường theo chuẩn thống nhất ---
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.jsonl")
    assert save_jsonl([ok(), ok(id="phoner_word_train_00002")], p) == 2
    back = list(load_jsonl(p))
    assert back[0] == ok() and len(back) == 2

    p3 = os.path.join(d, "3fields.jsonl")           # chỉ id/tokens/tags
    with open(p3, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "phoner_syllable_test_00007", "tokens": ["Hà", "Nội"],
                            "tags": ["B-LOCATION", "I-LOCATION"]}, ensure_ascii=False) + "\n\n")
    s = next(load_jsonl(p3))
    assert (s.dataset, s.granularity) == ("phoner_covid19", "syllable")

    pb = os.path.join(d, "bad.jsonl")               # báo lỗi kèm số dòng
    with open(pb, "w", encoding="utf-8") as f:
        f.write(ok().to_json_line() + "\n")
        f.write(json.dumps({"id": "phoner_word_train_00009", "tokens": ["a"], "tags": []}) + "\n")
    try: list(load_jsonl(pb)); raise SystemExit("phải lỗi")
    except ValueError as e: assert ":2:" in str(e), e
print("TẤT CẢ TEST schema ĐẠT")
