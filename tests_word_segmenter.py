"""Kiểm thử word_segmenter.py: phần chiếu nhãn (không cần Java) + phần VnCoreNLP thật (nếu có Java)."""
import os, shutil, tempfile
from src.common.schema import NERSample, load_jsonl, save_jsonl
from src.common.word_segmenter import (project_tags, WordSegmenter, AlignmentError,
                                       convert_file)

# --- 1. project_tags ---
# "Hà Nội" (B-LOC I-LOC) gộp thành 1 từ -> B-LOC
assert project_tags(["ở", "Hà", "Nội"], ["O", "B-LOC", "I-LOC"], ["ở", "Hà_Nội"]) == (["O", "B-LOC"], 0)
# thực thể 3 âm tiết, tách thành 2 từ -> B-PER, I-PER
assert project_tags(["Nguyễn", "Khắc", "Chúc"], ["B-PER", "I-PER", "I-PER"],
                    ["Nguyễn_Khắc", "Chúc"]) == (["B-PER", "I-PER"], 0)
# từ bắt đầu giữa token ("TP.HCM" -> "TP." + "HCM"): B- chuyển thành I-
assert project_tags(["ở", "TP.HCM"], ["O", "B-LOC"], ["ở", "TP.", "HCM"]) == (["O", "B-LOC", "I-LOC"], 0)
# từ vắt qua biên thực thể: "Nam" (B-NAME) + "bị" (O) gộp -> lấy nhãn đầu, đếm conflict
assert project_tags(["Nam", "bị", "sốt"], ["B-NAME", "O", "O"], ["Nam_bị", "sốt"]) == (["B-NAME", "O"], 1)
# O rồi B-X trong cùng một từ cũng là conflict
assert project_tags(["đã", "Hà", "Nội"], ["O", "B-LOC", "I-LOC"], ["đã_Hà", "Nội"]) == (["O", "I-LOC"], 1)
# lệch ký tự -> AlignmentError
for bad in (lambda: project_tags(["a", "b"], ["O", "O"], ["a", "c"]),
            lambda: project_tags(["a"], ["O"], ["a", "b"])):
    try: bad(); raise SystemExit("phải ném AlignmentError")
    except AlignmentError: pass

# --- 2. segment_sample với bộ tách giả (kiểm tra id, granularity, validate) ---
fake = WordSegmenter(segment_fn=lambda t: ["Bệnh_nhân số 204 ở Hà_Nội ."])
s = NERSample("vlsp_syllable_test_00003", "vlsp", "syllable",
              ["Bệnh", "nhân", "số", "204", "ở", "Hà", "Nội", "."],
              ["O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"])
new, c = fake.segment_sample(s)
assert new.id == "vlsp_word_test_00003" and new.granularity == "word" and c == 0
assert new.tokens == ["Bệnh_nhân", "số", "204", "ở", "Hà_Nội", "."]
assert new.tags == ["O", "O", "O", "O", "B-LOC", "O"]
try: fake.segment_sample(new); raise SystemExit("phải lỗi")      # đầu vào đã là mức word
except ValueError: pass
print("Test chiếu nhãn ĐẠT")

# --- 3. VnCoreNLP thật (bỏ qua nếu không có Java hoặc py_vncorenlp) ---
if shutil.which("java") is None:
    print("Bỏ qua test VnCoreNLP thật: không có Java"); raise SystemExit(0)
try:
    import py_vncorenlp  # noqa
except ImportError:
    print("Bỏ qua test VnCoreNLP thật: chưa cài py_vncorenlp"); raise SystemExit(0)

d = os.environ.get("VNCORENLP_TEST_DIR", os.path.join(tempfile.gettempdir(), "vncorenlp_min"))
cwd = os.getcwd()
seg = WordSegmenter(save_dir=d)
assert seg.segment_text("Bệnh nhân số 204 ở Hà Nội .")[:2] == ["Bệnh_nhân", "số"]
assert os.getcwd() == cwd, "py_vncorenlp đã đổi thư mục làm việc!"
assert not os.path.exists(os.path.join(d, "models", "ner")), "đã tải thừa mô hình NER"
print("VnCoreNLP:", seg.segment_text("Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội."))

# Chạy cả file: syllable -> word, câu có "TP.HCM" (VnCoreNLP tách thành 'TP.' + 'HCM')
syl = [
    NERSample("phoner_syllable_test_00001", "phoner_covid19", "syllable",
              ["Bệnh", "nhân", "số", "204", "ở", "Hà", "Nội", "."],
              ["O", "O", "O", "B-PATIENT_ID", "O", "B-LOCATION", "I-LOCATION", "O"]),
    NERSample("phoner_syllable_test_00002", "phoner_covid19", "syllable",
              ["Ca", "nhiễm", "ở", "TP.HCM", "tăng"], ["O", "O", "O", "B-LOCATION", "O"]),
]
with tempfile.TemporaryDirectory() as t:
    pin, pout = os.path.join(t, "in.jsonl"), os.path.join(t, "out.jsonl")
    save_jsonl(syl, pin)
    stats = convert_file(pin, pout, segmenter=seg)
    res = list(load_jsonl(pout))
    print(stats)
    for r in res: print(r.id, list(zip(r.tokens, r.tags)))
    assert [r.id for r in res] == ["phoner_word_test_00001", "phoner_word_test_00002"]
    assert ("Hà_Nội", "B-LOCATION") in list(zip(res[0].tokens, res[0].tags))
    assert stats.n_failed == 0
print("TẤT CẢ TEST word_segmenter ĐẠT")
