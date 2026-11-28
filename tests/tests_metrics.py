"""Kiểm thử metrics.py bằng dữ liệu giả + đối chiếu seqeval (nếu có cài)."""
import random
from src.common.metrics import compute_metrics, extract_entities, to_latex_table

# --- 1. Trích xuất thực thể ---
assert extract_entities(["O", "B-LOC", "I-LOC", "O", "B-PER"]) == {(1, 3, "LOC"), (4, 5, "PER")}
assert extract_entities(["I-LOC", "I-LOC"]) == {(0, 2, "LOC")}                # I- đầu câu
assert extract_entities(["B-LOC", "I-PER"]) == {(0, 1, "LOC"), (1, 2, "PER")}  # I- khác loại
assert extract_entities(["B-LOC", "B-LOC"]) == {(0, 1, "LOC"), (1, 2, "LOC")}
assert extract_entities(["B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE"]) == {(0, 2, "SYMPTOM_AND_DISEASE")}

# --- 2. Dữ liệu giả có đáp án tính tay ---
gold = [["B-PER", "I-PER", "O", "B-LOC"],      # 2 thực thể
        ["O", "B-ORG", "I-ORG", "O"]]          # 1 thực thể
pred = [["B-PER", "I-PER", "O", "O"],          # PER đúng, LOC bỏ sót
        ["O", "B-ORG", "O", "B-LOC"]]          # ORG sai biên, LOC thừa
r = compute_metrics(gold, pred)
o = r["overall"]
assert (o["tp"], o["fp"], o["fn"]) == (1, 2, 2)
assert abs(o["precision"] - 1/3) < 1e-9 and abs(o["recall"] - 1/3) < 1e-9
assert r["per_type"]["PER"]["f1"] == 1.0
assert r["per_type"]["LOC"]["f1"] == 0.0 and r["per_type"]["ORG"]["f1"] == 0.0

# Dự đoán hoàn hảo -> F1 = 1; không có thực thể nào -> 0 (không chia cho 0)
assert compute_metrics(gold, gold)["overall"]["f1"] == 1.0
assert compute_metrics([["O", "O"]], [["O", "O"]])["overall"]["f1"] == 0.0

# Độ dài lệch phải báo lỗi
for bad in (lambda: compute_metrics([["O"]], [["O", "O"]]), lambda: compute_metrics([["O"]], [])):
    try: bad(); raise SystemExit("phải báo lỗi")
    except ValueError: pass

print(to_latex_table({"Demo": r})[:60].replace("\n", " | "), "...")

# --- 3. Đối chiếu seqeval trên dữ liệu ngẫu nhiên ---
try:
    from seqeval.metrics import f1_score, precision_score, recall_score
except ImportError:
    print("Bỏ qua đối chiếu seqeval (chưa cài)"); raise SystemExit(0)

random.seed(0)
types = ["PER", "LOC", "ORG", "SYMPTOM_AND_DISEASE"]
def rand_seq(n):
    return [random.choice(["O", "O"] + [f"{p}-{t}" for p in "BI" for t in types]) for _ in range(n)]
G = [rand_seq(random.randint(1, 15)) for _ in range(2000)]
P = [[t if random.random() < .6 else random.choice(["O"] + [f"{p}-{x}" for p in "BI" for x in types])
      for t in g] for g in G]
mine = compute_metrics(G, P)["overall"]
for name, fn, key in [("P", precision_score, "precision"), ("R", recall_score, "recall"), ("F1", f1_score, "f1")]:
    ref = fn(G, P)
    assert abs(ref - mine[key]) < 1e-9, (name, ref, mine[key])
print("Khớp seqeval trên 2000 câu ngẫu nhiên (cả I- không có B- trước)")
print("TẤT CẢ TEST ĐỀU ĐẠT")
