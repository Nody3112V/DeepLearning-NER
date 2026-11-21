"""
metrics.py - Entity-level F1 dùng chung cho toàn dự án (TV3 phụ trách).

QUY ƯỚC (đã đóng băng từ cuối Tuần 1, không đổi cách tính về sau):
  * Một thực thể đúng khi KHỚP CHÍNH XÁC cả biên (start, end) lẫn loại thực thể.
  * Nhãn dạng BIO: "O", "B-<TYPE>", "I-<TYPE>" (TYPE có thể chứa dấu "_").
  * Một nhãn "I-X" đứng sau "O" hoặc sau nhãn loại khác được coi là bắt đầu
    thực thể mới (cách xử lý giống conlleval / seqeval chế độ mặc định).
  * Micro-F1 (gộp toàn bộ thực thể) là chỉ số chính; macro-F1 là chỉ số phụ.
  * Chia cho 0 -> trả về 0.0.

Cách dùng trong code:
    from src.common.metrics import compute_metrics
    res = compute_metrics(gold_tags, pred_tags)   # list[list[str]]
    print(res["overall"]["f1"], res["per_type"]["LOCATION"]["f1"])

Cách dùng từ dòng lệnh (2 file JSONL theo schema {id, tokens, tags}):
    python -m src.common.metrics --gold data/processed/phoner_covid19/word/test.jsonl \
                                 --pred outputs/predictions/bilstm_crf_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

Span = Tuple[int, int, str]  # (start, end_exclusive, entity_type)


# --------------------------------------------------------------------------- #
# 1. Trích xuất thực thể từ chuỗi nhãn BIO
# --------------------------------------------------------------------------- #
def _split_tag(tag: str) -> Tuple[str, str]:
    """'B-LOCATION' -> ('B', 'LOCATION'); 'O' -> ('O', '')."""
    if tag == "O":
        return "O", ""
    if len(tag) > 2 and tag[1] == "-" and tag[0] in ("B", "I"):
        return tag[0], tag[2:]
    raise ValueError(f"Nhãn BIO không hợp lệ: {tag!r}")


def extract_entities(tags: List[str]) -> Set[Span]:
    """Trả về tập các thực thể (start, end_exclusive, type) của một câu."""
    spans: Set[Span] = set()
    start, cur_type = None, None

    for i, tag in enumerate(tags):
        prefix, etype = _split_tag(tag)
        starts_new = prefix == "B" or (prefix == "I" and etype != cur_type)

        # đóng thực thể đang mở nếu gặp O, B-*, hoặc I- khác loại
        if cur_type is not None and (prefix == "O" or starts_new):
            spans.add((start, i, cur_type))
            start, cur_type = None, None

        if starts_new:
            start, cur_type = i, etype

    if cur_type is not None:  # thực thể kết thúc ở cuối câu
        spans.add((start, len(tags), cur_type))
    return spans


# --------------------------------------------------------------------------- #
# 2. Tính điểm
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def compute_metrics(gold: List[List[str]], pred: List[List[str]]) -> Dict:
    """
    gold, pred: danh sách các chuỗi nhãn BIO (mỗi câu một list), cùng thứ tự.
    Trả về dict:
        {"overall":  {precision, recall, f1, tp, fp, fn, support},   # micro
         "per_type": {TYPE: {...}},
         "macro_f1": float}
    """
    if len(gold) != len(pred):
        raise ValueError(f"Số câu lệch nhau: gold={len(gold)} pred={len(pred)}")

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for idx, (g_tags, p_tags) in enumerate(zip(gold, pred)):
        if len(g_tags) != len(p_tags):
            raise ValueError(f"Câu #{idx}: độ dài nhãn lệch "
                             f"(gold={len(g_tags)}, pred={len(p_tags)})")
        g_spans, p_spans = extract_entities(g_tags), extract_entities(p_tags)
        for s in g_spans & p_spans:
            counts[s[2]]["tp"] += 1
        for s in p_spans - g_spans:
            counts[s[2]]["fp"] += 1
        for s in g_spans - p_spans:
            counts[s[2]]["fn"] += 1

    per_type = {t: _prf(**c) for t, c in sorted(counts.items())}
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    macro_f1 = (sum(m["f1"] for m in per_type.values()) / len(per_type)
                if per_type else 0.0)
    return {"overall": _prf(tp, fp, fn), "per_type": per_type, "macro_f1": macro_f1}


# --------------------------------------------------------------------------- #
# 3. Đọc file JSONL và đối chiếu theo id (schema: id, tokens, tags)
# --------------------------------------------------------------------------- #
def _read_tags(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if len(rec["tokens"]) != len(rec["tags"]):
                raise ValueError(f"len(tokens) != len(tags) tại id={rec['id']}")
            out[rec["id"]] = rec["tags"]
    return out


def evaluate_files(gold_path: str, pred_path: str) -> Dict:
    """Chấm điểm 2 file JSONL, ghép câu theo `id` (không phụ thuộc thứ tự dòng)."""
    gold, pred = _read_tags(gold_path), _read_tags(pred_path)
    missing = set(gold) - set(pred)
    extra = set(pred) - set(gold)
    if missing or extra:
        raise ValueError(f"id không khớp: thiếu {len(missing)} id trong pred, "
                         f"thừa {len(extra)} id (ví dụ thiếu: {sorted(missing)[:3]})")
    ids = sorted(gold)
    return compute_metrics([gold[i] for i in ids], [pred[i] for i in ids])


# --------------------------------------------------------------------------- #
# 4. Định dạng kết quả
# --------------------------------------------------------------------------- #
def format_report(res: Dict) -> str:
    rows = [("TYPE", "P", "R", "F1", "support")]
    for t, m in res["per_type"].items():
        rows.append((t, m["precision"], m["recall"], m["f1"], m["support"]))
    o = res["overall"]
    rows.append(("MICRO", o["precision"], o["recall"], o["f1"], o["support"]))
    width = max(len(str(r[0])) for r in rows) + 2
    lines = []
    for r in rows:
        if r[0] == "TYPE":
            lines.append(f"{r[0]:<{width}}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
        else:
            lines.append(f"{r[0]:<{width}}{r[1]*100:8.2f}{r[2]*100:8.2f}"
                         f"{r[3]*100:8.2f}{r[4]:9d}")
    lines.append(f"{'MACRO-F1':<{width}}{'':>16}{res['macro_f1']*100:8.2f}")
    return "\n".join(lines)


def to_latex_table(results: Dict[str, Dict], caption: str = "Entity-level F1 (\\%).",
                   label: str = "tab:main") -> str:
    """
    results: {"BiLSTM-CRF": res1, "PhoBERT": res2, ...} (mỗi res từ compute_metrics).
    Sinh bảng LaTeX micro P / R / F1 + macro-F1 để ghi vào outputs/tables/.
    """
    lines = [r"\begin{table}[t]", r"\centering",
             r"\begin{tabular}{lcccc}", r"\hline",
             r"\textbf{Model} & \textbf{P} & \textbf{R} & \textbf{Micro-F1} "
             r"& \textbf{Macro-F1} \\", r"\hline"]
    for name, res in results.items():
        o = res["overall"]
        lines.append(f"{name} & {o['precision']*100:.2f} & {o['recall']*100:.2f} "
                     f"& {o['f1']*100:.2f} & {res['macro_f1']*100:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entity-level F1 cho 2 file JSONL")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    args = ap.parse_args()
    res = evaluate_files(args.gold, args.pred)
    print(json.dumps(res, ensure_ascii=False, indent=2) if args.json
          else format_report(res))


if __name__ == "__main__":
    main()
