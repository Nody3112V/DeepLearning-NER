"""
evaluate.py - Quy trình đánh giá chính thức cho nhánh Pretrained PLM (PhoBERT & XLM-R).

Tuân thủ nghiêm ngặt các quy định của đề bài:
  1. Kiểm tra Evaluation Protocol:
     - Xác nhận Test IDs, Test Split, Metrics chung, Label Mapping.
     - Kiểm tra tính hợp lệ của dữ liệu trước khi đánh giá.
  2. Phục hồi dự đoán từ Subwords về Tokens gốc:
     - Loại bỏ special tokens (<s>, </s>, <pad>).
     - Trích xuất nhãn dự đoán tại subword đầu tiên của từng từ gốc.
     - Đảm bảo ánh xạ 1-1: len(pred_tags) == len(sample.tokens).
     - Bảo toàn ID và thứ tự câu nguyên bản.
  3. Đánh giá chuẩn Entity-level qua src.common.metrics:
     - Micro-F1, Macro-F1, Precision, Recall.
     - Per-type F1 (PER, LOC, ORG, MISC).
     - Thống kê tổng số thực thể thực tế (Support) và dự đoán (TP + FP).
  4. Xử lý ngoại lệ & Kiểm định tính toàn vẹn:
     - Câu không có thực thể (all 'O'), câu nhiều thực thể.
     - Kiểm tra không trùng ID, không thiếu mẫu, không có ID lạ ngoài test set.
     - Kiểm tra round-trip đọc lại tệp dự đoán JSONL.
  5. Xuất kết quả chuẩn hóa:
     - Tệp dự đoán chính thức: outputs/predictions/pretrained_test.jsonl.
     - Tệp bảng số liệu: outputs/tables/official_pretrained_test_metrics.json.
     - Báo cáo định dạng Markdown & LaTeX.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.common.schema import load_jsonl, save_jsonl, NERSample, TAG_SETS
from src.common.metrics import compute_metrics, format_report, evaluate_files, to_latex_table
from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.subword_align import NERDataCollator


# --------------------------------------------------------------------------- #
# 1. Kiểm tra Evaluation Protocol
# --------------------------------------------------------------------------- #
def validate_evaluation_protocol(
    gold_path: Union[str, Path],
    dataset_name: str = "vlsp",
    expected_count: Optional[int] = 1872,
) -> List[NERSample]:
    """
    Kiểm định nghiêm ngặt protocol đánh giá trước khi thực hiện inference.
    Nếu thiếu file hoặc sai schema -> ném ngoại lệ (BLOCKED).
    """
    gold_file = Path(gold_path)
    if not gold_file.exists():
        raise FileNotFoundError(f"[BLOCKED] Không tìm thấy file dữ liệu test: {gold_file}")

    expected_tags = TAG_SETS.get(dataset_name)
    if not expected_tags:
        raise ValueError(f"[BLOCKED] Không tìm thấy bộ nhãn hợp lệ cho dataset '{dataset_name}'")

    samples = list(load_jsonl(str(gold_file), validate=True))

    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(
            f"[BLOCKED] Số lượng mẫu test ({len(samples)}) không khớp với số lượng chuẩn ({expected_count})!"
        )

    # Kiểm tra trùng lặp ID
    ids = [s.id for s in samples]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        raise ValueError(f"[BLOCKED] Phát hiện ID trùng lặp trong tập test: {len(ids) - len(unique_ids)} mẫu bị trùng!")

    # Kiểm tra tính hợp lệ của từng mẫu
    for idx, s in enumerate(samples):
        if len(s.tokens) == 0:
            raise ValueError(f"[BLOCKED] Mẫu #{idx} (id={s.id}) có danh sách tokens rỗng!")
        if len(s.tokens) != len(s.tags):
            raise ValueError(
                f"[BLOCKED] Mẫu #{idx} (id={s.id}) lệch độ dài: len(tokens)={len(s.tokens)} != len(tags)={len(s.tags)}"
            )
        for t in s.tags:
            if t not in expected_tags:
                raise ValueError(f"[BLOCKED] Mẫu #{idx} chứa nhãn không hợp lệ '{t}' ngoài TAG_SETS!")

    print(f"[*] Evaluation Protocol Verified: {len(samples)} mẫu test hợp lệ ({dataset_name}).")
    return samples


# --------------------------------------------------------------------------- #
# 2. Phục hồi dự đoán từ Subwords về Tokens gốc
# --------------------------------------------------------------------------- #
class InferenceSampleDataset(Dataset):
    """Dataset phục vụ dự đoán hàng loạt với subword mapping được theo dõi chính xác."""

    def __init__(self, samples: List[NERSample], tokenizer: Any, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tokens = sample.tokens

        subword_ids: List[int] = []
        token_to_sw_idx: List[int] = []

        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id

        if cls_id is not None:
            subword_ids.append(cls_id)

        for tok in tokens:
            sws = self.tokenizer.encode(tok, add_special_tokens=False)
            if not sws:
                sws = [self.tokenizer.unk_token_id]

            # Ghi nhận chỉ số subword đầu tiên nếu còn trong phạm vi max_length
            if len(subword_ids) < self.max_length - (1 if sep_id is not None else 0):
                token_to_sw_idx.append(len(subword_ids))
            else:
                token_to_sw_idx.append(-1)  # Token bị truncated

            subword_ids.extend(sws)

        # Truncation
        if sep_id is not None:
            if len(subword_ids) >= self.max_length:
                subword_ids = subword_ids[:self.max_length - 1] + [sep_id]
            else:
                subword_ids.append(sep_id)
        else:
            subword_ids = subword_ids[:self.max_length]

        attention_mask = [1] * len(subword_ids)

        return {
            "id": sample.id,
            "tokens": tokens,
            "gold_tags": sample.tags,
            "input_ids": subword_ids,
            "attention_mask": attention_mask,
            "token_to_sw_idx": token_to_sw_idx,
        }


def inference_collate_fn(pad_token_id: int):
    """Collate function cho inference (dynamic padding)."""
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        batch_input_ids = []
        batch_attention_mask = []

        for item in batch:
            cur_len = len(item["input_ids"])
            pad_len = max_len - cur_len
            batch_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)

        return {
            "ids": [item["id"] for item in batch],
            "tokens": [item["tokens"] for item in batch],
            "gold_tags": [item["gold_tags"] for item in batch],
            "token_to_sw_idx": [item["token_to_sw_idx"] for item in batch],
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }
    return collate


@torch.no_grad()
def predict_and_recover_tokens(
    model: PretrainedNERModel,
    tokenizer: Any,
    id2tag: Dict[int, str],
    samples: List[NERSample],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[List[NERSample], float]:
    """
    Thực hiện dự đoán và phục hồi nhãn về mức token gốc.

    Returns:
        Tuple gồm (danh_sách_NERSample_dự_đoán, thời_gian_suy_luận_giây)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ds = InferenceSampleDataset(samples, tokenizer, max_length=256)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=inference_collate_fn(pad_id))

    pred_samples: List[NERSample] = []
    t_start = time.time()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        for b in range(len(batch["ids"])):
            s_id = batch["ids"][b]
            tokens = batch["tokens"][b]
            gold_tags = batch["gold_tags"][b]
            token_map = batch["token_to_sw_idx"][b]
            sent_preds = preds[b]

            recovered_tags: List[str] = []
            for t_idx, sw_idx in enumerate(token_map):
                if sw_idx != -1 and sw_idx < len(sent_preds):
                    label_id = int(sent_preds[sw_idx])
                    recovered_tags.append(id2tag.get(label_id, "O"))
                else:
                    # Trường hợp bị truncation vượt quá 256 subwords
                    recovered_tags.append("O")

            # Bất biến: số nhãn phục hồi phải bằng đúng số token gốc
            assert len(recovered_tags) == len(tokens), (
                f"Lỗi phục hồi tại id={s_id}: len(tags)={len(recovered_tags)} != len(tokens)={len(tokens)}"
            )

            pred_sample = NERSample(
                id=s_id,
                dataset="vlsp",
                granularity="word",
                tokens=tokens,
                tags=recovered_tags,
            )
            # Kiểm tra hợp lệ định dạng schema
            pred_sample.validate()
            pred_samples.append(pred_sample)

    elapsed_time = time.time() - t_start
    return pred_samples, elapsed_time


# --------------------------------------------------------------------------- #
# 3. Phân tích Thực thể & Thống kê Chi tiết
# --------------------------------------------------------------------------- #
def analyze_entity_counts(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Tính toán tổng số thực thể thực tế (Support) và dự đoán (TP + FP)."""
    per_type_stats = {}
    total_gold = 0
    total_pred = 0

    for etype, data in metrics.get("per_type", {}).items():
        tp = data.get("tp", 0)
        fp = data.get("fp", 0)
        fn = data.get("fn", 0)
        support = data.get("support", tp + fn)
        pred_count = tp + fp
        total_gold += support
        total_pred += pred_count

        per_type_stats[etype] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_support": support,
            "pred_count": pred_count,
            "precision": data.get("precision", 0.0),
            "recall": data.get("recall", 0.0),
            "f1": data.get("f1", 0.0),
        }

    overall_tp = metrics.get("overall", {}).get("tp", 0)
    overall_fp = metrics.get("overall", {}).get("fp", 0)
    overall_fn = metrics.get("overall", {}).get("fn", 0)

    return {
        "total_gold_entities": total_gold,
        "total_predicted_entities": total_pred,
        "overall_tp": overall_tp,
        "overall_fp": overall_fp,
        "overall_fn": overall_fn,
        "per_type_breakdown": per_type_stats,
    }


# --------------------------------------------------------------------------- #
# 4. Hàm Thực thi Đánh giá Chính thức
# --------------------------------------------------------------------------- #
def run_official_evaluation(
    checkpoint_path: str = "outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt",
    gold_path: str = "data/processed/vlsp/word/test.jsonl",
    output_pred_path: str = "outputs/predictions/pretrained_test.jsonl",
    dataset_name: str = "vlsp",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Quy trình đánh giá chính thức khép kín.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  QUY TRÌNH ĐÁNH GIÁ CHÍNH THỨC - PRETRAINED PLM")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Gold Data : {gold_path}")
    print(f"  Output Pred: {output_pred_path}")
    print(f"  Thiết bị   : {device}")
    print(f"=======================================================\n")

    # BƯỚC 1: Xác nhận Evaluation Protocol
    gold_samples = validate_evaluation_protocol(gold_path, dataset_name=dataset_name, expected_count=1872)

    # BƯỚC 2: Nạp Checkpoint & Tokenizer
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[BLOCKED] Không tìm thấy file checkpoint: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt["config"]
    id2tag = {int(k): v for k, v in ckpt["id2tag"].items()}
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("model_name_or_path", "vinai/phobert-base-v2")
    pooling_strategy = model_cfg.get("pooling_strategy", "last")

    print(f"[*] Nạp Tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print(f"[*] Khởi tạo mô hình PretrainedNERModel (pooling={pooling_strategy}, labels={len(id2tag)})...")
    model = PretrainedNERModel(
        model_name_or_path=model_name,
        num_labels=len(id2tag),
        pooling_strategy=pooling_strategy,
        classifier_dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # BƯỚC 3: Dự đoán & Phục hồi về mức token gốc
    print(f"[*] Đang thực hiện suy luận trên 1,872 câu test...")
    pred_samples, infer_time = predict_and_recover_tokens(
        model=model,
        tokenizer=tokenizer,
        id2tag=id2tag,
        samples=gold_samples,
        batch_size=32,
        device=device,
    )

    # BƯỚC 4: Xuất file dự đoán chuẩn JSONL
    out_pred = Path(output_pred_path)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(pred_samples, str(out_pred))
    print(f"[*] Đã lưu {len(pred_samples)} dự đoán chuẩn vào: {out_pred}")

    # BƯỚC 5: Kiểm định tính toàn vẹn của tệp xuất
    reloaded_preds = list(load_jsonl(str(out_pred), validate=True))
    assert len(reloaded_preds) == len(gold_samples), "Lệch số lượng mẫu khi nạp lại!"
    for g, p in zip(gold_samples, reloaded_preds):
        assert g.id == p.id, f"Lệch ID: gold={g.id} != pred={p.id}"
        assert len(g.tokens) == len(p.tokens) == len(p.tags), f"Lệch tokens/tags tại id={g.id}"
    print(f"[*] Kiểm định Round-trip: 100% mẫu nạp lại hợp lệ và khớp ID.")

    # BƯỚC 6: Đánh giá bằng src.common.metrics (evaluate_files)
    metrics = evaluate_files(str(gold_path), str(out_pred))
    report_str = format_report(metrics)
    print("\n" + "=" * 50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC (ENTITY-LEVEL):")
    print("=" * 50)
    print(report_str)

    entity_analysis = analyze_entity_counts(metrics)
    print("\nTHỐNG KÊ CHI TIẾT SỐ LƯỢNG THỰC THỂ:")
    print(f"  - Tổng thực thể thực tế (Ground-Truth Support): {entity_analysis['total_gold_entities']:,}")
    print(f"  - Tổng thực thể mô hình dự đoán (TP + FP)     : {entity_analysis['total_predicted_entities']:,}")
    print(f"  - Số thực thể dự đoán đúng (True Positives)    : {entity_analysis['overall_tp']:,}")
    print(f"  - Số thực thể dự đoán sai (False Positives)    : {entity_analysis['overall_fp']:,}")
    print(f"  - Số thực thể bỏ sót (False Negatives)         : {entity_analysis['overall_fn']:,}")
    print(f"  - Thời gian suy luận toàn bộ 1,872 câu         : {infer_time:.2f}s ({infer_time/1872*1000:.2f} ms/câu)")

    # BƯỚC 7: Xuất bảng kết quả JSON, Markdown, LaTeX
    out_table_dir = Path("outputs/tables")
    out_table_dir.mkdir(parents=True, exist_ok=True)

    summary_results = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "pooling_strategy": pooling_strategy,
        "dataset": dataset_name,
        "granularity": "word",
        "num_test_samples": len(gold_samples),
        "inference_time_s": infer_time,
        "ms_per_sentence": infer_time / len(gold_samples) * 1000,
        "overall": metrics["overall"],
        "macro_f1": metrics["macro_f1"],
        "per_type": metrics["per_type"],
        "entity_counts": entity_analysis,
    }

    # 1. JSON
    json_path = out_table_dir / "official_pretrained_test_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã lưu bảng số liệu JSON vào: {json_path}")

    # 2. Markdown Report
    md_path = out_table_dir / "official_pretrained_test_report.md"
    md_content = [
        "# Báo cáo Đánh giá Chính thức - Nhánh Pretrained PLM",
        "",
        f"- **Mô hình**: `{model_name}` (Pooling: `{pooling_strategy}`)",
        f"- **Tập dữ liệu**: `{dataset_name}` (Mức: `word`)",
        f"- **Số lượng câu test**: `{len(gold_samples):,}`",
        f"- **Tốc độ suy luận**: `{infer_time:.2f}s` (`{infer_time/len(gold_samples)*1000:.2f} ms/câu` trên GPU)",
        "",
        "## 1. Bảng Điểm Entity-level Chuẩn (src.common.metrics)",
        "",
        "| Loại Thực thể | Thực tế (Support) | Dự đoán (TP+FP) | Đúng (TP) | Precision (%) | Recall (%) | F1-Score (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for etype, data in entity_analysis["per_type_breakdown"].items():
        md_content.append(
            f"| **{etype}** | {data['gold_support']} | {data['pred_count']} | {data['tp']} | "
            f"{data['precision']*100:.2f}% | {data['recall']*100:.2f}% | **{data['f1']*100:.2f}%** |"
        )
    o = metrics["overall"]
    md_content.extend([
        f"| **MICRO OVERALL** | **{entity_analysis['total_gold_entities']}** | **{entity_analysis['total_predicted_entities']}** | **{entity_analysis['overall_tp']}** | **{o['precision']*100:.2f}%** | **{o['recall']*100:.2f}%** | **{o['f1']*100:.2f}%** |",
        f"| **MACRO-F1** | - | - | - | - | - | **{metrics['macro_f1']*100:.2f}%** |",
        "",
        "## 2. Kiểm định Tính Toàn vẹn & Error Handling",
        "- [x] 100% mẫu test (1,872 câu) được phục hồi nhãn ở mức token gốc.",
        "- [x] Độ dài `tokens` và `tags` khớp 1-1 trên mọi câu.",
        "- [x] Mọi ID khớp 100% với file chuẩn `data/processed/vlsp/word/test.jsonl`.",
        "- [x] Xử lý an toàn các câu không có thực thể và câu có nhiều thực thể.",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"[*] Đã lưu báo cáo Markdown vào: {md_path}")

    # 3. LaTeX Table
    tex_path = out_table_dir / "official_pretrained_test_table.tex"
    latex_code = to_latex_table(
        {f"PhoBERT ({pooling_strategy})": metrics},
        caption="Kết quả đánh giá chính thức của PhoBERT trên tập test VLSP 2016."
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"[*] Đã lưu bảng LaTeX vào: {tex_path}")

    return summary_results


# --------------------------------------------------------------------------- #
# CLI Entrypoint
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Đánh giá chính thức mô hình Pretrained PLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt",
        help="Đường dẫn file checkpoint .pt",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default="data/processed/vlsp/word/test.jsonl",
        help="Đường dẫn file nhãn chuẩn test.jsonl",
    )
    parser.add_argument(
        "--output_pred",
        type=str,
        default="outputs/predictions/pretrained_test.jsonl",
        help="Đường dẫn xuất file dự đoán JSONL",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    run_official_evaluation(
        checkpoint_path=args.checkpoint,
        gold_path=args.gold,
        output_pred_path=args.output_pred,
        device=device,
    )


if __name__ == "__main__":
    main()
