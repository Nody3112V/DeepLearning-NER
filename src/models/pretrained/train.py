"""
train.py - CLI entrypoint huấn luyện và đánh giá mô hình Pretrained PLM (PhoBERT & XLM-R).

Thực hiện:
  1. Nạp cấu hình từ YAML (configs/pretrained_plm.yaml) và CLI arguments.
  2. Tải và căn chỉnh dữ liệu bằng subword_align.py theo chuẩn NERSample.
  3. Khởi tạo mô hình PretrainedNERModel hỗ trợ đa tầng ẩn (pooling_strategy) và đóng băng tầng.
  4. Huấn luyện bằng PLMTrainer (AMP autocast, GradScaler, differential LRs, early stopping).
  5. Đánh giá trên tập test bằng Entity-level F1 (src.common.metrics).
  6. Xuất kết quả dự đoán test chuẩn schema ra outputs/predictions/pretrained_test.jsonl.
  7. Xuất bảng kết quả và báo cáo LaTeX ra outputs/tables/.
"""
from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import json
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.common.schema import load_jsonl, TAG_SETS, NERSample, save_jsonl
from src.common.metrics import compute_metrics, format_report, to_latex_table
from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.subword_align import NERDataset, create_ner_collate_fn, recover_word_predictions
from src.models.pretrained.trainer import PLMTrainer, set_seed


def build_tag_mappings(dataset_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Tạo ánh xạ nhãn 2 chiều tag <-> id từ schema chuẩn."""
    tags = TAG_SETS.get(dataset_name, [])
    if not tags:
        raise ValueError(f"Không tìm thấy bộ nhãn cho dataset '{dataset_name}' trong schema.TAG_SETS")
    tag2id = {tag: idx for idx, tag in enumerate(tags)}
    id2tag = {idx: tag for idx, tag in enumerate(tags)}
    return tag2id, id2tag


def export_predictions(
    test_samples: List[NERSample],
    pred_tags: List[List[str]],
    out_path: Path,
) -> None:
    """Xuất tệp dự đoán chuẩn JSONL (id, tokens, tags) để dùng chung với metrics.py và TV3."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample, preds in zip(test_samples, pred_tags):
        rec = NERSample(
            id=sample.id,
            dataset=sample.dataset,
            granularity=sample.granularity,
            tokens=sample.tokens,
            tags=preds,
        )
        records.append(rec)
    save_jsonl(records, str(out_path))
    print(f"  -> Đã lưu kết quả dự đoán tập test ({len(records)} mẫu) vào: {out_path}")


def run_pipeline(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Quy trình chạy chính."""
    # 1. Cấu hình seed và device
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Thiết bị thực thi: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Dữ liệu & Ánh xạ nhãn
    data_cfg = cfg.get("data", {})
    dataset_name = args.dataset or data_cfg.get("dataset", "vlsp")
    granularity = args.granularity or data_cfg.get("granularity", "word")

    train_path = Path(args.train_path or data_cfg.get("train_path", f"data/processed/{dataset_name}/{granularity}/train.jsonl"))
    dev_path = Path(args.dev_path or data_cfg.get("dev_path", f"data/processed/{dataset_name}/{granularity}/dev.jsonl"))
    test_path = Path(args.test_path or data_cfg.get("test_path", f"data/processed/{dataset_name}/{granularity}/test.jsonl"))

    print(f"Nạp dữ liệu: {dataset_name} ({granularity})")
    print(f"  Train: {train_path}")
    print(f"  Dev  : {dev_path}")
    print(f"  Test : {test_path}")

    tag2id, id2tag = build_tag_mappings(dataset_name)
    num_labels = len(tag2id)
    print(f"Bộ nhãn ({num_labels} nhãn): {list(tag2id.keys())}")

    # 3. Tokenizer
    model_cfg = cfg.get("model", {})
    model_name = args.model_name or model_cfg.get("model_name_or_path", "vinai/phobert-base-v2")
    max_seq_len = int(args.max_seq_len or data_cfg.get("max_seq_len", 256))

    print(f"Đang nạp Tokenizer từ: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False if "phobert" in model_name.lower() else True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False if "phobert" in model_name.lower() else True,
        )

    train_samples = list(load_jsonl(str(train_path)))
    dev_samples = list(load_jsonl(str(dev_path)))
    test_samples = list(load_jsonl(str(test_path)))

    # Nếu chạy smoke test, chỉ lấy tập con nhỏ để kiểm định end-to-end nhanh
    if args.smoke_test:
        print("[!] Chế độ SMOKE TEST: giới hạn 32 mẫu train, 16 mẫu dev, 16 mẫu test.")
        train_samples = train_samples[:32]
        dev_samples = dev_samples[:16]
        test_samples = test_samples[:16]
        cfg.setdefault("train", {})["num_epochs"] = 1

    print(f"Số lượng mẫu: Train={len(train_samples)}, Dev={len(dev_samples)}, Test={len(test_samples)}")

    train_dataset = NERDataset(train_samples, tokenizer, tag2id, max_seq_len=max_seq_len)
    dev_dataset = NERDataset(dev_samples, tokenizer, tag2id, max_seq_len=max_seq_len)
    test_dataset = NERDataset(test_samples, tokenizer, tag2id, max_seq_len=max_seq_len)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = create_ner_collate_fn(pad_token_id=pad_token_id)

    train_cfg = cfg.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Khởi tạo mô hình
    pooling_strategy = args.pooling or model_cfg.get("pooling_strategy", "last")
    freeze_embeddings = args.freeze_embeddings if args.freeze_embeddings is not None else model_cfg.get("freeze_embeddings", False)
    freeze_layers = int(args.freeze_layers if args.freeze_layers is not None else model_cfg.get("freeze_encoder_layers", 0))
    freeze_entire_encoder = args.freeze_entire_encoder if args.freeze_entire_encoder is not None else model_cfg.get("freeze_entire_encoder", False)
    dropout = float(model_cfg.get("classifier_dropout", 0.2))

    print(f"Khởi tạo mô hình PretrainedNERModel...")
    print(f"  Backbone: {model_name} | Pooling: {pooling_strategy} | Freeze Layers: {freeze_layers}")
    model = PretrainedNERModel(
        model_name_or_path=model_name,
        num_labels=num_labels,
        pooling_strategy=pooling_strategy,
        freeze_embeddings=freeze_embeddings,
        freeze_encoder_layers=freeze_layers,
        freeze_entire_encoder=freeze_entire_encoder,
        classifier_dropout=dropout,
    )

    # Cập nhật số epochs nếu được ghi đè từ CLI
    if args.epochs:
        cfg.setdefault("train", {})["num_epochs"] = int(args.epochs)

    # 5. Huấn luyện hoặc nạp Checkpoint
    trainer = PLMTrainer(
        model=model,
        train_dataloader=train_loader,
        dev_dataloader=dev_loader,
        id2tag=id2tag,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )

    best_ckpt_path = trainer.checkpoint_dir / "best_model.pt"

    if not args.eval_only:
        train_results = trainer.train()
        print(f"\nHuấn luyện hoàn tất! Best Dev F1: {train_results.get('best_f1_micro', 0.0):.4f}")
        # Nạp lại checkpoint tốt nhất trước khi test
        if best_ckpt_path.exists():
            print(f"Nạp lại checkpoint tốt nhất từ: {best_ckpt_path}")
            model, _ = PLMTrainer.load_checkpoint(best_ckpt_path, model=model, device=device)
    else:
        ckpt_to_load = Path(args.checkpoint) if args.checkpoint else best_ckpt_path
        if not ckpt_to_load.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để đánh giá: {ckpt_to_load}")
        print(f"Nạp checkpoint từ: {ckpt_to_load}")
        model, _ = PLMTrainer.load_checkpoint(ckpt_to_load, model=model, device=device)

    # 6. Đánh giá toàn diện trên tập Test
    print(f"\n==========================================")
    print(f"ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print(f"==========================================")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    report_str = format_report(test_metrics)
    print("\nBÁO CÁO KẾT QUẢ TEST (Entity-level):")
    print(report_str)

    # 7. Dự đoán chi tiết và xuất kết quả (phục hồi chính xác 1-1 về mức token gốc)
    model.eval()
    all_pred_tags: List[List[str]] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_word_ids = batch["word_ids"]
            batch_tokens = batch["tokens"]

            for b in range(len(preds)):
                sent_preds = recover_word_predictions(
                    pred_ids=preds[b],
                    word_ids=batch_word_ids[b],
                    num_tokens=len(batch_tokens[b]),
                    id2tag=id2tag,
                    fallback_tag="O",
                )
                all_pred_tags.append(sent_preds)

    # Xuất file dự đoán theo cấu hình nhóm
    pred_out_file = Path(cfg.get("logging", {}).get("prediction_out", "outputs/predictions/pretrained_test.jsonl"))
    export_predictions(test_samples, all_pred_tags, pred_out_file)

    # Xuất bảng kết quả và tệp JSON
    table_dir = Path("outputs/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    exp_name = cfg.get("experiment_name", "phobert_ner")
    metrics_json_path = table_dir / f"{exp_name}_test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu chi tiết metrics test vào: {metrics_json_path}")

    # Sinh mã LaTeX cho bảng kết quả
    latex_code = to_latex_table({f"{model_name} ({pooling_strategy})": test_metrics}, caption=f"Kết quả {exp_name} trên tập test.")
    latex_table_path = table_dir / f"{exp_name}_latex_table.tex"
    with open(latex_table_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"Đã lưu bảng LaTeX vào: {latex_table_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá PhoBERT/XLM-R NER")
    parser.add_argument("--config", type=str, default="configs/pretrained_plm.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--dataset", type=str, choices=["vlsp", "phoner_covid19"], default=None)
    parser.add_argument("--granularity", type=str, choices=["word", "syllable"], default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--pooling", type=str, choices=["last", "concat_4", "mean_4", "weighted"], default=None)
    parser.add_argument("--freeze_embeddings", action="store_true", default=None)
    parser.add_argument("--freeze_layers", type=int, default=None)
    parser.add_argument("--freeze_entire_encoder", action="store_true", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Chỉ đánh giá checkpoint mà không huấn luyện")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Chạy thử 1 epoch trên tập mẫu nhỏ để kiểm định pipeline")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_pipeline(cfg, args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
