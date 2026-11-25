import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.common.logger import ExperimentLogger
from src.common.metrics import compute_metrics, format_report
from src.common.schema import NERSample, save_jsonl
from src.models.bilstm_crf.dataset import create_dataloader
from src.models.bilstm_crf.model import BiLSTMCRF
from src.models.bilstm_crf.vocab import Vocabulary



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: BiLSTMCRF,
    dataloader,
    vocab: Vocabulary,
    device: torch.device,
    save_predictions_path: Optional[str] = None,
    dataset_name: str = "phoner_covid19",
    granularity: str = "word",
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []
    pred_samples: List[NERSample] = []

    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)

            loss = model.compute_loss(
                word_ids, tag_ids, mask, lengths, char_ids=char_ids
            )
            total_loss += loss.item()

            pred_id_seqs = model.decode(word_ids, mask, lengths, char_ids=char_ids)
            for i, seq_len in enumerate(lengths.tolist()):
                true_ids = tag_ids[i, :seq_len].tolist()
                pred_tags = vocab.decode_tags(pred_id_seqs[i])
                all_true.append(vocab.decode_tags(true_ids))
                all_pred.append(pred_tags)

                if save_predictions_path is not None:
                    pred_samples.append(
                        NERSample(
                            id=batch["ids"][i],
                            dataset=dataset_name,
                            granularity=granularity,
                            tokens=batch["tokens"][i],
                            tags=pred_tags,
                        )
                    )

    if save_predictions_path is not None:
        out_path = Path(save_predictions_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(pred_samples, str(out_path))

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_true, all_pred)
    return avg_loss, metrics



def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện nhánh BiLSTM-CRF (Thành viên 1)")
    parser.add_argument("--config", type=str, default="configs/bilstm_crf.yaml")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-char", action="store_true", help="Tắt tầng CharEncoder")
    parser.add_argument("--no-crf", action="store_true", help="Tắt tầng CRF (dùng CrossEntropy)")
    parser.add_argument("--no-wandb", action="store_true", help="Tắt log lên Weights & Biases")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_char = cfg["model"].get("use_char_encoder", False) and not args.no_char
    use_crf = cfg["model"].get("use_crf", False) and not args.no_crf
    num_epochs = args.epochs or cfg["train"]["num_epochs"]
    exp_name = args.exp_name or cfg["experiment_name"]

    print(f"=== Thiết bị huấn luyện: {device} | Thí nghiệm: {exp_name} ===")

    # 1. Xây từ điển chỉ trên tập train
    vocab = Vocabulary()
    vocab.build_from_jsonl(
        cfg["data"]["train_path"],
        min_word_freq=cfg["data"].get("min_word_freq", 1),
    )
    print(
        f"Vocab -> Words: {len(vocab.word2idx)} | Chars: {len(vocab.char2idx)} | Tags: {len(vocab.tag2idx)}"
    )

    # 2. Khởi tạo DataLoader
    bs = cfg["train"]["batch_size"]
    max_seq_len = cfg["data"].get("max_seq_len", 256)
    max_word_len = cfg["data"].get("max_word_len", 25)

    train_loader = create_dataloader(
        cfg["data"]["train_path"], vocab, batch_size=bs, shuffle=True,
        max_seq_len=max_seq_len, max_word_len=max_word_len,
    )
    dev_loader = create_dataloader(
        cfg["data"]["dev_path"], vocab, batch_size=bs, shuffle=False,
        max_seq_len=None, max_word_len=max_word_len,
    )
    test_loader = create_dataloader(
        cfg["data"]["test_path"], vocab, batch_size=bs, shuffle=False,
        max_seq_len=None, max_word_len=max_word_len,
    )

    # 3. Khởi tạo CharEncoder và CRF nếu có (Tuần 3 sẽ tự cài đặt 2 module này)
    char_encoder, char_out_dim = None, 0
    if use_char:
        try:
            from src.models.bilstm_crf.char_encoder import build_char_encoder
            char_encoder, char_out_dim = build_char_encoder(cfg["model"], len(vocab.char2idx))
        except ImportError:
            print("[Thông báo Tuần 2] Chưa có char_encoder.py -> Chạy Baseline không có CharEncoder.")

    crf_layer = None
    if use_crf:
        try:
            from src.models.bilstm_crf.crf import CRF
            crf_layer = CRF(num_tags=len(vocab.tag2idx))
        except ImportError:
            print("[Thông báo Tuần 2] Chưa có crf.py -> Chạy Baseline với Masked CrossEntropyLoss.")

    # 4. Khởi tạo mô hình, bộ tối ưu và scheduler
    model = BiLSTMCRF(
        vocab_size=len(vocab.word2idx),
        num_tags=len(vocab.tag2idx),
        word_emb_dim=cfg["model"]["word_emb_dim"],
        bilstm_hidden_dim=cfg["model"]["bilstm_hidden_dim"],
        bilstm_num_layers=cfg["model"]["bilstm_num_layers"],
        dropout=cfg["model"]["dropout"],
        char_encoder=char_encoder,
        char_out_dim=char_out_dim,
        crf=crf_layer,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["train"]["lr_factor"],
        patience=cfg["train"]["lr_patience"],
    )

    # 5. Chuẩn bị thư mục lưu checkpoint và W&B
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"]) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(str(ckpt_dir / "vocab.json"))
    logger = ExperimentLogger(
        run_name=exp_name,
        project=cfg["logging"].get("project_name", "vietnamese-ner-acl"),
        config=cfg,
        member="tv1",
        tags=["bilstm_crf", cfg["data"]["dataset"], cfg["data"]["granularity"]],
        backends="local" if args.no_wandb else "auto",
        verbose=False,
    )

    best_dev_f1 = -1.0
    patience_counter = 0
    history = []
    last_epoch = 0

    for epoch in range(1, num_epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}", leave=False):
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(word_ids, tag_ids, mask, lengths, char_ids=char_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)
        dev_loss, dev_metrics = evaluate(model, dev_loader, vocab, device)
        dev_f1 = dev_metrics["overall"]["f1"]
        scheduler.step(dev_f1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "dev_loss": round(dev_loss, 4),
            "dev_precision": round(dev_metrics["overall"]["precision"], 4),
            "dev_recall": round(dev_metrics["overall"]["recall"], 4),
            "dev_f1": round(dev_f1, 4),
            "dev_macro_f1": round(dev_metrics["macro_f1"], 4),
            "lr": lr_now,
            "time_sec": round(elapsed, 1),
        }
        history.append(log_row)

        logger.log(
            {"train/loss": train_loss, "dev/loss": dev_loss, "lr": lr_now, "time_sec": elapsed},
            step=epoch,
        )
        logger.log_ner_metrics(dev_metrics, prefix="dev", step=epoch)

        improved = ""
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            improved = " <-- BEST"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | dev_loss: {dev_loss:.4f} | "
            f"dev_F1: {dev_f1*100:.2f}% | lr: {lr_now:.1e} | {elapsed:.1f}s{improved}"
        )

        if patience_counter >= cfg["train"]["early_stopping_patience"]:
            print(f"Dừng sớm (Early Stopping) tại epoch {epoch}.")
            break

    # 6. Lưu lịch sử huấn luyện & đánh giá mô hình tốt nhất trên tập test
    hist_path = Path("outputs/tables") / f"history_{exp_name}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    pred_out_path = (
        f"outputs/predictions/{exp_name}_test.jsonl"
        if args.exp_name is not None
        else cfg["logging"].get("prediction_out", f"outputs/predictions/{exp_name}_test.jsonl")
    )

    model.load_state_dict(
        torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        vocab,
        device,
        save_predictions_path=pred_out_path,
        dataset_name=cfg["data"]["dataset"],
        granularity=cfg["data"]["granularity"],
    )

    test_metrics_path = Path("outputs/tables") / f"{exp_name}_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": exp_name,
                "dataset": cfg["data"]["dataset"],
                "granularity": cfg["data"]["granularity"],
                "best_dev_f1": round(best_dev_f1, 4),
                "test_loss": round(test_loss, 4),
                **test_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n=== KẾT QUẢ TẬP TEST ({exp_name}) ===")
    print(f"Đã lưu metrics tại: {test_metrics_path}")
    print(f"Đã lưu dự đoán tại: {pred_out_path}")
    print(
        f"Test Loss: {test_loss:.4f} | Precision: {test_metrics['overall']['precision']*100:.2f}% | "
        f"Recall: {test_metrics['overall']['recall']*100:.2f}% | "
        f"Micro-F1: {test_metrics['overall']['f1']*100:.2f}% | "
        f"Macro-F1: {test_metrics['macro_f1']*100:.2f}%"
    )
    print(format_report(test_metrics))

    logger.log_ner_metrics(test_metrics, prefix="test", step=last_epoch + 1)
    logger.log_summary({
        "best_dev_f1": best_dev_f1,
        "test_loss": test_loss,
        "test_precision": test_metrics["overall"]["precision"],
        "test_recall": test_metrics["overall"]["recall"],
        "test_micro_f1": test_metrics["overall"]["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    })
    logger.finish()


if __name__ == "__main__":
    main()
