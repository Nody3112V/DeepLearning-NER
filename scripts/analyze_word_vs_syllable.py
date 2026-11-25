"""
analyze_word_vs_syllable.py - Phân tích, kiểm định chéo và trực quan hóa thực nghiệm
so sánh mức từ (word-level) và mức âm tiết (syllable-level) cho nhánh Pretrained PLM
(PhoBERT & XLM-R) và đối chiếu liên mô hình với BiLSTM-CRF (Thành viên 1).

Đầu ra:
  - outputs/tables/granularity_comparison_summary.json
  - outputs/tables/granularity_comparison_report.md
  - outputs/tables/granularity_comparison_vlsp.tex
  - outputs/tables/granularity_comparison_phoner.tex
  - outputs/tables/cross_model_phoner_comparison.tex
  - outputs/figures/granularity_f1_comparison.png
  - outputs/figures/granularity_runtime_memory.png
  - outputs/figures/granularity_sequence_length.png
  - outputs/figures/cross_model_per_entity_phoner.png
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from src.common.metrics import compute_metrics, evaluate_files, extract_entities
from src.common.schema import NERSample, load_jsonl, parse_sample_id
from src.data_prep.process_vlsp import word_to_syllables

ROOT = Path(".")
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Định nghĩa danh mục thực nghiệm chính thức
# --------------------------------------------------------------------------- #
PLM_EXPERIMENTS = [
    # VLSP 2016 (2x2 Controlled Matrix)
    {
        "exp_id": "exp01_phobert_baseline",
        "label": "PhoBERT (Word)",
        "model_short": "PhoBERT-base-v2",
        "model_name": "vinai/phobert-base-v2",
        "dataset": "vlsp",
        "granularity": "word",
        "tokenizer_type": "BPE (Slow, 64k)",
        "params": 135_005_193,
        "epochs": 4,
        "gold_path": "data/processed/vlsp/word/test.jsonl",
        "train_path": "data/processed/vlsp/word/train.jsonl",
        "pred_path": "outputs/predictions/exp01_phobert_baseline_test.jsonl",
        "log_dir": "outputs/logs/exp01_phobert_baseline",
    },
    {
        "exp_id": "exp13_phobert_vlsp_syllable",
        "label": "PhoBERT (Syllable)",
        "model_short": "PhoBERT-base-v2",
        "model_name": "vinai/phobert-base-v2",
        "dataset": "vlsp",
        "granularity": "syllable",
        "tokenizer_type": "BPE (Slow, 64k)",
        "params": 135_005_193,
        "epochs": 4,
        "gold_path": "data/processed/vlsp/syllable/test.jsonl",
        "train_path": "data/processed/vlsp/syllable/train.jsonl",
        "pred_path": "outputs/predictions/exp13_phobert_vlsp_syllable_test.jsonl",
        "log_dir": "outputs/logs/exp13_phobert_vlsp_syllable",
    },
    {
        "exp_id": "exp02_xlmr_baseline",
        "label": "XLM-R (Syllable)",
        "model_short": "XLM-RoBERTa-base",
        "model_name": "xlm-roberta-base",
        "dataset": "vlsp",
        "granularity": "syllable",
        "tokenizer_type": "SentencePiece (Fast, 250k)",
        "params": 278_050_569,
        "epochs": 4,
        "gold_path": "data/processed/vlsp/syllable/test.jsonl",
        "train_path": "data/processed/vlsp/syllable/train.jsonl",
        "pred_path": "outputs/predictions/exp02_xlmr_baseline_test.jsonl",
        "log_dir": "outputs/logs/exp02_xlmr_baseline",
    },
    {
        "exp_id": "exp14_xlmr_vlsp_word",
        "label": "XLM-R (Word)",
        "model_short": "XLM-RoBERTa-base",
        "model_name": "xlm-roberta-base",
        "dataset": "vlsp",
        "granularity": "word",
        "tokenizer_type": "SentencePiece (Fast, 250k)",
        "params": 278_050_569,
        "epochs": 4,
        "gold_path": "data/processed/vlsp/word/test.jsonl",
        "train_path": "data/processed/vlsp/word/train.jsonl",
        "pred_path": "outputs/predictions/exp14_xlmr_vlsp_word_test.jsonl",
        "log_dir": "outputs/logs/exp14_xlmr_vlsp_word",
    },
    # PhoNER_COVID19 (2x2 Controlled Matrix)
    {
        "exp_id": "exp15_phobert_phoner_word",
        "label": "PhoBERT (Word)",
        "model_short": "PhoBERT-base-v2",
        "model_name": "vinai/phobert-base-v2",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "tokenizer_type": "BPE (Slow, 64k)",
        "params": 135_014_421,
        "epochs": 6,
        "gold_path": "data/processed/phoner_covid19/word/test.jsonl",
        "train_path": "data/processed/phoner_covid19/word/train.jsonl",
        "pred_path": "outputs/predictions/exp15_phobert_phoner_word_test.jsonl",
        "log_dir": "outputs/logs/exp15_phobert_phoner_word",
    },
    {
        "exp_id": "exp16_phobert_phoner_syllable",
        "label": "PhoBERT (Syllable)",
        "model_short": "PhoBERT-base-v2",
        "model_name": "vinai/phobert-base-v2",
        "dataset": "phoner_covid19",
        "granularity": "syllable",
        "tokenizer_type": "BPE (Slow, 64k)",
        "params": 135_014_421,
        "epochs": 6,
        "gold_path": "data/processed/phoner_covid19/syllable/test.jsonl",
        "train_path": "data/processed/phoner_covid19/syllable/train.jsonl",
        "pred_path": "outputs/predictions/exp16_phobert_phoner_syllable_test.jsonl",
        "log_dir": "outputs/logs/exp16_phobert_phoner_syllable",
    },
    {
        "exp_id": "exp17_xlmr_phoner_syllable",
        "label": "XLM-R (Syllable)",
        "model_short": "XLM-RoBERTa-base",
        "model_name": "xlm-roberta-base",
        "dataset": "phoner_covid19",
        "granularity": "syllable",
        "tokenizer_type": "SentencePiece (Fast, 250k)",
        "params": 278_059_797,
        "epochs": 6,
        "gold_path": "data/processed/phoner_covid19/syllable/test.jsonl",
        "train_path": "data/processed/phoner_covid19/syllable/train.jsonl",
        "pred_path": "outputs/predictions/exp17_xlmr_phoner_syllable_test.jsonl",
        "log_dir": "outputs/logs/exp17_xlmr_phoner_syllable",
    },
    {
        "exp_id": "exp18_xlmr_phoner_word",
        "label": "XLM-R (Word)",
        "model_short": "XLM-RoBERTa-base",
        "model_name": "xlm-roberta-base",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "tokenizer_type": "SentencePiece (Fast, 250k)",
        "params": 278_059_797,
        "epochs": 6,
        "gold_path": "data/processed/phoner_covid19/word/test.jsonl",
        "train_path": "data/processed/phoner_covid19/word/train.jsonl",
        "pred_path": "outputs/predictions/exp18_xlmr_phoner_word_test.jsonl",
        "log_dir": "outputs/logs/exp18_xlmr_phoner_word",
    },
]

TV1_EXPERIMENTS = [
    {
        "exp_id": "baseline_bilstm_word",
        "label": "BiLSTM Baseline (TV1)",
        "model_short": "BiLSTM (TV1)",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "gold_path": "data/processed/phoner_covid19/word/test.jsonl",
        "pred_path": "outputs/predictions/baseline_bilstm_word_test.jsonl",
        "log_dir": "outputs/logs/baseline_bilstm_word",
    },
    {
        "exp_id": "bilstm_cnn_crf_phoner_word",
        "label": "BiLSTM-CNN-CRF (TV1)",
        "model_short": "BiLSTM-CNN-CRF (TV1)",
        "dataset": "phoner_covid19",
        "granularity": "word",
        "gold_path": "data/processed/phoner_covid19/word/test.jsonl",
        "pred_path": "outputs/predictions/bilstm_crf_test.jsonl",
        "log_dir": "outputs/logs/bilstm_cnn_crf_phoner_word",
    },
]


# --------------------------------------------------------------------------- #
# 2. Tiện ích đo lường log, độ dài chuỗi, và đơn vị đánh giá thống nhất
# --------------------------------------------------------------------------- #
def _clean(s: str) -> str:
    return re.sub(r"[\s_]+", "", s)


def extract_char_entities(tokens: List[str], tags: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Quy đổi thực thể BIO sang tọa độ ký tự chuẩn hóa (bỏ khoảng trắng và '_').
    Cho phép so sánh trực tiếp word-level và syllable-level trên CÙNG MỘT ĐƠN VỊ ĐÁNH GIÁ
    độc lập với cách tách từ.
    """
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        c = _clean(tok)
        offsets.append((pos, pos + len(c)))
        pos += len(c)

    char_spans: Set[Tuple[int, int, str]] = set()
    for start, end, etype in extract_entities(tags):
        c_start = offsets[start][0]
        c_end = offsets[end - 1][1]
        char_spans.add((c_start, c_end, etype))
    return char_spans


def compute_char_level_entity_f1(
    gold_samples: List[NERSample],
    pred_samples: List[NERSample],
) -> Dict[str, Any]:
    """Tính Entity-level P/R/F1 dựa trên tọa độ ký tự chuẩn hóa (độc lập word/syllable)."""
    assert len(gold_samples) == len(pred_samples)
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for g, p in zip(gold_samples, pred_samples):
        g_spans = extract_char_entities(g.tokens, g.tags)
        p_spans = extract_char_entities(p.tokens, p.tags)
        for s in g_spans & p_spans:
            counts[s[2]]["tp"] += 1
        for s in p_spans - g_spans:
            counts[s[2]]["fp"] += 1
        for s in g_spans - p_spans:
            counts[s[2]]["fn"] += 1

    def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}

    per_type = {t: _prf(**c) for t, c in sorted(counts.items())}
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    macro_f1 = sum(m["f1"] for m in per_type.values()) / len(per_type) if per_type else 0.0
    return {"overall": _prf(tp, fp, fn), "per_type": per_type, "macro_f1": macro_f1}


def paired_bootstrap_test(
    gold_samples_a: List[NERSample],
    pred_samples_a: List[NERSample],
    gold_samples_b: List[NERSample],
    pred_samples_b: List[NERSample],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Kiểm định ý nghĩa thống kê Paired Bootstrap (Berg-Kirkpatrick et al., 2012)
    giữa mô hình A và mô hình B trên cùng N câu kiểm thử (khớp 1-1 theo vị trí câu).
    """
    n = len(gold_samples_a)
    assert len(pred_samples_a) == n and len(gold_samples_b) == n and len(pred_samples_b) == n

    # Pre-compute per-sentence (tp, fp, fn) cho A và B
    stats_a = np.zeros((n, 3), dtype=np.int64)
    stats_b = np.zeros((n, 3), dtype=np.int64)
    a_wins, b_wins, ties = 0, 0, 0

    for i in range(n):
        ga, pa = extract_entities(gold_samples_a[i].tags), extract_entities(pred_samples_a[i].tags)
        gb, pb = extract_entities(gold_samples_b[i].tags), extract_entities(pred_samples_b[i].tags)
        tp_a, fp_a, fn_a = len(ga & pa), len(pa - ga), len(ga - pa)
        tp_b, fp_b, fn_b = len(gb & pb), len(pb - gb), len(gb - pb)
        stats_a[i] = (tp_a, fp_a, fn_a)
        stats_b[i] = (tp_b, fp_b, fn_b)

        err_a = fp_a + fn_a
        err_b = fp_b + fn_b
        if err_a < err_b:
            a_wins += 1
        elif err_b < err_a:
            b_wins += 1
        else:
            ties += 1

    def _f1_from_sums(s: np.ndarray) -> float:
        tp, fp, fn = s[0], s[1], s[2]
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    obs_f1_a = _f1_from_sums(stats_a.sum(axis=0))
    obs_f1_b = _f1_from_sums(stats_b.sum(axis=0))
    obs_diff = obs_f1_a - obs_f1_b

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        fa = _f1_from_sums(stats_a[idx].sum(axis=0))
        fb = _f1_from_sums(stats_b[idx].sum(axis=0))
        diffs[r] = fa - fb

    # Two-sided p-value qua bootstrap centering
    if obs_diff >= 0:
        p_one_sided = float(np.mean(diffs <= 0.0))
    else:
        p_one_sided = float(np.mean(diffs >= 0.0))
    p_two_sided = min(1.0, 2.0 * p_one_sided)

    return {
        "n_sentences": n,
        "f1_a": obs_f1_a,
        "f1_b": obs_f1_b,
        "delta_f1": obs_diff,
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_value_two_sided": p_two_sided,
        "significant_005": bool(p_two_sided < 0.05),
        "sentence_a_better": a_wins,
        "sentence_b_better": b_wins,
        "sentence_ties": ties,
    }


def read_training_resources(log_dir: Path, is_tv1: bool = False) -> Dict[str, Any]:
    """Đọc runtime và peak VRAM thực tế từ log huấn luyện."""
    summary_path = log_dir / "summary.json"
    metrics_path = log_dir / "metrics.jsonl"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    max_mem_mb = summary.get("max_gpu_memory_mb", None)
    runtime_s = summary.get("total_training_time_s", None)
    best_dev_f1 = summary.get("best_dev_f1_micro", summary.get("best_dev_f1", None))
    best_epoch = summary.get("best_epoch", None)

    epoch_times = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            m = rec.get("metrics", {})
            if "gpu/max_memory_mb" in m:
                val = float(m["gpu/max_memory_mb"])
                max_mem_mb = val if max_mem_mb is None else max(max_mem_mb, val)
            if "train/epoch_duration_s" in m:
                epoch_times.append(float(m["train/epoch_duration_s"]))
            elif "time_sec" in m:
                epoch_times.append(float(m["time_sec"]))

    if runtime_s is None and epoch_times:
        runtime_s = sum(epoch_times)

    return {
        "runtime_s": float(runtime_s) if runtime_s is not None else None,
        "max_gpu_memory_mb": float(max_mem_mb) if max_mem_mb is not None else None,
        "best_dev_f1": float(best_dev_f1) if best_dev_f1 is not None else None,
        "best_epoch": best_epoch,
        "num_logged_epochs": len(epoch_times),
    }


def compute_sequence_stats(
    samples: List[NERSample],
    model_name: str,
    tok_phobert: Any,
    tok_xlmr: Any,
    phobert_cache: Dict[str, int],
    max_seq_len: int = 256,
) -> Dict[str, Any]:
    """Đo chính xác số token gốc, độ dài subword đầu vào, tỷ lệ sw/tok và số câu bị cắt."""
    n = len(samples)
    raw_lens = [len(s.tokens) for s in samples]

    if "phobert" in model_name.lower():
        sw_lens, trunc_sents, trunc_toks = [], 0, 0
        for s in samples:
            cur = 1  # <s>
            covered = 0
            for t in s.tokens:
                if t not in phobert_cache:
                    sw = tok_phobert.encode(t, add_special_tokens=False)
                    phobert_cache[t] = max(1, len(sw))
                l = phobert_cache[t]
                if cur + l > max_seq_len - 1:
                    if (max_seq_len - 1) - cur > 0:
                        cur = max_seq_len - 1
                        covered += 1
                    break
                cur += l
                covered += 1
            cur += 1  # </s>
            sw_lens.append(cur)
            if covered < len(s.tokens):
                trunc_sents += 1
                trunc_toks += len(s.tokens) - covered
    else:
        enc = tok_xlmr(
            [s.tokens for s in samples],
            is_split_into_words=True,
            max_length=max_seq_len,
            truncation=True,
            return_attention_mask=False,
        )
        sw_lens, trunc_sents, trunc_toks = [], 0, 0
        for i, s in enumerate(samples):
            ids = enc["input_ids"][i]
            sw_lens.append(len(ids))
            wids = enc.word_ids(batch_index=i)
            covered = len({w for w in wids if w is not None})
            if covered < len(s.tokens):
                # Phân biệt truncation thật sự (len == max_seq_len) với ký tự zero-width
                if len(ids) >= max_seq_len:
                    trunc_sents += 1
                    trunc_toks += len(s.tokens) - covered

    sw_no_spec = sum(sw_lens) - 2 * n
    return {
        "num_sentences": n,
        "avg_tokens": round(sum(raw_lens) / n, 2),
        "max_tokens": int(max(raw_lens)),
        "avg_subwords": round(sum(sw_lens) / n, 2),
        "p95_subwords": round(float(np.percentile(sw_lens, 95)), 1),
        "max_subwords": int(max(sw_lens)),
        "subword_fertility": round(sw_no_spec / sum(raw_lens), 3),
        "truncated_sentences": int(trunc_sents),
        "truncated_tokens": int(trunc_toks),
    }


# --------------------------------------------------------------------------- #
# 3. Trực quan hóa (BƯỚC 6)
# --------------------------------------------------------------------------- #
def generate_charts(plm_records: List[Dict[str, Any]], tv1_records: List[Dict[str, Any]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 300,
    })

    by_id = {r["exp_id"]: r for r in plm_records}
    tv1_by_id = {r["exp_id"]: r for r in tv1_records}

    # --- Chart 1: F1 Comparison (Micro & Macro) across VLSP & PhoNER (+ TV1) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=False)

    # Panel A: VLSP 2016 (2x2 Matrix)
    vlsp_order = [
        ("exp01_phobert_baseline", "PhoBERT\n(Word)"),
        ("exp13_phobert_vlsp_syllable", "PhoBERT\n(Syllable)"),
        ("exp02_xlmr_baseline", "XLM-R\n(Syllable)"),
        ("exp14_xlmr_vlsp_word", "XLM-R\n(Word)"),
    ]
    x = np.arange(len(vlsp_order))
    w = 0.35
    v_micro = [by_id[eid]["metrics"]["overall"]["f1"] * 100 for eid, _ in vlsp_order]
    v_macro = [by_id[eid]["metrics"]["macro_f1"] * 100 for eid, _ in vlsp_order]

    b1 = axes[0].bar(x - w / 2, v_micro, w, label="Micro-F1", color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = axes[0].bar(x + w / 2, v_macro, w, label="Macro-F1", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    axes[0].set_title("VLSP 2016 (1,872 test sents | 4 entity types)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in vlsp_order])
    axes[0].set_ylabel("Entity-level F1 (%)")
    axes[0].set_ylim(82, 98)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(loc="lower right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: PhoNER_COVID19 (2x2 PLM + TV1 BiLSTM & BiLSTM-CNN-CRF)
    phoner_order = [
        ("tv1", "baseline_bilstm_word", "BiLSTM\n(Word, TV1)"),
        ("tv1", "bilstm_cnn_crf_phoner_word", "BiLSTM-CRF\n(Word, TV1)"),
        ("plm", "exp15_phobert_phoner_word", "PhoBERT\n(Word)"),
        ("plm", "exp16_phobert_phoner_syllable", "PhoBERT\n(Syllable)"),
        ("plm", "exp17_xlmr_phoner_syllable", "XLM-R\n(Syllable)"),
        ("plm", "exp18_xlmr_phoner_word", "XLM-R\n(Word)"),
    ]
    x2 = np.arange(len(phoner_order))
    p_micro, p_macro = [], []
    for src_kind, eid, _ in phoner_order:
        rec = tv1_by_id[eid] if src_kind == "tv1" else by_id[eid]
        p_micro.append(rec["metrics"]["overall"]["f1"] * 100)
        p_macro.append(rec["metrics"]["macro_f1"] * 100)

    b3 = axes[1].bar(x2 - w / 2, p_micro, w, label="Micro-F1", color="#2ca02c", edgecolor="black", linewidth=0.6)
    b4 = axes[1].bar(x2 + w / 2, p_macro, w, label="Macro-F1", color="#d62728", edgecolor="black", linewidth=0.6)
    axes[1].axvline(1.5, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_title("PhoNER_COVID19 (3,000 test sents | 10 entity types)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([lbl for _, _, lbl in phoner_order], fontsize=8.8)
    axes[1].set_ylabel("Entity-level F1 (%)")
    axes[1].set_ylim(80, 98)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(loc="lower right")
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    fig.suptitle("So sánh Entity-level F1 giữa Mức Từ (Word) và Mức Âm Tiết (Syllable)", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "granularity_f1_comparison.png", dpi=300)
    plt.close(fig)

    # --- Chart 2: Training Runtime & Peak GPU VRAM ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    all_plm_plot = [
        ("exp01_phobert_baseline", "VLSP\nPhoBERT-W"),
        ("exp13_phobert_vlsp_syllable", "VLSP\nPhoBERT-S"),
        ("exp02_xlmr_baseline", "VLSP\nXLM-R-S"),
        ("exp14_xlmr_vlsp_word", "VLSP\nXLM-R-W"),
        ("exp15_phobert_phoner_word", "PhoNER\nPhoBERT-W"),
        ("exp16_phobert_phoner_syllable", "PhoNER\nPhoBERT-S"),
        ("exp17_xlmr_phoner_syllable", "PhoNER\nXLM-R-S"),
        ("exp18_xlmr_phoner_word", "PhoNER\nXLM-R-W"),
    ]
    xp = np.arange(len(all_plm_plot))
    colors = ["#1f77b4", "#6baed6", "#ff7f0e", "#fdae6b"] * 2
    runtimes = [by_id[eid]["resources"]["runtime_s"] / 60.0 for eid, _ in all_plm_plot]  # minutes
    vrams = [by_id[eid]["resources"]["max_gpu_memory_mb"] / 1024.0 for eid, _ in all_plm_plot]  # GB

    bars_t = axes[0].bar(xp, runtimes, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].axvline(3.5, color="gray", linestyle="--", linewidth=1.1)
    axes[0].set_title("Tổng thời gian huấn luyện (phút)")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels([lbl for _, lbl in all_plm_plot], fontsize=8.5)
    axes[0].set_ylabel("Thời gian huấn luyện (phút)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    for bar in bars_t:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}m", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    bars_m = axes[1].bar(xp, vrams, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].axvline(3.5, color="gray", linestyle="--", linewidth=1.1)
    axes[1].axhline(8.0, color="#d62728", linestyle=":", linewidth=1.3, label="Giới hạn VRAM vật lý RTX 4060 (8.0 GB)")
    axes[1].set_title("Bộ nhớ GPU đỉnh (Peak VRAM Allocated - GB)")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels([lbl for _, lbl in all_plm_plot], fontsize=8.5)
    axes[1].set_ylabel("Peak GPU Memory (GB)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(loc="upper left", fontsize=8.5)
    for bar in bars_m:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    fig.suptitle("Chi phí Thời gian Huấn luyện & Bộ nhớ GPU theo Mô hình và Mức Biểu diễn", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "granularity_runtime_memory.png", dpi=300)
    plt.close(fig)

    # --- Chart 3: Input Sequence Length & Subword Fertility ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    raw_toks = [by_id[eid]["test_seq_stats"]["avg_tokens"] for eid, _ in all_plm_plot]
    avg_sws = [by_id[eid]["test_seq_stats"]["avg_subwords"] for eid, _ in all_plm_plot]
    ferts = [by_id[eid]["test_seq_stats"]["subword_fertility"] for eid, _ in all_plm_plot]

    w3 = 0.36
    b_raw = axes[0].bar(xp - w3 / 2, raw_toks, w3, label="Token đầu vào TB", color="#9467bd", edgecolor="black", linewidth=0.6)
    b_sw = axes[0].bar(xp + w3 / 2, avg_sws, w3, label="Độ dài chuỗi Subword TB", color="#17becf", edgecolor="black", linewidth=0.6)
    axes[0].axvline(3.5, color="gray", linestyle="--", linewidth=1.1)
    axes[0].set_title("Độ dài Token gốc vs Chuỗi Subword (tập Test)")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels([lbl for _, lbl in all_plm_plot], fontsize=8.5)
    axes[0].set_ylabel("Độ dài trung bình / câu")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(loc="upper left")
    for bars in (b_raw, b_sw):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8)

    bars_f = axes[1].bar(xp, ferts, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].axvline(3.5, color="gray", linestyle="--", linewidth=1.1)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="Tỷ lệ lý tưởng 1.0 subword/token")
    axes[1].set_title("Tỷ lệ phân mảnh Subword (Subword Fertility = subwords / token)")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels([lbl for _, lbl in all_plm_plot], fontsize=8.5)
    axes[1].set_ylabel("Subwords / Token")
    axes[1].set_ylim(0.8, 2.3)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(loc="upper left")
    for bar in bars_f:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}x", ha="center", va="bottom", fontsize=8.3, fontweight="bold")

    fig.suptitle("Ảnh hưởng của Mức Biểu diễn & Tokenizer lên Độ dài Chuỗi đầu vào", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "granularity_sequence_length.png", dpi=300)
    plt.close(fig)

    # --- Chart 4: Per-Entity-Type Cross-Model Comparison on PhoNER_COVID19 ---
    phoner_types = [
        "DATE", "PATIENT_ID", "GENDER", "AGE", "LOCATION",
        "TRANSPORTATION", "NAME", "SYMPTOM_AND_DISEASE", "ORGANIZATION", "JOB",
    ]
    models_plot = [
        ("BiLSTM-CNN-CRF (Word, TV1)", tv1_by_id["bilstm_cnn_crf_phoner_word"]["metrics"]["per_type"], "#7f7f7f"),
        ("PhoBERT (Word)", by_id["exp15_phobert_phoner_word"]["metrics"]["per_type"], "#1f77b4"),
        ("PhoBERT (Syllable)", by_id["exp16_phobert_phoner_syllable"]["metrics"]["per_type"], "#6baed6"),
        ("XLM-R (Syllable)", by_id["exp17_xlmr_phoner_syllable"]["metrics"]["per_type"], "#ff7f0e"),
        ("XLM-R (Word)", by_id["exp18_xlmr_phoner_word"]["metrics"]["per_type"], "#fdae6b"),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 5.8))
    xt = np.arange(len(phoner_types))
    bw = 0.16
    for idx, (m_lbl, pt_dict, col) in enumerate(models_plot):
        vals = [pt_dict[t]["f1"] * 100 for t in phoner_types]
        offset = (idx - (len(models_plot) - 1) / 2) * bw
        ax.bar(xt + offset, vals, bw, label=m_lbl, color=col, edgecolor="black", linewidth=0.45)

    ax.set_title("Đối chiếu Entity-level F1 (%) theo từng loại thực thể trên PhoNER_COVID19 (3,000 câu test chung)", pad=32)
    ax.set_xticks(xt)
    ax.set_xticklabels([t.replace("_", "\n") for t in phoner_types], fontsize=8.5)
    ax.set_ylabel("Entity-level F1 (%)")
    ax.set_ylim(45, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=True, fontsize=8.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_model_per_entity_phoner.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Tìm ví dụ phân tích lỗi định tính (Divergence Cases)
# --------------------------------------------------------------------------- #
def find_divergence_examples(
    gold_path: str,
    pred_a_path: str,
    pred_b_path: str,
    name_a: str,
    name_b: str,
    max_examples: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Tìm các câu điển hình nơi mô hình A đúng hoàn toàn còn B sai (và ngược lại)."""
    golds = list(load_jsonl(gold_path))
    preds_a = list(load_jsonl(pred_a_path))
    preds_b = list(load_jsonl(pred_b_path))

    a_better, b_better = [], []
    for g, pa, pb in zip(golds, preds_a, preds_b):
        gs = extract_entities(g.tags)
        if not gs:
            continue
        sa, sb = extract_entities(pa.tags), extract_entities(pb.tags)
        if sa == gs and sb != gs and len(a_better) < max_examples and len(g.tokens) <= 35:
            a_better.append({
                "id": g.id,
                "text": " ".join(g.tokens),
                "gold": [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(gs)],
                name_a: [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(sa)],
                name_b: [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(sb)],
            })
        elif sb == gs and sa != gs and len(b_better) < max_examples and len(g.tokens) <= 35:
            b_better.append({
                "id": g.id,
                "text": " ".join(g.tokens),
                "gold": [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(gs)],
                name_a: [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(sa)],
                name_b: [(" ".join(g.tokens[s:e]), t) for s, e, t in sorted(sb)],
            })
    return {f"{name_a}_better": a_better, f"{name_b}_better": b_better}


# --------------------------------------------------------------------------- #
# 5. Luồng thực thi chính
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=================================================================")
    print("  PHÂN TÍCH & KIỂM ĐỊNH CHÉO: WORD-LEVEL vs SYLLABLE-LEVEL (PLM)")
    print("=================================================================\n")

    # Kiểm tra toàn bộ file dự đoán tồn tại trước khi chạy
    missing_files = [e["pred_path"] for e in PLM_EXPERIMENTS + TV1_EXPERIMENTS if not Path(e["pred_path"]).exists()]
    if missing_files:
        raise FileNotFoundError(f"Thiếu file dự đoán: {missing_files}")

    # Nạp tokenizer để đo chính xác độ dài chuỗi subword
    tok_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False, local_files_only=True)
    tok_xlmr = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True, local_files_only=True)
    phobert_cache: Dict[str, int] = {}

    # Đọc các tập gold chính thức (dùng chung)
    golds_cache: Dict[Tuple[str, str], List[NERSample]] = {}
    for ds in ("vlsp", "phoner_covid19"):
        for gran in ("word", "syllable"):
            golds_cache[(ds, gran)] = list(load_jsonl(f"data/processed/{ds}/{gran}/test.jsonl"))

    # --- A. Đánh giá độc lập 8 thí nghiệm PLM ---
    plm_records: List[Dict[str, Any]] = []
    for cfg in PLM_EXPERIMENTS:
        metrics = evaluate_files(cfg["gold_path"], cfg["pred_path"])
        resources = read_training_resources(Path(cfg["log_dir"]))
        train_samples = list(load_jsonl(cfg["train_path"]))
        test_samples = golds_cache[(cfg["dataset"], cfg["granularity"])]
        pred_samples = list(load_jsonl(cfg["pred_path"]))

        train_seq = compute_sequence_stats(train_samples, cfg["model_name"], tok_phobert, tok_xlmr, phobert_cache)
        test_seq = compute_sequence_stats(test_samples, cfg["model_name"], tok_phobert, tok_xlmr, phobert_cache)

        # Kiểm định đơn vị đánh giá thống nhất bằng character-span entity F1 (so với syllable gold chuẩn)
        canon_gold = golds_cache[(cfg["dataset"], "syllable")]
        char_unified = compute_char_level_entity_f1(canon_gold, pred_samples)

        # Đối với VLSP word-level: kiểm định quy đổi word_to_syllables -> syllable gold
        vlsp_proj_f1 = None
        if cfg["dataset"] == "vlsp" and cfg["granularity"] == "word":
            proj_pred_tags = [word_to_syllables(p.tokens, p.tags)[1] for p in pred_samples]
            syl_gold_tags = [g.tags for g in golds_cache[("vlsp", "syllable")]]
            vlsp_proj_res = compute_metrics(syl_gold_tags, proj_pred_tags)
            vlsp_proj_f1 = vlsp_proj_res["overall"]["f1"]

        rec = {
            **cfg,
            "metrics": metrics,
            "resources": resources,
            "train_seq_stats": train_seq,
            "test_seq_stats": test_seq,
            "char_unified_metrics": char_unified["overall"],
            "char_unified_macro_f1": char_unified["macro_f1"],
            "vlsp_word_to_syllable_projected_f1": vlsp_proj_f1,
        }
        plm_records.append(rec)
        o = metrics["overall"]
        print(
            f"[PLM] {cfg['exp_id']:30s} | {cfg['dataset']:14s} | {cfg['granularity']:8s} | "
            f"P={o['precision']*100:5.2f}% R={o['recall']*100:5.2f}% "
            f"Micro-F1={o['f1']*100:5.2f}% Macro-F1={metrics['macro_f1']*100:5.2f}% | "
            f"Char-F1={char_unified['overall']['f1']*100:5.2f}% | "
            f"Time={resources['runtime_s']:6.1f}s | VRAM={resources['max_gpu_memory_mb']:7.1f}MB | "
            f"AvgTok={test_seq['avg_tokens']:5.2f} AvgSW={test_seq['avg_subwords']:5.2f} ({test_seq['subword_fertility']:.2f}x)"
        )
        if vlsp_proj_f1 is not None:
            diff_proj = abs(vlsp_proj_f1 - o["f1"])
            print(f"      -> Kiểm định quy đổi word->syllable VLSP: F1={vlsp_proj_f1*100:.4f}% (chênh lệch = {diff_proj:.6f})")

    # --- B. Kiểm định chéo với Thành viên 1 (BiLSTM & BiLSTM-CNN-CRF) ---
    print("\n--- KIỂM ĐỊNH CHÉO VỚI THÀNH VIÊN 1 (PhoNER_COVID19 - test_indices.json) ---")
    test_indices_data = json.loads(Path("data/splits/test_indices.json").read_text(encoding="utf-8"))
    expected_indices = set(test_indices_data["indices"])
    assert len(expected_indices) == 3000

    tv1_records: List[Dict[str, Any]] = []
    for cfg in TV1_EXPERIMENTS:
        metrics = evaluate_files(cfg["gold_path"], cfg["pred_path"])
        resources = read_training_resources(Path(cfg["log_dir"]), is_tv1=True)
        pred_samples = list(load_jsonl(cfg["pred_path"]))
        pred_indices = {int(parse_sample_id(s.id)["index"]) for s in pred_samples}
        assert pred_indices == expected_indices, f"Lệch test_indices.json ở {cfg['exp_id']}!"
        char_unified = compute_char_level_entity_f1(golds_cache[("phoner_covid19", "syllable")], pred_samples)

        rec = {
            **cfg,
            "metrics": metrics,
            "resources": resources,
            "test_indices_verified": True,
            "char_unified_metrics": char_unified["overall"],
            "char_unified_macro_f1": char_unified["macro_f1"],
        }
        tv1_records.append(rec)
        o = metrics["overall"]
        print(
            f"[TV1] {cfg['exp_id']:30s} | phoner_covid19 | word     | "
            f"P={o['precision']*100:5.2f}% R={o['recall']*100:5.2f}% "
            f"Micro-F1={o['f1']*100:5.2f}% Macro-F1={metrics['macro_f1']*100:5.2f}% | "
            f"Char-F1={char_unified['overall']['f1']*100:5.2f}% | "
            f"Time={resources['runtime_s']:6.1f}s | test_indices=OK (3000/3000)"
        )

    # Xác minh 4 mô hình PLM trên PhoNER cũng khớp 100% test_indices.json
    for cfg in PLM_EXPERIMENTS:
        if cfg["dataset"] == "phoner_covid19":
            preds = list(load_jsonl(cfg["pred_path"]))
            p_idx = {int(parse_sample_id(s.id)["index"]) for s in preds}
            assert p_idx == expected_indices, f"Lệch test_indices.json ở {cfg['exp_id']}!"
    print("[*] Xác nhận 100% mô hình (TV1 + TV2) trên PhoNER_COVID19 dùng chung đúng 3,000 test IDs trong data/splits/test_indices.json.")

    # --- C. Kiểm định ý nghĩa thống kê Paired Bootstrap ---
    by_id = {r["exp_id"]: r for r in plm_records}
    tv1_by_id = {r["exp_id"]: r for r in tv1_records}

    def _load(p: str) -> List[NERSample]:
        return list(load_jsonl(p))

    bootstrap_pairs = [
        ("vlsp_phobert_word_vs_syllable", by_id["exp01_phobert_baseline"], by_id["exp13_phobert_vlsp_syllable"]),
        ("vlsp_xlmr_syllable_vs_word", by_id["exp02_xlmr_baseline"], by_id["exp14_xlmr_vlsp_word"]),
        ("vlsp_phobert_word_vs_xlmr_syllable", by_id["exp01_phobert_baseline"], by_id["exp02_xlmr_baseline"]),
        ("phoner_phobert_word_vs_syllable", by_id["exp15_phobert_phoner_word"], by_id["exp16_phobert_phoner_syllable"]),
        ("phoner_xlmr_syllable_vs_word", by_id["exp17_xlmr_phoner_syllable"], by_id["exp18_xlmr_phoner_word"]),
        ("phoner_phobert_word_vs_xlmr_syllable", by_id["exp15_phobert_phoner_word"], by_id["exp17_xlmr_phoner_syllable"]),
        ("phoner_phobert_word_vs_bilstm_crf", by_id["exp15_phobert_phoner_word"], tv1_by_id["bilstm_cnn_crf_phoner_word"]),
        ("phoner_xlmr_syllable_vs_bilstm_crf", by_id["exp17_xlmr_phoner_syllable"], tv1_by_id["bilstm_cnn_crf_phoner_word"]),
        ("phoner_xlmr_word_vs_bilstm_crf", by_id["exp18_xlmr_phoner_word"], tv1_by_id["bilstm_cnn_crf_phoner_word"]),
    ]

    bootstrap_results: Dict[str, Any] = {}
    print("\n--- KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (PAIRED BOOTSTRAP, 1,000 RESAMPLES) ---")
    for pair_name, ra, rb in bootstrap_pairs:
        res_bs = paired_bootstrap_test(
            _load(ra["gold_path"]),
            _load(ra["pred_path"]),
            _load(rb["gold_path"]),
            _load(rb["pred_path"]),
            n_resamples=1000,
            seed=42,
        )
        bootstrap_results[pair_name] = res_bs
        sig_str = "CÓ Ý NGHĨA (p<0.05)" if res_bs["significant_005"] else "KHÔNG có ý nghĩa (p>=0.05)"
        print(
            f"  {pair_name:38s}: ΔF1={res_bs['delta_f1']*100:+6.2f}% "
            f"(95% CI [{res_bs['ci95_low']*100:+.2f}%, {res_bs['ci95_high']*100:+.2f}%]) | "
            f"p={res_bs['p_value_two_sided']:.4f} -> {sig_str} | "
            f"Win/Tie/Loss={res_bs['sentence_a_better']}/{res_bs['sentence_ties']}/{res_bs['sentence_b_better']}"
        )

    # --- D. Trích xuất ví dụ phân tích lỗi định tính ---
    divergences = {
        "phoner_phobert_word_vs_bilstm_crf": find_divergence_examples(
            "data/processed/phoner_covid19/word/test.jsonl",
            "outputs/predictions/exp15_phobert_phoner_word_test.jsonl",
            "outputs/predictions/bilstm_crf_test.jsonl",
            "PhoBERT_word",
            "BiLSTM_CRF",
            max_examples=3,
        ),
        "vlsp_phobert_word_vs_xlmr_word": find_divergence_examples(
            "data/processed/vlsp/word/test.jsonl",
            "outputs/predictions/exp01_phobert_baseline_test.jsonl",
            "outputs/predictions/exp14_xlmr_vlsp_word_test.jsonl",
            "PhoBERT_word",
            "XLMR_word",
            max_examples=3,
        ),
    }

    # --- E. Xuất JSON tổng hợp ---
    summary_payload = {
        "plm_experiments": plm_records,
        "tv1_bilstm_experiments": tv1_records,
        "paired_bootstrap": bootstrap_results,
        "divergence_examples": divergences,
    }
    json_out = TABLES_DIR / "granularity_comparison_summary.json"
    json_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[*] Đã lưu tổng hợp JSON vào: {json_out}")

    # --- F. Xuất bảng Markdown đầy đủ (BƯỚC 5) ---
    md_lines = [
        "# BÁO CÁO SO SÁNH MỨC TỪ (WORD-LEVEL) VÀ MỨC ÂM TIẾT (SYLLABLE-LEVEL) — NHÁNH PRETRAINED PLM",
        "",
        "## 1. Bảng Kết quả Tổng hợp Ma trận 2×2 — VLSP 2016 (1,872 câu test, 1,804 thực thể)",
        "",
        "| Model | Granularity | Precision (%) | Recall (%) | Micro-F1 (%) | Macro-F1 (%) | Char-Unified F1 (%) | Runtime (s) | Peak VRAM (MB) | Avg Tokens | Avg Subwords (Fertility) | Max SW |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in [x for x in plm_records if x["dataset"] == "vlsp"]:
        o = r["metrics"]["overall"]
        ts = r["test_seq_stats"]
        res = r["resources"]
        md_lines.append(
            f"| **{r['model_short']}** (`{r['exp_id']}`) | `{r['granularity']}` | "
            f"{o['precision']*100:.2f} | {o['recall']*100:.2f} | **{o['f1']*100:.2f}** | "
            f"{r['metrics']['macro_f1']*100:.2f} | {r['char_unified_metrics']['f1']*100:.2f} | "
            f"{res['runtime_s']:.1f} | {res['max_gpu_memory_mb']:.1f} | "
            f"{ts['avg_tokens']:.2f} | {ts['avg_subwords']:.2f} ({ts['subword_fertility']:.2f}×) | {ts['max_subwords']} |"
        )

    md_lines.extend([
        "",
        "## 2. Bảng Kết quả Tổng hợp Ma trận 2×2 & Kiểm định chéo BiLSTM-CRF — PhoNER_COVID19 (3,000 câu test chung, 11,735 thực thể)",
        "",
        "| Model | Granularity | Precision (%) | Recall (%) | Micro-F1 (%) | Macro-F1 (%) | Char-Unified F1 (%) | Runtime (s) | Peak VRAM (MB) | Avg Tokens | Avg Subwords (Fertility) | Max SW |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in tv1_records:
        o = r["metrics"]["overall"]
        res = r["resources"]
        md_lines.append(
            f"| **{r['model_short']}** (`{r['exp_id']}`) | `{r['granularity']}` | "
            f"{o['precision']*100:.2f} | {o['recall']*100:.2f} | **{o['f1']*100:.2f}** | "
            f"{r['metrics']['macro_f1']*100:.2f} | {r['char_unified_metrics']['f1']*100:.2f} | "
            f"{res['runtime_s']:.1f} | N/A (CPU/TV1) | 28.56 | N/A (Word+Char) | 162 |"
        )
    for r in [x for x in plm_records if x["dataset"] == "phoner_covid19"]:
        o = r["metrics"]["overall"]
        ts = r["test_seq_stats"]
        res = r["resources"]
        md_lines.append(
            f"| **{r['model_short']}** (`{r['exp_id']}`) | `{r['granularity']}` | "
            f"{o['precision']*100:.2f} | {o['recall']*100:.2f} | **{o['f1']*100:.2f}** | "
            f"{r['metrics']['macro_f1']*100:.2f} | {r['char_unified_metrics']['f1']*100:.2f} | "
            f"{res['runtime_s']:.1f} | {res['max_gpu_memory_mb']:.1f} | "
            f"{ts['avg_tokens']:.2f} | {ts['avg_subwords']:.2f} ({ts['subword_fertility']:.2f}×) | {ts['max_subwords']} |"
        )

    md_lines.extend([
        "",
        "## 3. Kiểm định Ý nghĩa Thống kê (Paired Bootstrap — 1,000 resamples)",
        "",
        "| Cặp đối chiếu (A vs B) | F1(A) (%) | F1(B) (%) | ΔF1 (A - B) | 95% CI | p-value (2-sided) | Kết luận |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for pname, bs in bootstrap_results.items():
        sig = "Có ý nghĩa (p < 0.05)" if bs["significant_005"] else "Không có ý nghĩa"
        md_lines.append(
            f"| `{pname}` | {bs['f1_a']*100:.2f} | {bs['f1_b']*100:.2f} | **{bs['delta_f1']*100:+.2f}%** | "
            f"[{bs['ci95_low']*100:+.2f}%, {bs['ci95_high']*100:+.2f}%] | {bs['p_value_two_sided']:.4f} | {sig} |"
        )

    md_out = TABLES_DIR / "granularity_comparison_report.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[*] Đã lưu báo cáo Markdown vào: {md_out}")

    # --- G. Xuất bảng LaTeX chuẩn ACL ---
    def _export_granularity_tex(records: List[Dict[str, Any]], out_path: Path, caption: str, label: str) -> None:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcccccccc}",
            r"\hline",
            r"\textbf{Model} & \textbf{Granularity} & \textbf{P (\%)} & \textbf{R (\%)} & \textbf{Micro-F1} & \textbf{Macro-F1} & \textbf{Time (s)} & \textbf{VRAM (MB)} & \textbf{Avg Tok} & \textbf{Avg SW} \\",
            r"\hline",
        ]
        best_f1 = max(r["metrics"]["overall"]["f1"] for r in records)
        for r in records:
            o = r["metrics"]["overall"]
            res = r["resources"]
            ts = r["test_seq_stats"]
            f1_str = f"\\textbf{{{o['f1']*100:.2f}}}" if abs(o["f1"] - best_f1) < 1e-6 else f"{o['f1']*100:.2f}"
            lines.append(
                f"{r['model_short']} & {r['granularity']} & {o['precision']*100:.2f} & {o['recall']*100:.2f} & "
                f"{f1_str} & {r['metrics']['macro_f1']*100:.2f} & {res['runtime_s']:.1f} & "
                f"{res['max_gpu_memory_mb']:.0f} & {ts['avg_tokens']:.2f} & {ts['avg_subwords']:.2f} ({ts['subword_fertility']:.2f}$\\times$) \\\\"
            )
        lines += [
            r"\hline",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table*}",
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _export_granularity_tex(
        [x for x in plm_records if x["dataset"] == "vlsp"],
        TABLES_DIR / "granularity_comparison_vlsp.tex",
        "So sánh hiệu năng và chi phí tài nguyên theo ma trận $2\\times 2$ (Mô hình $\\times$ Mức biểu diễn) trên tập test VLSP 2016 (1.872 câu).",
        "tab:granularity-vlsp",
    )
    _export_granularity_tex(
        [x for x in plm_records if x["dataset"] == "phoner_covid19"],
        TABLES_DIR / "granularity_comparison_phoner.tex",
        "So sánh hiệu năng và chi phí tài nguyên theo ma trận $2\\times 2$ (Mô hình $\\times$ Mức biểu diễn) trên tập test PhoNER\\_COVID19 (3.000 câu).",
        "tab:granularity-phoner",
    )

    # Bảng LaTeX kiểm định chéo liên mô hình trên PhoNER_COVID19 (TV1 vs TV2)
    phoner_types = [
        "AGE", "DATE", "GENDER", "JOB", "LOCATION",
        "NAME", "ORGANIZATION", "PATIENT_ID", "SYMPTOM_AND_DISEASE", "TRANSPORTATION",
    ]
    cross_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Loại thực thể} & \textbf{Support} & \textbf{BiLSTM-CRF (W)} & \textbf{PhoBERT (W)} & \textbf{PhoBERT (S)} & \textbf{XLM-R (S)} & \textbf{XLM-R (W)} \\",
        r"\hline",
    ]
    m_bilstm = tv1_by_id["bilstm_cnn_crf_phoner_word"]["metrics"]
    m_pb_w = by_id["exp15_phobert_phoner_word"]["metrics"]
    m_pb_s = by_id["exp16_phobert_phoner_syllable"]["metrics"]
    m_xl_s = by_id["exp17_xlmr_phoner_syllable"]["metrics"]
    m_xl_w = by_id["exp18_xlmr_phoner_word"]["metrics"]
    for t in phoner_types:
        supp = m_bilstm["per_type"][t]["support"]
        vals = [m["per_type"][t]["f1"] * 100 for m in (m_bilstm, m_pb_w, m_pb_s, m_xl_s, m_xl_w)]
        mx = max(vals)
        fmt = [f"\\textbf{{{v:.2f}}}" if abs(v - mx) < 1e-4 else f"{v:.2f}" for v in vals]
        t_tex = t.replace("_", r"\_")
        cross_lines.append(f"{t_tex} & {supp} & " + " & ".join(fmt) + r" \\")
    cross_lines.append(r"\hline")
    for row_lbl, getter in [
        (r"\textbf{Micro-F1}", lambda m: m["overall"]["f1"] * 100),
        (r"\textbf{Macro-F1}", lambda m: m["macro_f1"] * 100),
    ]:
        vals = [getter(m) for m in (m_bilstm, m_pb_w, m_pb_s, m_xl_s, m_xl_w)]
        mx = max(vals)
        fmt = [f"\\textbf{{{v:.2f}}}" if abs(v - mx) < 1e-4 else f"{v:.2f}" for v in vals]
        supp_str = "11735" if "Micro" in row_lbl else "---"
        cross_lines.append(f"{row_lbl} & {supp_str} & " + " & ".join(fmt) + r" \\")
    cross_lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Đối chiếu Entity-level $\text{F}_1$ (\%) theo từng loại thực thể giữa Nhánh 1 (BiLSTM-CNN-CRF của TV1) và Nhánh 2 (PhoBERT \& XLM-R mức từ/âm tiết của TV2) trên cùng 3.000 câu test PhoNER\_COVID19 (\texttt{test\_indices.json}).}",
        r"\label{tab:cross-model-phoner}",
        r"\end{table*}",
    ]
    (TABLES_DIR / "cross_model_phoner_comparison.tex").write_text("\n".join(cross_lines) + "\n", encoding="utf-8")
    print("[*] Đã xuất 3 bảng LaTeX chuẩn ACL vào outputs/tables/.")

    # --- H. Tạo biểu đồ trực quan hóa ---
    generate_charts(plm_records, tv1_records)
    print("[*] Đã xuất 4 biểu đồ 300-DPI vào outputs/figures/.")
    print("\n=== HOÀN TẤT PHÂN TÍCH VÀ KIỂM ĐỊNH CHÉO ===")


if __name__ == "__main__":
    main()
