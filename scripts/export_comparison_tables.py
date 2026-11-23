"""
export_comparison_tables.py - Tổng hợp kết quả thực nghiệm Pretrained PLM.
"""
import json
import os
from pathlib import Path

experiments_summary = [
    {
        "id": "exp01_phobert_baseline",
        "group": "Baseline",
        "model": "PhoBERT-base-v2",
        "granularity": "word",
        "freezing": "Full fine-tuning (0 freeze)",
        "pooling": "last",
        "scheduler": "linear",
        "epochs": 4,
        "trainable_params": 135005193,
        "total_params": 135005193,
        "trainable_pct": 100.0,
        "runtime_s": 413.8,
        "vram_mb": 4482.4,
        "dev_f1": 0.9456,
        "test_p": 0.9374,
        "test_r": 0.9468,
        "test_f1": 0.9421,
        "test_macro_f1": 0.8976,
        "per_type_f1": {"LOC": 0.9347, "MISC": 0.8657, "ORG": 0.8135, "PER": 0.9765}
    },
    {
        "id": "exp02_xlmr_baseline",
        "group": "Baseline",
        "model": "XLM-RoBERTa-base",
        "granularity": "syllable",
        "freezing": "Full fine-tuning (0 freeze)",
        "pooling": "last",
        "scheduler": "linear",
        "epochs": 4,
        "trainable_params": 278050569,
        "total_params": 278050569,
        "trainable_pct": 100.0,
        "runtime_s": 1479.8,
        "vram_mb": 7249.5,
        "dev_f1": 0.9357,
        "test_p": 0.9298,
        "test_r": 0.9396,
        "test_f1": 0.9347,
        "test_macro_f1": 0.8925,
        "per_type_f1": {"LOC": 0.9259, "MISC": 0.8923, "ORG": 0.7778, "PER": 0.9741}
    },
    {
        "id": "exp03_phobert_freeze_backbone",
        "group": "Nhóm A - Freezing",
        "model": "PhoBERT-base-v2",
        "granularity": "word",
        "freezing": "Frozen backbone (Head-only)",
        "pooling": "last",
        "scheduler": "linear",
        "epochs": 4,
        "trainable_params": 6921,
        "total_params": 135005193,
        "trainable_pct": 0.005,
        "runtime_s": 172.0,
        "vram_mb": 1349.2,
        "dev_f1": 0.2040,
        "test_p": 0.5538,
        "test_r": 0.1197,
        "test_f1": 0.1969,
        "test_macro_f1": 0.0878,
        "per_type_f1": {"LOC": 0.0026, "MISC": 0.0000, "ORG": 0.0000, "PER": 0.3485}
    },
    {
        "id": "exp05_phobert_freeze_embeddings",
        "group": "Nhóm A - Freezing",
        "model": "PhoBERT-base-v2",
        "granularity": "word",
        "freezing": "Freeze Embeddings only",
        "pooling": "last",
        "scheduler": "linear",
        "epochs": 4,
        "trainable_params": 85651977,
        "total_params": 135005193,
        "trainable_pct": 63.443,
        "runtime_s": 222.3,
        "vram_mb": 4089.8,
        "dev_f1": 0.9489,
        "test_p": 0.9347,
        "test_r": 0.9446,
        "test_f1": 0.9396,
        "test_macro_f1": 0.8826,
        "per_type_f1": {"LOC": 0.9401, "MISC": 0.8358, "ORG": 0.7805, "PER": 0.9741}
    },
    {
        "id": "exp08_phobert_sched_cosine",
        "group": "Nhóm C - Scheduler",
        "model": "PhoBERT-base-v2",
        "granularity": "word",
        "freezing": "Full fine-tuning (0 freeze)",
        "pooling": "last",
        "scheduler": "cosine",
        "epochs": 4,
        "trainable_params": 135005193,
        "total_params": 135005193,
        "trainable_pct": 100.0,
        "runtime_s": 568.1,
        "vram_mb": 4482.4,
        "dev_f1": 0.9455,
        "test_p": 0.9393,
        "test_r": 0.9440,
        "test_f1": 0.9417,
        "test_macro_f1": 0.8955,
        "per_type_f1": {"LOC": 0.9368, "MISC": 0.8657, "ORG": 0.8037, "PER": 0.9757}
    },
    {
        "id": "exp10_phobert_pooling_concat4",
        "group": "Nhóm E - Pooling",
        "model": "PhoBERT-base-v2",
        "granularity": "word",
        "freezing": "Full fine-tuning (0 freeze)",
        "pooling": "concat_4",
        "scheduler": "linear",
        "epochs": 4,
        "trainable_params": 135025929,
        "total_params": 135025929,
        "trainable_pct": 100.0,
        "runtime_s": 610.4,
        "vram_mb": 4559.8,
        "dev_f1": 0.9463,
        "test_p": 0.9414,
        "test_r": 0.9534,
        "test_f1": 0.9474,
        "test_macro_f1": 0.9114,
        "per_type_f1": {"LOC": 0.9422, "MISC": 0.9062, "ORG": 0.8182, "PER": 0.9788}
    }
]

out_dir = Path("outputs/tables")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Save JSON
with open(out_dir / "plm_experiment_comparison.json", "w", encoding="utf-8") as f:
    json.dump(experiments_summary, f, indent=2, ensure_ascii=False)

# 2. Save Markdown
md_lines = [
    "# Bảng Tổng hợp Kết quả Thực nghiệm Pretrained PLM (PhoBERT & XLM-R)",
    "",
    "| Exp ID | Nhóm Thử nghiệm | Mô hình & Cấu hình | Trainable Params | VRAM Đỉnh | Thời gian | Dev F1 (%) | Test P (%) | Test R (%) | Test F1 (%) | Test Macro-F1 (%) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
]
for exp in experiments_summary:
    md_lines.append(
        f"| `{exp['id']}` | {exp['group']} | **{exp['model']}** ({exp['freezing']}, pooling: `{exp['pooling']}`) | "
        f"{exp['trainable_params']:,} ({exp['trainable_pct']:.2f}%) | {exp['vram_mb']:.1f} MB | {exp['runtime_s']:.1f}s | "
        f"{exp['dev_f1']*100:.2f}% | {exp['test_p']*100:.2f}% | {exp['test_r']*100:.2f}% | **{exp['test_f1']*100:.2f}%** | {exp['test_macro_f1']*100:.2f}% |"
    )

with open(out_dir / "plm_experiment_comparison.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# 3. Save LaTeX Table
tex_lines = [
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\caption{Kết quả thực nghiệm so sánh các chiến lược fine-tuning cho PhoBERT và XLM-R trên tập VLSP 2016.}",
    "\\label{tab:plm_experiments}",
    "\\begin{tabular}{llcccccc}",
    "\\hline",
    "\\textbf{Exp ID} & \\textbf{Mô hình \\& Cấu hình} & \\textbf{Trainable} & \\textbf{VRAM} & \\textbf{Time} & \\textbf{Test P (\\%)} & \\textbf{Test R (\\%)} & \\textbf{Test F1 (\\%)} \\\\",
    "\\hline"
]
for exp in experiments_summary:
    tex_lines.append(
        f"`{exp['id']}` & {exp['model']} ({exp['freezing']}) & "
        f"{exp['trainable_pct']:.1f}\\% & {exp['vram_mb']:.0f} MB & {exp['runtime_s']:.0f}s & "
        f"{exp['test_p']*100:.2f} & {exp['test_r']*100:.2f} & \\textbf{{{exp['test_f1']*100:.2f}}} \\\\"
    )
tex_lines.extend([
    "\\hline",
    "\\end{tabular}",
    "\\end{table*}"
])

with open(out_dir / "plm_experiment_comparison.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

print("Exported JSON, Markdown, and LaTeX comparison tables successfully!")
