"""
generate_attention_maps.py - Chạy trích xuất, kiểm định và xuất bộ biểu đồ Attention Map
chất lượng cao (300 DPI) trên các câu kiểm thử thực tế của VLSP 2016 và PhoNER_COVID19
cho cả hai mô hình đã huấn luyện PhoBERT-base-v2 và XLM-RoBERTa-base.

Đầu ra:
  - Ảnh 300-DPI tại outputs/figures/attention/ (tên file gồm Model, Sentence ID, Layer, Head, Level)
  - Báo cáo số liệu định lượng tại outputs/tables/attention_analysis_summary.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.common.schema import NERSample, load_jsonl
from src.models.pretrained.attention_viz import (
    DEFAULT_OUTPUT_DIR,
    AttentionExtractionResult,
    aggregate_subword_attention_to_words,
    build_attention_filename,
    compute_attention_rollout,
    extract_sentence_attention,
    inspect_model_attention_spec,
    load_model_and_tokenizer_for_viz,
    plot_attention_heatmap,
    plot_head_comparison,
    plot_layer_comparison,
    plot_model_attention_comparison,
    select_layer_head_attention,
    validate_attention_result,
)

OUT_DIR = DEFAULT_OUTPUT_DIR
SUMMARY_JSON = Path("outputs/tables/attention_analysis_summary.json")


def find_sample(path: str, sample_id: str) -> NERSample:
    for s in load_jsonl(path):
        if s.id == sample_id:
            return s
    raise ValueError(f"Không tìm thấy {sample_id} trong {path}")


def summarize_attention_patterns(
    res: AttentionExtractionResult,
    layer_idx: int = -1,
    head: Any = "mean",
) -> Dict[str, Any]:
    """Trích xuất thống kê định lượng chính xác để làm căn cứ nhận xét không suy diễn."""
    sel, norm_l, _ = select_layer_head_attention(res.attentions, layer_idx=layer_idx, head=head)
    w_res = aggregate_subword_attention_to_words(
        sel,
        res.word_ids,
        res.original_tokens,
        subword_tokens=res.subword_tokens,
        query_agg="mean",
        key_agg="sum",
        include_special_tokens=False,
        renormalize=True,
    )

    # 1. Tỷ lệ chú ý vào special tokens (<s>/</s>) và entropy trung bình qua tất cả các tầng
    layer_stats = []
    for l in range(res.num_layers):
        l_mean = res.attentions[l].mean(dim=0)  # [L, L]
        lw = aggregate_subword_attention_to_words(
            l_mean, res.word_ids, res.original_tokens,
            subword_tokens=res.subword_tokens, query_agg="mean", key_agg="sum",
            include_special_tokens=False, renormalize=True,
        )
        p = np.clip(l_mean.numpy(), 1e-12, 1.0)
        ent = float(-(p * np.log(p)).sum(axis=-1).mean())
        diag_mean = float(np.diag(lw.matrix).mean())
        layer_stats.append({
            "layer": l + 1,
            "special_token_sink_pct": round(lw.special_token_mass_removed * 100, 2),
            "avg_entropy_nats": round(ent, 3),
            "word_self_attn_mean": round(diag_mean, 4),
        })

    # 2. Top-3 từ được chú ý nhiều nhất cho mỗi token thuộc thực thể (ở tầng đang xét)
    entity_focus = []
    tags_ref = res.predicted_tags or res.gold_tags or ["O"] * len(w_res.labels)
    for i, (tok, tag) in enumerate(zip(w_res.labels, tags_ref)):
        if tag != "O":
            row = w_res.matrix[i]
            top_idx = np.argsort(row)[::-1][:4]
            top_keys = [
                {"token": w_res.labels[j], "tag": tags_ref[j], "weight": round(float(row[j]), 4)}
                for j in top_idx
            ]
            entity_focus.append({
                "query_idx": i,
                "query_token": tok,
                "pred_tag": tag,
                "gold_tag": res.gold_tags[i] if res.gold_tags and i < len(res.gold_tags) else None,
                "top_attended_keys": top_keys,
            })

    # 3. Các từ bị tách thành >= 2 subwords
    multi_sw_words = {
        res.original_tokens[wid]: [res.subword_tokens[i] for i in sw_list]
        for wid, sw_list in w_res.word_to_subwords.items()
        if len(sw_list) > 1
    }

    return {
        "model_name": res.model_name,
        "sentence_id": res.sentence_id,
        "layer": norm_l + 1,
        "head": str(head),
        "num_original_tokens": len(res.original_tokens),
        "num_subwords": len(res.subword_tokens),
        "tokens": res.original_tokens,
        "subwords": res.subword_tokens,
        "gold_tags": res.gold_tags,
        "predicted_tags": res.predicted_tags,
        "exact_tag_match": (res.predicted_tags == res.gold_tags) if (res.predicted_tags and res.gold_tags) else None,
        "aggregation_method": w_res.method_description,
        "multi_subword_words": multi_sw_words,
        "entity_query_focus": entity_focus,
        "per_layer_stats": layer_stats,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== GENERATING ATTENTION VISUALIZATIONS (device={device}) ===\n", flush=True)

    report: Dict[str, Any] = {
        "model_specs": {},
        "validations": {},
        "saved_figures": [],
        "quantitative_patterns": {},
    }

    # ----------------------------------------------------------------------- #
    # CASE 1: VLSP 2016 — Sentence 00627 (ORG + PER + LOC)
    # "Ban biên_tập Tuổi_Trẻ quyết_định cử nữ phóng_viên Uyên Ly sang Mỹ ."
    # ----------------------------------------------------------------------- #
    vlsp_w_sample = find_sample("data/processed/vlsp/word/test.jsonl", "vlsp_word_test_00627")
    vlsp_s_sample = find_sample("data/processed/vlsp/syllable/test.jsonl", "vlsp_syllable_test_00627")

    print("[1/4] Loading PhoBERT VLSP best checkpoint (exp10_phobert_pooling_concat4)...", flush=True)
    pb_vlsp_model, pb_tok, pb_vlsp_id2tag, _ = load_model_and_tokenizer_for_viz(
        "outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt", device=device
    )
    report["model_specs"]["phobert_vlsp"] = asdict(inspect_model_attention_spec(pb_vlsp_model, pb_tok, device=device))

    res_pb_vlsp = extract_sentence_attention(
        pb_vlsp_model, pb_tok, vlsp_w_sample.tokens,
        layer_idx=-1, head="mean", id2tag=pb_vlsp_id2tag, gold_tags=vlsp_w_sample.tags,
        sentence_id=vlsp_w_sample.id, device=device,
    )
    report["validations"]["phobert_vlsp_00627"] = validate_attention_result(res_pb_vlsp)
    report["quantitative_patterns"]["phobert_vlsp_00627"] = summarize_attention_patterns(res_pb_vlsp, layer_idx=-1, head="mean")

    # Fig 1: PhoBERT VLSP Layer 12 Mean-Head Word-level
    w_pb_vlsp = aggregate_subword_attention_to_words(
        res_pb_vlsp.selected_attention, res_pb_vlsp.word_ids, res_pb_vlsp.original_tokens,
        subword_tokens=res_pb_vlsp.subword_tokens, query_agg="mean", key_agg="sum",
        include_special_tokens=False, renormalize=True,
    )
    f1 = build_attention_filename(OUT_DIR, res_pb_vlsp.model_name, vlsp_w_sample.id, 11, "mean", "word")
    plot_attention_heatmap(
        w_pb_vlsp.labels, w_pb_vlsp.matrix, output_path=f1,
        title=f"PhoBERT-base-v2 (concat_4) | {vlsp_w_sample.id} | Layer 12 | Head=Mean",
        subtitle=w_pb_vlsp.method_description,
        tags=res_pb_vlsp.predicted_tags, annot=True,
    )
    report["saved_figures"].append(str(f1))

    # Fig 2: PhoBERT VLSP Layer 12 Head Comparison (6 heads)
    f2 = build_attention_filename(OUT_DIR, res_pb_vlsp.model_name, vlsp_w_sample.id, 11, "heads_0_2_4_6_8_11", "word")
    plot_head_comparison(res_pb_vlsp, layer_idx=11, heads=(0, 2, 4, 6, 8, 11), word_level=True, output_path=f2)
    report["saved_figures"].append(str(f2))

    # Fig 3: PhoBERT VLSP Layer Comparison (Layers 1, 4, 8, 12)
    f3 = build_attention_filename(OUT_DIR, res_pb_vlsp.model_name, vlsp_w_sample.id, "1_4_8_12", "mean", "word")
    plot_layer_comparison(res_pb_vlsp, layers=(0, 3, 7, 11), head="mean", word_level=True, output_path=f3)
    report["saved_figures"].append(str(f3))

    del pb_vlsp_model
    torch.cuda.empty_cache()

    print("[2/4] Loading XLM-R VLSP checkpoint (exp02_xlmr_baseline)...", flush=True)
    xl_vlsp_model, xl_tok, xl_vlsp_id2tag, _ = load_model_and_tokenizer_for_viz(
        "outputs/checkpoints/pretrained/exp02_xlmr_baseline/best_model.pt", device=device
    )
    report["model_specs"]["xlmr_vlsp"] = asdict(inspect_model_attention_spec(xl_vlsp_model, xl_tok, device=device))

    res_xl_vlsp = extract_sentence_attention(
        xl_vlsp_model, xl_tok, vlsp_s_sample.tokens,
        layer_idx=-1, head="mean", id2tag=xl_vlsp_id2tag, gold_tags=vlsp_s_sample.tags,
        sentence_id=vlsp_s_sample.id, device=device,
    )
    report["validations"]["xlmr_vlsp_00627"] = validate_attention_result(res_xl_vlsp)
    report["quantitative_patterns"]["xlmr_vlsp_00627"] = summarize_attention_patterns(res_xl_vlsp, layer_idx=-1, head="mean")

    # Fig 4: XLM-R VLSP Layer 12 Raw Subword-level (giữ <s>/</s> để quan sát attention sink + SentencePiece subwords)
    f4 = build_attention_filename(OUT_DIR, res_xl_vlsp.model_name, vlsp_s_sample.id, 11, "mean", "subword")
    plot_attention_heatmap(
        res_xl_vlsp.subword_tokens, res_xl_vlsp.selected_attention, output_path=f4,
        title=f"XLM-RoBERTa-base (Syllable) | {vlsp_s_sample.id} | Layer 12 | Head=Mean (Raw Subwords)",
        subtitle=f"Raw subword-level attention ({len(res_xl_vlsp.subword_tokens)} subwords, bao gồm <s>/</s>)",
        annot=False,
    )
    report["saved_figures"].append(str(f4))

    # Fig 5: Side-by-side PhoBERT (Word) vs XLM-R (Syllable) on VLSP 00627
    f5 = OUT_DIR / "phobert_vs_xlmr_vlsp_test_00627_L12_Hmean_word.png"
    plot_model_attention_comparison(res_pb_vlsp, res_xl_vlsp, layer_idx=11, head="mean", output_path=f5)
    report["saved_figures"].append(str(f5))

    del xl_vlsp_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------------- #
    # CASE 2: PhoNER_COVID19 — Sentence 00277 (PATIENT_ID + JOB + LOCATION + DATE)
    # '" Bệnh_nhân 448 " , thực_tập_sinh tại Bệnh_viện Đà_Nẵng từ ngày 22/6 đến 17/7 .'
    # ----------------------------------------------------------------------- #
    ph_w_sample = find_sample("data/processed/phoner_covid19/word/test.jsonl", "phoner_word_test_00277")
    ph_s_sample = find_sample("data/processed/phoner_covid19/syllable/test.jsonl", "phoner_syllable_test_00277")

    print("[3/4] Loading PhoBERT PhoNER checkpoint (exp15_phobert_phoner_word)...", flush=True)
    pb_ph_model, pb_tok2, pb_ph_id2tag, _ = load_model_and_tokenizer_for_viz(
        "outputs/checkpoints/pretrained/exp15_phobert_phoner_word/best_model.pt", device=device
    )
    res_pb_ph = extract_sentence_attention(
        pb_ph_model, pb_tok2, ph_w_sample.tokens,
        layer_idx=-1, head="mean", id2tag=pb_ph_id2tag, gold_tags=ph_w_sample.tags,
        sentence_id=ph_w_sample.id, device=device,
    )
    report["validations"]["phobert_phoner_00277"] = validate_attention_result(res_pb_ph)
    report["quantitative_patterns"]["phobert_phoner_00277"] = summarize_attention_patterns(res_pb_ph, layer_idx=-1, head="mean")

    # Fig 6: PhoBERT PhoNER Subword-level (cho thấy 448, 22/6, 17/7 tách thành nhiều subwords)
    f6 = build_attention_filename(OUT_DIR, res_pb_ph.model_name, ph_w_sample.id, 11, "mean", "subword")
    plot_attention_heatmap(
        res_pb_ph.subword_tokens, res_pb_ph.selected_attention, output_path=f6,
        title=f"PhoBERT-base-v2 | {ph_w_sample.id} | Layer 12 | Head=Mean (Raw Subwords)",
        subtitle=f"Raw subword-level ({len(res_pb_ph.original_tokens)} từ -> {len(res_pb_ph.subword_tokens)} subwords, giữ <s>/</s>)",
        annot=False,
    )
    report["saved_figures"].append(str(f6))

    # Fig 7: PhoBERT PhoNER Word-level (sau khi tổng hợp subword -> word)
    w_pb_ph = aggregate_subword_attention_to_words(
        res_pb_ph.selected_attention, res_pb_ph.word_ids, res_pb_ph.original_tokens,
        subword_tokens=res_pb_ph.subword_tokens, query_agg="mean", key_agg="sum",
        include_special_tokens=False, renormalize=True,
    )
    f7 = build_attention_filename(OUT_DIR, res_pb_ph.model_name, ph_w_sample.id, 11, "mean", "word")
    plot_attention_heatmap(
        w_pb_ph.labels, w_pb_ph.matrix, output_path=f7,
        title=f"PhoBERT-base-v2 | {ph_w_sample.id} | Layer 12 | Head=Mean (Word-level)",
        subtitle=w_pb_ph.method_description,
        tags=res_pb_ph.predicted_tags, annot=True,
    )
    report["saved_figures"].append(str(f7))

    del pb_ph_model
    torch.cuda.empty_cache()

    print("[4/4] Loading XLM-R PhoNER checkpoint (exp17_xlmr_phoner_syllable)...", flush=True)
    xl_ph_model, xl_tok2, xl_ph_id2tag, _ = load_model_and_tokenizer_for_viz(
        "outputs/checkpoints/pretrained/exp17_xlmr_phoner_syllable/best_model.pt", device=device
    )
    res_xl_ph = extract_sentence_attention(
        xl_ph_model, xl_tok2, ph_s_sample.tokens,
        layer_idx=-1, head="mean", id2tag=xl_ph_id2tag, gold_tags=ph_s_sample.tags,
        sentence_id=ph_s_sample.id, device=device,
    )
    report["validations"]["xlmr_phoner_00277"] = validate_attention_result(res_xl_ph)
    report["quantitative_patterns"]["xlmr_phoner_00277"] = summarize_attention_patterns(res_xl_ph, layer_idx=-1, head="mean")

    # Fig 8: XLM-R PhoNER Layer Comparison (L1, L4, L8, L12)
    f8 = build_attention_filename(OUT_DIR, res_xl_ph.model_name, ph_s_sample.id, "1_4_8_12", "mean", "word")
    plot_layer_comparison(res_xl_ph, layers=(0, 3, 7, 11), head="mean", word_level=True, output_path=f8)
    report["saved_figures"].append(str(f8))

    # Fig 9: Side-by-side PhoBERT (Word) vs XLM-R (Syllable) on PhoNER 00277
    f9 = OUT_DIR / "phobert_vs_xlmr_phoner_test_00277_L12_Hmean_word.png"
    plot_model_attention_comparison(res_pb_ph, res_xl_ph, layer_idx=11, head="mean", output_path=f9)
    report["saved_figures"].append(str(f9))

    del xl_ph_model
    torch.cuda.empty_cache()

    SUMMARY_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[*] Đã lưu tổng cộng {len(report['saved_figures'])} biểu đồ Attention vào: {OUT_DIR}")
    for fp in report["saved_figures"]:
        print(f"    - {fp} ({Path(fp).stat().st_size / 1024:.1f} KB)")
    print(f"[*] Đã lưu báo cáo định lượng vào: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
