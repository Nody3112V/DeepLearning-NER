"""
examples_predictor_demo.py - Ví dụ tích hợp tối giản cho Thành viên 3 (demo/app.py).

Thành viên 3 KHÔNG cần hiểu chi tiết huấn luyện hay subword alignment; chỉ cần:
  1. Khởi tạo predictor 1 lần duy nhất khi khởi động app (có thể truyền chung 1 WordSegmenter).
  2. Gọi `predictor.predict(sentence)` để nhận danh sách Dict chuẩn:
       [{"text": ..., "type": ..., "start": ..., "end": ...}, ...]

Chạy thử:
    PYTHONUTF8=1 python examples_predictor_demo.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.common.word_segmenter import get_segmenter
from src.models.pretrained.predictor import PRESET_CHECKPOINTS, load_predictor


def run_demo() -> None:
    print("=================================================================")
    print("  VÍ DỤ TÍCH HỢP PHOBERT & XLM-R CHO DEMO (THÀNH VIÊN 3)")
    print("=================================================================\n")

    # 1. Khởi tạo 1 instance WordSegmenter dùng chung (nếu nạp nhiều mô hình word-level)
    #    Nếu máy chưa cài Java/VnCoreNLP, predictor vẫn tự fallback an toàn không sập app.
    shared_segmenter = get_segmenter()

    sample_sentences = [
        "Ban biên tập báo Tuổi Trẻ cử nữ phóng viên Uyên Ly sang Mỹ.",
        "Ông Nguyễn Văn An làm việc tại Bệnh viện Bạch Mai, Hà Nội rồi bay vào Đà Nẵng.",
        "Hôm nay trời mưa nhẹ và thời tiết rất mát mẻ.",
        "",  # Câu rỗng: demo không bị crash
    ]

    # 2. Nạp PhoBERT (VLSP - tốt nhất, concat_4)
    if Path(PRESET_CHECKPOINTS["phobert_vlsp_best"]).exists():
        t0 = time.time()
        phobert = load_predictor("phobert_vlsp_best", segmenter=shared_segmenter)
        print(f"[PhoBERT] Nạp xong trong {time.time() - t0:.2f}s | Info: {phobert.get_model_info()}\n")
        for sent in sample_sentences:
            t_inf = time.time()
            entities = phobert.predict(sent)
            dt_ms = (time.time() - t_inf) * 1000
            print(f"  Input  : {sent!r}")
            print(f"  Output ({dt_ms:.1f} ms): {json.dumps(entities, ensure_ascii=False)}")
        print()

    # 3. Nạp XLM-RoBERTa (VLSP - mức âm tiết, không cần tách từ)
    if Path(PRESET_CHECKPOINTS["xlmr_vlsp_syllable"]).exists():
        t0 = time.time()
        xlmr = load_predictor("xlmr_vlsp_syllable")
        print(f"[XLM-R]   Nạp xong trong {time.time() - t0:.2f}s | Info: {xlmr.get_model_info()}\n")
        for sent in sample_sentences[:3]:
            t_inf = time.time()
            entities = xlmr.predict(sent)
            dt_ms = (time.time() - t_inf) * 1000
            print(f"  Input  : {sent!r}")
            print(f"  Output ({dt_ms:.1f} ms): {json.dumps(entities, ensure_ascii=False)}")
        print()

    # 4. Ví dụ PhoNER_COVID19 (10 loại thực thể dịch tễ)
    if Path(PRESET_CHECKPOINTS["xlmr_phoner_syllable"]).exists():
        phoner_xlmr = load_predictor("xlmr_phoner_syllable")
        covid_sent = '"Bệnh nhân 448", thực tập sinh tại Bệnh viện Đà Nẵng từ ngày 22/6 đến 17/7.'
        entities = phoner_xlmr.predict(covid_sent)
        print(f"[XLM-R PhoNER_COVID19] Input : {covid_sent}")
        print(f"[XLM-R PhoNER_COVID19] Output: {json.dumps(entities, ensure_ascii=False)}")


if __name__ == "__main__":
    run_demo()
