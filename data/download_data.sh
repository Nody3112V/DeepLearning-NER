#!/usr/bin/env bash
set -e

# Đảm bảo đường dẫn thực thi luôn là thư mục gốc của dự án
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ==============================================================================
# Script hợp nhất tải toàn bộ dữ liệu dự án DeepLearning-NER
# Bao gồm:
#   1. PhoNER_COVID19 (từ GitHub VinAIResearch)
#   2. VLSP 2016 NER (từ HuggingFace Hub: datnth1709/VLSP2016-NER-data)
# ==============================================================================

# 1. Tải PhoNER_COVID19
PHONER_RAW_DIR="data/raw/phoner_covid19"
mkdir -p "$PHONER_RAW_DIR"

if [ ! -d "$PHONER_RAW_DIR/data" ]; then
    echo "=== [1/2] Đang tải PhoNER_COVID19 từ GitHub VinAIResearch... ==="
    git clone --depth 1 https://github.com/VinAIResearch/PhoNER_COVID19.git "$PHONER_RAW_DIR/repo_tmp"
    mv "$PHONER_RAW_DIR/repo_tmp/data" "$PHONER_RAW_DIR/data"
    rm -rf "$PHONER_RAW_DIR/repo_tmp"
    echo "-> Đã tải xong PhoNER_COVID19 vào $PHONER_RAW_DIR/data!"
else
    echo "-> Dữ liệu PhoNER_COVID19 đã tồn tại tại $PHONER_RAW_DIR/data."
fi

# 2. Tải VLSP 2016 NER
VLSP_RAW_DIR="data/raw/vlsp"
mkdir -p "$VLSP_RAW_DIR"

if [ ! -f "$VLSP_RAW_DIR/train.txt" ]; then
    echo "=== [2/2] Đang tải dữ liệu VLSP 2016 từ HuggingFace Hub... ==="
    python data/download_vlsp.py
    echo "-> Đã tải xong VLSP 2016 vào $VLSP_RAW_DIR!"
else
    echo "-> Dữ liệu VLSP 2016 đã tồn tại tại $VLSP_RAW_DIR."
fi

echo "=== HOÀN TẤT TẢI TOÀN BỘ DỮ LIỆU (PhoNER_COVID19 & VLSP 2016) ==="
