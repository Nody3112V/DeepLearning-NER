#!/usr/bin/env bash
set -e

RAW_DIR="data/raw/phoner_covid19"
mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/data" ]; then
    echo "Đang tải PhoNER_COVID19 từ GitHub VinAIResearch..."
    git clone --depth 1 https://github.com/VinAIResearch/PhoNER_COVID19.git "$RAW_DIR/repo_tmp"
    mv "$RAW_DIR/repo_tmp/data" "$RAW_DIR/data"
    rm -rf "$RAW_DIR/repo_tmp"
    echo "Đã tải xong PhoNER_COVID19 vào $RAW_DIR/data!"
else
    echo "Dữ liệu PhoNER_COVID19 đã tồn tại tại $RAW_DIR/data."
fi
