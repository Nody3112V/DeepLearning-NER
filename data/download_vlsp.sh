#!/usr/bin/env bash
set -e

RAW_DIR="data/raw/vlsp"
mkdir -p "$RAW_DIR"

if [ ! -f "$RAW_DIR/train.txt" ]; then
    echo "Đang tải dữ liệu VLSP 2016 từ HuggingFace Hub..."
    python data/download_vlsp.py
    echo "Đã tải xong VLSP 2016 vào $RAW_DIR!"
else
    echo "Dữ liệu VLSP 2016 đã tồn tại tại $RAW_DIR."
fi
