.PHONY: help install download prep test

PYTHON ?= python

help:
	@echo "Các lệnh khả dụng trong dự án DeepLearning-NER:"
	@echo "  make install    - Cài đặt các thư viện từ requirements.txt"
	@echo "  make download   - Tải toàn bộ dữ liệu (PhoNER_COVID19 & VLSP 2016)"
	@echo "  make prep       - Tiền xử lý dữ liệu về chuẩn JSONL (mức từ & âm tiết)"
	@echo "  make test       - Chạy toàn bộ các bài kiểm thử unit tests"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

download:
	bash data/download_data.sh

prep:
	$(PYTHON) src/data_prep/process_phoner.py
	$(PYTHON) src/data_prep/process_vlsp.py

test:
	$(PYTHON) -m unittest discover -s tests -p "tests_*.py" -v
