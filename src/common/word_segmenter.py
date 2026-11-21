"""
word_segmenter.py - Bộ tách từ tiếng Việt DUY NHẤT của dự án (VnCoreNLP / RDRSegmenter).

Dùng cho:
  * VLSP: đổi bản mức âm tiết (syllable) -> mức từ (word)   [TV2]
  * Suy luận / demo: tách từ câu thô trước khi đưa vào PhoBERT [TV2, TV3]

Yêu cầu: Java (JDK >= 1.8) và `pip install py_vncorenlp`.
Mô hình (~28 MB, chỉ jar + mô hình tách từ) tự tải về lần đầu vào
  <repo>/resources/vncorenlp   (đổi bằng biến môi trường VNCORENLP_DIR)
-> nhớ thêm `resources/vncorenlp/` vào .gitignore.

Quy ước đầu ra: các âm tiết của một từ nối bằng dấu "_" (vd "Hà_Nội"),
đúng định dạng PhoBERT yêu cầu.

CHIẾU NHÃN BIO khi tách từ mức âm tiết -> mức từ (xem project_tags):
  * Nhãn của từ = nhãn của token đầu tiên mà từ phủ lên.
  * Từ bắt đầu GIỮA một token (VnCoreNLP tách "TP.HCM" thành "TP." + "HCM")
    thì B-X đổi thành I-X.
  * Từ vắt qua biên thực thể (vd từ gồm 1 âm tiết trong thực thể + 1 âm tiết ngoài)
    vẫn lấy nhãn token đầu, và được ĐẾM vào thống kê "conflict" để báo cáo.

Dòng lệnh:
    python -m src.common.word_segmenter --input data/processed/vlsp/syllable/train.jsonl \
                                        --output data/processed/vlsp/word/train.jsonl
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from src.common.schema import NERSample, load_jsonl

logger = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
_REQUIRED_FILES = [
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
]
DEFAULT_SAVE_DIR = Path(os.environ.get(
    "VNCORENLP_DIR", Path(__file__).resolve().parents[2] / "resources" / "vncorenlp"))


class AlignmentError(ValueError):
    """Đầu ra của bộ tách từ không khớp ký tự với các token đầu vào."""


# --------------------------------------------------------------------------- #
# 1. Chiếu nhãn BIO từ token gốc sang từ đã tách (thuần Python, không cần Java)
# --------------------------------------------------------------------------- #
def _clean(s: str) -> str:
    """Bỏ khoảng trắng và '_' để so khớp ký tự giữa token gốc và từ đã tách."""
    return re.sub(r"[\s_]+", "", s)


def _char_spans(pieces: Sequence[str]) -> List[Tuple[int, int]]:
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans


def _entity_type(tag: str) -> str:
    return "" if tag == "O" else tag[2:]


def project_tags(tokens: Sequence[str], tags: Sequence[str],
                 words: Sequence[str]) -> Tuple[List[str], int]:
    """
    tokens/tags: câu gốc (vd mức âm tiết) và nhãn BIO của nó.
    words: các từ do bộ tách từ trả về (có thể chứa '_').
    Trả về (nhãn cho từng từ, số từ bị vắt qua biên thực thể).
    """
    if len(tokens) != len(tags):
        raise ValueError("len(tokens) != len(tags)")
    clean_tokens = [_clean(t) for t in tokens]
    clean_words = [_clean(w) for w in words]
    if any(not t for t in clean_tokens) or any(not w for w in clean_words):
        raise AlignmentError("Có token/từ rỗng sau khi bỏ khoảng trắng và '_'")
    if "".join(clean_tokens) != "".join(clean_words):
        raise AlignmentError(
            "Ký tự đầu ra không khớp đầu vào: "
            f"{''.join(clean_tokens)[:60]!r} != {''.join(clean_words)[:60]!r}")

    tok_spans = _char_spans(clean_tokens)
    word_spans = _char_spans(clean_words)

    word_tags: List[str] = []
    conflicts = 0
    ti = 0
    for ws, we in word_spans:
        while tok_spans[ti][1] <= ws:           # token chứa ký tự đầu của từ
            ti += 1
        tag = tags[ti]
        if ws > tok_spans[ti][0] and tag.startswith("B-"):   # từ bắt đầu giữa token
            tag = "I-" + tag[2:]
        word_tags.append(tag)

        # phát hiện từ vắt qua biên thực thể: các token tiếp theo mà từ phủ lên
        # phải là cùng một thực thể (I-X) hoặc cùng là "O"
        first_type = _entity_type(tags[ti])
        tj = ti
        crossed = False
        while tj + 1 < len(tok_spans) and tok_spans[tj + 1][0] < we:
            tj += 1
            nxt = tags[tj]
            same = (nxt == "O") if first_type == "" else (nxt == f"I-{first_type}")
            if not same:
                crossed = True
        conflicts += crossed
    return word_tags, conflicts


# --------------------------------------------------------------------------- #
# 2. Bộ tách từ VnCoreNLP
# --------------------------------------------------------------------------- #
def _ensure_model(save_dir: Path) -> None:
    """Tải jar + mô hình tách từ nếu chưa có (không dùng download_model của py_vncorenlp
    vì nó gọi wget, tải cả mô hình POS/NER/parse không cần thiết và lỗi nếu thư mục chưa tồn tại)."""
    for rel in _REQUIRED_FILES:
        dest = save_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Đang tải %s ...", rel)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(_BASE_URL + rel, tmp)
        tmp.replace(dest)


@dataclass
class SegmentationStats:
    n_samples: int = 0
    n_failed: int = 0                 # câu không căn chỉnh được (AlignmentError)
    n_samples_with_conflict: int = 0  # câu có >= 1 từ vắt qua biên thực thể
    n_words_with_conflict: int = 0
    failed_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.n_samples} câu | lỗi căn chỉnh: {self.n_failed} | "
                f"câu có từ vắt qua biên thực thể: {self.n_samples_with_conflict} "
                f"({self.n_words_with_conflict} từ)")


class WordSegmenter:
    """
    Bọc VnCoreNLP RDRSegmenter. Nạp mô hình lười (lần gọi đầu tiên).
    segment_fn: chỉ dùng để kiểm thử; nhận 1 chuỗi, trả về list câu đã tách (từ nối bằng '_').
    """

    def __init__(self, save_dir: Optional[str] = None,
                 segment_fn: Optional[Callable[[str], List[str]]] = None,
                 max_heap_size: str = "-Xmx2g"):
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        self.max_heap_size = max_heap_size
        self._segment_fn = segment_fn

    def _load(self) -> Callable[[str], List[str]]:
        if self._segment_fn is None:
            if "JAVA_HOME" not in os.environ and os.name == "nt":
                for java_dir in [
                    Path(r"C:\Program Files\Java\jdk-26.0.2"),
                    Path(r"C:\Program Files\Java\latest"),
                    Path(r"C:\Program Files\Java"),
                ]:
                    if java_dir.exists():
                        if (java_dir / "bin" / "java.exe").exists():
                            os.environ["JAVA_HOME"] = str(java_dir)
                            break
                        for sub in java_dir.glob("jdk*"):
                            if (sub / "bin" / "java.exe").exists():
                                os.environ["JAVA_HOME"] = str(sub)
                                break
            import py_vncorenlp  # import muộn để module vẫn import được khi chưa cài
            save_dir = self.save_dir.resolve()
            _ensure_model(save_dir)
            # py_vncorenlp âm thầm os.chdir(save_dir) khi khởi tạo -> làm hỏng mọi đường dẫn
            # tương đối của dự án (data/, configs/...). Khôi phục lại ngay sau khi nạp mô hình.
            cwd = os.getcwd()
            try:
                model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir),
                                               max_heap_size=self.max_heap_size)
            finally:
                os.chdir(cwd)
            self._segment_fn = model.word_segment
        return self._segment_fn

    # --- API ----------------------------------------------------------------
    def segment_text(self, text: str) -> List[str]:
        """Câu thô -> danh sách từ (âm tiết nối bằng '_'). Dùng cho suy luận/demo."""
        sentences = self._load()(text)
        return " ".join(sentences).split()

    def segment_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Danh sách âm tiết -> danh sách từ."""
        return self.segment_text(" ".join(tokens))

    def segment_sample(self, sample: NERSample) -> Tuple[NERSample, int]:
        """
        Mẫu mức âm tiết -> mẫu mức từ (id đổi '_syllable_' -> '_word_').
        Trả về (mẫu mới, số từ vắt qua biên thực thể). Ném AlignmentError nếu không căn chỉnh được.
        """
        if sample.granularity != "syllable":
            raise ValueError(f"Sample {sample.id}: cần granularity='syllable', "
                             f"nhận {sample.granularity!r}")
        words = self.segment_tokens(sample.tokens)
        word_tags, conflicts = project_tags(sample.tokens, sample.tags, words)
        new = NERSample(id=sample.id.replace("_syllable_", "_word_", 1),
                        dataset=sample.dataset, granularity="word",
                        tokens=words, tags=word_tags)
        new.validate()
        return new, conflicts


_default_segmenter: Optional[WordSegmenter] = None


def get_segmenter(save_dir: Optional[str] = None) -> WordSegmenter:
    """Singleton: JVM chỉ khởi động được MỘT lần trong mỗi tiến trình Python."""
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = WordSegmenter(save_dir=save_dir)
    return _default_segmenter


# --------------------------------------------------------------------------- #
# 3. Chuyển cả file JSONL: syllable -> word
# --------------------------------------------------------------------------- #
def convert_file(input_path: str, output_path: str, on_error: str = "raise",
                 segmenter: Optional[WordSegmenter] = None) -> SegmentationStats:
    """
    on_error="raise": dừng ở câu lỗi đầu tiên (mặc định - an toàn cho tập test).
    on_error="skip" : bỏ câu lỗi, ghi id ra <output>.failed_ids.txt và cảnh báo.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error phải là 'raise' hoặc 'skip'")
    seg = segmenter or get_segmenter()
    stats = SegmentationStats()
    out = []
    for sample in load_jsonl(input_path):
        stats.n_samples += 1
        try:
            new, conflicts = seg.segment_sample(sample)
        except AlignmentError as e:
            if on_error == "raise":
                raise AlignmentError(f"{sample.id}: {e}") from e
            stats.n_failed += 1
            stats.failed_ids.append(sample.id)
            continue
        if conflicts:
            stats.n_samples_with_conflict += 1
            stats.n_words_with_conflict += conflicts
        out.append(new)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in out:
            f.write(s.to_json_line() + "\n")
    if stats.failed_ids:
        Path(output_path + ".failed_ids.txt").write_text("\n".join(stats.failed_ids), encoding="utf-8")
        logger.warning("Bỏ qua %d câu không căn chỉnh được (xem %s.failed_ids.txt)",
                       stats.n_failed, output_path)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Tách từ VnCoreNLP: JSONL mức âm tiết -> mức từ")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--on-error", choices=["raise", "skip"], default="raise")
    ap.add_argument("--save-dir", default=None, help="thư mục chứa VnCoreNLP (mặc định resources/vncorenlp)")
    args = ap.parse_args()
    stats = convert_file(args.input, args.output, on_error=args.on_error,
                         segmenter=get_segmenter(args.save_dir))
    print(stats)


if __name__ == "__main__":
    main()
