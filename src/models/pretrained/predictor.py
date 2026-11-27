"""
predictor.py - Đóng gói PhoBERT và XLM-RoBERTa thành giao diện suy luận thống nhất cho Demo (TV3) & Eval.

Tuân thủ:
  * Hợp đồng dữ liệu chung (docs/pretrained_data_contract.md) và bộ nhãn chuẩn (src/common/schema.py).
  * Giao diện chính cho demo/app.py:
        predict(sentence: str) -> List[Dict[str, Any]]
    Mỗi phần tử Dict biểu diễn 1 thực thể theo đúng quy ước đã chốt với Thành viên 3:
        {
            "text":  str,  # Chuỗi gốc khớp chính xác sentence[start:end]
            "type":  str,  # Loại thực thể thuộc schema.py (PER/LOC/ORG/MISC hoặc 10 loại PhoNER)
            "start": int,  # Vị trí ký tự bắt đầu trong sentence (0-based, inclusive)
            "end":   int,  # Vị trí ký tự kết thúc trong sentence (0-based, exclusive)
        }
  * Giao diện cho luồng đánh giá & phân tích lỗi:
        predict_tokens(tokens: List[str]) -> List[str]
        predict_sample(sample: NERSample) -> NERSample
        predict_detailed(sentence: str) -> Dict[str, Any]

Đặc tính kỹ thuật:
  * Nạp model, tokenizer, label mapping một lần duy nhất; tuyệt đối không tải lại ở mỗi lần predict.
  * Tự động chọn CUDA/CPU (tự fallback sang CPU an toàn nếu CUDA không khả dụng).
  * Kiểm tra tính toàn vẹn checkpoint và tính tương thích model-config-schema trước khi nạp weights.
  * Suy luận với model.eval() + torch.inference_mode() + AMP trên GPU; không giữ computation graph.
  * Hỗ trợ câu dài vượt quá max_seq_len bằng chia cửa sổ trượt (sliding-window chunking) không cắt bỏ thực thể.
  * Ánh xạ ngược tọa độ ký tự tuần tự chính xác 100% (xử lý được thực thể lặp lại, dấu câu, từ ghép '_').
  * Xử lý an toàn mọi đầu vào bất thường (rỗng, sai kiểu, quá dài, thiếu VnCoreNLP) không làm sập demo.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer

from src.common.metrics import extract_entities
from src.common.schema import (
    ALLOWED_DATASETS,
    ALLOWED_GRANULARITIES,
    NERSample,
    TAG_SETS,
)
from src.common.word_segmenter import WordSegmenter, get_segmenter
from src.models.pretrained.model import PretrainedNERModel
from src.models.pretrained.subword_align import (
    recover_word_predictions,
    tokenize_and_get_word_ids,
)

logger = logging.getLogger(__name__)

# Danh mục checkpoint gợi ý sẵn để TV3 dễ tích hợp vào demo/app.py
PRESET_CHECKPOINTS: Dict[str, str] = {
    "phobert_vlsp_best": "outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt",
    "phobert_vlsp_baseline": "outputs/checkpoints/pretrained/exp01_phobert_baseline/best_model.pt",
    "xlmr_vlsp_syllable": "outputs/checkpoints/pretrained/exp02_xlmr_baseline/best_model.pt",
    "phobert_phoner_word": "outputs/checkpoints/pretrained/exp15_phobert_phoner_word/best_model.pt",
    "xlmr_phoner_syllable": "outputs/checkpoints/pretrained/exp17_xlmr_phoner_syllable/best_model.pt",
}

# Tách dấu câu đầu/cuối âm tiết cho mức syllable (bảo toàn 22/6, TP.HCM, 28-9-2004, H.)
_PUNCT_EDGE_RE = re.compile(
    r"""^([(\[{"'“‘«]+)?(.*?)([)\]},"';:!?…”’»]+)?$"""
)


def _tokenize_syllables(raw_text: str) -> List[str]:
    """
    Tách chuỗi văn bản thô thành danh sách âm tiết và dấu câu tách rời theo chuẩn VLSP/PhoNER.
    Giữ nguyên các cụm có dấu nội bộ như '22/6', '28-9-2004', 'TP.HCM', '3.5', 'H.'.
    """
    tokens: List[str] = []
    for piece in raw_text.strip().split():
        if not piece:
            continue
        # Tách riêng dấu mở đầu và dấu kết thúc (nhưng bảo toàn viết tắt kết thúc bằng 1 dấu chấm như 'H.', 'TP.')
        m = _PUNCT_EDGE_RE.match(piece)
        if not m:
            tokens.append(piece)
            continue
        lead, core, trail = m.group(1) or "", m.group(2) or "", m.group(3) or ""
        if lead:
            tokens.extend(list(lead))
        if core:
            # Nếu core kết thúc bằng '.' (vd 'Nam.' vs 'TP.'), tách '.' nếu trước nó dài >= 2 ký tự không viết hoa toàn bộ
            if core.endswith(".") and len(core) > 1 and not core.endswith(".."):
                stem = core[:-1]
                # Giữ viết tắt 1 chữ cái hoặc chữ viết hoa ngắn như 'H.', 'T.T.B.', 'TP.'
                if len(stem) <= 2 and stem.isupper() or "." in stem:
                    tokens.append(core)
                else:
                    tokens.append(stem)
                    tokens.append(".")
            else:
                tokens.append(core)
        if trail:
            tokens.extend(list(trail))
    return tokens


def _align_tokens_to_raw_text(raw_text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Ánh xạ tuần tự từng token (có thể chứa '_' từ VnCoreNLP) sang cặp tọa độ ký tự
    (start, end) chính xác trong raw_text.

    Đảm bảo:
      - Xử lý đúng khi cùng một thực thể xuất hiện nhiều lần trong câu.
      - Xử lý đúng khi từ ghép chứa '_' (vd 'Hà_Nội' khớp 'Hà Nội' trong raw_text).
      - Xử lý đúng các cụm dấu câu sát nhau như 'TP.HCM' hoặc '(Hà Nội)'.
    """
    spans: List[Tuple[int, int]] = []
    n = len(raw_text)
    cursor = 0

    for tok in tokens:
        # Bỏ khoảng trắng hiện tại
        while cursor < n and raw_text[cursor].isspace():
            cursor += 1

        clean_chars = [ch for ch in tok if not ch.isspace() and ch != "_"]
        if not clean_chars:
            spans.append((cursor, cursor))
            continue

        # Khớp tuần tự từng ký tự của token với raw_text kể từ cursor
        scan = cursor
        tok_start = -1
        matched_all = True
        for ch in clean_chars:
            while scan < n and (raw_text[scan].isspace() or raw_text[scan] == "_"):
                scan += 1
            if scan < n and raw_text[scan] == ch:
                if tok_start == -1:
                    tok_start = scan
                scan += 1
            else:
                # Thử tìm kiếm tiến trong cửa sổ nhỏ (trường hợp chuẩn hóa dấu câu nhẹ của bộ tách từ)
                look = raw_text.find(ch, scan, min(n, scan + 8))
                if look != -1:
                    if tok_start == -1:
                        tok_start = look
                    scan = look + 1
                else:
                    matched_all = False
                    break

        if matched_all and tok_start != -1:
            spans.append((tok_start, scan))
            cursor = scan
        else:
            # Fallback tìm theo chuỗi bỏ '_' kể từ cursor
            plain = tok.replace("_", " ")
            idx = raw_text.find(plain, cursor)
            if idx != -1:
                spans.append((idx, idx + len(plain)))
                cursor = idx + len(plain)
            else:
                spans.append((cursor, min(n, cursor + len(plain))))

    return spans


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Chọn thiết bị suy luận an toàn:
      - None / "auto" -> tự động chọn CUDA nếu khả dụng, ngược lại CPU.
      - Nếu yêu cầu CUDA nhưng máy không có CUDA -> cảnh báo và tự fallback sang CPU.
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng trên hệ thống này; tự động chuyển sang CPU.")
        return torch.device("cpu")
    return requested


def _validate_checkpoint_dict(ckpt: Any, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Kiểm định toàn vẹn cấu trúc checkpoint và tính tương thích model-config-schema."""
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint hỏng hoặc sai định dạng (không phải dict): {checkpoint_path}")

    required_keys = ("config", "id2tag", "model_state_dict")
    missing = [k for k in required_keys if k not in ckpt or ckpt[k] is None]
    if missing:
        raise ValueError(f"Checkpoint thiếu trường bắt buộc {missing}: {checkpoint_path}")

    cfg = ckpt["config"]
    if not isinstance(cfg, dict):
        raise ValueError(f"Trường 'config' trong checkpoint không hợp lệ: {checkpoint_path}")

    raw_id2tag = ckpt["id2tag"]
    if not isinstance(raw_id2tag, dict) or len(raw_id2tag) == 0:
        raise ValueError(f"Trường 'id2tag' trong checkpoint rỗng hoặc sai kiểu: {checkpoint_path}")

    id2tag = {int(k): str(v) for k, v in raw_id2tag.items()}
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    dataset = str(data_cfg.get("dataset", "vlsp"))
    granularity = str(data_cfg.get("granularity", "word"))
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Dataset '{dataset}' không hợp lệ (hợp lệ: {ALLOWED_DATASETS})")
    if granularity not in ALLOWED_GRANULARITIES:
        raise ValueError(f"Granularity '{granularity}' không hợp lệ (hợp lệ: {ALLOWED_GRANULARITIES})")

    # Kiểm tra tương thích bộ nhãn với src/common/schema.py
    schema_tags = TAG_SETS.get(dataset)
    if schema_tags is not None:
        unknown_tags = set(id2tag.values()) - set(schema_tags)
        if unknown_tags:
            raise ValueError(
                f"Nhãn trong checkpoint {sorted(unknown_tags)} không thuộc schema TAG_SETS['{dataset}']!"
            )

    # Kiểm tra tương thích số nhãn & pooling strategy giữa config và model_state_dict
    pooling = str(model_cfg.get("pooling_strategy", "last")).lower()
    if pooling not in PretrainedNERModel.VALID_POOLING_STRATEGIES:
        raise ValueError(
            f"pooling_strategy '{pooling}' không hợp lệ (hợp lệ: {PretrainedNERModel.VALID_POOLING_STRATEGIES})"
        )

    state_dict = ckpt["model_state_dict"]
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise ValueError(f"model_state_dict rỗng hoặc hỏng trong checkpoint: {checkpoint_path}")

    clf_weight = state_dict.get("classifier.weight")
    if clf_weight is not None and clf_weight.shape[0] != len(id2tag):
        raise ValueError(
            f"Không khớp kích thước classifier: state_dict có {clf_weight.shape[0]} nhãn "
            f"nhưng id2tag có {len(id2tag)} nhãn!"
        )

    return ckpt


class PLMPredictor:
    """
    Wrapper thống nhất cho PhoBERT và XLM-RoBERTa phục vụ tích hợp Demo (TV3) & Đánh giá.

    Cách dùng khuyến nghị cho Thành viên 3:
        >>> from src.models.pretrained.predictor import create_predictor
        >>> predictor = create_predictor("outputs/checkpoints/pretrained/exp10_phobert_pooling_concat4/best_model.pt")
        >>> predictor.predict("Ông Nguyễn Văn A làm việc tại Bệnh viện Bạch Mai, Hà Nội.")
        [
            {"text": "Nguyễn Văn A", "type": "PER", "start": 4, "end": 16},
            {"text": "Bệnh viện Bạch Mai", "type": "LOC", "start": 30, "end": 48},
            {"text": "Hà Nội", "type": "LOC", "start": 50, "end": 56},
        ]
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        segmenter: Optional[WordSegmenter] = None,
        max_seq_len: Optional[int] = None,
        use_amp: bool = True,
        *,
        model: Optional[PretrainedNERModel] = None,
        tokenizer: Optional[Any] = None,
        id2tag: Optional[Dict[int, str]] = None,
        dataset: str = "vlsp",
        granularity: str = "word",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo PLMPredictor theo một trong 2 cách:
          1. Truyền `checkpoint_path` (nạp checkpoint .pt + tokenizer + label map).
          2. Truyền trực tiếp `model`, `tokenizer`, `id2tag` đã có sẵn trong bộ nhớ (không tải lại).
        """
        self.device = _resolve_device(device)
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self._segmenter = segmenter
        self.checkpoint_path: Optional[Path] = Path(checkpoint_path) if checkpoint_path is not None else None

        self.model: Optional[PretrainedNERModel] = None
        self.tokenizer: Optional[Any] = None
        self.id2tag: Dict[int, str] = {}
        self.tag2id: Dict[str, int] = {}
        self.config: Dict[str, Any] = {}
        self.dataset: str = dataset
        self.granularity: str = granularity
        self.model_name: str = ""
        self.pooling_strategy: str = "last"
        self.max_seq_len: int = int(max_seq_len) if max_seq_len is not None else 256

        if model is not None:
            if tokenizer is None or not id2tag:
                raise ValueError("Khi truyền `model` trực tiếp, bắt buộc phải truyền kèm `tokenizer` và `id2tag`!")
            self._init_from_components(
                model=model,
                tokenizer=tokenizer,
                id2tag=id2tag,
                dataset=dataset,
                granularity=granularity,
                config=config,
                max_seq_len=max_seq_len,
            )
        elif checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, max_seq_len=max_seq_len)
        else:
            # Cho phép khởi tạo rỗng để nạp sau bằng load_checkpoint(), nhưng predict() sẽ báo rõ nếu chưa nạp
            logger.info("Khởi tạo PLMPredictor chưa nạp model. Hãy gọi load_checkpoint() trước khi predict().")

    def _init_from_components(
        self,
        model: PretrainedNERModel,
        tokenizer: Any,
        id2tag: Dict[int, str],
        dataset: str = "vlsp",
        granularity: str = "word",
        config: Optional[Dict[str, Any]] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        if dataset not in ALLOWED_DATASETS:
            raise ValueError(f"dataset không hợp lệ: {dataset!r}")
        if granularity not in ALLOWED_GRANULARITIES:
            raise ValueError(f"granularity không hợp lệ: {granularity!r}")

        self.id2tag = {int(k): str(v) for k, v in id2tag.items()}
        self.tag2id = {v: k for k, v in self.id2tag.items()}
        self.dataset = dataset
        self.granularity = granularity
        self.config = config or {}
        self.tokenizer = tokenizer
        self.model_name = str(getattr(model, "model_name_or_path", type(model).__name__))
        self.pooling_strategy = str(getattr(model, "pooling_strategy", "last"))

        cfg_max_len = int(self.config.get("data", {}).get("max_seq_len", 256))
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else cfg_max_len
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained_components(
        cls,
        model: PretrainedNERModel,
        tokenizer: Any,
        id2tag: Dict[int, str],
        dataset: str = "vlsp",
        granularity: str = "word",
        device: Optional[Union[str, torch.device]] = None,
        segmenter: Optional[WordSegmenter] = None,
        max_seq_len: int = 256,
    ) -> "PLMPredictor":
        """Tạo PLMPredictor từ model và tokenizer đã nạp sẵn (tránh nạp trùng lặp)."""
        return cls(
            checkpoint_path=None,
            device=device,
            segmenter=segmenter,
            max_seq_len=max_seq_len,
            model=model,
            tokenizer=tokenizer,
            id2tag=id2tag,
            dataset=dataset,
            granularity=granularity,
        )

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        max_seq_len: Optional[int] = None,
    ) -> "PLMPredictor":
        """Nạp checkpoint .pt, kiểm tra tương thích config/schema, nạp tokenizer và chuyển model sang eval()."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists() or not ckpt_file.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy tệp checkpoint: '{ckpt_file}'. "
                f"Kiểm tra lại đường dẫn hoặc chạy huấn luyện trước khi suy luận."
            )

        try:
            try:
                ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_file, map_location=self.device)
        except Exception as exc:
            raise ValueError(f"Không thể đọc tệp checkpoint bị hỏng '{ckpt_file}': {exc}") from exc

        ckpt = _validate_checkpoint_dict(ckpt, ckpt_file)
        cfg = ckpt["config"]
        id2tag = {int(k): str(v) for k, v in ckpt["id2tag"].items()}

        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        model_name = str(model_cfg.get("model_name_or_path", "vinai/phobert-base-v2"))
        pooling = str(model_cfg.get("pooling_strategy", "last")).lower()
        dataset = str(data_cfg.get("dataset", "vlsp"))
        granularity = str(data_cfg.get("granularity", "word"))

        # Nạp tokenizer (ưu tiên thư mục chứa checkpoint nếu đã lưu kèm, sau đó cache local, cuối cùng hub)
        use_fast = False if "phobert" in model_name.lower() else True
        tokenizer = self._load_tokenizer(model_name, ckpt_file.parent, use_fast=use_fast)

        # Khởi tạo kiến trúc tương thích và nạp trọng số (hỗ trợ cả fp32 lẫn fp16 state_dict)
        model = PretrainedNERModel(
            model_name_or_path=model_name,
            num_labels=len(id2tag),
            pooling_strategy=pooling,
            classifier_dropout=0.0,
        )
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Trọng số trong checkpoint '{ckpt_file}' không tương thích với kiến trúc "
                f"({model_name}, pooling={pooling}, num_labels={len(id2tag)}): {exc}"
            ) from exc

        self.checkpoint_path = ckpt_file
        self._init_from_components(
            model=model,
            tokenizer=tokenizer,
            id2tag=id2tag,
            dataset=dataset,
            granularity=granularity,
            config=cfg,
            max_seq_len=max_seq_len,
        )
        return self

    @staticmethod
    def _load_tokenizer(model_name: str, ckpt_dir: Path, use_fast: bool) -> Any:
        # 1. Thử nạp từ thư mục checkpoint (nếu tokenizer được lưu kèm khi train)
        if (ckpt_dir / "tokenizer_config.json").exists():
            try:
                return AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=use_fast, local_files_only=True)
            except Exception:
                pass
        # 2. Thử nạp từ cache HuggingFace local
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, local_files_only=True)
        except Exception:
            # 3. Fallback tải trực tuyến nếu chưa có cache
            return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and len(self.id2tag) > 0

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "PLMPredictor chưa được nạp mô hình! Hãy truyền `checkpoint_path` khi khởi tạo "
                "hoặc gọi `predictor.load_checkpoint(path)` trước khi suy luận."
            )

    def _get_segmenter(self) -> WordSegmenter:
        """Khởi tạo lười hoặc dùng chung singleton WordSegmenter để tiết kiệm RAM/JVM."""
        if self._segmenter is None:
            self._segmenter = get_segmenter()
        return self._segmenter

    def _preprocess_raw_text(self, raw_text: str) -> List[str]:
        """Tiền xử lý câu tiếng Việt thô đúng theo mức biểu diễn (word hoặc syllable) của mô hình."""
        text = raw_text.strip()
        if not text:
            return []

        if self.granularity == "word":
            try:
                seg = self._get_segmenter()
                tokens = seg.segment_text(text)
                if tokens:
                    return tokens
            except Exception as exc:
                logger.warning(
                    "Không thể chạy VnCoreNLP WordSegmenter (%s); fallback sang tách âm tiết/dấu câu an toàn.",
                    exc,
                )
            return _tokenize_syllables(text)

        return _tokenize_syllables(text)

    @torch.inference_mode()
    def _infer_window(self, tokens: List[str]) -> List[str]:
        """Suy luận một cửa sổ tokens đảm bảo vừa trong max_seq_len subwords."""
        input_ids_list, attention_mask_list, word_ids = tokenize_and_get_word_ids(
            tokenizer=self.tokenizer,
            tokens=tokens,
            max_length=self.max_seq_len,
        )
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)          # [1, L]
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long, device=self.device)  # [1, L]

        self.model.eval()
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
            )
            logits = outputs["logits"]  # [1, L, num_labels]

        pred_ids = torch.argmax(logits[0], dim=-1).cpu().tolist()  # [L]
        return recover_word_predictions(
            pred_ids=pred_ids,
            word_ids=word_ids,
            num_tokens=len(tokens),
            id2tag=self.id2tag,
            fallback_tag="O",
        )

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        """
        Dự đoán chuỗi nhãn BIO cho danh sách token gốc (luồng eval / phân tích lỗi).
        Tự động chia cửa sổ (chunking) khi câu quá dài vượt quá `max_seq_len` subwords
        để không bao giờ lỗi vượt vị trí embedding và không bỏ sót thực thể cuối câu.
        """
        self._ensure_loaded()
        if tokens is None:
            return []
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(f"tokens phải là list/tuple các chuỗi, nhận {type(tokens).__name__}")
        tok_list = [str(t) for t in tokens if str(t) != ""]
        if not tok_list:
            return []

        # Thử tokenize toàn câu 1 lần để kiểm tra có vượt max_seq_len hay không
        _, _, word_ids = tokenize_and_get_word_ids(
            tokenizer=self.tokenizer,
            tokens=tok_list,
            max_length=self.max_seq_len,
        )
        covered_ids = {w for w in word_ids if w is not None}
        # Nếu toàn bộ tokens nằm gọn trong 1 cửa sổ (hoặc chỉ khuyết ký tự zero-width như \u200b), suy luận 1 lượt
        if len(word_ids) < self.max_seq_len or (len(tok_list) - 1) in covered_ids:
            return self._infer_window(tok_list)

        # Câu quá dài (> max_seq_len subwords): chia cửa sổ theo bước an toàn (có overlap ngữ cảnh 16 tokens)
        all_tags: List[str] = ["O"] * len(tok_list)
        start = 0
        overlap = 16
        # Dự phòng kích thước cửa sổ ban đầu an toàn
        while start < len(tok_list):
            # Tìm điểm kết thúc lớn nhất sao cho không bị truncate
            sub_tokens = tok_list[start:]
            _, _, wids = tokenize_and_get_word_ids(
                tokenizer=self.tokenizer,
                tokens=sub_tokens,
                max_length=self.max_seq_len,
            )
            valid_wids = [w for w in wids if w is not None]
            if not valid_wids:
                # Token đơn lẻ cực dài vượt quá max_seq_len -> xử lý riêng 1 token
                all_tags[start] = self._infer_window([tok_list[start]])[0]
                start += 1
                continue

            max_local_wid = max(valid_wids)
            is_last_chunk = (start + max_local_wid + 1 >= len(tok_list))
            # Nếu chưa phải chunk cuối và có đủ tokens, lùi lại 1 token tránh cắt ngang subword cuối cửa sổ
            chunk_len = (max_local_wid + 1) if is_last_chunk else max(1, max_local_wid)
            chunk_tokens = tok_list[start : start + chunk_len]
            chunk_preds = self._infer_window(chunk_tokens)

            # Ghi kết quả: ở vùng overlap (start > 0), giữ nửa đầu của chunk trước và nửa sau của chunk mới
            write_offset = (overlap // 2) if (start > 0 and chunk_len > overlap) else 0
            for i in range(write_offset, chunk_len):
                all_tags[start + i] = chunk_preds[i]

            if is_last_chunk:
                break
            step = max(1, chunk_len - overlap)
            start += step

        return all_tags

    def predict_sample(self, sample: NERSample) -> NERSample:
        """Nhận một NERSample chuẩn schema.py và trả về NERSample chứa nhãn dự đoán."""
        self._ensure_loaded()
        pred_tags = self.predict_tokens(sample.tokens)
        out = NERSample(
            id=sample.id,
            dataset=sample.dataset,
            granularity=sample.granularity,
            tokens=list(sample.tokens),
            tags=pred_tags,
        )
        out.validate()
        return out

    def predict_detailed(self, sentence: str, strict: bool = False) -> Dict[str, Any]:
        """
        Dự đoán đầy đủ cho câu tiếng Việt thô, trả về cả tokens, nhãn BIO và danh sách thực thể chuẩn.
        """
        self._ensure_loaded()
        if not isinstance(sentence, str):
            msg = f"Đầu vào sentence phải là kiểu str, nhận {type(sentence).__name__}."
            if strict:
                raise TypeError(msg)
            logger.warning("%s Trả về kết quả rỗng để tránh làm sập demo.", msg)
            return {
                "text": "",
                "tokens": [],
                "tags": [],
                "entities": [],
                "model_name": self.model_name,
                "dataset": self.dataset,
                "granularity": self.granularity,
                "device": str(self.device),
            }

        if not sentence.strip():
            return {
                "text": sentence,
                "tokens": [],
                "tags": [],
                "entities": [],
                "model_name": self.model_name,
                "dataset": self.dataset,
                "granularity": self.granularity,
                "device": str(self.device),
            }

        tokens = self._preprocess_raw_text(sentence)
        tags = self.predict_tokens(tokens)
        tok_spans = _align_tokens_to_raw_text(sentence, tokens)

        # Trích xuất thực thể bằng đúng quy ước BIO của src/common/metrics.py (extract_entities)
        spans_sorted = sorted(extract_entities(tags), key=lambda x: (x[0], x[1]))
        entities: List[Dict[str, Any]] = []
        for s_idx, e_idx, etype in spans_sorted:
            char_start = tok_spans[s_idx][0]
            char_end = tok_spans[e_idx - 1][1]
            ent_text = sentence[char_start:char_end] if (0 <= char_start < char_end <= len(sentence)) else " ".join(
                t.replace("_", " ") for t in tokens[s_idx:e_idx]
            )
            entities.append({
                "text": ent_text,
                "type": etype,
                "start": int(char_start),
                "end": int(char_end),
            })

        return {
            "text": sentence,
            "tokens": tokens,
            "tags": tags,
            "entities": entities,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "granularity": self.granularity,
            "device": str(self.device),
        }

    def predict(self, sentence: str, strict: bool = False) -> List[Dict[str, Any]]:
        """
        Giao diện dự đoán thống nhất cho demo/app.py (Thành viên 3):
            predict(sentence: str) -> List[Dict[str, Any]]

        Args:
            sentence: Chuỗi văn bản tiếng Việt thô.
            strict: Nếu True, ném TypeError khi đầu vào không phải str; mặc định False (trả về [] an toàn cho demo).

        Returns:
            Danh sách thực thể: [{"text": str, "type": str, "start": int, "end": int}, ...]
        """
        return self.predict_detailed(sentence, strict=strict)["entities"]

    def get_model_info(self) -> Dict[str, Any]:
        """Trả về metadata mô hình phục vụ hiển thị thông tin trên giao diện Demo."""
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "granularity": self.granularity,
            "pooling_strategy": self.pooling_strategy,
            "num_labels": len(self.id2tag),
            "entity_types": sorted({t[2:] for t in self.id2tag.values() if t != "O" and "-" in t}),
            "max_seq_len": self.max_seq_len,
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


def create_predictor(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    segmenter: Optional[WordSegmenter] = None,
    max_seq_len: Optional[int] = None,
) -> PLMPredictor:
    """Factory function khởi tạo PLMPredictor (hỗ trợ dùng chung WordSegmenter trong demo/app.py)."""
    return PLMPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        segmenter=segmenter,
        max_seq_len=max_seq_len,
    )


def load_predictor(
    name_or_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    segmenter: Optional[WordSegmenter] = None,
    max_seq_len: Optional[int] = None,
) -> PLMPredictor:
    """
    Hàm tiện ích cho TV3 nạp nhanh theo tên preset (vd: 'phobert_vlsp_best', 'xlmr_phoner_syllable')
    hoặc đường dẫn trực tiếp tới tệp .pt.
    """
    resolved = PRESET_CHECKPOINTS.get(str(name_or_path), str(name_or_path))
    return create_predictor(
        checkpoint_path=resolved,
        device=device,
        segmenter=segmenter,
        max_seq_len=max_seq_len,
    )
