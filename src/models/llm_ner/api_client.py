"""
api_client.py - Gọi Gemini API và đo thời gian suy luận (nhánh LLM - TV3).

Dùng gói chính thức `google-genai`:   pip install -U google-genai
Khóa API (miễn phí, tạo trong Google AI Studio) đặt ở biến môi trường:
    GEMINI_API_KEY   (hoặc GOOGLE_API_KEY)
Tên mô hình đặt bằng tham số `model=` hoặc biến môi trường GEMINI_MODEL. KHÔNG có giá trị
mặc định, vì các mô hình Flash đổi tên/ngừng cấp phát vài lần mỗi năm. Xem mô hình khả dụng:
    python -m src.models.llm_ner.api_client --list-models

Cách dùng:
    from src.models.llm_ner.api_client import GeminiClient, DailyQuotaExceeded

    client = GeminiClient(model="<tên mô hình>", thinking="low", rpm=10)
    try:
        for sent in sentences:
            r = client.generate(prompt_for(sent), system=SYSTEM_PROMPT, json_mode=True)
            if r.ok: entities = json.loads(r.text)
    except DailyQuotaExceeded:
        pass            # hết hạn mức NGÀY: chạy lại ngày mai, các câu đã xong nằm sẵn trong cache
    print(client.stats())   # thời gian suy luận trung bình, số token, số lần thử lại...

Thiết kế cho gói miễn phí (hạn mức thấp, hay gặp lỗi 429):
  * CACHE đĩa: mỗi (mô hình, prompt, tham số) chỉ tốn 1 lượt gọi; chạy lại/tiếp tục sau khi
    sập hay hết hạn mức không mất lượt đã làm. Cache lưu cả độ trễ gốc nên số liệu thời gian
    vẫn dùng được.
  * THỬ LẠI tự động với 429/5xx/lỗi mạng: chờ theo `retryDelay` do server báo, nếu không có
    thì lùi dần theo cấp số nhân. Hết hạn mức NGÀY thì dừng ngay (DailyQuotaExceeded).
  * `rpm=` giãn cách yêu cầu phía client để tránh chạm trần số yêu cầu/phút.
  * Độ trễ `latency_s` chỉ tính lần gọi thành công (không tính thời gian chờ thử lại/giãn
    cách); `wall_s` tính cả. Dùng `latency_s` để báo cáo thời gian suy luận.

LƯU Ý về tham số sinh (theo tài liệu Gemini):
  * Gemini 3: Google khuyến nghị giữ temperature mặc định -> để `temperature=None`.
    Gemini 2.5: có thể đặt temperature=0.
  * Mức suy luận: `thinking` là int (thinking_budget, họ 2.5; 0 = tắt với Flash) hoặc chuỗi
    "minimal"/"low"/"medium"/"high" (thinking_level, họ 3). Đặt sai họ sẽ bị lỗi 400.
    Gemini 3 mặc định suy luận mức cao -> chậm và tốn token; cho NER nên thử "low".
    Token suy luận tính vào max_output_tokens -> nếu bị cắt (finish_reason=MAX_TOKENS) hãy tăng
    max_output_tokens hoặc giảm mức suy luận.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.environ.get(
    "NER_LLM_CACHE_DIR", Path(__file__).resolve().parents[3] / "outputs" / "llm_cache"))


# --------------------------------------------------------------------------- #
# Lỗi và kết quả
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    """Lỗi không thể khắc phục bằng thử lại (sai khóa/tham số/tên mô hình) hoặc hết số lần thử."""


class DailyQuotaExceeded(LLMError):
    """Hết hạn mức theo NGÀY của gói miễn phí -> dừng, chạy lại sau (cache giữ kết quả đã có)."""


@dataclass
class LLMResponse:
    text: str
    finish_reason: str          # "STOP" là bình thường; "MAX_TOKENS", "SAFETY", "PROMPT_BLOCKED:..." là bất thường
    latency_s: float            # thời gian của lần gọi THÀNH CÔNG (dùng cho báo cáo thời gian suy luận)
    wall_s: float               # tổng thời gian kể cả chờ/thử lại
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    thinking_tokens: Optional[int]
    attempts: int               # số lần gọi (1 = không phải thử lại)
    cached: bool                # True: lấy từ cache (latency_s là độ trễ đo ở lần chạy gốc)
    model: str

    @property
    def ok(self) -> bool:
        """Có nội dung và kết thúc bình thường (không bị cắt/chặn)."""
        return bool(self.text.strip()) and self.finish_reason == "STOP"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Raw:
    text: str
    finish_reason: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    thinking_tokens: Optional[int]


# --------------------------------------------------------------------------- #
# Cache đĩa (JSONL, chỉ ghi thêm)
# --------------------------------------------------------------------------- #
def _stable(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr)
    except Exception:
        return repr(obj)


class _DiskCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec
                    except Exception:      # dòng cuối bị cụt do sập giữa chừng -> bỏ qua
                        continue

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, rec: Dict[str, Any]) -> None:
        rec = {"key": key, **rec}
        with self._lock:
            self._data[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Khung chung: giãn cách, thử lại, cache, thống kê (để sau này thêm Groq... chỉ cần
# viết _call và _classify_error)
# --------------------------------------------------------------------------- #
class BaseLLMClient:
    provider = "base"

    def __init__(self, model: str, rpm: Optional[float], max_retries: int, base_delay: float,
                 max_delay: float, cache_path: Optional[Union[str, Path]]):
        self.model = model
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._cache = _DiskCache(Path(cache_path)) if cache_path else None
        self._last_start = 0.0
        self._records: List[LLMResponse] = []
        self._sleep: Callable[[float], None] = time.sleep       # thay được khi kiểm thử
        self._clock: Callable[[], float] = time.monotonic

    # --- lớp con phải cài ---------------------------------------------------
    def _call(self, prompt: str, system: Optional[str], json_mode: bool,
              response_schema: Any, max_output_tokens: Optional[int]) -> _Raw:
        raise NotImplementedError

    def _classify_error(self, exc: Exception) -> Tuple[str, Optional[float]]:
        """-> ("retry"|"daily"|"fatal", số giây server yêu cầu chờ hoặc None)."""
        raise NotImplementedError

    def _cache_params(self) -> Dict[str, Any]:
        return {}

    # --- khung chung ----------------------------------------------------------
    def _throttle(self) -> None:
        if self.min_interval:
            wait = self._last_start + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()

    def _cache_key(self, prompt, system, json_mode, schema, max_tokens) -> str:
        payload = _stable({"provider": self.provider, "model": self.model, "prompt": prompt,
                           "system": system, "json": bool(json_mode or schema),
                           "schema": schema, "max_tokens": max_tokens, **self._cache_params()})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
                 response_schema: Any = None, max_output_tokens: Optional[int] = None,
                 use_cache: bool = True) -> LLMResponse:
        """
        json_mode / response_schema: yêu cầu đầu ra JSON hợp lệ (schema: dict JSON-Schema hoặc lớp pydantic).
        Kết quả bất thường (bị cắt, bị chặn, rỗng) vẫn được TRẢ VỀ với ok=False và KHÔNG được cache.
        Ném DailyQuotaExceeded / LLMError cho lỗi không thể thử lại.
        """
        key = self._cache_key(prompt, system, json_mode, response_schema, max_output_tokens)
        if use_cache and self._cache is not None:
            hit = self._cache.get(key)
            if hit:
                resp = LLMResponse(text=hit["text"], finish_reason=hit["finish_reason"],
                                   latency_s=hit["latency_s"], wall_s=0.0,
                                   input_tokens=hit.get("input_tokens"),
                                   output_tokens=hit.get("output_tokens"),
                                   thinking_tokens=hit.get("thinking_tokens"),
                                   attempts=0, cached=True, model=self.model)
                self._records.append(resp)
                return resp

        wall0 = time.perf_counter()
        attempt, raw, latency = 0, None, 0.0
        while True:
            attempt += 1
            self._throttle()
            t0 = time.perf_counter()
            try:
                raw = self._call(prompt, system, json_mode, response_schema, max_output_tokens)
                latency = time.perf_counter() - t0
                break
            except Exception as exc:
                kind, server_wait = self._classify_error(exc)
                if kind == "daily":
                    raise DailyQuotaExceeded(
                        f"Hết hạn mức NGÀY của {self.provider}/{self.model}. Chạy lại sau; kết quả "
                        f"đã có nằm trong cache. Chi tiết: {exc}") from exc
                if kind == "fatal" or attempt > self.max_retries:
                    raise LLMError(f"{self.provider}/{self.model} lỗi sau {attempt} lần gọi: "
                                   f"{type(exc).__name__}: {exc}") from exc
                delay = (max(server_wait, 1.0) if server_wait is not None
                         else min(self.max_delay, self.base_delay * 2 ** (attempt - 1)))
                delay = min(delay, self.max_delay) * (1 + 0.2 * random.random())
                log.warning("Lần %d lỗi (%s: %s) - chờ %.1fs rồi thử lại.",
                            attempt, type(exc).__name__, str(exc)[:120], delay)
                self._sleep(delay)

        resp = LLMResponse(text=raw.text, finish_reason=raw.finish_reason, latency_s=latency,
                           wall_s=time.perf_counter() - wall0, input_tokens=raw.input_tokens,
                           output_tokens=raw.output_tokens, thinking_tokens=raw.thinking_tokens,
                           attempts=attempt, cached=False, model=self.model)
        if resp.ok and self._cache is not None:
            self._cache.put(key, {"text": resp.text, "finish_reason": resp.finish_reason,
                                  "latency_s": resp.latency_s, "input_tokens": resp.input_tokens,
                                  "output_tokens": resp.output_tokens,
                                  "thinking_tokens": resp.thinking_tokens, "model": self.model,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        elif not resp.ok:
            log.warning("Phản hồi bất thường (finish_reason=%s, rỗng=%s) - không lưu cache.",
                        resp.finish_reason, not resp.text.strip())
        self._records.append(resp)
        return resp

    # --- thống kê -------------------------------------------------------------
    def reset_stats(self) -> None:
        self._records.clear()

    def stats(self) -> Dict[str, Any]:
        """Thống kê trên các phản hồi đã trả về kể từ lần reset_stats() gần nhất."""
        recs = self._records
        good = [r for r in recs if r.ok]
        lat = [r.latency_s for r in good]

        def total(attr):
            vals = [getattr(r, attr) for r in recs if getattr(r, attr) is not None]
            return sum(vals) if vals else None

        return {
            "model": self.model,
            "n_responses": len(recs),
            "n_ok": len(good),
            "n_bad": len(recs) - len(good),              # bị cắt/bị chặn/rỗng
            "n_from_cache": sum(r.cached for r in recs),
            "n_retries": sum(max(r.attempts - 1, 0) for r in recs),
            "mean_latency_s": statistics.fmean(lat) if lat else None,
            "median_latency_s": statistics.median(lat) if lat else None,
            "max_latency_s": max(lat) if lat else None,
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "thinking_tokens": total("thinking_tokens"),
        }


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"|retry in ([\d.]+)\s*s', re.I)
_DAILY_RE = re.compile(r"PerDay|per day|daily", re.I)


class GeminiClient(BaseLLMClient):
    provider = "gemini"

    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: Optional[float] = None,
                 seed: Optional[int] = None,
                 thinking: Union[None, int, str] = None,
                 max_output_tokens: int = 4096,
                 rpm: Optional[float] = None,
                 max_retries: int = 5,
                 base_delay: float = 2.0,
                 max_delay: float = 120.0,
                 timeout_s: float = 120.0,
                 cache: bool = True,
                 cache_path: Optional[Union[str, Path]] = None,
                 client: Any = None):
        """
        model      : tên mô hình (hoặc biến môi trường GEMINI_MODEL) - bắt buộc.
        temperature: None = mặc định của mô hình (khuyến nghị cho Gemini 3).
        seed       : cố định hạt giống sinh (tăng tính tái lập, không đảm bảo tuyệt đối).
        thinking   : None | int (thinking_budget, họ 2.5) | "minimal"/"low"/"medium"/"high" (họ 3).
        rpm        : giới hạn số yêu cầu/phút phía client (None = không giãn cách).
        cache_path : mặc định outputs/llm_cache/gemini_<model>.jsonl.
        client     : đối tượng genai.Client dựng sẵn (dùng khi kiểm thử).
        """
        model = model or os.environ.get("GEMINI_MODEL")
        if not model:
            raise ValueError(
                "Chưa chọn mô hình. Truyền model=... hoặc đặt biến môi trường GEMINI_MODEL. "
                "Xem danh sách khả dụng: python -m src.models.llm_ner.api_client --list-models")
        if isinstance(thinking, bool):
            raise ValueError("thinking phải là int (budget), chuỗi (level) hoặc None - không phải bool")
        self.temperature, self.seed = temperature, seed
        self.thinking, self.max_output_tokens = thinking, max_output_tokens

        if cache and cache_path is None:
            safe = re.sub(r"[^\w.\-]+", "_", model)
            cache_path = DEFAULT_CACHE_DIR / f"gemini_{safe}.jsonl"
        super().__init__(model, rpm, max_retries, base_delay, max_delay, cache_path if cache else None)

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("Thiếu thư viện: pip install -U google-genai") from e
        self._types = types
        self._thinking_config()            # báo lỗi sớm nếu `thinking` sai, thay vì lúc đã chạy dở
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Chưa có khóa API: đặt biến môi trường GEMINI_API_KEY "
                                 "(tạo miễn phí tại Google AI Studio).")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000)))          # đơn vị: mili-giây

    # --- cấu hình sinh --------------------------------------------------------
    def _cache_params(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "seed": self.seed, "thinking": self.thinking}

    def _thinking_config(self):
        t, types = self.thinking, self._types
        if t is None:
            return None
        if isinstance(t, int):
            return types.ThinkingConfig(thinking_budget=t)
        level = getattr(types.ThinkingLevel, str(t).upper(), None)
        if level is None:
            raise ValueError(f"thinking_level không hợp lệ: {t!r} (minimal/low/medium/high)")
        return types.ThinkingConfig(thinking_level=level)

    def _gen_config(self, system, json_mode, schema, max_output_tokens):
        kw: Dict[str, Any] = {"max_output_tokens": max_output_tokens or self.max_output_tokens}
        if system:
            kw["system_instruction"] = system
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.seed is not None:
            kw["seed"] = self.seed
        if json_mode or schema is not None:
            kw["response_mime_type"] = "application/json"
        if schema is not None:
            kw["response_schema"] = schema
        thinking = self._thinking_config()
        if thinking is not None:
            kw["thinking_config"] = thinking
        return self._types.GenerateContentConfig(**kw)

    # --- gọi API --------------------------------------------------------------
    def _call(self, prompt, system, json_mode, response_schema, max_output_tokens) -> _Raw:
        config = self._gen_config(system, json_mode, response_schema, max_output_tokens)
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)

        try:
            text = resp.text or ""
        except Exception:                       # phản hồi không có phần văn bản
            text = ""
        finish = "UNKNOWN"
        cands = getattr(resp, "candidates", None)
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            finish = getattr(fr, "name", None) or (str(fr) if fr is not None else "UNKNOWN")
        else:
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            if block is not None:
                finish = f"PROMPT_BLOCKED:{getattr(block, 'name', block)}"
        um = getattr(resp, "usage_metadata", None)
        return _Raw(text=text, finish_reason=finish,
                    input_tokens=getattr(um, "prompt_token_count", None),
                    output_tokens=getattr(um, "candidates_token_count", None),
                    thinking_tokens=getattr(um, "thoughts_token_count", None))

    def _classify_error(self, exc: Exception) -> Tuple[str, Optional[float]]:
        from google.genai import errors
        if isinstance(exc, errors.APIError):
            blob = json.dumps(getattr(exc, "details", None), ensure_ascii=False, default=str) \
                if getattr(exc, "details", None) else str(exc)
            blob += " " + str(getattr(exc, "message", ""))
            code = getattr(exc, "code", None)
            if code == 429:
                if _DAILY_RE.search(blob):
                    return "daily", None
                m = _RETRY_DELAY_RE.search(blob)
                return "retry", (float(m.group(1) or m.group(2)) if m else None)
            if code in (408, 500, 502, 503, 504):
                return "retry", None
            return "fatal", None                # 400 sai tham số / 401-403 khóa / 404 tên mô hình
        mod = type(exc).__module__.split(".")[0]
        if isinstance(exc, (TimeoutError, ConnectionError)) or mod in ("httpx", "httpcore", "requests"):
            return "retry", None
        return "fatal", None

    # --- tiện ích -------------------------------------------------------------
    def list_models(self) -> List[str]:
        """Các mô hình khóa API này gọi được bằng generateContent."""
        names = []
        for m in self._client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(str(m.name).removeprefix("models/"))
        return sorted(names)


# --------------------------------------------------------------------------- #
# Dòng lệnh: kiểm tra khóa/mô hình trước khi chạy thật
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Kiểm tra kết nối Gemini")
    ap.add_argument("--model", default=None, help="tên mô hình (mặc định: biến GEMINI_MODEL)")
    ap.add_argument("--list-models", action="store_true", help="liệt kê mô hình khóa này dùng được")
    ap.add_argument("--ping", default=None, metavar="CÂU_HỎI", help="gửi thử một câu và in kết quả")
    ap.add_argument("--thinking", default=None, help="số (budget) hoặc minimal/low/medium/high")
    args = ap.parse_args()

    thinking: Union[None, int, str] = None
    if args.thinking is not None:
        thinking = int(args.thinking) if args.thinking.lstrip("-").isdigit() else args.thinking
    if args.list_models:
        c = GeminiClient(model=args.model or "placeholder", cache=False)
        print("\n".join(c.list_models()))
    if args.ping:
        c = GeminiClient(model=args.model, thinking=thinking, cache=False)
        r = c.generate(args.ping)
        print(r.text)
        print(f"\n[finish={r.finish_reason} | {r.latency_s:.2f}s | vào={r.input_tokens} "
              f"ra={r.output_tokens} suy luận={r.thinking_tokens} | thử={r.attempts}]")


if __name__ == "__main__":
    main()