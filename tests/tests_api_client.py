"""Kiểm thử api_client.py bằng client Gemini GIẢ (không gọi mạng, không cần khóa API).
Dùng các lớp lỗi và cấu hình THẬT của thư viện google-genai để kiểm tra việc ánh xạ."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

from google.genai import errors
from src.models.llm_ner.api_client import GeminiClient, DailyQuotaExceeded, LLMError

def resp(text="ok", finish="STOP", pin=10, pout=5, think=0):
    return NS(text=text, candidates=[NS(finish_reason=NS(name=finish))],
              usage_metadata=NS(prompt_token_count=pin, candidates_token_count=pout, thoughts_token_count=think))

def err(code, status="X", details=None, msg="m"):
    return errors.ClientError(code, {"error": {"code": code, "message": msg, "status": status,
                                               "details": details or []}}) if code < 500 else \
           errors.ServerError(code, {"error": {"code": code, "message": msg, "status": status}})

class FakeModels:
    def __init__(self, script): self.script, self.calls, self.configs = list(script), 0, []
    def generate_content(self, model, contents, config):
        self.calls += 1; self.configs.append(config)
        item = self.script.pop(0) if self.script else resp()
        if isinstance(item, Exception): raise item
        return item
    def list(self):
        return [NS(name="models/gemini-a", supported_actions=["generateContent"]),
                NS(name="models/embed-b", supported_actions=["embedContent"]),
                NS(name="models/gemini-c", supported_actions=["generateContent", "countTokens"])]

def make(script, tmp, **kw):
    fake = NS(models=FakeModels(script))
    c = GeminiClient(model="m1", client=fake, cache_path=Path(tmp) / "c.jsonl", **kw)
    c.sleeps = []; c._sleep = c.sleeps.append
    return c, fake.models

tmp = tempfile.mkdtemp()

# 1. Thành công: nội dung, token, độ trễ, cấu hình gửi đi
c, m = make([resp("xin chào", pin=7, pout=3, think=2)], tmp, temperature=0.0, seed=42, thinking="low")
r = c.generate("p", system="sys", json_mode=True)
assert (r.text, r.ok, r.attempts, r.cached) == ("xin chào", True, 1, False)
assert (r.input_tokens, r.output_tokens, r.thinking_tokens) == (7, 3, 2) and r.latency_s >= 0
cfg = m.configs[0]
assert cfg.temperature == 0.0 and cfg.seed == 42 and cfg.system_instruction == "sys"
assert cfg.response_mime_type == "application/json" and cfg.thinking_config.thinking_level.name == "LOW"

# 2. Ánh xạ thinking: int -> budget; sai -> lỗi; None -> không gửi; temperature None -> không gửi
c2, m2 = make([resp()], tmp + "/b", thinking=0); c2.generate("p")
assert m2.configs[0].thinking_config.thinking_budget == 0 and m2.configs[0].temperature is None
for bad in ("sieu", True):
    try: make([], tmp, thinking=bad); raise SystemExit("phải lỗi")
    except ValueError: pass
c3, m3 = make([resp()], tmp + "/c"); c3.generate("p"); assert m3.configs[0].thinking_config is None

# 3. Cache: lần 2 không gọi API, giữ độ trễ gốc; tiếp tục được sau khi 'sập' (instance mới)
c, m = make([resp("A"), resp("B")], tmp + "/d")
a1 = c.generate("câu 1"); assert m.calls == 1
a2 = c.generate("câu 1"); assert m.calls == 1 and a2.cached and a2.text == "A" and a2.latency_s == a1.latency_s
c_new, m_new = make([], tmp + "/d")                       # chạy lại từ đầu, cùng file cache
assert c_new.generate("câu 1").cached and m_new.calls == 0
assert not c_new.generate("câu 2").cached and m_new.calls == 1   # câu khác thì gọi
c_np, m_np = make([resp("X")], tmp + "/d"); c_np.generate("câu 1", use_cache=False); assert m_np.calls == 1

# 4. Tham số khác -> khóa cache khác (không dùng nhầm kết quả của cấu hình khác)
c_t, m_t = make([resp()], tmp + "/d", temperature=0.3)
c_t.generate("câu 1"); assert m_t.calls == 1

# 5. Thử lại 429: tôn trọng retryDelay của server; tính cả số lần thử
retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]
c, m = make([err(429, "RESOURCE_EXHAUSTED", retry_info), err(503), resp("sau thử lại")], tmp + "/e")
r = c.generate("p")
assert r.text == "sau thử lại" and r.attempts == 3 and m.calls == 3
assert len(c.sleeps) == 2 and 34 <= c.sleeps[0] <= 41 and c.sleeps[1] >= 2   # 503 -> lùi dần
assert r.wall_s >= r.latency_s

# 6. Hết hạn mức NGÀY: dừng ngay, không thử lại, không chờ
daily = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
          "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]
c, m = make([err(429, "RESOURCE_EXHAUSTED", daily)], tmp + "/f")
try: c.generate("p"); raise SystemExit("phải ném DailyQuotaExceeded")
except DailyQuotaExceeded: pass
assert m.calls == 1 and c.sleeps == []

# 7. Lỗi không thể thử lại (400/403/404): 1 lần gọi rồi LLMError; hết số lần thử cũng LLMError
for code in (400, 403, 404):
    c, m = make([err(code, "INVALID_ARGUMENT")], tmp + f"/g{code}")
    try: c.generate("p"); raise SystemExit("phải lỗi")
    except LLMError as e: assert not isinstance(e, DailyQuotaExceeded)
    assert m.calls == 1
c, m = make([err(503)] * 10, tmp + "/h", max_retries=2)
try: c.generate("p"); raise SystemExit("phải lỗi")
except LLMError: pass
assert m.calls == 3 and len(c.sleeps) == 2                  # 1 lần đầu + 2 lần thử lại

# 8. Lỗi mạng/timeout được thử lại
c, m = make([TimeoutError("hết giờ"), resp("ok")], tmp + "/i"); assert c.generate("p").attempts == 2

# 9. Phản hồi bất thường: trả về ok=False và KHÔNG cache
c, m = make([resp("{\"a\":", finish="MAX_TOKENS"), resp("", finish="STOP"), resp("tốt")], tmp + "/j")
x = c.generate("p"); assert not x.ok and x.finish_reason == "MAX_TOKENS"
y = c.generate("p"); assert not y.ok and not y.cached and m.calls == 2     # rỗng, không lấy từ cache
z = c.generate("p"); assert z.ok and m.calls == 3
blocked = NS(text=None, candidates=None, prompt_feedback=NS(block_reason=NS(name="SAFETY")), usage_metadata=None)
c, m = make([blocked], tmp + "/k"); b = c.generate("p")
assert not b.ok and b.finish_reason == "PROMPT_BLOCKED:SAFETY" and b.input_tokens is None

# 10. Giãn cách rpm: 2 yêu cầu liên tiếp phải cách nhau >= 60/rpm giây (đồng hồ giả)
now = [100.0]
c, m = make([resp(), resp()], tmp + "/l", rpm=6)            # 10 giây/yêu cầu
c._clock = lambda: now[0]; c._sleep = lambda s: (c.sleeps.append(s), now.__setitem__(0, now[0] + s))
c.generate("a"); now[0] += 3; c.generate("b")
assert abs(c.sleeps[-1] - 7.0) < 1e-6, c.sleeps             # đã trôi 3s -> chờ thêm 7s

# 11. Thống kê
c, m = make([resp("A", pin=10, pout=4, think=1), resp("B", pin=20, pout=6, think=3), resp("", finish="MAX_TOKENS")], tmp + "/m")
c.generate("1"); c.generate("2"); c.generate("3"); c.generate("1")           # câu 4 lấy từ cache
s = c.stats()
assert (s["n_responses"], s["n_ok"], s["n_bad"], s["n_from_cache"]) == (4, 3, 1, 1)
# câu 3 (bị cắt) vẫn tốn 10 token vào; câu 4 lấy từ cache nên cộng lại token của câu 1
assert s["input_tokens"] == 10 + 20 + 10 + 10 and s["thinking_tokens"] == 1 + 3 + 0 + 1
assert s["mean_latency_s"] is not None; c.reset_stats(); assert c.stats()["n_responses"] == 0

# 12. Thiếu cấu hình: báo lỗi rõ ràng
import os
for k in ("GEMINI_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY"): os.environ.pop(k, None)
for kw in ({}, {"model": "m"}):
    try: GeminiClient(cache=False, **kw); raise SystemExit("phải lỗi")
    except ValueError as e: assert ("mô hình" in str(e)) if not kw else ("GEMINI_API_KEY" in str(e))

# 13. list_models: chỉ lấy mô hình có generateContent, bỏ tiền tố 'models/'
c, _ = make([], tmp + "/n"); assert c.list_models() == ["gemini-a", "gemini-c"]
print("TẤT CẢ TEST api_client ĐẠT")