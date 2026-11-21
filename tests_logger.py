"""Kiểm thử logger.py (phần W&B/TensorBoard chỉ chạy khi đã cài thư viện)."""
import json, os, sys, tempfile, time
from pathlib import Path
from src.common.logger import ExperimentLogger
from src.common.metrics import compute_metrics

def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

class FakeScalar:                     # giả lập tensor/numpy scalar
    def __init__(self, v): self.v = v
    def item(self): return self.v

tmp = tempfile.mkdtemp()

# --- 1. Chỉ ghi cục bộ: step tự tăng, step tường minh, scalar giả, giá trị không hợp lệ ---
with ExperimentLogger(run_name="t1 bilstm/phoner", member="tv1", config={"lr": 1e-3, "nested": {"a": 1}},
                      backends="local", log_dir=tmp, verbose=False) as lg:
    assert lg.backend_names == ["local"] and lg.run_name == "t1_bilstm_phoner"
    lg.log({"train/loss": 0.9})                      # step 0
    lg.log({"train/loss": FakeScalar(0.7)})          # step 1
    lg.log({"dev/f1": 0.5, "bad": [1, 2]}, step=10)  # step 10, 'bad' bị bỏ qua
    lg.log({"x": 1.0})                               # step 11
    lg.log({"only_bad": [1]})                        # rỗng sau khi lọc -> không ghi
    lg.log_summary({"best_dev_f1": FakeScalar(0.5), "note": "ok"})
rows = read_jsonl(Path(tmp) / "t1_bilstm_phoner" / "metrics.jsonl")
assert [r["step"] for r in rows] == [0, 1, 10, 11], rows
assert rows[1]["metrics"] == {"train/loss": 0.7} and rows[2]["metrics"] == {"dev/f1": 0.5}
cfg = json.load(open(Path(tmp) / "t1_bilstm_phoner" / "config.json", encoding="utf-8"))
assert cfg["lr"] == 1e-3 and cfg["member"] == "tv1" and cfg["nested"] == {"a": 1}
assert json.load(open(Path(tmp) / "t1_bilstm_phoner" / "summary.json"))["best_dev_f1"] == 0.5

# --- 2. log_ner_metrics với kết quả thật của metrics.py ---
gold = [["B-LOCATION", "I-LOCATION", "O"], ["O", "B-NAME", "O"]]
pred = [["B-LOCATION", "I-LOCATION", "O"], ["O", "O", "O"]]
res = compute_metrics(gold, pred)
lg = ExperimentLogger(run_name="t2", backends="local", log_dir=tmp, verbose=False)
lg.log_ner_metrics(res, prefix="dev", step=3); lg.finish()
m = read_jsonl(Path(tmp) / "t2" / "metrics.jsonl")[0]["metrics"]
assert m["dev/f1"] == res["overall"]["f1"] and m["dev/macro_f1"] == res["macro_f1"]
assert m["dev/f1/LOCATION"] == 1.0 and m["dev/f1/NAME"] == 0.0 and "dev/precision" in m

# --- 3. Backend lỗi không làm hỏng huấn luyện ---
class Boom:
    name = "boom"
    def log(self, *a): raise ConnectionError("mất mạng")
    def summary(self, *a): raise ConnectionError("mất mạng")
    def finish(self): raise ConnectionError("mất mạng")
lg = ExperimentLogger(run_name="t3", backends="local", log_dir=tmp, verbose=False)
lg._backends.append(Boom())
lg.log({"a": 1.0}); lg.log({"a": 2.0}); lg.log_summary({"s": 1}); lg.finish()
assert lg.backend_names == ["local"]                                  # Boom đã bị tắt
assert len(read_jsonl(Path(tmp) / "t3" / "metrics.jsonl")) == 2       # file cục bộ vẫn đủ

# --- 4. finish: idempotent; log sau finish báo lỗi rõ ràng ---
lg.finish()
try: lg.log({"a": 3.0}); raise SystemExit("phải lỗi")
except RuntimeError: pass

# --- 5. 'auto' không bao giờ treo chờ đăng nhập; backend lạ/không cài thì bỏ qua ---
for k in ("WANDB_API_KEY", "WANDB_MODE", "NER_LOGGER"): os.environ.pop(k, None)
os.environ["HOME"] = os.environ["USERPROFILE"] = tmp                  # không có netrc
t0 = time.time()
lg = ExperimentLogger(run_name="t5", log_dir=tmp, verbose=False)      # backends mặc định = auto
assert "wandb" not in lg.backend_names and time.time() - t0 < 10; lg.finish()
lg = ExperimentLogger(run_name="t5b", backends="khong_biet", log_dir=tmp, verbose=False)
assert lg.backend_names == ["local"]; lg.finish()
saved = sys.modules.get("wandb"); sys.modules["wandb"] = None         # giả lập chưa cài wandb
lg = ExperimentLogger(run_name="t5c", backends="wandb", log_dir=tmp, verbose=False)
assert lg.backend_names == ["local"]; lg.log({"a": 1.0}); lg.finish()
if saved is not None: sys.modules["wandb"] = saved
else: del sys.modules["wandb"]
print("Test logger (cục bộ, lỗi backend, auto) ĐẠT")

# --- 6. W&B offline thật (nếu đã cài) ---
try:
    import wandb  # noqa
except ImportError:
    print("Bỏ qua test W&B: chưa cài wandb")
else:
    os.environ["WANDB_MODE"] = "offline"; os.environ["WANDB_SILENT"] = "true"
    with ExperimentLogger(run_name="t6_wandb", member="tv3", config={"k": 5}, backends="wandb",
                          log_dir=tmp, verbose=False) as lg:
        assert lg.backend_names == ["local", "wandb"], lg.backend_names
        lg.log({"train/loss": 1.0}, step=0); lg.log_ner_metrics(res, "dev", step=0)
        lg.log_summary({"best": 0.5})
    assert list((Path(tmp) / "t6_wandb").glob("wandb/offline-run-*")), "không thấy thư mục run offline"
    # 'auto' + WANDB_MODE=offline phải chọn wandb
    lg = ExperimentLogger(run_name="t6b", log_dir=tmp, verbose=False); assert "wandb" in lg.backend_names; lg.finish()
    os.environ.pop("WANDB_MODE")
    print("Test W&B offline ĐẠT")

# --- 7. TensorBoard thật (nếu đã cài) ---
try:
    import tensorboard, tensorboardX  # noqa
except ImportError:
    print("Bỏ qua test TensorBoard: chưa cài tensorboard/tensorboardX")
else:
    with ExperimentLogger(run_name="t7_tb", backends="tensorboard", log_dir=tmp, verbose=False) as lg:
        assert lg.backend_names == ["local", "tensorboard"], lg.backend_names
        lg.log({"train/loss": 1.0, "name": "text-bi-bo-qua"}, step=0); lg.log_summary({"best": 1})
    assert list((Path(tmp) / "t7_tb" / "tb").glob("events.out.tfevents.*")), "không có file event"
    print("Test TensorBoard ĐẠT")
print("TẤT CẢ TEST logger ĐẠT")
