"""
logger.py - Ghi nhật ký thực nghiệm dùng chung (Weights & Biases / TensorBoard).

Cả 3 nhánh (BiLSTM-CRF, PLM, LLM) dùng chung MỘT giao diện để số liệu thống nhất:

    from src.common.logger import ExperimentLogger
    from src.common.metrics import compute_metrics

    with ExperimentLogger(run_name="bilstm_crf_phoner_word_s42", member="tv1",
                          config=cfg_dict, tags=["bilstm_crf", "phoner", "word"]) as lg:
        for epoch in range(n_epochs):
            lg.log({"train/loss": loss, "lr": lr}, step=epoch)
            lg.log_ner_metrics(compute_metrics(gold, pred), prefix="dev", step=epoch)
        lg.log_summary({"best_dev_f1": best})

Chọn nơi ghi (tham số `backends` hoặc biến môi trường NER_LOGGER):
  "auto"        (mặc định) W&B nếu đã đăng nhập; không thì TensorBoard nếu đã cài;
                không thì chỉ ghi file cục bộ. KHÔNG bao giờ bật hộp thoại đăng nhập.
  "wandb"       chỉ W&B       (chạy offline: đặt WANDB_MODE=offline)
  "tensorboard" chỉ TensorBoard (xem bằng: tensorboard --logdir outputs/logs)
  "wandb,tensorboard"   cả hai
  "local"/"none"  chỉ ghi file cục bộ

LUÔN ghi thêm file cục bộ trong  outputs/logs/<run_name>/ :
  config.json, metrics.jsonl (mỗi lần log một dòng), summary.json
-> dùng để dựng bảng/hình cho báo cáo và tái tạo kết quả, kể cả khi không có mạng.

Lỗi của W&B/TensorBoard (mất mạng, hết quota...) KHÔNG làm hỏng quá trình huấn luyện:
backend lỗi sẽ bị tắt kèm một cảnh báo, các backend còn lại vẫn chạy.
"""
from __future__ import annotations

import json
import logging
import netrc
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

log = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path(os.environ.get(
    "NER_LOG_DIR", Path(__file__).resolve().parents[2] / "outputs" / "logs"))
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "vietnamese-ner")


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def _to_scalar(v: Any) -> Any:
    """int/float/bool/str giữ nguyên; tensor/numpy scalar -> số Python; còn lại -> None."""
    if isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "run"


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _wandb_logged_in() -> bool:
    """Đã có API key (biến môi trường hoặc file netrc) hay chưa - để 'auto' không bị treo chờ đăng nhập."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        return True
    home = Path.home()
    for name in (".netrc", "_netrc"):
        path = home / name
        if path.exists():
            try:
                if netrc.netrc(str(path)).authenticators("api.wandb.ai"):
                    return True
            except Exception:
                pass
    return False


def _module_available(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _tensorboard_available() -> bool:
    return _module_available("tensorboard") and (
        _module_available("torch") or _module_available("tensorboardX"))


# --------------------------------------------------------------------------- #
# Các backend (mỗi backend: log / summary / finish)
# --------------------------------------------------------------------------- #
class _LocalBackend:
    name = "local"

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.summary_path = run_dir / "summary.json"
        self._summary: Dict[str, Any] = {}
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        rec = {"step": step, "time": datetime.now().isoformat(timespec="seconds"), "metrics": metrics}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def summary(self, values: Dict[str, Any]) -> None:
        self._summary.update(values)
        self.summary_path.write_text(
            json.dumps(self._summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def finish(self) -> None:
        pass


class _WandbBackend:
    name = "wandb"

    def __init__(self, project: str, run_name: str, config: Dict[str, Any],
                 tags: Optional[List[str]], run_dir: Path, mode: Optional[str]):
        import wandb  # import muộn: không bắt buộc phải cài
        kwargs: Dict[str, Any] = dict(project=project, name=run_name, config=config,
                                      tags=tags, dir=str(run_dir))
        if mode:
            kwargs["mode"] = mode
        self._run = wandb.init(**kwargs)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self._run.log(metrics, step=step)

    def summary(self, values: Dict[str, Any]) -> None:
        self._run.summary.update(values)

    def finish(self) -> None:
        self._run.finish()


class _TensorBoardBackend:
    name = "tensorboard"

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            from tensorboardX import SummaryWriter  # không cần torch
        self._writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        self._writer.add_text(
            "config", "```json\n" + json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n```")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            if _is_number(v):
                self._writer.add_scalar(k, v, step)

    def summary(self, values: Dict[str, Any]) -> None:
        self._writer.add_text(
            "summary", "```json\n" + json.dumps(values, ensure_ascii=False, indent=2, default=str) + "\n```")

    def finish(self) -> None:
        self._writer.close()


# --------------------------------------------------------------------------- #
# Logger chính
# --------------------------------------------------------------------------- #
class ExperimentLogger:
    def __init__(self,
                 run_name: Optional[str] = None,
                 project: str = DEFAULT_PROJECT,
                 config: Optional[Dict[str, Any]] = None,
                 backends: Union[str, Sequence[str], None] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 tags: Optional[Sequence[str]] = None,
                 member: Optional[str] = None,
                 verbose: bool = True,
                 wandb_mode: Optional[str] = None):
        """
        run_name : tên lần chạy (nên gồm nhánh_tập_mức_seed). Bỏ trống -> tự sinh từ member + giờ.
        config   : dict siêu tham số (vd đọc từ configs/*.yaml) - ghi lại để tái tạo kết quả.
        member   : "tv1"/"tv2"/"tv3" - gắn vào config và tag để lọc trên W&B.
        wandb_mode: "online" | "offline" | "disabled" (mặc định theo biến môi trường WANDB_MODE).
        """
        self.verbose = verbose
        self.project = project
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        self.run_name = _safe_name(run_name or f"{member or 'run'}_{stamp}")
        self.run_dir = Path(log_dir or DEFAULT_LOG_DIR) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config: Dict[str, Any] = dict(config or {})
        if member:
            self.config.setdefault("member", member)
        self.tags = list(tags or []) + ([member] if member else [])
        self._step = 0
        self._t0 = time.time()
        self._finished = False
        self._warned_keys: set = set()

        self._backends: List[Any] = [_LocalBackend(self.run_dir, self.config)]
        for name in self._resolve_backends(backends):
            self._add_backend(name, wandb_mode)

    # --- khởi tạo backend ---------------------------------------------------
    @staticmethod
    def _resolve_backends(spec: Union[str, Sequence[str], None]) -> List[str]:
        spec = spec if spec is not None else os.environ.get("NER_LOGGER", "auto")
        names = [s.strip().lower() for s in (spec.split(",") if isinstance(spec, str) else spec)]
        if names == ["auto"]:
            if _module_available("wandb") and _wandb_logged_in():
                return ["wandb"]
            return ["tensorboard"] if _tensorboard_available() else []
        return [n for n in names if n not in ("local", "none", "")]

    def _add_backend(self, name: str, wandb_mode: Optional[str]) -> None:
        try:
            if name == "wandb":
                self._backends.append(_WandbBackend(
                    self.project, self.run_name, self.config, self.tags, self.run_dir, wandb_mode))
            elif name == "tensorboard":
                self._backends.append(_TensorBoardBackend(self.run_dir, self.config))
            else:
                log.warning("Backend không biết: %r (dùng: wandb, tensorboard, local)", name)
        except Exception as e:  # chưa cài / chưa đăng nhập / lỗi mạng -> bỏ qua, vẫn huấn luyện
            log.warning("Không khởi tạo được backend %r (%s: %s) - bỏ qua.", name, type(e).__name__, e)

    @property
    def backend_names(self) -> List[str]:
        return [b.name for b in self._backends]

    def _call(self, method: str, *args) -> None:
        """Gọi method trên mọi backend; backend nào lỗi thì tắt nó, không làm hỏng huấn luyện."""
        for b in list(self._backends):
            try:
                getattr(b, method)(*args)
            except Exception as e:
                if isinstance(b, _LocalBackend):
                    raise  # ghi file cục bộ lỗi (hết đĩa, sai quyền) thì nên dừng để biết
                log.warning("Backend %r lỗi khi %s (%s: %s) - tắt backend này.",
                            b.name, method, type(e).__name__, e)
                self._backends.remove(b)

    # --- API ----------------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Ghi một dict số liệu. step=None -> tự tăng dần; truyền step (vd epoch) để dùng làm trục x."""
        if self._finished:
            raise RuntimeError("Logger đã finish(); hãy tạo ExperimentLogger mới cho lần chạy khác.")
        clean: Dict[str, Any] = {}
        for k, v in metrics.items():
            sv = _to_scalar(v)
            if sv is None:
                if k not in self._warned_keys:
                    self._warned_keys.add(k)
                    log.warning("Bỏ qua %r: giá trị không phải số vô hướng (%s)", k, type(v).__name__)
                continue
            clean[k] = sv
        if not clean:
            return
        step = self._step if step is None else int(step)
        self._step = step + 1
        self._call("log", clean, step)
        if self.verbose:
            body = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in clean.items())
            print(f"[{self.run_name} | step {step} | {time.time() - self._t0:6.1f}s] {body}", flush=True)

    def log_ner_metrics(self, result: Dict[str, Any], prefix: str = "dev",
                        step: Optional[int] = None) -> None:
        """
        Ghi kết quả của src.common.metrics.compute_metrics:
          {prefix}/precision, /recall, /f1 (micro), /macro_f1 và {prefix}/f1/<LOẠI> cho từng loại thực thể.
        """
        o = result["overall"]
        flat = {f"{prefix}/precision": o["precision"], f"{prefix}/recall": o["recall"],
                f"{prefix}/f1": o["f1"], f"{prefix}/macro_f1": result["macro_f1"]}
        for etype, m in result.get("per_type", {}).items():
            flat[f"{prefix}/f1/{etype}"] = m["f1"]
        self.log(flat, step=step)

    def log_summary(self, values: Dict[str, Any]) -> None:
        """Ghi giá trị tổng kết của cả lần chạy (vd best_dev_f1, test_f1, thời gian suy luận trung bình)."""
        clean = {k: _to_scalar(v) for k, v in values.items() if _to_scalar(v) is not None}
        self._call("summary", clean)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._call("finish")

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()

    def __repr__(self) -> str:
        return f"ExperimentLogger(run={self.run_name!r}, backends={self.backend_names}, dir={str(self.run_dir)!r})"
