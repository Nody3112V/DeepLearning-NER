"""Integration tests: runner + real BaseLLMClient (cache, stats) + real metrics.py.

No API key or network needed: a scripted fake model replaces Gemini.
    python -m pytest tests/test_run_zeroshot.py
"""
import json

import pytest

from src.common.metrics import evaluate_files
from src.models.llm_ner.api_client import BaseLLMClient, _Raw
from src.models.llm_ner.postprocess import bio_to_entities
from src.models.llm_ner.run_zeroshot import run

TEMPLATES = [
    (["Bệnh_nhân", "669", "là", "bác_sĩ", "tại", "Bệnh_viện", "Đồng_Nai"],
     ["O", "B-PATIENT_ID", "O", "B-OCCUPATION", "O", "B-LOCATION", "I-LOCATION"]),
    (["BN91", "(", "nam", ",", "28", "tuổi", ")", "ngày", "12/3"],
     ["B-PATIENT_ID", "O", "B-GENDER", "O", "B-AGE", "I-AGE", "O", "O", "B-DATE"]),
    (["Chuyến_bay", "VN0054", "hạ_cánh", "Nội_Bài"],
     ["O", "B-TRANSPORTATION", "O", "B-LOCATION"]),
]


class QuotaErr(Exception):
    pass


class FakeClient(BaseLLMClient):
    provider = "fake"

    def __init__(self, script, cache_path):
        super().__init__("fake-model", None, 0, 0.0, 0.0, cache_path)
        self.script, self.calls = script, 0

    def _call(self, prompt, system, json_mode, schema, max_tokens):
        self.calls += 1
        sentence = prompt.rsplit("Sentence: ", 1)[1].split("\nOutput:")[0]
        return self.script(sentence)

    def _classify_error(self, exc):
        return ("daily", None) if isinstance(exc, QuotaErr) else ("fatal", None)


def make_env(tmp_path, n=6):
    rows = []
    for i in range(n):
        toks, tags = TEMPLATES[i % len(TEMPLATES)]
        rows.append({"id": f"s{i}", "tokens": toks + [f"#{i}"], "tags": tags + ["O"]})
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cfg = {
        "model": {"name": "fake-model", "thinking": None, "temperature": None, "seed": 1,
                  "max_output_tokens": 256, "json_mode": True},
        "rate_limit": {"rpm": None},
        "data": {"dataset": "phoner", "test_path": str(test_path),
                 "subset": {"n": n, "seed": 1, "gold_path": str(tmp_path / "gold.jsonl")}},
        "prompt": {"n_shot": 0, "include_guidelines": True},
        "postprocess": {"repeat": "all", "case_fallback": True},
        "output": {"predictions_path": str(tmp_path / "pred.jsonl"),
                   "diagnostics_path": str(tmp_path / "diag.jsonl"),
                   "stats_path": str(tmp_path / "stats.json")},
    }
    by_sentence = {" ".join(r["tokens"]).replace("_", " "): r for r in rows}
    return cfg, by_sentence


def oracle(by_sentence):
    def script(sentence):
        row = by_sentence[sentence]
        text = json.dumps({"entities": bio_to_entities(row["tokens"], row["tags"])}, ensure_ascii=False)
        return _Raw(text, "STOP", 10, 5, 0)
    return script


def test_oracle_run_scores_one_and_resumes_from_cache(tmp_path):
    cfg, by_s = make_env(tmp_path)
    cache = tmp_path / "cache.jsonl"
    c1 = FakeClient(oracle(by_s), cache)
    s1 = run(cfg, client=c1)
    assert c1.calls == 6 and s1["n_scored"] == 6
    assert s1["metrics"]["overall"]["f1"] == 1.0
    assert s1["postprocess"]["n_malformed"] == 0
    # the saved gold subset + predictions are accepted by the official CLI path
    assert evaluate_files(cfg["data"]["subset"]["gold_path"],
                          cfg["output"]["predictions_path"])["overall"]["f1"] == 1.0
    # second run: everything from cache, zero model calls
    c2 = FakeClient(oracle(by_s), cache)
    s2 = run(cfg, client=c2)
    assert c2.calls == 0 and s2["client_stats"]["n_from_cache"] == 6
    assert s2["metrics"]["overall"]["f1"] == 1.0


def test_truncated_response_is_malformed_and_not_cached(tmp_path):
    cfg, by_s = make_env(tmp_path, n=3)
    good = oracle(by_s)
    first = next(iter(by_s))

    def script(sentence):
        if sentence == first:
            return _Raw('{"entities": [{"text": "669", "type": "PATIENT_ID"}, {"te', "MAX_TOKENS", 10, 256, 200)
        return good(sentence)

    cache = tmp_path / "cache.jsonl"
    c1 = FakeClient(script, cache)
    s1 = run(cfg, client=c1)
    assert s1["postprocess"]["malformed_reasons"] == {"finish_max_tokens": 1}
    assert s1["finish_reasons"] == {"STOP": 2, "MAX_TOKENS": 1}
    assert s1["n_scored"] == 3
    c2 = FakeClient(good, cache)
    run(cfg, client=c2)
    assert c2.calls == 1  # only the truncated sentence is asked again


def test_aborts_after_consecutive_api_errors(tmp_path):
    cfg, _ = make_env(tmp_path)

    def boom(sentence):
        raise RuntimeError("HTTP 400 thinking level not supported")

    s = run(cfg, client=FakeClient(boom, tmp_path / "cache.jsonl"))
    assert s["api_errors"] == 3 and s["n_scored"] == 0 and "consecutive" in s["stopped_early"]
    assert s["metrics"] is None


def test_daily_quota_stops_and_keeps_finished_rows(tmp_path):
    cfg, by_s = make_env(tmp_path)
    good = oracle(by_s)
    seen = []

    def script(sentence):
        if len(seen) >= 2:
            raise QuotaErr("PerDay quota exceeded")
        seen.append(sentence)
        return good(sentence)

    s = run(cfg, client=FakeClient(script, tmp_path / "cache.jsonl"))
    assert s["n_scored"] == 2 and s["stopped_early"].startswith("daily quota")
    assert s["metrics"]["overall"]["f1"] == 1.0


def test_mock_modes_run_offline(tmp_path):
    cfg, _ = make_env(tmp_path, n=12)
    assert run(cfg, mock="oracle")["metrics"]["overall"]["f1"] == 1.0
    noisy = run(cfg, mock="noisy")
    assert noisy["n_scored"] == 12
