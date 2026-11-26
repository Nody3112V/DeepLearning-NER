"""Zero-shot LLM NER smoke run on a frozen subset (resumable).

    python -m src.models.llm_ner.run_zeroshot --config configs/llm_fewshot.yaml
    python -m src.models.llm_ner.run_zeroshot --config configs/llm_fewshot.yaml --mock oracle

Flow per sentence: tokens -> prompt -> GeminiClient (disk cache, rpm, retries live
in api_client.py) -> postprocess -> BIO tags -> predictions JSONL.

Resume: answered sentences come from the client's cache, so re-running costs no
quota for them. Not cached by the client (so re-queried on every run): responses
with finish_reason != STOP (cut off, blocked, empty). They are still scored, as
malformed under reason "finish_<reason>", and counted in the summary.

Stops early on DailyQuotaExceeded (run again tomorrow) or after 3 consecutive
LLMError (e.g. HTTP 400 because the model rejects the `thinking` value).

Scoring uses src.common.metrics.compute_metrics. Because the run covers a subset,
the subset's gold rows are saved to `data.subset.gold_path`; score the file with
    python -m src.common.metrics --gold <gold_path> --pred <predictions_path>
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.common.metrics import compute_metrics, format_report

from .api_client import DailyQuotaExceeded, GeminiClient, LLMError
from .postprocess import PostprocessStats, bio_to_entities, postprocess
from .prompt_templates import PROMPT_VERSION, TYPE_SETS, build_messages

log = logging.getLogger("llm.zeroshot")
MAX_CONSECUTIVE_ERRORS = 3


# ------------------------------------------------------------------- utils
def load_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_subset(rows: list[dict], n: int, seed: int, gold_path: str) -> list[dict]:
    """Frozen subset: sample once and save the gold rows; later runs reload that file."""
    if Path(gold_path).exists():
        return load_rows(gold_path)
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    subset = [rows[i] for i in idx]
    write_rows(gold_path, subset)
    return subset


def make_client(cfg: dict) -> GeminiClient:
    m = cfg["model"]
    return GeminiClient(
        model=m["name"],
        temperature=m.get("temperature"),      # None = model default (advised for Gemini 3)
        seed=m.get("seed"),
        thinking=m.get("thinking"),            # None | int budget | minimal/low/medium/high
        max_output_tokens=m["max_output_tokens"],
        rpm=cfg["rate_limit"]["rpm"],
    )


def mock_response(row: dict, mode: str, rng: random.Random) -> str:
    """Offline stand-in for the API, to test the pipeline end to end."""
    ents = bio_to_entities(row["tokens"], row["tags"])

    def dump(e):
        return json.dumps({"entities": e}, ensure_ascii=False)

    if mode == "oracle":
        return dump(ents)
    r = rng.random()
    if r < 0.08:
        return "I'm sorry, I can't help with that."
    if r < 0.16:
        return "```json\n" + dump(ents) + "\n```"
    if r < 0.26:
        return dump(ents + [{"text": "Thành phố Không Tồn Tại", "type": "LOCATION"}])
    if r < 0.34:
        return dump([{**e, "text": e["text"].lower()} for e in ents])
    if r < 0.40:
        return dump(ents + ents[:1])
    if r < 0.46:
        return dump([{**e, "type": e["type"].lower().replace("_and_", " & ")} for e in ents])
    return dump(ents)


# -------------------------------------------------------------------- main
def run(cfg: dict, mock: str | None = None, limit: int | None = None,
        out_override: str | None = None, client: Any = None) -> dict:
    mcfg, dcfg, pcfg = cfg["model"], cfg["data"], cfg["prompt"]
    ocfg, ppcfg = cfg["output"], cfg["postprocess"]
    types = TYPE_SETS[dcfg["dataset"]]
    allowed = list(types)

    sub = dcfg["subset"]
    subset = select_subset(load_rows(dcfg["test_path"]), sub["n"], sub["seed"], sub["gold_path"])
    if limit:
        subset = subset[:limit]
    log.info("subset: %d sentences (dataset=%s)", len(subset), dcfg["dataset"])

    pred_path = Path(out_override or ocfg["predictions_path"])
    diag_path = Path(ocfg["diagnostics_path"])
    stats_path = Path(ocfg["stats_path"])
    if mock:
        pred_path = pred_path.with_suffix(f".mock-{mock}.jsonl")
        diag_path = diag_path.with_suffix(f".mock-{mock}.jsonl")
        stats_path = stats_path.with_suffix(f".mock-{mock}.json")
    for p in (pred_path, diag_path, stats_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    if not mock and client is None:
        client = make_client(cfg)
    if client is not None:
        client.reset_stats()
    rng = random.Random(mcfg["seed"])

    total = PostprocessStats()
    finish_counts: Counter = Counter()
    gold_ok: list[dict] = []
    pred_ok: list[dict] = []
    api_errors = consecutive = 0
    stop_reason = None
    t0 = time.time()

    with open(pred_path, "w", encoding="utf-8") as fp, open(diag_path, "w", encoding="utf-8") as fd:
        for i, row in enumerate(subset):
            rid = str(row.get("id", i))
            system, prompt = build_messages(
                row["tokens"], include_guidelines=pcfg["include_guidelines"], entity_types=types)
            meta: dict = {}
            failure = None
            if mock:
                raw = mock_response(row, mock, rng)
            else:
                try:
                    r = client.generate(prompt, system=system, json_mode=mcfg["json_mode"])
                except DailyQuotaExceeded as exc:
                    stop_reason = f"daily quota exhausted: {exc}"
                    log.error("%s", stop_reason)
                    break
                except LLMError as exc:
                    api_errors += 1
                    consecutive += 1
                    log.warning("API error on %s: %s", rid, exc)
                    if consecutive >= MAX_CONSECUTIVE_ERRORS:
                        stop_reason = f"{consecutive} consecutive API errors; last: {exc}"
                        log.error("%s", stop_reason)
                        break
                    continue
                consecutive = 0
                raw = r.text
                finish_counts[r.finish_reason] += 1
                if r.finish_reason != "STOP":
                    failure = "finish_" + r.finish_reason.lower().split(":")[0]
                meta = {"finish_reason": r.finish_reason, "cached": r.cached,
                        "latency_s": r.latency_s, "thinking_tokens": r.thinking_tokens}

            res = postprocess(row["tokens"], raw, allowed, repeat=ppcfg["repeat"],
                              case_fallback=ppcfg["case_fallback"], failure_reason=failure)
            total.merge(res.stats)
            pred = {**row, "tags": res.tags}
            fp.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fd.write(json.dumps({"id": rid, "status": res.status, "issues": res.issues,
                                 **meta, "raw": raw}, ensure_ascii=False) + "\n")
            gold_ok.append(row)
            pred_ok.append(pred)
            if (i + 1) % 20 == 0:
                log.info("%d/%d done (api errors %d)", i + 1, len(subset), api_errors)

    metrics = (compute_metrics([g["tags"] for g in gold_ok], [p["tags"] for p in pred_ok])
               if gold_ok else None)
    summary = {
        "model": mcfg["name"], "thinking": mcfg.get("thinking"), "prompt_version": PROMPT_VERSION,
        "mock": mock, "n_requested": len(subset), "n_scored": len(gold_ok),
        "api_errors": api_errors, "stopped_early": stop_reason,
        "finish_reasons": dict(finish_counts), "seconds": round(time.time() - t0, 1),
        "client_stats": client.stats() if client is not None else None,
        "postprocess": total.to_dict(), "metrics": metrics,
    }
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nscored {len(gold_ok)}/{len(subset)} sentences | api errors: {api_errors} | "
          f"malformed rate: {total.malformed_rate:.1%} | unmatched entities: {total.unmatched_rate:.1%}")
    if summary["client_stats"]:
        cs = summary["client_stats"]
        print(f"mean latency: {cs['mean_latency_s']}s | thinking tokens: {cs['thinking_tokens']} "
              f"| from cache: {cs['n_from_cache']} | retries: {cs['n_retries']}")
    if metrics:
        print(format_report(metrics))
    if stop_reason:
        print(f"STOPPED EARLY: {stop_reason}\nRe-run to continue; answered sentences come from cache.")
    print(f"predictions -> {pred_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--mock", choices=["oracle", "noisy"], help="offline pipeline test, no API")
    ap.add_argument("--limit", type=int, help="only the first N sentences of the subset")
    ap.add_argument("--out", help="override predictions path")
    args = ap.parse_args()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, mock=args.mock, limit=args.limit, out_override=args.out)


if __name__ == "__main__":
    main()
